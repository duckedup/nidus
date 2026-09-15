---
title: Multi-tenancy
description: Give each tenant its own nidus store under one base location, either in process with Namespaces or over the network with `nidus serve --namespaced`, both backed by the same byte-bounded warm set.
---

A nidus store pins one dimension and one writer for its whole life. That is exactly right
for one tenant's data, but a host serving many tenants from one process cannot put them
all in one store: there is no partition inside a store cheap enough to isolate on (a
collection has no lock, no segments, and no memory-tier key of its own). What isolates
cleanly is a **store**. So the per-tenant unit is a store, one per tenant, and
`Namespaces` is the library handle that manages many of them under one base location.

Two ways to reach that: call `Namespaces` directly from an embedding host, in process,
or point `nidus serve` at one base location with `--namespaced` and let every request
name its tenant over HTTP or MCP. Both sit on the same warm set and the same eviction
policy; the server is a thin routing layer in front of the library handle, not a second
implementation.

## What a tenant gets

Each name passed to `Namespaces` resolves to its own store, opened at `base/<name>` (or
the equivalent child of `base` on S3 or GCS). Two tenants never share:

- **A writer lock.** One tenant's writer never contends with another's, because each
  store takes its own lock file (or object lease) under its own path.
- **Segments.** Each tenant's `data`/`log` objects, and any sealed segments, live under
  that tenant's own prefix. Nothing about one tenant's rows is visible while operating
  on another's store.
- **A memory-tier key.** If you point several tenants at one shared Redis (or Valkey,
  KeyDB, DragonflyDB) server, `Namespaces` derives a distinct `?prefix=` per tenant from
  its name, so their published working sets cannot collide on the shared
  `"workingset"` key. You do not have to set the prefix yourself per tenant.

## The API

```rust
Namespaces::new(template: Config, base: impl Into<String>) -> Self
    .budget_bytes(bytes: u64) -> Self
    .get(&mut self, name: &str) -> Result<&mut Nidus>
    .warm(&self) -> Vec<String>
    .evict(&mut self, name: &str) -> Result<bool>
    .warm_bytes(&self) -> u64
```

`new` takes a template `Config` (every setting except the path: dimension, fsync policy,
quantization, ann, and so on) and a base location under which every tenant's store lives.
Nothing is opened yet. `get(name)` is the only thing that opens a store: the first call
for a given name opens (or creates) it at `base/name`, applies the template, and hands
back a `&mut Nidus` to search and write through; a later call for the same name returns
the already-open store.

```rust
use nidus::{Config, Namespaces};

let template = Config::new("", 768); // path is filled in per tenant; the rest applies to all
let mut tenants = Namespaces::new(template, "./tenants").budget_bytes(512 * 1024 * 1024);

let store = tenants.get("acme-corp")?;
store.upsert("docs", &records)?;
```

## The byte budget

`budget_bytes` caps how much tenant state stays open across every namespace at once, not
how many tenant names exist. A host with more tenants than fit in that budget is the
normal case, not an error: when a `get` for a new tenant would cross the cap,
`Namespaces` evicts the coldest already-open tenant (flushing it first) to make room, the
same way an OS page cache reclaims the least-recently-used page rather than refusing the
new one. `warm()` lists which tenants are open right now, `warm_bytes()` reports the
current total against your budget, and `evict(name)` lets you reclaim one tenant by hand,
for instance right after a bulk job you know will not touch that tenant again soon.

There is no minimum: set the budget low to keep only a handful of stores open at a time
on a small host, or high on a box with room for most of your tenants resident at once.
Either way, a tenant currently evicted is not gone: its data is durable on disk (or S3,
or GCS) exactly as any nidus store's is, and the next `get` for its name reopens it.

## `mmap` defaults on here

A single-tenant `Config` defaults `mmap` to `false`: sealed segments are read fully into
RAM. `Namespaces` flips that default to `true` for any tenant whose template did not set
`mmap` explicitly, because the whole point of a byte budget across many tenants is that
most of them are cold most of the time, and a memory-mapped sealed segment costs no RAM
until a query actually touches its pages. If your template calls `.mmap(false)`
yourself, that choice is honored: `Namespaces` only fills in a default, it never
overrides an explicit setting.

## Over the network: `nidus serve --namespaced`

Everything above is the library handle. `nidus serve` puts it behind HTTP and MCP: start
it with `--namespaced` and `--dir`/`--persistence` names the base location instead of a
single store, so every request names its own tenant instead of the server opening one
store at startup.

```bash
nidus serve --namespaced --dir ./tenants --dim 768 --warm-budget-bytes 536870912
```

`--warm-budget-bytes` is `Namespaces::budget_bytes`; omit it for the library's own 1 GiB
default. `--dim` is still required, exactly as in single-store mode: it is the template
every namespace's store is created from, and the command refuses to start without it. No
store opens until the first request names one, so the listener (and `/ready`, see below)
comes up immediately, but the dimension is fixed for the whole base location up front.

A request addresses its tenant by putting `/ns/{namespace}` in front of the path it would
otherwise use, for every route this server answers:

```bash
curl -s -X POST localhost:7700/ns/acme-corp/collections/docs/upsert \
  -H 'content-type: application/json' \
  -d '{"records": [{"id": "a", "vector": [1,0,0]}]}'

curl -s localhost:7700/ns/acme-corp/search \
  -H 'content-type: application/json' -d '{"query": [1,0,0], "top_k": 5}'
```

`/ns/{namespace}` is stripped before the request reaches the ordinary handler, so every
field, error code, and response shape on the [HTTP API](/reference/http-api/) page is
unchanged; only the path grows a prefix. `/mcp` takes the same prefix
(`/ns/acme-corp/mcp`), and a namespaced-mode tool call may also pass an explicit
`namespace` argument instead, which is what a connection at bare `/mcp` needs to address
one; see [Namespaced mode](/guides/mcp/#namespaced-mode) in the MCP guide.

A namespaced-mode server also answers `GET /namespaces`: every namespace currently in the
warm set, its byte size, and its readiness, without opening anything the request did not
already touch.

```bash
curl -s localhost:7700/namespaces
```

```json
[{"name": "acme-corp", "bytes": 40960, "role": "Writer", "fenced": false, "staleness_secs": 0}]
```

This is where per-tenant health lives; see the next section for why `/ready` itself does
not carry it.

### Single-credential, not tenant isolation

**Namespaced mode is single-credential.** `--token` is the same one process-wide bearer
token it always was: whatever it protects, it protects for **every** namespace this
process serves. A caller holding the token can read and write any tenant by name; there
is no per-tenant token, and nothing here scopes one tenant away from another at the
network layer the way the store-per-tenant isolation above scopes them on disk. If your
tenants must not be able to reach each other's data even when both hold a valid
credential, this ticket does not give you that: `nidus-uyb` is the tracked follow-up for
per-tenant credentials, and until it ships, treat one namespaced `nidus serve` as one
trust boundary, not many.

### `/ready` stays process-level

`/ready` keeps exactly its single-store meaning: whether this process is up and can
accept a request at all. It does **not** open, and says nothing about, any one namespace,
because there is no one store to be ready or not. A namespace that has never been
requested is neither ready nor unready: it simply does not exist yet as far as this
process is concerned. Per-namespace state (role, fenced, staleness) is answered by
`GET /namespaces` instead, scoped to whichever namespaces are currently warm.

## What stays the same

Aliases and blue/green reindexing (see the [blue/green reindexing
guide](/guides/blue-green-reindex/)) are unaffected: both operate on collections within
one store, and a namespace-per-store model changes what a namespace is, not what a
collection inside one can do. A tenant's store still has its own aliases, still supports
the same repoint-and-drop sequence, and still pins one dimension for its life. Nothing
here changes single-tenant usage either: opening one store directly with `Nidus::open`,
or running `nidus serve` without `--namespaced`, needs no `Namespaces` at all.
