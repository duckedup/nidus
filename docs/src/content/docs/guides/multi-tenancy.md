---
title: Multi-tenancy
description: Give each tenant its own nidus store under one base location with Namespaces, a library handle that opens stores lazily and keeps only a byte-bounded warm set in RAM at once.
---

A nidus store pins one dimension and one writer for its whole life. That is exactly right
for one tenant's data, but a host serving many tenants from one process cannot put them
all in one store: there is no partition inside a store cheap enough to isolate on (a
collection has no lock, no segments, and no memory-tier key of its own). What isolates
cleanly is a **store**. So the per-tenant unit is a store, one per tenant, and
`Namespaces` is the library handle that manages many of them under one base location.

This is the library-side model only. `nidus serve` today still opens one store; routing
one HTTP or MCP server across many tenants by name is a later phase, not this one. What
follows applies to an embedding host calling nidus directly, in process.

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

## What stays the same

Aliases and blue/green reindexing (see the [blue/green reindexing
guide](/guides/blue-green-reindex/)) are unaffected: both operate on collections within
one store, and a namespace-per-store model changes what a namespace is, not what a
collection inside one can do. A tenant's store still has its own aliases, still supports
the same repoint-and-drop sequence, and still pins one dimension for its life. Nothing
here changes single-tenant usage either: opening one store directly with `Nidus::open`
needs no `Namespaces` at all.
