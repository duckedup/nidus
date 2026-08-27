---
title: HTTP server
description: "Run nidus as an HTTP server with `nidus serve`: start it, authenticate it, and drive a store over JSON with no Rust toolchain on the client."
---

`nidus serve` opens one store and exposes it over HTTP. Every library operation
has an endpoint, so a client that never links the crate can do the full job over
the network: create collections, upsert vectors, search, filter, inspect, and
maintain the store, all in JSON. The wire format is the same store directory the
[library](/guides/integrating/) and the [CLI](/guides/cli-and-server/) read and
write; the server is just another door into it.

The raw vector routes store and search the vectors you give them: you compute
embeddings with your own model (in any language, on the client), then send the
resulting vectors here to upsert and query. The server can also embed for you:
started with `--embed-provider` (the shipped binary includes every provider), it
answers `POST /collections/{name}/remember` and `/recall` with text in and ranked
text out, and serves the same memory layer to agents at `/mcp`. See
[remember & recall](/guides/remember-and-recall/) and [MCP](/guides/mcp/).

This page covers running the server. For the route-by-route reference, see the
[HTTP API](/reference/http-api/); for driving a store from your shell, see the
[command-line guide](/guides/cli-and-server/).

## Start the server

```bash
# Create the store on first run by passing --dim; afterwards it's inferred.
nidus serve --dir ./store --dim 768 --addr 127.0.0.1:7700
```

`nidus serve` prints its bind address and serves until you stop it with Ctrl-C,
flushing to disk on the way out. The store directory need not exist yet (the
first write creates it), and for the raw vector routes `--dim` is required until
it does, because the embedding dimension is pinned at creation. Started with
`--embed-provider` instead, `--dim` is optional for a store that does not exist
yet: the embedder already knows its own dimension, so nidus uses that. Either
way, an existing store's on-disk header wins, and a `--dim` that disagrees with
it is still a hard error.

Pass `--read-only` to serve without taking the writer lock: a search-only process
that can run beside a separate writer.

To serve approximate (ANN) search, add `--ann hnsw` or `--ann ivf` (with the
optional `--ann-*` knobs from the [command-line guide](/guides/cli-and-server/)), or
record it once as the store's default with `nidus configure --ann hnsw` (see
[Configure once](/guides/cli-and-server/#configure-once-recording-store-defaults))
so `serve` picks it up without the flag. The index lives in memory for the life of
the process; `GET /stats` reports the active configuration.

## A complete session over HTTP

From an empty directory to ranked results without ever touching the binary again
after launch. Start the server in one terminal:

```bash
nidus serve --dir ./store --dim 3 --addr 127.0.0.1:7700
```

Then drive it entirely over HTTP from another:

```bash
# 1. Create a collection.
curl -s -X POST localhost:7700/collections/docs

# 2. Upsert records: id + vector + any typed metadata.
curl -s localhost:7700/collections/docs/upsert \
  -H 'content-type: application/json' \
  -d '{"records": [
        {"id": "a", "vector": [1,0,0], "attrs": {"lang": {"Str": "rust"}}},
        {"id": "b", "vector": [0,1,0], "attrs": {"lang": {"Str": "go"}}}
      ]}'
# → {"upserted": 2}

# 3. Search for nearest neighbours.
curl -s localhost:7700/search \
  -H 'content-type: application/json' \
  -d '{"query": [1,0,0], "top_k": 2}'
# → [{"collection":"docs","id":"a","score":1.0,"attrs":{"lang":{"Str":"rust"}}}, …]

# 4. Inspect the store.
curl -s localhost:7700/stats
# → {"dimension":3,"distance":"Cosine","ann":null,"quantization":null,
#    "query_threads":1,"mmap":false,"collections":["docs"],"footprint":{…}}
```

That is a complete vector store over the network: no Rust toolchain on the client,
nothing but HTTP and JSON.

## Text-native ingest

The routes above take and return raw vectors: you embed on the client, in
whatever language you like, and send the result. `nidus serve` can also embed,
and optionally summarize, text itself, which is what powers
`POST /collections/{name}/remember` and `/recall`: text in, ranked text out, with
the vector math handled on the server. See
[remember & recall](/guides/remember-and-recall/) for the request and response
shapes, and [MCP](/guides/mcp/) for the same layer exposed as agent tools at
`/mcp`.

Configure an embedder with `--embed-provider` and, optionally, a summarizer with
`--summarize-provider`. Both take the same shape of flags: a provider name, a
model, an API key, and a base-URL override, each with a matching `NIDUS_*`
environment variable (see the embed and summarize rows in the
[environment table](#configuration-from-the-environment) below). The
[CLI reference](/reference/cli/) is canonical for the exact provider list and
per-flag syntax; here is the shape:

```bash
nidus serve --dir ./store \
  --embed-provider voyage --embed-model voyage-3.5 --embed-api-key "$VOYAGE_API_KEY" \
  --summarize-provider anthropic --summarize-api-key "$ANTHROPIC_API_KEY"
```

Omit `--embed-provider` and the server still starts, serving only the raw vector
routes; `/remember` and `/recall` then answer `400` (see below). A base-URL
override matters most for `openai-compat` and self-hosted gateways, which have no
default endpoint to fall back to.

### Feature gating

`--embed-*` and `--summarize-*` exist only in a build compiled with the `memory`
feature, which the `serve` feature umbrella pulls in along with every provider:
`serve = ["cli", "memory", "embed-all", "summarize-all", "rerank-all", "mcp",
"code"]`, and `default = ["serve"]`. `cargo install nidus`, `cargo binstall
nidus`, and the binaries `release.yml` publishes all build with `serve`, so the
shipped binary always has every provider. Only a `--no-default-features
--features cli` build opts out: it has no `--embed-provider` flag at all, and
clap rejects it as unrecognised.

### Failure modes

- **No embedder configured.** `/remember` and `/recall` both answer `400`, with a
  message naming `--embed-provider` (`missing_embedder_error` in
  `src/server/mod.rs`).
- **`mode: "summarize"` with no summarizer.** A separate `400`, naming
  `--summarize-provider` instead.
- **Dimension pinning.** A store's embedding dimension is fixed at creation, and
  the whole store shares one embedding space. If the configured embedder's
  dimension does not match, the write is refused, but unlike a raw-vector
  dimension mismatch (which is a `400`) this one surfaces as a `500`: the error
  message does not match the string the HTTP layer's error classifier checks for
  on the raw-vector path, so it falls through to the generic server-fault status.
  Point `--embed-provider` / `--embed-model` at the dimension the store was
  created with, or start a new store.
- **Cross-model recall.** A collection remembers which embedder first wrote to it
  (provider and model). Recalling into it with a different embedder is refused
  with `409 Conflict`, even at the same dimension, because a same-dimension,
  different-model space still ranks nonsense.
- **Retries against the provider.** Every embed and summarize call goes through
  the shared retry layer in `src/http.rs`: exponential backoff
  (`base_delay_ms * 2^attempt`), retrying on `429` and the common transient
  `5xx` statuses (`500`, `502`, `503`, `529`) for the hosted providers, or any
  `5xx` for Ollama specifically, since it is local and has no rate limit to
  respect. Retries are bounded (three attempts) and not configurable by flag;
  exhausting them fails the request rather than queuing the text for later.

## Authentication

The server is unauthenticated by default, which is fine on `127.0.0.1`. The moment
you bind a non-local address, set `--token <secret>` (or the `NIDUS_TOKEN` env
var). Every request except the probe endpoints (`GET /health`, `GET /ready`,
`GET /metrics`) must then carry `Authorization: Bearer <secret>`; anything else
gets `401`.

```bash
nidus serve --dir ./store --addr 0.0.0.0:7700 --token "$NIDUS_TOKEN"

curl -s localhost:7700/stats -H "authorization: Bearer $NIDUS_TOKEN"
```

The probes are open on purpose: an orchestrator that got a `401` from `/ready`
would read the instance as down and never route to it, and a metrics scraper would
report the same.

## Securing a deployment

**Treat nidus like a database: it belongs on a private network, and none of its
endpoints should be reachable from the public internet.** You would not put
Postgres on a public IP and rely on its password prompt; the same reasoning applies
here. Nothing in nidus assumes a hostile caller, and network placement, not
`--token`, is what keeps it safe. That boundary is yours to enforce.

Given that, the rest follows:

- **nidus serves plain HTTP, by design.** There is no `--tls` flag, because
  anywhere nidus is reachable off-box there is already an ingress, sidecar, or mesh
  terminating TLS, with rotation, SNI, and cipher policy handled by infrastructure
  you already operate, better than a TLS stack compiled into a vector store would.
- **`--token` authenticates a caller. It does not confer confidentiality.** Over
  plain HTTP the token crosses the network in cleartext on every request, alongside
  every vector, document id, and metadata value. It is a guard against a
  misconfigured neighbour on your own network, not a perimeter.
- **The bind address is the real control.** `--addr 0.0.0.0:7700` with no `--token`
  is an open, writable vector store for anyone who can route to it. nidus warns at
  startup on a non-loopback bind: with a token, that the credential is in
  cleartext; without one, that the store is open. It warns and starts; it never
  refuses, because refusing would break the proxy-terminated architecture above.
- **Rate limiting and slow-client protection belong at the proxy.** nidus bounds
  total in-flight work (see [backpressure](#backpressure)) but does nothing
  per-client, and deliberately so: that is a different layer.

A minimal nginx sidecar, as a worked example:

```nginx
server {
  listen 443 ssl;
  server_name nidus.internal;

  ssl_certificate     /etc/tls/tls.crt;
  ssl_certificate_key /etc/tls/tls.key;

  # nidus itself is bound to loopback and is unreachable from outside this host.
  location / {
    proxy_pass http://127.0.0.1:7700;
    proxy_read_timeout 600s;   # match --write-timeout: a large upsert is slow
  }
}
```

On Kubernetes, the equivalent is an `Ingress` with a TLS secret in front of the
chart's `Service`; see [Ingress and TLS](/guides/kubernetes/#ingress-and-tls) in
the Kubernetes guide.

### What the token model is, and is not

One static shared secret, and nothing more. Specifically:

- **No rotation.** Changing the token means restarting the instance. In a cluster,
  restarting every instance.
- **No scoping.** Any valid token can read every collection, write, compact, and
  call `/refresh`. There are no per-collection or read-only credentials.
- **No user model.** There is no identity attached to a request, so there is no
  per-caller audit trail beyond the access log's request id.

That is a deliberate boundary for a store of this size, not an oversight. If you
need scoped or rotatable credentials, issue them at the proxy and let it present the
single nidus token upstream.

## Request size

Each request body is buffered in memory, so the body-size limit is also the
largest single upsert. It defaults to 256 MiB; raise or lower it with
`--max-body-bytes <n>`. A body over the limit gets `413 Payload Too Large`.

## Backpressure

The body limit bounds how big one request can be. Two more flags bound how *many*
and how *long*.

| Flag | Default | What it does |
| --- | --- | --- |
| `--max-concurrent-requests <n>` | `0` (auto) | Cap on store-touching requests in flight. Past it, requests are **shed** with `503`. Auto is 8× CPU cores, floored at 64. |
| `--read-timeout <seconds>` | `30` | Deadline for a read (search, list, stats, a plain recall). `0` disables. |
| `--write-timeout <seconds>` | `600` | Deadline for a mutation (upsert, delete, compact, a `reinforce` recall). `0` disables. |
| `--body-idle-timeout <seconds>` | `15` | Abandon a request body that stops delivering data. `0` disables. |

**A shed request is a retryable `503`.** It carries `Retry-After: 1` and a body of
`{"error": …, "retryable": true}`. Nothing was attempted and the store is untouched,
so retrying after a brief backoff is the correct client behaviour: this is the
server saying "not right now", not "something went wrong".

The cap exists because the working set is in RAM: without it, in-flight request
bodies accumulate and compete for memory with the data itself, and the server
degrades until the allocator gives out rather than ever saying no. The auto default
is a small multiple of core count because search is CPU-bound brute force:
admitting far more concurrent scans than cores buys no throughput and costs memory.

**A request that outlives its deadline gets `504`**, with `"retryable": false`. The
distinction from `503` matters: a `504` means the work *was* admitted and may still
be running, so an immediate retry piles a second copy onto an instance that is
already behind.

**A deadline stops the work, not just the client.** When it fires, the caller gets
its `504` *and* the running scan is asked to stop: the scan kernels check a
cancellation flag every few thousand rows and bail out. Finishing a scan nobody is
waiting for is the worst possible use of a core under load. Cancellation is
cooperative, so it is prompt rather than instant: whatever chunk was in progress
completes. A per-row check would tax every query to spare the rare abandoned one.

**Read and write deadlines differ by design.** A search is milliseconds; a large
upsert legitimately runs for minutes under one write lock. One bound tight enough
for the first would abort the second mid-batch.

The probe endpoints (`/health`, `/ready`, `/metrics`) are **never** shed and never
time out. They take no store lock, so they cost nothing to admit, and shedding a
liveness probe under load would get a busy-but-healthy instance restarted, which is
the opposite of what you want when the server is saturated.

### Slow clients

A request is handled in two phases: its body is **received first**, and only then
does it take a concurrency permit to touch the store. That split is why a client
which sends headers and then goes quiet cannot stall your search traffic: it never
holds a store permit at all, however long it sits there.

Body reception has its own, larger pool (four times `--max-concurrent-requests`),
so an unbounded number of bodies still cannot accumulate in RAM: that is the memory
bound the concurrency cap exists for, kept in the phase that actually consumes the
memory.

`--body-idle-timeout` then bounds how long a single stalled body may occupy one of
those slots. It is an **idle** bound, not a total one: the clock resets on every
chunk, so a 256 MiB upsert over a slow link is never cut off however long it takes,
while a silent client dies in seconds. (Same semantic as nginx's
`client_body_timeout`.) Setting it to `0` removes that bound.

An oversized body is rejected with `413` before it is read, when the client sends a
`Content-Length`. A stalled one surfaces as `413` too, since the two are
indistinguishable at that point.

## Concurrency & durability

The server holds the store behind a read/write lock and runs each operation on a
blocking worker, the same pattern the library recommends for [driving it from
async code](/guides/integrating/). Reads (`/search`, `/list`, `/stats`, the `GET`
endpoints) run concurrently; writes take the store exclusively. Durability is
exactly the library's: each write batch is fsync'd before its response returns, so
a `200` means the data is on disk. The storage model and search semantics are
identical to the library: the server adds nothing and hides nothing.

You can take a hot [backup](/guides/cli-and-server/#backup-restore--verify) of a store
while `nidus serve` is running: `nidus backup` does not take the writer lock.

### Getting write throughput

Two client-side choices dominate ingest speed, and both are free:

**Batch your upserts.** Each `/collections/{name}/upsert` call pays a fixed cost
(a round trip and an fsync) no matter how many records it carries. Sending one
record per request is roughly *two hundred times* slower per vector than sending a
thousand:

| records per request | vectors/s (one client, 384-d) |
| --- | --- |
| 1 | ~130 |
| 10 | ~1k |
| 100 | ~9k |
| 1000 | ~33k |

**Use more than one connection.** A single client posting batches back-to-back
spends most of its time waiting: encode, send, decode, store, reply, all in series.
Several concurrent writers keep those stages overlapped, and one `nidus serve`
absorbs them: throughput roughly doubles at two clients and plateaus around
2.5–3.5× by four to eight, at which point the store's exclusive write lock is the
limit and more clients add nothing.

Both figures come from `just bench-write` on a development machine; treat them as
orders of magnitude, not promises. The shape is what matters: batch size first,
concurrency second.

## Metrics and logs

### `GET /metrics`

Prometheus text exposition, served without a credential (a scraper that got a `401`
would report the target as down). It reports:

- **Traffic**: `nidus_http_requests_total{route,status}` and
  `nidus_http_request_duration_seconds` (a histogram), by route; plus
  `nidus_http_requests_shed_total`, `nidus_http_requests_timed_out_total`,
  `nidus_http_requests_cancelled_total` (clients that disconnected before a
  response, invisible in a status breakdown, which only ever sees requests that
  finished), `nidus_http_requests_in_flight`, and `nidus_http_concurrency_limit`.
  See [the two in-flight gauges](#the-two-in-flight-gauges) for which is which.
- **Search path**: `nidus_search_queries_total` split by how each query was served
  (`nidus_search_ann_total`, `nidus_search_segmented_total`,
  `nidus_search_quantized_total`, `nidus_search_exact_total`), plus
  `nidus_search_vectors_scanned_total` and `nidus_search_reranked_total`. This is
  the difference between "queries are slow" and "queries are slow *because the
  index is not being used*".
- **Lease and fencing**: `nidus_lease_renew_attempts_total` and its outcomes,
  including the transient-vs-definitive split (`nidus_lease_renew_transient_failures_total`
  vs `nidus_lease_renew_lost_total`). A rising transient count is an object store
  misbehaving, visible long before anything actually breaks.
- **Backend health**: `nidus_backend_retries_total`, `nidus_refresh_failures_total`.
- **Instance state**: `nidus_ready`, `nidus_writer_fenced`, `nidus_staleness_seconds`.
- **Write path**: `nidus_write_batches_total` (batches that needed a durable barrier) against
  `nidus_durability_barriers_total` (barriers actually taken). See
  [group commit](#group-commit) below.

Route labels are **templates**: `/collections/{name}/upsert`, never the collection
name. That bounds the label cardinality, and it means the scrape exposes traffic
shape but not what is stored. Like every other endpoint, it belongs on your private
network; see [Securing a deployment](#securing-a-deployment).

Reading it takes no store lock, so a scrape answers instantly even during a
multi-minute upsert: the endpoints you consult during an incident must not be the
ones the incident blocks.

### Group commit

Concurrent writes share one disk barrier instead of each taking its own: the first request to
reach the store applies every write queued alongside it under one lock, one fsync covers them
all, and each request is answered only after that fsync succeeds. A `200` still means the
bytes are on disk. See [how it works](/guides/how-it-works/#group-commit) for the mechanism.

| Metric | Counts |
| --- | --- |
| `nidus_write_groups_total` | Groups committed, one shared barrier each. |
| `nidus_write_group_members_total` | Writes applied inside those groups. |
| `nidus_write_queue_depth` | Writes submitted and not yet applied: the current write backlog. |

Divide the second by the first for the **coalescing factor**: the average number of writes
that shared a barrier. `1.0` is not a fault: it means writes on this instance never overlap,
so there was never a group to form. Nothing waits for one, so a single writer is exactly as
fast as it would be without any of this.

The factor rises with write concurrency, which is where it matters: measured on one
developer machine, eight concurrent HTTP writers at 384 dimensions went from 85k to 134k
vectors/s at 3.0 writes per barrier.

### The two in-flight gauges

They count different things, and mixing them up will mislead you during an incident:

| Metric | Counts |
| --- | --- |
| `nidus_http_requests_in_flight` | Requests being handled right now, **probes included**. Drops when a client disconnects mid-request. |
| `nidus_http_admitted_in_flight` | **Concurrency permits held**: store-touching requests that passed admission control. |

Graph the second one against `nidus_http_concurrency_limit`: requests are shed with
a `503` exactly when it reaches the limit, so the two together explain every entry
in `nidus_http_requests_shed_total`. The first is the one to watch for clients
hanging up, alongside `nidus_http_requests_cancelled_total`.

`nidus_http_admitted_in_flight` reports the admission decision, not a headcount of
running work. When a request hits its `--read-timeout` / `--write-timeout` the
permit is released at once (continuing to hold it would shed live traffic on
behalf of a response nobody is waiting for), while the scan itself keeps going
until it notices the cancellation signal. During that window the gauge reads low.
The scan kernels check every few thousand rows, so the window is milliseconds, and
it is entered exactly `nidus_http_requests_timed_out_total` times: if that counter
is flat, it has never happened on your instance.

### Logs

Every diagnostic is one `key=value` line on stderr:

```text
ts=2026-07-25T18:04:11.482Z level=info target=http msg=request id=1a2b-7f method=POST route=/search status=200 duration_ms=3.418
```

`NIDUS_LOG` sets the threshold: `error`, `warn`, `info` (the default), `debug`,
`trace`, or `off`. `NIDUS_LOG=error` silences the per-request access log while
keeping failures; `NIDUS_LOG=debug` turns on lease tracing (the old
`NIDUS_LEASE_DEBUG=1` still works and now means the same thing).

Every request carries an `id`. nidus honours an inbound `X-Request-Id` header when
you send one and mints its own otherwise, and echoes it on the response, so the id
in your client's logs is the id to grep for in the server's.

## Configuration from the environment

Every `nidus serve` flag also reads from a matching `NIDUS_*` environment variable,
so the server can be configured without a command line at all, the natural fit for
a container or an orchestrator. An explicit flag always wins over the variable.

This table is the operator-facing surface: every variable, grouped by what it
governs, with its flag and default. The [CLI reference](/reference/cli/) is
canonical for **per-flag syntax** (value types, allowed strings, and how each flag
notes its own env binding); come here for the full picture and go there for the
detail on one flag.

### Store and open

| Variable | Flag | What it does | Default |
| --- | --- | --- | --- |
| `NIDUS_DIR` | `--dir` | Store directory (created on first write; unused, but still required, with an object store) | (required) |
| `NIDUS_DIM` | `--dim` | Embedding dimension. Required to create a store, unless `--embed-provider` supplies one | inferred from an existing store |
| `NIDUS_DISTANCE` | `--distance` | `cosine` \| `euclidean` \| `dot` | `cosine` (on create) |
| `NIDUS_PERSISTENCE` | `--persistence` | Where durable bytes live: `s3://…`, `gs://…`, or a local path | local files under `--dir` |
| `NIDUS_MEMORY` | `--memory` | Shared in-RAM working set: `redis://…` (or `valkey://…`, `keydb://…`, `dragonfly://…`) | process-local |
| `NIDUS_FSYNC` | `--fsync` | `per-batch` (durable per call) or `on-flush` (faster, weaker) | `per-batch` |
| `NIDUS_MMAP` | `--mmap` | Memory-map immutable segments instead of holding them in RAM | off |
| `NIDUS_NO_MMAP` | `--no-mmap` | Force mmap off, overriding a recorded `configure --mmap` default or a `NIDUS_MMAP` set in a shared env block | off |
| `NIDUS_QUERY_THREADS` | `--query-threads` | Worker threads splitting one query's scan (unrelated to serving concurrency) | `1` (serial) |
| `NIDUS_MAX_VECTOR_BYTES` | `--max-vector-bytes` | Refuse to open a store whose vector matrix would exceed this many bytes | no ceiling |
| `NIDUS_SEGMENT_MAX_ROWS` | `--segment-max-rows` | Seal the active segment once it reaches this many rows | never seal (one growing segment) |
| `NIDUS_SEGMENT_INDEX_MIN_ROWS` | `--segment-index-min-rows` | Minimum rows for a sealed segment to get its own IVF index | never index (exact brute-force) |
| `NIDUS_AUTO_COMPACT` | `--auto-compact` | Rewrite the data matrix once this fraction of rows is dead | `0.5` |
| `NIDUS_NO_AUTO_COMPACT` | `--no-auto-compact` | Never auto-compact; reclaim dead rows only on an explicit `compact` | off |

### Server, bind, and auth

| Variable | Flag | What it does | Default |
| --- | --- | --- | --- |
| `NIDUS_ADDR` | `--addr` | Bind address | `127.0.0.1:7700` |
| `NIDUS_TOKEN` | `--token` | Bearer token for auth | none (unauthenticated) |
| `NIDUS_READ_ONLY` | `--read-only` | Open without taking the writer lock; rejects mutations | off |

### Limits and backpressure

See [Backpressure](#backpressure) above for how these interact.

| Variable | Flag | What it does | Default |
| --- | --- | --- | --- |
| `NIDUS_MAX_BODY_BYTES` | `--max-body-bytes` | Request/upsert size limit; a body over it gets `413` | 256 MiB |
| `NIDUS_MAX_CONCURRENT_REQUESTS` | `--max-concurrent-requests` | In-flight cap; past it, requests are shed with `503` | `0` (auto: 8× CPU cores, floored at 64) |
| `NIDUS_READ_TIMEOUT` | `--read-timeout` | Read deadline in seconds (search, list, stats, a plain recall); `0` disables | `30` |
| `NIDUS_WRITE_TIMEOUT` | `--write-timeout` | Write deadline in seconds (upsert, delete, compact, a `reinforce` recall); `0` disables | `600` |
| `NIDUS_BODY_IDLE_TIMEOUT` | `--body-idle-timeout` | Abandon a request body that stops delivering data; `0` disables | `15` |

### Cluster and lease

See the [Kubernetes guide](/guides/kubernetes/) for running several instances.

| Variable | Flag | What it does | Default |
| --- | --- | --- | --- |
| `NIDUS_CLUSTER` | `--cluster` | Run as one of several cooperating instances over a shared object-store `--persistence` and Redis-family `--memory` tier | off |
| `NIDUS_NO_CLUSTER` | `--no-cluster` | Run standalone: the explicit off for `--cluster`, and it wins over a `NIDUS_CLUSTER` set in a shared env block | off |
| `NIDUS_LOCK_TTL` | `--lock-ttl` | Seconds before another process may reclaim a stale writer lock (also the writer-lease window in `--cluster` mode) | `60` |
| `NIDUS_WAIT_FOR_LEASE` | `--wait-for-lease` | Wait as a standby for the writer handle instead of exiting; bare flag means forever | unset (exit immediately if held) |
| `NIDUS_REQUIRE_REMOTE` | `--require-remote` | Refuse to start unless persistence and memory are both remote | off |
| `NIDUS_MAX_STALENESS` | `--max-staleness` | Fail the readiness probe once a `--read-only` instance has gone this many seconds without verifying it is current | no bound |
| `NIDUS_REFRESH_INTERVAL` | `--refresh-interval` | Auto-refresh this instance every N seconds instead of relying on `POST /refresh` | no auto-refresh |

### ANN (approximate search)

| Variable | Flag | What it does | Default |
| --- | --- | --- | --- |
| `NIDUS_ANN` | `--ann` | `hnsw` or `ivf`; omit for exact brute-force | none (exact) |
| `NIDUS_ANN_M` | `--ann-m` | HNSW: max neighbours per node above layer 0 | `16` |
| `NIDUS_ANN_EF_CONSTRUCTION` | `--ann-ef-construction` | HNSW: build-time beam width | `200` |
| `NIDUS_ANN_EF_SEARCH` | `--ann-ef-search` | HNSW: search-time beam width | `64` |
| `NIDUS_ANN_N_LISTS` | `--ann-n-lists` | IVF: number of k-means lists (`0` = auto `~sqrt(n)`) | `0` (auto) |
| `NIDUS_ANN_N_PROBE` | `--ann-n-probe` | IVF: lists probed per query | `8` |
| `NIDUS_ANN_OVERSCAN` | `--ann-overscan` | Candidate over-fetch multiple (`top_k * overscan`) before post-filter and rerank; both kinds | `4` |
| `NIDUS_ANN_SEED` | `--ann-seed` | Build PRNG seed for a deterministic index; both kinds | fixed default seed |

### Quantization

| Variable | Flag | What it does | Default |
| --- | --- | --- | --- |
| `NIDUS_QUANTIZATION` | `--quantization` | Quantize the first search pass: `int8` (4× less memory traffic) or `binary` (32×, cosine only), reranked in exact f32 | none (exact-only) |
| `NIDUS_QUANT_RESCORE` | `--quant-rescore` | Candidate over-fetch multiple for the quantized first pass, reranked in f32 | `4` (int8) / `16` (binary) |

### Embed

`memory` feature only (folded into `serve`); see [Text-native ingest](#text-native-ingest) above.

| Variable | Flag | What it does | Default |
| --- | --- | --- | --- |
| `NIDUS_EMBED_PROVIDER` | `--embed-provider` | `voyage`, `openai`, `ollama`, `cohere`, `gemini`, `mistral`, `jina`, or `openai-compat`. Omit to serve only the raw vector endpoints | none |
| `NIDUS_EMBED_MODEL` | `--embed-model` | Embedding model | provider's default (`openai-compat` has none; pass one) |
| `NIDUS_EMBED_API_KEY` | `--embed-api-key` | API key for the embedding provider (some, e.g. Ollama, need none) | none |
| `NIDUS_EMBED_BASE_URL` | `--embed-base-url` | Base-URL override (required for `openai-compat` and self-hosted gateways) | provider's default endpoint |
| `NIDUS_EMBED_DIMENSION` | `--embed-dimension` | Non-native embedding width: Voyage Matryoshka models (256, 512, 1024, 2048), or OpenAI `text-embedding-3-small`/`-large` (any width up to 1536/3072) | the model's native width |
| `NIDUS_STRICT_EMBEDDER_IDENTITY` | `--strict-embedder-identity` | Refuse a recall against a collection with no pinned embedder identity, instead of warning about it | off (warn) |

### Summarize

`memory` + `summarize` features (both folded into `serve`); enables `mode: "summarize"` on `/remember`.

| Variable | Flag | What it does | Default |
| --- | --- | --- | --- |
| `NIDUS_SUMMARIZE_PROVIDER` | `--summarize-provider` | `anthropic` or `openai`. Omit for raw-embed only | none |
| `NIDUS_SUMMARIZE_MODEL` | `--summarize-model` | Summarizer model | provider's default |
| `NIDUS_SUMMARIZE_API_KEY` | `--summarize-api-key` | API key for the summarizer provider | none |
| `NIDUS_SUMMARIZE_BASE_URL` | `--summarize-base-url` | Base-URL override | provider's default endpoint |

### Other

`NIDUS_LOG` sets the [log level](#logs) (`error` \| `warn` \| `info` \| `debug` \|
`trace` \| `off`, default `info`); it is read directly rather than bound through
clap, so it has no `--flag` form. The legacy `NIDUS_LEASE_DEBUG=1` still works and
now means `NIDUS_LOG=debug`.

Cloud credentials come from the [standard environment](/guides/storage-backends/)
for each backend (`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`, …).

## Running in a container

The published [`duckedup/nidus`](https://hub.docker.com/r/duckedup/nidus) image runs
`nidus serve` configured entirely from the environment. It is built for **shared,
non-local backends** (object-store persistence plus a Redis-family memory tier)
because a container has no durable local disk: a local-file or process-RAM store
would lose its data on every restart. The image bakes in `NIDUS_REQUIRE_REMOTE=true`,
so it fails fast with a clear message rather than start a store it cannot persist.

```bash
docker run --rm -p 7700:7700 \
  -e NIDUS_DIM=768 \
  -e NIDUS_PERSISTENCE=s3://my-bucket/store \
  -e NIDUS_MEMORY=redis://my-redis:6379 \
  -e NIDUS_TOKEN="$NIDUS_TOKEN" \
  -e AWS_ACCESS_KEY_ID=… -e AWS_SECRET_ACCESS_KEY=… -e AWS_REGION=… \
  duckedup/nidus:latest
```

The image binds `0.0.0.0:7700` and exposes the unauthenticated `GET /health` and
`GET /ready` for liveness/readiness probes. It handles `SIGTERM` (the signal an
orchestrator sends to stop a container): on stop it flushes, persists the ANN and
full-text caches so the next start is warm, and releases the writer lock, so a
replacement instance re-acquires it immediately instead of waiting out the lock TTL. Set a `NIDUS_TOKEN` whenever the port is reachable beyond localhost,
and read [Securing a deployment](#securing-a-deployment) first: a `0.0.0.0` bind is
plain HTTP, so the token and the data both cross the network in cleartext unless
something in front of the container terminates TLS.

## API reference

Every store operation is an HTTP route: `GET /stats`, `POST /search`,
`POST /collections/{name}/upsert`, and so on. The full route-by-route reference,
with a JSON body and a curl example for each, plus the error codes, is the
[**HTTP API**](/reference/http-api/) page.
