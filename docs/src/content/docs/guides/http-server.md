---
title: HTTP server
description: Run nidus as an HTTP server with `nidus serve` — start it, authenticate it, and drive a store over JSON with no Rust toolchain on the client.
---

`nidus serve` opens one store and exposes it over HTTP. Every library operation
has an endpoint, so a client that never links the crate can do the full job over
the network: create collections, upsert vectors, search, filter, inspect, and
maintain the store — all in JSON. The wire format is the same store directory the
[library](/guides/integrating/) and the [CLI](/guides/cli-and-server/) read and
write; the server is just another door into it.

nidus stores and searches the vectors you give it — it does not generate
embeddings. You compute embeddings with your own model (in any language, on the
client), then send the resulting vectors here to upsert and query.

This page covers running the server. For the route-by-route reference, see the
[HTTP API](/reference/http-api/); for driving a store from your shell, see the
[command-line guide](/guides/cli-and-server/).

## Start the server

```bash
# Create the store on first run by passing --dim; afterwards it's inferred.
nidus serve --dir ./store --dim 768 --addr 127.0.0.1:7700
```

`nidus serve` prints its bind address and serves until you stop it with Ctrl-C,
flushing to disk on the way out. The store directory need not exist yet — the
first write creates it — but `--dim` is required until it does, because the
embedding dimension is pinned at creation.

Pass `--read-only` to serve without taking the writer lock: a search-only process
that can run beside a separate writer.

To serve approximate (ANN) search, add `--ann hnsw` or `--ann ivf` (with the
optional `--ann-*` knobs from the [command-line guide](/guides/cli-and-server/)).
The index lives in memory for the life of the process; `GET /stats` reports the
active configuration.

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

# 2. Upsert records — id + vector + any typed metadata.
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
# → {"dimension":3,"distance":"Cosine","collections":["docs"],"footprint":{…}}
```

That is a complete vector store over the network: no Rust toolchain on the client,
nothing but HTTP and JSON.

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

**nidus serves plain HTTP, by design.** There is no `--tls` flag and no certificate
configuration, and that is a decision rather than an omission: in every deployment
where nidus is reachable off-box there is already an ingress, sidecar, or service
mesh that terminates TLS — and it does that job better than a TLS stack compiled
into a vector store would, with rotation, SNI, and modern cipher policy handled by
infrastructure you are already operating.

What that means in practice, stated plainly so you can decide with it:

- **`--token` authenticates a caller. It does not confer confidentiality.** Over
  plain HTTP the token itself crosses the network in cleartext on every request,
  alongside every vector, document id, and metadata value. Anyone who can observe
  the traffic has both the data and the credential.
- **Terminate TLS in front of nidus.** Bind nidus to loopback or to a pod-local
  address, and let the proxy be the only thing that accepts outside connections.
- **The bind address is the real perimeter.** `--addr 0.0.0.0:7700` with no
  `--token` is an unauthenticated, world-readable *and writable* vector store.
  nidus warns at startup when it binds a non-loopback address — with a token
  ("the credential crosses the network in cleartext") and without one ("this store
  is open to anyone who can reach it"). It warns and starts; it never refuses,
  because refusing would break the proxy-terminated architecture recommended here.
- **Rate limiting belongs at the proxy.** nidus deliberately does not do per-IP
  rate limiting — see [backpressure](#backpressure) for what it *does* bound, and
  why the two are different layers.

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
chart's `Service` — see the [Kubernetes guide](/guides/kubernetes/).

### What the token model is, and is not

One static shared secret, and nothing more. Specifically:

- **No rotation.** Changing the token means restarting the instance. In a cluster,
  restarting every instance.
- **No scoping.** Any valid token can read every collection, write, compact, and
  call `/refresh`. There are no per-collection or read-only credentials.
- **No user model.** There is no identity attached to a request, so there is no
  per-caller audit trail beyond the access log's request id.

That is a deliberate boundary for a store of this size, not an oversight — but it
is a boundary. If you need scoped or rotatable credentials, issue them at the proxy
and let it present the single nidus token upstream.

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
| `--read-timeout <seconds>` | `30` | Deadline for a read (search, list, stats). `0` disables. |
| `--write-timeout <seconds>` | `600` | Deadline for a mutation (upsert, delete, compact). `0` disables. |

**A shed request is a retryable `503`.** It carries `Retry-After: 1` and a body of
`{"error": …, "retryable": true}`. Nothing was attempted and the store is untouched,
so retrying after a brief backoff is the correct client behaviour — this is the
server saying "not right now", not "something went wrong".

The cap exists because the working set is in RAM: without it, in-flight request
bodies accumulate and compete for memory with the data itself, and the server
degrades until the allocator gives out rather than ever saying no. The auto default
is a small multiple of core count because search is CPU-bound brute force —
admitting far more concurrent scans than cores buys no throughput and costs memory.

**A request that outlives its deadline gets `504`**, with `"retryable": false`. The
distinction from `503` matters: a `504` means the work *was* admitted and may still
be running, so an immediate retry piles a second copy onto an instance that is
already behind.

Two limits of the timeouts, stated because it would be easy to assume otherwise:

- **A timeout frees the client, not the CPU.** When the deadline fires the caller
  gets its `504`, but a scan already running finishes anyway — nidus has no
  cooperative cancellation inside the scan loop. Abandoned work still costs a full
  scan.
- **Read and write deadlines differ by design.** A search is milliseconds; a large
  upsert legitimately runs for minutes under one write lock. One bound tight enough
  for the first would abort the second mid-batch.

The probe endpoints (`/health`, `/ready`, `/metrics`) are **never** shed and never
time out. They take no store lock, so they cost nothing to admit — and shedding a
liveness probe under load would get a busy-but-healthy instance restarted, which is
the opposite of what you want when the server is saturated.

## Concurrency & durability

The server holds the store behind a read/write lock and runs each operation on a
blocking worker — the same pattern the library recommends for [driving it from
async code](/guides/integrating/). Reads (`/search`, `/list`, `/stats`, the `GET`
endpoints) run concurrently; writes take the store exclusively. Durability is
exactly the library's: each write batch is fsync'd before its response returns, so
a `200` means the data is on disk. The storage model and search semantics are
identical to the library — the server adds nothing and hides nothing.

You can take a hot [backup](/guides/cli-and-server/#backup--restore) of a store
while `nidus serve` is running: `nidus backup` does not take the writer lock.

## Metrics and logs

### `GET /metrics`

Prometheus text exposition, served without a credential (a scraper that got a `401`
would report the target as down). It reports:

- **Traffic** — `nidus_http_requests_total{route,status}` and
  `nidus_http_request_duration_seconds` (a histogram), by route; plus
  `nidus_http_requests_shed_total`, `nidus_http_requests_timed_out_total`,
  `nidus_http_requests_cancelled_total` (clients that disconnected before a
  response — invisible in a status breakdown, which only ever sees requests that
  finished), `nidus_http_requests_in_flight`, and `nidus_http_concurrency_limit`.
- **Search path** — `nidus_search_queries_total` split by how each query was served
  (`nidus_search_ann_total`, `nidus_search_segmented_total`,
  `nidus_search_quantized_total`, `nidus_search_exact_total`), plus
  `nidus_search_vectors_scanned_total` and `nidus_search_reranked_total`. This is
  the difference between "queries are slow" and "queries are slow *because the
  index is not being used*".
- **Lease and fencing** — `nidus_lease_renew_attempts_total` and its outcomes,
  including the transient-vs-definitive split (`nidus_lease_renew_transient_failures_total`
  vs `nidus_lease_renew_lost_total`). A rising transient count is an object store
  misbehaving, visible long before anything actually breaks.
- **Backend health** — `nidus_backend_retries_total`, `nidus_refresh_failures_total`.
- **Instance state** — `nidus_ready`, `nidus_writer_fenced`, `nidus_staleness_seconds`.

Route labels are **templates**: `/collections/{name}/upsert`, never the collection
name. That bounds the label cardinality, and it means the scrape exposes traffic
shape but not what is stored. It is still traffic shape, so put `/metrics` on a
scrape-only path rather than the public ingress.

Reading it takes no store lock, so a scrape answers instantly even during a
multi-minute upsert — the endpoints you consult during an incident must not be the
ones the incident blocks.

### Logs

Every diagnostic is one `key=value` line on stderr:

```text
ts=2026-07-25T18:04:11.482Z level=info target=http msg=request id=1a2b-7f method=POST route=/search status=200 duration_ms=3.418
```

`NIDUS_LOG` sets the threshold — `error`, `warn`, `info` (the default), `debug`,
`trace`, or `off`. `NIDUS_LOG=error` silences the per-request access log while
keeping failures; `NIDUS_LOG=debug` turns on lease tracing (the old
`NIDUS_LEASE_DEBUG=1` still works and now means the same thing).

Every request carries an `id`. nidus honours an inbound `X-Request-Id` header when
you send one and mints its own otherwise, and echoes it on the response — so the id
in your client's logs is the id to grep for in the server's.

## Configuration from the environment

Every `nidus serve` flag also reads from a matching `NIDUS_*` environment variable,
so the server can be configured without a command line at all — the natural fit for
a container or an orchestrator. An explicit flag always wins over the variable.

| Variable | Flag | Purpose |
| --- | --- | --- |
| `NIDUS_DIR` | `--dir` | Store directory (unused, but still required, with an object store) |
| `NIDUS_DIM` | `--dim` | Embedding dimension (required to create a store) |
| `NIDUS_DISTANCE` | `--distance` | `cosine` \| `euclidean` \| `dot` |
| `NIDUS_PERSISTENCE` | `--persistence` | Where durable bytes live — `s3://…`, `gs://…`, or a local path |
| `NIDUS_MEMORY` | `--memory` | Shared working set — `redis://…` (or `valkey://…`, …) |
| `NIDUS_ADDR` | `--addr` | Bind address (default `127.0.0.1:7700`) |
| `NIDUS_TOKEN` | `--token` | Bearer token for auth |
| `NIDUS_MAX_BODY_BYTES` | `--max-body-bytes` | Request/upsert size limit |
| `NIDUS_MAX_CONCURRENT_REQUESTS` | `--max-concurrent-requests` | In-flight cap; past it, `503` (`0` = auto) |
| `NIDUS_READ_TIMEOUT` | `--read-timeout` | Read deadline in seconds (`0` = none) |
| `NIDUS_WRITE_TIMEOUT` | `--write-timeout` | Write deadline in seconds (`0` = none) |
| `NIDUS_LOG` | — | Log level: `error` \| `warn` \| `info` \| `debug` \| `trace` \| `off` |
| `NIDUS_READ_ONLY` | `--read-only` | Serve without the writer lock |
| `NIDUS_ANN`, `NIDUS_ANN_*` | `--ann`, `--ann-*` | Approximate-index selection and tuning |
| `NIDUS_REQUIRE_REMOTE` | `--require-remote` | Refuse to start on a local-only store (see below) |

Cloud credentials come from the [standard environment](/guides/storage-backends/)
for each backend (`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`, …).

## Running in a container

The published [`duckedup/nidus`](https://hub.docker.com/r/duckedup/nidus) image runs
`nidus serve` configured entirely from the environment. It is built for **shared,
non-local backends** — object-store persistence plus a Redis-family memory tier —
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
orchestrator sends to stop a container): on stop it flushes and releases the writer
lock, so a replacement instance re-acquires it immediately instead of waiting out
the lock TTL. Set a `NIDUS_TOKEN` whenever the port is reachable beyond localhost —
and read [Securing a deployment](#securing-a-deployment) first: a `0.0.0.0` bind is
plain HTTP, so the token and the data both cross the network in cleartext unless
something in front of the container terminates TLS.

## API reference

Every store operation is an HTTP route — `GET /stats`, `POST /search`,
`POST /collections/{name}/upsert`, and so on. The full route-by-route reference,
with a JSON body and a curl example for each, plus the error codes, is the
[**HTTP API**](/reference/http-api/) page.
