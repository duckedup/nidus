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
var). Every request except `GET /health` must then carry
`Authorization: Bearer <secret>`; anything else gets `401`.

```bash
nidus serve --dir ./store --addr 0.0.0.0:7700 --token "$NIDUS_TOKEN"

curl -s localhost:7700/stats -H "authorization: Bearer $NIDUS_TOKEN"
```

There is no TLS and no user model: nidus is one store behind one optional bearer
token. For anything beyond a trusted network, terminate TLS and add access control
at a reverse proxy in front of it.

## Request size

Each request body is buffered in memory, so the body-size limit is also the
largest single upsert. It defaults to 256 MiB; raise or lower it with
`--max-body-bytes <n>`. A body over the limit gets `413 Payload Too Large`.

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

The image binds `0.0.0.0:7700` and exposes the unauthenticated `GET /health` for a
readiness/liveness probe. It handles `SIGTERM` (the signal an orchestrator sends to
stop a container): on stop it flushes and releases the writer lock, so a replacement
instance re-acquires it immediately instead of waiting out the lock TTL. Set a
`NIDUS_TOKEN` whenever the port is reachable beyond localhost.

## API reference

Every store operation is an HTTP route — `GET /stats`, `POST /search`,
`POST /collections/{name}/upsert`, and so on. The full route-by-route reference,
with a JSON body and a curl example for each, plus the error codes, is the
[**HTTP API**](/reference/http-api/) page.
