---
title: HTTP server & API
description: Run nidus as an HTTP server and drive the whole store — create, upsert, search, inspect — over JSON, with no Rust toolchain on the client.
---

`nidus serve` opens one store and exposes it over HTTP. Every library operation
has an endpoint, so a client that never links the crate can do the full job over
the network: create collections, upsert vectors, search, filter, inspect, and
maintain the store — all in JSON. The wire format is the same store directory the
[library](/guides/integrating/) and the [CLI](/guides/cli-and-server/) read and
write; the server is just another door into it.

nidus stores and searches the vectors you give it — it does not generate
embeddings. You compute embeddings with your own model (in any language, on the
client), then send the resulting vectors here to upsert and query. "Interact
fully over the network" means exactly that: manage and search precomputed vectors
over HTTP, with no Rust toolchain and no embedding model on the nidus side.

This page is the network reference. For driving a store from your shell or a cron
job, see the [command-line guide](/guides/cli-and-server/).

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

## Endpoint reference

Every endpoint maps one-to-one onto a library method. Errors return
`{"error": "<message>"}` with a status that distinguishes a client fault from a
server fault (see [Errors](#errors)).

| Method & path | Operation | Library method |
| --- | --- | --- |
| `GET /health` | liveness check (always unauthenticated) | — |
| `GET /stats` | dimension, distance, ann config, collections, footprint | `dimension` / `footprint` |
| `GET /collections` | list collection names | `collections` |
| `POST /collections/{name}` | create a collection | `create_collection` |
| `DELETE /collections/{name}` | drop a collection and its records | `drop_collection` |
| `GET /collections/{name}/meta` | read collection metadata | `get_meta` |
| `PUT /collections/{name}/meta` | replace collection metadata | `set_meta` |
| `POST /collections/{name}/upsert` | insert or overwrite records | `upsert` |
| `POST /collections/{name}/delete` | delete by ids or by filter | `delete` / `delete_where` |
| `GET /collections/{name}/records` | every record in a collection | `get_all` |
| `POST /collections/{name}/fts-schema` | declare full-text-indexed fields | `set_fts_schema` |
| `POST /search` | nearest-neighbour search | `search` |
| `POST /text-search` | BM25 full-text search | `text_search` |
| `POST /hybrid-search` | fused vector + BM25 (RRF) | `hybrid_search` |
| `POST /list` | metadata-only query (no vector) | `list` |
| `POST /flush` | flush buffered writes to disk | `flush` |
| `POST /compact` | reclaim dead rows and superseded log records | `compact` |

### `GET /health`

Liveness probe. Returns `200` with the body `ok`. Always reachable without a
token, so a load balancer or `docker healthcheck` needs no credential.

### `GET /stats`

Store-wide introspection — the network equivalent of `nidus stats`.

```json
{
  "dimension": 768,
  "distance": "Cosine",
  "ann": null,
  "collections": ["docs", "notes"],
  "footprint": {
    "rows": 1240,
    "dead_rows": 12,
    "dimension": 768,
    "vector_bytes": 3809280,
    "doc_count": 1228
  }
}
```

`rows` counts every vector slot on disk (including superseded ones); `dead_rows`
is how many a `compact` would reclaim; `doc_count` is the live record count.
`ann` is `null` for exact brute-force search (the default), or echoes the active
ANN configuration when the server was started with `--ann hnsw`/`--ann ivf` (only
the knobs that apply to the chosen index are reported):

```json
"ann": { "kind": "Hnsw", "overscan": 4, "seed": 11400714819323198485,
         "m": 16, "ef_construction": 200, "ef_search": 64 }
```

### `GET /collections`

Returns the collection names as a JSON array: `["docs", "notes"]`.

### `POST /collections/{name}` · `DELETE /collections/{name}`

Create or drop a collection. The body is ignored.

```bash
curl -s -X POST   localhost:7700/collections/docs   # → {"created": "docs"}
curl -s -X DELETE localhost:7700/collections/docs   # → {"dropped": "docs"}
```

Upsert auto-creates a collection, so an explicit create is only needed to register
an empty one (e.g. to attach metadata before any records land).

### `GET /collections/{name}/meta` · `PUT /collections/{name}/meta`

Read or replace a collection's free-form string→string metadata map. `PUT`
replaces the whole map.

```bash
curl -s -X PUT localhost:7700/collections/docs/meta \
  -H 'content-type: application/json' \
  -d '{"model": "text-embedding-3-small", "owner": "search-team"}'
# → {"ok": true}

curl -s localhost:7700/collections/docs/meta
# → {"model": "text-embedding-3-small", "owner": "search-team"}
```

### `POST /collections/{name}/upsert`

Insert or overwrite records by id. Each record is `{id, vector, attrs}`; `vector`
length must match the store dimension. `attrs` values are tagged
(`{"Str": …}`, `{"Int": …}`, `{"Bool": …}`, `{"List": […]}`, `{"Null": null}`).

```bash
curl -s localhost:7700/collections/docs/upsert \
  -H 'content-type: application/json' \
  -d '{"records": [
        {"id": "a", "vector": [1,0,0], "attrs": {"lang": {"Str": "rust"}, "ts": {"Int": 1781000000}}}
      ]}'
# → {"upserted": 1}
```

The whole batch is all-or-nothing: a dimension mismatch or other fault rolls the
store back, and the call returns `400` having changed nothing.

### `POST /collections/{name}/delete`

Delete by explicit ids, or by an attribute filter — supply `ids` **or** `filter`;
`filter` wins if both are present.

```bash
# By id
curl -s localhost:7700/collections/docs/delete \
  -H 'content-type: application/json' -d '{"ids": ["a", "b"]}'

# By filter (delete everything archived)
curl -s localhost:7700/collections/docs/delete \
  -H 'content-type: application/json' \
  -d '{"filter": [{"Eq": ["status", {"Str": "archived"}]}]}'
# → {"deleted": 7}
```

### `GET /collections/{name}/records`

Every live record in the collection (id, vector, attrs) as a JSON array. Useful
for export or for re-embedding against a new model. There is no pagination here —
use [`POST /list`](#post-list) when you want filtering or paging.

### `POST /search`

Nearest-neighbour search. `query` is the only required field. An empty or omitted
`scope` searches every collection in one merged ranking (sound because all
collections share one embedding space).

```bash
curl -s localhost:7700/search \
  -H 'content-type: application/json' \
  -d '{
        "query": [1, 0, 0],
        "scope": ["docs"],
        "top_k": 5,
        "min_score": 0.2,
        "filter": [{"Eq": ["lang", {"Str": "rust"}]}]
      }'
```

| Field | Default | Meaning |
| --- | --- | --- |
| `query` | — (required) | query vector; length must equal the store dimension |
| `scope` | all collections | collection names to search |
| `top_k` | `10` | maximum hits to return |
| `min_score` | none | drop hits scoring below this similarity |
| `filter` | none | AND of predicates applied before scoring |

Returns hits sorted by descending score:

```json
[{"collection": "docs", "id": "a", "score": 1.0, "attrs": {"lang": {"Str": "rust"}}}]
```

### `POST /collections/{name}/fts-schema` · `POST /text-search` · `POST /hybrid-search`

Full-text search. First declare which attribute fields are full-text indexed (US
English analyzer), then query by text or fuse with a vector. See
[Full-text search](/guides/search/#full-text-search-bm25) for the ranking model.

```bash
# Declare the fields once.
curl -s -X POST localhost:7700/collections/docs/fts-schema \
  -H 'content-type: application/json' -d '{"fields": ["body"]}'

# BM25 text search. `min_score` here is a raw BM25 floor.
curl -s localhost:7700/text-search \
  -H 'content-type: application/json' \
  -d '{"field": "body", "query": "running quickly", "scope": ["docs"], "top_k": 5}'

# Hybrid: fuse a vector and a BM25 query with Reciprocal Rank Fusion.
curl -s localhost:7700/hybrid-search \
  -H 'content-type: application/json' \
  -d '{"vector": [1,0,0], "field": "body", "text": "vector database", "top_k": 5}'
```

Both search endpoints return the same hit shape as `/search`. `/text-search` takes the
search fields (`field`, `query`, `scope`, `top_k`, `min_score`, `filter`);
`/hybrid-search` takes `vector` + `field` + `text` plus `top_k`, `filter`, `rrf_k`
(default 60), and `candidates` (default 100), and has no `min_score` (a fused RRF score
has no absolute scale).

### `POST /list`

Metadata-only query — no vector, no scoring. Same `scope` and `filter` as search,
plus `offset`/`limit` for pagination.

```bash
curl -s localhost:7700/list \
  -H 'content-type: application/json' \
  -d '{"scope": ["docs"], "filter": [{"Eq": ["lang", {"Str": "rust"}]}], "offset": 0, "limit": 100}'
```

`limit` defaults to `100`, `offset` to `0`. The response shape matches search
(hits with a `score` of `0`, since nothing is scored).

The `filter` in both `/search` and `/list` is an AND of predicates: `Eq`, `Ne`,
`Glob`, `In`, `NotIn`, `Lt`, `Le`, `Gt`, `Ge`. See
[Search & filters](/guides/search/) for the full predicate grammar.

### `POST /flush` · `POST /compact`

Maintenance. `flush` forces buffered writes to disk; `compact` rewrites the store
to reclaim `dead_rows` and superseded log records. Both return `{"ok": true}`.

```bash
curl -s -X POST localhost:7700/compact   # → {"ok": true}
```

## Errors

Every error returns `{"error": "<message>"}`. The status code separates a client
mistake from a server fault:

| Status | When |
| --- | --- |
| `400 Bad Request` | malformed JSON, or a query/vector whose length ≠ store dimension |
| `401 Unauthorized` | missing or wrong bearer token (when a token is configured) |
| `403 Forbidden` | a write against a `--read-only` server |
| `409 Conflict` | the store's writer lock is held by another process |
| `413 Payload Too Large` | request body exceeds `--max-body-bytes` |
| `507 Insufficient Storage` | an allocation guard (`max_vector_bytes`) or OOM tripped |
| `500 Internal Server Error` | anything else (an IO fault, a bug) |
