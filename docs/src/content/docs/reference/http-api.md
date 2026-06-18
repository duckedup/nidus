---
title: HTTP API
description: The endpoint-by-endpoint reference for a running nidus server — every route, its JSON body, a curl example, and the error codes.
---

This is the endpoint reference for a running [`nidus serve`](/guides/http-server/). Every
route maps one-to-one onto a library method; bodies and responses are JSON. To run the
server, set a bind address, and configure auth, see the
[HTTP server guide](/guides/http-server/).

**Base URL** is wherever the server is bound (the examples use `localhost:7700`).
**Auth:** when the server is started with a token, every request except `GET /health` must
send `Authorization: Bearer <token>` — see [Authentication](/guides/http-server/#authentication).
**Errors** return `{"error": "<message>"}` with a status code; see [Errors](#errors).

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

## Health & introspection

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

## Collections & metadata

### `GET /collections`

Returns the collection names as a JSON array: `["docs", "notes"]`.

### `POST /collections/{name}`

Create a collection. The body is ignored. Upsert auto-creates a collection, so an
explicit create is only needed to register an empty one (e.g. to attach metadata
before any records land).

```bash
curl -s -X POST localhost:7700/collections/docs   # → {"created": "docs"}
```

### `DELETE /collections/{name}`

Drop a collection and its records. The body is ignored.

```bash
curl -s -X DELETE localhost:7700/collections/docs   # → {"dropped": "docs"}
```

### `GET /collections/{name}/meta`

Read a collection's free-form string→string metadata map.

```bash
curl -s localhost:7700/collections/docs/meta
# → {"model": "text-embedding-3-small", "owner": "search-team"}
```

### `PUT /collections/{name}/meta`

Replace a collection's metadata map wholesale.

```bash
curl -s -X PUT localhost:7700/collections/docs/meta \
  -H 'content-type: application/json' \
  -d '{"model": "text-embedding-3-small", "owner": "search-team"}'
# → {"ok": true}
```

### `POST /collections/{name}/fts-schema`

Declare which attribute fields of a collection are full-text indexed for BM25 (US
English analyzer). Run it once before (or after) upserting; see
[Full-text search](/guides/search/#full-text-search-bm25) for the ranking model.

```bash
curl -s -X POST localhost:7700/collections/docs/fts-schema \
  -H 'content-type: application/json' \
  -d '{"fields": ["body"]}'
# → {"ok": true}
```

## Records

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

## Search & queries

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

### `POST /text-search`

BM25 full-text search of a declared field. Returns the same hit shape as `/search`.
Takes `field`, `query`, `scope`, `top_k`, `filter`, and `min_score` — here a **raw
BM25** floor (not cosine).

```bash
curl -s localhost:7700/text-search \
  -H 'content-type: application/json' \
  -d '{"field": "body", "query": "running quickly", "scope": ["docs"], "top_k": 5}'
```

### `POST /hybrid-search`

Fuse a vector query and a BM25 text query with Reciprocal Rank Fusion. Takes `vector`
+ `field` + `text`, plus `top_k`, `filter`, `rrf_k` (default 60), and `candidates`
(default 100). There is no `min_score` (a fused RRF score has no absolute scale).
Returns the same hit shape as `/search`.

```bash
curl -s localhost:7700/hybrid-search \
  -H 'content-type: application/json' \
  -d '{"vector": [1,0,0], "field": "body", "text": "vector database", "top_k": 5}'
```

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

## Maintenance

### `POST /flush`

Force buffered writes to disk (relevant under `Fsync::OnFlush`). Returns `{"ok": true}`.

```bash
curl -s -X POST localhost:7700/flush   # → {"ok": true}
```

### `POST /compact`

Rewrite the store to reclaim `dead_rows` and superseded log records. Returns
`{"ok": true}`.

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
