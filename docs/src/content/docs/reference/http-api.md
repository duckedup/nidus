---
title: HTTP API
description: The endpoint-by-endpoint reference for a running nidus server (every route, its JSON body, a curl example, and the error codes).
---

This is the endpoint reference for a running [`nidus serve`](/guides/http-server/). Every
route maps one-to-one onto a library method; bodies and responses are JSON. To run the
server, set a bind address, and configure auth, see the
[HTTP server guide](/guides/http-server/).

**Base URL** is wherever the server is bound (the examples use `localhost:7700`).
**Auth:** when the server is started with a token, every request except the probe endpoints
(`GET /health`, `GET /ready`, `GET /metrics`) must send `Authorization: Bearer <token>`; see
[Authentication](/guides/http-server/#authentication).
**Errors** return `{"error": "<message>"}` with a status code; see [Errors](#errors).
**Correlation:** every response carries `X-Request-Id`. Send your own and nidus echoes it,
so the same id appears in your logs and the server's.

| Method & path | Operation | Library method |
| --- | --- | --- |
| `GET /health` | liveness check: `503` only when unrecoverably broken (always unauthenticated) | – |
| `GET /stats` | dimension, distance, the resolved open profile (ann, quantization, query_threads, mmap), collections, footprint | `dimension` / `footprint` |
| `GET /collections` | list collection names | `collections` |
| `POST /collections/{name}` | create a collection | `create_collection` |
| `DELETE /collections/{name}` | drop a collection and its records | `drop_collection` |
| `GET /collections/{name}/meta` | read collection metadata | `get_meta` |
| `PUT /collections/{name}/meta` | replace collection metadata | `set_meta` |
| `POST /collections/{name}/upsert` | insert or overwrite records | `upsert` |
| `POST /collections/{name}/delete` | delete by ids or by filter | `delete` / `delete_where` |
| `GET /collections/{name}/records` | every record in a collection | `get_all` |
| `POST /collections/{name}/fts-schema` | declare full-text-indexed fields | `set_fts_schema` |
| `POST /collections/{name}/filter-index` | declare filter-indexed fields | `set_filter_index` |
| `POST /search` | nearest-neighbour search | `search` |
| `POST /search/similar` | "more like this" using a stored record's own vector | `search_similar` |
| `POST /search/batch` | several queries in one round-trip, optionally RRF-fused | – |
| `POST /text-search` | BM25 full-text search | `text_search` |
| `POST /hybrid-search` | fused vector + BM25 (RRF) | `hybrid_search` |
| `POST /list` | metadata-only query (no vector) | `list` |
| `POST /aggregate` | count + sum over a filter, no records materialized | `aggregate` |
| `POST /flush` | flush buffered writes to disk | `flush` |
| `POST /compact` | reclaim dead rows and superseded log records | `compact` |
| `POST /refresh` | adopt another instance's newer committed state | `refresh` |
| `GET /ready` | whether this instance can serve (store open, not fenced, not stale) | – |
| `GET /cluster` | role, writer-handle state, fencing token, commit counter, staleness | `cluster_status` |
| `GET /metrics` | Prometheus scrape: traffic, search path, lease counters (always unauthenticated) | – |
| `POST /collections/{name}/remember`* | text in, optionally summarize, embed, and upsert | – |
| `POST /collections/{name}/recall`* | text in, embed, and search with TTL filtering | `search` |
| `/mcp`** | the Model Context Protocol surface, nested inside this router | – |

\* Present only in a `memory`-featured build (the `serve` umbrella; absent from a plain
`--features cli` build). See [Memory](#memory-remember--recall) below.
\*\* Needs the `mcp` feature on top of `memory`. See [`/mcp`](#mcp) below.

## Health & introspection

### `GET /health`

Liveness probe. Returns `200` with the body `ok`. Always reachable without a
token, so a load balancer or `docker healthcheck` needs no credential.

Says almost nothing about the store: only that the process is up, answering, and not
*unrecoverably* broken. An instance waiting for the writer handle (see `/ready`) is alive,
and killing it would be exactly wrong, so this keeps returning `200` throughout. So does an
instance that is merely **busy**: a large upsert holds the store's write guard for the length
of the batch, which is normal work, not a fault.

It returns `503` in exactly one case: the store's lock has been **poisoned** by a panic that
unwound while the store was locked for writing. That leaves the in-RAM index possibly out of
step with the durable bytes, and the condition never clears; every subsequent request would
fail. The instance cannot recover on its own, so liveness fails and a supervisor restarts it;
the durable data is intact and the fresh process rebuilds from it. In cluster mode that
restart is also what releases the writer lease, letting a `--wait-for-lease` standby take over.

```json
{
  "status": "unhealthy",
  "error": "store lock poisoned: a panic left this instance's in-RAM state untrustworthy — it must be restarted"
}
```

### `GET /ready`

Readiness probe. `200` once this instance can actually serve; `503` otherwise. It fails for
four distinct reasons, each of which should take an instance out of rotation:

- **no store yet**: still starting, or a standby waiting for the writer handle;
- **fenced**: this writer was superseded, so every write would fail and it must be replaced.
  A writer notices this on its own lease-renewal timer, so it stops reporting ready even if no
  write arrives to discover it;
- **stale**: a reader has gone longer than `--max-staleness` without verifying it is current
  (only when that bound is set);
- **poisoned**: a panic left this instance unrecoverable, so it leaves the load balancer as
  well as failing `/health` (see above).

What does **not** make an instance unready is being **busy**. A large upsert holds the store's
write guard for the whole batch, and readiness is answered without ever taking that lock, so a
writer stays in rotation while it works. This matters most where there is only one writer to
route to: dropping out mid-batch would take writes offline during exactly the operation the
instance exists to perform.

Also always reachable without a token: an orchestrator would read a `401` as "not ready"
and never route to a healthy instance.

Use this, not `/health`, to decide whether to send an instance traffic. The two differ
whenever an instance is *waiting*: the server binds its port before opening the store, so
`/health` answers immediately while `/ready` stays `503` until there is something to serve.
That gap is the whole point for a standby writer, which may wait indefinitely for the
active writer to release the handle.

Data routes answer `503` during that window too, with an error explaining that the
instance is waiting or still starting up.

### `GET /cluster`

Who this instance is and how current it is: the introspection to reach for during an
incident. Always unauthenticated-safe to scrape? No: unlike the probes, this one **does**
require the token when one is configured.

```json
{
  "role": "ClusterWriter",
  "cluster": true,
  "holds_writer_handle": true,
  "fenced": false,
  "lease_owner": "4131-1784992862827886000",
  "commit_version": 12,
  "staleness_secs": 0,
  "max_staleness_secs": null
}
```

`role` is one of `Writer`, `Reader`, `ClusterWriter`, `ClusterReader`, `InMemory`.
`lease_owner` is this instance's fencing token while it holds a cluster lease, and `null`
otherwise; comparing it across instances answers "who is the writer right now".
`commit_version` is the manifest commit counter being served, so a reader behind the writer
reports a lower number; the gap is replication lag. `staleness_secs` is `0` for a writer (it
*is* the current state) and, for a reader, the age of its last successful refresh.

Every field is read from memory: no object-store round trip, so this is cheap to poll.

### `GET /metrics`

Prometheus text exposition (`text/plain; version=0.0.4`). Always reachable without a
token: a scraper that got a `401` would report the target as down.

```bash
curl -s localhost:7700/metrics
```

```text
# HELP nidus_search_queries_total Vector searches served
# TYPE nidus_search_queries_total counter
nidus_search_queries_total 1483
# TYPE nidus_http_requests_total counter
nidus_http_requests_total{route="/search",status="2xx"} 1483
nidus_http_request_duration_seconds_bucket{route="/search",le="0.01"} 1402
…
```

Route labels are **templates** (`/collections/{name}/upsert`), never the collection name:
the scrape exposes traffic shape, not what is stored. The full metric list is in the
[server guide](/guides/http-server/#get-metrics). Reading it takes no store lock, so a
scrape answers instantly even during a long write.

### `GET /stats`

Store-wide introspection, the network equivalent of `nidus stats`.

```json
{
  "dimension": 768,
  "distance": "Cosine",
  "ann": null,
  "quantization": null,
  "query_threads": 1,
  "mmap": false,
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
ANN configuration when the server opened with one, whether from an explicit
`--ann hnsw`/`--ann ivf` flag or a default recorded earlier with `nidus configure`
(only the knobs that apply to the chosen index are reported):

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

Declare which attribute fields of a collection are full-text indexed for BM25. Run it
once before (or after) upserting; see
[Full-text search](/guides/search/#full-text-search-bm25) for the ranking model.

```bash
curl -s -X POST localhost:7700/collections/docs/fts-schema \
  -H 'content-type: application/json' \
  -d '{"fields": ["body"]}'
# → {"ok": true}
```

A field entry may also be an object, tuning BM25 and the analyzer for that field alone.
Every key but `field` is optional and defaults to what the bare-name form gets: `k1`
1.2, `b` 0.75, `language` `"english"`, no ASCII folding, no token-length cap
([details](/guides/search/#tuning-a-field)):

```bash
curl -s -X POST localhost:7700/collections/docs/fts-schema \
  -H 'content-type: application/json' \
  -d '{"fields": ["title", {"field": "body", "k1": 1.5, "b": 0.3, "ascii_folding": true}]}'
# → {"ok": true}
```

### `POST /collections/{name}/filter-index`

Declare which attribute fields are indexed for the text predicates (`Fuzzy`,
`ContainsAllTokens`, `ContainsAnyToken`, `ContainsTokenSequence`, `Regex`). Documents
already written are indexed as part of the declaration; see
[Indexing the text predicates](/guides/search/#indexing-the-text-predicates).

This changes how fast those predicates run, never what they return.

```bash
curl -s -X POST localhost:7700/collections/docs/filter-index \
  -H 'content-type: application/json' \
  -d '{"fields": ["body"]}'
# → {"ok": true}
```

A field entry may also be an object, choosing which structures to build. Both `tokens`
(the three token predicates) and `trigrams` (`Fuzzy` and `Regex`) default to `true`, and
a field with both off is rejected as a `400`. An empty `fields` list drops the
declaration:

```bash
curl -s -X POST localhost:7700/collections/docs/filter-index \
  -H 'content-type: application/json' \
  -d '{"fields": ["title", {"field": "tag", "trigrams": false}]}'
# → {"ok": true}
```

## Records

### `POST /collections/{name}/upsert`

Insert or overwrite records by id. Each record is `{id, vector, attrs}`; `vector`
length must match the store dimension, and may be **omitted** for a text-only document.
`attrs` values are tagged: `{"Str": …}`, `{"Int": …}`, `{"Bool": …}`, `{"List": […]}`,
`{"Float": …}`, `{"DateTime": …}` (epoch milliseconds), and the unit variant `Null` is
the bare string `"Null"`, not an object.

`Float` and `Int` are distinct types and never cross-compare in a filter, so a whole
number sent as `{"Float": 1.0}` stays a `Float` on the way back out. Pick one spelling per
attribute and keep to it.

```bash
curl -s localhost:7700/collections/docs/upsert \
  -H 'content-type: application/json' \
  -d '{"records": [
        {
          "id": "a",
          "vector": [1, 0, 0],
          "attrs": {"lang": {"Str": "rust"}, "ts": {"Int": 1781000000}}
        }
      ]}'
# → {"upserted": 1}
```

The whole batch is all-or-nothing: a dimension mismatch or other fault rolls the
store back, and the call returns `400` having changed nothing.

### `POST /collections/{name}/delete`

Delete by explicit ids, or by an attribute filter: supply `ids` **or** `filter`;
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
for export or for re-embedding against a new model. There is no pagination here;
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
| `query` | – (required) | query vector; length must equal the store dimension |
| `scope` | all collections | collection names to search |
| `top_k` | `10` | maximum hits to return |
| `offset` | `0` | top-ranked hits to skip, for pagination |
| `min_score` | none | drop hits scoring below this similarity |
| `filter` | none | AND of predicates applied before scoring |
| `exact` | `false` | force the exact scan, bypassing any index and quantization |
| `include_attributes` | all attrs | return only these attrs |
| `exclude_attributes` | all attrs | return every attr but these |
| `rank_by` | none | a [ranking expression](/guides/search/#ranking-by-recency) over the metric |
| `limit_per` | none | cap hits per distinct value of an attribute |
| `rerank` | none | [rerank](#reranking) the over-fetched window at a hosted cross-encoder; `rerank.query` is required here |

Returns hits ordered by `(score desc, collection, id)`; the tie-break is a guarantee,
which is what makes paging coherent:

```json
[{"collection": "docs", "id": "a", "score": 1.0, "attrs": {"lang": {"Str": "rust"}}}]
```

`offset` pages one ranking: `{"top_k": 20}` then `{"top_k": 20, "offset": 20}` tiles it
with no gap and no overlap. An `offset` past the last hit returns `[]` rather than an
error. `offset + top_k` may not exceed **10 000**: beyond that the request is a `400`,
never a silently shortened page. A page is stable only against an unchanging store;
concurrent writes shift the ranking under a paged walk.

`exact: true` runs the exact brute-force scan for that one request, bypassing the ANN
walk, the per-segment index, and the quantized first pass; the store keeps its index for
every other query.

`include_attributes` and `exclude_attributes` choose which attrs the hits carry; omit
both for every attr, exactly as before. Sending **both** in one request is a `400`, not a
precedence rule. The projection is applied where the hit is built, so an excluded attr is
never serialized, which is the point on a collection of long text bodies.

#### Ranking by recency

`rank_by` layers a recency decay over the metric: an age penalty **subtracted** from each
hit's score, so it works for every distance metric and for BM25 alike.

```bash
curl -s localhost:7700/search \
  -H 'content-type: application/json' \
  -d '{
        "query": [1, 0, 0],
        "rank_by": {"Decay": {"field": "updated_at",
                              "origin": 1770000000000,
                              "scale": 604800000,
                              "lambda": 0.2}}
      }'
```

| `Decay` field | Default | Meaning |
| --- | --- | --- |
| `field` | – (required) | timestamp attr: a `DateTime` or an `Int`, epoch milliseconds |
| `origin` | – (required) | "now" in epoch ms; ages are measured back from here |
| `scale` | `604800000` (7 days) | the age at which the factor equals `decay` |
| `decay` | `0.5` | the factor at one `scale` of age (`0.5` makes `scale` a half-life) |
| `lambda` | `1.0` | score a fully-decayed hit gives up |
| `missing` | `1.0` | factor for a record with no usable timestamp (**no penalty**) |

The score is `base − lambda × (1 − decay^(age / scale))`. `missing` defaults to `1.0`, so
enabling decay never buries records written before the field existed. `rank_by` does not
force an exact scan; over an ANN or quantized result set it reorders within an approximate
candidate set. A malformed expression (a non-positive `scale`, a `decay` outside `(0, 1)`, a
negative `lambda`) is a `400`.

#### Capping hits per attribute value

```bash
curl -s localhost:7700/search \
  -H 'content-type: application/json' \
  -d '{"query": [1, 0, 0], "top_k": 20, "limit_per": {"field": "path", "max": 2}}'
```

Records **missing** the attribute share one group, and the value is read from the stored
record, so `exclude_attributes` cannot lift the cap. The cap is exact only within an
over-fetch window, so a capped page may come back shorter than `top_k`; what is guaranteed
is that no page carries more than `max` hits for one value.

### `POST /search/similar`

"More like this": search using the vector already stored at `collection`/`id`, instead of
a caller-supplied `query`. Otherwise it takes the same fields as `POST /search`.

```bash
curl -s localhost:7700/search/similar \
  -H 'content-type: application/json' \
  -d '{"collection": "docs", "id": "a1", "top_k": 5}'
```

| Field | Default | Meaning |
| --- | --- | --- |
| `collection` | – (required) | collection the source record lives in |
| `id` | – (required) | id of the record to search with |
| `scope` | the source's own collection | collection names to search |
| `top_k` | `10` | maximum hits to return |
| `offset` | `0` | top-ranked hits to skip, for pagination |
| `min_score` | none | drop hits scoring below this similarity |
| `filter` | none | AND of predicates applied before scoring |
| `exact` | `false` | force the exact scan, bypassing any index and quantization |
| `include_attributes` | all attrs | return only these attrs |
| `exclude_attributes` | all attrs | return every attr but these |
| `rank_by` | none | a [ranking expression](/guides/search/#ranking-by-recency) over the metric |
| `limit_per` | none | cap hits per distinct value of an attribute |

The one difference from `/search`: an omitted or empty `scope` searches only the source's
own collection, not every collection in the store.

The source record is always excluded from its own results, by `(collection, id)` identity
rather than by score, so a genuine duplicate of the source (also scoring near 1.0) still
comes back. `collection`/`id` naming no record, or a record with no stored vector to search
with (a text-only entry), is a `400` naming the id and the reason, not an empty result.

### `POST /text-search`

BM25 full-text search of declared fields. Returns the same hit shape as `/search`.
Takes `scope`, `top_k`, `offset`, `filter`, `rank_by`, `limit_per`, `min_score` (here a
**raw BM25** floor, not cosine), the `include_attributes`/`exclude_attributes` projection,
and the query itself in one of two spellings.

```bash
curl -s localhost:7700/text-search \
  -H 'content-type: application/json' \
  -d '{"field": "body", "query": "running quickly", "scope": ["docs"], "top_k": 5}'
```

Name several fields with `clauses` instead; each clause carries its own text:

```bash
curl -s localhost:7700/text-search \
  -H 'content-type: application/json' \
  -d '{
        "clauses": [
          {"field": "title", "query": "rust"},
          {"field": "body",  "query": "async runtime"}
        ],
        "combine": "Sum",
        "top_k": 5
      }'
```

| field | default | meaning |
| --- | --- | --- |
| `field` + `query` | – | the single-clause spelling |
| `clauses` | – | `[{field, query}, …]`, one entry per field searched |
| `combine` | `"Sum"` | `"Sum"` adds every matched clause's score; `"Max"` takes the strongest |
| `explain` | `false` | report each matched clause's own BM25 score |
| `highlight` | absent | `{}` for defaults, or `{"max_fragments": 2, "fragment_chars": 120}` |
| `rerank` | none | [rerank](#reranking) the over-fetched window; `rerank.query` defaults to this request's own query text |

`field`+`query` and `clauses` are **mutually exclusive**, and an empty `clauses` list is a
`400`: an empty result set would otherwise read as "no matches" rather than "no query".

### `POST /hybrid-search`

Fuse a vector query and a BM25 text query with Reciprocal Rank Fusion. Takes `vector`
plus the text leg (`field` + `text`, or the same `clauses` + `combine` as `/text-search`),
plus `top_k`, `offset` (which pages the **fused** ranking, never a leg),
`filter`, `rrf_k` (default 60), `candidates` (default 100), `explain`/`highlight`, and
an optional [`rerank`](#reranking) (its `query` defaults to this request's own text leg).
There is no `min_score` (a fused RRF score has no absolute scale).
Returns the same hit shape as `/search`.

```bash
curl -s localhost:7700/hybrid-search \
  -H 'content-type: application/json' \
  -d '{"vector": [1,0,0], "field": "body", "text": "vector database", "top_k": 5}'
```

`vector_weight` and `text_weight` (both default `1.0`) scale each leg's contribution to the
fused score. Leaving them out (or sending `1.0` for both) reproduces the unweighted fusion
exactly. A non-finite or negative weight is a `400`.

```bash
curl -s localhost:7700/hybrid-search \
  -H 'content-type: application/json' \
  -d '{"vector": [1,0,0], "field": "body", "text": "CVE-2026-1234", "text_weight": 3.0}'
```

### Annotations: why a hit matched

`explain` and `highlight` add an `annotations` object to each hit. Both are opt-in, and the
key is **absent** otherwise: an unannotated response is byte-for-byte what it always was.

```json
{
  "collection": "docs", "id": "a", "score": 0.031, "attrs": {"title": {"Str": "vector search"}},
  "annotations": {
    "vector": {"rank": 0, "score": 0.98},
    "text": {"rank": 1, "score": 1.10},
    "clauses": [{"field": "title", "score": 0.49}, {"field": "body", "score": 0.61}],
    "highlights": [
      {"field": "body", "fragments": [
        {"text": "the engineers were running", "spans": [[19, 26]]}
      ]}
    ]
  }
}
```

`vector`/`text` are the fusion legs' own rank and score, and appear only on
`/hybrid-search`. `clauses` lists the clauses that actually matched, in query order.
`spans` are `[start, end)` **byte** offsets into that fragment's `text`, covering the word
as the document spells it: a query for `run` marks `running`. Highlighting reads the
stored text, so it still works on a field `include_attributes`/`exclude_attributes`
dropped from the payload: that pairing (drop the long body, keep the snippet) is the point.

### Reranking

`/search`, `/text-search`, `/hybrid-search`, and `/collections/{name}/recall` all
accept an optional `rerank` object: rank `(offset + top_k) * overscan` deep by the
plain metric, score each candidate's text against `query` at a hosted cross-encoder,
then trim to `top_k` by the provider's score. See the
[reranking guide](/guides/reranking/) for the full mechanism, including why this
`overscan` is unrelated to the ANN `overscan` and the quantization `rescore`.

| `rerank` field | Default | Meaning |
| --- | --- | --- |
| `query` | endpoint's own query text | Text scored against each candidate. **Required** on `/search`: a raw-vector query has no text of its own. |
| `text_field` | `nidus.text` | Attr holding each candidate's text. A candidate missing it is returned unranked, sorted below every reranked hit. |
| `overscan` | `4` | How many times deeper than `top_k` to rank before reranking. Maximum 64, and the resulting depth is capped at the same 10000 a plain query is. Past 64 the request is a `400` rather than a silent clamp. |
| `model` | – | Override the rerank model for this request only. Omit to use the server's configured model (`--rerank-model`). |

```bash
curl -s localhost:7700/collections/docs/recall \
  -H 'content-type: application/json' \
  -d '{"query": "how do I rotate the signing key", "top_k": 5, "rerank": {"overscan": 8}}'
```

A `rerank` field against a server started with no `--rerank-provider` is a `400`
naming the flag, not a silent no-op. `min_score` (where the endpoint has one) is
compared against the plain metric score while building the deep window, before the
rerank runs, since a cross-encoder score is on a different scale.

### `POST /list`

Metadata-only query: no vector, no scoring. Same `scope` and `filter` as search,
plus `offset`/`limit` for pagination.

```bash
curl -s localhost:7700/list \
  -H 'content-type: application/json' \
  -d '{
        "scope": ["docs"],
        "filter": [{"Eq": ["lang", {"Str": "rust"}]}],
        "offset": 0,
        "limit": 100
      }'
```

`limit` defaults to `100`, `offset` to `0`. `/list` takes the same
`include_attributes`/`exclude_attributes` projection as `/search`, with the same `400`
when both are sent. The response shape matches search (hits with a `score` of `0`, since
nothing is scored).

`order_by` sorts by an attribute instead of storage order: ORDER BY with no vector query.
Sorting runs over the whole match set before the page is cut, so `offset`/`limit` walk the
sorted order.

```bash
curl -s localhost:7700/list \
  -H 'content-type: application/json' \
  -d '{"order_by": {"field": "updated_at", "descending": true}, "limit": 20}'
```

`descending` defaults to `false`. Values that do not order against the first orderable one
(a different `Value` variant, an unorderable `Null`/`List`, or a record missing the
attribute) sort into one trailing bucket, which stays trailing when reversed.

### `POST /search/batch`

Answer up to **16** vector queries in one round-trip. Each entry of `queries` takes exactly
the fields `/search` does, with its own scope, filter and `top_k`.

```bash
curl -s localhost:7700/search/batch \
  -H 'content-type: application/json' \
  -d '{
        "queries": [
          {"query": [1, 0, 0], "top_k": 5},
          {"query": [0, 1, 0], "top_k": 5, "filter": [{"Eq": ["lang", {"Str": "rust"}]}]}
        ]
      }'
```

```json
{"results": [[{"collection": "docs", "id": "a", "score": 0.98, "attrs": {}}], []]}
```

`results` holds one ranking per query, **in request order**. The whole batch is validated
before any query runs, so a malformed leg answers `400` rather than returning a partial
result that cannot be told apart from a complete one.

Add `fuse` to merge the legs into a single ranking with the same RRF `/hybrid-search` uses;
the response then carries `fused` instead of `results`:

```bash
curl -s localhost:7700/search/batch \
  -H 'content-type: application/json' \
  -d '{
        "queries": [{"query": [1, 0, 0]}, {"query": [0, 1, 0]}],
        "fuse": {"rrf_k": 60, "weights": [1.0, 0.5], "top_k": 10}
      }'
```

| Field | Default | Meaning |
| --- | --- | --- |
| `queries` | – (required) | 1–16 search bodies, each shaped exactly like `/search` |
| `fuse` | none | merge the legs into one ranking instead of returning them side by side |
| `fuse.rrf_k` | `60` | RRF smoothing constant |
| `fuse.weights` | all `1.0` | per-leg weights; must be empty or exactly as long as `queries` |
| `fuse.top_k` | `10` | how many fused hits to return |

A `weights` list of the wrong length is a `400` rather than a zero-filled short list, which
would silently re-weight the wrong query.

### `POST /aggregate`

Count the filter-matching records and total numeric attributes, without materializing any
of them. Same `scope` and `filter` as `/list`.

```bash
curl -s localhost:7700/aggregate \
  -H 'content-type: application/json' \
  -d '{
        "scope": ["docs"],
        "filter": [{"Eq": ["lang", {"Str": "rust"}]}],
        "sum": ["bytes"]
      }'
```

```json
{"count": 12, "sums": {"bytes": {"Int": 40960}}}
```

`count` is always reported; `sum` names attributes to total. Each total is a tagged value:
`Int` while every addend was an `Int`, `Float` once any `Float` joined. A missing or
non-numeric value is skipped rather than counted as zero. A filter matching nothing answers
`{"count": 0, ...}` rather than erroring.

Add `group_by` to get the same figures **per distinct value** of an attribute, in the same
pass and beside the unchanged totals:

```bash
curl -s localhost:7700/aggregate \
  -H 'content-type: application/json' \
  -d '{"sum": ["bytes"], "group_by": "lang"}'
```

```json
{
  "count": 12,
  "sums": {"bytes": {"Int": 40960}},
  "groups": [
    {"value": {"Str": "rust"}, "count": 9, "sums": {"bytes": {"Int": 38000}}},
    {"value": null, "count": 3, "sums": {"bytes": {"Int": 2960}}}
  ]
}
```

Groups come back **largest first**, with ties broken deterministically so repeating a query
repeats the order. A `null` `value` is the group of records **missing** the attribute, not
the same as records holding a `Null`. `groups` is omitted entirely when no `group_by` was
asked for, so an existing client sees the response it always saw. If the distinct values
exceed the server's cap (10 000), later ones are dropped and `groups_truncated` is `true`.

### The `filter` grammar

Every route that takes a `filter` (`/search`, `/text-search`, `/hybrid-search`, `/list`,
`/aggregate`, and `/collections/{name}/delete`) takes the same one: a JSON array of
predicates, AND-combined. Each predicate is a single-key object naming the variant.

| Group | Predicates |
| --- | --- |
| Equality & sets | `Eq`, `Ne`, `In`, `NotIn` |
| Ranges (same-type, orderable) | `Lt`, `Le`, `Gt`, `Ge` |
| Patterns | `Glob`, `IGlob` (ASCII-case-insensitive), `Regex` |
| List containment | `Contains`, `NotContains`, `ContainsAny` |
| Text | `Fuzzy`, `ContainsAllTokens`, `ContainsAnyToken`, `ContainsTokenSequence` |
| Boolean groups | `All`, `Any`, `Not` |

```json
[
  {"Any": [{"Eq": ["lang", {"Str": "rust"}]}, {"Eq": ["lang", {"Str": "go"}]}]},
  {"Not": {"Contains": ["tags", {"Str": "generated"}]}},
  {"Ge": ["updated_at", {"DateTime": 1770000000000}]},
  {"Fuzzy": ["name", "nidus", 2]},
  {"Regex": ["path", "src/[a-z]+/mod\\.rs"]}
]
```

`All`/`Any`/`Not` take predicates rather than values, so any boolean shape nests inside
the outer AND. `Not` and `Ne` differ on a **missing** attribute: `Ne` requires the key
present, `Not` is a true complement. A filter is validated once per query, before any row
is scanned, so an unparseable `Regex` or a `Fuzzy` budget above 8 is **refused with an
error** rather than quietly matching nothing. See
[Search & filters](/guides/search/#filters) for the full semantics.

## Memory (remember & recall)

Two text-native routes over the same store: send text, not vectors, and the server embeds
(and optionally summarizes) it for you. Both are compiled in only under the `memory`
feature, which is part of the `serve` umbrella and therefore present in the prebuilt
`cargo binstall` binary. A `cargo install nidus --features cli` build has **neither** route;
hitting them there is a `404`, not a `400`. See the
[remember & recall guide](/guides/remember-and-recall/) for setup (an embedder, optionally a
summarizer) and [parity across the surfaces](/guides/remember-and-recall/#parity-across-the-surfaces)
for how these two routes compare to the Rust API, the CLI, and the MCP tools below.

### `POST /collections/{name}/remember`

Store a piece of text: embed it (summarizing first if asked), then upsert it under the
given id.

```bash
curl -s localhost:7700/collections/notes/remember \
  -H 'content-type: application/json' \
  -d '{"id": "note-1", "text": "the deploy window moved to Fridays",
       "attrs": {"project": {"Str": "nidus"}}, "ttl_seconds": 604800}'
# → {"ok": true, "upserted": 1, "id": "note-1", "deduped": false}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `id` | – (required) | the record id to write. Unlike the MCP `remember` tool, nothing here derives one for you; omit it and the request is a `400` |
| `text` | – (required) | the text to remember |
| `mode` | `"raw"` | `"raw"` embeds the text as given; `"summarize"` summarizes first and embeds the summary (needs a summarizer) |
| `attrs` | `{}` | structured metadata stored alongside the text |
| `ttl_seconds` | none (never expires) | seconds until this entry expires, counted from the moment it is written |
| `dedupe_threshold` | none (dedup disabled) | cosine-similarity floor above which this write updates the nearest existing entry instead of inserting a near-duplicate |

The response is `{"ok": true, "upserted": <n>, "id": "<id>", "deduped": <bool>}`.
**`id` and `deduped` are not an echo of what you sent.** When `dedupe_threshold` is set and
an existing entry scores above it, the write updates that entry in place instead of
inserting a competing near-duplicate, `deduped` comes back `true`, and `id` is the id of
the entry that was actually written, which may differ from the `id` you sent. Read `id` out
of the response rather than assuming it matches the request.

`mode: "summarize"` stamps the generated summary into `nidus.summary` in `attrs`,
alongside the original text.

Errors: no embedder configured at serve time is a `400` naming `--embed-provider`;
`mode: "summarize"` with no summarizer configured is a separate `400` naming
`--summarize-provider`.

### `POST /collections/{name}/recall`

Search by meaning: embed the query text, then rank the collection against it.

```bash
curl -s localhost:7700/collections/notes/recall \
  -H 'content-type: application/json' \
  -d '{"query": "when do we deploy", "top_k": 5,
       "filter": [{"Eq": ["project", {"Str": "nidus"}]}]}'
```

| Field | Default | Meaning |
| --- | --- | --- |
| `query` | – (required) | query text; embedded server-side |
| `top_k` | `10` | maximum hits to return |
| `min_score` | none | drop hits scoring below this cosine similarity |
| `filter` | none | AND of predicates applied before scoring |
| `rerank` | none | [rerank](#reranking) the over-fetched window; `rerank.query` defaults to this request's own recall query |

Returns the same `HitDto` shape as `/search`: an array of `{collection, id, score, attrs}`.

**TTL filtering applies here, and only here, of the routes on this page.** `/recall`
AND-s a not-expired predicate into your filter, so an entry past its `ttl_seconds` never
comes back from this route. `/search`, `/list`, `/text-search`, and `/hybrid-search` apply
no such predicate: an expired-but-unswept memory is still visible to those general-purpose
routes unless you filter `nidus.expires_at` yourself. Do not read TTL as a store-wide
property; it is a `/recall`-specific (and MCP-tool-specific) read filter, not a deletion.

Errors: no embedder configured is the same `400` as `/remember`. Recalling against a
collection embedded by a different provider/model is also refused, since the vectors would
not be comparable.

## `/mcp`

`nidus serve` also answers the [Model Context Protocol](https://modelcontextprotocol.io)
at `/mcp`, behind the `mcp` feature (also folded into the `serve` umbrella). It is
`nest_service`'d **inside** the same middleware stack as every route above, not layered
separately, so it inherits the body limit, backpressure, bearer auth, and metrics rather
than reimplementing any of them: a token required elsewhere on this server is required at
`/mcp` too.

Nine tools, all text-native: `remember`, `recall`, `text_search`, `hybrid_search`,
`list_collections`, `stats`, `forget`, `get`, `browse`. **No tool takes a raw vector**:
every argument is natural language, which is deliberate, since a model cannot emit a raw
float array as a tool call, and `tests/e2e/mcp/` asserts the surface stays that way.

This page does not restate the tool schemas, the transport details, or protocol
negotiation; see the [MCP guide](/guides/mcp/) for those, including the stdio transport
(`nidus mcp`) that does not go through this HTTP surface at all.

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

An optional body sweeps expired entries first: `{"expired": true}` deletes every
entry whose `nidus.expires_at` has passed, across every collection, then compacts to
reclaim the rows it freed, all in one call. A bodyless `POST /compact` (or
`{"expired": false}`) is still a plain compact with no sweep.

```bash
curl -s -X POST localhost:7700/compact \
  -H 'content-type: application/json' \
  -d '{"expired": true}'   # → {"ok": true}
```

### `POST /refresh`

Adopt newer state committed by another instance writing to the same shared store. A
read-only instance loads a snapshot when it starts and keeps serving that snapshot, so
this is how you advance it. `adopted` says whether there was anything new to take up,
which lets a poller distinguish "no change" from "moved forward".

```bash
curl -s -X POST localhost:7700/refresh   # → {"adopted": true}
```

Reads are deliberately not made to refresh on their own: that would put a metadata fetch
on every query, which is the opposite of what a read-heavy fan-out wants. Call this as
often as your staleness tolerance requires. It is harmless anywhere else: an instance
that does its own writing already has the freshest state, and answers `{"adopted": false}`.

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
| `503 Service Unavailable` | the store is not open yet (a standby waiting for promotion), **or** the request was [shed](/guides/http-server/#backpressure) at `--max-concurrent-requests` |
| `504 Gateway Timeout` | the request exceeded `--read-timeout` / `--write-timeout` |
| `507 Insufficient Storage` | an allocation guard (`max_vector_bytes`) or OOM tripped |
| `500 Internal Server Error` | anything else (an IO fault, a bug) |

### Retrying

A shed `503` carries `Retry-After: 1` and `{"retryable": true}` in the body. Nothing was
attempted and the store is untouched, so retrying after a brief backoff is correct.

A `504` carries `{"retryable": false}`. The request *was* admitted and the work may still
be running (a timeout frees the client, not the CPU), so an immediate retry piles a second
copy onto an instance that is already behind. Back off substantially, or don't retry.
