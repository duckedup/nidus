---
title: Python SDK
description: "nidus on PyPI — the official Python client for nidus. Connect to a local or remote nidus server over HTTP, upsert, and search, with a sync client that needs nothing but the standard library."
---

[`nidus`](https://pypi.org/project/nidus/) is the official Python client for nidus. It
drives a running [`nidus serve`](/guides/http-server/) instance over HTTP — local or
remote.

```sh
pip install nidus            # the sync client — pulls ZERO dependencies
pip install 'nidus[async]'   # adds AsyncNidusClient (httpx)
```

`NidusClient` is built on `urllib.request` from the standard library, so `pip install
nidus` brings **nothing** else with it — the same zero-dependency posture the
[JavaScript SDK](/sdks/javascript/) gets from the platform `fetch`. Only the async client
needs a third-party HTTP stack, and it lives behind the `async` extra. Python 3.9+, typed
(`py.typed` ships in the wheel).

The SDK is versioned in lockstep with nidus: the crate's version is the single source of
truth, so a given `nidus` release on PyPI is the client for the identically-numbered
nidus release. Match the two and the wire contract lines up.

## Connecting

"Local vs remote" is just the base URL — point the client at a local `nidus serve` or any
reachable host. When the server was started with a [token](/guides/http-server/), pass it
as `token`.

```python
import os

from nidus import NidusClient

# Local
db = NidusClient("http://127.0.0.1:7700")

# Remote, with bearer-token auth
db = NidusClient(
    "https://nidus.internal.example.com",
    token=os.environ["NIDUS_TOKEN"],
    timeout=5.0,  # per-request timeout in SECONDS; None (the default) means no timeout
)
```

Nothing is opened until the first request. The client is also a context manager, and the
default `urllib` transport is connectionless — so `close()` only really matters once a
[pooled transport](#bulk-ingest-supply-a-pooled-transport) or the async client is in play.
Using `with` means never having to remember which case you are in:

```python
with NidusClient("http://127.0.0.1:7700") as db:
    print(db.health())  # True when the server answers; never raises
```

## The async client

`AsyncNidusClient` mirrors the sync client method for method, with `async def` and
`aclose()`. It is the one part of the SDK that needs a dependency, so it requires:

```sh
pip install 'nidus[async]'
```

Without `httpx` the import raises an `ImportError` that names that fix, rather than an
opaque `ModuleNotFoundError`. Either spelling of the import works:

```python
from nidus.aio import AsyncNidusClient   # explicit
import nidus; nidus.AsyncNidusClient     # lazy — resolved on first attribute access
```

`import nidus` itself never touches `httpx` — that is what keeps the dependency genuinely
optional instead of one every caller pays for.

```python
import asyncio

from nidus import f
from nidus.aio import AsyncNidusClient


async def main():
    async with AsyncNidusClient("http://127.0.0.1:7700") as db:
        await db.create_collection("docs")
        await db.upsert("docs", [{"id": "a", "vector": [0.1, 0.2, 0.3]}])
        hits = await db.search(query=[0.1, 0.2, 0.3], top_k=5, filter=[f.eq("lang", "rust")])
        return hits


asyncio.run(main())
```

Everything below is written against the sync client; add `await` and the async client
behaves identically.

## Upserting and searching

`attrs` accept plain Python values — `str`, `int`, `bool`, lists of `str`, and `None` —
and are normalized to nidus's [typed values](/reference/api/) for you. Results come back
with `attrs` decoded to plain Python values.

```python
db.create_collection("docs")

db.upsert("docs", [
    {"id": "a", "vector": [0.1, 0.2, 0.3], "attrs": {"lang": "rust", "year": 2024}},
    {"id": "b", "vector": [0.4, 0.5, 0.6], "attrs": {"lang": "go", "year": 2023}},
    # text-only doc — omit the vector
    {"id": "c", "attrs": {"body": "vector stores are neat"}},
])

for hit in db.search(query=[0.1, 0.2, 0.3], top_k=5):
    print(hit.collection, hit.id, hit.score, hit.attrs.get("lang"))
```

`upsert` and `delete` return a count. The search family returns a list of frozen `Hit`
dataclasses (`collection`, `id`, `score`, `attrs`), so a typo in a field name fails at the
call site instead of quietly returning `None`. `attrs` itself is a plain `dict`: reach for
`.get` unless every record in scope is known to carry the key, since attrs are per-record
rather than a schema.

Python decides `Int` vs `Float` from the **runtime type**, so `2.0` is a `Float` and `2`
is an `Int`. That matters because comparisons are same-type only: a `Float` range filter
never matches a record whose value was stored as an `Int`. For an explicit type, use the
`v.*` helpers (`v.str`, `v.int`, `v.float`, `v.bool`, `v.list`, `v.datetime`, `v.nil`);
`v.datetime` takes a `datetime` and travels as UTC epoch milliseconds:

```python
from nidus import v

db.upsert("docs", [{"id": "d", "attrs": {"tags": v.list(["a", "b"]), "rank": v.int(7)}}])
```

`v.nil()` is the explicit `Null` value — "set, and empty" — which is a different fact from
an absent key ("not set / not indexed"). The SDK keeps the two apart in both directions.

## Filtering

Build an AND-filter with the `f.*` helpers. Each predicate is a positive assertion about a
**present** attribute — an absent key matches nothing, including the negative predicates.
See [Search & filters](/guides/search/) for the full semantics.

```python
from nidus import f

hits = db.search(
    query=[0.1, 0.2, 0.3],
    top_k=10,
    filter=f.and_(
        f.eq("lang", "rust"),
        f.ge("year", 2020),
        f.in_("status", ["published", "draft"]),
        f.glob("path", "src/*"),
    ),
)
```

Predicates: `eq`, `ne`, `glob`, `iglob`, `in_`, `not_in`, `lt`, `le`, `gt`, `ge`, plus
`and_`. `iglob` is `glob` with ASCII case folded on both sides.

Those three trailing underscores are not a style choice: `in` and `and` are **reserved
words** in Python, so `f.in_`, `f.not_in`, and `f.and_` are the JavaScript SDK's `f.in`,
`f.notIn`, and `f.and`. Nothing else in the surface deviates.

A `Filter` is just a `list` of predicates, AND-combined, so `f.and_(...)` is sugar for
building that list — `filter=[f.eq("lang", "rust")]` is equally valid, and `[]` matches
everything.

## Full-text and hybrid search

```python
db.set_fts_schema("docs", ["body"])

# BM25 text search over one indexed field
text_hits = db.text_search(field="body", query="vector store", top_k=10)

# Fuse a vector query and a BM25 query via reciprocal rank fusion
hybrid_hits = db.hybrid_search(
    vector=[0.1, 0.2, 0.3],
    field="body",
    text="vector store",
    top_k=10,
)
```

`hybrid_search` takes no `min_score`: its score is a fused RRF rank, not a similarity, so
there is no meaningful floor to set. `rrf_k` and `candidates` tune the fusion.

## Remembering and recalling

When the server is started with an embedder
([`nidus serve --embed-provider …`](/guides/remember-and-recall/)) you can send **text**
and let the server embed it — no need to compute vectors client-side. `remember` embeds
and upserts; `recall` embeds the query and vector-searches.

```python
# Embed "the quick brown fox" and store it under id "a"
db.remember("notes", "a", "the quick brown fox", attrs={"tag": "x"})

# Summarize first, then embed the summary (the server also needs --summarize-provider).
# The stored record additionally carries `nidus.summary` and `nidus.source` attrs.
db.remember("notes", "b", long_article, mode="summarize")

# Embed the query text and search, best first
hits = db.recall("notes", "quick fox", top_k=5, min_score=0.2, filter=[f.eq("tag", "x")])
```

Against a server started **without** an embedder both raise `NidusError` with status
`400`, and the message names `--embed-provider`; `mode="summarize"` without a summarizer
configured is likewise a `400`. The client only ever sends text — the embedding always
happens server-side.

## The rest of the API

Every endpoint of the [HTTP API](/reference/http-api/) has a typed method:

```python
db.collections()                    # list[str]
db.stats()                          # dimension, distance, ANN config, collections, footprint
db.list(scope=["docs"], filter=[f.eq("lang", "rust")], offset=0, limit=50)
db.records("docs")                  # every record, attrs decoded
db.get_meta("docs"); db.set_meta("docs", {"owner": "search-team"})
db.delete("docs", ["a"])            # by id
db.delete_where("docs", f.and_(f.lt("year", 2000)))
db.flush(); db.compact()
db.drop_collection("docs")
db.health()                         # bool
```

Optional arguments all default to `None`, which means "omit the key" so the **server's**
default applies (`top_k = 10`, `limit = 100`, `rrf_k = 60.0`, `candidates = 100`). Those
numbers are deliberately not restated in Python — and it is why the defaults are `None`
rather than a number: `top_k=0` is a legitimate request for zero results, so `0` cannot
double as "unset".

Two `None`s carry real information and are never flattened:

- `stats().ann` is `None` when the store does exact brute-force search, as opposed to an
  `AnnInfo` full of defaults.
- `Record.vector` is `None` — never `[]` — for a text-only document.

## Three things the client refuses to send

Python's type system cannot express two mistakes that produce a **well-formed** request the
server accepts and answers wrongly, so the SDK refuses them at the call site instead:

```python
db.delete("docs", "a")          # TypeError: a str IS a Sequence[str] — this asked to
                                # delete the ids "a"... one character at a time
db.search(query=vec, scope="docs")   # TypeError — same slip, five collections that
                                     # do not exist, an empty result and a 200
f.in_("lang", "rust")           # TypeError — one predicate value per character
db.delete_where("docs", [])     # ValueError: an empty filter matches EVERYTHING, so this
                                # deleted the whole collection; use drop_collection
```

None of these raise anywhere else in the stack: `mypy --strict` accepts all four, and the
server answers `200`. The list forms (`["a"]`, `["docs"]`, `["rust"]`) are what was meant.

Vectors, conversely, are *accepted* more widely than JSON allows: elements are coerced with
`float()`, so `numpy` arrays (`np.float32` is not a `float` subclass and `json` refuses it),
torch scalars and `Decimal` all work without a `.tolist()` first.

## Bulk ingest: supply a pooled transport

The honest cost of a standard-library-only client: `urllib.request` opens a **fresh
connection per request**. For interactive use that is invisible; over a long run of
sequential upserts the handshakes are measurable overhead.

The escape hatch is `transport=` — a callable
`(method, url, headers, body, timeout) -> (status, text)`. Hand in one backed by `httpx`
or `requests` and you get pooling, keep-alive, retries, or instrumentation, without the
SDK taking on a dependency for everyone:

```python
import httpx

from nidus import NidusClient


class PooledTransport:
    """A connection-pooling transport for bulk ingest."""

    def __init__(self):
        self._client = httpx.Client()

    def __call__(self, method, url, headers, body, timeout):
        # A transport RETURNS non-2xx statuses; only a failure to get any response at
        # all should raise. httpx already behaves that way.
        r = self._client.request(method, url, content=body, headers=headers, timeout=timeout)
        return r.status_code, r.text

    def close(self):
        # NidusClient.close() calls close() on the transport if it has one, so
        # `with NidusClient(...)` shuts the pool down too.
        self._client.close()


with NidusClient("http://127.0.0.1:7700", transport=PooledTransport()) as db:
    for batch in batches:
        db.upsert("docs", batch)
```

The same seam is what lets the SDK's own unit tests exercise every endpoint with no server
and no socket.

`AsyncNidusClient` takes the natural equivalent for its own stack: `transport=` there is
an `httpx.AsyncBaseTransport` — a pre-tuned pool, or an `httpx.MockTransport` for tests.
It pools by default, so an async caller needs nothing extra for bulk ingest.

Note also that batching is the bigger lever than pooling: `upsert` takes a list, and one
request per batch is [one fsync per batch](/guides/storage/) on the server.

## Errors

A failed request raises `NidusError` carrying the HTTP status the server reported, so you
can tell a client fault from a server fault:

```python
from nidus import NidusError

try:
    db.upsert("docs", records)
except NidusError as err:
    if err.is_bad_request:      # 400 — e.g. a vector dimension mismatch
        ...
    if err.is_locked:           # 409 — the writer lock is held by another process
        ...
    print(err.status, err.message)
```

Also available: `is_read_only` (403), `is_out_of_capacity` (507 — `max_vector_bytes`
exceeded, or OOM), and `is_transport_error`.

A status of `0` is the sentinel for **no response at all** — connection refused, DNS
failure, or the request exceeded `timeout`. Every nidus SDK uses the same sentinel, so
"was this even reachable?" is answered identically in all of them.

Bad attribute values are rejected locally, before any request is made: a `float`
attribute or a non-string list element raises `TypeError`, and an integer outside `i64`
raises `ValueError` (Python's ints are unbounded; the store's `Int` is not).
