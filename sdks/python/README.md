# nidus (Python)

The Python client for [nidus](https://nidus.duckedup.org) — a small, fast vector store.
This package drives a running `nidus serve` instance over HTTP, whether it is on your
laptop or a remote host.

```bash
pip install nidus            # the sync client — pulls ZERO dependencies
pip install 'nidus[async]'   # adds AsyncNidusClient (httpx)
```

`NidusClient` is built on `urllib.request`, so installing this package brings nothing
else with it — the same zero-dependency posture the JS SDK gets from the platform
`fetch`. Only the async client needs a third-party HTTP stack, and it is quarantined
behind the `async` extra. Python 3.9+.

This package is versioned in lockstep with nidus itself: the crate's version is the
single source of truth, so a given `nidus` release on PyPI is the client for the
identically-numbered nidus release. Match the two and the wire contract lines up.

## Connecting

"Local vs remote" is just the base URL — point the client at a local `nidus serve` or
any reachable host.

```python
import os

from nidus import NidusClient

# Local
db = NidusClient("http://127.0.0.1:7700")

# Remote, with the bearer token the server was started with (`nidus serve --token`)
db = NidusClient(
    "https://nidus.internal.example.com",
    token=os.environ["NIDUS_TOKEN"],
    timeout=5.0,  # per-request timeout in SECONDS; None (the default) means no timeout
)
```

Both clients work as context managers. Nothing is opened until the first request, and the
default `urllib` transport is connectionless, so `close()` matters only once a pooled
transport (below) or the async client is in play — using `with` means you never have to
remember which case you are in:

```python
with NidusClient("http://127.0.0.1:7700") as db:
    print(db.health())  # True when the server answers; never raises
```

### The async client

`AsyncNidusClient` mirrors the sync client method for method, with `async def` and
`aclose()`. It requires `pip install 'nidus[async]'`; without `httpx` the import fails
with an `ImportError` that names that fix. Either spelling works:

```python
from nidus.aio import AsyncNidusClient   # explicit
import nidus; nidus.AsyncNidusClient     # lazy — resolved on first attribute access
```

`import nidus` itself never touches `httpx`, which is what keeps the dependency
genuinely optional.

```python
import asyncio
from nidus import f
from nidus.aio import AsyncNidusClient

async def main():
    async with AsyncNidusClient("http://127.0.0.1:7700") as db:
        await db.create_collection("docs")
        await db.upsert("docs", [{"id": "a", "vector": [0.1, 0.2, 0.3]}])
        hits = await db.search(query=[0.1, 0.2, 0.3], top_k=5, filter=[f.eq("lang", "rust")])

asyncio.run(main())
```

## Upserting and searching

`attrs` accept plain Python values — `str`, `int`, `bool`, lists of `str`, and `None` —
and are normalized to nidus's typed values for you. Results come back with `attrs`
decoded to plain Python values.

```python
db.create_collection("docs")

db.upsert("docs", [
    {"id": "a", "vector": [0.1, 0.2, 0.3], "attrs": {"lang": "rust", "year": 2024}},
    {"id": "b", "vector": [0.4, 0.5, 0.6], "attrs": {"lang": "go", "year": 2023}},
    # a text-only doc — omit the vector
    {"id": "c", "attrs": {"body": "vector stores are neat"}},
])

for hit in db.search(query=[0.1, 0.2, 0.3], top_k=5):
    print(hit.collection, hit.id, hit.score, hit.attrs.get("lang"))
```

`upsert` and `delete` return a count; the search family returns a list of `Hit`
dataclasses (`collection`, `id`, `score`, `attrs`). `attrs` is a plain `dict`, so reach
for `.get` unless every record in scope is known to carry the key — a search spans
whatever the scope holds, and attrs are per-record, not a schema.

There is no float attribute type — floats belong in the vector, so passing one raises
`TypeError` rather than silently truncating. For an explicit type, use the `v.*`
helpers (`v.str`, `v.int`, `v.bool`, `v.list`, `v.nil`):

```python
from nidus import v

db.upsert("docs", [{"id": "d", "attrs": {"tags": v.list(["a", "b"]), "rank": v.int(7)}}])
```

`v.nil()` is the explicit `Null` value — "set, and empty" — which is a different fact
from an absent key ("not set / not indexed"). The SDK keeps them apart.

## Filtering

Build an AND-filter with the `f.*` helpers. Each predicate is a positive assertion about
a **present** attribute, so an absent key matches nothing — including the negative
predicates (`ne`, `not_in`) and the ranges.

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
The three trailing underscores are not style — `in` and `and` are reserved words in
Python, so `f.in_`, `f.not_in`, and `f.and_` are the JS SDK's `f.in`, `f.notIn`, and
`f.and`. Nothing else deviates.

A `Filter` is just a `list` of predicates, AND-combined, so `f.and_(...)` is sugar for
building that list — `filter=[f.eq("lang", "rust")]` is equally valid.

Comparisons are same-type only (int↔int numeric, str↔str lexical, bool↔bool). A range
predicate against a mismatched type matches nothing, which is the usual reason a filter
mysteriously returns no rows.

## Full-text and hybrid search

```python
db.set_fts_schema("docs", ["body"])
# Per-field tuning: db.set_fts_schema("docs", [{"field": "body", "k1": 1.5}])

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

`hybrid_search` takes no `min_score`: its score is a fused RRF rank, not a similarity,
so there is no meaningful floor to set. `rrf_k` and `candidates` tune the fusion.

## Remembering and recalling (text-native)

When the server is started with an embedder (`nidus serve --embed-provider …`) you can
send **text** and let the server embed it — no need to compute vectors client-side.
`remember` embeds and upserts; `recall` embeds the query and vector-searches.

```python
# Embed "the quick brown fox" and store it under id "a"
db.remember("notes", "a", "the quick brown fox", attrs={"tag": "x"})

# Summarize first, then embed the summary (the server also needs --summarize-provider).
# The stored record additionally carries `nidus.summary` and `nidus.source` attrs.
db.remember("notes", "b", long_article, mode="summarize")

# Embed the query text and search, best first
hits = db.recall("notes", "quick fox", top_k=5, min_score=0.2, filter=[f.eq("tag", "x")])
```

Both raise `NidusError` with status `400` against a server that has **no embedder
configured** (the message names `--embed-provider`), and `mode="summarize"` without a
summarizer is likewise a `400`. The client only ever sends text; the embedding always
happens server-side.

## Everything else

Every endpoint of the HTTP API has a method:

```python
db.collections()                    # list[str]
db.stats()                          # dimension, distance, ANN config, collections, footprint
db.list(scope=["docs"], filter=[f.eq("lang", "rust")], offset=0, limit=50)
db.records("docs")                  # every record, attrs decoded; vector is None for text-only
db.get_meta("docs"); db.set_meta("docs", {"owner": "search-team"})
db.delete("docs", ["a"])            # by id
db.delete_where("docs", f.and_(f.lt("year", 2000)))
db.flush(); db.compact()
db.drop_collection("docs")
db.health()                         # bool
```

Optional arguments all default to `None`, which means "omit the key" so the **server's**
default applies (`top_k = 10`, `limit = 100`, `rrf_k = 60.0`, `candidates = 100`). Those
numbers are deliberately not restated in Python, and it is why the defaults are `None`
rather than a number: `top_k=0` is a legitimate request for zero results, so `0` cannot
double as "unset".

`stats().ann` is `None` when the store does exact brute-force search, rather than an
`AnnInfo` full of defaults. Likewise `Record.vector` is `None` — never `[]` — for a
text-only document.

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
connection per request**. For interactive use that is invisible; for a long run of
sequential upserts the handshakes are measurable overhead.

The escape hatch is `transport=` — a callable
`(method, url, headers, body, timeout) -> (status, text)`. Hand in one backed by `httpx`
or `requests` and you get pooling, keep-alive, retries, or instrumentation without the
SDK taking on a dependency for everyone:

```python
import httpx
from nidus import NidusClient

class PooledTransport:
    """A connection-pooling transport for bulk ingest."""

    def __init__(self) -> None:
        self._client = httpx.Client()

    def __call__(self, method, url, headers, body, timeout):
        # A transport RETURNS non-2xx statuses; only a failure to get any response
        # at all should raise. httpx already behaves that way.
        r = self._client.request(method, url, content=body, headers=headers, timeout=timeout)
        return r.status_code, r.text

    def close(self) -> None:
        # NidusClient.close() calls close() on the transport if it has one, so
        # `with NidusClient(...)` shuts the pool down too.
        self._client.close()

with NidusClient("http://127.0.0.1:7700", transport=PooledTransport()) as db:
    for batch in batches:
        db.upsert("docs", batch)
```

The same seam is what lets the SDK's own unit tests exercise every endpoint with no
server and no socket. `AsyncNidusClient` takes the natural equivalent for its own stack —
`transport=` there is an `httpx.AsyncBaseTransport` (a pre-tuned pool, or an
`httpx.MockTransport` for tests); it pools by default, so nothing extra is needed for
bulk ingest.

## Errors

A failed request raises `NidusError` carrying the HTTP status the server reported, so
you can tell a client fault from a server fault:

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

Value errors are raised locally, before any request: a `float` attribute or a
non-string list element is a `TypeError`, and an integer outside `i64` is a `ValueError`
(Python's ints are unbounded; the store's `Int` is not).

## Documentation

Full documentation: <https://nidus.duckedup.org/sdks/python/>

## License

MIT
