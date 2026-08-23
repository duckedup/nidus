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

The Python type of an attribute decides its nidus type, and the pair that matters is
`Int` vs `Float`: `2` is an `Int`, `2.0` is a `Float`. They are separate types on the
server and comparisons are same-type only, so a `Float` attribute filtered with an `int`
operand matches nothing — keep a numeric field's Python type uniform across records.
`nan` and `inf` are refused (`ValueError`): JSON cannot spell them.

A `datetime` becomes a `DateTime` — a UTC instant carried as epoch **milliseconds**, so
sub-millisecond precision is truncated and the timezone is not stored. It must be
**aware**; a naive `datetime` raises `ValueError` rather than being assumed to be UTC,
because the wrong guess is off by hours in valid-looking JSON. Reading one back gives an
aware `datetime`, not an `int`, so a decoded `attrs` map re-encodes to what it came from.

For an explicit type, use the `v.*` helpers (`v.str`, `v.int`, `v.float`, `v.bool`,
`v.list`, `v.datetime`, `v.nil`):

```python
from datetime import datetime, timezone

from nidus import v

db.upsert("docs", [{"id": "d", "attrs": {
    "tags": v.list(["a", "b"]),
    "rank": v.int(7),
    "score": v.float(1),                      # a whole number, stored as a Float
    "seen": v.datetime(datetime.now(timezone.utc)),
}}])
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

Predicates: `eq`, `ne`, `glob`, `iglob`, `in_`, `not_in`, `lt`, `le`, `gt`, `ge`,
`contains`, `not_contains`, `contains_any`; the text ones `fuzzy`, `contains_all_tokens`,
`contains_any_token`, `contains_token_sequence`, `regex`; and the groups `all_`, `any_`,
`not_`, plus `and_`. The trailing underscores are not style — `in`, `and` and `not` are
reserved words in Python and `all`/`any` shadow builtins, so `f.in_`, `f.not_in`, `f.and_`,
`f.all_`, `f.any_` and `f.not_` are the JS SDK's `f.in`, `f.notIn`, `f.and`, `f.all`,
`f.any` and `f.not`. Nothing else deviates.

```python
f.iglob("path", "Src/*")                     # glob, ASCII case folded on both sides
f.regex("path", "src/.*[.]rs")               # anchored at BOTH ends, like glob
f.fuzzy("title", "levenshtein", 2)           # within N edits (N > 8 is an error)
f.contains_token_sequence("body", "async runtime")   # a phrase, in order
```

The text predicates tokenize (ASCII-case-folded runs of alphanumerics), so case and
punctuation do not count, and each of them matches a list attribute when any single
element does.

A `Filter` is just a `list` of predicates, AND-combined, so `f.and_(...)` is sugar for
building that list — `filter=[f.eq("lang", "rust")]` is equally valid.

Comparisons are same-type only (int↔int numeric, float↔float by IEEE, str↔str lexical,
bool↔bool, datetime↔datetime as instants). A range predicate against a mismatched type
matches nothing, which is the usual reason a filter mysteriously returns no rows — and
`f.gt("score", 2)` against a `Float` attribute is exactly that mismatch.

## Indexing the text predicates

`f.fuzzy`, `f.contains_all_tokens`, `f.contains_any_token`, `f.contains_token_sequence` and
`f.regex` are scanned per record by default. Declaring a filter index makes them a lot
faster and changes no results at all: the index proposes candidates and the predicate still
decides.

```python
db.set_filter_index("docs", ["body"])
# Per-field: only the token predicates on `tag`, no fuzzy or regex.
db.set_filter_index("docs", ["body", {"field": "tag", "trigrams": False}])
# An empty list drops it.
db.set_filter_index("docs", [])
```

It is opt-in per collection and per field, and it costs write time and memory. Documents
already written are indexed as part of the call. `AsyncNidusClient` has the same method.

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
so there is no meaningful floor to set. `rrf_k` and `candidates` tune the fusion, and
`vector_weight`/`text_weight` scale each leg's contribution to it (both default to 1.0,
which is the unweighted fusion exactly).

A query may name several fields instead of one, each with its own text, and ask *why* a
document matched:

```python
hits = db.text_search(
    clauses=[{"field": "title", "query": "rust"}, {"field": "body", "query": "async runtime"}],
    combine="Sum",      # or "Max" — Sum rewards matching in two fields, Max takes the best
    explain=True,       # each matched clause's own BM25 score
    highlight=True,     # or {"max_fragments": 2, "fragment_chars": 80}
)

for hit in hits:
    for clause in hit.annotations.clauses:
        print(clause.field, clause.score)
    for highlight in hit.annotations.highlights:
        for fragment in highlight.fragments:
            start, end = fragment.spans[0]
            print(fragment.text.encode()[start:end])   # spans are BYTE offsets into `text`
```

`field`+`query` and `clauses` are mutually exclusive, and an empty clause list is refused
at the call site rather than answered as "no matches". `hit.annotations` is `None` unless
`explain` or `highlight` asked for it — on a hybrid search it also carries each leg's own
`rank` and `score`, which is the only way to see a leg's rank, since the returned score is
the fused one.

`prefix=True` expands a clause's **final** term as a prefix instead of an exact stem
match, for typeahead:

```python
# Matches "running", "runtime", … — every indexed term "run" is a prefix of.
db.text_search(field="title", query="run", prefix=True)

# On the clauses spelling it is set per clause, not on the call as a whole.
db.text_search(clauses=[{"field": "title", "query": "run", "prefix": True}])
```

Only the clause's last term expands; earlier terms still need an exact stem match. The
index holds stems, so `"runn"` will not prefix-match indexed `"run"` — the fragment itself
is not stemmed, only fold-normalized (lowercased, accent-stripped).

## Query plans

`search`, `search_similar`, and `hybrid_search` each have a `_with_plan` sibling that
returns `(hits, plan)` instead of just `hits`, describing how the query actually ran:

```python
hits, plan = db.search_with_plan(query=[0.1, 0.2, 0.3], top_k=10)
print(plan.path, plan.timings.total_us)
if plan.candidates is not None:
    print(plan.candidates.surfaced, plan.candidates.survived)
```

`plan.path` is a plain string (`"ann"`, `"ann_prefilter_fallback"`, `"segmented"`,
`"quantized"`, or `"exact"`), not an enum, so a value a newer server introduces still
decodes rather than raising. `plan.rows_scanned` and `plan.candidates` are `None` when they
do not apply to the path taken; every `plan.timings` field is in microseconds and `None`
when that phase did not run, except `total_us`, which always does. `text_search` has no
`_with_plan` sibling: it has no plan to report.

## Remembering and recalling (text-native)

When the server is started with an embedder (`nidus serve --embed-provider …`) you can
send **text** and let the server embed it — no need to compute vectors client-side.
`remember` embeds and upserts; `recall` embeds the query and vector-searches.

```python
# Embed "the quick brown fox" and store it under id "a"
db.remember("notes", "a", "the quick brown fox", attrs={"tag": "x"})

# Summarize first, then embed the summary (the server also needs --summarize-provider).
# The stored record additionally carries a `nidus.summary` attr; the raw text is
# always stored under `nidus.text`.
db.remember("notes", "b", long_article, mode="summarize")

# Expire in an hour, and fold this write onto any entry it is >=0.95 similar to rather
# than storing a competing near-duplicate.
out = db.remember("notes", "c", "the quick brown fox", ttl_seconds=3600, dedupe_threshold=0.95)

# Embed the query text and search, best first
hits = db.recall("notes", "quick fox", top_k=5, min_score=0.2, filter=[f.eq("tag", "x")])

# reinforce=True bumps each returned entry's nidus.access_count (a write, so it needs a
# writable server) and extend_ttl_seconds pushes an existing expiry further out.
db.recall("notes", "quick fox", reinforce=True, extend_ttl_seconds=3600)
```

`remember` returns a `RememberResult` — `id`, `upserted`, `deduped`. Read `id` from it
rather than assuming the one you passed: a `dedupe_threshold` match redirects the write
onto the entry it matched, and that entry's id is the one that changed. An already-expired
entry is never a dedupe candidate, so a TTL that has run out cannot be revived by a later
near-duplicate.

Both raise `NidusError` with status `400` against a server that has **no embedder
configured** (the message names `--embed-provider`), and `mode="summarize"` without a
summarizer is likewise a `400`. Dedupe needs that same embedder — it is a vector search
under the hood. The client only ever sends text; the embedding always happens server-side.

## Everything else

Every endpoint of the HTTP API has a method:

```python
db.collections()                    # list[str]
db.stats()                          # dimension, distance, ANN config, collections, footprint
db.list(scope=["docs"], filter=[f.eq("lang", "rust")], offset=0, limit=50)
db.list(order_by={"field": "updated_at", "descending": True})   # sort by an attribute
db.aggregate(scope=["docs"], sum=["bytes"])   # -> Aggregation(count=…, sums={"bytes": …})
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
text-only document. `aggregate` is answered from the in-RAM index alone (no record is
built, no vector is read), and its sums keep the server's type: a run of `Int`s is an
`int`, a run that met one `Float` is a `float`.

### Ranking

`search` and `text_search` take four more knobs. `rank_by` layers a ranking expression over
the metric, `limit_per` caps how many hits may share one attribute value, `diversity`
spreads the page apart in vector space so near-duplicates stop filling it, and `expand`
widens each hit with the neighbouring chunks of its own document:

```python
from datetime import datetime, timedelta, timezone

from nidus import rank

hits = db.search(
    query=[0.1, 0.2, 0.3],
    rank_by=rank.decay("updated_at", datetime.now(timezone.utc), scale=timedelta(days=7)),
    limit_per={"field": "path", "max": 2},   # at most 2 hits per file
    diversity=0.5,                           # balance relevance against variety
)

# Read a chunked corpus as documents: the best chunk per file, widened with its
# neighbours into `hit.context`. Payload only, so the ranking is unchanged.
passages = db.search(
    query=[0.1, 0.2, 0.3],
    limit_per={"field": "nidus.parent_id", "max": 1},
    expand={"radius": 1},
)

# On recall the same pair has one text-native spelling.
hits = db.recall("docs", "how does the writer lock work", rollup={"neighbours": 1})
```

`diversity` is a Maximal Marginal Relevance lambda: `1.0` is pure relevance (the ranking you
get without it), `0.0` is pure variety, and values between trade one against the other. It
reorders a bounded window of candidates, so it never deepens the scan without limit.

Decay **subtracts** `lambda_ * (1 - factor)` from the base score rather than multiplying
it, so it stays meaningful where scores are negative or unbounded (Euclidean, dot product,
BM25). Ages are measured back from the `origin` you pass, never from the wall clock, so
the same query against an unchanged store ranks the same way twice. `scale` is a half-life
by default, and a record whose timestamp is missing or unusable is **not** penalized
(`missing` defaults to 1.0). `origin` takes an aware `datetime` or epoch milliseconds and
`scale` a `timedelta` or milliseconds; `lambda_` carries the underscore because `lambda`
is a reserved word, and travels as `lambda`.

`count_field`, `count_scale`, and `count_lambda` add a second, independent term that reads
a reinforcement count off `count_field` (see `reinforce` below) and subtracts a penalty the
same way: a much-recalled entry pays less, never more, and the term applies only when
`count_field` is set.

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

Value errors are raised locally, before any request: an attribute type the store has no
variant for (or a non-string list element) is a `TypeError`, and an integer outside `i64`,
a non-finite `float`, or a naive `datetime` is a `ValueError`.

## Documentation

Full documentation: <https://nidus.duckedup.org/sdks/python/>

## License

MIT
