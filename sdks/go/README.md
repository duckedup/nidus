# github.com/duckedup/nidus/sdks/go

The Go client for [nidus](https://nidus.duckedup.org) — a small, fast vector store.
This module connects to a running `nidus serve` instance over HTTP, whether it's on
your laptop or a remote host.

It is a **remote client**: zero dependencies, standard library only (`net/http`,
`encoding/json`, `context`), so there is no `go.sum` beside `go.mod` and nothing that
can pull a transitive surprise into your binary.

```sh
go get github.com/duckedup/nidus/sdks/go
```

The module path ends in `go`, but the package is `nidus` — so imports read the way you
want and need no alias (a final path element that differs from the package name is
legal Go; some editors will nonetheless offer to add one):

```go
import "github.com/duckedup/nidus/sdks/go"

db, err := nidus.NewClient("http://127.0.0.1:7700")
```

This module is versioned in lockstep with nidus itself, so `sdks/go@v0.39.0` is the
client for nidus `0.39.x`. Match the two and the wire contract lines up. Because the
module lives in a repository subdirectory, Go resolves it by a tag carrying that
prefix — `sdks/go/v0.39.0` — which is what `@v0.39.0` finds:

```sh
go get github.com/duckedup/nidus/sdks/go@v0.39.0
```

## Connecting

"Local vs remote" is just the base URL — point the client at a local `nidus serve` or
any reachable host. Every method takes a `context.Context` first and returns
`(T, error)`; cancellation and deadlines stay with the caller.

```go
// Local
db, err := nidus.NewClient("http://127.0.0.1:7700")

// Remote, with the bearer token the server was started with (`nidus serve --token`)
db, err := nidus.NewClient(
    "https://nidus.internal.example.com",
    nidus.WithToken(os.Getenv("NIDUS_TOKEN")),
    nidus.WithTimeout(5*time.Second), // applied as a per-request context deadline
)
```

A `*Client` is safe for concurrent use and holds the connection pool, so build one per
server and share it. `WithHTTPClient` takes your own `*http.Client` for a custom
transport, proxy, TLS config, retry wrapper, or instrumentation; `WithHeader` adds a
header sent on every request.

## Upserting and searching

Attributes are typed: `nidus.Str`, `nidus.Int`, `nidus.Float`, `nidus.Bool`,
`nidus.List`, `nidus.DateTime`, `nidus.Null`. `Int` and `Float` are separate types on
the server and comparisons are same-type only, so `nidus.Int(2)` never matches a
`Float` attribute; `nidus.ValueOf` decides between them from the Go type, which means
`float64(2024)` is a `Float`. A `DateTime` is a UTC instant carried as epoch
milliseconds — `nidus.DateTime(t)` from a `time.Time`, or `nidus.DateTimeMillis(ms)`
if you already have the count. NaN and ±Infinity are refused: JSON cannot spell them.

```go
ctx := context.Background()

if err := db.CreateCollection(ctx, "docs"); err != nil { /* … */ }

n, err := db.Upsert(ctx, "docs", []nidus.Record{
    {ID: "a", Vector: []float32{0.1, 0.2, 0.3},
        Attrs: nidus.Attrs{"lang": nidus.Str("rust"), "year": nidus.Int(2024),
            "score": nidus.Float(0.75), "seen": nidus.DateTime(time.Now())}},
    {ID: "b", Vector: []float32{0.4, 0.5, 0.6},
        Attrs: nidus.Attrs{"lang": nidus.Str("go"), "year": nidus.Int(2023)}},
    // a text-only doc — omit the vector
    {ID: "c", Attrs: nidus.Attrs{"body": nidus.Str("vector stores are neat")}},
})

hits, err := db.Search(ctx, nidus.SearchRequest{Query: []float32{0.1, 0.2, 0.3}, TopK: 5})
for _, hit := range hits {
    lang, _ := hit.Attrs["lang"].Str()
    fmt.Println(hit.ID, hit.Score, lang)
}
```

Leave `TopK` at zero to take the server's default (10) — a zero is omitted from the
request rather than sent, because `"top_k": 0` would be a request for no results. The
knobs whose zero the server *does* treat as a real value are pointers instead, so an
explicit zero can travel: `MinScore` (`nil` is "no floor", `&0` is a floor of exactly
zero), and hybrid search's `RRFK`, `Candidates`, `VectorWeight` and `TextWeight`.

A `Record` with a `nil` `Vector` is a text-only document. A non-nil but *empty* Vector is
refused by `Upsert` rather than sent: an empty slice encodes identically to an absent one
in Go, so it would silently store a text-only document that no vector search can see.

```go
floor := float32(0.2)
hits, err := db.Search(ctx, nidus.SearchRequest{
    Query:    []float32{0.1, 0.2, 0.3},
    Scope:    []string{"docs"}, // empty Scope searches every collection, one ranking
    MinScore: &floor,
})
```

If your attributes arrive as plain Go values — from your own JSON decode, say — use
`nidus.AttrsOf(map[string]any{…})` (or `nidus.ValueOf` for one value), which normalizes
and names the offending key when a value has no nidus type.

## Typed attributes on the way back

`Hit.Attrs` and `Record.Attrs` keep typed `Value`s rather than decoding to `any`. Read
one with the comma-ok accessors:

```go
lang, ok := hit.Attrs["lang"].Str()      // "", false if absent or not a string
year, ok := hit.Attrs["year"].Int()      // full int64 precision, never rounded via float64
tags, ok := hit.Attrs["tags"].List()
score, ok := hit.Attrs["score"].Float()  // an Int does NOT widen into this accessor
seen, ok := hit.Attrs["seen"].DateTime() // a time.Time in UTC
```

This is a deliberate deviation from the JavaScript SDK, which decodes attrs to plain JS
values: in a statically typed language the typed accessor is the better surface —
`hit.Attrs["lang"].Str()` beats an `any` and a type assertion, and a wrong-type read
gives you a testable `false` instead of a plausible-looking empty string. When you do
want the loose map, ask for it:

```go
plain := hit.Attrs.Decode() // map[string]any: string, int64, float64, bool, []string,
                            // time.Time, nil
```

## Filtering

Build an AND-filter with the predicate constructors. Each predicate is a positive
assertion about a **present** attribute — an absent key matches nothing, including the
negative predicates.

```go
hits, err := db.Search(ctx, nidus.SearchRequest{
    Query: []float32{0.1, 0.2, 0.3},
    TopK:  10,
    Filter: nidus.And(
        nidus.Eq("lang", "rust"),
        nidus.Ge("year", 2020),
        nidus.In("status", "published", "draft"),
        nidus.Glob("path", "src/*"),
    ),
})
```

Predicates: `Eq`, `Ne`, `Glob`, `IGlob`, `In`, `NotIn`, `Lt`, `Le`, `Gt`, `Ge`. `IGlob`
is `Glob` with ASCII case folded on both sides. They take `any`
so `nidus.Eq("year", 2024)` reads naturally; a value the store has no type for (a
`[]int`, say, or a NaN) is remembered on the predicate and surfaces as an ordinary error
from the call that used the filter. Check it earlier with `Predicate.Err()` /
`Filter.Err()`. Comparisons are same-type only, so give a range the same type the
attribute was written with — `Ge("score", 0.5)` for a `Float`, `Ge("year", 2020)` for an
`Int`, `Ge("seen", t)` for a `DateTime`.

Lists: `Contains`, `NotContains`, `ContainsAny`. Groups: `All`, `Any`, `Not` — `Not` is
genuine complement, so `Not(Eq(k, v))` matches a record with no `k` at all where
`Ne(k, v)` does not.

Text, over any string attribute (no full-text schema needed):

```go
nidus.Fuzzy("title", "serach", 2)              // within 2 edits, ASCII case folded
nidus.ContainsAllTokens("body", "async runtime") // both words, any order
nidus.ContainsAnyToken("body", "async runtime")  // either word
nidus.ContainsTokenSequence("body", "async runtime") // the phrase, in order
nidus.Regex("path", "^src/.*\\.rs")            // anchored BOTH ends, like Glob
```

`Fuzzy`'s edit budget is `0..8`; anything outside that is carried as a predicate error
like an unnormalizable value. `Regex` is compiled server-side and anchored at both ends,
so use `.*` to opt into a substring search — and an unparseable pattern comes back as a
request error, not a Go one.

## Indexing the text predicates

Fuzzy, ContainsAllTokens, ContainsAnyToken, ContainsTokenSequence and Regex are scanned per
record by default. Declaring a filter index makes them a lot faster and changes no results
at all: the index proposes candidates and the predicate still decides.

```go
if err := db.SetFilterIndex(ctx, "docs", []string{"body"}); err != nil { /* … */ }
// SetFilterIndexFields is the same call with per-field control over which structures
// are built, so a tag field can skip the trigrams that serve Fuzzy and Regex.
```

It is opt-in per collection and per field, and it costs write time and memory. Documents
already written are indexed as part of the call; passing no fields drops the declaration.

## Full-text and hybrid search

```go
if err := db.SetFtsSchema(ctx, "docs", []string{"body"}); err != nil { /* … */ }
// SetFtsFields is the same call with per-field BM25/analyzer tuning.

// BM25 text search. Scores are raw BM25 — unbounded, not comparable across queries.
text, err := db.TextSearch(ctx, nidus.TextSearchRequest{
    Field: "body", Query: "vector store", TopK: 10,
})

// Fuse vector + text via reciprocal rank fusion; the score is the fused RRF score.
hybrid, err := db.HybridSearch(ctx, nidus.HybridSearchRequest{
    Vector: []float32{0.1, 0.2, 0.3},
    Field:  "body",
    Text:   "vector store",
    TopK:   10,
})
```

`RRFK` and `Candidates` are `*float32` / `*int`: leave them `nil` for the server's
defaults (60 and 100). They are pointers because zero is a real request for both — the
server fuses with `1/(rrf_k + rank + 1)`, so `RRFK: &zero` is the maximally top-heavy
weighting, and `Candidates: &zero` fuses exactly `TopK` deep with no over-fetch. So are
`VectorWeight` and `TextWeight`, which scale each leg's contribution and default to
`1.0`: a weight of `&zero` drops that leg entirely, which a plain `float32` could not
tell apart from "unset".

Query several fields at once with `Clauses` instead of `Field`+`Query` (or `Field`+`Text`
on hybrid) — one or the other, never both:

```go
hits, err := db.TextSearch(ctx, nidus.TextSearchRequest{
    Clauses: []nidus.FtsClause{
        {Field: "title", Query: "rust"},
        {Field: "body", Query: "async runtime"},
    },
    Combine:   nidus.CombineMax, // or CombineSum, the default: add every matched clause
    Explain:   true,
    Highlight: &nidus.HighlightOpts{}, // zero value = the server's defaults
})

for _, hit := range hits {
    for _, c := range hit.Annotations.Clauses {
        fmt.Println(c.Field, c.Score)
    }
    for _, h := range hit.Annotations.Highlights {
        for _, f := range h.Fragments {
            for _, s := range f.Spans {
                fmt.Println(h.Field, f.Text[s.Start:s.End]) // spans are byte offsets
            }
        }
    }
}
```

`Hit.Annotations` is `nil` unless you asked for `Explain` or `Highlight`. On a hybrid
search `Explain` additionally fills `Annotations.Vector` and `Annotations.Text` with each
leg's own rank and score, before fusion flattened them into one number.

Set `Prefix` (on a clause, or on the `Field`+`Query`/`Field`+`Text` shorthand) to match
a truncated final term as a prefix instead of an exact word, for typeahead:

```go
on := true
hits, err := db.TextSearch(ctx, nidus.TextSearchRequest{
    Field: "body", Query: "sched", Prefix: &on, // matches "schedules", "scheduler", …
})
```

`Prefix` is a `*bool` so an unset (`nil`) field is dropped from the request entirely;
only the clause's or shorthand's own *final* term expands, and only against the index's
stemmed vocabulary (so `"runn"` will not find `"run"`).

## Reshaping the ranking

```go
// Prefer recent documents: subtract a penalty that grows with age, halving every week.
hits, err := db.Search(ctx, nidus.SearchRequest{
    Query: []float32{0.1, 0.2, 0.3},
    RankBy: nidus.DecayRank(nidus.Decay{
        Field:  "updated_at",              // a DateTime or Int attr, epoch ms
        Origin: time.Now().UnixMilli(),    // "now"; ages are measured back from here
        Scale:  7 * 24 * 60 * 60 * 1000,   // 0 takes the server's default of a week
    }),
    LimitPer: &nidus.LimitPer{Field: "path", Max: 2}, // at most 2 hits per file
})

// Read a chunked corpus as documents: the best chunk per file, widened with its
// neighbours into Hit.Context. Payload only, so the ranking is unchanged.
passages, err := db.Search(ctx, nidus.SearchRequest{
    Query:    []float32{0.1, 0.2, 0.3},
    LimitPer: &nidus.LimitPer{Field: "nidus.parent_id", Max: 1},
    Expand:   &nidus.Expand{Radius: 1},
})

// On recall the same pair has one text-native spelling.
hits, err = db.Recall(ctx, "docs", "how does the writer lock work", nidus.RecallOptions{
    Rollup: &nidus.Rollup{Neighbours: 1},
})

// Sort a metadata query by an attribute instead of storage order.
rows, err := db.List(ctx, nidus.ListRequest{
    OrderBy: &nidus.OrderBy{Field: "updated_at", Descending: true},
})
```

The decay penalty is `Lambda * (1 - Decay^(age/Scale))` and is **subtracted** from the
base score, never multiplied, so it stays meaningful where scores are negative or
unbounded (Euclidean, DotProduct, BM25). Ages are measured from `Origin` rather than the
wall clock, so the same query against an unchanged store ranks the same way twice. A
record whose timestamp is missing or unusable is *not* penalized by default; set
`Missing: &zero` to bury it instead. `RankBy`, `LimitPer` and `Expand` ride on `/search`
and `/text-search`; `OrderBy` on `/list`; `Rollup` on `/recall`.

## Explaining a query plan

`Search`, `SearchSimilar` and `HybridSearch` each have a `WithPlan` sibling that
returns a `*nidus.QueryPlan` alongside the hits: which code path answered the query,
how many rows it scanned, and where candidates were dropped along the way. The plain
methods never send the extra request field, so they stay byte-identical to before.

```go
hits, plan, err := db.SearchWithPlan(ctx, nidus.SearchRequest{
    Query: []float32{0.1, 0.2, 0.3},
})
fmt.Println(plan.Path) // e.g. "ann_prefilter_fallback"
if plan.Candidates != nil {
    fmt.Println(plan.Candidates.DroppedFiltered)
}
```

`RowsScanned` and `Candidates` are `nil`, not zero, when the path that ran doesn't
produce them; `QueryPlan.Path` and `PlanNarrowing.State` are plain strings rather than
closed enums, since a newer server may report a value this SDK predates.

## Remembering and recalling (text-native)

When the server is started with an embedder (`nidus serve --embed-provider …`), you can
send **text** and let the server embed it — no vectors client-side. `Remember` embeds
and upserts; `Recall` embeds the query and vector-searches.

```go
// Embed "the quick brown fox" and store it under id "a"
out, err := db.Remember(ctx, "notes", "a", "the quick brown fox",
    nidus.RememberOptions{Attrs: nidus.Attrs{"tag": nidus.Str("x")}})

// Summarize first, then embed the summary (the server also needs
// --summarize-provider). The stored record additionally carries nidus.summary
// attr; the raw text is always stored under nidus.text.
out, err = db.Remember(ctx, "notes", "b", longArticle,
    nidus.RememberOptions{Mode: "summarize"})

// Expire in an hour, and fold this write onto any entry it is >=0.95 similar to
// rather than storing a competing near-duplicate. out.ID is the record that
// actually changed — the match's id, not "c", whenever out.Deduped is true.
ttl, floor := int64(3600), float32(0.95)
out, err = db.Remember(ctx, "notes", "c", "the quick brown fox",
    nidus.RememberOptions{TTLSeconds: &ttl, DedupeThreshold: &floor})

// Embed the query text and search, best-first
hits, err := db.Recall(ctx, "notes", "quick fox", nidus.RecallOptions{
    TopK:     5,
    MinScore: &floor,
    Filter:   nidus.And(nidus.Eq("tag", "x")),
})

// Reinforce records that these hits proved useful: the server stamps
// nidus.access_count and nidus.last_accessed, and pushes any existing
// nidus.expires_at out by ExtendTTLSeconds. This makes the recall a write,
// so it takes the writer lock and is refused on a read-only server.
extend := int64(3600)
hits, err = db.Recall(ctx, "notes", "quick fox", nidus.RecallOptions{
    Reinforce: true, ExtendTTLSeconds: &extend,
})
```

Both knobs are pointers because zero means something in each: a `TTLSeconds` of `0`
expires the entry immediately, and a `DedupeThreshold` of `0` matches *any* entry rather
than disabling dedupe. Dedupe is a vector search server-side, so it needs the same
embedder `Remember` does, and an already-expired entry is never a candidate — a lapsed
TTL cannot be revived by a later near-duplicate.

Two different failures are worth telling apart when these do not work:

- **`404`, with no message** — the server binary was built without the `memory` feature,
  so `/remember` and `/recall` are not routes it has at all. That is what a plain
  `cli`-feature build (`just build-cli`) produces.
- **`400`** — the routes exist but the server was started without `--embed-provider`; the
  message names the flag. `Mode: "summarize"` without a summarizer is likewise a `400`.

Recalling a collection that was written with a different embedding model is a `409` rather
than a silently meaningless ranking.

## Everything else

```go
ok := db.Health(ctx)                       // bool — no token needed; "is it up", one answer
err = db.Ping(ctx)                         // the same call, keeping the reason it failed
stats, err := db.Stats(ctx)                // dimension, distance, ANN config, footprint
names, err := db.Collections(ctx)          // []string
hits, err := db.List(ctx, nidus.ListRequest{
    Scope:  []string{"docs"},
    Filter: nidus.And(nidus.Eq("lang", "rust")),
})                                          // metadata-only, paginated by Offset/Limit
agg, err := db.Aggregate(ctx, nidus.AggregateRequest{
    Scope: []string{"docs"},
    Sum:   []string{"bytes"},
})                                          // agg.Count, agg.Sums["bytes"].Int()
recs, err := db.Records(ctx, "docs")       // every record; Vector is nil for text-only
meta, err := db.GetMeta(ctx, "docs")
err = db.SetMeta(ctx, "docs", map[string]string{"owner": "search-team"}) // replaces, not merges
n, err := db.Delete(ctx, "docs", []string{"a"})
n, err = db.DeleteWhere(ctx, "docs", nidus.And(nidus.Lt("year", 2000))) // empty filter = all
err = db.Flush(ctx)
err = db.Compact(ctx)
err = db.DropCollection(ctx, "docs")
```

## Errors

A failed request returns a `*nidus.Error` carrying the HTTP status the server reported,
so you can tell a client fault from a server fault:

```go
if _, err := db.Upsert(ctx, "docs", records); err != nil {
    var nerr *nidus.Error
    if errors.As(err, &nerr) {
        switch {
        case nerr.IsBadRequest():     // 400/422 — the request is wrong; retrying cannot help
        case nerr.IsUnauthorized():   // 401 — missing or wrong bearer token
        case nerr.IsReadOnly():       // 403 — a write against a read-only store
        case nerr.IsLocked():         // 409 — the writer lock is held elsewhere
        case nerr.IsUnavailable():    // 503 — shed under backpressure, or store not open
        case nerr.IsOutOfCapacity():  // 507 — the store refused to grow; it is intact
        }
        log.Println(nerr.Status, nerr.Message)
    }
}
```

`IsBadRequest()` covers `400` **and** `422`, because they are one thing to a caller: the
request itself is wrong. The split is the server's HTTP layer — a JSON *syntax* error (and
the store's own client faults, like a dimension mismatch) is a `400`, while a body whose
*types* do not deserialize (`TopK: -1`) is a `422`. Retrying either forever is the bug this
grouping prevents. `409` and `503` are the two that a retry with backoff is the right
answer to.

`IsTransport()` (status `0`) means the request never got an answer at all — unreachable
server, timeout, cancelled context. Unlike the status-carrying cases it says nothing
about whether the write was applied: a timeout can fire after the server committed.

## License

MIT
