---
title: Go SDK
description: "github.com/duckedup/nidus/sdks/go: the official Go client for nidus. Connect to a local or remote nidus server over HTTP, upsert, and search."
---

[`github.com/duckedup/nidus/sdks/go`](https://pkg.go.dev/github.com/duckedup/nidus/sdks/go)
is the official Go client for nidus. It drives a running
[`nidus serve`](/guides/http-server/) instance over HTTP, local or remote.

It has **zero dependencies**: standard library only (`net/http`, `encoding/json`,
`context`), so there is no `go.sum` beside its `go.mod` and nothing that can pull a
transitive surprise into your binary.

```sh
go get github.com/duckedup/nidus/sdks/go
```

The module path ends in `go`, but the package is `nidus`, so imports read the way you
want and need no alias: a final path element that differs from the package name is
legal Go, though some editors will offer to add one anyway:

```go
import "github.com/duckedup/nidus/sdks/go"

db, err := nidus.NewClient("http://127.0.0.1:7700")
```

The SDK is versioned in lockstep with nidus, so `sdks/go@v0.39.0` is the client for
nidus `0.39.x`. Match the two and the wire contract lines up. Because the module lives
in a repository subdirectory, Go resolves it by a tag carrying that prefix
(`sdks/go/v0.39.0`), which is exactly what `@v0.39.0` finds:

```sh
go get github.com/duckedup/nidus/sdks/go@v0.39.0
```

## Connecting

"Local vs remote" is just the base URL: point the client at a local `nidus serve` or
any reachable host. When the server was started with a
[token](/guides/http-server/), pass it with `WithToken`.

Every method takes a `context.Context` first and returns `(T, error)`: cancellation and
deadlines belong to the caller, not to the SDK.

```go
// Local
db, err := nidus.NewClient("http://127.0.0.1:7700")

// Remote, with bearer-token auth
db, err := nidus.NewClient(
    "https://nidus.internal.example.com",
    nidus.WithToken(os.Getenv("NIDUS_TOKEN")),
    nidus.WithTimeout(5*time.Second), // optional per-request timeout
)
```

`WithTimeout` is applied as a context deadline per request, so it composes with a
caller's own context rather than overriding it; whichever deadline is earlier wins. A
`*Client` is safe for concurrent use and owns the connection pool, so build one per
server and share it. `WithHTTPClient` accepts your own `*http.Client` (custom transport,
proxy, TLS, retries, instrumentation); `WithHeader` adds a header sent on every request.

## Upserting and searching

Attributes are typed: `nidus.Str`, `nidus.Int`, `nidus.Float`, `nidus.Bool`,
`nidus.List`, `nidus.DateTime`, `nidus.Null`. The constructors keep an unstorable
attribute unrepresentable instead of turning it into a `400` at request time.

Go decides `Int` vs `Float` from the **static type**, so `2.0` (a `float64`) is a
`Float` and `2` is an `Int`. That matters because comparisons are same-type only: a
`Float` range filter never matches a record whose value was stored as an `Int`.
`nidus.DateTime` takes a `time.Time` and travels as UTC epoch milliseconds
(`DateTimeMillis` if you already hold the number).

```go
ctx := context.Background()

err := db.CreateCollection(ctx, "docs")

n, err := db.Upsert(ctx, "docs", []nidus.Record{
    {ID: "a", Vector: []float32{0.1, 0.2, 0.3},
        Attrs: nidus.Attrs{"lang": nidus.Str("rust"), "year": nidus.Int(2024)}},
    {ID: "b", Vector: []float32{0.4, 0.5, 0.6},
        Attrs: nidus.Attrs{"lang": nidus.Str("go"), "year": nidus.Int(2023)}},
    // text-only doc: omit the vector
    {ID: "c", Attrs: nidus.Attrs{"body": nidus.Str("vector stores are neat")}},
})

hits, err := db.Search(ctx, nidus.SearchRequest{Query: []float32{0.1, 0.2, 0.3}, TopK: 5})
for _, hit := range hits {
    lang, _ := hit.Attrs["lang"].Str()
    fmt.Println(hit.ID, hit.Score, lang)
}
```

Leave `TopK` zero to take the server's default (10): a zero is omitted from the request
rather than sent, since `"top_k": 0` would be a request for no results. The knobs whose
zero the server *does* treat as a real value are pointers, so an explicit zero can
travel: `MinScore` (`nil` is "no floor", `&0` is a floor of exactly zero) and hybrid
search's `RRFK` and `Candidates`.

A `Record` with a `nil` `Vector` is a text-only document. A non-nil but *empty* Vector is
refused by `Upsert` rather than sent: in Go an empty slice encodes identically to an
absent one, so it would quietly store a text-only document that no vector search can see.

```go
floor := float32(0.2)
hits, err := db.Search(ctx, nidus.SearchRequest{
    Query:    []float32{0.1, 0.2, 0.3},
    Scope:    []string{"docs"}, // an empty Scope searches every collection, one ranking
    MinScore: &floor,
})
```

If your attributes arrive as plain Go values (out of your own JSON decode, say), use
`nidus.AttrsOf(map[string]any{…})`, or `nidus.ValueOf` for a single value. Both reject
what the store has no type for and name the offending key.

## Typed attributes on the way back

`Hit.Attrs` and `Record.Attrs` keep typed `Value`s rather than decoding to `any`. Read
one with the comma-ok accessors:

```go
lang, ok := hit.Attrs["lang"].Str()  // "", false if the key is absent or not a string
year, ok := hit.Attrs["year"].Int()  // full int64 precision, never rounded via float64
tags, ok := hit.Attrs["tags"].List()
```

This is a deliberate divergence from the [JavaScript SDK](/sdks/javascript/), which
decodes attrs to plain values. In a statically typed language the typed accessor is the
better surface: `hit.Attrs["lang"].Str()` beats an `any` plus a type assertion, and a
wrong-type read gives you a testable `false` rather than a plausible-looking empty
string. When you *do* want the loose map, ask for it:

```go
plain := hit.Attrs.Decode() // map[string]any: string, int64, bool, []string, nil
```

## Similar records ("more like this")

`SearchSimilar` runs a search using the vector already stored at a record, instead of
a query you supply yourself:

```go
hits, err := db.SearchSimilar(ctx, nidus.SimilarRequest{Collection: "docs", ID: "a", TopK: 5})
```

Takes the same fields as `SearchRequest` (`TopK`, `Offset`, `MinScore`, `Filter`,
`Exact`, `RankBy`, `LimitPer`), plus `Scope`: which collections to search, defaulting
to the source record's own collection rather than every collection in the store the
way a plain `Search`'s empty `Scope` does.

The source record is never in its own results, dropped by id after ranking, not by a
score cutoff, so a genuine duplicate of it (also scoring near 1.0) still comes back. A
`Collection`/`ID` pair naming no record, or a record with no stored vector (a
text-only entry), returns an error naming the id and the reason, not an empty result.

## Filtering

Build an AND-filter with the predicate constructors. Each predicate is a positive
assertion about a **present** attribute: an absent key matches nothing (including the
negative predicates). See [Search & filters](/guides/search/) for the full semantics.

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
so `nidus.Eq("year", 2024)` reads naturally; a value with no nidus attribute type is
remembered on the predicate and comes back as an ordinary error from the call that
carried the filter. `Predicate.Err()` and `Filter.Err()` check it earlier.

## Full-text and hybrid search

```go
err := db.SetFtsSchema(ctx, "docs", []string{"body"})

// …or tune BM25 / the analyzer per field. Unset knobs take the server's defaults.
k1, folding := float32(1.5), true
err = db.SetFtsFields(ctx, "docs", []nidus.FtsField{
    {Field: "body", K1: &k1, AsciiFolding: &folding},
})

// BM25 text search: scores are raw BM25, not comparable across queries
text, err := db.TextSearch(ctx, nidus.TextSearchRequest{
    Field: "body", Query: "vector store", TopK: 10,
})

// Fuse vector + text via reciprocal rank fusion; the score is the fused RRF score
hybrid, err := db.HybridSearch(ctx, nidus.HybridSearchRequest{
    Vector: []float32{0.1, 0.2, 0.3},
    Field:  "body",
    Text:   "vector store",
    TopK:   10,
})

// Prefix match for typeahead: only the final word of Query expands
truePrefix := true
typeahead, err := db.TextSearch(ctx, nidus.TextSearchRequest{
    Field: "title", Query: "quick br", Prefix: &truePrefix, TopK: 10,
})
```

`RRFK` and `Candidates` are `*float32` / `*int`: `nil` takes the server's defaults (60 and
100). They are pointers because zero is a real request for both: the server fuses with
`1/(rrf_k + rank + 1)`, so an `RRFK` of `0` is the maximally top-heavy weighting, and a
`Candidates` of `0` fuses exactly `TopK` deep with no over-fetch.

`Prefix` is `*bool`, same idiom: `nil` (the zero value) means the final term must match a
word exactly, `true` expands it to any indexed term carrying it as a prefix (autocomplete),
capped at 256 expansions. `FtsClause` also carries its own `Prefix *bool`, so a multi-clause
query can prefix-match one field while another stays exact.

For an autocomplete dropdown itself, `Suggest` returns the completions directly, ranked
by how many live documents contain them (commonest first) rather than by BM25:

```go
sug, err := db.Suggest(ctx, "docs", nidus.SuggestRequest{
    Field: "title", Prefix: "quick br", Limit: 5,
})
for _, s := range sug.Suggestions {
    fmt.Println(s.Term, s.DF)
}
```

`Limit` is a plain `int`; leave it zero for the server's default of 10. `Suggest`
matches the same stemmed, folded vocabulary `Prefix` does.

`Fuzzy` is `*bool`, the same idiom as `Prefix`: `nil` (the zero value) leaves typo
tolerance on, so a mistyped fragment (`"runing"`) still completes to `"running"` when the
exact match finds nothing. Send `false` to opt out.

## Batch search and aggregation

`BatchSearch` answers several vector queries in one round-trip (16 max), saving a hop
per query when one question is fanned into several phrasings:

```go
queries := []nidus.SearchRequest{
    {Query: []float32{0.1, 0.2, 0.3}, TopK: 5},
    {Query: []float32{0.4, 0.5, 0.6}, TopK: 5,
        Filter: nidus.And(nidus.Eq("lang", "rust"))},
}
results, err := db.BatchSearch(ctx, nidus.BatchSearchRequest{Queries: queries})
for _, hits := range results {
    _ = hits
}

// Merge every leg into one ranking via reciprocal rank fusion
fused, err := db.BatchSearch(ctx, nidus.BatchSearchRequest{
    Queries: queries,
    Fuse:    &nidus.BatchFuse{},
})
```

With `Fuse` set the answer is still `[][]Hit`, holding the single fused ranking as its
one element, so indexing does not change shape with the flag. `BatchFuse.Weights` must
be empty or exactly as long as `Queries`.

`Aggregate` counts the records a filter matches and sums the named attributes,
answered from the in-RAM index alone: no record is built and no vector is read.

```go
totals, err := db.Aggregate(ctx, nidus.AggregateRequest{
    Scope:  []string{"docs"},
    Filter: nidus.And(nidus.Eq("lang", "rust")),
    Sum:    []string{"year"},
})
fmt.Println(totals.Count, totals.Sums["year"])

// One Group per distinct GroupBy value, alongside the unchanged whole-scope totals
byLang, err := db.Aggregate(ctx, nidus.AggregateRequest{
    Sum: []string{"year"}, GroupBy: "lang",
})
for _, g := range byLang.Groups {
    fmt.Println(g.Value, g.Count, g.Sums)
}
```

A missing or non-numeric value is skipped rather than counted as zero, so a field
nothing matched sums to `0`.

## Remembering and recalling

When the server is started with an embedder (`nidus serve --embed-provider …`), you can
send **text** and let the server embed it: no vectors client-side. `Remember` embeds
and upserts; `Recall` embeds the query and vector-searches. See
[Remember & recall](/guides/remember-and-recall/).

```go
// Embed "the quick brown fox" and store it under id "a"
res, err := db.Remember(ctx, "notes", "a", "the quick brown fox",
    nidus.RememberOptions{Attrs: nidus.Attrs{"tag": nidus.Str("x")}})

// Expire after an hour, and fold near-duplicates into the closest existing
// entry instead of inserting a competitor. On a dedupe match, res.Deduped is
// true and res.ID names the entry the write actually landed on.
ttl := int64(3600)
threshold := float32(0.95)
res, err = db.Remember(ctx, "notes", "a2", "the quick brown fox!",
    nidus.RememberOptions{TTLSeconds: &ttl, DedupeThreshold: &threshold})

// Summarize first, then embed the summary (the server also needs
// --summarize-provider). The record additionally carries the nidus.summary
// attr, with the raw input in nidus.text.
res, err = db.Remember(ctx, "notes", "b", longArticle,
    nidus.RememberOptions{Mode: "summarize"})

// Embed the query text and search, best-first
floor := float32(0.2)
hits, err := db.Recall(ctx, "notes", "quick fox", nidus.RecallOptions{
    TopK:     5,
    MinScore: &floor,
    Filter:   nidus.And(nidus.Eq("tag", "x")),
})

// Reinforce: stamp nidus.access_count / nidus.last_accessed on every hit
// returned, and push an existing expiry forward. Off by default, so a plain
// Recall stays a pure read.
extend := int64(86400)
reinforced, err := db.Recall(ctx, "notes", "quick fox", nidus.RecallOptions{
    TopK:             5,
    Reinforce:        true,
    ExtendTTLSeconds: &extend,
})
```

`Remember` returns a `RememberResult` (`ID`, `Upserted`, `Deduped`): `ID` is the record
that actually changed, which is not always the one passed in, and `Upserted` is the row
count from the underlying write. See
[Parity across the surfaces](/guides/remember-and-recall/#parity-across-the-surfaces)
for how these semantics line up with the other SDKs and the MCP surface.

Setting `Reinforce` makes the call a write: it takes the server's writer lock to apply
the stamp, and against a server started `--read-only` the stamp is skipped with a
warning rather than failing the recall. `ExtendTTLSeconds` only applies with
`Reinforce` set, and only pushes an **existing** `nidus.expires_at` forward; it never
gives an expiry to an entry that had none. See
[reinforcement](/guides/remember-and-recall/#reinforcement).

Two different failures are worth telling apart when these do not work. A **`404` with no
message** means the server binary was built without the `memory` feature, so `/remember`
and `/recall` are not routes it has at all. A **`400`** means the routes exist but the
server was started without `--embed-provider`; the message names the flag, and
`Mode: "summarize"` without a summarizer is likewise a `400`. Recalling a collection
written with a different embedding model is a `409`, not a silently meaningless ranking.

## The rest of the API

Every data-plane endpoint of the [HTTP API](/reference/http-api/) has a method. The
ops probes are methods too, with one exception: `Ready`, `Cluster`, and `Refresh`
each have a typed method, while `/metrics` stays unwrapped since it is a scraper's
endpoint, not something application code calls. `Ready` returns a verdict rather
than an error when the server reports not-ready, so a `503` is a value you check,
not a failure you unwrap:

```go
ok := db.Health(ctx)                       // bool: needs no token; "is it up", one answer
err = db.Ping(ctx)                         // the same call, keeping the reason it failed
stats, err := db.Stats(ctx)                // dimension, distance, ANN config, footprint
names, err := db.Collections(ctx)          // []string
hits, err := db.List(ctx, nidus.ListRequest{
    Scope:  []string{"docs"},
    Filter: nidus.And(nidus.Eq("lang", "rust")),
})                                          // metadata-only, paginated by Offset/Limit
recs, err := db.Records(ctx, "docs")       // every record; Vector is nil for a text-only doc
meta, err := db.GetMeta(ctx, "docs")
err = db.SetMeta(ctx, "docs", map[string]string{"owner": "search-team"}) // replaces, not merges
n, err := db.Delete(ctx, "docs", []string{"a"})
n, err = db.DeleteWhere(ctx, "docs", nidus.And(nidus.Lt("year", 2000))) // empty filter = everything
err = db.Flush(ctx)
err = db.Compact(ctx)
err = db.DropCollection(ctx, "docs")
aliases, err := db.Aliases(ctx)            // map[string]string: alias name -> concrete collection
err = db.SetAlias(ctx, "docs", "docs_v2")  // create or repoint, idempotent; target must exist
err = db.DropAlias(ctx, "docs")            // removes the alias, not the records
r, err := db.Ready(ctx)                    // 503 is a verdict: r.Ready false, r.Reason set
cs, err := db.Cluster(ctx)                 // role, lease state, commit version
adopted, err := db.Refresh(ctx)            // bool: did it adopt newer state
```

## Errors

A failed request returns a `*nidus.Error` carrying the HTTP status the server reported,
so you can tell a client fault from a server fault:

```go
if _, err := db.Upsert(ctx, "docs", records); err != nil {
    var nerr *nidus.Error
    if errors.As(err, &nerr) {
        switch {
        case nerr.IsBadRequest():     // 400/422: the request is wrong; retrying cannot help
        case nerr.IsUnauthorized():   // 401: missing or wrong bearer token
        case nerr.IsReadOnly():       // 403: a write against a read-only store
        case nerr.IsLocked():         // 409: the writer lock is held elsewhere
        case nerr.IsUnavailable():    // 503: shed under backpressure, or store not open
        case nerr.IsOutOfCapacity():  // 507: the store refused to grow; it is intact
        }
        log.Println(nerr.Status, nerr.Message)
    }
}
```

`IsBadRequest()` covers `400` **and** `422`. That split belongs to the server's HTTP layer
rather than to nidus: a JSON *syntax* error (and the store's own client faults, such as a
dimension mismatch) is a `400`, while a body whose *types* do not deserialize (`TopK: -1`)
is a `422`. To a caller they are one thing, since retrying neither can ever succeed.
`409` and `503` are the two a retry with backoff is the right answer to.

`IsTransport()` (status `0`) means the request never got an answer at all: the server
was unreachable, or the request outlived its deadline. Unlike the status-carrying cases
it says nothing about whether the write was applied, since a timeout can fire after the
server has committed.
