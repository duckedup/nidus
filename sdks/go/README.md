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

Attributes are typed: `nidus.Str`, `nidus.Int`, `nidus.Bool`, `nidus.List`,
`nidus.Null`. There is no float attribute — floats belong in the vector — and the
constructors say so at compile time rather than at the server's 400.

```go
ctx := context.Background()

if err := db.CreateCollection(ctx, "docs"); err != nil { /* … */ }

n, err := db.Upsert(ctx, "docs", []nidus.Record{
    {ID: "a", Vector: []float32{0.1, 0.2, 0.3},
        Attrs: nidus.Attrs{"lang": nidus.Str("rust"), "year": nidus.Int(2024)}},
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
zero), and hybrid search's `RRFK` and `Candidates`.

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
lang, ok := hit.Attrs["lang"].Str()   // "", false if absent or not a string
year, ok := hit.Attrs["year"].Int()   // full int64 precision, never rounded via float64
tags, ok := hit.Attrs["tags"].List()
```

This is a deliberate deviation from the JavaScript SDK, which decodes attrs to plain JS
values: in a statically typed language the typed accessor is the better surface —
`hit.Attrs["lang"].Str()` beats an `any` and a type assertion, and a wrong-type read
gives you a testable `false` instead of a plausible-looking empty string. When you do
want the loose map, ask for it:

```go
plain := hit.Attrs.Decode() // map[string]any: string, int64, bool, []string, nil
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

Predicates: `Eq`, `Ne`, `Glob`, `In`, `NotIn`, `Lt`, `Le`, `Gt`, `Ge`. They take `any`
so `nidus.Eq("year", 2024)` reads naturally; a value the store has no type for (a
float, say) is remembered on the predicate and surfaces as an ordinary error from the
call that used the filter. Check it earlier with `Predicate.Err()` / `Filter.Err()`.

## Full-text and hybrid search

```go
if err := db.SetFtsSchema(ctx, "docs", []string{"body"}); err != nil { /* … */ }

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
weighting, and `Candidates: &zero` fuses exactly `TopK` deep with no over-fetch.

## Remembering and recalling (text-native)

When the server is started with an embedder (`nidus serve --embed-provider …`), you can
send **text** and let the server embed it — no vectors client-side. `Remember` embeds
and upserts; `Recall` embeds the query and vector-searches.

```go
// Embed "the quick brown fox" and store it under id "a"
err := db.Remember(ctx, "notes", "a", "the quick brown fox",
    nidus.RememberOptions{Attrs: nidus.Attrs{"tag": nidus.Str("x")}})

// Summarize first, then embed the summary (the server also needs
// --summarize-provider). The stored record additionally carries nidus.summary
// and nidus.source attrs.
err = db.Remember(ctx, "notes", "b", longArticle,
    nidus.RememberOptions{Mode: "summarize"})

// Embed the query text and search, best-first
floor := float32(0.2)
hits, err := db.Recall(ctx, "notes", "quick fox", nidus.RecallOptions{
    TopK:     5,
    MinScore: &floor,
    Filter:   nidus.And(nidus.Eq("tag", "x")),
})
```

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
