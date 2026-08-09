---
title: Search & filters
description: Scoped search across nidus collections with three distance metrics, exact or approximate (HNSW/IVF) indexing, int8/binary quantization, BM25 full-text and hybrid (RRF) search, recency ranking, aggregation, and a filter language of equality, range, glob, set, containment, boolean, and text predicates.
---

Search in nidus runs over a scope you choose, using one of three distance metrics,
optionally narrowed by a metadata filter and a score floor. It is **exact by
default** (every in-scope vector is scored) and can opt into an
[approximate index](#approximate-search-ann) (HNSW or IVF) when a full scan is more
than you want to pay.

## Distance metrics

The distance metric is set at store creation via `Config::distance` and pinned
in the data header. Reopening with a different metric is an error.

| Metric | Normalization | Score | Range | Best for |
| --- | --- | --- | --- | --- |
| `Cosine` (default) | Vectors unit-normalized on insert | `dot(q, v)` | \[−1, 1\] | Embedding similarity |
| `Euclidean` | Stored as-is | `−‖q − v‖²` | (−∞, 0\] | Spatial distance |
| `DotProduct` | Stored as-is | `dot(q, v)` | (−∞, ∞) | When magnitude matters |

For all metrics, **higher score = more relevant**, so top-k, `min_score`, and
ranking all work the same way regardless of which metric you choose.

```rust
use nidus::{Config, Distance, Nidus};

// Cosine (default, same as before)
let db = Nidus::open(Config::new("./store", 384))?;

// Euclidean distance
let db = Nidus::open(Config::new("./store-l2", 384).distance(Distance::Euclidean))?;

// Raw dot product (magnitude carries signal)
let db = Nidus::open(Config::new("./store-dot", 384).distance(Distance::DotProduct))?;
# anyhow::Ok(())
```

## Scope

Every search runs over a [`Scope`](/reference/api/#scope): one collection, a
named subset of collections, or the whole store. Results from a multi-collection
scope are **merged into a single ranking**.

```rust
use nidus::Scope;

db.search("code", &q, &opts)?;                                // one collection
db.search(Scope::Collections(&["code", "docs"]), &q, &opts)?; // a named subset
db.search(Scope::All, &q, &opts)?;                            // the whole store
# anyhow::Ok(())
```

This is sound because **all collections share one embedding space**. The
dimension is pinned at store creation, so a vector in `code` and a vector in
`docs` are directly comparable: one ranking over both is meaningful, not a
category error.

## Scoring

`SearchOpts` controls the ranking:

```rust
use nidus::SearchOpts;

let opts = SearchOpts {
    top_k: 10,             // keep at most this many hits
    offset: 0,             // skip this many top-ranked hits (pagination)
    min_score: Some(0.5),  // drop anything below this score (None = no floor)
    ..Default::default()
};
# anyhow::Ok(())
```

`top_k` is enforced by a bounded heap, so memory stays flat regardless of how
many rows are scored.

Results are ordered by **score descending, then `collection`, then `id`**. That
tie-break is a guarantee, not a coincidence: it makes a query against an unchanged
store return the same ranking every time, which is what lets you page through it.

## Paginating a search

`offset` skips that many top-ranked hits, so successive pages tile one ranking with
no gap and no overlap. It works the same on `search`, `text_search`, and
`hybrid_search`.

```rust
use nidus::SearchOpts;

let query = vec![0.1_f32; 384];
let page1 = db.search("code", &query, &SearchOpts { top_k: 20, ..Default::default() })?;
let page2 = db.search(
    "code",
    &query,
    &SearchOpts { top_k: 20, offset: 20, ..Default::default() },
)?;
# anyhow::Ok(())
```

Things worth knowing:

- The ranking is computed `offset + top_k` deep, so a later page costs a little more
  than the first. `offset + top_k` may not exceed **10 000** over HTTP: past that a
  request is refused with a `400` rather than quietly shortened.
- An `offset` past the end returns an **empty** list, not an error. That is the signal
  to stop walking.
- For `hybrid_search` the page is cut on the *fused* ranking, never on one leg.
- A page is stable only against an **unchanging** store. Concurrent upserts and deletes
  shift the ranking, so a document can move between pages during a paged walk.

## Choosing the attrs a hit carries

By default a `Hit` carries every attr of the matched record. When the records hold long
text bodies, a top-50 search ships fifty full documents to answer a question about ids
and scores. `SearchOpts::projection` (and `ListOpts::projection`) trims that:

```rust
use nidus::{Projection, SearchOpts};

let query = vec![0.1_f32; 384];
// Only these attrs come back: nothing else is even cloned.
let lean = db.search(
    "code",
    &query,
    &SearchOpts {
        top_k: 10,
        projection: Projection::include(["path", "lang"]),
        ..Default::default()
    },
)?;

// Or: everything except the expensive one.
let no_body = db.search(
    "code",
    &query,
    &SearchOpts {
        top_k: 10,
        projection: Projection::exclude(["body"]),
        ..Default::default()
    },
)?;
# anyhow::Ok(())
```

- The default is `Projection::All`: every attr, exactly as before.
- Projection is applied where the hit is built, so an excluded attr is never copied.
- An included attr the record does not have is simply absent from the hit.
- Ranking, scores, and `top_k` are unaffected: this changes the payload, not the answer.
- Over HTTP the two are `include_attributes` and `exclude_attributes`. Sending both in
  one request is a `400`, since there is no precedence rule to remember.

## Forcing an exact search

A store configured with an [ANN index](#approximate-search-ann) or
[quantization](#quantization) answers every query approximately. `exact: true` opts one
query out of that, running the exact brute-force scan instead, useful when a caller
needs a guaranteed-exact answer over a small filtered subset but wants to keep the index
for everything else.

```rust
use nidus::SearchOpts;

let query = vec![0.1_f32; 384];
let certain = db.search(
    "code",
    &query,
    &SearchOpts { top_k: 10, exact: true, ..Default::default() },
)?;
# anyhow::Ok(())
```

It bypasses the ANN walk, the per-segment index fan-out, and the quantized first pass
alike. The default is `false`, so the store's configured path is unchanged for anyone who
does not ask.

## Ranking by recency

On pure similarity a two-year-old note beats a fresh one that says the same thing.
`SearchOpts::rank_by` layers a **recency decay** over the store's metric: an age penalty
computed from a timestamp attribute and subtracted from each hit's score.

```rust
use nidus::{Decay, RankBy, SearchOpts};

let query = vec![0.1_f32; 384];
let now = 1_770_000_000_000_i64; // epoch milliseconds
let week = 7 * 24 * 60 * 60 * 1000;

let hits = db.search(
    "notes",
    &query,
    &SearchOpts {
        top_k: 10,
        // Halve the decay factor every week; a fully-decayed hit gives up 0.2 of score.
        rank_by: Some(RankBy::Decay(Decay::new("updated_at", now, week).lambda(0.2))),
        ..Default::default()
    },
)?;
# anyhow::Ok(())
```

The timestamp attribute may be a `Value::DateTime` or a `Value::Int`, both **epoch
milliseconds**. `origin` is "now" as *you* supply it, not the wall clock, so the same query
against an unchanged store ranks the same way twice. The score is:

```text
age    = max(0, origin − attrs[field])   // a future timestamp is simply un-aged
factor = decay ^ (age / scale)           // `decay` (default 0.5) at exactly one `scale`
score  = base − lambda × (1 − factor)
```

The penalty **subtracts** rather than multiplying. That is what makes it valid for
`Euclidean` (which scores in (−∞, 0]) and `DotProduct` (which scores anywhere at all), not
just cosine, and for the unbounded BM25 scores of `text_search`, where `rank_by` works
identically. The cost is that `lambda` is in score units, so pick it against the metric in
use.

Two behaviours worth knowing:

- **A record with no usable timestamp is not penalized** (`missing` defaults to `1.0`), so
  switching decay on never silently buries data written before the field existed. Pass
  `.missing(0.0)` for the opposite.
- **`rank_by` does not force an exact search.** [`exact`](#forcing-an-exact-search) is the
  knob for that. Over an ANN or quantized result set the candidates were selected on the
  base score, so decay reorders within an approximate set. A ranked scan also runs
  single-threaded (it needs each record's attrs, which the parallel scan kernels do not
  see), so a `rank_by` query gives up `query_threads`.

`min_score` is compared against the final, decayed score on every path.

## Capping hits per attribute value

One verbose document can fill an entire recall window. `limit_per` caps how many hits may
carry any one value of an attribute:

```rust
use nidus::{LimitPer, SearchOpts};

let query = vec![0.1_f32; 384];
let hits = db.search(
    "code",
    &query,
    &SearchOpts {
        top_k: 20,
        limit_per: Some(LimitPer::new("path", 2)), // at most 2 hits per file
        ..Default::default()
    },
)?;
# anyhow::Ok(())
```

The group value is read from the stored record, so excluding the field with a
[projection](#choosing-the-attrs-a-hit-carries) does not lift the cap, and records
**missing** the attribute all share one group: otherwise dropping the attribute would be a
way to opt out.

It is a deliberately **approximate** cap. A capped search ranks eight pages deep, applies the
cap in rank order, then cuts the page, so a page can come back shorter than `top_k` even
though more uncapped matches exist further down. What is guaranteed is the upper bound: a
returned page never carries more than `max` hits for one value.

## Typed metadata

Each record carries an open map of typed [`Value`](/reference/api/#value)s:

| Variant       | Meaning                                              |
| ------------- | ---------------------------------------------------- |
| `Str(String)` | A string.                                            |
| `Int(i64)`    | A signed 64-bit integer.                             |
| `Bool(bool)`  | A boolean.                                           |
| `List(Vec<String>)` | A list of strings (e.g. tags).                 |
| `Float(f64)`  | A double, compared by IEEE rules.                    |
| `DateTime(i64)` | A UTC instant as epoch **milliseconds**.           |
| `Null`        | Set, but empty, **distinct from an absent key**.     |

The `Null`-vs-absent distinction is deliberate: absence means "not computed / not
indexed", while `Null` means "computed, and empty". A host app uses it to tell an
un-indexed field apart from a field that was indexed and found to be empty.

Comparison is **same-type only**, which has two consequences worth knowing up front:

- `Float` and `Int` do **not** cross-compare. `Ge("score", Float(0.5))` never matches a
  record that stored `Int(1)`, so pick one spelling per attribute and keep to it.
- A `Float` `NaN` is unordered and unequal to itself, so it matches *nothing*, `Eq("k",
  Float(NAN))` included.

`DateTime` is an absolute instant: there is no timezone and no local-time form, and
rendering it is the caller's business. It is a distinct variant from `Int` so a filter or
a [recency ranking](#ranking-by-recency) can tell a time from a number without relying on
a naming convention.

## Filters

A [`Filter`](/reference/api/#predicate--filter) is a conjunction (AND) of
[`Predicate`](/reference/api/#predicate--filter)s, applied **before scoring**: matching
rows are scored, the rest are skipped. An empty filter matches everything.

```rust
use nidus::{Filter, Predicate, Value};

let filter = Filter(vec![
    // attrs["path"] is a Str matching this glob
    Predicate::Glob("path".into(), "src/auth/*".into()),
    // AND attrs["lang"] equals one of these
    Predicate::In("lang".into(), vec![
        Value::Str("rust".into()),
        Value::Str("go".into()),
    ]),
    // AND attrs["ts"] is at least this (numeric range)
    Predicate::Ge("ts".into(), Value::Int(1_700_000_000)),
    // AND attrs["status"] is present and not "archived"
    Predicate::Ne("status".into(), Value::Str("archived".into())),
]);
# anyhow::Ok(())
```

The predicates:

- **`Eq(key, value)`** / **`Ne(key, value)`**: `attrs[key]` equals / does not equal
  `value` (any `Value` type, typed).
- **`Glob(key, pattern)`**: `attrs[key]` is a `Str` matching the glob. Supports
  `*`, `?`, and `[...]` character classes.
- **`IGlob(key, pattern)`**: the same, with **ASCII** case folded on both sides, so
  `"Src/Auth/*"` matches `"src/auth/mod.rs"`. Non-ASCII is not folded (`É` does not
  match `é`). The fold is context-free so a pattern means the same thing on every
  machine. Prefer this for path scoping, where an exact-case comparison mostly just
  returns nothing; keep `Glob` where case is a real distinction.
- **`In(key, values)`** / **`NotIn(key, values)`**: `attrs[key]` is / is not one of
  the values in the set.
- **`Lt` / `Le` / `Gt` / `Ge(key, value)`**: ordered range comparison, **same-type
  and orderable only** (`Int` numeric, `Str` lexical, `Bool` `false < true`). A
  cross-type or non-orderable (`Null`, `List`) comparison never matches.
- **`Contains(key, value)`** / **`NotContains(key, value)`**: `attrs[key]` is a
  `List` that does / does not hold `value`. Matching is whole-element, not substring:
  `Contains("tags", "rust")` does not match `["rustacean"]`, and `Contains` on a plain
  `Str` fails (a string is not a one-element list).
- **`ContainsAny(key, values)`**: `attrs[key]` is a `List` sharing at least one
  element with the set. There is no `ContainsAll`; `All` over several `Contains`
  already says it.

Every *leaf* predicate is a positive assertion about a **present** attribute: a record
that lacks `key` matches *nothing*, including `Ne` / `NotIn` / `NotContains` and the
range predicates. (So `Ne("status", "archived")` does not match a record with no
`status` at all.)

### Text predicates

Five more leaf predicates match *inside* the text an attribute carries, for the cases a
glob cannot express: a half-remembered identifier, a bag of words, a phrase, a pattern.
All five read a `Str` directly and a `List` element by element, matching when **any single
element** satisfies them (so a phrase never spans two elements).

```rust
use nidus::{Filter, Predicate};

let filter = Filter(vec![
    // Within 2 edits of "nidus", ASCII-case-folded on both sides.
    Predicate::Fuzzy("name".into(), "nidus".into(), 2),
    // Every one of these tokens appears, in any order.
    Predicate::ContainsAllTokens("body".into(), "vector store".into()),
    // These tokens appear consecutively and in order: a phrase.
    Predicate::ContainsTokenSequence("body".into(), "append only".into()),
    // Anchored at both ends, like a glob.
    Predicate::Regex("path".into(), r"src/[a-z]+/mod\.rs".into()),
]);
```

- **`Fuzzy(key, needle, n)`**: within `n` **Levenshtein** edits of `needle`, counting
  *characters* rather than bytes. It is the plain three-operation distance (substitute,
  insert, delete), so a **transposition costs 2**: `Fuzzy("word", "from", 1)` does not
  match `"form"`. Both sides are ASCII-case-folded, exactly as `IGlob` folds. `n` may not
  exceed **8**: a larger budget is an **error**, never a silent clamp, because a clamped
  budget quietly answers a different question than the one asked.
- **`ContainsAllTokens(key, text)`** / **`ContainsAnyToken(key, text)`** /
  **`ContainsTokenSequence(key, text)`**: every query token present in any order, at
  least one present, or the tokens consecutive and in order (a phrase). A token is a
  maximal run of alphanumerics, ASCII-case-folded, with **no stemming and no stopword
  removal**: these are *filter* predicates, where a term either is or is not there. So
  `ContainsAllTokens("body", "run")` does **not** match `"running"`, while
  [`text_search`](#full-text-search-bm25) for the same word does, because stemming belongs
  to ranking, where a partial-credit match means something.
- **`Regex(key, pattern)`**: a regular expression, **anchored at both ends** like `Glob`,
  so the whole attribute must match and `.*` on either side opts back into a substring
  search. Case-insensitivity is the pattern's own `(?i)` flag rather than a second
  variant, since a regex has somewhere to put a flag and a glob does not. An unparseable
  pattern is a caller-facing error, never a silently non-matching filter. The engine is
  finite-automata and **linear-time in the input**: it does not backtrack, so there is no
  ReDoS to guard against and no timeout knob to tune.

Empty queries take the usual identities: `ContainsAllTokens` with no tokens matches any
present text attribute (as `All(vec![])` does), `ContainsAnyToken` with none matches
nothing (as `Any(vec![])` does).

:::caution[There is no index behind these]
Every predicate here re-tokenizes or re-scans the attribute for **each record the scan
visits**: O(attribute length) per row for the token family, and O(needle × attribute) for
the `Fuzzy` DP. That is fine at the scale nidus targets, and it is *not* a substitute for
full-text search over a large corpus. When the field is a document, reach for
[`text_search`](#full-text-search-bm25), which does have an index.
:::

### Combining predicates

A `Filter` is a conjunction, but `All`, `Any`, and `Not` are themselves predicates that
take predicates, so any boolean shape nests inside one:

```rust
use nidus::{Filter, Predicate, Value};

// (lang = rust OR lang = go) AND NOT tags contains "generated"
let filter = Filter(vec![
    Predicate::Any(vec![
        Predicate::Eq("lang".into(), Value::Str("rust".into())),
        Predicate::Eq("lang".into(), Value::Str("go".into())),
    ]),
    Predicate::Not(Box::new(Predicate::Contains(
        "tags".into(),
        Value::Str("generated".into()),
    ))),
]);
```

Empty groups take the usual identities: `All(vec![])` matches everything (like an empty
`Filter`), `Any(vec![])` matches nothing.

:::caution[`Not` and `Ne` differ on a missing attribute]
`Ne("status", "archived")` asserts the attribute is **present and different**, so it
does not match a record with no `status`. `Not(Eq("status", "archived"))` negates the
truth value, so it **does** match that record: `Eq` was false, and `Not` inverted it.
Reach for `Ne` / `NotIn` / `NotContains` when the attribute must exist, and `Not` when
you want a genuine complement.
:::

Filters can also drive deletion without a search:

```rust
// Delete every record whose path is under src/legacy/
db.delete_where("code", &Filter(vec![
    Predicate::Glob("path".into(), "src/legacy/*".into()),
]))?;
# anyhow::Ok(())
```

## Metadata-only queries

Use `list` to retrieve records by metadata filter without a vector query. Results come
back in insertion order with `score: 0.0`, unless [`order_by`](#ordering-by-an-attribute)
says otherwise. `ListOpts`'s `offset` and `limit` paginate a stable ordering: advance
`offset` by `limit` to page through.

```rust
use nidus::{Filter, ListOpts, Predicate, Value};

let filter = Filter(vec![
    Predicate::Eq("lang".into(), Value::Str("rust".into())),
]);

// First page: offset 0, up to 100 matches (the default limit).
let page1 = db.list("code", &ListOpts { filter: filter.clone(), ..Default::default() })?;
// Next page.
let page2 = db.list("code", &ListOpts { offset: 100, filter, ..Default::default() })?;
# anyhow::Ok(())
```

`list` accepts a [`Scope`](/reference/api/#scope) just like `search`, so you can
list across multiple collections or the whole store. It also takes the same
[`projection`](#choosing-the-attrs-a-hit-carries), so a listing can return ids alone.

### Ordering by an attribute

`ListOpts::order_by` sorts by an attribute instead of storage order: ORDER BY with no
vector query at all. Sorting happens over the whole match set *before* the page is cut, so
`offset`/`limit` walk the sorted order.

```rust
use nidus::{ListOpts, OrderBy};

let newest = db.list(
    "notes",
    &ListOpts { order_by: Some(OrderBy::desc("updated_at")), limit: 20, ..Default::default() },
)?;
# anyhow::Ok(())
```

Cross-type ordering mirrors the filter's same-type rule: the first orderable value found
fixes the sort's type, and everything that does not order against it (a different `Value`
variant, an unorderable `Null` or `List`, or a record missing the attribute entirely) lands
in one **trailing bucket**. The bucket stays trailing under `descending` too.

## Aggregation

`aggregate` counts matching records and totals numeric attributes straight off the in-memory
index: no record is materialized and no vector is read.

```rust
use nidus::{AggregateOpts, Filter, Predicate, Scope, Value};

let stats = db.aggregate(
    Scope::All,
    &AggregateOpts {
        filter: Filter(vec![Predicate::Eq("lang".into(), Value::Str("rust".into()))]),
        sum: vec!["bytes".into()],
        ..Default::default()
    },
)?;
println!("{} records, {:?} bytes", stats.count, stats.sums["bytes"]);
# anyhow::Ok(())
```

`count` is always reported. Each entry in `sums` is a tagged `Value`: `Int` while every
addend was an `Int`, `Float` once any `Float` joined. A missing or non-numeric value is
skipped rather than counted as zero, so `sum` and `count` stay independently meaningful.

### Grouping

`group_by` reports the same figures per **distinct value** of an attribute, in the same
single pass, alongside the unchanged whole-scope totals.

```rust
use nidus::{AggregateOpts, Scope};

let by_lang = db.aggregate(
    Scope::All,
    &AggregateOpts {
        sum: vec!["bytes".into()],
        group_by: Some("lang".into()),
        ..Default::default()
    },
)?;
for g in &by_lang.groups {
    // `value` is None for the records with no `lang` attribute at all.
    println!("{:?}: {} records", g.value, g.count);
}
# anyhow::Ok(())
```

Groups arrive largest first, with a deterministic tie-break so repeating the query repeats
the order. Records missing the attribute form one group with a `None` value, distinct from
records holding `Value::Null`, the same absent-versus-null rule the filters follow. Past
10 000 distinct values further ones are dropped and `groups_truncated` is set.

## Quantization

For larger collections, enable quantization to speed up search. The store keeps a
compressed copy of every vector in RAM and search runs in two passes: a fast quantized
first pass selects candidates, overscanning by the `rescore` factor, then the original
f32 vectors re-rank those candidates for an exact final ranking.

Two schemes are available:

| Scheme | Size vs f32 | Default `rescore` | Metrics |
| --- | --- | --- | --- |
| `Quantization::int8()` | 4× smaller | 4 | any |
| `Quantization::binary()` | 32× smaller | 16 | **cosine only** |

int8 is global **symmetric** scalar quantization: one shared scale, no per-dimension
offset, so the int8 score stays monotonic with the true score. Binary keeps only each
dimension's sign bit and scores a Hamming distance; sign codes approximate *angular*
similarity and discard magnitude, which is why they are not a sound ranking proxy for
`DotProduct` or `Euclidean`. Binary is the coarser proxy, hence its larger default
overscan.

```rust
use nidus::{Config, Nidus, Quantization};

// int8: valid for any distance metric.
let db = Nidus::open(Config::new("./store", 768).quantization(Some(Quantization::int8())))?;

// Binary sign bits, with a wider candidate net (cosine stores only).
let db2 = Nidus::open(
    Config::new("./store2", 768).quantization(Some(Quantization::binary().rescore(24)))
)?;
# anyhow::Ok(())
```

The `rescore` factor trades recall for speed: `rescore: 4` means the first pass keeps
`top_k * 4` candidates before the f32 re-rank. Higher values widen the candidate net
(better recall, more f32 work).

**What to expect (int8).** In the `just bench-quant` sweep (uniform-random vectors,
a near-worst case for quantization recall), the two-pass search returns
essentially the exact neighbours (**~100% recall@10 at `rescore` ≥ 2**) for a
**~1.4× query speedup** at 1M × 768, in exchange for **~25% more RAM** (the int8
copy sits alongside the f32 matrix, which the re-rank still needs). The speedup
is bounded by the pure-safe-Rust scalar int8 kernel; the larger theoretical win
would need SIMD int8 dot-product intrinsics, which are `unsafe` and outside
nidus's zero-FFI design. Run `just bench-quant` to measure on your own shapes.

Quantization is purely a runtime optimization: it doesn't change the on-disk
format, and a store opened without it produces identical results (just slower
for large scans). Reach for it when search latency matters more than the extra
RAM. Vectors quantize incrementally on upsert, so adding records stays cheap.

## Approximate search (ANN)

By default every search is **exact**: it scores every in-scope vector. When a
collection grows past the point where a full scan is more than you want to pay,
`Config::ann` opts into an **approximate** index that walks a much smaller candidate
set instead. It is purely a runtime choice: nothing about the on-disk format changes,
and a store opened without it behaves exactly as before.

```rust
use nidus::{Config, Nidus, AnnConfig};

// HNSW: a navigable small-world graph (the default ANN index).
let db = Nidus::open(Config::new("./store", 768).ann(Some(AnnConfig::hnsw())))?;

// or IVF: k-means inverted lists.
let db2 = Nidus::open(Config::new("./store2", 768).ann(Some(AnnConfig::ivf())))?;
# anyhow::Ok(())
```

Both index types work the same way at query time: the index selects an over-fetched
candidate set (`top_k × overscan`), then nidus applies your scope, metadata filter,
and `min_score` to those candidates and ranks the survivors by the **exact** f32
score. So the *final ordering is always exact*; only the candidate *selection* is
approximate.

**Choosing an index.** `AnnConfig::hnsw()` gives high recall and supports cheap
incremental `upsert` (new vectors are inserted into the graph directly); it is the
right default. `AnnConfig::ivf()` uses less memory for its links but fits its k-means
partition from the data present at build time, so heavy incremental growth drifts the
partition until the next `compact()` rebuilds it.

**Tuning.** Each has builder setters: HNSW exposes `m`, `ef_construction`, and
`ef_search` (higher = better recall, more work); IVF exposes `n_lists` and `n_probe`
(more probed lists = better recall, slower). Both share `overscan` (how many
candidates to fetch before the post-filter) and a `seed` for reproducible builds.

```rust
use nidus::AnnConfig;
let cfg = AnnConfig::hnsw().m(32).ef_search(128).overscan(8);
# let _ = cfg;
```

**Persisting the index.** The graph is in-RAM and rebuilt from the vectors on
`open()`, and for HNSW that build is the expensive part. Call
[`db.persist_index()`](/reference/api/#nidus) to write it to an `ann` cache file so
the next `open()` *loads* it instead of rebuilding. The same call also writes the
full-text index's `fts` cache when a collection declares one, so it is worth calling
even on a store with no ANN index at all. It's an explicit, out-of-band operation
(also triggered by `compact()`), never on the `upsert`/`flush` write path,
so storage stays fast. Call it before shutting down a long-lived handle. `open()`
incrementally catches up any rows added since the cache was written, and silently
rebuilds if the cache is missing, stale, or for a different config. The caches are
derived data: deleting the `ann` or `fts` file only costs a one-time rebuild.

**Trade-offs to know.**

- **Approximate recall.** ANN may miss a true neighbour. Raise `ef_search`/`n_probe`
  and `overscan` to trade speed for recall.
- **Selective filters are handled exactly.** The filter is applied *after* the walk, so
  a very selective filter or a narrow collection-subset scope could starve the candidate
  set. nidus detects this case: when a narrowed query's survivor population drops below
  what the overscanned walk can reliably surface, it gathers the filter-passing rows
  directly and scores them exactly instead of walking the index. Selective filtered
  queries stay recall-complete; you pay an exact scan over the survivors, which is small
  by definition in exactly the case that triggers it.
- **Deletes.** A deleted vector leaves a stale node in the index that is skipped at
  query time and fully reclaimed on the next `compact()`.
- **Combine with quantization.** `ann` and `quantization` can be enabled together: the
  index walk scores quantized codes (a cheaper candidate selection) and the exact f32
  rerank over those candidates restores accuracy. Recall runs a little below the
  exact-walk index, so widen `ef_search`/`n_probe` and `overscan` if you need it back.

## Per-segment indexing at scale

`Config::ann` above is a **single global** index over every row. There is a second,
size-driven way to index that keeps the freshest data exact: when a store is split into
[segments](/guides/storage/#segments), nidus can IVF-index only the **cold, immutable**
segments and leave the recent write tail exhaustive.

```rust
use nidus::{Config, Nidus};

let db = Nidus::open(
    Config::new("./store", 768)
        .segment_max_rows(Some(100_000))        // seal a segment every 100k rows
        .segment_index_min_rows(Some(100_000)), // IVF-index each sealed segment
)?;
# anyhow::Ok(())
```

With [`segment_index_min_rows`](/reference/configuration/#segment_index_min_rows) set, a
sealed segment of at least that many rows gets its own IVF index (built when it seals and
rebuilt on each `open()`; unlike the `ann`/`fts` caches, per-segment indexes are not
persisted);
the **active** segment (everything written since the last seal) and any smaller segment
stay brute-forced. A search then **fans out**: it scans the exhaustive tail exactly and
walks each cold segment's index for candidates, merging both into one ranking with an exact
f32 rerank. So "exact vs approximate" follows size automatically: the fresh data is always
exact, only the cold bulk is approximate.

This is **off by default** (`segment_index_min_rows = None` → every segment brute-forced →
100% recall, zero knobs), and it is ignored when a global `ann` index is set (that index
already covers every row). The same approximate-recall and deleted-row notes above apply to
the cold segments.

## Full-text search (BM25)

Alongside vector search, a collection can declare **full-text-indexed fields** and be
queried by keyword with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) ranking. It
reuses the same `Hit` results, `Filter`, scope, and `top_k` heap as vector search;
only the scoring differs.

Declare which attribute fields are full-text indexed. You can do it up front at
collection creation (the recommended path: indexing is incremental from the first
upsert) or any time afterward (it indexes the docs already stored):

```rust
use nidus::{Config, FtsField, Nidus};

let mut db = Nidus::open(Config::new("./store", 384))?;

// Up front (recommended):
db.create_collection_with_fts("docs", &[FtsField::new("body")])?;

// …or declare/redeclare later on an existing collection:
db.set_fts_schema("docs", &[FtsField::new("title")])?;
# anyhow::Ok(())
```

Then query a field with [`text_search`](/reference/api/#nidus):

```rust
use nidus::{FtsQuery, SearchOpts};

let hits = db.text_search(
    "docs",
    &FtsQuery::new("body", "running quickly"),
    &SearchOpts { top_k: 10, ..Default::default() },
)?;
# anyhow::Ok(())
```

### Searching several fields at once

A query is a **list of clauses**, and each clause carries its own text, so a record with
both a title and a body can be searched across both in one query, with different words per
field:

```rust
use nidus::{FtsClause, FtsCombine, FtsQuery};

let q = FtsQuery::multi([
    FtsClause::new("title", "rust"),
    FtsClause::new("body", "async runtime"),
]);
let hits = db.text_search("docs", &q, &SearchOpts { top_k: 10, ..Default::default() })?;

// …or take the strongest single clause instead of adding them up:
let q = q.combine(FtsCombine::Max);
# anyhow::Ok(())
```

- **`FtsCombine::Sum`** (the default) adds every matched clause's BM25 score, so a document
  that hits the title *and* the body outranks one that hits either alone.
- **`FtsCombine::Max`** takes the strongest clause, so a long body cannot out-accumulate a
  precise title match.
- A clause naming a field the collection does not full-text index simply contributes
  nothing. An **empty clause list is an error** (over HTTP a `400`) because an empty
  result would otherwise read as "no matches" rather than "you sent no query".
- `FtsQuery::new(field, text)` is the one-clause shorthand, and a single clause scores
  exactly the same under either combine mode. `min_score` applies to the combined score.

### Tuning a field

Each declared field is an `FtsField`, and every knob has a default that reproduces
nidus's original scoring exactly (`FtsField::new("body")` is the untuned field):

```rust
use nidus::{Analyzer, FtsField, Language};

db.set_fts_schema("docs", &[
    // BM25: k1 is term-frequency saturation (default 1.2), b is length
    // normalization from 0 (off) to 1 (full; default 0.75).
    FtsField::new("body").k1(1.5).b(0.3),
    // Analyzer: fold Latin diacritics so "café" and "cafe" are one term, and drop
    // absurd tokens (a base64 blob, a minified bundle) before they reach the index.
    FtsField::new("title").ascii_folding(true).max_token_len(40),
    // The whole analyzer at once, if you prefer.
    FtsField::new("tags").analyzer(Analyzer::default().language(Language::English)),
])?;
# anyhow::Ok(())
```

Tuning is **per field**, not per store: `body` and `title` can score differently in the
same collection. Redeclaring a schema rebuilds the affected field indexes under the new
parameters: the parameters are part of the index cache's validity key, so a reopened
store never serves results scored under a schema you have since changed.

Over HTTP (and in the SDKs) the same knobs ride in the `fts-schema` body, where a bare
field name still means "all defaults":

```json
{"fields": ["title", {"field": "body", "k1": 1.5, "b": 0.3, "ascii_folding": true}]}
```

### What gets indexed, and how a query behaves

- **Analyzer.** US English today (`Language::English`): lowercase → Unicode word
  tokenization → English stopword removal → Porter stemming. Stemming means a query for
  `run` matches documents containing `running`, `runs`, or `ran`. The same analysis runs
  at index and query time, per field, so a field's own tuning applies to both. The
  `Language` enum is the seam for further languages.
- **What gets indexed.** `Str` attrs are indexed directly; `List` attrs are indexed
  per element. A document only lives in a field's index while it has text there.
- **`SearchOpts`.** `top_k`, `offset`, `filter`, `projection`, `rank_by`, and `limit_per`
  all work exactly as for vector search; only `min_score` differs, being a **raw BM25**
  floor rather than a cosine one. Results are tie-broken by `(collection, id)` for
  determinism, the same total order [pagination](#paginating-a-search) relies on.
- **Text-only documents.** A `Record` may carry no vector (`Record::text_only`), a
  pure full-text document. It is found by `text_search` and never by vector `search`.
  Vector-bearing and text-only docs coexist in one collection.

## Hybrid search (RRF)

[`hybrid_search`](/reference/api/#nidus) fuses a vector query and a BM25 text query
into one ranking using **Reciprocal Rank Fusion**: each leg is ranked independently,
and a document's fused score is `Σ 1 / (rrf_k + rank)` over the legs it appears in.

```rust
use nidus::{FtsQuery, HybridOpts};

let query_vector = vec![0.1_f32; 384];
let hits = db.hybrid_search(
    "docs",
    &query_vector,                       // the vector leg
    &FtsQuery::new("body", "vector database"), // the BM25 leg
    &HybridOpts { top_k: 10, ..Default::default() },
)?;
# anyhow::Ok(())
```

RRF fuses by **rank position**, not raw score, so the incomparable scales of cosine
(or euclidean/dot-product) and unbounded BM25 never need normalizing, and a document
that surfaces in only one leg (a strong vector match with weak text, or a text-only
doc) is still ranked. `HybridOpts` exposes `top_k`, `offset` (which pages the fused
ranking, never a leg), a `filter` applied to both legs, `rrf_k` (the rank-bias constant,
default 60), and `candidates` (how deep each leg is pulled before fusing, default 100).
There is no `min_score`: a fused RRF score has no absolute scale; threshold the
individual legs via `search` / `text_search` if you need a floor.

The text leg takes the same multi-clause `FtsQuery` as `text_search`: the clauses are
combined into one BM25 leg first, then fused with the vector leg, so a single-clause hybrid
query produces exactly the numbers it always did.

### Weighting the legs

`vector_weight` and `text_weight` scale each leg's contribution, so a document scores
`Σ wᵢ / (rrf_k + rankᵢ)`. Both default to `1.0`, which reproduces the unweighted fusion
exactly.

```rust
use nidus::{FtsQuery, HybridOpts};

let query_vector = vec![0.1_f32; 384];
// Lean on the keyword leg: exact terms matter more than semantic neighbourhood here.
let hits = db.hybrid_search(
    "docs",
    &query_vector,
    &FtsQuery::new("body", "CVE-2026-1234"),
    &HybridOpts { top_k: 10, text_weight: 3.0, ..Default::default() },
)?;
# anyhow::Ok(())
```

A weight must be finite and non-negative: a `NaN` would poison the sort and a negative
weight would invert a leg rather than de-emphasize it, so both are refused.

## Explaining a hit

A `Hit` carries one score and the record's attrs, which does not tell you *why* it matched.
`Hit::annotations` does. It is `None` unless the query asks, so nothing about the default
response changes.

```rust
use nidus::{FtsQuery, HighlightOpts, SearchOpts};

let q = FtsQuery::new("body", "run").highlight(HighlightOpts::default());
let hits = db.text_search(
    "docs",
    &q,
    &SearchOpts { top_k: 10, explain: true, ..Default::default() },
)?;

for hit in &hits {
    let a = hit.annotations.as_ref().unwrap();
    for clause in &a.clauses {
        println!("{} scored {}", clause.field, clause.score);
    }
    for hl in &a.highlights {
        for frag in &hl.fragments {
            for &(start, end) in &frag.spans {
                println!("{}: …{}…  matched {:?}", hl.field, frag.text, &frag.text[start..end]);
            }
        }
    }
}
# anyhow::Ok(())
```

- **`explain: true`** reports each *matched* clause's own BM25 score, in query order (a
  clause that did not match is absent, not a zero). A clause carries a **score only, no
  rank**: clauses are summed or maxed into one text score, so there is no separate ranking
  per clause for a rank to refer to. On `hybrid_search`, `HybridOpts::explain` additionally
  reports each fusion leg's own `(rank, score)` (the legs *are* ranked independently, which
  is exactly what RRF fuses), so you can see whether a document rode in on the vector leg,
  the text leg, or both. Vector `search` ignores `explain`: it has a single score to report.
- **`highlight`** returns fragments of the field's stored text with the byte ranges that
  matched. The offsets index the **original** text, not the analyzed token stream: a query
  for `run` highlights the word a document spells `running`, which no substring search would
  find. `HighlightOpts { max_fragments, fragment_chars }` bounds the output; fragments are
  snapped to word boundaries so they never open or close mid-word, and `spans` are byte
  offsets *within the fragment*.
- **Highlighting and [projection](#choosing-the-attrs-a-hit-carries) are independent.**
  Fragments are cut from the stored text, so dropping a 10 KB body from the payload and
  keeping only its snippet is the supported combination, not a silent no-op.
- Annotations are computed on the returned page, after pagination, so the cost scales with
  `top_k` rather than with the candidate set.

Full-text and hybrid search are a runtime feature over the same store. Like ANN and
quantization, they change nothing about the on-disk vector format, and a store opened
without declaring any FTS schema behaves exactly as before.
