---
title: Search & filters
description: Scoped search across nidus collections with three distance metrics, exact or approximate (HNSW/IVF) indexing, int8 quantization, metadata-only queries, and equality / glob / set / range filter predicates.
---

Search in nidus runs over a scope you choose, using one of three distance metrics,
optionally narrowed by a metadata filter and a score floor. It is **exact by
default** — every in-scope vector is scored — and can opt into an
[approximate index](#approximate-search-ann) (HNSW or IVF) when a full scan is more
than you want to pay.

## Distance metrics

The distance metric is set at store creation via `Config::distance` and pinned
in the data header — reopening with a different metric is an error.

| Metric | Normalization | Score | Range | Best for |
| --- | --- | --- | --- | --- |
| `Cosine` (default) | Vectors unit-normalized on insert | `dot(q, v)` | \[−1, 1\] | Embedding similarity |
| `Euclidean` | Stored as-is | `−‖q − v‖²` | (−∞, 0\] | Spatial distance |
| `DotProduct` | Stored as-is | `dot(q, v)` | (−∞, ∞) | When magnitude matters |

For all metrics, **higher score = more relevant**, so top-k, `min_score`, and
ranking all work the same way regardless of which metric you choose.

```rust
use nidus::{Config, Distance, Nidus};

// Cosine (default — same as before)
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

db.search("code", &q, &opts)?;                              // one collection (&str coerces)
db.search(Scope::Collections(&["code", "docs"]), &q, &opts)?; // a named subset
db.search(Scope::All, &q, &opts)?;                          // the whole store
# anyhow::Ok(())
```

This is sound because **all collections share one embedding space**. The
dimension is pinned at store creation, so a vector in `code` and a vector in
`docs` are directly comparable — one ranking over both is meaningful, not a
category error.

## Scoring

`SearchOpts` controls the ranking:

```rust
use nidus::SearchOpts;

let opts = SearchOpts {
    top_k: 10,             // keep at most this many hits
    min_score: Some(0.5),  // drop anything below this score (None = no floor)
    ..Default::default()
};
# anyhow::Ok(())
```

`top_k` is enforced by a bounded heap, so memory stays flat regardless of how
many rows are scored.

## Typed metadata

Each record carries an open map of typed [`Value`](/reference/api/#value)s:

| Variant       | Meaning                                              |
| ------------- | ---------------------------------------------------- |
| `Str(String)` | A string.                                            |
| `Int(i64)`    | A signed 64-bit integer.                             |
| `Bool(bool)`  | A boolean.                                           |
| `List(Vec<String>)` | A list of strings (e.g. tags).                 |
| `Null`        | Set, but empty — **distinct from an absent key**.    |

The `Null`-vs-absent distinction is deliberate: absence means "not computed / not
indexed", while `Null` means "computed, and empty". A host app uses it to tell an
un-indexed field apart from a field that was indexed and found to be empty.

## Filters

A [`Filter`](/reference/api/#filter) is a conjunction (AND) of
[`Predicate`](/reference/api/#predicate)s, applied **before scoring** — matching
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

- **`Eq(key, value)`** / **`Ne(key, value)`** — `attrs[key]` equals / does not equal
  `value` (any `Value` type, typed).
- **`Glob(key, pattern)`** — `attrs[key]` is a `Str` matching the glob. Supports
  `*`, `?`, and `[...]` character classes.
- **`In(key, values)`** / **`NotIn(key, values)`** — `attrs[key]` is / is not one of
  the values in the set.
- **`Lt` / `Le` / `Gt` / `Ge(key, value)`** — ordered range comparison, **same-type
  and orderable only**: `Int` numeric, `Str` lexical, `Bool` (`false < true`). A
  cross-type or non-orderable (`Null`, `List`) comparison never matches.

Every predicate is a positive assertion about a **present** attribute: a record that
lacks `key` matches *nothing* — including `Ne` / `NotIn` and the range predicates. (So
`Ne("status", "archived")` does not match a record with no `status` at all.) There is
no OR/disjunction — a `Filter` is always an AND; compose at the call site if you need
alternatives.

Filters can also drive deletion without a search:

```rust
// Delete every record whose path is under src/legacy/
db.delete_where("code", &Filter(vec![
    Predicate::Glob("path".into(), "src/legacy/*".into()),
]))?;
# anyhow::Ok(())
```

## Metadata-only queries

Use `list` to retrieve records by metadata filter without a vector query. Results
come back in insertion order with `score: 0.0`. The `offset` and `limit` arguments
paginate a stable ordering — advance `offset` by `limit` to page through.

```rust
use nidus::{Filter, Predicate, Value};

let filter = Filter(vec![
    Predicate::Eq("lang".into(), Value::Str("rust".into())),
]);

// First page: offset 0, up to 100 matches.
let page1 = db.list("code", &filter, 0, 100)?;
// Next page.
let page2 = db.list("code", &filter, 100, 100)?;
# anyhow::Ok(())
```

`list` accepts a [`Scope`](/reference/api/#scope) just like `search`, so you can
list across multiple collections or the whole store.

## int8 scalar quantization

For larger collections, enable int8 scalar quantization to speed up search.
The store keeps an int8 copy of every vector in RAM (global **symmetric**
quantization — one shared scale, no per-dimension offset, so the int8 score
stays monotonic with the true score). Search then runs in two passes: a fast
int8 first-pass selects candidates, overscanning by the `rescore` factor, then
the original f32 vectors re-rank those candidates for an exact final ranking.

```rust
use nidus::{Config, Quantization};

let db = Nidus::open(
    Config::new("./store", 768)
        .quantization(Some(Quantization { rescore: 4 }))
)?;
# anyhow::Ok(())
```

The `rescore` factor trades recall for speed: `rescore: 4` means the int8 pass
keeps `top_k * 4` candidates before the f32 re-rank. Higher values widen the
candidate net (better recall, more f32 work); the default is 4.

**What to expect.** In the `just bench-quant` sweep (uniform-random vectors,
a near-worst case for quantization recall), the two-pass search returns
essentially the exact neighbours — **~100% recall@10 at `rescore` ≥ 2** — for a
**~1.4× query speedup** at 1M × 768, in exchange for **~25% more RAM** (the int8
copy sits alongside the f32 matrix, which the re-rank still needs). The speedup
is bounded by the pure-safe-Rust scalar int8 kernel; the larger theoretical win
would need SIMD int8 dot-product intrinsics, which are `unsafe` and outside
nidus's zero-FFI design. Run `just bench-quant` to measure on your own shapes.

Quantization is purely a runtime optimization — it doesn't change the on-disk
format, and a store opened without it produces identical results (just slower
for large scans). Reach for it when search latency matters more than the extra
RAM. Vectors quantize incrementally on upsert, so adding records stays cheap.

## Approximate search (ANN)

By default every search is **exact** — it scores every in-scope vector. When a
collection grows past the point where a full scan is more than you want to pay,
`Config::ann` opts into an **approximate** index that walks a much smaller candidate
set instead. It is purely a runtime choice: nothing about the on-disk format changes,
and a store opened without it behaves exactly as before.

```rust
use nidus::{Config, Nidus, AnnConfig};

// HNSW — a navigable small-world graph (the default ANN index):
let db = Nidus::open(Config::new("./store", 768).ann(Some(AnnConfig::hnsw())))?;

// or IVF — k-means inverted lists:
let db2 = Nidus::open(Config::new("./store2", 768).ann(Some(AnnConfig::ivf())))?;
# anyhow::Ok(())
```

Both index types work the same way at query time: the index selects an over-fetched
candidate set (`top_k × overscan`), then nidus applies your scope, metadata filter,
and `min_score` to those candidates and ranks the survivors by the **exact** f32
score. So the *final ordering is always exact* — only the candidate *selection* is
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
`open()` — and for HNSW that build is the expensive part. Call
[`db.persist_index()`](/reference/api/#nidus) to write it to an `ann` cache file so
the next `open()` *loads* it instead of rebuilding. It's an explicit, out-of-band
operation (also triggered by `compact()`) — never on the `upsert`/`flush` write path,
so storage stays fast — so call it before shutting down a long-lived handle. `open()`
incrementally catches up any rows added since the cache was written, and silently
rebuilds if the cache is missing, stale, or for a different config. The cache is
derived data: deleting the `ann` file only costs a one-time rebuild.

**Trade-offs to know.**

- **Approximate recall.** ANN may miss a true neighbour. Raise `ef_search`/`n_probe`
  and `overscan` to trade speed for recall.
- **Selective filters.** Because the filter is applied *after* the walk, a very
  selective filter or a narrow collection-subset scope can leave too few matching
  candidates — recall drops in that case. If you mostly run highly-selective filtered
  queries, exact search (the default) may serve you better.
- **Deletes.** A deleted vector leaves a stale node in the index that is skipped at
  query time and fully reclaimed on the next `compact()`.
- **Combine with quantization.** `ann` and `quantization` can be enabled together: the
  index walk scores quantized codes (a cheaper candidate selection) and the exact f32
  rerank over those candidates restores accuracy. Recall runs a little below the
  exact-walk index, so widen `ef_search`/`n_probe` and `overscan` if you need it back.
