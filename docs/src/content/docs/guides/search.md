---
title: Search & filters
description: Scoped search across nidus collections with three distance metrics, int8 quantization, metadata-only queries, and equality / glob / set / range filter predicates.
---

Search in nidus is exact brute-force over a scope you choose, using one of three
distance metrics, optionally narrowed by a metadata filter and a score floor.

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
