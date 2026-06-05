---
title: Search & filters
description: Scoped search across nidus collections with three distance metrics, typed metadata, and the Eq / Glob / In filter predicates applied before scoring.
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
]);
# anyhow::Ok(())
```

The three predicates:

- **`Eq(key, value)`** — `attrs[key]` equals `value` exactly (any `Value` type).
- **`Glob(key, pattern)`** — `attrs[key]` is a `Str` matching the glob. Supports
  `*`, `?`, and `[...]` character classes.
- **`In(key, values)`** — `attrs[key]` equals one of the values in the set.

Filters can also drive deletion without a search:

```rust
// Delete every record whose path is under src/legacy/
db.delete_where("code", &Filter(vec![
    Predicate::Glob("path".into(), "src/legacy/*".into()),
]))?;
# anyhow::Ok(())
```
