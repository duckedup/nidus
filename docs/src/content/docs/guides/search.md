---
title: Search & filters
description: Scoped cosine search across nidus collections, typed metadata, and the Eq / Glob / In filter predicates applied before scoring.
---

Search in nidus is exact brute-force cosine, optionally narrowed by a metadata
filter and a score floor, over a scope you choose. This page covers all three.

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

Vectors are **unit-normalized on insert**, so the cosine similarity of a stored
vector `v` and a query `q` reduces to their dot product. A `Hit.score` is
therefore plain cosine in `[-1, 1]`: `1.0` is identical direction, `0.0`
orthogonal, `-1.0` opposite.

`SearchOpts` controls the ranking:

```rust
use nidus::SearchOpts;

let opts = SearchOpts {
    top_k: 10,             // keep at most this many hits
    min_score: Some(0.5),  // drop anything below this cosine (None = no floor)
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
