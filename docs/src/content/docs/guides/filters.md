---
title: Filters & metadata
description: "The typed metadata model and nidus's filter language: equality, range, glob, set, containment, boolean, and text predicates, applied before scoring, plus metadata-only queries and aggregation."
---

Every record in nidus carries an open map of typed metadata beside its vector, and a
filter narrows a search to the records worth scoring. Filters apply **before scoring**
and work identically across [vector](/guides/search/),
[full-text](/guides/full-text-search/) and [hybrid](/guides/hybrid-search/) search, so
this page is the one description of them all three share.

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
a [recency ranking](/guides/search/#ranking-by-recency) can tell a time from a number without relying on
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
  [`text_search`](/guides/full-text-search/) for the same word does, because stemming belongs
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

:::caution[By default there is no index behind these]
Every predicate here re-tokenizes or re-scans the attribute for **each record the scan
visits**: O(attribute length) per row for the token family, and O(needle × attribute) for
the `Fuzzy` DP. That is fine at the scale nidus targets, and it is *not* a substitute for
full-text search over a large corpus. When the field is a document, reach for
[`text_search`](/guides/full-text-search/), which has an index of its own.

For the filtering case, [`set_filter_index`](#indexing-the-text-predicates) below makes
these predicates a lot faster without changing a single result.
:::

<a id="indexing-the-text-predicates"></a>

### Indexing the text predicates

Declaring a filter index on a field speeds up all five predicates above. It is **opt-in,
per collection and per field**, and off until you ask for it:

```rust
use nidus::FilterIndexField;

db.set_filter_index("docs", &[
    FilterIndexField::new("body"),
    // Only the token predicates for this one: no Fuzzy or Regex on tags.
    FilterIndexField::new("tag").trigrams(false),
])?;
```

Documents already written are indexed as part of the declaration, so you can add one to an
existing collection. Passing an empty list drops it again.

**The index changes how fast a query runs, never what it returns.** It proposes candidate
documents and the predicate itself still decides, so an indexed field and an unindexed one
give identical results. That is what makes it safe to turn on for a field you are unsure
about.

What it costs: the index is built as you write and held in memory. On a 10,000 document
corpus of 32 tokens each, ingest roughly doubled (53 ms to 114 ms) and
`footprint().filter_index_bytes` reports the memory in use.

What it saves, on the same corpus:

| Predicate | Unindexed | Indexed |
| --- | --- | --- |
| `Fuzzy` (1 edit) | 91.6 ms | 1.4 ms |
| `ContainsAllTokens` (4 terms) | 4.8 ms | 0.2 ms |
| `ContainsTokenSequence` (3 terms) | 6.6 ms | 1.0 ms |

Some queries will not speed up, and that is the index working as intended. When a term
appears in much of the collection there is nothing to narrow, so nidus skips the index and
runs the ordinary scan (those cases measured within noise of the unindexed times). The same
happens for a `Regex` with no required literal such as `.*`, for a `Fuzzy` whose edit budget
is wide relative to a short needle, and for anything under a `Not`.

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
[`projection`](/guides/search/#choosing-the-attrs-a-hit-carries), so a listing can return ids alone.

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
