---
title: API reference
description: The full nidus public surface — Nidus, Config, Record, Value, Filter, Predicate, Scope, SearchOpts, FtsQuery, HybridOpts, ListOpts, Hit, Annotations, Footprint.
---

The complete public API. All fallible methods return `anyhow::Result`. For the
generated rustdoc, run `cargo doc --open` in the repository.

## `Nidus`

The open store. Synchronous — wrap in `Arc<RwLock<Nidus>>` for concurrent
searchers plus one writer (see
[Embedding in a host app](/guides/integrating/)).

### Opening

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `open` | `fn open(config: Config) -> Result<Self>` | Open, creating if absent. The full builder path. |
| `open_dir` | `fn open_dir(dir: impl AsRef<Path>, dimension: usize) -> Result<Self>` | Shorthand for `open(Config::new(dir, dimension))`. |
| `open_in_memory` | `fn open_in_memory(dimension: usize) -> Result<Self>` | No files, no lock — for tests and ephemeral use. |

### Introspection

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `dimension` | `fn dimension(&self) -> usize` | The pinned embedding dimension. |
| `config` | `fn config(&self) -> &Config` | The config the store was opened with. |
| `footprint` | `fn footprint(&self) -> Footprint` | A cheap snapshot of the vector footprint. |

### Collections

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `create_collection` | `fn create_collection(&mut self, name: &str) -> Result<()>` | |
| `drop_collection` | `fn drop_collection(&mut self, name: &str) -> Result<()>` | Drops the collection and its rows. |
| `has_collection` | `fn has_collection(&self, name: &str) -> bool` | |
| `collections` | `fn collections(&self) -> Vec<String>` | All collection names. |
| `get_meta` | `fn get_meta(&self, collection: &str) -> BTreeMap<String, String>` | Per-collection metadata. |
| `set_meta` | `fn set_meta(&mut self, collection: &str, meta: BTreeMap<String, String>) -> Result<()>` | |
| `create_collection_with_fts` | `fn create_collection_with_fts(&mut self, name: &str, fields: &[FtsField]) -> Result<()>` | Create + declare [full-text fields](/guides/search/#full-text-search-bm25) up front (incremental from the first upsert). |
| `set_fts_schema` | `fn set_fts_schema(&mut self, collection: &str, fields: &[FtsField]) -> Result<()>` | Declare/redeclare full-text fields any time; indexes existing docs once. Each [`FtsField`](/guides/search/#tuning-a-field) carries its own `k1`, `b`, and `Analyzer`. |

### Records

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `upsert` | `fn upsert(&mut self, collection: &str, records: &[Record]) -> Result<usize>` | Idempotent by id; all-or-nothing per batch. Returns rows written. |
| `delete` | `fn delete(&mut self, collection: &str, ids: &[&str]) -> Result<usize>` | Returns rows deleted. |
| `delete_where` | `fn delete_where(&mut self, collection: &str, filter: &Filter) -> Result<usize>` | Delete by metadata filter. |
| `get_all` | `fn get_all(&self, collection: &str) -> Vec<Record>` | Every live record in the collection. |

### Search & maintenance

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `list` | `fn list<'a>(&self, scope: impl Into<Scope<'a>>, opts: &ListOpts) -> Result<Vec<Hit>>` | Metadata-only query — no vector, returns filter-matched records in insertion order; `ListOpts`'s `offset`/`limit` paginate. |
| `search` | `fn search<'a>(&self, scope: impl Into<Scope<'a>>, query: &[f32], opts: &SearchOpts) -> Result<Vec<Hit>>` | Ranked search over a scope using the store's distance metric; `SearchOpts`'s `offset`/`top_k` paginate. |
| `text_search` | `fn text_search<'a>(&self, scope: impl Into<Scope<'a>>, query: &FtsQuery, opts: &SearchOpts) -> Result<Vec<Hit>>` | [BM25 full-text search](/guides/search/#full-text-search-bm25) over one or more field clauses; `min_score` is a raw BM25 floor. |
| `hybrid_search` | `fn hybrid_search<'a>(&self, scope: impl Into<Scope<'a>>, vector: &[f32], text: &FtsQuery, opts: &HybridOpts) -> Result<Vec<Hit>>` | [Hybrid vector + BM25](/guides/search/#hybrid-search-rrf), fused with Reciprocal Rank Fusion. |
| `aggregate` | `fn aggregate<'a>(&self, scope: impl Into<Scope<'a>>, opts: &AggregateOpts) -> Result<Aggregation>` | [Count and sum](/guides/search/#aggregation) over a filter, straight off the in-memory index — no record is materialized. |
| `flush` | `fn flush(&mut self) -> Result<()>` | Force an fsync (relevant under `Fsync::OnFlush`). |
| `deferred` | `fn deferred<T>(&mut self, f: impl FnOnce(&mut Nidus) -> Result<T>) -> Result<T>` | Run `f`'s mutations with their durable barrier deferred, so several can share one — see [group commit](/guides/how-it-works/#group-commit). **Report nothing successful until `commit` returns `Ok`**: until then the bytes are appended but not durable. |
| `commit` | `fn commit(&mut self) -> Result<()>` | Take one barrier covering everything appended by `deferred` (fsync `data`, then `log`). A no-op when no barrier is owed, so the ordinary path pays nothing. Narrower than `flush` — no segment seal, no working-set publish. |
| `compact` | `fn compact(&mut self) -> Result<()>` | Rewrite `data` to reclaim dead rows. |
| `refresh` | `fn refresh(&mut self) -> Result<bool>` | Adopt a separate writer's newer committed state into a lock-free [`ReadOnly`](/reference/configuration/#openmode) handle without reopening — picks up appends, deletes, seals, and compactions at one consistent point. Returns `true` when newer state was adopted, `false` when already current (the cheap case) or for a `ReadWrite`/in-memory handle. See [refreshing a reader](/guides/storage/#refreshing-a-reader). |
| `persist_index` | `fn persist_index(&mut self) -> Result<()>` | Write the [ANN index](#annconfig--annkind) to its `ann` cache so the next `open()` loads it instead of rebuilding the graph. Out-of-band (never on `upsert`/`flush`); no-op when ANN is off, in-memory, or read-only. `compact()` refreshes it too. |

## `Scope`

Which collections a search ranks over. Accepts `impl Into<Scope>`, so `&str` and
`&[&str]` coerce automatically.

```rust
pub enum Scope<'a> {
    Collection(&'a str),       // one collection — the common, fast path
    Collections(&'a [&'a str]), // a chosen subset
    All,                        // every collection in the store
}
```

Scores are comparable across collections because the whole store shares one
embedding space.

## `Record`

```rust
pub struct Record {
    pub id: String,                  // caller-supplied; the upsert key
    pub vector: Option<Vec<f32>>,    // Some: length == dimension; None: text-only
    pub attrs: BTreeMap<String, Value>,
}
```

Construct with `Record::new(id, vector, attrs)` for a vector-bearing document, or
`Record::text_only(id, attrs)` for a document with no embedding (indexed purely by
[full-text search](/guides/search/#full-text-search-bm25)). Over the wire / in backups
the `vector` field may be omitted, which deserializes to `None`.

## `Value`

A typed metadata value. `Null` is **distinct from an absent key** — see
[typed metadata](/guides/search/#typed-metadata).

```rust
pub enum Value {
    Null,
    Str(String),
    Int(i64),
    Bool(bool),
    List(Vec<String>),
    Float(f64),      // IEEE: NaN matches nothing, 0.0 == -0.0
    DateTime(i64),   // UTC epoch milliseconds
}
```

`Float` and `Int` are **not** interchangeable: comparisons are same-type only, so
`Ge("score", Float(0.5))` does not match a record storing `Int(1)`. `NaN` is unordered
and unequal to itself, so it fails every predicate, `Eq("k", NaN)` included.

`DateTime` is an absolute instant in UTC epoch milliseconds — there is no timezone and
no local-time form. It is distinct from `Int` so a filter or a recency ranking can tell
a time from a number without relying on a naming convention.

## `Predicate` & `Filter`

A `Filter` is a conjunction (AND) of predicates; an empty filter matches
everything. Every *leaf* predicate is a positive assertion about a **present**
attribute — a record lacking `key` matches no leaf predicate, including the negative
(`Ne`/`NotIn`/`NotContains`) and range ones.

```rust
pub enum Predicate {
    Eq(String, Value),      // attrs[key] == value
    Ne(String, Value),      // attrs[key] present and != value
    Glob(String, String),   // attrs[key] is a Str matching the glob (* ? [..])
    IGlob(String, String),  // same, ignoring ASCII case
    In(String, Vec<Value>), // attrs[key] is one of the values
    NotIn(String, Vec<Value>), // attrs[key] present and not one of the values
    Lt(String, Value),      // attrs[key] <  value  (same-type, orderable)
    Le(String, Value),      // attrs[key] <= value
    Gt(String, Value),      // attrs[key] >  value
    Ge(String, Value),      // attrs[key] >= value

    Contains(String, Value),         // attrs[key] is a List holding value
    NotContains(String, Value),      // attrs[key] is a present List not holding value
    ContainsAny(String, Vec<Value>), // attrs[key] is a List overlapping the set

    All(Vec<Predicate>),    // every sub-predicate holds  (empty = true)
    Any(Vec<Predicate>),    // some sub-predicate holds   (empty = false)
    Not(Box<Predicate>),    // the sub-predicate does not hold
}

pub struct Filter(pub Vec<Predicate>);
```

The range predicates (`Lt`/`Le`/`Gt`/`Ge`) compare **same-type, orderable** values
only: `Int` numerically, `Str` lexically, `Bool` as `false < true`. A cross-type or
non-orderable (`Null`, `List`) comparison never matches.

`Contains`/`NotContains`/`ContainsAny` look inside a `List`, matching whole elements
rather than substrings — `Contains("tags", "rust")` does not match `["rustacean"]`.

`All`/`Any`/`Not` are predicates over predicates, so arbitrary boolean shapes nest
without `Filter` itself changing. Note `Not` differs from `Ne` on a **missing**
attribute: `Ne(k, v)` is false (it requires `k` present), while `Not(Eq(k, v))` is
true. Use `Ne`/`NotIn`/`NotContains` to require presence, `Not` for set complement.

## `Distance`

The similarity / distance metric, set at store creation via `Config::distance`.
Pinned in the data header — reopening with a different metric is an error.

```rust
pub enum Distance {
    Cosine,      // default — vectors normalized on insert, score = dot(q, v)
    Euclidean,   // raw vectors, score = −‖q − v‖²
    DotProduct,  // raw vectors, score = dot(q, v)
}
```

For all metrics, higher score = more relevant.

## `SearchOpts`

```rust
pub struct SearchOpts {
    pub top_k: usize,            // maximum number of results
    pub offset: usize,           // top-ranked results to skip, for pagination
    pub filter: Filter,          // pre-scoring metadata filter
    pub min_score: Option<f32>,  // drop results below this score
    pub exact: bool,             // force the exact scan for this query
    pub projection: Projection,  // which attrs the hits carry
    pub explain: bool,           // annotate each hit with its per-clause BM25 scores
    pub rank_by: Option<RankBy>, // a ranking expression over the metric
    pub limit_per: Option<LimitPer>, // cap hits per attribute value
}
```

Implements `Default` (`offset: 0`, `exact: false`, `explain: false`,
`projection: Projection::All`, `rank_by: None`, `limit_per: None`) —
`SearchOpts { top_k: 5, ..Default::default() }` is the idiomatic call. Reused by
`text_search`, where `min_score` is a raw BM25 floor.

Results are ordered by `(score desc, collection, id)`. The ranking is computed
`offset + top_k` deep and the page cut once, at the end; an `offset` past the last
result is an empty `Vec`, not an error. See
[paginating a search](/guides/search/#paginating-a-search).

`exact: true` bypasses the ANN walk, the per-segment index, and the quantized first
pass, running the exact brute-force scan for that one query — the index stays in place
for every other. See [forcing an exact search](/guides/search/#forcing-an-exact-search).

## `RankBy` & `Decay`

An opt-in [ranking expression](/guides/search/#ranking-by-recency) layered over the
store's distance metric. `None` (the default) is the bare metric.

```rust
pub enum RankBy {
    Decay(Decay),   // subtract a recency penalty from every base score
}

pub struct Decay {
    pub field: String,  // timestamp attr: Value::DateTime or Value::Int, epoch millis
    pub origin: i64,    // "now", supplied by the caller so a ranking is reproducible
    pub scale: i64,     // the age (ms) at which the factor equals `decay`
    pub decay: f32,     // factor at one `scale` of age — default 0.5 (a half-life)
    pub lambda: f32,    // score a fully-decayed hit gives up — default 1.0
    pub missing: f32,   // factor when the attr is absent/unusable — default 1.0
}
```

Build one with `Decay::new(field, origin, scale)` plus `.decay(_)` / `.lambda(_)` /
`.missing(_)`. The score is `base − lambda × (1 − decay^(age / scale))` — the penalty
**subtracts**, which is what keeps it valid for `Euclidean` and `DotProduct` scores and
for raw BM25, not just cosine.

`missing` defaults to `1.0`, so a record with no timestamp is **not** penalized —
enabling decay never buries data that predates the field. `rank_by` does not force the
exact path; over an ANN or quantized result set it reorders within an approximate
candidate set. A ranked scan runs single-threaded.

## `LimitPer`

A cap on how many hits may carry any one value of an attribute — see
[capping hits per attribute value](/guides/search/#capping-hits-per-attribute-value).

```rust
pub struct LimitPer {
    pub field: String,  // the attribute whose distinct values define the groups
    pub max: usize,     // maximum hits per distinct value (at least 1)
}
```

Build with `LimitPer::new(field, max)`. Records **missing** the attribute form one shared
group, and the value is read from the stored record, so a `Projection` cannot lift the cap.
Deliberately approximate: exact only within the over-fetch window, so a capped page may come
back shorter than `top_k`.

## `OrderBy`

Sort a `list` by an attribute instead of storage order.

```rust
pub struct OrderBy {
    pub field: String,
    pub descending: bool,
}
```

Build with `OrderBy::asc(field)` / `OrderBy::desc(field)`. Values that do not order against
the first orderable one — a different variant, an unorderable `Null`/`List`, or an absent
attribute — sort into one trailing bucket, which stays trailing when reversed.

## `AggregateOpts` & `Aggregation`

[Count and sum](/guides/search/#aggregation) over a filter, answered from the in-memory
index without materializing a record.

```rust
pub struct AggregateOpts {
    pub filter: Filter,   // default matches every record
    pub sum: Vec<String>, // attributes to total
}

pub struct Aggregation {
    pub count: u64,
    pub sums: BTreeMap<String, Value>,  // Int while every addend was Int, else Float
}
```

A missing or non-numeric value is skipped, not counted as zero.

## `Projection`

Which attrs a returned [`Hit`](#hit) carries. Default `All`.

```rust
pub enum Projection {
    All,                    // every attr (the default)
    Include(Vec<String>),   // only these
    Exclude(Vec<String>),   // everything but these
}
```

Build one with `Projection::include([...])` / `Projection::exclude([...])`. It is
applied where a hit is materialized, so an excluded attr is never cloned — the payload
saving on a long-body collection is real. Ranking and scores are unaffected. An enum
rather than two lists, so "include and exclude at once" cannot be expressed; the HTTP
surface answers `400` for the wire form that sends both.

## `FtsQuery`, `FtsClause`, `FtsCombine` & `Language`

A [full-text query](/guides/search/#full-text-search-bm25): one or more clauses, each
naming an indexed field *and its own* raw query text (analyzed at query time the same way
documents were at index time).

```rust
pub struct FtsQuery {
    pub clauses: Vec<FtsClause>,           // at least one; empty is an error
    pub combine: FtsCombine,               // how clause scores fold (default Sum)
    pub highlight: Option<HighlightOpts>,  // None = no fragments (the default)
}

pub struct FtsClause {
    pub field: String,  // a full-text-indexed attribute field
    pub text: String,   // raw query text for this field
}

pub enum FtsCombine { Sum, Max }  // add every matched clause, or take the strongest
pub enum Language { English }     // the analyzer; extensible (US English today)
```

`FtsQuery::new(field, text)` is the one-clause shorthand; `FtsQuery::multi([...])` takes
several, with `.combine(...)` and `.highlight(...)` builders. See
[searching several fields at once](/guides/search/#searching-several-fields-at-once).

## `HighlightOpts`, `Annotations` & friends

The opt-in [explanation of a hit](/guides/search/#explaining-a-hit).

```rust
pub struct HighlightOpts {
    pub max_fragments: usize,   // fragments per field (default 1)
    pub fragment_chars: usize,  // characters per fragment (default 160)
}

pub struct Annotations {
    pub vector: Option<LegScore>,     // the vector leg's rank + score (hybrid only)
    pub text: Option<LegScore>,       // the BM25 leg's rank + score (hybrid only)
    pub clauses: Vec<ClauseScore>,    // each matched clause's own BM25 score
    pub highlights: Vec<Highlight>,   // fragments, one entry per matched field
}

pub struct LegScore    { pub rank: usize, pub score: f32 }
pub struct ClauseScore { pub field: String, pub score: f32 }
pub struct Highlight   { pub field: String, pub fragments: Vec<Fragment> }
pub struct Fragment {
    pub text: String,                 // an excerpt of the stored text
    pub spans: Vec<(usize, usize)>,   // matched byte ranges *within* `text`
}
```

Fragment offsets index the **original** text, not the analyzed tokens — a query for `run`
highlights a document's `running`. Highlighting reads the stored value, so it is unaffected
by [`Projection`](#projection).

## `HybridOpts`

Options for [hybrid search](/guides/search/#hybrid-search-rrf) (vector + BM25, fused
with Reciprocal Rank Fusion).

```rust
pub struct HybridOpts {
    pub top_k: usize,      // final result count
    pub offset: usize,     // fused results to skip, for pagination
    pub filter: Filter,    // applied to both legs
    pub rrf_k: f32,        // RRF rank-bias constant (default 60)
    pub candidates: usize, // depth pulled per leg before fusing (default 100)
    pub explain: bool,     // annotate each hit with per-leg and per-clause scores
    pub vector_weight: f32, // weight on the vector leg (default 1.0)
    pub text_weight: f32,   // weight on the BM25 leg (default 1.0)
}
```

Implements `Default` (`top_k: 10`, `offset: 0`, `explain: false`, both weights `1.0`).
`offset` pages the **fused** ranking, never a leg. There is no `min_score` — a fused RRF
score has no absolute scale. Both weights at `1.0` reproduce the unweighted fusion exactly;
a non-finite or negative weight is refused. See
[weighting the legs](/guides/search/#weighting-the-legs).

## `ListOpts`

Options for the metadata-only `list` query.

```rust
pub struct ListOpts {
    pub offset: usize,          // matches to skip, for pagination
    pub limit: usize,           // maximum records returned (default 100)
    pub filter: Filter,         // metadata filter; default matches everything
    pub projection: Projection, // which attrs the hits carry
    pub order_by: Option<OrderBy>, // sort by an attribute instead of storage order
}
```

Implements `Default` (`order_by: None`) — `ListOpts { limit: 20, ..Default::default() }`
is the idiomatic call. Sorting runs over the whole match set before the page is cut, so
`offset`/`limit` walk the sorted order.

## `Hit`

One search result. Carries its source collection and the matched attrs, but
**not** the vector. `#[non_exhaustive]`: build one with `Hit::new`.

```rust
#[non_exhaustive]
pub struct Hit {
    pub collection: String,
    pub id: String,
    pub score: f32,   // meaning depends on the store's Distance metric
    pub attrs: BTreeMap<String, Value>,
    pub annotations: Option<Annotations>,  // why it matched; None unless asked
}

impl Hit {
    pub fn new(
        collection: impl Into<String>,
        id: impl Into<String>,
        score: f32,
        attrs: BTreeMap<String, Value>,
    ) -> Self;
}
```

## `Footprint`

A cheap, allocation-free snapshot for deciding whether more data fits before a
memory ceiling. Pairs with
[`Config::max_vector_bytes`](/reference/configuration/#max_vector_bytes).

```rust
pub struct Footprint {
    pub rows: u64,          // physical rows (live + not-yet-compacted dead)
    pub dead_rows: u64,     // reclaimable by compact()
    pub dimension: usize,
    pub vector_bytes: u64,  // rows * dimension * 4 — what max_vector_bytes caps
    pub doc_count: usize,   // live documents across all collections
}
```

## `Quantization`

Configuration for int8 scalar quantization. Pass to `Config::quantization` to
enable two-pass search (int8 first-pass → f32 rerank).

```rust
pub struct Quantization {
    pub rescore: usize,  // overscan factor (default 4)
}
```

## `AnnConfig` & `AnnKind`

Configuration for the opt-in approximate-nearest-neighbour index. Pass to
`Config::ann` to walk an index instead of scanning every vector. Construct with
`AnnConfig::hnsw()` or `AnnConfig::ivf()` and adjust via the builder setters. See the
[approximate search guide](/guides/search/#approximate-search-ann).

```rust
pub enum AnnKind { Hnsw, Ivf }

pub struct AnnConfig {
    pub kind: AnnKind,
    pub m: usize,               // HNSW: neighbours/node (default 16)
    pub ef_construction: usize, // HNSW: build beam width (default 200)
    pub ef_search: usize,       // HNSW: query beam width (default 64)
    pub n_lists: usize,         // IVF: centroids; 0 = auto ~sqrt(n)
    pub n_probe: usize,         // IVF: lists scanned per query (default 8)
    pub overscan: usize,        // candidate over-fetch multiple (default 4)
    pub seed: u64,              // build PRNG seed (deterministic)
}

// Builders: AnnConfig::hnsw(), AnnConfig::ivf()
// Setters:  .m(), .ef_construction(), .ef_search(), .n_lists(), .n_probe(),
//           .overscan(), .seed()
```

May be combined with [`Quantization`](#quantization): the index walk then scores
quantized codes for cheaper candidate selection, and the exact f32 rerank over the
resulting candidates restores accuracy.
