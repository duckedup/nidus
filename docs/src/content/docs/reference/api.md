---
title: API reference
description: The full nidus public surface — Nidus, Config, Record, Value, Filter, Predicate, Scope, SearchOpts, FtsQuery, HybridOpts, Hit, Footprint.
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
| `create_collection_with_fts` | `fn create_collection_with_fts(&mut self, name: &str, fields: &[(String, Language)]) -> Result<()>` | Create + declare [full-text fields](/guides/search/#full-text-search-bm25) up front (incremental from the first upsert). |
| `set_fts_schema` | `fn set_fts_schema(&mut self, collection: &str, fields: &[(String, Language)]) -> Result<()>` | Declare/redeclare full-text fields any time; indexes existing docs once. |

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
| `list` | `fn list<'a>(&self, scope: impl Into<Scope<'a>>, filter: &Filter, offset: usize, limit: usize) -> Result<Vec<Hit>>` | Metadata-only query — no vector, returns filter-matched records in insertion order; `offset`/`limit` paginate. |
| `search` | `fn search<'a>(&self, scope: impl Into<Scope<'a>>, query: &[f32], opts: &SearchOpts) -> Result<Vec<Hit>>` | Ranked search over a scope using the store's distance metric. |
| `text_search` | `fn text_search<'a>(&self, scope: impl Into<Scope<'a>>, query: &FtsQuery, opts: &SearchOpts) -> Result<Vec<Hit>>` | [BM25 full-text search](/guides/search/#full-text-search-bm25); `min_score` is a raw BM25 floor. |
| `hybrid_search` | `fn hybrid_search<'a>(&self, scope: impl Into<Scope<'a>>, vector: &[f32], text: &FtsQuery, opts: &HybridOpts) -> Result<Vec<Hit>>` | [Hybrid vector + BM25](/guides/search/#hybrid-search-rrf), fused with Reciprocal Rank Fusion. |
| `flush` | `fn flush(&mut self) -> Result<()>` | Force an fsync (relevant under `Fsync::OnFlush`). |
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
}
```

## `Predicate` & `Filter`

A `Filter` is a conjunction (AND) of predicates; an empty filter matches
everything. Every predicate is a positive assertion about a **present** attribute —
a record lacking `key` matches no predicate, including the negative (`Ne`/`NotIn`)
and range ones.

```rust
pub enum Predicate {
    Eq(String, Value),      // attrs[key] == value
    Ne(String, Value),      // attrs[key] present and != value
    Glob(String, String),   // attrs[key] is a Str matching the glob (* ? [..])
    In(String, Vec<Value>), // attrs[key] is one of the values
    NotIn(String, Vec<Value>), // attrs[key] present and not one of the values
    Lt(String, Value),      // attrs[key] <  value  (same-type, orderable)
    Le(String, Value),      // attrs[key] <= value
    Gt(String, Value),      // attrs[key] >  value
    Ge(String, Value),      // attrs[key] >= value
}

pub struct Filter(pub Vec<Predicate>);
```

The range predicates (`Lt`/`Le`/`Gt`/`Ge`) compare **same-type, orderable** values
only: `Int` numerically, `Str` lexically, `Bool` as `false < true`. A cross-type or
non-orderable (`Null`, `List`) comparison never matches.

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
    pub filter: Filter,          // pre-scoring metadata filter
    pub min_score: Option<f32>,  // drop results below this score
}
```

Implements `Default` — `SearchOpts { top_k: 5, ..Default::default() }` is the
idiomatic call. Reused by `text_search`, where `min_score` is a raw BM25 floor.

## `FtsQuery` & `Language`

A [full-text query](/guides/search/#full-text-search-bm25): the indexed field and the
raw query text (analyzed at query time the same way documents were at index time).

```rust
pub struct FtsQuery {
    pub field: String,  // a full-text-indexed attribute field
    pub text: String,   // raw query text
}

pub enum Language { English }  // the analyzer; extensible (US English today)
```

Construct with `FtsQuery::new(field, text)`.

## `HybridOpts`

Options for [hybrid search](/guides/search/#hybrid-search-rrf) (vector + BM25, fused
with Reciprocal Rank Fusion).

```rust
pub struct HybridOpts {
    pub top_k: usize,      // final result count
    pub filter: Filter,    // applied to both legs
    pub rrf_k: f32,        // RRF rank-bias constant (default 60)
    pub candidates: usize, // depth pulled per leg before fusing (default 100)
}
```

Implements `Default` (`top_k: 10`). There is no `min_score` — a fused RRF score has no
absolute scale.

## `Hit`

One search result. Carries its source collection and the matched attrs, but
**not** the vector.

```rust
pub struct Hit {
    pub collection: String,
    pub id: String,
    pub score: f32,   // meaning depends on the store's Distance metric
    pub attrs: BTreeMap<String, Value>,
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
