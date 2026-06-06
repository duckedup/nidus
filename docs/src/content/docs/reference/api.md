---
title: API reference
description: The full nidus public surface — Nidus, Config, Record, Value, Filter, Predicate, Scope, SearchOpts, Hit, Footprint.
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
| `list` | `fn list<'a>(&self, scope: impl Into<Scope<'a>>, filter: &Filter, limit: usize) -> Result<Vec<Hit>>` | Metadata-only query — no vector, returns filter-matched records in insertion order. |
| `search` | `fn search<'a>(&self, scope: impl Into<Scope<'a>>, query: &[f32], opts: &SearchOpts) -> Result<Vec<Hit>>` | Ranked search over a scope using the store's distance metric. |
| `flush` | `fn flush(&mut self) -> Result<()>` | Force an fsync (relevant under `Fsync::OnFlush`). |
| `compact` | `fn compact(&mut self) -> Result<()>` | Rewrite `data` to reclaim dead rows. |

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
    pub id: String,               // caller-supplied; the upsert key
    pub vector: Vec<f32>,         // length must equal the store dimension
    pub attrs: BTreeMap<String, Value>,
}
```

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
everything.

```rust
pub enum Predicate {
    Eq(String, Value),     // attrs[key] == value
    Glob(String, String),  // attrs[key] is a Str matching the glob (* ? [..])
    In(String, Vec<Value>),// attrs[key] is one of the values
}

pub struct Filter(pub Vec<Predicate>);
```

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
idiomatic call.

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
