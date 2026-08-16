---
title: API reference
description: The full nidus public surface (Nidus, Config, OpenProfile, Record, Value, Filter, Predicate, Scope, SearchOpts, Projection, RankBy, Decay, LimitPer, OrderBy, AggregateOpts, FtsQuery, HybridOpts, ListOpts, Hit, Annotations, Footprint).
---

The complete public API. All fallible methods return `anyhow::Result`. For the
generated rustdoc, run `cargo doc --open` in the repository.

## `Nidus`

The open store. Synchronous: wrap in `Arc<RwLock<Nidus>>` for concurrent
searchers plus one writer (see
[Embedding in a host app](/guides/integrating/)).

### Opening

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `open` | `fn open(config: Config) -> Result<Self>` | Open, creating if absent. The full builder path. |
| `open_dir` | `fn open_dir(dir: impl AsRef<Path>, dimension: usize) -> Result<Self>` | Shorthand for `open(Config::new(dir, dimension))`. |
| `open_in_memory` | `fn open_in_memory(dimension: usize) -> Result<Self>` | No files, no lock; for tests and ephemeral use. |

### Introspection

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `dimension` | `fn dimension(&self) -> usize` | The pinned embedding dimension. |
| `config` | `fn config(&self) -> &Config` | The **effective** config: the caller's explicit settings merged with any [`OpenProfile`](#openprofile) defaults recorded in the store, an explicit setting always winning. |
| `footprint` | `fn footprint(&self) -> Footprint` | A cheap snapshot of the vector footprint. |
| `cluster_status` | `fn cluster_status(&self) -> ClusterStatus` | Role, writer-handle state, fencing token, commit counter, staleness (what [`GET /cluster`](/reference/http-api/#get-cluster) reports). |

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
| `set_filter_index` | `fn set_filter_index(&mut self, collection: &str, fields: &[FilterIndexField]) -> Result<()>` | Declare/redeclare [filter-indexed fields](/guides/search/#indexing-the-text-predicates) any time; indexes existing docs once. Speeds up the text predicates, changes no results. Empty list drops it. |

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
| `list` | `fn list<'a>(&self, scope: impl Into<Scope<'a>>, opts: &ListOpts) -> Result<Vec<Hit>>` | Metadata-only query: no vector, returns filter-matched records in insertion order (or by [`ListOpts::order_by`](#orderby)); `offset`/`limit` paginate. |
| `search` | `fn search<'a>(&self, scope: impl Into<Scope<'a>>, query: &[f32], opts: &SearchOpts) -> Result<Vec<Hit>>` | Ranked search over a scope using the store's distance metric; `SearchOpts`'s `offset`/`top_k` paginate. |
| `text_search` | `fn text_search<'a>(&self, scope: impl Into<Scope<'a>>, query: &FtsQuery, opts: &SearchOpts) -> Result<Vec<Hit>>` | [BM25 full-text search](/guides/search/#full-text-search-bm25) over one or more field clauses; `min_score` is a raw BM25 floor. |
| `hybrid_search` | `fn hybrid_search<'a>(&self, scope: impl Into<Scope<'a>>, vector: &[f32], text: &FtsQuery, opts: &HybridOpts) -> Result<Vec<Hit>>` | [Hybrid vector + BM25](/guides/search/#hybrid-search-rrf), fused with Reciprocal Rank Fusion. |
| `aggregate` | `fn aggregate<'a>(&self, scope: impl Into<Scope<'a>>, opts: &AggregateOpts) -> Result<Aggregation>` | [Count and sum](/guides/search/#aggregation) over a filter, straight off the in-memory index; no record is materialized. |
| `flush` | `fn flush(&mut self) -> Result<()>` | Force an fsync (relevant under `Fsync::OnFlush`). |
| `deferred` | `fn deferred<T>(&mut self, f: impl FnOnce(&mut Nidus) -> Result<T>) -> Result<T>` | Run `f`'s mutations with their durable barrier deferred, so several can share one; see [group commit](/guides/how-it-works/#group-commit). **Report nothing successful until `commit` returns `Ok`**: until then the bytes are appended but not durable. |
| `commit` | `fn commit(&mut self) -> Result<()>` | Take one barrier covering everything appended by `deferred` (fsync `data`, then `log`). A no-op when no barrier is owed, so the ordinary path pays nothing. Narrower than `flush`: no segment seal, no working-set publish. |
| `compact` | `fn compact(&mut self) -> Result<()>` | Rewrite `data` to reclaim dead rows. |
| `sweep_expired` | `fn sweep_expired(&mut self) -> Result<usize>` | Delete every entry across every collection whose `nidus.expires_at` has passed, then compact to reclaim the rows, in one call. Returns the number of entries deleted. Available in every build, no feature flag. |
| `refresh` | `fn refresh(&mut self) -> Result<bool>` | Adopt a separate writer's newer committed state into a lock-free [`ReadOnly`](/reference/configuration/#openmode) handle without reopening; it picks up appends, deletes, seals, and compactions at one consistent point. Returns `true` when newer state was adopted, `false` when already current (the cheap case) or for a `ReadWrite`/in-memory handle. See [refreshing a reader](/guides/storage/#refreshing-a-reader). |
| `persist_index` | `fn persist_index(&mut self) -> Result<()>` | Write the derived index caches: the [ANN index](#annconfig--annkind) to its `ann` cache and the full-text index to its `fts` cache, so the next `open()` loads them instead of rebuilding. Out-of-band (never on `upsert`/`flush`); no-op for whichever index is off, and when in-memory or read-only. `compact()` refreshes them too. |
| `open_profile` | `fn open_profile(&self) -> &OpenProfile` | The profile currently recorded in the manifest. Empty when nothing has been recorded. |
| `set_open_profile` | `fn set_open_profile(&mut self, p: &OpenProfile) -> Result<()>` | Record `p` as this store's open-time default for `ann`/`quantization`/`query_threads`/`mmap`, so a later `open()` with no explicit setting for a knob picks it up. Build `p` with [`Config::to_profile`](#config), which captures only the knobs that config set explicitly. Replaces the recorded profile wholesale, so merge onto `open_profile()` first if you mean to add one knob. Rejected on a read-only store, and rejected if the resulting combination could not be opened. |
| `clear_open_profile` | `fn clear_open_profile(&mut self) -> Result<()>` | Remove the recorded profile. Later opens fall back to built-in defaults unless a knob is set explicitly. |

## `Cancel`

A shared "stop what you are doing" flag for cooperative cancellation of a long scan,
e.g. what the HTTP server installs to enforce a request deadline. Cheap to clone:
every clone shares one signal.

```rust
pub struct Cancel(/* shared atomic flag */);

impl Cancel {
    pub fn new() -> Cancel;
    pub fn cancel(&self);             // signal every holder to stop; idempotent
    pub fn is_cancelled(&self) -> bool;
    pub fn scope<T>(&self, f: impl FnOnce() -> T) -> T;
}
```

`scope` installs this token as the ambient cancellation signal for the current thread
for the duration of `f`, restoring whatever was installed before (including across a
panic in `f`). The scan kernels check the ambient token every few thousand rows and
bail out with an error once it is cancelled, so cancellation is prompt rather than
instant, and never taxes the common uncancelled case with a per-row check.

## `Role` & `ClusterStatus`

What this instance is within a store, and how current it is. `Nidus::cluster_status`
(see [Introspection](#introspection)) returns a snapshot of both; the same facts back
[`GET /cluster`](/reference/http-api/#get-cluster).

```rust
pub enum Role {
    Writer,        // sole writer of a single-node store, holds the plain writer lock
    Reader,        // read-only opener of a single-node store, holds no lock
    ClusterWriter, // cluster writer, holds the renewable, fenced writer lease
    ClusterReader, // cluster reader, lock-free, advances via refresh()
    InMemory,      // in-memory store: no durability, no lock, no peers
}

pub struct ClusterStatus {
    pub role: Role,
    pub cluster: bool,               // whether cluster mode is on (Config::cluster)
    pub holds_writer_handle: bool,   // this instance believes it holds the writer handle
    pub fenced: bool,                // superseded: every subsequent write will fail
    pub lease_owner: Option<String>, // our fencing token while holding a cluster lease
    pub commit_version: u64,         // the manifest commit counter this instance is serving
    pub staleness_secs: u64,         // seconds since this instance last took up newer state
}
```

`fenced` latches once observed, because the condition is permanent: a fenced writer
never regains the lease, it has to reopen. `staleness_secs` is always `0` for a
writer (its own state is current by definition); for a reader it is the age of its
last successful `refresh()`, or of its open if it has never refreshed. Comparing
`commit_version` across instances shows replication lag.

## `LeaseWait`

What a would-be writer does when another instance already holds the writer handle,
under [`OpenMode::ReadWrite`](/reference/configuration/#open_mode). Set via
[`Config::lease_wait`](/reference/configuration/#lease_wait).

```rust
pub enum LeaseWait {
    Fail,               // fail immediately on contention (the default)
    Timeout(Duration),  // retry until acquired, or fail after this long
    Forever,            // retry indefinitely
}
```

`Forever` is what turns an extra `nidus serve` replica into a hot standby: it stays
live (but not ready) and promotes itself the moment the incumbent's lease lapses.

## `Scope`

Which collections a search ranks over. Accepts `impl Into<Scope>`, so `&str` and
`&[&str]` coerce automatically.

```rust
pub enum Scope<'a> {
    Collection(&'a str),       // one collection (the common, fast path)
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

A typed metadata value. `Null` is **distinct from an absent key**; see
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

`DateTime` is an absolute instant in UTC epoch milliseconds: there is no timezone and
no local-time form. It is distinct from `Int` so a filter or a recency ranking can tell
a time from a number without relying on a naming convention.

## `Predicate` & `Filter`

A `Filter` is a conjunction (AND) of predicates; an empty filter matches
everything. Every *leaf* predicate is a positive assertion about a **present**
attribute: a record lacking `key` matches no leaf predicate, including the negative
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

    Fuzzy(String, String, usize),        // within N Levenshtein edits (N ≤ 8)
    ContainsAllTokens(String, String),   // every query token present, any order
    ContainsAnyToken(String, String),    // at least one query token present
    ContainsTokenSequence(String, String), // the tokens consecutive, in order
    Regex(String, String),               // anchored at both ends, like Glob
}

pub struct Filter(pub Vec<Predicate>);
```

The range predicates (`Lt`/`Le`/`Gt`/`Ge`) compare **same-type, orderable** values
only: `Int` numerically, `Str` lexically, `Bool` as `false < true`. A cross-type or
non-orderable (`Null`, `List`) comparison never matches.

`Contains`/`NotContains`/`ContainsAny` look inside a `List`, matching whole elements
rather than substrings: `Contains("tags", "rust")` does not match `["rustacean"]`.

`All`/`Any`/`Not` are predicates over predicates, so arbitrary boolean shapes nest
without `Filter` itself changing. Note `Not` differs from `Ne` on a **missing**
attribute: `Ne(k, v)` is false (it requires `k` present), while `Not(Eq(k, v))` is
true. Use `Ne`/`NotIn`/`NotContains` to require presence, `Not` for set complement.

The [text predicates](/guides/search/#text-predicates) read any text the attribute
carries: a `Str` directly, a `List` element by element, matching when any *single*
element does. `Fuzzy` counts characters, not bytes, over the plain three-operation
Levenshtein distance (so a transposition costs 2) with both sides ASCII-case-folded; a
budget above **8** is an error, not a clamp. The token family tokenizes at query time on a
deliberately simpler rule than the FTS analyzer (maximal alphanumeric runs, ASCII-folded,
**no stemming or stopword removal**), so `ContainsAllTokens("body", "run")` does not match
`"running"` while `text_search` does. `Regex` is anchored at both ends like `Glob` (`.*`
opts back into a substring search), takes case-insensitivity from its own `(?i)` flag, and
runs on a linear-time non-backtracking engine; an unparseable pattern is a caller-facing
error. None of them is indexed; every one re-scans the attribute per row.

## `Distance`

The similarity / distance metric, set at store creation via `Config::distance`.
Pinned in the data header: reopening with a different metric is an error.

```rust
pub enum Distance {
    Cosine,      // default: vectors normalized on insert, score = dot(q, v)
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
`projection: Projection::All`, `rank_by: None`, `limit_per: None`);
`SearchOpts { top_k: 5, ..Default::default() }` is the idiomatic call. Reused by
`text_search`, where `min_score` is a raw BM25 floor.

Results are ordered by `(score desc, collection, id)`. The ranking is computed
`offset + top_k` deep and the page cut once, at the end; an `offset` past the last
result is an empty `Vec`, not an error. See
[paginating a search](/guides/search/#paginating-a-search).

`exact: true` bypasses the ANN walk, the per-segment index, and the quantized first
pass, running the exact brute-force scan for that one query; the index stays in place
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
    pub decay: f32,     // factor at one `scale` of age; default 0.5 (a half-life)
    pub lambda: f32,    // score a fully-decayed hit gives up; default 1.0
    pub missing: f32,   // factor when the attr is absent/unusable; default 1.0
}
```

Build one with `Decay::new(field, origin, scale)` plus `.decay(_)` / `.lambda(_)` /
`.missing(_)`. The score is `base − lambda × (1 − decay^(age / scale))`: the penalty
**subtracts**, which is what keeps it valid for `Euclidean` and `DotProduct` scores and
for raw BM25, not just cosine.

`missing` defaults to `1.0`, so a record with no timestamp is **not** penalized;
enabling decay never buries data that predates the field. `rank_by` does not force the
exact path; over an ANN or quantized result set it reorders within an approximate
candidate set. A ranked scan runs single-threaded.

## `LimitPer`

A cap on how many hits may carry any one value of an attribute; see
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
the first orderable one (a different variant, an unorderable `Null`/`List`, or an absent
attribute) sort into one trailing bucket, which stays trailing when reversed.

## `AggregateOpts` & `Aggregation`

[Count and sum](/guides/search/#aggregation) over a filter, answered from the in-memory
index without materializing a record.

```rust
pub struct AggregateOpts {
    pub filter: Filter,            // default matches every record
    pub sum: Vec<String>,          // attributes to total
    pub group_by: Option<String>,  // one Group per distinct value of this attribute
}

pub struct Aggregation {
    pub count: u64,
    pub sums: BTreeMap<String, Value>,  // Int while every addend was Int, else Float
    pub groups: Vec<Group>,             // empty unless group_by was set
    pub groups_truncated: bool,         // distinct values outran the cap
}

pub struct Group {
    pub value: Option<Value>,           // None = the records missing the attribute
    pub count: u64,
    pub sums: BTreeMap<String, Value>,
}
```

A missing or non-numeric value is skipped, not counted as zero.

`group_by` splits the same single pass into one `Group` per distinct value while still
reporting the whole-scope totals, so "how many per language, and how many overall" is one
query. Groups are ordered by `count` descending with a deterministic tie-break. A `None`
`value` is the group of records **missing** the attribute, distinct from those holding
`Value::Null`, matching how the filter predicates treat absent versus null. Distinct values
are capped at 10 000; past that, new values are dropped and `groups_truncated` is set rather
than letting a short list pass for a complete one.

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
applied where a hit is materialized, so an excluded attr is never cloned; the payload
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

## `FtsField` & `Analyzer`

The declared shape of one full-text-indexed field: BM25 tuning plus its analyzer.
Passed to [`create_collection_with_fts` / `set_fts_schema`](#collections); see
[tuning a field](/guides/search/#tuning-a-field).

```rust
pub struct FtsField {
    pub field: String,     // the attribute to index (a Str, or a List joined with spaces)
    pub k1: f32,           // BM25 term-frequency saturation (default 1.2)
    pub b: f32,            // BM25 length normalization, 0 = none, 1 = full (default 0.75)
    pub analyzer: Analyzer,
}

pub struct Analyzer {
    pub language: Language,           // picks the stopword set + stemmer (English today)
    pub ascii_folding: bool,          // fold Latin diacritics before stemming
    pub max_token_len: Option<usize>, // drop tokens longer than this many chars; None keeps every token
}

// Builders: FtsField::new(field), .k1(_), .b(_), .analyzer(_), .language(_),
//           .ascii_folding(_), .max_token_len(_)
// Analyzer builders: .language(_), .ascii_folding(_), .max_token_len(_)
// `&str` converts to FtsField::new(field) via `From`.
```

An analyzer is applied identically at index and query time, so a query term matches
a stored term only when both were analyzed the same way. `max_token_len` guards
against a base64 blob or a minified bundle inflating the term dictionary.

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

pub struct LegScore    { pub rank: usize, pub score: f32 }  // rank is 0-based
pub struct ClauseScore { pub field: String, pub score: f32 }
pub struct Highlight   { pub field: String, pub fragments: Vec<Fragment> }
pub struct Fragment {
    pub text: String,                 // an excerpt of the stored text
    pub spans: Vec<(usize, usize)>,   // matched byte ranges *within* `text`
}

// Builders: HighlightOpts::default().max_fragments(n).fragment_chars(n)
```

Fragment offsets index the **original** text, not the analyzed tokens: a query for `run`
highlights a document's `running`. Highlighting reads the stored value, so it is unaffected
by [`Projection`](#projection). Note the two units differ: `fragment_chars` budgets
**characters** (an excerpt is never cut mid-codepoint), while `spans` are **byte** offsets
into the fragment.

A `ClauseScore` carries a score but no rank: clauses are folded into one text score by
[`FtsCombine`](#ftsquery-ftsclause-ftscombine--language), so there is no per-clause ranking
for a rank to name. A `LegScore` does carry one, because the fusion legs *are* ranked
independently; only `hybrid_search` produces them.

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
`offset` pages the **fused** ranking, never a leg. There is no `min_score`: a fused RRF
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

Implements `Default` (`order_by: None`); `ListOpts { limit: 20, ..Default::default() }`
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
    pub vector_bytes: u64,  // rows * dimension * 4 (what max_vector_bytes caps)
    pub doc_count: usize,   // live documents across all collections
}
```

## `Quantization` & `QuantKind`

Configuration for vector quantization. Pass to `Config::quantization` to enable two-pass
search (quantized first pass → exact f32 rerank). See
[quantization](/guides/search/#quantization).

```rust
pub enum QuantKind {
    Int8,    // 4× smaller than f32; valid for any distance metric
    Binary,  // 32× smaller, Hamming first pass; COSINE ONLY
}

pub struct Quantization {
    pub kind: QuantKind,
    pub rescore: usize,  // overscan factor: int8 defaults to 4, binary to 16
}

// Builders: Quantization::int8() (also Default), Quantization::binary()
// Setter:   .rescore(n)
```

`Binary` keeps only each dimension's sign bit, which approximates *angular* similarity and
discards magnitude, so it is not a sound ranking proxy for `DotProduct` or `Euclidean`,
and is rejected for those metrics. Being the coarser proxy, it defaults to a larger
overscan than int8.

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

## `OpenProfile`

Recorded open-time defaults for `ann`, `quantization`, `query_threads`, and `mmap`,
carried in the store's manifest (SPEC §14.2). `Nidus::set_open_profile` writes the calling
config's currently-set knobs here; every later `open()` merges a recorded field in
wherever the caller left that knob unset, and an explicit [`Config`] setter for the
same knob always wins. See [Configure once](/guides/cli-and-server/#configure-once-recording-store-defaults).

```rust
pub struct OpenProfile {
    pub ann: Option<AnnConfig>,
    pub quantization: Option<Quantization>,
    pub query_threads: Option<usize>,
    pub mmap: Option<bool>,
}
```

Each `None` means "nothing recorded for this knob," not "explicitly off": there is
no recorded-off state for these four, only recorded-on or absent. A store that has
never been configured has an all-`None` profile and behaves exactly as before this
existed.

## `Memory`, `RememberOpts`, `RecallOpts`, `RememberMode` & `Remembered`

A text-native memory API layered over [`Nidus`](#nidus) and an embedder:
`remember(text)` writes a record, `recall(query_text)` searches by meaning. Gated on
the `memory` feature (`= embed`); see [remember and recall](/guides/remember-and-recall/).

```rust
pub struct Memory { /* db + embedder (+ summarizer) */ }

impl Memory {
    pub fn new(db: Nidus, embedder: AnyEmbedder) -> Self;
    #[cfg(feature = "summarize")]
    pub fn with_summarizer(mut self, summarizer: AnySummarizer) -> Self;
    pub async fn remember(&mut self, collection: &str, id: &str, text: &str, opts: RememberOpts) -> Result<Remembered>;
    pub async fn recall(&self, collection: &str, query_text: &str, opts: &RecallOpts) -> Result<Vec<Hit>>;
    pub fn db(&self) -> &Nidus;       // the raw Vec<f32> API escape hatch
    pub fn db_mut(&mut self) -> &mut Nidus;
    pub fn into_inner(self) -> Nidus; // drops the embedder/summarizer
}

pub enum RememberMode {
    Raw,                    // embed the text as given (the default)
    #[cfg(feature = "summarize")]
    Summarize,               // summarize first, embed the summary, store it under META_SUMMARY
}

pub struct RememberOpts {
    pub mode: RememberMode,
    pub attrs: BTreeMap<String, Value>, // reserved nidus.* recency keys are dropped before stamping
    pub ttl_seconds: Option<i64>,       // seconds until expiry, counted from the write; None never expires
    pub dedupe_threshold: Option<f32>,  // cosine floor above which a write redirects onto the nearest existing entry
}

pub struct Remembered {
    pub id: String,      // the record actually written; not the requested id when deduped
    pub deduped: bool,    // whether dedupe_threshold matched and redirected the write
    pub upserted: usize,  // rows the upsert touched
}

pub struct RecallOpts {
    pub top_k: usize,           // 0 means "use the default" (10)
    pub min_score: f32,         // drop hits scoring below this cosine similarity; 0.0 applies no floor
    pub filter: Option<Filter>, // optional pre-scoring metadata filter
}
```

`RememberOpts`/`RecallOpts`/`Remembered` all implement `Default`/the usual derives, so
`RememberOpts { ttl_seconds: Some(3600), ..Default::default() }` is the idiomatic call.
`RememberMode::Raw` embeds and stores the text as given; `Summarize` needs a summarizer
attached via `with_summarizer` and additionally requires the `summarize` feature.

### Memory metadata keys

Attr and collection-meta keys `remember`/`recall` stamp and read. All gated on
`memory` except `META_EXPIRES_AT`, which lives ungated in `src/meta.rs` so
[`Nidus::sweep_expired`](#nidus) compiles in every build, `memory` feature or not.

| Const | Key | Note |
| ----- | --- | ---- |
| `META_TEXT` | `nidus.text` | the raw remembered text, stamped on every `remember` write regardless of mode |
| `META_CREATED_AT` | `nidus.created_at` | `Value::DateTime`; carried forward unchanged on a dedup update-in-place |
| `META_UPDATED_AT` | `nidus.updated_at` | `Value::DateTime`; set to the write time on every write |
| `META_EMBEDDER` | `nidus.embedder` | collection meta: the `"provider/model"` identity of the embedder that produced its vectors |
| `META_DIM` | `nidus.dim` | collection meta: the embedding dimension, as a decimal string |
| `META_SUMMARY` | `nidus.summary` | `summarize` feature. The generated summary text when `RememberMode::Summarize` is used, i.e. what was actually embedded |
| `META_SOURCE` | `nidus.source` | `summarize` feature. **Legacy and read-only**: no longer stamped by any surface. `META_TEXT` carries the raw source text now; this is kept only so records written before nidus-133 remain readable |
| `META_EXPIRES_AT` | `nidus.expires_at` | ungated. `Value::DateTime` after which an entry is expired; absent means it never expires. Consulted by `Nidus::sweep_expired` |

## `Persistence`, `Appender`, `BackendLock` & `MemoryTier`

The pluggable storage and shared-memory-tier seam (SPEC §13): implement one of these
traits to plug in a backend nidus doesn't ship. See
[writing your own storage backend](/guides/storage-backends/#writing-your-own-backend)
and [writing your own memory store](/guides/memory-stores/#writing-your-own-memory-store).

```rust
pub trait Persistence: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()>;
    fn delete(&self, key: &str) -> Result<()>;
    fn list(&self) -> Result<Vec<String>>;
    fn try_lock(&self, key: &str, ttl: Duration) -> Result<Option<Box<dyn BackendLock>>>;

    // Optional; every one defaults to "not supported" (see the notes below).
    fn appender(&self, key: &str) -> Result<Option<Box<dyn Appender>>>;
    fn try_create_exclusive(&self, key: &str, bytes: &[u8]) -> Result<Option<bool>>;
    fn get_cas(&self, key: &str) -> Result<Option<(Vec<u8>, Option<String>)>>;
    fn put_cas(&self, key: &str, bytes: &[u8], expected: Option<&str>) -> Result<CasOutcome>;
    fn local_path(&self, key: &str) -> Option<PathBuf>;
    fn has_native_lock(&self) -> bool;
    fn supports_cas(&self) -> bool;
}
```

Whole named byte objects in two classes: source-of-truth (`data`/`log`, never
reconstructable) and derived caches (`ann`/`fts`, droppable, rebuilt on a stale or
torn load). Only the first five methods are required; the rest default to "not
supported" (`appender` → `None`, `put_cas` → `CasOutcome::Unsupported`,
`local_path` → `None`, `has_native_lock` → `true`, `supports_cas` → `false`), so a
minimal backend still works everywhere except cluster mode, which needs a real
`get_cas`/`put_cas`.

```rust
pub trait Appender: Send + Sync {
    fn len(&self) -> Result<u64>;
    fn is_empty(&self) -> Result<bool>;   // default: len()? == 0
    fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()>;
    fn append(&mut self, bytes: &[u8]) -> Result<()>;
    fn truncate_to(&mut self, offset: u64) -> Result<()>;
    fn sync(&mut self) -> Result<()>;
    fn rewrite(&mut self, bytes: &[u8]) -> Result<()>;
    fn read_to_end(&mut self, out: &mut Vec<u8>) -> Result<()>; // provided over read_exact_at

    // Required: len, read_exact_at, append, truncate_to, sync, rewrite.
}

pub trait BackendLock: Send + Sync {}

pub trait MemoryTier: Send + Sync {
    fn load(&self, key: &str) -> Result<Option<Vec<u8>>>;
    fn store(&self, key: &str, bytes: &[u8], ttl: Option<Duration>) -> Result<()>;
}
```

`Appender` is a durable, append-shaped byte stream: the native local-filesystem
capability `data` and `log` need (append is atomic, rolling back to the length
before the call on a partial write; `rewrite` is the atomic whole-file replace
`compact` uses). Object-store backends do not implement it; nidus wraps them in an
in-RAM `ObjectAppender` that rewrites the whole object on `sync` instead.

`BackendLock` is a held backend lock, released on `Drop`; the concrete guard owns
whatever the backend needs to release (a lock file, a conditional-PUT marker).

`MemoryTier` is where the in-RAM working set is held so it can be shared across
processes and reloaded without a rebuild (SPEC §13.3). Deliberately rebuildable: an
empty or evicted tier is never fatal, since the persistence tier is the source of
truth. `Arc<dyn MemoryTier>` (or `Arc<LocalRam>`) itself implements `MemoryTier`, so
several stores can publish to and adopt from one shared instance.

## `LocalFs`, `LocalRam`, `CasOutcome`, `open_persistence` & `open_memory_tier`

The bundled backend implementations and the two location-string dispatchers used to
pick one at runtime.

```rust
pub struct LocalFs { /* rooted at a directory */ }
impl LocalFs {
    pub fn new(dir: impl Into<PathBuf>) -> Result<LocalFs>; // creates dir (+ parents) if absent
    pub fn dir(&self) -> &Path;
}

pub struct LocalRam { /* a Mutex<HashMap<String, Vec<u8>>> */ }
impl LocalRam {
    pub fn new() -> LocalRam; // also Default; a fresh, empty tier
}

pub enum CasOutcome {
    Written(Option<String>), // committed; the new CAS token, when the backend reports one cheaply
    Stale,                   // precondition failed: lost the race, not an error
    Unsupported,              // this backend offers no compare-and-swap
}

pub fn open_persistence(location: &str) -> Result<Box<dyn Persistence>>;
pub fn open_memory_tier(location: &str) -> Result<Box<dyn MemoryTier>>;
```

`LocalFs` is the default `Persistence` backend, and the one every other backend is
checked against: each object is a file `<dir>/<key>`, whole-object writes are
atomic, and `try_lock` reuses the same `O_EXCL` lock file as the plain single-node
path. `LocalRam` is the trivial `MemoryTier`: shared between threads of one
process, never across processes.

`open_persistence` dispatches a location string to a backend: `s3://…` and
`gs://…`/`gcs://…` to the S3/GCS backends, anything else (`file://<path>` or a bare
path) to `LocalFs`. `open_memory_tier` dispatches `""`/`"local"`/`"ram"` to
`LocalRam`, and a Redis-family URL (`redis://`, `rediss://`, `valkey://`,
`valkeys://`, `keydb://`, `dragonfly://`) to a Redis-backed tier; anything else is
an error.

Cluster mode's writer lease (`ClusterLease`, and its non-owning `LeaseRenewer`) are
concrete types over a `Persistence` backend, not traits: `ClusterLease::acquire`
mints a fenced, heartbeated lease, `renew()` re-stamps it before each write batch
and fails once another writer has taken over, and `is_lease_lost` classifies that
failure (a `LeaseLost` error) out of the `anyhow` chain so a caller can tell it apart
from a transient backend failure. `ClusterLease::renewer` hands out a `LeaseRenewer`
that can keep the lease warm from a background task without being able to release it.
