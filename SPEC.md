# nidus — specification

> _nidus_ (Latin, "nest") — a small place where things are kept safe. A pure-Rust
> embeddable vector store, leaning on the bird theme.

This document is the source of truth for nidus's design. It records not just
*what* we build but *why*, including the decisions we deliberately deferred.
`CLAUDE.md` is the agent-facing summary; this is the long form. Keep them in sync.

---

## 1. Purpose & motivation

nidus is an **all-in-one memory** for semantic-search, RAG, and indexing tools:
remember text, recall the relevant bits. Classically that is a pipeline — chunk
some source content → embed each chunk into a dense vector → store the vectors plus
metadata → answer "nearest neighbours to this query vector" — and nidus can own the
whole thing (built-in embedding, optionally summarizing first, with the provider of
your choice) or just the storage-and-search core if you bring your own vectors. It
runs fast, in-process, with no hosted service. The source can be anything — code,
documents, issues, wiki pages — nidus does not care; it turns text into vectors,
stores vectors and metadata, and ranks them.

It exists because the obvious off-the-shelf options fail the *embedding* test —
not the functionality test, the **build-and-ship** test:

- **DuckDB** (a common embedded choice, via `libduckdb-sys`) **bundles a large C++
  source tree and compiles it from scratch via `cc`**. Costs: multi-minute cold
  builds, a required C++ toolchain, awkward cross-compilation, a bloated binary,
  and FFI that **cannot run under Miri**. A typical vector workload uses ~1% of
  DuckDB: one table, a brute-force cosine top-k, and equality/GLOB filters.
- **LanceDB** is "written in Rust" yet still compiles for ~10 minutes, because it
  drags in **arrow-rs + DataFusion (a full SQL engine) + the Lance columnar format
  + object_store**. Hundreds of crates and a query engine, to do a distance-ranked
  top-k. Same disease as DuckDB, transitively-Rust instead of FFI.

At its core the workload is a **vector store, not a database**: no joins, no SQL, no
analytics, no larger-than-RAM scans (at the target scale). nidus is that store —
plus an opt-in memory layer (embedding, optionally summarization) over it that stays
off the default build — and nothing more.

### Thesis (the product *is* the constraints)

**Core foundation — speed, testing, stable.** Everything below elaborates these three
commitments, and every change is judged against them; trading one away is a change to
*what nidus is*, not an implementation detail:

- **Speed** — builds in seconds, installs with a bare `cargo add`, answers queries at
  brute-force-fast latencies. The bar is empirical and CI-asserted, never aspirational.
- **Testing** — behaviour is *verified against the real artifact, never assumed* (§11).
  Every surface has a load-bearing test in CI — end-to-end where only end-to-end can
  prove it. A feature without its test is not shipped.
- **Stable** — durable by construction: crash-safe log, CRC'd codecs, torn-tail
  recovery, graceful ENOSPC/OOM (§6), additive on-disk formats (§9) with one deliberate
  exception, the versioned manifest (§14.2). Boring to depend on is the goal.

The hard bar on dependencies is **build-and-ship speed, not zero-C absolutism.** What
disqualified DuckDB and LanceDB was a *multi-minute* build (a large C/C++ tree, a whole
SQL engine) — not the mere presence of any C. nidus's bar is concrete and testable: **a
clean build stays under a minute** (it is ~seconds today).

1. **Pure-Rust-first, fast to build.** Prefer well-established pure-Rust crates
   (`anyhow`, `serde`/`bincode`, `crc32fast`, `regex`, …). A C-compiling/native-linking dep is
   acceptable **only** when it stays small and fast (e.g. `ring`'s TLS for the storage
   backends, §13) — **never** a crate that compiles a *large* C tree (DuckDB's C++,
   `aws-lc-sys`, vendored OpenSSL). `just deps` stays short; CI asserts the build-time
   ceiling.
2. **Near-zero `unsafe` in *our* code.** `#![deny(unsafe_code)]` with exactly **one** scoped
   `#[allow]`: the single `Mmap::map` call behind `Config::mmap` (the memory-mapped-segment
   seam, §9 / §14.6 phase 3 — a deliberate, opt-in FFI choice). No `flock`, no `extern "C"`
   written by us, and no other `unsafe` anywhere — every other use is still a hard compile
   error. (A dependency's internal `unsafe`/C is fine; ours, beyond that one site, is not.)
3. **Fast builds over zero-C.** `cargo build` stays in **seconds**. The pure-Rust core
   needs no C toolchain; the always-compiled storage backends (§13) add `ring` (small
   C/asm) to the default tree, so a C toolchain *is* required — but the build stays in
   seconds, which is the property that actually matters.
4. **Miri covers all *our* logic.** Our code — codecs, filters, distance math, file IO
   — runs under Miri. A dependency's native/FFI paths (a backend's TLS) and the `mmap`
   syscall cannot, so the tests that exercise them are `#[cfg_attr(miri, ignore)]` like the
   fsync tests (§11). This is narrower than "the whole crate runs under Miri," and a
   deliberate trade for frictionless pluggable backends (§13.6) and the mmap seam (§9).

Compiling a *large* C tree, or adding a *second* `unsafe` site to *our* code, is a change to
*what nidus is*. File an issue and decide deliberately.

---

## 2. Goals & non-goals

**Goals**
- Embeddable, in-process, single-store-per-directory.
- Exact (100% recall) brute-force cosine search, fast at the target scale
  (≤ a few million vectors, comfortably in RAM).
- Many logical collections (namespaces) in one store, sharing one dimension.
- **Scoped search**: query one collection, a chosen subset, or the entire store in
  a single call, with results merged into one ranking. The API must not lock callers
  into a single namespace per query, and the storage layout must not make
  whole-store search expensive beyond the unavoidable scan cost.
- Crash-safe writes; lock-free, consistent cross-process reads.
- Idempotent upserts by caller-supplied id.

**Non-goals (for v0.1)**
- Approximate nearest neighbour (HNSW/IVF) was a deferred seam; it has since shipped
  as the opt-in `Config::ann` mode (§9). Exact brute-force remains the default —
  whole-store search makes the scanned `N` potentially large, which is exactly what
  motivated the seam.
- Larger-than-RAM / memory-mapped operation was a deferred seam; it has since shipped as
  the opt-in `Config::mmap` mode — immutable segments served from a read-only memory-map
  while the active segment stays in RAM (§9 / §14.6 phase 3).
- Quantization — int8 scalar and binary (sign-bit) quantization have since shipped
  (§9, opt-in via `Config::quantization`).
- SQL, a query planner, transactions spanning multiple operations, multi-writer
  concurrency, or replication.
- A query *protocol* over the network was a non-goal; the opt-in `nidus serve` (§9)
  has since shipped as a separate `cli`-feature wrapper, not a core change. Pluggable
  *persistence* backends (S3/GCS) and a shared *memory tier* (Redis/Valkey/Memcached)
  are now a designed seam (§13) — that is where the *bytes* live and where the *warm
  working set* is shared, still distinct from a query protocol.

---

## 3. Data model

```rust
pub struct Record {
    pub id: String,             // caller-supplied; upsert key (idempotent)
    pub vector: Vec<f32>,       // length must equal the store dimension
    pub attrs: BTreeMap<String, Value>,
}

pub enum Value {
    Null,                       // distinct from "absent" — see below
    Str(String),
    Int(i64),
    Bool(bool),
    List(Vec<String>),
    Float(f64),                 // IEEE: NaN matches nothing, 0.0 == -0.0
    DateTime(i64),              // UTC epoch milliseconds; no timezone, no local time
}
```

- **Collections** are logical partitions (namespaces) identified by a `&str`. There
  are many; each is created/dropped independently; all share the store's single
  pinned dimension.
- **Dimension** is fixed for the life of the store, recorded in the `data` header.
  Reopening with a different dimension is a hard error. One store = one embedding
  model = one comparable vector space. A **query** vector of the wrong length is a
  hard error too, symmetrically with `upsert` — never a silent empty result. The
  usual cause is swapping embedding models without re-indexing, and reporting that as
  "no matches" would be indistinguishable from an empty store (nidus-c5v).
- `Value` is rich enough to hold any scalar/metadata a caller attaches. The
  `Null`-vs-absent distinction is meaningful and preserved across disk round-trips:
  it lets a caller tell apart "this field was not computed/indexed" (absent) from
  "computed, and the value is empty" (e.g. `List([])`) — a distinction that matters
  for things like optional graph edges or tags.
- **`Value` is append-only.** The op-log encodes a value by its *variant index*
  (bincode), so a new variant may only be added at the end; inserting one above an
  existing variant would silently reinterpret every value in every existing store.
  `Float` and `DateTime` were appended for exactly this reason, and a test asserts the
  pre-existing variants still occupy indices 0..=4.
- **`Float` is IEEE, and deliberately not interchangeable with `Int`.** Comparison is
  same-type only (§7), so `Ge("score", Float(0.5))` does not match a record storing
  `Int(1)` — the same rule that already separates `Str` from `Int`. `NaN` is not
  ordered and is not equal to itself, so it fails every predicate including
  `Eq(k, NaN)`, rather than being forced into a total order to make it sortable.
- **`DateTime` is UTC epoch milliseconds** — an absolute instant. There is no timezone
  field and no local-time form: a timezone is a rendering concern, and storing one
  would make the same instant compare unequal to itself. It is a distinct type from
  `Int` so that a range filter and a recency ranking can tell "a number" from "a time"
  without a naming convention the store cannot enforce.

---

## 4. Public API (sketch)

Synchronous (see §6.5 for why — the hot path is CPU-bound, so async would only add
overhead and a runtime dependency). Mutations take `&mut self`; reads take `&self`,
so `Arc<RwLock<Nidus>>` yields many concurrent searchers + one writer. Async callers
bridge with `spawn_blocking`.

```rust
impl Nidus {
    pub fn open(config: Config) -> Result<Self>;                            // canonical
    pub fn open_dir(dir: impl AsRef<Path>, dimension: usize) -> Result<Self>; // = open(Config::new(dir, dim))
    pub fn open_in_memory(dimension: usize) -> Result<Self>;                // tests; no files, no lock
    pub fn dimension(&self) -> usize;
    pub fn config(&self) -> &Config;
    pub fn footprint(&self) -> Footprint;   // cheap vector-footprint snapshot (§6.6)

    // collections
    pub fn create_collection(&mut self, name: &str) -> Result<()>;     // idempotent
    pub fn drop_collection(&mut self, name: &str) -> Result<()>;
    pub fn has_collection(&self, name: &str) -> bool;
    pub fn collections(&self) -> Vec<String>;

    // per-collection metadata (small string map; e.g. a high-water mark, model id)
    pub fn get_meta(&self, collection: &str) -> BTreeMap<String, String>;
    pub fn set_meta(&mut self, collection: &str, meta: BTreeMap<String, String>) -> Result<()>;

    // documents — upsert is idempotent by id; fsynced per call (a batch)
    pub fn upsert(&mut self, collection: &str, records: &[Record]) -> Result<usize>;
    pub fn delete(&mut self, collection: &str, ids: &[&str]) -> Result<usize>;
    pub fn delete_where(&mut self, collection: &str, filter: &Filter) -> Result<usize>;
    pub fn get_all(&self, collection: &str) -> Vec<Record>;            // includes vectors

    // search one collection, a subset, or the whole store — one merged ranking.
    // `scope` accepts `impl Into<Scope>`, so a bare &str / &[&str] also works.
    pub fn search(&self, scope: Scope, query: &[f32], opts: &SearchOpts) -> Result<Vec<Hit>>;

    // count + sum over a filter, straight off the in-RAM index — no Record built (§7.7)
    pub fn aggregate(&self, scope: Scope, opts: &AggregateOpts) -> Result<Aggregation>;

    pub fn flush(&mut self) -> Result<()>;     // fsync both files
    pub fn compact(&mut self) -> Result<()>;   // reclaim dead rows / log churn
    pub fn refresh(&mut self) -> Result<bool>; // ReadOnly: adopt a writer's newer state (§14.6)
}

/// Which collections a search ranks over. Scores are comparable across
/// collections because the whole store shares one embedding space (§3).
pub enum Scope<'a> {
    Collection(&'a str),       // the common, fast path
    Collections(&'a [&'a str]),
    All,                       // every collection in the store
}
// impl From<&str> / From<&[&str]> for Scope — ergonomic single- and multi-collection calls.

// `offset` skips that many top-ranked hits (§7 pagination); 0 is the whole first page.
// `exact` forces the brute-force scan for one query; `projection` picks the attrs (§7).
// `rank_by`/`limit_per` are the opt-in ranking expression and per-value hit cap (§7.6, §7.7).
pub struct SearchOpts { pub top_k: usize, pub offset: usize, pub filter: Filter, pub min_score: Option<f32>, pub exact: bool, pub projection: Projection, pub rank_by: Option<RankBy>, pub limit_per: Option<LimitPer> }

// Which attrs a Hit carries. An enum, so "include and exclude at once" cannot be built.
pub enum Projection { All, Include(Vec<String>), Exclude(Vec<String>) }

// A ranking expression layered over the metric. The penalty SUBTRACTS (§7.6).
pub enum RankBy { Decay(Decay) }
pub struct Decay { pub field: String, pub origin: i64, pub scale: i64,
                   pub decay: f32, pub lambda: f32, pub missing: f32 }   // missing defaults to 1.0

// "At most `max` hits per distinct value of `field`" — approximate; see §7.7.
pub struct LimitPer { pub field: String, pub max: usize }

// ORDER BY for `list`. Cross-type and missing values sort into a trailing bucket (§7.6).
pub struct OrderBy { pub field: String, pub descending: bool }

// count is always reported; `sum` names attributes to total, each a tagged Value (§7.7).
// `group_by` adds one Group per distinct value of that attribute, beside the totals.
pub struct AggregateOpts { pub filter: Filter, pub sum: Vec<String>, pub group_by: Option<String> }
pub struct Aggregation { pub count: u64, pub sums: BTreeMap<String, Value>,
                         pub groups: Vec<Group>, pub groups_truncated: bool }
// `value` is None for the records missing the attribute — not the same as Value::Null (§7.7).
pub struct Group { pub value: Option<Value>, pub count: u64, pub sums: BTreeMap<String, Value> }

// `collection` identifies the source namespace — required when a query spans more
// than one, and (id) is only unique within a collection.
pub struct Hit { pub collection: String, pub id: String, pub score: f32, pub attrs: BTreeMap<String, Value> } // no vector

pub struct Filter(pub Vec<Predicate>);   // AND of predicates
pub enum Predicate {
    Eq(String, Value),                   // attr == value
    Ne(String, Value),                   // attr present and != value
    Glob(String, String),                // attr (Str) matches glob pattern
    IGlob(String, String),               // same, ignoring ASCII case
    In(String, Vec<Value>),              // attr ∈ set
    NotIn(String, Vec<Value>),           // attr present and ∉ set
    Lt(String, Value),                   // attr <  value  (same-type, orderable)
    Le(String, Value),                   // attr <= value
    Gt(String, Value),                   // attr >  value
    Ge(String, Value),                   // attr >= value
    Contains(String, Value),             // attr (List) holds value
    NotContains(String, Value),          // attr present, a List, and does not hold value
    ContainsAny(String, Vec<Value>),     // attr (List) overlaps the set
    All(Vec<Predicate>),                 // every sub-predicate holds (empty = true)
    Any(Vec<Predicate>),                 // some sub-predicate holds  (empty = false)
    Not(Box<Predicate>),                 // sub-predicate does not hold
}

// A cheap, allocation-free footprint snapshot (§6.6). `vector_bytes` is the
// dominant, predictable cost — what `Config::max_vector_bytes` caps.
pub struct Footprint {
    pub rows: u64, pub dead_rows: u64, pub dimension: usize,
    pub vector_bytes: u64, pub doc_count: usize,
}
```

`Hit` deliberately omits the vector (search returns many; nobody needs the floats
back). `get_all` includes vectors (callers re-upserting with new metadata need them).

### 4.1 Configuration — the store location is the caller's choice

A store has **no hardcoded location**. `open` takes a `Config`; the directory is
always supplied by the caller — an application's own config, a user-facing flag, or
(for an embedding tool) whatever path that tool manages. nidus never picks a path,
an env var, or a default directory for you. Only the file *names inside* the
directory (`data`/`log`/`lock`) are fixed; they are an internal detail.

```rust
#[derive(Clone, Debug)]
pub struct Config {
    pub path: PathBuf,             // REQUIRED — the store directory
    pub dimension: usize,          // REQUIRED — pinned embedding dimension
    pub fsync: Fsync,              // default PerBatch (decision B)
    pub open_mode: OpenMode,       // default ReadWrite
    pub auto_compact: Option<f32>, // dead-row ratio that triggers compaction on open;
                                   //   None = never. default Some(0.5)
    pub lock_ttl: Duration,        // stale writer-lock reclamation window. default 60s
    pub max_vector_bytes: Option<u64>, // hard ceiling on the vector matrix
                                   //   (rows*dim*4); None = unbounded (default). §6.6
    pub quantization: Option<Quantization>, // two-pass quantized search; None disables (default)
    pub ann: Option<AnnConfig>,    // approximate index (HNSW/IVF); None = exact brute force (default)
    pub query_threads: usize,      // worker threads for one search. default 1 (serial)
    pub mmap: bool,                // memory-map sealed segments instead of RAM. default false
}
pub enum Fsync { PerBatch, OnFlush }
pub enum OpenMode { ReadWrite, ReadOnly }   // ReadOnly takes no writer lock; rejects writes

impl Config {
    pub fn new(path: impl Into<PathBuf>, dimension: usize) -> Self;  // all else defaulted
    pub fn fsync(self, f: Fsync) -> Self;                            // builder setters
    pub fn open_mode(self, m: OpenMode) -> Self;
    pub fn auto_compact(self, ratio: Option<f32>) -> Self;
    pub fn lock_ttl(self, ttl: Duration) -> Self;
    pub fn max_vector_bytes(self, bytes: Option<u64>) -> Self;
    pub fn quantization(self, q: Option<Quantization>) -> Self;
    pub fn ann(self, ann: Option<AnnConfig>) -> Self;
    pub fn query_threads(self, n: usize) -> Self;
    pub fn mmap(self, on: bool) -> Self;
}
```

These four knobs (`quantization`, `ann`, `query_threads`, `mmap`) are also the ones
`OpenProfile` can record as store-level defaults (§14.2, nidus-141): an explicit
setter here always wins over a recorded default for the same knob, at every open.

- `OpenMode::ReadOnly` opens **without** taking the writer lock and rejects
  mutations — the basis for many concurrent search-only processes over a store
  another process writes (the lock-free snapshot model, §6.2), and the foundation
  for the search server (§9, shipped as `nidus serve` behind the `cli` feature).
- Defaults are chosen so `Config::new(path, dim)` "just works" for the embedded
  single-writer case; every field is overridable for callers (or a server) that
  need different durability/lock/compaction behavior.

---

## 5. On-disk format

A store is, by default, a **local directory** — the first of the pluggable storage
backends designed in §13. Its objects:

```
<dir>/
  data    flat f32 matrix, append-only, never rewritten in place   (source of truth)
  log     append-only op stream (the commit record)                (source of truth)
  ann     persisted ANN index — derived cache, reconstructable, best-effort (§9)
  fts     persisted BM25 index — derived cache, shares the ann cache codec (§9)
  lock    writer-exclusion lock file (present only while a writer holds it)
```

`data` and `log` are the **source of truth**; `ann`/`fts` are **derived caches**
(an absent/stale/corrupt cache is rebuilt, never fatal — see `index_cache.rs`, §9).
That two-tier split is what lets object-store persistence backends (S3/GCS) durably hold a
store without shipping the large, append-hostile index — see §13.

### 5.1 `data` — the vector segment

```
┌─ header (64 bytes, fixed) ─────────────────────────────────────────────┐
│ magic   "NIDUS\0" + format version (u16)                               │
│ dimension (u32)                                                        │
│ reserved zero padding to 64 bytes (cache-line alignment for rows)      │
├─ rows ─────────────────────────────────────────────────────────────────┤
│ row 0:  dim × f32 (little-endian)   at byte offset 64                  │
│ row 1:  dim × f32                    at offset 64 + 1·dim·4             │
│ ...                                  row i at 64 + i·dim·4              │
└────────────────────────────────────────────────────────────────────────┘
```

- Fixed stride (`dim·4` bytes) → row `i` is pure arithmetic; rows are 64-byte
  aligned → friendly to autovectorized dot products and (later) a sound
  reinterpret for mmap.
- **Append-only.** New vectors append at the tail. Existing rows are never mutated,
  so a concurrent reader always sees fully-written rows. Deletes/overwrites do not
  remove rows here — the row is simply no longer referenced (reclaimed by §8).
- Vectors are **unit-normalized before writing** (§7), so `data` stores unit vectors.

### 5.2 `log` — the operation stream

A sequence of records; each record is:

```
[ len: u32 ][ payload: len bytes ][ crc32: u32 ]   (all little-endian)
```

`crc32` covers the payload (the `crc32fast` dep). Payload is a
tagged op:

| tag | op | payload |
|----|----|---------|
| 0 | `CreateCollection` | name |
| 1 | `DropCollection`   | name |
| 2 | `SetMeta`          | collection, `{k:v}` string map |
| 3 | `Upsert`           | collection, id, **row_index (u64)**, attrs |
| 4 | `Delete`           | collection, id |

- The **`Upsert` log record is the commit point**: it references a `row_index`
  into `data`. A vector exists in the store iff a committed `Upsert` points at its
  row. Orphan rows (vector written, process died before the log record) are inert
  and reclaimed on compaction.
- **Replay** (`open`): read records sequentially, applying each to the in-RAM
  index. If the final record is short (truncated `len`/`payload`) or fails CRC, the
  log was torn by a crash mid-append → **truncate to the last good record** and
  continue. This is the crash-recovery mechanism.
- Strings/maps/attrs use the explicit little-endian codec in `model.rs`
  (`u32` length prefixes; `Value` tag byte + payload). No serde.

### 5.3 In-RAM state (rebuilt on open)

```
dimension: usize
vectors:   Vec<f32>                              // the data rows, row-major
collections: HashMap<String, Collection>
struct Collection {
    meta: BTreeMap<String, String>,
    docs: HashMap<String, DocEntry>,             // id → entry
}
struct DocEntry { row: u64, attrs: BTreeMap<String, Value> }
```

`open` cost = `read(data)` (one bulk read of a flat blob; **no parsing**) + replay
`log` (small). The big vector file is never deserialized field-by-field.

---

## 6. Durability & concurrency

Three guarantees, in priority order.

### 6.1 Crash safety — guaranteed
Append-only files + commit-via-log + CRC'd, length-prefixed records. A writer
killed at any point leaves: a valid `data` prefix (possibly with inert orphan rows
at the tail) and a `log` whose last record is either complete or detectably torn.
Reopening recovers to "last fully-committed op." The worst loss is the in-flight
batch — acceptable because the index is reproducible from source.

Crucially this also holds for an *in-process* write failure (e.g. ENOSPC), not just
a kill: each `append` is atomic per row/frame (a partial write is rolled back to the
boundary), and `upsert` is **all-or-nothing** — every fallible step rolls `data` and
`log` back to the marks taken at entry, so a failed batch leaves the store
byte-identical to its pre-call state. A caught error never leaves a torn row/frame
for the next write to build on. See §6.6.

### 6.2 Cross-process reader/writer isolation — lock-free
**Write order is load-bearing:** append vectors to `data` → **fsync `data`** →
append committing `log` records → **fsync `log`**. Therefore any committed `Upsert`
record's referenced row is already durable in `data`.

A reader process opens by reading `data` (size S → `S/dim` rows) then replaying
`log`, and **ignores any record referencing a row ≥ S/dim**. Result: a consistent,
possibly-slightly-stale snapshot of whatever was committed when it read — never a
torn vector, never a half-record. No read lock required. The snapshot is advanced in
place by `Nidus::refresh()` (§14.6 phase 4), which re-applies this same rule at a newer
manifest version without reopening — the basis for a search-only process tracking a store
another process is writing.

### 6.3 Writer/writer exclusion — best-effort, pure std
Two concurrent writers would corrupt the append stream. A writer acquires
`<dir>/lock` via `OpenOptions::new().write(true).create_new(true)` (atomic O_EXCL
create — pure `std`, no `flock`/FFI), writing its PID + start timestamp inside. On
conflict: error with a clear message; a lock older than a TTL is treated as stale
and reclaimed (git's `index.lock` pattern). The lock is removed on clean close.

> Decision (C): pure-std lock file, not `flock`. `flock` would auto-release on
> process death (no stale wart) but is FFI. Per the zero-FFI thesis we take the
> lock file and the mild stale-lock TTL instead. Indexing is typically serial, so
> contention is rare.

### 6.4 fsync policy
Decision (B): **per-batch fsync** — every `upsert`/`delete` call fsyncs. Batches are
large (e.g. hundreds of docs) and infrequent during indexing, so the cost is
negligible and durability is real. `flush()` exists for callers that want an
explicit barrier.

That assumption — batches large and infrequent — holds for an indexer and breaks for a
*served* store, where the barrier is a per-call cost measured at ~7.6ms regardless of
payload, and several writes are genuinely in flight at once. **Group commit** (nidus-xb9.1)
is the answer, and it is a change to *who calls the barrier*, not to the policy:

- `Nidus::deferred(f)` runs `f`'s mutations with their barrier deferred — appended, indexed,
  not synced.
- `Nidus::commit()` then takes **one** barrier for all of them, in the §6.2 order (`data`,
  then `log`), plus one commit-counter publish in cluster mode instead of one per batch.

The obligation this shifts onto the caller is explicit and is the whole of the correctness
argument: **nothing may be reported successful until `commit()` returns `Ok`.** Until then
the bytes are appended but not durable, which is exactly the tail state §6.2's reader rule
already drops on replay — so a crash in that window loses the un-acknowledged batch and
nothing else. A failed barrier fails every member of its group.

`nidus serve` is the host that does this: concurrent writes queue, the first to reach the
store applies the whole queue under one exclusive guard, one barrier covers them, and only
then does each request get its `200`. There is no timed window — a lone write forms a group
of one and pays exactly what it always did — so the uncontended path is untouched while the
contended one stops paying N barriers for N writes. `nidus_write_groups_total` /
`nidus_write_group_members_total` report the resulting coalescing factor.

Leadership is a `bool` under the queue's mutex rather than a dedicated commit thread: less
machinery for the same behaviour, with no lifecycle to own and no thread idling in a
read-only instance. The election happens *together with the enqueue*, so a write either
finds a leader guaranteed to come back for it or becomes the leader — there is no third
outcome, which is what rules out a lost wakeup. The queue is unbounded because it is
already bounded: the §6.5 concurrency semaphore caps store-touching requests in flight, and
nothing reaches the queue without holding a permit. Under write load the blocking pool then
holds roughly one task per *group* rather than one per request.

### 6.5 Concurrency & speed
The hot path (`search`) is pure CPU over in-RAM data — there is no IO to await — so
the core API is **synchronous on purpose**. An `async` core would add executor
overhead, risk blocking the runtime with a CPU loop, and force a runtime dependency
on every user (breaking zero-deps and runtime-agnosticism). Speed and concurrency
come from elsewhere, in order:

- **`&self` reads ⇒ concurrent searchers.** `Arc<RwLock<Nidus>>` gives many parallel
  searches with one exclusive writer (`Arc<Mutex<…>>` if simplicity is preferred).
- **Parallel scan (shipped, §9).** `search(&self)` fans the row scan across cores
  with `std::thread::scope` (**zero-dep std, no rayon**) into per-thread top-k heaps,
  then merges — no API or format change. Opt-in via `Config::query_threads` (default
  `1`, serial); a dim-aware work floor keeps small scans serial so thread spawn/join
  never dominates. The flat, aligned `f32` matrix is laid out for this and for
  autovectorized dot products. Leave it at `1` when query-level concurrency already
  saturates the cores (many readers under `Arc<RwLock<Nidus>>`).
- **Async callers** bridge with `spawn_blocking` (their runtime, their choice). The
  core never exposes `async fn`.

### 6.6 Resource exhaustion — graceful failure

nidus holds the whole vector matrix in RAM and on disk, so "out of room" has two
forms; neither may corrupt the store or silently abort the process.

**Disk full (ENOSPC).** Appends are atomic and batches are all-or-nothing (§6.1).
`DataSegment::append` / `OpLog::append` capture the file offset, and on a partial
`write_all` roll the file back to the row/frame boundary — without this the next
append would write past the partial bytes, misaligning the matrix or producing a
mid-file torn frame that `log` replay rejects as hard corruption. `upsert` captures
`(data_rows, log_offset)` at entry and, on any failure through data-append →
data-fsync → log-append → log-fsync, truncates both files back to those marks before
returning the original error. The in-RAM index is mutated only in a final,
infallible commit phase (its map capacity is reserved up-front), after both files
are durable.

**Out of RAM.** Growth of the vector matrix and the index maps uses `try_reserve`,
so an allocator-null OOM becomes an `Err`, not a `handle_alloc_error` abort. `open`
streams the data file into a single pre-reserved `Vec<f32>` (no raw-bytes +
decoded-floats double allocation), so reopening peaks at ≈ steady state and fails
cleanly if it won't fit. **Limit:** `attrs` (`BTreeMap`) and id (`String`) clones
have no `try_reserve` in std and can still abort — these are small metadata next to
the `N·dim·4` matrix, which *is* covered. `get_all` returns a `Vec` (not `Result`)
and so is likewise not fallible; it is a bulk-read convenience, not a write/open
path.

**The real risk is constrained/containerized deployments**, not roomy laptops (1M ×
768-dim ≈ 3 GB fits fine). Under a cgroup limit with memory overcommit, the kernel
SIGKILLs before an allocation ever fails, so `try_reserve` never fires. The only
reliable guard there is to refuse work *before* allocating:

- **`Config::max_vector_bytes: Option<u64>`** (default `None` — no behavior change)
  caps `rows · dim · 4`. `upsert` projects the post-batch size and refuses
  (cleanly, no rollback) anything over the cap; `open` refuses a data file already
  over it before allocating. The cap counts physical rows incl. not-yet-compacted
  dead rows, so `compact` reclaims headroom.
- **`footprint() -> Footprint`** is the cheap introspection hook (rows, dead rows,
  `vector_bytes`, live `doc_count`) a host reads to decide whether more data fits.

---

## 7. Search semantics

- **Cosine via unit vectors.** Vectors are normalized to unit length on insert; the
  query is normalized once per search. Then `score = dot(stored, query)` ∈ [−1, 1],
  identical to `1 − cosine_distance`. No per-vector norms stored, no per-query norm
  loop. Zero vectors store as-is and score 0.
  > Observable caveat: `get_all` returns unit-scaled vectors, not the caller's
  > originals. A re-upsert flow that round-trips vectors is idempotent under this
  > (re-normalizing a unit vector is a no-op). Documented, intentional.
- **Scoped scan.** A search ranks over a `Scope` — one collection, a chosen subset,
  or `All`. The scan walks the `docs` of each in-scope collection, slicing into the
  shared global `vectors` matrix (rows are global, so no per-collection vector
  storage and nothing to gather). Single-collection is the fast path (cost scales
  with that collection); whole-store search costs the union scan — the unavoidable
  price of exact search, and the reason the ANN seam (§9) exists for later. Merging
  across collections is **sound because every collection shares one embedding
  space** (§3): a `score` means the same thing everywhere. Each `Hit` carries its
  source `collection` (ids are unique only within a collection).
- **Top-k** via a single bounded min-heap of size `k` fed by every in-scope
  collection (don't sort all N, and don't merge per-collection result lists).
  `min_score` filters during selection. `f32` isn't `Ord`: scores are ordered with
  `f32::total_cmp`, and `NaN` is treated as the lowest possible score so it never
  displaces a real result. `normalize` leaves a zero / non-finite / near-zero
  (`< ~1e-12`) vector unchanged, so it scores 0 against everything.
- **Ranking is a total order — a contract, not an implementation detail.** Results are
  ordered by `(score descending, collection ascending, id ascending)`. The tie-break is
  applied *inside* the bounded heap, not only when sorting the survivors, so which of two
  equal-scoring documents is retained at the `k` boundary does not depend on the order the
  scan happened to visit them in. Every ranked surface obeys it: vector `search`, BM25
  `text_search`, the RRF fusion in `hybrid_search`, and the ANN / per-segment paths.
  Consequence, and the reason it is stated here: the same query against an unchanged store
  returns the same ranking every time, which is what makes pagination coherent.
- **Pagination is offset/limit** (`SearchOpts::offset` + `top_k`, `HybridOpts::offset`,
  `ListOpts::offset` + `limit`) — deliberately *not* an opaque cursor. A search ranks
  `offset + top_k` deep and then drops the first `offset`; the drop happens in exactly one
  place, after the top-k cap, because applying it before would silently return a short page.
  For hybrid search the cut is on the **fused** ranking, never on a leg: a leg's rank is an
  input to the fused score, so paginating a leg would change which documents fuse at all.
  An `offset` past the end of the results is an empty page, not an error — a caller walking
  pages must be able to stop. `offset + top_k` is bounded by 10 000 at the HTTP boundary,
  and exceeding it is a `400`, never a clamp (a shortened page is indistinguishable from
  the end of the results). The memory API's `RecallOpts` deliberately has no `offset`.
  > Stability caveat: a page is stable only against an **unchanging** store. Concurrent
  > upserts and deletes shift the ranking, so a document can move between pages or be
  > seen twice across a paged walk. Offset/limit cannot deliver more than that, and
  > pretending otherwise with a cursor would only hide where the guarantee ends.
- **The exact/approximate choice is per query, not per store** (`SearchOpts::exact`,
  nidus-m50.12). `Config::ann` and `Config::quantization` decide what an instance *has*;
  `exact: true` decides, for one query, not to use it — the ANN walk, the per-segment IVF
  fan-out, and the quantized first pass are all bypassed and the query runs the exact f32
  brute-force scan. The case it exists for is a caller who wants a guaranteed-exact answer
  over a small filtered subset while keeping the index for everything else. Default `false`,
  so an instance's configured path is unchanged for every caller who does not ask.
- **Projection selects the attrs a `Hit` carries** (`SearchOpts::projection`,
  `ListOpts::projection`, nidus-m50.7). `Projection::All` (the default) is every attr;
  `Include`/`Exclude` name a subset. It is applied where a hit is *materialized* — one
  place, `hits_from_topk`, plus the `list` tail — so an excluded attr is never cloned and
  the saving on a long-body collection is real rather than cosmetic. It is an enum, not a
  pair of lists, so "include and exclude at once" is unrepresentable in the library; the
  wire form carries `include_attributes`/`exclude_attributes` and answers `400` when both
  are sent, rather than inventing a precedence rule (nidus-m50.15). Ranking is untouched:
  projection changes the payload, never the order or the scores. **Highlighting is
  independent of it** (§7.8): fragments are cut from the *stored* text, so projecting a long
  body away and keeping only its snippet is the supported combination, not a silent no-op.
  `RecallOpts` has neither knob, for the same reason it has no `offset` — the memory API
  stays lean.
- **A full-text query is a list of clauses** (`FtsQuery { clauses, combine, highlight }`,
  nidus-m50.10). Each `FtsClause { field, text }` names one indexed field *and its own query
  text*, so `title:"rust"` + `body:"async runtime"` is a single query — the reason the shape
  is `clauses: [{field, text}]` rather than `fields: [name]`. A document's clause scores fold
  by `FtsCombine`: `Sum` (the default — matching two fields beats matching one harder) or
  `Max` (the strongest single clause, so a long body cannot out-accumulate a precise title).
  A clause naming a field the collection does not index contributes nothing; a single clause
  scores exactly as the one-field query always did under either fold, because BM25 here is
  strictly positive so both folds are identities on one value. An **empty clause list is an
  error**, not a match-all — over HTTP a `400`, since an empty result would otherwise read as
  "the corpus has no matches" rather than "you sent no query". `min_score` applies to the
  folded score. `hybrid_search` fuses one vector leg with the *combined* text leg, so the RRF
  numbers for a single-clause query are unchanged and per-clause weights stay out of scope
  (they belong to the ranking ticket).
- **Ranking expressions are additive and off by default** (§7.6): `SearchOpts::rank_by`
  layers a recency decay over the metric (subtracting an age penalty, so it holds for every
  `Distance` and for BM25), `HybridOpts::vector_weight`/`text_weight` weight the fused legs,
  and `ListOpts::order_by` is a plain ORDER BY with no vector query at all. A query that sets
  none of them returns exactly what it returned before they existed. **Result diversity** is
  `SearchOpts::limit_per` — a cap on hits per attribute value, exact only within an over-fetch
  window (§7.7). Aggregation (`count`/`sum`) is answered without materializing a record.
- **Filters** (`Filter` = AND of `Predicate`s) are evaluated against `attrs` before
  scoring: `Eq` (typed equality), `Ne` (typed inequality), `Glob` / `IGlob` (pattern
  match on a `Str` attr, case-sensitive and ASCII-case-insensitive, §7.1), `In` /
  `NotIn` (set membership), `Lt`/`Le`/`Gt`/`Ge` (ordered
  range comparison), and the text predicates `Fuzzy` / `ContainsAllTokens` /
  `ContainsAnyToken` / `ContainsTokenSequence` (§7.4) and `Regex` (§7.5).
  This covers typical needs: path-prefix scoping (`Glob "path*"`),
  type/language/kind equality, exact-path matches, glob-based bulk deletes, presence
  sweeps, numeric/date ranges (`Ge "ts" 1700000000`), and exclusions (`Ne "status"
  "archived"`). The range predicates are **same-type and orderable only**: `Int`
  numeric, `Str` lexical, `Bool` (`false < true`); a cross-type or non-orderable
  (`Null`, `List`) comparison never matches. OR/disjunction is intentionally absent —
  compose at the call site, or it is a future additive extension.

### 7.1 Glob subset
`glob.rs` implements the GLOB subset callers actually use: `*` (any run, incl.
empty), `?` (exactly one char, never empty), `[...]` / `[!...]` / `[^...]` (char
class / negation, with ranges). Recursive matcher with `*` backtracking — fine for
short keys like file paths. The pattern is **anchored at both ends** (the whole
pattern must match the whole text); an unterminated `[` (no closing `]`) is treated
as a literal `[`. This matches common SQL `GLOB` semantics so an application
migrating off such a backend behaves identically.

`IGlob` is the same subset with **ASCII** case folded on both sides — the pattern and
the attribute are each mapped through `to_ascii_lowercase` before matching. Every
metacharacter (`*`, `?`, `[`, `]`, `!`, `^`, `-`) is outside `A-Z`, so folding cannot
disturb pattern structure, and an uppercase range like `[A-Z]` folds to `[a-z]` rather
than becoming unmatchable. The fold is deliberately ASCII-only and context-free: `É`
does not match `é`, because a locale-dependent fold would make the same pattern mean
different things on different machines. Case-*sensitive* path comparison is usually
not a useful distinction — on a case-insensitive filesystem a caller can easily hold a
path in a casing the store does not have — so `IGlob` is the right default for
path scoping, and `Glob` for anything that must compare exactly.

`filter::matches` AND-combines predicates (empty filter matches everything); an
absent key fails **every** leaf predicate — including the negative ones (`Ne`, `NotIn`,
`NotContains`) and the range ones. Each leaf predicate is a positive assertion about a
*present* attribute, so a record lacking the key is never a match (e.g. `Ne "status"
"archived"` does not match a record with no `status`). `Eq(key, Null)` likewise
requires the key to be present and equal to `Null` (absent ≠ `Null`, per §3).

### 7.2 Array containment

`Contains`, `NotContains`, and `ContainsAny` look *inside* a `List`. Lists hold strings
(§3), so a non-`Str` needle is unfindable rather than coerced — `Contains("tags", Int 1)`
does not match the list `["1"]`. Matching is whole-element equality, not substring:
`Contains("tags", "rust")` does not match `["rustacean"]`; `Glob` is the tool for
substrings, and it remains `Str`-only (a `List` fails it). `Contains` on a scalar `Str`
fails — a string is not a one-element list. `NotContains` requires the key present and
list-typed, mirroring `Ne`; `ContainsAny` with an empty candidate set is `false`, since
nothing can overlap. There is deliberately **no** `ContainsAll` variant: `All` over
several `Contains` already expresses it.

### 7.3 Boolean composition

`All`, `Any`, and `Not` are predicates that take predicates, so arbitrary boolean shapes
nest without changing `Filter` itself — it stays a conjunction, and the existing flat
`[p, q]` wire form keeps its exact meaning and its scan fast path. Empty groups take the
standard identities: `All([])` is `true` (matching `Filter`'s empty case), `Any([])` is
`false`.

`Not` negates the *truth value* of its sub-expression, which makes it behave differently
from the negative leaf predicates on an absent key. `Ne(k, v)` is `false` when `k` is
missing (it asserts a present-and-different attribute); `Not(Eq(k, v))` is `true`, because
`Eq` was false. Both are useful — reach for `Ne`/`NotIn`/`NotContains` to require the
attribute exist, and `Not` for genuine set complement. This is the one asymmetry in the
filter language that reliably surprises, so it is tested explicitly.

Nesting depth is bounded before evaluation: filters reach the store only through
serde_json (HTTP body, CLI `--where`), which caps nesting at 128, and no op-log `Op`
carries a filter — `delete_where` resolves to ids before logging. So a filter can be
neither deep enough to exhaust the stack nor persisted to blow up on replay.

### 7.4 Fuzzy and token text predicates

`Fuzzy(key, needle, n)` matches when the attribute is within `n` **Levenshtein** edits of
`needle` — the memory-store case of an agent recalling a half-remembered identifier. The
distance is the plain three-operation one (substitution, insertion, deletion), computed by
a two-row DP with no dependency, so a **transposition costs 2**, not 1: `Fuzzy("word",
"from", 1)` does not match `"form"`. Both sides are **ASCII-case-folded**, following `IGlob`
(§7.1) and for the same reason — a locale-dependent fold would make one query mean different
things on different machines, so `É` is still not `é`. Distance counts *characters*, not
bytes. `n` is capped by `MAX_FUZZY_EDITS` (**8**); a larger budget is an **error surfaced to
the caller, never a silent clamp**, because a clamped budget quietly answers a different
question than the one that was asked.

The token family — `ContainsAllTokens` (every query token present, in any order),
`ContainsAnyToken` (at least one), `ContainsTokenSequence` (the query tokens consecutive and
in order: a phrase) — tokenizes the attribute **at query time**. That tokenizer is
deliberately **simpler than the FTS analyzer** (§9): a token is a maximal run of
alphanumerics, ASCII-case-folded, with **no stemming and no stopword removal**. These are
*filter* predicates, where a term either is or is not present; stemming belongs to ranking,
where a partial-credit match is meaningful. Consequence to know: `ContainsAllTokens("body",
"run")` does not match `"running"`, and `text_search` for the same word does.

All four read **any text the attribute carries**: a `Str` offers itself, a `List` offers each
element and matches if **any single element** satisfies the predicate (consistent with
`Contains`, §7.2). A phrase therefore never spans two list elements, and "all tokens" is not
the union across elements. Every other `Value` variant offers nothing, so — like every leaf
(§7.1) — an absent or wrong-type attribute never matches. Empty queries take the standard
identities: `ContainsAllTokens` with no tokens is `true` for any present text attribute
(as `All([])` is), `ContainsAnyToken` with none is `false` (as `Any([])` is).

**Cost, stated plainly: there is no index behind any of this.** Every predicate here
re-tokenizes or re-scans the attribute for **each record the scan visits** — O(attribute
length) per row for the token predicates, and O(needle × attribute) for the `Fuzzy` DP
(skipped outright when the two lengths differ by more than the budget). That is fine at the
scale nidus targets and it is *not* fine as a substitute for full-text search over a large
corpus; reach for `text_search` when the field is a document. Indexing them is future work.

### 7.5 Regular expressions

`Regex(key, pattern)` matches the attribute against a regular expression, over the same
"any text the attribute carries" rule as §7.4. The engine is the `regex` crate — pure Rust,
no C, no native linking, and adding it moved the whole-crate clean build by under a second
(11.4s → 11.6s for the library, 14.5s → 15.1s with `--features cli`), well inside the
build-cost bar of §1.

**There is no ReDoS mitigation because there is no ReDoS.** `regex` is a finite-automata
engine with a documented linear-time guarantee in the length of the input and does not
backtrack, so a pathological pattern cannot blow up. Adding a timeout or a complexity
heuristic would buy nothing and would make the predicate's behaviour depend on machine
speed. Nesting/repetition is bounded instead by the engine's own compile-time size limit,
which surfaces as an ordinary invalid-pattern error.

Two decisions worth stating because they will otherwise bite someone:

- **Anchored at both ends**, matching `Glob`/`IGlob` rather than the unanchored default of
  most regex APIs. The whole attribute must match; `.*` on either side opts back into a
  substring search. Consistency inside one filter language wins here — a caller choosing
  between `Glob "src/*"` and `Regex "src/.*"` should get the same *shape* of answer.
  Implemented by wrapping the caller's pattern as `^(?:…)$`, so an alternation is anchored
  as a whole and a caller's own `^`/`$` remain harmless.
- **Case sensitivity is the pattern's own `(?i)` flag**, not a second predicate variant.
  `Glob`/`IGlob` are a pair because a glob has nowhere to put a flag; a regex does.

An unparseable pattern is a **caller-facing error**, never a panic and never a silently
non-matching filter. Compilation happens **once per query**, not once per record: every
public query method prepares its filter first (`filter::validate`), which compiles each
pattern into a process-wide cache keyed by the pattern text; the per-record path is then a
read-locked lookup. The cache is capped (256 distinct patterns) and cleared wholesale on
overflow, since patterns arrive from untrusted request bodies.

### 7.6 Ranking expressions: recency decay, leg weights, ORDER BY

Until nidus-m50.3 there were exactly three rankings: cosine (or the store's metric), BM25,
and an RRF fusion of the two at fixed leg weights. Three additive knobs widen that, and
**every one of them is off by default** — an untouched query returns byte-identical results.

**Recency decay** (`SearchOpts::rank_by = Some(RankBy::Decay(…))`) is the one that most
changes result quality for a memory store: on pure cosine a two-year-old note beats a fresh
one that says the same thing. A `Decay` names a timestamp attribute (`Value::DateTime` or a
`Value::Int`, both epoch milliseconds), an `origin` ("now", supplied by the caller so a
ranking is reproducible rather than clock-dependent), a `scale`, and three tuning knobs.
The factor is

```
age    = max(0, origin − attrs[field])          # milliseconds; a future stamp is un-aged
factor = decay ^ (age / scale)                  # `decay` at exactly one `scale` of age
score  = base − lambda × (1 − factor)
```

**The penalty SUBTRACTS; it does not multiply.** This is the load-bearing decision. A
multiplicative factor (`base × factor`) is only meaningful when scores are non-negative and
larger-is-better on a fixed scale — i.e. Cosine, and only after a clamp, because multiplying
a *negative* cosine score by a decayed factor makes it larger. nidus has three metrics:
`Euclidean` scores in (−∞, 0] and `DotProduct` scores anywhere at all. A subtraction is a
translation, and a translation preserves order under every one of them and for both signs.
It also composes with the raw BM25 scores of `text_search`, which have no bounded range
either. The cost is that `lambda` is expressed in *score units* rather than as a fraction,
so it must be chosen against the metric in use — a real trade, and the right one.

Three defaults matter:

- `decay` defaults to `0.5`, which makes `scale` a **half-life**.
- `lambda` defaults to `1.0` — one full unit of score for a fully-decayed record.
- **`missing` defaults to `1.0`: a record with no usable timestamp is not penalized at all.**
  The alternative (treating absent as infinitely old) would mean that switching decay on
  silently buries every document written before the field existed. Callers who want the
  opposite set `missing: 0.0` explicitly.

`rank_by` **does not force the exact path.** `SearchOpts::exact` is the explicit opt-in for
that, and conflating the two would mean enabling decay silently disabled the index. The
expression is applied where each candidate's final score is decided, which every path
already reaches with the record's attrs in hand: the brute-force scan, the ANN walk's
post-filter rerank, the per-segment IVF fan-out, and the quantized two-pass rerank. Over an
approximate result set decay therefore **inherits that path's approximation** — the
candidate *set* was chosen on the base score, and only the ordering within it is decayed.
`min_score` is compared against the final, decayed number on every path, not the base one.

One performance note, stated rather than hidden: a ranked scan runs **serial**. The chunk
kernels that feed `parallel_topk` see only the vector matrix, never the attrs, so the
expression path takes a separate loop that does a per-row index lookup. Generalizing the
parallel engine to carry attrs is deliberately deferred; a query with `rank_by` set gives up
`Config::query_threads` and pays an index lookup per scanned row.

**Per-leg hybrid weights** (`HybridOpts::vector_weight` / `text_weight`) scale each leg's
contribution to the fused score: a document scores `Σ wᵢ / (rrf_k + rankᵢ + 1)`. Both at the
default `1.0` reproduces unweighted RRF **bit for bit** — multiplying by `1.0` is exact, not
approximately exact — so the default fusion is unchanged. A non-finite or negative weight is
refused: `NaN` makes every comparison in the sort false, and a negative weight inverts a leg
rather than de-emphasizing it.

**ORDER BY with no vector query** is `ListOpts::order_by`. Sorting runs over the whole match
set *before* the page is cut, so `offset`/`limit` walk the sorted order. Cross-type ordering
mirrors the filter's same-type rule (§7): the **witness** — the first value in the incoming
stable order that compares against itself — fixes the sort's type, and every value that does
not order against it (a different variant, an unorderable `Null`/`List`/`NaN`, or an absent
attribute) falls into one **trailing bucket**, in the order `list` built it. The bucket stays
trailing under `descending`, which is why direction is applied to the value comparison only
and never to the bucket split.

### 7.7 Aggregation and result diversity

**`count` and `sum`** (`Nidus::aggregate`, `POST /aggregate`) answer from the in-RAM index in
one pass: no `Record` is built, no vector row is read. `count` is always reported;
`AggregateOpts::sum` names attributes to total. A sum reports a **tagged `Value`**, like the
rest of the API — `Int` while every addend was an `Int`, `Float` once any `Float` joined (the
integer part is accumulated in `i128`, so a long `i64` run cannot wrap, and a total past
`i64` is reported as `Float` rather than silently truncated). A missing or non-numeric value
is **skipped, not counted as zero**, which is the only reading that keeps `sum` and `count`
independently meaningful.

**`group_by`** (`AggregateOpts::group_by`) reports the same `count`/`sum` per **distinct value
of one attribute** rather than only over the whole scope — turbopuffer's `ForEachUnique`. It
rides the same single pass, and the whole-scope totals are still reported beside the groups, so
"how many per language, and how many overall" is one query rather than two. Rows come back
ordered by `count` descending, ties broken by the group key, because a `HashMap` has no order
and an answer that reshuffles between identical calls is not reportable. Records **missing**
the attribute form one group with a `null` value — distinct from those holding `Value::Null`,
mirroring the absent-vs-`Null` rule the filter predicates already follow. Distinct values are
bounded by the same `MAX_GROUPS`; past it new values are dropped and **`groups_truncated` says
so**, since a silently short list of groups reads exactly like a complete one.

**`limit_per`** caps how many hits may carry any one value of an attribute — "at most 2 hits
per file". For a memory store this is the highest-value piece here: it stops one verbose
document from filling the whole recall window. The group value is read from the **live
record**, not from the returned hit, so a `Projection` that excludes the field cannot lift
the cap. Records **missing** the attribute form ONE shared group (`MAX_GROUPS`, currently
10 000, bounds the distinct values tracked; past it further values pass uncapped). Without
that shared group, deleting the attribute from a document would be a way to opt out of the
cap entirely.

**`limit_per` is exact only within the over-fetch window, and that trade is deliberate.** A
capped search ranks `(offset + top_k) × 8` deep, applies the cap in rank order, then cuts the
page. Documents past that window were never scored into the ranking, so a page can come back
**shorter than `top_k`** even though more matching, uncapped documents exist further down.
Making it exact would mean ranking the entire match set for every capped query — an unbounded
scan to satisfy a diversity knob. The cap that *is* guaranteed is the upper bound: no
returned page ever carries more than `max` hits for one value.

### 7.8 Result annotations — why a hit matched

A `Hit` carries one score and an attrs map, which does not answer "which clause fired, and
what part of the text matched". `Hit::annotations` does, and is `None` unless the query asked
(nidus-m50.5) — the default response is byte-identical to a nidus without the feature.

- **Per-leg sub-scores** (`SearchOpts::explain` / `HybridOpts::explain`). A `text_search` hit
  reports each *matched* clause's own BM25 score in query order (an unmatched clause is
  absent, not a zero row). A `hybrid_search` hit additionally reports each fusion leg's own
  `(rank, score)` — the data `rrf_fuse` already computes per leg and used to discard at the
  call site. The clause breakdown is carried across fusion in a side map keyed by
  `(collection, id)`, because fusion rebuilds the `Hit` and cannot see inside a leg. Vector
  `search` ignores `explain`: it has a single score, and reporting it twice explains nothing.
- **Highlighted fragments** (`FtsQuery::highlight`). The hard part is that the analyzer stems
  and folds, so a matched *term* is generally **not a substring of the stored text** — a query
  for "run" matches a document spelling it "running", and searching the text for the term
  finds nothing. So the analyzer is the offset source: `analyze_spans` is the single analysis
  path (`analyze` is it with the offsets dropped, so a highlighter can never disagree with the
  index about what was a term) and every emitted term carries the byte range of the original
  text it came from. A fragment is a character window around the matched spans, snapped to
  token boundaries at both ends so it does not open or close mid-word, with `spans` rebased to
  byte offsets **within the fragment**. `HighlightOpts { max_fragments, fragment_chars }`
  bounds it; a `List` field is highlighted over the same space-joined text the index saw.

Both run on the **final page**, after pagination and after the fused truncate, so the cost is
bounded by `top_k` rather than by the candidate set.

---

### 7.9 Multi-query batching

`POST /search/batch` answers up to **16** vector queries in one request (nidus-m50.11). Each
entry is an ordinary search body with its own scope, filter, and `top_k`, and each is validated
by the same code the single-query route runs — **all of it before any leg executes**, so a
malformed leg fails the request rather than returning a partial answer a caller cannot tell
apart from a complete one.

The batch runs under **one** lock acquisition, one blocking task, and one request deadline. That
is the whole point (an agent fanning one question into several phrasings pays one round-trip and
one queueing delay), and it is also why the count is capped: a request that holds one concurrency
permit must not be able to buy unbounded scan.

Optional `fuse` merges the legs into a single ranking with the **same RRF** `/hybrid-search`
uses — N query legs instead of a vector leg and a text leg — with optional per-leg weights. A
weights list that is neither empty nor exactly as long as `queries` is a `400`, not a
zero-filled shorter list, because the silent version re-weights the wrong leg.

This is a **transport** concern, not a storage one: there is no library counterpart, since an
embedding caller already has the loop and the fusion is `rrf_fuse` either way.

## 8. Compaction

Deletes and overwrites leave dead rows in `data` and superseded records in `log`.
`compact()` rewrites both, live-only:

1. Walk the in-RAM index; assign fresh contiguous row indices to live docs.
2. Write `data.tmp` (live vectors) and `log.tmp` (`CreateCollection` + `SetMeta` +
   one `Upsert` per live doc with new row indices).
3. fsync both → atomically rename over `data`/`log` → swap in-RAM `vectors`.

Triggered by a dead-row-ratio threshold on `open`, and on explicit `compact()`.
Full reindexes churn little; incremental indexing needs this to bound growth.

---

## 9. Seams: shipped and still-deferred

Every seam here is purely additive over the format in §5 — **none changed the vector
data layout.** The manifest is versioned separately and *has* changed: v2 carries the
open profile (§14.2), and a v2 manifest is unreadable by a pre-0.60 binary. Several were designed-for here and have since been built; the design
rationale is kept so the choices stay legible. The rest stay deferred: do **not**
build until a real need exists.

### Shipped (was a deferred seam)

- **Parallel scan.** `search(&self)` fans the row scan across `Config::query_threads`
  workers via `std::thread::scope` (zero-dep std — no rayon, no added dependency) into
  per-worker bounded top-k heaps, merged at the end. No API or format change. Opt-in:
  `query_threads` defaults to `1` (serial, zero behavior change), and a dim-aware work
  floor (`rows × dim` below a threshold) keeps small scans serial so spawn/join cost
  never dominates. Each worker sorts its own chunk by physical row, so per-chunk access
  stays prefetcher-friendly and the global sort is skipped on the parallel path (no
  Amdahl tax). Both the exact f32 scan **and** the int8 first pass parallelize. The f32
  scan is bandwidth-bound (sublinear gain past a few cores); the int8 first pass is
  compute-bound and scales better. See §6.5.
- **Cached scan order.** A whole-store search/`list` scans every live doc in physical-row
  order for prefetcher-friendly `data` access (the nidus-33k win), which means a
  `(row, collection, id)` scan sorted by row. That order only changes on a write, so it
  is cached in RAM (`RwLock<Option<…>>`) and reused across the many queries between
  writes instead of being re-sorted every query — a ~27% serial-search win at n=100k,
  dim=768 (the sort was ~2.16 ms of a ~8 ms query). Built lazily on the first whole-store
  query after a write (subset-only scopes keep the direct iterate-and-sort path, so they
  never build it), and invalidated by `upsert`/`delete`/`drop_collection`/`compact`. No
  API or format change; in-RAM only. The int8/binary serial first passes drop their own
  per-query sort too. The parallel path is unchanged (it sorts per-chunk, §6.5).
- **Scalar (int8) quantization.** `Config::quantization` maintains an in-RAM int8
  matrix mirroring the f32 rows one-for-one and runs a two-pass search: an int8
  first-pass — monotonic with the f32 score under a single shared symmetric scale, so
  it picks the right candidate set — selects an overscanned candidate set (`top_k ×
  rescore`), then f32 reranks those for exact scores. ~4× less memory traffic on the
  first pass. The scale refits geometrically on growth so incremental upsert stays
  amortized O(1)/row. Affects only the in-RAM matrix + the scoring kernel, never the
  `data` segment on disk. The scheme is selected by `Quantization::int8()` (the default).
- **Binary (sign-bit) quantization.** `Quantization::binary()` maintains an in-RAM
  packed-bit matrix (`dim/8` bytes/row, ~32× smaller than f32, 8× smaller than int8) and
  runs the same two-pass shape with a Hamming-distance first pass (`u64::count_ones` —
  pure Rust, autovectorizes, Miri-clean, no new deps), then an exact f32 rerank. **Cosine
  only:** sign codes are an angular (SimHash) proxy that discards magnitude, so binary is
  rejected at `open()` for dot-product/Euclidean. Scale-free (a row's code is just its
  signs), so incremental upsert is plain append — no scale, no refit. Parallelizes harder
  than int8 (32× less first-pass traffic). The first pass overscans more (`rescore`
  defaults to 16 vs int8's 4) to offset the coarser proxy.
- **Lightweight server.** `nidus serve` (behind the opt-in `cli` feature) wraps a
  long-lived `Nidus` in a thin axum/tokio HTTP layer — exactly the separate-wrapper
  shape this seam called for, not a change to nidus core. The enabling pieces were
  already here: the cross-process lock + lock-free read snapshots (§6.2) and
  `OpenMode::ReadOnly` (§4.1) let a writer process and one-or-more search servers share
  one store. The core API stayed operation-centric, with no process-wide assumptions.
  Its deps (`clap`, `tokio`, `axum`, `tower`, `serde_json` — all pure Rust, zero FFI)
  compile only under `--features cli`, so `cargo add nidus` stays lean.
- **MCP surface, two transports.** Behind the opt-in `mcp` feature (folds `cli` +
  `memory`), one `NidusMcp` adapter (`src/server/mcp/`) speaks MCP 2026-07-28 both
  nested inside `nidus serve`'s HTTP stack at `/mcp` (inheriting its auth, body
  limits, backpressure, and metrics) and standalone over stdio via `nidus mcp
  --dir …`, for a client that spawns its own server process (e.g. `claude mcp add
  nidus -- nidus mcp --dir ~/.nidus`). Both hand the same tool list to rmcp's
  `ServerHandler` blanket impl, so nothing forks by transport. Unlike `serve`,
  `nidus mcp` opens the store eagerly and fails fast — there is no listener to
  keep answering health probes while a standby waits — so a second process on the
  same directory exits immediately, naming the lock conflict. A stdio session
  skips `limits.rs`/`metrics.rs` entirely (both are axum `.layer()`-only): one
  local client needs neither an admission cap nor a scrape endpoint.
- **Agent-memory write path (nidus-k28.7/.5/.6).** `remember` provisions on first
  write (collection plus a default full-text schema over `nidus.text`, gated on
  `Nidus::has_fts_schema` — `set_fts_schema` rebuilds the field index from every
  live doc, so an ungated call would make each write O(collection size)), stamps a
  reserved attr vocabulary, and optionally suppresses near-duplicates. The reserved
  keys are `nidus.text` (the raw text, always stamped, and what the default schema
  indexes), `nidus.created_at` / `nidus.updated_at` / `nidus.expires_at` (all
  `Value::DateTime`, UTC epoch ms). `nidus.source` predates `nidus.text`, carried
  exactly the same value, and is retained read-only so records written before this
  change still resolve — nothing stamps it now. Because `upsert` replaces a doc's
  attrs wholesale, both preserving `created_at` and merging a dedupe match's
  untouched attrs require an explicit read-before-write, done inside the same
  `run_write` closure so it is atomic against every other queued write.

  **TTL is enforced at read time, and that is not one option of two.** nidus runs
  no background threads and has no periodic sweep, so nothing else *can* hide an
  expired entry: a compaction-only design would leave expired entries recallable
  until whenever `compact` next ran, which is not expiry. The guard is a predicate
  AND-ed into the caller's filter on every memory read — all MCP read tools,
  `Memory::recall`, and HTTP `/recall` — in the true-complement form
  `Not(Le(nidus.expires_at, now))` — a bare `Gt`/`Ge` is false on an absent
  key (§7 range semantics) and would silently hide every entry that never got a
  TTL. `get` bypasses `Filter` entirely, so it carries the same check by hand.
  The raw store surfaces (`search`/`list`/`get_all` and their HTTP routes) stay
  unguarded on purpose: TTL is a memory-layer contract, and raw access must see
  what the store holds.
  Physical reclaim stays a separate, caller-triggered concern reached through
  `delete_where` + `compact` (§8); it is deliberately not new logic inside
  `compact`'s per-doc re-emission loop.
- **ANN index (HNSW/IVF).** `Config::ann` opts a store into an in-RAM approximate
  index over the same `data` rows; `search` walks it instead of scanning. Two
  algorithms, selected by `AnnKind`: **HNSW** (`AnnConfig::hnsw`, the default — a
  navigable small-world graph with native incremental insert) and **IVF**
  (`AnnConfig::ivf` — k-means inverted lists). Both are pure safe Rust with no new
  deps; the only randomness is a hand-rolled seeded splitmix64 PRNG, so builds are
  deterministic and the logic runs under Miri. The index only *picks* an over-fetched
  candidate set (`top_k × overscan`); the store then post-filters those candidates by
  scope + metadata filter + `min_score` and ranks them by the exact f32 score, so
  final ordering is always exact even though candidate *selection* is approximate.
  **Approximation cost:** a very selective filter or collection-subset scope can
  starve the candidate set (the graph walk surfaces too few matching rows). The
  **exact-prefilter fallback** (nidus-0ou, shipped) closes this: when a narrowed
  query's survivor population drops below what the overscanned walk can reliably
  surface (`total/overscan`), the store gathers the filter-passing rows directly and
  scores them exactly instead of walking the index — so selective filtered searches
  stay exact rather than silently losing recall. The e2e canary
  (`scale::ann_filtered_search_recall_stays_above_the_floor`) pins this. Deletes
  leave stale nodes in the index that are skipped at query time (the candidate→doc
  resolution is re-verified against the live index) and reclaimed on the next
  `compact` rebuild. ANN and quantization **may be combined** (nidus-ndu): when
  `Config::quantization` is also set, the index walk — both the build heuristics and
  the query traversal — scores the store's int8/binary codes (the graph/lists are built
  *in* that quantized space) for a cheaper candidate selection, and `search` then reranks
  the candidate rows with the exact f32 score, so accuracy is restored while the walk
  stays cheap. IVF keeps its k-means fit and centroids in f32 (a mean of codes is
  meaningless) and only its per-row list scan goes quantized. Recall runs a touch below
  the exact-walk index — widen `ef_search`/`n_probe`/`overscan` to recover it. The index
  is extended in O(batch) on `upsert`.
  **Persistence (derived cache).** The graph/lists are reconstructable from the
  vectors, so they are persisted only as an optimization: a separate `ann` file
  (`NIDUS\0` header + `bincode` + CRC32, atomically written) lets `open` *load* the
  index instead of rebuilding it (the expensive part — HNSW build is scalar/
  single-threaded). It is written strictly **out-of-band** — on `compact`, the
  explicit `Nidus::persist_index()`, and the clean-shutdown path of `nidus serve` /
  `nidus mcp` (§9) — **never** on the `upsert`/`flush` hot path, so
  writes stay fast and there is no background thread. `open` loads the cache, validates
  it against the current `(dim, distance, kind, params, quantization)` + a CRC, and **incrementally
  catches up** any rows appended since it was written; an absent, stale, over-long, or
  corrupt cache is silently discarded and the index rebuilt from the vectors. The
  `data`/`log` format is unchanged.
  **Parallel build.** `Config::query_threads > 1` also parallelizes the from-scratch
  HNSW build (on a cacheless `open` and on `compact`): node levels are assigned
  serially, then per-node neighbour search + linking run across `std::thread::scope`
  workers with one `Mutex` per node's adjacency and an `RwLock` entry point, edges
  locked in node-id order (deadlock-free; safe Rust precludes data races). The serial
  build at `query_threads == 1` is unchanged and deterministic; a parallel build
  varies slightly with thread count (insertion order), with equivalent recall.
  Incremental `upsert` stays serial. IVF build is already cheap and stays serial.
- **Full-text search (BM25) + hybrid + optional vectors.** A collection can declare
  full-text-indexed attribute fields (`create_collection_with_fts` / `set_fts_schema`,
  persisted as a `SetFtsFields` op). nidus then maintains an in-RAM inverted index per
  `(collection, field)` and answers `text_search(FtsQuery)` by BM25, reusing the same
  `Hit`/`Filter`/scope/top-k machinery as vector search. A query is a **list of clauses**
  over several fields, folded by `Sum` or `Max` (§7). `hybrid_search` fuses a vector
  and a BM25 query with **Reciprocal Rank Fusion** (rank-based, so the incomparable
  cosine/BM25 scales need no normalization), and either surface can annotate its hits with
  per-leg sub-scores and highlighted fragments (§7.8). The analyzer is pure-Rust, zero-FFI
  (lowercase → Unicode tokenize → optional ASCII folding + token-length cap → English
  stopwords → Porter stem) behind a `Language` enum (US English today; the seam is open
  for more).
  **Per-field tuning.** Each declared field is an `FtsField { field, k1, b, analyzer }`
  (`Analyzer { language, ascii_folding, max_token_len }`), defaulting to BM25's textbook
  `k1 = 1.2` / `b = 0.75` over the US English analyzer with no folding and no token-length
  cap — bit-identical to the store-wide constants these replaced. The parameters are part
  of the persisted schema (`SetFtsFields`) **and** of the `fts` cache's validity key, so a
  redeclared schema rebuilds the index rather than serving postings scored under the old
  parameters. The superseded `SetFtsSchema` op (a language per field, no params) is still
  replayed — with the defaults — so a store written before this opens unchanged. To support
  pure-text corpora,
  `Record.vector` is now `Option`: a **text-only** doc (`Record::text_only`) carries no
  embedding, occupies no data row, and is found by full-text/metadata queries but never
  by vector search — coexisting with vector-bearing docs in one collection (a new
  append-only `UpsertText` op carries it; the `data` format is unchanged). The FTS index
  is a derived cache like ANN; today it rebuilds from the replayed docs on `open` (an
  on-disk `fts` cache, sharing the ANN cache codec, is the planned follow-up).
- **Shared index-cache codec.** ANN and the forthcoming FTS cache share one framing
  module (`NIDUS\0` header + validity key + watermark + `bincode` + CRC32, atomic
  temp/fsync/rename), so a derived index persists/loads through a single source.

- **mmap immutable segments.** `Config::mmap` swaps the single "row `i` → `&[f32]`"
  accessor for a memory-map of each **immutable** (sealed) segment instead of reading it
  into an in-RAM `Vec<f32>`; the active (appendable) segment stays in RAM. The OS pages a
  cold segment in on touch, so a store can hold more vectors than fit in RAM, with zero-copy
  load and cross-process page sharing. The cost is the one conscious FFI/`unsafe` opt-in nidus
  permits: `memmap2` (a thin, fast-compiling wrapper over the platform `mmap` — no C to build),
  with the only `unsafe` in the crate being the single `Mmap::map` call (`#![deny(unsafe_code)]`
  + one scoped `#[allow]`); the byte→f32 reinterpret is a safe alignment-checked `bytemuck`
  cast. Default **off** (all-RAM, unchanged). Effective only for a **local-FS** store with
  sealed segments (it needs `segment_max_rows` to produce immutable segments and a mappable
  local file; an object-store / in-memory store silently stays all-RAM), and **little-endian**
  hosts (the on-disk f32 layout, §5.1). Applied per segment, it is the §14.6 phase-3 leg of the
  segment scale model. Reads map through the same accessor, so results are identical to the
  RAM path — exact, filter-respecting, ANN/quant-compatible. (Compaction of a mapped store
  materializes the live set in RAM like any compaction, so it is bounded by RAM even when the
  store is not.)

### Still deferred (designed-for, not built)

- **Pluggable storage & memory backends.** Generalize the §5 local directory along two
  orthogonal axes behind two sync traits: a **persistence** backend (durable `data`/`log`
  — local files / S3 / GCS) and a shared **memory tier** (the warm working set — local RAM
  / Redis / Valkey / KeyDB / Dragonfly), selected by URL scheme (`file://`, `s3://`, `gs://`;
  `redis://`, `valkey://`, …). The on-disk *object set* (`data`/`log` + derived caches) is
  unchanged — only *where the bytes live* and *whether the working set is shared*. Search
  stays in local RAM, never over the wire. Built — see §13.
- **Segment-based storage (the scale model).** Evolve the single in-RAM matrix + one log
  into immutable **segments** + a write-ahead log + a manifest, so scaling (datasets past
  one node's RAM, incremental cloud writes, cooperating instances) becomes a *quantity* of
  one architecture rather than a separate mode. Brute-force stays the exact default for the
  small/recent tail; an IVF index covers the cold bulk. **Phases 1–5 (the segment format +
  manifest + WAL→segment sealing, per-segment IVF, per-segment mmap, manifest-versioned reader
  refresh, and cooperating-instances cluster mode) are built — see §14.**

#### Exotic vector types — DECIDED, 2026-08 (nidus-m50.14)

One dense vector per record, one dimension per store, stays the model. Three variants
were evaluated against turbopuffer's surface and answered separately:

- **`f16` storage — rejected, not deferred.** It buys ~2× on the vector matrix, which
  int8 quantization (shipped, `Config::quantization`) already beats at 4× with a rerank
  pass that restores accuracy. Adding a third storage width would multiply the codec,
  scan-kernel, and quantization matrix for a strictly worse trade. Nothing is waiting on
  this; the branch is closed.
- **Multi-vector late interaction (ColBERT-style) — deferred as a rerank-only feature.**
  The useful form scores a candidate set produced by ordinary dense retrieval, so it
  belongs on the rerank seam, not in the segment format. It needs no format change if
  scoped that way, and it should not be built until a caller wants it.
- **Sparse vectors (`SparseKNN`) — deferred, and deliberately not built now.** The byte
  format could carry them additively, but the surrounding cost is out of proportion to
  demand: a breaking `Record` change, a second `Op` append, a working-set key bump that
  discards every deployed memory-tier snapshot, the sparse payload held twice in RAM
  outside `max_vector_bytes`, and no worst-case bound on a query over a common dimension.
  No user has asked for SPLADE-style retrieval. Revisit when one does — the decision is
  "not yet", not "never", and nothing in the format forecloses it.

The general rule this encodes: a change to the *segment format* needs a named caller,
because it is the one layer where being wrong is expensive to walk back. Query-path
features do not carry that burden and are judged on their own merits.

### 9.1 A bundled zero-config local embedder — RESOLVED (not shipping)

Every shipped embedder (voyage, openai, ollama, cohere, gemini, mistral, jina,
openai-compat) needs an API key or a running daemon, so `nidus mcp` cannot be
zero-config. The question was whether to ship a local embedder to close that gap.

**The premise this was filed under turned out to be wrong**, and that is worth
recording because it will come up again. The assumption was that a local embedder
collides with the §1/§13.6 build-cost bar. It does not. That bar rules out an
*inference engine* — ONNX Runtime is exactly the multi-minute native-toolchain
dependency nidus exists to avoid. But a **static** embedder (model2vec-style) is not
an inference engine: it is a token→vector lookup table plus mean-pooling, which is
tens of lines of pure Rust and costs nothing to compile. Measured for reference: the
clean default build is ~16s against a "well under a minute" budget.

The real cost is **weight distribution**, a constraint the build bar does not speak to:

- **Bundled in the crate** — ~8 MB (int8) to ~32 MB (f32) for a table worth using.
  Every `cargo add nidus` pays it, including the majority who use Voyage or OpenAI,
  and crates.io's default package limit is 10 MB. This directly contradicts the
  dependency-lean promise the crate is built around.
- **Fetched on first use** — keeps the crate lean, but needs a cache directory,
  checksum verification, and a network round trip, and "zero-config" that reaches the
  network on first run is not obviously better than an API key.
- **User-supplied** — not zero-config by definition.

**Decision: ship nothing.** nidus does not embed text and does not plan to; it stores
vectors and answers queries over them. The gap is closed by documentation, not code:
the docs state plainly that an embedder is the caller's to provide, and point at
Ollama as the fully local, keyless option that already exists (`embed-ollama`, no API
key, `base_url` to a local daemon). That covers the "I don't want to phone a hosted
API" case today without putting a model table inside a vector store.

The quality ceiling reinforces it: static embeddings retain roughly 85% of a small
transformer's benchmark performance, so a bundled default would be *both* a size cost
*and* the worst-scoring option in the list — a poor thing to make the zero-config path.

Revisit only if the trade changes shape — a table small enough to be unremarkable, or
a caller for whom the Ollama route is genuinely unavailable. The decision is "no",
not "never", and nothing in the format or the `AnyEmbedder` enum forecloses it.

---

## 10. Module layout

A module that has grown to span several distinct concerns is a *directory* of
sibling files (each owning one concern) with `mod.rs` holding the core type + glue,
rather than one ever-growing file — `store/` and `backend/` are the worked examples.
Child files see the parent's private items, so a split costs no extra `pub`.

```
src/
├── lib.rs        Public API (Nidus, Scope); #![deny(unsafe_code)] (one scoped allow:
│                 the Mmap::map call in data/mmap.rs, §9); re-exports
├── config.rs     Config, Fsync, OpenMode, ann/quant/memory/persistence settings (§4.1)
├── model.rs      Shared type vocabulary: Value (+ its little-endian codec), Record,
│                 Predicate/Filter, Op, Distance, Quantization, AnnConfig,
│                 FtsQuery/FtsClause/FtsCombine (pure defs + serde)
├── glob/         minimal * ? [..] matcher (§7.1)
├── filter/       Filter/Predicate evaluation against a record's attrs: mod.rs
│                 (dispatch + per-query validate/prepare), text.rs (Levenshtein +
│                 the filter tokenizer, §7.4), pattern.rs (regex + compile cache, §7.5)
├── search/       distance kernels (cosine/dot/euclidean; f32/int8/binary Hamming) +
│                 bounded top-k heap + min_score; SearchOpts, Hit
├── data/         segment store: mod.rs (DataSegment — header, append, row accessor),
│                 segments.rs (Segments — the live segment set as one dense global row
│                 space; seal/rewrite over the manifest, §14), mmap.rs (the opt-in
│                 memory-mapped accessor for sealed segments, §9)
├── manifest/     the manifest object: live-segment set + pins, [crc32][bincode],
│                 atomic put = the seal/compaction commit point (§14.2)
├── log/          op-log codec (the WAL): len + payload + crc32, replay, torn-tail recovery
├── lock/         O_EXCL writer lock (pure std)
├── index_cache.rs  shared codec for derived caches (ann/fts): framed, CRC'd,
│                 validity-keyed; a stale/missing/torn load → rebuild (never fatal)
├── ann/          opt-in ANN index (Config::ann): hnsw.rs (graph) + ivf.rs (lists) +
│                 persist.rs (cache round-trip)
├── fts/          opt-in full-text (BM25) index: mod.rs + analyzer.rs (tokenize/stem/
│                 span) + fold.rs + schema.rs (FtsField) + highlight.rs (fragments, §7.8)
├── annotate.rs   opt-in result annotations: LegScore/ClauseScore/Highlight/Fragment
│                 + HighlightOpts (§7.8)
├── fuse.rs       Reciprocal Rank Fusion for hybrid search (§7.6)
├── cancel.rs     cooperative query cancellation: Cancel token + ambient scope
├── metrics.rs    in-process counters/gauges (pub mod metrics; scraped by the server)
├── diag.rs       levelled logfmt diagnostics to stderr (NIDUS_LOG)
├── backend/      pluggable storage & memory (§13): mod.rs (Persistence/Appender/
│                 MemoryTier/BackendLock traits + URL routing), local.rs (LocalFs +
│                 FileAppender), ram.rs (LocalRam + MemAppender), object.rs
│                 (ObjectAppender + object_try_lock), cloud.rs (shared ureq Http),
│                 s3.rs, aws_creds.rs, gcs.rs, redis.rs, tests.rs
└── store/        the integrator: mod.rs (Store type, open/in_memory ctors, lock +
                  ANN lifecycle glue), scoring.rs (scan kernels + parallel engine),
                  quant.rs (int8/binary state + quantized two-pass search), read.rs
                  (accessors, exact + ANN search incl. the exact-prefilter fallback),
                  text.rs (multi-clause BM25, hybrid fusion, annotations, §7.8),
                  rank.rs (recency decay + ORDER BY, §7.6), aggregate.rs (count/sum +
                  group_by + limit_per, §7.7), write.rs (upsert/delete/flush/compact),
                  memtier.rs (working-set publish/adopt), tests.rs

# ── AI ingest layer (feature-gated: embed-*/summarize-*/memory) ──
├── embed/        Embedder trait + per-provider adapters: mod.rs, voyage.rs, openai.rs,
│                 openai_compat.rs, ollama.rs, cohere.rs, gemini.rs, mistral.rs, jina.rs
├── summarize/    Summarizer trait + adapters: mod.rs, anthropic.rs, openai.rs, prompts.rs
├── memory.rs     Memory (remember/recall over a Nidus + an embedder); reserved
│                 nidus.* attr vocabulary, recency stamps, the TTL read guard (§9)
├── providers.rs  provider capability registry (Embed/Summarize)
├── http.rs       shared HTTP retry infrastructure for the ingest adapters

# ── `cli` feature only (the `nidus` binary, --features cli) ──
├── bin/nidus.rs  thin entry point: parse args → cli::run
├── cli/          clap subcommands over a store dir: mod.rs + backup.rs (snapshot)
└── server/       axum/tokio HTTP wrapper over one Nidus: mod.rs (routes + handlers),
                  dto.rs (wire types), auth.rs (bearer token), limits.rs (backpressure,
                  deadlines, body-idle timeout), commit.rs (group commit), metrics.rs
                  (Prometheus scrape + access log), mcp/ (the MCP 2026-07-28 adapter,
                  `mcp` feature: mod.rs, args.rs, remember.rs, search.rs, hygiene.rs,
                  admin.rs, stdio.rs)

tests/            file-backed integration (temp dirs; #[cfg_attr(miri, ignore)] on fsync
                  paths); tests/e2e/ drives the real binary (one test target, §11)
examples/         demo.rs — end-to-end smoke: open → upsert → search (single + All
                  scope); memory.rs — remember/recall over real providers (ingest features)
```

Errors propagate via `anyhow::Result` everywhere (`anyhow!`/`bail!`/`.context()`),
matching the common convention; no hand-rolled error enum.

Build order (bottom-up, each with tests, keeping `cargo build` in seconds):
`config → model → glob → filter → search → data → log → lock → index_cache →
ann/fts → backend → manifest → store → lib` (the `data` segment aggregator and `manifest`
sit over `backend`; the ingest layer and the `cli`/`server` binary layers sit above
`lib`, behind their features). The shared type vocabulary in `model` is frozen as
signatures first so the modules above can be implemented independently and still
compile together.

---

## 11. Testing strategy

- **Verify, never assume.** Every claim about behaviour — in this SPEC, in an issue,
  in a PR body — is held to what a test demonstrates against the real artifact, not
  to what the prose asserts. When a surface can only be proven by driving the real
  thing (a spawned binary, a real socket, an SDK against a live server), that
  end-to-end test is the load-bearing one and runs in CI; a unit tier that would
  pass while the contract is broken does not count as coverage.
- **Pure-logic tests run under Miri** (cosine, glob, filter, CRC, and the
  value/op-log codecs exercised against in-memory `Vec<u8>` buffers). These must
  never be `#[cfg_attr(miri, ignore)]`.
- **File-backed tests** (`open`, durability, recovery, compaction) use temp dirs.
  Mark with `#[cfg_attr(miri, ignore)]` only where they fsync / hit syscalls Miri
  lacks.
- **Crash-recovery tests**: hand-write a `log` with a truncated/CRC-broken tail
  record and assert `open` recovers to the prior good state and re-truncates the file.
- **Determinism**: same inserts → same on-disk bytes (modulo timestamps); same
  query → identical ranked ids and scores.

---

## 12. Integrating nidus into a host application

A consuming tool maps its own document type onto a nidus `Record` and (if it is
async) wraps `Nidus` in `Arc<Mutex<Nidus>>` + `spawn_blocking`. nidus knows nothing
about the application's domain — it stores `id + vector + attrs` and ranks them.

A typical semantic-index document maps cleanly, e.g.:

| Application concept | nidus `Record` |
|---|---|
| stable document id | `id` |
| embedding | `vector` |
| the embedded text / summary | `attrs["text"] = Str` |
| source locator (path, URI) | `attrs["path"] = Str` |
| chunk type ("file"/"section"/…) | `attrs["kind"] = Str` |
| name / title | `Str` or absent |
| line/section range | `Int` or absent |
| language / mime / labels | `Str` / `List` or absent |
| content hash (change detection) | `Str` or absent |
| optional edges/relations | `List` (present) · `Null` (computed-empty) · absent (un-indexed) |

The application's notion of a namespace → a nidus collection; any per-namespace
bookkeeping (a sync high-water mark, the embedding model id) → the collection's
string `meta` map; query options (top-k, path scoping, type/language filters,
min-score) → `SearchOpts` with `Glob`/`Eq`/`In` predicates.

The host owns the **store location**: it maps its own configured path (e.g. a
`store.<name>.path` setting or a user flag) and any durability/lock preferences into
a `Config` (§4.1) and calls `Nidus::open(config)`. A search-only process opens with
`OpenMode::ReadOnly`. nidus contributes no path defaults of its own.

---

## 13. Storage & memory backends (built)

§5 describes one configuration: a local directory, with vectors held in local process
RAM. This section generalizes that along **two independent axes**, behind two small sync
traits — and **both axes are now built end to end** (`src/backend/`).

- The **trait surface + local backends + live-store wiring** (`Persistence` + `Appender` +
  `BackendLock` and `MemoryTier`, `LocalFs`/`LocalRam`, URL-scheme selection): nidus-870.2
  Phase 1 + nidus-vnu — the store's `data`/`log` segments run over `Persistence::appender`,
  its `ann`/`fts` caches over `get`/`put`/`try_lock`; snapshot/backup is routed through
  `Persistence` too (§13.7).
- The **S3 + GCS persistence backends** (`s3.rs`/`gcs.rs`: whole-object get/put/delete/list
  over sans-IO `rusty-s3` / `tame-gcs`+`tame-oauth` + `ureq`): nidus-870.4. A store also
  **runs live** on them via an `ObjectAppender` (in-RAM segment buffer, whole-object rewrite
  on sync) plus a race-free object lock — nidus-cgr, made race-free in nidus-a7c via atomic
  create-if-absent (S3 `If-None-Match: *`, GCS `ifGenerationMatch=0`); a backend without that
  primitive falls back to the original advisory get-then-put.
- The **shared Redis memory tier** (`redis.rs`: one blocking `redis-rs` client over the RESP
  family — Redis/Valkey/KeyDB/DragonflyDB — plain or TLS), with the store publishing the
  serialized working set on `flush` and adopting it on `open`: nidus-870.3. Memcached is
  intentionally **not** built (eviction-only, the weakest cache fit).

- **Persistence** — where the durable *source-of-truth* bytes (`data`/`log`) live:
  **local files** (default), **S3**, or **GCS**. Optimized for durability and cost.
- **Memory tier** — where the in-RAM *working set* is held for serving: **local process
  RAM** (default), or a shared external store — **Redis / Valkey / KeyDB / DragonflyDB**.
  Optimized for fast access and sharing.

The axes are orthogonal and compose (e.g. truth in S3, working set shared via Valkey, scan
in local RAM); both default to local.

### 13.1 Two axes, and why search is independent of both

nidus searches with **CPU SIMD over a local, contiguous `Vec<f32>`** (exact cosine, the
ANN walk, the quantized scan, BM25). You cannot run that over bytes on a socket, so
**neither axis is ever in the query hot path**:

- The **persistence** tier is the durable truth, read on `open` and synced on write.
- The **memory** tier is a **shared, rebuildable cache of the in-RAM working set**, always
  materialized into local RAM *before* a scan — never searched over the wire.

So Redis/Valkey/Memcached are **not** persistence (S3 is cheaper and durable for that) and
**not** a way to search data that never enters local RAM. They are the shared-memory
**model (a)**: the working set lives in the external store so many stateless workers share
one copy and a cold start skips the rebuild — but each worker still loads it into its own
RAM to serve. Neither axis makes nidus larger-than-RAM — that is what the opt-in `Config::mmap`
mode is for (§9 / §14.6 phase 3): immutable segments mapped from disk, paged in on touch.

### 13.2 Persistence backends (the durable source of truth)

A persisted store is not intrinsically a directory; it is a small, fixed set of **named
byte objects** in two **classes**:

| object | class | reconstructable? | local discipline (§5/§6) |
|---|---|---|---|
| `data` | **source of truth** | no | append + fsync |
| `log` | **source of truth** | no | append + fsync |
| `ann` | derived cache | yes (from data/log) | atomic temp+fsync+rename, best-effort load |
| `fts` | derived cache | yes (from data/log) | atomic temp+fsync+rename, best-effort load |
| `data.crc` / `seg-NNNNNNNN.crc` | integrity sidecar | yes, but never rebuilt automatically | atomic whole-object write, stamped once at seal/compaction |

- **Source of truth** (`data`, `log`) must be durable and is append-shaped — small,
  incremental streams.
- **Derived caches** (`ann`, `fts`, the in-RAM quant matrices) are *reconstructable* from
  data/log; a missing/stale/corrupt cache is never fatal (§9, `index_cache.rs`). A backend
  may persist them or **drop and rebuild on open**.
- **Integrity sidecars** (`data.crc`, one per sealed `seg-NNNNNNNN.crc`) sit in a third
  class: unlike `ann`/`fts`, a *mismatch* is never silently discarded and rebuilt — doing
  that over already-corrupted row bytes would launder the corruption into a fresh,
  valid-looking checksum. A missing/stale one is only "unverified," never fatal, and is the
  one derived object `nidus check` (§13.7) reads rather than rebuilds on demand.

Only `data`+`log` must be shipped durably; the large, append-hostile HNSW graph is exactly
the artifact a backend is free to discard. Members:

| backend | crate (pure-Rust, sans-IO / blocking) | append? | auth | TLS |
|---|---|---|---|---|
| Local FS (default) | `std::fs` | native (fsync) | — | — |
| Amazon S3 (+ R2/MinIO) | `rusty-s3` (sans-IO) + blocking HTTP | none (whole-object) | Sigv4 = HMAC-SHA256 (pure RustCrypto) | yes (§13.6) |
| Google Cloud Storage | `tame-gcs` (sans-IO) + blocking HTTP | none (whole-object) | OAuth2 svc-account = RSA-JWT | yes (§13.6) |

S3 and GCS are **different APIs needing different clients** — not one "cloud" backend.
`rusty-s3`/`tame-gcs` are sans-IO (build/sign requests, parse responses; bring your own
transport), so the SDK layer itself adds no async and no TLS.

### 13.3 Memory tier (the shared working set)

Where the in-RAM working state (the vectors + the derived indexes) is held so it can be
**shared across processes** and **reloaded without a rebuild**:

| tier | crate | role |
|---|---|---|
| Local process RAM (default) | — (`Vec<f32>`) | the working set *is* the process heap; nothing shared |
| Redis / Valkey / KeyDB / Dragonfly | `redis` (blocking; plain TCP or TLS) | shared, rebuildable cache of the working set across workers (**built**) |
| Memcached | — | **not built** — eviction-only, no durability/structures; the weakest cache fit |

It is **model (a)** throughout: the external store is a *shared cache* of the serialized
in-RAM state (the same framing `index_cache.rs` writes locally, §9 — just shared/remote). It
is **rebuildable from the persistence tier**, so an empty or evicted store is never fatal —
exactly the derived-cache contract. Wins: N stateless search workers share one copy instead
of each rebuilding; restart is a load, not a rebuild.

**What is shared** is the replay-derived index (per-collection `id → (row, attrs)`, dead-row
count, FTS schemas) — the one in-RAM structure with no other cache, reusing the
`index_cache.rs` frame/decode codec. It is **watermark-guarded** (the log byte offset + data
row count): a store adopts the blob on `open` only when it matches the just-opened
`data`/`log` exactly, else replays the log; it publishes a fresh blob on `flush`.

Pure-Rust and sync: one `redis-rs` blocking client speaks RESP, so it covers Redis and its
wire-compatible kin — **Valkey, KeyDB, DragonflyDB** — selected by URL scheme. Plain
`redis://` needs **no TLS**; `rediss://` reuses the same rustls + `ring` as S3/GCS.

### 13.4 The traits — object-granular, sync (both axes)

Two small traits, one per axis. The persistence trait is the **common denominator** of
local files / S3 / GCS — whole-object put/get/list/delete plus an **optional append**
capability (local native; object stores emulate). The memory trait is load/store of the
shared working-set blob:

```rust
// sketch — not final
pub trait Persistence: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;     // whole object
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()>;    // atomic whole-object write
    fn delete(&self, key: &str) -> Result<()>;
    fn list(&self) -> Result<Vec<String>>;
    fn appender(&self, key: &str) -> Option<Box<dyn Appender>> { None } // local native; cloud None
    fn try_create_exclusive(&self, key: &str, bytes: &[u8]) -> Result<Option<bool>> { Ok(None) }
                                  // atomic create-if-absent (S3 If-None-Match / GCS ifGenerationMatch=0)
    fn try_lock(&self, key: &str, ttl: Duration) -> Result<Option<Lock>>; // O_EXCL (local native lock)
}

pub trait MemoryTier: Send + Sync {        // local RAM is the trivial impl
    fn load(&self, key: &str) -> Result<Option<Vec<u8>>>;     // pull the shared working set
    fn store(&self, key: &str, bytes: &[u8], ttl: Option<Duration>) -> Result<()>;
}
```

**Sync, deliberately** (consistent with §6.5):

1. Search is CPU-over-RAM and never touches either backend (§13.1) — no query IO to make async.
2. Every backend *can* be sync: sans-IO S3/GCS + a blocking HTTP client; `redis-rs`
   blocking; local files and local RAM are sync.
3. A sync trait is `dyn`-safe out of the box → genuine runtime plug-and-play. Async trait
   methods are not `dyn`-safe without `async-trait` (boxed futures) — worse plug-and-play.
4. An async core has an enormous blast radius against the §6.5 sync design, for no payoff
   on the non-bottleneck path.

> Decision: a backend whose only client is async owns a small runtime and `block_on`s
> **internally** — quarantined inside that one impl; the traits and core stay sync.
> (`block_on` panics inside a caller's async context; the `spawn_blocking` integration
> (§12) avoids that.)

Selection is by **URL scheme** — persistence: `file://`, `s3://`, `gs://`; memory: local
(default), `redis://`/`valkey://`/`keydb://`/`dragonfly://` (plain) and `rediss://`/`valkeys://` (TLS).

### 13.5 Effect on speed (independent of both axes)

**Search is backend-independent.** Every path — exact cosine, ANN walk (HNSW/IVF),
int8/binary two-pass, BM25, hybrid RRF — runs over **in-RAM** structures, so query
*results and latency are identical* regardless of persistence or memory tier. What
changes is only `open`/cold-start and writes:

| op | local FS + local RAM | network persistence (S3/GCS) | shared memory tier (Redis/Valkey) |
|---|---|---|---|
| search | RAM | identical — RAM | identical — RAM |
| `open` / cold start | bulk read + (maybe) rebuild caches | download `data`/`log`, rebuild caches | **load prebuilt working set** (skip rebuild) |
| write durability | fsync (~0.1–2 ms) | round-trip (~20–100 ms+); append → whole-object rewrite **O(store)** or segments | n/a (truth is the persistence tier) |

So the persistence tradeoff is **cheaper near-incremental writes** (ship the small
`data`/`log` delta; drop+rebuild `ann`/`fts`) for a **costlier `open`**; the memory tier's
payoff is the opposite end — a **fast, shared cold start** (load the working set instead of
rebuilding it).

> Neither axis makes nidus larger-than-RAM. The working set still lives in each process's
> RAM to be scanned; the tiers only change *where the durable bytes live* and *whether the
> warm working set is shared*. Larger-than-RAM is a separate axis — the opt-in `Config::mmap`
> mode, which maps immutable segments from disk (§9 / §14.6 phase 3).

### 13.6 Persistence build-time / TLS decision (S3/GCS only) — RESOLVED

This concerns **only the S3/GCS persistence backends** — they need HTTPS, hence a TLS
stack. The memory tier (Redis/Valkey/KeyDB/Dragonfly) is plain TCP or TLS, pure-Rust, and **unaffected**.
Resolved dependency trees confirm the wall and the one escape:

- `rustls` defaults to `aws-lc-sys` (C); its `ring` provider compiles C+asm; `reqwest`'s
  rustls path pulls `ring`; `google-cloud-storage` pulls `ring` **and** OpenSSL.
- A **pure** path exists: `rustls` + the `rustls-rustcrypto` provider + a hand-built
  `hyper`/`tokio-rustls` transport resolve with **no C** — but `rustls-rustcrypto` is
  **unaudited**. S3 auth (Sigv4) is pure HMAC (`rusty-s3` is clean); GCS auth
  (`tame-oauth`) pulls `ring` for RSA-JWT signing and would need replacing with
  RustCrypto's `rsa`.

The two candidates were **(A) pragmatic** — allow an audited C TLS dep, feature-gated —
and **(B) purist** — hold zero-FFI via `rustls-rustcrypto` + hand-rolled auth.

> **Decision: (A) pragmatic, NOT purist — and NOT feature-gated.** All backends compile
> into the **default build**; there is no per-backend or `cloud` feature flag. The
> reason is the whole point of a pluggable backend: a user who *outgrows local FS and
> needs S3* should switch with a **one-line runtime change** (`file://…` → `s3://…`),
> never a `Cargo.toml`-edit-and-recompile event that first confronts them with a new
> toolchain requirement at the worst moment. Compiling everything in makes the upgrade
> path frictionless for *every* consumer.
>
> The cost — accepted deliberately (§1): `ring` enters the default tree, so a C
> toolchain is required for all consumers (even local-only), and Miri now covers only
> *our* logic (§1.4). The thesis bar is *build speed*, not zero-C: the enemy is the
> *multi-minute* C tree, not `ring`'s small one.
>
> - **Forbidden** (slow compile): `rustls`'s default `aws-lc-sys`, **vendored** OpenSSL,
>   and the `reqwest`+`tokio`+`hyper` stack.
> - **Chosen** (fast compile): the **sans-IO clients** `rusty-s3`/`tame-gcs` + a
>   **lightweight blocking HTTP client** (`ureq`) with TLS via `ring`. Measured: a clean
>   debug build of the full core + all four backends (sync Redis, no tokio/aws-lc/OpenSSL
>   in the tree) is **~7.5 s** — ~8× under the one-minute budget.
>
> `(B)` purist (`rustls-rustcrypto`) is rejected: it would keep no-C + Miri, but at the
> cost of bespoke **unaudited** crypto on a credentialed path — a worse trade. Note this
> is *not* a capacity upgrade: S3/GCS change where bytes durably live, not the
> RAM-bound search model (§13.4) — "outgrow" means durability/sharing, not larger-than-RAM.

The memory tier over plain TCP (`redis://`, `valkey://`) needs no TLS; only a TLS'd
persistence target (`s3://`, `gs://`, or `rediss://` if ever used) exercises `ring`. Local
FS needs nothing. **The standing guardrail: the whole crate's clean build stays under a
minute — CI asserts it (§9, the build-time gate).**

### 13.7 Persistence usage modes (both supported)

- **Live backing store (built).** A store's `data`/`log` live on the persistence backend;
  writes durably round-trip per §13.5. On an object store with no native append, each
  segment is an in-RAM buffer rewritten as one whole object on sync (`ObjectAppender`,
  `O(object)` per flush) under a race-free object lock — atomic create-if-absent (S3
  `If-None-Match: *`, GCS `ifGenerationMatch=0`, nidus-a7c), falling back to an advisory
  get-then-put on a backend lacking the primitive. Best for low-write-rate / dev /
  small-scale, single-writer use (nidus's positioning).
- **Snapshot / backup (built).** PUT/GET the whole store as one archive (the `cli`-feature
  `tar.gz`). This is *exactly* object-granular, so every persistence backend does it
  trivially. `nidus backup --out <loc>` reads the source store's `data`/`log` objects via
  its backend and PUTs the archive to the destination backend named by `<loc>` (a local
  path / `file://` today, `s3://` once that backend lands); `nidus restore --in <loc>` GETs
  it and PUTs the objects into the target store. The capture order (`data` then `log`) plus
  the lock-free reader rule (§6.2) keep a hot snapshot consistent without a writer lock.
- **Verify (built).** An archive is self-describing: `nidus-backup.json` carries a
  per-object `{name, bytes, crc32}` baseline computed over the exact archived bytes.
  `nidus verify -i <archive>` extracts to a scratch location (never a real store), checks
  that baseline, drives the gzip stream to EOF so its own trailer CRC32 actually fires, and
  reopens the extracted store read-only to confirm the expected dimension, distance,
  collections, and record count; `nidus backup --verify` runs the same check right after
  writing, reading the archive back from its destination rather than trusting the local
  write. Archives written before 0.57 predate the baseline; verify falls back to the
  structural check and reports `objects_checked: 0`. **What it covers, and what it always
  left out:** the archive-level CRC is a claim about *archives*. It has never said anything
  about a store that rots in place on local disk between backups — that is a separate gap,
  closed separately by the checksum sidecars below, checked by `nidus check`, not by
  `verify` growing a new mode.
- **Live checksum sidecars (built, nidus-160).** Closes the in-place gap named above: before
  this, `src/data/*.rs` validated only the 64-byte header's magic and version, never the
  vector row bytes, so a flipped byte on disk changed no row count and no header and the
  store opened clean and scored wrong forever. Now every **sealed** segment (`data` once
  something supersedes it as active, and every `seg-NNNNNNNN`) gets a per-segment sidecar
  object (`<segment>.crc`) stamped the instant it becomes immutable — at seal time
  (`Segments::seal`) and again at compaction, which restamps the rewritten base and drops
  the sidecars of the segments it collapses away. `nidus check` recomputes each sealed
  segment's checksum and compares it to its sidecar; a mismatch is reported and never
  silently recomputed-and-resaved (that would launder real corruption into a fresh,
  valid-looking checksum — see `src/data/checksum.rs`). **What still does not hold, by
  design:** a sidecar is stamped only when a segment goes immutable, not on every append, so
  rows written to the still-open active segment after the last seal/compaction are
  unverified rather than vouched-for-clean until the next seal covers them. `src/log/`'s
  deliberate tolerance of a CRC-bad *tail* record as a torn write (§6.1, correct crash
  recovery) is unchanged and out of scope for this check. This adds a new sidecar *object*
  per segment; it changes nothing about the `data` segment's own byte layout (§5.1, §9's
  "none changed the vector data layout" still holds). The backup archive's object set does
  not capture the sidecar either (`src/cli/backup.rs`'s `is_store_object` excludes it, the
  same way it excludes the `ann`/`fts` caches): a sidecar is read where it lives, by
  `nidus check` against a live store, not by `nidus verify`/`restore` against an archive.

---

## 14. Scaling the storage model: segments (Phases 1–5 built)

nidus's thesis is **ease and a local→cloud continuum** (§1): the same store and the same
API run on a laptop and, by changing a location string, on a shared object store. The
original engine was the simplest thing that satisfies that at local scale — one in-RAM
`data` matrix + one `log`, the whole working set loaded on `open` (§5), the whole object
rewritten on each cloud sync (§13.7). That monolith was also the root of every scaling
limit. This section describes the storage model that turns **scale into a quantity of one
architecture rather than a separate mode**. It evolves over the existing seams (the §9 mmap
seam, the append-only format), not a rewrite, and changes no public API (§4).

**Phases 1–5 are built** (the segment format + manifest + WAL→segment sealing; the
per-segment IVF / exhaustive-tail split; per-segment mmap; manifest-versioned reader refresh;
and cluster mode — §14.6), each additive over the same segment format. The manifest that
names those segments is the one versioned object: it went to v2 for the open profile.

### 14.1 Principle: the durable objects are the store; a process is a cache over them

The source of truth is a set of objects on the persistence backend (§13.2). A running nidus
is a **cache-and-serve layer** over those objects: it loads what it needs into local RAM
(later, mmap'd disk) to score, and never treats RAM as authoritative — RAM is reconstructable
from the objects, always.

This *is* the local→cloud continuum, expressed in the data layout. Local: the objects sit on
the local FS and the cache is your process RAM. Cloud: the objects sit on S3/GCS and the
cache is the node's RAM. Same architecture; the only difference is **where the bytes live and
how much is resident**. Scale is a quantity, not a mode — the property the rest of the design
exists to deliver.

### 14.2 Segments: the unit of everything

A **segment** is a small, **immutable**, self-contained chunk of records — its vectors, its
attrs, and its own optional index (IVF lists, FTS postings). Once written it is never
mutated: only created, merged, or dropped. The store becomes three things:

- a **write-ahead log** (the op-log, §5.2, evolved) holding in-flight, not-yet-segmented writes;
- a set of **immutable segments** (the bulk of the data);
- a tiny **manifest** naming the live segments (the atomic commit point — swapping it
  publishes a new state).

**The manifest is FORMAT VERSION 2** (nidus-141): besides the enforced pins (dimension,
distance — a mismatch on open is a hard error, never silently reconciled) it now also
carries an *advisory* `OpenProfile`: recorded defaults for `ann`, `quantization`,
`query_threads`, and `mmap`. These are plain defaults, not invariants — an explicit flag
or `Config` setter on a later open always overrides the recorded value for that knob, and
an absent profile (the pre-141 case) just means "use the built-in defaults," never an
error. `nidus configure` (or `Nidus::set_open_profile`) is what writes a profile.

**Compatibility is one-way.** A v2 decode accepts a v1 manifest and lifts it in place with
an empty profile, so upgrading nidus is transparent: every existing store keeps opening,
with no recorded defaults, exactly as before. The reverse does not hold: once anything
rewrites the manifest at v2, a pre-0.60 binary that opens it hard-fails with "manifest
format version 2 is not supported." That is an accepted, one-time cost of the upgrade, not
a bug: the alternative (silently downgrading a v2 manifest's profile away to stay
v1-readable) would make configuring a store next to a mixed fleet of binary versions look
like it worked and then silently stop applying.

Immutability is what unlocks everything else, because each scaling limit of the monolith
dissolves into a segment operation:

| Limit (monolith) | Resolution (segments) |
|---|---|
| whole dataset must fit in RAM | hold / mmap a *subset* of segments; the rest stay cold on the backend |
| cloud sync rewrites the whole object (`O(store)`, §13.7) | a write is a **new small segment object** — `O(write)`, append-only |
| no unit of distribution | a shard is a **set of segments** |
| ANN index hard to maintain over a mutable store | build an index **once per immutable segment**; never mutate it |
| compaction rewrites everything | **background merge** of small segments into bigger ones |

### 14.3 Brute-force is the tail, not the engine *(built)*

Exactness stays the default by making brute-force the strategy for the **small/recent**
slice rather than the whole store. The active (appendable) segment — the recent WAL tail —
and any small sealed segment are scored **exhaustively** (exact, zero build, zero parameters);
a large **immutable** segment carries an **IVF index** built once when it is sealed (and
rebuilt at compaction). So "exact vs approximate" stops being a global mode the user selects
— it is a per-segment property that follows size:

- a laptop store is a few small segments, all brute-forced → **100% recall, no knobs**;
- a large store is mostly indexed segments plus a brute-forced tail → fast, still exact on
  the fresh data, with the same code path.

The trigger is an opt-in size threshold, `Config::segment_index_min_rows` (default `None` →
**no segment is ever indexed**, so the local default stays 100%-recall exact). When set, a
sealed segment with at least that many rows is IVF-indexed; the active segment never is. This
keeps "automatic-by-size **or** opt-in" (§14.5) honest: segmentation alone (for incremental
cloud writes) stays exact; indexing the cold bulk is a separate, explicit choice. A search
**fans out** — brute-force the exhaustive tail, walk each cold segment's IVF for an
over-fetched candidate set — and merges both legs into one bounded top-k with an exact f32
rerank (the two row sets are disjoint, so no doc is double-counted). When a global `ann`
index (§9) is configured it already covers every row, so it takes precedence and per-segment
indexing stays off.

The segment index is **IVF (centroid/list), not the HNSW graph**: list-structured indexes
have far lower roundtrip count and write-amplification against object storage than a
pointer-chasing graph, and they rebuild cleanly per immutable segment. nidus reuses the
existing `ivf.rs` for the per-segment index.

*(Built with a deliberate seam: per-segment IVF indexes are **rebuilt on `open`** from the
immutable segments rather than cached to their own objects — IVF's k-means build is cheap.
Persisting per-segment indexes as cache objects, parallel/quantized per-segment walks, and a
background segment-merge step are additive follow-ups over this same format.)

### 14.4 Writes: log first, index asynchronously

A write appends to the WAL and is durable on fsync/PUT (§6.4) — and **immediately queryable**
via the exhaustive tail scan, before any index exists. Turning WAL records into a segment
(and building that segment's IVF index) happens **off the commit path** — lazily on
flush/compact, or a future background step — so write latency never waits on index build.
Batched/group commits amortize the per-batch fsync/PUT (already the §6.4 per-batch policy;
a segment is the natural object boundary for a batch).

### 14.5 What stays fixed (non-negotiables)

- **Exact-by-default, zero-config locally.** No tuning knob (`ef`/`nprobe`/…) is ever a
  precondition to getting answers; any index is automatic-by-size or opt-in, never required.
- **Near-zero `unsafe` in our code; clean build well under a minute; no heavy/native deps**
  (§1, §13.6). mmap is the one conscious FFI opt-in (§9) — `#![deny(unsafe_code)]` plus the
  single scoped `Mmap::map` site — now built and applied per segment (phase 3, opt-in, off by
  default). No other `unsafe` is permitted.
- **One embedding space per store; the §4 public API is unchanged.** This is an internal
  storage rearchitecture — `open`/`upsert`/`search`/`flush`/`compact` keep their signatures.

### 14.6 What it unlocks, and the phasing

- **Larger than one node's RAM:** hold/mmap a subset of segments; cold segments stay on the
  backend until touched.
- **Incremental cloud writes:** one new segment per batch — no whole-object rewrite (§13.7).
- **Cooperating instances (cluster):** *(built — `Config::cluster`)* the segments + WAL +
  manifest on a *shared* backend are the shared truth; instances are stateless caches over
  them. One writer appends segments and advances the manifest — holding a heartbeated lease
  evolved from the §6.3 writer lock (race-free over object stores, nidus-a7c), renewed
  op-driven and fencing a superseded writer; readers serve from their cached subset and
  **refresh when the manifest version advances** (which it does on every commit — the
  universal commit counter). Cluster mode is a *consequence* of this model — a shared backend
  plus a versioned manifest — not a parallel architecture. It requires a shared persistence
  backend **and** a shared memory tier; local FS / local RAM are single-node by definition and
  are rejected for cluster mode.

Each phase is additive over the format and shippable alone; the order front-loads the
single-node payoff before any distribution work:

1. **Segment format + manifest + WAL→segment.** *(built)* The monolith is gone: vectors live
   in an ordered set of segment objects presented as one dense **global row space**
   (so search/quant/ANN still address vectors by global row, unchanged); a `manifest` object
   names the live segments, the **last** being the active appendable one. The §6.2 reader
   rule generalizes from "ignore rows past size S" to "read the manifest, open the segments
   it names." See the on-disk details below.
2. **Per-segment IVF + exhaustive tail.** *(built)* The brute-force-tail / indexed-cold split,
   opt-in by size via `Config::segment_index_min_rows` (default off → fully exact). See §14.3.
3. **mmap per segment.** *(built)* >RAM on one node: with `Config::mmap`, each **immutable**
   segment is served from a read-only memory-map (the §9 mmap seam scoped to a segment) while
   the active segment stays in RAM; cold segments page in on touch. Local-FS + little-endian +
   sealed-segments only; off by default; results are identical to the RAM path. See §9.
4. **Manifest-versioned reader refresh.** *(built)* A lock-free `ReadOnly` reader adopts a
   separate writer's newer committed state in place via `Nidus::refresh()` — no reopen. It
   re-reads the manifest and, when the version advanced (a commit) or the `log` grew, moves to
   the newer state at one consistent point, swapping it in atomically (a failure leaves the prior
   snapshot serving). Returns whether newer state was adopted; a writer / in-memory store is a
   no-op. Two fast paths keep it cheap (nidus-bdg): when a **shared memory tier** holds a snapshot
   matching the new row-count + log watermark, the reader **adopts** it and skips the log replay
   (mirroring `open`); and when the segment **list** is unchanged (plain appends, no
   seal/compaction), it re-reads **only the active segment** object and reuses every immutable
   segment — avoiding the dominant cost (re-fetching the whole set) on an object store. A
   restructure (seal/compaction changes the list) takes the full re-open.
5. **Cluster mode.** *(built)* Cooperating instances over one **shared** backend, enabled by
   `Config::cluster` (rejected unless persistence is a shared object store **and** a shared
   memory tier is set — local FS / process RAM are single-node). One `ReadWrite` writer holds
   a **lease** (the §6.3 object lock evolved: it carries an owner token and is **renewed on
   every write batch** — op-driven, no background thread in the library — so an active writer
   keeps it while an idle one past the TTL can be taken over); the renewal **fences** a
   superseded writer at the start of each batch. `nidus serve` additionally renews on a timer at
   `lock_ttl/3` through a `Drop`-free `LeaseRenewer` that does not take the store lock, so an
   idle writer — or one inside a batch longer than the TTL — is not mistaken for a dead one now
   that standbys wait to take over (`--wait-for-lease`).

   Renewal is itself a **compare-and-swap** on the lease object, not a blind put (nidus-lp4.7).
   Get-then-put was a read-modify-write race: a peer reclaiming the lease between the holder's
   read and its write would be silently overwritten, leaving *two* instances each believing they
   held the writer handle. The conditional write turns that into a detected loss — and a lost CAS
   is re-read before concluding anything, so an instance racing *its own* other renewer (the
   op-driven one against the timer) is not fenced by mistake. Relatedly, the staleness verdict
   treats an age *equal* to the TTL as still-live: stamps are whole seconds, so a truncated age
   can overstate the real elapsed time by nearly a second, and a strict comparison would declare
   a live, renewing holder dead. Takeover therefore lands in `ttl..=ttl+1` seconds — availability
   pays a bounded cost so mutual exclusion does not.

   A renewal failure is classified, too: only the store naming a *different* owner is a lost
   lease (permanent — the instance latches `fenced` and reports NOT ready). A transient backend
   error fails the write in flight and nothing more, because permanently retiring a healthy
   writer over a dropped connection trades an outage for a blip.

   The narrower window the per-batch renew cannot cover — a writer that
   stalls past the lease TTL *mid-batch* while a replacement takes over — is closed by
   **compare-and-swap on every durable object write** (nidus-ahw): each cluster write of a
   segment/`log`/`manifest` object is conditional on the version the writer last saw (S3
   `If-Match`, GCS `ifGenerationMatch`; create-if-absent for a fresh object), so a superseded
   writer's write is *refused* — it fails cleanly rather than clobbering the peer's committed
   bytes. A backend without CAS degrades to the per-batch lease fence alone. Every commit advances
   the manifest version (the universal **commit counter**), so any number of `ReadOnly` readers
   pick up the writer's changes with a single manifest read via `refresh()` (phase 4). It is not a
   managed cluster — no coordinator, replication, or rebalancing; the object store plus the
   versioned manifest *are* the coordination. (Remaining hardening: verification against real
   S3/GCS buckets — the in-RAM/offline coverage exercises the logic and request construction.)

**Phase-1 on-disk model (built).** A store is `manifest` + N segment objects + `log` (the
WAL). Each segment carries the existing §5.1 header (magic/version/dim/distance) + f32 rows;
the first segment keeps the name `data` so a single-segment store is byte-compatible with the
pre-segment layout (and `peek_header`/snapshot/legacy readers keep resolving `data`). Sealed
segments mint monotonic `seg-NNNNNNNN` names. The `manifest` is a `[crc32][bincode]` object
holding the pinned dimension/distance, the ordered segment names, the next-id counter, and a
monotonic version; it is published with an atomic whole-object `put` — the **commit point**.
**Sealing** (`Config::segment_max_rows`, default off → a single-segment store identical to the
old monolith) rotates the active segment to immutable and starts a fresh one — no data is
moved — then publishes the new manifest; a crash before that publish leaves the prior manifest
in force and the fresh segment an ignored orphan. **Compaction** collapses every segment back
into one fresh `data` segment, republishes the manifest, and reclaims the now-unreferenced
objects. A store opened with no manifest (a pre-segment `data`+`log` store) is **transparently
migrated**: `data` becomes the base segment and a manifest is written on open (ReadWrite only —
a ReadOnly open reads through a synthesized in-RAM manifest and writes nothing).

**Consistency.** A reader always sees a single manifest version — the exact live segment set
at the version it loaded, never a torn mix — and (phase 4, `Nidus::refresh()`) moves to a
newer manifest atomically: it builds the new segment set + replayed index into locals and
swaps them in only once every fallible step succeeds, then drops segments no longer
referenced (a failure mid-refresh leaves the prior snapshot serving). This preserves the §6
crash-safety and lock-free-reader guarantees: a half-written or not-yet-named segment is
invisible until its manifest commit, exactly as a row past size `S` is today.
