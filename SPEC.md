# nidus — specification

> _nidus_ (Latin, "nest") — a small place where things are kept safe. A pure-Rust
> embeddable vector store, leaning on the bird theme.

This document is the source of truth for nidus's design. It records not just
*what* we build but *why*, including the decisions we deliberately deferred.
`CLAUDE.md` is the agent-facing summary; this is the long form. Keep them in sync.

---

## 1. Purpose & motivation

nidus is the **local storage leg** for semantic-search and indexing tools: chunk
some source content → embed each chunk into a dense vector → store the vectors plus
metadata → answer "nearest neighbours to this query vector" fast, in-process, with
no hosted service. The source can be anything — code, documents, issues, wiki
pages — nidus does not care; it stores vectors and metadata and ranks them.

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

The workload is a **vector store, not a database**: no joins, no SQL, no analytics,
no larger-than-RAM scans (at the target scale). nidus is that store and nothing
more.

### Thesis (the product *is* the constraints)

The hard bar is **build-and-ship speed, not zero-C absolutism.** What disqualified
DuckDB and LanceDB was a *multi-minute* build (a large C/C++ tree, a whole SQL engine)
— not the mere presence of any C. nidus's bar is concrete and testable: **a clean build
stays under a minute** (it is ~seconds today).

1. **Pure-Rust-first, fast to build.** Prefer well-established pure-Rust crates
   (`anyhow`, `serde`/`bincode`, `crc32fast`, …). A C-compiling/native-linking dep is
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
}
```

- **Collections** are logical partitions (namespaces) identified by a `&str`. There
  are many; each is created/dropped independently; all share the store's single
  pinned dimension.
- **Dimension** is fixed for the life of the store, recorded in the `data` header.
  Reopening with a different dimension is a hard error. One store = one embedding
  model = one comparable vector space.
- `Value` is rich enough to hold any scalar/metadata a caller attaches. The
  `Null`-vs-absent distinction is meaningful and preserved across disk round-trips:
  it lets a caller tell apart "this field was not computed/indexed" (absent) from
  "computed, and the value is empty" (e.g. `List([])`) — a distinction that matters
  for things like optional graph edges or tags.

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

pub struct SearchOpts { pub top_k: usize, pub filter: Filter, pub min_score: Option<f32> }

// `collection` identifies the source namespace — required when a query spans more
// than one, and (id) is only unique within a collection.
pub struct Hit { pub collection: String, pub id: String, pub score: f32, pub attrs: BTreeMap<String, Value> } // no vector

pub struct Filter(pub Vec<Predicate>);   // AND of predicates
pub enum Predicate {
    Eq(String, Value),                   // attr == value
    Ne(String, Value),                   // attr present and != value
    Glob(String, String),                // attr (Str) matches glob pattern
    In(String, Vec<Value>),              // attr ∈ set
    NotIn(String, Vec<Value>),           // attr present and ∉ set
    Lt(String, Value),                   // attr <  value  (same-type, orderable)
    Le(String, Value),                   // attr <= value
    Gt(String, Value),                   // attr >  value
    Ge(String, Value),                   // attr >= value
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
}
```

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

`crc32` covers the payload (table-based, hand-rolled, `crc.rs`). Payload is a
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
- Strings/maps/attrs use the explicit little-endian codec in `value.rs`
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
- **Filters** (`Filter` = AND of `Predicate`s) are evaluated against `attrs` before
  scoring: `Eq` (typed equality), `Ne` (typed inequality), `Glob` (pattern match on a
  `Str` attr, §7.1), `In` / `NotIn` (set membership), and `Lt`/`Le`/`Gt`/`Ge` (ordered
  range comparison). This covers typical needs: path-prefix scoping (`Glob "path*"`),
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

`filter::matches` AND-combines predicates (empty filter matches everything); an
absent key fails **every** predicate — including the negative ones (`Ne`, `NotIn`)
and the range ones. Each predicate is a positive assertion about a *present*
attribute, so a record lacking the key is never a match (e.g. `Ne "status"
"archived"` does not match a record with no `status`). `Eq(key, Null)` likewise
requires the key to be present and equal to `Null` (absent ≠ `Null`, per §3).

---

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

Every seam here is purely additive over the format in §5 — **none changed the on-disk
byte layout.** Several were designed-for here and have since been built; the design
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
  starve the candidate set (the graph walk surfaces too few matching rows) — recall
  degrades silently there; an exact-prefilter path is the planned follow-up. Deletes
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
  single-threaded). It is written strictly **out-of-band** — on `compact` and the
  explicit `Nidus::persist_index()`, **never** on the `upsert`/`flush` hot path, so
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
  persisted as a `SetFtsSchema` op). nidus then maintains an in-RAM inverted index per
  `(collection, field)` and answers `text_search(FtsQuery)` by BM25, reusing the same
  `Hit`/`Filter`/scope/top-k machinery as vector search. `hybrid_search` fuses a vector
  and a BM25 query with **Reciprocal Rank Fusion** (rank-based, so the incomparable
  cosine/BM25 scales need no normalization). The analyzer is pure-Rust, zero-FFI
  (lowercase → Unicode tokenize → English stopwords → Porter stem) behind a `Language`
  enum (US English today; the seam is open for more). To support pure-text corpora,
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
  small/recent tail; an IVF index covers the cold bulk. **Phases 1–4 (the segment format +
  manifest + WAL→segment sealing, per-segment IVF, per-segment mmap, and manifest-versioned
  reader refresh) are built — see §14**; the remaining phase (cluster) is designed in §14 but
  not yet built.

---

## 10. Module layout

A module that has grown to span several distinct concerns is a *directory* of
sibling files (each owning one concern) with `mod.rs` holding the core type + glue,
rather than one ever-growing file — `store/` and `backend/` are the worked examples.
Child files see the parent's private items, so a split costs no extra `pub`.

```
src/
├── lib.rs        Public API (Nidus, Scope); #![forbid(unsafe_code)]; re-exports
├── config.rs     Config, Fsync, OpenMode, ann/quant/memory/persistence settings (§4.1)
├── model.rs      Shared type vocabulary: Value, Record, Predicate/Filter, Op,
│                 Distance, Quantization, AnnConfig, FtsQuery (pure defs + serde)
├── crc.rs        table CRC32 (zero-dep)
├── glob/         minimal * ? [..] matcher (§7.1)
├── filter/       Filter/Predicate evaluation against a record's attrs
├── search/       distance kernels (cosine/dot/euclidean; f32/int8/binary Hamming) +
│                 bounded top-k heap + min_score; SearchOpts, Hit
├── data/         segment store: mod.rs (DataSegment — header, append, row accessor;
│                 the mmap seam) + segments.rs (Segments — the live segment set as one
│                 dense global row space; seal/rewrite over the manifest, §14)
├── manifest/     the manifest object: live-segment set + pins, [crc32][bincode],
│                 atomic put = the seal/compaction commit point (§14.2)
├── log/          op-log codec (the WAL): len + payload + crc32, replay, torn-tail recovery
├── lock/         O_EXCL writer lock (pure std)
├── index_cache.rs  shared codec for derived caches (ann/fts): framed, CRC'd,
│                 validity-keyed; a stale/missing/torn load → rebuild (never fatal)
├── ann/          opt-in ANN index (Config::ann): hnsw.rs (graph) + ivf.rs (lists) +
│                 persist.rs (cache round-trip)
├── fts/          opt-in full-text (BM25) index: mod.rs + analyzer.rs (tokenize/stem)
├── backend/      pluggable storage & memory (§13): mod.rs (Persistence/Appender/
│                 MemoryTier/BackendLock traits + URL routing), local.rs (LocalFs +
│                 FileAppender), ram.rs (LocalRam + MemAppender), object.rs
│                 (ObjectAppender + object_try_lock), cloud.rs (shared ureq Http),
│                 s3.rs, gcs.rs, redis.rs, tests.rs
└── store/        the integrator: mod.rs (Store type, open/in_memory ctors, lock +
                  ANN lifecycle glue), scoring.rs (scan kernels + parallel engine),
                  quant.rs (int8/binary state + quantized two-pass search), read.rs
                  (accessors, exact + ANN search), write.rs (upsert/delete/flush/
                  compact), memtier.rs (working-set publish/adopt), tests.rs

# ── `cli` feature only (the `nidus` binary, --features cli) ──
├── bin/nidus.rs  thin entry point: parse args → cli::run
├── cli/          clap subcommands over a store dir: mod.rs + backup.rs (snapshot)
└── server/       axum/tokio HTTP wrapper over one Nidus: mod.rs + dto.rs (wire types)

tests/            file-backed integration (temp dirs; #[cfg_attr(miri, ignore)] on fsync paths)
examples/         demo.rs — end-to-end smoke: open → upsert → search (single + All scope)
```

Errors propagate via `anyhow::Result` everywhere (`anyhow!`/`bail!`/`.context()`),
matching the common convention; no hand-rolled error enum.

Build order (bottom-up, each with tests, keeping `cargo build` in seconds):
`config → crc → model → glob → filter → search → data → log → lock → index_cache →
ann/fts → backend → manifest → store → lib` (the `data` segment aggregator and `manifest`
sit over `backend`; the `cli`/`server` binary layers sit above `lib`,
behind the `cli` feature). The shared type vocabulary in `model` is frozen as
signatures first so the modules above can be implemented independently and still
compile together.

---

## 11. Testing strategy

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

- **Source of truth** (`data`, `log`) must be durable and is append-shaped — small,
  incremental streams.
- **Derived caches** (`ann`, `fts`, the in-RAM quant matrices) are *reconstructable* from
  data/log; a missing/stale/corrupt cache is never fatal (§9, `index_cache.rs`). A backend
  may persist them or **drop and rebuild on open**.

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
and cluster mode — §14.6), each additive over the same on-disk format.

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
   re-reads the manifest and, when the version advanced (a seal/compaction) or the `log` grew
   (an append/delete), re-opens the segment set and replays the log into a fresh in-RAM index
   at one consistent point, swapping it in atomically (a failure leaves the prior snapshot
   serving). Returns whether newer state was adopted; a writer / in-memory store is a no-op.
5. **Cluster mode.** *(built)* Cooperating instances over one **shared** backend, enabled by
   `Config::cluster` (rejected unless persistence is a shared object store **and** a shared
   memory tier is set — local FS / process RAM are single-node). One `ReadWrite` writer holds
   a **lease** (the §6.3 object lock evolved: it carries an owner token and is **renewed on
   every write batch** — op-driven, no background thread — so an active writer keeps it while
   an idle one past the TTL can be taken over); the renewal **fences** a superseded writer,
   which fails its next write rather than clobbering. Every commit advances the manifest
   version (the universal **commit counter**), so any number of `ReadOnly` readers pick up the
   writer's changes with a single manifest read via `refresh()` (phase 4). It is not a managed
   cluster — no coordinator, replication, or rebalancing; the object store plus the versioned
   manifest *are* the coordination.

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
