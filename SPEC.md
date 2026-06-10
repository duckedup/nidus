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

1. **Popular, pure-Rust dependencies only.** Lean on well-established pure-Rust
   crates where they earn their keep (e.g. `anyhow` for errors, `serde`/`bincode`
   for codecs, `crc32fast` for checksums) — but **never a crate that compiles C or
   links a native library** (no `*-sys`, no bundled C++). The bar is *build-and-ship*:
   fast compile, no C toolchain, no FFI. `just deps` should stay short and every
   crate in it pure Rust.
2. **Zero FFI, zero `unsafe` in our code.** `#![forbid(unsafe_code)]`. No `flock`,
   no `mmap`, no `extern "C"` written by us. (A dependency's internal `unsafe` is
   fine; ours is not.)
3. **No C to compile.** `cargo build` is rustc — seconds, no C toolchain.
4. **Fully Miri-checkable.** Our logic, including file IO, runs under Miri.

Pulling in a crate that compiles C, links native code, or adds an `unsafe` block to
*our* code is a change to *what nidus is*. File an issue and decide deliberately.

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
- Larger-than-RAM / memory-mapped operation — deferred seam (§9).
- Quantization — int8 scalar and binary (sign-bit) quantization have since shipped
  (§9, opt-in via `Config::quantization`).
- SQL, a query planner, transactions spanning multiple operations, multi-writer
  concurrency, networking, or replication.

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

A store is a **directory**. Three files:

```
<dir>/
  data    flat f32 matrix, append-only, never rewritten in place
  log     append-only op stream (the commit record)
  lock    writer-exclusion lock file (present only while a writer holds it)
```

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
torn vector, never a half-record. No read lock required.

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
  `compact` rebuild. ANN and quantization both replace the search path and are
  **mutually exclusive** (rejected at `open`); combining them (a quantized walk + f32
  rerank) is a deferred optimization. The index is in-RAM only — no `data`/`log`
  format change, rebuilt from the vectors on `open` and `compact`, extended in
  O(batch) on `upsert`.

### Still deferred (designed-for, not built)

- **mmap.** Replace the single "row `i` → `&[f32]`" accessor: index into a mapped
  region instead of the in-RAM `Vec<f32>`. Gains zero-copy load, cross-process page
  sharing, >RAM. Cost: FFI (`unsafe`) — would relax the zero-FFI thesis, so it is a
  conscious future choice, not a default.

---

## 10. Module layout

```
src/
├── lib.rs       Public API (Nidus, Scope); #![forbid(unsafe_code)]; re-exports
├── config.rs    Config, Fsync, OpenMode (§4.1)
├── value.rs     Value + little-endian encode/decode
├── record.rs    Record
├── filter.rs    Predicate / Filter + matching against attrs
├── glob.rs      minimal * ? [..] matcher (§7.1)
├── search.rs    cosine kernel + bounded top-k heap + min_score; SearchOpts, Hit
├── ann/         opt-in ANN index (Config::ann): hnsw.rs (graph) + ivf.rs (lists)
├── data.rs      flat f32 segment: header, append, row accessor (the mmap seam)
├── log.rs       op-log codec: len + payload + crc32, replay, torn-tail recovery
├── lock.rs      O_EXCL writer lock (pure std)
├── crc.rs       table CRC32 (zero-dep)
└── store.rs     in-RAM index, write/read glue, compaction
tests/           file-backed integration (temp dirs; #[cfg_attr(miri, ignore)] on fsync paths)
examples/        demo.rs — end-to-end smoke: open → upsert → search (single + All scope)
```

Errors propagate via `anyhow::Result` everywhere (`anyhow!`/`bail!`/`.context()`),
matching the common convention; no hand-rolled error enum.

Build order (bottom-up, each with tests, keeping `cargo build` in seconds):
`config → crc → value → record → glob → filter → search → data → log → lock → store → lib`.
The shared type vocabulary (`config`, `value`, `record`, `filter`, `search` types)
is frozen as signatures first so modules can be implemented independently and still
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
