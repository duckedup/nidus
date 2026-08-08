//! The integrator: in-RAM index + write/read glue + compaction, composing `Segments`,
//! `OpLog`, `Manifest`, and an optional `WriteLock`. Holds [`Store`] and its constructors;
//! behaviour splits across the child modules below. Contract: `SPEC.md` §3, §5–§8, §14.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};

use crate::ann::{Ann, IvfIndex, Walk};
use crate::backend::{
    BackendLock, ClusterLease, MemoryTier, Persistence, appender_for, locked_error,
    object_try_lock, open_memory_tier, open_persistence,
};
use crate::config::{Config, LeaseWait, OpenMode};
use crate::data::Segments;
use crate::fts::{Fts, FtsField};
use crate::log::OpLog;
use crate::manifest::{MANIFEST_KEY, Manifest};
use crate::model::{AnnConfig, ClusterStatus, Distance, Op, Role};

mod aggregate;
mod memtier;
mod quant;
mod rank;
mod read;
mod scoring;
mod text;
mod write;

#[cfg(test)]
mod tests;

use quant::Quant;
/// The HTTP layer's only reader — `server::classify` maps it to a `400`.
#[cfg(feature = "cli")]
pub(crate) use read::BAD_QUERY;

// ── In-RAM types ─────────────────────────────────────────────────────────────

/// The cached row-sorted scan order: `(row, collection, id)` for every live doc,
/// sorted by `row` (see [`Store::scan_order`]).
type ScanOrder = Vec<(u64, String, String)>;

/// One document's entry within a collection. `row` is `None` for a text-only doc, which stays
/// out of the vector scan and ANN index. Serializable so the index can be published to a
/// shared [`MemoryTier`](crate::backend::MemoryTier).
#[derive(serde::Serialize, serde::Deserialize)]
struct DocEntry {
    row: Option<u64>,
    attrs: BTreeMap<String, crate::model::Value>,
}

/// One logical namespace within the store.
#[derive(serde::Serialize, serde::Deserialize)]
struct Collection {
    meta: BTreeMap<String, String>,
    docs: HashMap<String, DocEntry>,
}

impl Collection {
    fn new() -> Self {
        Self {
            meta: BTreeMap::new(),
            docs: HashMap::new(),
        }
    }
}

/// Map a failed `try_reserve` into a clear out-of-memory error rather than letting
/// the global allocator abort the process. `count` is the number of elements the
/// reservation was for (units depend on the collection — vectors, rows, entries).
fn oom(what: &str, count: usize) -> anyhow::Error {
    anyhow!("out of memory reserving capacity for {count} {what}")
}

/// Monotonic clock base for the lock-free staleness stamp: `Instant` cannot live in an atomic,
/// and staleness must read without the store lock. NOT `SystemTime` — the wall clock can jump
/// backwards, making a reader look *younger* than it is.
fn mono_base() -> Instant {
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *BASE.get_or_init(Instant::now)
}

/// Milliseconds on the [`mono_base`] clock — the value stored in a staleness stamp.
fn mono_millis() -> u64 {
    mono_base().elapsed().as_millis() as u64
}

/// The facts a readiness probe needs, readable with no lock — every field is an atomic or
/// fixed at open. Busy is not unhealthy, so readiness must not observe busy-ness
/// (nidus-abx.3); liveness treats a *poisoned* lock as unhealthy, readiness a held one as fine.
#[derive(Clone)]
pub struct Readiness {
    /// Fixed once the store is open: promotion happens *during* open, so an instance's role
    /// never changes afterwards.
    role: Role,
    /// Shared with [`Store::fenced`] — latched by a failing write or a background lease
    /// renewal (nidus-lp4.7), so readiness sees a fencing the moment it is detected.
    fenced: Arc<std::sync::atomic::AtomicBool>,
    /// Shared staleness stamp, `None` for a writer or an in-memory store — neither can lag.
    last_verified: Option<Arc<std::sync::atomic::AtomicU64>>,
}

impl Readiness {
    /// What this instance is. Fixed at open.
    pub fn role(&self) -> Role {
        self.role
    }

    /// Whether this writer has been superseded and can never write again.
    pub fn fenced(&self) -> bool {
        self.fenced.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Seconds since this instance last verified itself current — `0` for a writer, whose
    /// own state *is* the current state.
    pub fn staleness_secs(&self) -> u64 {
        match &self.last_verified {
            None => 0,
            Some(stamp) => {
                let then = stamp.load(std::sync::atomic::Ordering::Acquire);
                mono_millis().saturating_sub(then) / 1000
            }
        }
    }
}

/// A pseudo-random duration in `0..=span`, spreading standby retry polls. Not a PRNG: it only
/// decorrelates instances that would wake in lockstep, and a poor draw costs one redundant
/// lock read.
fn jitter(span: Duration) -> Duration {
    if span.is_zero() {
        return Duration::ZERO;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos() as u64);
    Duration::from_nanos(nanos % (span.as_nanos().max(1) as u64))
}

// ── Store ─────────────────────────────────────────────────────────────────────

/// The on-disk + in-RAM store backing [`Nidus`](crate::Nidus). Implementers choose
/// the internal layout (per-collection `id → (row, attrs)` maps, dead-row counter,
/// held lock, etc.) but must keep these signatures — `lib.rs` calls them verbatim.
pub struct Store {
    config: Config,
    data: Segments,
    log: OpLog,
    /// Where the store's objects live: `LocalFs`, or `S3`/`Gcs` for an object-backed store.
    /// `None` in memory (the cache/lock paths short-circuit). `Arc` so an [`ObjectAppender`]
    /// shares the handle to rewrite whole objects on sync.
    persistence: Option<Arc<dyn Persistence>>,
    /// The shared memory tier (SPEC §13.3); `None` = process heap only. When set, the working
    /// set is published on `flush` and adopted on `open`, skipping replay + rebuild. A
    /// rebuildable cache, so tier errors are never fatal.
    memory: Option<Box<dyn MemoryTier>>,
    /// Held for its `Drop` effect (releases the writer lock on close). `ReadOnly` stores
    /// and in-memory stores hold `None`.
    #[allow(dead_code)]
    lock: Option<Box<dyn BackendLock>>,
    /// The cluster writer **lease** (SPEC §14.6 phase 5), held in place of `lock`. Renewed per
    /// write batch (op-driven, no background thread), fences a superseded writer, released on
    /// drop. `None` outside cluster mode and for readers.
    lease: Option<ClusterLease>,
    collections: HashMap<String, Collection>,
    /// Rows no longer referenced (deleted or overwritten), for compaction tracking.
    dead_rows: usize,
    /// Quantization state (None when quantization is off — the f32 brute-force default).
    quant: Option<Quant>,
    /// Approximate-nearest-neighbour index (None when ANN is off — the exact default).
    /// May coexist with `quant`: the index walk then scores `quant`'s codes and the
    /// f32 rerank in `search_ann` restores accuracy (nidus-ndu).
    ann: Option<Ann>,
    /// Per-segment IVF indexes, position-aligned with `data`'s segments. `None` = that segment
    /// is brute-forced (always the active one). Empty when per-segment indexing is off or a
    /// global `ann` covers every row. The brute-force-tail / indexed-cold split, SPEC §14.3.
    seg_indexes: Vec<Option<IvfIndex>>,
    /// The in-RAM ANN index has unpersisted changes (rows inserted since the last
    /// `persist_index`/load). Lets `persist_index` skip a redundant write and tracks
    /// whether the on-disk `ann` cache is current. Meaningless when ANN is off.
    ann_dirty: bool,
    /// Full-text (BM25) index, keyed per declared `(collection, field)`. Empty (inert)
    /// until a collection declares an FTS schema; loaded from the `fts` cache on open
    /// when current, else rebuilt from the live docs.
    fts: Fts,
    /// The in-RAM FTS index has changes not yet written to the `fts` cache (mirrors
    /// `ann_dirty`). Meaningless when FTS is inactive.
    fts_dirty: bool,
    /// True for in-memory stores (no backing directory) — they never persist the ANN
    /// cache. `open`ed (file-backed) stores set this false.
    in_memory: bool,
    /// Reverse map row → `(collection, id)`, ANN-only, so a candidate resolves to its doc in
    /// O(1). A *hint*: every lookup is re-verified against `docs[id].row`, so deletes and
    /// overwrites need no invalidation. Rebuilt wholesale on `compact`.
    row_to_doc: Vec<Option<(String, String)>>,
    /// Row-sorted scan order over all live docs, so a whole-store scan reads the matrix in
    /// storage order without re-sorting per query (nidus-dxt). Built lazily, `None` = stale.
    /// `RwLock` because searches take `&self` and run concurrently.
    scan_order: std::sync::RwLock<Option<ScanOrder>>,
    /// The committed `log` length the in-RAM index reflects — the reader-refresh watermark
    /// (SPEC §14.6 phase 4). [`refresh`](Self::refresh) compares it plus the manifest version
    /// to detect a separate writer's commits cheaply. Only a `ReadOnly` reader reads it back.
    loaded_log_offset: u64,
    /// CAS token (S3 `ETag` / GCS generation) of the `manifest` as last written — the fence for
    /// the commit point (SPEC §14.6, nidus-ahw). A writer superseded mid-batch finds it changed
    /// and fails its commit rather than making a stale segment set the truth.
    manifest_cas: Option<String>,
    /// Superseded and can never write again (SPEC §14.6) — latched, since it is permanent.
    /// Observable via `cluster_status` so a probe can pull the instance from rotation
    /// (nidus-lp4.1); `Arc<Atomic>` so a background `LeaseRenewer` shares it (nidus-lp4.7).
    fenced: Arc<std::sync::atomic::AtomicBool>,
    /// When this instance last *verified* it was current — open, or the last successful
    /// [`refresh`](Self::refresh), reset even when nothing was adopted (nidus-lp4.4). Millis not
    /// `Instant`, so a probe reads it without the store lock (nidus-abx.3).
    last_verified: Arc<std::sync::atomic::AtomicU64>,
    /// The host owns the durable barrier — group commit (nidus-xb9.1). Set only inside
    /// [`Nidus::deferred`](crate::Nidus::deferred), where mutations append without fsync so N
    /// share one barrier. It says who calls the barrier, not what the durability policy is.
    defer_barrier: bool,
    /// Appended bytes no barrier has covered, so `commit`/`flush` owes them an fsync (plus a
    /// commit-counter bump in cluster mode). Set by any mutation that did not sync itself and
    /// cleared by the covering barrier, which is what lets `commit` no-op when nothing is owed.
    pending_barrier: bool,
}

impl Store {
    /// Open per `config`: take the writer lock unless `ReadOnly`, open data + log, replay the
    /// log into the in-RAM index (ignoring `Upsert`s past the data file — the lock-free reader
    /// rule, §6.2), and auto-compact past `config.auto_compact`.
    pub fn open(config: Config) -> Result<Store> {
        // 1. Open the persistence backend (SPEC §13.2): empty location → local files under
        //    `config.path`, `s3://…`/`gs://…` → object store. `Arc` so the appenders below
        //    share the handle to rewrite whole objects on sync.
        let location = if config.persistence.is_empty() {
            config.path.to_string_lossy().into_owned()
        } else {
            config.persistence.clone()
        };
        let persistence: Arc<dyn Persistence> = open_persistence(&location)?.into();

        // 2. Open the optional shared memory tier (SPEC §13.3). Empty/`local`/`ram` →
        //    `None` (the working set is the process heap). A `redis://…`/`valkey://…`
        //    URL → a shared, rebuildable working-set cache.
        let memory = Self::open_memory(&config.memory)?;

        Self::open_with(config, &location, persistence, memory)
    }

    /// Open over already-resolved backends — shared by [`open`](Self::open) and the
    /// backend-injection tests. `location` only feeds the "store is locked" message. The backend
    /// may be local or a whole-object store.
    pub(crate) fn open_with(
        config: Config,
        location: &str,
        persistence: Arc<dyn Persistence>,
        memory: Option<Box<dyn MemoryTier>>,
    ) -> Result<Store> {
        // 2a. Backend-independent cross-field config invariants (e.g. quantization vs
        //     per-segment indexing) — reject before any IO so a bad config fails fast.
        config.validate()?;

        // 2b. Cluster mode (SPEC §14.6 phase 5) needs a *shared* backend, so local files and
        //     process RAM are rejected here — for readers and writers alike, since all
        //     instances must agree on the mode.
        if config.cluster {
            if persistence.has_native_lock() {
                bail!(
                    "cluster mode requires a shared object-store persistence backend \
                     (s3://… or gs://…); a local-filesystem store is single-node"
                );
            }
            if memory.is_none() {
                bail!(
                    "cluster mode requires a shared memory tier (e.g. redis://…); the \
                     process-local working set cannot be shared between instances"
                );
            }
            // Cluster safety IS the conditional write: without CAS the lease is advisory, two
            // instances can both believe they hold it, and a stalled writer waking up
            // superseded can clobber a peer's committed bytes (nidus-lp4.2).
            if !persistence.supports_cas() {
                bail!(
                    "cluster mode requires a persistence backend with compare-and-swap \
                     (S3 If-Match / GCS ifGenerationMatch); this backend has none, so the \
                     writer lease would be advisory only and a superseded writer could \
                     overwrite committed data"
                );
            }
        }

        // 3. Acquire the writer handle (ReadWrite only): a heartbeated lease in cluster mode,
        //    else the plain writer lock (`O_EXCL` locally, the object lock otherwise). Under
        //    `Config::lease_wait` a loser waits for promotion — see `await_writer_handle`.
        let (lock, lease) = if config.open_mode == OpenMode::ReadWrite {
            if config.cluster {
                let lease = Self::await_writer_handle(location, &config, || {
                    ClusterLease::acquire(&persistence, "lock", config.lock_ttl)
                })?;
                (None, Some(lease))
            } else {
                let lock = Self::await_writer_handle(location, &config, || {
                    if persistence.has_native_lock() {
                        persistence.try_lock("lock", config.lock_ttl)
                    } else {
                        object_try_lock(&persistence, "lock", config.lock_ttl)
                    }
                })?;
                (Some(lock), None)
            }
        } else {
            (None, None)
        };

        // 4. Read the manifest naming the live segments (SPEC §14.2). Absent → a fresh store or
        //    a legacy `data`+`log` one; both synthesize a single-segment manifest over the base
        //    `data` object, persisted below (ReadWrite only).
        let on_disk = Manifest::load(persistence.as_ref())?;
        let manifest = match &on_disk {
            Some(m) => {
                if m.dimension as usize != config.dimension {
                    bail!(
                        "store dimension mismatch: manifest has {}, requested {}",
                        m.dimension,
                        config.dimension
                    );
                }
                if m.distance != config.distance {
                    bail!(
                        "store distance metric mismatch: manifest has {:?}, requested {:?}",
                        m.distance,
                        config.distance
                    );
                }
                m.clone()
            }
            None => Manifest::fresh(config.dimension, config.distance),
        };

        // CAS fencing applies to a cluster *writer*'s object writes: each rewrite is conditional
        // on the version it last saw, so one superseded mid-batch is fenced rather than clobbering
        // a peer (SPEC §14.6, nidus-ahw). Readers never write.
        let cas = config.cluster && config.open_mode == OpenMode::ReadWrite;

        // 5. Open every segment the manifest names into one global row space. The cap is
        //    enforced before any segment loads into RAM (§6.6, generalized across segments).
        let data = Segments::open(
            persistence.clone(),
            &manifest,
            config.max_vector_bytes,
            config.mmap,
            cas,
        )?;

        // No manifest on disk → write one now, initializing a fresh store and migrating a legacy
        // one in the same step. ReadOnly never writes; it reads the synthesized manifest in RAM.
        if on_disk.is_none() && config.open_mode == OpenMode::ReadWrite {
            data.manifest().store(persistence.as_ref())?;
        }

        // Capture the manifest's CAS token for a cluster writer — the fence anchor for every
        // later conditional commit (nidus-ahw). Read whether we just wrote it or adopted an
        // existing one; `None` on a backend without CAS (publish then degrades to a plain put).
        let manifest_cas = if cas {
            persistence.get_cas(MANIFEST_KEY)?.and_then(|(_, t)| t)
        } else {
            None
        };

        // 6. Open the op log through the backend's appender (replaying torn tails). The
        //    decoded `ops` are the fallback source for building the in-RAM index.
        let log_ap = appender_for(&persistence, "log", cas)?;
        let (log, ops) = OpLog::open_with(log_ap)?;

        let row_count = data.row_count();

        // 6. Build the in-RAM index. Prefer the shared memory tier's serialized working
        //    set when it is exactly current (skipping the replay) — SPEC §13.3; otherwise
        //    replay the log ops, then publish the fresh working set so peers can adopt it.
        let watermark = log.offset()?;
        let key = memtier::working_set_key(&config);
        let adopted = memtier::try_adopt(memory.as_deref(), &key, row_count, watermark)?;
        let from_tier = adopted.is_some();
        let (collections, dead_rows, fts) = match adopted {
            Some(index) => index.into_parts(),
            None => Self::replay_ops(ops, row_count),
        };

        let quant = match config.quantization {
            Some(q) => Some(Quant::empty(q.kind, data.dimension(), config.distance)?),
            None => None,
        };
        let ann = config
            .ann
            .map(|a| Ann::empty(a, data.dimension(), config.distance));

        let mut store = Store {
            config,
            data,
            log,
            persistence: Some(persistence),
            memory,
            lock,
            lease,
            collections,
            dead_rows,
            quant,
            ann,
            seg_indexes: Vec::new(),
            ann_dirty: false,
            fts,
            fts_dirty: false,
            in_memory: false,
            row_to_doc: Vec::new(),
            scan_order: std::sync::RwLock::new(None),
            loaded_log_offset: watermark,
            manifest_cas,
            fenced: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            last_verified: Arc::new(std::sync::atomic::AtomicU64::new(mono_millis())),
            defer_barrier: false,
            pending_barrier: false,
        };

        // Whether the in-RAM index now differs from any tier snapshot — true if we built
        // it from the log, or if a compaction below rewrote `data`/`log` (new watermark).
        let mut tier_stale = !from_tier;

        // 6. Auto-compact if the dead-row ratio exceeds the threshold.
        if let Some(threshold) = store.config.auto_compact {
            let total_rows = store.data.row_count() as usize;
            let ratio = store.dead_rows as f32 / total_rows.max(1) as f32;
            if ratio > threshold {
                store.compact()?;
                tier_stale = true;
            }
        }

        // 7. Build the quantized matrix from the loaded vectors, if enabled.
        store.rebuild_quant();
        // 8. Load the ANN index from its cache (incrementally catching up any rows
        //    added since), or rebuild it from the vectors if there is no valid cache.
        store.load_or_build_ann()?;
        // 8b. Build the per-segment IVF indexes over the cold (immutable) segments, when
        //     per-segment indexing is on (SPEC §14.3). No-op for the default exact store
        //     and when a global ANN index is configured.
        store.build_segment_indexes();
        // 9. Load the FTS index from its `fts` cache when it is exactly current, else
        //    rebuild it from the replayed docs (the schema was restored during replay).
        //    A no-op when no collection declares FTS.
        store.load_or_build_fts()?;

        // 10. Auto-compact for FTS tombstone pressure too: text-only docs occupy no data rows, so
        //     step 6 cannot see their churn and dead postings would grow without bound. Checked
        //     after the index is built so the ratio is meaningful.
        if let Some(threshold) = store.config.auto_compact
            && store.fts.tombstone_ratio() > threshold
        {
            store.compact()?;
            tier_stale = true;
        }

        // 11. Warm the shared memory tier: if the index came from the log, or a compaction above
        //     rewrote the store, publish the working set so peers adopt it instead of replaying.
        //     Best-effort — the tier is a rebuildable cache.
        if tier_stale {
            store.publish_working_set();
        }

        Ok(store)
    }

    /// An in-memory store (no files, no lock). For tests.
    pub fn in_memory(dimension: usize) -> Result<Store> {
        Self::in_memory_with(dimension, Distance::default())
    }

    /// An in-memory store with a specific distance metric.
    pub fn in_memory_with(dimension: usize, distance: Distance) -> Result<Store> {
        Self::in_memory_cfg(
            Config::new("/dev/null/in-memory", dimension)
                .distance(distance)
                .open_mode(OpenMode::ReadWrite)
                .auto_compact(None),
        )
    }

    /// An in-memory store with full config control.
    pub fn in_memory_cfg(config: Config) -> Result<Store> {
        config.validate()?;
        let quant = match config.quantization {
            Some(q) => Some(Quant::empty(q.kind, config.dimension, config.distance)?),
            None => None,
        };
        let ann = config
            .ann
            .map(|a| Ann::empty(a, config.dimension, config.distance));
        let mut store = Store {
            data: Segments::in_memory_with(config.dimension, config.distance),
            log: OpLog::in_memory(),
            persistence: None,
            memory: None,
            lock: None,
            lease: None,
            collections: HashMap::new(),
            dead_rows: 0,
            quant,
            ann,
            seg_indexes: Vec::new(),
            ann_dirty: false,
            fts: Fts::default(),
            fts_dirty: false,
            in_memory: true,
            row_to_doc: Vec::new(),
            scan_order: std::sync::RwLock::new(None),
            loaded_log_offset: 0,
            manifest_cas: None,
            fenced: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            last_verified: Arc::new(std::sync::atomic::AtomicU64::new(mono_millis())),
            defer_barrier: false,
            pending_barrier: false,
            config,
        };
        // Align `seg_indexes` to the (single, empty) segment so a later seal can update it
        // incrementally. No-op unless per-segment indexing is on.
        store.build_segment_indexes();
        Ok(store)
    }

    /// Adopt a writer's newer committed state into this lock-free reader without a full reopen
    /// (SPEC §14.6 phase 4). `Ok(false)` when already current or unable to refresh. Atomic against
    /// compaction: built into locals first, so a failure leaves the prior snapshot intact.
    pub fn refresh(&mut self) -> Result<bool> {
        // Only a lock-free ReadOnly reader over a durable backend tracks a separate writer.
        // A writer holds the only mutating handle (the §6.3 lock excludes other writers), so
        // its in-RAM state already is the truth; an in-memory store has no backend at all.
        if self.in_memory || self.config.open_mode != OpenMode::ReadOnly {
            return Ok(false);
        }
        let Some(persistence) = self.persistence.clone() else {
            return Ok(false);
        };

        // Read the current manifest (synthesizing one for a legacy `data`+`log` store, as
        // `open` does). Its pins must still match — a store's dimension/metric never change
        // in place, so a mismatch here means the directory was swapped under us.
        let manifest = match Manifest::load(persistence.as_ref())? {
            Some(m) => {
                if m.dimension as usize != self.config.dimension {
                    bail!(
                        "store dimension mismatch on refresh: manifest has {}, store opened \
                         with {}",
                        m.dimension,
                        self.config.dimension
                    );
                }
                if m.distance != self.config.distance {
                    bail!(
                        "store distance metric mismatch on refresh: manifest has {:?}, store \
                         opened with {:?}",
                        m.distance,
                        self.config.distance
                    );
                }
                m
            }
            None => Manifest::fresh(self.config.dimension, self.config.distance),
        };

        // Cheap currency check: the version advances on every seal/compaction, every other write
        // appends to the `log`, so the reader is current exactly when both are unchanged. A
        // compaction can stale the cached log length, but it bumps the version too.
        let on_disk_log_len = self.log.offset()?;
        let changed =
            manifest.version != self.data.version() || on_disk_log_len != self.loaded_log_offset;
        if !changed {
            // Verified current: nothing new to adopt, which is just as much proof of
            // freshness as adopting would be. Reset the staleness clock.
            self.last_verified
                .store(mono_millis(), std::sync::atomic::Ordering::Release);
            return Ok(false);
        }
        // What kind of change: a restructured segment list needs a full re-open, an unchanged one
        // means only the active segment grew. Keyed on the list, not the version — in cluster mode
        // the version is the commit counter and advances every batch (nidus-bdg).
        let restructured = !self.data.segment_names_match(&manifest.segments);

        // Re-read the segments. The incremental path (nidus-bdg) re-reads only the active one and
        // reuses the immutable ones, avoiding the whole-set fetch that dominates object-store cost.
        // `self.data` is untouched until the final swap.
        let mut replaced: Option<Segments> = None;
        let mut pending = None;
        let row_count = if restructured {
            // The §6.6 cap is re-enforced before any segment loads into RAM.
            let data = Segments::open(
                persistence.clone(),
                &manifest,
                self.config.max_vector_bytes,
                self.config.mmap,
                false, // a reader never writes — plain appenders, no CAS fencing
            )?;
            let rows = data.row_count();
            replaced = Some(data);
            rows
        } else {
            let staged = self.data.reopen_active(self.config.max_vector_bytes)?;
            let rows = staged.row_count();
            pending = Some(staged);
            rows
        };

        // Re-read the log on a fresh handle (a compaction may have replaced the object) and
        // rebuild the index bounded by the freshly-sized segments (§6.2). Prefer the memory tier's
        // snapshot when exactly current, which skips the replay entirely.
        let (log, ops) = OpLog::open_with(appender_for(&persistence, "log", false)?)?;
        let watermark = log.offset()?;
        let key = memtier::working_set_key(&self.config);
        let (collections, dead_rows, fts) =
            match memtier::try_adopt(self.memory.as_deref(), &key, row_count, watermark)? {
                Some(index) => index.into_parts(),
                None => Self::replay_ops(ops, row_count),
            };

        // Every fallible load has succeeded — swap the new snapshot in atomically (the active
        // segment in place, or the whole set on a restructure), then rebuild the derived indexes
        // over it with the same builders `open` uses.
        match (replaced, pending) {
            (Some(data), _) => self.data = data,
            (None, Some(staged)) => self.data.install_active(staged, manifest.version),
            (None, None) => unreachable!("refresh staged neither a full set nor an active segment"),
        }
        self.log = log;
        self.collections = collections;
        self.dead_rows = dead_rows;
        self.fts = fts;
        self.loaded_log_offset = watermark;
        self.row_to_doc = Vec::new();
        self.invalidate_scan_order();

        self.rebuild_quant();
        self.load_or_build_ann()?;
        self.build_segment_indexes();
        self.load_or_build_fts()?;

        self.last_verified
            .store(mono_millis(), std::sync::atomic::Ordering::Release);
        Ok(true)
    }

    // ── Backend wiring helpers ───────────────────────────────────────────────────

    /// Open the configured shared memory tier (SPEC §13.3). Empty / `local` / `ram` →
    /// `None` (the working set is the process heap only, the default — no external tier,
    /// no publish/adopt overhead). Any other location → the resolved tier.
    fn open_memory(location: &str) -> Result<Option<Box<dyn MemoryTier>>> {
        match location {
            "" | "local" | "ram" => Ok(None),
            loc => Ok(Some(open_memory_tier(loc)?)),
        }
    }

    /// Who this instance is and how current it is (SPEC §14.6) — see [`ClusterStatus`]. In-RAM
    /// only, no IO: a readiness probe calls this every few seconds and must not touch the object
    /// store.
    pub fn cluster_status(&self) -> ClusterStatus {
        let writer = self.config.open_mode == OpenMode::ReadWrite;
        let role = if self.in_memory {
            Role::InMemory
        } else {
            match (self.config.cluster, writer) {
                (true, true) => Role::ClusterWriter,
                (true, false) => Role::ClusterReader,
                (false, true) => Role::Writer,
                (false, false) => Role::Reader,
            }
        };
        let fenced = self.fenced.load(std::sync::atomic::Ordering::Acquire);
        ClusterStatus {
            role,
            cluster: self.config.cluster,
            // A fenced writer no longer holds anything, whatever it once thought.
            holds_writer_handle: (self.lease.is_some() || self.lock.is_some()) && !fenced,
            fenced,
            lease_owner: self.lease.as_ref().map(|l| l.owner().to_string()),
            commit_version: self.data.version(),
            // A writer IS the current state, so it is never stale. Only a reader lags.
            staleness_secs: if writer {
                0
            } else {
                mono_millis().saturating_sub(
                    self.last_verified
                        .load(std::sync::atomic::Ordering::Acquire),
                ) / 1000
            },
        }
    }

    /// A lock-free handle to the facts a readiness probe needs — see [`Readiness`]. Taken once at
    /// open so a long write cannot make a healthy instance report unready (nidus-abx.3); it shares
    /// the store's atomics rather than snapshotting, so a mid-batch fencing shows on the next probe.
    pub fn readiness(&self) -> Readiness {
        let writer = self.config.open_mode == OpenMode::ReadWrite;
        Readiness {
            role: self.cluster_status().role,
            fenced: self.fenced.clone(),
            // A writer is its own source of truth and an in-memory store has no peer to lag
            // behind; only a reader over a durable backend can be stale.
            last_verified: (!writer && !self.in_memory).then(|| self.last_verified.clone()),
        }
    }

    /// A `Drop`-free renewal handle for the writer lease, independent of the store lock so a batch
    /// longer than `lock_ttl` cannot let a peer fence a merely-slow writer (nidus-lp4.3). NOT a
    /// lease clone: a cloned owning guard would delete the lease object on drop.
    pub fn lease_renewer(&self) -> Option<crate::backend::LeaseRenewer> {
        // Hand over a clone of *this store's* fenced latch, so a lease lost on a background
        // renewal is recorded where `cluster_status` — and therefore the readiness probe —
        // will see it, instead of only reaching stderr (nidus-lp4.7).
        self.lease.as_ref().map(|l| l.renewer(self.fenced.clone()))
    }

    /// Claim the writer handle, honouring [`Config::lease_wait`]. `Ok(None)` = a live holder owns
    /// it: an error under the default [`LeaseWait::Fail`], else retried so the process becomes a
    /// standby. Safe because `claim` is atomic, so retrying changes only how a loser responds.
    fn await_writer_handle<T>(
        location: &str,
        config: &Config,
        claim: impl Fn() -> Result<Option<T>>,
    ) -> Result<T> {
        // A stale handle cannot be reclaimed before its TTL lapses, so faster polling only spends
        // requests. An eighth of the TTL, clamped so a tiny one does not spin and a large one still
        // notices a clean release promptly.
        let base = (config.lock_ttl / 8).clamp(Duration::from_millis(250), Duration::from_secs(2));
        let deadline = match config.lease_wait {
            LeaseWait::Fail => {
                return claim()?.ok_or_else(|| locked_error(location));
            }
            LeaseWait::Timeout(limit) => Some(Instant::now() + limit),
            LeaseWait::Forever => None,
        };

        // Remembered so a wait that ends in repeated backend errors reports the real cause
        // rather than a misleading "store is locked".
        let mut last_error: Option<anyhow::Error> = None;
        loop {
            // A transient backend error must NOT end the wait: a standby may sit here for hours,
            // and treating the first dropped connection as fatal causes exactly the crash-loop
            // waiting exists to avoid. Only acquisition or the deadline ends the loop.
            match claim() {
                Ok(Some(handle)) => return Ok(handle),
                Ok(None) => {}
                Err(e) => {
                    crate::metrics::metrics().lease_wait_errors.inc();
                    crate::diag::diag!(
                        crate::diag::Level::Warn,
                        "lease",
                        "waiting for the writer handle — backend error, will retry",
                        "err" => format!("{e:#}"),
                    );
                    last_error = Some(e);
                }
            }
            if let Some(deadline) = deadline
                && Instant::now() >= deadline
            {
                // Surface the real cause when the wait ended in errors rather than
                // contention — "store is locked" would be a misleading diagnosis.
                return Err(last_error.unwrap_or_else(|| locked_error(location)));
            }
            // Jitter up to +25%: several standbys would otherwise wake together the
            // instant a TTL lapses and stampede the same lock object.
            std::thread::sleep(base + jitter(base / 4));
        }
    }

    /// Replay the decoded log `ops` into the in-RAM index — the source of truth when no shared
    /// snapshot is adopted. Returns collections, dead-row count, and the FTS index with schemas
    /// restored. `Upsert`s past the data file are ignored (the lock-free reader rule, §6.2).
    fn replay_ops(ops: Vec<Op>, row_count: u64) -> (HashMap<String, Collection>, usize, Fts) {
        let mut collections: HashMap<String, Collection> = HashMap::new();
        let mut dead_rows: usize = 0;
        let mut fts = Fts::default();

        for op in ops {
            match op {
                Op::CreateCollection { collection } => {
                    collections
                        .entry(collection)
                        .or_insert_with(Collection::new);
                }
                Op::DropCollection { collection } => {
                    if let Some(col) = collections.remove(&collection) {
                        // Only rowed docs leave a reclaimable data row behind.
                        dead_rows += col.docs.values().filter(|e| e.row.is_some()).count();
                    }
                }
                Op::SetMeta { collection, meta } => {
                    let col = collections
                        .entry(collection)
                        .or_insert_with(Collection::new);
                    col.meta = meta;
                }
                Op::Upsert {
                    collection,
                    id,
                    row,
                    attrs,
                } => {
                    // Ignore rows beyond the data file — lock-free reader rule (§6.2).
                    if row >= row_count {
                        continue;
                    }
                    let col = collections
                        .entry(collection)
                        .or_insert_with(Collection::new);
                    // Overwriting a *rowed* doc leaves its old row dead.
                    if let Some(old) = col.docs.insert(
                        id,
                        DocEntry {
                            row: Some(row),
                            attrs,
                        },
                    ) && old.row.is_some()
                    {
                        dead_rows += 1;
                    }
                }
                Op::UpsertText {
                    collection,
                    id,
                    attrs,
                } => {
                    let col = collections
                        .entry(collection)
                        .or_insert_with(Collection::new);
                    if let Some(old) = col.docs.insert(id, DocEntry { row: None, attrs })
                        && old.row.is_some()
                    {
                        dead_rows += 1;
                    }
                }
                Op::Delete { collection, id } => {
                    if let Some(col) = collections.get_mut(&collection)
                        && let Some(old) = col.docs.remove(&id)
                        && old.row.is_some()
                    {
                        dead_rows += 1;
                    }
                }
                // Legacy shape: a language per field, no BM25/analyzer params. Adopting the
                // defaults here is what makes a pre-nidus-m50.13 log open unchanged.
                Op::SetFtsSchema { collection, fields } => {
                    let fields: Vec<FtsField> = fields
                        .into_iter()
                        .map(|(field, lang)| FtsField::new(field).language(lang))
                        .collect();
                    collections
                        .entry(collection.clone())
                        .or_insert_with(Collection::new);
                    fts.set_schema(&collection, &fields);
                }
                Op::SetFtsFields { collection, fields } => {
                    // The collection exists implicitly (matches SetMeta leniency); the
                    // field indexes are (re)built from the live docs once replay finishes.
                    collections
                        .entry(collection.clone())
                        .or_insert_with(Collection::new);
                    fts.set_schema(&collection, &fields);
                }
            }
        }
        (collections, dead_rows, fts)
    }

    // ── ANN index lifecycle ─────────────────────────────────────────────────────

    /// Rebuild the `row → (collection, id)` reverse map from the live index and return
    /// the live physical rows. Sized to the physical row count; dead rows stay `None`.
    /// Shared by the ANN rebuild and the snapshot-load paths.
    fn rebuild_row_to_doc(&mut self) -> Vec<u64> {
        let mut row_to_doc: Vec<Option<(String, String)>> =
            vec![None; self.data.row_count() as usize];
        let mut live_rows: Vec<u64> = Vec::new();
        for (col_name, col) in &self.collections {
            for (id, entry) in &col.docs {
                // Text-only docs (row None) have no vector — they never enter the index.
                if let Some(row) = entry.row
                    && (row as usize) < row_to_doc.len()
                {
                    row_to_doc[row as usize] = Some((col_name.clone(), id.clone()));
                    live_rows.push(row);
                }
            }
        }
        self.row_to_doc = row_to_doc;
        live_rows
    }

    /// Rebuild the ANN index and its reverse map from *all* current live docs. O(N) —
    /// used after `compact` renumbers rows and when no valid cache exists on `open`.
    /// No-op when ANN is off. Marks the index dirty (the on-disk cache is now stale).
    fn rebuild_ann(&mut self) {
        if self.ann.is_none() {
            return;
        }
        let live_rows = self.rebuild_row_to_doc();
        let workers = self.config.query_threads;
        let walk = quant::ann_walk_for(self.quant.as_ref(), &self.data, self.config.distance);
        if let Some(ann) = self.ann.as_mut() {
            ann.build(&walk, &live_rows, workers);
        }
        self.ann_dirty = true;
    }

    /// On `open`: load the ANN index from its `ann` cache when valid for this config, then insert
    /// rows added since — so even a stale cache keeps open cheap. Falls back to a full
    /// `rebuild_ann`; no-op when ANN is off.
    fn load_or_build_ann(&mut self) -> Result<()> {
        let Some(cfg) = self.config.ann else {
            return Ok(());
        };
        if self.ann.is_none() {
            return Ok(());
        }
        let dim = self.data.dimension();
        let distance = self.config.distance;
        let total = self.data.row_count();
        let quant = self.config.quantization.map(|q| q.kind);

        // Load the cache in its own scope so the immutable borrow of `persistence` ends
        // before we mutate `self` below. No backend (in-memory) → rebuild from vectors.
        let loaded = {
            let Some(p) = self.persistence.as_deref() else {
                self.rebuild_ann();
                return Ok(());
            };
            crate::ann::load_index(p, dim, distance, &cfg, quant)?
        };
        match loaded {
            // Valid cache that doesn't claim more rows than the data file holds (a
            // larger `covered` would mean dangling node→row refs — treat as stale).
            Some((ann, covered)) if covered <= total => {
                self.ann = Some(ann);
                self.rebuild_row_to_doc();
                if total > covered {
                    // Catch up rows appended after the cache was written.
                    let new_rows: Vec<u64> = (covered..total).collect();
                    let walk =
                        quant::ann_walk_for(self.quant.as_ref(), &self.data, self.config.distance);
                    if let Some(ann) = self.ann.as_mut() {
                        ann.insert_rows(&walk, &new_rows);
                    }
                    self.ann_dirty = true; // the delta isn't persisted yet
                } else {
                    self.ann_dirty = false; // on-disk cache is exactly current
                }
            }
            // No cache, or stale/corrupt/over-long → rebuild from the vectors.
            _ => self.rebuild_ann(),
        }
        Ok(())
    }

    /// Write the ANN index to its `ann` cache so the next `open` skips the rebuild. Out-of-band by
    /// design — called explicitly or by `compact`, *never* on the `upsert`/`flush` path. No-op when
    /// ANN is off, the store is in-memory or read-only, or nothing changed.
    pub fn persist_index(&mut self) -> Result<()> {
        // The on-disk caches are never written for an in-memory or read-only store.
        if self.in_memory || self.config.open_mode == OpenMode::ReadOnly {
            return Ok(());
        }
        self.persist_ann()?;
        self.persist_fts()?;
        Ok(())
    }

    /// Persist the ANN cache if dirty (gating shared via [`Self::persist_index`]).
    fn persist_ann(&mut self) -> Result<()> {
        let Some(cfg) = self.config.ann else {
            return Ok(());
        };
        if !self.ann_dirty {
            return Ok(());
        }
        let Some(ann) = self.ann.as_ref() else {
            return Ok(());
        };
        let Some(p) = self.persistence.as_deref() else {
            return Ok(());
        };
        crate::ann::save_index(
            p,
            ann,
            self.data.row_count(),
            self.data.dimension(),
            self.config.distance,
            &cfg,
            self.config.quantization.map(|q| q.kind),
        )?;
        self.ann_dirty = false;
        Ok(())
    }

    /// Persist the FTS index to the `fts` cache if dirty. Keyed on the declared schema plus
    /// analyzer/BM25 params, watermarked by the log offset, so open adopts it only when nothing has
    /// been written since. Reuses the [`crate::index_cache`] codec.
    fn persist_fts(&mut self) -> Result<()> {
        if !self.fts.is_active() || !self.fts_dirty {
            return Ok(());
        }
        let watermark = self.log.offset()?;
        let Some(p) = self.persistence.as_deref() else {
            return Ok(());
        };
        crate::index_cache::save(p, "fts", &self.fts.cache_key(), watermark, &self.fts)?;
        self.fts_dirty = false;
        Ok(())
    }

    /// On `open`: adopt the `fts` cache when valid for the current schema *and* its watermark
    /// matches the log offset — the clean-reopen fast path. Otherwise rebuild from the replayed
    /// docs; no-op when FTS is inactive.
    fn load_or_build_fts(&mut self) -> Result<()> {
        if !self.fts.is_active() {
            return Ok(());
        }
        let key = self.fts.cache_key();
        let current = self.log.offset()?;
        let loaded = {
            let Some(p) = self.persistence.as_deref() else {
                self.rebuild_fts();
                return Ok(());
            };
            crate::index_cache::load::<Fts>(p, "fts", &key)?
        };
        if let Some((cached, watermark)) = loaded
            && watermark == current
        {
            // The cache reflects the store exactly as it stands.
            self.fts = cached;
            self.fts_dirty = false;
            return Ok(());
        }
        // Absent, stale (schema/params changed), or the store changed since persist.
        self.rebuild_fts();
        Ok(())
    }

    /// Incrementally index the rows `upsert` just appended (`[prev_rows, row_count())`),
    /// all owned by `collection`, recording their owners in the reverse map — O(batch),
    /// not O(N). No-op when ANN is off. `new_owners` is `(row, id)` captured at commit.
    fn extend_ann(&mut self, collection: &str, prev_rows: u64, new_owners: &[(u64, String)]) {
        if self.ann.is_none() {
            return;
        }
        let total = self.data.row_count();
        if self.row_to_doc.len() < total as usize {
            self.row_to_doc.resize(total as usize, None);
        }
        for (row, id) in new_owners {
            self.row_to_doc[*row as usize] = Some((collection.to_string(), id.clone()));
        }
        let new_rows: Vec<u64> = (prev_rows..total).collect();
        let walk = quant::ann_walk_for(self.quant.as_ref(), &self.data, self.config.distance);
        if let Some(ann) = self.ann.as_mut() {
            ann.insert_rows(&walk, &new_rows);
        }
        self.ann_dirty = true;
    }

    // ── Per-segment index lifecycle (SPEC §14.3) ─────────────────────────────────

    /// Per-segment IVF indexing is active when [`Config::segment_index_min_rows`] is set *and* no
    /// global `ann` is configured — a global index already covers every row, so it takes precedence
    /// and the per-segment split stays off.
    fn seg_indexing_on(&self) -> bool {
        self.ann.is_none() && self.config.segment_index_min_rows.is_some()
    }

    /// The IVF tuning every per-segment index is built with — size-driven defaults
    /// (`n_lists = 0` → ~√rows). One tuning point keeps the knob a single concept.
    fn segment_ivf_config() -> AnnConfig {
        AnnConfig::ivf()
    }

    /// Rebuild the per-segment IVF indexes over the current segment set: one per immutable segment
    /// at or above `segment_index_min_rows`, `None` for the active and smaller ones. O(indexed
    /// rows), for `open` and `compact`; a single seal takes the cheaper `index_just_sealed`.
    fn build_segment_indexes(&mut self) {
        self.seg_indexes = Vec::new();
        if !self.seg_indexing_on() {
            return;
        }
        let min = self.config.segment_index_min_rows.unwrap();
        // The IVF walk resolves candidate rows through the reverse map; rebuild it so it
        // covers every live row of the segments we are about to index.
        self.rebuild_row_to_doc();
        let ranges = self.data.segment_ranges();
        let active = ranges.len() - 1;
        let cfg = Self::segment_ivf_config();
        let dim = self.data.dimension();
        let distance = self.config.distance;
        let workers = self.config.query_threads;
        let walk = Walk::exact(&self.data, distance);
        let mut indexes: Vec<Option<IvfIndex>> = Vec::with_capacity(ranges.len());
        for (i, &(base, rows)) in ranges.iter().enumerate() {
            if i != active && rows >= min {
                let mut ix = IvfIndex::new(cfg, dim, distance);
                let segment_rows: Vec<u64> = (base..base + rows).collect();
                ix.build(&walk, &segment_rows, workers);
                indexes.push(Some(ix));
            } else {
                indexes.push(None);
            }
        }
        self.seg_indexes = indexes;
    }

    /// After a seal, index the just-sealed segment if it meets the threshold and append a `None`
    /// slot for the fresh active one — keeping `seg_indexes` aligned without re-running k-means on
    /// segments already built. Falls back to `build_segment_indexes` if alignment is off.
    fn index_just_sealed(&mut self) {
        if !self.seg_indexing_on() {
            return;
        }
        let ranges = self.data.segment_ranges();
        let sealed = ranges.len() - 2; // the seal pushed a new active; this just froze.
        if self.seg_indexes.len() != sealed + 1 {
            // Not aligned to the pre-seal segment count — rebuild defensively.
            self.build_segment_indexes();
            return;
        }
        let min = self.config.segment_index_min_rows.unwrap();
        let (base, rows) = ranges[sealed];
        // The sealed segment's rows were active until now — make sure they resolve.
        self.rebuild_row_to_doc();
        let built = if rows >= min {
            let mut ix = IvfIndex::new(
                Self::segment_ivf_config(),
                self.data.dimension(),
                self.config.distance,
            );
            let walk = Walk::exact(&self.data, self.config.distance);
            let segment_rows: Vec<u64> = (base..base + rows).collect();
            ix.build(&walk, &segment_rows, self.config.query_threads);
            Some(ix)
        } else {
            None
        };
        self.seg_indexes[sealed] = built;
        self.seg_indexes.push(None);
    }

    // ── FTS index lifecycle ───────────────────────────────────────────────────────

    /// Rebuild the full-text index from all live docs — on `open` after replay, and after
    /// `compact` renumbers. Re-indexes in a deterministic order (sorted collection, then id) so
    /// docnums are reproducible. No-op when FTS is inactive.
    fn rebuild_fts(&mut self) {
        if !self.fts.is_active() {
            return;
        }
        self.fts.clear_indexes();
        let mut col_names: Vec<String> = self.collections.keys().cloned().collect();
        col_names.sort();
        for col_name in &col_names {
            if self.fts.schema_for(col_name).is_none() {
                continue;
            }
            let col = &self.collections[col_name];
            let mut ids: Vec<&String> = col.docs.keys().collect();
            ids.sort();
            for id in ids {
                let attrs = &col.docs[id].attrs;
                self.fts.index_doc(col_name, id, attrs);
            }
        }
        // The rebuilt index isn't on disk yet.
        self.fts_dirty = true;
    }
}
