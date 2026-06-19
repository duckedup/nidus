//! The integrator: in-RAM index + write/read glue + compaction. Composes
//! [`Segments`](crate::data::Segments) (the live segment set as one global row space),
//! [`OpLog`](crate::log::OpLog), the [`Manifest`](crate::manifest::Manifest), and an
//! optional [`WriteLock`](crate::lock::WriteLock). Contract: see the root `SPEC.md`
//! §3, §5–§8, §14.
//!
//! This module holds the [`Store`] type, its constructors (`open`/`in_memory*`), and
//! the ANN index lifecycle glue. The behaviour splits across child modules — each can
//! see `Store`'s private fields because they descend from this one:
//!
//! - [`scoring`] — the scan kernels and the parallel-scan engine.
//! - [`quant`]   — int8/binary state + the quantized two-pass search.
//! - [`read`]    — accessors, scan plumbing, exact + ANN search.
//! - [`write`]   — `upsert`/`delete`, `flush`, `compact`, collection lifecycle.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};

use crate::ann::{Ann, IvfIndex, Walk};
use crate::backend::{
    BackendLock, ClusterLease, MemoryTier, Persistence, appender_for, locked_error,
    object_try_lock, open_memory_tier, open_persistence,
};
use crate::config::{Config, OpenMode};
use crate::data::Segments;
use crate::fts::Fts;
use crate::log::OpLog;
use crate::manifest::{MANIFEST_KEY, Manifest};
use crate::model::{AnnConfig, Distance, Op};

mod memtier;
mod quant;
mod read;
mod scoring;
mod write;

#[cfg(test)]
mod tests;

use quant::Quant;

// ── In-RAM types ─────────────────────────────────────────────────────────────

/// The cached row-sorted scan order: `(row, collection, id)` for every live doc,
/// sorted by `row` (see [`Store::scan_order`]).
type ScanOrder = Vec<(u64, String, String)>;

/// One document's entry within a collection. `row` is `None` for a text-only doc (no
/// embedding, so no row in the data matrix); such docs are full-text/metadata only and
/// never enter the vector scan or ANN index.
///
/// Serializable so the whole index can be published to / loaded from a shared
/// [`MemoryTier`](crate::backend::MemoryTier) (the working-set snapshot, see [`memtier`]).
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

// ── Store ─────────────────────────────────────────────────────────────────────

/// The on-disk + in-RAM store backing [`Nidus`](crate::Nidus). Implementers choose
/// the internal layout (per-collection `id → (row, attrs)` maps, dead-row counter,
/// held lock, etc.) but must keep these signatures — `lib.rs` calls them verbatim.
pub struct Store {
    config: Config,
    data: Segments,
    log: OpLog,
    /// The persistence backend the store's objects (`data`/`log`/`ann`/`fts`/`lock`)
    /// live on — a [`LocalFs`](crate::backend::LocalFs) for a file-backed store, or an
    /// object store ([`S3`](crate::backend::S3)/[`Gcs`](crate::backend::Gcs)) for a live
    /// object-backed store. `None` for an in-memory store (no durable backing; the
    /// cache/lock paths short-circuit). Held as an `Arc` so an [`ObjectAppender`] can
    /// share the same backend handle to rewrite `data`/`log` whole-objects on sync.
    persistence: Option<Arc<dyn Persistence>>,
    /// The shared memory tier (SPEC §13.3), when a non-local one is configured. `None`
    /// means the working set is the process heap only (the default). When `Some`, the
    /// serialized working set is published on `flush` and adopted on `open` (skipping
    /// the log replay + index rebuild). A rebuildable cache: tier errors are never fatal.
    memory: Option<Box<dyn MemoryTier>>,
    /// Held for its `Drop` effect (releases the writer lock on close). `ReadOnly` stores
    /// and in-memory stores hold `None`.
    #[allow(dead_code)]
    lock: Option<Box<dyn BackendLock>>,
    /// The cluster writer **lease** (SPEC §14.6 phase 5), held in place of `lock` by a
    /// cluster-mode [`ReadWrite`](crate::OpenMode::ReadWrite) writer. Renewed before every
    /// write batch (op-driven, no background thread) and fences a superseded writer; released
    /// on drop. `None` outside cluster mode and for readers / in-memory stores.
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
    /// Per-segment IVF indexes, aligned by position with `data`'s segment set
    /// (`seg_indexes[i]` indexes segment `i`'s global row range). `None` = that segment is
    /// brute-forced — always the active (last) segment, plus any immutable segment below
    /// [`Config::segment_index_min_rows`](crate::Config::segment_index_min_rows). Empty
    /// when per-segment indexing is off (the default) or a global `ann` index is configured
    /// (which already covers every row). This is the "brute-force tail / indexed cold
    /// segments" split (SPEC §14.3): the index walk picks candidates from the cold segments,
    /// the exhaustive scan covers the fresh tail, and both feed one merged top-k.
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
    /// Reverse map physical-row → `(collection, id)`, maintained only when ANN is on,
    /// so an ANN candidate row resolves to its owning doc in O(1) for the post-filter
    /// and `Hit` build. It is a *hint*: every lookup is re-verified against the
    /// authoritative index (`docs[id].row == row`), so deletions and overwrites need no
    /// special invalidation — a stale entry simply fails the check and is skipped.
    /// Dense + append-only (rebuilt wholesale on `compact`, which renumbers rows).
    row_to_doc: Vec<Option<(String, String)>>,
    /// Row-sorted scan order over *all* live docs — `(row, collection, id)` sorted by
    /// `row` — so a whole-store scan reaches the data matrix in storage order without
    /// re-sorting on every query (nidus-dxt). Built lazily on the first whole-store
    /// `search`/`list` after a write and reused until the next write invalidates it
    /// (`None` = stale). Subset-only workloads never build it, so they pay nothing.
    /// Behind a `RwLock` because searches take `&self` and may run concurrently
    /// (the store is shared as `Arc<RwLock<Nidus>>`); writers hold `&mut self` and
    /// invalidate via `get_mut` (no lock contention). The duplicated id/collection
    /// strings cost ~one extra copy of the key set in RAM while the cache is live.
    scan_order: std::sync::RwLock<Option<ScanOrder>>,
    /// The committed `log` byte-length the in-RAM index currently reflects — the
    /// reader-refresh watermark (SPEC §14.6 phase 4). [`refresh`](Self::refresh) compares it
    /// (together with the manifest version) against the on-disk state to tell, cheaply,
    /// whether a separate writer has committed anything since this reader last loaded.
    /// Set after each replay; only a [`ReadOnly`](crate::OpenMode::ReadOnly) reader reads it
    /// back (a writer is itself the source of truth and never refreshes).
    loaded_log_offset: u64,
    /// The CAS token (S3 `ETag` / GCS generation) of the `manifest` object as this writer last
    /// wrote or read it — the compare-and-swap fence for the **commit point** (SPEC §14.6,
    /// nidus-ahw). A cluster writer publishes each manifest conditionally on this token, so a
    /// writer superseded mid-batch finds the token changed and fails its commit rather than
    /// making its stale segment set the truth. `None` outside cluster mode, for readers, and on
    /// a backend without CAS (the publish then degrades to a plain put, fenced only per-batch).
    manifest_cas: Option<String>,
}

impl Store {
    /// Open per `config`: acquire the writer lock (unless `ReadOnly`), open the
    /// data + log files, replay the log into the in-RAM index (ignoring `Upsert`s
    /// that reference rows beyond the data file — the lock-free reader rule, §6.2),
    /// and auto-compact if the dead-row ratio exceeds `config.auto_compact`.
    pub fn open(config: Config) -> Result<Store> {
        // 1. Open the persistence backend (SPEC §13.2). Empty location → local files
        //    under `config.path` (created if absent); `s3://…`/`gs://…` → a live
        //    object-store-backed store. Held as `Arc` so the object-store appenders below
        //    can share the same handle to rewrite whole objects on sync.
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

    /// Open over already-resolved backends — the body shared by [`open`](Self::open) and
    /// the backend-injection tests. `location` is only used in the "store is locked"
    /// message. The persistence backend may be local (native append + `O_EXCL` lock) or a
    /// whole-object store (an [`ObjectAppender`] per segment + the advisory object lock).
    pub(crate) fn open_with(
        config: Config,
        location: &str,
        persistence: Arc<dyn Persistence>,
        memory: Option<Box<dyn MemoryTier>>,
    ) -> Result<Store> {
        // 2b. Cluster mode (SPEC §14.6 phase 5) needs a *shared* backend: every cooperating
        //     instance must reach the same durable objects and the same working set. Local
        //     files / process RAM are single-node by definition, so reject them here — for
        //     readers and writers alike (all instances must agree on the mode).
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
        }

        // 3. Acquire the writer handle (ReadWrite only). In cluster mode this is a heartbeated
        //    lease (renewed per write batch, fences a superseded writer); otherwise the plain
        //    writer lock — native `O_EXCL` on local files, or the object lock on a whole-object
        //    store. Readers take neither.
        let (lock, lease) = if config.open_mode == OpenMode::ReadWrite {
            if config.cluster {
                let lease = ClusterLease::acquire(&persistence, "lock", config.lock_ttl)?
                    .ok_or_else(|| locked_error(location))?;
                (None, Some(lease))
            } else {
                (
                    Some(Self::acquire_lock(&persistence, location, config.lock_ttl)?),
                    None,
                )
            }
        } else {
            (None, None)
        };

        // 4. Read the manifest naming the live segments (SPEC §14.2). Absent → this is a
        //    fresh store or a legacy `data`+`log` store that predates the manifest; both
        //    synthesize a single-segment manifest over the base `data` object (transparent
        //    migration). The synthesized manifest is persisted below (ReadWrite only).
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

        // Compare-and-swap fencing applies to a cluster **writer**'s object writes (manifest,
        // segments, log): each durable rewrite is conditional on the version it last saw, so a
        // writer superseded mid-batch is fenced instead of clobbering a peer (SPEC §14.6,
        // nidus-ahw). Readers never write, so they take the plain path.
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

        // A store with no manifest on disk gets one written now — initializing a fresh
        // store and migrating a legacy one in the same step. ReadOnly stores never write
        // (lock-free readers stay strictly read-only); they read through the synthesized
        // manifest in RAM.
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

        // 10. Auto-compact for FTS tombstone pressure too. Text-only docs occupy no data
        //     rows, so their deletes/overwrites never raise `dead_rows` and the step-6
        //     check can't see them — a churning text-only collection would otherwise let
        //     dead postings grow without bound. `compact` rebuilds the index (dropping
        //     tombstones). Checked after the index is built, so the ratio is meaningful;
        //     if step 6 already compacted, the ratio is ~0 and this is a no-op.
        if let Some(threshold) = store.config.auto_compact
            && store.fts.tombstone_ratio() > threshold
        {
            store.compact()?;
            tier_stale = true;
        }

        // 11. Warm the shared memory tier for peers: if we built the index from the log
        //     (didn't adopt a tier snapshot) or a compaction above rewrote the store,
        //     publish the fresh working set so peers can adopt the current state instead
        //     of replaying. Best-effort — the tier is a rebuildable cache.
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
            config,
        };
        // Align `seg_indexes` to the (single, empty) segment so a later seal can update it
        // incrementally. No-op unless per-segment indexing is on.
        store.build_segment_indexes();
        Ok(store)
    }

    /// Adopt a separate writer's newer committed state into this lock-free
    /// [`ReadOnly`](crate::OpenMode::ReadOnly) reader **without a full reopen** (SPEC §14.6
    /// phase 4). Re-reads the `manifest` and, when it names a newer version or the `log` has
    /// grown, re-opens the segment set and replays the log into a fresh in-RAM index at a
    /// single consistent point — the §6.2 lock-free reader rule, advanced in place.
    ///
    /// Returns `Ok(true)` when newer state was adopted, `Ok(false)` when the reader was
    /// already current (the cheap common case — one small manifest read plus a `log` stat,
    /// no segment or index work) or the handle cannot refresh: a writer (already the source
    /// of truth, its in-RAM state never trailing the disk) or an in-memory store (no backend
    /// to track).
    ///
    /// **Atomic against a concurrent compaction.** The new segment set and replayed index are
    /// built into locals first, and every step a concurrent seal/compaction can break —
    /// re-opening the segments (one it may have already deleted) and re-reading the log — runs
    /// *before* any field of `self` is touched. So that class of failure leaves the reader
    /// serving its prior consistent snapshot, never a torn mix. The derived-index rebuilds run
    /// after the swap over already-consistent data (a cache miss there rebuilds in RAM rather
    /// than failing).
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

        // Cheap currency check. The manifest version advances on every seal/compaction (the
        // structural commits); every other write (upsert/delete/meta) appends to the `log`.
        // So the reader is current exactly when both are unchanged. `self.log.offset()` stats
        // the reader's existing log handle — live for plain appends (the writer extends the
        // same object). A compaction replaces the log object, which can leave that cached
        // length stale, but it also bumps the version, so we reload via the version check
        // regardless.
        let on_disk_log_len = self.log.offset()?;
        let changed =
            manifest.version != self.data.version() || on_disk_log_len != self.loaded_log_offset;
        if !changed {
            return Ok(false);
        }
        // What *kind* of change: a restructure (seal/compaction altered the segment list) needs a
        // full re-open; an unchanged list means only the active segment grew (the incremental fast
        // path). Keyed on the segment list, not the version — in cluster mode the version is the
        // commit counter and advances on every batch (nidus-bdg).
        let restructured = !self.data.segment_names_match(&manifest.segments);

        // Re-read the segment objects. The **incremental** fast path (nidus-bdg) handles the
        // common case — only the active segment grew (plain appends, manifest version unchanged)
        // — by re-reading just the active segment object and reusing every immutable segment
        // (they never change), which avoids re-fetching the whole set (the dominant cost on an
        // object store). A version change means a seal/compaction restructured the set, so re-open
        // it whole. Both build into locals first so a failure leaves the live snapshot untouched;
        // `replaced` carries the whole new set on the version-change path, else `None`.
        // Exactly one of these is populated: the whole new set (restructure) or the staged
        // active segment (incremental). Both are locals — `self.data` is not touched until the
        // final swap, so a failure above leaves the reader on its prior consistent snapshot.
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

        // Re-read the log (a fresh handle — the cached object may have been replaced by a
        // compaction) and rebuild the in-RAM index, bounded by the freshly-sized segments
        // (§6.2: ignore any `Upsert` past the row count we just observed). Prefer the shared
        // memory tier's snapshot when it is exactly current — adopting it skips the log replay
        // entirely (tier-aware refresh, mirroring the open path), else replay the ops.
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

    /// Acquire the writer lock: the backend's native `O_EXCL` lock (local files) or, on a
    /// whole-object store with no native lock, the object lock (race-free conditional-PUT
    /// where the backend supports it, advisory otherwise). Contention is a clear "store is
    /// locked" error in both cases.
    fn acquire_lock(
        persistence: &Arc<dyn Persistence>,
        location: &str,
        ttl: Duration,
    ) -> Result<Box<dyn BackendLock>> {
        let acquired = if persistence.has_native_lock() {
            persistence.try_lock("lock", ttl)?
        } else {
            object_try_lock(persistence, "lock", ttl)?
        };
        acquired.ok_or_else(|| locked_error(location))
    }

    /// Replay the decoded log `ops` into the in-RAM index — the source of truth when no
    /// shared working-set snapshot is adopted. Returns the collections, the dead-row
    /// count, and the FTS index with its declared schemas restored (postings are rebuilt
    /// later by [`load_or_build_fts`](Self::load_or_build_fts)). `Upsert`s referencing a
    /// row beyond the data file are ignored (the lock-free reader rule, §6.2).
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
                Op::SetFtsSchema { collection, fields } => {
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

    /// On `open`: load the ANN index from its `ann` cache if one is present and valid
    /// for this store's config, then incrementally insert any rows added since the
    /// cache was written (so a stale/partial cache still makes open cheap). With no
    /// valid cache, fall back to a full `rebuild_ann`. No-op when ANN is off.
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

    /// Write the ANN index to its `ann` cache file so the next `open` skips the
    /// rebuild. Out-of-band by design — call it explicitly (e.g. before shutdown) or
    /// let `compact` trigger it; it is *never* on the `upsert`/`flush` write path. A
    /// no-op when ANN is off, the store is in-memory or read-only, or nothing changed
    /// since the last persist.
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

    /// Persist the FTS index to the `fts` cache if dirty. The validity key is the
    /// declared schema + analyzer/BM25 params; the watermark is the current log offset,
    /// so on open the cache is adopted only when nothing has been written since (any
    /// later write → the offset differs → rebuild). Reuses the shared
    /// [`crate::index_cache`] codec.
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

    /// On `open`: adopt the `fts` cache when it is valid for the current schema **and**
    /// its watermark equals the current log offset (i.e. nothing was written after it
    /// was persisted — the clean-reopen fast path). Otherwise rebuild from the replayed
    /// docs. No-op when FTS is inactive.
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

    /// Per-segment IVF indexing is active when [`Config::segment_index_min_rows`] is set
    /// **and** no global `ann` index is configured. A global index already covers every
    /// row, so the per-segment split would be redundant — that path takes precedence and
    /// per-segment indexing stays off.
    fn seg_indexing_on(&self) -> bool {
        self.ann.is_none() && self.config.segment_index_min_rows.is_some()
    }

    /// The IVF tuning every per-segment index is built with — size-driven defaults
    /// (`n_lists = 0` → ~√rows). One tuning point keeps the knob a single concept.
    fn segment_ivf_config() -> AnnConfig {
        AnnConfig::ivf()
    }

    /// (Re)build the per-segment IVF indexes from scratch over the current segment set:
    /// an [`IvfIndex`] over each **immutable** segment holding at least
    /// `segment_index_min_rows` rows, `None` for the active segment and any smaller one.
    /// Refreshes the reverse map first so walked candidates resolve to docs. O(indexed
    /// rows) — runs on `open` and after `compact`; the cheaper incremental
    /// [`index_just_sealed`](Self::index_just_sealed) handles a single seal. No-op unless
    /// per-segment indexing is on.
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

    /// After a successful seal, index the just-sealed segment (now immutable, the
    /// second-to-last) if it meets the threshold and append a `None` slot for the fresh
    /// active segment — keeping `seg_indexes` aligned with the segment set without
    /// re-running k-means on the already-built segments. Falls back to a full
    /// [`build_segment_indexes`](Self::build_segment_indexes) if the alignment is somehow
    /// off. No-op unless per-segment indexing is on.
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

    /// Rebuild the full-text index from all live docs (used on `open` after replay and
    /// after `compact` renumbers). Clears the field indexes (keeping the declared
    /// schema), then re-indexes every doc of every FTS collection in a deterministic
    /// order (sorted collection, then sorted id) so docnums are reproducible. No-op when
    /// FTS is inactive.
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
