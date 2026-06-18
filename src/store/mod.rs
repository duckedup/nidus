//! The integrator: in-RAM index + write/read glue + compaction. Composes
//! [`DataSegment`](crate::data::DataSegment), [`OpLog`](crate::log::OpLog), and an
//! optional [`WriteLock`](crate::lock::WriteLock). Contract: see the root `SPEC.md`
//! §3, §5–§8.
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

use crate::ann::Ann;
use crate::backend::{
    Appender, BackendLock, MemoryTier, ObjectAppender, Persistence, advisory_try_lock,
    locked_error, open_memory_tier, open_persistence,
};
use crate::config::{Config, OpenMode};
use crate::data::DataSegment;
use crate::fts::Fts;
use crate::log::OpLog;
use crate::model::{Distance, Op};

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
    data: DataSegment,
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
    collections: HashMap<String, Collection>,
    /// Rows no longer referenced (deleted or overwritten), for compaction tracking.
    dead_rows: usize,
    /// Quantization state (None when quantization is off — the f32 brute-force default).
    quant: Option<Quant>,
    /// Approximate-nearest-neighbour index (None when ANN is off — the exact default).
    /// May coexist with `quant`: the index walk then scores `quant`'s codes and the
    /// f32 rerank in `search_ann` restores accuracy (nidus-ndu).
    ann: Option<Ann>,
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
        // 3. Acquire the writer lock (ReadWrite only) — native `O_EXCL` on local files,
        //    or the advisory object lock on a whole-object store.
        let lock = if config.open_mode == OpenMode::ReadWrite {
            Some(Self::acquire_lock(&persistence, location, config.lock_ttl)?)
        } else {
            None
        };

        // 4. Open the data segment through the backend's appender (a native file handle
        //    on local FS; an in-RAM buffer rewritten whole on sync for an object store).
        //    First refuse — before allocating — to load a data file whose vectors already
        //    exceed the configured cap, turning a would-be allocation abort into a clear
        //    error.
        let data_ap = Self::appender_for(&persistence, "data")?;
        if let Some(cap) = config.max_vector_bytes {
            let vector_bytes = data_ap
                .len()?
                .saturating_sub(crate::data::HEADER_LEN as u64);
            if vector_bytes > cap {
                bail!(
                    "data file holds {vector_bytes} bytes of vectors, exceeding \
                     max_vector_bytes ({cap} bytes)"
                );
            }
        }
        let data = DataSegment::open_with(data_ap, config.dimension, config.distance)?;

        // 5. Open the op log through the backend's appender (replaying torn tails). The
        //    decoded `ops` are the fallback source for building the in-RAM index.
        let log_ap = Self::appender_for(&persistence, "log")?;
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
            collections,
            dead_rows,
            quant,
            ann,
            ann_dirty: false,
            fts,
            fts_dirty: false,
            in_memory: false,
            row_to_doc: Vec::new(),
            scan_order: std::sync::RwLock::new(None),
        };

        // 6. Auto-compact if the dead-row ratio exceeds the threshold.
        if let Some(threshold) = store.config.auto_compact {
            let total_rows = store.data.row_count() as usize;
            let ratio = store.dead_rows as f32 / total_rows.max(1) as f32;
            if ratio > threshold {
                store.compact()?;
            }
        }

        // 7. Build the quantized matrix from the loaded vectors, if enabled.
        store.rebuild_quant();
        // 8. Load the ANN index from its cache (incrementally catching up any rows
        //    added since), or rebuild it from the vectors if there is no valid cache.
        store.load_or_build_ann()?;
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
        }

        // 11. Warm the shared memory tier for peers: if we built the index from the log
        //     (didn't adopt a tier snapshot, or a compaction above changed it), publish
        //     the fresh working set. Best-effort — the tier is a rebuildable cache.
        if !from_tier {
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
        Ok(Store {
            data: DataSegment::in_memory_with(config.dimension, config.distance),
            log: OpLog::in_memory(),
            persistence: None,
            memory: None,
            lock: None,
            collections: HashMap::new(),
            dead_rows: 0,
            quant,
            ann,
            ann_dirty: false,
            fts: Fts::default(),
            fts_dirty: false,
            in_memory: true,
            row_to_doc: Vec::new(),
            scan_order: std::sync::RwLock::new(None),
            config,
        })
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
    /// whole-object store with no native lock, the advisory object lock. Contention is a
    /// clear "store is locked" error in both cases.
    fn acquire_lock(
        persistence: &Arc<dyn Persistence>,
        location: &str,
        ttl: Duration,
    ) -> Result<Box<dyn BackendLock>> {
        let acquired = if persistence.has_native_lock() {
            persistence.try_lock("lock", ttl)?
        } else {
            advisory_try_lock(persistence, "lock", ttl)?
        };
        acquired.ok_or_else(|| locked_error(location))
    }

    /// The `data`/`log` append handle for `key`: the backend's native [`Appender`] when
    /// it has one (local files), else an [`ObjectAppender`] — an in-RAM buffer rewritten
    /// as a whole object on sync — over the shared backend handle (object stores).
    fn appender_for(persistence: &Arc<dyn Persistence>, key: &str) -> Result<Box<dyn Appender>> {
        match persistence.appender(key)? {
            Some(native) => Ok(native),
            None => Ok(Box::new(ObjectAppender::open(persistence.clone(), key)?)),
        }
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
