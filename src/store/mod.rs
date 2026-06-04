//! The integrator: in-RAM index + write/read glue + compaction. Composes
//! [`DataSegment`](crate::data::DataSegment), [`OpLog`](crate::log::OpLog), and an
//! optional [`WriteLock`](crate::lock::WriteLock). Contract: see the root `SPEC.md`
//! §3, §5–§8.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context, Result, anyhow, bail};

use crate::config::{Config, Fsync, OpenMode};
use crate::data::DataSegment;
use crate::filter;
use crate::lock::WriteLock;
use crate::log::OpLog;
use crate::model::{Filter, Footprint, Hit, Op, Record, SearchOpts};
use crate::search::{TopK, dot, normalize};

// ── In-RAM types ─────────────────────────────────────────────────────────────

/// One document's entry within a collection.
struct DocEntry {
    row: u64,
    attrs: BTreeMap<String, crate::model::Value>,
}

/// One logical namespace within the store.
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
    /// Held for its `Drop` effect (removes the lock file on close). `ReadOnly` stores
    /// and in-memory stores hold `None`.
    #[allow(dead_code)]
    lock: Option<WriteLock>,
    collections: HashMap<String, Collection>,
    /// Rows no longer referenced (deleted or overwritten), for compaction tracking.
    dead_rows: usize,
}

impl Store {
    /// Open per `config`: acquire the writer lock (unless `ReadOnly`), open the
    /// data + log files, replay the log into the in-RAM index (ignoring `Upsert`s
    /// that reference rows beyond the data file — the lock-free reader rule, §6.2),
    /// and auto-compact if the dead-row ratio exceeds `config.auto_compact`.
    pub fn open(config: Config) -> Result<Store> {
        // 1. Create the store directory if needed.
        std::fs::create_dir_all(&config.path).map_err(|e| {
            anyhow::anyhow!("failed to create store directory {:?}: {}", config.path, e)
        })?;

        // 2. Acquire the writer lock (ReadWrite only).
        let lock = if config.open_mode == OpenMode::ReadWrite {
            Some(WriteLock::acquire(&config.path, config.lock_ttl)?)
        } else {
            None
        };

        // 3. Open the data segment. First refuse — before allocating — to load a
        //    data file whose vectors already exceed the configured cap, turning a
        //    would-be allocation abort into a clear error.
        let data_path = config.path.join("data");
        if let Some(cap) = config.max_vector_bytes {
            let on_disk = std::fs::metadata(&data_path).map(|m| m.len()).unwrap_or(0);
            let vector_bytes = on_disk.saturating_sub(crate::data::HEADER_LEN as u64);
            if vector_bytes > cap {
                bail!(
                    "data file holds {vector_bytes} bytes of vectors, exceeding \
                     max_vector_bytes ({cap} bytes)"
                );
            }
        }
        let data = DataSegment::open(&data_path, config.dimension)?;

        // 4. Open and replay the op log.
        let (log, ops) = OpLog::open(&config.path.join("log"))?;

        let row_count = data.row_count();

        // 5. Replay ops into the in-RAM index.
        let mut collections: HashMap<String, Collection> = HashMap::new();
        let mut dead_rows: usize = 0;

        for op in ops {
            match op {
                Op::CreateCollection { collection } => {
                    collections
                        .entry(collection)
                        .or_insert_with(Collection::new);
                }
                Op::DropCollection { collection } => {
                    if let Some(col) = collections.remove(&collection) {
                        dead_rows += col.docs.len();
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
                    // If overwriting an existing id, the old row becomes dead.
                    if col.docs.contains_key(&id) {
                        dead_rows += 1;
                    }
                    col.docs.insert(id, DocEntry { row, attrs });
                }
                Op::Delete { collection, id } => {
                    if let Some(col) = collections.get_mut(&collection)
                        && col.docs.remove(&id).is_some()
                    {
                        dead_rows += 1;
                    }
                }
            }
        }

        let mut store = Store {
            config,
            data,
            log,
            lock,
            collections,
            dead_rows,
        };

        // 6. Auto-compact if the dead-row ratio exceeds the threshold.
        if let Some(threshold) = store.config.auto_compact {
            let total_rows = store.data.row_count() as usize;
            let ratio = store.dead_rows as f32 / total_rows.max(1) as f32;
            if ratio > threshold {
                store.compact()?;
            }
        }

        Ok(store)
    }

    /// An in-memory store (no files, no lock). For tests.
    pub fn in_memory(dimension: usize) -> Result<Store> {
        // Use a dummy path — in-memory stores don't use the file system.
        // We still need a Config, but with in_memory there are no files.
        let config = Config::new("/dev/null/in-memory", dimension)
            .open_mode(OpenMode::ReadWrite)
            .auto_compact(None);

        Ok(Store {
            config,
            data: DataSegment::in_memory(dimension),
            log: OpLog::in_memory(),
            lock: None,
            collections: HashMap::new(),
            dead_rows: 0,
        })
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Reject mutations when in ReadOnly mode.
    fn check_writable(&self) -> Result<()> {
        if self.config.open_mode == OpenMode::ReadOnly {
            bail!("read-only store: mutations are not allowed");
        }
        Ok(())
    }

    /// Apply the fsync policy after a mutation: sync data then log under PerBatch.
    fn maybe_sync(&mut self) -> Result<()> {
        if self.config.fsync == Fsync::PerBatch {
            self.data.sync()?;
            self.log.sync()?;
        }
        Ok(())
    }

    // ── Public API ────────────────────────────────────────────────────────────

    pub fn dimension(&self) -> usize {
        self.data.dimension()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// A cheap snapshot of the store's vector footprint (see [`Footprint`]).
    pub fn footprint(&self) -> Footprint {
        let rows = self.data.row_count();
        let dimension = self.data.dimension();
        let doc_count = self.collections.values().map(|c| c.docs.len()).sum();
        Footprint {
            rows,
            dead_rows: self.dead_rows as u64,
            dimension,
            vector_bytes: rows * dimension as u64 * 4,
            doc_count,
        }
    }

    pub fn create_collection(&mut self, name: &str) -> Result<()> {
        self.check_writable()?;
        // Idempotent: only create if absent.
        if !self.collections.contains_key(name) {
            self.collections.insert(name.to_string(), Collection::new());
            self.log.append(&Op::CreateCollection {
                collection: name.to_string(),
            })?;
            self.maybe_sync()?;
        }
        Ok(())
    }

    pub fn drop_collection(&mut self, name: &str) -> Result<()> {
        self.check_writable()?;
        if let Some(col) = self.collections.remove(name) {
            self.dead_rows += col.docs.len();
            self.log.append(&Op::DropCollection {
                collection: name.to_string(),
            })?;
            self.maybe_sync()?;
        }
        Ok(())
    }

    pub fn has_collection(&self, name: &str) -> bool {
        self.collections.contains_key(name)
    }

    /// Returns collection names sorted alphabetically.
    pub fn collections(&self) -> Vec<String> {
        let mut names: Vec<String> = self.collections.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn get_meta(&self, collection: &str) -> BTreeMap<String, String> {
        self.collections
            .get(collection)
            .map(|c| c.meta.clone())
            .unwrap_or_default()
    }

    pub fn set_meta(&mut self, collection: &str, meta: BTreeMap<String, String>) -> Result<()> {
        self.check_writable()?;
        // Implicitly create collection if absent (matches replay leniency).
        let col = self
            .collections
            .entry(collection.to_string())
            .or_insert_with(Collection::new);
        col.meta = meta.clone();
        self.log.append(&Op::SetMeta {
            collection: collection.to_string(),
            meta,
        })?;
        self.maybe_sync()?;
        Ok(())
    }

    /// Upsert a batch. **All-or-nothing:** every fallible step (vector append,
    /// data fsync, log append, log fsync) rolls `data` and `log` back to the marks
    /// captured at entry on failure, then returns the original error — a failed
    /// batch (e.g. ENOSPC mid-write) leaves the store byte-identical to its
    /// pre-call state and never corrupts it. The in-RAM index is mutated only in
    /// the final, infallible commit phase, after both files are durable.
    pub fn upsert(&mut self, collection: &str, records: &[Record]) -> Result<usize> {
        self.check_writable()?;

        let dim = self.data.dimension();

        // Validate all vectors first (fail fast before any mutation).
        for rec in records {
            if rec.vector.len() != dim {
                bail!(
                    "vector length {} does not match store dimension {}",
                    rec.vector.len(),
                    dim
                );
            }
        }

        let need_create = !self.collections.contains_key(collection);

        // Empty batch: preserve the implicit-create contract, transactionally.
        if records.is_empty() {
            if need_create {
                self.log.append(&Op::CreateCollection {
                    collection: collection.to_string(),
                })?;
                self.maybe_sync()?;
                self.collections
                    .insert(collection.to_string(), Collection::new());
            }
            return Ok(0);
        }

        // Capacity gate: refuse — before any append — a batch that would grow the
        // vector matrix past the cap. Clean refusal, no rollback, store stays fully
        // usable for reads/search. (Counts physical rows incl. dead ones; compact()
        // reclaims headroom.)
        if let Some(cap) = self.config.max_vector_bytes {
            let projected =
                (self.data.row_count() + records.len() as u64) * self.data.dimension() as u64 * 4;
            if projected > cap {
                bail!(
                    "upsert would grow the vector matrix to {projected} bytes, exceeding \
                     max_vector_bytes ({cap} bytes); compact() can reclaim dead rows"
                );
            }
        }

        // Rollback marks: where data and log stood before this batch touched them.
        let data_mark = self.data.row_count();
        let log_mark = self.log.offset()?;

        // Phase 0: reserve every growable buffer up-front, fallibly, so the commit
        // phase (Phase 5) can never reallocate / OOM. Nothing is mutated here, so an
        // OOM just returns — no rollback needed (data + log untouched).
        let mut staged: Vec<(String, u64, BTreeMap<String, crate::model::Value>)> = Vec::new();
        staged
            .try_reserve_exact(records.len())
            .map_err(|_| oom("upsert staging entries", records.len()))?;
        // Index capacity: for a not-yet-created collection, build it locally with a
        // reserved docs map and stash it; for an existing one, grow its docs map now
        // (pure capacity — harmless if the batch later rolls back).
        let mut pending_collection: Option<Collection> = None;
        if need_create {
            self.collections
                .try_reserve(1)
                .map_err(|_| oom("collections map", 1))?;
            let mut col = Collection::new();
            col.docs
                .try_reserve(records.len())
                .map_err(|_| oom("collection docs map", records.len()))?;
            pending_collection = Some(col);
        } else {
            self.collections
                .get_mut(collection)
                .unwrap()
                .docs
                .try_reserve(records.len())
                .map_err(|_| oom("collection docs map", records.len()))?;
        }

        // Phase 1: append all vectors to data (SPEC §6.2 write order). Roll back on
        // any failure — nothing else has been touched yet.
        // NOTE: `rec.attrs.clone()` (BTreeMap) and `rec.id.clone()` (String) can
        // still abort on OOM — std offers no `try_reserve` for either. These are
        // small metadata next to the N×dim×4 vector matrix, which `data.append`
        // reserves fallibly; the `max_vector_bytes` cap guards the dominant memory.
        for rec in records {
            let mut v = rec.vector.clone();
            normalize(&mut v);
            match self.data.append(&v) {
                Ok(row) => staged.push((rec.id.clone(), row, rec.attrs.clone())),
                Err(e) => {
                    self.data
                        .truncate_to(data_mark)
                        .context("rollback data after failed append")?;
                    return Err(e);
                }
            }
        }

        // Phase 2: fsync data before writing log records.
        if let Err(e) = self.data.sync() {
            self.data
                .truncate_to(data_mark)
                .context("rollback data after failed sync")?;
            return Err(e);
        }

        // Phase 3: append log records (CreateCollection, if needed, then the
        // Upserts). On any failure, roll back both files to their marks.
        let log_ops = need_create
            .then(|| Op::CreateCollection {
                collection: collection.to_string(),
            })
            .into_iter()
            .chain(staged.iter().map(|(id, row, attrs)| Op::Upsert {
                collection: collection.to_string(),
                id: id.clone(),
                row: *row,
                attrs: attrs.clone(),
            }));
        for op in log_ops {
            if let Err(e) = self.log.append(&op) {
                self.rollback(data_mark, log_mark)?;
                return Err(e);
            }
        }

        // Phase 4: fsync log (or defer to flush()).
        if self.config.fsync == Fsync::PerBatch
            && let Err(e) = self.log.sync()
        {
            self.rollback(data_mark, log_mark)?;
            return Err(e);
        }

        // Phase 5: commit to the in-RAM index — infallible. Both files are durable,
        // and the maps' capacity was reserved in Phase 0, so no insert reallocates.
        if let Some(col) = pending_collection {
            self.collections.insert(collection.to_string(), col);
        }
        let col = self.collections.get_mut(collection).unwrap();
        let mut count = 0usize;
        for (id, row, attrs) in staged {
            if col.docs.contains_key(&id) {
                self.dead_rows += 1; // overwriting: the old row becomes dead
            }
            col.docs.insert(id, DocEntry { row, attrs });
            count += 1;
        }

        Ok(count)
    }

    /// Roll both append-only files back to the given marks (batch-rollback for a
    /// failed `upsert`). Surfaces a rollback failure rather than masking it.
    fn rollback(&mut self, data_mark: u64, log_mark: u64) -> Result<()> {
        self.log
            .truncate_to(log_mark)
            .context("rollback log after failed upsert")?;
        self.data
            .truncate_to(data_mark)
            .context("rollback data after failed upsert")?;
        Ok(())
    }

    pub fn delete(&mut self, collection: &str, ids: &[&str]) -> Result<usize> {
        self.check_writable()?;

        let Some(col) = self.collections.get_mut(collection) else {
            return Ok(0);
        };

        let mut count = 0usize;
        for &id in ids {
            if col.docs.remove(id).is_some() {
                self.dead_rows += 1;
                self.log.append(&Op::Delete {
                    collection: collection.to_string(),
                    id: id.to_string(),
                })?;
                count += 1;
            }
        }

        if count > 0 {
            self.maybe_sync()?;
        }

        Ok(count)
    }

    pub fn delete_where(&mut self, collection: &str, filter: &Filter) -> Result<usize> {
        self.check_writable()?;

        let Some(col) = self.collections.get(collection) else {
            return Ok(0);
        };

        // Collect matching ids first.
        let to_delete: Vec<String> = col
            .docs
            .iter()
            .filter(|(_, entry)| filter::matches(filter, &entry.attrs))
            .map(|(id, _)| id.clone())
            .collect();

        if to_delete.is_empty() {
            return Ok(0);
        }

        // Now delete them via the normal delete path.
        let refs: Vec<&str> = to_delete.iter().map(String::as_str).collect();
        self.delete(collection, &refs)
    }

    // NOTE: `get_all` materializes the whole collection (vector + attr clones) into
    // a fresh Vec and returns it directly, so it is not fallible — an OOM here can
    // still abort. Making it `Result` would break the public API for a bulk-read
    // convenience; hosts holding huge collections should prefer `search`/scoped
    // reads. The write and open paths (the exhaustion-critical ones) are fallible.
    pub fn get_all(&self, collection: &str) -> Vec<Record> {
        let Some(col) = self.collections.get(collection) else {
            return Vec::new();
        };

        col.docs
            .iter()
            .map(|(id, entry)| Record {
                id: id.clone(),
                vector: self.data.row(entry.row).to_vec(),
                attrs: entry.attrs.clone(),
            })
            .collect()
    }

    /// Brute-force cosine over the union of `collections`, merged into one ranking
    /// (one bounded top-k heap fed by every in-scope collection).
    pub fn search(
        &self,
        collections: &[&str],
        query: &[f32],
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        // Normalize the query once.
        let mut q = query.to_vec();
        normalize(&mut q);

        // Gather the in-scope, filter-passing rows as cheap borrowed tuples — no
        // string allocation on this path — then sort by physical row. The scoring
        // loop below then walks the data matrix in storage order, so the CPU
        // prefetcher streams the (row-major, contiguous) `f32` buffer instead of
        // chasing the arbitrary order a `HashMap` iterator hands back. At small
        // dimensions, where the dot is cheap relative to a row-boundary cache miss,
        // this is the difference between memory-bandwidth-bound and miss-bound.
        let mut scan: Vec<(u64, &str, &str)> = Vec::new();
        let scan_cap: usize = collections
            .iter()
            .filter_map(|c| self.collections.get(*c))
            .map(|c| c.docs.len())
            .sum();
        scan.try_reserve(scan_cap)
            .map_err(|_| oom("search scan buffer", scan_cap))?;
        for &col_name in collections {
            let Some(col) = self.collections.get(col_name) else {
                // Named collection that does not exist — skip silently.
                continue;
            };
            for (id, entry) in &col.docs {
                // Apply filter before scoring.
                if !filter::matches(&opts.filter, &entry.attrs) {
                    continue;
                }
                scan.push((entry.row, col_name, id.as_str()));
            }
        }
        scan.sort_unstable_by_key(|&(row, _, _)| row);

        // The heap carries only *borrowed* identifiers (`&str` into `self`/`collections`)
        // during the scan — offering one is a cheap pointer copy, so no allocation
        // happens on the hot per-row path. Strings (and attr clones) are materialized
        // once, at the end, for the ≤ top_k survivors.
        let mut topk: TopK<(&str, &str)> = TopK::new(opts.top_k);
        for (row, col_name, id) in scan {
            let stored = self.data.row(row);
            let score = dot(&q, stored);
            // Apply min_score gate.
            if let Some(min) = opts.min_score
                && score < min
            {
                continue;
            }
            topk.offer(score, (col_name, id));
        }

        // Build results — only clone ids/attrs for surviving top-k entries.
        let results = topk
            .into_sorted_desc()
            .into_iter()
            .map(|(score, (collection, id))| {
                let attrs = self
                    .collections
                    .get(collection)
                    .and_then(|c| c.docs.get(id))
                    .map(|e| e.attrs.clone())
                    .unwrap_or_default();
                Hit {
                    collection: collection.to_string(),
                    id: id.to_string(),
                    score,
                    attrs,
                }
            })
            .collect();

        Ok(results)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.check_writable()?;
        self.data.sync()?;
        self.log.sync()?;
        Ok(())
    }

    pub fn compact(&mut self) -> Result<()> {
        self.check_writable()?;

        // 1. Assign fresh contiguous row indices to live docs.
        //    Walk collections in sorted order for determinism.
        let live_rows: usize = self.collections.values().map(|c| c.docs.len()).sum();
        let mut new_rows: Vec<f32> = Vec::new();
        new_rows
            .try_reserve_exact(live_rows * self.data.dimension())
            .map_err(|_| oom("compacted vector matrix", live_rows * self.data.dimension()))?;
        let mut next_row: u64 = 0;

        // Build the new ops list for the log: CreateCollection + SetMeta + Upserts.
        let mut log_ops: Vec<Op> = Vec::new();

        // Sort collection names for determinism.
        let mut col_names: Vec<String> = self.collections.keys().cloned().collect();
        col_names.sort();

        // Collect all the row updates we need to apply to each collection's docs.
        // We map: (collection_name, id) -> new_row
        struct PendingUpdate {
            col: String,
            id: String,
            new_row: u64,
        }
        let mut updates: Vec<PendingUpdate> = Vec::new();

        for col_name in &col_names {
            let col = self.collections.get(col_name).unwrap();

            // Emit CreateCollection.
            log_ops.push(Op::CreateCollection {
                collection: col_name.clone(),
            });

            // Emit SetMeta if non-empty.
            if !col.meta.is_empty() {
                log_ops.push(Op::SetMeta {
                    collection: col_name.clone(),
                    meta: col.meta.clone(),
                });
            }

            // Assign new rows to live docs (sorted by id for determinism).
            let mut doc_ids: Vec<&String> = col.docs.keys().collect();
            doc_ids.sort();

            for id in doc_ids {
                let entry = &col.docs[id];
                // Copy the vector from the old data segment.
                let vec_slice = self.data.row(entry.row);
                new_rows.extend_from_slice(vec_slice);

                let new_row = next_row;
                next_row += 1;

                // Emit Upsert with new row index.
                log_ops.push(Op::Upsert {
                    collection: col_name.clone(),
                    id: id.clone(),
                    row: new_row,
                    attrs: entry.attrs.clone(),
                });

                updates.push(PendingUpdate {
                    col: col_name.clone(),
                    id: id.clone(),
                    new_row,
                });
            }
        }

        // 2. Rewrite data and log atomically (delegated to their modules).
        self.data.rewrite(&new_rows)?;
        self.log.rewrite(&log_ops)?;

        // 3. Update in-RAM DocEntry rows.
        for update in updates {
            if let Some(col) = self.collections.get_mut(&update.col)
                && let Some(entry) = col.docs.get_mut(&update.id)
            {
                entry.row = update.new_row;
            }
        }

        // 4. Reset dead-rows counter.
        self.dead_rows = 0;

        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::model::{Filter, Predicate, Record, SearchOpts, Value};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn rec(id: &str, vector: Vec<f32>) -> Record {
        Record {
            id: id.to_string(),
            vector,
            attrs: BTreeMap::new(),
        }
    }

    fn rec_with(id: &str, vector: Vec<f32>, attrs: BTreeMap<String, Value>) -> Record {
        Record {
            id: id.to_string(),
            vector,
            attrs,
        }
    }

    fn default_opts(top_k: usize) -> SearchOpts {
        SearchOpts {
            top_k,
            filter: Filter::default(),
            min_score: None,
        }
    }

    // ── Pure-logic tests (Miri-clean) ─────────────────────────────────────

    #[test]
    fn in_memory_dimension() {
        let store = Store::in_memory(4).unwrap();
        assert_eq!(store.dimension(), 4);
    }

    #[test]
    fn create_and_has_collection() {
        let mut store = Store::in_memory(3).unwrap();
        assert!(!store.has_collection("docs"));
        store.create_collection("docs").unwrap();
        assert!(store.has_collection("docs"));
    }

    #[test]
    fn create_collection_idempotent() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("docs").unwrap();
        store.create_collection("docs").unwrap(); // should not error
        assert!(store.has_collection("docs"));
        assert_eq!(store.collections().len(), 1);
    }

    #[test]
    fn drop_collection() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("docs").unwrap();
        store.drop_collection("docs").unwrap();
        assert!(!store.has_collection("docs"));
    }

    #[test]
    fn drop_nonexistent_collection_is_noop() {
        let mut store = Store::in_memory(3).unwrap();
        store.drop_collection("ghost").unwrap(); // no error
    }

    #[test]
    fn collections_sorted() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("zebra").unwrap();
        store.create_collection("apple").unwrap();
        store.create_collection("mango").unwrap();
        let names = store.collections();
        assert_eq!(names, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn metadata_round_trip() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("col").unwrap();
        let mut meta = BTreeMap::new();
        meta.insert("model".to_string(), "text-embed-v1".to_string());
        meta.insert("hwm".to_string(), "42".to_string());
        store.set_meta("col", meta.clone()).unwrap();
        assert_eq!(store.get_meta("col"), meta);
    }

    #[test]
    fn get_meta_absent_collection_returns_empty() {
        let store = Store::in_memory(2).unwrap();
        assert!(store.get_meta("nope").is_empty());
    }

    #[test]
    fn upsert_and_search_exact_match() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        // A vector pointing along x.
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        let hits = store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "doc1");
        assert!(
            (hits[0].score - 1.0).abs() < 1e-6,
            "exact match should score ~1.0"
        );
    }

    #[test]
    fn upsert_orthogonal_scores_zero() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        // Query along y — orthogonal to doc1's vector.
        let hits = store
            .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].score.abs() < 1e-6,
            "orthogonal vectors should score ~0.0"
        );
    }

    #[test]
    fn search_ranking_order() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        // doc_a is closest to query [1,0,0], doc_b is farther.
        store
            .upsert(
                "col",
                &[
                    rec("doc_a", vec![1.0, 0.0, 0.0]),
                    rec("doc_b", vec![0.0, 1.0, 0.0]),
                ],
            )
            .unwrap();
        let hits = store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "doc_a", "highest scorer should be first");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn upsert_is_idempotent_by_id() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        // Insert doc1 twice with different vectors.
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        store
            .upsert("col", &[rec("doc1", vec![0.0, 1.0, 0.0])])
            .unwrap();
        // Count stays at 1.
        assert_eq!(store.get_all("col").len(), 1);
        // The newest vector wins — query along y should give score ~1.0.
        let hits = store
            .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn delete_removes_doc() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        let removed = store.delete("col", &["doc1"]).unwrap();
        assert_eq!(removed, 1);
        assert!(store.get_all("col").is_empty());
    }

    #[test]
    fn delete_nonexistent_returns_zero() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        let removed = store.delete("col", &["ghost"]).unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn delete_where_by_attr() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        let mut attrs_a = BTreeMap::new();
        attrs_a.insert("kind".to_string(), Value::Str("file".to_string()));
        let mut attrs_b = BTreeMap::new();
        attrs_b.insert("kind".to_string(), Value::Str("section".to_string()));
        store
            .upsert(
                "col",
                &[
                    rec_with("doc_a", vec![1.0, 0.0, 0.0], attrs_a),
                    rec_with("doc_b", vec![0.0, 1.0, 0.0], attrs_b),
                ],
            )
            .unwrap();
        // Delete only files.
        let filter = Filter(vec![Predicate::Eq(
            "kind".to_string(),
            Value::Str("file".to_string()),
        )]);
        let removed = store.delete_where("col", &filter).unwrap();
        assert_eq!(removed, 1);
        let remaining = store.get_all("col");
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, "doc_b");
    }

    #[test]
    fn min_score_filters_low_results() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        // Query along y — score will be ~0.0, below min_score of 0.5.
        let opts = SearchOpts {
            top_k: 5,
            filter: Filter::default(),
            min_score: Some(0.5),
        };
        let hits = store.search(&["col"], &[0.0, 1.0, 0.0], &opts).unwrap();
        assert!(hits.is_empty(), "doc should be filtered by min_score");
    }

    #[test]
    fn filter_scoping_in_search() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        let mut attrs_rust = BTreeMap::new();
        attrs_rust.insert("lang".to_string(), Value::Str("rust".to_string()));
        let mut attrs_go = BTreeMap::new();
        attrs_go.insert("lang".to_string(), Value::Str("go".to_string()));
        store
            .upsert(
                "col",
                &[
                    rec_with("rust_doc", vec![1.0, 0.0, 0.0], attrs_rust),
                    rec_with("go_doc", vec![1.0, 0.0, 0.0], attrs_go),
                ],
            )
            .unwrap();
        // Search with a filter restricting to Rust only.
        let opts = SearchOpts {
            top_k: 5,
            filter: Filter(vec![Predicate::Eq(
                "lang".to_string(),
                Value::Str("rust".to_string()),
            )]),
            min_score: None,
        };
        let hits = store.search(&["col"], &[1.0, 0.0, 0.0], &opts).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "rust_doc");
    }

    #[test]
    fn multi_collection_merged_search() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col_a").unwrap();
        store.create_collection("col_b").unwrap();
        // col_a has the nearest doc to query [1,0,0].
        store
            .upsert("col_a", &[rec("best", vec![1.0, 0.0, 0.0])])
            .unwrap();
        // col_b has a less-close doc.
        let h = std::f32::consts::FRAC_1_SQRT_2;
        store
            .upsert("col_b", &[rec("ok", vec![h, h, 0.0])])
            .unwrap();
        let hits = store
            .search(&["col_a", "col_b"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 2);
        // The first hit should be "best" from col_a.
        assert_eq!(hits[0].id, "best");
        assert_eq!(hits[0].collection, "col_a");
        assert_eq!(hits[1].id, "ok");
        assert_eq!(hits[1].collection, "col_b");
    }

    #[test]
    fn multi_collection_hit_collection_field() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("alpha").unwrap();
        store.create_collection("beta").unwrap();
        store.upsert("alpha", &[rec("a1", vec![1.0, 0.0])]).unwrap();
        store.upsert("beta", &[rec("b1", vec![0.0, 1.0])]).unwrap();
        let hits = store
            .search(&["alpha", "beta"], &[1.0, 0.0], &default_opts(5))
            .unwrap();
        // Each hit should carry the correct collection field.
        for hit in &hits {
            if hit.id == "a1" {
                assert_eq!(hit.collection, "alpha");
            } else if hit.id == "b1" {
                assert_eq!(hit.collection, "beta");
            } else {
                panic!("unexpected id: {}", hit.id);
            }
        }
    }

    #[test]
    fn search_missing_collection_skipped() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("real").unwrap();
        store
            .upsert("real", &[rec("doc1", vec![1.0, 0.0])])
            .unwrap();
        // Include a non-existent collection — should not error.
        let hits = store
            .search(&["real", "phantom"], &[1.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "doc1");
    }

    #[test]
    fn upsert_wrong_dimension_errors() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        let result = store.upsert("col", &[rec("doc1", vec![1.0, 0.0])]);
        assert!(result.is_err());
    }

    #[test]
    fn upsert_implicitly_creates_collection() {
        let mut store = Store::in_memory(3).unwrap();
        // No explicit create_collection — upsert should auto-create.
        store
            .upsert("newcol", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        assert!(store.has_collection("newcol"));
    }

    #[test]
    fn get_all_includes_vector() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        let records = store.get_all("col");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "doc1");
        // Vector should be unit-normalized (already unit here).
        assert_eq!(records[0].vector.len(), 3);
    }

    #[test]
    fn compact_in_memory_preserves_live_docs() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        store
            .upsert("col", &[rec("doc2", vec![0.0, 1.0, 0.0])])
            .unwrap();
        // Overwrite doc1 — creates a dead row.
        store
            .upsert("col", &[rec("doc1", vec![0.0, 0.0, 1.0])])
            .unwrap();
        store.compact().unwrap();
        assert_eq!(store.dead_rows, 0);
        // Both docs should still be searchable.
        let hits = store
            .search(&["col"], &[0.0, 0.0, 1.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "doc1");
    }

    #[test]
    fn drop_increments_dead_rows() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        store
            .upsert("col", &[rec("doc2", vec![0.0, 1.0, 0.0])])
            .unwrap();
        assert_eq!(store.dead_rows, 0);
        store.drop_collection("col").unwrap();
        assert_eq!(store.dead_rows, 2);
    }

    #[test]
    fn top_k_limits_results() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("col").unwrap();
        for i in 0..10u32 {
            let v = vec![i as f32, 0.0];
            store.upsert("col", &[rec(&format!("doc{i}"), v)]).unwrap();
        }
        let hits = store
            .search(&["col"], &[1.0, 0.0], &default_opts(3))
            .unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn upsert_rolls_back_on_mid_batch_failure() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("col").unwrap();
        store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();

        let rows_before = store.data.row_count();
        let docs_before = store.get_all("col").len();
        let dead_before = store.dead_rows;

        // A 2-record batch where the first append succeeds and the second fails.
        store.data.fail_after(1);
        let res = store.upsert("col", &[rec("b", vec![0.0, 1.0]), rec("c", vec![1.0, 1.0])]);
        assert!(res.is_err());

        // Everything restored: no orphan row, index untouched, dead-count untouched.
        assert_eq!(
            store.data.row_count(),
            rows_before,
            "orphan row must be rolled back"
        );
        assert_eq!(store.get_all("col").len(), docs_before, "index unchanged");
        assert_eq!(store.dead_rows, dead_before);

        // Store remains usable for subsequent writes (disarm the seam first).
        store.data.fail_after(usize::MAX);
        store.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
        assert_eq!(store.get_all("col").len(), 2);
    }

    #[test]
    fn footprint_tracks_rows_dead_and_docs() {
        let mut store = Store::in_memory(4).unwrap();
        store.create_collection("col").unwrap();

        let fp0 = store.footprint();
        assert_eq!(fp0.rows, 0);
        assert_eq!(fp0.dead_rows, 0);
        assert_eq!(fp0.dimension, 4);
        assert_eq!(fp0.vector_bytes, 0);
        assert_eq!(fp0.doc_count, 0);

        store
            .upsert("col", &[rec("a", vec![1.0, 0.0, 0.0, 0.0])])
            .unwrap();
        store
            .upsert("col", &[rec("b", vec![0.0, 1.0, 0.0, 0.0])])
            .unwrap();
        let fp1 = store.footprint();
        assert_eq!(fp1.rows, 2);
        assert_eq!(fp1.dead_rows, 0);
        assert_eq!(fp1.vector_bytes, 2 * 4 * 4); // 2 rows × dim 4 × 4 bytes
        assert_eq!(fp1.doc_count, 2);

        // Overwrite "a": a dead row appears, doc_count stays at 2.
        store
            .upsert("col", &[rec("a", vec![0.0, 0.0, 1.0, 0.0])])
            .unwrap();
        let fp2 = store.footprint();
        assert_eq!(fp2.rows, 3);
        assert_eq!(fp2.dead_rows, 1);
        assert_eq!(fp2.doc_count, 2);

        // Compaction reclaims the dead row.
        store.compact().unwrap();
        let fp3 = store.footprint();
        assert_eq!(fp3.rows, 2);
        assert_eq!(fp3.dead_rows, 0);
        assert_eq!(fp3.doc_count, 2);
    }

    #[test]
    fn max_vector_bytes_refuses_over_budget_upsert() {
        // Cap at exactly 2 rows (dim 2 × 4 bytes × 2 rows = 16 bytes).
        let config = Config::new("/dev/null/in-memory", 2)
            .open_mode(OpenMode::ReadWrite)
            .auto_compact(None)
            .max_vector_bytes(Some(16));
        let mut store = Store {
            config,
            data: DataSegment::in_memory(2),
            log: OpLog::in_memory(),
            lock: None,
            collections: HashMap::new(),
            dead_rows: 0,
        };
        store.create_collection("col").unwrap();
        store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
        store.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
        assert_eq!(store.footprint().vector_bytes, 16);

        // The third row would exceed the cap — refuse, leaving the store intact.
        let res = store.upsert("col", &[rec("c", vec![1.0, 1.0])]);
        assert!(res.is_err());
        assert_eq!(store.footprint().rows, 2, "refused batch must not append");

        // Store stays usable for reads.
        let hits = store
            .search(&["col"], &[1.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 2);
    }

    // ── File-backed tests (ignored under Miri) ────────────────────────────

    #[cfg_attr(miri, ignore)]
    #[test]
    fn open_refuses_data_file_over_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Write 3 rows (dim 2) with no cap.
        {
            let mut store = Store::open(Config::new(&path, 2)).unwrap();
            store.create_collection("col").unwrap();
            store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
            store.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
            store.upsert("col", &[rec("c", vec![1.0, 1.0])]).unwrap();
        }

        // Reopen with a cap below the on-disk size → clean Err, not a panic.
        let res = Store::open(Config::new(&path, 2).max_vector_bytes(Some(8)));
        assert!(res.is_err());
        let msg = res.err().unwrap().to_string();
        assert!(
            msg.contains("max_vector_bytes"),
            "error should mention the cap: {msg}"
        );

        // A cap at/above the size still opens fine.
        let ok = Store::open(Config::new(&path, 2).max_vector_bytes(Some(24)));
        assert!(ok.is_ok());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn upsert_rollback_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        {
            let mut store = Store::open(Config::new(&path, 2)).unwrap();
            store.create_collection("col").unwrap();
            store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();

            // Next append fails immediately; the batch must fully roll back.
            store.data.fail_after(0);
            assert!(store.upsert("col", &[rec("b", vec![0.0, 1.0])]).is_err());
            assert_eq!(store.data.row_count(), 1);
            assert_eq!(store.get_all("col").len(), 1);
        }

        // Reopen: only "a" is present, replayed cleanly with no corruption.
        let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
        let recs = store.get_all("col");
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].id, "a");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn reopen_sees_prior_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Write some data.
        {
            let mut store = Store::open(Config::new(&path, 3)).unwrap();
            store.create_collection("col").unwrap();
            store
                .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
                .unwrap();
            store
                .upsert("col", &[rec("doc2", vec![0.0, 1.0, 0.0])])
                .unwrap();
        }

        // Reopen and verify.
        {
            let store = Store::open(Config::new(&path, 3).open_mode(OpenMode::ReadOnly)).unwrap();
            assert!(store.has_collection("col"));
            let records = store.get_all("col");
            assert_eq!(records.len(), 2);
            let ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
            assert!(ids.contains(&"doc1".to_string()));
            assert!(ids.contains(&"doc2".to_string()));
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn readonly_rejects_writes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Create a store and write something.
        {
            Store::open(Config::new(&path, 2)).unwrap();
        }

        // Open read-only.
        let mut store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();

        assert!(store.create_collection("col").is_err());
        assert!(store.drop_collection("col").is_err());
        assert!(store.set_meta("col", BTreeMap::new()).is_err());
        assert!(store.upsert("col", &[rec("doc1", vec![1.0, 0.0])]).is_err());
        assert!(store.delete("col", &["doc1"]).is_err());
        assert!(store.delete_where("col", &Filter::default()).is_err());
        assert!(store.flush().is_err());
        assert!(store.compact().is_err());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn compaction_preserves_live_docs_and_results() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();

        {
            let mut store = Store::open(Config::new(&path, 3).auto_compact(None)).unwrap();
            store.create_collection("col").unwrap();
            store
                .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
                .unwrap();
            store
                .upsert("col", &[rec("doc2", vec![0.0, 1.0, 0.0])])
                .unwrap();
            // Overwrite doc1 — creates a dead row.
            store
                .upsert("col", &[rec("doc1", vec![0.0, 0.0, 1.0])])
                .unwrap();
            assert_eq!(store.dead_rows, 1);
            store.compact().unwrap();
            assert_eq!(store.dead_rows, 0);

            // Verify search still works after compact.
            let hits = store
                .search(&["col"], &[0.0, 0.0, 1.0], &default_opts(5))
                .unwrap();
            assert_eq!(hits.len(), 2);
            assert_eq!(hits[0].id, "doc1");
        }

        // Reopen and verify compacted state persists.
        {
            let store = Store::open(
                Config::new(&path, 3)
                    .open_mode(OpenMode::ReadOnly)
                    .auto_compact(None),
            )
            .unwrap();
            let records = store.get_all("col");
            assert_eq!(records.len(), 2);
            let hits = store
                .search(&["col"], &[0.0, 0.0, 1.0], &default_opts(5))
                .unwrap();
            assert_eq!(hits.len(), 2);
            assert_eq!(hits[0].id, "doc1");
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn metadata_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();

        let mut meta = BTreeMap::new();
        meta.insert("model".to_string(), "text-v3".to_string());

        {
            let mut store = Store::open(Config::new(&path, 2)).unwrap();
            store.create_collection("col").unwrap();
            store.set_meta("col", meta.clone()).unwrap();
        }

        {
            let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
            assert_eq!(store.get_meta("col"), meta);
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn auto_compact_triggers_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Write with enough dead rows to trigger auto-compact (ratio > 0.5).
        {
            let mut store = Store::open(
                Config::new(&path, 3).auto_compact(None), // disable for setup
            )
            .unwrap();
            store.create_collection("col").unwrap();
            // Insert 3 docs then overwrite 2 of them → 2 dead of 5 total rows = 40%.
            // Then delete 1 more → 3 dead of 5 total > 50%.
            store
                .upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])])
                .unwrap();
            store
                .upsert("col", &[rec("b", vec![0.0, 1.0, 0.0])])
                .unwrap();
            store
                .upsert("col", &[rec("c", vec![0.0, 0.0, 1.0])])
                .unwrap();
            store
                .upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])])
                .unwrap(); // overwrite a
            store
                .upsert("col", &[rec("b", vec![0.0, 1.0, 0.0])])
                .unwrap(); // overwrite b
            // Now we have 5 rows, 2 dead (ratio = 0.4), 3 live docs.
            assert_eq!(store.dead_rows, 2);
        }

        // Reopen with auto_compact = Some(0.3) — should trigger compaction.
        {
            let store = Store::open(Config::new(&path, 3).auto_compact(Some(0.3))).unwrap();
            assert_eq!(store.dead_rows, 0, "auto-compact should have run");
            assert_eq!(store.get_all("col").len(), 3);
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn upsert_idempotent_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();

        {
            let mut store = Store::open(Config::new(&path, 2)).unwrap();
            store.create_collection("col").unwrap();
            store.upsert("col", &[rec("doc1", vec![1.0, 0.0])]).unwrap();
            // Overwrite with a different vector.
            store.upsert("col", &[rec("doc1", vec![0.0, 1.0])]).unwrap();
        }

        {
            let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
            let records = store.get_all("col");
            assert_eq!(records.len(), 1);
            // The newest vector should win — search along y should score ~1.0.
            let hits = store
                .search(&["col"], &[0.0, 1.0], &default_opts(5))
                .unwrap();
            assert_eq!(hits.len(), 1);
            assert!((hits[0].score - 1.0).abs() < 1e-5);
        }
    }
}
