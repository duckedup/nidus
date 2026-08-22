//! Mutations: collection lifecycle, `upsert`/`delete`, `flush`, `compact`. Everything here
//! funnels through [`check_writable`](Store::check_writable) and the §6.2 durable write order;
//! `upsert` is all-or-nothing. Read/search lives in [`super::read`].

use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail};

use super::{Collection, DocEntry, Store, oom};
use crate::backend::CasOutcome;
use crate::config::{Fsync, OpenMode};
use crate::data::SegmentIntegrity;
use crate::filter;
use crate::findex::FilterIndexField;
use crate::fts::FtsField;
use crate::manifest::history::{self, HistoryEntry, HistoryFloor};

/// History-entry deletes issued per call, both when pruning and when compaction reclaims.
/// The floor already makes every excluded entry unreachable, so a backlog is cheap to leave.
const HISTORY_PRUNE_BATCH: usize = 32;
use crate::manifest::{BASE_SEGMENT, MANIFEST_KEY, Manifest};
use crate::model::{Distance, Filter, Op, Record, Value};
use crate::profile::OpenProfile;
use crate::search::normalize;

/// One live segment's checksum-sidecar finding, named so a mismatch identifies exactly which
/// segment's bytes disagree with its sidecar (#160). Order matches the manifest: oldest first,
/// active (normally only partially covered) last.
#[derive(Debug, Clone)]
pub struct SegmentReport {
    pub segment: String,
    pub integrity: SegmentIntegrity,
}

impl Store {
    /// Reject mutations in ReadOnly mode and, in cluster mode, renew/fence the writer lease
    /// before any durable write (SPEC §14.6 phase 5). Every mutating op calls this first, so it is
    /// the single point where a superseded writer is stopped before it can clobber the store.
    fn check_writable(&self) -> Result<()> {
        if self.config.open_mode == OpenMode::ReadOnly {
            bail!("read-only store: mutations are not allowed");
        }
        self.renew_lease()?;
        Ok(())
    }

    /// Renew (and fence) the cluster writer lease — op-driven, no background thread. A no-op
    /// outside cluster mode (no lease held); errors if this writer was superseded.
    fn renew_lease(&self) -> Result<()> {
        if let Some(lease) = &self.lease
            && let Err(e) = lease.renew()
        {
            // Only a *definitive* loss latches the fence (nidus-lp4.7), so a readiness probe can
            // see it and pull the instance from rotation (nidus-lp4.1). A transient backend blip
            // must not: that would retire a healthy writer permanently. Fail the write, stay up.
            if crate::backend::is_lease_lost(&e) {
                crate::backend::latch_fenced(&self.fenced);
                crate::diag::diag!(
                    crate::diag::Level::Error,
                    "lease",
                    "writer lease LOST — this instance is fenced and must be replaced",
                    "err" => format!("{e:#}"),
                );
            } else {
                crate::diag::diag!(
                    crate::diag::Level::Warn,
                    "lease",
                    "writer lease renewal failed transiently — this write is refused, but the \
                     lease is still ours",
                    "err" => format!("{e:#}"),
                );
            }
            return Err(e);
        }
        Ok(())
    }

    /// Advance the manifest version and republish it, making this batch an addressable commit
    /// point: the cluster commit counter (SPEC §14.6 phase 5), and — since seals are rare and
    /// `segment_max_rows` defaults to off — the only thing a pinned read could pin to.
    fn note_commit_point(&mut self) -> Result<()> {
        if self.in_memory || self.config.open_mode == OpenMode::ReadOnly {
            return Ok(());
        }
        if !self.config.cluster && self.config.history_versions.is_none() {
            return Ok(());
        }
        self.data.bump_version();
        self.persist_manifest()
    }

    /// Whether *this* mutation issues its own durable barrier.
    fn barrier_now(&self) -> bool {
        self.config.fsync == Fsync::PerBatch && !self.defer_barrier
    }

    /// Apply the fsync policy after a mutation: sync data then log under PerBatch, then (in
    /// cluster mode) advance the published commit counter so peers see the batch. When the
    /// barrier is not ours to take, record that one is owed instead (see `pending_barrier`).
    fn maybe_sync(&mut self) -> Result<()> {
        crate::metrics::metrics().write_batches.inc();
        if !self.barrier_now() {
            self.pending_barrier = true;
            return Ok(());
        }
        self.data.sync()?;
        self.log.sync()?;
        crate::metrics::metrics().durability_barriers.inc();
        self.note_commit_point()
    }

    /// Enter a deferred-barrier scope (**group commit**, nidus-xb9.1), returning the previous
    /// setting for [`end_deferred`](Self::end_deferred) to restore. Returning the old value
    /// rather than just clearing makes nesting safe.
    pub(crate) fn begin_deferred(&mut self) -> bool {
        std::mem::replace(&mut self.defer_barrier, true)
    }

    /// Leave a deferred-barrier scope, restoring what [`begin_deferred`](Self::begin_deferred)
    /// found. Does **not** take the barrier — [`commit`](Self::commit) does, and keeping them
    /// separate is what lets the caller decide when the group is closed.
    pub(crate) fn end_deferred(&mut self, prev: bool) {
        self.defer_barrier = prev;
    }

    /// The group barrier: make everything appended since the last barrier durable.
    pub fn commit(&mut self) -> Result<()> {
        if !self.pending_barrier || self.config.fsync == Fsync::OnFlush {
            return Ok(());
        }
        self.data.sync()?;
        self.log.sync()?;
        crate::metrics::metrics().durability_barriers.inc();
        self.note_commit_point()?;
        // Last: a failure above must leave the barrier still owed, so the next `commit` (or
        // `flush`) retries it rather than reporting durability nobody achieved.
        self.pending_barrier = false;
        Ok(())
    }

    /// Seal the active segment and publish the new manifest once it passes
    /// [`Config::segment_max_rows`] (SPEC §14.4). Called *before* a batch appends, so a seal
    /// failure leaves the store byte-identical — the fresh segment goes live only on commit.
    fn maybe_seal(&mut self) -> Result<()> {
        let Some(max) = self.config.segment_max_rows else {
            return Ok(());
        };
        if self.data.active_rows() >= max && self.data.seal()? {
            self.persist_manifest()?;
            // Index the segment that just froze (cold), if it meets the size threshold —
            // SPEC §14.3. No-op unless per-segment indexing is on.
            self.index_just_sealed();
        }
        Ok(())
    }

    /// Publish the current segment set as the `manifest` — the atomic commit point for a
    /// seal/compaction (and, in cluster mode, every durable batch). No-op for in-memory or
    /// read-only stores (no durable manifest).
    fn persist_manifest(&mut self) -> Result<()> {
        if self.in_memory || self.config.open_mode == OpenMode::ReadOnly {
            return Ok(());
        }
        let Some(p) = self.persistence.clone() else {
            return Ok(());
        };
        let manifest = self.data.manifest(self.open_profile.clone());
        let result = if !self.config.cluster {
            manifest.store(p.as_ref())
        } else {
            let bytes = manifest.encode()?;
            match p.put_cas(MANIFEST_KEY, &bytes, self.manifest_cas.as_deref())? {
                CasOutcome::Written(new) => {
                    self.manifest_cas = match new {
                        Some(t) => Some(t),
                        None => p.get_cas(MANIFEST_KEY)?.and_then(|(_, t)| t),
                    };
                    Ok(())
                }
                CasOutcome::Stale => Err(anyhow!(
                    "writer lease lost: the manifest was committed by another writer — this \
                     instance was superseded (its lease was taken over while it stalled mid-batch); \
                     stop writing and reopen"
                )),
                // No CAS on this backend — publish plainly (advisory; the per-batch lease fence
                // still applies). Keep `manifest_cas` `None` so we stay on this path.
                CasOutcome::Unsupported => manifest.store(p.as_ref()),
            }
        };
        // After the commit point, off the durability path (nidus-bnf): a failure here loses
        // history, never correctness. Before it would be worse — a CAS-stale writer would
        // clobber the entry of the peer that actually won this version.
        if result.is_ok() {
            self.record_history_entry(&manifest, p.as_ref());
        }
        result
    }

    /// Best-effort: record `manifest`'s commit point as a history entry (nidus-bnf), then
    /// prune past `Config::history_versions`. No-op when history is off. A failure here
    /// costs only introspection — the manifest itself is already durable.
    fn record_history_entry(&mut self, manifest: &Manifest, p: &dyn crate::backend::Persistence) {
        let Some(n) = self.config.history_versions else {
            return;
        };
        let log_offset = match self.log.offset() {
            Ok(o) => o,
            Err(e) => {
                crate::diag::diag!(
                    crate::diag::Level::Warn,
                    "history",
                    "failed to read log offset while recording history entry",
                    "err" => format!("{e:#}"),
                );
                return;
            }
        };
        let entry = HistoryEntry {
            format_version: history::HIST_FORMAT_VERSION,
            version: manifest.version,
            dimension: manifest.dimension,
            distance: manifest.distance,
            segments: manifest.segments.clone(),
            next_id: manifest.next_id,
            row_count: self.data.row_count(),
            log_offset,
            profile: manifest.profile.clone(),
            commit_millis: crate::meta::now_ms().max(0) as u64,
        };
        if let Err(e) = history::store_entry(p, &entry) {
            crate::diag::diag!(
                crate::diag::Level::Warn,
                "history",
                "failed to write history entry",
                "err" => format!("{e:#}"),
            );
            return;
        }
        self.prune_history(manifest.version, n as u64, p);
    }

    /// Delete history entries the retention window (`n` versions) has aged past, capped at
    /// [`HISTORY_PRUNE_BATCH`] keys per call. Advances `pruned_through` unconditionally (nidus-bnf) — a failing delete
    /// must never freeze the floor, and a leaked object is already unreachable once excluded.
    fn prune_history(&mut self, version: u64, n: u64, p: &dyn crate::backend::Persistence) {
        if version <= n {
            return;
        }
        let target = version - n;
        if target <= self.pruned_through {
            return;
        }
        let start = self.pruned_through + 1;
        let end = target.min(start + HISTORY_PRUNE_BATCH as u64 - 1);
        for v in start..=end {
            let _ = history::delete_entry(p, v);
            self.pruned_through = v;
        }
        let floor = HistoryFloor {
            format_version: history::HIST_FORMAT_VERSION,
            oldest_readable: self.pruned_through + 1,
        };
        if let Err(e) = history::store_floor(p, &floor) {
            crate::diag::diag!(
                crate::diag::Level::Warn,
                "history",
                "failed to advance the history floor",
                "err" => format!("{e:#}"),
            );
        }
    }

    /// The store's currently recorded open-time profile (nidus-141), empty when nothing is
    /// recorded and every knob resolves to its built-in default (or an explicit flag).
    pub fn open_profile(&self) -> &OpenProfile {
        &self.open_profile
    }

    /// Record `p` as the store's open-time profile so every later open resolves
    /// ann/quantization/query_threads/mmap without re-passing them — an explicit act, never
    /// implied by `open` itself. `ReadOnly` rejects it, matching every other mutation.
    pub fn set_open_profile(&mut self, p: &OpenProfile) -> Result<()> {
        self.check_writable()?;
        // A recorded profile resolves at open, not here, so an unbuildable combination would
        // fail every future open — including the `clear` that would undo it. Reject it now.
        let mut resolved = self.baseline_config.clone();
        resolved.apply_profile(p);
        resolved.validate()?;
        self.open_profile = p.clone();
        self.data.bump_version();
        self.persist_manifest()
    }

    /// Clear the recorded profile (write an empty one) — later opens fall back to built-in
    /// defaults unless overridden by an explicit flag.
    pub fn clear_open_profile(&mut self) -> Result<()> {
        self.set_open_profile(&OpenProfile::default())
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
            // Only rowed docs leave a reclaimable data row behind.
            self.dead_rows += col.docs.values().filter(|e| e.row.is_some()).count();
            // Drop the collection's FTS schema + field indexes.
            if self.fts.is_active() {
                self.fts.drop_collection(name);
                self.fts_dirty = true;
            }
            if self.findex.is_active() {
                self.findex.drop_collection(name);
                self.findex_dirty = true;
            }
            self.log.append(&Op::DropCollection {
                collection: name.to_string(),
            })?;
            self.maybe_sync()?;
            // The collection's docs left the scan order — drop the cache.
            self.invalidate_scan_order();
        }
        Ok(())
    }

    /// Declare `collection`'s full-text-indexed fields, then build the field indexes from its
    /// live docs. Settable any time; `create_collection_with_fts` shares this code so a fresh
    /// collection indexes from its first upsert. Redeclaring rebuilds the affected indexes.
    pub fn set_fts_schema(&mut self, collection: &str, fields: &[FtsField]) -> Result<()> {
        self.check_writable()?;
        // Validate before the log append: an unusable k1/b persisted here would be replayed
        // on every subsequent open.
        crate::fts::validate(fields)?;
        // Implicitly create the collection if absent (matches set_meta / replay leniency).
        self.collections
            .entry(collection.to_string())
            .or_insert_with(Collection::new);
        self.log.append(&Op::SetFtsFields {
            collection: collection.to_string(),
            fields: fields.to_vec(),
        })?;
        self.maybe_sync()?;
        self.fts.set_schema(collection, fields);
        // Build the field indexes from docs already in the collection (sorted ids →
        // reproducible docnums). For a brand-new collection this loop is empty.
        let col = &self.collections[collection];
        let mut ids: Vec<&String> = col.docs.keys().collect();
        ids.sort();
        for id in ids {
            let attrs = &col.docs[id].attrs;
            self.fts.index_doc(collection, id, attrs);
        }
        self.fts_dirty = true;
        Ok(())
    }

    /// Declare `collection`'s filter-indexed fields (SPEC §7.4/§7.5), then build the index
    /// from its live docs. Settable any time; redeclaring rebuilds. An empty `fields` drops
    /// the declaration, which is how a caller turns the index off.
    pub fn set_filter_index(
        &mut self,
        collection: &str,
        fields: &[FilterIndexField],
    ) -> Result<()> {
        self.check_writable()?;
        // Validate before the log append: an unusable declaration persisted here would be
        // replayed on every subsequent open.
        crate::findex::validate(fields)?;
        self.collections
            .entry(collection.to_string())
            .or_insert_with(Collection::new);
        self.log.append(&Op::SetFilterIndex {
            collection: collection.to_string(),
            fields: fields.to_vec(),
        })?;
        self.maybe_sync()?;
        self.findex.set_schema(collection, fields);
        // Docs written before the declaration are not in the index; build from them now
        // (sorted ids → reproducible docnums). Empty for a brand-new collection.
        let col = &self.collections[collection];
        let mut ids: Vec<&String> = col.docs.keys().collect();
        ids.sort();
        for id in ids {
            let attrs = &col.docs[id].attrs;
            self.findex.index_doc(collection, id, attrs);
        }
        self.findex_dirty = true;
        Ok(())
    }

    /// Create `collection` (idempotent) and declare its full-text fields up front. The
    /// recommended FTS path: indexing is fully incremental from the first upsert.
    pub fn create_collection_with_fts(&mut self, name: &str, fields: &[FtsField]) -> Result<()> {
        self.create_collection(name)?;
        self.set_fts_schema(name, fields)
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

    /// Upsert a batch, all-or-nothing: every fallible step rolls `data` and `log` back to their
    /// entry marks and returns the original error, so a failed batch (ENOSPC mid-write) leaves the
    /// store byte-identical. The in-RAM index changes only in the final infallible phase.
    pub fn upsert(&mut self, collection: &str, records: &[Record]) -> Result<usize> {
        self.check_writable()?;

        let dim = self.data.dimension();

        // Validate all present vectors first (fail fast before any mutation). A
        // text-only record (`vector: None`) is exempt — it occupies no data row.
        for rec in records {
            if let Some(v) = &rec.vector
                && v.len() != dim
            {
                bail!(
                    "vector length {} does not match store dimension {}",
                    v.len(),
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

        // Seal the active segment first if it has outgrown the threshold, so this batch's
        // rows land in a fresh segment (SPEC §14.4). Before any append + before the marks
        // below, so a seal failure leaves the store unchanged.
        self.maybe_seal()?;

        // Capacity gate: refuse, before any append, a batch that would grow the matrix past the
        // cap — clean refusal, no rollback. Counts physical rows including dead ones, so
        // `compact` reclaims headroom; text-only records cost no rows.
        let vector_count = records.iter().filter(|r| r.vector.is_some()).count() as u64;
        if let Some(cap) = self.config.max_vector_bytes {
            let projected =
                (self.data.row_count() + vector_count) * self.data.dimension() as u64 * 4;
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
        let mut staged: Vec<(String, Option<u64>, BTreeMap<String, Value>)> = Vec::new();
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
        let should_normalize = self.config.distance == Distance::Cosine;
        for rec in records {
            let row = match &rec.vector {
                Some(v) => {
                    let mut v = v.clone();
                    if should_normalize {
                        normalize(&mut v);
                    }
                    match self.data.append(&v) {
                        Ok(row) => Some(row),
                        Err(e) => {
                            self.data
                                .truncate_to(data_mark)
                                .context("rollback data after failed append")?;
                            return Err(e);
                        }
                    }
                }
                // Text-only doc: no embedding, no data row.
                None => None,
            };
            staged.push((rec.id.clone(), row, rec.attrs.clone()));
        }

        // Whether this batch takes its own barrier, decided once so phases 2 and 4 cannot
        // disagree (they must sync as a pair or not at all).
        let barrier_now = self.barrier_now();
        crate::metrics::metrics().write_batches.inc();

        // Phase 2: fsync data before writing log records — under `PerBatch` only.
        if barrier_now && let Err(e) = self.data.sync() {
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
            .chain(staged.iter().map(|(id, row, attrs)| match row {
                Some(row) => Op::Upsert {
                    collection: collection.to_string(),
                    id: id.clone(),
                    row: *row,
                    attrs: attrs.clone(),
                },
                None => Op::UpsertText {
                    collection: collection.to_string(),
                    id: id.clone(),
                    attrs: attrs.clone(),
                },
            }));
        for op in log_ops {
            if let Err(e) = self.log.append(&op) {
                self.rollback(data_mark, log_mark)?;
                return Err(e);
            }
        }

        // Phase 4: fsync log (or defer to commit()/flush()).
        if barrier_now {
            if let Err(e) = self.log.sync() {
                self.rollback(data_mark, log_mark)?;
                return Err(e);
            }
            crate::metrics::metrics().durability_barriers.inc();
        } else {
            // Appended, not yet durable — `commit()`/`flush()` owes this batch a barrier.
            self.pending_barrier = true;
        }

        // Phase 5: commit to the in-RAM index — infallible. Both files are durable,
        // and the maps' capacity was reserved in Phase 0, so no insert reallocates.
        if let Some(col) = pending_collection {
            self.collections.insert(collection.to_string(), col);
        }
        let col = self.collections.get_mut(collection).unwrap();
        let ann_on = self.ann.is_some();
        let fts_on = self.fts.is_active();
        let findex_on = self.findex.is_active();
        let mut new_owners: Vec<(u64, String)> = Vec::new();
        let mut count = 0usize;
        for (id, row, attrs) in staged {
            // Only a vector-bearing new doc joins the ANN index.
            if ann_on && let Some(r) = row {
                new_owners.push((r, id.clone()));
            }
            // Index the doc's text into any FTS fields (no-op if this collection has no
            // schema). Done before the attrs move into the index. O(batch).
            if fts_on {
                self.fts.index_doc(collection, &id, &attrs);
            }
            // A missed upsert here is a silently wrong query result, not a slow one: the
            // doc would never become a candidate. A missed delete is only a false positive.
            if findex_on {
                self.findex.index_doc(collection, &id, &attrs);
            }
            // Overwriting a *rowed* doc leaves its old row dead.
            if let Some(old) = col.docs.insert(id, DocEntry { row, attrs })
                && old.row.is_some()
            {
                self.dead_rows += 1;
            }
            count += 1;
        }

        if fts_on {
            self.fts_dirty = true;
        }

        // Quantize only the rows this batch appended (O(batch)); refits lazily.
        self.extend_quant(data_mark);
        // Index the new rows in the ANN graph/lists (O(batch)). No-op when ANN is off.
        self.extend_ann(collection, data_mark, &new_owners);
        // The doc set changed — drop the cached scan order (rebuilt on next query).
        self.invalidate_scan_order();
        // Cluster: announce this committed batch via the manifest commit counter so reader
        // instances detect it (this path commits durably itself, bypassing `maybe_sync`).
        if barrier_now {
            self.note_commit_point()?;
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
            let Some(old) = col.docs.remove(id) else {
                continue;
            };
            // Only a rowed doc leaves a reclaimable data row.
            if old.row.is_some() {
                self.dead_rows += 1;
            }
            // Tombstone the doc in any FTS field indexes (no-op when none).
            self.fts.remove_doc(collection, id);
            self.findex.remove_doc(collection, id);
            self.log.append(&Op::Delete {
                collection: collection.to_string(),
                id: id.to_string(),
            })?;
            count += 1;
        }

        if count > 0 {
            self.maybe_sync()?;
            // Docs were removed — drop the cached scan order.
            self.invalidate_scan_order();
            if self.fts.is_active() {
                self.fts_dirty = true;
            }
            if self.findex.is_active() {
                self.findex_dirty = true;
            }
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

    /// Delete across *every* collection as one all-or-nothing batch (nidus-166). Looping
    /// `delete_where` instead renews the cluster lease per collection, so a transient renewal
    /// failure mid-loop left earlier collections deleted while the caller was told it failed.
    pub fn delete_where_all(&mut self, filter: &Filter) -> Result<usize> {
        self.check_writable()?;

        // Phase 1: collect every match up front — pure, so an empty sweep touches nothing.
        let mut targets: Vec<(String, Vec<String>)> = Vec::new();
        for (name, col) in &self.collections {
            let ids: Vec<String> = col
                .docs
                .iter()
                .filter(|(_, entry)| filter::matches(filter, &entry.attrs))
                .map(|(id, _)| id.clone())
                .collect();
            if !ids.is_empty() {
                targets.push((name.clone(), ids));
            }
        }
        if targets.is_empty() {
            return Ok(0);
        }

        let log_mark = self.log.offset()?;

        // Phase 2: append every Delete record, rolling the log back as a unit on failure.
        // Deletes append no vectors, so the log is the only file to unwind.
        for (collection, ids) in &targets {
            for id in ids {
                let op = Op::Delete {
                    collection: collection.clone(),
                    id: id.clone(),
                };
                if let Err(e) = self.log.append(&op) {
                    self.log
                        .truncate_to(log_mark)
                        .context("rollback log after failed sweep")?;
                    return Err(e);
                }
            }
        }

        // Phase 3: durable barrier (or defer it to commit()/flush()). Before the in-RAM
        // commit, so a sync failure unwinds to a store that never saw the sweep.
        if let Err(e) = self.maybe_sync() {
            self.log
                .truncate_to(log_mark)
                .context("rollback log after failed sweep sync")?;
            return Err(e);
        }

        // Phase 4: commit to the in-RAM index — infallible, both files are durable.
        let mut count = 0usize;
        for (collection, ids) in &targets {
            let Some(col) = self.collections.get_mut(collection) else {
                continue;
            };
            for id in ids {
                let Some(old) = col.docs.remove(id) else {
                    continue;
                };
                // Only a rowed doc leaves a reclaimable data row.
                if old.row.is_some() {
                    self.dead_rows += 1;
                }
                self.fts.remove_doc(collection, id);
                self.findex.remove_doc(collection, id);
                count += 1;
            }
        }

        if count > 0 {
            self.invalidate_scan_order();
            if self.fts.is_active() {
                self.fts_dirty = true;
            }
            if self.findex.is_active() {
                self.findex_dirty = true;
            }
        }

        Ok(count)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.check_writable()?;
        self.data.sync()?;
        self.log.sync()?;
        crate::metrics::metrics().durability_barriers.inc();
        // Under OnFlush — or when a group-commit scope left a barrier owed — the commit-counter
        // bump was deferred to here (PerBatch already did it in `maybe_sync`). Advance it now the
        // batch is durable so peers see it.
        if self.config.fsync == Fsync::OnFlush || self.pending_barrier {
            self.note_commit_point()?;
        }
        self.pending_barrier = false;
        // Seal a large active-segment tail into an immutable segment (SPEC §14.4). No-op
        // unless `segment_max_rows` is set and the tail is over it.
        self.maybe_seal()?;
        // Refresh the active segment's checksum sidecar (#160) — after the durable barrier
        // above, never inside it. Best-effort: a save failure must not fail the flush, since
        // the next flush simply covers more rows.
        let _ = self.refresh_active_checksum();
        // Refresh the shared working set so peers skip a rebuild (SPEC §13.3). Best-effort
        // and a no-op without an external memory tier — never fails the durable flush.
        self.publish_working_set();
        Ok(())
    }

    /// Recompute and save the checksum sidecar for the *active* segment, covering its rows as
    /// of right now — the same on-disk encoding `data::checksum` uses (private to
    /// `crate::data`), so `Segments::verify_checksums` reads it unchanged. No-op in-memory/RO.
    fn refresh_active_checksum(&mut self) -> Result<()> {
        if self.in_memory || self.config.open_mode == OpenMode::ReadOnly {
            return Ok(());
        }
        let Some(p) = self.persistence.clone() else {
            return Ok(());
        };
        let name = self
            .data
            .manifest(self.open_profile.clone())
            .segments
            .last()
            .cloned()
            .ok_or_else(|| anyhow!("manifest names no segments"))?;
        let dim = self.data.dimension();
        let rows = self.data.active_rows();
        let base = self.data.row_count() - rows;

        let mut hasher = crc32fast::Hasher::new();
        for r in base..base + rows {
            for &f in self.data.row(r) {
                hasher.update(&f.to_le_bytes());
            }
        }
        let crc = hasher.finalize();

        let key = crate::data::checksum_key(&name, dim, self.config.distance);
        crate::data::checksum_save(p.as_ref(), &name, &key, rows, crc)
    }

    /// Verify every live segment's checksum sidecar (#160), most-recent (active) last. A
    /// mismatch is reported, never repaired — recomputing over corrupted bytes would launder
    /// the corruption. The public half; a `Nidus` wrapper in `src/lib.rs` is out of scope here.
    pub fn verify_integrity(&mut self) -> Result<Vec<SegmentReport>> {
        let names = self.data.manifest(self.open_profile.clone()).segments;
        let integrities = self.data.verify_checksums()?;
        Ok(names
            .into_iter()
            .zip(integrities)
            .map(|(segment, integrity)| SegmentReport { segment, integrity })
            .collect())
    }

    pub fn compact(&mut self) -> Result<()> {
        self.check_writable()?;

        // 1. Assign fresh contiguous row indices to live *rowed* docs (text-only docs
        //    carry no vector and are re-emitted as `UpsertText`). Walk collections in
        //    sorted order for determinism.
        let rowed: usize = self
            .collections
            .values()
            .flat_map(|c| c.docs.values())
            .filter(|e| e.row.is_some())
            .count();
        let mut new_rows: Vec<f32> = Vec::new();
        new_rows
            .try_reserve_exact(rowed * self.data.dimension())
            .map_err(|_| oom("compacted vector matrix", rowed * self.data.dimension()))?;
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

            // Re-emit the FTS schema so a post-compact replay restores it.
            if let Some(fields) = self.fts.schema_for(col_name) {
                log_ops.push(Op::SetFtsFields {
                    collection: col_name.clone(),
                    fields: fields.to_vec(),
                });
            }

            // Re-emit the filter-index schema too, or a post-compact replay loses it.
            if let Some(fields) = self.findex.schema_for(col_name) {
                log_ops.push(Op::SetFilterIndex {
                    collection: col_name.clone(),
                    fields: fields.to_vec(),
                });
            }

            // Assign new rows to live docs (sorted by id for determinism).
            let mut doc_ids: Vec<&String> = col.docs.keys().collect();
            doc_ids.sort();

            for id in doc_ids {
                let entry = &col.docs[id];
                match entry.row {
                    Some(old_row) => {
                        // Copy the vector from the old data segment to its new row.
                        let vec_slice = self.data.row(old_row);
                        new_rows.extend_from_slice(vec_slice);

                        let new_row = next_row;
                        next_row += 1;

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
                    // Text-only doc: no vector to relocate; re-emit as UpsertText.
                    None => log_ops.push(Op::UpsertText {
                        collection: col_name.clone(),
                        id: id.clone(),
                        attrs: entry.attrs.clone(),
                    }),
                }
            }
        }

        // 1c. The compaction fence (nidus-bnf), written *before* the rewrite so a crash
        // refuses history rather than serving it. Keyed on the store *having* history, not on
        // this process's flag: `nidus compact` without it must still fence.
        let new_floor = match self.persistence.clone() {
            Some(p) if !history::list_versions(p.as_ref())?.is_empty() => Some((
                p,
                HistoryFloor {
                    format_version: history::HIST_FORMAT_VERSION,
                    oldest_readable: self.data.version() + 1,
                },
            )),
            _ => None,
        };
        // Captured before advancing `pruned_through` below, so the best-effort cleanup loop
        // after the rewrite still knows the true start of the range to delete.
        let prior_pruned_through = self.pruned_through;
        if let Some((p, floor)) = &new_floor {
            history::store_floor(p.as_ref(), floor)?;
            // Advance the in-memory floor now, before `persist_manifest` runs below — otherwise
            // its own count-based pruning would see the stale value and could write a *smaller*
            // floor back, reopening a window onto the segments this rewrite just invalidated.
            self.pruned_through = self
                .pruned_through
                .max(floor.oldest_readable.saturating_sub(1));
        }

        // 2. Rewrite data and log atomically. Compaction collapses every segment into one fresh
        //    base; `rewrite` returns the now-unreferenced names so their objects can be reclaimed,
        //    and the new manifest is the commit point (SPEC §14.2).
        let dropped = self.data.rewrite(&new_rows)?;
        self.log.rewrite(&log_ops)?;
        self.persist_manifest()?;
        if let Some(p) = self.persistence.as_deref() {
            for name in &dropped {
                // Best-effort: a leftover unreferenced segment object wastes space but is
                // already invisible (the manifest no longer names it), so a delete failure
                // must not fail the compaction.
                let _ = p.delete(name);
            }
        }
        // nidus-bnf: reclaim the history entries the new floor just excluded — best-effort,
        // since each is already unreachable the instant the floor excludes it.
        if let Some((p, floor)) = &new_floor {
            // Capped like `prune_history`: the floor already makes every one of these
            // unreachable, so a long backlog must not turn one compaction into a million
            // synchronous backend deletes.
            for v in (prior_pruned_through + 1..floor.oldest_readable).take(HISTORY_PRUNE_BATCH) {
                let _ = history::delete_entry(p.as_ref(), v);
            }
        }

        // 2b. Drop every per-segment IVF sidecar, the dropped segments' and the base's alike.
        //     `rewrite` replaced the base's bytes in place at the same base row, so its old
        //     sidecar would key-match a later seal at the same row count (nidus-143).
        let mut stale: Vec<String> = dropped.clone();
        stale.push(BASE_SEGMENT.to_string());
        self.delete_seg_index_sidecars(&stale);

        // 3. Update in-RAM DocEntry rows.
        for update in updates {
            if let Some(col) = self.collections.get_mut(&update.col)
                && let Some(entry) = col.docs.get_mut(&update.id)
            {
                entry.row = Some(update.new_row);
            }
        }

        // 4. Reset dead-rows counter.
        self.dead_rows = 0;

        // 5. Rebuild quantization state with compacted vectors.
        self.rebuild_quant();

        // 5b. Rebuild the ANN index + reverse map (rows were renumbered) and refresh
        //     its on-disk cache. Best effort: the cache is derived, so a persist
        //     failure must not fail the compaction.
        self.rebuild_ann();

        // 5b-ii. Rebuild the per-segment IVF indexes. Compaction left one fresh active segment, so
        //     the store is fully exact until it seals again and the next over-threshold write
        //     re-indexes. No-op unless per-segment indexing is on.
        self.build_segment_indexes();

        // 5c. Rebuild the FTS index from the live docs (drops tombstones, renumbers
        //     docnums). Reads attrs, so it is unaffected by the row renumbering. Done
        //     before persist so the refreshed `fts` cache matches the rewritten log.
        self.rebuild_fts();
        self.rebuild_findex();

        // 5d. Refresh both on-disk caches. Best effort: a persist failure must not fail
        //     the compaction (the caches are derived).
        let _ = self.persist_index();

        // 6. Rows were renumbered — drop the cached scan order.
        self.invalidate_scan_order();

        Ok(())
    }
}
