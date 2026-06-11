//! Mutations: collection lifecycle, `upsert`/`delete`, `flush`, and `compact`. Every
//! method here funnels through [`check_writable`](Store::check_writable) and the §6.2
//! durable write order; `upsert` is all-or-nothing (rolls `data`+`log` back to their
//! entry marks on any failure). Read/search lives in [`super::read`].

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};

use super::{Collection, DocEntry, Store, oom};
use crate::config::{Fsync, OpenMode};
use crate::filter;
use crate::model::{Distance, Filter, Op, Record, Value};
use crate::search::normalize;

impl Store {
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
            // The collection's docs left the scan order — drop the cache.
            self.invalidate_scan_order();
        }
        Ok(())
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
        let mut staged: Vec<(String, u64, BTreeMap<String, Value>)> = Vec::new();
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
        let should_normalize = self.config.distance == Distance::Cosine;
        for rec in records {
            let mut v = rec.vector.clone();
            if should_normalize {
                normalize(&mut v);
            }
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
        let ann_on = self.ann.is_some();
        let mut new_owners: Vec<(u64, String)> = Vec::new();
        let mut count = 0usize;
        for (id, row, attrs) in staged {
            if col.docs.contains_key(&id) {
                self.dead_rows += 1; // overwriting: the old row becomes dead
            }
            if ann_on {
                new_owners.push((row, id.clone()));
            }
            col.docs.insert(id, DocEntry { row, attrs });
            count += 1;
        }

        // Quantize only the rows this batch appended (O(batch)); refits lazily.
        self.extend_quant(data_mark);
        // Index the new rows in the ANN graph/lists (O(batch)). No-op when ANN is off.
        self.extend_ann(collection, data_mark, &new_owners);
        // The doc set changed — drop the cached scan order (rebuilt on next query).
        self.invalidate_scan_order();
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
            // Docs were removed — drop the cached scan order.
            self.invalidate_scan_order();
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

        // 5. Rebuild quantization state with compacted vectors.
        self.rebuild_quant();

        // 5b. Rebuild the ANN index + reverse map (rows were renumbered) and refresh
        //     its on-disk cache. Best effort: the cache is derived, so a persist
        //     failure must not fail the compaction.
        self.rebuild_ann();
        let _ = self.persist_index();

        // 6. Rows were renumbered — drop the cached scan order.
        self.invalidate_scan_order();

        Ok(())
    }
}
