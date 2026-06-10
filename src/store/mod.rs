//! The integrator: in-RAM index + write/read glue + compaction. Composes
//! [`DataSegment`](crate::data::DataSegment), [`OpLog`](crate::log::OpLog), and an
//! optional [`WriteLock`](crate::lock::WriteLock). Contract: see the root `SPEC.md`
//! §3, §5–§8.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context, Result, anyhow, bail};

use crate::ann::Ann;
use crate::config::{Config, Fsync, OpenMode};
use crate::data::DataSegment;
use crate::filter;
use crate::lock::WriteLock;
use crate::log::OpLog;
use crate::model::{Distance, Filter, Footprint, Hit, Op, QuantKind, Record, SearchOpts};
use crate::search::{
    QuantParams, TopK, dot, dot_i8, euclidean_neg_sq, euclidean_neg_sq_i8, hamming, normalize,
    pack_signs, pack_signs_into,
};

// ── In-RAM types ─────────────────────────────────────────────────────────────

/// The cached row-sorted scan order: `(row, collection, id)` for every live doc,
/// sorted by `row` (see [`Store::scan_order`]).
type ScanOrder = Vec<(u64, String, String)>;

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

/// Minimum total scan *work* — candidate rows × dimension — before a parallel search
/// splits across worker threads. Below this, thread spawn/join overhead outweighs the
/// scan, so we stay serial even when `Config::query_threads > 1`. The floor is on work
/// rather than a flat row count because per-row scan cost scales with dimension: a fixed
/// row floor over-parallelizes narrow vectors and under-parallelizes wide ones. ~1.05M
/// units ≈ 4096 rows at dim 256, or ~1365 rows at dim 768.
const PARALLEL_SCAN_WORK_FLOOR: usize = 1 << 20;

/// Score a slice of candidate rows into a fresh bounded top-k heap. The unit of
/// parallel work: each worker scores one chunk independently, then the caller
/// merges the per-chunk heaps. Pure read of `data` (shared `&` across threads).
fn score_chunk<'a>(
    data: &DataSegment,
    chunk: &[(u64, &'a str, &'a str)],
    q: &[f32],
    score_fn: fn(&[f32], &[f32]) -> f32,
    top_k: usize,
    min_score: Option<f32>,
) -> TopK<(&'a str, &'a str)> {
    let mut topk: TopK<(&'a str, &'a str)> = TopK::new(top_k);
    for &(row, col_name, id) in chunk {
        let score = score_fn(q, data.row(row));
        if let Some(min) = min_score
            && score < min
        {
            continue;
        }
        topk.offer(score, (col_name, id));
    }
    topk
}

/// Score a chunk against the **int8** matrix into a bounded top-k of `overscan`
/// candidates — the quantized first-pass unit of parallel work, mirroring
/// [`score_chunk`] for the f32 path. The int8 score is monotonic with the f32 score
/// (shared symmetric scale), so it picks the right candidate set; exact scores come
/// from the caller's f32 rerank. Carries `row` in the item so the rerank can re-read
/// the f32 vector. `min_score` is *not* applied here — the int8 score is only an
/// ordering proxy, so the floor is enforced on the exact f32 score during rerank.
fn score_chunk_i8<'a>(
    quant_vectors: &[i8],
    dim: usize,
    chunk: &[(u64, &'a str, &'a str)],
    q_i8: &[i8],
    is_euclidean: bool,
    overscan: usize,
) -> TopK<(u64, &'a str, &'a str)> {
    let mut topk: TopK<(u64, &'a str, &'a str)> = TopK::new(overscan);
    for &(row, col_name, id) in chunk {
        let base = row as usize * dim;
        let end = base + dim;
        if end > quant_vectors.len() {
            continue;
        }
        let stored_i8 = &quant_vectors[base..end];
        let approx_score = if is_euclidean {
            euclidean_neg_sq_i8(q_i8, stored_i8) as f32
        } else {
            dot_i8(q_i8, stored_i8) as f32
        };
        topk.offer(approx_score, (row, col_name, id));
    }
    topk
}

/// Score a chunk against the **binary** (sign-bit) matrix into a bounded top-k of
/// `overscan` candidates — the binary first-pass unit of parallel work, mirroring
/// [`score_chunk_i8`]. Score is `-(hamming)` (higher = better), monotone with cosine
/// rank for unit vectors, so it picks the right candidate set; exact scores come from
/// the caller's f32 rerank. Carries `row` so the rerank can re-read the f32 vector.
/// `min_score` is *not* applied here — Hamming is only an ordering proxy.
fn score_chunk_bin<'a>(
    words: &[u64],
    words_per_row: usize,
    chunk: &[(u64, &'a str, &'a str)],
    q_words: &[u64],
    overscan: usize,
) -> TopK<(u64, &'a str, &'a str)> {
    let mut topk: TopK<(u64, &'a str, &'a str)> = TopK::new(overscan);
    for &(row, col_name, id) in chunk {
        let base = row as usize * words_per_row;
        let end = base + words_per_row;
        if end > words.len() {
            continue;
        }
        let approx_score = -(hamming(q_words, &words[base..end]) as f32);
        topk.offer(approx_score, (row, col_name, id));
    }
    topk
}

/// Split `scan` across `workers` threads, score each chunk with `score_one` into its
/// own bounded top-k of capacity `cap`, then merge the per-worker results into one.
/// The shared parallel-scan engine behind both the f32 and int8 first passes.
///
/// Each worker sorts its own chunk by physical row before scoring, so the per-chunk
/// sweep stays storage-ordered for the prefetcher — the global row-sort is skipped on
/// the parallel path (the prefetch win is per-chunk, and per-chunk sorts run in
/// parallel instead of as serial pre-work, cutting the Amdahl tax). Reads of `data` /
/// the quant matrix are shared `&` across threads; the only mutation is each worker
/// reordering its disjoint `&mut` chunk.
fn parallel_topk<'a, T, F>(
    scan: &mut [(u64, &'a str, &'a str)],
    workers: usize,
    cap: usize,
    score_one: F,
) -> Result<TopK<T>>
where
    T: Send,
    F: Fn(&[(u64, &'a str, &'a str)]) -> TopK<T> + Sync,
{
    let chunk_len = scan.len().div_ceil(workers);
    let score_one = &score_one;
    let locals = std::thread::scope(|s| -> Result<Vec<Vec<(f32, T)>>> {
        let handles: Vec<_> = scan
            .chunks_mut(chunk_len)
            .map(|chunk| {
                s.spawn(move || {
                    chunk.sort_unstable_by_key(|&(row, _, _)| row);
                    score_one(chunk).into_sorted_desc()
                })
            })
            .collect();
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(
                h.join()
                    .map_err(|_| anyhow!("search worker thread panicked"))?,
            );
        }
        Ok(out)
    })?;

    let mut merged: TopK<T> = TopK::new(cap);
    for local in locals {
        for (score, item) in local {
            merged.offer(score, item);
        }
    }
    Ok(merged)
}

// ── Store ─────────────────────────────────────────────────────────────────────

/// int8 scalar quantization state. `vectors` mirrors the f32 `data` rows one-for-one
/// (same physical row indices).
struct Int8State {
    params: QuantParams,
    /// Quantized vectors, flat and row-major, `data.row_count() * dim` int8 values.
    vectors: Vec<i8>,
    /// How many rows `params` was fit from. Upserts quantize new rows against the
    /// current `params` and only refit (rescan for a fresh scale) once the row count
    /// outgrows this by [`REFIT_GROWTH`], keeping incremental upsert amortized O(1)/row.
    params_rows: u64,
}

/// Binary (sign-bit) quantization state. **Scale-free:** each row's code is just its
/// sign bits, so there is no scale to fit and no refit — incremental upsert is a plain
/// append. `words` mirrors the f32 `data` rows one-for-one, `words_per_row` u64 each.
struct BinState {
    /// Packed sign-bit codes, flat and row-major, `row_count * words_per_row` u64 values.
    words: Vec<u64>,
    /// `dim.div_ceil(64)` — words per row's code.
    words_per_row: usize,
}

/// The active quantization scheme's in-RAM state, maintained when `Config::quantization`
/// is set (`None` when quantization is off — the f32 brute-force default).
enum Quant {
    Int8(Int8State),
    Binary(BinState),
}

impl Quant {
    /// An empty quant state for `kind`, validating metric compatibility up front.
    /// Binary codes are an angular proxy (they ignore magnitude), so they are rejected
    /// for any metric but cosine — a clear error beats a silently wrong ranking.
    fn empty(kind: QuantKind, dim: usize, distance: Distance) -> Result<Self> {
        match kind {
            QuantKind::Int8 => Ok(Quant::Int8(Int8State {
                params: QuantParams::from_vectors(&[]),
                vectors: Vec::new(),
                params_rows: 0,
            })),
            QuantKind::Binary => {
                if distance != Distance::Cosine {
                    bail!(
                        "binary quantization requires Distance::Cosine (sign codes are an \
                         angular proxy and ignore magnitude); use int8 quantization for a \
                         dot-product or Euclidean store"
                    );
                }
                Ok(Quant::Binary(BinState {
                    words: Vec::new(),
                    words_per_row: dim.div_ceil(64),
                }))
            }
        }
    }
}

/// Pack a flat row-major f32 matrix (`dim` floats per row) into row-major sign-bit
/// codes, `dim.div_ceil(64)` u64 per row. The whole-matrix build used by `rebuild_quant`.
fn pack_matrix(vectors: &[f32], dim: usize) -> Vec<u64> {
    if dim == 0 {
        return Vec::new();
    }
    let words_per_row = dim.div_ceil(64);
    let rows = vectors.len() / dim;
    let mut out = vec![0u64; rows * words_per_row];
    for r in 0..rows {
        let src = &vectors[r * dim..(r + 1) * dim];
        let dst = &mut out[r * words_per_row..(r + 1) * words_per_row];
        pack_signs_into(src, dst);
    }
    out
}

/// Refit the quantization scale once the live row count grows past this multiple of
/// the count it was last fit from. Geometric (doubling) → amortized O(1) per row over
/// a full incremental build, while bounding how stale the shared scale can get.
const REFIT_GROWTH: u64 = 2;

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
    /// Quantization state (None when quantization is off — the f32 brute-force default).
    quant: Option<Quant>,
    /// Approximate-nearest-neighbour index (None when ANN is off — the exact default).
    /// Mutually exclusive with `quant` (rejected at `open`).
    ann: Option<Ann>,
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
        let data = DataSegment::open(&data_path, config.dimension, config.distance)?;

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

        Self::reject_ann_with_quant(&config)?;
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
            lock,
            collections,
            dead_rows,
            quant,
            ann,
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
        // 8. Build the ANN index (and its reverse map) from the loaded vectors.
        store.rebuild_ann();

        Ok(store)
    }

    /// ANN and quantization both replace the search path and may not run together.
    fn reject_ann_with_quant(config: &Config) -> Result<()> {
        if config.ann.is_some() && config.quantization.is_some() {
            bail!(
                "Config::ann and Config::quantization cannot both be set — ANN and \
                 quantization both replace the search path; enable only one"
            );
        }
        Ok(())
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
        Self::reject_ann_with_quant(&config)?;
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
            lock: None,
            collections: HashMap::new(),
            dead_rows: 0,
            quant,
            ann,
            row_to_doc: Vec::new(),
            scan_order: std::sync::RwLock::new(None),
            config,
        })
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Rebuild the quantized matrix from *all* current vectors. O(N) — used on `open`,
    /// `compact`, and the occasional int8 geometric refit, not per upsert batch. int8
    /// re-fits the scale and re-quantizes; binary repacks sign bits (scale-free).
    fn rebuild_quant(&mut self) {
        let dim = self.data.dimension();
        let all = self.data.vectors();
        match self.quant {
            None => {}
            Some(Quant::Int8(ref mut s)) => {
                s.params = QuantParams::from_vectors(all);
                s.vectors = s.params.quantize_all(all);
                s.params_rows = self.data.row_count();
            }
            Some(Quant::Binary(ref mut s)) => {
                s.words_per_row = dim.div_ceil(64);
                s.words = pack_matrix(all, dim);
            }
        }
    }

    /// Incrementally extend the quantized matrix after `upsert` appended rows
    /// `[prev_rows, row_count())` — O(batch), not O(N). int8 quantizes the new rows
    /// against the existing scale, falling back to a full [`rebuild_quant`] when there
    /// is no scale yet or the row count has grown past [`REFIT_GROWTH`]× the fit set (so
    /// a drifting distribution can't keep saturating a stale scale). Binary is scale-free
    /// — it just packs the new rows' sign bits, never refits.
    fn extend_quant(&mut self, prev_rows: u64) {
        let total = self.data.row_count();
        let dim = self.data.dimension();
        // Decide whether int8 needs a full refit before taking the mutable state borrow.
        let refit = match self.quant {
            None => return,
            Some(Quant::Int8(ref s)) => s.params_rows == 0 || total > s.params_rows * REFIT_GROWTH,
            Some(Quant::Binary(_)) => false, // scale-free: never refits
        };
        if refit {
            self.rebuild_quant();
            return;
        }
        let all = self.data.vectors();
        match self.quant {
            None => {}
            Some(Quant::Int8(ref mut s)) => {
                s.vectors.resize(total as usize * dim, 0);
                for row in prev_rows as usize..total as usize {
                    let base = row * dim;
                    let (src, dst) = (&all[base..base + dim], &mut s.vectors[base..base + dim]);
                    s.params.quantize(src, dst);
                }
            }
            Some(Quant::Binary(ref mut s)) => {
                let wpr = s.words_per_row;
                s.words.resize(total as usize * wpr, 0);
                for row in prev_rows as usize..total as usize {
                    let src = &all[row * dim..(row + 1) * dim];
                    pack_signs_into(src, &mut s.words[row * wpr..(row + 1) * wpr]);
                }
            }
        }
    }

    /// Rebuild the ANN index and its reverse map from *all* current live docs. O(N) —
    /// used on `open` and after `compact` renumbers rows. No-op when ANN is off.
    fn rebuild_ann(&mut self) {
        if self.ann.is_none() {
            return;
        }
        // Reverse map sized to the physical row count; live docs fill their slot, dead
        // rows stay `None`. Also collect the live rows to (re)build the index over.
        let mut row_to_doc: Vec<Option<(String, String)>> =
            vec![None; self.data.row_count() as usize];
        let mut live_rows: Vec<u64> = Vec::new();
        for (col_name, col) in &self.collections {
            for (id, entry) in &col.docs {
                if (entry.row as usize) < row_to_doc.len() {
                    row_to_doc[entry.row as usize] = Some((col_name.clone(), id.clone()));
                    live_rows.push(entry.row);
                }
            }
        }
        self.row_to_doc = row_to_doc;
        if let Some(ann) = self.ann.as_mut() {
            ann.build(&self.data, &live_rows);
        }
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
        if let Some(ann) = self.ann.as_mut() {
            ann.insert_rows(&self.data, &new_rows);
        }
    }

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
            // The collection's docs left the scan order — drop the cache.
            self.invalidate_scan_order();
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

    /// List records matching `filter` across `collections`, without vector scoring.
    /// Skips the first `offset` matches and returns up to `limit` more, in insertion
    /// order (row index), all with `score: 0.0`. `offset`/`limit` paginate a stable
    /// ordering: the full match set is ordered by physical row, then the window
    /// `[offset, offset + limit)` is returned.
    pub fn list(
        &self,
        collections: &[&str],
        filter: &Filter,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<Hit>> {
        self.with_sorted_scan(collections, filter, |scan| {
            let results = scan
                .iter()
                .skip(offset)
                .take(limit)
                .map(|&(_, collection, id)| {
                    let attrs = self
                        .collections
                        .get(collection)
                        .and_then(|c| c.docs.get(id))
                        .map(|e| e.attrs.clone())
                        .unwrap_or_default();
                    Hit {
                        collection: collection.to_string(),
                        id: id.to_string(),
                        score: 0.0,
                        attrs,
                    }
                })
                .collect();
            Ok(results)
        })
    }

    /// How many worker threads to split a scan of `scan_len` candidates across: the
    /// configured `query_threads` when that is `> 1` *and* the total work
    /// (`scan_len × dimension`) clears [`PARALLEL_SCAN_WORK_FLOOR`], else `1` (serial).
    fn parallel_workers(&self, scan_len: usize) -> usize {
        let threads = self.config.query_threads.max(1);
        if threads > 1 && scan_len.saturating_mul(self.data.dimension()) >= PARALLEL_SCAN_WORK_FLOOR
        {
            threads
        } else {
            1
        }
    }

    /// Total live docs across all collections — the scan-order cache's length and the
    /// yardstick for "does this scope cover the whole store?" (`scan_cap == live count`).
    fn live_doc_count(&self) -> usize {
        self.collections.values().map(|c| c.docs.len()).sum()
    }

    /// Drop the cached scan order. Called from every write that changes the doc set
    /// (`upsert`, `delete`, `drop_collection`, `compact`); `&mut self`, so it takes the
    /// lock uncontended via `get_mut` and clears even a poisoned lock.
    fn invalidate_scan_order(&mut self) {
        *self.scan_order.get_mut().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// A read guard over the cached row-sorted scan order, rebuilding it first if stale.
    /// The returned guard always holds `Some`. Double-checked under the write lock so
    /// concurrent searchers rebuild at most once. Fallible only on the rebuild's
    /// `try_reserve` (OOM) — the per-entry `String` clones share the codebase's
    /// no-`try_reserve`-for-clones caveat (small next to the vector matrix).
    fn scan_order(&self) -> Result<std::sync::RwLockReadGuard<'_, Option<ScanOrder>>> {
        // Fast path: already built and current.
        {
            let guard = self.scan_order.read().unwrap_or_else(|e| e.into_inner());
            if guard.is_some() {
                return Ok(guard);
            }
        }
        // Rebuild under the write lock; another searcher may have raced us (re-check).
        {
            let mut w = self.scan_order.write().unwrap_or_else(|e| e.into_inner());
            if w.is_none() {
                let n = self.live_doc_count();
                let mut order: ScanOrder = Vec::new();
                order
                    .try_reserve_exact(n)
                    .map_err(|_| oom("scan-order cache", n))?;
                for (col_name, col) in &self.collections {
                    for (id, entry) in &col.docs {
                        order.push((entry.row, col_name.clone(), id.clone()));
                    }
                }
                order.sort_unstable_by_key(|&(row, _, _)| row);
                *w = Some(order);
            }
        }
        Ok(self.scan_order.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// Build the in-scope, filter-passing scan **in row order** and hand it to `f`.
    /// This is the single place row-sorted access is established for `search` and
    /// `list`, so both reach the data matrix storage-ordered (nidus-33k) — and skip the
    /// per-query sort when they can (nidus-dxt).
    ///
    /// Two ways there. When the scope covers every live doc (`scan_cap == live count` —
    /// the single-collection store and `Scope::All`, the common cases), the scan is
    /// drawn from the lazily-cached global order, so the sort is amortized across all
    /// queries between writes rather than redone each time. Otherwise (a strict subset)
    /// it falls back to iterating just the in-scope collections and sorting that smaller
    /// scan — which is cheaper than walking the whole-store cache to extract a small
    /// slice. Either way `f` receives an already row-sorted `&mut` scan.
    fn with_sorted_scan<R>(
        &self,
        collections: &[&str],
        filter: &Filter,
        f: impl for<'b> FnOnce(&mut [(u64, &'b str, &'b str)]) -> Result<R>,
    ) -> Result<R> {
        let scan_cap: usize = collections
            .iter()
            .filter_map(|c| self.collections.get(*c))
            .map(|c| c.docs.len())
            .sum();
        let mut scan: Vec<(u64, &str, &str)> = Vec::new();
        scan.try_reserve(scan_cap)
            .map_err(|_| oom("search scan buffer", scan_cap))?;

        if scan_cap == self.live_doc_count() {
            // Whole-store scope: draw from the cached row-sorted order (no per-query
            // sort). The cache covers every live doc, so every entry is in scope.
            let guard = self.scan_order()?;
            let order = guard
                .as_ref()
                .expect("scan_order() guarantees Some on success");
            let match_all = filter.0.is_empty();
            for (row, col, id) in order {
                if !match_all {
                    // Non-empty filter needs the attrs; look the live entry up (cheaper
                    // than a sort at scale, and skipped entirely for the common
                    // empty-filter search).
                    let Some(attrs) = self
                        .collections
                        .get(col)
                        .and_then(|c| c.docs.get(id))
                        .map(|e| &e.attrs)
                    else {
                        continue;
                    };
                    if !filter::matches(filter, attrs) {
                        continue;
                    }
                }
                scan.push((*row, col.as_str(), id.as_str()));
            }
            // `scan` inherits the cache's row order — already sorted, no sort call.
            f(&mut scan)
        } else {
            // Strict subset: iterate only the in-scope collections, then sort that
            // (smaller) scan.
            for &col_name in collections {
                let Some(col) = self.collections.get(col_name) else {
                    continue;
                };
                for (id, entry) in &col.docs {
                    if !filter::matches(filter, &entry.attrs) {
                        continue;
                    }
                    scan.push((entry.row, col_name, id.as_str()));
                }
            }
            scan.sort_unstable_by_key(|&(row, _, _)| row);
            f(&mut scan)
        }
    }

    /// Brute-force search over the union of `collections`, merged into one ranking
    /// (one bounded top-k heap fed by every in-scope collection). The scoring
    /// function is determined by the store's [`Distance`] metric.
    ///
    /// When int8 quantization is enabled, the first pass scores with int8 vectors
    /// (overscanning by the rescore factor), then re-ranks the candidates with f32.
    pub fn search(
        &self,
        collections: &[&str],
        query: &[f32],
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        let mut q = query.to_vec();
        if self.config.distance == Distance::Cosine {
            normalize(&mut q);
        }

        let score_fn: fn(&[f32], &[f32]) -> f32 = match self.config.distance {
            Distance::Cosine | Distance::DotProduct => dot,
            Distance::Euclidean => euclidean_neg_sq,
        };

        // ANN path: walk the index for an over-fetched candidate set, then post-filter
        // by scope + filter + min_score and rerank. Approximate — recall is traded for
        // speed, and a selective filter/scope can starve the candidate set (the
        // exact-prefilter follow-up addresses that). Skips the linear scan entirely.
        if self.ann.is_some() {
            return Ok(self.search_ann(collections, &q, opts));
        }

        // Gather the in-scope, filter-passing rows in physical-row order (for
        // cache-friendly sequential `data` access — nidus-33k). `with_sorted_scan`
        // hands back an already row-sorted scan, reusing the cached whole-store order
        // where it can so the sort is not redone every query (nidus-dxt).
        self.with_sorted_scan(collections, &opts.filter, |scan| {
            // Decide once whether this query splits across workers (configured threads +
            // enough scan work to amortize spawn cost). On the parallel path each worker
            // re-sorts its own (already-ordered) chunk; the serial path scores in place,
            // so the global sort never caps speedup (Amdahl).
            let workers = self.parallel_workers(scan.len());

            // Two-pass quantized search if enabled and the quantized matrix is populated.
            match self.quant {
                Some(Quant::Int8(ref s)) if !s.vectors.is_empty() && self.data.dimension() > 0 => {
                    return self.search_int8(&q, scan, opts, s, score_fn, workers);
                }
                Some(Quant::Binary(ref s)) if !s.words.is_empty() && self.data.dimension() > 0 => {
                    return self.search_binary(&q, scan, opts, s, score_fn, workers);
                }
                _ => {}
            }

            // Standard f32 brute-force path. Split across worker threads when engaged;
            // otherwise score serially. Both yield the same bounded top-k (ties aside).
            let topk = if workers > 1 {
                parallel_topk(scan, workers, opts.top_k, |chunk| {
                    score_chunk(&self.data, chunk, &q, score_fn, opts.top_k, opts.min_score)
                })?
            } else {
                score_chunk(&self.data, scan, &q, score_fn, opts.top_k, opts.min_score)
            };

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
        })
    }

    /// The configured overscan factor (the first pass keeps `top_k × rescore`
    /// candidates for the f32 rerank). 1 when quantization is off — unused on that path.
    fn rescore(&self) -> usize {
        self.config.quantization.map_or(1, |q| q.rescore)
    }

    /// Two-pass int8 search: int8 first-pass selects candidates, f32 reranks. The int8
    /// first pass is the lever that scales with threads — int8 moves 4× fewer bytes than
    /// f32, so it is compute- not bandwidth-bound — so it splits across `workers` (when
    /// engaged), while the f32 rerank stays serial (only `top_k × rescore` rows, too few
    /// to amortize a second fan-out).
    fn search_int8<'a>(
        &self,
        q: &[f32],
        scan: &mut [(u64, &'a str, &'a str)],
        opts: &SearchOpts,
        s: &Int8State,
        score_fn: fn(&[f32], &[f32]) -> f32,
        workers: usize,
    ) -> Result<Vec<Hit>> {
        let dim = self.data.dimension();
        let overscan = opts.top_k.saturating_mul(self.rescore()).max(opts.top_k);

        // Quantize the query vector with the same shared scale as the stored rows.
        let mut q_i8 = vec![0i8; dim];
        s.params.quantize(q, &mut q_i8);

        // First pass: int8 scoring to select overscan candidates. The int8 score is
        // monotonic with the f32 score (shared symmetric scale), so it picks the right
        // candidate set; exact scores come from the f32 rerank below. Parallel when
        // engaged (the int8 sweep is the part that scales with threads), else serial.
        let is_euclidean = self.config.distance == Distance::Euclidean;
        let topk_q = if workers > 1 {
            parallel_topk(scan, workers, overscan, |chunk| {
                score_chunk_i8(&s.vectors, dim, chunk, &q_i8, is_euclidean, overscan)
            })?
        } else {
            // `scan` arrives row-sorted from `with_sorted_scan` — score it in place.
            score_chunk_i8(&s.vectors, dim, scan, &q_i8, is_euclidean, overscan)
        };

        let candidates = topk_q.into_sorted_desc();
        Ok(self.rerank_candidates(q, &candidates, score_fn, opts))
    }

    /// Two-pass binary search: a Hamming first-pass over the 32×-smaller sign-bit matrix
    /// selects candidates, f32 reranks. Cosine only (enforced at `open`), so the query is
    /// already unit-normalized when it reaches here; its sign code is invariant to that.
    /// The binary first pass moves 32× fewer bytes than f32, so it scales with `workers`
    /// even harder than int8; the f32 rerank stays serial.
    fn search_binary<'a>(
        &self,
        q: &[f32],
        scan: &mut [(u64, &'a str, &'a str)],
        opts: &SearchOpts,
        s: &BinState,
        score_fn: fn(&[f32], &[f32]) -> f32,
        workers: usize,
    ) -> Result<Vec<Hit>> {
        let overscan = opts.top_k.saturating_mul(self.rescore()).max(opts.top_k);

        // Pack the query's sign bits with the same rule as the stored rows.
        let q_words = pack_signs(q);
        let wpr = s.words_per_row;

        // First pass: Hamming scoring selects overscan candidates. Score = -(hamming),
        // monotone with cosine rank for unit vectors; exact scores come from the rerank.
        let topk_q = if workers > 1 {
            parallel_topk(scan, workers, overscan, |chunk| {
                score_chunk_bin(&s.words, wpr, chunk, &q_words, overscan)
            })?
        } else {
            // `scan` arrives row-sorted from `with_sorted_scan` — score it in place.
            score_chunk_bin(&s.words, wpr, scan, &q_words, overscan)
        };

        let candidates = topk_q.into_sorted_desc();
        Ok(self.rerank_candidates(q, &candidates, score_fn, opts))
    }

    /// ANN search: walk the index for `top_k × overscan` candidate rows, then resolve
    /// each to its owning doc, keep only those in scope and passing the filter, and
    /// rank by the exact f32 score (the candidate scores returned by the index are
    /// already the exact metric — both the HNSW beam and the IVF probe score real
    /// rows). Over-fetching gives the post-filter survivors to rank; recall still
    /// degrades when a filter/scope is very selective (see the exact-prefilter
    /// follow-up). Candidate→doc resolution is verified against the live index, so
    /// stale graph nodes (deleted/overwritten rows) are skipped.
    fn search_ann(&self, collections: &[&str], q: &[f32], opts: &SearchOpts) -> Vec<Hit> {
        let Some(ann) = self.ann.as_ref() else {
            return Vec::new();
        };
        if opts.top_k == 0 {
            return Vec::new();
        }
        let scope: std::collections::HashSet<&str> = collections.iter().copied().collect();
        let overscan = self.config.ann.map_or(1, |a| a.overscan).max(1);
        let n_candidates = opts.top_k.saturating_mul(overscan).max(opts.top_k);

        let candidates = ann.search(&self.data, q, n_candidates);

        let mut topk: TopK<(&str, &str)> = TopK::new(opts.top_k);
        for (row, score) in &candidates {
            // Resolve the candidate row to its owning doc via the reverse map, then
            // verify the doc still lives at this row (catches deletes/overwrites).
            let Some(Some((col_name, id))) = self.row_to_doc.get(*row as usize) else {
                continue;
            };
            if !scope.contains(col_name.as_str()) {
                continue;
            }
            let Some(col) = self.collections.get(col_name) else {
                continue;
            };
            let Some(entry) = col.docs.get(id) else {
                continue;
            };
            if entry.row != *row {
                continue; // stale reverse-map hint — row was overwritten
            }
            if !filter::matches(&opts.filter, &entry.attrs) {
                continue;
            }
            if let Some(min) = opts.min_score
                && *score < min
            {
                continue;
            }
            topk.offer(*score, (col_name.as_str(), id.as_str()));
        }

        topk.into_sorted_desc()
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
            .collect()
    }

    /// Exact f32 rerank of first-pass candidates → final ranked `Hit`s. Shared by both
    /// two-pass paths: the first pass is only an ordering proxy, so the true score (and
    /// `min_score`) is computed here from the original f32 vectors.
    fn rerank_candidates(
        &self,
        q: &[f32],
        candidates: &[(f32, (u64, &str, &str))],
        score_fn: fn(&[f32], &[f32]) -> f32,
        opts: &SearchOpts,
    ) -> Vec<Hit> {
        let mut topk: TopK<(&str, &str)> = TopK::new(opts.top_k);
        for (_, (row, col_name, id)) in candidates {
            let score = score_fn(q, self.data.row(*row));
            if let Some(min) = opts.min_score
                && score < min
            {
                continue;
            }
            topk.offer(score, (*col_name, *id));
        }
        topk.into_sorted_desc()
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
            .collect()
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

        // 5b. Rebuild the ANN index + reverse map (rows were renumbered).
        self.rebuild_ann();

        // 6. Rows were renumbered — drop the cached scan order.
        self.invalidate_scan_order();

        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::model::{Filter, Predicate, Quantization, Record, SearchOpts, Value};

    /// Extract the int8 state from a store's quant slot, panicking if it is off or binary.
    fn int8_state(store: &Store) -> &Int8State {
        match store
            .quant
            .as_ref()
            .expect("quantization should be enabled")
        {
            Quant::Int8(s) => s,
            Quant::Binary(_) => panic!("expected int8 quant state, found binary"),
        }
    }

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
            quant: None,
            ann: None,
            row_to_doc: Vec::new(),
            scan_order: std::sync::RwLock::new(None),
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

    // ── Euclidean distance tests ─────────────────────────────────────────

    #[test]
    fn euclidean_exact_match_scores_zero() {
        let mut store = Store::in_memory_with(3, Distance::Euclidean).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 2.0, 3.0])])
            .unwrap();
        let hits = store
            .search(&["col"], &[1.0, 2.0, 3.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].score.abs() < 1e-6,
            "identical vectors should score 0.0, got {}",
            hits[0].score
        );
    }

    #[test]
    fn euclidean_ranking_closer_first() {
        let mut store = Store::in_memory_with(3, Distance::Euclidean).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert(
                "col",
                &[
                    rec("close", vec![0.9, 0.1, 0.0]),
                    rec("far", vec![0.0, 1.0, 0.0]),
                ],
            )
            .unwrap();
        let hits = store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits[0].id, "close", "closer vector should rank first");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn euclidean_does_not_normalize() {
        let mut store = Store::in_memory_with(2, Distance::Euclidean).unwrap();
        store.create_collection("col").unwrap();
        store.upsert("col", &[rec("doc1", vec![3.0, 4.0])]).unwrap();
        let records = store.get_all("col");
        assert_eq!(records[0].vector, vec![3.0, 4.0], "raw vectors preserved");
    }

    #[test]
    fn euclidean_min_score_filters() {
        let mut store = Store::in_memory_with(2, Distance::Euclidean).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![10.0, 0.0])])
            .unwrap();
        let opts = SearchOpts {
            top_k: 5,
            filter: Filter::default(),
            min_score: Some(-1.0),
        };
        let hits = store.search(&["col"], &[0.0, 0.0], &opts).unwrap();
        assert!(
            hits.is_empty(),
            "score should be -100, below min_score of -1"
        );
    }

    // ── DotProduct distance tests ────────────────────────────────────────

    #[test]
    fn dotproduct_raw_dot() {
        let mut store = Store::in_memory_with(3, Distance::DotProduct).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert(
                "col",
                &[rec("a", vec![2.0, 0.0, 0.0]), rec("b", vec![1.0, 0.0, 0.0])],
            )
            .unwrap();
        let hits = store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits[0].id, "a", "higher magnitude should score higher");
        assert!(
            (hits[0].score - 2.0).abs() < 1e-6,
            "score = raw dot product"
        );
        assert!((hits[1].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dotproduct_does_not_normalize() {
        let mut store = Store::in_memory_with(2, Distance::DotProduct).unwrap();
        store.create_collection("col").unwrap();
        store.upsert("col", &[rec("doc1", vec![3.0, 4.0])]).unwrap();
        let records = store.get_all("col");
        assert_eq!(records[0].vector, vec![3.0, 4.0], "raw vectors preserved");
    }

    #[test]
    fn dotproduct_ranking_by_magnitude() {
        let mut store = Store::in_memory_with(2, Distance::DotProduct).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert(
                "col",
                &[rec("big", vec![10.0, 0.0]), rec("small", vec![1.0, 0.0])],
            )
            .unwrap();
        let hits = store
            .search(&["col"], &[1.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits[0].id, "big");
        assert!(hits[0].score > hits[1].score);
    }

    // ── Distance metric persistence tests ────────────────────────────────

    #[cfg_attr(miri, ignore)]
    #[test]
    fn euclidean_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        {
            let mut store =
                Store::open(Config::new(&path, 3).distance(Distance::Euclidean)).unwrap();
            store.create_collection("col").unwrap();
            store
                .upsert("col", &[rec("doc1", vec![1.0, 2.0, 3.0])])
                .unwrap();
        }
        {
            let store = Store::open(
                Config::new(&path, 3)
                    .distance(Distance::Euclidean)
                    .open_mode(OpenMode::ReadOnly),
            )
            .unwrap();
            let records = store.get_all("col");
            assert_eq!(records[0].vector, vec![1.0, 2.0, 3.0]);
            let hits = store
                .search(&["col"], &[1.0, 2.0, 3.0], &default_opts(5))
                .unwrap();
            assert!(hits[0].score.abs() < 1e-6);
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn distance_mismatch_on_reopen_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        {
            Store::open(Config::new(&path, 3).distance(Distance::Euclidean)).unwrap();
        }
        let res = Store::open(Config::new(&path, 3).distance(Distance::Cosine));
        assert!(res.is_err());
        let msg = res.err().unwrap().to_string();
        assert!(
            msg.contains("distance"),
            "error should mention distance: {msg}"
        );
    }

    // ── list (metadata-only query) tests ─────────────────────────────────

    #[test]
    fn list_returns_all_matching() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        let mut a_rust = BTreeMap::new();
        a_rust.insert("lang".to_string(), Value::Str("rust".to_string()));
        let mut a_go = BTreeMap::new();
        a_go.insert("lang".to_string(), Value::Str("go".to_string()));
        store
            .upsert(
                "col",
                &[
                    rec_with("r1", vec![1.0, 0.0, 0.0], a_rust.clone()),
                    rec_with("r2", vec![0.0, 1.0, 0.0], a_rust),
                    rec_with("g1", vec![0.0, 0.0, 1.0], a_go),
                ],
            )
            .unwrap();
        let filter = Filter(vec![Predicate::Eq(
            "lang".to_string(),
            Value::Str("rust".to_string()),
        )]);
        let hits = store.list(&["col"], &filter, 0, 100).unwrap();
        assert_eq!(hits.len(), 2);
        let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        assert!(ids.contains(&"r1"));
        assert!(ids.contains(&"r2"));
    }

    #[test]
    fn list_respects_limit() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("col").unwrap();
        for i in 0..10u32 {
            store
                .upsert("col", &[rec(&format!("d{i}"), vec![i as f32, 0.0])])
                .unwrap();
        }
        let hits = store.list(&["col"], &Filter::default(), 0, 3).unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn list_scores_are_zero() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("col").unwrap();
        store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
        let hits = store.list(&["col"], &Filter::default(), 0, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].score, 0.0);
    }

    #[test]
    fn list_empty_filter_returns_all() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("col").unwrap();
        store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
        store.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
        let hits = store.list(&["col"], &Filter::default(), 0, 100).unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn list_multi_collection() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("a").unwrap();
        store.create_collection("b").unwrap();
        store.upsert("a", &[rec("x", vec![1.0, 0.0])]).unwrap();
        store.upsert("b", &[rec("y", vec![0.0, 1.0])]).unwrap();
        let hits = store.list(&["a", "b"], &Filter::default(), 0, 100).unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn list_insertion_order() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("first", vec![1.0, 0.0])])
            .unwrap();
        store
            .upsert("col", &[rec("second", vec![0.0, 1.0])])
            .unwrap();
        let hits = store.list(&["col"], &Filter::default(), 0, 100).unwrap();
        assert_eq!(hits[0].id, "first");
        assert_eq!(hits[1].id, "second");
    }

    #[test]
    fn list_offset_paginates() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("col").unwrap();
        for i in 0..10u32 {
            store
                .upsert("col", &[rec(&format!("d{i}"), vec![i as f32, 0.0])])
                .unwrap();
        }
        // Page through in windows of 3; concatenating the pages reproduces the
        // full insertion-ordered list with no gaps or repeats.
        let mut paged: Vec<String> = Vec::new();
        for page in 0..4 {
            let hits = store
                .list(&["col"], &Filter::default(), page * 3, 3)
                .unwrap();
            paged.extend(hits.into_iter().map(|h| h.id));
        }
        let full: Vec<String> = store
            .list(&["col"], &Filter::default(), 0, 100)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(
            paged, full,
            "paginated windows must reconstruct the full list"
        );
        assert_eq!(paged.len(), 10);
    }

    #[test]
    fn list_offset_past_end_is_empty() {
        let mut store = Store::in_memory(2).unwrap();
        store.create_collection("col").unwrap();
        store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
        let hits = store.list(&["col"], &Filter::default(), 5, 10).unwrap();
        assert!(hits.is_empty());
    }

    // ── scan-order cache (nidus-dxt) ─────────────────────────────────────
    //
    // The whole-store fast path caches a row-sorted scan across queries; these pin
    // that it stays consistent with the doc set — i.e. every write that changes the
    // docs invalidates it, so a search after a write never reads a stale order.

    #[test]
    fn scan_cache_reflects_upsert_between_searches() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        // First search builds the cache.
        let hits = store
            .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 1);
        // A new doc lands on a fresh row — the cache must pick it up next query.
        store
            .upsert("col", &[rec("doc2", vec![0.0, 1.0, 0.0])])
            .unwrap();
        let hits = store
            .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 2, "second search must see the upserted doc");
        assert_eq!(hits[0].id, "doc2", "new doc is the nearest to the query");
    }

    #[test]
    fn scan_cache_reflects_delete_between_searches() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert(
                "col",
                &[
                    rec("doc1", vec![1.0, 0.0, 0.0]),
                    rec("doc2", vec![0.0, 1.0, 0.0]),
                ],
            )
            .unwrap();
        // Build the cache.
        assert_eq!(
            store
                .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
                .unwrap()
                .len(),
            2
        );
        // Delete and re-search: a stale cache would still rank the dead row.
        store.delete("col", &["doc1"]).unwrap();
        let hits = store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "doc2");
    }

    #[test]
    fn scan_cache_overwrite_uses_new_vector() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        // Build the cache against the original row.
        store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        // Overwrite doc1 — old row goes dead, new row is appended.
        store
            .upsert("col", &[rec("doc1", vec![0.0, 1.0, 0.0])])
            .unwrap();
        let hits = store
            .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            (hits[0].score - 1.0).abs() < 1e-6,
            "search must score the overwritten vector, not the dead row"
        );
    }

    #[test]
    fn scan_cache_survives_compact() {
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert(
                "col",
                &[
                    rec("a", vec![1.0, 0.0, 0.0]),
                    rec("b", vec![0.0, 1.0, 0.0]),
                    rec("c", vec![0.0, 0.0, 1.0]),
                ],
            )
            .unwrap();
        store.delete("col", &["b"]).unwrap();
        // Build the cache while a dead row exists.
        store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        // Compaction renumbers every live row — the cache must be rebuilt against them.
        store.compact().unwrap();
        let hits = store
            .search(&["col"], &[0.0, 0.0, 1.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "c");
    }

    #[test]
    fn scan_cache_whole_store_filter_matches_subset_path() {
        // The whole-store cache path filters via a per-entry attr lookup; the subset
        // path filters inline. Both must agree. Build one collection with attrs and
        // compare a filtered whole-store search against the same filter via subset.
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("col").unwrap();
        let tag = |t: &str| {
            let mut m = BTreeMap::new();
            m.insert("tag".to_string(), Value::Str(t.to_string()));
            m
        };
        store
            .upsert(
                "col",
                &[
                    rec_with("a", vec![1.0, 0.0, 0.0], tag("keep")),
                    rec_with("b", vec![0.9, 0.1, 0.0], tag("drop")),
                    rec_with("c", vec![0.8, 0.2, 0.0], tag("keep")),
                ],
            )
            .unwrap();
        let opts = SearchOpts {
            top_k: 5,
            filter: Filter(vec![Predicate::Eq(
                "tag".to_string(),
                Value::Str("keep".to_string()),
            )]),
            min_score: None,
        };
        let hits = store.search(&["col"], &[1.0, 0.0, 0.0], &opts).unwrap();
        let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "c"], "filter must keep only tagged docs");
    }

    #[test]
    fn scan_cache_subset_scope_excludes_other_collections() {
        // A strict subset scope takes the direct (non-cache) path; it must not leak
        // docs from out-of-scope collections, and the cache (built by a prior whole-
        // store search) must not interfere.
        let mut store = Store::in_memory(3).unwrap();
        store.create_collection("a").unwrap();
        store.create_collection("b").unwrap();
        store
            .upsert("a", &[rec("a1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        store
            .upsert("b", &[rec("b1", vec![0.0, 1.0, 0.0])])
            .unwrap();
        // Whole-store search builds the global cache.
        assert_eq!(
            store
                .search(&["a", "b"], &[1.0, 0.0, 0.0], &default_opts(5))
                .unwrap()
                .len(),
            2
        );
        // Subset search must see only collection "a".
        let hits = store
            .search(&["a"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "a1");
    }

    // ── int8 scalar quantization tests ───────────────────────────────────

    fn quantized_store(dim: usize) -> Store {
        Store::in_memory_cfg(
            Config::new("/dev/null/in-memory", dim)
                .open_mode(OpenMode::ReadWrite)
                .auto_compact(None)
                .quantization(Some(Quantization::default())),
        )
        .unwrap()
    }

    #[test]
    fn quantized_search_ranking_matches_exact() {
        let mut store = quantized_store(3);
        store.create_collection("col").unwrap();
        store
            .upsert(
                "col",
                &[
                    rec("close", vec![0.9, 0.1, 0.0]),
                    rec("mid", vec![0.5, 0.5, 0.0]),
                    rec("far", vec![0.0, 0.0, 1.0]),
                ],
            )
            .unwrap();
        let hits = store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(3))
            .unwrap();
        assert_eq!(
            hits[0].id, "close",
            "quantized search should rank correctly"
        );
    }

    #[test]
    fn quantized_search_respects_top_k() {
        let mut store = quantized_store(2);
        store.create_collection("col").unwrap();
        for i in 0..20u32 {
            store
                .upsert("col", &[rec(&format!("d{i}"), vec![i as f32, 0.0])])
                .unwrap();
        }
        let hits = store
            .search(&["col"], &[19.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 5);
    }

    #[test]
    fn quantized_search_with_filter() {
        let mut store = quantized_store(3);
        store.create_collection("col").unwrap();
        let mut a_rust = BTreeMap::new();
        a_rust.insert("lang".to_string(), Value::Str("rust".to_string()));
        let mut a_go = BTreeMap::new();
        a_go.insert("lang".to_string(), Value::Str("go".to_string()));
        store
            .upsert(
                "col",
                &[
                    rec_with("r1", vec![1.0, 0.0, 0.0], a_rust),
                    rec_with("g1", vec![1.0, 0.0, 0.0], a_go),
                ],
            )
            .unwrap();
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
        assert_eq!(hits[0].id, "r1");
    }

    #[test]
    fn quantized_search_euclidean() {
        let mut store = Store::in_memory_cfg(
            Config::new("/dev/null/in-memory", 3)
                .distance(Distance::Euclidean)
                .open_mode(OpenMode::ReadWrite)
                .auto_compact(None)
                .quantization(Some(Quantization::default())),
        )
        .unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert(
                "col",
                &[
                    rec("exact", vec![1.0, 2.0, 3.0]),
                    rec("far", vec![10.0, 20.0, 30.0]),
                ],
            )
            .unwrap();
        let hits = store
            .search(&["col"], &[1.0, 2.0, 3.0], &default_opts(2))
            .unwrap();
        assert_eq!(hits[0].id, "exact");
    }

    #[test]
    fn quantized_survives_compact() {
        let mut store = quantized_store(3);
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])])
            .unwrap();
        store
            .upsert("col", &[rec("a", vec![0.0, 1.0, 0.0])])
            .unwrap();
        store.compact().unwrap();
        let hits = store
            .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn quantized_empty_store_searches_ok() {
        let store = quantized_store(3);
        let hits = store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn quantized_incremental_matches_bulk() {
        // The int8 matrix must stay correct whether built in one batch or many.
        // Build the same data two ways and assert identical search rankings.
        let make = |incremental: bool| {
            let mut store = quantized_store(4);
            store.create_collection("col").unwrap();
            let recs: Vec<Record> = (0..50u32)
                .map(|i| {
                    let a = i as f32 * 0.01;
                    rec(&format!("d{i}"), vec![a, 1.0 - a, 0.5, -a])
                })
                .collect();
            if incremental {
                for r in &recs {
                    store.upsert("col", std::slice::from_ref(r)).unwrap();
                }
            } else {
                store.upsert("col", &recs).unwrap();
            }
            store
        };
        let bulk = make(false);
        let incr = make(true);
        let q = vec![0.2, 0.8, 0.5, -0.2];
        let hb = bulk.search(&["col"], &q, &default_opts(10)).unwrap();
        let hi = incr.search(&["col"], &q, &default_opts(10)).unwrap();
        let ids_b: Vec<&str> = hb.iter().map(|h| h.id.as_str()).collect();
        let ids_i: Vec<&str> = hi.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids_b, ids_i, "incremental and bulk must rank identically");
    }

    #[test]
    fn quantized_incremental_keeps_full_recall() {
        // Drip-feed rows one at a time, then confirm an exact-match query still
        // finds its target (incremental quantization must not lose the vector).
        let mut store = quantized_store(3);
        store.create_collection("col").unwrap();
        for i in 0..30u32 {
            let v = vec![i as f32, (30 - i) as f32, 1.0];
            store.upsert("col", &[rec(&format!("d{i}"), v)]).unwrap();
        }
        // Query exactly matches d7.
        let hits = store
            .search(&["col"], &[7.0, 23.0, 1.0], &default_opts(1))
            .unwrap();
        assert_eq!(hits[0].id, "d7");
    }

    #[test]
    fn quantized_refit_tracks_row_growth() {
        // params_rows must follow the geometric-refit rule: it only jumps when the
        // row count crosses REFIT_GROWTH× the last fit set, not on every batch.
        let mut store = quantized_store(2);
        store.create_collection("col").unwrap();
        // First batch (2 rows): refit from 0 → params_rows = 2.
        store
            .upsert("col", &[rec("a", vec![1.0, 0.0]), rec("b", vec![0.0, 1.0])])
            .unwrap();
        assert_eq!(int8_state(&store).params_rows, 2);
        // One more row (total 3): 3 <= 2*2, so NO refit — params_rows stays 2.
        store.upsert("col", &[rec("c", vec![1.0, 1.0])]).unwrap();
        assert_eq!(int8_state(&store).params_rows, 2);
        // Push past 2*2=4 (total 5): refit fires → params_rows = 5.
        store
            .upsert("col", &[rec("d", vec![2.0, 0.0]), rec("e", vec![0.0, 2.0])])
            .unwrap();
        assert_eq!(int8_state(&store).params_rows, 5);
        // The int8 matrix always covers every physical row.
        let dim = store.data.dimension();
        assert_eq!(
            int8_state(&store).vectors.len(),
            store.data.row_count() as usize * dim
        );
    }

    // ── binary (sign-bit) quantization tests ─────────────────────────────

    /// A deterministic xorshift pseudo-random vector in roughly [-0.5, 0.5)^dim, for
    /// recall/parallel tests where structured modulo data would produce Hamming ties.
    fn pseudo_vec(seed: u64, dim: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..dim)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn binary_store(dim: usize) -> Store {
        Store::in_memory_cfg(
            Config::new("/dev/null/in-memory", dim)
                .distance(Distance::Cosine)
                .open_mode(OpenMode::ReadWrite)
                .auto_compact(None)
                .quantization(Some(Quantization::binary())),
        )
        .unwrap()
    }

    /// Extract the binary state, panicking if quant is off or int8.
    fn bin_state(store: &Store) -> &BinState {
        match store
            .quant
            .as_ref()
            .expect("quantization should be enabled")
        {
            Quant::Binary(s) => s,
            Quant::Int8(_) => panic!("expected binary quant state, found int8"),
        }
    }

    #[test]
    fn binary_rejects_non_cosine() {
        // Sign codes are an angular proxy; binary must refuse dot-product / Euclidean.
        for distance in [Distance::DotProduct, Distance::Euclidean] {
            let result = Store::in_memory_cfg(
                Config::new("/dev/null/in-memory", 4)
                    .distance(distance)
                    .open_mode(OpenMode::ReadWrite)
                    .quantization(Some(Quantization::binary())),
            );
            let err = match result {
                Ok(_) => panic!("binary quantization must be rejected for {distance:?}"),
                Err(e) => e,
            };
            assert!(
                err.to_string()
                    .contains("binary quantization requires Distance::Cosine"),
                "expected cosine-only rejection, got: {err}"
            );
        }
        // Cosine is accepted.
        assert!(
            Store::in_memory_cfg(
                Config::new("/dev/null/in-memory", 4)
                    .distance(Distance::Cosine)
                    .open_mode(OpenMode::ReadWrite)
                    .quantization(Some(Quantization::binary())),
            )
            .is_ok()
        );
    }

    #[test]
    fn binary_search_ranks_correctly() {
        let mut store = binary_store(3);
        store.create_collection("col").unwrap();
        store
            .upsert(
                "col",
                &[
                    rec("close", vec![0.9, 0.1, 0.0]),
                    rec("mid", vec![0.6, 0.5, 0.1]),
                    rec("far", vec![-1.0, -0.2, 0.3]),
                ],
            )
            .unwrap();
        let hits = store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(3))
            .unwrap();
        assert_eq!(
            hits[0].id, "close",
            "binary first-pass + f32 rerank should rank correctly"
        );
        // The reranked score is the exact f32 cosine, not a Hamming proxy.
        assert!(hits[0].score <= 1.0 + 1e-6 && hits[0].score >= -1.0 - 1e-6);
    }

    #[test]
    fn binary_state_covers_all_rows_multiword() {
        // dim 130 → 3 u64 words per row; words must cover every physical row.
        let mut store = binary_store(130);
        store.create_collection("col").unwrap();
        for i in 0..7u32 {
            store
                .upsert(
                    "col",
                    &[rec(&format!("d{i}"), pseudo_vec(i as u64 + 1, 130))],
                )
                .unwrap();
        }
        assert_eq!(bin_state(&store).words_per_row, 130usize.div_ceil(64)); // == 3
        assert_eq!(
            bin_state(&store).words.len(),
            store.data.row_count() as usize * 3
        );
    }

    // Ignored under Miri: builds thousands of rows to make recall meaningful — far too
    // slow at Miri's ~100x. Pure in-RAM logic, covered amply by the f32/serial path.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn binary_search_recall_high_vs_exact() {
        let dim = 128;
        let n = 2000usize;
        let k = 10usize;
        let mut exact = Store::in_memory_with(dim, Distance::Cosine).unwrap();
        let mut bin = binary_store(dim);
        exact.create_collection("c").unwrap();
        bin.create_collection("c").unwrap();
        for i in 0..n {
            let r = rec(&format!("d{i}"), pseudo_vec(i as u64 + 1, dim));
            exact.upsert("c", std::slice::from_ref(&r)).unwrap();
            bin.upsert("c", &[r]).unwrap();
        }
        let (mut hit, mut total) = (0usize, 0usize);
        for qi in 0..20u64 {
            let q = pseudo_vec(1_000_000 + qi, dim);
            let truth: Vec<String> = exact
                .search(&["c"], &q, &default_opts(k))
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect();
            let got: std::collections::HashSet<String> = bin
                .search(&["c"], &q, &default_opts(k))
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect();
            for id in &truth {
                if got.contains(id) {
                    hit += 1;
                }
                total += 1;
            }
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.6, "binary recall@{k} too low: {recall:.3}");
    }

    /// Build a binary-quantized store with `n` pseudo-random rows and the given threads.
    fn binary_pseudo_store(dim: usize, n: usize, threads: usize) -> Store {
        let mut store = Store::in_memory_cfg(
            Config::new("/dev/null/in-memory", dim)
                .distance(Distance::Cosine)
                .open_mode(OpenMode::ReadWrite)
                .auto_compact(None)
                .query_threads(threads)
                .quantization(Some(Quantization::binary())),
        )
        .unwrap();
        store.create_collection("col").unwrap();
        let recs: Vec<Record> = (0..n)
            .map(|i| rec(&format!("d{i}"), pseudo_vec(i as u64 + 1, dim)))
            .collect();
        store.upsert("col", &recs).unwrap();
        store
    }

    // Ignored under Miri — needs to clear PARALLEL_SCAN_WORK_FLOOR to engage threads.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn binary_parallel_matches_serial() {
        // Pseudo-random sign codes make Hamming ties near the overscan boundary
        // vanishingly unlikely, so serial and parallel select the same candidates and
        // rerank to byte-identical ordered results.
        let dim = 768;
        let n = rows_to_parallelize(dim) + 100;
        let serial = binary_pseudo_store(dim, n, 1);
        let parallel = binary_pseudo_store(dim, n, 4);
        let q = pseudo_vec(7_000_001, dim);
        let hs: Vec<String> = serial
            .search(&["col"], &q, &default_opts(20))
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let hp: Vec<String> = parallel
            .search(&["col"], &q, &default_opts(20))
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(hs, hp, "binary parallel scan must match serial");
    }

    // ── parallel scan tests ──────────────────────────────────────────────

    /// Rows needed at `dim` to clear [`PARALLEL_SCAN_WORK_FLOOR`], so the threaded path
    /// actually engages. Keeps the parallel tests robust to the constant's value (and
    /// fast: a wide dim hits the work floor at far fewer rows than a narrow one).
    fn rows_to_parallelize(dim: usize) -> usize {
        PARALLEL_SCAN_WORK_FLOOR.div_ceil(dim) + 1
    }

    /// Build an in-memory store with `n` deterministic pseudo-random rows, the given
    /// `query_threads`, and optional int8 quantization.
    fn threaded_store_cfg(dim: usize, n: usize, threads: usize, quant: bool) -> Store {
        let mut cfg = Config::new("/dev/null/in-memory", dim)
            .open_mode(OpenMode::ReadWrite)
            .auto_compact(None)
            .query_threads(threads);
        if quant {
            cfg = cfg.quantization(Some(Quantization::default()));
        }
        let mut store = Store::in_memory_cfg(cfg).unwrap();
        store.create_collection("col").unwrap();
        let recs: Vec<Record> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| ((i * 31 + d * 7) % 97) as f32 - 48.0)
                    .collect();
                rec(&format!("d{i}"), v)
            })
            .collect();
        store.upsert("col", &recs).unwrap();
        store
    }

    fn threaded_store(dim: usize, n: usize, threads: usize) -> Store {
        threaded_store_cfg(dim, n, threads, false)
    }

    // Ignored under Miri: needs enough work to clear PARALLEL_SCAN_WORK_FLOOR to engage
    // the threaded path, which Miri runs at ~100x slowdown (minutes). The thread::scope
    // scan is `#![forbid(unsafe_code)]` safe Rust over shared `&` reads — the borrow
    // checker already proves it data-race-free, so Miri adds no coverage here.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn parallel_search_matches_serial() {
        // A wide dim clears the work floor at ~1.4k rows — far cheaper than narrow dims.
        let dim = 768;
        let n = rows_to_parallelize(dim) + 100; // exceed the floor so threading engages
        let serial = threaded_store(dim, n, 1);
        let parallel = threaded_store(dim, n, 4);
        let q: Vec<f32> = (0..dim).map(|d| (d * 5 % 13) as f32 - 6.0).collect();
        let hs = serial.search(&["col"], &q, &default_opts(20)).unwrap();
        let hp = parallel.search(&["col"], &q, &default_opts(20)).unwrap();
        assert_eq!(hs.len(), hp.len());
        // The sorted score sequence must be byte-identical (exact f32 over the same
        // data); only tie-breaking among equal scores may differ.
        for (a, b) in hs.iter().zip(&hp) {
            assert!(
                (a.score - b.score).abs() < 1e-6,
                "score mismatch: serial {} vs parallel {}",
                a.score,
                b.score
            );
        }
    }

    // Ignored under Miri — same reason as `parallel_search_matches_serial`.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn parallel_search_respects_filter_and_min_score() {
        let dim = 768;
        let n = rows_to_parallelize(dim) + 100;
        let parallel = threaded_store(dim, n, 4);
        let q: Vec<f32> = (0..dim).map(|d| (d * 5 % 13) as f32 - 6.0).collect();
        // A min_score floor must be honored across all worker chunks.
        let opts = SearchOpts {
            top_k: 30,
            filter: Filter::default(),
            min_score: Some(0.99),
        };
        let hits = parallel.search(&["col"], &q, &opts).unwrap();
        assert!(hits.iter().all(|h| h.score >= 0.99));
    }

    // The quantized first pass scales across threads; its parallel and serial candidate
    // sets must produce the same final ranking. Ignored under Miri (same cost reason).
    #[cfg_attr(miri, ignore)]
    #[test]
    fn parallel_quantized_matches_serial() {
        let dim = 768;
        let n = rows_to_parallelize(dim) + 100;
        let serial = threaded_store_cfg(dim, n, 1, true);
        let parallel = threaded_store_cfg(dim, n, 4, true);
        let q: Vec<f32> = (0..dim).map(|d| (d * 5 % 13) as f32 - 6.0).collect();
        let hs = serial.search(&["col"], &q, &default_opts(20)).unwrap();
        let hp = parallel.search(&["col"], &q, &default_opts(20)).unwrap();
        assert_eq!(hs.len(), hp.len());
        // Same int8 candidate set (just scored in chunks) → same f32 rerank scores.
        for (a, b) in hs.iter().zip(&hp) {
            assert!(
                (a.score - b.score).abs() < 1e-6,
                "score mismatch: serial {} vs parallel {}",
                a.score,
                b.score
            );
        }
    }

    #[test]
    fn parallel_below_floor_falls_back_to_serial() {
        // Fewer rows than the floor: the parallel branch is skipped, but results
        // must still be correct.
        let store = threaded_store(4, 10, 8);
        let hits = store
            .search(&["col"], &[1.0, 0.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 5);
        // Scores are non-increasing.
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn parallel_search_with_quantization() {
        // query_threads is set and quantization is on, but the scan is below the work
        // floor: the quantized path runs single-threaded and must still be correct.
        let store = threaded_store_cfg(8, 200, 4, true);
        let q: Vec<f32> = (0..8).map(|d| (d * 2 % 7) as f32).collect();
        let hits = store.search(&["col"], &q, &default_opts(10)).unwrap();
        assert_eq!(hits.len(), 10);
    }

    // ── ANN ─────────────────────────────────────────────────────────────────────

    use crate::ann::SplitMix64;
    use crate::model::AnnConfig;

    /// `n` deterministic random unit vectors of dimension `dim`.
    fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = SplitMix64::new(seed);
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
                    .collect();
                normalize(&mut v);
                v
            })
            .collect()
    }

    fn ann_store(dim: usize, cfg: AnnConfig, vectors: &[Vec<f32>]) -> Store {
        let mut s = Store::in_memory_cfg(
            Config::new("/dev/null/in-memory", dim)
                .auto_compact(None)
                .ann(Some(cfg)),
        )
        .unwrap();
        let recs: Vec<Record> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
            .collect();
        s.upsert("col", &recs).unwrap();
        s
    }

    fn exact_store(dim: usize, vectors: &[Vec<f32>]) -> Store {
        let mut s = Store::in_memory(dim).unwrap();
        let recs: Vec<Record> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
            .collect();
        s.upsert("col", &recs).unwrap();
        s
    }

    /// Mean recall@k of `ann` against the exact brute-force `truth` over `queries`.
    fn mean_recall(ann: &Store, truth: &Store, queries: &[Vec<f32>], k: usize) -> f32 {
        let mut total = 0.0f32;
        for q in queries {
            let exact: std::collections::HashSet<String> = truth
                .search(&["col"], q, &default_opts(k))
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect();
            let got = ann.search(&["col"], q, &default_opts(k)).unwrap();
            let hit = got.iter().filter(|h| exact.contains(&h.id)).count();
            total += hit as f32 / k as f32;
        }
        total / queries.len() as f32
    }

    #[test]
    #[cfg_attr(miri, ignore)] // N=2000 build is too slow under Miri; logic is covered in ann/.
    fn hnsw_recall_matches_exact() {
        let (n, dim, k) = (2000, 32, 10);
        let data = random_unit_vectors(n, dim, 1);
        let queries = random_unit_vectors(50, dim, 2);
        let ann = ann_store(dim, AnnConfig::hnsw(), &data);
        let truth = exact_store(dim, &data);
        let recall = mean_recall(&ann, &truth, &queries, k);
        assert!(
            recall >= 0.90,
            "HNSW recall@{k} = {recall:.3}, expected >= 0.90"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn ivf_recall_matches_exact() {
        let (n, dim, k) = (2000, 32, 10);
        let data = random_unit_vectors(n, dim, 3);
        let queries = random_unit_vectors(50, dim, 4);
        // Probe a generous fraction of lists so recall is solid.
        let ann = ann_store(dim, AnnConfig::ivf().n_probe(12), &data);
        let truth = exact_store(dim, &data);
        let recall = mean_recall(&ann, &truth, &queries, k);
        assert!(
            recall >= 0.70,
            "IVF recall@{k} = {recall:.3}, expected >= 0.70"
        );
    }

    /// Small-N correctness that stays Miri-clean (no fsync, tiny build).
    #[test]
    fn ann_finds_exact_match_small() {
        let data = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        for cfg in [AnnConfig::hnsw(), AnnConfig::ivf().n_probe(8)] {
            let s = ann_store(3, cfg, &data);
            let hits = s
                .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(1))
                .unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].id, "d1", "{cfg:?} should find the exact match");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // N=200 HNSW build is slow under Miri; tiny cases cover the path.
    fn ann_post_filter_returns_only_matching() {
        // Half the docs carry kind=a, half kind=b; an ANN query filtered to kind=b must
        // never return a kind=a doc.
        let dim = 16;
        let data = random_unit_vectors(200, dim, 5);
        let mut s = Store::in_memory_cfg(
            Config::new("/dev/null/in-memory", dim)
                .auto_compact(None)
                .ann(Some(AnnConfig::hnsw().overscan(8))),
        )
        .unwrap();
        let recs: Vec<Record> = data
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut attrs = BTreeMap::new();
                let kind = if i % 2 == 0 { "a" } else { "b" };
                attrs.insert("kind".to_string(), Value::Str(kind.to_string()));
                rec_with(&format!("d{i}"), v.clone(), attrs)
            })
            .collect();
        s.upsert("col", &recs).unwrap();

        let opts = SearchOpts {
            top_k: 10,
            filter: Filter(vec![Predicate::Eq(
                "kind".to_string(),
                Value::Str("b".to_string()),
            )]),
            min_score: None,
        };
        let hits = s.search(&["col"], &data[1], &opts).unwrap();
        assert!(!hits.is_empty(), "filtered ANN should still return results");
        for h in &hits {
            // d1, d3, … are odd indices = kind b.
            let idx: usize = h.id.trim_start_matches('d').parse().unwrap();
            assert_eq!(idx % 2, 1, "{} leaked into a kind=b query", h.id);
        }
    }

    #[test]
    fn ann_skips_deleted_rows() {
        let data = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let mut s = ann_store(3, AnnConfig::hnsw(), &data);
        // Delete the nearest doc to a +x query; its graph node is now stale.
        s.delete("col", &["d0"]).unwrap();
        let hits = s
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(3))
            .unwrap();
        assert!(
            hits.iter().all(|h| h.id != "d0"),
            "deleted doc must not appear: {hits:?}"
        );
        // The next-nearest live doc should now lead.
        assert_eq!(hits[0].id, "d1");
    }

    #[test]
    fn ann_rejects_combination_with_quantization() {
        let result = Store::in_memory_cfg(
            Config::new("/dev/null/in-memory", 4)
                .ann(Some(AnnConfig::hnsw()))
                .quantization(Some(Quantization::default())),
        );
        let err = match result {
            Ok(_) => panic!("expected ann+quantization to be rejected"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("cannot both be set"),
            "unexpected error: {err}"
        );
    }
}
