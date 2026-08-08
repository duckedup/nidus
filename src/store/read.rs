//! Reads and search: cheap accessors, the row-sorted scan plumbing that feeds every query the
//! data matrix in storage order, exact f32 brute force, and the approximate `search_ann`. The
//! quantized first pass lives in [`super::quant`].

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Result, bail};

use super::scoring::{PARALLEL_SCAN_WORK_FLOOR, parallel_topk, score_chunk};
use super::{ScanOrder, Store, oom};
use crate::ann::Walk;
use crate::config::Config;
use crate::filter;
use crate::fts::Analyzer;
use crate::fuse::{FusionLeg, rrf_fuse};
use crate::model::{
    AnnConfig, Distance, Filter, Footprint, FtsQuery, Hit, HybridOpts, ListOpts, SearchOpts,
};
use crate::search::{TopK, dot, euclidean_neg_sq, normalize};

/// The options a search *path* runs with: rank `offset + top_k` deep, then let the caller's
/// single tail drop `offset`. `offset` is zeroed so no inner path can paginate a second time.
fn deepened(opts: &SearchOpts) -> SearchOpts {
    SearchOpts {
        top_k: opts.offset.saturating_add(opts.top_k),
        offset: 0,
        ..opts.clone()
    }
}

/// Drop the first `offset` ranked hits — the ONE place a page boundary is cut. An offset past
/// the end is an empty page, never an error: a caller walking pages must be able to stop.
fn paginate(mut hits: Vec<Hit>, offset: usize) -> Vec<Hit> {
    if offset == 0 {
        return hits;
    }
    if offset >= hits.len() {
        return Vec::new();
    }
    hits.drain(..offset);
    hits
}

impl Store {
    // ── Cheap accessors ─────────────────────────────────────────────────────────

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

    // `get_all` materializes the whole collection into a fresh Vec and is not fallible, so an
    // OOM here can still abort. Making it `Result` would break the public API for a bulk-read
    // convenience; huge collections should prefer `search`. The write/open paths are fallible.
    pub fn get_all(&self, collection: &str) -> Vec<crate::model::Record> {
        let Some(col) = self.collections.get(collection) else {
            return Vec::new();
        };

        col.docs
            .iter()
            .map(|(id, entry)| crate::model::Record {
                id: id.clone(),
                // Text-only docs (row None) have no embedding.
                vector: entry.row.map(|r| self.data.row(r).to_vec()),
                attrs: entry.attrs.clone(),
            })
            .collect()
    }

    /// List records matching `opts.filter` across `collections`, without vector scoring.
    pub fn list(&self, collections: &[&str], opts: &ListOpts) -> Result<Vec<Hit>> {
        let cap: usize = collections
            .iter()
            .filter_map(|c| self.collections.get(*c))
            .map(|c| c.docs.len())
            .sum();
        let mut scan: Vec<(Option<u64>, &str, &str)> = Vec::new();
        scan.try_reserve(cap)
            .map_err(|_| oom("list scan buffer", cap))?;
        for &col_name in collections {
            let Some(col) = self.collections.get(col_name) else {
                continue;
            };
            for (id, entry) in &col.docs {
                if !filter::matches(&opts.filter, &entry.attrs) {
                    continue;
                }
                scan.push((entry.row, col_name, id.as_str()));
            }
        }
        // Rowed docs by row, then text-only docs by id — a stable order for pagination.
        scan.sort_unstable_by(|a, b| match (a.0, b.0) {
            (Some(x), Some(y)) => x.cmp(&y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.2.cmp(b.2),
        });
        let results = scan
            .iter()
            .skip(opts.offset)
            .take(opts.limit)
            .map(|&(_, collection, id)| {
                let attrs = self
                    .collections
                    .get(collection)
                    .and_then(|c| c.docs.get(id))
                    .map(|e| e.attrs.clone())
                    .unwrap_or_default();
                Hit::new(collection, id, 0.0, attrs)
            })
            .collect();
        Ok(results)
    }

    // ── Scan plumbing ─────────────────────────────────────────────────────────

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

    /// Total scannable (vector-bearing) docs across all collections — the scan-order cache's
    /// length, and the yardstick for "does this scope cover every vector doc?". Text-only docs
    /// carry no row and never enter a vector scan.
    fn scannable_doc_count(&self) -> usize {
        self.collections
            .values()
            .flat_map(|c| c.docs.values())
            .filter(|e| e.row.is_some())
            .count()
    }

    /// Scannable (vector-bearing) docs within `collections` — the in-scope half of the
    /// whole-store yardstick and the ANN selectivity population.
    fn scannable_in_scope(&self, collections: &[&str]) -> usize {
        collections
            .iter()
            .filter_map(|c| self.collections.get(*c))
            .flat_map(|c| c.docs.values())
            .filter(|e| e.row.is_some())
            .count()
    }

    /// Drop the cached scan order. Called from every write that changes the doc set
    /// (`upsert`, `delete`, `drop_collection`, `compact`); `&mut self`, so it takes the
    /// lock uncontended via `get_mut` and clears even a poisoned lock.
    pub(super) fn invalidate_scan_order(&mut self) {
        *self.scan_order.get_mut().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// A read guard over the cached row-sorted scan order, rebuilding it first if stale.
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
                let n = self.scannable_doc_count();
                let mut order: ScanOrder = Vec::new();
                order
                    .try_reserve_exact(n)
                    .map_err(|_| oom("scan-order cache", n))?;
                for (col_name, col) in &self.collections {
                    for (id, entry) in &col.docs {
                        // Only vector-bearing docs belong in the scan order.
                        if let Some(row) = entry.row {
                            order.push((row, col_name.clone(), id.clone()));
                        }
                    }
                }
                order.sort_unstable_by_key(|&(row, _, _)| row);
                *w = Some(order);
            }
        }
        Ok(self.scan_order.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// Build the in-scope, filter-passing scan **in row order** and hand it to `f`.
    fn with_sorted_scan<R>(
        &self,
        collections: &[&str],
        filter: &Filter,
        f: impl for<'b> FnOnce(&mut [(u64, &'b str, &'b str)]) -> Result<R>,
    ) -> Result<R> {
        // Count only vector-bearing docs — text-only docs never enter a vector scan.
        let scan_cap: usize = self.scannable_in_scope(collections);
        let mut scan: Vec<(u64, &str, &str)> = Vec::new();
        scan.try_reserve(scan_cap)
            .map_err(|_| oom("search scan buffer", scan_cap))?;

        if scan_cap == self.scannable_doc_count() {
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
                    let Some(row) = entry.row else { continue };
                    if !filter::matches(filter, &entry.attrs) {
                        continue;
                    }
                    scan.push((row, col_name, id.as_str()));
                }
            }
            scan.sort_unstable_by_key(|&(row, _, _)| row);
            f(&mut scan)
        }
    }

    // ── Search ──────────────────────────────────────────────────────────────────

    /// Reject a query vector whose length is not the store's pinned dimension (nidus-c5v).
    fn check_query_dim(&self, query: &[f32]) -> Result<()> {
        let dim = self.data.dimension();
        if query.len() != dim {
            bail!(
                "query length {} does not match store dimension {}",
                query.len(),
                dim
            );
        }
        Ok(())
    }

    /// Brute-force search over the union of `collections`, merged into one ranking
    /// (one bounded top-k heap fed by every in-scope collection). The scoring function
    /// is determined by the store's [`Distance`] metric.
    pub fn search(
        &self,
        collections: &[&str],
        query: &[f32],
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        // Before the metric: a request the store refuses is not a query it served, so
        // counting it would overstate `search_queries` and skew every ratio built on it.
        self.check_query_dim(query)?;

        // Which path served a query is the difference between "queries are slow" and
        // "queries are slow BECAUSE the index is not being used" (nidus-abx.4). One relaxed
        // atomic add per query, off the per-vector inner loop entirely.
        let m = crate::metrics::metrics();
        m.search_queries.inc();

        let mut q = query.to_vec();
        if self.config.distance == Distance::Cosine {
            normalize(&mut q);
        }

        let score_fn: fn(&[f32], &[f32]) -> f32 = match self.config.distance {
            Distance::Cosine | Distance::DotProduct => dot,
            Distance::Euclidean => euclidean_neg_sq,
        };

        // Each branch ranks `offset + top_k` deep and hands the ranking to ONE tail that drops
        // `offset`. Single-tail on purpose: a branch that paginated itself — or that applied the
        // offset before the top-k cap — still compiles and is silently wrong (nidus-m50.8).
        let deep = deepened(opts);
        let ranked = if self.ann.is_some() {
            // ANN: walk the index for an over-fetched candidate set, then post-filter and rerank —
            // recall traded for speed. A selective filter/scope can starve the walk, so `search_ann`
            // falls back to an exact prefilter when survivors are few (nidus-0ou).
            m.search_ann.inc();
            self.search_ann(collections, &q, &deep, score_fn)?
        } else if self.seg_indexes.iter().any(Option::is_some) {
            // Per-segment fan-out: walk each cold segment's IVF index and brute-force the tail (the
            // active segment plus any sub-threshold sealed one), merged into one ranking (SPEC
            // §14.3). Engaged only once a sealed segment has crossed `segment_index_min_rows`.
            m.search_segmented.inc();
            self.search_segmented(collections, &q, &deep, score_fn)?
        } else {
            // Gather in-scope, filter-passing rows in physical-row order, for sequential `data`
            // access (nidus-33k). `with_sorted_scan` reuses the cached whole-store order where it
            // can, so the sort is not redone every query (nidus-dxt).
            self.with_sorted_scan(collections, &deep.filter, |scan| {
                // Only the brute-force paths reach here, which is exactly why the counter lives
                // here: "rows scanned" is a meaningful cost on a linear scan and meaningless on
                // an ANN walk, so counting it in one place keeps the metric honest.
                m.search_vectors_scanned.add(scan.len() as u64);

                // Decide once whether this query splits across workers (configured threads +
                // enough scan work to amortize spawn cost).
                let workers = self.parallel_workers(scan.len());

                // Two-pass quantized search if enabled and the quantized matrix is populated;
                // otherwise the standard exact f32 brute-force path.
                if let Some(res) = self.search_quantized(&q, scan, &deep, score_fn, workers) {
                    m.search_quantized.inc();
                    return res;
                }
                m.search_exact.inc();
                self.rank_scan(&q, scan, score_fn, &deep)
            })?
        };
        Ok(paginate(ranked, opts.offset))
    }

    /// Full-text (BM25) search over `collections`, reusing the same `Hit`/`Filter`/top-k
    /// machinery as vector `search`. `min_score` here is a raw BM25 floor, not cosine. Text-only
    /// and vector-bearing docs are both eligible; ties break on `(collection, id)`.
    pub fn text_search(
        &self,
        collections: &[&str],
        query: &FtsQuery,
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        if opts.top_k == 0 {
            return Ok(Vec::new());
        }
        // Same shape as `search`: rank `offset + top_k` deep, cut the page in one place at the end.
        let mut topk: TopK<(&str, &str)> = TopK::new(opts.offset.saturating_add(opts.top_k));
        // Analyze the query text once per distinct field analyzer across the scope
        // (collections usually share one), not once per collection.
        let mut analyzed: HashMap<Analyzer, Vec<String>> = HashMap::new();
        for &col_name in collections {
            let Some(col) = self.collections.get(col_name) else {
                continue;
            };
            let Some(cfg) = self.fts.field_analyzer(col_name, &query.field) else {
                continue; // this collection doesn't full-text-index the field
            };
            analyzed
                .entry(cfg)
                .or_insert_with(|| crate::fts::analyze(&query.text, cfg));
            let terms = &analyzed[&cfg];
            for (id, score) in self.fts.score(col_name, &query.field, terms) {
                if let Some(min) = opts.min_score
                    && score < min
                {
                    continue;
                }
                // Hint-verify the id against the live index and apply the metadata
                // filter (the FTS index can lag a delete until the next rebuild).
                let Some(entry) = col.docs.get(id) else {
                    continue;
                };
                if !filter::matches(&opts.filter, &entry.attrs) {
                    continue;
                }
                topk.offer(score, (col_name, id));
            }
        }
        // `TopK` already resolves ties on `(collection, id)`, so no re-sort is needed here.
        Ok(paginate(self.hits_from_topk(topk), opts.offset))
    }

    /// Hybrid search: fuse a vector and a BM25 leg with Reciprocal Rank Fusion. Each leg runs
    /// independently `candidates` deep, then a doc's fused score is the sum of
    /// `1 / (rrf_k + rank + 1)`; a doc in only one leg is carried by it. Ties break on `(collection, id)`.
    pub fn hybrid_search(
        &self,
        collections: &[&str],
        vector: &[f32],
        text: &FtsQuery,
        opts: &HybridOpts,
    ) -> Result<Vec<Hit>> {
        // Ahead of the `top_k == 0` shortcut, not after: the vector leg validates, but the
        // shortcut returns before the leg runs. Validating here means a bad query does not change
        // verdict based on `top_k`.
        self.check_query_dim(vector)?;

        if opts.top_k == 0 {
            return Ok(Vec::new());
        }
        // Pull each leg at least a full page deep (`offset + top_k`) so fusion can fill it.
        let page = opts.offset.saturating_add(opts.top_k);
        let leg_opts = SearchOpts {
            top_k: opts.candidates.max(page),
            filter: opts.filter.clone(),
            ..Default::default()
        };
        let vector_leg = self.search(collections, vector, &leg_opts)?;
        let text_leg = self.text_search(collections, text, &leg_opts)?;

        // The per-leg detail is dropped here; `hybrid_search` reports only the fused score.
        let fused = rrf_fuse(
            vec![FusionLeg::new(vector_leg), FusionLeg::new(text_leg)],
            opts.rrf_k,
        );
        let hits: Vec<Hit> = fused.into_iter().map(|(hit, _per_leg)| hit).collect();
        // The page is cut on the *fused* ranking, never per leg — a leg's rank is an input to
        // the fused score, so paginating a leg would change which documents fuse at all.
        let mut hits = paginate(hits, opts.offset);
        hits.truncate(opts.top_k);
        Ok(hits)
    }

    /// Score an already-gathered, filter-passing scan exactly into ranked [`Hit`]s — the shared
    /// tail of the brute-force path and the ANN exact-prefilter fallback. Splits across workers
    /// once the scan clears the parallel floor, else scores serially, for the same top-k.
    fn rank_scan<'b>(
        &self,
        q: &[f32],
        scan: &mut [(u64, &'b str, &'b str)],
        score_fn: fn(&[f32], &[f32]) -> f32,
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        let workers = self.parallel_workers(scan.len());
        let topk = if workers > 1 {
            parallel_topk(scan, workers, opts.top_k, |chunk| {
                score_chunk(&self.data, chunk, q, score_fn, opts.top_k, opts.min_score)
            })?
        } else {
            score_chunk(&self.data, scan, q, score_fn, opts.top_k, opts.min_score)?
        };
        Ok(self.hits_from_topk(topk))
    }

    /// Resolve a bounded top-k of `(collection, id)` into ranked [`Hit`]s, cloning each
    /// winner's attrs from the live index. Shared by every search path.
    pub(super) fn hits_from_topk<'b>(&self, topk: TopK<(&'b str, &'b str)>) -> Vec<Hit> {
        topk.into_sorted_desc()
            .into_iter()
            .map(|(score, (collection, id))| {
                let attrs = self
                    .collections
                    .get(collection)
                    .and_then(|c| c.docs.get(id))
                    .map(|e| e.attrs.clone())
                    .unwrap_or_default();
                Hit::new(collection, id, score, attrs)
            })
            .collect()
    }

    /// ANN search: walk the index for `top_k × overscan` candidate rows, resolve each to its doc,
    /// keep those in scope and passing the filter, and rank by exact f32 score. Resolution is
    /// verified against the live index, so stale graph nodes are skipped.
    fn search_ann(
        &self,
        collections: &[&str],
        q: &[f32],
        opts: &SearchOpts,
        score_fn: fn(&[f32], &[f32]) -> f32,
    ) -> Result<Vec<Hit>> {
        let Some(ann) = self.ann.as_ref() else {
            return Ok(Vec::new());
        };
        if opts.top_k == 0 {
            return Ok(Vec::new());
        }
        let scope: HashSet<&str> = collections.iter().copied().collect();
        let overscan = self.config.ann.map_or(1, |a| a.overscan).max(1);
        let n_candidates = opts.top_k.saturating_mul(overscan).max(opts.top_k);

        // Exact-prefilter fallback: only a narrowed query can starve the walk. The post-filter
        // surfaces `top_k` survivors reliably only when the survivor population ≥ total/overscan;
        // below that, gather survivors directly and score them exactly.
        let total = self.scannable_doc_count();
        let in_scope = self.scannable_in_scope(collections);
        let narrowed = !opts.filter.0.is_empty() || in_scope < total;
        if narrowed {
            let cap = (total / overscan).max(n_candidates);
            if let Some(mut scan) = self.collect_selective_scan(collections, &opts.filter, cap) {
                // Row-sort for cache-friendly sequential `data` access, then score
                // exactly through the shared brute-force tail.
                scan.sort_unstable_by_key(|&(row, _, _)| row);
                return self.rank_scan(q, &mut scan, score_fn, opts);
            }
        }

        // Walk the index in the configured space — quantized codes when quantization is
        // on (the graph/lists were built in that space), else exact f32 (nidus-ndu).
        let walk =
            super::quant::ann_walk_for(self.quant.as_ref(), &self.data, self.config.distance);
        let candidates = ann.search(&walk, q, n_candidates);

        let mut topk: TopK<(&str, &str)> = TopK::new(opts.top_k);
        self.offer_candidates(&candidates, &scope, q, score_fn, opts, &mut topk);
        Ok(self.hits_from_topk(topk))
    }

    /// Resolve walked candidates to their docs, drop the out-of-scope/filtered/stale ones,
    /// exact-rerank, and offer into `topk` — the shared tail of both index-walk searches. The
    /// walk's score is only a selection proxy, so the true f32 score is recomputed here.
    fn offer_candidates<'b>(
        &'b self,
        candidates: &[(u64, f32)],
        scope: &std::collections::HashSet<&str>,
        q: &[f32],
        score_fn: fn(&[f32], &[f32]) -> f32,
        opts: &SearchOpts,
        topk: &mut TopK<(&'b str, &'b str)>,
    ) {
        for (row, _) in candidates {
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
            if entry.row != Some(*row) {
                continue; // stale reverse-map hint — row was overwritten/cleared
            }
            if !filter::matches(&opts.filter, &entry.attrs) {
                continue;
            }
            let score = score_fn(q, self.data.row(*row));
            if let Some(min) = opts.min_score
                && score < min
            {
                continue;
            }
            topk.offer(score, (col_name.as_str(), id.as_str()));
        }
    }

    /// Per-segment fan-out search (SPEC §14.3): brute-force the **exhaustive tail** (the
    /// active segment plus any sealed segment below `segment_index_min_rows`) and **walk
    /// each cold segment's IVF** for candidates, fusing both into one ranking.
    fn search_segmented(
        &self,
        collections: &[&str],
        q: &[f32],
        opts: &SearchOpts,
        score_fn: fn(&[f32], &[f32]) -> f32,
    ) -> Result<Vec<Hit>> {
        if opts.top_k == 0 {
            return Ok(Vec::new());
        }

        // Global row ranges of the indexed (cold) segments — used to exclude their rows
        // from the exhaustive tail leg (the IVF leg covers them).
        let ranges = self.data.segment_ranges();
        let indexed: Vec<(u64, u64)> = ranges
            .iter()
            .zip(&self.seg_indexes)
            .filter_map(|(&(base, rows), ix)| ix.as_ref().map(|_| (base, base + rows)))
            .collect();

        // Exhaustive tail: in-scope, filter-passing rows outside every indexed segment, gathered
        // from the live index, row-sorted for sequential `data` access, and scored exactly through
        // the shared brute-force tail.
        let mut scan: Vec<(u64, &str, &str)> = Vec::new();
        for &col_name in collections {
            let Some(col) = self.collections.get(col_name) else {
                continue;
            };
            for (id, entry) in &col.docs {
                let Some(row) = entry.row else { continue };
                if indexed.iter().any(|&(s, e)| row >= s && row < e) {
                    continue; // covered by the IVF leg
                }
                if !filter::matches(&opts.filter, &entry.attrs) {
                    continue;
                }
                scan.push((row, col_name, id.as_str()));
            }
        }
        scan.sort_unstable_by_key(|&(row, _, _)| row);
        let mut hits = self.rank_scan(q, &mut scan, score_fn, opts)?;

        // Indexed cold segments: walk each IVF for an over-fetched candidate set, then
        // verify/filter/rerank into a bounded heap.
        let scope: std::collections::HashSet<&str> = collections.iter().copied().collect();
        let overscan = AnnConfig::ivf().overscan.max(1);
        let n_candidates = opts.top_k.saturating_mul(overscan).max(opts.top_k);
        let walk = Walk::exact(&self.data, self.config.distance);
        let mut ivf_topk: TopK<(&str, &str)> = TopK::new(opts.top_k);
        for ix in self.seg_indexes.iter().flatten() {
            let candidates = ix.search(&walk, q, n_candidates);
            self.offer_candidates(&candidates, &scope, q, score_fn, opts, &mut ivf_topk);
        }
        hits.extend(self.hits_from_topk(ivf_topk));

        // Merge the two legs into one ranking: highest exact score first, deterministic
        // tie-break, then keep `top_k`.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.collection.cmp(&b.collection))
                .then_with(|| a.id.cmp(&b.id))
        });
        hits.truncate(opts.top_k);
        Ok(hits)
    }

    /// Gather in-scope, filter-passing rows for the exact-prefilter fallback, bailing once the
    /// population exceeds `cap`. `Some` = selective enough to score exactly and stay
    /// recall-complete; `None` = permissive, so walk the graph. The bail keeps it `O(cap)`.
    fn collect_selective_scan<'b>(
        &'b self,
        collections: &[&'b str],
        filter: &Filter,
        cap: usize,
    ) -> Option<Vec<(u64, &'b str, &'b str)>> {
        let mut scan: Vec<(u64, &str, &str)> = Vec::new();
        for &col_name in collections {
            let Some(col) = self.collections.get(col_name) else {
                continue;
            };
            for (id, entry) in &col.docs {
                let Some(row) = entry.row else { continue };
                if !filter::matches(filter, &entry.attrs) {
                    continue;
                }
                if scan.len() == cap {
                    return None; // population exceeds the selective threshold
                }
                scan.push((row, col_name, id.as_str()));
            }
        }
        Some(scan)
    }
}
