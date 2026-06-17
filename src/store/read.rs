//! Reads and search. Cheap accessors (`dimension`, `collections`, `footprint`, …),
//! the row-sorted scan plumbing (`scan_order`/`with_sorted_scan`) that feeds every
//! query the data matrix in storage order, the exact f32 brute-force [`search`](Store::search),
//! and the approximate [`search_ann`](Store::search_ann). The quantized first-pass
//! search it dispatches to lives in [`super::quant`].

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;

use super::scoring::{PARALLEL_SCAN_WORK_FLOOR, parallel_topk, score_chunk};
use super::{ScanOrder, Store, oom};
use crate::config::Config;
use crate::filter;
use crate::fts::Language;
use crate::model::{Distance, Filter, Footprint, FtsQuery, Hit, HybridOpts, SearchOpts, Value};
use crate::search::{TopK, dot, euclidean_neg_sq, normalize};

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

    // NOTE: `get_all` materializes the whole collection (vector + attr clones) into
    // a fresh Vec and returns it directly, so it is not fallible — an OOM here can
    // still abort. Making it `Result` would break the public API for a bulk-read
    // convenience; hosts holding huge collections should prefer `search`/scoped
    // reads. The write and open paths (the exhaustion-critical ones) are fallible.
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

    /// List records matching `filter` across `collections`, without vector scoring.
    /// Skips the first `offset` matches and returns up to `limit` more, all with
    /// `score: 0.0`. Unlike `search`, this **includes text-only docs** (no vector) —
    /// it is a metadata query. The window `[offset, offset + limit)` paginates a stable
    /// ordering: vector-bearing docs first by physical row, then text-only docs by id.
    pub fn list(
        &self,
        collections: &[&str],
        filter: &Filter,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<Hit>> {
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
                if !filter::matches(filter, &entry.attrs) {
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

    /// Total **scannable** (vector-bearing) docs across all collections — the scan-order
    /// cache's length and the yardstick for "does this scope cover every vector doc?"
    /// (`scan_cap == scannable count`). Text-only docs carry no row and are excluded:
    /// they never enter a vector scan.
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

    /// Brute-force search over the union of `collections`, merged into one ranking
    /// (one bounded top-k heap fed by every in-scope collection). The scoring function
    /// is determined by the store's [`Distance`] metric.
    ///
    /// Dispatches to the approximate [`search_ann`](Self::search_ann) when an ANN index
    /// is configured, then to the quantized two-pass path
    /// ([`search_quantized`](Self::search_quantized)) when quantization is on, and
    /// otherwise scores the row-sorted scan exactly via [`rank_scan`](Self::rank_scan).
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
        // speed. A selective filter/scope can starve that candidate walk, so
        // `search_ann` first falls back to an exact prefilter when the survivor set is
        // small enough to score directly (nidus-0ou). Skips the linear scan otherwise.
        if self.ann.is_some() {
            return self.search_ann(collections, &q, opts, score_fn);
        }

        // Gather the in-scope, filter-passing rows in physical-row order (for
        // cache-friendly sequential `data` access — nidus-33k). `with_sorted_scan`
        // hands back an already row-sorted scan, reusing the cached whole-store order
        // where it can so the sort is not redone every query (nidus-dxt).
        self.with_sorted_scan(collections, &opts.filter, |scan| {
            // Decide once whether this query splits across workers (configured threads +
            // enough scan work to amortize spawn cost).
            let workers = self.parallel_workers(scan.len());

            // Two-pass quantized search if enabled and the quantized matrix is populated;
            // otherwise the standard exact f32 brute-force path.
            if let Some(res) = self.search_quantized(&q, scan, opts, score_fn, workers) {
                return res;
            }
            self.rank_scan(&q, scan, score_fn, opts)
        })
    }

    /// Full-text (BM25) search over `collections`, ranked by BM25 relevance for
    /// `query.field`. Reuses the same `Hit`/`Filter`/top-k machinery as vector
    /// `search`: each in-scope collection's matches are scored, the metadata `filter`
    /// and `min_score` (here a **raw BM25** floor, not cosine) are applied, and one
    /// bounded heap merges them into a single ranking. Text-only and vector-bearing
    /// docs are both eligible — relevance is purely textual. Results are tie-broken by
    /// `(collection, id)` for determinism.
    pub fn text_search(
        &self,
        collections: &[&str],
        query: &FtsQuery,
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        if opts.top_k == 0 {
            return Ok(Vec::new());
        }
        let mut topk: TopK<(&str, &str)> = TopK::new(opts.top_k);
        // Analyze the query text once per distinct field language across the scope
        // (collections usually share one), not once per collection.
        let mut analyzed: HashMap<Language, Vec<String>> = HashMap::new();
        for &col_name in collections {
            let Some(col) = self.collections.get(col_name) else {
                continue;
            };
            let Some(lang) = self.fts.field_language(col_name, &query.field) else {
                continue; // this collection doesn't full-text-index the field
            };
            analyzed
                .entry(lang)
                .or_insert_with(|| crate::fts::analyze(&query.text, lang));
            let terms = &analyzed[&lang];
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
        let mut hits = self.hits_from_topk(topk);
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.collection.cmp(&b.collection))
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(hits)
    }

    /// Hybrid search: fuse a vector query and a BM25 text query into one ranking with
    /// Reciprocal Rank Fusion. Each leg is run independently (reusing
    /// [`search`](Self::search) and [`text_search`](Self::text_search) with the shared
    /// `filter`, pulled `candidates` deep), then a doc's fused score is the sum over the
    /// legs of `1 / (rrf_k + rank + 1)`. A doc present in only one leg is carried by it.
    /// Results are tie-broken by `(collection, id)` for determinism.
    pub fn hybrid_search(
        &self,
        collections: &[&str],
        vector: &[f32],
        text: &FtsQuery,
        opts: &HybridOpts,
    ) -> Result<Vec<Hit>> {
        if opts.top_k == 0 {
            return Ok(Vec::new());
        }
        // Pull each leg at least `top_k` deep so fusion can fill `top_k`.
        let leg_opts = SearchOpts {
            top_k: opts.candidates.max(opts.top_k),
            filter: opts.filter.clone(),
            min_score: None,
        };
        let vector_leg = self.search(collections, vector, &leg_opts)?;
        let text_leg = self.text_search(collections, text, &leg_opts)?;

        // Fuse by reciprocal rank, keyed by (collection, id), carrying attrs.
        let k = opts.rrf_k;
        let mut fused: HashMap<(String, String), (f32, BTreeMap<String, Value>)> = HashMap::new();
        for (rank, h) in vector_leg.into_iter().enumerate() {
            let e = fused.entry((h.collection, h.id)).or_insert((0.0, h.attrs));
            e.0 += 1.0 / (k + rank as f32 + 1.0);
        }
        for (rank, h) in text_leg.into_iter().enumerate() {
            let e = fused.entry((h.collection, h.id)).or_insert((0.0, h.attrs));
            e.0 += 1.0 / (k + rank as f32 + 1.0);
        }

        let mut hits: Vec<Hit> = fused
            .into_iter()
            .map(|((collection, id), (score, attrs))| Hit {
                collection,
                id,
                score,
                attrs,
            })
            .collect();
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

    /// Score an already-gathered, in-scope, filter-passing scan exactly (f32) into
    /// ranked [`Hit`]s. The shared tail of the brute-force path and the ANN
    /// exact-prefilter fallback ([`Self::search_ann`]): both arrive with a row-sorted
    /// scan and need the same bounded top-k + hit assembly. Splits across worker
    /// threads when the scan clears the parallel work floor, else scores serially —
    /// both yield the same bounded top-k (ties aside).
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
            score_chunk(&self.data, scan, q, score_fn, opts.top_k, opts.min_score)
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
                Hit {
                    collection: collection.to_string(),
                    id: id.to_string(),
                    score,
                    attrs,
                }
            })
            .collect()
    }

    /// ANN search: walk the index for `top_k × overscan` candidate rows, then resolve
    /// each to its owning doc, keep only those in scope and passing the filter, and
    /// rank by the exact f32 score (the candidate scores returned by the index are
    /// already the exact metric — both the HNSW beam and the IVF probe score real
    /// rows). Candidate→doc resolution is verified against the live index, so stale
    /// graph nodes (deleted/overwritten rows) are skipped.
    ///
    /// The walk over-fetches so a *permissive* filter/scope still leaves `top_k`
    /// survivors to rank. A **selective** one would starve it (the survivors are too
    /// sparse among the nearest `n_candidates` overall), silently dropping recall — so
    /// when the in-scope, filter-passing population is small enough to score directly
    /// (`≤ total/overscan`, below which the walk can't be trusted), we skip the graph
    /// and brute-force exactly those rows instead. That fallback is cheap *because*
    /// the filter is selective, and it restores exact recall (nidus-0ou).
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

        // Exact-prefilter fallback. Only a narrowed query (a filter or a strict scope
        // subset) can starve the walk; an unfiltered whole-store search always takes
        // the graph. The ANN post-filter reliably surfaces `top_k` survivors only when
        // selectivity ≥ 1/overscan, i.e. the survivor population ≥ total/overscan;
        // below that we gather the survivors directly (bailing out as soon as the
        // population proves it is *not* selective) and score them exactly.
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
        for (row, _) in &candidates {
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
            if entry.row != Some(*row) {
                continue; // stale reverse-map hint — row was overwritten/cleared
            }
            if !filter::matches(&opts.filter, &entry.attrs) {
                continue;
            }
            // Rerank exactly: the walk's score is only a selection proxy (approximate
            // under quantization), so the true f32 score — and `min_score` — is computed
            // here from the original vectors, exactly as the quantized brute-force path
            // reranks its first-pass candidates.
            let score = score_fn(q, self.data.row(*row));
            if let Some(min) = opts.min_score
                && score < min
            {
                continue;
            }
            topk.offer(score, (col_name.as_str(), id.as_str()));
        }

        Ok(self.hits_from_topk(topk))
    }

    /// Gather in-scope, filter-passing rows for the exact-prefilter fallback, bailing
    /// out the moment the population exceeds `cap`. `Some(scan)` means the filter/scope
    /// is selective enough that the whole survivor set fits within `cap` — exact
    /// scoring over it is cheap *and* recall-complete, which the ANN post-filter walk
    /// cannot guarantee once it starves. `None` means the query is permissive
    /// (population > `cap`), so the caller should walk the graph. Pure metadata work,
    /// no vector scoring; the early bail keeps the permissive case `O(cap)` rather than
    /// `O(scope)`.
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
