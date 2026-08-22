//! Reads and search: cheap accessors, the row-sorted scan plumbing that feeds every query the
//! data matrix in storage order, exact f32 brute force, and the approximate `search_ann`. The
//! quantized first pass lives in [`super::quant`], full-text and hybrid in [`super::text`].

use std::collections::{BTreeMap, HashSet};

use anyhow::{Context, Result, bail};

use super::aggregate::LIMIT_PER_OVERFETCH;
use super::diversity::MMR_OVERFETCH;
use super::plan::{Phase, PlanRec};
use super::rank;
use super::scoring::{PARALLEL_SCAN_WORK_FLOOR, parallel_topk, score_chunk};
use super::{ScanOrder, Store, oom};
use crate::ann::Walk;
use crate::config::Config;
use crate::filter;
use crate::model::{
    AnnConfig, Distance, Filter, Footprint, Hit, HybridOpts, ListOpts, Projection, SearchOpts,
};
use crate::plan::{Candidates, Narrowing, QueryPath, QueryPlan};
use crate::search::{TopK, dot, euclidean_neg_sq, normalize};

/// The query-side inputs `offer_candidates` needs, bundled so the two index-walk callers
/// pass one borrow rather than four.
struct CandidateCtx<'a> {
    scope: &'a std::collections::HashSet<&'a str>,
    q: &'a [f32],
    score_fn: fn(&[f32], &[f32]) -> f32,
    opts: &'a SearchOpts,
}

/// How deep a search path must rank: one page (`offset + top_k`), multiplied by the over-fetch
/// factor when a per-value cap will thin the ranking (nidus-m50.6) or MMR will reorder it
/// (nidus-tx2). The **larger** factor wins, since one deep ranking serves both.
pub(super) fn depth(opts: &SearchOpts) -> usize {
    let page = opts.offset.saturating_add(opts.top_k);
    let cap = if opts.limit_per.is_some() {
        LIMIT_PER_OVERFETCH
    } else {
        1
    };
    let spread = if opts.diversity.is_some() {
        MMR_OVERFETCH
    } else {
        1
    };
    page.saturating_mul(cap.max(spread))
}

/// The options a search *path* runs with: rank [`depth`] deep, then let the caller's single
/// tail cap and paginate. `offset` is zeroed so no inner path can paginate a second time.
fn deepened(opts: &SearchOpts) -> SearchOpts {
    SearchOpts {
        top_k: depth(opts),
        offset: 0,
        ..opts.clone()
    }
}

/// The marker every caller-fault query rejection carries, so the HTTP layer answers `400`
/// rather than `500` for a malformed knob (`server::classify` keys off it).
pub(crate) const BAD_QUERY: &str = "invalid query option";

/// How [`Store::narrowed_scan`] fared, for [`QueryPlan::narrowing`] to report.
#[derive(Clone, Copy)]
enum NarrowOutcome {
    Inactive,
    Declined,
    Narrowed(usize),
}

/// Refuse the knobs that shape a ranking, once per query rather than once per record. Beside
/// `filter::validate`, which the `Nidus` entry points already run.
pub(super) fn check_query_opts(opts: &SearchOpts) -> Result<()> {
    rank::validate(opts.rank_by.as_ref())
        .and_then(|()| super::aggregate::validate(opts.limit_per.as_ref()))
        .and_then(|()| super::diversity::validate(opts.diversity))
        .and_then(|()| super::expand::validate(opts.expand.as_ref()))
        .context(BAD_QUERY)
}

/// Refuse a fusion weight that would poison the fused score. A `NaN` weight makes every
/// comparison in the sort false and a negative one inverts a leg rather than de-emphasizing it.
pub(super) fn check_weight(name: &str, w: f32) -> Result<()> {
    if !w.is_finite() || w < 0.0 {
        bail!("{BAD_QUERY}: {name} must be finite and non-negative, got {w}");
    }
    Ok(())
}

/// Drop the first `offset` ranked entries — the ONE place a page boundary is cut. An offset past
/// the end is an empty page, never an error: a caller walking pages must be able to stop. Generic
/// so hybrid can page the fused ranking while it still carries each leg's detail.
pub(super) fn paginate<T>(mut ranked: Vec<T>, offset: usize) -> Vec<T> {
    if offset == 0 {
        return ranked;
    }
    if offset >= ranked.len() {
        return Vec::new();
    }
    ranked.drain(..offset);
    ranked
}

impl Store {
    /// The ONE tail every ranked surface funnels through: cap by value, spread by MMR, cut the
    /// page, hold `top_k`. Both precede pagination (reshaping a page moves the boundary, not
    /// what crosses it), and the cap precedes MMR so a hard constraint outranks an objective.
    pub(crate) fn finish(&self, ranked: Vec<Hit>, opts: &SearchOpts) -> Vec<Hit> {
        let capped = match &opts.limit_per {
            Some(cap) => self.cap_per_value(ranked, cap),
            None => ranked,
        };
        let spread = match opts.diversity {
            Some(lambda) => self.diversify(capped, lambda),
            None => capped,
        };
        let mut hits = paginate(spread, opts.offset);
        hits.truncate(opts.top_k);
        // Last, and after the page is cut: expansion is payload-only, so it must not be able
        // to influence anything that reorders or thins the ranking.
        if let Some(e) = &opts.expand {
            self.expand_hits(&mut hits, e);
        }
        hits
    }

    /// The hybrid tail for an already-fused `Vec<Hit>`: no `limit_per` on `HybridOpts`, so
    /// just the page cut (SPEC §7). Used by `crate::rerank`'s async wrapper post-rerank;
    /// `hybrid_search` itself cuts inline, since its fused list still carries per-leg detail.
    #[cfg_attr(not(feature = "rerank"), allow(dead_code))]
    pub(crate) fn finish_hybrid(&self, ranked: Vec<Hit>, opts: &HybridOpts) -> Vec<Hit> {
        let mut hits = paginate(ranked, opts.offset);
        hits.truncate(opts.top_k);
        if let Some(e) = &opts.expand {
            self.expand_hits(&mut hits, e);
        }
        hits
    }

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
            filter_index_bytes: self.findex.heap_bytes() as u64,
        }
    }

    pub fn has_collection(&self, name: &str) -> bool {
        self.collections.contains_key(name)
    }

    /// Whether `collection` already has a declared FTS schema. The truthful gate for
    /// callers that must call `set_fts_schema` at most once per collection, since that
    /// call rebuilds the field index from every live doc (`store/write.rs`).
    pub fn has_fts_schema(&self, collection: &str) -> bool {
        self.fts.schema_for(collection).is_some()
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

    /// O(1) id-keyed lookup; a missing collection or id is `None`, not an error.
    pub fn get(&self, collection: &str, id: &str) -> Option<crate::model::Record> {
        let entry = self.collections.get(collection)?.docs.get(id)?;
        Some(crate::model::Record {
            id: id.to_string(),
            // Text-only docs (row None) have no embedding.
            vector: entry.row.map(|r| self.data.row(r).to_vec()),
            attrs: entry.attrs.clone(),
        })
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
            // Narrow through the filter index when it can help, then verify each survivor
            // exactly as the full walk does — the index proposes, `filter::matches` decides.
            if let Some(ids) = filter::candidate_ids(&self.findex, col_name, &opts.filter) {
                for id in ids {
                    let Some((id, entry)) = col.docs.get_key_value(&id) else {
                        continue;
                    };
                    if !filter::matches(&opts.filter, &entry.attrs) {
                        continue;
                    }
                    scan.push((entry.row, col_name, id.as_str()));
                }
                continue;
            }
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
        // ORDER BY runs over the whole match set, before the page is cut — sorting a page
        // would only reorder rows storage order had already chosen (nidus-m50.3).
        if let Some(order) = &opts.order_by {
            self.order_scan(&mut scan, order);
        }
        let results = scan
            .iter()
            .skip(opts.offset)
            .take(opts.limit)
            .map(|&(_, collection, id)| {
                let attrs = self
                    .collections
                    .get(collection)
                    .and_then(|c| c.docs.get(id))
                    .map(|e| opts.projection.apply(&e.attrs))
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

    /// Build the in-scope, filter-passing scan **in row order** and hand it to `f` along
    /// with `rec` (a parameter, not a capture — capturing it would borrow it mutably twice
    /// at once, here and inside `f`). Passing `&mut PlanRec::Off` costs nothing extra.
    fn with_sorted_scan<R>(
        &self,
        collections: &[&str],
        filter: &Filter,
        rec: &mut PlanRec,
        f: impl for<'b> FnOnce(&mut [(u64, &'b str, &'b str)], &mut PlanRec) -> Result<R>,
    ) -> Result<R> {
        // Count only vector-bearing docs — text-only docs never enter a vector scan.
        let scan_cap: usize = self.scannable_in_scope(collections);
        let mut scan: Vec<(u64, &str, &str)> = Vec::new();
        scan.try_reserve(scan_cap)
            .map_err(|_| oom("search scan buffer", scan_cap))?;

        // Try the filter index first. It applies to whole-store and subset scope alike,
        // and it only reports success when every in-scope collection could be narrowed —
        // a partial narrowing would silently omit the collections it skipped.
        let outcome = rec.phase(Phase::Narrow, || {
            self.narrowed_scan(collections, filter, &mut scan)
        });
        if let NarrowOutcome::Narrowed(n) = outcome {
            // Unconditional: the aggregate must not depend on whether a plan was asked for.
            crate::metrics::metrics().search_findex_narrowed.inc();
            rec.narrowing(Narrowing::Narrowed {
                candidates: n as u64,
            });
            rec.phase(Phase::Gather, || {
                scan.sort_unstable_by_key(|&(row, _, _)| row)
            });
            return f(&mut scan, rec);
        }
        rec.narrowing(match outcome {
            NarrowOutcome::Inactive => Narrowing::Inactive,
            NarrowOutcome::Declined => Narrowing::Declined,
            NarrowOutcome::Narrowed(_) => unreachable!("handled above"),
        });

        let gather = rec.start();
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
            rec.stop(Phase::Gather, gather);
            f(&mut scan, rec)
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
            rec.stop(Phase::Gather, gather);
            f(&mut scan, rec)
        }
    }

    /// Fill `scan` from the filter index, or report why not. All-or-nothing: a scan built
    /// from only the narrowable collections would be missing the others' matches entirely.
    fn narrowed_scan<'b>(
        &'b self,
        collections: &[&'b str],
        filter: &Filter,
        scan: &mut Vec<(u64, &'b str, &'b str)>,
    ) -> NarrowOutcome {
        if !self.findex.is_active() {
            return NarrowOutcome::Inactive;
        }
        let mut narrowed: Vec<(&'b str, Vec<String>)> = Vec::new();
        for &col_name in collections {
            if !self.collections.contains_key(col_name) {
                continue;
            }
            match filter::candidate_ids(&self.findex, col_name, filter) {
                Some(ids) => narrowed.push((col_name, ids)),
                None => return NarrowOutcome::Declined,
            }
        }
        for (col_name, ids) in narrowed {
            let Some(col) = self.collections.get(col_name) else {
                continue;
            };
            for id in ids {
                let Some((id, entry)) = col.docs.get_key_value(&id) else {
                    continue;
                };
                let Some(row) = entry.row else { continue };
                // The index narrows; this decides. Skipping it would ship the
                // over-approximation straight to the caller.
                if !filter::matches(filter, &entry.attrs) {
                    continue;
                }
                scan.push((row, col_name, id.as_str()));
            }
        }
        NarrowOutcome::Narrowed(scan.len())
    }

    // ── Search ──────────────────────────────────────────────────────────────────

    /// Reject a query vector whose length is not the store's pinned dimension (nidus-c5v).
    pub(super) fn check_query_dim(&self, query: &[f32]) -> Result<()> {
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
        Ok(self
            .traced(opts.plan, |rec| {
                self.search_inner(collections, query, opts, rec)
            })?
            .0)
    }

    /// Like [`Store::search`], but also returns the [`QueryPlan`] describing how the query
    /// ran: path taken, rows scanned, candidate survival, phase timings (nidus-cvz).
    pub fn search_with_plan(
        &self,
        collections: &[&str],
        query: &[f32],
        opts: &SearchOpts,
    ) -> Result<(Vec<Hit>, QueryPlan)> {
        let (hits, plan) =
            self.traced(true, |rec| self.search_inner(collections, query, opts, rec))?;
        Ok((hits, plan.expect("traced(true, _) always finishes a plan")))
    }

    /// The instrumented body shared by [`Store::search`] and [`Store::search_with_plan`].
    /// `pub(super)`: `super::text::hybrid_search_inner` also drives it directly, so the
    /// hybrid path's vector leg lands in the *same* recorder instead of starting its own.
    pub(super) fn search_inner(
        &self,
        collections: &[&str],
        query: &[f32],
        opts: &SearchOpts,
        rec: &mut PlanRec,
    ) -> Result<Vec<Hit>> {
        // Before the metric: a request the store refuses is not a query it served, so
        // counting it would overstate `search_queries` and skew every ratio built on it.
        self.check_query_dim(query)?;
        check_query_opts(opts)?;

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

        // Each branch ranks `depth` deep and hands the ranking to ONE tail that caps and cuts the
        // page. Single-tail on purpose: a branch that paginated itself — or that applied the offset
        // before the top-k cap — still compiles and is silently wrong (nidus-m50.8).
        let deep = deepened(opts);
        // `opts.exact` gates every approximate branch below (nidus-m50.12). Store-level config
        // decides what exists; this decides, per query, whether to use it.
        let ranked = if self.ann.is_some() && !deep.exact {
            // ANN: walk the index for an over-fetched candidate set, then post-filter and rerank —
            // recall traded for speed. A selective filter/scope can starve the walk, so `search_ann`
            // falls back to an exact prefilter when survivors are few (nidus-0ou).
            m.search_ann.inc();
            rec.path(QueryPath::Ann);
            self.search_ann(collections, &q, &deep, score_fn, rec)?
        } else if self.seg_indexes.iter().any(Option::is_some) && !deep.exact {
            // Per-segment fan-out: walk each cold segment's IVF index and brute-force the tail (the
            // active segment plus any sub-threshold sealed one), merged into one ranking (SPEC
            // §14.3). Engaged only once a sealed segment has crossed `segment_index_min_rows`.
            m.search_segmented.inc();
            rec.path(QueryPath::Segmented);
            self.search_segmented(collections, &q, &deep, score_fn, rec)?
        } else {
            // Gather in-scope, filter-passing rows in physical-row order, for sequential `data`
            // access (nidus-33k). `with_sorted_scan` reuses the cached whole-store order where it
            // can, so the sort is not redone every query (nidus-dxt).
            self.with_sorted_scan(collections, &deep.filter, rec, |scan, rec| {
                // Only the brute-force paths reach here, which is exactly why the counter lives
                // here: "rows scanned" is a meaningful cost on a linear scan and meaningless on
                // an ANN walk, so counting it in one place keeps the metric honest.
                m.search_vectors_scanned.add(scan.len() as u64);
                rec.rows_scanned(scan.len() as u64);

                // Decide once whether this query splits across workers (configured threads +
                // enough scan work to amortize spawn cost).
                let workers = self.parallel_workers(scan.len());

                // Two-pass quantized search if enabled and the quantized matrix is populated;
                // otherwise the standard exact f32 brute-force path.
                if !deep.exact
                    && let Some(res) =
                        self.search_quantized(&q, scan, &deep, score_fn, workers, rec)
                {
                    m.search_quantized.inc();
                    rec.path(QueryPath::Quantized);
                    return res;
                }
                m.search_exact.inc();
                rec.path(QueryPath::Exact);
                rec.phase(Phase::Score, || self.rank_scan(&q, scan, score_fn, &deep))
            })?
        };
        Ok(self.finish(ranked, opts))
    }

    /// "More like this": look up `collection`/`id`'s stored (already unit-normalized) vector and
    /// search `collections` with it, dropping the source record itself by `(collection, id)`
    /// identity — never a score test, so a true duplicate (also scoring ~1.0) still comes back.
    pub fn search_similar(
        &self,
        collections: &[&str],
        collection: &str,
        id: &str,
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        Ok(self
            .traced(opts.plan, |rec| {
                self.search_similar_inner(collections, collection, id, opts, rec)
            })?
            .0)
    }

    /// Like [`Store::search_similar`], but also returns the [`QueryPlan`] for the vector
    /// search it runs under the hood (nidus-cvz).
    pub fn search_similar_with_plan(
        &self,
        collections: &[&str],
        collection: &str,
        id: &str,
        opts: &SearchOpts,
    ) -> Result<(Vec<Hit>, QueryPlan)> {
        let (hits, plan) = self.traced(true, |rec| {
            self.search_similar_inner(collections, collection, id, opts, rec)
        })?;
        Ok((hits, plan.expect("traced(true, _) always finishes a plan")))
    }

    /// The instrumented body shared by [`Store::search_similar`] and
    /// [`Store::search_similar_with_plan`].
    fn search_similar_inner(
        &self,
        collections: &[&str],
        collection: &str,
        id: &str,
        opts: &SearchOpts,
        rec: &mut PlanRec,
    ) -> Result<Vec<Hit>> {
        let entry = self
            .collections
            .get(collection)
            .and_then(|c| c.docs.get(id))
            .with_context(|| {
                format!("{BAD_QUERY}: no record `{id}` in collection `{collection}`")
            })?;
        let Some(row) = entry.row else {
            bail!(
                "{BAD_QUERY}: record `{collection}/{id}` is text-only and has no vector to search with"
            );
        };
        let query = self.data.row(row).to_vec();

        // Rank one extra slot deep with NO cap or page, so the source is gone before the single
        // tail runs. Capping first would let the source — always rank 1 — spend its own value's
        // `limit_per` quota and starve the real neighbour sharing it.
        let wide = SearchOpts {
            offset: 0,
            top_k: depth(opts).saturating_add(1),
            limit_per: None,
            diversity: None,
            ..opts.clone()
        };
        let ranked = self.search_inner(collections, &query, &wide, rec)?;
        let ranked: Vec<Hit> = ranked
            .into_iter()
            .filter(|h| !(h.collection == collection && h.id == id))
            .collect();
        Ok(self.finish(ranked, opts))
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
        // A ranking expression needs each record's attrs, which the chunk kernels never see:
        // that path scans serially (nidus-m50.15 #11).
        if opts.rank_by.is_some() {
            return self.rank_scan_expr(q, scan, score_fn, opts);
        }
        let workers = self.parallel_workers(scan.len());
        let topk = if workers > 1 {
            parallel_topk(scan, workers, opts.top_k, |chunk| {
                score_chunk(&self.data, chunk, q, score_fn, opts.top_k, opts.min_score)
            })?
        } else {
            score_chunk(&self.data, scan, q, score_fn, opts.top_k, opts.min_score)?
        };
        Ok(self.hits_from_topk(topk, &opts.projection))
    }

    /// Resolve a bounded top-k of `(collection, id)` into ranked [`Hit`]s, materializing each
    /// winner's projected attrs from the live index. Shared by every search path — the one
    /// place a hit is built, so projection is applied instead of a full map being trimmed.
    pub(super) fn hits_from_topk<'b>(
        &self,
        topk: TopK<(&'b str, &'b str)>,
        projection: &Projection,
    ) -> Vec<Hit> {
        topk.into_sorted_desc()
            .into_iter()
            .map(|(score, (collection, id))| {
                let attrs = self
                    .collections
                    .get(collection)
                    .and_then(|c| c.docs.get(id))
                    .map(|e| projection.apply(&e.attrs))
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
        rec: &mut PlanRec,
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
                // exactly through the shared brute-force tail. Overwrites the `Ann` path
                // the caller set, since this query never actually walked the graph.
                rec.path(QueryPath::AnnPrefilterFallback);
                scan.sort_unstable_by_key(|&(row, _, _)| row);
                return self.rank_scan(q, &mut scan, score_fn, opts);
            }
        }

        // Walk the index in the configured space — quantized codes when quantization is
        // on (the graph/lists were built in that space), else exact f32 (nidus-ndu).
        let walk =
            super::quant::ann_walk_for(self.quant.as_ref(), &self.data, self.config.distance);
        let candidates = rec.phase(Phase::Walk, || ann.search(&walk, q, n_candidates));

        let mut topk: TopK<(&str, &str)> = TopK::new(opts.top_k);
        let mut acc = Candidates::default();
        rec.phase(Phase::Resolve, || {
            let ctx = CandidateCtx {
                scope: &scope,
                q,
                score_fn,
                opts,
            };
            self.offer_candidates(&candidates, &ctx, &mut topk, &mut acc);
        });
        if let Some(c) = rec.candidates() {
            *c = acc;
        }
        Ok(self.hits_from_topk(topk, &opts.projection))
    }

    /// Resolve walked candidates to their docs, drop the out-of-scope/filtered/stale ones,
    /// exact-rerank, and offer into `topk` — the shared tail of both index-walk searches. The
    /// walk's score is only a selection proxy, so the true f32 score is recomputed here.
    fn offer_candidates<'b>(
        &'b self,
        candidates: &[(u64, f32)],
        ctx: &CandidateCtx<'_>,
        topk: &mut TopK<(&'b str, &'b str)>,
        acc: &mut Candidates,
    ) {
        let CandidateCtx {
            scope,
            q,
            score_fn,
            opts,
        } = ctx;
        acc.surfaced += candidates.len() as u64;
        for (row, _) in candidates {
            let Some(Some((col_name, id))) = self.row_to_doc.get(*row as usize) else {
                acc.dropped_stale += 1;
                continue;
            };
            if !scope.contains(col_name.as_str()) {
                acc.dropped_out_of_scope += 1;
                continue;
            }
            let Some(col) = self.collections.get(col_name) else {
                acc.dropped_stale += 1;
                continue;
            };
            let Some(entry) = col.docs.get(id) else {
                acc.dropped_stale += 1;
                continue;
            };
            if entry.row != Some(*row) {
                acc.dropped_stale += 1; // stale reverse-map hint — row was overwritten/cleared
                continue;
            }
            if !filter::matches(&opts.filter, &entry.attrs) {
                acc.dropped_filtered += 1;
                continue;
            }
            // The entry is already in hand, so the ranking expression costs nothing extra here —
            // which is what lets decay apply over an ANN result set without forcing exact.
            let base = score_fn(q, self.data.row(*row));
            let score = rank::adjust(opts.rank_by.as_ref(), base, &entry.attrs);
            if let Some(min) = opts.min_score
                && score < min
            {
                acc.dropped_min_score += 1;
                continue;
            }
            acc.survived += 1;
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
        rec: &mut PlanRec,
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
        let mut acc = Candidates::default();
        let ctx = CandidateCtx {
            scope: &scope,
            q,
            score_fn,
            opts,
        };
        rec.phase(Phase::Walk, || {
            for ix in self.seg_indexes.iter().flatten() {
                let candidates = ix.search(&walk, q, n_candidates);
                self.offer_candidates(&candidates, &ctx, &mut ivf_topk, &mut acc);
            }
        });
        if let Some(c) = rec.candidates() {
            *c = acc;
        }
        hits.extend(self.hits_from_topk(ivf_topk, &opts.projection));

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
