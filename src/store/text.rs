//! Full-text and hybrid search: multi-clause BM25 (nidus-m50.10) and the RRF fusion of a
//! vector leg with it, plus the opt-in annotations that explain a hit (nidus-m50.5).

use std::collections::HashMap;

use anyhow::Result;

use super::Store;
use super::plan::PlanRec;
use super::rank;
use super::read::{check_query_opts, check_weight, depth, paginate};
use crate::annotate::{Annotations, ClauseScore, Highlight, HighlightOpts, LegScore};
use crate::filter;
use crate::fts::Analyzer;
use crate::fuse::{FusionLeg, rrf_fuse};
use crate::model::{FtsClause, FtsCombine, FtsQuery, Hit, HybridOpts, SearchOpts};
use crate::plan::QueryPlan;
use crate::search::TopK;

/// A document's BM25 score per clause, in clause order; `None` where that clause did not match.
type PerClause = Vec<Option<f32>>;

/// Per-hit clause breakdowns keyed by `(collection, id)` — how the text leg's detail reaches a
/// fused hit, since fusion rebuilds the `Hit` and cannot see inside a leg.
type Breakdown = HashMap<(String, String), Vec<ClauseScore>>;

/// Fold a doc's per-clause scores into its text score. Unmatched clauses contribute nothing, and
/// either mode reproduces a single clause's score exactly (BM25 here is strictly positive, so
/// folding `Max` from `0.0` cannot mask a clause).
fn combine(per_clause: &PerClause, mode: FtsCombine) -> f32 {
    let scored = per_clause.iter().flatten().copied();
    match mode {
        FtsCombine::Sum => scored.sum(),
        FtsCombine::Max => scored.fold(0.0, f32::max),
    }
}

/// The matched clauses' scores, in query order — the reportable form of a [`PerClause`].
fn clause_scores(per_clause: &PerClause, clauses: &[FtsClause]) -> Vec<ClauseScore> {
    clauses
        .iter()
        .zip(per_clause)
        .filter_map(|(c, s)| {
            s.map(|score| ClauseScore {
                field: c.field.clone(),
                score,
            })
        })
        .collect()
}

impl Store {
    /// Full-text (BM25) search over `collections`, reusing vector `search`'s
    /// `Hit`/`Filter`/top-k machinery. Every clause scores against its own field and folds by
    /// `query.combine`; `min_score` is a raw BM25 floor on the folded score, not cosine.
    pub fn text_search(
        &self,
        collections: &[&str],
        query: &FtsQuery,
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        query.validate()?;
        check_query_opts(opts)?;
        if opts.top_k == 0 {
            return Ok(Vec::new());
        }
        // Same shape as `search`: rank `depth` deep, then hand the ranking to the one tail.
        let mut topk: TopK<(&str, &str)> = TopK::new(depth(opts));
        // Analyze each clause once per distinct field analyzer across the scope (collections
        // usually share one), not once per collection.
        let mut analyzed: HashMap<(usize, Analyzer), Vec<String>> = HashMap::new();
        let mut breakdown = Breakdown::new();

        for &col_name in collections {
            let Some(col) = self.collections.get(col_name) else {
                continue;
            };
            // Accumulate every clause's score per doc before folding, so `Sum`/`Max` see the
            // whole row and a doc matching two clauses is ranked once, not twice.
            let mut acc: HashMap<&str, PerClause> = HashMap::new();
            for (ci, clause) in query.clauses.iter().enumerate() {
                let Some(cfg) = self.fts.field_analyzer(col_name, &clause.field) else {
                    continue; // this collection doesn't full-text-index the clause's field
                };
                analyzed
                    .entry((ci, cfg))
                    .or_insert_with(|| crate::fts::analyze(&clause.text, cfg));
                let terms = &analyzed[&(ci, cfg)];
                for (id, score) in self.fts.score(col_name, &clause.field, terms) {
                    acc.entry(id)
                        .or_insert_with(|| vec![None; query.clauses.len()])[ci] = Some(score);
                }
            }
            for (id, per_clause) in acc {
                let base = combine(&per_clause, query.combine);
                // Hint-verify the id against the live index and apply the metadata filter (the
                // FTS index can lag a delete until the next rebuild).
                let Some(entry) = col.docs.get(id) else {
                    continue;
                };
                if !filter::matches(&opts.filter, &entry.attrs) {
                    continue;
                }
                // A subtracted penalty is metric-agnostic, so the same expression applies to a
                // folded BM25 score as to a cosine one. `min_score` gates the final number.
                let score = rank::adjust(opts.rank_by.as_ref(), base, &entry.attrs);
                if let Some(min) = opts.min_score
                    && score < min
                {
                    continue;
                }
                topk.offer(score, (col_name, id));
                if opts.explain {
                    breakdown.insert(
                        (col_name.to_string(), id.to_string()),
                        clause_scores(&per_clause, &query.clauses),
                    );
                }
            }
        }
        // `TopK` already resolves ties on `(collection, id)`, so no re-sort is needed here.
        let mut hits = self.finish(self.hits_from_topk(topk, &opts.projection), opts);
        self.annotate(&mut hits, query, opts.explain.then_some(&mut breakdown));
        Ok(hits)
    }

    /// Hybrid search: fuse a vector and a BM25 leg with Reciprocal Rank Fusion. Each leg runs
    /// independently `candidates` deep, then a doc's fused score is the sum of
    /// `1 / (rrf_k + rank + 1)`; a doc in only one leg is carried by it.
    pub fn hybrid_search(
        &self,
        collections: &[&str],
        vector: &[f32],
        text: &FtsQuery,
        opts: &HybridOpts,
    ) -> Result<Vec<Hit>> {
        Ok(self
            .traced(opts.plan, |rec| {
                self.hybrid_search_inner(collections, vector, text, opts, rec)
            })?
            .0)
    }

    /// Like [`Store::hybrid_search`], but also returns the [`QueryPlan`] for the vector leg
    /// (nidus-cvz). `text_search` alone has no plan; the hybrid plan describes its vector leg.
    pub fn hybrid_search_with_plan(
        &self,
        collections: &[&str],
        vector: &[f32],
        text: &FtsQuery,
        opts: &HybridOpts,
    ) -> Result<(Vec<Hit>, QueryPlan)> {
        let (hits, plan) = self.traced(true, |rec| {
            self.hybrid_search_inner(collections, vector, text, opts, rec)
        })?;
        Ok((hits, plan.expect("traced(true, _) always finishes a plan")))
    }

    /// The instrumented body shared by [`Store::hybrid_search`] and
    /// [`Store::hybrid_search_with_plan`].
    fn hybrid_search_inner(
        &self,
        collections: &[&str],
        vector: &[f32],
        text: &FtsQuery,
        opts: &HybridOpts,
        rec: &mut PlanRec,
    ) -> Result<Vec<Hit>> {
        // Ahead of the `top_k == 0` shortcut, not after: the vector leg validates, but the
        // shortcut returns before the leg runs. Validating here means a bad query does not change
        // verdict based on `top_k`.
        self.check_query_dim(vector)?;
        text.validate()?;
        check_weight("vector_weight", opts.vector_weight)?;
        check_weight("text_weight", opts.text_weight)?;

        if opts.top_k == 0 {
            return Ok(Vec::new());
        }
        // Pull each leg at least a full page deep (`offset + top_k`) so fusion can fill it.
        let page = opts.offset.saturating_add(opts.top_k);
        let leg_opts = SearchOpts {
            top_k: opts.candidates.max(page),
            filter: opts.filter.clone(),
            explain: opts.explain,
            ..Default::default()
        };
        let vector_leg = self.search_inner(collections, vector, &leg_opts, rec)?;
        // The leg is scored but not highlighted: highlighting the whole candidate set would be
        // work thrown away on the documents fusion drops.
        let mut text_leg = self.text_search(
            collections,
            &FtsQuery {
                highlight: None,
                ..text.clone()
            },
            &leg_opts,
        )?;
        let mut breakdown: Breakdown = text_leg
            .iter_mut()
            .filter_map(|h| {
                let a = h.annotations.take()?;
                Some(((h.collection.clone(), h.id.clone()), a.clauses))
            })
            .collect();

        // The page is cut on the *fused* ranking, never per leg — a leg's rank is an input to
        // the fused score, so paginating a leg would change which documents fuse at all.
        let fused = rrf_fuse(
            vec![
                FusionLeg::new(vector_leg).weight(opts.vector_weight),
                FusionLeg::new(text_leg).weight(opts.text_weight),
            ],
            opts.rrf_k,
        );
        let mut page = paginate(fused, opts.offset);
        page.truncate(opts.top_k);

        let mut hits: Vec<Hit> = page
            .into_iter()
            .map(|(mut hit, per_leg)| {
                if opts.explain {
                    let leg = |i: usize| per_leg[i].map(|(rank, score)| LegScore { rank, score });
                    hit.annotations = Some(Annotations {
                        vector: leg(0),
                        text: leg(1),
                        clauses: breakdown
                            .remove(&(hit.collection.clone(), hit.id.clone()))
                            .unwrap_or_default(),
                        highlights: Vec::new(),
                    });
                }
                hit
            })
            .collect();
        self.annotate(&mut hits, text, None);
        if let Some(e) = &opts.expand {
            self.expand_hits(&mut hits, e);
        }
        Ok(hits)
    }

    /// Attach the opt-in annotations to a final page: each matched clause's BM25 score (drained
    /// from `breakdown`, when explaining) and the highlighted fragments. Runs on the page rather
    /// than the candidate set, so the cost is bounded by `top_k`.
    fn annotate(&self, hits: &mut [Hit], query: &FtsQuery, breakdown: Option<&mut Breakdown>) {
        if breakdown.is_none() && query.highlight.is_none() {
            return;
        }
        let mut breakdown = breakdown;
        for hit in hits.iter_mut() {
            let a = hit.annotations.get_or_insert_with(Annotations::default);
            if let Some(b) = breakdown.as_deref_mut() {
                a.clauses = b
                    .remove(&(hit.collection.clone(), hit.id.clone()))
                    .unwrap_or_default();
            }
            if let Some(opts) = &query.highlight {
                a.highlights = self.highlights_for(&hit.collection, &hit.id, query, opts);
            }
        }
    }

    /// Highlighted fragments for one hit, one entry per clause field that matched. Reads the
    /// **live** record's text, not the hit's projected attrs, so projecting a long body away and
    /// keeping only its snippet is the supported combination rather than a silent no-op.
    fn highlights_for(
        &self,
        collection: &str,
        id: &str,
        query: &FtsQuery,
        opts: &HighlightOpts,
    ) -> Vec<Highlight> {
        let Some(entry) = self
            .collections
            .get(collection)
            .and_then(|c| c.docs.get(id))
        else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for clause in &query.clauses {
            let Some(cfg) = self.fts.field_analyzer(collection, &clause.field) else {
                continue;
            };
            let text = crate::fts::field_text(&entry.attrs, &clause.field);
            let terms = crate::fts::analyze(&clause.text, cfg);
            let fragments = crate::fts::fragments(&text, cfg, &terms, opts);
            if !fragments.is_empty() {
                out.push(Highlight {
                    field: clause.field.clone(),
                    fragments,
                });
            }
        }
        out
    }
}
