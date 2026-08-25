//! Full-text and hybrid search: multi-clause BM25 (nidus-m50.10) and the RRF fusion of a
//! vector leg with it, plus the opt-in annotations that explain a hit (nidus-m50.5).

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;

use super::Store;
use super::plan::PlanRec;
use super::rank;
use super::read::{check_query_opts, check_weight, depth, paginate};
use crate::annotate::{Annotations, ClauseScore, Expansion, Highlight, HighlightOpts, LegScore};
use crate::filter;
use crate::fts::Analyzer;
use crate::fuse::{FusionLeg, rrf_fuse};
use crate::model::{
    FtsClause, FtsCombine, FtsQuery, Hit, HybridOpts, SearchOpts, SuggestOpts, Suggestion,
    Suggestions,
};
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
/// `expansions` carries each prefix clause's truncation report, `None` for a non-prefix clause
/// or one this collection doesn't index — aligned with `clauses` the same way `per_clause` is.
fn clause_scores(
    per_clause: &PerClause,
    clauses: &[FtsClause],
    expansions: &[Option<Expansion>],
) -> Vec<ClauseScore> {
    clauses
        .iter()
        .zip(per_clause)
        .zip(expansions)
        .filter_map(|((c, s), exp)| {
            s.map(|score| ClauseScore {
                field: c.field.clone(),
                score,
                expansion: *exp,
            })
        })
        .collect()
}

/// Which pass of `Store::suggest` a collection sweep answers for (nidus-972): the exact-prefix
/// scan, or the fallback fuzzy scan that only runs when the prefix scan found nothing at all.
enum Leg {
    Prefix,
    Fuzzy,
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
        // Head terms + fragment for a prefix clause are analyzer-keyed like `analyzed`, but the
        // *expansion* against them is per `(collection, field)` and must not be cached here (D6).
        let mut prefix_analyzed: HashMap<(usize, Analyzer), (Vec<String>, Option<String>)> =
            HashMap::new();
        let mut breakdown = Breakdown::new();

        for &col_name in collections {
            let Some(col) = self.collections.get(col_name) else {
                continue;
            };
            // Accumulate every clause's score per doc before folding, so `Sum`/`Max` see the
            // whole row and a doc matching two clauses is ranked once, not twice.
            let mut acc: HashMap<&str, PerClause> = HashMap::new();
            let mut expansions: Vec<Option<Expansion>> = vec![None; query.clauses.len()];
            for (ci, clause) in query.clauses.iter().enumerate() {
                let Some(cfg) = self.fts.field_analyzer(col_name, &clause.field) else {
                    continue; // this collection doesn't full-text-index the clause's field
                };
                if clause.prefix {
                    let (heads, fragment) = prefix_analyzed
                        .entry((ci, cfg))
                        .or_insert_with(|| crate::fts::analyze_with_prefix(&clause.text, cfg));
                    let mut terms = heads.clone();
                    if let Some(fragment) = fragment {
                        let (expanded, matched) =
                            self.fts.expand_prefix(col_name, &clause.field, fragment);
                        expansions[ci] = Some(Expansion {
                            matched,
                            scored: expanded.len(),
                        });
                        terms.extend(expanded);
                    }
                    for (id, score) in self.fts.score(col_name, &clause.field, &terms) {
                        acc.entry(id)
                            .or_insert_with(|| vec![None; query.clauses.len()])[ci] = Some(score);
                    }
                } else {
                    analyzed
                        .entry((ci, cfg))
                        .or_insert_with(|| crate::fts::analyze(&clause.text, cfg));
                    let terms = &analyzed[&(ci, cfg)];
                    for (id, score) in self.fts.score(col_name, &clause.field, terms) {
                        acc.entry(id)
                            .or_insert_with(|| vec![None; query.clauses.len()])[ci] = Some(score);
                    }
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
                        clause_scores(&per_clause, &query.clauses, &expansions),
                    );
                }
            }
        }
        // `TopK` already resolves ties on `(collection, id)`, so no re-sort is needed here.
        let mut hits = self.finish(self.hits_from_topk(topk, &opts.projection), opts);
        self.annotate(&mut hits, query, opts.explain.then_some(&mut breakdown));
        Ok(hits)
    }

    /// One collection's completions for `leg` as `(term, df, distance)`, `distance` always `0`
    /// for [`Leg::Prefix`]. Empty when the collection doesn't index `field`, the fragment folds
    /// away to nothing, head conditioning excludes everything, or (fuzzy) the budget is `0`.
    fn suggest_one(
        &self,
        col_name: &str,
        field: &str,
        prefix: &str,
        opts: &SuggestOpts,
        leg: &Leg,
    ) -> Vec<(String, usize, usize)> {
        let Some(cfg) = self.fts.field_analyzer(col_name, field) else {
            return Vec::new();
        };
        let Some(col) = self.collections.get(col_name) else {
            return Vec::new();
        };
        // Fold-only, exactly as a prefix clause folds its fragment. The heads ARE kept here,
        // unlike before nidus-ucl; they condition the df rather than being scored.
        let (heads, Some(fragment)) = crate::fts::analyze_with_prefix(prefix, cfg) else {
            return Vec::new();
        };
        let head_docs = self.fts.head_docs(col_name, field, &heads);
        // No heads means unconditioned; an empty head set means nothing continues the phrase,
        // so every completion is dropped rather than falling back to the whole corpus.
        if head_docs.as_ref().is_some_and(HashSet::is_empty) {
            return Vec::new();
        }
        let filtered = !opts.filter.0.is_empty();
        let admit = |docnum: u32, id: &str| {
            if let Some(docs) = head_docs.as_ref()
                && !docs.contains(&docnum)
            {
                return false;
            }
            if !filtered {
                return true;
            }
            col.docs
                .get(id)
                .is_some_and(|entry| filter::matches(&opts.filter, &entry.attrs))
        };
        // Un-conditioned stays on the cheap path: no closure, so a tombstone-free index reads
        // each posting list's length instead of walking it (nidus-clv stage 1).
        let unconditioned = head_docs.is_none() && !filtered;
        match leg {
            Leg::Prefix => {
                let scored = if unconditioned {
                    self.fts.suggest(col_name, field, &fragment, None)
                } else {
                    self.fts.suggest(col_name, field, &fragment, Some(&admit))
                };
                scored.into_iter().map(|(term, df)| (term, df, 0)).collect()
            }
            Leg::Fuzzy => {
                // Budget from this collection's own fragment: analyzers differ per field, so
                // fragments (and their char counts) can differ collection to collection.
                let max_edits = crate::fts::fuzzy_budget_for(&fragment);
                if max_edits == 0 {
                    return Vec::new();
                }
                if unconditioned {
                    self.fts
                        .suggest_fuzzy(col_name, field, &fragment, max_edits, None)
                } else {
                    self.fts
                        .suggest_fuzzy(col_name, field, &fragment, max_edits, Some(&admit))
                }
            }
        }
    }

    /// Ranked completions from `field`'s vocabulary across `collections` (SPEC §7). The exact-
    /// prefix leg ranks `df` desc then term asc; if and only if it finds nothing at all does the
    /// fallback fuzzy leg run (`opts.fuzzy`), ranked `distance` asc, `df` desc, term asc.
    pub fn suggest(
        &self,
        collections: &[&str],
        field: &str,
        prefix: &str,
        opts: &SuggestOpts,
    ) -> Result<Suggestions> {
        filter::validate(&opts.filter)?;
        // Merged by term across the scope: one dropdown, so a completion two collections share is
        // one row whose `df` is the sum and whose distance is the best any collection found.
        let mut merged: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for &col_name in collections {
            for (term, df, distance) in
                self.suggest_one(col_name, field, prefix, opts, &Leg::Prefix)
            {
                merged
                    .entry(term)
                    .and_modify(|(d, existing)| {
                        *d = (*d).min(distance);
                        *existing += df;
                    })
                    .or_insert((distance, df));
            }
        }
        // Only when the prefix leg found nothing at all does the fuzzy leg sweep; a zero limit
        // has nothing to fill, so it is skipped too (nidus-972).
        let fuzzy = merged.is_empty() && opts.fuzzy && opts.limit > 0;
        if fuzzy {
            for &col_name in collections {
                for (term, df, distance) in
                    self.suggest_one(col_name, field, prefix, opts, &Leg::Fuzzy)
                {
                    merged
                        .entry(term)
                        .and_modify(|(d, existing)| {
                            *d = (*d).min(distance);
                            *existing += df;
                        })
                        .or_insert((distance, df));
                }
            }
        }
        let matched = merged.len();
        let mut scored: Vec<(String, usize, usize)> = merged
            .into_iter()
            .map(|(term, (distance, df))| (term, df, distance))
            .collect();
        if fuzzy {
            // (distance asc, df desc, term asc): a near-miss belongs above a common but distant
            // word. The prefix ranking never mixes with this leg's, since it only fires when
            // the prefix leg answered nothing.
            scored.sort_by(|a, b| {
                a.2.cmp(&b.2)
                    .then_with(|| b.1.cmp(&a.1))
                    .then_with(|| a.0.cmp(&b.0))
            });
        } else {
            // (df desc, term asc), unchanged: the tie-break is load-bearing, or two equal-df
            // completions truncate in whatever order the sort happened to leave them.
            scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        }
        scored.truncate(crate::fts::MAX_PREFIX_EXPANSION);
        scored.truncate(opts.limit);
        Ok(Suggestions {
            suggestions: scored
                .into_iter()
                .map(|(term, df, _)| Suggestion { term, df })
                .collect(),
            matched,
        })
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
        // A prefix expansion is the same for every hit sharing a (collection, field, clause),
        // so expand once per page rather than once per hit — highlighting is typeahead's
        // usual companion, and the page would otherwise repeat the range scan `top_k` times.
        let mut expansions: HashMap<(String, usize), Vec<String>> = HashMap::new();
        for hit in hits.iter_mut() {
            let a = hit.annotations.get_or_insert_with(Annotations::default);
            if let Some(b) = breakdown.as_deref_mut() {
                a.clauses = b
                    .remove(&(hit.collection.clone(), hit.id.clone()))
                    .unwrap_or_default();
            }
            if let Some(opts) = &query.highlight {
                a.highlights =
                    self.highlights_for(&hit.collection, &hit.id, query, opts, &mut expansions);
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
        expansions: &mut HashMap<(String, usize), Vec<String>>,
    ) -> Vec<Highlight> {
        let Some(entry) = self
            .collections
            .get(collection)
            .and_then(|c| c.docs.get(id))
        else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (ci, clause) in query.clauses.iter().enumerate() {
            let Some(cfg) = self.fts.field_analyzer(collection, &clause.field) else {
                continue;
            };
            let text = crate::fts::field_text(&entry.attrs, &clause.field);
            // Highlight the same expanded term list a prefix clause scored against, so a
            // prefix match highlights with no change to `highlight.rs` itself (D6).
            let terms = if clause.prefix {
                expansions
                    .entry((collection.to_string(), ci))
                    .or_insert_with(|| {
                        let (mut terms, fragment) =
                            crate::fts::analyze_with_prefix(&clause.text, cfg);
                        if let Some(fragment) = fragment {
                            let (expanded, _) =
                                self.fts.expand_prefix(collection, &clause.field, &fragment);
                            terms.extend(expanded);
                        }
                        terms
                    })
                    .clone()
            } else {
                crate::fts::analyze(&clause.text, cfg)
            };
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
