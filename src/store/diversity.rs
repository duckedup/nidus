//! Vector-space result diversity (nidus-tx2): Maximal Marginal Relevance over an over-fetched
//! ranking, so near-identical passages stop crowding a page. Sibling to [`super::aggregate`],
//! which diversifies by *attribute value* instead. See `SPEC.md` §7.7.

use anyhow::{Result, bail};

use super::Store;
use crate::model::Hit;

/// How much deeper than a page a diversified search ranks, so MMR has candidates to spread
/// across. Smaller than `LIMIT_PER_OVERFETCH`: a cap deletes hits, MMR only reorders them.
pub(super) const MMR_OVERFETCH: usize = 4;

/// The documented quadratic bound. MMR needs pairwise similarity, so it is O(W² · dim); past
/// this many head candidates the tail keeps its score order rather than growing the cost.
pub(super) const MAX_DIVERSITY_WINDOW: usize = 512;

/// Reject a lambda that cannot weight anything, once per query. `NaN` would make every
/// comparison in the greedy selection false, silently degrading MMR to "keep the input order".
pub(super) fn validate(diversity: Option<f32>) -> Result<()> {
    let Some(lambda) = diversity else {
        return Ok(());
    };
    if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
        bail!("diversity must be a finite lambda in [0.0, 1.0], got {lambda}");
    }
    Ok(())
}

/// Cosine similarity from a precomputed norm pair. Explicitly cosine rather than a bare dot
/// product: vectors are unit-normalized on insert only when the store's distance is `Cosine`.
fn cosine(a: &[f32], na: f32, b: &[f32], nb: f32) -> f32 {
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (na * nb)
}

/// One candidate's vector and its norm, or `None` for a text-only record.
type Candidate<'a> = Option<(&'a [f32], f32)>;

/// Greedily reorder `window` by MMR, walking candidates in their existing ranked order so a
/// tie always resolves the way the input already did. Returns the new index order.
fn select(window: &[Candidate<'_>], scores: &[f32], lambda: f32) -> Vec<usize> {
    let n = window.len();
    let mut order = Vec::with_capacity(n);
    let mut taken = vec![false; n];
    // Redundancy against everything already selected, updated incrementally so the whole
    // selection stays O(n²) similarity computations rather than O(n³).
    let mut max_sim = vec![0.0f32; n];

    for _ in 0..n {
        let mut best: Option<(usize, f32)> = None;
        for i in 0..n {
            if taken[i] {
                continue;
            }
            let mmr = lambda * scores[i] - (1.0 - lambda) * max_sim[i];
            // Strictly greater: the first candidate in ranked order wins every tie, which is
            // what keeps the SPEC §7 total order (and therefore pagination) coherent.
            if best.is_none_or(|(_, b)| mmr > b) {
                best = Some((i, mmr));
            }
        }
        let Some((pick, _)) = best else { break };
        taken[pick] = true;
        order.push(pick);
        // A text-only record has no vector, so nothing can be measurably redundant with it.
        if let Some((pv, pn)) = window[pick] {
            for i in 0..n {
                if taken[i] {
                    continue;
                }
                if let Some((v, nrm)) = window[i] {
                    max_sim[i] = max_sim[i].max(cosine(pv, pn, v, nrm));
                }
            }
        }
    }
    order
}

impl Store {
    /// Reorder the head of a ranking by MMR. The head is bounded by [`MAX_DIVERSITY_WINDOW`];
    /// anything deeper keeps its score order, so the cost stays bounded rather than quadratic
    /// in whatever depth the caller over-fetched to.
    pub(super) fn diversify(&self, mut ranked: Vec<Hit>, lambda: f32) -> Vec<Hit> {
        let width = ranked.len().min(MAX_DIVERSITY_WINDOW);
        if width < 2 {
            return ranked;
        }
        let tail = ranked.split_off(width);

        // Vectors are read from the live index, never from the hit, so a `Projection` cannot
        // change what MMR measures.
        let window: Vec<Candidate<'_>> = ranked
            .iter()
            .map(|h| {
                let row = self
                    .collections
                    .get(&h.collection)
                    .and_then(|c| c.docs.get(&h.id))
                    .and_then(|e| e.row)?;
                let v = self.data.row(row);
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                Some((v, norm))
            })
            .collect();
        let scores: Vec<f32> = ranked.iter().map(|h| h.score).collect();
        let order = select(&window, &scores, lambda);

        let mut slots: Vec<Option<Hit>> = ranked.into_iter().map(Some).collect();
        let mut out: Vec<Hit> = order.into_iter().filter_map(|i| slots[i].take()).collect();
        out.extend(tail);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(vector, norm)` for a candidate, computed the way `diversify` does.
    fn cand(v: &[f32]) -> Candidate<'_> {
        Some((v, v.iter().map(|x| x * x).sum::<f32>().sqrt()))
    }

    #[test]
    fn lambda_one_is_pure_relevance() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        let w = [cand(&a), cand(&b), cand(&c)];
        assert_eq!(select(&w, &[0.9, 0.8, 0.7], 1.0), vec![0, 1, 2]);
    }

    #[test]
    fn a_near_duplicate_is_pushed_behind_a_novel_candidate() {
        // Rank 2 duplicates rank 1; rank 3 is orthogonal and scores lower. MMR must promote it.
        let a = [1.0, 0.0];
        let dup = [1.0, 0.0];
        let novel = [0.0, 1.0];
        let w = [cand(&a), cand(&dup), cand(&novel)];
        assert_eq!(select(&w, &[0.9, 0.89, 0.5], 0.5), vec![0, 2, 1]);
    }

    #[test]
    fn the_top_hit_is_never_displaced() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let w = [cand(&a), cand(&b)];
        for lambda in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(select(&w, &[0.9, 0.1], lambda)[0], 0, "lambda {lambda}");
        }
    }

    #[test]
    fn identical_candidates_keep_their_input_order() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        let c = [1.0, 0.0];
        let w = [cand(&a), cand(&b), cand(&c)];
        assert_eq!(select(&w, &[0.5, 0.5, 0.5], 0.5), vec![0, 1, 2]);
    }

    #[test]
    fn a_vectorless_candidate_is_never_redundant() {
        // Two duplicates plus a text-only hit: the vectorless one carries no penalty, so at
        // lambda 0 it outranks the duplicate despite the lower score.
        let a = [1.0, 0.0];
        let dup = [1.0, 0.0];
        let w = [cand(&a), cand(&dup), None];
        assert_eq!(select(&w, &[0.9, 0.8, 0.1], 0.0), vec![0, 2, 1]);
    }

    #[test]
    fn cosine_is_computed_on_unnormalized_vectors() {
        let a = [3.0, 0.0];
        let b = [10.0, 0.0];
        let na = 3.0;
        let nb = 10.0;
        assert!((cosine(&a, na, &b, nb) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_zero_vector_is_similar_to_nothing() {
        let z = [0.0, 0.0];
        let a = [1.0, 0.0];
        assert_eq!(cosine(&z, 0.0, &a, 1.0), 0.0);
    }

    #[test]
    fn validate_accepts_the_closed_unit_interval_and_nothing_else() {
        for ok in [None, Some(0.0), Some(0.5), Some(1.0)] {
            assert!(validate(ok).is_ok(), "{ok:?}");
        }
        for bad in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(validate(Some(bad)).is_err(), "{bad}");
        }
    }
}
