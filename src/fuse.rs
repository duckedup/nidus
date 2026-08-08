//! Reciprocal Rank Fusion: merge several ranked legs into one ranking, keeping each leg's
//! own rank and score for every document so a caller can explain the fused number.

use std::collections::{BTreeMap, HashMap};

use crate::model::{Hit, Value};

/// One document's `(rank, score)` inside each input leg, in leg order; `None` where that
/// leg did not return the document.
pub(crate) type LegScores = Vec<Option<(usize, f32)>>;

/// Per-document accumulator: fused score, attrs from the first leg to see it, per-leg detail.
type Acc = (f32, BTreeMap<String, Value>, LegScores);

/// A ranked leg feeding [`rrf_fuse`], with the weight its contribution carries.
pub(crate) struct FusionLeg {
    pub(crate) hits: Vec<Hit>,
    pub(crate) weight: f32,
}

impl FusionLeg {
    /// A leg at the neutral weight `1.0`. Multiplying by `1.0` is exact, so fusing only
    /// neutral legs reproduces unweighted RRF bit for bit.
    pub(crate) fn new(hits: Vec<Hit>) -> Self {
        Self { hits, weight: 1.0 }
    }

    /// Scale this leg's contribution to the fused score.
    pub(crate) fn weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

/// Fuse `legs` by reciprocal rank: a document scores `Σ wᵢ / (rrf_k + rankᵢ + 1)` over the legs
/// that returned it, and a document in one leg is carried by that leg alone. Sorted by fused
/// score descending, ties on `(collection, id)`; attrs come from the first leg to return it.
pub(crate) fn rrf_fuse(legs: Vec<FusionLeg>, rrf_k: f32) -> Vec<(Hit, LegScores)> {
    let leg_count = legs.len();
    let mut fused: HashMap<(String, String), Acc> = HashMap::new();
    for (leg_idx, leg) in legs.into_iter().enumerate() {
        for (rank, h) in leg.hits.into_iter().enumerate() {
            let leg_score = h.score;
            let e = fused
                .entry((h.collection, h.id))
                .or_insert_with(|| (0.0, h.attrs, vec![None; leg_count]));
            e.0 += leg.weight * (1.0 / (rrf_k + rank as f32 + 1.0));
            e.2[leg_idx] = Some((rank, leg_score));
        }
    }

    let mut out: Vec<(Hit, LegScores)> = fused
        .into_iter()
        .map(|((collection, id), (score, attrs, per_leg))| {
            (Hit::new(collection, id, score, attrs), per_leg)
        })
        .collect();
    out.sort_by(|a, b| {
        b.0.score
            .partial_cmp(&a.0.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.collection.cmp(&b.0.collection))
            .then_with(|| a.0.id.cmp(&b.0.id))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str, score: f32) -> Hit {
        Hit::new("c", id, score, BTreeMap::new())
    }

    #[test]
    fn a_doc_in_both_legs_sums_both_contributions() {
        let fused = rrf_fuse(
            vec![
                FusionLeg::new(vec![hit("a", 0.9), hit("b", 0.5)]),
                FusionLeg::new(vec![hit("a", 3.0)]),
            ],
            60.0,
        );
        assert_eq!(fused[0].0.id, "a");
        assert_eq!(fused[0].0.score, 1.0 / 61.0 + 1.0 / 61.0);
        assert_eq!(fused[1].0.score, 1.0 / 62.0);
    }

    #[test]
    fn each_leg_keeps_its_own_rank_and_score() {
        let fused = rrf_fuse(
            vec![
                FusionLeg::new(vec![hit("a", 0.9), hit("b", 0.5)]),
                FusionLeg::new(vec![hit("b", 3.0)]),
            ],
            60.0,
        );
        let b = fused.iter().find(|(h, _)| h.id == "b").unwrap();
        assert_eq!(b.1, vec![Some((1, 0.5)), Some((0, 3.0))]);
        let a = fused.iter().find(|(h, _)| h.id == "a").unwrap();
        assert_eq!(a.1, vec![Some((0, 0.9)), None]);
    }

    #[test]
    fn a_leg_weight_scales_only_that_legs_contribution() {
        let legs = vec![
            FusionLeg {
                hits: vec![hit("a", 0.9)],
                weight: 2.0,
            },
            FusionLeg::new(vec![hit("b", 3.0)]),
        ];
        let fused = rrf_fuse(legs, 60.0);
        assert_eq!(fused[0].0.id, "a");
        assert_eq!(fused[0].0.score, 2.0 * (1.0 / 61.0));
        assert_eq!(fused[1].0.score, 1.0 / 61.0);
    }

    #[test]
    fn equal_scores_break_on_collection_then_id() {
        // Two single-hit legs: both docs land at rank 0, so only the tie-break orders them.
        let tied = rrf_fuse(
            vec![
                FusionLeg::new(vec![Hit::new("z", "1", 1.0, BTreeMap::new())]),
                FusionLeg::new(vec![Hit::new("a", "9", 1.0, BTreeMap::new())]),
            ],
            0.0,
        );
        assert_eq!(tied[0].0.collection, "a");
        assert_eq!(tied[1].0.collection, "z");
    }

    #[test]
    fn no_legs_fuses_to_nothing() {
        assert!(rrf_fuse(Vec::new(), 60.0).is_empty());
    }
}
