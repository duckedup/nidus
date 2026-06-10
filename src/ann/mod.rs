//! Opt-in approximate-nearest-neighbour index (SPEC.md §9).
//!
//! Exact brute-force is the default; this module is reached only when
//! [`crate::Config::ann`] is set. It builds an in-RAM index over the `data` rows and
//! answers a query by *walking* the index for an over-fetched candidate set rather
//! than scanning every row. The store then post-filters those candidates by
//! scope/filter/`min_score` and reranks them with the exact f32 score — so the index
//! only has to pick a good candidate *set*; final ordering is always exact.
//!
//! Two algorithms are selected by [`AnnKind`]: [`HnswGraph`] (a navigable small-world
//! graph, the default) and [`IvfIndex`] (k-means inverted lists). Both are pure
//! safe Rust with no dependencies — the only randomness is a hand-rolled seeded
//! [`SplitMix64`] PRNG, so builds are deterministic and the logic runs under Miri.

use crate::data::DataSegment;
use crate::model::{AnnConfig, AnnKind, Distance};

mod hnsw;
mod ivf;

pub(crate) use hnsw::HnswGraph;
pub(crate) use ivf::IvfIndex;

/// Scoring function shared with the brute-force path: **higher = nearer**. Cosine and
/// dot-product use the raw dot product (vectors are unit-normalized on insert for
/// cosine); Euclidean uses negative squared distance. Picking the candidate set and
/// the final rerank both use this, so the index orders candidates exactly as the
/// exact path would.
pub(crate) type ScoreFn = fn(&[f32], &[f32]) -> f32;

/// The score function for a metric (mirrors the dispatch in [`crate::store`]).
pub(crate) fn score_fn_for(distance: Distance) -> ScoreFn {
    match distance {
        Distance::Cosine | Distance::DotProduct => crate::search::dot,
        Distance::Euclidean => crate::search::euclidean_neg_sq,
    }
}

/// A small, fast, fully-deterministic PRNG (splitmix64). Pure arithmetic, no deps —
/// used for HNSW level assignment and IVF centroid seeding so a build is reproducible
/// from [`AnnConfig::seed`](crate::AnnConfig::seed).
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform `f64` in `[0, 1)` (53-bit mantissa).
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    /// A uniform `usize` in `[0, n)` (`n` must be > 0).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// The active ANN index, maintained in RAM when [`crate::Config::ann`] is set. Built
/// on `open`, extended on `upsert`, rebuilt on `compact` — mirroring the quantization
/// state machine in [`crate::store`].
pub(crate) enum Ann {
    Hnsw(HnswGraph),
    Ivf(IvfIndex),
}

impl Ann {
    /// An empty index for `cfg`. Every metric is valid (the graph/lists score with the
    /// same f32 metric as exact search), so unlike binary quantization this never
    /// rejects a metric.
    pub(crate) fn empty(cfg: AnnConfig, dim: usize, distance: Distance) -> Self {
        match cfg.kind {
            AnnKind::Hnsw => Ann::Hnsw(HnswGraph::new(cfg, dim, distance)),
            AnnKind::Ivf => Ann::Ivf(IvfIndex::new(cfg, dim, distance)),
        }
    }

    /// Full (re)build from `live_rows` (physical row indices into `data`). Used on
    /// `open` and after `compact` renumbers rows.
    pub(crate) fn build(&mut self, data: &DataSegment, live_rows: &[u64]) {
        match self {
            Ann::Hnsw(g) => g.build(data, live_rows),
            Ann::Ivf(i) => i.build(data, live_rows),
        }
    }

    /// Incrementally index `rows` (already appended to `data`) without a full rebuild.
    pub(crate) fn insert_rows(&mut self, data: &DataSegment, rows: &[u64]) {
        match self {
            Ann::Hnsw(g) => g.insert_rows(data, rows),
            Ann::Ivf(i) => i.insert_rows(data, rows),
        }
    }

    /// Walk the index for up to `n_candidates` rows, best-first as `(row, score)`. The
    /// caller post-filters and reranks; recall rises with `n_candidates`.
    pub(crate) fn search(
        &self,
        data: &DataSegment,
        query: &[f32],
        n_candidates: usize,
    ) -> Vec<(u64, f32)> {
        match self {
            Ann::Hnsw(g) => g.search(data, query, n_candidates),
            Ann::Ivf(i) => i.search(data, query, n_candidates),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_is_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn splitmix64_f64_in_unit_range() {
        let mut r = SplitMix64::new(7);
        for _ in 0..10_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x), "{x} out of [0,1)");
        }
    }

    #[test]
    fn splitmix64_below_is_bounded() {
        let mut r = SplitMix64::new(99);
        for _ in 0..1000 {
            assert!(r.below(5) < 5);
        }
    }
}
