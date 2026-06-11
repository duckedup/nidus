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

use serde::{Deserialize, Serialize};

use crate::data::DataSegment;
use crate::model::{AnnConfig, AnnKind, Distance};
use crate::search::{QuantParams, dot_i8, euclidean_neg_sq_i8, hamming, pack_signs};

mod hnsw;
mod ivf;
mod persist;

pub(crate) use hnsw::HnswGraph;
pub(crate) use ivf::IvfIndex;
pub(crate) use persist::{load as load_index, save as save_index};

/// Borrowed view of an index's durable state, for zero-copy serialization on save.
/// Mirrors [`AnnSnapshot`] but holds references so a snapshot write never clones the
/// (potentially large) graph.
#[derive(Serialize)]
pub(crate) enum AnnSnapshotRef<'a> {
    Hnsw {
        rows: &'a [u64],
        links: &'a [Vec<Vec<u32>>],
        entry: Option<u32>,
        max_level: usize,
    },
    Ivf {
        centroids: &'a [f32],
        lists: &'a [Vec<u64>],
    },
}

/// Owned durable state of an index, decoded from a snapshot on load. Reconstituted
/// into a live [`Ann`] via [`Ann::from_snapshot`] (the config-derived fields — score
/// function, PRNG, tunables — come from the `AnnConfig`, not the file).
#[derive(Deserialize)]
pub(crate) enum AnnSnapshot {
    Hnsw {
        rows: Vec<u64>,
        links: Vec<Vec<Vec<u32>>>,
        entry: Option<u32>,
        max_level: usize,
    },
    Ivf {
        centroids: Vec<f32>,
        lists: Vec<Vec<u64>>,
    },
}

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

/// The metric the [`Walk`] scores in. `None` is exact f32 (the default, unchanged); the
/// quantized variants score the store's int8/binary codes for a cheaper walk. The codes
/// are borrowed from the store's quantization state by *physical row* — the same layout
/// the brute-force two-pass search uses — so this adds no new state or persistence.
enum WalkQuant<'a> {
    /// Exact f32: rows are scored straight from the `data` matrix.
    None,
    /// int8 scalar codes, flat row-major (`dim` per row). `euclidean` picks
    /// [`euclidean_neg_sq_i8`] over [`dot_i8`]; both are monotonic with the f32 score.
    /// `params` is the store's shared scale — the query must be quantized with the same
    /// scale the stored codes were, so the int8 scores stay mutually comparable.
    Int8 {
        codes: &'a [i8],
        dim: usize,
        euclidean: bool,
        params: QuantParams,
    },
    /// Binary sign-bit codes, flat row-major (`wpr` u64 words per row). Score is
    /// `-hamming`, monotone with cosine rank for unit vectors (cosine only).
    Binary { words: &'a [u64], wpr: usize },
}

/// How the ANN index measures "nearness" during build and search (nidus-ndu).
///
/// It always carries the f32 `data` matrix (IVF fits its centroids in f32, and the exact
/// variant scores rows straight from it) plus an optional quantized codebook. When a
/// codebook is present the *walk* — the graph hops / list scans the index does to pick a
/// candidate set — scores cheap quantized codes instead of f32; the store then reranks the
/// resulting candidate rows with the exact f32 score, so the walk only has to choose a good
/// candidate *set*. Higher score = nearer in every variant.
///
/// Borrows the store's `data` and (when quantized) its code matrix, so it is cheap to
/// build per query/build and `Sync` for the parallel HNSW build.
pub(crate) struct Walk<'a> {
    data: &'a DataSegment,
    score_fn: ScoreFn,
    quant: WalkQuant<'a>,
}

/// A query encoded once for a [`Walk`], reused to score it against many rows. Created by
/// [`Walk::query_scorer`]; for the exact variant it just borrows the f32 query.
pub(crate) struct QueryScorer<'a> {
    walk: &'a Walk<'a>,
    query: &'a [f32],
    code: QueryCode,
}

enum QueryCode {
    F32,
    Int8(Vec<i8>),
    Binary(Vec<u64>),
}

/// int8 row score as f32 — monotonic with the f32 metric under the shared symmetric scale.
fn i8_score(euclidean: bool, a: &[i8], b: &[i8]) -> f32 {
    if euclidean {
        euclidean_neg_sq_i8(a, b) as f32
    } else {
        dot_i8(a, b) as f32
    }
}

impl<'a> Walk<'a> {
    /// The exact f32 walk (no quantization) — byte-for-byte the pre-nidus-ndu behavior.
    pub(crate) fn exact(data: &'a DataSegment, distance: Distance) -> Self {
        Walk {
            data,
            score_fn: score_fn_for(distance),
            quant: WalkQuant::None,
        }
    }

    /// A walk that scores int8 codes (`codes` is row-major, `data.dimension()` per row);
    /// `params` is the store's shared quantization scale.
    pub(crate) fn int8(
        data: &'a DataSegment,
        codes: &'a [i8],
        params: QuantParams,
        distance: Distance,
    ) -> Self {
        Walk {
            data,
            score_fn: score_fn_for(distance),
            quant: WalkQuant::Int8 {
                codes,
                dim: data.dimension(),
                euclidean: distance == Distance::Euclidean,
                params,
            },
        }
    }

    /// A walk that scores binary sign-bit codes (`words` is row-major, `wpr` words/row).
    /// Cosine only (binary codes are an angular proxy — enforced at `open`).
    pub(crate) fn binary(data: &'a DataSegment, words: &'a [u64], wpr: usize) -> Self {
        Walk {
            data,
            score_fn: score_fn_for(Distance::Cosine),
            quant: WalkQuant::Binary { words, wpr },
        }
    }

    /// The f32 data matrix (IVF k-means fits centroids from it; the exact variant scores
    /// rows from it).
    pub(crate) fn data(&self) -> &DataSegment {
        self.data
    }

    /// Score physical rows `a` and `b` against each other (HNSW build heuristics). Uses the
    /// quantized codes when present, else exact f32.
    pub(crate) fn score_rows(&self, a: u64, b: u64) -> f32 {
        match &self.quant {
            WalkQuant::None => (self.score_fn)(self.data.row(a), self.data.row(b)),
            WalkQuant::Int8 {
                codes,
                dim,
                euclidean,
                ..
            } => {
                let (a, b) = (a as usize, b as usize);
                i8_score(
                    *euclidean,
                    &codes[a * dim..(a + 1) * dim],
                    &codes[b * dim..(b + 1) * dim],
                )
            }
            WalkQuant::Binary { words, wpr } => {
                let (a, b) = (a as usize, b as usize);
                -(hamming(
                    &words[a * wpr..(a + 1) * wpr],
                    &words[b * wpr..(b + 1) * wpr],
                ) as f32)
            }
        }
    }

    /// Encode `query` once into a scorer that scores it against any row (search path).
    pub(crate) fn query_scorer(&'a self, query: &'a [f32]) -> QueryScorer<'a> {
        let code = match &self.quant {
            WalkQuant::None => QueryCode::F32,
            WalkQuant::Int8 { dim, params, .. } => {
                let mut q = vec![0i8; *dim];
                params.quantize(query, &mut q);
                QueryCode::Int8(q)
            }
            WalkQuant::Binary { .. } => QueryCode::Binary(pack_signs(query)),
        };
        QueryScorer {
            walk: self,
            query,
            code,
        }
    }
}

impl QueryScorer<'_> {
    /// Score the encoded query against physical row `row`. Higher = nearer.
    pub(crate) fn score(&self, row: u64) -> f32 {
        match (&self.walk.quant, &self.code) {
            (WalkQuant::None, _) => (self.walk.score_fn)(self.query, self.walk.data.row(row)),
            (
                WalkQuant::Int8 {
                    codes,
                    dim,
                    euclidean,
                    ..
                },
                QueryCode::Int8(q),
            ) => {
                let r = row as usize;
                i8_score(*euclidean, q, &codes[r * dim..(r + 1) * dim])
            }
            (WalkQuant::Binary { words, wpr }, QueryCode::Binary(q)) => {
                let r = row as usize;
                -(hamming(q, &words[r * wpr..(r + 1) * wpr]) as f32)
            }
            // `query_scorer` always pairs a code with its own walk, so the cross
            // combinations are unreachable.
            _ => unreachable!("query code does not match walk metric"),
        }
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
    /// `open` and after `compact` renumbers rows. `workers` (from
    /// `Config::query_threads`) caps build concurrency; `1` builds serially and
    /// deterministically. The [`Walk`] decides whether the build scores exact f32 or
    /// quantized codes (nidus-ndu).
    pub(crate) fn build(&mut self, walk: &Walk, live_rows: &[u64], workers: usize) {
        match self {
            Ann::Hnsw(g) => g.build(walk, live_rows, workers),
            Ann::Ivf(i) => i.build(walk, live_rows, workers),
        }
    }

    /// Incrementally index `rows` (already appended to `data`) without a full rebuild.
    pub(crate) fn insert_rows(&mut self, walk: &Walk, rows: &[u64]) {
        match self {
            Ann::Hnsw(g) => g.insert_rows(walk, rows),
            Ann::Ivf(i) => i.insert_rows(walk, rows),
        }
    }

    /// A borrowed view of this index's durable state, for serialization on save.
    pub(crate) fn snapshot_ref(&self) -> AnnSnapshotRef<'_> {
        match self {
            Ann::Hnsw(g) => g.snapshot_ref(),
            Ann::Ivf(i) => i.snapshot_ref(),
        }
    }

    /// Rebuild a live index from a decoded snapshot. The kind must match `cfg.kind`
    /// (the caller validates this against the file header first).
    pub(crate) fn from_snapshot(
        cfg: AnnConfig,
        dim: usize,
        distance: Distance,
        snap: AnnSnapshot,
    ) -> Self {
        match snap {
            AnnSnapshot::Hnsw {
                rows,
                links,
                entry,
                max_level,
            } => Ann::Hnsw(HnswGraph::from_parts(
                cfg, dim, distance, rows, links, entry, max_level,
            )),
            AnnSnapshot::Ivf { centroids, lists } => {
                Ann::Ivf(IvfIndex::from_parts(cfg, dim, distance, centroids, lists))
            }
        }
    }

    /// Walk the index for up to `n_candidates` rows, best-first as `(row, score)`. The
    /// caller post-filters and reranks; recall rises with `n_candidates`. Scores are
    /// quantized-approximate when the [`Walk`] carries a codebook (the store reranks the
    /// rows exactly), exact f32 otherwise.
    pub(crate) fn search(
        &self,
        walk: &Walk,
        query: &[f32],
        n_candidates: usize,
    ) -> Vec<(u64, f32)> {
        match self {
            Ann::Hnsw(g) => g.search(walk, query, n_candidates),
            Ann::Ivf(i) => i.search(walk, query, n_candidates),
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
