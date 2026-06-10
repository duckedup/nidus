//! Inverted-file (IVF) index: k-means centroids partition the vectors into lists; a
//! query scores the centroids, probes the nearest `n_probe` lists, and scores only the
//! rows in those lists. Pure safe Rust, no dependencies.
//!
//! Lower edge memory than HNSW, but the centroids are fit from the rows present at
//! build time. Incremental `insert_rows` assigns each new row to its nearest existing
//! centroid without refitting, so heavy growth drifts the partition until the next
//! `compact` triggers a full rebuild. Candidate scores returned here are already the
//! exact f32 metric (we score real rows), so recall loss comes only from rows sitting
//! in unprobed lists.

use crate::ann::{AnnSnapshotRef, ScoreFn, SplitMix64, score_fn_for};
use crate::data::DataSegment;
use crate::model::{AnnConfig, Distance};

/// Fixed Lloyd's-iteration count for the k-means fit. Enough to settle on random
/// embedding data without making the build pass open-ended.
const KMEANS_ITERS: usize = 12;

pub(crate) struct IvfIndex {
    dim: usize,
    distance: Distance,
    score_fn: ScoreFn,
    n_lists_cfg: usize,
    n_probe: usize,
    seed: u64,
    /// `n_lists * dim` flat centroids (unit-normalized when the metric is cosine).
    centroids: Vec<f32>,
    /// One inverted list of physical rows per centroid.
    lists: Vec<Vec<u64>>,
}

impl IvfIndex {
    pub(crate) fn new(cfg: AnnConfig, dim: usize, distance: Distance) -> Self {
        IvfIndex {
            dim,
            distance,
            score_fn: score_fn_for(distance),
            n_lists_cfg: cfg.n_lists,
            n_probe: cfg.n_probe.max(1),
            seed: cfg.seed,
            centroids: Vec::new(),
            lists: Vec::new(),
        }
    }

    fn n_centroids(&self) -> usize {
        self.centroids.len() / self.dim.max(1)
    }

    /// Reconstruct an index from decoded durable state (snapshot load).
    pub(crate) fn from_parts(
        cfg: AnnConfig,
        dim: usize,
        distance: Distance,
        centroids: Vec<f32>,
        lists: Vec<Vec<u64>>,
    ) -> Self {
        let mut ix = IvfIndex::new(cfg, dim, distance);
        ix.centroids = centroids;
        ix.lists = lists;
        ix
    }

    /// A borrowed view of the durable index state, for serialization.
    pub(crate) fn snapshot_ref(&self) -> AnnSnapshotRef<'_> {
        AnnSnapshotRef::Ivf {
            centroids: &self.centroids,
            lists: &self.lists,
        }
    }

    pub(crate) fn build(&mut self, data: &DataSegment, live_rows: &[u64]) {
        self.centroids.clear();
        self.lists.clear();
        if self.dim == 0 || live_rows.is_empty() {
            return;
        }

        // List count: configured, or ~sqrt(n); never more than the live rows.
        let auto = (live_rows.len() as f64).sqrt().round() as usize;
        let k = self.n_lists_cfg.max(1).min(live_rows.len());
        let k = if self.n_lists_cfg == 0 {
            auto.max(1).min(live_rows.len())
        } else {
            k
        };

        // Seed centroids from k distinct rows (Fisher–Yates prefix over a row index
        // copy), then run fixed Lloyd iterations.
        let mut rng = SplitMix64::new(self.seed);
        let mut idx: Vec<u64> = live_rows.to_vec();
        for i in 0..k {
            let j = i + rng.below(idx.len() - i);
            idx.swap(i, j);
        }
        self.centroids = vec![0.0; k * self.dim];
        for (c, &row) in idx[..k].iter().enumerate() {
            self.centroids[c * self.dim..(c + 1) * self.dim].copy_from_slice(data.row(row));
        }

        let mut assign = vec![0usize; live_rows.len()];
        for _ in 0..KMEANS_ITERS {
            // Assignment step.
            let mut changed = false;
            for (n, &row) in live_rows.iter().enumerate() {
                let c = self.nearest_centroid(data.row(row));
                if assign[n] != c {
                    assign[n] = c;
                    changed = true;
                }
            }
            // Update step: centroid = mean of its members (re-normalized for cosine).
            let mut sums = vec![0.0f32; k * self.dim];
            let mut counts = vec![0u32; k];
            for (n, &row) in live_rows.iter().enumerate() {
                let c = assign[n];
                counts[c] += 1;
                let v = data.row(row);
                let dst = &mut sums[c * self.dim..(c + 1) * self.dim];
                for (d, &x) in dst.iter_mut().zip(v) {
                    *d += x;
                }
            }
            for c in 0..k {
                if counts[c] == 0 {
                    continue; // keep an empty centroid where it was
                }
                let inv = 1.0 / counts[c] as f32;
                let cen = &mut self.centroids[c * self.dim..(c + 1) * self.dim];
                for (d, &s) in cen.iter_mut().zip(&sums[c * self.dim..(c + 1) * self.dim]) {
                    *d = s * inv;
                }
                if self.distance == Distance::Cosine {
                    crate::search::normalize(cen);
                }
            }
            if !changed {
                break;
            }
        }

        // Materialize the inverted lists from the final assignment.
        self.lists = vec![Vec::new(); k];
        for (n, &row) in live_rows.iter().enumerate() {
            let c = self.nearest_centroid(data.row(row));
            assign[n] = c;
            self.lists[c].push(row);
        }
    }

    pub(crate) fn insert_rows(&mut self, data: &DataSegment, rows: &[u64]) {
        if self.dim == 0 || self.n_centroids() == 0 {
            // No centroids yet (index built empty) — fold these in via a full build.
            if !rows.is_empty() {
                self.build(data, rows);
            }
            return;
        }
        for &row in rows {
            let c = self.nearest_centroid(data.row(row));
            self.lists[c].push(row);
        }
    }

    /// Index of the centroid with the highest score for `v`.
    fn nearest_centroid(&self, v: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..self.n_centroids() {
            let cen = &self.centroids[c * self.dim..(c + 1) * self.dim];
            let s = (self.score_fn)(v, cen);
            if s > best_score {
                best_score = s;
                best = c;
            }
        }
        best
    }

    pub(crate) fn search(
        &self,
        data: &DataSegment,
        query: &[f32],
        n_candidates: usize,
    ) -> Vec<(u64, f32)> {
        if n_candidates == 0 || self.dim == 0 || self.n_centroids() == 0 {
            return Vec::new();
        }

        // Rank centroids by score, take the nearest `n_probe`.
        let mut centroid_scores: Vec<(f32, usize)> = (0..self.n_centroids())
            .map(|c| {
                let cen = &self.centroids[c * self.dim..(c + 1) * self.dim];
                ((self.score_fn)(query, cen), c)
            })
            .collect();
        let probe = self.n_probe.min(centroid_scores.len());
        centroid_scores.select_nth_unstable_by(probe - 1, |a, b| b.0.total_cmp(&a.0));

        // Score every row in the probed lists; keep the best `n_candidates`.
        let mut scored: Vec<(u64, f32)> = Vec::new();
        for &(_, c) in &centroid_scores[..probe] {
            for &row in &self.lists[c] {
                let s = (self.score_fn)(query, data.row(row));
                scored.push((row, s));
            }
        }
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(n_candidates);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(dim: usize, rows: &[Vec<f32>]) -> DataSegment {
        let mut d = DataSegment::in_memory(dim);
        for r in rows {
            d.append(r).unwrap();
        }
        d
    }

    fn build(data: &DataSegment, n: u64, cfg: AnnConfig) -> IvfIndex {
        let mut ix = IvfIndex::new(cfg, data.dimension(), Distance::Cosine);
        ix.build(data, &(0..n).collect::<Vec<_>>());
        ix
    }

    #[test]
    fn empty_index_returns_nothing() {
        let data = seg(3, &[]);
        let ix = build(&data, 0, AnnConfig::ivf());
        assert!(ix.search(&data, &[1.0, 0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn finds_match_with_full_probe() {
        // Two tight clusters of unit vectors (cosine needs normalized inputs); probe
        // enough lists to cover both.
        let mut rows = Vec::new();
        for i in 0..8 {
            let a = i as f32 * 0.01; // cluster A near angle 0 (+x)
            rows.push(vec![a.cos(), a.sin()]);
        }
        for i in 0..8 {
            let a = std::f32::consts::FRAC_PI_2 + i as f32 * 0.01; // cluster B near +y
            rows.push(vec![a.cos(), a.sin()]);
        }
        let data = seg(2, &rows);
        let ix = build(&data, 16, AnnConfig::ivf().n_lists(2).n_probe(2));
        let hits = ix.search(&data, &rows[3], 1);
        assert_eq!(hits[0].0, 3, "exact row should surface with full probe");
    }

    #[test]
    fn candidates_are_best_first() {
        let rows: Vec<Vec<f32>> = (0..30)
            .map(|i| {
                let t = i as f32 / 30.0 * std::f32::consts::TAU;
                vec![t.cos(), t.sin()]
            })
            .collect();
        let data = seg(2, &rows);
        let ix = build(&data, 30, AnnConfig::ivf().n_lists(4).n_probe(4));
        let hits = ix.search(&data, &[1.0, 0.0], 5);
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1, "not best-first: {hits:?}");
        }
    }

    #[test]
    fn incremental_insert_lands_in_a_list() {
        let rows: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let t = i as f32 / 20.0;
                vec![t.cos(), t.sin()]
            })
            .collect();
        let data = seg(2, &rows);
        let mut ix = IvfIndex::new(AnnConfig::ivf().n_lists(3), 2, Distance::Cosine);
        ix.build(&data, &(0..15).collect::<Vec<_>>());
        ix.insert_rows(&data, &(15..20).collect::<Vec<_>>());
        let total: usize = ix.lists.iter().map(|l| l.len()).sum();
        assert_eq!(total, 20, "all rows should be assigned to some list");
    }
}
