//! Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016), the default
//! ANN index. Pure safe Rust, no dependencies.
//!
//! Internal node ids are dense (`0..nodes`), assigned in build/insert order; `rows`
//! maps each node id back to its physical `data` row. Adjacency is `links[node][level]`
//! of neighbour node ids, with `links[node].len()` being that node's top level + 1.
//! Higher layers are progressively sparser; layer 0 holds every node. A query greedily
//! descends from the entry point through the upper layers, then runs an `ef`-width beam
//! search at layer 0; the best candidates (mapped back to rows) are returned for the
//! store to filter and rerank.
//!
//! Everything scores with [`ScoreFn`] where **higher = nearer**, so the beam is a
//! "keep the highest scores" collector and "closer" means "higher score".

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::ann::{AnnSnapshotRef, ScoreFn, SplitMix64, score_fn_for};
use crate::data::DataSegment;
use crate::model::{AnnConfig, Distance};

/// A `(score, node)` pair ordered by score, for the search heaps. `total_cmp` keeps
/// NaN from ever ranking as nearest.
#[derive(Clone, Copy)]
struct Scored {
    score: f32,
    node: u32,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}

/// `Scored` reversed, so a `BinaryHeap` (max-heap) yields the *lowest* score first —
/// used for the bounded result set whose worst element we must inspect.
#[derive(Clone, Copy)]
struct MinScored(Scored);
impl PartialEq for MinScored {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MinScored {}
impl PartialOrd for MinScored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinScored {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

pub(crate) struct HnswGraph {
    dim: usize,
    score_fn: ScoreFn,
    /// Tunables from [`AnnConfig`].
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    /// `1 / ln(m)` — the level-assignment normaliser.
    level_mult: f64,
    rng: SplitMix64,
    /// node id → physical data row.
    rows: Vec<u64>,
    /// node id → per-level adjacency (`links[n][l]` = neighbours of `n` at level `l`).
    links: Vec<Vec<Vec<u32>>>,
    /// Top-layer entry node, or `None` when empty.
    entry: Option<u32>,
    /// Level of the entry node.
    max_level: usize,
}

impl HnswGraph {
    pub(crate) fn new(cfg: AnnConfig, dim: usize, distance: Distance) -> Self {
        let m = cfg.m.max(1);
        // ln(1) == 0 would divide by zero; m == 1 degenerates to a single layer.
        let level_mult = if m > 1 { 1.0 / (m as f64).ln() } else { 0.0 };
        HnswGraph {
            dim,
            score_fn: score_fn_for(distance),
            m,
            m_max0: m.saturating_mul(2).max(1),
            ef_construction: cfg.ef_construction.max(1),
            ef_search: cfg.ef_search.max(1),
            level_mult,
            rng: SplitMix64::new(cfg.seed),
            rows: Vec::new(),
            links: Vec::new(),
            entry: None,
            max_level: 0,
        }
    }

    /// Reconstruct a graph from decoded durable state (snapshot load). Config-derived
    /// fields (tunables, score fn, PRNG) come from `cfg`/`distance`, not the file.
    pub(crate) fn from_parts(
        cfg: AnnConfig,
        dim: usize,
        distance: Distance,
        rows: Vec<u64>,
        links: Vec<Vec<Vec<u32>>>,
        entry: Option<u32>,
        max_level: usize,
    ) -> Self {
        let mut g = HnswGraph::new(cfg, dim, distance);
        g.rows = rows;
        g.links = links;
        g.entry = entry;
        g.max_level = max_level;
        g
    }

    /// A borrowed view of the durable graph state, for serialization.
    pub(crate) fn snapshot_ref(&self) -> AnnSnapshotRef<'_> {
        AnnSnapshotRef::Hnsw {
            rows: &self.rows,
            links: &self.links,
            entry: self.entry,
            max_level: self.max_level,
        }
    }

    pub(crate) fn build(&mut self, data: &DataSegment, live_rows: &[u64]) {
        self.rows.clear();
        self.links.clear();
        self.entry = None;
        self.max_level = 0;
        self.insert_rows(data, live_rows);
    }

    pub(crate) fn insert_rows(&mut self, data: &DataSegment, rows: &[u64]) {
        if self.dim == 0 {
            return;
        }
        for &row in rows {
            self.insert_one(data, row);
        }
    }

    /// Draw a node level: `floor(-ln(U) * level_mult)`, the geometric distribution that
    /// makes each layer ~`1/m` the size of the one below.
    fn random_level(&mut self) -> usize {
        if self.level_mult == 0.0 {
            return 0;
        }
        // `next_f64()` is in [0,1); shift to (0,1] so ln is finite.
        let u = 1.0 - self.rng.next_f64();
        (-u.ln() * self.level_mult).floor() as usize
    }

    fn insert_one(&mut self, data: &DataSegment, row: u64) {
        let node = self.rows.len() as u32;
        let level = self.random_level();
        self.rows.push(row);
        self.links.push(vec![Vec::new(); level + 1]);

        let Some(entry) = self.entry else {
            // First node becomes the entry point.
            self.entry = Some(node);
            self.max_level = level;
            return;
        };

        let q = data.row(row);
        let mut ep = entry;

        // Greedy descent through the layers above `level` (beam of 1).
        let mut lc = self.max_level;
        while lc > level {
            ep = self.greedy_descend(data, q, ep, lc);
            lc -= 1;
        }

        // From min(level, max_level) down to 0: beam search, connect, prune.
        let mut lc = level.min(self.max_level) as isize;
        let mut entry_points = vec![ep];
        while lc >= 0 {
            let level_u = lc as usize;
            let found = self.search_layer(data, q, &entry_points, self.ef_construction, level_u);
            let m = if level_u == 0 { self.m_max0 } else { self.m };
            let selected = self.select_neighbors(data, &found, m);

            for &nbr in &selected {
                self.connect(node, nbr, level_u);
                self.connect(nbr, node, level_u);
                self.prune(data, nbr, level_u);
            }

            entry_points = found.iter().map(|s| s.node).collect();
            if entry_points.is_empty() {
                entry_points.push(ep);
            }
            lc -= 1;
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(node);
        }
    }

    /// One greedy hop-to-the-best walk at `level`, returning the nearest node found.
    fn greedy_descend(&self, data: &DataSegment, q: &[f32], entry: u32, level: usize) -> u32 {
        let mut cur = entry;
        let mut cur_score = (self.score_fn)(q, data.row(self.rows[cur as usize]));
        loop {
            let mut improved = false;
            if let Some(neighbors) = self.links[cur as usize].get(level) {
                for &nbr in neighbors {
                    let s = (self.score_fn)(q, data.row(self.rows[nbr as usize]));
                    if s > cur_score {
                        cur_score = s;
                        cur = nbr;
                        improved = true;
                    }
                }
            }
            if !improved {
                return cur;
            }
        }
    }

    /// Beam search at `level`: explore from `entry_points`, keeping the `ef` nearest
    /// nodes. Returns them best-first.
    fn search_layer(
        &self,
        data: &DataSegment,
        q: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: usize,
    ) -> Vec<Scored> {
        let mut visited = vec![false; self.rows.len()];
        // `candidates`: max-heap, explore the best next. `result`: min-heap, so we can
        // pop the worst once it exceeds `ef`.
        let mut candidates: BinaryHeap<Scored> = BinaryHeap::new();
        let mut result: BinaryHeap<MinScored> = BinaryHeap::new();

        for &ep in entry_points {
            if visited[ep as usize] {
                continue;
            }
            visited[ep as usize] = true;
            let s = (self.score_fn)(q, data.row(self.rows[ep as usize]));
            let sc = Scored { score: s, node: ep };
            candidates.push(sc);
            result.push(MinScored(sc));
        }

        while let Some(cand) = candidates.pop() {
            // Worst score currently kept. If the best remaining candidate is worse than
            // that and the result set is already full, no further node can improve it.
            let worst = result
                .peek()
                .map(|m| m.0.score)
                .unwrap_or(f32::NEG_INFINITY);
            if result.len() >= ef && cand.score < worst {
                break;
            }
            if let Some(neighbors) = self.links[cand.node as usize].get(level) {
                for &nbr in neighbors {
                    if visited[nbr as usize] {
                        continue;
                    }
                    visited[nbr as usize] = true;
                    let s = (self.score_fn)(q, data.row(self.rows[nbr as usize]));
                    let worst = result
                        .peek()
                        .map(|m| m.0.score)
                        .unwrap_or(f32::NEG_INFINITY);
                    if result.len() < ef || s > worst {
                        let sc = Scored {
                            score: s,
                            node: nbr,
                        };
                        candidates.push(sc);
                        result.push(MinScored(sc));
                        if result.len() > ef {
                            result.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<Scored> = result.into_iter().map(|m| m.0).collect();
        out.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        out
    }

    /// The HNSW neighbour-selection heuristic: walk candidates nearest-first and keep
    /// one only if it is nearer to the base than to any already-kept neighbour. This
    /// spreads links across directions (better navigability) instead of clustering them
    /// all toward the single nearest region.
    fn select_neighbors(&self, data: &DataSegment, candidates: &[Scored], m: usize) -> Vec<u32> {
        let mut selected: Vec<u32> = Vec::with_capacity(m);
        // `candidates` arrives best-first from `search_layer`.
        for cand in candidates {
            if selected.len() >= m {
                break;
            }
            let cand_row = data.row(self.rows[cand.node as usize]);
            let keep = selected.iter().all(|&r| {
                let r_row = data.row(self.rows[r as usize]);
                // cand→base score vs cand→neighbour score: keep if cand is closer to the
                // base than to this neighbour (i.e. it adds a genuinely new direction).
                cand.score >= (self.score_fn)(cand_row, r_row)
            });
            if keep {
                selected.push(cand.node);
            }
        }
        // If the heuristic was too strict to fill `m`, top up with the remaining
        // nearest candidates so the node is not under-connected.
        if selected.len() < m {
            for cand in candidates {
                if selected.len() >= m {
                    break;
                }
                if !selected.contains(&cand.node) {
                    selected.push(cand.node);
                }
            }
        }
        selected
    }

    /// Add `to` to `from`'s adjacency at `level` (deduplicated).
    fn connect(&mut self, from: u32, to: u32, level: usize) {
        if from == to {
            return;
        }
        let adj = &mut self.links[from as usize][level];
        if !adj.contains(&to) {
            adj.push(to);
        }
    }

    /// Trim `node`'s level-`level` adjacency back to the level cap using the same
    /// selection heuristic, so bidirectional inserts never let a node's degree grow
    /// without bound.
    fn prune(&mut self, data: &DataSegment, node: u32, level: usize) {
        let cap = if level == 0 { self.m_max0 } else { self.m };
        let adj = &self.links[node as usize][level];
        if adj.len() <= cap {
            return;
        }
        let base = data.row(self.rows[node as usize]);
        let mut scored: Vec<Scored> = adj
            .iter()
            .map(|&n| Scored {
                score: (self.score_fn)(base, data.row(self.rows[n as usize])),
                node: n,
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        let kept = self.select_neighbors(data, &scored, cap);
        self.links[node as usize][level] = kept;
    }

    pub(crate) fn search(
        &self,
        data: &DataSegment,
        query: &[f32],
        n_candidates: usize,
    ) -> Vec<(u64, f32)> {
        if n_candidates == 0 || self.dim == 0 {
            return Vec::new();
        }
        let Some(entry) = self.entry else {
            return Vec::new();
        };

        // Descend the upper layers greedily, then beam-search layer 0 with a beam wide
        // enough to surface `n_candidates`.
        let mut ep = entry;
        let mut lc = self.max_level;
        while lc > 0 {
            ep = self.greedy_descend(data, query, ep, lc);
            lc -= 1;
        }
        let ef = self.ef_search.max(n_candidates);
        let mut found = self.search_layer(data, query, &[ep], ef, 0);
        found.truncate(n_candidates);
        found
            .into_iter()
            .map(|s| (self.rows[s.node as usize], s.score))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::AnnConfig;

    /// A tiny in-memory `DataSegment` of unit-ish vectors for Miri-clean graph tests.
    fn seg(dim: usize, rows: &[Vec<f32>]) -> DataSegment {
        let mut d = DataSegment::in_memory(dim);
        for r in rows {
            d.append(r).unwrap();
        }
        d
    }

    fn build_graph(data: &DataSegment, n: u64) -> HnswGraph {
        let mut g = HnswGraph::new(AnnConfig::hnsw(), data.dimension(), Distance::Cosine);
        let live: Vec<u64> = (0..n).collect();
        g.build(data, &live);
        g
    }

    #[test]
    fn empty_graph_returns_nothing() {
        let data = seg(3, &[]);
        let g = build_graph(&data, 0);
        assert!(g.search(&data, &[1.0, 0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn finds_exact_match_among_orthogonal_axes() {
        // Three orthogonal unit vectors; the query equals axis 1.
        let data = seg(
            3,
            &[
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        );
        let g = build_graph(&data, 3);
        let hits = g.search(&data, &[0.0, 1.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1, "row 1 is the exact match");
    }

    #[test]
    fn returns_candidates_best_first() {
        let data = seg(
            2,
            &[
                vec![1.0, 0.0],
                vec![0.9, 0.1],
                vec![0.0, 1.0],
                vec![-1.0, 0.0],
            ],
        );
        let g = build_graph(&data, 4);
        let hits = g.search(&data, &[1.0, 0.0], 4);
        // Scores must be non-increasing.
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1, "candidates not best-first: {hits:?}");
        }
        // Nearest is row 0, then row 1.
        assert_eq!(hits[0].0, 0);
    }

    #[test]
    fn level_assignment_is_deterministic_for_seed() {
        let mut a = HnswGraph::new(AnnConfig::hnsw().seed(123), 4, Distance::Cosine);
        let mut b = HnswGraph::new(AnnConfig::hnsw().seed(123), 4, Distance::Cosine);
        for _ in 0..200 {
            assert_eq!(a.random_level(), b.random_level());
        }
    }

    #[test]
    fn incremental_insert_matches_membership() {
        let rows: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let t = i as f32 / 20.0;
                vec![t.cos(), t.sin()]
            })
            .collect();
        let data = seg(2, &rows);
        let mut g = HnswGraph::new(AnnConfig::hnsw(), 2, Distance::Cosine);
        // Insert in two incremental batches.
        g.insert_rows(&data, &(0..10).collect::<Vec<_>>());
        g.insert_rows(&data, &(10..20).collect::<Vec<_>>());
        let hits = g.search(&data, &rows[7], 1);
        assert_eq!(hits[0].0, 7, "self should be the nearest neighbour");
    }
}
