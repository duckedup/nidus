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
//! Everything scores through the [`Walk`] where **higher = nearer**, so the beam is a
//! "keep the highest scores" collector and "closer" means "higher score". The walk is
//! exact f32 by default, or quantized int8/binary codes when the store combines ANN with
//! quantization (nidus-ndu) — build and search then run in that same quantized space.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use crate::ann::{AnnSnapshotRef, SplitMix64, Walk};
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
    pub(crate) fn new(cfg: AnnConfig, dim: usize, _distance: Distance) -> Self {
        let m = cfg.m.max(1);
        // ln(1) == 0 would divide by zero; m == 1 degenerates to a single layer.
        let level_mult = if m > 1 { 1.0 / (m as f64).ln() } else { 0.0 };
        HnswGraph {
            dim,
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

    pub(crate) fn build(&mut self, walk: &Walk, live_rows: &[u64], workers: usize) {
        self.rows.clear();
        self.links.clear();
        self.entry = None;
        self.max_level = 0;
        // Parallelize only a large from-scratch build; small builds and incremental
        // upserts stay serial (and deterministic). The serial path is byte-identical
        // to before this knob existed.
        if workers > 1 && self.dim > 0 && live_rows.len() >= PARALLEL_BUILD_MIN {
            self.build_parallel(walk, live_rows, workers);
        } else {
            self.insert_rows(walk, live_rows);
        }
    }

    pub(crate) fn insert_rows(&mut self, walk: &Walk, rows: &[u64]) {
        if self.dim == 0 {
            return;
        }
        for &row in rows {
            self.insert_one(walk, row);
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

    fn insert_one(&mut self, walk: &Walk, row: u64) {
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

        // The "query" is the inserting row; score every node against it via the walk
        // (quantized codes or exact f32). `score(r)` = nearness of physical row `r` to it.
        let score = |r: u64| walk.score_rows(row, r);
        let mut ep = entry;

        // Greedy descent through the layers above `level` (beam of 1).
        let mut lc = self.max_level;
        while lc > level {
            ep = self.greedy_descend(&score, ep, lc);
            lc -= 1;
        }

        // From min(level, max_level) down to 0: beam search, connect, prune.
        let mut lc = level.min(self.max_level) as isize;
        let mut entry_points = vec![ep];
        while lc >= 0 {
            let level_u = lc as usize;
            let found = self.search_layer(&score, &entry_points, self.ef_construction, level_u);
            let m = if level_u == 0 { self.m_max0 } else { self.m };
            let selected = self.select_neighbors(walk, &found, m);

            for &nbr in &selected {
                self.connect(node, nbr, level_u);
                self.connect(nbr, node, level_u);
                self.prune(walk, nbr, level_u);
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
    /// `score(row)` gives the nearness of physical `row` to the fixed probe (an inserting
    /// row during build, the encoded query during search).
    fn greedy_descend(&self, score: &impl Fn(u64) -> f32, entry: u32, level: usize) -> u32 {
        let mut cur = entry;
        let mut cur_score = score(self.rows[cur as usize]);
        loop {
            let mut improved = false;
            if let Some(neighbors) = self.links[cur as usize].get(level) {
                for &nbr in neighbors {
                    let s = score(self.rows[nbr as usize]);
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
    /// nodes. Returns them best-first. `score` is the probe scorer (see [`greedy_descend`]).
    fn search_layer(
        &self,
        score: &impl Fn(u64) -> f32,
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
            let s = score(self.rows[ep as usize]);
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
                    let s = score(self.rows[nbr as usize]);
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
    fn select_neighbors(&self, walk: &Walk, candidates: &[Scored], m: usize) -> Vec<u32> {
        select_neighbors(&self.rows, walk, candidates, m)
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
    fn prune(&mut self, walk: &Walk, node: u32, level: usize) {
        let cap = if level == 0 { self.m_max0 } else { self.m };
        let adj = &self.links[node as usize][level];
        if adj.len() <= cap {
            return;
        }
        let base = self.rows[node as usize];
        let mut scored: Vec<Scored> = adj
            .iter()
            .map(|&n| Scored {
                score: walk.score_rows(base, self.rows[n as usize]),
                node: n,
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        let kept = self.select_neighbors(walk, &scored, cap);
        self.links[node as usize][level] = kept;
    }

    pub(crate) fn search(
        &self,
        walk: &Walk,
        query: &[f32],
        n_candidates: usize,
    ) -> Vec<(u64, f32)> {
        if n_candidates == 0 || self.dim == 0 {
            return Vec::new();
        }
        let Some(entry) = self.entry else {
            return Vec::new();
        };

        // Encode the query once for the walk's metric, then score every visited row
        // against it. Quantized when the walk carries a codebook, exact f32 otherwise.
        let qs = walk.query_scorer(query);
        let score = |r: u64| qs.score(r);

        // Descend the upper layers greedily, then beam-search layer 0 with a beam wide
        // enough to surface `n_candidates`.
        let mut ep = entry;
        let mut lc = self.max_level;
        while lc > 0 {
            ep = self.greedy_descend(&score, ep, lc);
            lc -= 1;
        }
        let ef = self.ef_search.max(n_candidates);
        let mut found = self.search_layer(&score, &[ep], ef, 0);
        found.truncate(n_candidates);
        found
            .into_iter()
            .map(|s| (self.rows[s.node as usize], s.score))
            .collect()
    }

    /// Build the graph across `workers` threads. Node levels are assigned serially
    /// (deterministic, cheap), then the expensive per-node neighbour search + linking
    /// runs concurrently: adjacency is guarded by one `Mutex` per node and the entry
    /// point by an `RwLock`. Edges always lock the two endpoints in node-id order, so
    /// there is no deadlock; safe Rust rules out data races, so the only effect of
    /// concurrency is that the graph (and thus exact recall) varies slightly with the
    /// thread count — determinism holds only on the serial path.
    fn build_parallel(&mut self, walk: &Walk, live_rows: &[u64], workers: usize) {
        let n = live_rows.len();
        self.rows = live_rows.to_vec();
        let levels: Vec<usize> = (0..n).map(|_| self.random_level()).collect();
        let links: Vec<Mutex<Vec<Vec<u32>>>> = levels
            .iter()
            .map(|&l| Mutex::new(vec![Vec::new(); l + 1]))
            .collect();
        // Seed the entry with node 0; higher-level nodes promote it as they insert.
        let entry_state = RwLock::new((0u32, levels[0]));
        let counter = AtomicUsize::new(1); // node 0 is the seed; insert 1..n
        let (m, m_max0, ef_c) = (self.m, self.m_max0, self.ef_construction);
        let rows = &self.rows;
        let links_ref = &links;
        let entry_ref = &entry_state;
        let levels_ref = &levels;
        let counter_ref = &counter;

        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(move || {
                    loop {
                        let i = counter_ref.fetch_add(1, AtomicOrdering::Relaxed);
                        if i >= n {
                            break;
                        }
                        insert_locked(
                            i as u32, rows, walk, links_ref, entry_ref, levels_ref, m, m_max0, ef_c,
                        );
                    }
                });
            }
        });

        self.links = links.into_iter().map(|m| m.into_inner().unwrap()).collect();
        let (entry, max_level) = entry_state.into_inner().unwrap();
        self.entry = Some(entry);
        self.max_level = max_level;
    }
}

/// Below this node count a parallel build isn't worth the thread/lock overhead — the
/// serial path is used instead (also the case for incremental upserts).
const PARALLEL_BUILD_MIN: usize = 1024;

/// The HNSW neighbour-selection heuristic (free function so the serial method and the
/// parallel builder share one implementation). Walks candidates nearest-first, keeping
/// one only if it is nearer to the base than to any already-kept neighbour; tops up
/// with the nearest remaining if the heuristic underfills `m`.
fn select_neighbors(rows: &[u64], walk: &Walk, candidates: &[Scored], m: usize) -> Vec<u32> {
    let mut selected: Vec<u32> = Vec::with_capacity(m);
    for cand in candidates {
        if selected.len() >= m {
            break;
        }
        let cand_row = rows[cand.node as usize];
        let keep = selected
            .iter()
            .all(|&r| cand.score >= walk.score_rows(cand_row, rows[r as usize]));
        if keep {
            selected.push(cand.node);
        }
    }
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

/// A snapshot copy of `node`'s level-`level` neighbours, taken under its lock and
/// released immediately (so scoring never holds a lock).
fn nbrs_locked(links: &[Mutex<Vec<Vec<u32>>>], node: u32, level: usize) -> Vec<u32> {
    let g = links[node as usize].lock().unwrap();
    g.get(level).cloned().unwrap_or_default()
}

/// Locked variant of `greedy_descend` for the parallel build. `score(row)` is the
/// inserting row's walk scorer (see [`HnswGraph::greedy_descend`]).
fn greedy_descend_locked(
    rows: &[u64],
    score: &impl Fn(u64) -> f32,
    links: &[Mutex<Vec<Vec<u32>>>],
    entry: u32,
    level: usize,
) -> u32 {
    let mut cur = entry;
    let mut cur_score = score(rows[cur as usize]);
    loop {
        let mut improved = false;
        for nbr in nbrs_locked(links, cur, level) {
            let s = score(rows[nbr as usize]);
            if s > cur_score {
                cur_score = s;
                cur = nbr;
                improved = true;
            }
        }
        if !improved {
            return cur;
        }
    }
}

/// Locked variant of `search_layer` for the parallel build.
#[allow(clippy::too_many_arguments)]
fn search_layer_locked(
    rows: &[u64],
    score: &impl Fn(u64) -> f32,
    links: &[Mutex<Vec<Vec<u32>>>],
    entry_points: &[u32],
    ef: usize,
    level: usize,
    n_nodes: usize,
) -> Vec<Scored> {
    let mut visited = vec![false; n_nodes];
    let mut candidates: BinaryHeap<Scored> = BinaryHeap::new();
    let mut result: BinaryHeap<MinScored> = BinaryHeap::new();

    for &ep in entry_points {
        if visited[ep as usize] {
            continue;
        }
        visited[ep as usize] = true;
        let s = score(rows[ep as usize]);
        let sc = Scored { score: s, node: ep };
        candidates.push(sc);
        result.push(MinScored(sc));
    }

    while let Some(cand) = candidates.pop() {
        let worst = result
            .peek()
            .map(|m| m.0.score)
            .unwrap_or(f32::NEG_INFINITY);
        if result.len() >= ef && cand.score < worst {
            break;
        }
        for nbr in nbrs_locked(links, cand.node, level) {
            if visited[nbr as usize] {
                continue;
            }
            visited[nbr as usize] = true;
            let s = score(rows[nbr as usize]);
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

    let mut out: Vec<Scored> = result.into_iter().map(|m| m.0).collect();
    out.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
    out
}

/// Add the bidirectional edge `node ↔ nbr` at `level` and prune `nbr` back to its
/// degree cap, all under both endpoints' locks (acquired in node-id order to avoid
/// deadlock).
#[allow(clippy::too_many_arguments)]
fn link_and_prune(
    rows: &[u64],
    walk: &Walk,
    links: &[Mutex<Vec<Vec<u32>>>],
    node: u32,
    nbr: u32,
    level: usize,
    m: usize,
    m_max0: usize,
) {
    let (a, b) = (node.min(nbr), node.max(nbr));
    let mut ga = links[a as usize].lock().unwrap();
    let mut gb = links[b as usize].lock().unwrap();
    // `ga`/`gb` are disjoint guards, so &mut to each at once is sound.
    let (g_node, g_nbr): (&mut Vec<Vec<u32>>, &mut Vec<Vec<u32>>) = if node == a {
        (&mut ga, &mut gb)
    } else {
        (&mut gb, &mut ga)
    };

    if level < g_node.len() && !g_node[level].contains(&nbr) {
        g_node[level].push(nbr);
    }
    if level < g_nbr.len() {
        if !g_nbr[level].contains(&node) {
            g_nbr[level].push(node);
        }
        let cap = if level == 0 { m_max0 } else { m };
        if g_nbr[level].len() > cap {
            let base = rows[nbr as usize];
            let mut scored: Vec<Scored> = g_nbr[level]
                .iter()
                .map(|&x| Scored {
                    score: walk.score_rows(base, rows[x as usize]),
                    node: x,
                })
                .collect();
            scored.sort_unstable_by(|x, y| y.score.total_cmp(&x.score));
            g_nbr[level] = select_neighbors(rows, walk, &scored, cap);
        }
    }
}

/// Insert one node into the locked graph (the concurrent counterpart of `insert_one`).
#[allow(clippy::too_many_arguments)]
fn insert_locked(
    node: u32,
    rows: &[u64],
    walk: &Walk,
    links: &[Mutex<Vec<Vec<u32>>>],
    entry_state: &RwLock<(u32, usize)>,
    levels: &[usize],
    m: usize,
    m_max0: usize,
    ef_c: usize,
) {
    // Score every node against this inserting row, via the walk (quantized or exact).
    let node_row = rows[node as usize];
    let score = |r: u64| walk.score_rows(node_row, r);
    let node_level = levels[node as usize];
    let (mut ep, top) = {
        let g = entry_state.read().unwrap();
        (g.0, g.1)
    };

    // Greedy descent through the layers above this node's level.
    let mut lc = top;
    while lc > node_level {
        ep = greedy_descend_locked(rows, &score, links, ep, lc);
        lc -= 1;
    }

    // From min(node_level, top) down to 0: beam search, connect, prune.
    let mut lc = node_level.min(top) as isize;
    let mut entry_points = vec![ep];
    while lc >= 0 {
        let level = lc as usize;
        let found =
            search_layer_locked(rows, &score, links, &entry_points, ef_c, level, rows.len());
        let cap = if level == 0 { m_max0 } else { m };
        let selected = select_neighbors(rows, walk, &found, cap);
        for &nbr in &selected {
            if nbr != node {
                link_and_prune(rows, walk, links, node, nbr, level, m, m_max0);
            }
        }
        entry_points = found.iter().map(|s| s.node).collect();
        if entry_points.is_empty() {
            entry_points.push(ep);
        }
        lc -= 1;
    }

    if node_level > top {
        let mut g = entry_state.write().unwrap();
        if node_level > g.1 {
            *g = (node, node_level);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Segments;
    use crate::model::AnnConfig;

    /// A tiny in-memory `Segments` of unit-ish vectors for Miri-clean graph tests.
    fn seg(dim: usize, rows: &[Vec<f32>]) -> Segments {
        let mut d = Segments::in_memory_with(dim, Distance::Cosine);
        for r in rows {
            d.append(r).unwrap();
        }
        d
    }

    fn build_graph(data: &Segments, n: u64) -> HnswGraph {
        let mut g = HnswGraph::new(AnnConfig::hnsw(), data.dimension(), Distance::Cosine);
        let live: Vec<u64> = (0..n).collect();
        g.build(&Walk::exact(data, Distance::Cosine), &live, 1);
        g
    }

    #[test]
    fn empty_graph_returns_nothing() {
        let data = seg(3, &[]);
        let g = build_graph(&data, 0);
        assert!(
            g.search(&Walk::exact(&data, Distance::Cosine), &[1.0, 0.0, 0.0], 5)
                .is_empty()
        );
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
        let hits = g.search(&Walk::exact(&data, Distance::Cosine), &[0.0, 1.0, 0.0], 1);
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
        let hits = g.search(&Walk::exact(&data, Distance::Cosine), &[1.0, 0.0], 4);
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
        g.insert_rows(
            &Walk::exact(&data, Distance::Cosine),
            &(0..10).collect::<Vec<_>>(),
        );
        g.insert_rows(
            &Walk::exact(&data, Distance::Cosine),
            &(10..20).collect::<Vec<_>>(),
        );
        let hits = g.search(&Walk::exact(&data, Distance::Cosine), &rows[7], 1);
        assert_eq!(hits[0].0, 7, "self should be the nearest neighbour");
    }
}
