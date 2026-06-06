//! nidus-bench — cross-engine exact-KNN performance-parity harness.
//!
//! The goal is **parity, not winning**: confirm nidus's exact brute-force cosine KNN
//! stays in line with DuckDB and LanceDB, and catch regressions over time. Every engine
//! is pinned to *exact* search (no ANN index). The harness computes its own independent
//! exact ground truth and reports each engine's **recall@k** against it — including
//! nidus's, so no engine is trusted as the oracle; ~100% confirms the configs are exact.

use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;

pub mod data;
pub mod engines;
pub mod metrics;
pub mod report;

use data::Dataset;
use metrics::Timings;

/// A vector store under test. Sync by design — the LanceDB adapter hides its async
/// behind a shared runtime so the harness stays uniform.
pub trait VectorStore: Sized {
    /// Stable display name (also the JSON key).
    const NAME: &'static str;

    /// Create an empty store of the given dimension, backed by files under `dir`.
    fn create(dim: usize, dir: &Path) -> Result<Self>;

    /// Bulk-insert `ids.len()` vectors. `vectors` is row-major, stride = dim.
    fn ingest(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()>;

    /// Exact top-k nearest neighbours by cosine similarity, highest score first.
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>>;

    /// Bytes the store occupies on disk after ingest.
    fn disk_bytes(&self) -> u64;
}

/// One point in the benchmark matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub n: usize,
    pub dim: usize,
    pub top_k: usize,
}

impl std::fmt::Display for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n={} dim={} top_k={}", self.n, self.dim, self.top_k)
    }
}

/// How hard to drive each measurement.
#[derive(Clone, Copy, Debug)]
pub struct RunCfg {
    pub warmup: usize,
    pub iters: usize,
}

/// Everything measured for one (engine × cell) run.
#[derive(Clone, Debug)]
pub struct EngineResult {
    pub engine: &'static str,
    pub cell: Cell,
    /// create + ingest wall time.
    pub build: Duration,
    pub ingest_per_s: f64,
    pub query: Timings,
    pub disk_bytes: u64,
    /// recall@k vs the harness's independent exact ground truth, averaged over queries.
    /// 1.0 = every query returned exactly the true neighbours.
    pub recall: f64,
}

/// Run one engine across one cell: create → ingest → warm up → time queries.
///
/// `truth` is the harness's own exact top-k per query (see [`exact_ground_truth`]) — the
/// recall reference, computed independently of every engine so none (not even nidus) is
/// trusted as the oracle.
pub fn run_engine<E: VectorStore>(
    cell: Cell,
    data: &Dataset,
    cfg: &RunCfg,
    truth: &[Vec<u64>],
) -> Result<EngineResult> {
    let tmp = tempfile::Builder::new()
        .prefix(&format!("nidus-bench-{}-", E::NAME))
        .tempdir()?;

    let t0 = Instant::now();
    let mut store = E::create(cell.dim, tmp.path())?;
    store.ingest(&data.ids, &data.vectors)?;
    let build = t0.elapsed();
    let ingest_per_s = cell.n as f64 / build.as_secs_f64();

    // Warm up caches / JITs / page-ins without recording.
    for q in data.queries.iter().take(cfg.warmup) {
        store.search(q, cell.top_k)?;
    }

    let mut samples = Vec::with_capacity(cfg.iters * data.queries.len());
    let mut topk_ids = Vec::with_capacity(data.queries.len());
    for iter in 0..cfg.iters {
        for q in &data.queries {
            let t = Instant::now();
            let hits = store.search(q, cell.top_k)?;
            samples.push(t.elapsed());
            if iter == 0 {
                topk_ids.push(hits.into_iter().map(|(id, _)| id).collect());
            }
        }
    }

    let disk_bytes = store.disk_bytes();
    // Keep `tmp` alive until after disk measurement, then let it clean up.
    drop(store);
    drop(tmp);

    Ok(EngineResult {
        engine: E::NAME,
        cell,
        build,
        ingest_per_s,
        query: Timings::summarize(samples),
        disk_bytes,
        recall: recall_at_k(&topk_ids, truth),
    })
}

/// The harness's **independent** exact top-k per query, by full brute-force cosine in f64
/// — the unbiased ground truth for recall. Computed straight from the raw dataset, never
/// from an engine's output, so it doesn't privilege whichever engine we compare against.
pub fn exact_ground_truth(data: &Dataset, top_k: usize) -> Vec<Vec<u64>> {
    let dim = data.dim;
    let n = data.n();
    let norms: Vec<f64> = (0..n)
        .map(|i| {
            data.vectors[i * dim..(i + 1) * dim]
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    data.queries
        .iter()
        .map(|q| {
            let qn: f64 = q
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt();
            let mut scored: Vec<(f64, u64)> = (0..n)
                .map(|i| {
                    let row = &data.vectors[i * dim..(i + 1) * dim];
                    let dot: f64 = row
                        .iter()
                        .zip(q)
                        .map(|(&a, &b)| (a as f64) * (b as f64))
                        .sum();
                    let denom = norms[i] * qn;
                    let cos = if denom > 1e-12 { dot / denom } else { 0.0 };
                    (cos, data.ids[i])
                })
                .collect();
            // Highest cosine first; total_cmp is a stable total order on ties.
            scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            scored.into_iter().take(top_k).map(|(_, id)| id).collect()
        })
        .collect()
}

/// recall@k = mean over queries of |returned ∩ truth| / |truth|.
pub fn recall_at_k(returned: &[Vec<u64>], truth: &[Vec<u64>]) -> f64 {
    let q = returned.len().min(truth.len());
    if q == 0 {
        return 1.0;
    }
    let mut acc = 0.0;
    for i in 0..q {
        let t: HashSet<u64> = truth[i].iter().copied().collect();
        let hit = returned[i].iter().filter(|id| t.contains(id)).count();
        acc += hit as f64 / truth[i].len().max(1) as f64;
    }
    acc / q as f64
}
