//! nidus-bench — cross-engine exact-KNN performance-parity harness.
//!
//! The goal is **parity, not winning**: confirm nidus's exact brute-force cosine KNN
//! stays in line with DuckDB and LanceDB, and catch regressions over time. Every engine
//! is pinned to *exact* search (no ANN index), so the top-k results must agree — the
//! harness asserts that as a fairness check.

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
    /// Top-k id list per query (first measured iteration), for the parity cross-check.
    pub topk_ids: Vec<Vec<u64>>,
}

/// Run one engine across one cell: create → ingest → warm up → time queries.
pub fn run_engine<E: VectorStore>(
    cell: Cell,
    data: &Dataset,
    cfg: &RunCfg,
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
        topk_ids,
    })
}

/// Fraction of top-k ids shared between two result sets, averaged over queries.
/// 1.0 means every query agreed exactly. Exact engines may differ only at the k-th
/// boundary on near-ties, so values just below 1.0 are expected and acceptable.
pub fn topk_agreement(a: &EngineResult, b: &EngineResult) -> f64 {
    let q = a.topk_ids.len().min(b.topk_ids.len());
    if q == 0 {
        return 1.0;
    }
    let mut acc = 0.0;
    for i in 0..q {
        let sa: std::collections::HashSet<u64> = a.topk_ids[i].iter().copied().collect();
        let overlap = b.topk_ids[i].iter().filter(|id| sa.contains(id)).count();
        let denom = a.topk_ids[i].len().max(1);
        acc += overlap as f64 / denom as f64;
    }
    acc / q as f64
}
