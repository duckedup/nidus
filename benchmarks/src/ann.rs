//! nidus-bench-ann — approximate-nearest-neighbour recall & speed sweep.
//!
//! Mirrors `nidus-bench-quant`: builds one exact (f32) nidus store and one store
//! per ANN variant over identical data, then reports each variant's **recall@k**
//! against an independent exact ground truth, plus build time, query latency, and
//! speedup vs the exact brute-force path. This is the evidence for where `Config::ann`
//! stands at nidus's target scale and how to set `ef_search` / `n_probe`.
//!
//! Run via `just bench-ann [key=value ...]`:
//!   n=100000           corpus size(s), comma-separated
//!   dim=384,768        embedding dimension(s)
//!   top_k=10           neighbours to retrieve
//!   ef_search=64,128   HNSW query beam widths to sweep
//!   n_probe=8,32       IVF probe counts to sweep
//!   queries=100        distinct query vectors
//!   warmup=10          unrecorded warmup queries
//!   iters=5            measured passes over the query set
//!   seed=42            PRNG seed

use std::collections::BTreeMap;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use nidus::{AnnConfig, Config, Nidus, Record, SearchOpts};
use nidus_bench::metrics::Timings;
use nidus_bench::{data, exact_ground_truth, recall_at_k};

const COLLECTION: &str = "bench";

/// splitmix64, matching the bench data generator — used to synthesize *clustered*
/// data (the realistic case for ANN; uniform-random high-dim data is a near-worst
/// case where all cosines concentrate near zero and no index can navigate well).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

/// A clustered dataset: `k` random centers, each corpus vector = a center plus small
/// Gaussian-ish noise, then unit-normalized. Queries sit near random centers too.
/// This mimics the low-intrinsic-dimensional manifold real embeddings live on.
fn generate_clustered(
    seed: u64,
    n: usize,
    dim: usize,
    num_queries: usize,
    k: usize,
) -> data::Dataset {
    let mut rng = Rng::new(seed ^ 0xC1075735);
    let mut centers = vec![0.0f32; k * dim];
    for c in centers.iter_mut() {
        *c = rng.unit();
    }
    let normalize = |v: &mut [f32]| {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    };
    for c in 0..k {
        normalize(&mut centers[c * dim..(c + 1) * dim]);
    }
    // ~6 noise samples averaged ≈ a mild Gaussian; scale keeps points near the center.
    let noisy = |center: &[f32], rng: &mut Rng| -> Vec<f32> {
        let mut v: Vec<f32> = center.to_vec();
        for x in v.iter_mut() {
            let noise = (0..6).map(|_| rng.unit()).sum::<f32>() / 6.0;
            *x += 0.25 * noise;
        }
        normalize(&mut v);
        v
    };
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = (i % k) * dim;
        vectors.extend_from_slice(&noisy(&centers[c..c + dim], &mut rng));
    }
    let mut qr = Rng::new(seed ^ 0x5151_1A2B);
    let queries = (0..num_queries)
        .map(|i| {
            let c = (i % k) * dim;
            noisy(&centers[c..c + dim], &mut qr)
        })
        .collect();
    data::Dataset {
        dim,
        ids: (0..n as u64).collect(),
        vectors,
        queries,
    }
}

struct Args {
    n: Vec<usize>,
    dim: Vec<usize>,
    top_k: usize,
    ef_search: Vec<usize>,
    n_probe: Vec<usize>,
    queries: usize,
    warmup: usize,
    iters: usize,
    seed: u64,
    clustered: bool,
    clusters: usize,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            n: vec![100_000],
            dim: vec![384, 768],
            top_k: 10,
            ef_search: vec![64, 128, 256],
            n_probe: vec![8, 32, 64],
            queries: 100,
            warmup: 10,
            iters: 5,
            seed: 42,
            clustered: false,
            clusters: 64,
        }
    }
}

fn parse_list(v: &str) -> Result<Vec<usize>> {
    v.split(',')
        .map(|s| Ok(s.trim().parse::<usize>()?))
        .collect()
}

fn parse_args() -> Result<Args> {
    let mut a = Args::default();
    for tok in std::env::args().skip(1) {
        if tok == "help" || tok == "--help" || tok == "-h" {
            println!("nidus-bench-ann — ANN (HNSW + IVF) recall & speed sweep");
            println!(
                "args: n=, dim=, top_k=, ef_search=, n_probe=, queries=, warmup=, iters=, seed="
            );
            std::process::exit(0);
        }
        let Some((key, val)) = tok.split_once('=') else {
            bail!("expected key=value, got `{tok}` (try `help`)");
        };
        match key {
            "n" => a.n = parse_list(val)?,
            "dim" => a.dim = parse_list(val)?,
            "top_k" | "k" => a.top_k = val.parse()?,
            "ef_search" | "ef" => a.ef_search = parse_list(val)?,
            "n_probe" | "probe" => a.n_probe = parse_list(val)?,
            "queries" => a.queries = val.parse()?,
            "warmup" => a.warmup = val.parse()?,
            "iters" => a.iters = val.parse()?,
            "seed" => a.seed = val.parse()?,
            "clustered" => a.clustered = val.parse::<u8>()? != 0,
            "clusters" => a.clusters = val.parse()?,
            other => bail!("unknown arg `{other}` (try `help`)"),
        }
    }
    Ok(a)
}

/// Build a file-backed nidus store (optionally with an ANN index) and ingest the
/// dataset. Returns the store + its `TempDir` guard and the ingest (build) duration.
fn build(
    data: &data::Dataset,
    ann: Option<AnnConfig>,
) -> Result<(Nidus, tempfile::TempDir, Duration)> {
    let dir = tempfile::tempdir()?;
    let cfg = Config::new(dir.path().join("store"), data.dim)
        .ann(ann)
        .auto_compact(None);
    let mut db = Nidus::open(cfg)?;
    db.create_collection(COLLECTION)?;
    let dim = data.dim;
    let records: Vec<Record> = data
        .ids
        .iter()
        .enumerate()
        .map(|(i, &id)| Record {
            id: id.to_string(),
            vector: data.vectors[i * dim..(i + 1) * dim].to_vec(),
            attrs: BTreeMap::new(),
        })
        .collect();
    let t = Instant::now();
    db.upsert(COLLECTION, &records)?;
    let build_time = t.elapsed();
    Ok((db, dir, build_time))
}

/// Measure recall@k and per-query latency for one store over the query set.
fn measure(
    db: &Nidus,
    data: &data::Dataset,
    top_k: usize,
    args: &Args,
    truth: &[Vec<u64>],
) -> Result<(f64, Timings)> {
    let opts = SearchOpts {
        top_k,
        ..Default::default()
    };
    for q in data.queries.iter().take(args.warmup) {
        db.search(COLLECTION, q, &opts)?;
    }
    let mut samples = Vec::with_capacity(args.iters * data.queries.len());
    let mut returned: Vec<Vec<u64>> = Vec::with_capacity(data.queries.len());
    for iter in 0..args.iters {
        for q in &data.queries {
            let t = Instant::now();
            let hits = db.search(COLLECTION, q, &opts)?;
            samples.push(t.elapsed());
            if iter == 0 {
                returned.push(
                    hits.into_iter()
                        .map(|h| h.id.parse::<u64>())
                        .collect::<std::result::Result<Vec<_>, _>>()?,
                );
            }
        }
    }
    Ok((recall_at_k(&returned, truth), Timings::summarize(samples)))
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = parse_args()?;
    println!("nidus-bench-ann — ANN (HNSW + IVF) recall & speed sweep");
    println!(
        "seed={}  queries={}  warmup={}  iters={}  top_k={}",
        args.seed, args.queries, args.warmup, args.iters, args.top_k
    );

    if args.clustered {
        println!("dataset=clustered ({} clusters)", args.clusters);
    } else {
        println!("dataset=uniform-random (near-worst case for ANN recall)");
    }

    for &n in &args.n {
        for &dim in &args.dim {
            let data = if args.clustered {
                generate_clustered(args.seed, n, dim, args.queries, args.clusters)
            } else {
                data::generate(args.seed, n, dim, args.queries)
            };
            let truth = exact_ground_truth(&data, args.top_k);

            println!("\nn={n} dim={dim} top_k={}", args.top_k);
            println!(
                "  {:<22}{:>9}  {:>10}  {:>10}  {:>7}",
                "variant", "recall", "build", "p50", "speedup"
            );

            // Exact (f32) baseline.
            let (exact_db, _exact_guard, exact_build) = build(&data, None)?;
            let (exact_recall, exact_t) = measure(&exact_db, &data, args.top_k, &args, &truth)?;
            let base = exact_t.p50.as_secs_f64();
            println!(
                "  {:<22}{:>9.4}  {:>9.2}s  {:>8.3}ms  {:>6.2}x",
                "exact (f32)",
                exact_recall,
                exact_build.as_secs_f64(),
                base * 1e3,
                1.0
            );

            // HNSW variants (sweep ef_search).
            for &ef in &args.ef_search {
                let (db, _guard, bt) = build(&data, Some(AnnConfig::hnsw().ef_search(ef)))?;
                let (recall, t) = measure(&db, &data, args.top_k, &args, &truth)?;
                let p50 = t.p50.as_secs_f64();
                let speedup = if p50 > 0.0 { base / p50 } else { 0.0 };
                println!(
                    "  {:<22}{:>9.4}  {:>9.2}s  {:>8.3}ms  {:>6.2}x",
                    format!("hnsw ef={ef}"),
                    recall,
                    bt.as_secs_f64(),
                    p50 * 1e3,
                    speedup
                );
            }

            // IVF variants (sweep n_probe).
            for &probe in &args.n_probe {
                let (db, _guard, bt) = build(&data, Some(AnnConfig::ivf().n_probe(probe)))?;
                let (recall, t) = measure(&db, &data, args.top_k, &args, &truth)?;
                let p50 = t.p50.as_secs_f64();
                let speedup = if p50 > 0.0 { base / p50 } else { 0.0 };
                println!(
                    "  {:<22}{:>9.4}  {:>9.2}s  {:>8.3}ms  {:>6.2}x",
                    format!("ivf probe={probe}"),
                    recall,
                    bt.as_secs_f64(),
                    p50 * 1e3,
                    speedup
                );
            }
        }
    }
    Ok(())
}
