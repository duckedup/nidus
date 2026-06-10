//! nidus-bench-quant — int8 scalar-quantization recall & speed sweep.
//!
//! Holds quantization to the same bar as the parity harness: it builds one exact
//! (f32) nidus store and one quantized store per `rescore` factor over identical
//! data, then reports each variant's **recall@k** against an independent exact
//! ground truth plus its query latency and speedup vs the exact path. This is the
//! evidence for whether quantization wins at nidus's target scale and what the
//! default `rescore` should be.
//!
//! Run via `just bench-quant [key=value ...]`:
//!   n=100000           corpus size(s), comma-separated
//!   dim=384,768        embedding dimension(s)
//!   top_k=10           neighbours to retrieve
//!   rescore=1,2,4,8    overscan factors to sweep
//!   queries=100        distinct query vectors
//!   warmup=10          unrecorded warmup queries
//!   iters=5            measured passes over the query set
//!   seed=42            PRNG seed

use std::collections::BTreeMap;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Result, bail};
use nidus::{Config, Nidus, Quantization, Record, SearchOpts};
use nidus_bench::metrics::Timings;
use nidus_bench::{data, exact_ground_truth, recall_at_k};

const COLLECTION: &str = "bench";

struct Args {
    n: Vec<usize>,
    dim: Vec<usize>,
    top_k: usize,
    rescore: Vec<usize>,
    queries: usize,
    warmup: usize,
    iters: usize,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            n: vec![100_000],
            dim: vec![384, 768],
            top_k: 10,
            rescore: vec![1, 2, 4, 8],
            queries: 100,
            warmup: 10,
            iters: 5,
            seed: 42,
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
            println!("nidus-bench-quant — int8 quantization recall & speed sweep");
            println!("args: n=, dim=, top_k=, rescore=, queries=, warmup=, iters=, seed=");
            std::process::exit(0);
        }
        let Some((key, val)) = tok.split_once('=') else {
            bail!("expected key=value, got `{tok}` (try `help`)");
        };
        match key {
            "n" => a.n = parse_list(val)?,
            "dim" => a.dim = parse_list(val)?,
            "top_k" | "k" => a.top_k = val.parse()?,
            "rescore" => a.rescore = parse_list(val)?,
            "queries" => a.queries = val.parse()?,
            "warmup" => a.warmup = val.parse()?,
            "iters" => a.iters = val.parse()?,
            "seed" => a.seed = val.parse()?,
            other => bail!("unknown arg `{other}` (try `help`)"),
        }
    }
    Ok(a)
}

/// Build a file-backed nidus store (optionally quantized) and ingest the dataset.
/// Returns the `TempDir` guard alongside the store so the caller keeps both alive.
fn build(data: &data::Dataset, quant: Option<Quantization>) -> Result<(Nidus, tempfile::TempDir)> {
    let dir = tempfile::tempdir()?;
    let cfg = Config::new(dir.path().join("store"), data.dim)
        .quantization(quant)
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
    db.upsert(COLLECTION, &records)?;
    Ok((db, dir))
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
    println!("nidus-bench-quant — int8 quantization recall & speed sweep");
    println!(
        "seed={}  queries={}  warmup={}  iters={}  top_k={}",
        args.seed, args.queries, args.warmup, args.iters, args.top_k
    );

    for &n in &args.n {
        for &dim in &args.dim {
            let data = data::generate(args.seed, n, dim, args.queries);
            let truth = exact_ground_truth(&data, args.top_k);

            println!("\nn={n} dim={dim} top_k={}", args.top_k);

            // Exact (f32) baseline.
            let (exact_db, _exact_guard) = build(&data, None)?;
            let (exact_recall, exact_t) = measure(&exact_db, &data, args.top_k, &args, &truth)?;
            let base = exact_t.p50.as_secs_f64();
            println!(
                "  exact (f32)      recall={:.4}  p50={:>8.3}ms  1.00x",
                exact_recall,
                base * 1e3
            );

            // Quantized variants.
            for &rescore in &args.rescore {
                let (db, _guard) = build(&data, Some(Quantization::int8().rescore(rescore)))?;
                let (recall, t) = measure(&db, &data, args.top_k, &args, &truth)?;
                let p50 = t.p50.as_secs_f64();
                let speedup = if p50 > 0.0 { base / p50 } else { 0.0 };
                println!(
                    "  quant rescore={rescore:<2}  recall={:.4}  p50={:>8.3}ms  {:.2}x",
                    recall,
                    p50 * 1e3,
                    speedup
                );
            }
        }
    }
    Ok(())
}
