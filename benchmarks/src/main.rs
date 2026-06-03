//! nidus-bench — cross-engine exact-KNN performance-parity benchmark.
//!
//! Run via `just bench [engines] [key=value ...]`. All engines are pinned to exact
//! brute-force cosine KNN, so this measures parity (and tracks regressions), not a race.
//!
//! Args (all `key=value`, repeatable lists are comma-separated):
//!   n=10000,100000     corpus sizes
//!   dim=384,768        embedding dimensions
//!   top_k=10           neighbours to retrieve
//!   queries=100        distinct query vectors per cell
//!   warmup=10          unrecorded warmup queries
//!   iters=5            measured passes over the query set
//!   seed=42            PRNG seed (determinism)
//!   threshold=1.25     max nidus_p50 / best_p50 before a cell is a FAIL
//!   json=<path>        override the JSON artifact path

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use nidus_bench::engines::nidus::NidusEngine;
use nidus_bench::{Cell, EngineResult, RunCfg, data, report, run_engine};

struct Args {
    n: Vec<usize>,
    dim: Vec<usize>,
    top_k: Vec<usize>,
    queries: usize,
    cfg: RunCfg,
    seed: u64,
    threshold: f64,
    json: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            n: vec![10_000, 100_000],
            dim: vec![384, 768],
            top_k: vec![10],
            queries: 100,
            cfg: RunCfg {
                warmup: 10,
                iters: 5,
            },
            seed: 42,
            threshold: 1.25,
            json: None,
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
            println!("{}", include_str!("usage.txt"));
            std::process::exit(0);
        }
        let Some((key, val)) = tok.split_once('=') else {
            bail!("expected key=value, got `{tok}` (try `help`)");
        };
        match key {
            "n" => a.n = parse_list(val)?,
            "dim" => a.dim = parse_list(val)?,
            "top_k" | "k" => a.top_k = parse_list(val)?,
            "queries" => a.queries = val.parse()?,
            "warmup" => a.cfg.warmup = val.parse()?,
            "iters" => a.cfg.iters = val.parse()?,
            "seed" => a.seed = val.parse()?,
            "threshold" => a.threshold = val.parse()?,
            "json" => a.json = Some(PathBuf::from(val)),
            other => bail!("unknown arg `{other}` (try `help`)"),
        }
    }
    Ok(a)
}

fn compiled_engines() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut v = vec!["nidus"];
    #[cfg(feature = "duckdb")]
    v.push("duckdb");
    #[cfg(feature = "lancedb")]
    v.push("lancedb");
    v
}

/// Run every compiled-in engine over one cell, scoring recall against one shared,
/// engine-independent exact ground truth.
fn run_cell(cell: Cell, args: &Args) -> Result<Vec<EngineResult>> {
    let data = data::generate(args.seed, cell.n, cell.dim, args.queries);
    let truth = nidus_bench::exact_ground_truth(&data, cell.top_k);
    #[allow(unused_mut)]
    let mut results = vec![run_engine::<NidusEngine>(cell, &data, &args.cfg, &truth)?];
    #[cfg(feature = "duckdb")]
    results.push(run_engine::<nidus_bench::engines::duckdb::DuckdbEngine>(
        cell, &data, &args.cfg, &truth,
    )?);
    #[cfg(feature = "lancedb")]
    results.push(run_engine::<nidus_bench::engines::lancedb::LancedbEngine>(
        cell, &data, &args.cfg, &truth,
    )?);
    Ok(results)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode> {
    let args = parse_args()?;
    let engines = compiled_engines();

    println!("nidus-bench — exact brute-force cosine KNN parity");
    println!(
        "engines: {}   seed={}  queries={}  warmup={}  iters={}",
        engines.join(", "),
        args.seed,
        args.queries,
        args.cfg.warmup,
        args.cfg.iters
    );
    if engines.len() == 1 {
        println!(
            "(only nidus compiled in — run `just bench all` to compare against DuckDB + LanceDB)"
        );
    }

    let mut all: Vec<(Cell, Vec<EngineResult>)> = Vec::new();
    let mut failures: Vec<(Cell, f64)> = Vec::new();

    for &n in &args.n {
        for &dim in &args.dim {
            for &top_k in &args.top_k {
                let cell = Cell { n, dim, top_k };
                let results = run_cell(cell, &args)?;
                report::print_cell(cell, &results);

                if let Some(pass) = report::cell_passes(&results, args.threshold) {
                    let nidus = results.iter().find(|r| r.engine == "nidus").unwrap();
                    let best = results.iter().map(|r| r.query.p50).min().unwrap();
                    let ratio = nidus.query.p50.as_secs_f64() / best.as_secs_f64();
                    if pass {
                        println!(
                            "  -> nidus p50 {ratio:.2}x best (<= {:.2}x)  PASS",
                            args.threshold
                        );
                    } else {
                        println!(
                            "  -> nidus p50 {ratio:.2}x best (> {:.2}x)  FAIL",
                            args.threshold
                        );
                        failures.push((cell, ratio));
                    }
                }
                all.push((cell, results));
            }
        }
    }

    // JSON artifact (stamped so successive runs can be diffed).
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let json_path = args
        .json
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("target/bench-results/{stamp}.json")));
    if let Some(parent) = json_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let doc = report::to_json(args.seed, &args.cfg, &all, stamp);
    std::fs::write(&json_path, serde_json::to_vec_pretty(&doc)?)?;
    println!("\nwrote {}", json_path.display());

    if failures.is_empty() {
        Ok(ExitCode::SUCCESS)
    } else {
        println!(
            "\n{} cell(s) exceeded the {:.2}x parity threshold:",
            failures.len(),
            args.threshold
        );
        for (cell, ratio) in &failures {
            println!("  {cell}: {ratio:.2}x");
        }
        Ok(ExitCode::FAILURE)
    }
}
