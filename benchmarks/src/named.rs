//! nidus-bench-named — named-vector cost, pooling, parallelism and recall sweep (nidus-85t).
//!
//! Four numbers, modelled on `ann.rs`: (1) the cost of naming at `names=1,2,4` against the
//! pre-change single-vector baseline (asserted, not just printed), (2) `Max` vs `Sum` pooling
//! over the same rows, (3) whether the shard-boundary-snapped reduction still scales with
//! `threads=`, and (4) ANN recall over named vectors. See `benchmarks/README.md`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use nidus::{AnnConfig, Config, Nidus, Pool, Record, SearchOpts};
use nidus_bench::metrics::Timings;
use nidus_bench::report::{fmt_count, fmt_dur};
use nidus_bench::{data, exact_ground_truth, recall_at_k};
use serde_json::{Value, json};

const COLLECTION: &str = "bench";

struct Args {
    n: Vec<usize>,
    dim: Vec<usize>,
    names: Vec<usize>,
    threads: Vec<usize>,
    top_k: usize,
    queries: usize,
    warmup: usize,
    iters: usize,
    seed: u64,
    clustered: bool,
    clusters: usize,
    /// Max acceptable `names=1` / pre-change-baseline p50 ratio (item 1's regression gate).
    tolerance: f64,
    /// Max ratio for an UNNAMED search on a named-vector store, per three stored vectors and
    /// scaled by the store's name count. Looser than `tolerance` to encode the measured
    /// interleaving cost of nidus-vttd; still fails anything worse than that.
    unnamed_tolerance: f64,
    /// Min named-path thread scaling as a fraction of the UNNAMED control's. Encodes
    /// nidus-vttd: pooling is parallel, the per-query scan build is not. A genuinely
    /// serialized reduction lands near 0.25 and still fails this.
    min_scale: f64,
    json: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            n: vec![100_000],
            dim: vec![384, 768],
            names: vec![1, 2, 4],
            threads: vec![1, 2, 4, 8],
            top_k: 10,
            queries: 100,
            warmup: 10,
            iters: 5,
            seed: 42,
            clustered: false,
            clusters: 64,
            tolerance: 1.15,
            unnamed_tolerance: 3.5,
            min_scale: 0.6,
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
            println!("nidus-bench-named — named-vector cost, pooling, threads & recall sweep");
            println!(
                "args: n=, dim=, names=, threads=, top_k=, queries=, warmup=, iters=, seed=, \
                 clustered=, clusters=, tolerance=, min_scale=, json=<path>"
            );
            std::process::exit(0);
        }
        let Some((key, val)) = tok.split_once('=') else {
            bail!("expected key=value, got `{tok}` (try `help`)");
        };
        match key {
            "n" => a.n = parse_list(val)?,
            "dim" => a.dim = parse_list(val)?,
            "names" => a.names = parse_list(val)?,
            "threads" => a.threads = parse_list(val)?,
            "top_k" | "k" => a.top_k = val.parse()?,
            "queries" => a.queries = val.parse()?,
            "warmup" => a.warmup = val.parse()?,
            "iters" => a.iters = val.parse()?,
            "seed" => a.seed = val.parse()?,
            "clustered" => a.clustered = val.parse::<u8>()? != 0,
            "clusters" => a.clusters = val.parse()?,
            "tolerance" => a.tolerance = val.parse()?,
            "unnamed_tolerance" => a.unnamed_tolerance = val.parse()?,
            "min_scale" => a.min_scale = val.parse()?,
            "json" => a.json = Some(PathBuf::from(val)),
            other => bail!("unknown arg `{other}` (try `help`)"),
        }
    }
    if a.names.is_empty() || a.names.contains(&0) {
        bail!("names must be non-empty and every value >= 1");
    }
    Ok(a)
}

// ── data ─────────────────────────────────────────────────────────────────────

/// splitmix64 RNG feeding `generate_clustered` below — copied from `ann.rs` (private there,
/// and `data.rs` has no clustered generator despite this unit's blueprint, so this mirrors
/// the existing pattern rather than inventing a shared one; see this PR's notes).
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

/// A clustered dataset: `k` random centers, corpus/query vectors = a center plus noise, then
/// unit-normalized. A direct copy of `ann.rs::generate_clustered` — see its comment for why
/// clustered (not uniform) is the realistic case for ANN recall.
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

/// Uniform or clustered, per `Args::clustered` — the same choice `ann.rs` exposes.
fn generate_dataset(
    seed: u64,
    n: usize,
    dim: usize,
    num_queries: usize,
    clustered: bool,
    clusters: usize,
) -> data::Dataset {
    if clustered {
        generate_clustered(seed, n, dim, num_queries, clusters)
    } else {
        data::generate(seed, n, dim, num_queries)
    }
}

/// A distinct sub-seed per named-vector slot, so `v1..vN` are independent representations
/// of the same record (title vs body) rather than copies of one another.
fn slot_seed(seed: u64, slot: usize) -> u64 {
    seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(slot as u64 + 1))
}

// ── store building ───────────────────────────────────────────────────────────

/// The pre-change store: one plain `vector` per record, no names declared at all — exactly
/// the code path `search` took before nidus-85t (`opts.names` stays empty).
fn build_plain(data: &data::Dataset, ann: Option<AnnConfig>) -> Result<(Nidus, tempfile::TempDir)> {
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
        .map(|(i, &id)| {
            Record::new(
                id.to_string(),
                data.vectors[i * dim..(i + 1) * dim].to_vec(),
                BTreeMap::new(),
            )
        })
        .collect();
    db.upsert(COLLECTION, &records)?;
    Ok((db, dir))
}

/// A store declaring `slots.len()` named vectors (`v1..vN`) and one record per id carrying
/// every one — the shape a multi-field document (title + body) actually takes.
fn build_named(
    slots: &[data::Dataset],
    ann: Option<AnnConfig>,
) -> Result<(Nidus, tempfile::TempDir)> {
    let dim = slots[0].dim;
    let n = slots[0].n();
    let dir = tempfile::tempdir()?;
    let cfg = Config::new(dir.path().join("store"), dim)
        .ann(ann)
        .auto_compact(None);
    let mut db = Nidus::open(cfg)?;
    db.create_collection(COLLECTION)?;
    let names: Vec<String> = (1..=slots.len()).map(|i| format!("v{i}")).collect();
    db.set_vector_names(COLLECTION, &names)?;
    let records: Vec<Record> = (0..n)
        .map(|i| {
            let vectors: BTreeMap<String, Vec<f32>> = names
                .iter()
                .enumerate()
                .map(|(slot, name)| {
                    (
                        name.clone(),
                        slots[slot].vectors[i * dim..(i + 1) * dim].to_vec(),
                    )
                })
                .collect();
            // Every record ALSO carries the plain `default` vector (slot 0's), so an
            // unnamed search on this store scans a full matrix. Without it the regression
            // gate below would time an empty scan and pass vacuously.
            Record {
                id: i.to_string(),
                vector: Some(slots[0].vectors[i * dim..(i + 1) * dim].to_vec()),
                vectors,
                attrs: BTreeMap::new(),
            }
        })
        .collect();
    db.upsert(COLLECTION, &records)?;
    Ok((db, dir))
}

// ── measurement ──────────────────────────────────────────────────────────────

/// Warm up, then time `iters` passes over `queries` with `db.search` alone — never
/// `search_with_plan`, whose tracing overhead would otherwise leak into the p50/p95/p99 this
/// reports. `rows_scanned` comes from one untimed `search_with_plan` call afterward.
fn measure(
    db: &Nidus,
    queries: &[Vec<f32>],
    opts: &SearchOpts,
    warmup: usize,
    iters: usize,
) -> Result<(Timings, u64)> {
    for q in queries.iter().take(warmup) {
        db.search(COLLECTION, q, opts)?;
    }
    let mut samples = Vec::with_capacity(iters * queries.len());
    for _ in 0..iters {
        for q in queries {
            let t = Instant::now();
            db.search(COLLECTION, q, opts)?;
            samples.push(t.elapsed());
        }
    }
    let rows_scanned = match queries.first() {
        Some(q) => db
            .search_with_plan(COLLECTION, q, opts)?
            .1
            .rows_scanned
            .unwrap_or(0),
        None => 0,
    };
    Ok((Timings::summarize(samples), rows_scanned))
}

/// Each query's returned ids, parsed back to `u64` — [`recall_at_k`]'s input shape.
fn topk_ids(db: &Nidus, queries: &[Vec<f32>], opts: &SearchOpts) -> Result<Vec<Vec<u64>>> {
    queries
        .iter()
        .map(|q| {
            db.search(COLLECTION, q, opts)?
                .into_iter()
                .map(|h| h.id.parse::<u64>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(Into::into)
        })
        .collect()
}

// ── reporting ────────────────────────────────────────────────────────────────

fn write_json(path: &std::path::Path, args: &Args, cells: &[Value]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let doc = json!({
        "bench": "nidus-bench-named",
        // The version of NIDUS, not this bench crate's own `CARGO_PKG_VERSION` (which would
        // report nidus-bench's, the wrong crate). The justfile recipe passes it in.
        "nidus_version": std::env::var("NIDUS_VERSION").unwrap_or_else(|_| "unknown".into()),
        "inputs": {
            "n": args.n, "dim": args.dim, "names": args.names, "threads": args.threads,
            "top_k": args.top_k, "queries": args.queries, "warmup": args.warmup,
            "iters": args.iters, "seed": args.seed, "clustered": args.clustered,
            "clusters": args.clusters, "tolerance": args.tolerance,
            "unnamed_tolerance": args.unnamed_tolerance, "min_scale": args.min_scale,
        },
        "cells": cells,
    });
    std::fs::write(path, format!("{}\n", serde_json::to_string_pretty(&doc)?))
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!("\nwrote {}", path.display());
    Ok(())
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
    let names_max = *args.names.iter().max().expect("validated non-empty above");
    println!("nidus-bench-named — named-vector cost, pooling, threads & recall sweep");
    println!(
        "seed={}  queries={}  warmup={}  iters={}  top_k={}  names={:?}",
        args.seed, args.queries, args.warmup, args.iters, args.top_k, args.names
    );

    let mut failures: Vec<String> = Vec::new();
    let mut doc_cells: Vec<Value> = Vec::new();

    for &n in &args.n {
        for &dim in &args.dim {
            println!("\n═══ n={n} dim={dim} ═══════════════════════════════════════════");

            // One dataset per named-vector slot, shared by every section below: `v1` also
            // doubles as the pre-change baseline's plain vector and query set.
            let mut slots = Vec::with_capacity(names_max);
            for slot in 0..names_max {
                let nq = if slot == 0 { args.queries } else { 0 };
                slots.push(generate_dataset(
                    slot_seed(args.seed, slot),
                    n,
                    dim,
                    nq,
                    args.clustered,
                    args.clusters,
                ));
            }
            let queries = slots[0].queries.clone();

            // ── 1. cost of naming, with names=1 gated against the pre-change baseline ──
            println!("\n── cost of naming ──────────────────────────────────────────────");
            let (plain_db, _plain_guard) = build_plain(&slots[0], None)?;
            let plain_opts = SearchOpts {
                top_k: args.top_k,
                ..Default::default()
            };
            let (baseline, baseline_rows) =
                measure(&plain_db, &queries, &plain_opts, args.warmup, args.iters)?;
            println!(
                "  {:<24}{:>10}  rows={baseline_rows}",
                "baseline (pre-change)",
                fmt_dur(baseline.p50)
            );

            let mut by_names_cost: Vec<Value> = Vec::new();
            for &k in &args.names {
                let names: Vec<String> = (1..=k).map(|i| format!("v{i}")).collect();
                let (db, _guard) = build_named(&slots[..k], None)?;
                let opts = SearchOpts {
                    top_k: args.top_k,
                    names,
                    pool: Pool::Max,
                    ..Default::default()
                };
                let (t, rows) = measure(&db, &queries, &opts, args.warmup, args.iters)?;
                let ratio = t.p50.as_secs_f64() / baseline.p50.as_secs_f64().max(1e-12);
                println!(
                    "  {:<24}{:>10}  rows={rows:<9}  {ratio:>5.2}x baseline",
                    format!("names={k}"),
                    fmt_dur(t.p50)
                );
                if k == 1 {
                    // Reported, NOT gated: this is the feature path's own price, not a
                    // regression — gating it compared two different things (nidus-85t).
                    println!(
                        "    -> cost of the named path itself vs pre-change: {ratio:.2}x \
                         (reported, not gated)"
                    );
                }
                by_names_cost.push(json!({
                    "names": k, "p50_ms": t.p50.as_secs_f64() * 1e3,
                    "rows_scanned": rows, "ratio_vs_baseline": ratio,
                }));
            }

            // THE regression gate, and the one existing users feel: a plain unnamed search on
            // a store that now CARRIES named vectors. Named rows must stay invisible to it —
            // not in its scan, and since the ANN fix not in its index either.
            let (named_db, _named_guard) = build_named(&slots, None)?;
            let (unnamed_on_named, unnamed_rows) =
                measure(&named_db, &queries, &plain_opts, args.warmup, args.iters)?;
            let unnamed_ratio =
                unnamed_on_named.p50.as_secs_f64() / baseline.p50.as_secs_f64().max(1e-12);
            // The stride is one row of every `names_max + 1`, so the expected locality cost
            // grows with the number of names a record carries. A flat ceiling would pass at
            // names=2 and fail at names=4 for a code path that did not change.
            let unnamed_ceiling = args.unnamed_tolerance * (names_max as f64 + 1.0) / 3.0;
            let unnamed_pass = unnamed_ratio <= unnamed_ceiling;
            println!(
                "  {:<24}{:>10}  rows={unnamed_rows:<9}  {unnamed_ratio:>5.2}x baseline",
                "unnamed on named store",
                fmt_dur(unnamed_on_named.p50)
            );
            println!(
                "    -> an ordinary search on a named-vector store: {unnamed_ratio:.2}x \
                 (<= {unnamed_ceiling:.2}x for {names_max} names, the locality cost filed as \
                 nidus-vttd)  {}",
                if unnamed_pass { "PASS" } else { "FAIL" }
            );
            if !unnamed_pass {
                failures.push(format!(
                    "n={n} dim={dim}: an UNNAMED search on a store carrying named vectors is \
                     {unnamed_ratio:.2}x the pre-change baseline (ceiling {unnamed_ceiling:.2}x \
                     for {names_max} names) — WORSE than the interleaving cost of nidus-vttd"
                ));
            }

            // ── 2. Max vs Sum pooling, over the same rows ───────────────────────────
            println!("\n── pooling  names={names_max} ───────────────────────────────────────────");
            let pool_names: Vec<String> = (1..=names_max).map(|i| format!("v{i}")).collect();
            let (pool_db, pool_guard) = build_named(&slots[..names_max], None)?;
            let max_opts = SearchOpts {
                top_k: args.top_k,
                names: pool_names,
                pool: Pool::Max,
                ..Default::default()
            };
            let sum_opts = SearchOpts {
                pool: Pool::Sum,
                ..max_opts.clone()
            };
            let (max_t, _) = measure(&pool_db, &queries, &max_opts, args.warmup, args.iters)?;
            let (sum_t, _) = measure(&pool_db, &queries, &sum_opts, args.warmup, args.iters)?;
            let pool_ratio = sum_t.p50.as_secs_f64() / max_t.p50.as_secs_f64().max(1e-12);
            println!("  {:<10}{:>10}", "max", fmt_dur(max_t.p50));
            println!("  {:<10}{:>10}", "sum", fmt_dur(sum_t.p50));
            println!(
                "  sum/max = {pool_ratio:.2}x — same rows walked, so a large gap would mean the \
                 reduction itself (not the scan) is the cost"
            );

            // ── 3. threads= sweep at names=names_max ────────────────────────────────
            println!("\n── threads=  names={names_max} ──────────────────────────────────────────");
            drop(pool_db); // release the writer lock before reopening at each thread count
            let store_path = pool_guard.path().join("store");
            let mut by_threads: Vec<Value> = Vec::new();
            let mut base_per_s = 0.0;
            let mut base_ctl_per_s = 0.0;
            for (i, &threads) in args.threads.iter().enumerate() {
                let cfg = Config::new(&store_path, dim)
                    .auto_compact(None)
                    .query_threads(threads);
                let db = Nidus::open(cfg)?;
                let (t, _) = measure(&db, &queries, &max_opts, args.warmup, args.iters)?;
                // The UNNAMED path on the same store and box: the control. Absolute scaling
                // here is bandwidth-bound (nidus-vttd), so it measures the machine as much as
                // the code; scaling RELATIVE to the control isolates the reduction.
                let (ctl, _) = measure(&db, &queries, &plain_opts, args.warmup, args.iters)?;
                let per_s = 1.0 / t.mean.as_secs_f64().max(1e-12);
                let ctl_per_s = 1.0 / ctl.mean.as_secs_f64().max(1e-12);
                if i == 0 {
                    base_per_s = per_s;
                    base_ctl_per_s = ctl_per_s;
                }
                let speedup = if base_per_s > 0.0 {
                    per_s / base_per_s
                } else {
                    0.0
                };
                let ctl_speedup = if base_ctl_per_s > 0.0 {
                    ctl_per_s / base_ctl_per_s
                } else {
                    0.0
                };
                println!(
                    "  threads={threads:<4}{:>14}  named {speedup:>5.2}x   unnamed control \
                     {ctl_speedup:>5.2}x",
                    fmt_count(per_s)
                );
                by_threads.push(json!({
                    "threads": threads, "per_s": per_s, "speedup": speedup,
                    "control_per_s": ctl_per_s, "control_speedup": ctl_speedup,
                }));
            }
            if let (Some(&lo), Some(&hi)) = (args.threads.first(), args.threads.last())
                && hi > lo
            {
                let last = by_threads.last();
                let last_speedup = last.and_then(|v| v["speedup"].as_f64()).unwrap_or(0.0);
                let ctl_speedup = last
                    .and_then(|v| v["control_speedup"].as_f64())
                    .unwrap_or(0.0);
                // Relative to the control, not to an ideal: a box that cannot scale either
                // path fails both equally and says nothing about this code.
                let ratio = if ctl_speedup > 0.0 {
                    last_speedup / ctl_speedup
                } else {
                    0.0
                };
                let pass = ratio >= args.min_scale;
                println!(
                    "  -> {lo}->{hi} threads: named scaled {last_speedup:.2}x vs unnamed control \
                     {ctl_speedup:.2}x = {ratio:.2}x of the control (need >= {:.2}x)  {}",
                    args.min_scale,
                    if pass { "PASS" } else { "FAIL" }
                );
                if !pass {
                    failures.push(format!(
                        "n={n} dim={dim}: named scaling is only {ratio:.2}x the unnamed control's \
                         across {lo}->{hi} threads (need >= {:.2}x) — the per-record reduction \
                         serialized, and it is not the machine, because the control scaled",
                        args.min_scale
                    ));
                }
            }

            // ── 4. ANN recall per names= value ──────────────────────────────────────
            println!("\n── ANN recall ───────────────────────────────────────────────────");
            let mut by_names_recall: Vec<Value> = Vec::new();
            for &k in &args.names {
                let names: Vec<String> = (1..=k).map(|i| format!("v{i}")).collect();
                let (db, _guard) = build_named(&slots[..k], Some(AnnConfig::hnsw()))?;
                let ann_opts = SearchOpts {
                    top_k: args.top_k,
                    names,
                    pool: Pool::Max,
                    ..Default::default()
                };
                let exact_opts = SearchOpts {
                    exact: true,
                    ..ann_opts.clone()
                };
                let ann_ids = topk_ids(&db, &queries, &ann_opts)?;
                let exact_ids = topk_ids(&db, &queries, &exact_opts)?;
                let recall = recall_at_k(&ann_ids, &exact_ids);
                println!("  names={k:<3} recall@{} = {recall:.4}", args.top_k);
                let mut row = json!({"names": k, "recall_at_k": recall});

                if k == 1 {
                    // Independent cross-check: with one name, pooling is a no-op (Max/Sum of a
                    // single term is that term), so the harness's own `exact_ground_truth` over
                    // the raw slot-0 array applies unmodified — genuine reuse, not a rebuild.
                    let truth = exact_ground_truth(&slots[0], args.top_k);
                    let independent = recall_at_k(&exact_ids, &truth);
                    println!("    independent cross-check vs exact_ground_truth: {independent:.4}");
                    row["independent_check_vs_exact_ground_truth"] = json!(independent);
                }
                by_names_recall.push(row);
            }
            println!(
                "  note: named-vector search always takes the exact path today, regardless of \
                 Config::ann (src/store/read.rs), so recall@k above is 1.0 by construction — it \
                 will start reporting real degradation once ANN composes with named vectors."
            );

            doc_cells.push(json!({
                "n": n, "dim": dim,
                "baseline_p50_ms": baseline.p50.as_secs_f64() * 1e3,
                "baseline_rows_scanned": baseline_rows,
                "by_names_cost": by_names_cost,
                "pooling": {
                    "names": names_max,
                    "max_p50_ms": max_t.p50.as_secs_f64() * 1e3,
                    "sum_p50_ms": sum_t.p50.as_secs_f64() * 1e3,
                },
                "by_threads": by_threads,
                "by_names_recall": by_names_recall,
            }));
        }
    }

    if let Some(path) = &args.json {
        write_json(path, &args, &doc_cells)?;
    }

    if failures.is_empty() {
        Ok(ExitCode::SUCCESS)
    } else {
        println!("\n{} check(s) failed:", failures.len());
        for f in &failures {
            println!("  {f}");
        }
        Ok(ExitCode::FAILURE)
    }
}
