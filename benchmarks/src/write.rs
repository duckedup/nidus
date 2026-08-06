//! nidus-bench-write — where a single writer's throughput actually goes (nidus-xb9).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use nidus::{Config, Fsync, Nidus, Record};
use nidus_bench::data::{self, Dataset};
use nidus_bench::report::fmt_count;
use serde_json::{Value, json};

const COLLECTION: &str = "bench";

// ── args ─────────────────────────────────────────────────────────────────────

struct Args {
    n: usize,
    dim: Vec<usize>,
    batch: Vec<usize>,
    clients: Vec<usize>,
    /// Ceiling on upsert calls per pass — see [`Plan::new`] for why one exists.
    max_requests: usize,
    seed: u64,
    /// Where to write the machine-readable run, if anywhere.
    json: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            n: 50_000,
            dim: vec![384, 768],
            batch: vec![1, 10, 100, 1_000],
            clients: vec![1, 2, 4, 8],
            max_requests: 500,
            seed: 42,
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
            println!("nidus-bench-write — single-writer ingest decomposition (nidus-xb9)");
            println!("args: n=, dim=, batch=, clients=, max_requests=, seed=, json=<path>");
            std::process::exit(0);
        }
        let Some((key, val)) = tok.split_once('=') else {
            bail!("expected key=value, got `{tok}` (try `help`)");
        };
        match key {
            "n" => a.n = val.parse()?,
            "dim" => a.dim = parse_list(val)?,
            "batch" => a.batch = parse_list(val)?,
            "clients" => a.clients = parse_list(val)?,
            "max_requests" => a.max_requests = val.parse()?,
            "seed" => a.seed = val.parse()?,
            "json" => a.json = Some(PathBuf::from(val)),
            _ => bail!("unknown arg `{key}` (try `help`)"),
        }
    }
    if a.n == 0 || a.dim.is_empty() || a.batch.is_empty() {
        bail!("n, dim and batch must be non-empty");
    }
    Ok(a)
}

// ── the wire body ────────────────────────────────────────────────────────────

/// Mirror of the server's `UpsertRequest` (`src/server/dto.rs`).
#[derive(serde::Serialize, serde::Deserialize)]
struct WireBody {
    records: Vec<Record>,
}

/// The records for rows `start..start + len`, built outside every timed section: this is
/// the caller's own data-marshalling cost, not nidus's.
fn batch_records(dataset: &Dataset, start: usize, len: usize) -> Vec<Record> {
    let dim = dataset.dim;
    (start..start + len)
        .map(|row| {
            Record::new(
                row.to_string(),
                dataset.vectors[row * dim..(row + 1) * dim].to_vec(),
                BTreeMap::new(),
            )
        })
        .collect()
}

/// The exact batches a pass will ingest, and how many vectors that adds up to.
struct Plan {
    /// `(start_row, len)` per upsert call.
    batches: Vec<(usize, usize)>,
    /// Rows actually ingested — the denominator for every per-vector figure.
    vectors: usize,
}

impl Plan {
    /// Cover `n` rows in `batch`-sized calls, stopping after `max_requests` of them.
    fn new(n: usize, batch: usize, max_requests: usize) -> Plan {
        let batches: Vec<(usize, usize)> = (0..n)
            .step_by(batch)
            .map(|start| (start, batch.min(n - start)))
            .take(max_requests)
            .collect();
        let vectors = batches.iter().map(|&(_, len)| len).sum();
        Plan { batches, vectors }
    }
}

// ── stages ───────────────────────────────────────────────────────────────────

/// Time JSON encode and decode of every batch, over the exact bytes the wire carries.
fn codec_pass(dataset: &Dataset, plan: &Plan) -> Result<(Duration, Duration)> {
    let (mut encode, mut decode) = (Duration::ZERO, Duration::ZERO);
    for &(start, len) in &plan.batches {
        let body = WireBody {
            records: batch_records(dataset, start, len),
        };

        let t = Instant::now();
        let bytes = serde_json::to_vec(&body)?;
        encode += t.elapsed();

        let t = Instant::now();
        let back: WireBody = serde_json::from_slice(&bytes)?;
        decode += t.elapsed();
        std::hint::black_box(&back);
    }
    Ok((encode, decode))
}

/// Time `Nidus::upsert` over every batch under one fsync policy.
fn store_pass(dataset: &Dataset, plan: &Plan, fsync: Fsync) -> Result<Duration> {
    let tmp = tempfile::Builder::new()
        .prefix("nidus-bench-write-")
        .tempdir()?;
    let mut db = Nidus::open(Config::new(tmp.path(), dataset.dim).fsync(fsync))?;

    let mut total = Duration::ZERO;
    for &(start, len) in &plan.batches {
        let records = batch_records(dataset, start, len);
        let t = Instant::now();
        db.upsert(COLLECTION, &records)?;
        total += t.elapsed();
    }
    let t = Instant::now();
    db.flush()?;
    total += t.elapsed();
    Ok(total)
}

// ── the HTTP half ────────────────────────────────────────────────────────────

/// What one HTTP ingest pass measured.
#[cfg(feature = "server")]
struct HttpPass {
    /// Wall clock for the whole pass — the number a client actually experiences, and the
    /// only meaningful one once more than one client is running.
    wall: Duration,
    /// Summed client-side JSON encoding, excluded from `wall`'s attribution below.
    encode: Duration,
    /// Writes per durable barrier over the pass — the group-commit coalescing factor
    /// (nidus-xb9.1), read from the server's own counters. `1.0` means every write took its
    /// own fsync, which is what this pass looked like before group commit.
    coalescing: f64,
}

/// Ingest the plan over HTTP with `clients` concurrent writers.
#[cfg(feature = "server")]
fn http_pass(dataset: &Dataset, plan: &Plan, fsync: Fsync, clients: usize) -> Result<HttpPass> {
    use std::sync::atomic::{AtomicU64, Ordering};

    use nidus_bench::serve::{ServeProcess, post_on};
    use serde_json::json;

    let tmp = tempfile::Builder::new()
        .prefix("nidus-bench-write-http-")
        .tempdir()?;
    let fsync_arg = match fsync {
        Fsync::PerBatch => "per-batch",
        Fsync::OnFlush => "on-flush",
    };
    let proc = ServeProcess::spawn(
        &tmp.path().join("store"),
        dataset.dim,
        &["--fsync", fsync_arg],
    )?;
    proc.post(&format!("/collections/{COLLECTION}"), &json!({}))?;

    let path = format!("/collections/{COLLECTION}/upsert");
    let all = &plan.batches;
    let encode_nanos = AtomicU64::new(0);

    let t0 = Instant::now();
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(clients);
        for worker in 0..clients {
            // Round-robin rather than contiguous ranges: batches are uniform here, but an
            // interleave keeps every worker active for the whole pass even when they are
            // not, so the wall clock reflects sustained concurrency and not a tail.
            let mine: Vec<(usize, usize)> =
                all.iter().skip(worker).step_by(clients).copied().collect();
            let (agent, base, path) = (proc.agent.clone(), proc.base.clone(), path.as_str());
            let encode_nanos = &encode_nanos;
            handles.push(scope.spawn(move || -> Result<()> {
                let mut encode = Duration::ZERO;
                for (start, len) in mine {
                    let body = WireBody {
                        records: batch_records(dataset, start, len),
                    };
                    let t = Instant::now();
                    let bytes = serde_json::to_vec(&body)?;
                    encode += t.elapsed();

                    let res = post_on(&agent, &base, path, &bytes)?;
                    let upserted = res["upserted"].as_u64().unwrap_or(0) as usize;
                    if upserted != len {
                        bail!("upsert reported {upserted} of {len} records");
                    }
                }
                encode_nanos.fetch_add(encode.as_nanos() as u64, Ordering::Relaxed);
                Ok(())
            }));
        }
        for h in handles {
            h.join().map_err(|_| anyhow::anyhow!("writer panicked"))??;
        }
        Ok(())
    })?;
    let wall = t0.elapsed();

    // Under `on-flush` the server has acknowledged writes it has not yet synced; make it
    // pay that before the clock is read, or the policy looks free rather than deferred.
    if fsync == Fsync::OnFlush {
        proc.post("/flush", &json!({}))?;
    }

    // Scraped after the clock is read, so the extra request never lands inside the timing.
    let groups = proc.metric("nidus_write_groups_total")?;
    let members = proc.metric("nidus_write_group_members_total")?;

    Ok(HttpPass {
        wall,
        encode: Duration::from_nanos(encode_nanos.load(Ordering::Relaxed)),
        coalescing: if groups > 0.0 { members / groups } else { 0.0 },
    })
}

// ── reporting ────────────────────────────────────────────────────────────────

fn per_vec_us(d: Duration, n: usize) -> f64 {
    d.as_secs_f64() * 1e6 / n as f64
}

fn per_s(d: Duration, n: usize) -> f64 {
    if d.is_zero() {
        return f64::INFINITY;
    }
    n as f64 / d.as_secs_f64()
}

/// One row of the decomposition table.
fn stage_row(name: &str, d: Duration, n: usize, total: Duration) {
    let share = if total.is_zero() {
        0.0
    } else {
        100.0 * d.as_secs_f64() / total.as_secs_f64()
    };
    println!(
        "{:<28} {:>12.2} {:>14} {:>9.1}%",
        name,
        per_vec_us(d, n),
        fmt_count(per_s(d, n)),
        share
    );
}

/// Write the run as JSON for diffing against a later one.
fn write_json(path: &Path, args: &Args, cells: &[Value]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let doc = json!({
        "bench": "nidus-bench-write",
        // The version of NIDUS, not of this bench crate. `env!("CARGO_PKG_VERSION")` here
        // would report nidus-bench's own 0.1.0 — which is worse than nothing on a
        // baseline, since it looks authoritative while naming the wrong crate. nidus does
        // not expose its version, so `just bench-write` passes it in; a direct `cargo run`
        // gets "unknown" rather than a confident lie.
        "nidus_version": std::env::var("NIDUS_VERSION").unwrap_or_else(|_| "unknown".into()),
        "server_feature": cfg!(feature = "server"),
        "inputs": {
            "n": args.n,
            "dim": args.dim,
            "batch": args.batch,
            "clients": args.clients,
            "max_requests": args.max_requests,
            "seed": args.seed,
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
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = parse_args()?;
    let n = args.n;
    let mut doc_cells: Vec<Value> = Vec::new();

    #[cfg(not(feature = "server"))]
    eprintln!(
        "note: built without the `server` feature — reporting the in-process \
         decomposition only. Use `just bench-write` for the HTTP half."
    );

    for &dim in &args.dim {
        // Queries are unused here; ask for none.
        let dataset = data::generate(args.seed, n, dim, 0);

        // ── the decomposition, at the largest batch size ────────────────────
        //
        // Largest, because per-request costs are amortised there and what remains is the
        // per-vector floor — the thing that cannot be batched away and therefore the thing
        // that decides whether a single writer has a ceiling worth scaling out past.
        let batch = *args.batch.iter().max().expect("non-empty");
        let plan = Plan::new(n, batch, args.max_requests);
        let v = plan.vectors;
        let (encode, decode) = codec_pass(&dataset, &plan)?;
        let append = store_pass(&dataset, &plan, Fsync::OnFlush)?;
        let durable = store_pass(&dataset, &plan, Fsync::PerBatch)?;
        let fsync = durable.saturating_sub(append);

        println!("\n── ingest decomposition  n={v} dim={dim} batch={batch}  ───────────────");
        println!(
            "{:<28} {:>12} {:>14} {:>10}",
            "stage", "us/vector", "vectors/s", "share"
        );

        #[cfg(feature = "server")]
        let http = http_pass(&dataset, &plan, Fsync::PerBatch, 1)?;
        // Shares are of the full HTTP round trip when we have it, of the durable
        // in-process write otherwise — always of the total actually being decomposed.
        #[cfg(feature = "server")]
        let total = http.wall;
        #[cfg(not(feature = "server"))]
        let total = durable;

        stage_row("client JSON encode", encode, v, total);
        stage_row("server JSON decode", decode, v, total);
        stage_row("store append (no fsync)", append, v, total);
        stage_row("fsync (per batch)", fsync, v, total);
        stage_row("= in-process durable write", durable, v, total);

        // `mut` only under the `server` feature, which adds the http/residual keys.
        #[allow(unused_mut)]
        let mut stages = json!({
            "encode": per_vec_us(encode, v),
            "decode": per_vec_us(decode, v),
            "append": per_vec_us(append, v),
            "fsync": per_vec_us(fsync, v),
            "durable": per_vec_us(durable, v),
        });

        #[cfg(feature = "server")]
        {
            stage_row("HTTP round trip (measured)", http.wall, v, total);
            let residual = http.wall.saturating_sub(durable + decode);
            stage_row("  of which transport+runtime", residual, v, total);
            println!(
                "  (client-side encode ran concurrently and cost {:.2} us/vector)",
                per_vec_us(http.encode, v)
            );
            stages["http"] = json!(per_vec_us(http.wall, v));
            stages["residual"] = json!(per_vec_us(residual, v));
        }

        // ── batch-size sweep ────────────────────────────────────────────────
        //
        // `n` varies down the column: small batches are capped at `--max-requests` calls,
        // so the row reports the vectors it actually ingested rather than implying it
        // covered the same corpus as the row below it.
        println!("\n── batch-size sweep  dim={dim}  (vectors/s) ──────────────────────────");
        #[cfg(feature = "server")]
        println!(
            "{:<10} {:>10} {:>14} {:>14} {:>14}",
            "batch", "n", "append", "durable", "http"
        );
        #[cfg(not(feature = "server"))]
        println!(
            "{:<10} {:>10} {:>14} {:>14}",
            "batch", "n", "append", "durable"
        );

        let mut by_batch: Vec<Value> = Vec::new();
        for &b in &args.batch {
            let plan = Plan::new(n, b, args.max_requests);
            let v = plan.vectors;
            let append = store_pass(&dataset, &plan, Fsync::OnFlush)?;
            let durable = store_pass(&dataset, &plan, Fsync::PerBatch)?;
            #[allow(unused_mut)] // `http_per_s` is added only under the `server` feature
            let mut row = json!({
                "batch": b,
                "vectors": v,
                "append_per_s": per_s(append, v),
                "durable_per_s": per_s(durable, v),
            });
            #[cfg(feature = "server")]
            {
                let http = http_pass(&dataset, &plan, Fsync::PerBatch, 1)?;
                println!(
                    "{:<10} {:>10} {:>14} {:>14} {:>14}",
                    b,
                    v,
                    fmt_count(per_s(append, v)),
                    fmt_count(per_s(durable, v)),
                    fmt_count(per_s(http.wall, v))
                );
                row["http_per_s"] = json!(per_s(http.wall, v));
            }
            #[cfg(not(feature = "server"))]
            println!(
                "{:<10} {:>10} {:>14} {:>14}",
                b,
                v,
                fmt_count(per_s(append, v)),
                fmt_count(per_s(durable, v))
            );
            by_batch.push(row);
        }

        // ── concurrent-writer sweep ─────────────────────────────────────────
        #[allow(unused_mut)]
        let mut by_clients: Vec<Value> = Vec::new();
        #[cfg(feature = "server")]
        {
            println!("\n── concurrent writers  n={v} dim={dim} batch={batch}  ────────────────");
            // `writes/barrier` is the group-commit coalescing factor read off the server's
            // own counters (nidus-xb9.1) — without it a rising throughput curve cannot be
            // told apart from the machine simply having more cores to spend, and a *flat*
            // one cannot be told apart from group commit having silently stopped working.
            println!(
                "{:<10} {:>14} {:>12} {:>16}",
                "clients", "vectors/s", "vs 1 client", "writes/barrier"
            );
            let mut baseline = 0.0;
            for (i, &c) in args.clients.iter().enumerate() {
                let http = http_pass(&dataset, &plan, Fsync::PerBatch, c)?;
                let rate = per_s(http.wall, v);
                if i == 0 {
                    baseline = rate;
                }
                println!(
                    "{:<10} {:>14} {:>11.2}x {:>15.2}",
                    c,
                    fmt_count(rate),
                    if baseline > 0.0 { rate / baseline } else { 0.0 },
                    http.coalescing
                );
                by_clients.push(json!({
                    "clients": c,
                    "per_s": rate,
                    "writes_per_barrier": http.coalescing,
                }));
            }
        }

        doc_cells.push(json!({
            "dim": dim,
            "batch": batch,
            "vectors": v,
            "stages_us_per_vector": stages,
            "by_batch": by_batch,
            "by_clients": by_clients,
        }));
    }

    if let Some(path) = &args.json {
        write_json(path, &args, &doc_cells)?;
    }

    Ok(())
}
