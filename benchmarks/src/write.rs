//! nidus-bench-write — where a single writer's throughput actually goes (nidus-xb9).
//!
//! Epic nidus-xb9 asks whether writes should be scaled out past the one fenced writer
//! lease. Its stated prerequisite is a measurement, because nidus-8fn had already found a
//! 10–12× gap between in-process ingest (~139–169k vec/s) and ingest over HTTP (~14k/s)
//! and attributed it, unverified, to "JSON float encoding". Scaling writers out before
//! knowing which layer actually saturates would be optimising the wrong one.
//!
//! So this bench decomposes one writer's ingest path into the stages a vector actually
//! passes through, and times each in isolation:
//!
//! | stage | what it is |
//! | --- | --- |
//! | `encode` | client turns `dim` floats into JSON — pure CPU, client side |
//! | `decode` | server turns that JSON back into `Record`s — pure CPU, server side |
//! | `append` | `Nidus::upsert` with [`Fsync::OnFlush`]: normalise, append, index |
//! | `fsync` | the same upsert with [`Fsync::PerBatch`], minus `append` |
//! | `http` | the real round trip against a live `nidus serve` |
//!
//! `http - (decode + append + fsync)` is then the residual: sockets, the tokio hop, the
//! middleware stack, and the `RwLock`. Naming it as a residual rather than measuring it
//! directly is deliberate — it is whatever is left, so nothing can hide in it.
//!
//! Two sweeps answer the questions the decomposition raises:
//!
//! * **Batch size.** Per-request costs (one fsync, one round trip) amortise over a batch;
//!   per-vector costs (JSON) do not. The sweep separates them, which is the direct test of
//!   option 1 in nidus-xb9 — batch coalescing / group commit.
//! * **Concurrent clients.** Every write takes the store exclusively, so if throughput is
//!   flat as clients rise the writer is lock-bound and only scale-out or a faster critical
//!   section helps; if it rises, the bottleneck was above the store and a single writer
//!   still has headroom.
//!
//! Run via `just bench-write [key=value ...]`:
//!   n=50000              vectors ingested per pass
//!   dim=384,768          embedding dimension(s)
//!   batch=1,10,100,1000  records per upsert call/request
//!   clients=1,2,4,8      concurrent HTTP writers (needs the `server` feature)
//!   max_requests=500     ceiling on upsert calls per pass (see `Plan::new`)
//!   seed=42              PRNG seed
//!
//! The HTTP half needs the `server` feature and a release `nidus` binary; without them
//! the in-process decomposition still runs and says so.

use std::collections::BTreeMap;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use nidus::{Config, Fsync, Nidus, Record};
use nidus_bench::data::{self, Dataset};
use nidus_bench::report::fmt_count;

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
            println!("args: n=, dim=, batch=, clients=, max_requests=, seed=");
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
///
/// Copied rather than imported because that DTO is behind nidus's `cli` feature and is not
/// public API. The point of this bench is that the bytes are identical to what the server
/// parses, so if the two ever drift the decode measurement quietly stops measuring the
/// server's work — the shape is one field and has been stable since the endpoint shipped.
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
///
/// Built once per `(n, batch)` and handed to every pass, so the in-process and HTTP
/// numbers on one line are always over identical work — a decomposition whose stages
/// measured different amounts of data would subtract to nonsense.
struct Plan {
    /// `(start_row, len)` per upsert call.
    batches: Vec<(usize, usize)>,
    /// Rows actually ingested — the denominator for every per-vector figure.
    vectors: usize,
}

impl Plan {
    /// Cover `n` rows in `batch`-sized calls, stopping after `max_requests` of them.
    ///
    /// The cap exists because small batches are dominated by *per-request* cost, and on
    /// macOS `File::sync_all` is `F_FULLFSYNC` — a real drive barrier of several
    /// milliseconds. At `batch=1` the uncapped sweep would be 50k barriers per pass, tens
    /// of minutes to re-measure a per-request cost that a few thousand samples already
    /// pin down. Throughput per vector is what is being measured, so a shorter pass costs
    /// precision, not validity — and the effective `n` is printed rather than implied.
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
///
/// Store creation is untimed; the closing `flush` is included, because under
/// [`Fsync::OnFlush`] that is where durability is actually paid for and leaving it out
/// would flatter the policy by exactly the cost being measured.
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

    Ok(HttpPass {
        wall,
        encode: Duration::from_nanos(encode_nanos.load(Ordering::Relaxed)),
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

        #[cfg(feature = "server")]
        {
            stage_row("HTTP round trip (measured)", http.wall, v, total);
            let residual = http.wall.saturating_sub(durable + decode);
            stage_row("  of which transport+runtime", residual, v, total);
            println!(
                "  (client-side encode ran concurrently and cost {:.2} us/vector)",
                per_vec_us(http.encode, v)
            );
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

        for &b in &args.batch {
            let plan = Plan::new(n, b, args.max_requests);
            let v = plan.vectors;
            let append = store_pass(&dataset, &plan, Fsync::OnFlush)?;
            let durable = store_pass(&dataset, &plan, Fsync::PerBatch)?;
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
            }
            #[cfg(not(feature = "server"))]
            println!(
                "{:<10} {:>10} {:>14} {:>14}",
                b,
                v,
                fmt_count(per_s(append, v)),
                fmt_count(per_s(durable, v))
            );
        }

        // ── concurrent-writer sweep ─────────────────────────────────────────
        #[cfg(feature = "server")]
        {
            println!("\n── concurrent writers  n={v} dim={dim} batch={batch}  ────────────────");
            println!(
                "{:<10} {:>14} {:>12}",
                "clients", "vectors/s", "vs 1 client"
            );
            let mut baseline = 0.0;
            for (i, &c) in args.clients.iter().enumerate() {
                let http = http_pass(&dataset, &plan, Fsync::PerBatch, c)?;
                let rate = per_s(http.wall, v);
                if i == 0 {
                    baseline = rate;
                }
                println!(
                    "{:<10} {:>14} {:>11.2}x",
                    c,
                    fmt_count(rate),
                    if baseline > 0.0 { rate / baseline } else { 0.0 }
                );
            }
        }
    }

    Ok(())
}
