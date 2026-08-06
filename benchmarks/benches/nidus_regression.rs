//! Criterion regression benchmarks for nidus — the "are we getting better / did we
//! regress?" signal, complementing the cross-engine parity table.

use std::collections::BTreeMap;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use nidus::{Config, Distance, Fsync, Nidus, QuantKind, Quantization, Record, SearchOpts};
use nidus_bench::data;
use std::hint::black_box;

const SEED: u64 = 42;

/// Materialize `n` vectors at `dim` from the shared generator into records.
fn records(n: usize, dim: usize) -> Vec<Record> {
    let ds = data::generate(SEED, n, dim, 0);
    (0..n)
        .map(|i| Record {
            id: i.to_string(),
            vector: Some(ds.vectors[i * dim..(i + 1) * dim].to_vec()),
            attrs: BTreeMap::new(),
        })
        .collect()
}

/// Build an in-memory store of `n` vectors at `dim` from the shared generator.
fn build_store(n: usize, dim: usize) -> Nidus {
    let mut db = Nidus::open_in_memory(dim).expect("open in-memory");
    db.create_collection("bench").expect("create collection");
    db.upsert("bench", &records(n, dim)).expect("upsert");
    db
}

/// Build a file-backed store (in a tempdir) with a specific `query_threads` and
/// optional quantization, so the parallel-scan path (f32, int8, or binary) can be
/// driven through the public `Config` API. Returns the `TempDir` guard alongside the
/// store to keep the backing files alive. Binary quantization pins cosine distance
/// (its only supported metric).
fn build_store_threaded(
    n: usize,
    dim: usize,
    threads: usize,
    quant: Option<Quantization>,
) -> (Nidus, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = Config::new(dir.path().join("store"), dim)
        .query_threads(threads)
        .auto_compact(None);
    if let Some(q) = quant {
        cfg = cfg.quantization(Some(q));
        if q.kind == QuantKind::Binary {
            cfg = cfg.distance(Distance::Cosine);
        }
    }
    let mut db = Nidus::open(cfg).expect("open store");
    db.create_collection("bench").expect("create collection");
    db.upsert("bench", &records(n, dim)).expect("upsert");
    (db, dir)
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");
    for &(n, dim) in &[(10_000usize, 384usize), (100_000, 768)] {
        let db = build_store(n, dim);
        let query = data::generate(SEED ^ 1, 1, dim, 0).vectors;
        let opts = SearchOpts {
            top_k: 10,
            ..Default::default()
        };
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{n}_dim{dim}")),
            &(),
            |b, _| {
                b.iter(|| {
                    let hits = db.search("bench", black_box(&query), &opts).unwrap();
                    black_box(hits);
                })
            },
        );
    }
    group.finish();
}

/// Same large search, swept across `query_threads` — the reproducible measurement
/// behind the parallel-scan speedup claim, across the f32, int8, and binary first
/// passes. The f32 scan is memory-bandwidth-bound, so its gain is sublinear; int8
/// moves 4× fewer bytes and binary 32× (compute- not bandwidth-bound), so those are
/// the paths that should scale with threads — binary hardest. One group each so they
/// diff separately, and the int8/binary groups also expose the recall/latency trade.
fn bench_parallel_search(c: &mut Criterion) {
    let (n, dim) = (100_000usize, 768usize);
    let query = data::generate(SEED ^ 1, 1, dim, 0).vectors;
    let opts = SearchOpts {
        top_k: 10,
        ..Default::default()
    };
    for (group_name, quant) in [
        ("parallel_search", None),
        ("parallel_search_quant", Some(Quantization::int8())),
        ("parallel_search_binary", Some(Quantization::binary())),
    ] {
        let mut group = c.benchmark_group(group_name);
        for &threads in &[1usize, 2, 4, 8] {
            let (db, _dir) = build_store_threaded(n, dim, threads, quant);
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("threads{threads}")),
                &(),
                |b, _| {
                    b.iter(|| {
                        let hits = db.search("bench", black_box(&query), &opts).unwrap();
                        black_box(hits);
                    })
                },
            );
            // `_dir` stays alive through the synchronous bench above, then drops here.
        }
        group.finish();
    }
}

fn bench_ingest(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest");
    let (n, dim) = (10_000usize, 384usize);
    let ds = data::generate(SEED, n, dim, 0);
    let records: Vec<Record> = (0..n)
        .map(|i| Record {
            id: i.to_string(),
            vector: Some(ds.vectors[i * dim..(i + 1) * dim].to_vec()),
            attrs: BTreeMap::new(),
        })
        .collect();
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function(format!("n{n}_dim{dim}"), |b| {
        b.iter_batched(
            || {
                let mut db = Nidus::open_in_memory(dim).unwrap();
                db.create_collection("bench").unwrap();
                db
            },
            |mut db| {
                db.upsert("bench", black_box(&records)).unwrap();
                black_box(db)
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
}

/// The **file-backed** write path, across fsync policy and batch size — the regression
/// lane for nidus-xb9.1 (group commit).
fn bench_write_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_path");
    // The floor criterion allows. These are IO benchmarks whose variance comes from the
    // filesystem, not from sampling noise, so more samples mostly buys wall time.
    group.sample_size(10);
    let dim = 384;
    // Deliberately modest: this measures per-CALL cost, which is constant in store size
    // (verified while fixing nidus-4h2), so a bigger corpus buys noise, not signal.
    let n = 200;
    let recs = records(n, dim);

    for (label, fsync) in [("per_batch", Fsync::PerBatch), ("on_flush", Fsync::OnFlush)] {
        for batch in [1usize, 10, 100] {
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(
                BenchmarkId::new(label, format!("b{batch}")),
                &(batch, fsync),
                |b, &(batch, fsync)| {
                    b.iter_batched(
                        // Setup is untimed: a fresh store per iteration, so no run
                        // inherits the previous one's rows or dirty pages.
                        || {
                            let tmp = tempfile::Builder::new()
                                .prefix("nidus-crit-write-")
                                .tempdir()
                                .expect("tempdir");
                            let db = Nidus::open(Config::new(tmp.path(), dim).fsync(fsync))
                                .expect("open store");
                            (tmp, db)
                        },
                        |(tmp, mut db)| {
                            for chunk in recs.chunks(batch) {
                                db.upsert("bench", black_box(chunk)).expect("upsert");
                            }
                            // Included, not incidental: under OnFlush this is where
                            // durability is actually paid for, and omitting it would
                            // flatter that policy by exactly the cost being measured.
                            db.flush().expect("flush");
                            black_box(&db);
                            drop(db);
                            drop(tmp);
                        },
                        criterion::BatchSize::PerIteration,
                    )
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_search,
    bench_parallel_search,
    bench_ingest,
    bench_write_path
);
criterion_main!(benches);
