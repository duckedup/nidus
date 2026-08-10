//! Criterion regression benchmarks for nidus — the "are we getting better / did we
//! regress?" signal, complementing the cross-engine parity table.

use std::collections::BTreeMap;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use nidus::{
    Config, Decay, Distance, Fsync, FtsField, FtsQuery, HybridOpts, Nidus, QuantKind, Quantization,
    RankBy, Record, SearchOpts, Value,
};
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

/// Build a file-backed store in a tempdir with a specific `query_threads` and optional
/// quantization, driving the parallel-scan path through the public `Config` API. Returns the
/// `TempDir` guard too, to keep the backing files alive.
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

/// Synthetic FTS vocabulary size and fixed token count per document — a few hundred terms
/// and 32 tokens/doc is plenty for BM25 to see real term-frequency variation.
const TEXT_VOCAB: usize = 300;
const TOKENS_PER_DOC: usize = 32;

/// splitmix64, mirroring `nidus_bench::data::Rng` (whose random-producing methods are
/// private outside that crate) so the text corpus below can sample deterministically too.
struct TermRng(u64);

impl TermRng {
    fn new(seed: u64) -> Self {
        TermRng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0, 1)`, for sampling against a cumulative weight table.
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Zipf-ish cumulative weights over `vocab` terms (`weight(rank) = 1/(rank+1)`): a handful
/// of hot terms dominate and the rest form a long tail, so BM25's IDF actually varies —
/// uniform term frequencies would make it degenerate and measure nothing interesting.
fn zipf_cumulative(vocab: usize) -> Vec<f64> {
    let mut cum = Vec::with_capacity(vocab);
    let mut acc = 0.0;
    for rank in 0..vocab {
        acc += 1.0 / (rank as f64 + 1.0);
        cum.push(acc);
    }
    let total = *cum.last().unwrap();
    for c in &mut cum {
        *c /= total;
    }
    cum
}

/// Sample one vocabulary index from `cumulative` via inverse-CDF lookup.
fn sample_term(rng: &mut TermRng, cumulative: &[f64]) -> usize {
    let u = rng.next_unit();
    match cumulative.binary_search_by(|c| c.partial_cmp(&u).unwrap()) {
        Ok(i) => i,
        Err(i) => i.min(cumulative.len() - 1),
    }
}

/// `n` deterministic documents of exactly `TOKENS_PER_DOC` space-joined tokens each, drawn
/// from a fixed `TEXT_VOCAB`-term Zipf-ish vocabulary — reproducible from `seed` alone.
fn text_corpus(seed: u64, n: usize) -> Vec<String> {
    let vocab: Vec<String> = (0..TEXT_VOCAB).map(|i| format!("term{i}")).collect();
    let cumulative = zipf_cumulative(TEXT_VOCAB);
    let mut rng = TermRng::new(seed);
    (0..n)
        .map(|_| {
            (0..TOKENS_PER_DOC)
                .map(|_| vocab[sample_term(&mut rng, &cumulative)].as_str())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Build an in-memory store of `n` records at `dim`, each carrying a vector plus a `text`
/// attr from the Zipf-ish synthetic corpus, with FTS declared via `set_fts_schema` before
/// the upsert — the shared fixture for the text/hybrid benches below.
fn build_text_store(n: usize, dim: usize) -> Nidus {
    let ds = data::generate(SEED, n, dim, 0);
    let texts = text_corpus(SEED, n);
    let mut db = Nidus::open_in_memory(dim).expect("open in-memory");
    db.set_fts_schema("bench", &[FtsField::new("text")])
        .expect("set fts schema");
    let recs: Vec<Record> = (0..n)
        .map(|i| {
            let mut attrs = BTreeMap::new();
            attrs.insert("text".to_string(), Value::Str(texts[i].clone()));
            Record {
                id: i.to_string(),
                vector: Some(ds.vectors[i * dim..(i + 1) * dim].to_vec()),
                attrs,
            }
        })
        .collect();
    db.upsert("bench", &recs).expect("upsert");
    db
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

/// The same large search swept across `query_threads` — the reproducible measurement behind the
/// parallel-scan claim. The f32 scan is bandwidth-bound so its gain is sublinear; int8 and binary
/// move 4×/32× fewer bytes and should scale with threads. One group each so they diff separately.
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

/// BM25 full-text search, single- vs multi-term — term count is the dominant cost.
fn bench_text_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_search");
    let (n, dim) = (10_000usize, 384usize);
    let db = build_text_store(n, dim);
    let opts = SearchOpts {
        top_k: 10,
        ..Default::default()
    };
    group.throughput(Throughput::Elements(n as u64));
    for (label, text) in [
        ("single_hot_term", "term0"),
        ("multi_term", "term0 term5 term20 term100"),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(label), &(), |b, _| {
            b.iter(|| {
                let query = FtsQuery::new("text", black_box(text));
                let hits = db.text_search("bench", &query, &opts).unwrap();
                black_box(hits);
            })
        });
    }
    group.finish();
}

/// RRF fusion of the vector and BM25 legs, swept across `HybridOpts::candidates` — how deep
/// each leg is pulled before fusing, and the knob whose cost is linear.
fn bench_hybrid(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid");
    let (n, dim) = (10_000usize, 384usize);
    let db = build_text_store(n, dim);
    let query_vec = data::generate(SEED ^ 1, 1, dim, 0).vectors;
    let text = FtsQuery::new("text", "term0 term5 term20");
    group.throughput(Throughput::Elements(n as u64));
    for &candidates in &[50usize, 100, 400] {
        let opts = HybridOpts {
            candidates,
            ..Default::default()
        };
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("candidates{candidates}")),
            &(),
            |b, _| {
                b.iter(|| {
                    let hits = db
                        .hybrid_search("bench", black_box(&query_vec), &text, &opts)
                        .unwrap();
                    black_box(hits);
                })
            },
        );
    }
    group.finish();
}

/// The `RankBy::Decay` overhead: the same search with and without the expression, so the
/// published number is the expression's own cost. Single-threaded (`open_in_memory` leaves
/// `query_threads` at 1), named so it is never compared against a parallel number.
fn bench_rank_by(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_by_decay_single_threaded");
    let (n, dim) = (10_000usize, 384usize);
    let mut db = Nidus::open_in_memory(dim).expect("open in-memory");
    db.create_collection("bench").expect("create collection");
    let ds = data::generate(SEED, n, dim, 0);
    let recs: Vec<Record> = (0..n)
        .map(|i| {
            let mut attrs = BTreeMap::new();
            attrs.insert("ts".to_string(), Value::DateTime(i as i64 * 60_000));
            Record {
                id: i.to_string(),
                vector: Some(ds.vectors[i * dim..(i + 1) * dim].to_vec()),
                attrs,
            }
        })
        .collect();
    db.upsert("bench", &recs).expect("upsert");
    let query = data::generate(SEED ^ 1, 1, dim, 0).vectors;

    group.throughput(Throughput::Elements(n as u64));
    for label in ["without_decay", "with_decay"] {
        let rank_by = (label == "with_decay")
            .then(|| RankBy::Decay(Decay::new("ts", n as i64 * 60_000, 7 * 24 * 60 * 60 * 1000)));
        let opts = SearchOpts {
            top_k: 10,
            rank_by,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(label), &(), |b, _| {
            b.iter(|| {
                let hits = db.search("bench", black_box(&query), &opts).unwrap();
                black_box(hits);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_search,
    bench_parallel_search,
    bench_ingest,
    bench_write_path,
    bench_text_search,
    bench_hybrid,
    bench_rank_by
);
criterion_main!(benches);
