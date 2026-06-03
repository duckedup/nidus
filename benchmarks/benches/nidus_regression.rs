//! Criterion regression benchmarks for nidus — the "are we getting better / did we
//! regress?" signal, complementing the cross-engine parity table.
//!
//! These exercise nidus through its PUBLIC API (open_in_memory → upsert → search), which
//! drives the same dot-product + top-k hot paths as a real query, without any bench-only
//! surface on the crate. criterion is a dev-dependency of nidus-bench only, so it never
//! enters nidus's own build/test/CI path.
//!
//!   just bench-crit                        run all
//!   just bench-crit --save-baseline main   record a baseline to diff later runs against

use std::collections::BTreeMap;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use nidus::{Nidus, Record, SearchOpts};
use nidus_bench::data;
use std::hint::black_box;

const SEED: u64 = 42;

/// Build an in-memory store of `n` vectors at `dim` from the shared generator.
fn build_store(n: usize, dim: usize) -> Nidus {
    let ds = data::generate(SEED, n, dim, 0);
    let mut db = Nidus::open_in_memory(dim).expect("open in-memory");
    db.create_collection("bench").expect("create collection");
    let records: Vec<Record> = (0..n)
        .map(|i| Record {
            id: i.to_string(),
            vector: ds.vectors[i * dim..(i + 1) * dim].to_vec(),
            attrs: BTreeMap::new(),
        })
        .collect();
    db.upsert("bench", &records).expect("upsert");
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

fn bench_ingest(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest");
    let (n, dim) = (10_000usize, 384usize);
    let ds = data::generate(SEED, n, dim, 0);
    let records: Vec<Record> = (0..n)
        .map(|i| Record {
            id: i.to_string(),
            vector: ds.vectors[i * dim..(i + 1) * dim].to_vec(),
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

criterion_group!(benches, bench_search, bench_ingest);
criterion_main!(benches);
