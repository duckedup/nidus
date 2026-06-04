---
title: Performance
description: Exact brute-force cosine KNN benchmarked against DuckDB and LanceDB — nidus is the fastest in every cell, at 100% recall, while compiling in seconds.
---

Every vector store ships a benchmark proving it's the fastest, on synthetic data
that looks nothing like your workload. It's a genre. Here's ours — and yes, we
win our own benchmark, that's how this works.

## The numbers

Exact brute-force cosine KNN, 100k vectors, single thread, measured against
DuckDB (`array_cosine_similarity`) and LanceDB (`bypass_vector_index`) — both
pinned to the same exact search, so all three return the same neighbours. The
harness computes its own independent ground truth and reports **recall@k** for
every engine (including nidus), so none is trusted as the oracle. Numbers are
query p50; lower is better.

| n=100k   | top_k | **nidus**   | LanceDB  | DuckDB   | recall |
| -------- | ----: | ----------: | -------: | -------: | :----: |
| dim=384  |    10 | **5.44 ms** | 12.29 ms | 32.29 ms |  100%  |
| dim=384  |   100 | **5.53 ms** | 28.52 ms | 30.59 ms |  100%  |
| dim=768  |    10 | **8.09 ms** | 24.78 ms | 69.54 ms |  100%  |
| dim=768  |   100 | **8.57 ms** | 53.16 ms | 64.99 ms |  100%  |

All three are exact (recall 100%); nidus is the fastest in every cell while being
the one that compiles in seconds with zero FFI.

## Why it's fast

The scoring kernel is plain safe Rust the optimizer can vectorize:

- An **8-lane chunked dot product** the compiler auto-vectorizes to SIMD.
- An **allocation-free top-k scan** backed by a bounded heap.
- A **storage-order, prefetcher-friendly sweep** of the row-major matrix.

No FFI boundary to cross, no columnar decode, no query planner — just a tight
loop over a contiguous `f32` matrix that is already resident in RAM.

## The target regime

nidus is built for **exact** search at the scale where brute force wins: up to a
few million vectors, comfortably in RAM. At that size, 100% recall with no index
to build or tune beats an approximate index — and you never pay for an ANN
structure you don't need.

ANN/HNSW is a [deferred seam](/guides/how-it-works/#what-it-deliberately-is-not),
not a missing feature: it stays unbuilt until a real consumer hits a scale and
latency budget brute force can't meet — and even then it must be pure-Rust,
optional, and additive over the same append-only file.

## Reproduce it

```bash
just bench all
```

The heavy DuckDB/LanceDB dependencies are **quarantined off nidus's own build
path** (in `benchmarks/`), so they never touch the seconds-long build of nidus
itself. Synthetic data on an Apple Silicon laptop — useless, like all
benchmarks, but there it is.
