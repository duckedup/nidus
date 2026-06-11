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

## Optional speed levers

Two opt-in knobs trade a little for more speed when the exact single-threaded
sweep isn't enough. Both stay pure-safe-Rust and are off by default:

- **[int8 quantization](/guides/search/#int8-scalar-quantization)** — a two-pass
  search (int8 first-pass → f32 rerank) returns essentially the exact neighbours
  (**~100% recall@10 at `rescore` ≥ 2**) for a **~1.4× speedup** at 1M × 768, at
  the cost of ~25% more RAM. Reproduce: `just bench-quant`.
- **[parallel scan](/guides/integrating/#two-kinds-of-parallelism)**
  (`Config::query_threads`) — splits one large search across worker threads. The
  plain **f32** scan is memory-bandwidth-bound, so its gain is sublinear and plateaus
  early (**~1.3–1.4×** at 4–8 threads). Combined with **int8 quantization**, threads
  pay off: the int8 first pass moves 4× fewer bytes, so it is compute- not
  bandwidth-bound and scales to **~2.4×** at 4 threads. Reproduce:
  `just bench-crit parallel_search` (the `parallel_search_quant` group is the
  quantized sweep).
- **[approximate index (HNSW / IVF)](/guides/search/#approximate-search-ann)**
  (`Config::ann`) — walks an index instead of scanning every vector, for when the
  collection outgrows a full scan. On realistic clustered data (n=20k, dim=768) HNSW
  returns **~0.99–1.0 recall@10 at ~7–10× the query speed** of the exact scan; IVF
  ~1.0 recall at ~3×. The graph is in-RAM but [persisted](/guides/search/#approximate-search-ann)
  so a warm `open()` is **~0.05 s** instead of rebuilding (~36 s here), and a cold
  rebuild parallelizes over `query_threads` (**~36 s → ~5 s at 8 threads**). Recall is
  data-dependent — uniform-random vectors are a near-worst case — so measure your own:
  `just bench-ann clustered=1`.

Neither int8 nor threads is the headline multiplier its theory suggests — the 4× from int8 and the
linear scaling from threads both want SIMD/bandwidth headroom nidus doesn't chase
within its zero-FFI design. The f32 scan is bandwidth-bound; threads help most when
paired with the int8 first pass, which has the compute headroom to scale. They're
honest latency wins for the right workload, measured by benchmarks you can run
yourself.

## The target regime

nidus is tuned for **exact** search at the scale where a full scan wins: up to a
few million vectors, comfortably in RAM. At that size, 100% recall with no index
to build or tune beats an approximate index — and exact search is the default, so
you never pay for an index you don't need.

Past that scale, an [approximate index](/guides/search/#approximate-search-ann)
(HNSW or IVF, via `Config::ann`) is available as an opt-in: it trades some recall
for a smaller candidate walk instead of a full scan. It is pure-Rust, optional, and
additive over the same append-only file — exact search is unchanged when it is off.
The benchmark above measures the exact path; run `just bench-ann` to sweep the
approximate variants' recall and latency on your own shapes.

## Reproduce it

```bash
just bench all                  # cross-engine parity table (nidus vs DuckDB vs LanceDB)
just bench-quant                # int8 quantization recall & speed sweep
just bench-crit parallel_search # query_threads scaling (criterion)
```

The heavy DuckDB/LanceDB dependencies are **quarantined off nidus's own build
path** (in `benchmarks/`), so they never touch the seconds-long build of nidus
itself. Synthetic data on an Apple Silicon laptop — useless, like all
benchmarks, but there it is.
