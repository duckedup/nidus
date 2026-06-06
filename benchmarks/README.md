# nidus-bench

A quarantined benchmark that confirms nidus's **exact brute-force cosine KNN** stays *in
line* with DuckDB and LanceDB. The goal is **parity, not winning** — nidus serves a
different purpose (tiny, pure-Rust, zero-FFI, seconds-long builds); this tool just makes
sure we match on the search path and catches regressions over time.

## Why it's a separate crate

`nidus-bench` deliberately pulls the heavy dependencies nidus exists to avoid — **bundled
DuckDB** (compiles the C++ engine from source) and **LanceDB** (pure Rust, but Arrow +
DataFusion + lance is a long compile). The root workspace pins `default-members = ["."]`,
so `cargo build` / `cargo test` / `just ci` touch **only nidus** and stay fast. The heavy
engines compile **only** when you explicitly ask for them here, behind cargo features.

## Running

All via `just` (engine deps are gated — only what you ask for is compiled):

```bash
just bench                 # nidus only (quick, no heavy deps)
just bench duckdb          # nidus + DuckDB
just bench lancedb         # nidus + LanceDB
just bench all             # nidus + DuckDB + LanceDB
just bench all top_k=100 n=1000000   # extra key=value args pass through
just bench all help        # list all harness args
```

First build of a heavy engine is slow (bundled DuckDB ≈ minutes of C++; LanceDB ≈ minutes
of Rust) and then cached in `target/`. No system setup is required — both engines are
fully self-contained (no `brew install`, no system libduckdb).

### What it measures

Per `(engine × n × dim × top_k)` cell, over a deterministic seeded dataset: build +
ingest time, ingest throughput, query latency `p50/p95/p99`, on-disk size, and
**recall@k**. All three engines are pinned to **exact** search (LanceDB
`bypass_vector_index`, DuckDB `array_cosine_similarity` scan, nidus native).

### Fairness

- **Recall is scored against an independent ground truth.** The harness computes its own
  exact top-k by full brute-force cosine in `f64` (in `lib::exact_ground_truth`), straight
  from the raw dataset — *not* from any engine's output. recall@k is then reported for
  **every** engine, nidus included, so none is trusted as the oracle. ~100% across the
  board confirms the configs are genuinely exact (no accidental ANN).
- **Identical inputs.** Every engine sees the same seeded vectors and the same queries;
  the timed region is exactly the `search` call; warmup and iteration counts are equal.

What it deliberately does **not** control (single-process micro-benchmark, so read with
this in mind): engines run sequentially in a fixed order (mild cache/thermal effects);
ingest durability semantics differ per engine (nidus fsyncs per batch); and at small `n`
the per-query fixed overheads (nidus parses string ids, LanceDB enters its async runtime)
are visible — the comparison is most meaningful where the scan dominates (larger `n`).

A configurable threshold (`threshold=1.25`: nidus p50 ≤ 1.25× the best engine) sets the
process exit code, so the run doubles as a pass/fail guard. Each run also writes a JSON
artifact under `target/bench-results/<stamp>.json` for diffing over time.

## int8 quantization sweep

```bash
just bench-quant                       # recall + speed across rescore=1,2,4,8
just bench-quant n=1000000 dim=768      # extra key=value args pass through
```

`bench-quant` builds one exact (f32) store and one quantized store per `rescore`
factor over identical data, then reports each variant's **recall@k** against the
harness's independent exact ground truth plus query latency and speedup vs the
exact path. It's nidus-only (no engine deps) and is how the default `rescore` and
the documented recall/speed expectations were chosen.

## nidus regression tracking (criterion)

```bash
just bench-crit                        # all groups: search, parallel_search, ingest
just bench-crit parallel_search        # just the query_threads scaling group
just bench-crit --save-baseline main   # record a baseline; later runs report the delta
```

`bench-crit` benchmarks nidus through its public API with criterion's statistical
sampling and baseline comparison — the "did we regress?" signal. It covers single-threaded
`search`, the `parallel_search` sweep across `query_threads` (1/2/4/8 — the reproducible
parallel-scan measurement), and `ingest`. criterion is a dev-dependency of *this* crate
only and never touches nidus's build.
