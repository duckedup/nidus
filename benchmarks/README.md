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
ingest time, ingest throughput, query latency `p50/p95/p99`, and on-disk size. All three
engines are pinned to **exact** search (LanceDB `bypass_vector_index`, DuckDB
`array_cosine_similarity` scan, nidus native), so the harness also reports **top-k id
agreement vs nidus** as a fairness check — it should be ~100%.

A configurable threshold (`threshold=1.25`: nidus p50 ≤ 1.25× the best engine) sets the
process exit code, so the run doubles as a pass/fail guard. Each run also writes a JSON
artifact under `target/bench-results/<stamp>.json` for diffing over time.

## nidus regression tracking (criterion)

```bash
just bench-crit                        # nidus-internal benchmarks
just bench-crit --save-baseline main   # record a baseline; later runs report the delta
```

`bench-crit` benchmarks nidus through its public API (search + ingest) with criterion's
statistical sampling and baseline comparison — the "did we regress?" signal. criterion is
a dev-dependency of *this* crate only and never touches nidus's build.
