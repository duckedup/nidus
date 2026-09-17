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

## Named-vector cost, pooling, parallelism & recall sweep

```bash
just bench-named                                      # n=100k, dim=384/768, names=1,2,4
just bench-named names=1,2,4 n=100000 dim=768          # extra key=value args pass through
just bench-named names=4 threads=1,2,4,8               # parallelism sweep at names=4
just bench-named help                                  # list every arg
```

`bench-named` (nidus-85t) answers whether scoring several named vectors per record, then
reducing them to one score before top-k selection, is *worth* what it costs. Four numbers,
one binary:

1. **Cost of naming.** Per-query p50 and rows scanned at `names=1,2,4` over a fixed record
   count, exact path, against a **pre-change baseline**: the same vectors and queries, but
   ingested as a plain `vector` with no name declared at all (the code path `search` took
   before this feature). `names=1` is asserted, not just printed, **not to exceed
   `tolerance` (default 1.15x)** the baseline p50 — the number most likely to embarrass us,
   since a store with one named vector must not be measurably slower than before.
2. **`Max` vs `Sum` pooling**, over the identical rows. They walk the same scan, so a large
   gap between them means the reduction itself, not the scan, is the cost.
3. **A `threads=` sweep at `names=4`.** The core design snaps shard boundaries to record
   boundaries so pooling stays exact *and* parallel. Throughput is asserted to reach at
   least `min_scale` (default 40%) of ideal linear speedup at the top of the sweep; if it
   doesn't, the reduction has serialized and the design claim needs revisiting, not the
   benchmark.
4. **ANN recall per `names=` value**, via the existing `exact_ground_truth` + `recall_at_k`.
   At `names=1`, pooling is a no-op, so the harness's own independent brute-force ground
   truth applies directly, as a genuine correctness cross-check. **Caveat, read before
   trusting this number:** named-vector search always takes the exact path today,
   regardless of `Config::ann` (`src/store/read.rs`), so recall@k is 1.0 by construction
   for every `names=` value — this benchmark will start reporting real degradation only
   once ANN composes with named vectors.

**What this does NOT prove.** Retrieval *quality* (nDCG@10, Recall@100 against labeled
relevance judgements) is covered separately, by `bench-retrieval` below, not by this suite.
This suite proves named vectors are fast enough and approximation-safe; it does not prove
records are retrieved *better*.

### Baselines

```bash
just bench-named json=benchmarks/baselines/named-<version>.json   # record
```

Same convention as `bench-write`'s baselines: the file records the *inputs* next to the
results, and a comparison only means something **on the same box** and against a baseline
recorded with the same knobs.

## BEIR retrieval quality

```bash
just bench-retrieval                                          # all three datasets
just bench-retrieval dataset=scifact                          # one dataset
```

`bench-retrieval` (`nidus-yq9p.5`) answers whether nidus's results are good, not just fast.
It runs against real labeled relevance judgements (BEIR's qrels) rather than synthetic
data, which makes it the only bench here that does. Per dataset (SciFact, NFCorpus,
FiQA-2018), it reports nDCG@10 and Recall@100 for four legs: FTS only, vector only, RRF
fusion at the shipped defaults (`rrf_k` 60, `candidates` 100, both weights 1.0), and fusion
plus rerank, embedding through Voyage `voyage-4` and reranking with `rerank-2.5`.

It requires a live `VOYAGE_API_KEY` and network access, and downloads roughly 60k
documents' worth of corpora on first run. Corpora and embeddings cache under
`benchmarks/.cache/` (gitignored), so a rerun costs no API calls.

**What this does NOT prove.** Three small English datasets, one embedding model, and one
recorded run on one developer machine. Nothing in CI re-verifies these numbers
(`nidus-yq9p.7` will add floors); a rerun is the only way to check they still hold.

### Baselines

```bash
just bench-retrieval json=benchmarks/baselines/retrieval-<dataset>-<version>.json   # record
```

The committed baselines are named `0.104.0` because that is the nidus they measured: this
lane's PR changes nothing under `src/`, so the library under test is the released 0.104.0
library even though the PR ships as 0.105.0. They were also recorded just before
`rerank_overscan` was added to the JSON inputs block, so that one field is absent from them.
Its value for those runs was 1, the same value the code uses today, and every later run
records it.

## Single-writer ingest decomposition

```bash
just bench-write                              # n=50k, dim=384/768, batch=1..1000
just bench-write n=100000 dim=768 batch=1000  # extra key=value args pass through
```

`bench-write` answers *which layer a single writer saturates on*, by timing each stage a
vector passes through on its way into the store — client JSON encode, server JSON decode,
`Nidus::upsert` without fsync, the fsync itself, and the real HTTP round trip. Everything
the round trip costs beyond the sum of the others is reported as a **residual**
("transport+runtime": sockets, the tokio hop, the middleware stack, the `RwLock`) rather
than measured directly, so nothing can hide in it.

Two sweeps come with it. **Batch size** separates per-request costs (one fsync, one round
trip), which amortise, from per-vector costs (JSON), which do not. **Concurrent writers**
shows whether one server process still has headroom above the store's exclusive lock.

This is the measurement epic `nidus-xb9` asked for before deciding whether nidus needs
more than one writer; the recorded findings live on that issue.

### Baselines

```bash
just bench-write json=benchmarks/baselines/write-<version>.json   # record
```

A printed table is for reading; a baseline is for **diffing**. `benchmarks/baselines/`
holds committed runs so a change can be argued against a number rather than a memory:

| baseline | what it is |
| --- | --- |
| `write-0.36.0.json` | before group commit, right after `nidus-4h2` removed the stray per-call fsync — every write pays its own barrier |
| `write-0.37.0.json` | after group commit (`nidus-xb9.1`) |

The concurrent-writer sweep is where the two differ, and the `writes/barrier` column says
why: on the same box, 8 clients at 384-d went **85k → 134k vectors/s** at 3.0 writes per
barrier, and at 768-d **70k → 85k** at 2.3. At 1 and 2 clients the column reads `1.00` —
group commit forms no group when nothing is waiting, so the uncontended path is untouched,
which is the property to check first if that number ever drifts above 1 at one client.

The file records the *inputs* (`n`, `dim`, `batch`, `clients`, `max_requests`, `seed`)
next to the results, because a baseline compared against a run with different knobs is
not a comparison and there would otherwise be nothing to catch it. These are wall-clock
numbers from one developer machine: compare runs **on the same box**, and treat a
cross-machine diff as meaningless.

## nidus regression tracking (criterion)

```bash
just bench-crit                        # all groups
just bench-crit parallel_search        # just the query_threads scaling group
just bench-crit write_path             # just the file-backed write path
just bench-crit --save-baseline main   # record a baseline; later runs report the delta
```

`bench-crit` benchmarks nidus through its public API with criterion's statistical
sampling and baseline comparison — the "did we regress?" signal. It covers single-threaded
`search`, the `parallel_search` sweep across `query_threads` (1/2/4/8 — the reproducible
parallel-scan measurement), `ingest`, and `write_path`. criterion is a dev-dependency of
*this* crate only and never touches nidus's build.

`write_path` is the file-backed write lane, swept across fsync policy and batch size. It
exists because `ingest` **cannot** catch a write regression: `ingest` uses
`open_in_memory` and a single 10k-record call, so it touches no filesystem, takes no disk
barrier, and has no batch-size axis — every quantity `nidus-xb9.1` is meant to move is
invisible to it. `write_path` runs at `sample_size(10)` over only 200 records because a
`PerBatch` call costs a real disk barrier (~3.8ms where `sync_all` is `F_FULLFSYNC`);
criterion's defaults would make the `b1` row alone several minutes. Expect the group to
take roughly a minute and a half.

The shape to watch: `on_flush` is flat across batch size (no per-call barrier since
`nidus-4h2`), while `per_batch` scales with the number of calls. Group commit should pull
`per_batch` toward `on_flush`.
