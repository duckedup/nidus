---
title: Performance
description: Exact brute-force cosine KNN benchmarked against DuckDB and LanceDB. nidus is the fastest in every cell, at 100% recall.
---

Every vector store ships a benchmark proving it's the fastest, on synthetic data
that looks nothing like your workload. It's a genre. Here's ours, and yes, we
win our own benchmark, that's how this works.

## The numbers

Exact brute-force cosine KNN, 100k vectors, single thread, measured against
DuckDB (`array_cosine_similarity`) and LanceDB (`bypass_vector_index`), both
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
the one with a pure-Rust core and no bundled C++ tree.

### Text search, hybrid, and rank_by

Single-threaded, same machine as above: 10,000 documents at dim 384, each 32 tokens
drawn from a 300-term Zipf-weighted vocabulary, one FTS field on the default
analyzer.

| bench                                      |    time |
| ------------------------------------------- | ------: |
| text_search, single hot term                | 1.08 ms |
| text_search, multi-term                     | 1.13 ms |
| hybrid, candidates=50                       | 1.82 ms |
| hybrid, candidates=100                      | 1.89 ms |
| hybrid, candidates=400                      | 2.38 ms |
| rank_by recency decay, without (baseline)   |  530 µs |
| rank_by recency decay, with                 | 1.26 ms |

Term count barely moves the text-search cost: a multi-term query is only marginally
dearer than a single hot term. Hybrid's `candidates` sweep is the actionable knob:
going 8× deeper (50 → 400) costs about 1.3×, so buying fusion recall is cheap.
`rank_by`'s recency-decay expression is the expensive lever here, roughly
**2.4×**ing a single-threaded vector search at this size (530 µs → 1.26 ms); read
it as that with/without delta rather than an absolute, and note it is
single-threaded (`query_threads` unset).

## Retrieval quality

Speed numbers say nothing about whether the results are any good. This section measures
that directly: nDCG@10 and Recall@100 over three BEIR datasets (SciFact, NFCorpus,
FiQA-2018), for four legs (FTS only, vector only, RRF fusion at the shipped defaults, and
fusion plus rerank). Fusion weights and analyzer choices were previously tuned by feel;
this gives them a number.

**Methodology**, so the table below is reproducible from this page alone:

- Embeddings: Voyage `voyage-4` (1024 dimensions). Rerank: Voyage `rerank-2.5`.
- Queries are embedded through the same path as documents, so both share the disk cache.
  That means query vectors are document-tagged, not the query-tagged vectors Voyage's own
  published numbers were measured with, which costs a little nDCG relative to those
  numbers: a reader comparing against Voyage's published figures needs this caveat. The
  size of that gap is not yet measured (`nidus-ocdm`).
- Fusion runs at the shipped defaults: `rrf_k` 60, `candidates` 100, both weights 1.0.
- Document text is the BEIR title and body concatenated.
- A document counts as relevant for Recall@100 when its qrel score is above zero; nDCG
  uses the graded qrel score directly.
- One store per dataset, one collection carrying both the FTS index and the vectors.
- The rerank leg reranks the 100 fusion candidates, not a widened pool. Its Recall@100 is
  therefore identical to the fusion leg's by construction, since reordering 100 candidates
  cannot change which 100 they are. Only its nDCG@10 carries information.

The BEIR paper's published BM25 nDCG@10 is shown per dataset for scale, not as a
reproduction: BEIR's baseline runs Elasticsearch's default analysis, while nidus uses k1
1.2 and b 0.75 over a Porter English analyzer. Treat the comparison as directional.

**SciFact**

| leg               | nDCG@10 | Recall@100 |
| ------------------ | :-----: | :--------: |
| BM25 (BEIR paper)  |  0.665  |    n/a     |
| FTS only           |  0.691  |   0.931    |
| vector only        |  0.737  |   0.967    |
| fusion (defaults)  |  0.761  |   0.973    |
| fusion + rerank    |  0.809  |   0.973    |

**NFCorpus**

| leg               | nDCG@10 | Recall@100 |
| ------------------ | :-----: | :--------: |
| BM25 (BEIR paper)  |  0.325  |    n/a     |
| FTS only           |  0.329  |   0.249    |
| vector only        |  0.316  |   0.351    |
| fusion (defaults)  |  0.384  |   0.351    |
| fusion + rerank    |  0.432  |   0.351    |

**FiQA-2018**

| leg               | nDCG@10 | Recall@100 |
| ------------------ | :-----: | :--------: |
| BM25 (BEIR paper)  |  0.236  |    n/a     |
| FTS only           |  0.253  |   0.561    |
| vector only        |  0.288  |   0.699    |
| fusion (defaults)  |  0.393  |   0.705    |
| fusion + rerank    |  0.545  |   0.705    |

Two things to read off these tables. nidus's FTS leg lands at or slightly above the BEIR
paper's published BM25 on all three datasets, which is a useful independent check: a different
engine, scored on the same judgements, agrees to within a few points, so the analyzer and the
scoring are behaving as BM25 should. And fusion beats both single legs everywhere, which is
what RRF is for, with rerank adding the largest jump on FiQA (0.393 to 0.545).

**What this does not prove.** This lane is not CI verified: it needs network and a paid
API key, so these numbers are a recorded run, not a continuously enforced claim
(`nidus-yq9p.7` will add floors against them). And three small datasets in English are not
a general retrieval-quality claim; they say fusion beats either leg alone on these corpora,
nothing broader.

## Why it's fast

The scoring kernel is plain safe Rust the optimizer can vectorize:

- An **8-lane chunked dot product** the compiler auto-vectorizes to SIMD.
- An **allocation-free top-k scan** backed by a bounded heap.
- A **storage-order, prefetcher-friendly sweep** of the row-major matrix.

No FFI boundary to cross, no columnar decode, no query planner: just a tight
loop over a contiguous `f32` matrix that is already resident in RAM.

## Optional speed levers

Two opt-in knobs trade a little for more speed when the exact single-threaded
sweep isn't enough. Both stay pure-safe-Rust and are off by default:

- **[int8 quantization](/guides/vector-search/#quantization)**: a two-pass
  search (int8 first-pass → f32 rerank) returns essentially the exact neighbours
  (**~100% recall@10 at `rescore` ≥ 2**) for a **~1.4× speedup** at 1M × 768, at
  the cost of ~25% more RAM. Reproduce: `just bench-quant`.
- **[parallel scan](/guides/integrating/#two-kinds-of-parallelism)**
  (`Config::query_threads`): splits one large search across worker threads. The
  plain **f32** scan is memory-bandwidth-bound, so its gain is sublinear and plateaus
  early (**~1.3–1.4×** at 4–8 threads). Combined with **int8 quantization**, threads
  pay off: the int8 first pass moves 4× fewer bytes, so it is compute- not
  bandwidth-bound and scales to **~2.4×** at 4 threads. Reproduce:
  `just bench-crit parallel_search` (the `parallel_search_quant` group is the
  quantized sweep).
- **[approximate index (HNSW / IVF)](/guides/vector-search/#approximate-search-ann)**
  (`Config::ann`): walks an index instead of scanning every vector, for when the
  collection outgrows a full scan. On realistic clustered data (n=20k, dim=768) HNSW
  returns **~0.99–1.0 recall@10 at ~7–10× the query speed** of the exact scan; IVF
  ~1.0 recall at ~3×. The graph is in-RAM but [persisted](/guides/vector-search/#approximate-search-ann)
  so a warm `open()` is **~0.05 s** instead of rebuilding (~36 s here). `nidus serve`
  persists the cache when it stops cleanly, so a restart is warm without any
  out-of-band call. A cold
  rebuild parallelizes over `query_threads` (**~36 s → ~5 s at 8 threads**). Recall is
  data-dependent (uniform-random vectors are a near-worst case), so measure your own:
  `just bench-ann clustered=1`.

Neither int8 nor threads is the headline multiplier its theory suggests: the 4× from int8 and the
linear scaling from threads both want SIMD/bandwidth headroom nidus doesn't chase
within its zero-FFI design. The f32 scan is bandwidth-bound; threads help most when
paired with the int8 first pass, which has the compute headroom to scale. They're
honest latency wins for the right workload, measured by benchmarks you can run
yourself.

### `persist_index` is manual, not automatic

`persist_index` is the only writer of the on-disk ANN and FTS caches, and it is
strictly out-of-band: it is **never** called from `upsert` or `flush`. It fires
only from `compact()` (best-effort, so a persist failure never fails the
compaction), the public `Nidus::persist_index`, and `nidus serve`/`nidus mcp`'s
clean-shutdown path, which persists right after a final flush. A long-running
writer that never compacts and never calls it pays a full index rebuild on the
next `open`. Per-segment IVF indexes are never persisted at all and rebuild on
every open, a real startup cost at scale. See [index cache
lifecycle](/guides/how-it-works/#index-cache-lifecycle) for the full watermark
and staleness contract.

## The target regime

nidus is tuned for **exact** search at the scale where a full scan wins. Scan cost scales
with the rows scanned, so where a full sweep is cheap, 100% recall with no index to build
or tune beats an approximate index. Exact search is the default, so you never pay for an
index you don't need.

Past that scale, an [approximate index](/guides/vector-search/#approximate-search-ann)
(HNSW or IVF, via `Config::ann`) is available as an opt-in: it trades some recall
for a smaller candidate walk instead of a full scan. It is pure-Rust, optional, and
additive over the same append-only file; exact search is unchanged when it is off.
The benchmark above measures the exact path; run `just bench-ann` to sweep the
approximate variants' recall and latency on your own shapes.

## Benchmarks vs. `nidus tune`

The numbers on this page and `nidus tune` (see [Measuring recall against your own
data](/guides/vector-search/#measuring-recall-against-your-own-data)) measure two
different things, and it is easy to conflate them:

- **`benchmarks/`** measures **throughput on synthetic data** across engines
  (nidus, DuckDB, LanceDB), on shapes chosen to be reproducible and comparable.
  It answers "how fast is nidus in general, relative to the alternatives."
- **`nidus tune`** measures **recall and latency against your own store**: it
  samples your actual vectors, not a synthetic distribution, and reports what an
  `ef_search`/`n_probe`/`overscan` setting costs and buys *for that data*. It
  answers "what should I set these to, here."

Use the benchmarks to decide whether nidus fits at all; use `tune` once it does,
to pick settings for the collection you actually have.

## Reproduce it

```bash
just bench all                  # cross-engine parity table (nidus vs DuckDB vs LanceDB)
just bench-quant                # int8 quantization recall & speed sweep
just bench-crit parallel_search # query_threads scaling (criterion)
cargo bench -p nidus-bench --bench nidus_regression -- 'text_search|hybrid|rank_by'
                                 # text search, hybrid, and rank_by (criterion)
just bench-retrieval            # BEIR retrieval quality (needs VOYAGE_API_KEY and network)
```

The heavy DuckDB/LanceDB dependencies are **quarantined off nidus's own build
path** (in `benchmarks/`), so they never touch the build of nidus itself.
Synthetic data on an Apple Silicon laptop: useless, like all benchmarks, but
there it is.
