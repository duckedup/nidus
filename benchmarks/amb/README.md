# nidus on the Agent Memory Benchmark

Tooling to run nidus against [AMB](https://github.com/vectorize-io/agent-memory-benchmark)
(leaderboard: [agentmemorybenchmark.ai](https://agentmemorybenchmark.ai)), find where nidus
falls short, and track the gap closing. Epic: `nidus-7d5`.

We do not open PRs against AMB. Everything nidus-side lives here and is symlinked into an
AMB checkout.

## What is here

| File | What it is |
|---|---|
| `nidus_provider.py` | Three AMB `MemoryProvider` implementations, each driving a real `nidus serve` subprocess over the published Python SDK |
| `nidus_embed.py` | Client-side batch embedding. Exists only as a workaround for `nidus-7d5.1`; delete it when that lands |
| `gapscan.py` | Retrieval-only scorer: recall@k against AMB's gold document ids, no LLM, no API spend |
| `results-*.json` | Recorded gap-scan runs |

### The three providers

| Name | What it exercises | Needs a key |
|---|---|---|
| `nidus-fts` | BM25 only, over a dimension-0 store (no vectors at all). `POST /text-search` | no |
| `nidus` | Dense + BM25 hybrid RRF via `POST /hybrid-search`. Vectors embedded client-side in batches | yes |
| `nidus-memory` | The all-in-one memory layer end to end: `POST /remember` per document, `POST /recall` per query, embedding server-side | yes |

`nidus-fts` is the one that runs anywhere, which is why every number below is BM25 only.

## Why a retrieval-only scorer

AMB scores accuracy end to end: retrieve, answer with Gemini, judge with a second Gemini
call. That is the leaderboard number, and it costs real money and hours of wall time per
run. It also hides retrieval, because a provider that fetches the right chunk and one that
fetches a plausible neighbour can land on the same judged answer.

`gapscan.py` scores the retrieval step alone, against the gold document ids AMB already
ships. No LLM, seconds per provider. It answers the question worth answering before
spending on judging: does nidus put the right documents in front of the model at all, and
how fast? Chunking is AMB's own 512-token `cl100k` chunker on both sides, so the comparison
is about retrieval and not about chunk boundaries.

## Results

Retrieval-only, `k=10` unless noted, nidus BM25 vs AMB's `bm25` baseline (`rank_bm25`).

| dataset | provider | hit@k | recall@k | MRR | ingest docs/s | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| longmemeval/s, 500q, 23867 docs | bm25 | 0.923 | 0.862 | 0.806 | 1556 | 12.4 | 13.2 |
| longmemeval/s | **nidus-fts** | **0.979** | **0.950** | **0.907** | 873 | 13.8 | 29.4 |
| locomo/locomo10, 200q | bm25 | 0.873 | 0.814 | 0.550 | 2064 | 2.0 | 2.1 |
| locomo/locomo10 | **nidus-fts** | **0.985** | **0.945** | **0.822** | 1793 | **0.3** | **0.4** |
| personamem/32k, 200q, k=3 | bm25 | 1.000 | 0.451 | 1.000 | 630 | 4.8 | 5.7 |
| personamem/32k, k=3 | **nidus-fts** | 1.000 | 0.458 | 1.000 | 395 | **0.8** | **0.9** |

nidus's BM25 beats `rank_bm25` on retrieval quality on every scorable dataset, hardest on
MRR: 0.822 vs 0.550 on locomo means nidus puts the gold session near the top of the list
rather than merely somewhere in it.

Two things it does not win. Ingest throughput is roughly half `rank_bm25`'s, which is what
an HTTP store costs against an in-process index build. And BM25 tail latency degrades with
corpus size: at 1/5 of longmemeval-s nidus was 7x faster (1.5 / 3.0 ms) than `rank_bm25`,
and at full scale p50 is a wash while p95 is 2.2x worse. That is `nidus-7d5.5`.

### Not measured

The dense and hybrid legs (`nidus-7d5.6`), which is the half that competes with AMB's
qdrant baseline and with hindsight. And three of the six datasets (`nidus-7d5.8`):
lifebench's loader 404s upstream, beam's gold ids are conversation ids rather than document
ids so retrieval-only scoring does not apply, and sdebench needs a real coding-agent
harness.

### The numbers to beat

AMB's published end-to-end accuracy, `rag` mode, gemini-3.1-pro-preview answering and
judging:

| dataset | split | qdrant (hybrid-search) | hindsight | cognee |
|---|---|---|---|---|
| personamem | 32k | 0.844 | 0.866 | 0.818 |
| locomo | locomo10 | 0.791 | 0.920 | 0.803 |
| longmemeval | s | 0.740 | 0.946 | |
| lifebench | en | 0.610 | 0.715 | |
| beam | 100k / 500k / 1m / 10m | | 0.734 / 0.711 / 0.739 / 0.641 | |

## Reproducing

Needs `uv` and a release build of nidus.

```bash
cargo build --release

git clone https://github.com/vectorize-io/agent-memory-benchmark.git amb
cd amb
uv python pin 3.13     # onnxruntime has no 3.14 wheel
uv sync
uv pip install -e /path/to/nidus/sdks/python

# wire this directory in
ln -sf /path/to/nidus/benchmarks/amb/nidus_provider.py src/memory_bench/memory/
ln -sf /path/to/nidus/benchmarks/amb/nidus_embed.py    src/memory_bench/memory/
ln -sf /path/to/nidus/benchmarks/amb/gapscan.py        .
```

Then register the providers in `src/memory_bench/memory/__init__.py`:

```python
from .nidus_provider import NidusFtsProvider, NidusProvider, NidusServerMemoryProvider

REGISTRY["nidus-fts"] = NidusFtsProvider
REGISTRY["nidus"] = NidusProvider
REGISTRY["nidus-memory"] = NidusServerMemoryProvider
```

Retrieval-only, no keys:

```bash
export NIDUS_BIN=/path/to/nidus/target/release/nidus
uv run python gapscan.py --dataset longmemeval --split s --providers bm25,nidus-fts -k 10
uv run python gapscan.py --dataset locomo --split locomo10 --providers bm25,nidus-fts -k 10 --query-limit 200
uv run python gapscan.py --dataset personamem --split 32k --providers bm25,nidus-fts -k 3 --query-limit 200
```

With a key, adding the dense and hybrid legs and AMB's qdrant baseline:

```bash
export VOYAGE_API_KEY=...
uv run python gapscan.py --dataset locomo --split locomo10 \
    --providers bm25,nidus-fts,nidus,qdrant -k 10 --query-limit 200
```

The full judged run, which needs `GEMINI_API_KEY` in `amb/.env` and costs money:

```bash
uv run amb run --dataset personamem --domain 32k --memory nidus
uv run amb view      # browse results
```

### Configuration

`nidus_provider.py` reads environment only, since AMB providers take no constructor args.

| Variable | Default | What it does |
|---|---|---|
| `NIDUS_BIN` | `nidus` on PATH | which binary to serve |
| `NIDUS_EMBED_PROVIDER` | `voyage` | voyage, openai, gemini, ollama, cohere, jina, mistral |
| `NIDUS_EMBED_MODEL` | provider default | |
| `NIDUS_EMBED_API_KEY` | `VOYAGE_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | |
| `NIDUS_EMBED_DIMENSION` | native | Voyage Matryoshka, OpenAI v3 |
| `NIDUS_RERANK_PROVIDER` | off | voyage or cohere, enables the cross-encoder stage |
| `NIDUS_ANN` | off (exact) | `hnsw` or `ivf` |
| `NIDUS_TOP_K` | 50 | retrieval depth, matching the qdrant baseline |

A provider whose native embedding width is not in `_NATIVE_DIMS` needs
`NIDUS_EMBED_DIMENSION` set, because the store's dimension is pinned at creation and so has
to be known before the first embed call.
