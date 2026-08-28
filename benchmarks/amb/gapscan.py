"""Retrieval-only gap scan against the Agent Memory Benchmark datasets.

AMB scores accuracy end to end: retrieve, then feed the context to Gemini, then judge with
a second Gemini call. That is the number on the leaderboard, and it costs real money and
real wall time per run. It also *hides* retrieval: a provider that fetches the right chunk
and a provider that fetches a plausible neighbour can land on the same judged answer.

This harness scores the retrieval step alone, against the gold document ids AMB already
ships. No LLM, no API spend, seconds per provider instead of hours. It answers the only
question worth answering before spending on judging: does nidus put the right documents in
front of the model at all, and how fast?

    uv run python gapscan.py --dataset personamem --split 32k --providers bm25,nidus-fts
    uv run python gapscan.py --dataset locomo --split locomo10 --query-limit 100 \
        --providers bm25,nidus-fts,nidus,qdrant

Run it from the AMB checkout (it imports `memory_bench`), with this directory importable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from memory_bench.dataset import get_dataset
from memory_bench.memory import get_memory_provider
from memory_bench.utils import count_tokens


def _metrics(retrieved: list[str], gold: list[str], k: int) -> dict:
    """hit@k, recall@k and reciprocal rank for one query."""
    top = retrieved[:k]
    goldset = set(gold)
    found = [i for i, d in enumerate(top) if d in goldset]
    return {
        "hit": 1.0 if found else 0.0,
        "recall": len(goldset & set(top)) / len(goldset) if goldset else 0.0,
        "rr": 1.0 / (found[0] + 1) if found else 0.0,
    }


def scan(dataset_name: str, split: str, provider_name: str, k: int, query_limit: int | None,
         doc_limit: int | None, store_root: Path) -> dict:
    dataset = get_dataset(dataset_name)
    memory = get_memory_provider(provider_name)

    queries = dataset.load_queries(split, limit=query_limit)
    if not any(q.gold_ids for q in queries):
        raise SystemExit(
            f"{dataset_name}/{split} carries no gold_ids — retrieval cannot be scored "
            "without the judge. Use `amb run` for this dataset."
        )
    queries = [q for q in queries if q.gold_ids]

    documents = None
    if dataset.isolation_unit is not None and query_limit is not None:
        user_ids = {q.user_id for q in queries if q.user_id}
        try:
            documents = dataset.load_documents(split, limit=doc_limit, user_ids=user_ids)
        except TypeError:
            # Not every AMB dataset honours the `user_ids` kwarg its own base class
            # declares (locomo does not); fall back to the whole split and let the
            # per-query user filter do the isolation.
            documents = None
    if documents is None:
        documents = dataset.load_documents(split, limit=doc_limit)

    unit_ids = None
    if dataset.isolation_unit is not None:
        unit_ids = {u for d in documents if (u := dataset.get_isolation_id(d)) is not None}

    store = store_root / dataset_name / provider_name / split
    store.mkdir(parents=True, exist_ok=True)

    memory.initialize()
    try:
        memory.prepare(store, unit_ids=unit_ids, reset=True)

        t0 = time.perf_counter()
        memory.ingest(documents)
        ingest_s = time.perf_counter() - t0

        per_query, latencies, ctx_tokens = [], [], []
        for q in queries:
            t = time.perf_counter()
            docs, _ = memory.retrieve(q.query, k=k, user_id=q.user_id)
            latencies.append((time.perf_counter() - t) * 1000)
            ctx_tokens.append(sum(count_tokens(d.content) for d in docs))
            # de-duplicate parent ids, keeping rank order
            seen, ranked = set(), []
            for d in docs:
                if d.id not in seen:
                    seen.add(d.id)
                    ranked.append(d.id)
            per_query.append(_metrics(ranked, q.gold_ids, k))
    finally:
        memory.cleanup()

    total_tokens = sum(count_tokens(d.content) for d in documents)
    return {
        "dataset": dataset_name,
        "split": split,
        "provider": provider_name,
        "k": k,
        "queries": len(queries),
        "documents": len(documents),
        "corpus_tokens": total_tokens,
        "ingest_s": round(ingest_s, 2),
        "ingest_docs_per_s": round(len(documents) / ingest_s, 1) if ingest_s else 0.0,
        "hit_at_k": round(statistics.mean(m["hit"] for m in per_query), 4),
        "recall_at_k": round(statistics.mean(m["recall"] for m in per_query), 4),
        "mrr": round(statistics.mean(m["rr"] for m in per_query), 4),
        "retrieve_p50_ms": round(statistics.median(latencies), 1),
        "retrieve_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1),
        "avg_context_tokens": round(statistics.mean(ctx_tokens), 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--providers", default="bm25,nidus-fts", help="comma-separated AMB provider names")
    ap.add_argument("-k", type=int, default=50, help="retrieval depth (default 50, the qdrant baseline's)")
    ap.add_argument("--query-limit", type=int, default=None)
    ap.add_argument("--doc-limit", type=int, default=None)
    ap.add_argument("--store", default="gapscan-store", help="scratch directory for provider stores")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    args = ap.parse_args()

    rows = []
    for name in [p.strip() for p in args.providers.split(",") if p.strip()]:
        print(f"── {name} ─────────────────────────────────", flush=True)
        try:
            row = scan(args.dataset, args.split, name, args.k, args.query_limit,
                       args.doc_limit, Path(args.store))
        except Exception as exc:  # a broken provider must not sink the whole scan
            print(f"   FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            rows.append({"provider": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    print()
    hdr = f"{'provider':<16}{'hit@k':>8}{'recall@k':>10}{'MRR':>8}{'ingest_s':>10}{'docs/s':>9}{'p50_ms':>9}{'p95_ms':>9}{'ctx_tok':>9}"
    print(hdr)
    print("─" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['provider']:<16}{'— ' + r['error'][:60]}")
            continue
        print(f"{r['provider']:<16}{r['hit_at_k']:>8.3f}{r['recall_at_k']:>10.3f}{r['mrr']:>8.3f}"
              f"{r['ingest_s']:>10.1f}{r['ingest_docs_per_s']:>9.1f}"
              f"{r['retrieve_p50_ms']:>9.1f}{r['retrieve_p95_ms']:>9.1f}{r['avg_context_tokens']:>9.0f}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
