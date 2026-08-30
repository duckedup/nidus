---
title: Hybrid search (RRF)
description: "Combine BM25 full-text and vector search into one ranking with reciprocal rank fusion, and weight the two legs."
---

Keyword search finds the exact term. Vector search finds the thing you meant.
[`hybrid_search`](/reference/api/#nidus) runs both and fuses them into one ranking
using **Reciprocal Rank Fusion**: each leg is ranked independently, and a document's
fused score is `Σ 1 / (rrf_k + rank)` over the legs it appears in.

RRF fuses *ranks*, not scores, which is what makes it safe here: a BM25 score and a
cosine similarity are not on one scale and cannot be added, but their positions in
two result lists can. See [full-text search](/guides/full-text-search/) for the BM25
leg and [vector search](/guides/search/) for the other.

```rust
use nidus::{FtsQuery, HybridOpts};

let query_vector = vec![0.1_f32; 384];
let hits = db.hybrid_search(
    "docs",
    &query_vector,                       // the vector leg
    &FtsQuery::new("body", "vector database"), // the BM25 leg
    &HybridOpts { top_k: 10, ..Default::default() },
)?;
# anyhow::Ok(())
```

RRF fuses by **rank position**, not raw score, so the incomparable scales of cosine
(or euclidean/dot-product) and unbounded BM25 never need normalizing, and a document
that surfaces in only one leg (a strong vector match with weak text, or a text-only
doc) is still ranked. `HybridOpts` exposes `top_k`, `offset` (which pages the fused
ranking, never a leg), a `filter` applied to both legs, `rrf_k` (the rank-bias constant,
default 60), and `candidates` (how deep each leg is pulled before fusing, default 100).
There is no `min_score`: a fused RRF score has no absolute scale; threshold the
individual legs via `search` / `text_search` if you need a floor.

The text leg takes the same multi-clause `FtsQuery` as `text_search`: the clauses are
combined into one BM25 leg first, then fused with the vector leg, so a single-clause hybrid
query produces exactly the numbers it always did.

### Weighting the legs

`vector_weight` and `text_weight` scale each leg's contribution, so a document scores
`Σ wᵢ / (rrf_k + rankᵢ)`. Both default to `1.0`, which reproduces the unweighted fusion
exactly.

```rust
use nidus::{FtsQuery, HybridOpts};

let query_vector = vec![0.1_f32; 384];
// Lean on the keyword leg: exact terms matter more than semantic neighbourhood here.
let hits = db.hybrid_search(
    "docs",
    &query_vector,
    &FtsQuery::new("body", "CVE-2026-1234"),
    &HybridOpts { top_k: 10, text_weight: 3.0, ..Default::default() },
)?;
# anyhow::Ok(())
```

A weight must be finite and non-negative: a `NaN` would poison the sort and a negative
weight would invert a leg rather than de-emphasize it, so both are refused.
