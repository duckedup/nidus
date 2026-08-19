---
title: Reranking
description: Re-score a search's top candidates with a hosted cross-encoder for higher precision than embedding similarity alone, as an opt-in stage over vector, hybrid, and recall search.
---

Vector search ranks by comparing two independently-computed embeddings: the query's and
each candidate's. A **cross-encoder** reranker instead reads the query and one candidate
**together** in a single pass, so it can weigh interactions between them that two separate
embeddings throw away before they are ever compared. That makes it markedly more accurate
than embedding similarity alone, at the cost of one extra provider call per query and a
per-candidate scoring fee, so it is not run over the whole corpus: retrieve a wide
candidate set the ordinary way, then rerank just that set down to the page you actually
return.

Reach for it when precision matters more than latency and cost: the last mile of a search
UI, a RAG pipeline picking context for a model, or anywhere tuning fusion weights on the
same corpus has stopped paying off. If a query is latency-sensitive or runs at very high
volume, the plain vector or hybrid ranking is usually the better trade.

## Turning it on

Reranking is **off by default**, exactly like the embedders: `cargo add nidus` stays a
pure, dependency-lean sync vector store until you opt in. Enable the `rerank` feature plus
one `rerank-<name>` feature per provider you want (or `rerank-all` for both):

```toml
[dependencies]
nidus = { version = "0.67", features = ["rerank-voyage"] }
```

| Provider | Feature | Enum | Default model |
|---|---|---|---|
| Voyage | `rerank-voyage` | `RerankProvider::Voyage` | `rerank-2.5` |
| Cohere | `rerank-cohere` | `RerankProvider::Cohere` | `rerank-v3.5` |

Over HTTP and MCP, `nidus serve` takes the provider from flags instead of code:
`--rerank-provider` (`voyage` or `cohere`), `--rerank-model`, `--rerank-api-key`, and
`--rerank-base-url`, with `NIDUS_RERANK_PROVIDER` / `NIDUS_RERANK_MODEL` /
`NIDUS_RERANK_API_KEY` / `NIDUS_RERANK_BASE_URL` as their environment equivalents. Omit
`--rerank-provider` and the server still starts; a request that then asks for reranking
answers `400` naming the missing flag rather than silently returning the un-reranked order.

## The over-fetch window

A rerank call only ever sees a bounded window of candidates, not the whole corpus:
`overscan` multiplies the page depth, so a `top_k: 10` query with the default `overscan: 10`
retrieves and scores 100 candidates before cutting back down to 10. Raise it to widen the
net the cross-encoder gets to choose from; each unit costs one more document in the
provider call, so larger finds more and costs more.

## The text it reads

The reranker scores each candidate's **text**, not its vector, so it needs somewhere to
read that text from. By default it reads the `nidus.text` attr (the one `remember` always
stamps); name a different attr with `text_attr` if your records carry the text elsewhere.

A candidate missing that attr is not an error: it is passed through **unranked**, appended
after every successfully reranked hit, in its original metric order and with its original
metric score untouched. This is the behaviour most likely to surprise you if you don't
expect it, since a page that mixes text-bearing and text-less records ends up split into
two blocks rather than one interleaved ranking. Reranked scores and metric scores are on
different scales (see below), so they cannot be interleaved by score in the first place;
this is the deliberate, documented alternative to failing the whole query over one record.

## What happens to scores

A rerank score **replaces** `Hit::score`, and it is on the provider's own scale, not
cosine. `min_score` and `rank_by` are cosine-scale (or BM25-scale for a text leg) and are
evaluated **before** reranking, inside the ordinary sync search; nothing about them changes
after that. `limit_per`, on the other hand, is **re-applied after** reranking: the store
caps the over-fetched window as usual, but the cap has to run again on the reranked order
or a diversity guarantee you asked for would be silently undone by the reordering. Ties
still break on `(collection, id)`, exactly as everywhere else in nidus.

## How it composes

Reranking runs strictly **after** the store's own top-k selection, so it composes with
everything that selection can already do: ANN, quantization, and the hybrid RRF fusion all
hand it an ordinary ranked candidate set, unaware that anything downstream is going to
re-score it. Nothing about the on-disk format, an index, or a store's configuration
changes; a rerank request is a property of one query, not of the store.

## The surfaces

### Library

Build a reranker the same way you'd build an embedder, and pass it to one of the async
free functions in `nidus::rerank` alongside your (still-synchronous) `Nidus`:

```rust
use nidus::rerank::{AnyReranker, RerankConfig, RerankProvider, search_reranked};
use nidus::{Config, Nidus, RerankOpts, Scope, SearchOpts};

# async fn run(query_vector: Vec<f32>) -> anyhow::Result<()> {
let reranker = AnyReranker::build(
    RerankProvider::Voyage,
    RerankConfig::new("").api_key(std::env::var("VOYAGE_API_KEY")?),
)?;

let db = Nidus::open(Config::new("./store", 1024))?;
let opts = SearchOpts {
    top_k: 10,
    rerank: Some(RerankOpts::default()),
    ..Default::default()
};
let hits = search_reranked(
    &db, &reranker, Scope::All, &query_vector, "how do users sign in?", &opts,
).await?;
# anyhow::Ok(())
# }
```

`hybrid_reranked` is the same shape over `HybridOpts`. `Nidus` itself gains no async
method: the sync search runs first, and only the resulting candidate texts cross the
network.

### HTTP

`rerank` is an additive field on `POST /search`, `POST /hybrid-search`, and
`POST /collections/{name}/recall`; omitting it leaves the response byte-identical to
before. See the [HTTP API reference](/reference/http-api/) for the exact request shape on
each route.

### MCP

The `recall`, `text_search`, and `hybrid_search` tools each take a plain `rerank` boolean
and an optional `rerank_overscan` integer. Every one of these tools already carries the
query as text, so there is nothing extra to plumb: set `rerank: true` and the server reads
the query it already has.

## Where to next

- [Search & filters](/guides/search/): the ranking a rerank stage runs over.
- [Remember & recall](/guides/remember-and-recall/): where `nidus.text` comes from.
- [HTTP API reference](/reference/http-api/): the exact wire shape.
