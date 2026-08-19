---
title: Reranking
description: Rerank a retrieved candidate window through a hosted cross-encoder (Voyage, Cohere, or Jina) before returning top_k, for when retrieval accuracy matters more than one extra API call on the request path.
---

A plain nidus search ranks by comparing two **independently computed** vectors: the
query's and the candidate's. A cross-encoder reranker scores the query and the
candidate **together**, in one forward pass, which lets it weigh interactions between
the two that two separate embeddings can never capture. That extra accuracy is why
retrieving deep and reranking down to `top_k` reliably beats tuning fusion weights on
the same corpus: the reranker sees pairs a fusion formula never does.

The trade is cost, not accuracy: a cross-encoder call scores every candidate against
the query, so it is one more network round trip per search, on the request path. Reach
for it when precision matters more than latency; nidus's own similarity ranking is
still the cheaper default.

## What a cross-encoder does differently

An embedding model turns text into a vector once, independent of any particular
query; search then compares two of those vectors by cosine (or another metric). A
cross-encoder reranker instead takes the query and one candidate's text **together**
and returns a single relevance score for that pair, which is why it cannot be
precomputed: the score does not exist until both sides are known. It is slower per
pair for exactly that reason, so nidus uses it as a second pass over a shortlist a
cheap first pass already narrowed, never as the primary index.

## The overscan window

A reranked query ranks deeper than `top_k` before handing the window to the
provider: `(offset + top_k) * overscan` candidates by the plain metric, scored by
the cross-encoder, then trimmed back to `top_k`. The default `overscan` is **4** and the
maximum is **64**. The resulting depth is capped at 10000, the same ceiling a plain query
has, so a large overscan cannot be a way around it. A value past 64 is refused rather than
clamped, because the window size decides how many candidates reach a paid provider.
Set it higher to widen the net the reranker gets to choose from (more accurate,
more provider cost); `0` and `1` both mean no over-fetch at all.

This overscan is unrelated to two similarly-named knobs elsewhere in these docs:
see [not the quantized rerank, not the ANN overscan](#not-the-quantized-rerank-not-the-ann-overscan)
below.

## Where the text comes from

Each candidate's text is read from the `nidus.text` attr by default, the same key
[`remember`](/guides/remember-and-recall/) stamps on every write, so a memory store
reranks with no extra configuration at all. Name a different attr with `text_field`
if your records carry their text elsewhere.

A candidate whose text attr is absent, or not a string, is **not** an error: it
passes through the response unranked, keeping its plain metric score, and sorts
below every candidate the provider did score. A page can come back with a mix of
reranked and un-reranked hits when only some records have the text field.

## `min_score` is a floor before the rerank, not after

`min_score` is compared against the **metric** score (cosine, BM25, and so on)
while building the deep candidate window, before the reranker ever runs. A
cross-encoder score is not on the same scale as a similarity score, so carrying a
`min_score` floor past the rerank step would be comparing two different units.
There is no floor on the reranker's own score.

## Starting a server with a reranker

Reranking is opt-in at serve time, one provider per server, configured the same
way as the embed/summarize providers: a `--rerank-provider` flag and its matching
`NIDUS_RERANK_*` environment variables. With no `--rerank-provider`, the server
starts exactly as before and a request that sends `rerank` gets a `400` naming the
flag.

| Flag | Env | Meaning |
| --- | --- | --- |
| `--rerank-provider <NAME>` | `NIDUS_RERANK_PROVIDER` | `voyage`, `cohere`, or `jina`. Omit to leave reranking off. |
| `--rerank-model <MODEL>` | `NIDUS_RERANK_MODEL` | Defaults to the provider's own default model. |
| `--rerank-api-key <KEY>` | `NIDUS_RERANK_API_KEY` | API key for the rerank provider. |
| `--rerank-base-url <URL>` | `NIDUS_RERANK_BASE_URL` | Base-URL override, for a self-hosted gateway or a mock. |
| `--rerank-overscan <N>` | `NIDUS_RERANK_OVERSCAN` | The server-wide default overscan when a request leaves it unset. Defaults to `4`. |
| `--rerank-text-field <ATTR>` | `NIDUS_RERANK_TEXT_FIELD` | The server-wide default text attr. Defaults to `nidus.text`. |

```bash
nidus serve --dir ./store --dim 1024 \
  --rerank-provider voyage --rerank-api-key "$VOYAGE_API_KEY"
```

| Provider | Feature | Default model |
| --- | --- | --- |
| Voyage | `rerank-voyage` | `rerank-2.5` |
| Cohere | `rerank-cohere` | `rerank-v3.5` |
| Jina | `rerank-jina` | `jina-reranker-v2-base-multilingual` |

Enable one provider's feature (`rerank-voyage`, `rerank-cohere`, `rerank-jina`) or
all three at once with `rerank-all`. Prebuilt binaries (`cargo binstall nidus`, the
install script, and the `serve` feature umbrella) carry every provider already.

## A worked example

Ask for a reranked search by adding a `rerank` object to the request. On
`/search` (a raw-vector query) `rerank.query` is required, since a vector carries
no text of its own; on `/text-search`, `/hybrid-search`, and `/recall` it defaults
to that request's own query text.

### Over HTTP

```bash
curl -s localhost:7700/collections/docs/recall \
  -H 'content-type: application/json' \
  -d '{"query": "how do I rotate the signing key", "top_k": 5,
       "rerank": {"overscan": 8}}'
```

### From the JavaScript SDK

```ts
const hits = await db.recall("docs", "how do I rotate the signing key", {
  topK: 5,
  rerank: { overscan: 8 },
});
```

See the [HTTP API reference](/reference/http-api/) for the full `rerank` object
on each of the four endpoints that accept it, and the SDK pages under
[SDKs](/sdks/javascript/) for the equivalent option in each client.

## The cost

Reranking adds exactly one extra call to the provider per search: every
candidate that has usable text is sent to the cross-encoder in one request (or a
few, chunked to the provider's own per-request document limit), scored, and
merged back in. That call sits on the request's own critical path, so it adds the
provider's own latency to every reranked query, on top of whatever the plain
search already cost to rank the deeper `overscan` window.

## Not the quantized rerank, not the ANN overscan

nidus already uses the word "rerank" for something else, and the two are
unrelated mechanisms. Elsewhere in these docs:

- [Quantization](/guides/search/#quantization)'s `rescore` factor controls a
  **second exact f32 pass** over a quantized first pass's candidates, entirely
  in-process, with no network call and no provider involved.
- [Approximate search](/guides/search/#approximate-search-ann)'s `overscan`
  controls how many candidates the **ANN index walk** fetches before nidus
  applies your filter and scores the survivors exactly, also in-process.

The hosted-reranker `overscan` documented on this page is a third, distinct
knob: how deep a search ranks before handing the window to an external
cross-encoder API. Widening it does not change what the ANN index fetches or
how deep the quantized second pass reaches, and widening either of those does
not change what the reranker provider sees.
