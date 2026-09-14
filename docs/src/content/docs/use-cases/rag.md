---
title: Ground a model in your own documents
description: Index a directory once, ask in plain language, and get back the passages that answer it and the file each came from. Any model, any framework.
---

Retrieval is the hard half of a RAG pipeline, and it is the half nidus is. Index a
directory once, ask a question in plain language, and get back the few passages
that answer it, each carrying the file it came from. What comes back is JSON from a
command or an HTTP call, so it works with any model and any framework: nothing here
is tied to one.

## What you need

- [Ingest a directory](/guides/ingest/): walk, chunk, embed and store a tree in one
  command, re-runnable for free.
- [Remember & recall](/guides/remember-and-recall/): the same store, for text that
  is not files on disk.
- [Hybrid search (RRF)](/guides/hybrid-search/): fuse keyword and vector legs into
  one ranking, so an exact term and a paraphrase both land.
- [Reranking](/guides/rerank/): a cross-encoder pass over the retrieved set, for
  when precision on the last mile matters more than latency.
- [HTTP server](/guides/http-server/): the same pipeline over JSON, for a client
  that never links the crate.

## Doing it

```sh
nidus ingest ./docs \
  --collection docs \
  --glob '**/*.md' \
  --dir ./store \
  --embed-provider voyage
```

```sh
nidus recall docs "how does compaction work" --dir ./store --embed-provider voyage
```

Over HTTP, the same query is a POST:

```sh
curl -s localhost:7700/collections/docs/recall \
  -H 'content-type: application/json' \
  -d '{"query": "how does compaction work", "rollup": {"neighbours": 1}}'
```

## What to tune

- [`--rollup` and `--neighbours`](/guides/ingest/#reading-a-chunked-corpus-back):
  one readable passage per document instead of three overlapping fragments.
- [Weighting the legs](/guides/hybrid-search/#weighting-the-legs): favour the exact
  term or the paraphrase depending on the corpus.
- [Turning on reranking](/guides/rerank/#turning-it-on): re-score the retrieved set
  before it reaches the model.

## Where it stops

nidus retrieves; it does not generate. You bring the model, and you decide what to
do with what comes back.
