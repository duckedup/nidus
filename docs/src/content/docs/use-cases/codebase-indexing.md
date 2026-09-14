---
title: Search a codebase
description: Chunk a repository AST-aware and answer with a file and a line range, not a copy of the source, so a coding assistant opens the real file for ground truth.
---

A coding assistant has a hard limit on what it can hold in context, and it blows past
that limit on any real codebase. Point nidus at the repo instead: it chunks by
function, struct and trait rather than by character count, and every hit comes back
as a symbol with the lines it lives on, never the body itself.

## What you need

- [Code search](/guides/code-search/): AST-aware chunking, one chunk per symbol,
  results grouped by file.
- [MCP](/guides/mcp/): one line connects the store to Claude Code, Cursor, or
  anything else that speaks the protocol, with no glue code.
- [Full-text search (BM25)](/guides/full-text-search/): the run below never called
  an embedding provider; BM25 keyword matching over the chunked text is what
  answered it.

## Doing it

**read the repo, then ask it something**

```sh
nidus code ingest . --dir ./store

nidus code search "where do we fsync the write ahead log" \
  --dir ./store
```

```json
[
  { "path": "store/write.rs", "language": "rust",
    "symbols": [
      { "symbol": "commit", "kind": "function",
        "start_line": 136, "end_line": 148, "score": 10.09 },
      { "symbol": "maybe_sync", "kind": "function",
        "start_line": 109, "end_line": 119, "score": 7.75 }
    ] }
]
```

138 files became 3,403 symbols in 1.7 seconds, with no API key, and it found the two
functions that do the work.

**hand it to your assistant**

```sh
claude mcp add nidus -- nidus mcp --dir ~/.nidus \
  --embed-provider voyage --embed-model voyage-4
```

That is the entire integration. No SDK to install, no glue code to write.

## What to tune

- [Adding an embedder](/guides/code-search/#add-an-embedder): search by meaning
  instead of exact keywords once a provider is configured.
- [Summarize-then-embed](/guides/code-search/#summarize-then-embed): embed what a
  symbol means rather than its literal tokens, for a query that never spells the
  words the code itself uses.

## Where it stops

Results are symbols and line spans, never source bodies: the assistant always opens
the file for the real text. `code` is part of the default build;
`--no-default-features` excludes it along with the rest of the ingest layer.
