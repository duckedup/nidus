---
title: Code search
description: Index a codebase and its docs together, chunked for what each file actually is, and search it grouped by file and symbol. Off by default, enabled with the code feature.
---

`nidus code` indexes source and documentation in one corpus, each chunked for what it
is: one chunk per function/struct/trait/… for a recognised language, heading-aware for
markdown, and nidus's own generic splitter for everything else. Search comes back
grouped by file, each hit carrying a symbol name, kind and line span, never the source
body itself: read the file at that line span for ground truth.

It ships behind the off-by-default `code` feature, so `cargo add nidus` is unaffected
whether or not you ever touch this page. See [why it is off by default](#why-off-by-default)
before turning it on.

## Install with the feature on

```bash
cargo binstall nidus --features code
# …or: cargo install nidus --features code
```

`code` needs `memory` underneath it (the walk/digest/embed pipeline `code ingest` and
`code search` are front doors over), so either enable both or reach for the `serve`
umbrella, which already includes `memory`:

```bash
cargo install nidus --features serve,code
```

## Index a repo, no provider

The no-provider path needs no API key and no network call: every query answers from
BM25 keyword matching over the chunked text.

```bash
nidus code ingest . --dir ./store
```

Dot-entries are walked by default (`.github`, `.claude`, …), because a repo scan that
skipped its own config and docs would miss half the corpus; `.git` is always skipped,
at any depth, regardless. Symlinks are skipped and non-UTF-8 files are counted and
skipped rather than failing the run, the same as a plain `nidus ingest`.

## Search it

```bash
nidus code search "where do we release commission payments" --dir ./store
```

```json
[
  {
    "path": "internal/finance/commission/release.go",
    "symbols": [
      { "symbol": "ReleaseCommission", "kind": "function",
        "start_line": 42, "end_line": 78, "score": 8.31 }
    ]
  }
]
```

That is the whole output shape: `path`, and one entry per matching symbol with its
`kind`, its `start_line`/`end_line`, and its score. There is no source body in the
response on purpose. A code hit is a pointer, not a quote: the agent reading the
result opens the file at those lines for the real, current text, rather than trusting
a copy that chunking or a stale index could have gotten wrong.

## Add an embedder

Pass an embedding provider to search by meaning instead of exact keywords, the same
flags `nidus ingest` takes:

```bash
nidus code ingest . --dir ./store --embed-provider voyage
nidus code search "where do we release commission payments" \
  --dir ./store --embed-provider voyage
```

With an embedder configured, a query embeds and searches by vector; a store ingested
with no embedder (dimension 0) falls back to BM25 automatically. `--vector` on
`code search` forces the vector leg and surfaces the refusal by name, rather than
silently falling back, when no embedder is available or the store holds no vectors.

## Summarize-then-embed

The code engine carries `wdpkr-core`'s own code-summarization prompts (one for a
whole file, one per symbol), built for embedding what code *means* rather than its
literal tokens, which is what closes a conceptual query like "where do we release
commission payments" onto a function that never spells any of those words. Pass `--summarize` with a summarize provider and `code ingest` embeds each symbol's
summary instead of its body. The body is still stored and still BM25-searchable; the
summary sits beside it under `nidus.summary`.

```bash
nidus code ingest . --dir ./index \
  --embed-provider voyage --embed-api-key "$VOYAGE_API_KEY" \
  --summarize --summarize-provider anthropic --summarize-api-key "$ANTHROPIC_API_KEY" \
  --summarize-budget 500
```

It costs one model call per file plus one per symbol, so a whole repo is thousands of
calls. `--summarize-budget` is the ceiling (500 by default). A file whose remaining
budget cannot cover it is embedded raw rather than half-summarized, and the report
counts those under `summarize.files_over_budget`, so a truncated run says so instead of
leaving two kinds of vector in one corpus with nothing recording which is which.

The prompts are also exposed as library-level `SummarizeOpts` builders
(`feature = "summarize"`) for a custom ingest pipeline built on `Nidus` directly.

## Known limitation: exported TypeScript classes

An exported class (`export class Foo { ... }`) currently chunks as one chunk for the
whole export rather than one per method. A plain `class Foo { ... }` chunks per method,
as do Python, Java and C# classes. The cause is upstream, in how `wdpkr-core` walks an
export statement ([wdpkr-core#7](https://github.com/duckedup/wdpkr-core/issues/7)), so it
is fixed there rather than worked around here.

## Why off by default

`nidus code` depends on [`wdpkr-core`](https://crates.io/crates/wdpkr-core) for its
tree-sitter AST chunking across eight languages. That dependency sits entirely behind
the `code` feature, so a plain `cargo add nidus` never sees it: `cargo add` stays the
few-hundred-millisecond build it always was.

The reason it is a feature and not the default is measured, not a guess (see
[D0014](https://github.com/duckedup/nidus/blob/main/decisions/0014-nidus-depends-on-wdpkr-core-behind-code.md)
for the full record). Clean, offline, debug builds, one machine:

| lane | wall | CPU |
| --- | ---: | ---: |
| default features | 13.4s | 50.5s |
| `serve,embed-all,summarize-all,memory,mcp` | 24.8s | 96.3s |
| the same, plus `wdpkr-core` | 26.3s | 122.1s |
| tree-sitter plus the eight grammars alone | 2.6s | 9.3s |
| `wdpkr-core` alone | 13.1s | 68.9s |

The eight grammars are not the expensive part: 2.48M lines of generated, table-driven
C that costs 9.3s CPU at `-O0`. What costs is `wdpkr-core`'s own Rust dependency
graph, which arrives whole (it ships no cargo features of its own) the moment
anything pulls it in. A dedicated `build-budget-code` CI job holds the `code` lane to
its own 60s-class bound, the same way the default lane is held to its 60s bound,
without moving either bound onto the other.
