---
title: Ingest a directory
description: One command from a folder of files to a searchable corpus. nidus ingest walks a tree, chunks each file, embeds the chunks and upserts them, and a re-run over an unchanged tree costs nothing.
---

`nidus ingest` is the whole pipeline in one command: walk a directory, split each
file into chunks, embed the chunks with the provider you choose, and store them.

Without it, everyone writes this script themselves, and most write it slightly
wrong: no dedupe, no resume, and every run re-embeds the entire corpus.

Want keyword search and nothing else? `--fts-only` runs the same pipeline with no
embedding provider at all: no API key, no network call, works offline and in CI.
Jump to [Keyword-only, with no provider](#keyword-only-with-no-provider).

## The one command

```bash
nidus ingest ./docs \
  --collection docs \
  --glob '**/*.md' \
  --dir ./store \
  --embed-provider voyage
```

Then search it:

```bash
nidus recall docs "how does compaction work" --dir ./store --embed-provider voyage
```

That is the whole story. The collection is created on first write, its full-text
index is provisioned for you, and the records carry the original text so `recall`
gives you back something readable.

## Re-running costs nothing

This is the part that makes `nidus ingest` safe to put in a file watcher, a
git hook, or a CI job. Run it twice over an unchanged tree and the second run
makes **no embedding calls and no writes**:

```json
{ "matched": 340, "ingested": 0, "unchanged": 340, "chunks": 0,
  "cache": { "hits": 0, "misses": 0, "evicted": 0 } }
```

Two independent mechanisms do that, and it is worth knowing which is which
because they cover different cases.

**A per-file digest** decides whether a file needs any work at all. It is stored
on the file's own records as `nidus.source_hash`, and it covers more than the
text: the chunking strategy, `--max-chars`, `--overlap-chars`, the embedder
identity, and the dimension. So an unchanged file is skipped whole (not read
past the hash, not chunked, not embedded, not written), while changing
`--max-chars` or switching models correctly re-ingests. Serving vectors produced
under one set of options beside vectors produced under another would be worse
than paying to redo them.

A matching digest is not on its own enough to skip a file, because the digest
lives on the file's first chunk and that chunk is written first. If a run is
killed partway through a long document, the first chunk is on disk and the rest
are not. So the check also confirms the file's last chunk is present, using the
chunk count stored alongside the digest. A half-written document re-ingests
instead of looking finished forever.

**A content-hash embedding cache** covers the files that did change. Edit one
paragraph of a twenty-chunk document and only the chunks that actually differ go
to the provider. The chunks above your edit are byte-identical, and the ones
below it have merely shifted position, so the cache answers for all of them:

```json
{ "ingested": 1, "unchanged": 339,
  "cache": { "hits": 19, "misses": 1, "evicted": 0 } }
```

The cache is a sidecar object alongside the store's other files, keyed by the
model and dimension it was built with, so a model change invalidates it rather
than silently serving vectors from the wrong model. It is never the source of
truth: if it is missing or damaged, the worst case is that you pay to embed
again. It holds `--cache-max-entries` vectors (50,000 by default) and evicts the
oldest beyond that. `--no-cache` turns it off completely, and the per-file digest keeps
working either way.

## Record ids, so a re-ingest replaces

Each chunk is stored as `<path>#<chunk-index>`, where the path is relative to the
directory you pointed at. `docs/guides/search.md` chunk 3 becomes
`guides/search.md#3`.

That makes a re-ingest a replacement rather than a duplication, and it means a
file that got shorter has its leftover high-index chunks removed. Every chunk
also carries `nidus.parent_id` (the path) and `nidus.chunk_index`, so you can
filter to one file or pull a chunk's neighbours.

## Removing files that are gone: `--prune`

Deleting a file from disk does not remove its records. Pass `--prune` to do that:

```bash
nidus ingest ./docs --collection docs --glob '**/*.md' --dir ./store \
  --embed-provider voyage --prune
```

It is opt-in on purpose. Pointing `ingest` at a partial tree (a half-finished
checkout, a filtered copy, the wrong directory) must not empty your collection,
and an automatic delete would do exactly that.

`--prune` only ever removes records that this command wrote, which it can tell
because they carry `nidus.source_hash`. Facts you added by hand with
`nidus remember` into the same collection are left alone.

One real limitation to know about: prune decides what is stale by comparing
against the files it just walked, so if you ingest two different source
directories into one collection, each run's `--prune` will delete the other's
records. Give each source root its own collection.

Reach for `--dry-run` first if you are unsure. It reports what would happen and
touches nothing, making no embedding calls and no writes.

## Choosing what gets ingested

`--glob` is matched against each path relative to the directory you named. nidus
uses SQL GLOB semantics, where `*` crosses `/`, so `*.md` is already recursive:

```bash
--glob '*.md'          # every .md at any depth
--glob '**/*.md'       # the same thing, if you prefer this spelling
--glob 'guides/*.md'   # only under guides/
```

Three things are skipped without being asked:

- **Entries whose name starts with a dot**, so `nidus ingest .` does not walk
  `.git` or `.venv`.
- **Symlinks**, so a link pointing back up the tree cannot make the walk loop
  forever.
- **Files that are not valid UTF-8.** A stray binary file in the tree is counted
  as `skipped_non_utf8` and reported, not treated as a failure. One PNG should
  not abort an ingest of nine hundred documents.

Empty and whitespace-only files produce no chunks, so they are counted as
`skipped_empty` rather than ingested.

## Chunking

Sizes are in **characters, not tokens**. nidus does not tokenize for a model it
does not own, and a character budget with a sensible margin is honest about
that.

| Flag | Default | Description |
| --- | --- | --- |
| `--strategy` | `recursive` | `recursive`, `markdown`, or `sentence`. |
| `--max-chars` | `1000` | Chunk budget in characters. |
| `--overlap-chars` | `100` | Characters of backward overlap. Must be below `--max-chars`. |

`recursive` splits on progressively finer separators (paragraphs, then lines,
then words) so it breaks at the most natural boundary that fits.
`markdown` splits on headings and never splits inside a fenced code block, which
makes it the right choice for documentation. `sentence` is the finest, useful
when you want precise citations rather than context.

Overlap exists so a sentence spanning a chunk boundary is still findable from
either side. With `markdown`, overlap never reaches back across a heading.

## What it prints

The summary is JSON, so a CI job can assert on it:

| Field | Meaning |
| --- | --- |
| `matched` | Files matching the glob. |
| `ingested` | Files chunked, embedded and written this run. |
| `unchanged` | Files skipped by the per-file digest. |
| `skipped_non_utf8` | Files that were not valid UTF-8. |
| `skipped_empty` | Files that were empty or whitespace only. |
| `chunks` | Chunks written. |
| `stale_tail_pruned` | Leftover chunks removed from files that got shorter. |
| `pruned` | Records removed by `--prune`. |
| `would_ingest` | Under `--dry-run`, files that would be ingested. |
| `cache` | `hits`, `misses` and `evicted` from the embedding cache. |

## Reading a chunked corpus back

Chunks are how the corpus is stored, not how you want to read it. A plain recall
over a chunked collection returns chunk hits: several fragments of the same
document competing for the page, each a sentence or two out of context.

`--rollup` collapses that. It keeps the best-matching chunk per document, and
`--neighbours` widens each survivor with the chunks around it:

```bash
nidus recall docs "how does the writer lock work" \
  --dir ./store --rollup 1 --neighbours 1
```

Each hit gains a `context` field: the winning chunk plus its neighbours, stitched
back into the passage they came from. The overlap two adjacent chunks share is
dropped rather than repeated, so `context` is the source once.

Nothing else about the result changes. `context` is extra payload, so the ids,
the scores and the order are exactly what the same query returns without it, and
a hit that is not part of a chunked document simply has no `context`.

Over HTTP the same knob is `rollup` on the recall body, and the `recall` MCP tool
takes it too:

```bash
curl -s localhost:8080/collections/docs/recall \
  -d '{"query": "how does the writer lock work", "rollup": {"neighbours": 1}}'
```

Corpora ingested before 0.75.0 have no stored chunk offsets, so their windows are
joined with a blank line and keep whatever overlap the chunker left. The per-file
digest covers the chunk options, so changing any of them re-ingests the tree and
picks the offsets up.

## Keyword-only, with no provider

Every embedding provider is a network provider, so an ordinary `ingest` needs an API
key or a local ollama. When you only want BM25, `--fts-only` skips that entirely:

```bash
nidus ingest ./docs \
  --collection docs \
  --glob '**/*.md' \
  --dir ./store \
  --strategy markdown \
  --fts-only nidus.text

nidus text-search --dir ./store nidus.text "how does compaction work"
```

It walks, chunks, skips unchanged files and prunes exactly as above. The difference is
what it writes: each chunk becomes a **text-only record**, carrying its text and
provenance attrs, taking no vector row and staying out of every vector scan. No filler
vectors, which would otherwise poison `search` and `hybrid-search` with meaningless
cosine scores.

Pass the flag once per attr you want indexed. The chunk text lands under `nidus.text`;
`nidus.source_path` is useful when you want to find a chunk by its file name. Change
the set later and the corpus re-ingests and the schema is redeclared, because the
sorted field list is folded into the per-file digest.

Pointed at a directory with no store, this creates one with **dimension 0**: it
declares no embedding space, and a vector query against it is refused naming that
reason (HTTP `400`) rather than answered with an empty ranking. Pass `--dim`, or point
`--fts-only` at a store that already exists, to keep room for vectors alongside.

`--fts-only` and `--embed-provider` are mutually exclusive, rejected at parse time.

## Scope

`ingest` reads your local filesystem, so it is a command-line feature and has no
HTTP or MCP equivalent: a server cannot walk your disk on your behalf. If you
have the text already, `POST /collections/{name}/remember` and the `remember` MCP
tool cover the same ground one document at a time.
