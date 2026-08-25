---
title: CLI
description: Every nidus subcommand and flag, with its environment variable and the feature that unlocks it.
---

Every flag the `nidus` binary accepts, generated from `nidus --help` and each subcommand's
own `--help`. For a guided tour with worked examples, see the [command-line
guide](/guides/cli-and-server/); this page is the exhaustive reference.

The binary has **31 subcommands**: `serve`, `mcp`, `collections`, `create`, `drop`,
`upsert`, `search`, `similar`, `aggregate`, `list`, `set-fts-schema`, `suggest`,
`text-search`, `hybrid-search`, `get`, `delete`, `compact`, `versions`, `configure`,
`backup`, `restore`, `verify`, `check`, `stats`, `tune`, `ingest`, `remember`, `recall`,
`aliases`, `set-alias`, `drop-alias`.

## Feature gating

Not every install has every subcommand or flag. Which flags exist depends on how the
binary was built:

| Install | Command | Surface |
| --- | --- | --- |
| `cargo binstall nidus` (prebuilt), or the install script | n/a | Everything below: all 31 subcommands, every `--embed-*`/`--summarize-*` flag, `mcp`, `ingest`, `remember`, `recall` |
| `cargo install nidus --features cli` | build from source | No `mcp`, `ingest`, `remember`, or `recall` subcommand, and `serve` has **no** `--embed-*`/`--summarize-*` flags |

The prebuilt binaries (`cargo binstall`, the install script, and the release
tarballs) are all built with `--features serve`, which is the umbrella feature
`cli + memory + embed-all + summarize-all + mcp`. A plain `--features cli`
source build gets the store-operation subcommands and `serve` for the raw vector
routes, but none of the AI-ingest layer: `mcp`, `remember`, and `recall` do not
exist as subcommands at all, and `serve`'s embedder/summarizer flags are absent
rather than merely inert. A reference that listed them anyway would be wrong for
that install, so every flag below tagged **memory** or **mcp** is only present
when the binary was built with that feature (`serve` pulls in both).

See [Install](/guides/cli-and-server/#install) for the two paths, and
[Remember & recall](/guides/remember-and-recall/#turn-it-on) for the Cargo
features behind the memory layer if you are building from source yourself.

## Store flags

Every subcommand that opens a store (all of them except `backup`, `restore`, `verify`,
and `check`, which take their own `--dir`/`--persistence` instead) accepts this shared
set. For an existing store the dimension and distance are read from the on-disk header;
`--dim`/`--distance` are only needed when creating one, or to double-check an existing
one, where a mismatch is a hard error.

| Flag | Env | Description |
| --- | --- | --- |
| `-d, --dir <DIR>` | `NIDUS_DIR` | Store directory. Required, even when `--persistence` names an object store. |
| `--dim <DIM>` | `NIDUS_DIM` | Embedding dimension. Inferred from an existing store; required to create one. |
| `--distance <cosine\|euclidean\|dot>` | `NIDUS_DISTANCE` | Distance metric. Inferred from an existing store; defaults to `cosine` at creation. |
| `--read-only` | `NIDUS_READ_ONLY` | Open without taking the writer lock (rejects mutations). |
| `--history-versions <N>` | `NIDUS_HISTORY_VERSIONS` | Keep the last N commit points addressable by `--at-version`. Off by default: enabling it makes every durable batch a commit point, so the write path pays for it. |
| `--at-version <N>` | `NIDUS_AT_VERSION` | Open a read-only snapshot pinned to that past commit version instead of the current state. Forces the store read-only, so a subcommand that writes is refused. |
| `--ann <hnsw\|ivf>` | `NIDUS_ANN` | Opt into an approximate-nearest-neighbour index. Omit for exact brute-force. |
| `--ann-m <N>` | `NIDUS_ANN_M` | HNSW: max neighbours per node above layer 0. |
| `--ann-ef-construction <N>` | `NIDUS_ANN_EF_CONSTRUCTION` | HNSW: build-time beam width. |
| `--ann-ef-search <N>` | `NIDUS_ANN_EF_SEARCH` | HNSW: search-time beam width. |
| `--ann-n-lists <N>` | `NIDUS_ANN_N_LISTS` | IVF: number of k-means lists (`0` = auto). |
| `--ann-n-probe <N>` | `NIDUS_ANN_N_PROBE` | IVF: lists probed per query. |
| `--ann-overscan <N>` | `NIDUS_ANN_OVERSCAN` | Candidate over-fetch multiple before rerank. Applies to both ANN kinds. |
| `--ann-seed <N>` | `NIDUS_ANN_SEED` | Build PRNG seed (deterministic index). Applies to both ANN kinds. |
| `--persistence <LOCATION>` | `NIDUS_PERSISTENCE` | Where the durable bytes live: a path/`file://` (default) or `s3://…`/`gs://…`. |
| `--memory <LOCATION>` | `NIDUS_MEMORY` | Share the in-RAM working set: `local` (default) or a `redis://…`-family URL. |
| `--cluster` | `NIDUS_CLUSTER` | Run as one of several cooperating instances over a shared backend. |
| `--no-cluster` | `NIDUS_NO_CLUSTER` | Run standalone: the explicit off for `--cluster`. |
| `--mmap` | `NIDUS_MMAP` | Memory-map immutable segments instead of holding them in RAM. |
| `--no-mmap` | `NIDUS_NO_MMAP` | Turn mmap off, overriding a recorded `configure --mmap` default. |
| `--quantization <int8\|binary>` | `NIDUS_QUANTIZATION` | Quantize the search first pass, then rerank in exact f32. |
| `--quant-rescore <N>` | `NIDUS_QUANT_RESCORE` | Over-fetch multiple for the quantized first pass. |
| `--query-threads <N>` | `NIDUS_QUERY_THREADS` | Worker threads for a single exact search (`1` = serial, default). |
| `--segment-max-rows <N>` | `NIDUS_SEGMENT_MAX_ROWS` | Seal the active segment past this many rows. |
| `--segment-index-min-rows <N>` | `NIDUS_SEGMENT_INDEX_MIN_ROWS` | Minimum rows for a sealed segment to get its own IVF index. |
| `--fsync <per-batch\|on-flush>` | `NIDUS_FSYNC` | fsync policy. `per-batch` (default) is durable per call; `on-flush` is faster, weaker. |
| `--auto-compact <RATIO>` | `NIDUS_AUTO_COMPACT` | Rewrite the data matrix when this fraction of rows is dead (default `0.5`). |
| `--no-auto-compact` | `NIDUS_NO_AUTO_COMPACT` | Never auto-compact; reclaim dead rows only on explicit `compact`. |
| `--lock-ttl <SECONDS>` | `NIDUS_LOCK_TTL` | Seconds before a stale writer lock may be reclaimed (default `60`). |
| `--wait-for-lease [<SECONDS>]` | `NIDUS_WAIT_FOR_LEASE` | Wait for the writer handle instead of exiting; becomes a standby. Bare flag waits forever. |
| `--max-staleness <SECONDS>` | `NIDUS_MAX_STALENESS` | Fail the readiness probe once a `--read-only` instance is this stale. |
| `--max-vector-bytes <N>` | `NIDUS_MAX_VECTOR_BYTES` | Refuse to open a store whose vector matrix would exceed this many bytes. |
| `--strict-embedder-identity` | `NIDUS_STRICT_EMBEDDER_IDENTITY` | Refuse a recall against a collection with no pinned embedder, instead of warning. |

### Turning a flag back off

Each `--no-…` flag is the explicit off for its positive twin, and it wins wherever the
positive one came from: a value recorded by `nidus configure`, an inherited
`NIDUS_CLUSTER`/`NIDUS_MMAP`/`NIDUS_AUTO_COMPACT` in a shared environment block, or a
default. That is what makes one pod in a Deployment able to run standalone without
editing the env block it shares. Typing both sides on the same command line is still
refused, since only there is the contradiction something you wrote yourself.

See [Configuration](/reference/configuration/) for what each of these does to the
open store, and [Approximate search (ANN)](/guides/cli-and-server/#approximate-search-ann)
/ [Configure once](/guides/cli-and-server/#configure-once-recording-store-defaults) for
how the `--ann`/`--quantization`/`--query-threads`/`--mmap` knobs can be recorded as a
store's own defaults instead of repeated on every call.

## Ingest flags (`memory` feature)

`serve`, `mcp`, `remember`, and `recall` additionally take this set, present only when
the binary was built with the `memory` feature (the `serve` umbrella). With no
`--embed-provider`, `serve`/`mcp` still start, serving only the raw vector endpoints.

| Flag | Env | Description |
| --- | --- | --- |
| `--embed-provider <NAME>` | `NIDUS_EMBED_PROVIDER` | Embedding provider: `voyage`, `openai`, `ollama`, `cohere`, `gemini`, `mistral`, `jina`, or `openai-compat`. |
| `--embed-model <MODEL>` | `NIDUS_EMBED_MODEL` | Embedding model. Defaults to the provider's default (`openai-compat` has none). |
| `--embed-api-key <KEY>` | `NIDUS_EMBED_API_KEY` | API key for the embedding provider (some, e.g. Ollama, need none). |
| `--embed-base-url <URL>` | `NIDUS_EMBED_BASE_URL` | Base-URL override (required for `openai-compat` and self-hosted gateways). |
| `--embed-dimension <N>` | `NIDUS_EMBED_DIMENSION` | Ask for a non-native embedding width. Voyage Matryoshka models (the `voyage-4` family, `voyage-3-large`, `voyage-3.5`, `voyage-3.5-lite`, `voyage-code-3`) accept 256, 512, 1024, or 2048; OpenAI's `text-embedding-3-small`/`-large` accept any width up to their native 1536/3072. Other providers and models reject the flag. |
| `--summarize-provider <NAME>` | `NIDUS_SUMMARIZE_PROVIDER` | Summarizer provider enabling `mode: "summarize"`: `anthropic` or `openai`. Needs the `summarize` feature. |
| `--summarize-model <MODEL>` | `NIDUS_SUMMARIZE_MODEL` | Summarizer model. Defaults to the provider's default. |
| `--summarize-api-key <KEY>` | `NIDUS_SUMMARIZE_API_KEY` | API key for the summarizer provider. |
| `--summarize-base-url <URL>` | `NIDUS_SUMMARIZE_BASE_URL` | Base-URL override for the summarizer provider. |

See [Remember & recall](/guides/remember-and-recall/) for the provider table and
[`remember`](#remember-memory-feature)/[`recall`](#recall-memory-feature) below for
parity between the CLI, HTTP, and MCP surfaces at
[Parity across the surfaces](/guides/remember-and-recall/#parity-across-the-surfaces).

## Rerank flags (`rerank` feature)

`search`, `text-search`, `hybrid-search`, and `recall` additionally take these flags,
present only when the binary was built with the `rerank` feature. They configure the
opt-in **cross-encoder** re-scoring stage; this is unrelated to `--ann-overscan`'s
ANN candidate over-fetch and to `--quantization`'s quantized-then-f32 rescore above,
which use the word "rerank" for something else entirely.

| Flag | Env | Description |
| --- | --- | --- |
| `--rerank-provider <NAME>` | `NIDUS_RERANK_PROVIDER` | Cross-encoder reranking provider: `voyage` or `cohere`. |
| `--rerank-model <MODEL>` | `NIDUS_RERANK_MODEL` | Reranking model. Defaults to the provider's default. |
| `--rerank-api-key <KEY>` | `NIDUS_RERANK_API_KEY` | API key for the reranking provider. |
| `--rerank-base-url <URL>` | `NIDUS_RERANK_BASE_URL` | Base-URL override for the reranking provider. |

`--rerank` (below, per subcommand) with no `--rerank-provider` configured is an error
naming the flag, never a silent un-reranked result.

## Subcommands

Flags already listed above (store flags, ingest flags) are not repeated per
subcommand; only what that subcommand adds is shown.

### `serve`

Run the HTTP server. Usage: `nidus serve [OPTIONS] --dir <DIR>`.

| Flag | Env | Description |
| --- | --- | --- |
| `--addr <ADDR>` | `NIDUS_ADDR` | Address to bind (default `127.0.0.1:7700`). |
| `--token <TOKEN>` | `NIDUS_TOKEN` | Require `Authorization: Bearer <token>` on every request except `/health`. |
| `--max-body-bytes <N>` | `NIDUS_MAX_BODY_BYTES` | Maximum request body size in bytes (default 256 MiB). |
| `--max-concurrent-requests <N>` | `NIDUS_MAX_CONCURRENT_REQUESTS` | Cap on store-touching requests in flight (default `0` = auto: 8x CPU cores, floored at 64). |
| `--read-timeout <SECONDS>` | `NIDUS_READ_TIMEOUT` | Deadline for a read request, a plain recall included (default `30`; `0` disables). |
| `--write-timeout <SECONDS>` | `NIDUS_WRITE_TIMEOUT` | Deadline for a mutating request, a `reinforce` recall included (default `600`; `0` disables). |
| `--body-idle-timeout <SECONDS>` | `NIDUS_BODY_IDLE_TIMEOUT` | Abandon a stalled request body (default `15`; `0` disables). |
| `--refresh-interval <SECONDS>` | `NIDUS_REFRESH_INTERVAL` | Refresh this instance on a timer instead of leaving it to the caller. |
| `--require-remote` | `NIDUS_REQUIRE_REMOTE` | Refuse to start unless `--persistence` and `--memory` are both shared, non-local backends. |
| `--embed-*`/`--summarize-*` | see above | Present only under the `memory` feature; see [Ingest flags](#ingest-flags-memory-feature). |

The full operator-facing environment table, authentication model, and request
lifecycle live on the [HTTP server](/guides/http-server/) page; this page only lists
the flags.

### `mcp` (`mcp` feature)

Speak MCP over stdio, for `claude mcp add nidus -- nidus mcp --dir ~/.nidus`. Usage:
`nidus mcp [OPTIONS] --dir <DIR>`. Takes the [store flags](#store-flags) and, under
the `memory` feature, the [ingest flags](#ingest-flags-memory-feature). No flags of
its own. See [MCP](/guides/mcp/).

### `collections`

List collections. Usage: `nidus collections [OPTIONS] --dir <DIR>`. No flags beyond
the store flags.

### `create`

Create a collection. Usage: `nidus create [OPTIONS] --dir <DIR> <NAME>`. No flags
beyond the store flags.

### `drop`

Drop a collection and its records. Usage: `nidus drop [OPTIONS] --dir <DIR> <NAME>`.
No flags beyond the store flags.

### `aliases`

List aliases and the collections they resolve to. Usage:
`nidus aliases [OPTIONS] --dir <DIR>`. No flags beyond the store flags.

```
nidus aliases --dir ./mystore
```

### `set-alias`

Create or repoint an alias at a concrete collection, in one call. Usage:
`nidus set-alias [OPTIONS] --dir <DIR> <NAME> <TARGET>`. No flags beyond the store
flags. The target must already exist as a concrete collection, and aliases never
chain: pointing one alias at another is rejected.

```
nidus set-alias --dir ./mystore docs docs_v2
```

### `drop-alias`

Remove an alias. Deletes no records; the collection it pointed at is untouched.
Usage: `nidus drop-alias [OPTIONS] --dir <DIR> <NAME>`. No flags beyond the store
flags.

```
nidus drop-alias --dir ./mystore docs
```

### `upsert`

Upsert records (JSON array) from a file or stdin. Usage:
`nidus upsert [OPTIONS] --dir <DIR> <COLLECTION>`.

| Flag | Env | Description |
| --- | --- | --- |
| `--file <FILE>` | none | Read records from this file instead of stdin. |

### `search`

Nearest-neighbour search; the query vector is a JSON array of floats. Usage:
`nidus search [OPTIONS] --dir <DIR> [COLLECTIONS]...`.

| Flag | Env | Description |
| --- | --- | --- |
| `--query-file <FILE>` | none | Read the query vector from this file instead of stdin. |
| `-k, --top-k <N>` | none | Hits to return (default `10`). |
| `--offset <N>` | none | Skip this many top-ranked hits before returning (default `0`). |
| `--min-score <SCORE>` | none | Drop hits scoring below this cosine similarity. |
| `--where <FILTER>` | none | AND-filter as JSON. |
| `--exact` | none | Force the exact scan, bypassing any ANN index and quantized first pass. |
| `--include-attr <ATTR>` | none | Return only this attr (repeatable). Exclusive with `--exclude-attr`. |
| `--exclude-attr <ATTR>` | none | Return every attr but this one (repeatable). Exclusive with `--include-attr`. |
| `--rank-by <EXPR>` | none | Ranking expression as JSON, e.g. a recency-decay expression. |
| `--limit-per <ATTR>` | none | Cap hits per distinct value of this attribute; needs `--limit-per-max`. |
| `--limit-per-max <N>` | none | Maximum hits kept per distinct `--limit-per` value. |
| `--diversity <LAMBDA>` | none | MMR lambda spreading hits in vector space: `1.0` pure relevance, `0.0` pure spread. |
| `--expand-radius <N>` | none | Widen each hit with this many neighbouring chunks of its own document, either side. Adds a `context` field; changes nothing about the ranking. |
| `--expand-parent-field <ATTR>` | none | Attr grouping a document's chunks (default `nidus.parent_id`); needs `--expand-radius`. |
| `--expand-index-field <ATTR>` | none | Attr ordering the chunks within a document (default `nidus.chunk_index`); needs `--expand-radius`. |
| `--expand-text-field <ATTR>` | none | Attr holding each chunk's text (default `nidus.text`); needs `--expand-radius`. |
| `--rerank` (`rerank` feature) | none | Re-score the candidate window with the configured cross-encoder. Requires `--rerank-query`, since a vector query carries no text of its own. |
| `--rerank-query <TEXT>` (`rerank` feature) | none | Text scored against each candidate by the cross-encoder. |
| `--rerank-overscan <N>` (`rerank` feature) | none | Candidates retrieved per `top_k` before the cross-encoder rerank (default `10`). |
| `--rerank-text-attr <ATTR>` (`rerank` feature) | none | Attr holding each candidate's text for the cross-encoder rerank (default `nidus.text`). |
| `--plan` | none | Print `{hits, plan}` instead of the bare hit array: path taken, rows scanned, candidate survival, timings. See [Query plans](/reference/http-api/#query-plans-how-a-query-ran). |

### `similar`

"More like this": nearest-neighbour search using the vector already stored at
`COLLECTION`/`ID`, instead of a caller-supplied query vector. Usage:
`nidus similar [OPTIONS] --dir <DIR> <COLLECTION> <ID>`. Takes the same flags as
`search` above, plus:

| Flag | Env | Description |
| --- | --- | --- |
| `--scope <COLLECTION>` | none | Collections to search (repeatable); omit to search only the source's own collection. |

The source record is always excluded from its own results, by id rather than by
score, so a genuine duplicate of the source still comes back. `COLLECTION`/`ID`
naming no record, or a record with no stored vector (a text-only entry), is an error
naming the reason, not an empty result.

`similar` takes no `--rerank`: "more like this" starts from a stored vector with no
query text, the same reason `/search/similar` excludes it over HTTP.

### `aggregate`

Count records matching a filter, and sum numeric attributes, without listing them.
Usage: `nidus aggregate [OPTIONS] --dir <DIR> [COLLECTIONS]...`.

| Flag | Env | Description |
| --- | --- | --- |
| `--where <FILTER>` | none | AND-filter as JSON (same form as `search --where`). |
| `--sum <ATTR>` | none | Attribute to sum (repeatable). Missing/non-numeric values are skipped. |
| `--group-by <ATTR>` | none | Report one row per distinct value of this attribute, alongside the totals. |

### `list`

List records by metadata filter, no vector query. Usage:
`nidus list [OPTIONS] --dir <DIR> [COLLECTIONS]...`.

| Flag | Env | Description |
| --- | --- | --- |
| `--offset <N>` | none | Skip this many matches before returning (default `0`). |
| `-n, --limit <N>` | none | Maximum results (default `100`). |
| `--where <FILTER>` | none | AND-filter as JSON. |
| `--include-attr <ATTR>` | none | Return only this attr (repeatable). Exclusive with `--exclude-attr`. |
| `--exclude-attr <ATTR>` | none | Return every attr but this one (repeatable). Exclusive with `--include-attr`. |
| `--order-by <ATTR>` | none | Sort by this attribute instead of storage order. |
| `--desc` | none | Sort `--order-by` descending; requires `--order-by`. |

### `set-fts-schema`

Declare a collection's full-text-indexed fields (BM25). Usage:
`nidus set-fts-schema [OPTIONS] --dir <DIR> <COLLECTION>`. The tuning flags apply to
every `--field`; use `--field-spec` to tune one field on its own. Re-running rebuilds
the affected field indexes.

| Flag | Env | Description |
| --- | --- | --- |
| `--field <NAME>` | none | Attribute field to index, taking the tuning flags below (repeatable). |
| `--field-spec <SPEC>` | none | One field with its own tuning, e.g. `body:k1=1.5,b=0.3`. Keys: `k1`, `b`, `ascii_folding`, `max_token_len`. |
| `--k1 <N>` | none | BM25 term-frequency saturation (default `1.2`). |
| `--b <N>` | none | BM25 length normalization, `0..=1` (default `0.75`). |
| `--ascii-folding` | none | Fold Latin diacritics to ASCII, so "café" and "cafe" share a term. |
| `--max-token-len <N>` | none | Drop tokens longer than this many characters (default: no limit). |

### `suggest`

Ranked term completions for a prefix from a full-text-indexed field's vocabulary, the kind
of list an autocomplete dropdown shows. Usage:
`nidus suggest [OPTIONS] --dir <DIR> <FIELD> <PREFIX> [COLLECTIONS]...`. Omit the collection
list to complete from every collection. Completions are ranked by document frequency
(commonest first), the opposite of how a prefix clause ranks *documents* by idf. The
response's `matched` exceeds the returned count when the 256-term cap truncated the match
set. Completions are real words: the field's surface forms are indexed alongside its stems,
so every keystroke of `running` completes to `running`.

Only the final token of `<PREFIX>` is completed, but the words before it are not discarded:
a completion's `df` counts only documents that also carry them, so `"quick br"` completes
against the documents that say "quick" and a `brown` that never co-occurs with it is not
offered. A single-token prefix, or one whose earlier words are all stopwords, is
unconditioned.

| Flag | Env | Description |
| --- | --- | --- |
| `-n, --limit <N>` | none | How many completions to return (default `10`). |
| `--where <JSON>` | none | AND-filter (same form as `search --where`). Each completion's `df` counts only matching documents, so a completion no match carries is not offered. |
| `--no-fuzzy` | none | Turn off typo tolerance. On by default: when the exact prefix matches nothing, `suggest` retries within a short edit budget that grows with the fragment's length (none below 4 characters, 1 at 4 to 7, 2 at 8 or more). |

```
nidus suggest --dir ./store body nid docs --limit 5
```
```json
{
  "suggestions": [
    { "term": "nidus", "df": 42 },
    { "term": "nidification", "df": 3 }
  ],
  "matched": 2
}
```

```
nidus suggest --dir ./store body "quick br" docs \
  --where '[{"Eq":["tenant",{"Str":"acme"}]}]'
```

### `text-search`

Full-text (BM25) search of fields declared via `set-fts-schema`. Usage:
`nidus text-search [OPTIONS] --dir <DIR> [FIELD] [QUERY]`.

| Flag | Env | Description |
| --- | --- | --- |
| `--clause <FIELD=TEXT>` | none | An extra query clause (repeatable). Use instead of the positional field/query pair, never alongside it. |
| `--prefix-clause <FIELD=TEXT>` | none | An extra clause whose final term is a prefix (repeatable, for typeahead). Combine with `--clause`; never with the positional pair. |
| `--prefix` | none | Match the positional `QUERY`'s final term as a prefix instead of a complete word. Applies only to the positional field/query pair; use `--prefix-clause` with `--clause`. |
| `--combine <sum\|max>` | none | How several clauses fold into one score (default `sum`). |
| `--explain` | none | Annotate each hit with its per-clause (and, for hybrid, per-leg) scores. |
| `--highlight` | none | Return highlighted fragments of the matched text. |
| `--max-fragments <N>` | none | Fragments per field when highlighting (default `1`). |
| `--fragment-chars <N>` | none | Characters per fragment when highlighting (default `160`). |
| `--in <COLLECTION>` | none | Collections to search (repeatable); omit to search every collection. |
| `-k, --top-k <N>` | none | Hits to return (default `10`). |
| `--offset <N>` | none | Skip this many top-ranked hits before returning (default `0`). |
| `--min-score <SCORE>` | none | Drop hits scoring below this raw BM25 score. |
| `--where <FILTER>` | none | AND-filter as JSON (same form as `search --where`). |
| `--include-attr <ATTR>` | none | Return only this attr (repeatable). Exclusive with `--exclude-attr`. |
| `--exclude-attr <ATTR>` | none | Return every attr but this one (repeatable). Exclusive with `--include-attr`. |
| `--rank-by <EXPR>` | none | Ranking expression as JSON (same form as `search --rank-by`), applied to the BM25 score. |
| `--limit-per <ATTR>` | none | Cap hits per distinct value of this attribute; needs `--limit-per-max`. |
| `--limit-per-max <N>` | none | Maximum hits kept per distinct `--limit-per` value. |
| `--diversity <LAMBDA>` | none | MMR lambda spreading hits in vector space: `1.0` pure relevance, `0.0` pure spread. |
| `--expand-radius <N>` | none | Widen each hit with this many neighbouring chunks of its own document, either side. Adds a `context` field; changes nothing about the ranking. |
| `--expand-parent-field <ATTR>` | none | Attr grouping a document's chunks (default `nidus.parent_id`); needs `--expand-radius`. |
| `--expand-index-field <ATTR>` | none | Attr ordering the chunks within a document (default `nidus.chunk_index`); needs `--expand-radius`. |
| `--expand-text-field <ATTR>` | none | Attr holding each chunk's text (default `nidus.text`); needs `--expand-radius`. |
| `--rerank` (`rerank` feature) | none | Re-score the candidate window with the configured cross-encoder. |
| `--rerank-query <TEXT>` (`rerank` feature) | none | Text scored against each candidate by the cross-encoder. Defaults to the positional `QUERY` when the `--clause` spelling is not used; with `--clause`, omitting it is an error. |
| `--rerank-overscan <N>` (`rerank` feature) | none | Candidates retrieved per `top_k` before the cross-encoder rerank (default `10`). |
| `--rerank-text-attr <ATTR>` (`rerank` feature) | none | Attr holding each candidate's text for the cross-encoder rerank (default `nidus.text`). |

A prefix match expands only the clause's final term, capped at 256 expansions (past the
cap, the commonest completions win rather than the command erroring). With `--explain`,
each hit's clause score reports `expansion: {matched, scored}` when the clause was a
prefix match. See [prefix matching for typeahead](/guides/search/#prefix-matching-search-as-you-type).

### `hybrid-search`

Fuse a vector query and a BM25 text query with Reciprocal Rank Fusion. Usage:
`nidus hybrid-search [OPTIONS] --dir <DIR> [FIELD] [TEXT]`. Takes the same
`--clause`/`--prefix-clause`/`--prefix`/`--combine`/`--explain`/`--highlight`/
`--max-fragments`/`--fragment-chars` flags as `text-search` above, plus:

| Flag | Env | Description |
| --- | --- | --- |
| `--query-file <FILE>` | none | Read the query vector (JSON array) from this file instead of stdin. |
| `--in <COLLECTION>` | none | Collections to search (repeatable); omit to search every collection. |
| `-k, --top-k <N>` | none | Fused hits to return (default `10`). |
| `--offset <N>` | none | Skip this many top-ranked fused hits before returning (default `0`). |
| `--where <FILTER>` | none | AND-filter as JSON, applied to both legs. |
| `--rrf-k <N>` | none | RRF rank-bias constant (default `60`). |
| `--candidates <N>` | none | Candidates pulled per leg before fusing (default `100`). |
| `--vector-weight <N>` | none | Weight on the vector leg's fused contribution (default `1`). |
| `--text-weight <N>` | none | Weight on the BM25 leg's fused contribution (default `1`). |
| `--expand-radius <N>` | none | Widen each fused hit with this many neighbouring chunks of its own document, either side. Adds a `context` field; changes nothing about the ranking. |
| `--expand-parent-field <ATTR>` | none | Attr grouping a document's chunks (default `nidus.parent_id`); needs `--expand-radius`. |
| `--expand-index-field <ATTR>` | none | Attr ordering the chunks within a document (default `nidus.chunk_index`); needs `--expand-radius`. |
| `--expand-text-field <ATTR>` | none | Attr holding each chunk's text (default `nidus.text`); needs `--expand-radius`. |
| `--rerank` (`rerank` feature) | none | Re-score the fused candidate window with the configured cross-encoder. Requires `--rerank-query`, since the fused ranking has no single natural query text. |
| `--rerank-query <TEXT>` (`rerank` feature) | none | Text scored against each candidate by the cross-encoder. |
| `--rerank-overscan <N>` (`rerank` feature) | none | Candidates retrieved per `top_k` before the cross-encoder rerank (default `10`). |
| `--rerank-text-attr <ATTR>` (`rerank` feature) | none | Attr holding each candidate's text for the cross-encoder rerank (default `nidus.text`). |
| `--plan` | none | Print `{hits, plan}` instead of the bare hit array: path taken, rows scanned, candidate survival, timings. See [Query plans](/reference/http-api/#query-plans-how-a-query-ran). |

### `get`

Print every record in a collection as JSON. Usage:
`nidus get [OPTIONS] --dir <DIR> <COLLECTION>`. No flags beyond the store flags.

### `delete`

Delete records by id, or by `--where` filter. Usage:
`nidus delete [OPTIONS] --dir <DIR> <COLLECTION> [IDS]...`.

| Flag | Env | Description |
| --- | --- | --- |
| `--where <FILTER>` | none | Delete by filter (JSON) instead of ids. Mutually exclusive with positional ids. |

### `compact`

Reclaim dead rows and superseded log records. Usage:
`nidus compact [OPTIONS] --dir <DIR>`.

| Flag | Env | Description |
| --- | --- | --- |
| `--expired` | none | First delete every entry whose `nidus.expires_at` has passed, then reclaim the freed rows. |

### `configure`

Record `--ann`/`--quantization`/`--query-threads`/`--mmap` as this store's own
open-time defaults, so later opens (including `serve`) need not repeat them. Usage:
`nidus configure [OPTIONS] --dir <DIR>`. See
[Configure once](/guides/cli-and-server/#configure-once-recording-store-defaults).

| Flag | Env | Description |
| --- | --- | --- |
| `--clear` | none | Remove the recorded profile instead of writing one. |

### `backup`

Snapshot a store into a single compressed `.tar.gz` archive. Usage:
`nidus backup [OPTIONS] --dir <DIR>`. This subcommand does **not** take the shared
[store flags](#store-flags); it has its own `--dir`/`--persistence`.

| Flag | Env | Description |
| --- | --- | --- |
| `-d, --dir <DIR>` | none | Store directory to back up (the source when `--persistence` is omitted). |
| `--persistence <LOCATION>` | none | Read the source store from this persistence location instead of `--dir`. |
| `-o, --out <LOCATION>` | none | Output archive location: a local path, `file://…`, `s3://…`, or `gs://…`. Defaults to `<dir-name>-<unix-secs>.tar.gz`. |
| `--verify` | none | After writing the archive, re-read it and prove it is restorable. |

See [Backup, restore & verify](/guides/cli-and-server/#backup-restore--verify).

### `restore`

Restore a store from a `nidus backup` archive. Usage:
`nidus restore [OPTIONS] --in <INPUT> --dir <DIR>`. Does not take the shared store
flags.

| Flag | Env | Description |
| --- | --- | --- |
| `-i, --in <LOCATION>` | none | Backup archive location to restore from. |
| `-d, --dir <DIR>` | none | Target store directory (created if absent; the target when `--persistence` is omitted). |
| `--persistence <LOCATION>` | none | Restore into this persistence location instead of `--dir`. |
| `-y, --yes` | none | Overwrite an existing store without prompting (for cron / scripts). |

### `verify`

Prove a backup archive is restorable: check its integrity and open it read-only in a
scratch location. Usage: `nidus verify --in <INPUT>`. Does not take the shared store
flags.

| Flag | Env | Description |
| --- | --- | --- |
| `-i, --in <LOCATION>` | none | Backup archive location to verify. |

### `check`

Verify every live segment's checksum sidecar against its on-disk bytes: a
live-store integrity check, unlike `verify` (which checks a backup archive). Exits
non-zero, naming the segment, on the first mismatch. Usage:
`nidus check [OPTIONS] --dir <DIR>`. Does not take the shared store flags.

| Flag | Env | Description |
| --- | --- | --- |
| `-d, --dir <DIR>` | none | Store directory to check (the source when `--persistence` is omitted). |
| `--persistence <LOCATION>` | none | Check a store at this persistence location instead of `--dir`. |

See [Checking a live store](/guides/cli-and-server/#checking-a-live-store).

### `stats`

Print store footprint and collections as JSON. Usage:
`nidus stats [OPTIONS] --dir <DIR>`. No flags beyond the store flags.

### `versions`

Print the commit-version landscape as JSON: `commit_version` (now), `oldest_readable`,
`pinned`, and `readable` (every addressable commit point). Usage:
`nidus versions [OPTIONS] --dir <DIR>`. No flags beyond the store flags.

```bash
nidus versions --dir ./store
# → {"commit_version":42,"oldest_readable":31,"pinned":null,"readable":[31, ..., 42]}

# Re-run a query against the index as it was at version 31.
nidus search --dir ./store --at-version 31 docs --text "…"
```

`readable` is empty unless the store was written with `--history-versions N`. See
[Point-in-time reads](/guides/storage/#point-in-time-reads).

### `tune`

Sweep `ef_search`/`n_probe`/`overscan` (and, optionally, quantization) against the
store's own vectors, scoring each candidate setting with recall@k measured against
the same binary's exact search, and print a recommended `Config`. Opens read-only,
so it runs alongside a `nidus serve` holding the writer lock. Usage:
`nidus tune [OPTIONS] --dir <DIR>`.

| Flag | Env | Description |
| --- | --- | --- |
| `--collection <NAME>` | none | Sample from this collection only. Omit to sample the whole store. |
| `-k, --top-k <N>` | none | `k` in recall@k (default `10`). |
| `--sample <N>` | none | Number of sampled queries (default `200`). |
| `--ef-search <LIST>` | none | Comma-separated `ef_search` values to sweep (HNSW). |
| `--n-probe <LIST>` | none | Comma-separated `n_probe` values to sweep (IVF). |
| `--overscan <LIST>` | none | Comma-separated overscan values to sweep (both ANN kinds). |
| `--sweep-quantization` | none | Also sweep `none`/`int8`/`binary`. Forces an index rebuild per cell, unlike the other flags above. |
| `--target-recall <F>` | none | Recall@k the recommendation must clear (default `0.95`). |
| `--seed <U64>` | none | Sampling seed, for a reproducible sweep. |

Each sampled query is a vector already in the store, so it has a guaranteed
distance-0 match against itself; `tune` drops that self-hit from both the exact and
approximate results before scoring, and the output says so in words rather than
reporting a flattered number. The result is print-only: it names `nidus configure`
as the way to persist the recommendation, but never writes to the store itself.

### `ingest` (`memory` feature)

Walk a directory, chunk each file, embed the chunks and upsert them. Idempotent: a
re-run over an unchanged tree makes no embedding calls and no writes. Usage:
`nidus ingest [OPTIONS] --collection <NAME> --dir <DIR> <PATH>`. Takes the store flags
and the [ingest flags](#ingest-flags-memory-feature) (it needs an embedder), plus:

| Flag | Env | Description |
| --- | --- | --- |
| `--collection <NAME>` | none | Collection to ingest into. Required. |
| `--glob <GLOB>` | none | GLOB over each path relative to `PATH` (default `*`). `*` crosses `/`, so `*.md` is already recursive; a leading `**/` is optional. |
| `--strategy <S>` | none | `recursive` (default), `markdown`, or `sentence`. |
| `--max-chars <N>` | none | Chunk budget in characters, not tokens (default `1000`). |
| `--overlap-chars <N>` | none | Characters of backward overlap per chunk (default `100`). Must be below `--max-chars`. |
| `--prune` | none | Delete previously-ingested records whose source file is gone. Opt-in, because pointing `ingest` at a partial tree must not empty the collection. Only removes records this command wrote. |
| `--dry-run` | none | Report what would happen without embedding or writing anything. |
| `--no-cache` | none | Skip the content-hash embedding cache. The per-file skip still applies. |
| `--cache-max-entries <N>` | none | Cached vectors to keep before evicting the oldest (default `50000`). |
| `--fts-only <FIELD>` | none | Ingest for BM25 only: store each chunk as a text-only record and declare these attrs as the collection's full-text schema, with no embedder at all. Repeatable. Conflicts with `--embed-provider`. |

Dot-entries, symlinks, and files that are not valid UTF-8 are skipped; the last of
these is counted as `skipped_non_utf8` rather than failing the run. See
[Ingest a directory](/guides/ingest/) for the full behaviour.

#### Keyword-only ingest, with no embedding provider

`--fts-only` is the on-ramp for a corpus you want to search by keyword and nothing
else. It needs no API key, makes no network call, and works offline and in CI:

```bash
nidus ingest --dir ./store --collection docs --glob '*.md' \
  --strategy markdown --fts-only nidus.text ./docs
nidus text-search --dir ./store nidus.text "crash safety"
```

Every chunk is stored as a **text-only record**: it carries the chunk text and its
provenance attrs, occupies no vector row, and stays out of every vector scan. The
chunk text lands under `nidus.text`, which is what the example indexes.

Pointed at a directory with no store yet, this creates one with **dimension 0** -
the store declares that it has no embedding space, and a vector query against it is
refused with that reason rather than answered with an empty ranking. Pass `--dim` (or
point `--fts-only` at a store that already exists) to keep room for vectors, and the
text-only chunks sit alongside them.

Re-running is a no-op on an unchanged tree, the same as an embedding ingest. Changing
which fields you pass to `--fts-only` re-ingests, because the declared field set is
folded into the per-file digest.

### `remember` (`memory` feature)

Embed `text` (optionally summarizing first) and store it. Usage:
`nidus remember [OPTIONS] --dir <DIR> <COLLECTION> <TEXT>`. Takes the store flags
and the [ingest flags](#ingest-flags-memory-feature) (it needs an embedder), plus:

| Flag | Env | Description |
| --- | --- | --- |
| `--id <ID>` | none | Id to store under. Omit to derive a stable one from the text, making re-remembering idempotent. |
| `--attrs <JSON>` | none | Extra attrs as a JSON object of typed values, e.g. `{"tag":{"Str":"ops"}}`. |
| `--ttl-seconds <N>` | none | Seconds until this memory expires. Omit to never expire. |
| `--dedupe-threshold <SCORE>` | none | Cosine floor (0-1) above which this write updates the nearest existing entry instead of inserting a near-duplicate. |
| `--summarize` | none | Summarize the text first and embed the summary, storing both. Needs the `summarize` feature. |

### `recall` (`memory` feature)

Recall the nearest remembered text to `query`. Opens read-only by default, so a plain
`nidus recall` runs alongside a `nidus serve` holding the writer lock. Usage:
`nidus recall [OPTIONS] --dir <DIR> <COLLECTION> <QUERY>`. Takes the store flags and
the [ingest flags](#ingest-flags-memory-feature), plus:

| Flag | Env | Description |
| --- | --- | --- |
| `-k, --top-k <N>` | none | Hits to return (default `10`). |
| `--min-score <SCORE>` | none | Drop hits scoring below this cosine similarity. |
| `--where <FILTER>` | none | AND-filter as JSON (same form as `search --where`). |
| `--diversity <LAMBDA>` | none | MMR lambda spreading hits in vector space: `1.0` pure relevance, `0.0` pure spread. |
| `--rollup <N>` | none | Read the collection as a chunked corpus: keep this many chunks per document. |
| `--neighbours <N>` | none | Chunks stitched either side of each survivor, into the hit's `context`; needs `--rollup`. |
| `--rerank` (`rerank` feature) | none | Re-score the candidate window with the configured cross-encoder. |
| `--rerank-query <TEXT>` (`rerank` feature) | none | Text scored against each candidate by the cross-encoder. Defaults to `QUERY` above. |
| `--rerank-overscan <N>` (`rerank` feature) | none | Candidates retrieved per `top_k` before the cross-encoder rerank (default `10`). |
| `--rerank-text-attr <ATTR>` (`rerank` feature) | none | Attr holding each candidate's text for the cross-encoder rerank (default `nidus.text`). |
| `--reinforce` | none | Stamp `nidus.access_count` / `nidus.last_accessed` on every returned entry; see [reinforcement](/guides/remember-and-recall/#reinforcement). |
| `--extend-ttl-seconds <SECS>` | none | With `--reinforce`, push an existing `nidus.expires_at` forward to now plus this many seconds. Never creates an expiry on an entry that had none. |
| `--rank-by <JSON>` | none | Ranking expression, the same form `search --rank-by` takes, so a recall can rank on the reinforcement counters. |

**`--reinforce` opens the store read-write instead of read-only**, since stamping the
counters is a write. That means `nidus recall --reinforce` cannot run beside a live
`nidus serve` holding the writer lock: the open fails outright with a lock error. A
plain `nidus recall`, with no `--reinforce`, is unaffected and keeps opening read-only.

`remember`/`recall` are the CLI door onto the same memory layer HTTP's
`/remember`/`/recall` and MCP's tools use; see
[Parity across the surfaces](/guides/remember-and-recall/#parity-across-the-surfaces)
for what matches and what does not, yet, between the three.
