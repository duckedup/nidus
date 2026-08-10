---
title: CLI
description: Every nidus subcommand and flag, with its environment variable and the feature that unlocks it.
---

Every flag the `nidus` binary accepts, generated from `nidus --help` and each subcommand's
own `--help`. For a guided tour with worked examples, see the [command-line
guide](/guides/cli-and-server/); this page is the exhaustive reference.

The binary has **23 subcommands**: `serve`, `mcp`, `collections`, `create`, `drop`,
`upsert`, `search`, `aggregate`, `list`, `set-fts-schema`, `text-search`,
`hybrid-search`, `get`, `delete`, `compact`, `configure`, `backup`, `restore`,
`verify`, `check`, `stats`, `remember`, `recall`.

## Feature gating

Not every install has every subcommand or flag. Which flags exist depends on how the
binary was built:

| Install | Command | Surface |
| --- | --- | --- |
| `cargo binstall nidus` (prebuilt), or the install script | n/a | Everything below: all 23 subcommands, every `--embed-*`/`--summarize-*` flag, `mcp`, `remember`, `recall` |
| `cargo install nidus --features cli` | build from source | No `mcp`, `remember`, or `recall` subcommand, and `serve` has **no** `--embed-*`/`--summarize-*` flags |

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
| `--summarize-provider <NAME>` | `NIDUS_SUMMARIZE_PROVIDER` | Summarizer provider enabling `mode: "summarize"`: `anthropic` or `openai`. Needs the `summarize` feature. |
| `--summarize-model <MODEL>` | `NIDUS_SUMMARIZE_MODEL` | Summarizer model. Defaults to the provider's default. |
| `--summarize-api-key <KEY>` | `NIDUS_SUMMARIZE_API_KEY` | API key for the summarizer provider. |
| `--summarize-base-url <URL>` | `NIDUS_SUMMARIZE_BASE_URL` | Base-URL override for the summarizer provider. |

See [Remember & recall](/guides/remember-and-recall/) for the provider table and
[`remember`](#remember-memory-feature)/[`recall`](#recall-memory-feature) below for
parity between the CLI, HTTP, and MCP surfaces at
[Parity across the surfaces](/guides/remember-and-recall/#parity-across-the-surfaces).

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
| `--read-timeout <SECONDS>` | `NIDUS_READ_TIMEOUT` | Deadline for a read request (default `30`; `0` disables). |
| `--write-timeout <SECONDS>` | `NIDUS_WRITE_TIMEOUT` | Deadline for a mutating request (default `600`; `0` disables). |
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

### `text-search`

Full-text (BM25) search of fields declared via `set-fts-schema`. Usage:
`nidus text-search [OPTIONS] --dir <DIR> [FIELD] [QUERY]`.

| Flag | Env | Description |
| --- | --- | --- |
| `--clause <FIELD=TEXT>` | none | An extra query clause (repeatable). Use instead of the positional field/query pair, never alongside it. |
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

### `hybrid-search`

Fuse a vector query and a BM25 text query with Reciprocal Rank Fusion. Usage:
`nidus hybrid-search [OPTIONS] --dir <DIR> [FIELD] [TEXT]`. Takes the same
`--clause`/`--combine`/`--explain`/`--highlight`/`--max-fragments`/`--fragment-chars`
flags as `text-search` above, plus:

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

Recall the nearest remembered text to `query`. Opens read-only, so it runs alongside
a `nidus serve` holding the writer lock. Usage:
`nidus recall [OPTIONS] --dir <DIR> <COLLECTION> <QUERY>`. Takes the store flags and
the [ingest flags](#ingest-flags-memory-feature), plus:

| Flag | Env | Description |
| --- | --- | --- |
| `-k, --top-k <N>` | none | Hits to return (default `10`). |
| `--min-score <SCORE>` | none | Drop hits scoring below this cosine similarity. |
| `--where <FILTER>` | none | AND-filter as JSON (same form as `search --where`). |

`remember`/`recall` are the CLI door onto the same memory layer HTTP's
`/remember`/`/recall` and MCP's tools use; see
[Parity across the surfaces](/guides/remember-and-recall/#parity-across-the-surfaces)
for what matches and what does not, yet, between the three.
