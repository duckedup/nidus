---
paths:
  - "src/**"
---

# Architecture, in the small

nidus is an **embeddable vector store**: dense vectors plus typed metadata in a single on-disk
store, answering nearest-neighbour queries by exact brute-force cosine. It is the local storage
leg for semantic-search and indexing tools — a pure-Rust replacement for an embedded
DuckDB/LanceDB. No SQL, no query engine, no background threads.

**For the module map, run `just spec 10`** — it is the same tree with more detail per module,
and it cannot drift from the spec the way a second copy here would. `just spec find <words>`
finds the section that owns any behaviour.

## The four facts worth holding without a lookup

**Storage model.** A store is a set of objects behind a `Persistence` backend (SPEC §13) — a
local directory by default, an `s3://`/`gs://` prefix by URL: `manifest` (the live-segment set
plus the pinned dimension/distance, and the atomic commit point), one or more fixed-stride
`f32` segments (base is `data`; sealing mints `seg-NNNNNNNN`), `log` (the append-only op stream
and the commit record), and `lock`. Segments are append-only except in `compact()`, which
rewrites the base segment in place.

**Search.** `open` loads the live segments into one global row space and replays `log` into an
in-RAM index (`collection → { id → (row, attrs) }`). Search is brute-force over a `Scope` — one
collection, a subset, or the whole store — merged into one ranking, which is sound because all
collections share one embedding space. Vectors are unit-normalized on insert, so
`score = dot(v, q)`.

**Durability.** Per-batch fsync, and the write order is load-bearing: append vectors to the
active segment → fsync it → append committing log records → fsync `log`. So a committed
`Upsert`'s row is durable before anything references it, and a crash loses at most the
in-flight batch. Cross-process readers are lock-free: read the manifest, open the segments it
names for N rows, replay `log`, ignore any record referencing a row ≥ N. Consistent,
possibly-stale, never torn (SPEC §6.2, §14.6).

**Graceful failure (SPEC §6.6).** Appends are atomic and `upsert` is all-or-nothing, so a
caught ENOSPC never corrupts the store. RAM growth uses `try_reserve` — except `attrs`/`id`
clones, which std gives no `try_reserve` for. The overcommit-proof guard is
`Config::max_vector_bytes`; `Nidus::footprint()` is the introspection hook.

## Before proposing something new

`just spec 9` is the shipped-and-deferred seam list, with the reasoning for each. Much of what
looks missing has shipped (int8 and binary quantization, parallel scan, the HTTP server, ANN,
mmap), so check there rather than assuming.
