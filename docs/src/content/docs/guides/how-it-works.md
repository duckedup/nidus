---
title: How it works
description: The nidus storage model and search path, end to end, from upsert to ranked hits.
---

nidus holds dense vectors plus typed metadata in a single on-disk directory and
answers nearest-neighbour queries over cosine (the default), dot, or Euclidean.
Scoring is **exact by default** (every in-scope vector is compared), and you can
opt into an [approximate index](/guides/search/#approximate-search-ann) (HNSW or IVF)
for larger collections. There is no query planner and no background thread: the
whole thing is a RAM-resident matrix, an optional in-RAM index, and a small amount of
write glue.

## The storage model

A store is a directory:

```
<dir>/
  manifest  names the live segments (the first is `data`); the atomic commit point
  data      append-only, fixed-stride, row-major f32 matrix (header pins dimension)
  log       append-only framed op stream: [len][bincode(Op)][crc32] (the commit record)
  lock      O_EXCL writer-exclusion lock file
  seg-…     additional immutable segments, once a store grows past the seal threshold
```

- **`data`** is the vectors: a flat `f32` matrix with a fixed stride (the pinned
  dimension), row-major, **never rewritten in place**. New rows are appended. A store
  holds one or more such **segments** presented as a single dense row space; by default
  it is just `data` (see [Storage → Segments](/guides/storage/#segments)).
- **`manifest`** names the live segments and pins the dimension/metric: a tiny
  CRC-checked object, replaced atomically when a segment is sealed or compacted.
- **`log`** is the commit record: an append-only stream of framed,
  CRC32-checked, bincode-encoded operations (`CreateCollection`, `Upsert`,
  `Delete`, …). This is what makes a write durable.
- **`lock`** excludes concurrent writers via an `O_EXCL` lock file: pure std,
  no `flock`, no FFI.

## Open

`open` reads `data` into RAM and **replays `log`** into an in-RAM index:

```
collection → { id → (row, attrs) }
```

The index is fully reproducible from the two files, so it is never itself
persisted. After open, **search never touches disk**: it sweeps the in-RAM
matrix. (The opt-in [`Config::mmap`](/guides/storage/#larger-than-ram-memory-mapped-segments)
mode trades this for capacity: cold segments are mapped from disk and paged in on
touch, so a store can outgrow RAM.)

## Upsert

A batch upsert is a fixed sequence designed so a crash can never corrupt the
store:

1. Append the new vectors to `data`.
2. fsync `data`.
3. Append the committing `Upsert` records to `log`.
4. fsync `log`.

The `log` append is the commit point. A vector that made it into `data` but
whose `Upsert` record never landed in `log` is simply ignored on the next open:
it is an unreferenced row, reclaimed by [compaction](/guides/storage/#compaction).
Upsert is **all-or-nothing**: any failure mid-batch rolls `data` and `log` back
to the entry marks, so a caught `ENOSPC` leaves the store exactly as it was.

### Group commit

Steps 2 and 4 are one disk barrier, and a barrier costs the same whether the batch
carries one record or a thousand. That is fine for an indexer writing big batches
occasionally; it is the ceiling for a server taking many small writes at once, where every
concurrent write used to pay its own.

So the barrier can be **shared**. `deferred` runs a group of mutations without syncing;
`commit` then takes one barrier for all of them, still `data` before `log`:

```rust
db.deferred(|db| {
    db.upsert("docs", &first)?;
    db.upsert("docs", &second)?;
    Ok(())
})?;
db.commit()?; // one fsync pair for both batches
```

The rule that keeps this honest: **do not tell anyone a write succeeded until `commit`
returns `Ok`.** Before that its bytes are appended but not durable: the same tail state a
crash leaves behind, which the next open discards. Deferring the barrier is a way to pay for
durability once instead of N times, never a way to skip it.

[`nidus serve`](/guides/http-server/) does this for you: concurrent writes are applied
together under one lock, share one barrier, and each request is answered only after it
succeeds. Nothing waits for a group to form, so a lone write is exactly as fast as before.

## Search

Search scores (cosine, dot, or Euclidean) over a
[`Scope`](/reference/api/#scope) (one collection, a named subset, or the whole
store), merged into a single ranking. By default it is exact (every in-scope row is
scored); with [`Config::ann`](/guides/search/#approximate-search-ann) set it instead
walks an approximate index for a candidate set and applies the same scope/filter/rerank
to those. The exact path is:

1. Resolve the scope to a set of candidate rows.
2. Apply the metadata [`Filter`](/guides/filters/#filters) (before any dot
   product: cheap rows are discarded first).
3. Score each surviving row against the query with a dot product. Because
   vectors are **unit-normalized on insert**, `score = dot(v, q)` *is* cosine
   similarity in `[-1, 1]`.
4. Keep the top-k in a bounded heap, optionally dropping anything below
   `min_score`.
5. Cut the page. The ranking is a **total order** (score descending, then
   `collection`, then `id`) computed `offset + top_k` deep, so
   [pagination](/guides/search/#paginating-a-search) tiles it with no gap and no overlap.

Steps 3–5 are where the opt-in ranking knobs sit: a
[`rank_by`](/guides/search/#ranking-by-recency) recency penalty is subtracted from each
base score before the heap sees it, and
[`limit_per`](/guides/search/#capping-hits-per-attribute-value) caps hits per attribute
value as the page is cut, with
[`diversity`](/guides/search/#spreading-near-duplicates-apart) reordering the survivors in
vector space between the two. All are off by default, and an untouched query returns exactly
what it always did.

Scoping the whole store in one call is sound because **every collection shares
one embedding space**: one dimension is pinned for the life of the store, so
all vectors are directly comparable.

The scoring kernel is plain safe Rust the optimizer can vectorize: an 8-lane
chunked dot product, an allocation-free top-k scan, and a storage-order
(prefetcher-friendly) sweep of the matrix. See
[Performance](/reference/performance/) for the numbers.

## Index cache lifecycle

Two derived structures are cached to disk on top of `data`/`log`: the ANN graph/lists
and the FTS postings. A third, per-segment IVF, deliberately is not. None of the three
is ever the source of truth, so every path that reads a cache treats a bad read as a
signal to rebuild, never as an error.

### One codec, one rule: a bad load returns `Ok(None)`

Both caches serialize through the same framed codec (`src/index_cache.rs`): magic
bytes, a version, a watermark, a caller-supplied validity key, the payload, and a
CRC32. `load` decodes and checks all of it, and on any mismatch, missing object, or
torn tail it returns `Ok(None)`, never `Err`. So "the cache is corrupt" and "the cache
is absent" are the same event to every caller: fall back to rebuilding from `data`/`log`.

### The ANN cache: query-time knobs excluded from the validity key

`src/ann/persist.rs`'s `validity_key` folds in the ANN kind, distance metric,
quantization, dimension, `m`, `ef_construction`, `n_lists`, and seed: change any of
those and the cache is stale, discarded, and rebuilt. It deliberately excludes
`ef_search`, `n_probe`, and `overscan`, the knobs that only steer a query over an
already-built structure. Raising `ef_search` to tune recall never invalidates the
on-disk graph; only a change to how the structure is built does.

### The FTS cache: watermarked by the log offset

`persist_fts` and `load_or_build_fts` (`src/store/mod.rs`) key the cache on the FTS
schema and analyzer/BM25 params, and watermark it with the log offset at persist time.
On open, the cache is adopted only when that watermark equals the store's current log
offset exactly: any write since the last persist makes it stale, and the whole index
is rebuilt from the replayed docs rather than incrementally caught up.

### `persist_index` is out-of-band: never called from upsert or flush

`Store::persist_index` is the only thing that writes either cache to disk, and it is
explicit by design. Nothing in the `upsert` or `flush` path calls it. What does:

- `compact()` calls it best-effort, via `let _ =`: a persist failure must not fail the
  compaction, since the cache is derived and disposable.
- The public `Nidus::persist_index` API, for a caller that wants it written on its own
  schedule.
- The clean-shutdown path shared by both `nidus serve` (HTTP) and `nidus mcp` (stdio),
  which flushes and then persists the index as the process exits, also best-effort.

So a long-running writer that never compacts and never shuts down cleanly (and never
calls `persist_index` directly) keeps both indexes in RAM only. The next open finds no
cache, or a stale one, and pays a full rebuild.

### Per-segment IVF is never cached at all

`build_segment_indexes` (`src/store/mod.rs`) is the one derived structure with no
`index_cache` path whatsoever. It recomputes from scratch on every open, on a
lock-free reader's `refresh`, and on every `compact`; a single freshly sealed segment
takes the cheaper incremental `index_just_sealed` instead. None of these paths write
anything to disk. At small scale this is invisible; with `Config::segment_index_min_rows`
set on a large store it is not, since every one of those rebuilds pays a full,
k-means-driven IVF build over every eligible segment, on every restart.

The only tests behind any of the above are in-memory codec round-trips
(`src/index_cache.rs`, `src/ann/persist.rs`): corrupt CRC, key mismatch, config
mismatch, a truncated buffer, and a roundtrip-and-searches-the-same check. Nothing
drives `persist_index` over HTTP and asserts that a stale cache gets rebuilt rather
than served, so there is no e2e test of this lifecycle: the claims above come from
reading the source, not from a test suite.

## What it deliberately is not

- **Exact by default.** The default search compares every in-scope vector: 100%
  recall, by construction. Approximate indexing (HNSW/IVF) is opt-in via
  [`Config::ann`](/guides/search/#approximate-search-ann) when you want speed over
  exactness at larger scale.
- **Not a database.** No SQL, no joins, no transactions across calls.
- **Not async.** The hot path is CPU-bound; the library API is synchronous (see
  [Embedding in a host app](/guides/integrating/)).
- **In-process by default.** You embed it and call methods directly; when you
  want it over the wire, [`nidus serve`](/guides/http-server/) exposes the whole
  store over HTTP.

None of those are walls: they are *seams*, additive over the same append-only
format. Several have since shipped as opt-in modes: an [ANN
index](/guides/search/#approximate-search-ann), [scalar/binary
quantization](/guides/search/#quantization), and [memory-mapped
larger-than-RAM stores](/guides/storage/#larger-than-ram-memory-mapped-segments).
Each stays off by default, so the simple exact-in-RAM store is what you get until
you opt in.
