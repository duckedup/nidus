---
title: How it works
description: The nidus storage model and search path, end to end — from upsert to ranked hits.
---

nidus holds dense vectors plus typed metadata in a single on-disk directory and
answers nearest-neighbour queries over cosine (the default), dot, or Euclidean.
Scoring is **exact by default** — every in-scope vector is compared — and you can
opt into an [approximate index](/guides/search/#approximate-search-ann) (HNSW or IVF)
for larger collections. There is no query planner and no background thread — the
whole thing is a RAM-resident matrix, an optional in-RAM index, and a small amount of
write glue.

## The storage model

A store is a directory:

```
<dir>/
  manifest  names the live segments (the first is `data`); the atomic commit point
  data      append-only, fixed-stride, row-major f32 matrix (header pins dimension)
  log       append-only framed op stream: [len][bincode(Op)][crc32] — the commit record
  lock      O_EXCL writer-exclusion lock file
  seg-…     additional immutable segments, once a store grows past the seal threshold
```

- **`data`** is the vectors: a flat `f32` matrix with a fixed stride (the pinned
  dimension), row-major, **never rewritten in place**. New rows are appended. A store
  holds one or more such **segments** presented as a single dense row space; by default
  it is just `data` (see [Storage → Segments](/guides/storage/#segments)).
- **`manifest`** names the live segments and pins the dimension/metric — a tiny
  CRC-checked object, replaced atomically when a segment is sealed or compacted.
- **`log`** is the commit record: an append-only stream of framed,
  CRC32-checked, bincode-encoded operations (`CreateCollection`, `Upsert`,
  `Delete`, …). This is what makes a write durable.
- **`lock`** excludes concurrent writers via an `O_EXCL` lock file — pure std,
  no `flock`, no FFI.

## Open

`open` reads `data` into RAM and **replays `log`** into an in-RAM index:

```
collection → { id → (row, attrs) }
```

The index is fully reproducible from the two files, so it is never itself
persisted. After open, **search never touches disk** — it sweeps the in-RAM
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
whose `Upsert` record never landed in `log` is simply ignored on the next open —
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
returns `Ok`.** Before that its bytes are appended but not durable — the same tail state a
crash leaves behind, which the next open discards. Deferring the barrier is a way to pay for
durability once instead of N times, never a way to skip it.

[`nidus serve`](/guides/http-server/) does this for you: concurrent writes are applied
together under one lock, share one barrier, and each request is answered only after it
succeeds. Nothing waits for a group to form, so a lone write is exactly as fast as before.

## Search

Search scores (cosine, dot, or Euclidean) over a
[`Scope`](/reference/api/#scope) — one collection, a named subset, or the whole
store — merged into a single ranking. By default it is exact (every in-scope row is
scored); with [`Config::ann`](/guides/search/#approximate-search-ann) set it instead
walks an approximate index for a candidate set and applies the same scope/filter/rerank
to those. The exact path is:

1. Resolve the scope to a set of candidate rows.
2. Apply the metadata [`Filter`](/guides/search/#filters) (before any dot
   product — cheap rows are discarded first).
3. Score each surviving row against the query with a dot product. Because
   vectors are **unit-normalized on insert**, `score = dot(v, q)` *is* cosine
   similarity in `[-1, 1]`.
4. Keep the top-k in a bounded heap, optionally dropping anything below
   `min_score`.
5. Cut the page. The ranking is a **total order** — score descending, then
   `collection`, then `id` — computed `offset + top_k` deep, so
   [pagination](/guides/search/#paginating-a-search) tiles it with no gap and no overlap.

Steps 3–5 are where the opt-in ranking knobs sit: a
[`rank_by`](/guides/search/#ranking-by-recency) recency penalty is subtracted from each
base score before the heap sees it, and
[`limit_per`](/guides/search/#capping-hits-per-attribute-value) caps hits per attribute
value as the page is cut. Both are off by default, and an untouched query returns exactly
what it always did.

Scoping the whole store in one call is sound because **every collection shares
one embedding space** — one dimension is pinned for the life of the store, so
all vectors are directly comparable.

The scoring kernel is plain safe Rust the optimizer can vectorize: an 8-lane
chunked dot product, an allocation-free top-k scan, and a storage-order
(prefetcher-friendly) sweep of the matrix. See
[Performance](/reference/performance/) for the numbers.

## What it deliberately is not

- **Exact by default.** The default search compares every in-scope vector — 100%
  recall, by construction. Approximate indexing (HNSW/IVF) is opt-in via
  [`Config::ann`](/guides/search/#approximate-search-ann) when you want speed over
  exactness at larger scale.
- **Not a database.** No SQL, no joins, no transactions across calls.
- **Not async.** The hot path is CPU-bound; the library API is synchronous (see
  [Embedding in a host app](/guides/integrating/)).
- **In-process by default.** You embed it and call methods directly; when you
  want it over the wire, [`nidus serve`](/guides/http-server/) exposes the whole
  store over HTTP.

None of those are walls — they are *seams*, additive over the same append-only
format. Several have since shipped as opt-in modes: an [ANN
index](/guides/search/#approximate-search-ann), [scalar/binary
quantization](/guides/search/#quantization), and [memory-mapped
larger-than-RAM stores](/guides/storage/#larger-than-ram-memory-mapped-segments).
Each stays off by default, so the simple exact-in-RAM store is what you get until
you opt in.
