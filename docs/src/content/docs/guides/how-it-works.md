---
title: How it works
description: The nidus storage model and search path, end to end — from upsert to ranked hits.
---

nidus holds dense vectors plus typed metadata in a single on-disk directory and
answers nearest-neighbour queries by **exact brute-force cosine**. There is no
ANN index, no query planner, and no background thread — the whole thing is a
RAM-resident matrix and a small amount of write glue.

## The storage model

A store is a directory with three files:

```
<dir>/
  data    append-only, fixed-stride, row-major f32 matrix (header pins dimension)
  log     append-only framed op stream: [len][bincode(Op)][crc32] — the commit record
  lock    O_EXCL writer-exclusion lock file
```

- **`data`** is the vectors: a flat `f32` matrix with a fixed stride (the pinned
  dimension), row-major, **never rewritten in place**. New rows are appended.
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
matrix.

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

## Search

Search is brute-force cosine over a [`Scope`](/reference/api/#scope) — one
collection, a named subset, or the whole store — merged into a single ranking:

1. Resolve the scope to a set of candidate rows.
2. Apply the metadata [`Filter`](/guides/search/#filters) (before any dot
   product — cheap rows are discarded first).
3. Score each surviving row against the query with a dot product. Because
   vectors are **unit-normalized on insert**, `score = dot(v, q)` *is* cosine
   similarity in `[-1, 1]`.
4. Keep the top-k in a bounded heap, optionally dropping anything below
   `min_score`.

Scoping the whole store in one call is sound because **every collection shares
one embedding space** — one dimension is pinned for the life of the store, so
all vectors are directly comparable.

The scoring kernel is plain safe Rust the optimizer can vectorize: an 8-lane
chunked dot product, an allocation-free top-k scan, and a storage-order
(prefetcher-friendly) sweep of the matrix. See
[Performance](/reference/performance/) for the numbers.

## What it deliberately is not

- **Not approximate.** No HNSW/IVF index. 100% recall, by construction.
- **Not a database.** No SQL, no joins, no transactions across calls.
- **Not async.** The hot path is CPU-bound; the API is synchronous (see
  [Embedding in a host app](/guides/integrating/)).
- **Not networked.** It is a library you call in-process.

Each of those is a deferred *seam*, not a wall — mmap, an ANN index, and scalar
quantization are all designed-for and additive over the same append-only file.
They are simply not built until a real workload needs them.
