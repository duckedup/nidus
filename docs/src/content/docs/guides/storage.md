---
title: Storage & durability
description: The nidus on-disk format, the per-batch fsync contract, crash recovery, compaction, and lock-free cross-process readers.
---

nidus is durable by design with a tiny surface: two append-only files and a lock.
This page covers the on-disk format, what survives a crash, how dead rows are
reclaimed, and how a second process reads a store another is writing.

This page is about a store on **local disk**. To keep the durable data somewhere
else — Amazon S3 or Google Cloud Storage — see [Storage backends](/guides/storage-backends/);
to share the in-memory index across processes via Redis, see
[Memory stores](/guides/memory-stores/).

## On-disk format

```
<dir>/
  manifest  names the live segments (the first is `data`) + pins dimension/metric
  data      append-only, fixed-stride, row-major f32 matrix (header pins dimension)
  log       append-only framed op stream: [len][bincode(Op)][crc32] — the commit record
  lock      O_EXCL writer-exclusion lock file
  seg-…     additional immutable segments (only once a store seals past the threshold)
```

All on-disk encoding is **little-endian and explicit**. Every `log` record is
length-prefixed and CRC32-checked, so a torn tail (a half-written final record
after a crash) is detectable and is dropped on the next open. The `manifest` is a
`[crc32][bincode]` object replaced atomically — a reader sees one whole manifest
version, never a torn mix.

The `data` header pins the embedding **dimension** at creation. Reopening a store
with a different dimension is a hard error — one embedding space per store, for
the life of the store.

## The durability contract

The default fsync policy is [`Fsync::PerBatch`](/reference/configuration/#fsync):
each `upsert`/`delete` call appends vectors, fsyncs `data`, appends the
committing `log` records, then fsyncs `log`. **A crash loses at most the
in-flight batch** — everything fsynced before it is intact, and the in-RAM index
is fully reproducible from the files.

`Fsync::OnFlush` defers the fsync to an explicit
[`flush()`](/reference/api/#flush) or close. This is faster but weaker: an
unflushed batch can be lost on a crash. Use it when you are bulk-loading and can
afford to redo the load on failure.

## Graceful failure

Resource exhaustion never corrupts a store:

- **Appends are atomic.** A partial write rolls back to the row/frame boundary.
- **Upsert is all-or-nothing.** Any failure mid-batch rolls `data` and `log` back
  to the entry marks, so a caught `ENOSPC` leaves the store exactly as it was.
- **RAM growth uses `try_reserve`.** Out-of-memory surfaces as an `Err`, not an
  abort (except `attrs`/`id` clones, which std gives no fallible reserve for).
- **A hard ceiling holds under overcommit.**
  [`Config::max_vector_bytes`](/reference/configuration/#max_vector_bytes)
  refuses a batch *before* allocating — the only guard that works on systems
  where the kernel SIGKILLs before an allocation fails.
  [`Nidus::footprint()`](/reference/api/#footprint) is the introspection hook for
  deciding whether you can afford more data.

## Compaction

Because segments are never rewritten in place, a `delete` or an overwriting
`upsert` leaves the old row behind as a **dead row** — still on disk, no
longer referenced by the index. [`compact()`](/reference/api/#compact) collapses
every [segment](#segments) into one fresh `data` segment that drops the dead rows,
publishes the new manifest, and reclaims the old segment objects.

Compaction also runs automatically on `open` when the dead-row ratio exceeds
[`Config::auto_compact`](/reference/configuration/#auto_compact) (default `0.5` —
half the rows dead). Set it to `None` to disable and compact only on demand.

[`footprint()`](/reference/api/#footprint) reports `rows`, `dead_rows`, and
`vector_bytes` so you can decide when a manual compaction is worth it.

## Segments

A store's vectors live in one or more **segments** — self-contained, immutable chunks of
rows — named in order by the `manifest`. The last one is the **active** segment that new
rows append to; the rest are sealed and never rewritten. The segments are presented to
search as a single dense row space, so this is invisible to queries: the same exact
brute-force scan runs whether a store is one segment or many.

By default a store is a **single segment** (`data`) and behaves exactly as it always has.
Set [`Config::segment_max_rows`](/reference/configuration/#segment_max_rows) to roll the
active segment into a sealed one once it grows past *N* rows and start a fresh one — no
data is copied; sealing just publishes a new `manifest`. Sealing and
[compaction](#compaction) (which collapses every segment back into one) replace the
manifest atomically, which is the store's commit point.

A store that predates this format (just `data` + `log`, no `manifest`) is migrated
transparently on the first read-write open — `data` becomes the base segment and a
manifest is written. A read-only open of such a store writes nothing.

Segments are also the unit of **indexing at scale**: with
[`Config::segment_index_min_rows`](/reference/configuration/#segment_index_min_rows) set, a
sealed segment large enough to cross that threshold gets its own IVF index (the active tail
stays exact), so searches over a large store walk the cold segments and brute-force only the
fresh data. See [per-segment indexing](/guides/search/#per-segment-indexing-at-scale).

## Larger than RAM: memory-mapped segments

By default nidus loads every segment into RAM on `open`. Set
[`Config::mmap(true)`](/reference/configuration/#mmap) and each **immutable** (sealed)
segment is instead served from a read-only **memory-map** of its file — the operating
system pages a segment in on demand and reclaims it under pressure, so a store can hold
more vectors than fit in memory. The **active** segment (the one still taking appends)
stays in RAM.

```rust
use nidus::{Config, Nidus};

// A local store with sealed segments, served larger-than-RAM.
let store = Nidus::open(
    Config::new("/path/to/store", 768)
        .segment_max_rows(Some(1_000_000)) // produce immutable segments to map
        .mmap(true),
)?;
# anyhow::Ok(())
```

Search over mapped segments goes through the same row accessor as the in-RAM path, so
**results are identical** — still exact (or, with an [index](/guides/search/), the same
approximate set), still filter- and `min_score`-respecting. It composes with quantization
and the [per-segment indexes](/guides/search/#per-segment-indexing-at-scale): a cold
segment can be both mapped and indexed.

A few conditions apply:

- It is effective only for a **local-filesystem** store with **sealed** segments — it
  needs [`segment_max_rows`](/reference/configuration/#segment_max_rows) to create
  immutable segments and a mappable local file. An object-store (`s3://`/`gs://`) or
  in-memory store silently stays all-RAM.
- The host must be **little-endian** (the on-disk `f32` layout). Other hosts fall back to
  loading into RAM.
- [Compaction](#compaction) still materializes the live set in RAM, so it is bounded by
  memory even when the store as a whole is not — keep it infrequent on a very large store.

## Cross-process readers

A store can be opened **read-only** by other processes while one process holds
the writer lock:

```rust
use nidus::{Config, OpenMode};

let reader = Nidus::open(
    Config::new("/path/to/store", 768).open_mode(OpenMode::ReadOnly)
)?;
# anyhow::Ok(())
```

A `ReadOnly` open takes **no lock**. It reads the `manifest`, loads the segments it
names to their current total size *S*, replays `log`, and ignores any record that
references a row ≥ *S*/dim. The result is a **consistent, possibly-stale snapshot** —
never a torn read — even while the writer is mid-append: a not-yet-named segment or a
row past *S* is simply invisible until its commit. This is the lock-free basis for
search-only processes reading a store another process is writing.

Only one **writer** (`OpenMode::ReadWrite`, the default) may hold a store at a
time, enforced by the `O_EXCL` `lock` file. A stale lock left by a crashed writer
is reclaimed after [`Config::lock_ttl`](/reference/configuration/#lock_ttl)
(default 60s).
