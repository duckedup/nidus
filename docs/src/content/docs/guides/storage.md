---
title: Storage & durability
description: The nidus on-disk format, the per-batch fsync contract, crash recovery, compaction, and lock-free cross-process readers.
---

nidus is durable by design with a tiny surface: two append-only files and a lock.
This page covers the on-disk format, what survives a crash, how dead rows are
reclaimed, and how a second process reads a store another is writing.

## On-disk format

```
<dir>/
  data    append-only, fixed-stride, row-major f32 matrix (header pins dimension)
  log     append-only framed op stream: [len][bincode(Op)][crc32] — the commit record
  lock    O_EXCL writer-exclusion lock file
```

All on-disk encoding is **little-endian and explicit**. Every `log` record is
length-prefixed and CRC32-checked, so a torn tail (a half-written final record
after a crash) is detectable and is dropped on the next open.

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

Because `data` is never rewritten in place, a `delete` or an overwriting
`upsert` leaves the old row in the file as a **dead row** — still on disk, no
longer referenced by the index. [`compact()`](/reference/api/#compact) rewrites
`data` to drop dead rows and reclaim the space.

Compaction also runs automatically on `open` when the dead-row ratio exceeds
[`Config::auto_compact`](/reference/configuration/#auto_compact) (default `0.5` —
half the rows dead). Set it to `None` to disable and compact only on demand.

[`footprint()`](/reference/api/#footprint) reports `rows`, `dead_rows`, and
`vector_bytes` so you can decide when a manual compaction is worth it.

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

A `ReadOnly` open takes **no lock**. It reads `data` to its current size *S*,
replays `log`, and ignores any record that references a row ≥ *S*/dim. The
result is a **consistent, possibly-stale snapshot** — never a torn read — even
while the writer is mid-append. This is the lock-free basis for search-only
processes and a future read-only search server.

Only one **writer** (`OpenMode::ReadWrite`, the default) may hold a store at a
time, enforced by the `O_EXCL` `lock` file. A stale lock left by a crashed writer
is reclaimed after [`Config::lock_ttl`](/reference/configuration/#lock_ttl)
(default 60s).
