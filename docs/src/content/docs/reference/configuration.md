---
title: Configuration
description: Every knob on nidus Config — distance metric, fsync policy, open mode, auto-compaction, lock TTL, and the max_vector_bytes ceiling.
---

`Config` carries everything needed to open a store. Construct it with
`Config::new(path, dimension)` — the two required fields — and adjust the rest
with chainable builder setters. The store **location is always the caller's
choice**: nidus contributes no path defaults, env vars, or hidden directories.

```rust
use std::time::Duration;
use nidus::{Config, Distance, Fsync, OpenMode, Quantization};

let cfg = Config::new("/path/to/store", 768)
    .distance(Distance::Cosine)      // similarity metric (default)
    .fsync(Fsync::PerBatch)          // durability granularity (default)
    .open_mode(OpenMode::ReadWrite)  // ReadOnly = no lock, search-only
    .auto_compact(Some(0.5))         // compact on open above this dead-row ratio
    .lock_ttl(Duration::from_secs(60))
    .max_vector_bytes(None)          // no ceiling (default)
    .quantization(None);             // int8 two-pass search (default: off)
# let _ = cfg;
```

## Fields

### `path`

`PathBuf` — **required.** The store directory; created if absent.

### `dimension`

`usize` — **required.** The pinned embedding dimension. It is written to the
`data` header at creation and must match on every reopen — reopening with a
different dimension is a hard error. One embedding space per store.

### `distance`

[`Distance`](/reference/api/#distance) — default `Distance::Cosine`. The
similarity / distance metric used for scoring. Like dimension, it is pinned in
the data header at creation — reopening with a different metric is a hard error.
See [distance metrics](/guides/search/#distance-metrics) for details on each
metric.

### `fsync`

[`Fsync`](#fsync) — default `Fsync::PerBatch`. Durability granularity.

### `open_mode`

[`OpenMode`](#openmode) — default `OpenMode::ReadWrite`. Whether this handle may
write (and thus takes the writer lock).

### `auto_compact`

`Option<f32>` — default `Some(0.5)`. Dead-row ratio that triggers
[compaction](/guides/storage/#compaction) on open. `None` disables auto-compaction
(compact only via `compact()`).

### `lock_ttl`

`Duration` — default 60s. The window after which a stale writer lock (left by a
crashed process) may be reclaimed.

### `max_vector_bytes`

`Option<u64>` — default `None` (no ceiling). A hard cap on the vector matrix
(`rows * dimension * 4` bytes), enforced **before** allocating: `upsert` refuses
a batch that would exceed it, and `open` refuses a `data` file already over it.

This is the only exhaustion guard that holds under **memory overcommit**, where
the kernel SIGKILLs the process before an allocation fails and `try_reserve`
never fires. It counts physical rows including not-yet-compacted dead rows, so
`compact()` can reclaim headroom. Pair it with
[`footprint()`](/reference/api/#footprint) to decide whether more data fits.

### `quantization`

`Option<Quantization>` — default `None` (disabled). When set, the store
maintains an in-memory int8 copy of all vectors and uses a two-pass search:
int8 first-pass → f32 rerank. See
[int8 scalar quantization](/guides/search/#int8-scalar-quantization) for details.

## `Fsync`

```rust
pub enum Fsync {
    PerBatch,  // fsync after every upsert/delete (durable per batch). Default.
    OnFlush,   // fsync only on explicit flush()/close (faster, weaker durability)
}
```

`PerBatch` loses at most the in-flight batch on a crash. `OnFlush` trades that
guarantee for speed — useful for bulk loads you can afford to redo. See
[the durability contract](/guides/storage/#the-durability-contract).

## `OpenMode`

```rust
pub enum OpenMode {
    ReadWrite,  // takes the writer lock; mutations allowed. Default.
    ReadOnly,   // no lock taken; mutations rejected — for search-only processes
}
```

A `ReadOnly` handle reads a consistent, possibly-stale, lock-free snapshot — many
can coexist with a single writer. See
[cross-process readers](/guides/storage/#cross-process-readers).
