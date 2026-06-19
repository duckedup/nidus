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
    .quantization(None)              // int8 two-pass search (default: off)
    .ann(None)                       // approximate-nearest-neighbour index (default: off)
    .query_threads(1)                // worker threads per exact search (default: 1)
    .segment_max_rows(None)          // seal the active segment past N rows (default: off)
    .persistence("")                 // durable bytes: "" = local; "s3://…"/"gs://…"
    .memory("");                     // shared working set: "" = local; "redis://…"
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

### `ann`

`Option<AnnConfig>` — default `None` (disabled; exact search). When set,
the store builds an in-memory approximate-nearest-neighbour index and `search` walks
it for an over-fetched candidate set, then applies the scope/filter/`min_score` and an
exact f32 rerank — trading recall for speed when a scan over every vector is more than
you need. Two algorithms, via `AnnConfig::hnsw()` (a navigable small-world graph, the
default) and `AnnConfig::ivf()` (k-means inverted lists). May be combined with
`quantization` (a quantized index walk plus an exact f32 rerank). See
[approximate search](/guides/search/#approximate-search-ann) for details and tuning.

### `query_threads`

`usize` — default `1` (single-threaded; no behavior change). When `> 1`, a single
large search is split across this many `std::thread::scope` workers to cut one
query's latency — both the exact f32 scan and, when
[int8 quantization](/guides/search/#int8-scalar-quantization) is on, its int8 first
pass. The f32 scan is memory-bandwidth-bound (sublinear speedup); the int8 first
pass is compute-bound and scales better with threads. Leave it at `1` if you already
run concurrent searches under `Arc<RwLock<Nidus>>` — see
[two kinds of parallelism](/guides/integrating/#two-kinds-of-parallelism).

When an [HNSW index](/guides/search/#approximate-search-ann) is enabled, `> 1` also
parallelizes the from-scratch graph **build** (on `open` with no cache, and on
`compact`) across this many threads — the expensive part of opening an ANN store.
Incremental `upsert` and the serial build at `1` are unchanged; note a parallel build
is non-deterministic (insertion order varies), so a graph built with threads can
differ slightly from the serial one (recall stays equivalent).

### `segment_max_rows`

`Option<u64>` — default `None`. A store keeps its vectors in one or more **segments**
named by a small `manifest` (the first is the base `data` segment). When this is set,
the active (appendable) segment is sealed into an **immutable** segment once it grows past
this many rows, and a fresh active segment begins; the new manifest is published
atomically (the commit point). `None` — the default — keeps the store a single segment,
behaving exactly as it always has. A soft bound: a single `upsert` batch is never split,
so a segment can exceed it by one batch. Most stores never need this; see
[Storage](/guides/storage/#segments) for the on-disk picture.

### `persistence`

`String` — where the durable `data`/`log` bytes live (default `""` = local files under
[`path`](#path)). An [`open_persistence`](/guides/storage-backends/) location: a path / `file://`,
or `s3://<bucket>[/<prefix>]` / `gs://<bucket>[/<prefix>]` for a **live object-store-backed
store** (each segment is rewritten as one whole object on flush — `O(object)`, suited to
low write rates, under an advisory writer lock). With an object store, pass `dimension`
explicitly — the remote header is not peeked. See [Storage backends](/guides/storage-backends/).

### `memory`

`String` — where the in-RAM working set is *shared* (default `""`/`local`/`ram` = the
process heap; nothing shared). A Redis-family URL — `redis://` / `rediss://` /
`valkey://` / `valkeys://` / `keydb://` / `dragonfly://`, optionally `?prefix=<ns>` —
publishes the serialized working set on `flush` and adopts it on `open`, so other workers
skip the log replay. A rebuildable cache: an unreachable or evicted tier is never fatal.
See [Memory stores](/guides/memory-stores/).

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
