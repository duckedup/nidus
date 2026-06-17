---
title: Storage backends
description: The two-axis backend model — where durable bytes live (persistence) and where the in-RAM working set is held (memory tier) — and the small sync traits behind it.
---

A nidus store is described along **two independent axes**, each behind a small,
synchronous, object-safe trait in the [`nidus::backend`](/reference/api/) module.
Both default to local, and they compose:

- **Persistence** — where the durable *source-of-truth* bytes live (`data`, `log`)
  along with the derived caches (`ann`, `fts`). The [`Persistence`] trait.
- **Memory tier** — where the in-RAM *working set* is held for serving, so it can be
  shared across processes and reloaded without a rebuild. The [`MemoryTier`] trait.

Today nidus ships the local implementations of both — `LocalFs` (a directory) and
`LocalRam` (the process heap) — and exposes the traits as the seam any other backend
plugs into. A file-backed store **runs over `LocalFs`**: its `data`/`log` segments are
append handles the backend hands out, and its `ann`/`fts` caches and writer lock go
through the same object operations — so the trait is the real substrate, not a wrapper
bolted on the side.

## Why the two axes are separate

They optimize for different things, so nidus keeps them independent:

- Persistence is optimized for **durability and cost** — the bytes that must survive.
- The memory tier is optimized for **fast access and sharing** — the warm working set.

Crucially, **neither axis is ever in the query path**. nidus searches with CPU SIMD
over a local, contiguous `Vec<f32>` (exact cosine, the ANN walk, the quantized scan,
BM25). You cannot run that over bytes on a socket, so the working set is always
materialized into local RAM *before* a scan. A backend only changes `open`/cold-start
and write durability — never search results or latency.

## Persistence: named objects, two classes

A persisted store is not intrinsically a directory; it is a small, fixed set of
**named byte objects** in two classes:

| object | class | reconstructable? |
|--------|-------|------------------|
| `data` | **source of truth** | no |
| `log`  | **source of truth** | no |
| `ann`  | derived cache | yes — rebuilt from `data`/`log` |
| `fts`  | derived cache | yes — rebuilt from `data`/`log` |

Only `data` + `log` must be shipped durably. The derived caches are
*reconstructable*, so a missing, stale, or corrupt cache is never fatal — a backend
may persist them or simply drop and rebuild them on open. (This is the same contract
the local [index caches](/guides/search/) already follow.)

The [`Persistence`] trait is the common denominator of local files and object stores
— whole-object operations plus an **optional** native append capability:

```rust
use std::time::Duration;
use nidus::Result;
use nidus::backend::{Appender, BackendLock};

pub trait Persistence: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;   // whole object
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()>;  // atomic whole-object write
    fn delete(&self, key: &str) -> Result<()>;
    fn list(&self) -> Result<Vec<String>>;
    // Native in-place append (local files); object stores return Ok(None) and
    // rewrite whole objects via `put` instead.
    fn appender(&self, key: &str) -> Result<Option<Box<dyn Appender>>>;
    // Best-effort exclusive lock; Ok(None) on contention, never a hard error.
    fn try_lock(&self, key: &str, ttl: Duration) -> Result<Option<Box<dyn BackendLock>>>;
}
```

`key` is a single flat object name (`"data"`, `"ann"`, …); path separators and `..`
are rejected, so keys behave identically on local files and (any future) object store.

### `LocalFs`

The built-in, default persistence backend. Each object is a file `<dir>/<key>`;
whole-object writes are atomic (temp + fsync + rename), the native `Appender` is a
plain file handle — so the live `data`/`log` path keeps its exact append + fsync
discipline with zero overhead — and `try_lock` is the same `O_EXCL` writer-exclusion
lock described in [Storage & durability](/guides/storage/#cross-process-readers).

```rust
use nidus::backend::{LocalFs, Persistence};

let fs = LocalFs::new("/path/to/store")?;
fs.put("note", b"hello")?;
assert_eq!(fs.get("note")?.as_deref(), Some(b"hello".as_slice()));
# anyhow::Ok(())
```

## Memory tier: a shared, rebuildable working set

The memory tier holds the warm in-RAM state so it can be **shared across processes**
and **reloaded without a rebuild**. It is a *cache* of the serialized working set, not
a source of truth: an empty or evicted tier is never fatal, because the working set is
always rebuildable from the persistence tier.

```rust
use std::time::Duration;
use nidus::Result;

pub trait MemoryTier: Send + Sync {
    fn load(&self, key: &str) -> Result<Option<Vec<u8>>>;
    fn store(&self, key: &str, bytes: &[u8], ttl: Option<Duration>) -> Result<()>;
}
```

### `LocalRam`

The built-in, default memory tier: the working set *is* the process heap, shared
between threads of one process (so it composes with `Arc<RwLock<Nidus>>`) but not
across processes. `ttl` is ignored — local RAM never evicts.

```rust
use nidus::backend::{LocalRam, MemoryTier};

let tier = LocalRam::new();
tier.store("warm", b"...serialized working set...", None)?;
# anyhow::Ok(())
```

## Selection by URL scheme

Backends are addressed by a location string, so a store's home is a value, not a
compile-time choice:

```rust
use nidus::{open_persistence, open_memory_tier};

// file://<path> — or a bare path — selects LocalFs.
let store = open_persistence("file:///path/to/store")?;
// "local" (or "" / "ram") selects LocalRam.
let tier = open_memory_tier("local")?;
# anyhow::Ok(())
```

`file://` (local files), `local` (local RAM), and **`s3://<bucket>[/<prefix>]`**
(Amazon S3 and S3-compatible stores — R2, MinIO) are available today; `s3://` reads
credentials, region, and an optional custom endpoint from the standard AWS environment
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ENDPOINT_URL`). Other
schemes (`gs://`) are recognized and rejected with a clear error rather than a silent
fallback, so a location string is always validated up front.

```bash
# Snapshot a local store straight to S3 — the destination is just an s3:// location.
AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… AWS_REGION=us-east-1 \
  nidus backup --dir ./store --out s3://my-bucket/backups/store.tar.gz
```

S3 is a whole-object backend (`get`/`put`/`delete`/`list`) — there is no native
append, so it serves snapshots and whole-object use, not a live append-backed
`data`/`log` store.

## Snapshots are object-granular

Because a store is just a few named objects, a **snapshot is one object too** — a
single `.tar.gz` holding the source-of-truth `data` and `log`. The
[`nidus backup`](/guides/cli-and-server/) command reads those objects from the source
store's backend and `put`s the archive to whatever backend its destination names:

```bash
# A local path or file:// today; another backend's URL once it lands.
nidus backup  --dir ./store --out ./store.tar.gz
nidus backup  --dir ./store --out file:///backups/store.tar.gz
nidus restore --in  ./store.tar.gz --dir ./restored
```

The snapshot is consistent without taking the writer lock: it captures `data` then
`log`, and a log record referencing a row not yet in the captured `data` is ignored on
restore (the same lock-free rule that lets a reader run beside a writer — see
[Storage & durability](/guides/storage/#cross-process-readers)).

## Why the traits are synchronous

Both traits are sync, on purpose:

1. Search is CPU-over-RAM and never touches a backend, so there is no query I/O to
   make asynchronous.
2. Every backend *can* be expressed synchronously.
3. A sync trait is object-safe out of the box (`Box<dyn Persistence>`), which is what
   makes runtime selection by URL scheme work cleanly.

This matches nidus's [synchronous core](/guides/integrating/): async callers wrap the
store in `spawn_blocking`, exactly as they would a blocking embedded database.
