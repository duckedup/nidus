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

nidus ships several implementations of each axis, all selected by a location string:

- **Persistence** — `LocalFs` (a directory, the default), plus `S3` (`s3://`, also R2/MinIO)
  and `Gcs` (`gs://`) object stores.
- **Memory tier** — `LocalRam` (the process heap, the default), plus `RedisTier`
  (`redis://` and the RESP-compatible kin: Valkey, KeyDB, DragonflyDB).

A file-backed store **runs over `LocalFs`**: its `data`/`log` segments are append handles
the backend hands out, and its `ann`/`fts` caches and writer lock go through the same
object operations — so the trait is the real substrate, not a wrapper bolted on the side.
An object-backed store runs over the same trait: each segment is an in-RAM buffer the
backend rewrites as one whole object on sync.

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

What nidus publishes here is the **replay-derived index** — the per-collection
`id → (row, attrs)` maps, the dead-row count, and the declared FTS schemas: the one piece
of in-RAM state with no other cache, since every process would otherwise rebuild it by
replaying the whole op log. The blob is **watermark-guarded** (the log byte offset + the
data row count), so a store adopts it on `open` only when it matches the just-opened
`data`/`log` exactly; otherwise it falls back to a normal replay. The store publishes a
fresh snapshot on `flush`.

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

### `RedisTier`

A **shared** tier over the Redis wire protocol, so several stateless workers pointed at
the same server skip the log replay on cold start — the first to open publishes the
working set, the rest adopt it. One blocking client (`redis-rs`, sync — no async runtime)
covers the whole RESP-compatible family: **Redis, Valkey, KeyDB, and DragonflyDB**.
Selected by URL scheme, plain TCP or TLS:

- `redis://host:6379` · `valkey://…` · `keydb://…` · `dragonfly://…` — plain TCP
- `rediss://…` · `valkeys://…` — TLS (reuses the same rustls + `ring` as the S3/GCS path)

Append `?prefix=<ns>` to namespace the keys (`<ns>:workingset`), so distinct stores can
share one server. The tier is a rebuildable cache: an unreachable or evicted server is
never fatal — the store just replays the log locally.

```rust
use nidus::open_memory_tier;

let tier = open_memory_tier("valkey://cache.internal:6379?prefix=docs")?;
# anyhow::Ok(())
```

> Memcached is intentionally **not** supported: it is eviction-only with no durability
> guarantees, the weakest fit for even a rebuildable cache.

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

**Persistence** locations: `file://<path>` or a bare path (local files),
**`s3://<bucket>[/<prefix>]`** (Amazon S3 and S3-compatible stores — R2, MinIO), and
**`gs://<bucket>[/<prefix>]`** (Google Cloud Storage). **Memory-tier** locations: `local`
(or `""` / `ram`) for the process heap, and the Redis family
**`redis://` / `rediss://` / `valkey://` / `valkeys://` / `keydb://` / `dragonfly://`**.

Credentials come from the standard environment — S3 from
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION` (plus `AWS_ENDPOINT_URL` for
R2/MinIO), GCS from a service-account key at `GOOGLE_APPLICATION_CREDENTIALS`, Redis from
the URL itself. An unrecognized scheme is rejected with a clear error rather than a silent
fallback, so a location string is always validated up front. The two axes compose: truth
in S3, working set shared via Valkey, search always in local RAM.

```bash
# Snapshot a local store straight to the cloud — the destination is just a location.
AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… AWS_REGION=us-east-1 \
  nidus backup --dir ./store --out s3://my-bucket/backups/store.tar.gz

GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json \
  nidus backup --dir ./store --out gs://my-bucket/backups/store.tar.gz
```

S3 and GCS are whole-object backends (`get`/`put`/`delete`/`list`) with no native append.
A store can still **run live** on them: each `data`/`log` segment is buffered in RAM and
rewritten as one whole object on every sync (an `ObjectAppender`). That makes a flush
`O(object)` rather than an `O(batch)` append — fine for the low-write-rate / dev /
small-scale use nidus targets, costly for write-heavy workloads. The writer lock on an
object store is **advisory** (a TTL'd `lock` object via get-then-put, not race-free), which
suits a single writer; concurrent writers want the snapshot mode below instead. Search is
unaffected (it is always CPU-over-local-RAM).

```bash
# A live store whose data/log live in S3 (creds from the AWS environment):
nidus upsert --dir ./meta --dim 768 --persistence s3://my-bucket/store docs < recs.json
nidus search --dir ./meta --dim 768 --persistence s3://my-bucket/store docs -k 5 < q.json
```

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
