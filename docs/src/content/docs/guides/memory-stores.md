---
title: Memory stores
description: Share a store's in-RAM working set across processes with Redis (or Valkey, KeyDB, DragonflyDB) so each worker loads it instead of rebuilding — or keep it in local memory, the default.
---

When nidus opens a store, it builds an in-memory index of your data and serves searches
from it. By default that index lives only in the current process. If you run **several
processes against one store** — say a few `nidus serve` workers behind a load balancer —
you can have them **share that index through Redis** (or a drop-in alternative) so each
worker loads it on startup instead of rebuilding it from scratch.

```bash
# Default: the index lives only in this process.
nidus serve --dir ./store --dim 768

# Shared: workers publish the warm index to Redis and load it on startup.
nidus serve --dir ./store --dim 768 --memory redis://cache:6379
```

From the Rust library it is one builder call:

```rust
use nidus::{Config, Nidus};

// Local memory (the default):
let db = Nidus::open(Config::new("./store", 768))?;

// Share the warm index via Redis:
let shared = Nidus::open(Config::new("./store", 768).memory("redis://cache:6379"))?;
# anyhow::Ok(())
```

The first worker to open the store publishes the warm index; the rest **load it instead
of replaying the whole change log**. Search itself never touches Redis — every worker
still scans its own local copy of the vectors — so this changes startup time and sharing,
never search results or speed. And it's only ever a cache: if the server is down, empty,
or evicts the entry, the worker just rebuilds from the store's own data, so you never risk
anything by pointing at it.

## Choosing a memory store

| You want… | Use this location | Notes |
|---|---|---|
| Local memory (default) | omit it, or `local` | The index lives in this process only. |
| Shared via Redis | `redis://host:6379` | Plain connection. |
| Shared via Valkey / KeyDB / Dragonfly | `valkey://…`, `keydb://…`, `dragonfly://…` | Same Redis protocol; one client covers them all. |
| Shared over TLS | `rediss://…` (or `valkeys://…`) | Encrypted connection. |
| Redis / Valkey **Cluster** | `redis://seed:6379?cluster=true` | Slot-routed across nodes; the host is a seed. |

### Local memory (the default)

The index lives in the current process and is shared between its threads, but no other
process sees it. Nothing to configure — just don't pass `--memory`. This is the fastest
option and the right one for a single process.

```bash
nidus serve --dir ./store --dim 768
```

### Redis

Point `--memory` at a Redis server and several workers over the same store share one warm
index. The first to start publishes it; the rest skip the log replay and load it.

```bash
# Two workers over the same store, sharing the warm index:
nidus serve --dir ./store --dim 768 --memory redis://cache:6379   # worker 1
nidus serve --dir ./store --dim 768 --memory redis://cache:6379   # worker 2
```

### Valkey, KeyDB, and DragonflyDB

These speak the same wire protocol as Redis, so the same client drives them — just change
the scheme. Use whichever you already run; nidus treats them identically.

```bash
nidus serve --dir ./store --dim 768 --memory valkey://cache:6379
nidus serve --dir ./store --dim 768 --memory keydb://cache:6379
nidus serve --dir ./store --dim 768 --memory dragonfly://cache:6379
```

### TLS connections

For an encrypted connection to a managed or remote server, use `rediss://` (or
`valkeys://`). It reuses the same TLS stack as the S3/GCS backends.

```bash
nidus serve --dir ./store --dim 768 --memory rediss://user:pass@cache.example.com:6380
```

### Redis / Valkey Cluster

For a sharded, highly-available deployment, point `--memory` at a **cluster** with
`?cluster=true`. The host is a seed node — nidus discovers the rest of the topology and the
client routes each key to its slot's node (handling `MOVED`/`ASK` redirects). Works with any
RESP-compatible cluster (Redis Cluster, Valkey Cluster), plain or over TLS.

```bash
nidus serve --dir ./store --dim 768 --memory redis://seed-node:6379?cluster=true
nidus serve --dir ./store --dim 768 --memory rediss://seed:6380?cluster=true&prefix=docs
```

List several **comma-separated seed nodes** so discovery still bootstraps if one is down at
startup (they share the scheme, credentials, and database):

```bash
nidus serve --dir ./store --dim 768 \
  --memory 'redis://node-a:6379,node-b:6379,node-c:6379?cluster=true'
```

### Sharing one server across stores

Add `?prefix=<name>` to namespace the keys, so several different stores can share one
Redis server without colliding:

```bash
nidus serve --dir ./docs  --dim 768 --memory redis://cache:6379?prefix=docs
nidus serve --dir ./notes --dim 768 --memory redis://cache:6379?prefix=notes
```

## When to share (and when not to)

- **Share** when many stateless search workers run over one store and restart often — they
  load one warm copy instead of each replaying the change log on boot.
- **Don't bother** for a single process: local memory is already the fastest thing, and
  adding Redis just puts a network hop on startup for no benefit.

This pairs with the [storage backend](/guides/storage-backends/): you might keep the
durable data in S3 and share the warm index via Redis, while search always runs in each
worker's local RAM.

> **Why not Memcached?** It evicts whenever it likes and offers no guarantees, the weakest
> fit even for a throwaway cache — so nidus doesn't support it. Use Redis or one of its
> drop-in kin above.

## Writing your own memory store

The options above implement one small, synchronous Rust trait,
`nidus::backend::MemoryTier` — just `load` and `store` of an opaque blob. To share the
warm index through something nidus doesn't ship, implement that trait; see the
[API reference](/reference/api/).
