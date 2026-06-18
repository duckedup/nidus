---
title: Memory stores
description: Share a store's in-RAM working set across processes with Redis (or Valkey, KeyDB, DragonflyDB) so each worker loads it instead of rebuilding — or keep it in local memory, the default.
---

When nidus opens a store, it builds an in-memory index of your data and serves searches
from it. By default that index lives in the current process and nothing else sees it. If
you run **several processes against the same store** — say a few `nidus serve` workers
behind a load balancer — you can have them **share that index through Redis** (or a
drop-in alternative: **Valkey**, **KeyDB**, **DragonflyDB**) instead of each rebuilding it
from scratch on startup.

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

// Share the warm index via Redis / Valkey / KeyDB / Dragonfly:
let shared = Nidus::open(Config::new("./store", 768).memory("redis://cache:6379"))?;
# anyhow::Ok(())
```

The first worker to open the store publishes the warm index; the rest **load it instead
of replaying the whole change log** — a faster cold start when you have many workers.
Search itself never touches Redis: every worker still scans its own local copy of the
vectors, so this changes startup time and sharing, never search results or speed.

## Choosing a memory store

| You want… | Use this location | Notes |
|---|---|---|
| Local memory (default) | omit it, or `local` | The index lives in this process only. |
| Shared via Redis | `redis://host:6379` | Plain connection. |
| Shared via Valkey / KeyDB / Dragonfly | `valkey://…`, `keydb://…`, `dragonfly://…` | Same Redis protocol; one client covers them all. |
| Shared over TLS | `rediss://…` (or `valkeys://…`) | Encrypted connection. |

Add `?prefix=<name>` to the URL to namespace the keys, so several stores can share one
Redis server without colliding: `redis://cache:6379?prefix=docs`.

It is a **cache, never the source of truth.** If the Redis server is down, evicts the
entry, or is empty, nothing breaks — the worker just rebuilds the index from the store's
own `data`/`log` (exactly as the default does). So you never risk data by pointing at it.

> **Why not Memcached?** It evicts whenever it likes and offers no guarantees, which makes
> it the weakest fit even for a throwaway cache — so nidus doesn't support it. Use Redis or
> any of its drop-in kin above.

## When this helps (and when it doesn't)

- **Helps:** many stateless search workers over one store, restarted often — they share
  one warm copy instead of each replaying the change log on boot.
- **Doesn't matter:** a single process. The default local memory is already the fastest
  thing; adding Redis just adds a network hop on startup for no benefit.

This pairs with the [storage backend](/guides/storage-backends/): you might keep the
durable data in S3 and share the warm index via Redis, while search always runs in each
worker's local RAM.

## Writing your own memory store

The options above implement one small, synchronous Rust trait,
`nidus::backend::MemoryTier` — just `load` and `store` of an opaque blob. If you want to
share the warm index through something nidus doesn't ship, implement that trait; see the
[API reference](/reference/api/).
