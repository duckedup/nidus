---
title: Embedding in a host app
description: How a consuming tool maps its document type onto a nidus Record and bridges the synchronous API into an async runtime.
---

nidus knows nothing about your domain — it is a general-purpose vector store. A
consuming tool maps its own document type onto a nidus
[`Record`](/reference/api/#record) and, if it is async, wraps the store to bridge
into its runtime.

## Map your document onto a Record

A `Record` is an `id`, a `vector`, and an open `attrs` map. Every field of your
document either *is* one of those or fits an attr. Pick a stable `id` (it is the
upsert key), embed the content into `vector`, and project the rest into typed
attrs:

```rust
use std::collections::BTreeMap;
use nidus::{Record, Value};

struct Chunk {
    chunk_id: String,
    embedding: Vec<f32>,
    path: String,
    symbol: Option<String>,
    tags: Vec<String>,
}

fn to_record(c: Chunk) -> Record {
    let mut attrs = BTreeMap::new();
    attrs.insert("path".into(), Value::Str(c.path));
    // Null vs absent matters: emit Null for "computed, but none",
    // omit the key entirely for "not computed".
    attrs.insert("symbol".into(), match c.symbol {
        Some(s) => Value::Str(s),
        None => Value::Null,
    });
    attrs.insert("tags".into(), Value::List(c.tags));
    Record::new(c.chunk_id, c.embedding, attrs)
}
```

The [`Null`-vs-absent](/guides/search/#typed-metadata) distinction preserves
"computed-empty" versus "un-indexed" semantics — don't collapse them.

## Many collections, one dimension

A store has **one embedding space**. Use collections to partition documents that
share that space — e.g. `code`, `docs`, `commits` — and search any subset or all
of them in one ranked call. If you need two genuinely different embedding models
(different dimensions), use two stores.

## Bridging into async

nidus is **synchronous**: the hot path is CPU-bound (the dot-product sweep) and
file IO is blocking, so there is no async core to lock you into a runtime. This
is the same shape as a blocking embedded database handle, and you wrap it the
same way.

For concurrent searchers plus a single writer, share an `RwLock`:

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use nidus::Nidus;

let db = Arc::new(RwLock::new(Nidus::open_dir("/path/to/store", 768)?));

// In an async handler, run the blocking call on the blocking pool:
let db = Arc::clone(&db);
let hits = tokio::task::spawn_blocking(move || {
    let guard = db.blocking_read();
    guard.search("code", &query, &opts)
}).await??;
# anyhow::Ok(())
```

Reads (`search`, `get_all`, `footprint`) take `&self`; writes (`upsert`,
`delete`, `compact`, …) take `&mut self`. An `RwLock` therefore admits many
concurrent searchers and one exclusive writer. Use `spawn_blocking` (or your
runtime's equivalent) so a long sweep never blocks the async reactor.

## Two kinds of parallelism

There are two independent ways to put cores to work, and they suit opposite
workloads:

- **Query-level (the default).** Many `&self` searches run at once under
  `Arc<RwLock<Nidus>>` — one core per in-flight query. This is what you want for a
  server fielding concurrent requests; it maxes out **throughput**.
- **Intra-query (`Config::query_threads`).** A *single* large exact search is split
  across `query_threads` worker threads (`std::thread::scope`, per-chunk heaps
  merged at the end), cutting **one query's latency**. Opt-in — the default of `1`
  keeps search single-threaded.

```rust
use nidus::{Config, Nidus};

// Split each exact search across 4 worker threads.
let db = Nidus::open(Config::new("/path/to/store", 768).query_threads(4))?;
# anyhow::Ok(())
```

Pick one. Intra-query threads help when queries arrive **one at a time** against a
large store on otherwise-idle cores. The plain f32 scan is memory-bandwidth-bound,
so its speedup is real but sublinear (≈1.3–1.4× at 4–8 threads on 100k × 768).
Threads pay off best paired with [int8 quantization](/guides/search/#int8-scalar-quantization):
the int8 first pass moves 4× fewer bytes (compute- not bandwidth-bound), so it
splits across the same workers and scales to ≈2.4× at 4 threads. If you already have
query-level concurrency, leave `query_threads` at `1` — splitting each query then
just oversubscribes the cores your concurrent readers are already using.
