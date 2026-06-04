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
    Record { id: c.chunk_id, vector: c.embedding, attrs }
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
