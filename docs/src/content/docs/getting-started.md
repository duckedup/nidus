---
title: Getting started
description: Add nidus to a Rust project, open a store, index records, and run your first search.
---

nidus is built for development and small-scale use. You add it as a dependency,
open a store in a directory of your choosing, and call methods. There is nothing
to install, no daemon to run, and no network.

## Add the dependency

```toml
# Cargo.toml
[dependencies]
nidus = "0.4"
anyhow = "1"     # nidus returns anyhow::Result
```

nidus requires **Rust 1.96+** (edition 2024). It pulls in only popular pure-Rust
crates — there is no C to compile and no native library to link, so the build is
seconds, not minutes.

## Open a store

A store is a single directory. The **location is always your choice** — nidus
contributes no path defaults, env vars, or hidden directories. The **embedding
dimension is pinned** at creation and checked on every reopen.

```rust
use nidus::{Nidus, Config};

// Open (or create) a store with a pinned 768-dimensional embedding space.
let mut db = Nidus::open(Config::new("/path/to/store", 768))?;
db.create_collection("code")?;
# anyhow::Ok(())
```

Shorthand constructors exist for the common cases:

```rust
use nidus::Nidus;

// Same as Config::new(dir, dim) with all defaults.
let db = Nidus::open_dir("/path/to/store", 768)?;

// A throwaway store with no files — handy for tests.
let db = Nidus::open_in_memory(768)?;
# anyhow::Ok(())
```

## Index records

A `Record` is a caller-supplied `id`, its `vector` (length must equal the store
dimension), and an open map of typed `attrs`. Upserts are **idempotent by id**
within a collection — re-upserting the same id replaces it.

```rust
use std::collections::BTreeMap;
use nidus::{Record, Value};

let mut attrs = BTreeMap::new();
attrs.insert("path".into(), Value::Str("src/auth/login.rs".into()));
attrs.insert("lines".into(), Value::Int(42));

db.upsert("code", &[Record {
    id: "a".into(),
    vector: vec![/* 768 f32s */],
    attrs,
}])?;
# anyhow::Ok(())
```

Pass a slice to upsert a whole batch in one durable, all-or-nothing call.

## Search

`search` takes a [`Scope`](/reference/api/#scope), a query vector, and
[`SearchOpts`](/reference/api/#searchopts). It returns ranked `Hit`s — each
carrying its source collection, id, cosine score, and the matched record's
attrs.

```rust
use nidus::SearchOpts;

let hits = db.search("code", &query, &SearchOpts {
    top_k: 5,
    ..Default::default()
})?;

for h in &hits {
    println!("{:.3}  [{}] {}", h.score, h.collection, h.id);
}
# anyhow::Ok(())
```

Search the **whole store at once**, with a metadata filter and a score floor:

```rust
use nidus::{Scope, SearchOpts, Filter, Predicate};

let opts = SearchOpts {
    top_k: 10,
    filter: Filter(vec![Predicate::Glob("path".into(), "src/auth/*".into())]),
    min_score: Some(0.5),
};
let hits = db.search(Scope::All, &query, &opts)?;
# anyhow::Ok(())
```

Scoping the whole store is sound because every collection shares one embedding
space — see [Search & filters](/guides/search/).

## Run the example

The repository ships an end-to-end demo:

```bash
cargo run --example demo
```

## Where to next

- [How it works](/guides/how-it-works/) — the storage model and search path.
- [Storage & durability](/guides/storage/) — the on-disk format and crash safety.
- [Configuration](/reference/configuration/) — every knob on `Config`.
- [API reference](/reference/api/) — the full surface.
