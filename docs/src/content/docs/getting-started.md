---
title: Getting started
description: Add nidus to a Rust project, open a store, index records, and run your first vector, full-text, or hybrid search.
---

nidus is a pure-Rust vector store with full-text search that runs anywhere Rust runs: in
process as a library, behind `nidus serve` over HTTP, as an MCP server, or in a browser on
wasm. Its bytes live on local disk or in object storage (S3, GCS), with an optional shared
memory tier (Redis, Valkey).

This guide takes the in-process library path: add nidus as a dependency, open a store in a
directory of your choosing, and call methods directly. There is nothing to install, no
daemon to run, and no network for this path.

A store holds records: a vector, an id, and an open map of typed metadata. You can
search it three ways, and they share one set of results, filters, and scoping:

- **[Full-text search](/guides/full-text-search/)** ranks by keyword with BM25. No
  embedder, no model, no API key.
- **[Vector search](/guides/vector-search/)** ranks by meaning, over embeddings you supply.
- **[Hybrid search](/guides/hybrid-search/)** runs both legs and fuses them into one
  ranking.

This page takes the vector path, because it exercises the most of the store. If you
only want keyword search, [full-text search](/guides/full-text-search/) stands alone
and needs no embeddings at all.

:::tip[Prefer not to write Rust?]
Install the `nidus` command-line tool (no toolchain required) and stand up a working
local store in four commands. See
[Quickstart: local search in four commands](/guides/command-line/#quickstart-local-search-in-four-commands).
:::

## Add the dependency

```toml
# Cargo.toml
[dependencies]
nidus = "0.104"
anyhow = "1"     # nidus returns anyhow::Result
```

nidus requires **Rust 1.98+** (edition 2024). It pulls in only popular, mostly
pure-Rust crates: the local store and search path are pure Rust, and the bundled
S3/GCS backends add only a small TLS compile (`ring`), never a bundled C++ tree.

If you only want the storage and search core, `cargo add nidus --no-default-features`
gives you the four core crates plus the storage backends: no HTTP server, no async
runtime, no embed, summarize, or rerank providers.

## Open a store

A store is a single directory. The **location is always your choice**: nidus
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

// A throwaway store with no files, handy for tests.
let db = Nidus::open_in_memory(768)?;
# anyhow::Ok(())
```

## Index records

A `Record` is a caller-supplied `id`, its `vector` (length must equal the store
dimension), and an open map of typed `attrs`. Upserts are **idempotent by id**
within a collection: re-upserting the same id replaces it.

```rust
use std::collections::BTreeMap;
use nidus::{Record, Value};

let mut attrs = BTreeMap::new();
attrs.insert("path".into(), Value::Str("src/auth/login.rs".into()));
attrs.insert("lines".into(), Value::Int(42));

db.upsert("code", &[Record::new("a", vec![/* 768 f32s */], attrs)])?;
# anyhow::Ok(())
```

Pass a slice to upsert a whole batch in one durable, all-or-nothing call. A record
may also carry no embedding (`Record::text_only(id, attrs)`) for a document indexed
purely by [full-text search](/guides/full-text-search/).

## Search

`search` takes a [`Scope`](/reference/api/#scope), a query vector, and
[`SearchOpts`](/reference/api/#searchopts). It returns ranked `Hit`s, each
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
    ..Default::default()
};
let hits = db.search(Scope::All, &query, &opts)?;
# anyhow::Ok(())
```

Scoping the whole store is sound because every collection shares one embedding
space; see [Search & filters](/guides/vector-search/).

## Query with SQL

The same search compiles from a `SELECT` statement instead of a typed call, for
anywhere a string is easier to reach for than a struct (a CLI one-liner, a notebook):

```rust
use nidus::QueryAnswer;

let QueryAnswer::Hits { hits, .. } = db.query(
    "SELECT * FROM code WHERE path LIKE 'src/auth/*' \
     ORDER BY knn([/* 768 f32s */]) WITH (min_score = 0.5) LIMIT 10",
)?
else {
    unreachable!("a search-ranked query always answers Hits")
};
# anyhow::Ok(())
```

`query` compiles the statement to the same `SearchOpts`/`Filter` values the typed call
above builds by hand, then runs it. See [Query with SQL](/guides/query-with-sql/) for
the full grammar and what each clause compiles to.

## Run the example

The repository ships an end-to-end demo:

```bash
cargo run --example demo
```

## Where to next

- [How it works](/guides/how-it-works/): the storage model and search path.
- [Storage & durability](/guides/storage/): the on-disk format and crash safety.
- [Configuration](/reference/configuration/): every knob on `Config`.
- [API reference](/reference/api/): the full surface.
