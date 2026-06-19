# nidus

A small, pure-Rust **vector store for development and small-scale use**.
Nearest-neighbour search — cosine, dot, or Euclidean — over a single append-only
directory, exact by default or approximate (HNSW/IVF) when you opt in, with typed
metadata filters and many logical collections sharing one embedding space. No SQL,
no query engine, and a build measured in seconds, not minutes.

> _nidus_ (Latin, "nest") — a small place where things are kept safe.

## Why it exists

nidus is the local storage leg for semantic-search and indexing tools: chunk some
source → embed each chunk → store the vectors + metadata → ask for nearest
neighbours. The obvious off-the-shelf options fail the **build-and-ship** test, not
the functionality test:

- **DuckDB** (via `libduckdb-sys`) bundles a large C++ source tree and compiles it
  from scratch — multi-minute cold builds, a required C++ toolchain, a bloated
  binary, and FFI that can't run under Miri. A vector workload uses ~1% of it.
- **LanceDB** is "written in Rust" yet still takes ~10 minutes to compile, because
  it drags in Arrow + DataFusion (a full SQL engine) + a columnar format — hundreds
  of crates to do `ORDER BY distance LIMIT k`.

The workload is a *vector store, not a database*. nidus is that store and nothing
more, so it **compiles in seconds** and embeds as a normal Rust dependency.

### The constraints are the product

The bar is **build-and-ship speed**, not zero-C absolutism. The enemy is the
*multi-minute* C/C++ tree (DuckDB) or hundred-crate graph (LanceDB) — not a small,
fast dependency.

- **Builds in seconds** — the whole crate, with every backend (local files, S3, GCS, and
  the Redis/Valkey memory tier), compiles in seconds (CI asserts well under a minute). The
  only native code is `ring` (the TLS used by the S3/GCS backends and `rediss://` — a small
  C+asm compile); never a bundled C++ tree, vendored OpenSSL, or `aws-lc`. The Redis client
  is sync/pure-Rust (no tokio).
- **Near-zero `unsafe` in our code** (`#![deny(unsafe_code)]`). The one exception is the
  opt-in [`Config::mmap`](https://nidus.duckedup.org/reference/configuration/#mmap) path —
  a single scoped `mmap` call for serving large stores from disk; every other `unsafe` is a
  hard compile error.
- **Pure-Rust core** — the local store and search path are pure Rust with no native
  library; the cloud backends are sans-IO clients (`rusty-s3`/`tame-gcs`) over a small
  blocking HTTP client.
- **Miri-checkable** — all of nidus's own logic, including the local file IO, runs
  under Miri (only the network TLS paths are excluded).

## Quick start

```toml
[dependencies]
nidus = "0.26"
```

```rust
use std::collections::BTreeMap;
use nidus::{Nidus, Config, Record, SearchOpts, Scope, Value, Filter, Predicate};

// Open (or create) a store. The directory is always the caller's choice;
// the dimension is pinned for the life of the store.
let mut db = Nidus::open(Config::new("/path/to/store", 4))?;
db.create_collection("code")?;

// Index some records: id + embedding + arbitrary typed metadata.
let mut attrs = BTreeMap::new();
attrs.insert("path".into(), Value::Str("src/auth/login.rs".into()));
db.upsert("code", &[Record { id: "a".into(), vector: vec![1.0, 0.0, 0.0, 0.0], attrs }])?;

// Nearest neighbours (cosine), top-k.
let hits = db.search("code", &[1.0, 0.0, 0.0, 0.0], &SearchOpts { top_k: 5, ..Default::default() })?;
for h in &hits {
    println!("{:.3}  [{}] {}", h.score, h.collection, h.id);
}

// Search the whole store at once, with a metadata filter + score floor.
let opts = SearchOpts {
    top_k: 10,
    filter: Filter(vec![Predicate::Glob("path".into(), "src/auth/*".into())]),
    min_score: Some(0.5),
};
let hits = db.search(Scope::All, &[1.0, 0.0, 0.0, 0.0], &opts)?;
# anyhow::Ok(())
```

See [`examples/demo.rs`](examples/demo.rs) for an end-to-end run (`cargo run
--example demo`).

## What it does

- **Exact or approximate search** — exact by default (100% recall, fast at the target
  scale of ≤ a few million vectors, comfortably in RAM). Score by cosine, dot, or
  Euclidean (cosine the default; cosine vectors are unit-normalized on insert, so a
  score is plain similarity in `[-1, 1]`). Opt into an approximate index (HNSW or IVF)
  or int8 quantization to trade some recall for speed at larger scale.
- **Scoped search** — query one collection, a subset, or the **whole store** in one
  call, merged into a single ranking. Sound because every collection shares one
  embedding space (one pinned dimension).
- **Typed metadata + filters** — attach `Str`/`Int`/`Bool`/`List`/`Null` attributes
  and filter with `Eq` / `Glob` / `In` predicates before scoring.
- **Idempotent upserts** by caller-supplied id; `delete`, `delete_where`, per-
  collection metadata.
- **Crash-safe & durable** — an append-only flat-`f32` `data` segment plus a framed,
  CRC-checked op `log` (the commit record). A crash loses at most the in-flight
  batch; a torn tail is recovered on open. Cross-process readers get a consistent,
  lock-free snapshot (`OpenMode::ReadOnly`).
- **Synchronous, runtime-agnostic** — the hot path is CPU-bound, so there's no async
  core to lock you into a runtime. `Arc<RwLock<Nidus>>` gives concurrent searchers +
  one writer; async callers bridge with `spawn_blocking`.

## Command line & server

The same crate ships an optional `nidus` binary: a CLI for working with a store
directly, and `nidus serve`, an HTTP server exposing the full store — create,
upsert, search, inspect, maintain — over JSON. The binary is built behind a `cli`
feature, so `cargo add nidus` stays pure: the library never pulls the binary's
dependencies.

```bash
# Install — no Rust toolchain needed (prebuilt binary for your platform)
curl -fsSL https://raw.githubusercontent.com/duckedup/nidus/main/install.sh | sh
# …or, with cargo: `cargo binstall nidus` / `cargo install nidus --features cli`

# Use it on a store directory (records/queries are JSON). --dim is pinned at
# creation, then inferred from the store, so later commands don't repeat it.
nidus create  --dir ./store --dim 3 docs
echo '[{"id":"a","vector":[1,0,0],"attrs":{}}]' | nidus upsert --dir ./store docs
echo '[1,0,0]' | nidus search --dir ./store docs -k 5

# Snapshot the whole store to one portable .tar.gz (safe while a writer runs),
# and restore it — handy before an upgrade or as a cron job.
nidus backup  --dir ./store --out ./store.tar.gz
nidus restore --in ./store.tar.gz --dir ./restored
```

Or drive the same store over the network — no Rust toolchain on the client, just
HTTP and JSON:

```bash
nidus serve --dir ./store --dim 3 --addr 127.0.0.1:7700

curl -s -X POST localhost:7700/collections/docs
curl -s localhost:7700/collections/docs/upsert -H 'content-type: application/json' \
  -d '{"records": [{"id": "a", "vector": [1,0,0], "attrs": {}}]}'
curl -s localhost:7700/search -H 'content-type: application/json' \
  -d '{"query": [1,0,0], "top_k": 5}'
```

The server shares the library's storage model, durability, and search semantics.
See the [command-line](https://nidus.duckedup.org/guides/cli-and-server/) and
[HTTP server & API](https://nidus.duckedup.org/guides/http-server/) guides.

## Performance

Every vector store ships a benchmark proving it's the fastest, on synthetic data
that looks nothing like your workload. It's a genre. Here's ours — and yes, we win
our own benchmark, that's how this works.

Exact brute-force cosine KNN, 100k vectors, single thread, measured against
DuckDB (`array_cosine_similarity`) and LanceDB (`bypass_vector_index`) — both pinned
to the same exact search, so all three return the same neighbours. The harness
computes its own independent ground truth and reports **recall@k** for every engine
(including nidus), so none is trusted as the oracle. Numbers are query p50; lower is
better.

| n=100k | top_k | **nidus** | LanceDB | DuckDB | recall |
|--------|------:|----------:|--------:|-------:|:------:|
| dim=384 |  10 | **5.44 ms** | 12.29 ms | 32.29 ms | 100% |
| dim=384 | 100 | **5.53 ms** | 28.52 ms | 30.59 ms | 100% |
| dim=768 |  10 | **8.09 ms** | 24.78 ms | 69.54 ms | 100% |
| dim=768 | 100 | **8.57 ms** | 53.16 ms | 64.99 ms | 100% |

All three are exact (recall 100%); nidus is the fastest in every cell while being the
one that compiles in seconds. The kernel is plain safe Rust — an
8-lane chunked dot the optimizer can vectorize, an allocation-free top-k scan, and a
storage-order (prefetcher-friendly) sweep of the matrix. Reproduce with
`just bench all` (see [`benchmarks/`](benchmarks/); the heavy DuckDB/LanceDB deps are
quarantined off nidus's own build path). Synthetic data on an Apple Silicon laptop —
useless, like all benchmarks, but there it is.

## On-disk layout

A store is a directory:

```
<dir>/
  data    append-only, fixed-stride, row-major f32 matrix (header pins dimension)
  log     append-only framed op stream: [len][bincode(Op)][crc32] — the commit record
  lock    O_EXCL writer-exclusion lock file
```

`open` reads `data` into RAM and replays `log` into an in-RAM index
(`collection → { id → (row, attrs) }`). Search never touches disk.

## Configuration

```rust
use std::time::Duration;
use nidus::{Config, Fsync, OpenMode};

let cfg = Config::new("/path/to/store", 768)
    .fsync(Fsync::PerBatch)          // durability granularity (default)
    .open_mode(OpenMode::ReadWrite)  // ReadOnly = no lock, search-only
    .auto_compact(Some(0.5))         // compact on open above this dead-row ratio
    .lock_ttl(Duration::from_secs(60));
```

The store **location is always the caller's choice** — nidus contributes no path
defaults, env vars, or hidden directories.

## Development

```bash
just test    # all tests (pure library)
just ci       # fmt-check + clippy (-D warnings) + test (pure library)
just miri     # undefined-behavior check (nightly; all of nidus's own logic runs)
just demo     # the end-to-end example
just deps     # the dependency tree (stays short)

just ci-cli   # the same gate for the opt-in `cli` feature (binary + server)
just serve ./store 768   # run `nidus serve` from the checkout
```

The core recipes keep the seconds-long build path intact; the `cli` feature (which
pulls clap + the tokio/axum stack) has its own opt-in recipes. Miri runs all of
nidus's own logic, including the local file IO and the in-RAM object-store/memory-tier
paths — only the network paths (S3/GCS TLS, the Redis socket) and the opt-in `mmap`
syscall are outside its reach.

Rust 1.96+ (pinned via `rust-toolchain.toml`), edition 2024.

## Design

The full design — data model, on-disk format, durability/concurrency model, the
opt-in modes (approximate ANN/HNSW + IVF, scalar/binary quantization, memory-mapped
larger-than-RAM stores), and the remaining deferred seams — lives in
[`SPEC.md`](SPEC.md). Each module also carries its own contract in `src/<module>/SPEC.md`.

## License

[MIT](LICENSE)
