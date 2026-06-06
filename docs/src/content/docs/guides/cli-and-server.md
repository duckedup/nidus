---
title: Command line & server
description: Use nidus from the terminal, or run it as a small HTTP server, with the `nidus` binary.
---

Besides the Rust library, nidus ships a `nidus` binary: a command-line tool for
working with a store directly, and `nidus serve`, a small HTTP server that exposes
the same operations over JSON. Both operate on an ordinary store directory — the
very same format the library reads and writes.

The binary is optional. The library has no dependency on it: `cargo add nidus`
pulls in only the pure-Rust core. The binary is built behind a `cli` feature, so
its extra dependencies are compiled only when you ask for them.

## Install

The fastest path needs **no Rust toolchain** — one command fetches a prebuilt
`nidus` binary for your platform from the latest release and drops it in
`~/.local/bin`:

```bash
curl -fsSL https://raw.githubusercontent.com/duckedup/nidus/main/install.sh | sh
```

Set `NIDUS_BIN_DIR` to install elsewhere, or `NIDUS_VERSION=vX.Y.Z` to pin a
version. Prefer not to pipe to a shell? [Read the script first](https://github.com/duckedup/nidus/blob/main/install.sh),
or grab the tarball straight from the [releases page](https://github.com/duckedup/nidus/releases/latest)
(`nidus-<target>.tar.gz`, or `.zip` on Windows), extract it, and put the `nidus`
binary on your `PATH`.

If you already have a Rust toolchain, either of these works too:

```bash
cargo binstall nidus                 # prebuilt binary, via cargo
cargo install nidus --features cli   # build from source
```

Every route installs the same single `nidus` executable.

## Quickstart: local search in four commands

From an empty directory to a working nearest-neighbour query — no Rust, no
config files, no daemon to register. Pick a store directory and an embedding
dimension once (here a toy `3`); after the store exists, `--dim` is remembered.

```bash
# 1. Install (see above)
curl -fsSL https://raw.githubusercontent.com/duckedup/nidus/main/install.sh | sh

# 2. Create a collection. The dimension is pinned here, at creation.
nidus create --dir ./store --dim 3 docs

# 3. Add a couple of records (id + vector + any typed metadata).
echo '[
  {"id":"a","vector":[1,0,0],"attrs":{"lang":{"Str":"rust"}}},
  {"id":"b","vector":[0,1,0],"attrs":{"lang":{"Str":"go"}}}
]' | nidus upsert --dir ./store docs

# 4. Search. No --dim needed — it is read from the store.
echo '[1,0,0]' | nidus search --dir ./store docs -k 2
```

The last command prints ranked hits as JSON. That is a complete local vector
store: a single `./store` directory you can copy, back up, or delete. To query
it over HTTP instead, point [`nidus serve`](#server) at the same directory.

## Command line

Every command takes the store directory (`--dir`/`-d`). The embedding dimension
is pinned in the store the first time you create a collection, so **`--dim` is
only needed at creation** — afterwards it is read from the store. (Pass it anyway
and it is checked: a mismatch is a hard error, not a silent surprise.) The
`--distance` metric (`cosine`, `euclidean`, or `dot`) works the same way: chosen
at creation (default `cosine`), inferred thereafter. Records, query vectors, and
filters are JSON; output is JSON on stdout.

```bash
# Create a collection — dimension pinned here (default cosine distance)
nidus create --dir ./store --dim 3 docs

# Or with Euclidean distance
nidus create --dir ./store --dim 3 --distance euclidean docs

# Upsert records (JSON array) from stdin or a --file — no --dim needed
echo '[{"id":"a","vector":[1,0,0],"attrs":{"lang":{"Str":"rust"}}}]' \
  | nidus upsert --dir ./store docs

# Nearest-neighbour search — query vector on stdin
echo '[1,0,0]' | nidus search --dir ./store docs -k 5

# Search every collection at once (omit the collection names)
echo '[1,0,0]' | nidus search --dir ./store

# Filter while searching (an AND of predicates, as JSON)
echo '[1,0,0]' | nidus search --dir ./store docs \
  --where '[{"Eq":["lang",{"Str":"rust"}]}]'

# List records by metadata filter (no vector query); --offset/-n paginate
nidus list --dir ./store docs --where '[{"Eq":["lang",{"Str":"rust"}]}]'
nidus list --dir ./store docs --offset 100 -n 100   # next page

# Inspect, maintain
nidus collections --dir ./store
nidus get        --dir ./store docs
nidus stats      --dir ./store
nidus compact    --dir ./store
nidus delete     --dir ./store docs a b
```

Read-only commands (`search`, `get`, `collections`, `stats`) open the store
without taking the writer lock, so they can run alongside a writer such as a
running server.

## Server

`nidus serve` opens one store and serves it over HTTP until you stop it
(Ctrl-C, which flushes on the way out).

```bash
# --dim is only needed if the store doesn't exist yet; otherwise it's inferred.
nidus serve --dir ./store --dim 768 --addr 127.0.0.1:7700
```

Pass `--read-only` to serve without taking the writer lock — useful for a
search-only process beside a separate writer.

### Authentication

The server is unauthenticated by default, which is fine on `127.0.0.1`. The moment
you bind a non-local address, set `--token <secret>` (or the `NIDUS_TOKEN` env
var): every request except `GET /health` must then carry
`Authorization: Bearer <secret>`, and anything else gets `401`.

```bash
nidus serve --dir ./store --addr 0.0.0.0:7700 --token "$NIDUS_TOKEN"
curl -s localhost:7700/search -H "authorization: Bearer $NIDUS_TOKEN" -d '…'
```

### Request size

Each request body is buffered in memory, so the body-size limit is also the
largest single upsert. It defaults to 256 MiB; raise or lower it with
`--max-body-bytes <n>`.

The endpoints map one-to-one onto the library API:

| Method & path | Operation |
| --- | --- |
| `GET /health` | liveness check |
| `GET /collections` | list collections |
| `POST /collections/{name}` | create a collection |
| `DELETE /collections/{name}` | drop a collection |
| `GET /collections/{name}/meta` | read collection metadata |
| `PUT /collections/{name}/meta` | set collection metadata |
| `POST /collections/{name}/upsert` | upsert records |
| `POST /collections/{name}/delete` | delete by ids or filter |
| `GET /collections/{name}/records` | all records in a collection |
| `POST /search` | nearest-neighbour search |
| `POST /list` | metadata-only query (no vector) |
| `POST /flush` | flush to disk |
| `POST /compact` | reclaim dead rows |

A search request takes a query vector, an optional `scope` (a list of collection
names; empty means the whole store), and the usual options:

```bash
curl -s localhost:7700/search -H 'content-type: application/json' -d '{
  "query": [1, 0, 0],
  "scope": ["docs"],
  "top_k": 5,
  "min_score": 0.2,
  "filter": [{"Eq": ["lang", {"Str": "rust"}]}]
}'
```

Upsert and delete mirror the library:

```bash
curl -s localhost:7700/collections/docs/upsert \
  -H 'content-type: application/json' \
  -d '{"records": [{"id": "a", "vector": [1,0,0], "attrs": {}}]}'

curl -s localhost:7700/collections/docs/delete \
  -H 'content-type: application/json' -d '{"ids": ["a"]}'
```

The server holds the store behind a read/write lock and runs each operation on a
blocking worker, the same pattern the library recommends for [driving it from
async code](/guides/integrating/). Reads (search, list, get) run concurrently;
writes take the store exclusively. It is a thin wrapper: the storage model,
durability, and search semantics are exactly those of the library.
