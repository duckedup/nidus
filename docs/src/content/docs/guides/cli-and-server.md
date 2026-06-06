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

```bash
# Prebuilt binary (fastest):
cargo binstall nidus

# Or build from source:
cargo install nidus --features cli
```

Both install a single `nidus` executable.

## Command line

Every command takes the store directory (`--dir`/`-d`) and the embedding
dimension (`--dim`), which must match the store. An optional `--distance` flag
selects the metric (`cosine`, `euclidean`, or `dot`; default `cosine`). Records,
query vectors, and filters are JSON; output is JSON on stdout.

```bash
# Create a collection (default cosine distance)
nidus create --dir ./store --dim 3 docs

# Or with Euclidean distance
nidus create --dir ./store --dim 3 --distance euclidean docs

# Upsert records (JSON array) from stdin or a --file
echo '[{"id":"a","vector":[1,0,0],"attrs":{"lang":{"Str":"rust"}}}]' \
  | nidus upsert --dir ./store --dim 3 docs

# Nearest-neighbour search — query vector on stdin
echo '[1,0,0]' | nidus search --dir ./store --dim 3 docs -k 5

# Search every collection at once (omit the collection names)
echo '[1,0,0]' | nidus search --dir ./store --dim 3

# Filter while searching (an AND of predicates, as JSON)
echo '[1,0,0]' | nidus search --dir ./store --dim 3 docs \
  --where '[{"Eq":["lang",{"Str":"rust"}]}]'

# List records by metadata filter (no vector query)
nidus list --dir ./store --dim 3 docs --where '[{"Eq":["lang",{"Str":"rust"}]}]'

# Inspect, maintain
nidus collections --dir ./store --dim 3
nidus get        --dir ./store --dim 3 docs
nidus stats      --dir ./store --dim 3
nidus compact    --dir ./store --dim 3
nidus delete     --dir ./store --dim 3 docs a b
```

Read-only commands (`search`, `get`, `collections`, `stats`) open the store
without taking the writer lock, so they can run alongside a writer such as a
running server.

## Server

`nidus serve` opens one store and serves it over HTTP until you stop it
(Ctrl-C, which flushes on the way out).

```bash
nidus serve --dir ./store --dim 768 --addr 127.0.0.1:7700
```

Pass `--read-only` to serve without taking the writer lock — useful for a
search-only process beside a separate writer.

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

The server holds the store behind a lock and runs each operation on a blocking
worker, the same pattern the library recommends for [driving it from async
code](/guides/integrating/). It is a thin wrapper: the storage model, durability,
and search semantics are exactly those of the library.
