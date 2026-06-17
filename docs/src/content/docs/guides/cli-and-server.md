---
title: Command line
description: Use nidus from the terminal with the `nidus` binary — create, upsert, search, inspect, back up, and restore a store directory.
---

Besides the Rust library, nidus ships a `nidus` binary: a command-line tool for
working with a store directly. It operates on an ordinary store directory — the
very same format the library reads and writes. The same binary also runs an HTTP
server; that has its own [HTTP server & API](/guides/http-server/) page.

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
store: a single `./store` directory you can copy, back up, or delete. To drive the
same store over the network instead, point
[`nidus serve`](/guides/http-server/) at the same directory.

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

# Full-text search (BM25): declare which fields are indexed, then query by text
nidus set-fts-schema --dir ./store docs --field body --field title
nidus text-search --dir ./store body "running quickly" -k 5
nidus text-search --dir ./store body "rust" --in docs --where '[{"Eq":["lang",{"Str":"rust"}]}]'

# Hybrid search: fuse a vector (stdin) and a BM25 text query with RRF
echo '[1,0,0]' | nidus hybrid-search --dir ./store body "vector database" -k 5

# Inspect, maintain
nidus collections --dir ./store
nidus get        --dir ./store docs
nidus stats      --dir ./store
nidus compact    --dir ./store
nidus delete     --dir ./store docs a b

# Snapshot the whole store to one portable .tar.gz, and restore it
nidus backup     --dir ./store --out ./store.tar.gz
nidus restore    --in ./store.tar.gz --dir ./restored
```

Read-only commands (`search`, `list`, `get`, `collections`, `stats`, and
`backup`) open the store without taking the writer lock, so they can run
alongside a writer such as a running server.

## Approximate search (ANN)

By default search is exact brute-force. For larger stores you can opt into an
in-memory approximate-nearest-neighbour index with `--ann hnsw` or `--ann ivf`.
Unlike `--dim` and `--distance`, the ANN choice is **not** recorded in the store
header — it is a property of how you *open* the store, so pass it on every command
that should build or consult the index (including `serve`):

```bash
# Upsert into a store whose index you maintain as an HNSW graph
echo '[{"id":"a","vector":[1,0,0],"attrs":{}}]' \
  | nidus upsert --dir ./store --dim 3 --ann hnsw docs

# Search via the index (over-fetch, then exact rerank of the survivors)
echo '[1,0,0]' | nidus search --dir ./store --ann hnsw docs -k 5

# Tune the knobs (all optional; defaults are sensible)
echo '[1,0,0]' | nidus search --dir ./store --ann hnsw \
  --ann-ef-search 128 --ann-overscan 8 docs
```

HNSW knobs: `--ann-m`, `--ann-ef-construction`, `--ann-ef-search`. IVF knobs:
`--ann-n-lists`, `--ann-n-probe`. `--ann-overscan` and `--ann-seed` apply to both.
Candidate *selection* is approximate, but the final ranking is always the exact
score over the over-fetched survivors. `nidus stats --ann …` echoes the active
configuration. ANN cannot be combined with quantization.

## Backup & restore

A store is just a directory, so you can always copy it by hand — but `nidus
backup` packages the whole thing into a single compressed `.tar.gz` you can
stash before an upgrade or hand to a cron job, and `nidus restore` brings it
back.

```bash
# Snapshot ./store into one portable archive.
nidus backup --dir ./store --out ./store.tar.gz

# Omit --out and you get a sortable, timestamped name in the current directory,
# e.g. store-1781063324.tar.gz — handy for keeping a series of snapshots.
nidus backup --dir ./store

# Restore into a directory. If the target already holds a store you are asked to
# confirm; pass -y to overwrite without prompting.
nidus restore --in ./store.tar.gz --dir ./restored
nidus restore --in ./store.tar.gz --dir ./store --yes
```

The archive's `--out`/`--in` is a [storage-backend](/guides/backends/) location, so
besides a plain path it accepts a `file://` URL, an `s3://` bucket, or a `gs://` bucket
— the snapshot is written and read as one object on whatever backend the location names:

```bash
nidus backup  --dir ./store --out file:///backups/store.tar.gz
# Straight to S3 (creds from the AWS environment):
nidus backup  --dir ./store --out s3://my-bucket/backups/store.tar.gz
# …or Google Cloud Storage (GOOGLE_APPLICATION_CREDENTIALS):
nidus backup  --dir ./store --out gs://my-bucket/backups/store.tar.gz
```

The archive is an ordinary gzip-compressed tarball — `tar tzf store.tar.gz`
lists the `data` and `log` files plus a small `nidus-backup.json` manifest
(version, timestamp, dimension), so you can inspect or extract it with standard
tools too. Restore reopens the store afterwards to confirm it loads, and never
carries over a stale writer lock.

**Backup is a safe hot snapshot.** It does not take the writer lock, so it can
run while a writer — including `nidus serve` — is busy. It captures the same
consistent, possibly-slightly-stale view a [lock-free reader](/guides/storage/)
sees: never a torn or half-written store.

Because a backup is one self-contained command on a single directory, a periodic
snapshot is a one-line cron entry:

```bash
# Every night at 02:00, snapshot into a dated file and keep the last 14.
0 2 * * *  nidus backup --dir /srv/nidus/store --out /backups/store-$(date +\%F).tar.gz && \
           ls -1t /backups/store-*.tar.gz | tail -n +15 | xargs -r rm
```

## Over the network

The same `nidus` binary serves a store over HTTP, so a client with no Rust
toolchain can do the full job — create, upsert, search, inspect, maintain — in
JSON:

```bash
# --dim is only needed if the store doesn't exist yet; otherwise it's inferred.
nidus serve --dir ./store --dim 768 --addr 127.0.0.1:7700
```

The complete network workflow, authentication, request limits, and the
endpoint-by-endpoint reference live on the dedicated
[HTTP server & API](/guides/http-server/) page.
