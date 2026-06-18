---
title: Storage backends
description: Keep a store's durable data on local disk (the default), in Amazon S3 (or R2 / MinIO), or in Google Cloud Storage — chosen with a single location string.
---

By default, a nidus store is a folder on your local disk. You can instead keep its
durable data in a cloud object store — **Amazon S3** (or any S3-compatible store like
**Cloudflare R2** and **MinIO**) or **Google Cloud Storage** — by passing one location
string. Nothing else about how you use nidus changes.

```bash
# Local folder (the default)
nidus search --dir ./store docs -k 5 < query.json

# The same store, kept in Amazon S3 instead
nidus search --dir ./meta --dim 768 --persistence s3://my-bucket/store docs -k 5 < query.json
```

From the Rust library it is one builder call:

```rust
use nidus::{Config, Nidus};

// Local disk (the default — just a path):
let local = Nidus::open(Config::new("./store", 768))?;

// Amazon S3:
let cloud = Nidus::open(Config::new("./meta", 768).persistence("s3://my-bucket/store"))?;
# anyhow::Ok(())
```

That's the whole feature: **where the durable bytes live is a value, not a rebuild.**
Search itself never touches the backend — nidus always scans an in-RAM copy of your
vectors — so moving a store to the cloud changes startup and write cost, never search
results or speed.

## Choosing a backend

| You want… | Use this location | Notes |
|---|---|---|
| Local disk (default) | `./store` or `file:///abs/path` | A plain folder; fastest, simplest. |
| Amazon S3 | `s3://bucket/prefix` | Also Cloudflare R2 and MinIO (set `AWS_ENDPOINT_URL`). |
| Google Cloud Storage | `gs://bucket/prefix` | `gcs://…` works too. |

An unrecognized location is rejected with a clear error, so a typo never silently falls
back to local disk.

### Local files (the default)

A store is a folder; each piece is a file inside it (`data`, `log`, and rebuildable
caches). Writes are crash-safe and a second writer is locked out — see
[Storage & durability](/guides/storage/). Nothing to configure: just give a path.

```bash
nidus create --dir ./store --dim 768 docs
```

### Amazon S3 (and R2, MinIO)

Point `--persistence` at an `s3://` bucket. Credentials come from the standard AWS
environment variables — the same ones the AWS CLI uses:

```bash
export AWS_ACCESS_KEY_ID=…
export AWS_SECRET_ACCESS_KEY=…
export AWS_REGION=us-east-1
# For Cloudflare R2 or MinIO, also point at the endpoint:
# export AWS_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com

nidus upsert --dir ./meta --dim 768 --persistence s3://my-bucket/store docs < recs.json
nidus search --dir ./meta --dim 768 --persistence s3://my-bucket/store docs -k 5 < query.json
```

`AWS_SESSION_TOKEN` is used if set. Pass `--dim` when the store lives in the cloud — nidus
reads the local folder's header to learn the dimension, and there isn't one for a remote
store.

### Google Cloud Storage

Point `--persistence` at a `gs://` bucket and supply a service-account key, either as a
file path or inline JSON:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
# or: export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",…}'

nidus upsert --dir ./meta --dim 768 --persistence gs://my-bucket/store docs < recs.json
```

## Local vs. cloud: what changes

A cloud object store has no "append to the end of a file" operation, so a store kept in
S3 or GCS rewrites the whole `data`/`log` object each time you flush. In return you get
durable, off-box storage with a one-line switch. The trade-offs:

- **Write cost.** A flush rewrites the whole object (cost grows with store size), versus a
  tiny append on local disk. Fine for occasional writes / dev / small stores; not for
  write-heavy workloads.
- **One writer.** The cloud writer lock is best-effort (a short-lived marker object), so a
  cloud-backed store assumes a single writer. For many concurrent writers, keep the store
  on local disk and [back it up](#backups) to the cloud instead.
- **Search is identical.** Either way, search runs over local RAM — same results, same
  latency.

This makes cloud backing a good fit for nidus's sweet spot: a personal- or small-team
store you want to keep somewhere durable and shareable, written occasionally.

## Backups

A store is just a few files, so a backup is one `.tar.gz`. `nidus backup` reads a store
and writes the archive to any location — a local path or a cloud bucket — and
`nidus restore` brings it back:

```bash
# Snapshot a local store straight to S3:
nidus backup  --dir ./store --out s3://my-bucket/backups/store.tar.gz
nidus restore --in s3://my-bucket/backups/store.tar.gz --dir ./restored
```

A backup is a safe hot snapshot — it doesn't take the writer lock, so it can run while a
writer (or `nidus serve`) is busy. See the [command-line guide](/guides/cli-and-server/#backup--restore)
for the full story.

## What gets stored

Whichever backend you choose, a store is the same small set of named pieces:

| piece | what it is | survives a backup? |
|---|---|---|
| `data` | your vectors (the durable matrix) | **yes — required** |
| `log`  | the record of every change | **yes — required** |
| `ann`, `fts` | search-index caches | no — rebuilt automatically on open |

Only `data` and `log` matter for durability; the caches can be deleted and are rebuilt
from scratch when the store next opens, so a missing or stale cache is never a problem.

## Writing your own backend

The backends above are implementations of one small, synchronous Rust trait,
`nidus::backend::Persistence` (whole-object `get`/`put`/`delete`/`list`, plus an optional
native append for local files). If you want a store to live somewhere nidus doesn't ship
— another object store, a database, a tmpfs — implement that trait. The full method
surface is in the [API reference](/reference/api/); the trait is sync on purpose, so it
drops straight into a `Box<dyn Persistence>` chosen at runtime.
