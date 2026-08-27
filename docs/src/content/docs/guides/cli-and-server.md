---
title: Command line
description: "Use nidus from the terminal with the `nidus` binary: create, upsert, search, inspect, back up, restore, and check a store directory."
---

Besides the Rust library, nidus ships a `nidus` binary: a command-line tool for
working with a store directly. It operates on an ordinary store directory, the
very same format the library reads and writes. The same binary also runs an HTTP
server; that has its own [HTTP server](/guides/http-server/) page.

The binary is optional. The library has no dependency on it: `cargo add nidus
--no-default-features` gives you the storage-and-search core alone. The binary
is built behind a `cli` feature, part of the default build.

This page is a tour, with worked examples. For every subcommand and flag, its
`NIDUS_*` environment variable, and which Cargo feature unlocks it, see the
[CLI reference](/reference/cli/).

## Install

The fastest path needs **no Rust toolchain**: one command fetches a prebuilt
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
cargo install nidus                  # build from source
```

Every route installs the same single `nidus` executable.

## Quickstart: local search in four commands

From an empty directory to a working nearest-neighbour query: no Rust, no
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

# 4. Search. No --dim needed: it is read from the store.
echo '[1,0,0]' | nidus search --dir ./store docs -k 2
```

The last command prints ranked hits as JSON. That is a complete local vector
store: a single `./store` directory you can copy, back up, or delete. To drive the
same store over the network instead, point
[`nidus serve`](/guides/http-server/) at the same directory.

## Command line

Every command takes the store directory (`--dir`/`-d`). The embedding dimension
is pinned in the store the first time you create a collection, so **`--dim` is
only needed at creation**. Afterwards it is read from the store. (Pass it anyway
and it is checked: a mismatch is a hard error, not a silent surprise.) The
`--distance` metric (`cosine`, `euclidean`, or `dot`) works the same way: chosen
at creation (default `cosine`), inferred thereafter. Records, query vectors, and
filters are JSON; output is JSON on stdout.

```bash
# Create a collection: dimension pinned here (default cosine distance)
nidus create --dir ./store --dim 3 docs

# Or with Euclidean distance
nidus create --dir ./store --dim 3 --distance euclidean docs

# Upsert records (JSON array) from stdin or a --file (no --dim needed)
echo '[{"id":"a","vector":[1,0,0],"attrs":{"lang":{"Str":"rust"}}}]' \
  | nidus upsert --dir ./store docs

# Nearest-neighbour search: query vector on stdin
echo '[1,0,0]' | nidus search --dir ./store docs -k 5

# Search every collection at once (omit the collection names)
echo '[1,0,0]' | nidus search --dir ./store

# Filter while searching (an AND of predicates, as JSON)
echo '[1,0,0]' | nidus search --dir ./store docs \
  --where '[{"Eq":["lang",{"Str":"rust"}]}]'

# Force the exact scan, even on a store with an ANN index or quantization
echo '[1,0,0]' | nidus search --dir ./store docs --exact

# Trim the attrs each hit carries (repeatable; the two flags are mutually exclusive)
echo '[1,0,0]' | nidus search --dir ./store docs --include-attr path --include-attr lang
echo '[1,0,0]' | nidus search --dir ./store docs --exclude-attr body

# Rank by recency: subtract an age penalty from each hit's score
echo '[1,0,0]' | nidus search --dir ./store docs \
  --rank-by '{"Decay":{"field":"updated_at","origin":1770000000000,"lambda":0.2}}'

# Diversify: at most 2 hits per distinct value of an attribute
echo '[1,0,0]' | nidus search --dir ./store docs --limit-per path --limit-per-max 2

# List records by metadata filter (no vector query); --offset/-n paginate
nidus list --dir ./store docs --where '[{"Eq":["lang",{"Str":"rust"}]}]'
nidus list --dir ./store docs --offset 100 -n 100   # next page
nidus list --dir ./store docs --include-attr path   # ids and one attr only
nidus list --dir ./store docs --order-by updated_at --desc  # ORDER BY, no vector

# Count matches and total numeric attrs without listing anything
nidus aggregate --dir ./store docs --sum bytes \
  --where '[{"Eq":["lang",{"Str":"rust"}]}]'

# Full-text search (BM25): declare which fields are indexed, then query by text
nidus set-fts-schema --dir ./store docs --field body --field title
# …with BM25/analyzer tuning applied to every --field in the call
nidus set-fts-schema --dir ./store docs --field body \
  --k1 1.5 --b 0.3 --ascii-folding --max-token-len 40
nidus text-search --dir ./store body "running quickly" -k 5
nidus text-search --dir ./store body "rust" --in docs \
  --where '[{"Eq":["lang",{"Str":"rust"}]}]'

# Search several fields at once: --clause field=text, repeatable. Use it instead of
# the positional field/text pair, never alongside. --combine sum (default) or max.
nidus text-search --dir ./store --clause title=rust --clause body="async runtime" \
  --combine max -k 5

# Ask why a hit matched: per-clause BM25 scores, and highlighted fragments
nidus text-search --dir ./store body "running" --explain --highlight \
  --max-fragments 2 --fragment-chars 120

# Autocomplete: ranked term completions for a prefix, commonest first
nidus suggest --dir ./store body run docs --limit 5
# …scoped to what this caller may see, and completing the phrase rather than the last word
nidus suggest --dir ./store body "quick br" docs --where '[{"Eq":["tenant",{"Str":"acme"}]}]'

# Hybrid search: fuse a vector (stdin) and a BM25 text query with RRF
echo '[1,0,0]' | nidus hybrid-search --dir ./store body "vector database" -k 5
# …leaning on the keyword leg (both weights default to 1.0)
echo '[1,0,0]' | nidus hybrid-search --dir ./store body "CVE-2026-1234" --text-weight 3
# --clause/--combine/--explain/--highlight work here too; --explain additionally
# reports each fusion leg's own rank and score.
echo '[1,0,0]' | nidus hybrid-search --dir ./store body "vector database" --explain

# Inspect, maintain
nidus collections --dir ./store
nidus get        --dir ./store docs
nidus stats      --dir ./store
nidus compact    --dir ./store
nidus compact    --dir ./store --expired  # delete lapsed entries, then reclaim rows
nidus delete     --dir ./store docs a b

# Record the open-time knobs (--ann, --quantization, --query-threads, --mmap) as
# this store's own defaults; see "Configure once" below
nidus configure  --dir ./store --ann hnsw

# Snapshot the whole store to one portable .tar.gz, and restore it
nidus backup     --dir ./store --out ./store.tar.gz
nidus restore    --in ./store.tar.gz --dir ./restored

# Verify sealed segments against their checksum sidecars, in place, no archive involved
nidus check      --dir ./store
```

Read-only commands (`search`, `list`, `get`, `collections`, `stats`, `check`, and
`backup`) open the store without taking the writer lock, so they can run
alongside a writer such as a running server.

`compact --expired` deletes every entry across the store whose `nidus.expires_at`
has passed, then reclaims the rows those deletes freed, in one call: the manual
"filter by timestamp, delete, then compact" sequence in one step. It is a plain
write command like `compact` and `delete`: it opens the store read-write and blocks
on the writer lock if a server already holds it, so against a running server the
equivalent is `POST /compact {"expired": true}` (see [HTTP
API](/reference/http-api/#post-compact)).

## Approximate search (ANN)

By default search is exact brute-force. For larger stores you can opt into an
in-memory approximate-nearest-neighbour index with `--ann hnsw` or `--ann ivf`. Pass
it on the commands that should use it, or record it once as the store's own default
with [`nidus configure`](#configure-once-recording-store-defaults) so later commands,
including `serve`, pick it up without repeating the flag:

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
configuration.

## Storage & memory backends

By default a store is a local directory and its working set is the process heap. Two
optional flags point each axis elsewhere. See [Storage backends](/guides/storage-backends/)
and [Memory stores](/guides/memory-stores/) for the full model; both work on every
store-opening command, including `serve`:

```bash
# --persistence: where the durable data/log live. A live object-store-backed store
# (whole-object rewrite on flush). --dim only to create one; an existing store's
# dimension is read from its remote header, so any --dir opens it cold.
nidus upsert --dir ./meta --dim 768 --persistence s3://my-bucket/store docs < recs.json
nidus search --dir ./meta --persistence s3://my-bucket/store docs -k 5 < q.json

# --memory: share the in-RAM working set across processes via Redis/Valkey/KeyDB.
# Each worker publishes on flush and adopts on open, skipping the log replay.
nidus serve --dir ./store --dim 768 --memory redis://cache:6379?prefix=docs
```

`--persistence` accepts a path / `file://` (local, the default), `s3://<bucket>[/<prefix>]`,
or `gs://<bucket>[/<prefix>]`; cloud credentials come from the standard environment
(`AWS_*` / `GOOGLE_APPLICATION_CREDENTIALS`). `--memory` accepts `local` (the default) or a
`redis://`/`rediss://`/`valkey://`/`valkeys://`/`keydb://`/`dragonfly://` URL. The axes
compose, and search is identical either way: it always runs over local RAM.

A live object-store-backed store rewrites the whole `data`/`log` object on each flush
(`O(object)`, fine for low write rates) and takes an **advisory** writer lock, suited to a
single writer; for many concurrent writers, prefer a local store and snapshot to the cloud.

## Speed, memory & durability flags

Defaults are exact, all-RAM, and durable per batch. These flags trade along each of those
axes. Pass them on the commands that should use them, or record them once with
[`nidus configure`](#configure-once-recording-store-defaults) so later commands,
including `serve`, pick them up without repeating the flag:

```bash
# Quantize the search first pass, then rerank candidates in exact f32.
# int8 = 4x less memory traffic; binary = 32x, cosine only.
echo '[1,0,0]' | nidus search --dir ./store --quantization int8 docs -k 5
echo '[1,0,0]' | nidus search --dir ./store --quantization binary --quant-rescore 24 docs

# Split ONE query's scan across threads (unrelated to serving concurrent requests)
nidus serve --dir ./store --dim 768 --query-threads 8

# Seal the active segment every N rows, and memory-map the sealed ones so the
# store can outgrow RAM on a single node (local filesystem only)
nidus serve --dir ./store --dim 768 --segment-max-rows 250000 --mmap

# Weaken durability for speed: fsync on explicit flush instead of every batch
nidus upsert --dir ./store --fsync on-flush docs < recs.json
```

`--quantization` accepts `int8` or `binary` (`binary` is cosine-only: sign codes discard
magnitude), with `--quant-rescore` tuning the over-fetch before the exact rerank. Candidate
selection is approximate; the final ranking is always exact. `--segment-index-min-rows`
gives each sufficiently large sealed segment its own IVF index. Housekeeping knobs:
`--auto-compact <ratio>` (or `--no-auto-compact`) controls dead-row reclamation,
`--lock-ttl <seconds>` how long before a stale writer lock may be reclaimed, and
`--max-vector-bytes` refuses to open a store whose matrix would exceed a ceiling.

Every flag also reads from a `NIDUS_*` environment variable (`NIDUS_QUANTIZATION`,
`NIDUS_MMAP`, `NIDUS_QUERY_THREADS`, …), so a container can be configured entirely through
the environment with no command line.

## Configure once: recording store defaults

`--ann`, `--quantization`, `--query-threads`, and `--mmap` are open-time knobs: which
index to walk, how the search first pass scores, how many threads split one query's
scan, and whether sealed segments are memory-mapped. `nidus configure` records the
flags you pass it as the store's own defaults, so later commands, including `serve`,
inherit them without repeating the flag:

```bash
# Record ANN + quantization as this store's defaults, once
nidus configure --dir ./store --ann hnsw --quantization int8

# Every later command picks them up automatically
echo '[1,0,0]' | nidus search --dir ./store docs -k 5
nidus serve --dir ./store --dim 768
```

Precedence is: an explicit flag (or its `NIDUS_*` environment variable, since clap
resolves the variable into the flag) always wins for that one call; otherwise a
recorded default applies; otherwise the built-in default. So `nidus search --dir
./store --exact docs` still forces brute force even after `--ann` is configured.
`mmap` is the one knob that is a bare on/off rather than a value with its own "off"
state, so once `--mmap` is recorded, `--no-mmap` opts a single command back out.

Recording defaults writes the profile into the manifest, which moves it to format
version 2. Every nidus binary from this release onward reads a v2 manifest fine
(and lifts an older v1 manifest transparently, with no recorded defaults), but the
reverse does not hold: an older, **pre-0.60** binary refuses a v2 manifest outright
with `format version 2 is not supported`. Configuring a store is a one-way upgrade:
harmless for a store one version of nidus owns, but worth knowing before you
configure a store that a mixed fleet of older and newer binaries all open.

## Running several instances over one shared store

`--cluster` runs nidus as one of several cooperating instances over the *same* store. It
requires both shared axes (an object-store `--persistence` **and** a Redis-family
`--memory` tier) and is refused with a clear error otherwise, since a local directory or a
process-local working set cannot be shared:

```bash
# The writer: holds a renewing lease; only one may hold it at a time
nidus serve --dir ./meta --dim 768 --cluster \
  --persistence s3://my-bucket/store --memory redis://cache:6379

# Readers: no lease, so run as many as you like
nidus serve --dir ./meta --dim 768 --cluster --read-only \
  --persistence s3://my-bucket/store --memory redis://cache:6379
```

A reader loads the store's committed state when it starts and keeps serving that snapshot;
`POST /refresh` advances it to whatever the writer has committed since, and answers
`{"adopted": true}` when there was something new. Reads deliberately do not refresh on
their own (that would add a metadata fetch to every query, which is the opposite of what
a read-heavy fan-out wants), so poll it as often as your staleness tolerance requires.

### Failover

Only one instance may hold the writer handle. By default a second writer exits at once with
a "locked" error, which is right for a one-off command but means a would-be standby just
dies. `--wait-for-lease` changes that: the instance stays up, waits, and is **promoted
automatically** within roughly `--lock-ttl` of the active writer dying.

```bash
# A standby: same store, same flags, but it waits for the handle instead of exiting
nidus serve --dir ./meta --dim 768 --cluster --wait-for-lease \
  --persistence s3://my-bucket/store --memory redis://cache:6379 --lock-ttl 15
```

While waiting, a standby reports `200` on `/health` (it is alive: waiting is its job) and
`503` on `/ready` (it has no store, so it should get no traffic). Point your liveness probe
at `/health` and your readiness probe at `/ready` and an orchestrator does the right thing
on its own: keep the standby running, route around it, and route to it once promoted.

The two probes answer different questions, and both are deliberately blind to an instance
being merely **busy**: a large upsert holds the store's write guard for the whole batch, and
neither probe takes that lock, so a working writer is never mistaken for a broken one.
`/ready` additionally reports `503` for a fenced writer or a reader past `--max-staleness`;
`/health` reports `503` only when a panic has left the process unrecoverable, which is the one
case where restarting it is the right response.

Pass a number of seconds (`--wait-for-lease 300`) to give up after that long instead of
waiting indefinitely, useful in a script that should not hang.

### Choosing `--lock-ttl`

`--lock-ttl` bounds how long a dead writer's handle stays un-reclaimable, and therefore how
long promotion takes (within a second either side: lease stamps have one-second granularity,
and the reclaim rule errs towards leaving a live writer alone). Lower is faster failover.
The default is 60s; 10–30s is a reasonable cluster setting.

**It is failover latency, not a write budget.** The writer renews its lease on a timer at a
third of the TTL, out of band and without the store lock, so renewal continues *during* a
long write. You do not need to size the TTL against your largest batch, and a slow
object-store `PUT` no longer risks a healthy writer being replaced mid-flight. An idle writer
is kept alive by the same timer: issuing no writes for hours does not put its lease at risk.

What the two directions actually cost:

| | Too low | Too high |
| --- | --- | --- |
| Cost | A run of failed renewals (an object store having a bad minute) can expire a lease that a healthy writer still believes it holds | A crashed writer's slice of the store is read-only for that long before a standby takes over |
| Floor | Renewal is `TTL/3`, so anything under a few seconds gives the renewer no room to retry through a blip | (none) |

That first case is *safe*, just disruptive: the superseded writer is
[fenced](/guides/storage-backends/#cooperating-instances-cluster), not allowed to clobber:
every durable write is compare-and-swapped against the version it last saw, so its next batch
is refused rather than applied. You lose the writer, never the data. It reports `503` from
`/ready` immediately (the background renewer latches the state; nobody has to wait for a
write to discover it), and an orchestrator recycles it. Set the TTL so that does not happen
on an ordinary hiccup.

A renewal that fails because the object store was briefly unreachable does **not** fence the
writer: only the store actually reporting a different lease owner does. That distinction is
why a blip costs the write in flight rather than the instance.

If failover latency matters more than the TTL can give you, run a standby with
`--wait-for-lease`: it is already up, already warm, and takes over the moment the handle
becomes claimable, so promotion is the TTL and not the TTL plus a cold start.

### Keeping readers current, and noticing when they are not

A reader adopts state at open and advances only when refreshed. Two flags make that
operable rather than something you have to build around:

```bash
# Refresh itself every 5s, and report NOT ready if it ever falls 30s behind
nidus serve --dir ./meta --dim 768 --cluster --read-only \
  --refresh-interval 5 --max-staleness 30 \
  --persistence s3://my-bucket/store --memory redis://cache:6379
```

`--refresh-interval` removes the need for a sidecar or cron calling `POST /refresh`.
`--max-staleness` is the safety net: if refreshing stops working, readiness fails and the
instance leaves the load balancer rather than quietly serving ever-older results. Reads
themselves are never rejected: the bound governs *routing*, not correctness.

`GET /cluster` reports each instance's role, whether it holds the writer handle, whether it
has been fenced, the commit counter it is serving, and its staleness. That is what to check
first during an incident: comparing `commit_version` across instances shows replication lag,
and `lease_owner` answers who the writer is.

There is no election and no coordinator, and none is needed: the object store's conditional
writes (`If-Match` on S3, `ifGenerationMatch` on GCS) are a linearizable compare-and-swap,
so exactly one claimant can win the handle even when several try at the same instant. That
is the same primitive a consensus protocol would give you, already durable and already
shared, which is why a writer that stalls and wakes up superseded is *refused* rather than
allowed to clobber its successor's commits.

This is deliberately **not** a managed cluster: there is no coordinator, no replication, and
no rebalancing. Writes are fenced (a superseded writer is refused rather than allowed to
clobber committed data) and `--lock-ttl` sets the lease window. If you only want more
capacity across a few machines, a simpler shape needs none of this: run one independent
instance per box and fan queries out client-side, merging the top-k yourself (sound because
every instance shares one embedding space). See
[running across a few boxes](/guides/multi-box/) for the recipe.

## Backup, restore & verify

A store is just a directory, so you can always copy it by hand, but `nidus
backup` archives the whole durable object set (`data`, `log`, `manifest`, and
every sealed `seg-*`) into a single compressed `.tar.gz` you can stash before
an upgrade or hand to a cron job, and `nidus restore` brings it back.

```bash
# Snapshot ./store into one portable archive.
nidus backup --dir ./store --out ./store.tar.gz

# Omit --out and you get a sortable, timestamped name in the current directory,
# e.g. store-1781063324.tar.gz, handy for keeping a series of snapshots.
nidus backup --dir ./store

# Restore into a directory. If the target already holds a store you are asked to
# confirm; pass -y to overwrite without prompting.
nidus restore --in ./store.tar.gz --dir ./restored
nidus restore --in ./store.tar.gz --dir ./store --yes
```

The archive's `--out`/`--in` is a [storage-backend](/guides/storage-backends/) location, so
besides a plain path it accepts a `file://` URL, an `s3://` bucket, or a `gs://` bucket.
The snapshot is written and read as one object on whatever backend the location names:

```bash
nidus backup  --dir ./store --out file:///backups/store.tar.gz
# Straight to S3 (creds from the AWS environment):
nidus backup  --dir ./store --out s3://my-bucket/backups/store.tar.gz
# …or Google Cloud Storage (GOOGLE_APPLICATION_CREDENTIALS):
nidus backup  --dir ./store --out gs://my-bucket/backups/store.tar.gz
```

The *source* store is a backend location too: pass `--persistence` to back up an
object-store-backed store, and `nidus restore --persistence …` to restore into one:

```bash
nidus backup  --persistence s3://my-bucket/store --out ./store.tar.gz
nidus restore --in ./store.tar.gz --persistence s3://my-bucket/store --dir ./meta
```

The archive is an ordinary gzip-compressed tarball: `tar tzf store.tar.gz`
lists `data`, `log`, `manifest`, and any sealed `seg-*` files, plus a small
`nidus-backup.json` manifest (version, timestamp, dimension, and a per-object
`{name, bytes, crc32}` baseline that `nidus verify` checks below), so you can
inspect or extract it with standard tools too. Restore reopens the store
afterwards to confirm it loads, and never carries over a stale writer lock.

**Backup is a safe hot snapshot.** It does not take the writer lock, so it can
run while a writer (including `nidus serve`) is busy. It captures the same
consistent, possibly-slightly-stale view a [lock-free reader](/guides/storage/)
sees: never a torn or half-written store.

Because a backup is one self-contained command on a single directory, a periodic
snapshot is a one-line cron entry:

```bash
# Every night at 02:00, snapshot into a dated file and keep the last 14.
0 2 * * *  nidus backup --dir /srv/nidus/store --out /backups/$(date +\%F).tar.gz && \
           ls -1t /backups/store-*.tar.gz | tail -n +15 | xargs -r rm
```

### Verify

`nidus verify` proves an archive is restorable without restoring it into a
real store: it extracts into a scratch location, never a real store, checks
the archive's own bytes, and confirms the extracted store reopens cleanly.

```bash
nidus verify -i ./store.tar.gz
```

On success it prints a JSON report to stdout and exits `0`:

```json
{
  "archive": "./store.tar.gz",
  "dimension": 768,
  "distance": "Cosine",
  "collections": ["docs", "notes"],
  "records": 15000,
  "objects_checked": 5,
  "archive_bytes": 10485760
}
```

`objects_checked` is how many archived objects had their checksum rechecked.
Archives written before 0.57 carry no checksum baseline, so it reads `0` and
verify falls back to the structural check.

On any mismatch it prints nothing to stdout, writes `error: …` to stderr, and
exits `1`, so scripting against the exit code alone is safe:

```bash
nidus verify -i ./store.tar.gz || echo "backup is bad, alert on-call"
```

You can also verify a backup as part of taking it, so a bad upload or a
truncated write is caught immediately rather than at restore time:

```bash
nidus backup --dir ./store -o ./store.tar.gz --verify
```

This takes the backup, then re-reads it from its destination (`./store.tar.gz`)
and runs the same verification the standalone command does.

**What verify proves, and what it does not.** It checks the archive's
per-object CRC32 baseline recorded in `nidus-backup.json`, drives the gzip
stream to EOF so its own trailer checksum fires too, and confirms the store
reopens read-only with the expected dimension, distance, collections, and
record count. It is not a semantic diff against the live source store, and it
says nothing about a store corrupted in place on disk, only about the
archive's own bytes. Archives written before nidus 0.57 predate the baseline;
verify falls back to the structural check and reports `objects_checked: 0`.
For a store's own bytes on disk between backups, see `nidus check` below.

## Checking a live store

`nidus verify` proves an archive is restorable. It says nothing about the
working store on disk right now, between backups, because a store corrupted
in place (a bad sector, a stray write from something else) still opens
cleanly and returns wrong scores: nothing about it changes the row count or
the header. `nidus check` closes that gap, in place, with no archive step:

```bash
nidus check --dir ./store
```

Every **sealed** segment (a `seg-…` file, or `data` once something else
becomes the active segment) carries a small sidecar object, `<segment>.crc`,
written the moment the segment becomes immutable. `check` recomputes each
sealed segment's checksum and compares it against its sidecar, then reports
one entry per segment: whether it matched, how many rows the sidecar covers
versus how many the segment now holds, and whether the sidecar is missing or
unusable. It exits `1` on any mismatch, so you can script against the exit
code the same way as `verify`:

```bash
nidus check --dir ./store || echo "store is corrupted, alert on-call"
```

**What it does and does not cover.** A checksum is stamped only when a
segment becomes immutable (seal or compaction), never on every append, so a
segment can legitimately show fewer covered rows than it holds: the
difference is the tail written to the active segment since the last seal,
which is unverified, not vouched-for-clean, until the next seal covers it. A
store with no sealed segments yet (nothing has grown past
[`segment_max_rows`](/reference/configuration/#segment_max_rows)) reports
every segment as having no sidecar at all, which means unverified, not
corrupt. `check` never recomputes and re-saves a checksum on its own: doing
that over already-corrupted bytes would launder the corruption into a fresh,
valid-looking checksum, so a real mismatch stays reported until you
investigate and, if you choose to, rebuild the segment yourself (compaction
restamps it). `check` also does not cover `log`, whose own tolerance of a
CRC-bad *tail* record is deliberate crash recovery (see [Storage &
durability](/guides/storage/#the-durability-contract)), not corruption. It is
a different tool from `verify` for a different question: `verify` asks "is
this archive restorable," `check` asks "has this store's disk rotted since
its last seal."

## Over the network

The same `nidus` binary serves a store over HTTP, so a client with no Rust
toolchain can do the full job (create, upsert, search, inspect, maintain) in
JSON:

```bash
# --dim is only needed if the store doesn't exist yet; otherwise it's inferred.
nidus serve --dir ./store --dim 768 --addr 127.0.0.1:7700
```

Started with `--embed-provider` instead, `--dim` drops from required to optional for
a store directory that does not exist yet: the embedder has its own dimension, so
nidus reads it from there. This is a narrower rule than the general one above, and
only covers the not-yet-created case: point either form at a store that already
exists and the on-disk header wins, so passing a `--dim` that disagrees with it is
still a hard error.

The complete network workflow, authentication, and request limits are on the
[HTTP server](/guides/http-server/) page; the endpoint-by-endpoint reference is the
[HTTP API](/reference/http-api/).
