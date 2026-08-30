---
title: Storage & durability
description: The nidus on-disk format, the per-batch fsync contract, crash recovery, checking a store for in-place corruption, compaction, and lock-free cross-process readers.
---

nidus is durable by design with a tiny surface: two append-only files and a lock.
This page covers the on-disk format, what survives a crash, how dead rows are
reclaimed, and how a second process reads a store another is writing.

This page is about a store on **local disk**. To keep the durable data somewhere
else (Amazon S3 or Google Cloud Storage), see [Storage backends](/guides/storage-backends/);
to share the in-memory index across processes via Redis, see
[Memory stores](/guides/in-memory-tier/).

## On-disk format

```
<dir>/
  manifest  names the live segments (the first is `data`) + pins dimension/metric
  data      append-only, fixed-stride, row-major f32 matrix (header pins dimension)
  log       append-only framed op stream: [len][bincode(Op)][crc32] (the commit record)
  lock      O_EXCL writer-exclusion lock file
  seg-…     additional immutable segments (only once a store seals past the threshold)
  data.crc, seg-….crc   checksum sidecars for sealed segments, see "Checking for corruption" below
```

All on-disk encoding is **little-endian and explicit**. Every `log` record is
length-prefixed and CRC32-checked, so a torn tail (a half-written final record
after a crash) is detectable and is dropped on the next open. The `manifest` is a
`[crc32][bincode]` object replaced atomically: a reader sees one whole manifest
version, never a torn mix.

The `data` header pins the embedding **dimension** at creation. Reopening a store
with a different dimension is a hard error: one embedding space per store, for
the life of the store.

## The durability contract

The default fsync policy is [`Fsync::PerBatch`](/reference/configuration/#fsync):
each `upsert`/`delete` call appends vectors, fsyncs `data`, appends the
committing `log` records, then fsyncs `log`. **A crash loses at most the
in-flight batch**: everything fsynced before it is intact, and the in-RAM index
is fully reproducible from the files.

`Fsync::OnFlush` defers **both** fsyncs to an explicit
[`flush()`](/reference/api/#search--maintenance) or close, which re-establishes the same
data-then-log ordering in one go. This is much faster but weaker: an unflushed
batch can be lost on a crash. Use it when you are bulk-loading and can afford to
redo the load on failure.

How much faster depends entirely on how big your batches are, because what
`OnFlush` removes is a fixed per-call cost. Ingesting one record per `upsert`
call, it is a couple of hundred times quicker; at a thousand records per call
the barrier is already amortised and the gap is closer to 2×.

A crash under `OnFlush` can leave the `log` durable while the rows it references
are not. That is not a torn store: replay ignores any record pointing past the
end of `data`, so recovery drops the tail and opens cleanly on the prefix, the
same rule that lets readers work lock-free.

## Graceful failure

Resource exhaustion never corrupts a store:

- **Appends are atomic.** A partial write rolls back to the row/frame boundary.
- **Upsert is all-or-nothing.** Any failure mid-batch rolls `data` and `log` back
  to the entry marks, so a caught `ENOSPC` leaves the store exactly as it was.
- **RAM growth uses `try_reserve`.** Out-of-memory surfaces as an `Err`, not an
  abort (except `attrs`/`id` clones, which std gives no fallible reserve for).
- **A hard ceiling holds under overcommit.**
  [`Config::max_vector_bytes`](/reference/configuration/#max_vector_bytes)
  refuses a batch *before* allocating, the only guard that works on systems
  where the kernel SIGKILLs before an allocation fails.
  [`Nidus::footprint()`](/reference/api/#footprint) is the introspection hook for
  deciding whether you can afford more data.

## Checking for corruption

Everything above protects against a crash mid-write. It says nothing about bytes that
change after a successful write: a bad sector, a stray write from something else on the
disk, a copy that dropped a byte. Before this, `data` and `seg-…` carried no checksum
over the vector rows themselves (only the 64-byte header's magic and version were
validated), so a flipped byte inside a row changed no row count and no header: the
store opened cleanly and returned wrong scores forever.

Every **sealed** segment (an immutable `seg-…`, or `data` once something else becomes
the active segment) now gets a small sidecar object, `<segment>.crc`, stamped the
moment the segment becomes immutable: at seal time, and again at compaction, which
restamps the rewritten base segment and drops the sidecars of the segments it collapses
away. [`nidus check`](/guides/cli-and-server/#checking-a-live-store) recomputes each
sealed segment's checksum and compares it against its sidecar.

A mismatch is real corruption, reported plainly and never silently recomputed and
re-saved (doing that over already-corrupted bytes would just launder the corruption
into a fresh, valid-looking checksum). What it does not cover, on purpose: a sidecar is
only stamped once a segment is immutable, so rows appended to the still-open active
segment since the last seal are unverified, not vouched-for-clean, until the next seal
covers them. `log`'s own tolerance of a CRC-bad *tail* record as a torn write is
deliberate crash recovery (above) and stays exactly as it is: this check does not touch
it.

## Compaction

Because segments are never rewritten in place, a `delete` or an overwriting
`upsert` leaves the old row behind as a **dead row**: still on disk, no
longer referenced by the index. [`compact()`](/reference/api/#search--maintenance) collapses
every [segment](#segments) into one fresh `data` segment that drops the dead rows,
publishes the new manifest, and reclaims the old segment objects.

Compaction also runs automatically on `open` when the dead-row ratio exceeds
[`Config::auto_compact`](/reference/configuration/#auto_compact) (default `0.5`,
half the rows dead). Set it to `None` to disable and compact only on demand.

[`footprint()`](/reference/api/#footprint) reports `rows`, `dead_rows`, and
`vector_bytes` so you can decide when a manual compaction is worth it.

## Segments

A store's vectors live in one or more **segments** (self-contained, immutable chunks of
rows) named in order by the `manifest`. The last one is the **active** segment that new
rows append to; the rest are sealed and never rewritten. The segments are presented to
search as a single dense row space, so this is invisible to queries: the same exact
brute-force scan runs whether a store is one segment or many.

By default a store is a **single segment** (`data`) and behaves exactly as it always has.
Set [`Config::segment_max_rows`](/reference/configuration/#segment_max_rows) to roll the
active segment into a sealed one once it grows past *N* rows and start a fresh one. No
data is copied; sealing just publishes a new `manifest`. Sealing and
[compaction](#compaction) (which collapses every segment back into one) replace the
manifest atomically, which is the store's commit point.

A store that predates this format (just `data` + `log`, no `manifest`) is migrated
transparently on the first read-write open: `data` becomes the base segment and a
manifest is written. A read-only open of such a store writes nothing.

Segments are also the unit of **indexing at scale**: with
[`Config::segment_index_min_rows`](/reference/configuration/#segment_index_min_rows) set, a
sealed segment large enough to cross that threshold gets its own IVF index (the active tail
stays exact), so searches over a large store walk the cold segments and brute-force only the
fresh data. See [per-segment indexing](/guides/search/#per-segment-indexing-at-scale).

## Larger than RAM: memory-mapped segments

By default nidus loads every segment into RAM on `open`. Set
[`Config::mmap(true)`](/reference/configuration/#mmap) and each **immutable** (sealed)
segment is instead served from a read-only **memory-map** of its file: the operating
system pages a segment in on demand and reclaims it under pressure, so a store can hold
more vectors than fit in memory. The **active** segment (the one still taking appends)
stays in RAM.

```rust
use nidus::{Config, Nidus};

// A local store with sealed segments, served larger-than-RAM.
let store = Nidus::open(
    Config::new("/path/to/store", 768)
        .segment_max_rows(Some(1_000_000)) // produce immutable segments to map
        .mmap(true),
)?;
# anyhow::Ok(())
```

Search over mapped segments goes through the same row accessor as the in-RAM path, so
**results are identical**: still exact (or, with an [index](/guides/search/), the same
approximate set), still filter- and `min_score`-respecting. It composes with quantization
and the [per-segment indexes](/guides/search/#per-segment-indexing-at-scale): a cold
segment can be both mapped and indexed.

A few conditions apply:

- It is effective only for a **local-filesystem** store with **sealed** segments: it
  needs [`segment_max_rows`](/reference/configuration/#segment_max_rows) to create
  immutable segments and a mappable local file. An object-store (`s3://`/`gs://`) or
  in-memory store silently stays all-RAM.
- The host must be **little-endian** (the on-disk `f32` layout). Other hosts fall back to
  loading into RAM.
- [Compaction](#compaction) still materializes the live set in RAM, so it is bounded by
  memory even when the store as a whole is not. Keep it infrequent on a very large store.

## Cross-process readers

A store can be opened **read-only** by other processes while one process holds
the writer lock:

```rust
use nidus::{Config, OpenMode};

let reader = Nidus::open(
    Config::new("/path/to/store", 768).open_mode(OpenMode::ReadOnly)
)?;
# anyhow::Ok(())
```

A `ReadOnly` open takes **no lock**. It reads the `manifest`, loads the segments it
names to their current total size *S*, replays `log`, and ignores any record that
references a row ≥ *S*/dim. The result is a **consistent, possibly-stale snapshot**
(never a torn read), even while the writer is mid-append: a not-yet-named segment or a
row past *S* is simply invisible until its commit. This is the lock-free basis for
search-only processes reading a store another process is writing.

### Refreshing a reader

A `ReadOnly` snapshot is fixed at the moment it opened. To pick up a writer's later
commits (appends, deletes, seals, compactions) without reopening, call
[`refresh()`](/reference/api/#search--maintenance):

```rust
use nidus::{Config, Nidus, OpenMode, Scope, SearchOpts};

let mut reader = Nidus::open(
    Config::new("/path/to/store", 768).open_mode(OpenMode::ReadOnly),
)?;

// … later, after another process has written more …
let query = vec![0.0; 768];
if reader.refresh()? {
    // newer state adopted; queries now see it
}
let hits = reader.search(Scope::All, &query, &SearchOpts::default())?;
# let _ = hits;
# anyhow::Ok(())
```

`refresh` re-reads the `manifest` and, when a newer version is published or the `log`
has grown, moves to the newer state at a single consistent point, then swaps it in
**atomically** (a failure leaves the prior snapshot serving, never a torn mix). It
returns `true` when newer state was adopted and `false` when the reader was already
current, the cheap common case (a small `manifest` read plus a `log` stat, no segment
or index work), so it is safe to call before a batch of queries. A `ReadWrite` handle is
already the source of truth, so its `refresh` is always a no-op.

It also stays cheap *when* there is new state. If only the active segment grew (plain
appends, with no seal or compaction changing the segment list), `refresh` re-reads just that
one segment object and reuses every immutable segment, instead of re-fetching the whole
set; a seal/compaction takes the full re-open. And when a shared
[memory tier](/guides/in-memory-tier/) holds a snapshot matching the new state, the reader
adopts it and skips the log replay entirely.

## Point-in-time reads

A reader can also be pinned to the store as it was at an earlier **commit version**:
re-run last week's queries against last week's index, bisect a ranking regression across
store versions, or hold one consistent view across a batch job that outlives the writes
running beside it.

History is **off by default** and has to be recorded before it can be read:

```rust
use nidus::{Config, Nidus, OpenMode};

// The writer records the last 128 commit points.
let mut writer = Nidus::open(Config::new("/path/to/store", 768).history_versions(Some(128)))?;
# let _ = &mut writer;

// A reader pinned to one of them. Always read-only.
let pinned = Nidus::open_at(
    Config::new("/path/to/store", 768).open_mode(OpenMode::ReadOnly),
    31,
)?;
# let _ = pinned;
# anyhow::Ok(())
```

`Nidus::versions()` (and `GET /versions`, and `nidus versions`) reports what is
addressable: the current `commit_version`, the `oldest_readable` one, this handle's `pinned`
version, and the full `readable` set.

Three things are worth knowing before you rely on it.

**You cannot travel back to before you turned it on.** Recording history makes every
durable batch a commit point, which costs a small object write per batch, so it is opt-in
rather than something every store pays for. A store written without it has no addressable
past, and `versions()` reports an empty `readable` set.

**Compaction is a hard floor.** `compact()` rewrites the base segment in place and
renumbers rows, so every version older than the last compaction is gone for good. There is
no knob that keeps them. Asking for one is an error naming the oldest version that is still
readable; a nearest surviving version is never served in its place, because a snapshot that
quietly answers for a different point in time is worse than no answer.

**A pin does not drift.** `refresh()` returns `false` on a pinned handle rather than
advancing it, and `refresh_to(version)` is the explicit way to move. Every write path
refuses, whether you reach it through the library, `nidus serve --at-version N` (which runs
the whole instance pinned and read-only), or the CLI.

Only one **writer** (`OpenMode::ReadWrite`, the default) may hold a store at a
time, enforced by the `O_EXCL` `lock` file. A stale lock left by a crashed writer
is reclaimed after [`Config::lock_ttl`](/reference/configuration/#lock_ttl)
(default 60s).
