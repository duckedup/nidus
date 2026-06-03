# `store` module — spec (the integrator)

Implement `Store` in `mod.rs`. **Do not change the public signatures** — `lib.rs`
calls them verbatim. Root design: `../../SPEC.md` (read it in full — §3, §5, §6, §7,
§8). This is the only module that composes the others; build it last.

## Dependencies (all implemented before this runs)
- `crate::config::{Config, Fsync, OpenMode}`
- `crate::model::{Value, Record, Filter, Predicate, SearchOpts, Hit, Op}`
- `crate::data::DataSegment` — vector segment (RAM + `data` file)
- `crate::log::OpLog` — op stream (`open` returns `(OpLog, Vec<Op>)`)
- `crate::lock::WriteLock` — `acquire(dir, ttl)`, releases on drop
- `crate::filter::matches` — predicate evaluation
- `crate::search::{normalize, dot, TopK}` — cosine + bounded top-k

## In-RAM state
```
config: Config
data:   DataSegment                      // global row-major f32 matrix
log:    OpLog
lock:   Option<WriteLock>                // None when ReadOnly / in-memory
collections: HashMap<String, Collection>
dead_rows: usize                         // rows no longer referenced (for compaction)
struct Collection { meta: BTreeMap<String,String>, docs: HashMap<String, DocEntry> }
struct DocEntry { row: u64, attrs: BTreeMap<String, Value> }
```

## open(config)
1. `std::fs::create_dir_all(&config.path)`.
2. If `open_mode == ReadWrite`, `lock = Some(WriteLock::acquire(&config.path, config.lock_ttl)?)`; else `None`.
3. `data = DataSegment::open(&path.join("data"), config.dimension)?`.
4. `(log, ops) = OpLog::open(&path.join("log"))?`.
5. Replay `ops` in order into `collections`:
   - `CreateCollection` → ensure entry exists. `DropCollection` → remove it.
   - `SetMeta` → replace that collection's `meta`.
   - `Upsert{collection,id,row,attrs}` → **ignore if `row >= data.row_count()`**
     (lock-free reader rule, §6.2); else insert/overwrite `docs[id]` (overwriting an
     existing id increments `dead_rows`).
   - `Delete{collection,id}` → remove `docs[id]` (increment `dead_rows` if present).
   - An op naming a missing collection: create it implicitly (be lenient on replay).
6. If `config.auto_compact == Some(t)` and `dead_rows / max(1,total_rows) > t`, call
   `compact()`.

## Writes (reject all with `Error`/`bail!("read-only")` when `OpenMode::ReadOnly`)
Per `Fsync::PerBatch` (default) each mutating call fsyncs `data` then `log` before
returning; for `Fsync::OnFlush` defer fsync to `flush()`.
- `create_collection` / `drop_collection`: update RAM + append the op. `drop` marks
  all its docs' rows dead.
- `set_meta`: update RAM + append `SetMeta`.
- `upsert(collection, records)`: ensure collection exists; for each record: `bail!`
  if `record.vector.len() != dimension`; normalize a copy of the vector
  (`search::normalize`); `row = data.append(&v)?`; if an id already exists, the old
  row becomes dead; set `docs[id] = {row, attrs}`; append `Op::Upsert`. **Order:
  append all vectors, `data.sync()`, then append all log records, `log.sync()`**
  (§6.2). Return count upserted.
- `delete(collection, ids)`: remove present ids (mark rows dead), append `Op::Delete`
  per removed id, fsync per policy. Return count removed.
- `delete_where(collection, filter)`: collect ids whose `attrs` satisfy
  `filter::matches`, then delete them as above. Return count.

## Reads
- `dimension`, `config`, `has_collection`, `collections` (sorted or insertion order —
  document which; sorted is fine), `get_meta` (clone; empty map if absent),
  `get_all(collection)` → `Vec<Record>` rebuilding each `Record { id, vector:
  data.row(row).to_vec(), attrs }`.
- `search(collections, query, opts)`:
  - Make a normalized copy of `query`.
  - One `TopK<(String /*collection*/, String /*id*/)>` (or row ref) of size
    `opts.top_k`.
  - For each named collection that exists, for each `(id, entry)`: if
    `filter::matches(&opts.filter, &entry.attrs)`, compute `score = dot(query,
    data.row(entry.row))`; if `min_score` is `None` or `score >= min`, `offer` it.
  - `into_sorted_desc` → build `Hit { collection, id, score, attrs: clone }`.
  - A named collection that does not exist is skipped (not an error).

## compact()
Reassign contiguous row indices to live docs; gather their vectors into one `Vec<f32>`;
`data.rewrite(&rows)`; rebuild every `DocEntry.row`; `log.rewrite(&ops)` where `ops`
is `CreateCollection` + `SetMeta` + one `Upsert` per live doc (new rows); reset
`dead_rows = 0`. Atomic + fsynced (delegated to `data`/`log` rewrite).

## Constraints
Pure safe Rust (`#![forbid(unsafe_code)]`), `anyhow` errors. Hold no lock across a
panic boundary beyond `WriteLock`'s own `Drop`. Keep `search` allocation-light on
the hot path (only clone attrs for the surviving top-k).

## Tests (`tests` submodule)
Use `Store::in_memory(dim)` for pure-logic where possible; `tempfile::tempdir()` +
`#[cfg_attr(miri, ignore)]` for file-backed durability/replay/compaction. Cover:
create/upsert/search ranking (exact match scores ~1.0, orthogonal ~0.0); idempotent
overwrite by id (count stays, newest wins); delete + delete_where; `min_score` and
filter scoping; multi-collection search merges into one ranking and each `Hit` has
the right `collection`; metadata round-trip; reopen sees prior data (file-backed);
ReadOnly rejects writes; compaction preserves all live docs and search results.
There is also a crate-level integration test in `../../tests/` and an
`examples/demo.rs` exercising the public API — keep them green.
