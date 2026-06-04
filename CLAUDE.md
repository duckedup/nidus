# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
just test          # run all tests
just ci            # fmt-check + clippy (-D warnings) + test
just lint          # clippy only
just miri          # undefined behavior check via Miri (requires nightly)
just fmt           # format code
just build         # debug build
just release       # optimized release build
just doc           # build + open API docs
just deps          # assert the dependency tree is empty
```

Rust 1.96+ required (pinned via `rust-toolchain.toml`). Edition 2024.

### Pure-Rust dependencies only, zero FFI — enforced

nidus may depend on **popular pure-Rust crates** (`anyhow`, `serde`/`bincode`,
`crc32fast`, …), but **never a crate that compiles C or links a native library**
(no `*-sys`, no bundled C/C++), and our own code carries `#![forbid(unsafe_code)]`.
These are not stylistic preferences; they are the product. Pulling in a C-compiling
crate or adding an `unsafe` block to our code is a design change, not an
implementation detail — raise it as an issue first. `just deps` should stay short
and every crate in it pure Rust. This is the entire reason nidus exists instead of
DuckDB (bundled C++, multi-minute builds, FFI) or LanceDB (Arrow + DataFusion,
~10-minute builds).

### Miri (Undefined Behavior Checker)

`just miri` runs the test suite under [Miri](https://github.com/rust-lang/miri/).
Because nidus is pure safe Rust with **no FFI**, the *entire* crate compiles and
runs under Miri — unlike a C-backed store, where Miri can't execute the FFI and
the backend must be feature-gated out. Miri runs with `-Zmiri-disable-isolation`
so file-backed tests can touch a temp dir.

**When to add `#[cfg_attr(miri, ignore)]`** to a test:
- It calls `File::sync_all`/`sync_data` (fsync) or other filesystem syscalls Miri
  does not implement. Keep these in the file-backed integration tests.

**Do NOT ignore** pure-logic tests (cosine math, glob matching, filter evaluation,
the op-log/value codec round-trips). These operate on in-memory byte buffers, are
pure Rust, and must run under Miri. Prefer testing the codec against `Vec<u8>`
rather than a real file so coverage stays Miri-clean.

## Architecture Overview

nidus is an **embeddable vector store**: a library that holds dense vectors plus
typed metadata in a single on-disk store and answers nearest-neighbour queries by
**exact brute-force cosine**. It is the local storage leg for semantic-search and
indexing tools — a pure-Rust replacement for an embedded DuckDB/LanceDB. No SQL,
no query engine, no network, no background threads.

```
src/
├── lib.rs       # Public API: Nidus::{open, upsert, delete, delete_where, get_all, search, flush, compact}
├── value.rs     # Value (Null|Str|Int|Bool|List) + binary encode/decode
├── record.rs    # Record { id, vector, attrs }
├── filter.rs    # Predicate (Eq|Glob|In) / Filter + matching
├── glob.rs      # minimal * ? [..] matcher (covers the GLOB subset callers use)
├── search.rs    # cosine scoring + bounded top-k heap + min_score
├── data.rs      # flat f32 segment: header, append, row accessor (the future mmap seam)
├── log.rs       # op-log codec: len + payload + crc32, replay, torn-tail recovery
├── lock.rs      # writer exclusion via O_EXCL lock file (pure std, no flock/FFI)
├── crc.rs       # ~15-line table CRC32 (zero-dep checksum)
└── store.rs     # in-RAM index, write/read glue, compaction
```

**Storage model.** A store is a directory: `data` (append-only flat `f32` matrix,
fixed stride, never rewritten in place), `log` (append-only op stream — the commit
record), and `lock`. `open` reads `data` into RAM and replays `log` into an in-RAM
index (`collection → { id → (row, attrs) }`). Search is brute-force cosine over a
`Scope` — one collection, a subset, or the whole store — merged into one ranking
(sound because all collections share one embedding space); vectors are
unit-normalized on insert so `score = dot(v, q)`.

**Durability.** Per-batch fsync: append vectors → fsync `data` → append committing
log records → fsync `log`. A crash loses at most the in-flight batch (the index is
reproducible). Cross-process readers are lock-free: read `data` to size S, replay
`log`, ignore any record referencing a row ≥ S/dim — a consistent, possibly-stale
snapshot, never torn.

**Graceful failure (SPEC §6.6).** Appends are atomic (a partial write rolls back to
the row/frame boundary) and `upsert` is all-or-nothing (rolls `data`+`log` back to
entry marks on any failure), so a caught ENOSPC never corrupts the store. RAM growth
uses `try_reserve` (OOM → `Err`, not an abort) — except `attrs`/`id` clones, which
std gives no `try_reserve` for. The overcommit-proof guard is
`Config::max_vector_bytes` (refuse before allocating); `Nidus::footprint()` is the
introspection hook. Still in-RAM brute-force — not spill-to-disk/mmap (deferred).

**Deferred-but-seamed** (do NOT build until needed; each is additive over the same
file format): mmap (swap the one row accessor), an ANN/HNSW index, scalar
quantization. See `SPEC.md` for the full rationale and the decisions behind each.

## Conventions & Patterns

- **Pure safe Rust**: `#![forbid(unsafe_code)]`; popular pure-Rust crates only,
  never a C-compiling / native-linking crate. Non-negotiable.
- **Sync API**: nidus is synchronous (CPU + blocking file IO). Async callers wrap
  it in `Arc<Mutex<Nidus>>` + `spawn_blocking` (the same pattern used to wrap a
  blocking embedded DB connection).
- **One embedding space per store**: dimension is pinned in the `data` header at
  creation; reopening with a different dimension is a hard error. Many collections,
  one dimension.
- **Error handling**: `anyhow::Result` everywhere (`anyhow!`/`bail!`/`.context()`),
  matching the common Rust convention. No hand-rolled error enum, no `thiserror`.
- **Codec discipline**: all on-disk encoding is little-endian and explicit; every
  record is length-prefixed and CRC32-checked so a torn tail is detectable and
  recoverable. Test codecs against in-memory buffers (Miri-clean).
- **Commit style**: emoji prefix + short description (e.g. `🪺 op-log codec`).
- **Issue tracking**: `bd` (beads) — run `bd ready` for available work.
- **Branch workflow**: one branch per issue or bundled epic, push for PR review.
- **Tests**: pure-logic unit tests live inline per module; file-backed behavior in
  `tests/` against temp dirs (and `#[cfg_attr(miri, ignore)]` where they fsync).

### Integrating into a host application

A consuming tool maps its own document type onto a nidus `Record` (`id`, `vector`,
and an open `attrs` map — every field fits an attr; the `List`/`Null`/absent
distinction preserves "computed-empty" vs "un-indexed" semantics) and, if async,
wraps `Nidus` in `Arc<Mutex<Nidus>>` + `spawn_blocking`. nidus itself knows nothing
about the application's domain — it is a general-purpose vector store. See `SPEC.md`
§12 for the mapping pattern.

## Documentation site

The docs live in `docs/` — an Astro + Starlight site (`just docs` / `docs-build`
/ `docs-preview`), deployed to GitHub Pages at **nidus.duckedup.org** by
`.github/workflows/docs.yml` on push to `main` under `docs/**`.

**Positioning.** nidus is a vector store **for development and small-scale use**,
and is meant to support more than one usage mode: an embedded library today, with
a **standalone (read-only) search server** as a planned/designed-for seam (see
`SPEC.md` §9). Do NOT frame it as "an embeddable library, not a server" — the
server is a roadmap item, not a rejected idea. Keep the docs' wording open to it.

**Version sync — on every crate version bump, bump the docs too.** When you change
`version` in `Cargo.toml`, update the install-snippet version string in BOTH the
docs (`docs/src/content/docs/getting-started.md`) and `README.md` to match (e.g.
`nidus = "0.3"`). Those `[dependencies]` examples must not lag the released crate.
