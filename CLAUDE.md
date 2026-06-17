# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
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

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
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
just test          # run all tests (pure library — no cli feature)
just ci            # fmt-check + clippy (-D warnings) + test (pure library)
just lint          # clippy only (pure library)
just miri          # undefined behavior check via Miri (requires nightly)
just fmt           # format code
just build         # debug build
just release       # optimized release build
just doc           # build + open API docs
just deps          # assert the dependency tree is empty
```

Rust 1.96+ required (pinned via `rust-toolchain.toml`). Edition 2024.

### The `nidus` binary lives behind the opt-in `cli` feature

The crate ships an optional binary — the CLI plus `nidus serve` (an axum/tokio HTTP
wrapper, SPEC.md §9). It is gated behind the **non-default `cli` feature**, exactly
like the benchmarks are a separate member: the core recipes above (`just test`,
`ci`, `lint`, and Miri) build ONLY the pure library, so `cargo add nidus` and `just
deps` pull nothing beyond the four core crates and the FFI-free, seconds-long build
path stays intact. The binary's deps (`clap`, `tokio`, `axum`, `tower`,
`serde_json` — all still pure Rust, zero FFI) compile only under `--features cli`.

```bash
just ci-cli        # fmt-check + clippy + test, all with --features cli
just test-cli      # cargo test --features cli
just build-cli     # release build of the nidus binary
just serve DIR DIM # cargo run --features cli -- serve --dir DIR --dim DIM
just install       # cargo install --path . --features cli
```

When you touch `src/cli/`, `src/server/`, or `src/bin/`, gate it on the `cli`
feature and verify with `just ci-cli` (the core `just ci` does not compile it).
Do NOT move these deps into the default feature set or use them from the library
modules — that would break the pure-`cargo add nidus` install. The binary adapts
to the library (wire DTOs mirror `Hit`/`Footprint` in `src/server/dto.rs`), never
the reverse. `cargo binstall nidus` fetches prebuilt binaries via
`[package.metadata.binstall]`; `cargo install nidus --features cli` builds from
source.

### The dependency bar is BUILD-AND-SHIP SPEED, not zero-C — enforced

The real constraint is **build-and-ship cost** (fast compiles, no heavy toolchain,
no binary bloat), not language purity (SPEC §1, §13.6). nidus's core is **popular
pure-Rust crates** (`anyhow`, `serde`/`bincode`, `crc32fast`, …); the S3/GCS
persistence backends add sans-IO clients (`rusty-s3`/`tame-gcs`) over `ureq`, whose
default TLS is rustls + **`ring`** — a *small* C+asm compile. `ring` is **allowed**
(in the default build, not feature-gated, so `file://`→`s3://` is a runtime switch).
Our own code still carries `#![forbid(unsafe_code)]`.

**FORBIDDEN — the multi-minute C trees nidus exists to avoid:** bundled C/C++
(DuckDB's `libduckdb-sys`), vendored OpenSSL, `aws-lc-sys`, or a transitively-huge
graph (Arrow + DataFusion). The guardrail is empirical: **the whole-crate clean build
stays well under a minute** (measured ~7s; CI asserts it). Adding a dependency that
blows that budget — or a bundled-C / native-linking crate — is a design change, not
an implementation detail: raise it as an issue first. Judge a dep by "does it blow up
compile time / require a heavy toolchain / bloat the binary," not "is it pure Rust."

### Miri (Undefined Behavior Checker)

`just miri` runs the test suite under [Miri](https://github.com/rust-lang/miri/).
**All of nidus's own logic** runs under Miri — the codecs, search kernels, filters,
and the local file IO. Only the network paths in the S3/GCS backends are outside its
reach (and their unit tests — presigned-URL/request construction — are pure and DO run
under Miri; the localhost-mock round-trips are `#[cfg_attr(miri, ignore)]`). Miri runs
with `-Zmiri-disable-isolation` so file-backed tests can touch a temp dir.

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
├── lib.rs       # Public API: Nidus::{open, upsert, delete, delete_where, get_all, list, search, flush, compact}
├── value.rs     # Value (Null|Str|Int|Bool|List) + binary encode/decode
├── record.rs    # Record { id, vector, attrs }
├── filter.rs    # Predicate (Eq|Ne|Glob|In|NotIn|Lt|Le|Gt|Ge) / Filter + matching
├── glob.rs      # minimal * ? [..] matcher (covers the GLOB subset callers use)
├── search.rs    # distance kernels (cosine/dot/euclidean, f32 + int8 + binary Hamming) + bounded top-k heap + min_score
├── ann/         # opt-in ANN index (Config::ann): hnsw.rs graph + ivf.rs lists; seeded PRNG, no deps
├── data.rs      # flat f32 segment: header, append, row accessor (the future mmap seam)
├── log.rs       # op-log codec: len + payload + crc32, replay, torn-tail recovery
├── lock.rs      # writer exclusion via O_EXCL lock file (pure std, no flock/FFI)
├── crc.rs       # ~15-line table CRC32 (zero-dep checksum)
├── store/       # the integrator, split by concern (see "Keep files focused"):
│   ├── mod.rs    #   Store type + open/in_memory constructors + ANN index lifecycle glue
│   ├── scoring.rs#   scan kernels (f32/int8/binary chunk scorers) + parallel-scan engine
│   ├── quant.rs  #   int8/binary quant state + the quantized two-pass search
│   ├── read.rs   #   accessors, scan plumbing, exact + ANN search
│   ├── write.rs  #   upsert/delete/flush/compact + collection lifecycle
│   └── tests.rs  #   store tests (pure-logic + file-backed + quant/ANN)
│
│   # ── `cli` feature only (the `nidus` binary) — compiled with --features cli ──
├── bin/nidus.rs # thin entry point: parse args → cli::run
├── cli/         # clap subcommands over a store dir (serve, upsert, search, …)
└── server/      # axum/tokio HTTP wrapper over one Nidus; server/dto.rs = wire types
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
file format): mmap (swap the one row accessor). (int8 *and* binary quantization,
opt-in parallel scan via `Config::query_threads`, the HTTP server, and the opt-in
ANN index — `Config::ann`, HNSW + IVF in `src/ann/` — have since shipped; they were
once on this list.) See `SPEC.md` §9 for the full rationale and the decisions behind
each.

## Conventions & Patterns

- **Safe Rust, fast builds**: `#![forbid(unsafe_code)]` in our code; deps judged by
  build-and-ship cost, not purity (`ring`'s small TLS compile is allowed for the
  S3/GCS backends; multi-minute C/C++ trees are not — see above). Non-negotiable: the
  whole-crate clean build stays well under a minute.
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
- **Keep files focused — split by concern, not by size cap.** There is no hard line
  limit, but a module that has grown to cover several distinct concerns should be
  broken into a directory of sibling files, each owning one concern, with `mod.rs`
  holding the core type + the glue. `store/` is the worked example: `scoring`,
  `quant`, `read`, `write`, and `tests` each stand alone. **In Rust this costs almost
  nothing**: child modules see the parent's private items and private struct fields,
  so an inherent `impl Store` can span several files with **no** field made `pub`;
  only a method/type/fn *called across sibling submodules* needs `pub(super)` (e.g.
  `hits_from_topk`, `rebuild_quant`). Keep state types beside the code that reads
  their internals (e.g. `Int8State` lives with the quantized search) so their fields
  stay private. When you add a big new concern to an already-large module, prefer a
  new sibling file over appending to it — and move the matching tests into the
  module's own `tests.rs` rather than growing one giant test block.
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

**Positioning.** nidus is a vector store **for development and small-scale use**.
Keep the public framing open: do NOT pin it down as "an embeddable library" (or
"a library, not a server") and do NOT make public promises about future modes
(no "server planned / on the roadmap" in the docs/README). Describe what it does
today, neutrally, without limiting where it can go. (A server is one of the
deferred seams in `SPEC.md` §9 — internal context, not a public commitment.)

**Bump the version in EVERY PR — releases are automatic.** `.github/workflows/release.yml`
runs on push to `main`: it reads `version` from `Cargo.toml`, and releases (tag
`v<version>`, GitHub release, prebuilt `cargo binstall` binaries) **only if that tag
does not already exist**. So a PR that doesn't bump `version` ships nothing — the tag
is already there and the release is silently skipped. Every PR with a user-visible or
behavioural change MUST bump `Cargo.toml` `version` (semver: patch for fixes/refactors,
minor for new features/behaviour, major for breaking API). Pure-internal no-op churn
is the only exception.

**Version sync — on every crate version bump, bump the docs too.** When you change
`version` in `Cargo.toml`, update the install-snippet version string in BOTH the
docs (`docs/src/content/docs/getting-started.md`) and `README.md` to match (e.g.
`nidus = "0.3"`) — but only when the `major.minor` changes, since the snippets pin
`major.minor` (a patch bump like `0.12.1 → 0.12.2` leaves `nidus = "0.12"` correct).
Those `[dependencies]` examples must not lag the released crate.
