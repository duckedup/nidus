# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

## Core Foundation: Speed, Testing, Stable

Three commitments every change is judged against (SPEC §1 — trading one away is a
design change, not an implementation detail):

1. **Speed** — the clean build stays in seconds and CI asserts it; dependencies are
   judged by build-and-ship cost. Never trade this away silently.
2. **Testing** — verify against the real artifact, never assume (SPEC §11). Every
   behaviour claim is backed by a test that runs in CI: surfaces only a real binary
   can prove get e2e tests (`tests/e2e/`), and the SDK↔server contract runs against a
   real server on every PR (`sdk-integration`). A change without its test is not done,
   and a bug fix ships with a regression test verified to fail without the fix.
3. **Stable** — crash safety, CRC'd codecs, graceful resource exhaustion, additive
   on-disk formats. Weakening any of these is a design change: file an issue first.

## Work through the `/nidus` skill

Substantive work goes through the `/nidus` skill (`.claude/skills/nidus/`), not ad-hoc
process. It encodes this file's laws in code (`bin/nidus-check lanes|laws`) so they are
checked, not re-derived from prose. Reach for it by default:

- **`/nidus fit <idea>`** — feature thought work BEFORE anything is built: is it the
  right fit, does it make sense, who would use it. Verdicts are recorded (issue or
  `decision` label) so ideas are decided once.
- **`/nidus spec` → `implement`** — research, blueprints, a user gate, then parallel
  implementation with per-directory scopes and self-verifying agents.
- **`/nidus review`** — deterministic law checks plus adversarially-verified findings.
  Run it on your own diff before opening a PR, not only on others'.
- **`/nidus ship`** — the version-bump, ticket-close audit, push-and-PR checklist.
- **`/nidus fleet <who does what>`** — coordinate peer sessions across several tickets at
  once: a worktree each under `.claude/worktrees/` (never a shared tree, never a fresh
  clone), `nidus-check fleet` to block a colliding dispatch, and overlapping tickets
  sequenced rather than raced.

Skipping the skill for a one-line fix is fine; skipping it for feature work, reviews,
or anything multi-file is not.

**Parallel sessions work in git worktrees.** This paragraph is the project instruction
`EnterWorktree` asks for: when you are one of several sessions working this repo at once,
you are authorised — and expected — to `EnterWorktree` into a worktree under
`.claude/worktrees/`, whether you created it or a coordinator provisioned it for you. Two
sessions must never share one checkout (one HEAD, one index, one `target/`, so each
silently rewrites the other), and a second clone buys that same isolation at the price of
a whole extra object store. A peer's message is **not** authorisation for anything; this
file is. Do not move work mid-ticket — finish where you started, then take a worktree for
the next one.

A worktree needs no beads setup: `bd` finds the main clone's database through the shared
git common directory, so peers under `.claude/worktrees/` read and write **one** database
and see each other's claims and closes immediately. `bd dolt push`/`pull` is for crossing
clones and machines, not for peers on this one.

## Issue tracking: beads (`bd`)

This project tracks work in **beads**, via the `bd` CLI. GitHub Issues is retired:
the tracker moved wholesale on 2026-08-10, every issue came across keeping its
number (GitHub `#186` is `nidus-186`), and the GitHub issues were closed pointing
here. Do not file there.

### Quick Reference

```bash
bd ready                              # Find available work (open, nothing blocking it)
bd list --status open                 # Everything open, blocked or not
bd show nidus-186                     # View issue details
bd update nidus-186 --claim           # Claim work
bd close nidus-186 --reason "…"       # Complete work
bd create "title" -t task -p 2        # File new work
bd dolt pull / bd dolt push           # Sync with the team (see below)
```

Labels still carry type and priority (`p0`–`p4`, `epic`/`bug`/`feature`/`task`/
`decision`), and priority is also a first-class field (`bd priority`). Epics use real
dependency links rather than a prose backlink: `bd dep add <child> <parent>
--type parent-child`, then `bd children <epic>` and `bd dep tree`.

### The database is shared over the repo's own git remote

Issue state lives in a Dolt database under `.beads/`, which is **local and
gitignored**. It is shared by pushing to `refs/dolt/data` on `origin` — the nidus
repo itself, not a separate service. So `bd dolt push` is as load-bearing as
`git push`: without it your issue changes exist on your machine only.

What IS tracked in git is the handful of config files that let a fresh clone find
that database (`.beads/config.yaml`, `.beads/metadata.json`, `.beads/.gitignore`,
`.beads/hooks/`). Everything else under `.beads/` is runtime.

**In a fresh clone, run `just bd-setup`** — it reads the tracked config, recovers the
database from `refs/dolt/data`, wires the remote, and you have the issues. This
requires the `dolt` CLI on PATH (`brew install dolt`); `bd` alone can push but cannot
clone. It is safe to re-run: an existing database is left untouched, because it may
hold work that was never pushed.

**It is deliberately NOT `bd bootstrap`** (`scripts/bd-setup.sh` does that job
directly). Bootstrap reads the tracked `sync.remote`, rejects its `git+ssh://` form as
"not a Dolt remote", and so never reaches its own `refs/dolt/data` branch — leaving a
fresh clone with an **empty** tracker and no error naming the cause (nidus-1oq). That
silence is the danger: an empty database reads as divergent history, and the recovery
`bd dolt pull` then offers is `bd dolt push --force`, which would force-push nothing
over everyone's issues. If you ever see that prompt, stop.

Never `bd init` against this repo either: it mints a new identity and can force-push
over everyone else's history (`bd help init-safety`). A **worktree** needs none of
this — it shares the main clone's database directly. `just bd-sync` is pull-then-push
when you finish.

### Rules

- Use beads for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Durable knowledge goes in the issue that owns it, or in `SPEC.md` — do NOT use MEMORY.md files
- **Never let the JSONL export become tracked.** `.beads/issues.jsonl` is a local
  viewer/backup artifact and is gitignored, with `export.git-add` off. The retired
  pre-migration tracker committed that file and rewrote it from each branch's local
  database, so any branch could silently revert another's closes (#83). The shared
  state is the Dolt ref; a tracked export recreates the bug.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   bd dolt push                 # issue state — a separate ref, NOT carried by git push
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
   Issue state is not in your commit, so `git push` does not carry it: closing a
   ticket and pushing the branch still leaves the close on your machine. Push both,
   and close anything the PR does not itself close.
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds


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
just deps          # print the dependency tree (cargo tree -p nidus)
```

Rust 1.96+ required (pinned via `rust-toolchain.toml`). Edition 2024.

### The `nidus` binary lives behind the opt-in `cli` feature

The crate ships an optional binary — the CLI plus `nidus serve` (an axum/tokio HTTP
wrapper, SPEC.md §9). It is gated behind the **non-default `cli` feature**, exactly
like the benchmarks are a separate member: the core recipes above (`just test`,
`ci`, `lint`, and Miri) build ONLY the pure library, so `cargo add nidus` keeps the
seconds-long build path intact. The binary's deps (`clap`, `tokio`, `axum`, `tower`,
`serde_json`, `tar`, `flate2` — all pure Rust, zero FFI) compile only under
`--features cli`; the AI ingest layer (`embed`/`summarize`/`memory`/`mcp`, which add
`reqwest` + `rmcp`) is likewise off by default.

The default build is **not** the four-crate core any more, and CI does not claim it
is: alongside `anyhow`/`serde`/`bincode`/`crc32fast` it carries `regex` (§7.5), the
S3/GCS/redis backend stack (`rusty-s3`, `tame-gcs`, `tame-oauth`, `ureq`, `url`,
`http`, `redis` — deliberately un-gated so `file://`→`s3://` is a runtime switch, not
a recompile), and `memmap2` + `bytemuck` for the mmap seam. So the default build is
not FFI-free either — `ring` (ureq's TLS) and `memmap2` are the two conscious
opt-ins. What is enforced is the *budget*, not the crate count: whole-crate clean
build well under a minute (see the dependency bar below).

```bash
just ci-cli        # fmt-check + clippy + test, all with --features cli
just test-cli      # cargo test --features cli
just build-cli     # release build of the nidus binary
just serve DIR DIM # cargo run --features cli -- serve --dir DIR --dim DIM
just install       # cargo install --path . --features cli
```

`nidus serve` also answers **MCP `2026-07-28`** at `/mcp` behind the `mcp` feature
(folded into `serve`), so any MCP client can use the memory layer as agent memory.
`src/server/mcp/` is an *adapter*: every tool routes through the same
`run_read`/`run_write` helpers the HTTP handlers use, and the service is
`nest_service`'d **inside** the middleware stack so it inherits the body limit,
backpressure, bearer auth, and metrics rather than reimplementing any of them. Two
things there are load-bearing and easy to break: the tool surface is **text-native**
(no tool may take a raw `vector` — a model cannot emit one, and `tests/e2e/mcp/`
asserts it), and tool schemas are **hand-written JSON**, never `schemars`-derived,
because the descriptions drive tool-selection quality. The same holds for resources
and prompts: resource content and prompt messages carry `{id, attrs}`, never a
vector. Verify with
`cargo clippy --all-targets --features mcp -- -D warnings` and `just test-e2e`.

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
Our own code still carries `#![deny(unsafe_code)]` (see "Safe Rust" below for why
`deny` and not `forbid`).

**FORBIDDEN — the multi-minute C trees nidus exists to avoid:** bundled C/C++
(DuckDB's `libduckdb-sys`), vendored OpenSSL, `aws-lc-sys`, or a transitively-huge
graph (Arrow + DataFusion). The guardrail is empirical and **CI-enforced**: the
`build-budget` job times a clean, uncached, offline build of the default features on
every PR and fails past 60s (measured ~7s; the bound is order-of-magnitude on purpose,
like the scale.rs timings, so it never flakes and still catches a bundled-C tree).
Adding a dependency that
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

**When to add `#[cfg_attr(miri, ignore)]`** to a test — and **always say which reason
applies**, either trailing the attribute (`#[cfg_attr(miri, ignore)] // <why>`) or on the
line directly above it, both of which the checker reads. There are only three:
- **Syscalls Miri does not implement** — `File::sync_all`/`sync_data` (fsync) and friends.
  Keep these in the file-backed integration tests.
- **Runtime cost.** The interpreter is orders of magnitude slower, so an N=2000 ANN build
  or a threaded scan will not finish. Say so; do not leave it looking like reason one.
- **Float ULP.** Miri's libm differs from the host in the last bit, so a test asserting
  *exact* `f32` score bits through a transcendental (BM25's IDF uses `ln`) fails under
  Miri while the ranking is identical. Compare ranks, or ignore with this reason named.

The comment is load-bearing, not decoration: `nidus-check laws` treats a bare ignore as
unresolved and a documented one as settled, so an undocumented ignore nags forever and a
documented one stops. That is the whole difference between an actionable check and
ambient noise. It also cuts the other way — the reason is the only thing standing between
a correct ignore and someone deleting it, which is exactly what happened to two float-ULP
tests here when the checker read only one of the two comment positions.

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
├── lib.rs        # Public API: Nidus::{open, upsert, delete, delete_where, get_all, list, search, flush, compact}
├── config.rs     # Config: Fsync, OpenMode, ann/quant/memory/persistence settings (SPEC §4.1)
├── model.rs      # the shared type vocabulary — Value, Record, Predicate/Filter, Op, Distance,
│                 #   Quantization, AnnConfig, FtsQuery/FtsClause/… (pure defs + serde)
├── glob/         # minimal * ? [..] matcher (covers the GLOB subset callers use, SPEC §7.1)
├── filter/       # filter evaluation: mod.rs (dispatch + per-query validate/prepare),
│                 #   text.rs (Levenshtein + the filter tokenizer, §7.4), pattern.rs (regex, §7.5)
├── search/       # distance kernels (cosine/dot/euclidean, f32 + int8 + binary Hamming)
│                 #   + bounded top-k heap + min_score
├── data/         # the vector segments: mod.rs (DataSegment — header, append, row accessor),
│                 #   segments.rs (the live set as one global row space), mmap.rs (the ONE
│                 #   memory-map seam — the crate's only scoped `allow(unsafe_code)`)
├── manifest/     # the atomic commit point naming the live segments (SPEC §14.2)
├── log/          # op-log codec (the WAL): len + payload + crc32, replay, torn-tail recovery
├── lock/         # writer exclusion via O_EXCL lock file (pure std, no flock/FFI)
├── index_cache.rs# shared codec for derived caches (ann/fts); a stale/torn load rebuilds, never fatal
├── ann/          # opt-in ANN index (Config::ann): hnsw.rs graph + ivf.rs lists + persist.rs
├── fts/          # opt-in BM25 index: analyzer.rs, fold.rs, schema.rs, highlight.rs
├── annotate.rs   # opt-in result annotations — why a hit matched (SPEC §7.8)
├── fuse.rs       # Reciprocal Rank Fusion: merge several ranked legs into one ranking
├── backend/      # pluggable storage & memory tiers (SPEC §13): local, ram, object, s3, gcs, redis
├── embed/        # embedding providers: voyage, openai, ollama, cohere, gemini, mistral,
│                 #   jina, openai_compat
├── summarize/    # single-shot text summarization (anthropic, openai)
├── memory.rs     # the text-native memory API the MCP surface is built on
├── providers.rs  # provider capability registry
├── http.rs       # shared HTTP retry infrastructure for the reqwest-based adapters
├── cancel.rs     # cooperative cancellation for long scans
├── diag.rs       # levelled logfmt diagnostics on stderr
├── metrics.rs    # process-wide counters, exported as Prometheus text by GET /metrics
├── store/        # the integrator, split by concern (see "Keep files focused"):
│   ├── mod.rs    #   Store type + open/in_memory constructors + lock/ANN lifecycle glue
│   ├── scoring.rs#   scan kernels (f32/int8/binary chunk scorers) + parallel-scan engine
│   ├── quant.rs  #   int8/binary quant state + the quantized two-pass search
│   ├── read.rs   #   accessors, scan plumbing, exact + ANN search
│   ├── text.rs   #   multi-clause BM25 + the hybrid RRF fusion + annotations (§7.8)
│   ├── rank.rs   #   ranking expressions: recency decay, leg weights, ORDER BY (§7.6)
│   ├── aggregate.rs # count/sum from the in-RAM index + result diversity via limit_per (§7.7)
│   ├── write.rs  #   upsert/delete/flush/compact + collection lifecycle
│   ├── memtier.rs#   publish/adopt the in-RAM working set against the memory tier (§13.3)
│   └── tests.rs  #   store tests (pure-logic + file-backed + quant/ANN)
│
│   # ── `cli` feature only (the `nidus` binary) — compiled with --features cli ──
├── bin/nidus.rs  # thin entry point: parse args → cli::run
├── cli/          # clap subcommands over a store dir (serve, upsert, search, …) + backup.rs
└── server/       # axum/tokio HTTP wrapper over one Nidus; dto.rs = wire types, alongside
                  #   auth.rs, limits.rs, commit.rs, metrics.rs
                  # server/mcp/ = the MCP 2026-07-28 surface at /mcp (`mcp` feature):
                  #   mod.rs, remember.rs, search.rs, admin.rs, hygiene.rs, args.rs,
                  #   stdio.rs (the stdio transport), resources.rs, prompts.rs, uri.rs
```

SPEC.md §10 carries the same map with more detail on each module's contract; when the
two disagree, the tree wins and both are stale.

**Storage model.** A store is a set of objects behind a `Persistence` backend (SPEC
§13) — a local directory by default, an `s3://`/`gs://` prefix by URL: `manifest`
(the live-segment set + the pinned dimension/distance — the atomic commit point,
§14.2), one or more fixed-stride `f32` segments (the base segment is still named
`data`; sealing mints `seg-NNNNNNNN`), `log` (append-only op stream — the commit
record), and `lock`. Segments are append-only in normal operation; the one exception
is `compact()`, which collapses the live set and **rewrites the base segment in
place** (`Segments::rewrite`), leaving the sealed ones unreferenced for deletion.
`open` reads the manifest, loads the live segments into one global row space, and
replays `log` into an in-RAM index (`collection → { id → (row, attrs) }`). Search is
brute-force cosine over a `Scope` — one collection, a subset, or the whole store — merged into
one ranking (sound because all collections share one embedding space); vectors are
unit-normalized on insert so `score = dot(v, q)`.

**Durability.** Per-batch fsync, and the write order is load-bearing: append vectors
to the active segment → fsync it → append committing log records → fsync `log`. So a
committed `Upsert`'s row is already durable before anything references it, and a crash
loses at most the in-flight batch (the index is reproducible). Cross-process readers
are lock-free: read the manifest, open the segments it names for a total of N rows,
replay `log`, and ignore any record referencing a row ≥ N — a consistent,
possibly-stale snapshot, never torn. `Nidus::refresh()` advances that snapshot in place
by re-applying the same rule at a newer manifest version (SPEC §6.2, §14.6).

**Graceful failure (SPEC §6.6).** Appends are atomic (a partial write rolls back to
the row/frame boundary) and `upsert` is all-or-nothing (rolls `data`+`log` back to
entry marks on any failure), so a caught ENOSPC never corrupts the store. RAM growth
uses `try_reserve` (OOM → `Err`, not an abort) — except `attrs`/`id` clones, which
std gives no `try_reserve` for. The overcommit-proof guard is
`Config::max_vector_bytes` (refuse before allocating); `Nidus::footprint()` is the
introspection hook.

**Deferred-but-seamed** (do NOT build until needed; each is additive over the same
file format): see `SPEC.md` §9 "Still deferred" for the current list and the
reasoning behind each. Much of what this section used to name has since shipped —
int8 *and* binary quantization, opt-in parallel scan via `Config::query_threads`,
the HTTP server, the opt-in ANN index (`Config::ann`, HNSW + IVF in `src/ann/`), and
mmap (`src/data/mmap.rs`) — so check §9 rather than trusting a list here.

## Conventions & Patterns

- **Safe Rust, fast builds**: `#![deny(unsafe_code)]` in our code — `deny`, not
  `forbid`, because the single memory-map call in `src/data/mmap.rs` (SPEC §9/§14.6)
  carries a scoped `#[allow(unsafe_code)]`. That module is the **only** place the
  allow appears; every other `unsafe` in the crate stays a hard compile error, and a
  second scoped allow is a design change, not an implementation detail. Deps judged by
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
- **Comments: 3 lines maximum, and they must add clarity.** This is a hard cap on every
  comment and doc comment — `//` and `///` — counting the whole block, including any
  `///` blank separators. A comment earns its place by saying something the code cannot:
  the non-obvious *why*, a constraint that will bite, a bug it guards against. It does not
  earn its place by restating the code, justifying the design at length, arguing with an
  imagined reviewer, or recording the history of how the decision was reached. **Rationale
  that needs more than three lines belongs in the commit message, the PR, `SPEC.md`, or a
  `bd` issue — not above the code.** When trimming, keep the fact and drop the argument:
  "rmcp reports `rmcp 3.1.1` here, not this crate" beats a paragraph explaining why that
  matters. Long comments are not thoroughness; they push the code off the screen and go
  stale where prose in a commit cannot. **Two exceptions.** A doc example (a ```` ``` ````
  fence) is test code, not commentary, so it does not count toward the cap — the prose
  around it still does. And **`//!` module/crate docs are exempt entirely**: that block is
  the crate's published rustdoc landing page (what a reader meets on docs.rs), not
  commentary sitting between a reader and the code, so the reason for the cap — keeping
  code on screen — does not apply. `//!` earns no licence to ramble; it is just reviewed
  as documentation rather than counted. The exemption is **per line, not per block**: a
  `///` doc that happens to abut a `//!` one — no blank line between, which is ordinary
  Rust — is still counted, or a stray `//!` would be a one-line way to dodge the cap.
  `lib/laws.mjs` `commentCap` implements both exceptions.
- **Commit style**: emoji prefix + short description (e.g. `🪺 op-log codec`).
- **Issue tracking**: beads — run `bd ready` for available work.
- **Branch workflow**: one branch per issue or bundled epic, push for PR review.
- **The merge queue is what makes parallel PRs safe, so keep it fed.** Each PR is
  retested against `main` *plus the entries queued ahead of it*, which is the only
  thing that catches two individually-green PRs that are jointly broken (#135 and
  #137 each passed, then broke `main` together until #146). The mechanism is
  fragile in one specific way: a queued PR builds on a temporary
  `gh-readonly-queue/**` ref and fires a **`merge_group`** event, so any workflow
  owning a required check MUST list `merge_group:` under `on:`. A required check
  that does not run there never reports and the entry stalls until it is ejected —
  the queue looks broken when it is really just waiting. Adding a required check
  and adding its `merge_group` trigger are one change, never two.
  **What the queue re-runs is narrowed on purpose:** only `fmt`, `clippy`, `test`
  and `release` do real work there. The other eleven required checks still trigger
  on `merge_group` and still report — they short-circuit via a per-step
  `if: env.QUEUE_LITE != 'true'` guard, never a job-level `if`, because a job that
  is skipped outright is exactly the check that never reports. So the rule above is
  unchanged: the trigger is mandatory, the *work* is what got trimmed. The trade is
  deliberate — a pair of PRs that are jointly broken only in e2e, Miri, or an SDK
  lane will now land, and `main`'s own push build is what catches it.
- **CLOSE THE TICKET YOURSELF WHEN THE PR MERGES — NOTHING AUTO-CLOSES.** A
  `Closes nidus-186` line in a PR body is documentation and nothing more: GitHub
  cannot close a bead, so the trailer that used to do the work now only *looks*
  like it did. Run `bd close nidus-186 --reason "…"` and `bd dolt push` as part of
  shipping. **This law has been broken in both directions and they are the same
  failure.** Under the pre-migration tracker finished tickets sat silently open —
  a sweep in PR #63 found ten, three of them P1 bugs fixed weeks earlier, two
  whose work had never landed at all. GitHub's auto-close then introduced the
  opposite lie, firing on merge whether or not the work survived review, so a PR
  gutted down to a fraction of an issue still closed it. Both are the tracker
  asserting something the tree does not support. So the constant survives the
  move, and the manual close makes the first direction the live risk again:
  verify the claim against the tree, never against the note on the issue, and
  before you close, confirm the merged diff actually finishes that issue.
- **Tests**: pure-logic unit tests live inline per module; file-backed behavior in
  `tests/` against temp dirs (and `#[cfg_attr(miri, ignore)]` where they fsync).
  **End-to-end tests that drive the real binary** live in `tests/e2e/`
  (`just test-e2e`, `cli`-gated): `harness.rs` spawns `nidus serve` via
  `env!("CARGO_BIN_EXE_nidus")` on `--addr 127.0.0.1:0`, learns the port from the
  startup line, polls `/health`, and kills + reaps the child on `Drop`; the suites
  beside it (`server.rs`, …) hold only assertions. Deliberately **one** test binary
  (`tests/e2e/main.rs` + sibling modules), because each `tests/*.rs` file is its own
  crate — a second file would mean a second copy of the harness. Add a new suite as a
  module here, not as a new `tests/*.rs`. These cover what the in-process
  `tower::oneshot` server tests structurally cannot: the real bind, the CLI-flag →
  `ServeConfig` wiring, socket framing, cross-process locking, and restart.
  `cluster.rs` goes further — several real instances over a **real** object store and
  memory tier — and is therefore `#[ignore]`d: bring the services up with `just
  e2e-services-up` (defined once in `scripts/e2e-services.sh`, shared with CI so the
  two cannot drift) and run `just test-e2e-cluster`. `.github/workflows/integration.yml`
  runs the whole target with `--include-ignored` on every PR, so anything added here
  is enforced. When a cluster test fails, read the panic first: the harness attaches
  the offending process's own stderr. `scale.rs` is the ranking-correctness lane: 10k
  384-d vectors ingested over HTTP, top-k checked against cosine ground truth computed
  in-test, so a scoring/normalisation/JSON-round-trip bug fails loudly where a
  three-vector smoke test cannot see it. Its timing assertions are **order-of-magnitude
  only** and must stay that way — it is a debug build on a shared runner, so a tight
  bound flakes and proves nothing. Real performance work belongs in `benchmarks/`
  (`just bench`, `--release`); note that harness drives the library **in-process**, so
  the HTTP path's own cost is still unbenchmarked (nidus-8fn).

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

**Style: NO em dashes in user-facing prose** (docs site, README, SDK READMEs).
Reword with a period, comma, colon, or parentheses. A sweep removed them all;
do not reintroduce them. (En dashes in numeric ranges are fine.)

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

**The client SDKs ship at the crate's version — and you do NOT hand-edit theirs
either.** Every SDK under `sdks/` is released at `Cargo.toml`'s `version`, so "which
nidus does this client speak to" is answerable from the version alone. `release.yml`
invokes each SDK's release workflow (`sdk-js-release.yml`, `sdk-go-release.yml`,
`sdk-py-release.yml`) via `workflow_call` after it cuts a release — same mechanism, and
same two reasons, as `publish-docker`/`publish-helm`: a tag pushed with `GITHUB_TOKEN`
cannot trigger another workflow, and calling directly means an SDK can only publish a
version the crate actually released. Any version string in an SDK manifest
(`sdks/js/package.json`, `sdks/python/src/nidus/_version.py`) is **stamped from
`Cargo.toml` at release time and committed back to `main`**, exactly like the chart
below; the Go SDK has no version file at all, since a Go module's version *is* its tag.
So an SDK-only fix still ships by bumping `Cargo.toml` — there is no separate SDK bump
to remember, and no CI assertion to trip over. Tag namespaces are distinct from the
crate's `v*`: `js-v*`, `py-v*`, and `sdks/go/v*` (that last form is forced — Go resolves
a module in a repo subdirectory only via a `<subdir>/v<semver>` tag).

**Do NOT hand-edit `charts/nidus/Chart.yaml` versions.** Both `version` and
`appVersion` are stamped from `Cargo.toml` by `.github/workflows/helm-publish.yml` at
release time, and the release job commits the stamp back (nidus-yap). A PR that bumps
the crate should leave the chart alone — editing it just creates a conflict with the
bot commit. This used to be a hand-edit enforced by a CI assertion; that fired on
essentially every PR, which is why it is derived now.

**Known caveat: the stamp-back has never actually landed (#82).** `main`'s ruleset
rejects the `github-actions[bot]` push with `GH013`, so `charts/nidus/Chart.yaml`,
`sdks/js/package.json` and `sdks/python/src/nidus/_version.py` on `main` lag what was
really published. **Only the repo is wrong — the stamp is applied *before* publish, so
everything that did publish (PyPI, the OCI chart) got the correct version.** npm is the
exception, and for an unrelated reason: #81 has blocked every JS publish since 0.2.0, so
that job fails at the publish step and never reaches stamp-back at all. Do not "fix" the
drift by hand: a hand-stamp asserts a version that may never have been published, and
`sdks/js/package.json` is exactly that case — stamping it to the crate version would
claim an npm release that does not exist. The mechanism is right; it
needs a one-time bypass on `main`'s ruleset. Until then `scripts/stamp-back.sh` makes
each failure loud in the release run instead of silently green, and setting
`STAMP_BACK_STRICT=1` turns it into a hard job failure once the bypass is in place.
