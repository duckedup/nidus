# Project Instructions for AI Agents

Each rule here is one line. The argument and the incident history behind it live in
`decisions/` (the `D####` pointers), and path-scoped detail lives in `.claude/rules/`, which
loads only when you touch matching files. `.claude/skills/nidus/bin/nidus-check laws`
enforces what a script can enforce — run it rather than re-deriving a rule from prose.

<!-- Maintainer note: HTML comments are stripped before this file enters context, so notes
     here are free. This file loads into every session AND every subagent, so a line that
     only matters when someone challenges a rule belongs in decisions/, not here. The cap is
     200 lines and `nidus-check laws` enforces it. -->

## Communication style

Be direct, short, and straight to the point. Lead with the answer or the result, then only
the detail that changes what the reader does next. No preamble, no restating the request,
no summarising work the diff already shows. Say plainly when something failed or was
skipped — brevity is not a licence to omit bad news.

## Core Foundation: Speed, Testing, Stable

Three commitments every change is judged against (SPEC §1 — trading one away is a
design change, not an implementation detail):

1. **Speed** — the clean build stays in seconds and CI asserts it; dependencies are
   judged by build-and-ship cost (D0005). Never trade this away silently.
2. **Testing** — verify against the real artifact, never assume (SPEC §11). Every
   behaviour claim is backed by a test that runs in CI: surfaces only a real binary
   can prove get e2e tests (`tests/e2e/`), and the SDK↔server contract runs against a
   real server on every PR (`sdk-integration`). A change without its test is not done,
   and a bug fix ships with a regression test verified to fail without the fix.
3. **Stable** — crash safety, CRC'd codecs, graceful resource exhaustion, additive
   on-disk formats. Weakening any of these is a design change: file an issue first.

## Work through the `/nidus` skill

Substantive work goes through the `/nidus` skill (`.claude/skills/nidus/`), not ad-hoc
process. `SKILL.md` routes to one lane file per subcommand: `fit` (is this the right thing to
build), `spec` → `implement` (research, blueprints, a user gate, then parallel agents),
`review` (deterministic law checks plus adversarially-verified findings — run it on your own
diff before opening a PR), `ship` (version bump, ticket-close audit, push and PR), `fleet`
(peer sessions across several tickets at once).

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

## Issue tracking: beads (`bd`)

This project tracks work in **beads**, via the `bd` CLI. GitHub Issues is retired: the
tracker moved wholesale on 2026-08-10, every issue came across keeping its number (GitHub
`#186` is `nidus-186`), and the GitHub issues were closed pointing here. Do not file there.

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

Labels carry type and priority (`p0`–`p4`, `epic`/`bug`/`feature`/`task`/`decision`), and
priority is also a first-class field (`bd priority`). Epics use real dependency links rather
than a prose backlink: `bd dep add <child> <parent> --type parent-child`, then
`bd children <epic>` and `bd dep tree`.

### The database is shared over the repo's own git remote

Issue state lives in a Dolt database under `.beads/`, which is **local and gitignored**. It is
shared by pushing to `refs/dolt/data` on `origin` — the nidus repo itself, not a separate
service. So **`bd dolt push` is as load-bearing as `git push`**: without it your issue changes
exist on your machine only. What IS tracked in git is the handful of config files that let a
fresh clone find that database (`.beads/config.yaml`, `.beads/metadata.json`,
`.beads/.gitignore`, `.beads/hooks/`); everything else under `.beads/` is runtime.

**In a fresh clone, run `just bd-setup`** — it reads the tracked config, recovers the database
from `refs/dolt/data`, and wires the remote. Needs the `dolt` CLI (`brew install dolt`); `bd`
alone can push but cannot clone. Safe to re-run: an existing database is left untouched,
because it may hold work that was never pushed.

**It is deliberately NOT `bd bootstrap`, and never `bd init`.** Bootstrap cannot reach this
repo's own `refs/dolt/data` and leaves a fresh clone with an *empty* tracker and no error
naming the cause; the recovery it then offers would force-push nothing over everyone's
issues. `bd init` mints a new identity and can do the same. If you ever see a
`bd dolt push --force` prompt, stop (D0002). A **worktree** needs none of this — it shares the
main clone's database directly, so peers see each other's claims and closes immediately.
`just bd-sync` is pull-then-push when you finish.

### Rules

- Use beads for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Durable knowledge goes in the issue that owns it, in `SPEC.md`, or in `decisions/` — do NOT
  use MEMORY.md files
- **Never let the JSONL export become tracked.** `.beads/issues.jsonl` is a local
  viewer/backup artifact, gitignored, with `export.git-add` off. A tracked export is rewritten
  from each branch's local database, so any branch can silently revert another's closes — and
  one did (D0001). The shared state is the Dolt ref.
- **CLOSE THE TICKET YOURSELF WHEN THE PR MERGES — NOTHING AUTO-CLOSES.** A `Closes nidus-186`
  line in a PR body is documentation and nothing more: GitHub cannot close a bead. Run
  `bd close nidus-186 --reason "…"` and `bd dolt push` as part of shipping, and before you
  close, confirm the merged diff actually finishes that issue (D0004).

## Session Completion

Work is NOT complete until `git push` succeeds. Do not stop before pushing, and do not say
"ready to push when you are" — you push.

1. File issues for anything that needs follow-up.
2. Run the quality gates for what changed (`just ci`, plus `just ci-cli` if you touched the
   binary).
3. Close finished work, update in-progress items.
4. Push **both**, because they travel separately:
   ```bash
   bd dolt push                 # issue state — a separate ref, NOT carried by git push
   git pull --rebase && git push
   git status                   # MUST show "up to date with origin"
   ```
5. Clear stashes, prune remote branches, and hand off context for the next session.

## Build & Test

```bash
just test          # all tests (pure library — no cli feature)
just ci            # fmt-check + clippy (-D warnings) + test (pure library)
just lint          # clippy only          just fmt      # format
just miri          # UB check via Miri (nightly)
just build         # debug                just release  # optimized
just doc           # build + open API docs
just deps          # dependency tree (cargo tree -p nidus)
just spec toc      # SPEC.md section index — see below
```

Rust 1.96+ (pinned via `rust-toolchain.toml`). Edition 2024. `just --list` has the rest,
including the `cli`, `serve`, and wasm lanes.

**Do not read `SPEC.md` whole — it is 2577 lines.** `just spec toc` is the index,
`just spec find <words>` says which section covers a topic, `just spec <ref>` prints one
section. It works on any tracked doc: `just spec --file CLAUDE.md find miri`.

## Laws that apply before you touch a file

- **Bump `Cargo.toml` `version` in every PR** with a user-visible or behavioural change.
  Releases fire on push to `main` only if the tag does not exist, so a PR that does not bump
  ships nothing, silently. Do NOT hand-edit chart or SDK version files (D0007).
- **A feature ships whole, in one PR** — core, HTTP, CLI, MCP, all three SDKs, docs. The SDKs
  are not a follow-up and not out of scope (D0012).
- **These are design changes, not implementation details**, and each needs an issue first: a
  dependency that blows the build budget (D0005), a second `allow(unsafe_code)` (D0006), a
  non-additive on-disk format change, or weakening a Core Foundation commitment.
- **No em dashes in user-facing prose** (docs site, README, SDK READMEs). Reword with a
  period, comma, colon, or parentheses. En dashes in numeric ranges are fine.
- **Positioning:** nidus is a vector store for development and small-scale use. Describe what
  it does today; do not pin it to "an embeddable library" and do not promise future modes.
- Commit style: emoji prefix + short description (e.g. `🪺 op-log codec`).
- One branch per issue or bundled epic; push for PR review.

## Where the rest lives

| Looking for | Read |
|---|---|
| Why a rule exists, and what broke to produce it | `decisions/README.md` |
| Module map, storage model, durability | `just spec 10`, `just spec 6`, `.claude/rules/architecture.md` |
| Rust conventions, comment cap, error handling | `.claude/rules/rust-style.md` |
| Miri discipline | `.claude/rules/miri.md` |
| The `cli`/`serve`/`mcp` feature gates | `.claude/rules/cli-feature.md` |
| Test placement, e2e, benchmarks | `.claude/rules/testing.md` |
| CI, the merge queue, required checks | `.claude/rules/ci.md` |
| Releases, versions, SDKs, the docs site | `.claude/rules/release.md` |
| The full product spec | `just spec toc`, then `just spec <ref>` |

Rules files load automatically when you open a file they match, so you do not need to fetch
them by hand — but they are plain markdown and readable any time.
