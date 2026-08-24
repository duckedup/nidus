---
name: nidus
description: Carry work from a thought to a shipped PR — assess whether it belongs, research and blueprint it, implement it with parallel agents, review it, ship it, or coordinate a fleet of peer sessions doing all of that. Use when the user invokes /nidus with a subcommand (fit, spec, implement, review, ship, fleet) or with an issue number (beads, e.g. nidus-186 or #186) or description.
argument-hint: "[fit|spec|implement|review|ship|fleet] <issue number | description | PR number | who does what>"
model: opus
allowed-tools: [Read, Write, Edit, Bash, Grep, Glob, Agent, Workflow, AskUserQuestion, ReportFindings, TodoWrite, ListAgents, SendMessage, EnterWorktree, ExitWorktree]
---

Arguments: $ARGUMENTS

The judgement lives here; the rules live in code. `bin/nidus-check` decides which
verification lanes a change needs and which repo laws it breaks — never re-derive either
by reading CLAUDE.md and guessing.

```
.claude/skills/nidus/bin/nidus-check preflight [--issue <id>] [--no-fetch] [--json]
.claude/skills/nidus/bin/nidus-check lanes [--base <ref>|--pr <n>|--paths a,b] [--json]
.claude/skills/nidus/bin/nidus-check laws  [--base <ref>|--pr <n>] [--json] [--strict]
.claude/skills/nidus/bin/nidus-check selftest
```

**`preflight` runs first, always — before you read the ticket, before you read a line of
code.** It fetches `origin` and then answers the one question every judgement below rests
on: is this tree fit to reason from. It is a script rather than a paragraph because the
failure it prevents is not one anybody reasons their way out of.

## Routing

Read the first word of `$ARGUMENTS`, then **read that lane's file and follow it**. Only the
lane you need is loaded. A bold lane name anywhere in these files means the same thing:
open `lanes/<name>.md` and follow it.

| First word | Read |
|---|---|
| `fit` | `lanes/fit.md` |
| `spec` | `lanes/spec.md` |
| `implement` | `lanes/implement.md` |
| `review` | `lanes/review.md` |
| `ship` | `lanes/ship.md` |
| `fleet` | `lanes/fleet.md` |
| anything else | **Full pipeline**: `lanes/spec.md` → scope gate → plan gate → `lanes/implement.md` → `lanes/review.md` → offer `lanes/ship.md` |

Paths are relative to `.claude/skills/nidus/`. The rest of `$ARGUMENTS` is the target: an
issue number (`#42`), a PR number (review only), a path, or a freeform description. With no
arguments at all, run `lanes/review.md` against the working tree — that is the cheapest
useful thing.

## Preflight (every subcommand, `review` and `fit` included)

1. **`nidus-check preflight --issue <id>` — first, before anything else.** It fetches
   `origin`, then blocks on every premise that would otherwise be silently stale: HEAD
   behind `origin/main`, on `main`, the ticket already closed, already carried by a merged
   or open PR, already claimed by a remote branch or another assignee. It also prints the
   next free `Cargo.toml` version, which is the number **Ship** step 2 needs and which no
   branch can work out from inside its own tree.

   **Errors block. Do not proceed and do not reason around them** — every one of them
   means a fact you are about to rely on describes a `main` that has since moved. The
   motivating case: a session evaluating `nidus-lvo.2` read its dependency `nidus-lvo.1` as
   "committed locally, unpushed, no PR" and planned a whole branch strategy on it. `lvo.1`
   had merged as #218 an hour earlier. Nothing about that tree looked wrong, which is the
   point — a stale premise produces confident, coherent, wrong work, and the only tell is a
   fetch nobody remembered to do. `--no-fetch` exists for an offline box and reports itself
   as an error, because a preflight that did not look is not a preflight.

   `review` and `fit` need this as much as the rest: `review` compares against
   `origin/main` (a stale base examines a range nobody meant, nidus-qko), and `fit` step 1
   asks whether an idea is already decided, which is a question about issue state.
2. Resolve the target. If it matches `#?\d+` or `nidus-\d+`, `bd show nidus-<n>`; read the
   title, description, labels, and comments (`bd comments nidus-<n>`). Issues kept their
   GitHub numbers, so `#186` and `nidus-186` are the same ticket. If `bd show` cannot find
   it, say so and ask whether to proceed from the description alone — do not invent the
   issue's contents.
3. If the target is a description with no issue, file one before writing code
   (`bd create "…" -t <type> -p <0-4> -d "…"`) and `bd update nidus-<n> --claim`.
4. If not already on a branch for this work, create `austin/<n>-<slug>` (3–5 word kebab
   slug) from an up-to-date `main`. If the tree is dirty, ask before touching it: stash and
   branch, commit here first, or stop.

## Rules

- Blueprints are `BLUEPRINT-<id>.md`; they are transient, gitignored, and deleted once
  implemented. `SPEC.md` is the product spec and is never touched by this skill.
- Never cross a gate the user has not approved, and never commit to `main`. Writing a
  blueprint before the scope gate is crossing one: the file is the plan.
- Track work in beads (`bd`). No TodoWrite lists, no markdown checklists, no MEMORY.md —
  durable knowledge goes in the issue that owns it, or in `SPEC.md`. Issue state is not in
  the repo: `bd dolt push` publishes it, `bd bootstrap` sets up a fresh clone.
- Implementation agents are sonnet in worktrees; merging, verifying, and reviewing stay on
  the main thread so one context has seen the whole change.
- `nidus-check` is the source of truth for lanes, laws and dispatch safety. If it is wrong, fix
  the checker and its selftest — do not work around it in prose.
- **Assert the behaviour, not that the machinery ran.** The rule below is reactive: it catches a
  no-failing-mode check once you construct the counterfactual. This one is preventive and reads
  off the assertion itself — does it name what would be wrong if the change were absent, or only
  that something happened? "The store opened", "the command ran", "the diff was checked" are all
  true whether or not the code works, which is why they go green and why they do not look wrong.
  Worked example: an e2e test drove `nidus search` across a restart to prove an open-time profile
  merge, and passed with the merge disabled, because `search` has no profile-dependent output. It
  asserted the store opened. Eight instances in one fleet run, so treat it as the default failure
  of a test written in a hurry.
- **Ask whether a check could have failed, not whether it passed.** Three times in two days a
  check ran green and proved nothing: regression tests for a p1 that passed without the fix
  (they corrupted a compressed byte, so they tested the decompressor), a fails-without-fix
  signal that only ever appeared by accident, and a `git log -6` sample that could not have
  disconfirmed the convention it was sampling. A green result from a check that had no failing
  mode is indistinguishable from no check. For a regression test, go and watch it go red.
- **A green lane run is evidence only about a tree nobody was writing to.** Anything sharing
  the checkout while the lanes run — an agent, another session, you — can turn a lane red for
  a reason that is nowhere in the diff and gone before you look. When a lane fails on
  something the change cannot explain, check whether the tree moved under it (`git status`, a
  hash against `HEAD`) before believing either "my change broke it" or "the suite is flaky".
- Peers get worktrees under `.claude/worktrees/`, never a shared tree and never a fresh clone.
  Prune them when the ticket ships.
