---
name: nidus
description: Carry work from a thought to a shipped PR — research and blueprint it, implement it with parallel agents, review it, ship it. Use when the user invokes /nidus with a subcommand (spec, implement, review, ship) or with a GitHub issue number or description.
argument-hint: "[spec|implement|review|ship] <issue number | description | PR number>"
model: opus
allowed-tools: [Read, Write, Edit, Bash, Grep, Glob, Agent, Workflow, AskUserQuestion, ReportFindings, TodoWrite]
---

Arguments: $ARGUMENTS

The judgement lives here; the rules live in code. `bin/nidus-check` decides which
verification lanes a change needs and which repo laws it breaks — never re-derive either
by reading CLAUDE.md and guessing.

```
.claude/skills/nidus/bin/nidus-check lanes [--base <ref>|--pr <n>|--paths a,b] [--json]
.claude/skills/nidus/bin/nidus-check laws  [--base <ref>|--pr <n>] [--json] [--strict]
.claude/skills/nidus/bin/nidus-check selftest
```

## Routing

Read the first word of `$ARGUMENTS`:

| First word | Go to |
|---|---|
| `spec` | **Spec** |
| `implement` | **Implement** |
| `review` | **Review** |
| `ship` | **Ship** |
| anything else | **Full pipeline**: Spec → gate → Implement → Review → offer Ship |

The rest of `$ARGUMENTS` is the target: an issue number (`#42`), a PR number (review only),
a path, or a freeform description. With no arguments at all, run **Review** on the working
tree — that is the cheapest useful thing.

## Preflight (every subcommand except `review`)

1. `git branch --show-current`. Never work on `main`.
2. Resolve the target. If it matches `#?\d+`, `gh issue view <n>`; read the title, body,
   labels, and comments. If `gh issue view` cannot find it, say so and ask whether to
   proceed from the description alone — do not invent the issue's contents.
3. If the target is a description with no issue, file one before writing code
   (`gh issue create --title=… --body=… --label=…`) and `gh issue edit <n> --add-assignee @me`.
4. If not already on a branch for this work, create `austin/<n>-<slug>` (3–5 word kebab
   slug) from an up-to-date `main`. If the tree is dirty, ask before touching it: stash and
   branch, commit here first, or stop.

## Spec

1. Run the research workflow. It fans out four fixed lenses (modules, tests, laws, prior art)
   and returns a proposed directory partition:
   `Workflow({ scriptPath: ".claude/skills/nidus/spec.workflow.js", args: { id, ask } })`
2. **You** write the blueprints from what it returns — do not delegate this. The gate the
   user approves must be yours.
   - `BLUEPRINT-<id>.md` in **each directory** that will change.
   - `BLUEPRINT-<id>.md` at the **repo root**: summary, the table of sub-blueprints, complete
     file create/modify/remove list, group ordering and why, and the global verification lanes
     from `nidus-check lanes`.
   - Never name these `SPEC-*.md` — `SPEC.md` at the root is nidus's product spec.
   - Each sub-blueprint carries: context, files to modify/create/remove, concrete code
     patterns to mirror (path + line range + snippet, so the agent never re-explores), the
     test pattern for that area, acceptance criteria, its exact `verify` lanes, and a scope
     boundary naming the files it may NOT touch.
3. **The gate.** One `AskUserQuestion` carrying a 2–3 sentence understanding summary plus any
   decision that genuinely forks the implementation and would be expensive to walk back.
   Options: approve / refine (they edit, then re-ask) / reject (delete the blueprints, stop).
   Include a decision only if it is real — small reversible details belong in the blueprint's
   open questions instead.

**Do not implement anything until the user picks approve.**

## Implement

Needs blueprints. If none exist for this target, run **Spec** first (including its gate).

1. `mktemp -d` for patches and verify logs.
2. Read every sub-blueprint. For each, compute its lanes:
   `nidus-check lanes --paths <that dir's files> --json`.
3. Fan out:
   `Workflow({ scriptPath: ".claude/skills/nidus/implement.workflow.js",
               args: { id, scratchDir, groups } })`
   Each agent is sonnet, runs in its own worktree, verifies itself (a nidus worktree is a
   complete checkout, so its lanes really do run), and returns a patch. Tell the user to watch
   with `/workflows`.
4. **You merge — this is not delegated.** For each returned patch:
   `git apply --whitespace=nowarn <patch_file>`. On conflict, resolve it yourself or re-run
   that one unit; never abandon a patch silently.
5. Revert anything outside the blueprints' scope, then run the full lane set from
   `nidus-check lanes --json` against the merged tree. Agents passing individually does not
   mean the merged result passes.
6. Report failures from the workflow with their blockers and log paths, and ask whether to
   investigate, skip, or abort.
7. On success delete the blueprint files, then continue to **Review**.

## Review

Target resolution, in order: a number in `$ARGUMENTS` → that PR; a branch name → that branch
against `main`; a path → those files; nothing → the working tree.

1. **Deterministic first.** `nidus-check laws --pr <n>` (or `--base main`, or bare). These are
   already true — they need no verification and no agent.
2. **Eligibility** (PR targets only): skip closed, draft, or bot PRs, and say so rather than
   reviewing them anyway.
3. **Fan out.** `Workflow({ scriptPath: ".claude/skills/nidus/review.workflow.js",
   args: { ref, diffCmd, changed, laws, effort } })` — six lenses (durability, concurrency,
   build thesis, API contract, a diff-only bug scan, and git history / prior review comments),
   each finding then attacked by a skeptic that scores confidence 0–100. Only ≥80 survives.
4. **Report** the law violations plus the confirmed findings with `ReportFindings`, most
   severe first. Say plainly when nothing survived — that is a real result.
5. `--fix`: apply the findings to the working tree, then re-run the affected lanes.
   `--comment`: post them as PR comments, citing code by permalink with the full commit SHA
   (`https://github.com/duckedup/nidus/blob/<sha>/<path>#L10-L15` — a `$(git rev-parse)`
   substitution does not render in a comment).

## Ship

1. `nidus-check laws --strict` and the full `nidus-check lanes` set. Do not ship red.
2. **Version.** Every user-visible or behavioural change bumps `Cargo.toml` `version`
   (patch for fixes/refactors, minor for features, major for breaks) — `release.yml` cuts a
   release only when the `v<version>` tag is new, so an un-bumped PR silently ships nothing.
   If `major.minor` changed, update the `nidus = "M.m"` snippet in `README.md` and
   `docs/src/content/docs/getting-started.md`. Never touch `charts/nidus/Chart.yaml` or the
   SDK version files — CI stamps those and commits them back.
3. **Close the issue in this PR** — put `Closes #<n>` in the PR body so the merge closes
   it, not a later pass. `Closes` fires on merge whether or not the work survived review,
   so re-read every such line against the diff first and downgrade it to `Refs #<n>` if
   this change does not actually finish the issue.
4. **Commit.** Subject `🪺 <area>: <terse description>` — an emoji prefix and the area, no
   issue or PR number (the squash merge appends `(#<pr>)`; a second `(#<n>)` for the issue
   would be unreadable next to it). The issue ref belongs in the PR body. The commit body
   explains why, not what. This repo does keep `Co-Authored-By` and `Claude-Session`
   trailers — match `git log`, do not assume.
5. **Push.** `git pull --rebase` then `git push -u origin <branch>`. Issue state lives on
   GitHub, so nothing tracker-related ships with the commit.
6. **Offer the PR** with `AskUserQuestion` — open it, or stop with the branch pushed. When
   opening: `gh pr create --assignee @me`, title = the commit subject verbatim, body = what
   changed and why plus `Closes #<n>`. Print the URL.

Ask before the commit. Never commit or push without the user choosing to.

## Rules

- Blueprints are `BLUEPRINT-<id>.md`; they are transient, gitignored, and deleted once
  implemented. `SPEC.md` is the product spec and is never touched by this skill.
- Never cross a gate the user has not approved, and never commit to `main`.
- Track work in GitHub Issues. No TodoWrite lists, no markdown checklists, no MEMORY.md —
  durable knowledge goes in the issue that owns it, or in `SPEC.md`.
- Implementation agents are sonnet in worktrees; merging, verifying, and reviewing stay on
  the main thread so one context has seen the whole change.
- `nidus-check` is the source of truth for lanes and laws. If it is wrong, fix the checker and
  its selftest — do not work around it in prose.
