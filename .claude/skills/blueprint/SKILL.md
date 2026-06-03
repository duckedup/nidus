---
name: blueprint
description: Lay down a root spec plus per-directory sub-specs, then orchestrate sonnet implementation agents (one per directory, isolated via worktrees when they would collide). Use when the user invokes /blueprint with a description of what they want built or changed.
argument-hint: <description of what to build or change>
model: opus
allowed-tools: [Read, Write, Edit, Bash, Grep, Glob, Agent, AskUserQuestion]
---

You are running **blueprint**: turn a description of work into a set of spec files, then orchestrate parallel implementation agents to execute them.

User's request: $ARGUMENTS

Follow all rules in the project's CLAUDE.md. In particular: track all work with **bd (beads)** — do NOT use TodoWrite, TaskCreate, or markdown TODO lists.

The flow has two halves:
- **Phase A — Plan (you, opus):** write a root `spec.md` and per-directory sub-specs.
- **Phase B — Build (you orchestrate, sonnet executes):** after the user approves, spawn one sonnet agent per sub-spec, each scoped to its own directory so they don't collide. Use git worktrees when isolation is needed.

---

## Step 0: Preflight

**Validate input:** If `$ARGUMENTS` is empty or too vague to act on (e.g., a single word with no context), STOP and ask the user to describe what they want in more detail.

**Repo check:** Run `git rev-parse --is-inside-work-tree 2>/dev/null` and `git status --porcelain`. If the working tree is dirty, note it — worktree-based isolation in Phase B works best from a clean tree. If not in a git repo, continue (specs can still be written, but worktree isolation is unavailable).

## Step 1: Understand the request

1. Parse the description. Identify the core goal, constraints, and implied requirements.
2. Note what existing features, bugs, or systems you'll need to explore.
3. Summarize your understanding in 2-3 sentences and confirm with the user via `AskUserQuestion`.

Do not proceed until the user confirms.

## Step 2: Explore the codebase

Launch up to 3 `Explore` agents in parallel, each with a specific focus derived from the request.

Goals:
1. Identify relevant services, components, and code paths.
2. Read key files to understand current behavior and patterns.
3. Find existing utilities, helpers, types, or infrastructure to reuse. Record exact file paths and line numbers.
4. Identify test patterns in the relevant areas (framework, fixtures, mocks, conventions) with concrete examples.
5. Note code-generation dependencies (protobuf, GraphQL, mocks) that may need regeneration.
6. **Group findings by directory/package.** This grouping is the key output — each group becomes one sub-spec and, in Phase B, one implementation agent. Aim for groups whose file changes do not overlap.

**Do NOT start implementing yet.**

## Step 3: Write the sub-specs

For each directory requiring changes, write a `spec.md` in that directory. Each must be self-contained enough that an implementation agent can execute it without further exploration.

```markdown
# Spec: [short description]

## Context
Why changes are needed here, and how it ties back to the user's request.

## Files to modify
- `path/to/file:symbol` — what to change and why

## Files to create
- `path/to/new_file` — purpose and what it should contain

## Files to remove
- `path/to/old_file` — why

## Code patterns to follow
Exact snippets from the existing codebase (with file path + line range) the agent should mirror. This prevents the agent from re-exploring.

### Test patterns
The test framework, fixtures, mocks, and conventions used here, with concrete examples.

## Boundaries (collision avoidance)
- This agent may ONLY edit files under `this/directory/`.
- Shared/generated files it must NOT touch directly: <list>. If a shared file needs changes, note it here so the orchestrator handles it (or assigns a worktree).

## Acceptance criteria
- [ ] Behavioral requirement 1
- [ ] Behavioral requirement 2
- [ ] Test: exact test case description

## Verification commands
Exact commands to run after implementation (e.g. `go test ./this/package/...`, build commands).
```

## Step 4: Write the root spec

Write `spec.md` at the repository root:
- Overall summary of the change.
- Table of every sub-spec: its path, one-line description, and the directory it owns.
- Complete list of files to create / modify / remove across all sub-specs.
- **Ordering constraints** — what must happen before what (e.g., "schema changes land before application code"). These become bd dependencies.
- **Collision map** — which sub-specs touch shared or generated files. Anything that overlaps cannot run as a plain parallel agent; flag it for serialization or a worktree.
- Whether codegen (`just generate` or equivalent) is needed.
- Global verification: the full build + test commands to run once everything is implemented.

## Step 5: File issues and present the plan

1. Create one bd issue per sub-spec: `bd create --title="..." --description="implement <path>/spec.md" --type=task`. Add `bd dep add` links matching the ordering constraints from the root spec. (Run creates in parallel where possible.)
2. Use `AskUserQuestion` to present the blueprint: list every spec path with a one-line summary, the execution plan (which agents run in parallel, which serialize, which need worktrees), and ask the user to choose:
   - **Plan only** — stop here; the user will review/implement themselves.
   - **Plan + build** — proceed to Phase B and orchestrate the implementation agents.

Do not enter Phase B unless the user explicitly approves building.

---

## Phase B: Orchestrate implementation

You remain the opus orchestrator. You do **not** write the implementation yourself — you spawn sonnet agents, one per sub-spec, and integrate their results.

### Decide isolation per agent

For each sub-spec, pick the lightest isolation that prevents collisions:

1. **Plain parallel agent (default):** the sub-spec owns a disjoint directory and touches no shared/generated files. Multiple of these run concurrently with no isolation — they edit different files in the same working tree.
2. **Worktree-isolated agent:** use `isolation: "worktree"` on the `Agent` call when the agent must mutate shared/generated files, run a full build/test that would conflict with siblings, or otherwise can't be guaranteed file-disjoint. Each gets its own git worktree so parallel writes don't clobber each other. Worktrees cost setup time and disk — use only when plain parallelism is unsafe.
3. **Serialized:** when sub-spec B depends on A's output (per the ordering constraints), run A to completion before starting B.

### Spawn the agents

- Issue all independent agents in a **single message with multiple `Agent` tool calls** so they run concurrently. Respect serialization from the ordering constraints.
- Each agent is `subagent_type: "general-purpose"`, `model: "sonnet"`.
- Each agent's prompt must:
  - Point to its `spec.md` (absolute path) as the source of truth, and tell it to read CLAUDE.md.
  - State its directory boundary explicitly: "Only edit files under `<dir>/`. Do not touch <shared files>."
  - Have it claim its bd issue (`bd update <id> --claim`) at start and close it (`bd close <id>`) when its acceptance criteria and verification commands pass.
  - Tell it to run its spec's verification commands and report pass/fail with output. It must NOT commit or push.
  - Return a structured summary: files changed, verification result, anything it couldn't do.

### Collect

1. Wait for every spawned agent to finish. Gather each one's structured summary (files changed, verification result, anything it couldn't do).
2. For worktree-isolated agents, review each agent's diff and merge its changes back into the main working tree. Resolve any conflicts. Re-run codegen (`just generate` or equivalent) if any spec required it.
3. Confirm every sub-spec's bd issue was actually closed by its agent. Reopen or follow up on any that weren't.

## Step C: Validate all the work (the main agent owns this)

This is the orchestrator's final and non-delegable responsibility. **You do not trust the agents' self-reported pass/fail — you re-verify the whole, integrated result yourself.**

1. **Per-spec acceptance:** for each sub-spec, confirm its acceptance criteria are met and re-run its verification commands against the integrated tree (not in isolation). Individually-passing pieces can still fail once combined.
2. **Global verification:** run the **global build + test commands** from the root spec across the entire integrated tree. This is the authoritative gate.
3. **Cross-cutting check:** verify the pieces actually fit together — shared/generated files are consistent, no duplicate or conflicting definitions, ordering constraints held, and nothing one agent did broke another's directory.
4. **Remediate and re-validate:** if anything fails, fix it directly or spawn a focused follow-up agent, then re-run global verification. Loop until the whole tree passes (or escalate to the user if it can't be made to pass).
5. **Report:** give the user a final summary — what was implemented, the global verification status with actual command output, which bd issues are closed vs. still open, and anything that needs a human decision.

Do NOT commit or push unless the user asks. When they do, follow the CLAUDE.md session-close protocol.

---

## Skill-specific rules

- Spec files are always named `spec.md`. The root spec lives at the repo root; each sub-spec lives in the directory it governs.
- In Phase A you only write spec files — no implementation, no branches, no commits.
- Never enter Phase B without explicit user approval.
- Implementation agents are sonnet; you (the orchestrator) are opus.
- Keep agents file-disjoint by default; escalate to worktrees only when they would collide.
- If a `spec.md` already exists in a target directory, read it first and ask whether to replace or merge.
- Track everything in bd, never in TodoWrite or markdown checklists.
