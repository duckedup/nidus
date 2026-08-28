## Simplify

Staff-level cleanup: find where nidus says the same thing twice, or says a simple thing in a
complicated way, and fix it **without changing what nidus does**. Not a bug hunt (that is
**Review**) and not a perf pass (that is **Optimize**).

The target is a scope, not a ticket: a path (`src/store`), a subsystem in words ("the MCP
surface"), or nothing at all, which means the whole repository.

**The one law of this lane: no functionality is removed.** Not a public item, not a CLI flag,
not a response field, not an error message a test asserts on, not an on-disk byte. If the best
candidate needs one of those, it becomes a filed issue and somebody decides deliberately. The
sharpest tell that this line has been crossed is an **edited test**: if a pre-existing test has
to change for the refactor to pass, the refactor changed behaviour and is wearing a refactor's
clothes. Stop and say so.

1. **Preflight, then the control run.** SKILL.md's preflight has already run. Before you touch
   anything, get the tree green and record it: `nidus-check lanes --paths <scope>` names the
   lanes, run them, keep the output. This is the control. A refactor's whole claim is "the same
   tests pass", which is worth nothing if nobody checked they passed first. Note anything
   already red — you are not fixing it here, and it must not be mistaken for your damage later.
2. **Sweep.** `Workflow({ scriptPath: ".claude/skills/nidus/sweep.workflow.js",
   args: { mode: "simplify", scope, perLens: 2 } })`
   Five opus lenses read the codebase in parallel — literal duplication, the same behaviour
   re-implemented per surface, code at the wrong altitude, vestigial code, and duplicated test
   scaffolding — and every candidate they rank highest is then corroborated by a fresh sonnet
   agent whose only job is to refute it. That second pass is the point: two blocks that look
   identical usually differ in one guard, one default or one error string, and the corroborator
   returns those differences as `objections`. Anything under 70, or that the corroborator says
   changes behaviour, never reaches the shortlist. Tell the user to watch with `/workflows`.
   Raise `perLens` to corroborate deeper, pass `only: ["duplication"]` to run one lens. The
   default shape is 16 agents (5 opus + 10 sonnet + 1 partition); `perLens: 1` halves the
   corroboration half of that, and narrowing `scope` is the cheaper lever than either.
3. **The candidate gate — ask before you write, not after.** This is **Spec**'s scope gate with
   a different question, and the same positional rule: **no `BLUEPRINT-*.md` exists on disk
   until it is answered.** Show the ranked shortlist and what it removes, in two or three
   sentences plus a list; then one `AskUserQuestion` (four maximum) built from
   `partition.scope_questions` plus the shortlist/deferred split itself, since how far this PR
   reaches is the decision that changes every blueprint. Name any candidate that would remove a
   `pub` item explicitly — that is a breaking change and it is the user's call, not yours.
4. **File the work, both halves.** One bead for what is being taken now
   (`bd create "…" -t task -p <0-4> -d "…"`, then `--claim`); the sweep's `deferred` list and
   its `design_changes` become their own beads, each carrying the sites and the evidence in the
   body, so a codebase-wide sweep is not thrown away because one PR could only hold a slice of
   it. Then branch per SKILL.md preflight step 4.
5. **You write the blueprints** — `lanes/spec.md` step 3 governs the format exactly, including
   the root blueprint, the per-directory files, and the `docs/src/content/docs/` exception.
   Two things every sub-blueprint in this lane also carries:
   - the corroborator's `objections` for its candidates, verbatim, as **invariants the agent
     must preserve** — those are the differences a careless merge erases;
   - an explicit "behaviour that must not change" section, and the instruction that **no
     existing test may be edited or deleted**. A new test is fine; a changed one is a stop.
6. **The plan gate.** `lanes/spec.md` step 4, unchanged. Include the net line delta you expect
   (a simplify PR that grows the codebase needs a sentence explaining why).
7. **Implement** — `lanes/implement.md`, unchanged, sonnet agents in worktrees. Its step 5
   (fails-without-fix) does not apply, since nothing here fixes a bug. Its step 6 does, twice
   over: refactors are where out-of-scope hunks ride along.
8. **Prove nothing moved.** Re-run step 1's lanes on the merged tree and diff the result
   against the control. Then, specifically:
   - `git diff --stat main...HEAD` — is it net negative, and does the file list match the
     blueprints?
   - `git diff main...HEAD -- tests/ src/**/tests.rs` and any `#[cfg(test)]` hunk — **every
     changed or deleted assertion is a finding**, not a detail. Report them rather than
     explaining them away.
   - the lanes `just ci` cannot see: `just ci-cli`, the SDK lanes, `just test-e2e`. A shared
     helper that broke a caller in `src/cli` is the classic failure of this lane, and the core
     lane is green while it happens.
9. **Review** — `lanes/review.md`, unchanged, with `issues` set to the bead from step 4. The
   `seams` lens is the one that matters most here and it is already in the set; do not narrow
   with `only`. Then **Ship** (`lanes/ship.md`): a refactor with no user-visible change is
   still a `patch` bump, because `release.yml` ships nothing without one.
