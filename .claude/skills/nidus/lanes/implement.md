## Implement

Needs blueprints. If none exist for this target, run **Spec** first (including its gate).

1. `mktemp -d` for patches and verify logs.
2. Read every sub-blueprint. For each, compute its lanes:
   `nidus-check lanes --paths <that dir's files> --json`.
3. Fan out:
   `Workflow({ scriptPath: ".claude/skills/nidus/implement.workflow.js",
               args: { id, scratchDir, groups } })`
   Each unit is `{ dir, content, path }`. **Pass `path` — the absolute path to that
   sub-blueprint** — as well as `content`. `content` is captured for *every* group the moment
   you launch, so without `path` a blueprint edited mid-run reaches nobody, including groups
   that have not started; the agent reads `path` at its own start instead. The file is
   gitignored, so it is never in the agent's worktree and the path must be absolute (#175).
   **Groups sequence state, not just timing.** A later group is handed every earlier patch and
   applies them before it starts, because that dependency is the only reason it is a later
   group. So put a blueprint in group N+1 exactly when it needs group N's code to exist —
   "implement the thing" then "test the thing" is the usual split.
   Each agent is sonnet, runs in its own worktree, and returns a patch. They do **not** build
   or run lanes — see "Workers do not build" under Fleet; you verify once, on the merged tree.
   Tell the user to watch with `/workflows`.
4. **You merge — this is not delegated.** For each returned patch:
   `git apply --whitespace=nowarn <patch_file>`. On conflict, resolve it yourself or re-run
   that one unit; never abandon a patch silently.
5. **A bug fix owes a deliberate fails-without-fix check.** CLAUDE.md requires the regression
   test to be verified against unpatched code, and nothing in this pipeline proves it for you
   any more: workers do not build, and later groups now start from earlier patches, so a test
   can no longer fail by accident for want of the code it covers. Revert the fix in the merged
   tree, watch the test go red, restore it. Once observed, say so — an unverified regression
   test claimed as verified is worse than none.
6. **Check scope before you trust the merge.** A patch is cut with `git add -A`, so it carries
   everything in that worktree, not just the blueprint's directory. The workflow returns
   `out_of_scope` per patch, but it is derived from the agent's own `files_changed` — confirm
   it against the patch itself (`git apply --numstat <patch_file>`) rather than believing it,
   and revert what does not belong. Then run the `run` lanes from `nidus-check lanes --json`
   against the merged tree: agents passing individually does not mean the merged result passes.
   Lanes the checker reports as `ci` (Miri today) are required PR checks — do not run them
   locally; the PR is where they run once. Debugging a red CI lane is the exception.
   In the full pipeline, start these lanes in the background and launch **Review**'s fan-out
   immediately, so the wall clock is the slower of the two and not the sum. **What makes that
   safe is a rule, not a property of reviewing**: `review.workflow.js` forbids its agents from
   writing any tracked file in this checkout, and tells one that must run *changed* code to
   copy it into its own worktree first. Reviewing is not inherently read-only — a skeptic
   confirming a defect by executing it is the review working as intended, and one that added a
   temporary `#[test]`, ran it and reverted it failed a concurrent `just ci-cli` on a test
   attributable to nothing in the diff, leaving a byte-identical file behind to diagnose from
   (nidus-jni). So if you fan out anything else beside the lanes, give it the same rule. The
   overlap binds you too: do not edit the tree — review fixes included — until the lanes
   report, or the green proves a tree nobody has.
7. Report failures from the workflow with their blockers and log paths, and ask whether to
   investigate, skip, or abort.
8. On success delete the blueprint files, then continue to **Review**.

