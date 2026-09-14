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

   This is the one thing CI cannot do for you: a counterfactual needs the fix removed, and no
   check ever runs that tree. It is **one targeted test**, not a lane — `cargo test <filter>`,
   not `just ci`. **Commit first.** Reverting a file to mutate it and then restoring it with
   `git checkout -- <file>` throws away every *other* uncommitted change in that file, which
   is a real way to lose work you are about to push.
6. **Check scope before you trust the merge.** A patch is cut with `git add -A`, so it carries
   everything in that worktree, not just the blueprint's directory. The workflow returns
   `out_of_scope` per patch, but it is derived from the agent's own `files_changed` — confirm
   it against the patch itself (`git apply --numstat <patch_file>`) rather than believing it,
   and revert what does not belong.

   **Do not run the verification lanes here. CI runs them.** `nidus-check lanes` still tells
   you which CI jobs cover the files you touched, and blueprints still carry that list, but it
   is a coverage map, not a script: reading it tells you what will be exercised and what will
   not. What the main thread owes instead is a tree that is worth pushing — the patches
   merged, scope confirmed, nothing obviously half-applied — and then **Ship**, which pushes
   and watches the checks.

   The reason is not just wall clock. A local run proves something about *your* machine and
   *your* feature flags, and this pipeline writes to the tree constantly — reverting a file to
   test a mutation, applying the next group's patch, formatting. A lane run overlapping any of
   that is evidence about a tree nobody has, and a hang or a red result costs a debugging
   detour into a machine-local artefact. CI runs a clean checkout, every lane, once.
7. Report failures from the workflow with their blockers and log paths, and ask whether to
   investigate, skip, or abort. A worker that could not finish is a different thing from a
   red lane, and it is the only failure you can see before CI.
8. On success delete the blueprint files, then continue to **Review**.

