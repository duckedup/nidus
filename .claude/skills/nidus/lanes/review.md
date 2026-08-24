## Review

Target resolution, in order: a number in `$ARGUMENTS` → that PR; a branch name → that branch
against `main`; a path → those files; nothing → the working tree.

Then resolve the **tickets** the change claims, because two of the lenses below need them:
the `Closes`/`Refs` lines in the PR body (`gh pr view <n> --json body`) or in the commit
messages on the branch (`git log --format=%B main..HEAD`), and the issue id in the branch
name. Confirm each with `bd show <id>` and drop what does not resolve; a guessed ticket is
worse than none, since `scope` would then review against the wrong requirements.

1. **Deterministic first.** `nidus-check laws --pr <n>` (or `--base origin/main`, or bare). These are
   already true — they need no verification and no agent.
2. **Eligibility** (PR targets only): skip closed, draft, or bot PRs, and say so rather than
   reviewing them anyway.
3. **Fan out.** `Workflow({ scriptPath: ".claude/skills/nidus/review.workflow.js",
   args: { ref, diffCmd, changed, laws, issues, effort } })`. Ten lenses, because a review
   that asks one question finds one kind of bug:

   | Lens | The angle nobody else takes |
   |---|---|
   | `durability` | fsync order, torn tails, rollback, the reader snapshot rule |
   | `concurrency` | the writer lock, the cluster lease, interleavings |
   | `build-thesis` | does the DEFAULT build stay fast and feature-gated |
   | `api-contract` | DTOs, the text-native MCP surface, the three SDKs |
   | `bugs` | the diff alone, no wider context |
   | `scope` | the **ticket**: criteria not delivered, behaviour nobody asked for, hunks that rode along |
   | `seams` | callers **outside** the changed directory — including the ones `just ci` never compiles |
   | `security` | what an untrusted request or untrusted bytes on disk can do |
   | `test-efficacy` | would the new test fail without the fix; is it flaky by construction |
   | `history` | `git log -p`, `git blame`, and objections reviewers already made |

   Pass `issues: ["nidus-<n>"]` whenever the change claims a ticket: `scope` drops out
   without it, and the criteria pass below never runs. Then every finding is attacked by a
   skeptic scoring confidence 0–100 — one asking all the refutation questions at `medium`,
   three with different failure modes (misreading, prior art, reproduction) at `high`, where
   the **median** decides. Only ≥80 survives.
4. **The criteria pass runs it, rather than reading it.** With `issues` set, a fresh-context
   agent takes each acceptance criterion, demonstrates it with a command it actually runs
   (a test, the real binary, `just test-e2e`), and pastes the output — the denial criteria
   especially, the ones a happy-path test never touches. It comes back as `unmet`, and a
   criterion nobody could demonstrate sends the work back; it does not become a note in the
   PR. Whatever wrote the code is the worst judge of whether it meets the ticket.
5. **Report** the law violations plus the confirmed findings with `ReportFindings`, most
   severe first, and list any `unmet` criteria beside them. Say plainly when nothing
   survived — that is a real result.
6. **Act on it.** Apply the straightforward fixes. Surface anything material rather than
   quietly deciding it: a finding that changes the approach, contradicts the issue, or is a
   judgment call. A finding that reveals the **ticket** was wrong is worth saying out loud —
   this is the cheapest moment to learn the plan was off, and which to fix is the user's call.
7. **Re-verify only what changed.** Skip this entirely if steps 5–6 changed no files: the
   suite was already green and re-running it proves nothing. Otherwise run the lanes
   `nidus-check lanes` names for the files you touched, not the full set.
8. `--fix`: apply the findings to the working tree, then re-run the affected lanes.
   `--comment`: post them as PR comments, citing code by permalink with the full commit SHA
   (`https://github.com/duckedup/nidus/blob/<sha>/<path>#L10-L15` — a `$(git rev-parse)`
   substitution does not render in a comment).

