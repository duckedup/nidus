# D0003 — Every required check must trigger on `merge_group`

**Status:** accepted
**Rule:** Adding a required check and adding its `merge_group:` trigger are one change, never two. Narrow the *work* with a per-step `if`, never a job-level one.

## Why

The merge queue is what makes parallel PRs safe: each PR is retested against `main` *plus
the entries queued ahead of it*, which is the only thing that catches two individually-green
PRs that are jointly broken.

The mechanism is fragile in one specific way. A queued PR builds on a temporary
`gh-readonly-queue/**` ref and fires a `merge_group` event, so any workflow owning a required
check MUST list `merge_group:` under `on:`. A required check that does not run there never
reports, and the entry stalls until it is ejected. The queue looks broken when it is really
just waiting.

What the queue re-runs is narrowed on purpose: only `fmt`, `clippy` and `release` do real
work there. Every other required check still triggers on `merge_group` and still reports,
short-circuiting via a per-step `if: env.QUEUE_LITE != 'true'` guard. Never a job-level
`if` — a job skipped outright is exactly the check that never reports.

The test and Miri lanes are excluded because queue entries are serialized and those are the
slowest lanes, so their cost is paid per entry. What the queue still buys is `clippy` +
`release` compiling every feature set of the merged tree, which is where two green PRs
usually collide.

The trade is deliberate and it is wider than it looks: a pair of PRs that is jointly broken
only at *test* time (a unit test, e2e, Miri, an SDK lane) will land, and `main`'s own push
build is what catches it.

## Evidence

- #135 and #137 each passed, then broke `main` together until #146 — a compile break.
- nidus-0bs — the test and Miri lanes split into two required jobs each.
