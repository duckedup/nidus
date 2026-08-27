---
paths:
  - ".github/workflows/**"
  - "scripts/**"
---

# CI and the merge queue

**Adding a required check and adding its `merge_group:` trigger are one change, never two.**
A queued PR builds on a temporary `gh-readonly-queue/**` ref and fires a `merge_group` event,
so a workflow owning a required check that does not list `merge_group:` under `on:` never
reports there, and the entry stalls until it is ejected. The queue looks broken when it is
really just waiting (D0003).

**Narrow the work with a per-step `if: env.QUEUE_LITE != 'true'`, never a job-level `if`.** A
job skipped outright is exactly the check that never reports. Only `fmt`, `clippy` and
`release` do real work in the queue; everything else still triggers and still reports.

Two jobs time a clean, uncached, offline build on every PR. `build-budget` now measures the
**lean library build** (`--no-default-features`) and fails past 60s (measured ~7s — the bound
is order-of-magnitude on purpose so it never flakes and still catches a bundled-C tree).
`build-budget-default` measures the **default build** (`serve`, D0015) against a 120s bound.
Adding a dependency that blows either, or any bundled-C / native-linking crate, is a design
change: file an issue first (D0005).

`release.yml` invokes the SDK and chart publish workflows via `workflow_call` rather than
letting a tag trigger them: a tag pushed with `GITHUB_TOKEN` cannot trigger another workflow,
and calling directly means a downstream artifact can only publish a version the crate actually
released (D0007).

Scripts shared between a `just` recipe and CI are defined **once** in `scripts/` and called
from both, so the two cannot drift (`scripts/e2e-services.sh` is the pattern).
