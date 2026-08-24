# D0004 — Close the ticket yourself; nothing auto-closes

**Status:** accepted
**Rule:** `bd close nidus-<n> --reason "…"` plus `bd dolt push` are part of shipping. Verify the claim against the merged tree, never against the note on the issue.

## Why

A `Closes nidus-186` line in a PR body is documentation and nothing more. GitHub cannot
close a bead, so the trailer that used to do the work now only *looks* like it did.

This law has been broken in both directions, and they are the same failure: the tracker
asserting something the tree does not support.

Under the pre-migration tracker, finished tickets sat silently open. A sweep found ten of
them, three P1 bugs fixed weeks earlier, and two whose work had never landed at all.
GitHub's auto-close then introduced the opposite lie, firing on merge whether or not the
work survived review, so a PR gutted down to a fraction of an issue still closed it.

The manual close makes the first direction the live risk again. So before closing, confirm
the merged diff actually finishes that issue.

## Evidence

- PR #63 — the sweep that found ten silently-open finished tickets.
