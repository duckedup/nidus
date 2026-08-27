# Decision records

One file per decision: the rule, why it exists, and the evidence that produced it.

CLAUDE.md carries the imperative one-liner and a `D####` pointer; the argument and the
incident history live here. That split is deliberate — CLAUDE.md loads into every session
*and* every subagent, so prose that only matters when someone challenges a rule is paid for
thousands of times and read once.

These are markdown at the repo root rather than in a store or a tracker: git-tracked, visible
in a PR diff, greppable, and readable in a fresh clone with no `dolt` and no API key. Not
under `docs/` — that path triggers the site deploy (D0013).

Fetch one with `just spec --file decisions/<file> toc`, or just read it.

| # | Decision |
|---|---|
| D0001 | [The tracker is beads, and its JSONL export stays untracked](0001-tracker-is-beads-and-the-export-stays-untracked.md) |
| D0002 | [A fresh clone runs `just bd-setup`, never `bd bootstrap` or `bd init`](0002-bd-setup-not-bd-bootstrap.md) |
| D0003 | [Every required check must trigger on `merge_group`](0003-merge-queue-needs-the-merge-group-trigger.md) |
| D0004 | [Close the ticket yourself; nothing auto-closes](0004-close-the-ticket-yourself.md) |
| D0005 | [The dependency bar is build-and-ship speed, not zero-C](0005-the-dependency-bar-is-build-and-ship-speed.md) |
| D0006 | [`deny(unsafe_code)`, not `forbid`, for exactly one scoped allow](0006-deny-unsafe-not-forbid.md) |
| D0007 | [Chart and SDK versions are stamped from `Cargo.toml`](0007-version-stamps-are-derived-never-hand-edited.md) |
| D0008 | [A Miri ignore must name its reason](0008-a-miri-ignore-must-name-its-reason.md) |
| D0009 | [Comments cap at three lines, with two exceptions](0009-comments-cap-at-three-lines.md) |
| D0010 | [One e2e test binary, order-of-magnitude timings](0010-one-e2e-test-binary.md) |
| D0011 | [The binary is gated behind the non-default `cli` feature](0011-the-cli-feature-keeps-cargo-add-fast.md) |
| D0012 | [A feature ships whole, in one PR](0012-a-feature-ships-whole.md) |
| D0013 | [Docs retrieval is derived, and never committed](0013-the-docs-index-is-derived-and-never-tracked.md) |
| D0014 | [nidus depends on `wdpkr-core`, behind the off-by-default `code` feature](0014-nidus-depends-on-wdpkr-core-behind-code.md) |
| D0015 | [The default build ships the whole binary](0015-the-default-build-ships-the-whole-binary.md) |
