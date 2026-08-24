# D0009 — Comments cap at three lines, with two exceptions

**Status:** accepted
**Rule:** Every `//` and `///` block caps at 3 lines, counting `///` blank separators. Rationale longer than that belongs in the commit message, the PR, `SPEC.md`, or a `bd` issue.

## Why

A comment earns its place by saying something the code cannot: the non-obvious *why*, a
constraint that will bite, a bug it guards against. It does not earn its place by restating
the code, justifying the design at length, arguing with an imagined reviewer, or recording
the history of how the decision was reached.

When trimming, keep the fact and drop the argument. "rmcp reports `rmcp 3.1.1` here, not this
crate" beats a paragraph explaining why that matters. Long comments are not thoroughness:
they push the code off the screen, and they go stale where prose in a commit cannot.

**Exception 1 — doc examples.** A ``` fence is test code, not commentary, so its lines do not
count. The prose around it still does.

**Exception 2 — `//!` module and crate docs, exempt entirely.** That block is the crate's
published rustdoc landing page, what a reader meets on docs.rs, not commentary sitting
between a reader and the code. The reason for the cap does not apply. It earns no licence to
ramble; it is reviewed as documentation rather than counted.

The `//!` exemption is per line, not per block. A `///` doc that abuts a `//!` one with no
blank line between, which is ordinary Rust, is still counted — otherwise a stray `//!` would
be a one-line way to dodge the cap.

## Evidence

- `lib/laws.mjs` — `commentCap` implements both exceptions and the per-line rule.
