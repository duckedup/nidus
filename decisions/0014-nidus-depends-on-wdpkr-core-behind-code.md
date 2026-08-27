# D0014 — nidus depends on `wdpkr-core`, behind the off-by-default `code` feature

**Status:** accepted
**Superseded in part by [D0015](0015-the-default-build-ships-the-whole-binary.md):** `code` is part of the default `serve` set now, so three claims below are stale. The budget table's "default features" row now names the *lean* `--no-default-features` lane, not the default one. The Miri paragraph reasons from "the CI Miri jobs never pass `--features`" meaning Miri never saw `code`; post-flip that same fact means the opposite, so the jobs now pass `--no-default-features` to preserve this record's own stated invariant. And the Evidence line citing D0011 for "the binary … is gated behind non-default features" is stale in the same way D0011 itself now is. Everything else — the cycle argument, the measurement methodology, the 167-crate cost, the exit plan — still holds.
**Rule:** nidus may depend on `wdpkr-core` behind the off-by-default `code` feature. The cargo-cycle argument that forbade this is dead, the build-budget argument is satisfied by measurement plus a dedicated CI job, and the default `cargo add nidus` graph is unchanged.

## Why

`nidus-gmy.7`'s bead description said:

> Note on wdpkr: it already depends on nidus, so a dependency back is a cargo cycle, and it
> would drag tree-sitter plus eight grammars into the default build graph against a 60s budget
> (D0005). Cross-invoke the binary if its ranked/decision layer is wanted; never cross-depend.

That sentence is superseded, and this record is why. D0013's docs-index design deferred to it
and inherits the same correction.

**The cycle argument died on 2026-08-24.** `duckedup/wdpkr-core` 0.2.0 extracted wdpkr's
engine — walk, tree-sitter chunk, summarize, embed, search orchestration — with `VectorStore`
as a trait and zero backends shipped. Nothing in `wdpkr-core`'s dependency graph points back
at nidus, so `nidus` → `wdpkr-core` is no longer a cycle. Cross-invoking the binary was never
the goal; it was the workaround for a graph that no longer exists.

**The build-budget argument is not dead, but it is measured rather than assumed.** Clean,
offline, debug builds, one machine, `rm -rf target` each time, plus the CI anchor for
extrapolation:

| lane | wall | CPU |
|---|---|---|
| default features (the CI-guarded lane) | 13.4s | 50.5s |
| `serve,embed-all,summarize-all,memory,mcp` | 24.8s | 96.3s |
| the same, plus `wdpkr-core` | 26.3s | 122.1s |
| tree-sitter plus the eight grammars alone | 2.6s | 9.3s |
| `wdpkr-core` alone | 13.1s | 68.9s |

CI anchor: the `build-budget` job reported 25s for the default lane on the same commit range,
against its 60s bound (D0005). The default lane is untouched by any of this, because `code` is
off by default; a `build-budget-code` job carries the 60s-class bound to the `code` lane
instead of leaving it unguarded.

**The finding that matters for D0005's wording**: the eight grammars are not the cost. They
are 2.48M lines of generated C across 81MB of sources, and they cost 9.3s CPU, because
table-driven parsers are cheap to compile at `-O0`. Tree-sitter is not the "large C tree"
D0005 forbids by name — that clause names DuckDB's bundled C++, vendored OpenSSL, and
`aws-lc-sys`, none of which describe a generated table walked by a small C runtime. What
actually costs is `wdpkr-core`'s own Rust graph, not its C dependency.

**The accepted cost, stated plainly**: `wdpkr-core` 0.2.0 ships zero cargo features
(crates.io reports `{}`), so depending on it at all pulls in all 167 crates unconditionally —
`ai_providers`, `http` (reqwest/hyper/rustls), `indexer`, `eval`, `tap`, `decision`, plus
`clap` with derive and `serde_yaml` — even though only `chunk/` and the summarize prompts are
called. Two alternatives were on the table and rejected for now: asking upstream to
feature-gate first, or vendoring tree-sitter directly and porting `chunk/`'s and
`summarize/`'s logic into nidus. Both were rejected as premature — the first because it blocks
on someone else's release, the second because it throws away `wdpkr-core`'s tested AST
chunking and its embed-summaries-not-code prompts to save a cargo feature that upstream may
add anyway. The exit is named rather than left implicit: the day `wdpkr-core` grows features,
nidus narrows its dependency with `default-features = false` and picks only `chunk`/
`summarize` equivalents. Until then, the whole 167-crate graph sits behind `code`, which is
off by default, so it costs nothing to `cargo add nidus`.

**The Miri consequence**: `src/chunk/` is ungated today, so Miri covers all of it. The CI Miri
jobs never pass `--features` (`.github/workflows/ci.yml:638,689`), so gating the AST-chunking
half behind `feature = "code"` keeps Miri's coverage of the existing pure strategies intact
and puts tree-sitter's C permanently out of Miri's reach — Miri cannot instrument foreign
code, so there was never a version of this feature Miri could check. That is a deliberate
narrowing of what "Miri covers `src/chunk/`" means, recorded here rather than discovered
later by someone wondering why a `code`-gated strategy has no Miri job.

**Pinning**: `wdpkr-core = "0.2"`, with `Cargo.lock` committed, so 0.x churn on upstream's side
shows up as a reviewable lockfile diff in a nidus PR rather than a silent resolution change on
the next build.

**What is still red, and outside this repo**: `~/Projects/wdpkr` HEAD (6915e17) still declares
`nidus = "0.65"` and vendors the eight grammars itself — it has not migrated onto the
extracted `wdpkr-core` crate. The three-repo invariant (`wdpkr-core` depends on no concrete
backend) is proven on `wdpkr-core`'s side, not yet on wdpkr's. nidus's side of this decision
does not depend on that migration landing: `wdpkr-core` already depends on no backend, which
is the only fact this record needs. It is stated here so the record does not imply the
migration landed when it has not.

## Evidence

- `nidus-3gm` and its comments — the epic gating implementation on this record, and the
  measurement table above, taken verbatim from the 2026-08-25 comment.
- `nidus-gmy.7` — the bead description carrying the superseded sentence.
- D0005 — the dependency bar is build-and-ship speed, not zero-C.
- D0011 — the binary (and its heavier feature stacks) is gated behind non-default features.
- D0012 — a feature ships whole, in one PR; applies to `nidus code` once implemented.
- the `build-budget-code` CI job — the dedicated budget for the `code` lane.
- commit c55d7e5 — the wdpkr → nidus retrieval-loop eval that first showed flat cosine ranking
  losing to wdpkr's file/symbol grouping, which is why grouping stays a presentation-layer
  concern in nidus rather than something `wdpkr-core`'s indexer is asked to own.
