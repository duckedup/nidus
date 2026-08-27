# D0015 — The default build ships the whole binary

**Status:** accepted
**Rule:** The default feature set is `serve`, so `cargo install nidus` ships the whole binary. Every `#[cfg(feature = …)]` gate stays in place and `--no-default-features` remains the supported lean library build; a lane that means a narrower slice must pass `--no-default-features` explicitly, because `--features X` is additive.

## Why

`cargo install nidus` installed nothing, silently. `[[bin]]` carries
`required-features = ["cli"]` and the default feature set was empty, so a user following the
crate's own front page got no binary and no error explaining why. That is the motivating
failure this record fixes.

**The measurement, on the record.** Apple M5, 10 cores, clean offline debug build, fresh
target dir each time:

| lane | wall | user CPU |
|---|---|---|
| lean library build (`--no-default-features`) | 12.2s | 46.3s |
| default build (`serve`) | 27.4s | 121.9s |

CI anchor: `build-budget` reported 25s for a lane measured at 12.2s locally, so the default
lane extrapolates to roughly 55s on CI runners against `build-budget-default`'s 120s bound.

**Why the old bar no longer binds.** D0005's rule was written against DuckDB's bundled C++
tree, a multi-hundred-second compile, and LanceDB's Arrow + DataFusion graph. The heaviest
nidus lane is 27.4s wall. The gate was defending a budget it clears by a wide margin, and it
was charging every user a flag to do so.

**What stays true.** This matters more than the change itself, because a future reader needs
to know what they may still rely on:

- `--no-default-features` yields the same lean tree as before, unchanged: the four core direct
  dependencies plus the storage backends. No feature edge moved. This was verified rather than
  assumed: `cargo tree -e no-dev` on `origin/main`'s default build and on this branch's
  `--no-default-features` build differ in exactly one line, nidus's own version stamp.
- Every `#[cfg(feature = …)]` gate is untouched. No file under `src/` changed in this PR.
- `nidus-check laws`' `featureGating` and `modGating` are structural, keyed on which gates and
  imports exist rather than on which features are default. They remain valid and load-bearing.
- Miri covers exactly the code it covered before, now reached via an explicit
  `--no-default-features`. Tree-sitter's C was always outside Miri's reach (D0014 recorded
  this); that is unchanged.

**The costs, stated plainly and without softening:**

- `cargo add nidus` now resolves 250 crates where `--no-default-features` resolves 120
  (`cargo tree -e no-dev`, unique crate names, measured on this change; 254 and 121 if you count
  name-and-version, since four crates resolve at two versions in the default tree). The additions are
  reqwest/hyper/rustls, tokio, axum, clap with derive, serde_yaml, rmcp, and wdpkr-core's
  `ai_providers`/`indexer`/`eval`/`tap`/`decision` modules that nidus never calls. Note the
  older "four core crates" phrasing elsewhere in this repo counts *direct* dependencies, not
  the resolved tree; both numbers here are resolved-tree counts, so they are comparable to each
  other and not to that four. The exit stays open: wdpkr-core 0.2.0 ships
  zero cargo features, so this is all-or-nothing today; the day upstream feature-gates, nidus
  narrows with `default-features = false` and takes only the chunker (D0014 already names this
  exit).
- A wasm32 consumer must now write `default-features = false`, because tokio/axum/reqwest do
  not build for that target. `bindings/wasm/Cargo.toml` does; an external consumer has to
  learn it.
- The published Docker image gains the full memory/embed/summarize/rerank/MCP surface. It
  shipped CLI-only before. This PR pins `--features serve` explicitly so the change is
  deliberate rather than inherited, and so no future default change moves the image again.
- `cargo publish` now compiles the full stack on every release, including tree-sitter's C,
  because it verifies against the default set.
- The additive-features hazard is permanent and load-bearing: any future lane meaning a
  narrower slice must pass `--no-default-features`, or it silently re-tests `serve` while
  staying green. This is the thing most likely to be got wrong later.

## Evidence

- `nidus-rwz` — the epic implementing this flip.
- D0005 — the dependency bar is build-and-ship speed, not zero-C.
- D0011 and D0014 — superseded in part, see the notes at the top of each.
- The `build-budget` / `build-budget-default` CI jobs.
- `tests/build_thesis.rs`'s `default_build_is_pure` — asserts both directions:
  `--no-default-features` pulls no reqwest/tokio/hyper, and the default tree does.
