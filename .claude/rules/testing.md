---
paths:
  - "tests/**"
  - "benchmarks/**"
  - "src/**/tests.rs"
---

# Test placement

**`--features X` is additive to the default set, never a replacement (D0015).** The default
build now ships everything (`serve`), so a lane meant to isolate one feature slice — `cli`,
`code`, `embed-all`, whatever — must pass `--no-default-features --features X` explicitly. Omit
`--no-default-features` and the lane silently re-tests the full default build instead: it keeps
passing and its name keeps promising isolation it no longer provides.

- **Pure-logic unit tests** live inline per module.
- **File-backed behaviour** goes in `tests/` against temp dirs, with
  `#[cfg_attr(miri, ignore)]` where they fsync (name the reason — `.claude/rules/miri.md`).
- **End-to-end tests that drive the real binary** live in `tests/e2e/` (`just test-e2e`,
  `cli`-gated). Add a new suite as a **module** there, never as a new `tests/*.rs`: each
  `tests/*.rs` file is its own crate, so a second file means a second copy of the harness
  (D0010).

`tests/e2e/harness.rs` spawns `nidus serve` via `env!("CARGO_BIN_EXE_nidus")` on
`--addr 127.0.0.1:0`, learns the port from the startup line, polls `/health`, and kills and
reaps the child on `Drop`. The suites beside it hold only assertions. They cover what the
in-process `tower::oneshot` server tests structurally cannot: the real bind, the CLI-flag →
`ServeConfig` wiring, socket framing, cross-process locking, and restart.

`cluster.rs` runs several real instances over a **real** object store and memory tier, so it is
`#[ignore]`d: bring services up with `just e2e-services-up` (defined once in
`scripts/e2e-services.sh`, shared with CI so the two cannot drift) and run
`just test-e2e-cluster`. `integration.yml` runs the whole target with `--include-ignored` on
every PR, so anything added here is enforced. When a cluster test fails, read the panic
first — the harness attaches the offending process's own stderr.

`scale.rs` is the ranking-correctness lane: 10k 384-d vectors ingested over HTTP, top-k checked
against cosine ground truth computed in-test, so a scoring, normalisation, or JSON-round-trip
bug fails loudly where a three-vector smoke test cannot see it.

**Its timing assertions are order-of-magnitude only and must stay that way** — a debug build on
a shared runner, so a tight bound flakes and proves nothing. Real performance work belongs in
`benchmarks/` (`just bench`, `--release`). Note the bench harness drives the library
**in-process**, so the HTTP path's own cost is still unbenchmarked (nidus-8fn).
