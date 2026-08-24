# D0010 — One e2e test binary, and its timing assertions stay order-of-magnitude

**Status:** accepted
**Rule:** Add an e2e suite as a module under `tests/e2e/`, never as a new `tests/*.rs`. Timing assertions in `scale.rs` stay order-of-magnitude only.

## Why

Each `tests/*.rs` file is its own crate, so a second file would mean a second copy of the
harness. `tests/e2e/main.rs` plus sibling modules is deliberately one binary:
`harness.rs` spawns `nidus serve` via `env!("CARGO_BIN_EXE_nidus")` on `--addr 127.0.0.1:0`,
learns the port from the startup line, polls `/health`, and kills and reaps the child on
`Drop`. The suites beside it hold only assertions.

These cover what the in-process `tower::oneshot` server tests structurally cannot: the real
bind, the CLI-flag → `ServeConfig` wiring, socket framing, cross-process locking, and restart.

`cluster.rs` goes further, running several real instances over a real object store and memory
tier, and is therefore `#[ignore]`d. Bring the services up with `just e2e-services-up`
(defined once in `scripts/e2e-services.sh`, shared with CI so the two cannot drift) and run
`just test-e2e-cluster`. `integration.yml` runs the whole target with `--include-ignored` on
every PR, so anything added here is enforced. When a cluster test fails, read the panic
first: the harness attaches the offending process's own stderr.

`scale.rs` is the ranking-correctness lane: 10k 384-d vectors ingested over HTTP, top-k
checked against cosine ground truth computed in-test, so a scoring, normalisation, or
JSON-round-trip bug fails loudly where a three-vector smoke test cannot see it.

Its timing assertions must stay order-of-magnitude. It is a debug build on a shared runner,
so a tight bound flakes and proves nothing. Real performance work belongs in `benchmarks/`
(`just bench`, `--release`).

## Evidence

- nidus-8fn — the benchmark harness drives the library in-process, so the HTTP path's own
  cost is still unbenchmarked.
