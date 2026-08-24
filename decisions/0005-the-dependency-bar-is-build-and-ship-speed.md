# D0005 — The dependency bar is build-and-ship speed, not zero-C

**Status:** accepted
**Rule:** Judge a dep by "does it blow up compile time, require a heavy toolchain, or bloat the binary", not "is it pure Rust". A dep that blows the budget, or any bundled-C / native-linking crate, is a design change: file an issue first.

## Why

The real constraint is build-and-ship cost, not language purity (SPEC §1, §13.6). nidus's
core is popular pure-Rust crates (`anyhow`, `serde`/`bincode`, `crc32fast`, …). The S3/GCS
persistence backends add sans-IO clients (`rusty-s3`/`tame-gcs`) over `ureq`, whose default
TLS is rustls + `ring` — a small C+asm compile. `ring` is allowed, and deliberately not
feature-gated, so `file://` → `s3://` stays a runtime switch rather than a recompile.

Forbidden are the multi-minute C trees nidus exists to avoid: bundled C/C++ (DuckDB's
`libduckdb-sys`), vendored OpenSSL, `aws-lc-sys`, or a transitively-huge graph
(Arrow + DataFusion).

The guardrail is empirical and CI-enforced rather than a crate count. The `build-budget` job
times a clean, uncached, offline build of the default features on every PR and fails past
60s. Measured around 7s; the bound is order-of-magnitude on purpose, like the `scale.rs`
timings, so it never flakes and still catches a bundled-C tree.

The default build is not the four-crate core any more, and CI does not claim it is. Alongside
`anyhow`/`serde`/`bincode`/`crc32fast` it carries `regex` (§7.5), the S3/GCS/redis backend
stack (`rusty-s3`, `tame-gcs`, `tame-oauth`, `ureq`, `url`, `http`, `redis`), and
`memmap2` + `bytemuck` for the mmap seam. It is not FFI-free either: `ring` and `memmap2`
are the two conscious opt-ins. What is enforced is the budget.

## Evidence

- `.github/workflows/` — the `build-budget` job.
- `Cargo.lock` is committed so a new `*-sys` dependency shows up as a reviewable diff.
