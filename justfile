# nidus — task runner.
# Run `just` (or `just --list`) to see available recipes.

# Default: list recipes
default:
    @just --list

# ── Local dev ──────────────────────────────────────────────────────────────

# Quick compile check across all targets
check:
    cargo check --all-targets

# Format all code
fmt:
    cargo fmt --all

# Verify formatting is clean (CI guard)
fmt-check:
    cargo fmt --all -- --check

# Lint with clippy, deny all warnings (pure library — no cli feature)
lint:
    cargo clippy --all-targets -- -D warnings

# Run all tests (pure library — no cli feature)
test:
    cargo test

# Run tests within a single module/path (e.g. `just test-mod log`)
test-mod MOD:
    cargo test {{ MOD }}

# Generate and open the API docs
doc:
    cargo doc --no-deps --open

# Run the end-to-end demo (open → index → search → reopen)
demo:
    cargo run --example demo

# Run Miri to check for undefined behavior (requires nightly).
# nidus is pure safe Rust with zero FFI, so — unlike a C-backed store — the
# WHOLE crate compiles and runs under Miri. Isolation is disabled so the
# file-backed integration tests can touch a temp dir; pure-logic tests
# (cosine, glob, filter, codec) need no special flags.
miri:
    MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-permissive-provenance -Zmiri-ignore-leaks" cargo +nightly miri test

# Show nidus's dependency tree — it must stay minimal and pure Rust. Scoped to
# `-p nidus` so the heavy, opt-in benchmark crate (nidus-bench) never shows up here.
deps:
    cargo tree -p nidus

# Pre-commit / pre-PR checks: format clean, no clippy warnings, tests green.
# Pure library only — the opt-in `cli` feature has its own gate (`just ci-cli`),
# kept separate so the core stays a seconds-long, FFI-free build path.
ci: fmt-check lint test

# ── CLI / server binary (the opt-in `cli` feature) ──────────────────────────
# Off the core build path on purpose: these pull the binary-only deps (clap +
# the tokio/axum stack). All pure Rust, but heavier than the library alone.

# Lint the cli feature (binary + server)
lint-cli:
    cargo clippy --all-targets --features cli -- -D warnings

# Test the cli feature (binary + server)
test-cli:
    cargo test --features cli

# Release build of the `nidus` binary
build-cli:
    cargo build --release --features cli

# Install the `nidus` binary from this checkout
install:
    cargo install --path . --features cli

# Run `nidus serve` against a store (e.g. `just serve /tmp/store 384`)
serve DIR DIM *ARGS:
    cargo run --features cli -- serve --dir {{ DIR }} --dim {{ DIM }} {{ ARGS }}

# Pre-PR checks for the cli feature: format clean, no clippy warnings, tests green
ci-cli: fmt-check lint-cli test-cli

# ── Docs site (Astro + Starlight, in docs/) ─────────────────────────────────

# Run the docs dev server with live reload (installs deps on first run)
docs:
    cd docs && bun install && bun run dev

# Build the docs site to docs/dist/
docs-build:
    cd docs && bun install && bun run build

# Preview the production docs build locally
docs-preview:
    cd docs && bun run preview

# ── Build ──────────────────────────────────────────────────────────────────

# Debug build for the current host
build:
    cargo build

# Release build for the current host
release:
    cargo build --release

# Remove all build artifacts
clean:
    cargo clean

# ── Benchmarks (quarantined: heavy deps, NEVER on nidus's own build path) ──────

# Cross-engine exact-KNN performance-parity benchmark. Engine deps are heavy and gated,
# so only the engines you ask for are compiled (nidus is always in). Extra key=value args
# pass straight through to the harness (`just bench all help` lists them).
#   just bench                   nidus only (quick)
#   just bench duckdb            nidus + DuckDB (bundled C++)
#   just bench lancedb           nidus + LanceDB
#   just bench all               nidus + DuckDB + LanceDB
#   just bench all top_k=100 n=1000000   pass-through args
bench ENGINES="nidus" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{ENGINES}}" in
      nidus) feats="" ;;
      all)   feats="duckdb,lancedb" ;;
      *)     feats="$(echo '{{ENGINES}}' | tr ' ' ',' | sed -E 's/(^|,)nidus(,|$)/\1/g; s/^,+|,+$//g; s/,,+/,/g')" ;;
    esac
    cargo run -p nidus-bench --release ${feats:+--features "$feats"} -- {{ARGS}}

# int8 quantization recall & speed sweep (nidus-only, no engine deps). Sweeps the
# rescore factor and reports recall@k vs exact ground truth + query speedup.
#   just bench-quant                         defaults (n=100k, dim=384/768, rescore=1,2,4,8)
#   just bench-quant n=10000 rescore=2,4,8   pass-through args
bench-quant *ARGS:
    cargo run -p nidus-bench --release --bin nidus-bench-quant -- {{ARGS}}

# nidus-internal regression benchmarks (criterion); compares against saved baselines.
# Targets the criterion bench directly so harness args reach it (the lib's libtest
# harness would otherwise reject them).
#   just bench-crit                          run all
#   just bench-crit --save-baseline main     record a baseline to diff later runs against
bench-crit *ARGS:
    cargo bench -p nidus-bench --bench nidus_regression -- {{ARGS}}

# ── Project setup ────────────────────────────────────────────────────────────

# Initialize beads issue tracking for this project
bd-init:
    bd init --reinit-local --prefix nidus
    git config beads.role contributor
    chmod 700 .beads
