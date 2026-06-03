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

# Lint with clippy, deny all warnings
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Run all tests
test:
    cargo test --all-features

# Run tests within a single module/path (e.g. `just test-mod log`)
test-mod MOD:
    cargo test --all-features {{ MOD }}

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

# Pre-commit / pre-PR checks: format clean, no clippy warnings, tests green
ci: fmt-check lint test

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
