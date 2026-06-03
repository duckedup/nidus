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

# ── Project setup ────────────────────────────────────────────────────────────

# Initialize beads issue tracking for this project
bd-init:
    bd init --reinit-local --prefix nidus
    git config beads.role contributor
    chmod 700 .beads
