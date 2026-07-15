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

# Install the `nidus` binary from this checkout — with the memory endpoints, so
# the installed `nidus serve` matches the shipped (binstall/Docker) binary.
install:
    cargo install --path . --features serve

# Run `nidus serve` against a store (e.g. `just serve /tmp/store 384`)
serve DIR DIM *ARGS:
    cargo run --features cli -- serve --dir {{ DIR }} --dim {{ DIM }} {{ ARGS }}

# Pre-PR checks for the cli feature: format clean, no clippy warnings, tests green
ci-cli: fmt-check lint-cli test-cli

# ── `nidus serve` WITH memory endpoints (the `serve` umbrella feature) ───────
# `serve` = `cli` + the full AI-ingest layer (memory + every embedder +
# summarizer). This is the binary that ships via `cargo binstall` / Docker: the
# `/remember` + `/recall` routes are compiled in, so the SDKs can send TEXT and
# the server embeds/summarizes. Its own lane so a `cli`-only change never waits
# on the heavier async-edge compile.

# Lint the serve feature (server + memory routes + every provider adapter)
lint-serve:
    cargo clippy --all-targets --features serve -- -D warnings

# Test the serve feature (server + memory routes, driven offline against mocks)
test-serve:
    cargo test --features serve

# Release build of the shipped `nidus` binary (serve = cli + memory endpoints)
build-serve:
    cargo build --release --features serve

# Run the memory-capable `nidus serve` (e.g. `just serve-memory /tmp/store 1536
# --embed-provider openai --embed-api-key $OPENAI_API_KEY`)
serve-memory DIR DIM *ARGS:
    cargo run --features serve -- serve --dir {{ DIR }} --dim {{ DIM }} {{ ARGS }}

# Pre-PR checks for the serve feature: format clean, no clippy warnings, tests green
ci-serve: fmt-check lint-serve test-serve

# ── AI ingest layer (the opt-in embed/summarize/memory features) ────────────
# The off-by-default async network edge (reqwest → hyper → rustls/ring; NO new C
# tree, NO aws-lc/OpenSSL) that turns nidus into an all-in-one "memory". Like
# `cli`/`serve`, it lives OFF the core build path on purpose: the DEFAULT build
# (`just ci`) pulls NONE of reqwest/tokio/hyper, so `cargo add nidus` stays the
# pure, seconds-fast, sync vector store. The `build-thesis` guard (CI + the
# integration test `tests/build_thesis.rs`) asserts that invariant. Being an
# async-edge feature set, this layer is skipped by the Miri lanes — the same as
# the cli/server stack.

# Lint the embed feature set (base infra + every provider adapter)
lint-embed:
    cargo clippy --all-targets --features embed-all -- -D warnings

# Test the embed feature set (base infra + every provider adapter)
test-embed:
    cargo test --features embed-all

# Pre-PR checks for the embed features: format clean, no clippy warnings, tests green
ci-embed: fmt-check lint-embed test-embed

# Lint the summarize feature set (base infra + every provider adapter)
lint-summarize:
    cargo clippy --all-targets --features summarize-all -- -D warnings

# Test the summarize feature set (base infra + every provider adapter)
test-summarize:
    cargo test --features summarize-all

# Pre-PR checks for the summarize features: format clean, no clippy warnings, tests green
ci-summarize: fmt-check lint-summarize test-summarize

# Pre-PR checks for the FULL ingest layer (memory + every embedder + summarizer): build, lint, test
ci-ingest: fmt-check
    cargo clippy --all-targets --features memory,embed-all,summarize-all -- -D warnings
    cargo test --features memory,embed-all,summarize-all
    cargo build --release --features memory,embed-all,summarize-all

# ── Docs site (Astro + Starlight, in docs/) ─────────────────────────────────

# Run the docs dev server with live reload (installs deps on first run)
# `--bun` forces Bun's runtime so Astro doesn't shell out to system Node
# (Astro needs Node >=22.12; many machines still ship an older system Node).
docs:
    cd docs && bun install && bun --bun run dev

# Build the docs site to docs/dist/
docs-build:
    cd docs && bun install && bun --bun run build

# Preview the production docs build locally
docs-preview:
    cd docs && bun --bun run preview

# ── Client SDKs (in sdks/) ───────────────────────────────────────────────────

# JS/TS SDK: typecheck + unit tests (mocked fetch — no server needed)
sdk-js-test:
    cd sdks/js && npm install && npm run typecheck && npm run test:unit

# JS/TS SDK: full test incl. integration against a real `nidus serve`.
# Builds the binary first and points the suite at it via NIDUS_BIN.
sdk-js-test-all: build-cli
    cd sdks/js && npm install && npm run typecheck && \
      NIDUS_BIN={{justfile_directory()}}/target/release/nidus npm test

# JS/TS SDK: build the dual ESM/CJS bundle + type declarations
sdk-js-build:
    cd sdks/js && npm install && npm run build

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

# ANN (HNSW + IVF) recall & speed sweep (nidus-only, no engine deps). Sweeps
# ef_search (HNSW) and n_probe (IVF), reporting recall@k vs exact ground truth,
# build time, and query speedup vs exact brute force.
#   just bench-ann                              defaults (n=100k, dim=384/768)
#   just bench-ann n=50000 dim=768 ef_search=64,128   pass-through args
bench-ann *ARGS:
    cargo run -p nidus-bench --release --bin nidus-bench-ann -- {{ARGS}}

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
