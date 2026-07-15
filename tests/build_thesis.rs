//! Build-thesis guard for the AI ingest layer (epic nidus-54l).
//!
//! nidus's identity is a **pure-Rust, dependency-lean, seconds-fast** embeddable
//! vector store. The off-by-default AI ingest layer (`embed`/`summarize`/`memory`
//! and their per-provider features) adds an async network edge — `reqwest` +
//! `tokio` + `hyper` — exactly like the `cli`/`serve` feature adds the tokio/axum
//! stack. That edge must never leak into the DEFAULT build: `cargo add nidus` has
//! to stay the pure sync store with none of those crates compiled.
//!
//! This file enforces that thesis at COMPILE TIME via `cfg`. It has two halves,
//! selected by the active feature set, so it stays green on both CI lanes:
//!
//!   * DEFAULT lane (`cargo test`) — none of the ingest features are on, so the
//!     test binary links with NO reqwest/tokio/hyper. The mere fact that this
//!     integration crate compiles and runs under plain `cargo test` (reqwest is
//!     not a dev-dependency, and nothing here can reach it) is itself part of the
//!     proof; the `default_build_is_pure` assertion pins the intent.
//!
//!   * INGEST lane (`cargo test --features memory,embed-all,summarize-all`, etc.)
//!     — the feature-implication graph is asserted: every provider feature pulls
//!     its base (`embed`/`summarize`), and each base is what gates `reqwest` +
//!     `tokio`. This keeps a provider feature from ever being wired up WITHOUT the
//!     async edge it needs (or, conversely, a stray edge with no provider).
//!
//! The AUTHORITATIVE, empirical dep-absence check — `cargo tree -e no-dev` must
//! show none of reqwest/tokio/hyper on the default build, and rustls/ring (never
//! aws-lc/OpenSSL) on the ingest build — lives in CI (`.github/workflows/ci.yml`,
//! the `build-thesis` step) and the `just ci-ingest` recipe. Doing the tree grep
//! there rather than shelling out from a test keeps this crate a fast, offline,
//! pure compile-time invariant. This whole layer is miri-skipped (async edge).

// Every assertion in this file is a DELIBERATE compile-time `cfg!` guard — its
// operand is a constant on purpose (that is the whole point: pin the feature
// graph at build time). `clippy::assertions_on_constants` would flag each one,
// so it is allowed crate-wide for this guard file.
#![allow(clippy::assertions_on_constants)]

/// DEFAULT build: none of the AI-ingest features are enabled, so the async edge
/// (reqwest/tokio/hyper) is not compiled. This test only exists on the pure lane.
#[cfg(not(any(feature = "embed", feature = "summarize")))]
#[test]
fn default_build_is_pure() {
    // Base infra features — both gate `dep:reqwest` + `dep:tokio`.
    assert!(!cfg!(feature = "embed"), "embed must be off by default");
    assert!(
        !cfg!(feature = "summarize"),
        "summarize must be off by default"
    );

    // Headline memory surface + umbrellas.
    assert!(!cfg!(feature = "memory"), "memory must be off by default");
    assert!(
        !cfg!(feature = "embed-all"),
        "embed-all must be off by default"
    );
    assert!(
        !cfg!(feature = "summarize-all"),
        "summarize-all must be off by default"
    );

    // Every shipped provider adapter is likewise off.
    assert!(!cfg!(feature = "embed-voyage"));
    assert!(!cfg!(feature = "embed-openai"));
    assert!(!cfg!(feature = "embed-ollama"));
    assert!(!cfg!(feature = "embed-cohere"));
    assert!(!cfg!(feature = "embed-gemini"));
    assert!(!cfg!(feature = "embed-mistral"));
    assert!(!cfg!(feature = "embed-jina"));
    assert!(!cfg!(feature = "embed-openai-compat"));
    assert!(!cfg!(feature = "summarize-anthropic"));
    assert!(!cfg!(feature = "summarize-openai"));
}

// ── Ingest lane: the feature-implication graph that wires the async edge. ──────
//
// Each `assert!` is compiled ONLY when its provider feature is active, so on the
// pure lane none of these exist; on the ingest lane they pin that a provider can
// never be enabled without the base feature that pulls reqwest + tokio.

/// A provider/umbrella feature must always drag in its base `embed` feature —
/// that base is what enables `dep:reqwest` + `dep:tokio`. If any embedder is on
/// but `embed` is not, the async edge would be missing: a hard compile error.
#[cfg(any(
    feature = "embed-voyage",
    feature = "embed-openai",
    feature = "embed-ollama",
    feature = "embed-cohere",
    feature = "embed-gemini",
    feature = "embed-mistral",
    feature = "embed-jina",
    feature = "embed-openai-compat",
    feature = "embed-all",
    feature = "memory",
))]
const _: () = {
    assert!(
        cfg!(feature = "embed"),
        "an embed provider / memory feature must enable the `embed` base (reqwest + tokio edge)"
    );
};

/// Likewise every summarizer must enable the `summarize` base.
#[cfg(any(
    feature = "summarize-anthropic",
    feature = "summarize-openai",
    feature = "summarize-all",
))]
const _: () = {
    assert!(
        cfg!(feature = "summarize"),
        "a summarize provider feature must enable the `summarize` base (reqwest + tokio edge)"
    );
};

/// The `embed-all` umbrella must turn on every shipped embedder.
#[cfg(feature = "embed-all")]
const _: () = {
    assert!(cfg!(feature = "embed-voyage"));
    assert!(cfg!(feature = "embed-openai"));
    assert!(cfg!(feature = "embed-ollama"));
    assert!(cfg!(feature = "embed-cohere"));
    assert!(cfg!(feature = "embed-gemini"));
    assert!(cfg!(feature = "embed-mistral"));
    assert!(cfg!(feature = "embed-jina"));
    assert!(cfg!(feature = "embed-openai-compat"));
};

/// The `summarize-all` umbrella must turn on every shipped summarizer.
#[cfg(feature = "summarize-all")]
const _: () = {
    assert!(cfg!(feature = "summarize-anthropic"));
    assert!(cfg!(feature = "summarize-openai"));
};

/// On the ingest lane, at least one base edge is present — a sanity anchor so the
/// file has a live `#[test]` under `--features …` too (not just `const _` checks).
#[cfg(any(feature = "embed", feature = "summarize"))]
#[test]
fn ingest_lane_enables_the_async_edge() {
    assert!(
        cfg!(feature = "embed") || cfg!(feature = "summarize"),
        "ingest lane must enable at least one async-edge base feature"
    );
}
