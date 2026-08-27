//! Build-thesis guard for the AI ingest layer (epic nidus-54l, D0015). The file itself
//! always compiles; the tests inside are individually `cfg`-gated. `default_build_is_pure`
//! exists only on the LEAN lane (`--no-default-features`) and asserts the async edge is
//! absent there, while `chunker_ships_in_default_build` is ungated and
//! `ingest_lane_enables_the_async_edge` runs whenever the ingest features are on.

// Every assertion here is a deliberate compile-time `cfg!` guard, so its operand is a constant on
// purpose — pinning the feature graph at build time. `clippy::assertions_on_constants` would flag
// each one, hence the crate-wide allow in this guard file.
#![allow(clippy::assertions_on_constants)]

/// LEAN library build: none of the AI-ingest features are enabled, so the async edge
/// (reqwest/tokio/hyper) is not compiled. This test only exists on the lean lane.
#[cfg(not(any(
    feature = "embed",
    feature = "summarize",
    feature = "rerank",
    feature = "code"
)))]
#[test]
fn default_build_is_pure() {
    // Base infra features — both gate `dep:reqwest` + `dep:tokio`.
    assert!(!cfg!(feature = "embed"), "embed must be off by default");
    assert!(
        !cfg!(feature = "summarize"),
        "summarize must be off by default"
    );
    assert!(!cfg!(feature = "rerank"), "rerank must be off by default");
    assert!(!cfg!(feature = "code"), "code must be off by default");

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

    // Rerank umbrella + provider adapters, mirroring the embed/summarize checks above.
    assert!(
        !cfg!(feature = "rerank-all"),
        "rerank-all must be off by default"
    );
    assert!(!cfg!(feature = "rerank-voyage"));
    assert!(!cfg!(feature = "rerank-cohere"));
}

// ── Ingest lane: the feature-implication graph that wires the async edge. ──────
// Each `assert!` compiles only when its provider feature is active, so none exist on the pure lane;
// on the ingest lane they pin that a provider cannot be enabled without its reqwest+tokio base.

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

/// The `rerank-all` umbrella must turn on every shipped reranker.
#[cfg(feature = "rerank-all")]
const _: () = {
    assert!(cfg!(feature = "rerank-voyage"));
    assert!(cfg!(feature = "rerank-cohere"));
};

/// On the ingest lane, at least one base edge is present — a sanity anchor so the
/// file has a live `#[test]` under `--features …` too (not just `const _` checks).
#[cfg(any(feature = "embed", feature = "summarize", feature = "rerank"))]
#[test]
fn ingest_lane_enables_the_async_edge() {
    assert!(
        cfg!(feature = "embed") || cfg!(feature = "summarize") || cfg!(feature = "rerank"),
        "ingest lane must enable at least one async-edge base feature"
    );
}

/// `src/chunk` is deliberately ungated (no Cargo feature), so it must compile and split
/// text with zero features enabled. A `#[cfg(feature = "…")]` added above `pub mod chunk`
/// would fail this test to compile on the pure lane, catching the regression this guards.
#[test]
fn chunker_ships_in_default_build() {
    let opts = nidus::chunk::ChunkOpts {
        max_chars: 20,
        overlap_chars: 0,
        ..Default::default()
    };
    let chunks = nidus::chunk::chunk_text(
        "This text is comfortably longer than twenty characters and must split.",
        &opts,
    )
    .unwrap();
    assert!(
        chunks.len() > 1,
        "expected more than one chunk from text longer than max_chars"
    );
    assert_eq!(chunks[0].index, 0);
}
