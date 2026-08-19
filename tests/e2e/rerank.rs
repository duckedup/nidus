//! Real-binary proof that `--rerank-provider`/`--rerank-base-url` wire into a working
//! cross-encoder stage (nidus-4ss): the in-process `tower::oneshot` tests cannot see the
//! CLI-flag -> `ServeConfig` wiring this proves.

#![cfg(all(feature = "mcp", feature = "embed-ollama", feature = "rerank-cohere"))]

use serde_json::{Value, json};

use crate::harness::{RunningServer, Server};
use crate::mcp::support::{DIM, mock_embedder_per_text, mock_reranker_inverting};

/// A server with a per-text mock embedder and the mock inverting reranker wired in
/// through the real `--rerank-provider`/`--rerank-base-url` flags.
fn reranking_server(dir: &std::path::Path) -> RunningServer {
    let embed_url = mock_embedder_per_text(DIM);
    let rerank_url = mock_reranker_inverting();
    Server::new(dir, DIM)
        .args([
            "--embed-provider",
            "ollama",
            "--embed-base-url",
            &embed_url,
            "--rerank-provider",
            "cohere",
            "--rerank-api-key",
            "mock-key",
            "--rerank-base-url",
            &rerank_url,
        ])
        .start()
}

/// `POST /collections/docs/remember`, asserting success.
fn remember(server: &RunningServer, id: &str, text: &str) {
    let (status, body) = server.post(
        "/collections/docs/remember",
        &json!({"id": id, "text": text}),
    );
    assert_eq!(status, 200, "remember({id}) failed: {body}");
}

/// `POST /collections/docs/recall`, returning the raw hits array.
fn recall(server: &RunningServer, body: Value) -> Value {
    let (status, resp) = server.post("/collections/docs/recall", &body);
    assert_eq!(status, 200, "recall failed: {resp}");
    resp
}

fn ids(hits: &Value) -> Vec<String> {
    hits.as_array()
        .expect("recall returns an array")
        .iter()
        .map(|h| h["id"].as_str().expect("hit id").to_string())
        .collect()
}

fn scores(hits: &Value) -> Vec<f32> {
    hits.as_array()
        .expect("recall returns an array")
        .iter()
        .map(|h| h["score"].as_f64().expect("hit score") as f32)
        .collect()
}

/// The acceptance criterion no in-process test can prove: a rerank flips the returned
/// order of a REAL server, wired through the real `--rerank-*` flags on the real binary.
#[test]
fn rerank_flips_the_order_of_a_real_server() {
    let dir = tempfile::tempdir().unwrap();
    let server = reranking_server(dir.path());

    for (id, text) in [
        ("bug", "the ranking bug is in the upsert path"),
        ("groceries", "a completely unrelated grocery list"),
        ("weather", "tomorrow's forecast calls for rain"),
    ] {
        remember(&server, id, text);
    }

    let query = "the ranking bug is in the upsert path";

    let baseline_hits = recall(&server, json!({"query": query, "top_k": 3}));
    let baseline = ids(&baseline_hits);
    assert!(
        baseline.len() >= 3,
        "baseline order must have at least 3 distinct ids to make reversal unmistakable: \
         {baseline:?}"
    );

    let reranked_hits = recall(&server, json!({"query": query, "top_k": 3, "rerank": {}}));
    let reranked = ids(&reranked_hits);
    let expected: Vec<String> = baseline.iter().rev().cloned().collect();
    assert_eq!(
        reranked, expected,
        "rerank must reverse the metric order: baseline {baseline:?}, reranked {reranked:?}"
    );

    // The mock's scores run 1..=3 here, past cosine's [-1, 1] range — proof the
    // substitution happened, not just that something reordered.
    let reranked_scores = scores(&reranked_hits);
    assert!(
        reranked_scores.iter().any(|&s| s > 1.0),
        "reranked scores should be on the provider's scale, not cosine's: {reranked_scores:?}"
    );
}

/// A rerank request against a server started without `--rerank-provider` is a `400`
/// naming the flag, never a `500` and never a silent unreranked pass-through.
#[test]
fn rerank_requested_on_a_server_started_without_one_is_400() {
    let dir = tempfile::tempdir().unwrap();
    let embed_url = mock_embedder_per_text(DIM);
    let server = Server::new(dir.path(), DIM)
        .args(["--embed-provider", "ollama", "--embed-base-url", &embed_url])
        .start();

    remember(&server, "only", "anything at all");

    let (status, body) = server.post(
        "/collections/docs/recall",
        &json!({"query": "anything at all", "rerank": {}}),
    );
    assert_eq!(status, 400, "expected a 400, got {status}: {body}");
    let message = body["error"].as_str().unwrap_or_default();
    assert!(
        message.contains("--rerank-provider"),
        "the error must name the flag that fixes it: {message}"
    );
}
