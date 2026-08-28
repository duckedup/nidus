//! Shared embedder mocks for `tests/e2e/mcp/*`. The mocks in `attrs.rs` and `filters.rs`
//! return one vector for any text, so everything collides at 1.0; this adds a per-text
//! variant (ported from `memory.rs`'s `FakeEmbedder`) plus a fixed one for parity.

#![cfg(feature = "embed-ollama")]

use std::io::Write;
use std::net::{TcpListener, TcpStream};

use serde_json::{Value, json};

use crate::harness::{Server, read_request_body};

/// The dimension the mcp e2e suites use (shared with `memory_http.rs`).
pub(crate) const DIM: usize = 3;

/// The per-text hash `src/memory.rs`'s inline `FakeEmbedder` uses: byte contributions spread
/// across buckets, `+0.1` so an all-zero vector (unnormalizable by the store) never occurs.
pub(crate) fn vector_for(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.1f32; dim];
    for (i, b) in text.bytes().enumerate() {
        v[i % dim] += (b as f32) + 1.0;
    }
    v
}

/// The embedding every call returns regardless of input text — `attrs.rs`'s original mock,
/// reproduced here so a test can opt into the collide-everything behaviour on purpose.
pub(super) fn fixed_vector(dim: usize) -> Vec<f32> {
    (0..dim).map(|i| (i + 1) as f32 * 0.1).collect()
}

/// Answer one request with `body` as a `200 application/json` response.
fn write_json_response(mut stream: TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
}

/// The text Ollama's `/api/embed` request carries in its `input` field (`src/embed/ollama.rs`).
fn requested_text(body: &[u8]) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v["input"].as_str().map(str::to_string))
        .unwrap_or_default()
}

/// A persistent mock answering every request with [`fixed_vector`] in Ollama's wire shape
/// (`{"embeddings": [[...]]}`), regardless of the text it was asked to embed.
pub(super) fn mock_embedder_fixed(dim: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock embedder");
    let addr = listener.local_addr().expect("mock embedder addr");
    let vector: Vec<f64> = fixed_vector(dim).into_iter().map(f64::from).collect();
    let body = json!({ "embeddings": [vector] }).to_string();
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            read_request_body(&mut stream);
            write_json_response(stream, &body);
        }
    });
    format!("http://{addr}")
}

/// A persistent mock answering each request with [`vector_for`] the text carried in that
/// request's own body, so identical texts embed identically and distinct texts embed apart —
/// the property [`mock_embedder_fixed`] cannot give a near-duplicate/dedupe test.
pub(crate) fn mock_embedder_per_text(dim: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock embedder");
    let addr = listener.local_addr().expect("mock embedder addr");
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let body = read_request_body(&mut stream);
            let text = requested_text(&body);
            let vector: Vec<f64> = vector_for(&text, dim).into_iter().map(f64::from).collect();
            let response = json!({ "embeddings": [vector] }).to_string();
            write_json_response(stream, &response);
        }
    });
    format!("http://{addr}")
}

/// A per-text mock that sleeps `delay` before answering, but **only** for a text containing
/// `marker` — so a test can make one request outlive a deadline while its setup stays fast.
/// One thread per connection, so a slow answer never delays a later request.
pub(crate) fn mock_embedder_slow_for(
    dim: usize,
    marker: &str,
    delay: std::time::Duration,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock embedder");
    let addr = listener.local_addr().expect("mock embedder addr");
    let marker = marker.to_string();
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let marker = marker.clone();
            std::thread::spawn(move || {
                let body = read_request_body(&mut stream);
                let text = requested_text(&body);
                if text.contains(&marker) {
                    std::thread::sleep(delay);
                }
                let vector: Vec<f64> = vector_for(&text, dim).into_iter().map(f64::from).collect();
                let response = json!({ "embeddings": [vector] }).to_string();
                write_json_response(stream, &response);
            });
        }
    });
    format!("http://{addr}")
}

/// A server with a [`mock_embedder_fixed`]-backed embedder, over a fresh store directory.
pub(super) fn fixed_embedder_server(
    dir: &std::path::Path,
    dim: usize,
) -> crate::harness::RunningServer {
    let embed_url = mock_embedder_fixed(dim);
    Server::new(dir, dim)
        .args(["--embed-provider", "ollama", "--embed-base-url", &embed_url])
        .start()
}

/// A server with a [`mock_embedder_per_text`]-backed embedder, over a fresh store directory.
pub(crate) fn per_text_embedder_server(
    dir: &std::path::Path,
    dim: usize,
) -> crate::harness::RunningServer {
    let embed_url = mock_embedder_per_text(dim);
    Server::new(dir, dim)
        .args(["--embed-provider", "ollama", "--embed-base-url", &embed_url])
        .start()
}

/// A persistent mock answering the Cohere `/v2/rerank` wire shape (`src/rerank/cohere.rs`).
/// `documents` arrive in metric order (best first) and the store re-sorts descending by
/// score, so a score *increasing* with `index` (worst metric match scores highest) inverts.
#[cfg(feature = "rerank-cohere")]
pub(crate) fn mock_reranker_inverting() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock reranker");
    let addr = listener.local_addr().expect("mock reranker addr");
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let body = read_request_body(&mut stream);
            let n = serde_json::from_slice::<Value>(&body)
                .ok()
                .and_then(|v| v["documents"].as_array().map(|d| d.len()))
                .unwrap_or(0);
            let results: Vec<Value> = (0..n)
                .map(|i| json!({ "index": i, "relevance_score": (i + 1) as f64 }))
                .collect();
            let response = json!({ "results": results }).to_string();
            write_json_response(stream, &response);
        }
    });
    format!("http://{addr}")
}

/// Cosine similarity, used only to phrase the acceptance criterion below in the same terms
/// the store scores with.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// The acceptance criterion for this unit: two identical texts must embed to the same
/// vector, and two very different texts must embed far enough apart that a similarity
/// threshold can tell them apart — the property `attrs.rs`'s fixed-vector mock cannot give.
#[test]
fn per_text_vectors_match_on_identical_text_and_diverge_on_different_text() {
    let a1 = vector_for("the ranking bug is in the upsert path", DIM);
    let a2 = vector_for("the ranking bug is in the upsert path", DIM);
    assert_eq!(a1, a2, "identical text must embed identically");

    let b = vector_for("x", DIM);
    assert!(
        cosine(&a1, &b) < 0.9,
        "distinct texts should not cosine-collide at ~1.0: {a1:?} vs {b:?}"
    );
}

/// The fixed-vector mock, driven through a real server exactly as [`fixed_embedder_server`]
/// wires it up for a caller that wants the deterministic collide-everything behaviour on
/// purpose: `remember`/`recall` under it must still work end to end.
#[test]
fn fixed_embedder_server_round_trips_a_remember_and_recall() {
    let dir = tempfile::tempdir().unwrap();
    let server = fixed_embedder_server(dir.path(), DIM);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("remember"),
        &super::call(
            1,
            "remember",
            json!({"collection": "notes", "text": "anything at all", "id": "only"}),
        ),
    );
    assert_eq!(status, 200, "remember failed: {body}");

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("recall"),
        &super::call(
            2,
            "recall",
            json!({"collection": "notes", "query": "anything"}),
        ),
    );
    assert_eq!(status, 200, "recall failed: {body}");
    let hits: Value =
        serde_json::from_str(&super::text(&super::result(&body))).expect("recall hits JSON");
    assert!(
        hits.as_array()
            .is_some_and(|a| a.iter().any(|h| h["id"] == "only")),
        "the remembered entry should come back: {hits}"
    );
}

/// The mock actually used by a running server: two very different texts, remembered into a
/// fresh collection, must not tie for first place when queried by one of them — a check
/// [`mock_embedder_fixed`] would fail trivially since every entry scores 1.0.
#[test]
fn per_text_mock_produces_distinguishable_search_results_through_a_real_server() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    for (id, text) in [
        ("bug", "the ranking bug is in the upsert path"),
        ("groceries", "a completely unrelated grocery list"),
    ] {
        let (status, body) = super::mcp(
            &server,
            "tools/call",
            Some("remember"),
            &super::call(
                1,
                "remember",
                json!({"collection": "notes", "text": text, "id": id}),
            ),
        );
        assert_eq!(status, 200, "remember({id}) failed: {body}");
    }

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("recall"),
        &super::call(
            2,
            "recall",
            json!({"collection": "notes", "query": "the ranking bug is in the upsert path"}),
        ),
    );
    assert_eq!(status, 200, "recall failed: {body}");
    let hits: Value =
        serde_json::from_str(&super::text(&super::result(&body))).expect("recall hits JSON");
    let top = hits
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or_else(|| panic!("expected at least one hit: {hits}"));
    assert_eq!(
        top["id"], "bug",
        "the matching text should rank first, not tie with an unrelated one: {hits}"
    );
}
