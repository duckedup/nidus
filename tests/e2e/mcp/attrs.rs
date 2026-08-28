//! MCP `remember`'s attrs (nidus-k28.2): a metadata map must round-trip through `recall`
//! unchanged and reach the HTTP `/search` filter path. Needs an embedder, so this module
//! compiles away outside `embed-ollama` (`just ci-serve`), like `stdio::round_trip`.

#![cfg(feature = "embed-ollama")]

use std::net::TcpListener;

use serde_json::{Value, json};

use crate::harness::{Server, respond_once};

use super::{call, mcp, result, text};

const DIM: usize = 3;

/// The embedding every mock call returns, regardless of the text it was asked to embed.
fn fixed_vector() -> Vec<f32> {
    (0..DIM).map(|i| (i + 1) as f32 * 0.1).collect()
}

/// A persistent mock answering every request with [`fixed_vector`] in Ollama's wire shape
/// (`{"embeddings": [[...]]}`) — needs no API key.
fn mock_embedder() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock embedder");
    let addr = listener.local_addr().expect("mock embedder addr");
    let vector: Vec<f64> = fixed_vector().into_iter().map(f64::from).collect();
    let body = json!({ "embeddings": [vector] }).to_string();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            respond_once(stream, &body);
        }
    });
    format!("http://{addr}")
}

/// A persistent mock answering every request in the OpenAI chat-completions shape the
/// summarizer expects, always returning `summary` as the assistant's message content.
fn mock_summarizer(summary: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock summarizer");
    let addr = listener.local_addr().expect("mock summarizer addr");
    let body =
        json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": summary}}]})
            .to_string();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            respond_once(stream, &body);
        }
    });
    format!("http://{addr}")
}

/// A server with a working (mock) embedder, over a fresh store directory.
fn embedder_backed_server(dir: &std::path::Path) -> crate::harness::RunningServer {
    let embed_url = mock_embedder();
    Server::new(dir, DIM)
        .args(["--embed-provider", "ollama", "--embed-base-url", &embed_url])
        .start()
}

/// The `recall` hit for `id`, from a `tools/call` response body — panics naming every id
/// actually present if it is missing, since that is the first thing worth knowing.
fn find_hit(recall_body: &Value, id: &str) -> Value {
    let hits: Value = serde_json::from_str(&text(&result(recall_body))).expect("recall hits JSON");
    hits.as_array()
        .expect("hits array")
        .iter()
        .find(|h| h["id"] == id)
        .unwrap_or_else(|| panic!("no hit with id `{id}` in {hits}"))
        .clone()
}

/// The acceptance criterion: `remember` with `Int`/`Bool`/`List` attrs, read back via
/// `recall`, must come back byte-for-byte the same tagged values — not stringified, not
/// coerced, not dropped.
#[test]
fn attrs_round_trip_through_recall_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let server = embedder_backed_server(dir.path());

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("remember"),
        &call(
            1,
            "remember",
            json!({
                "collection": "notes",
                "text": "the ranking bug is in the upsert path",
                "id": "ranking-bug",
                "attrs": {
                    "project": {"Str": "nidus"},
                    "count": {"Int": 7},
                    "urgent": {"Bool": true},
                    "tags": {"List": ["mcp", "memory"]}
                }
            }),
        ),
    );
    assert_eq!(status, 200, "remember failed: {body}");

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("recall"),
        &call(
            2,
            "recall",
            json!({"collection": "notes", "query": "ranking bug"}),
        ),
    );
    assert_eq!(status, 200, "recall failed: {body}");
    let hit = find_hit(&body, "ranking-bug");

    assert_eq!(hit["attrs"]["project"], json!({"Str": "nidus"}));
    assert_eq!(
        hit["attrs"]["count"],
        json!({"Int": 7}),
        "an Int attr must round-trip as Int, not a string or a float: {hit}"
    );
    assert_eq!(
        hit["attrs"]["urgent"],
        json!({"Bool": true}),
        "a Bool attr must round-trip unchanged: {hit}"
    );
    assert_eq!(
        hit["attrs"]["tags"],
        json!({"List": ["mcp", "memory"]}),
        "a List attr must round-trip unchanged: {hit}"
    );
}

/// The bug this closes: attrs written through MCP must be reachable by the HTTP `/search`
/// filter path, not just echoed back to the same tool that wrote them.
#[test]
fn attrs_written_over_mcp_are_visible_to_the_http_search_filter() {
    let dir = tempfile::tempdir().unwrap();
    let server = embedder_backed_server(dir.path());

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("remember"),
        &call(
            1,
            "remember",
            json!({
                "collection": "notes",
                "text": "filterable via http",
                "id": "http-filter-me",
                "attrs": {"project": {"Str": "nidus"}, "kind": {"Str": "decision"}}
            }),
        ),
    );
    assert_eq!(status, 200, "remember failed: {body}");

    let query = fixed_vector();
    let (status, body) = server.post(
        "/search",
        &json!({
            "query": query,
            "top_k": 5,
            "filter": [{"Eq": ["project", {"Str": "nidus"}]}]
        }),
    );
    assert_eq!(status, 200, "HTTP /search failed: {body}");
    assert!(
        body.as_array()
            .expect("hits array")
            .iter()
            .any(|h| h["id"] == "http-filter-me"),
        "HTTP /search should see attrs an MCP remember wrote: {body}"
    );

    // A non-matching filter proves this is real filtering, not an unfiltered dump.
    let (status, body) = server.post(
        "/search",
        &json!({
            "query": query,
            "top_k": 5,
            "filter": [{"Eq": ["project", {"Str": "nope"}]}]
        }),
    );
    assert_eq!(status, 200, "HTTP /search failed: {body}");
    assert!(
        !body
            .as_array()
            .expect("hits array")
            .iter()
            .any(|h| h["id"] == "http-filter-me"),
        "a non-matching filter should exclude the entry: {body}"
    );
}

/// In summarize mode, caller attrs must survive alongside the `META_SUMMARY`/`META_SOURCE`
/// keys the summarizer stamps in — matching the HTTP handler's merge order, not replacing it.
#[cfg(feature = "summarize-openai")]
#[test]
fn summarize_mode_preserves_caller_attrs_alongside_meta() {
    let dir = tempfile::tempdir().unwrap();
    let embed_url = mock_embedder();
    let summarize_url = mock_summarizer("A dense summary.");
    let server = Server::new(dir.path(), DIM)
        .args([
            "--embed-provider",
            "ollama",
            "--embed-base-url",
            &embed_url,
            "--summarize-provider",
            "openai",
            "--summarize-base-url",
            &summarize_url,
            "--summarize-api-key",
            "test-key",
        ])
        .start();

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("remember"),
        &call(
            1,
            "remember",
            json!({
                "collection": "notes",
                "text": "a very long document worth summarizing",
                "id": "summarized",
                "summarize": true,
                "attrs": {"project": {"Str": "nidus"}, "kind": {"Str": "doc"}}
            }),
        ),
    );
    assert_eq!(status, 200, "remember (summarize) failed: {body}");

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("recall"),
        &call(
            2,
            "recall",
            json!({"collection": "notes", "query": "summary"}),
        ),
    );
    assert_eq!(status, 200, "recall failed: {body}");
    let hit = find_hit(&body, "summarized");

    assert_eq!(
        hit["attrs"]["project"],
        json!({"Str": "nidus"}),
        "caller attrs must survive summarize mode: {hit}"
    );
    assert_eq!(hit["attrs"]["kind"], json!({"Str": "doc"}));
    assert_eq!(
        hit["attrs"]["nidus.summary"],
        json!({"Str": "A dense summary."}),
        "the summarizer's own stamped attr must still be present: {hit}"
    );
    // The raw text rides in `nidus.text` on every write now (nidus-k28.7); `nidus.source`
    // is no longer stamped, since it carried exactly this value.
    assert_eq!(
        hit["attrs"]["nidus.text"],
        json!({"Str": "a very long document worth summarizing"})
    );
    assert_eq!(hit["attrs"]["nidus.source"], json!(null));
}

/// A malformed attrs map is a caller fault (`-32602`), the same classification
/// `missing_required_argument_names_the_argument` (in `mod.rs`) uses for a bad argument —
/// never an internal error, since no retry without changing the call can fix it.
#[test]
fn malformed_attrs_map_is_a_caller_fault() {
    let dir = tempfile::tempdir().unwrap();
    let server = embedder_backed_server(dir.path());

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("remember"),
        &call(
            1,
            "remember",
            json!({
                "collection": "notes",
                "text": "this attrs value is not a valid tagged Value",
                "attrs": {"broken": {"NotAVariant": 1}}
            }),
        ),
    );
    assert_eq!(
        status, 400,
        "a malformed attrs map is a caller fault: {body}"
    );
    assert_eq!(
        body["error"]["code"].as_i64(),
        Some(-32602),
        "malformed attrs should be invalid_params: {body}"
    );
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("attrs"),
        "the error should name attrs: {body}"
    );
}
