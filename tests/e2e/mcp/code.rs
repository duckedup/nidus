//! `code_search` and `POST /code-search` (nidus-3gm unit 5). The two positive tests are
//! gated on the `code` feature (off in the default `just test-e2e` lane); the negative one
//! runs exactly there, proving a binary built without `code` answers 404, not 500.

#[cfg(feature = "code")]
use serde_json::Value;
use serde_json::json;

#[cfg(feature = "code")]
use super::{call, mcp, result, text};
#[cfg(feature = "code")]
use crate::harness::RunningServer;
use crate::harness::Server;

/// Create `collection`, declare `nidus.text` as full-text, and seed it with `records` —
/// hand-crafted attrs mirroring what a real `code` ingest would stamp (`src/code/mod.rs`'s
/// `META_*` keys), since that ingest wiring is a different unit.
#[cfg(feature = "code")]
fn seed_code(server: &RunningServer, collection: &str, records: Value) {
    assert_eq!(
        server
            .post(&format!("/collections/{collection}"), &json!({}))
            .0,
        200
    );
    assert_eq!(
        server
            .post(
                &format!("/collections/{collection}/fts-schema"),
                &json!({"fields": ["nidus.text"]})
            )
            .0,
        200
    );
    let (status, body) = server.post(
        &format!("/collections/{collection}/upsert"),
        &json!({ "records": records }),
    );
    assert_eq!(status, 200, "seeding {collection} failed: {body}");
}

/// The load-bearing MCP assertion: a symbol's path, kind, and line span come back, and
/// neither a vector nor the record's source body ever does. `semantic: false` forces BM25
/// so the test needs no embedder.
#[cfg(feature = "code")]
#[test]
fn code_search_finds_a_symbol_by_path_kind_and_line_span_never_a_vector_or_source_body() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed_code(
        &server,
        "code",
        json!([
            {
                "id": "src/commit.rs#0",
                "vector": [1, 0, 0],
                "attrs": {
                    "code.path": {"Str": "src/commit.rs"},
                    "code.symbol": {"Str": "commit_batch"},
                    "code.kind": {"Str": "function"},
                    "code.language": {"Str": "rust"},
                    "code.start_line": {"Int": 10},
                    "code.end_line": {"Int": 42},
                    "nidus.text": {"Str": "commit_batch fsyncs the active segment before appending log records"}
                }
            },
            {
                "id": "src/other.rs#0",
                "vector": [0, 1, 0],
                "attrs": {
                    "code.path": {"Str": "src/other.rs"},
                    "code.symbol": {"Str": "unrelated_thing"},
                    "code.kind": {"Str": "function"},
                    "code.language": {"Str": "rust"},
                    "code.start_line": {"Int": 1},
                    "code.end_line": {"Int": 5},
                    "nidus.text": {"Str": "nothing to do with the query at all"}
                }
            }
        ]),
    );

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("code_search"),
        &call(
            1,
            "code_search",
            json!({"collection": "code", "query": "fsyncs", "semantic": false}),
        ),
    );
    assert_eq!(status, 200, "code_search failed: {body}");
    let rendered = text(&result(&body));

    assert!(
        rendered.contains("\"path\": \"src/commit.rs\""),
        "the matched file's path should be reported: {rendered}"
    );
    assert!(
        rendered.contains("\"symbol\": \"commit_batch\""),
        "the matched symbol's name should be reported: {rendered}"
    );
    assert!(
        rendered.contains("\"kind\": \"function\""),
        "the symbol's kind should be reported: {rendered}"
    );
    assert!(
        rendered.contains("\"start_line\": 10") && rendered.contains("\"end_line\": 42"),
        "the symbol's line span should be reported: {rendered}"
    );
    assert!(
        !rendered.contains("src/other.rs"),
        "the non-matching file must not appear: {rendered}"
    );

    // The counterfactual this exists to catch: a handler that returns raw hits (which carry
    // the full `attrs`, including the source text and, over HTTP, a vector).
    assert!(
        !rendered.contains("\"vector\""),
        "a code-search result must never carry a vector: {rendered}"
    );
    assert!(
        !rendered.contains("fsyncs the active segment"),
        "a code-search result must never carry the source body — the agent reads the file \
         for that: {rendered}"
    );
}

/// `POST /code-search` directly, against a store pinned at dimension 0 (no embedding space
/// at all): it must answer with BM25 results, not the "does not match store dimension"
/// error a vector query would get.
#[cfg(feature = "code")]
#[test]
fn code_search_on_a_dimension_zero_store_answers_bm25_not_a_dimension_error() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 0).start();
    seed_code(
        &server,
        "code",
        json!([{
            "id": "src/widen.rs#0",
            "vector": [],
            "attrs": {
                "code.path": {"Str": "src/widen.rs"},
                "code.symbol": {"Str": "widen_scope"},
                "code.kind": {"Str": "function"},
                "code.start_line": {"Int": 3},
                "code.end_line": {"Int": 9},
                "nidus.text": {"Str": "narrows a query scope back down to size"}
            }
        }]),
    );

    // No `vector` field at all: the handler must default to BM25 from the store's own
    // dimension rather than attempting (and failing) a vector search.
    let (status, body) = server.post(
        "/code-search",
        &json!({"collection": "code", "query": "narrows", "limit": 5}),
    );
    assert_eq!(
        status, 200,
        "a dim-0 store should answer BM25, not error: {body}"
    );
    let rendered = body.to_string();
    assert!(
        rendered.contains("widen_scope"),
        "the matched symbol should be reported: {rendered}"
    );
    assert!(
        !rendered.to_lowercase().contains("dimension"),
        "a dim-0 store must not fall through to the dimension-mismatch error: {rendered}"
    );
}

/// The negative half, run exactly in the lane that has `code` off (the default `just
/// test-e2e` feature set): `/code-search` must be a plain 404, never a 500, on a binary
/// that never compiled the route in.
#[cfg(not(feature = "code"))]
#[test]
fn code_search_route_is_a_404_not_a_500_without_the_code_feature() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = server.post(
        "/code-search",
        &json!({"collection": "code", "query": "anything"}),
    );
    assert_eq!(
        status, 404,
        "a route absent from this build must 404, not 500: {body}"
    );
}
