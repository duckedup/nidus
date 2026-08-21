//! E2E tests for the `related` tool (nidus-9gs): "more like this" by id, needing no
//! embedder. Every case seeds raw records over plain HTTP, the same pattern `hygiene.rs`
//! uses, and drives `related` over MCP.

use serde_json::json;

use crate::harness::{RunningServer, Server};

/// Milliseconds since the epoch, `offset_ms` away from now (negative = in the past).
fn epoch_ms(offset_ms: i64) -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
        + offset_ms
}

/// A `sim` collection with a source, a near neighbour, an identical-vector duplicate, a
/// text-only record, and a record whose TTL already expired.
fn seed(server: &RunningServer) {
    let (status, body) = server.post(
        "/collections/sim/upsert",
        &json!({"records": [
            {"id": "src", "vector": [1, 0, 0], "attrs": {"kind": {"Str": "note"}}},
            {"id": "near", "vector": [0.9, 0.1, 0], "attrs": {"kind": {"Str": "note"}}},
            {"id": "dup", "vector": [1, 0, 0], "attrs": {"kind": {"Str": "note"}}},
            {"id": "txt", "attrs": {"kind": {"Str": "text-only"}}},
            {
                "id": "old",
                "vector": [0, 0, 1],
                "attrs": {"nidus.expires_at": {"DateTime": epoch_ms(-60_000)}}
            }
        ]}),
    );
    assert_eq!(status, 200, "seed upsert failed: {body}");
}

fn call_related(server: &RunningServer, args: serde_json::Value) -> (u16, serde_json::Value) {
    super::mcp(
        server,
        "tools/call",
        Some("related"),
        &super::call(1, "related", args),
    )
}

#[test]
fn excludes_the_source_and_returns_a_near_neighbour() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = call_related(&server, json!({"collection": "sim", "id": "src"}));
    assert_eq!(status, 200, "{body}");
    let rendered = super::text(&super::result(&body));

    assert!(
        rendered.contains("\"near\""),
        "the near neighbour should be returned: {rendered}"
    );
    assert!(
        !rendered.contains("\"src\""),
        "the source id must not appear in its own results: {rendered}"
    );
}

/// Exclusion is by id, not by score: a true duplicate scores 1.0, identical to the
/// self-match `search_similar` drops, but it must still come back.
#[test]
fn an_identical_vector_duplicate_is_still_returned() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = call_related(&server, json!({"collection": "sim", "id": "src"}));
    assert_eq!(status, 200, "{body}");
    let rendered = super::text(&super::result(&body));
    assert!(
        rendered.contains("\"dup\""),
        "an identical-vector duplicate must still be returned: {rendered}"
    );
}

/// The rendered response is the text-native contract's own proof: no float array ever
/// rides along, even though the tool searched with one internally.
#[test]
fn the_rendered_response_never_contains_a_float_array() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = call_related(&server, json!({"collection": "sim", "id": "src"}));
    assert_eq!(status, 200, "{body}");
    let rendered = super::text(&super::result(&body));
    assert!(
        !rendered.contains("\"vector\""),
        "related must never emit a vector: {rendered}"
    );
}

#[test]
fn a_text_only_source_errors_naming_the_reason() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = call_related(&server, json!({"collection": "sim", "id": "txt"}));
    assert_eq!(status, 400, "a text-only source is a caller fault: {body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("text-only"),
        "the error should name why: {message}"
    );
}

#[test]
fn an_unknown_id_errors_naming_the_reason() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = call_related(&server, json!({"collection": "sim", "id": "ghost"}));
    assert_eq!(status, 400, "an unknown id is a caller fault: {body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("ghost"),
        "the error should name the missing id: {message}"
    );
}

/// The D5-shaped case in reverse: an expired *source* must be refused rather than
/// silently used as a query, even though `search_similar`'s own id lookup bypasses
/// `Filter` and would otherwise never see the expiry.
#[test]
fn an_expired_source_is_refused_rather_than_used_as_a_query() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = call_related(&server, json!({"collection": "sim", "id": "old"}));
    assert_eq!(status, 400, "an expired source is a caller fault: {body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("expired"),
        "the error should name why: {message}"
    );
}

/// `diversity` must reach the MCP tool surface and change what an agent gets back, not
/// merely be accepted. Seeded here because `related` needs no embedder: `dup` is identical
/// to `src`, `near` almost so, and `far` is the outlier MMR has to promote.
#[test]
fn diversity_spreads_the_related_results() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    let (status, body) = server.post(
        "/collections/sim/upsert",
        &json!({"records": [
            {"id": "src", "vector": [1, 0, 0], "attrs": {}},
            {"id": "dup", "vector": [1, 0.01, 0], "attrs": {}},
            {"id": "near", "vector": [1, 0.02, 0], "attrs": {}},
            {"id": "far", "vector": [0.6, 0.8, 0], "attrs": {}}
        ]}),
    );
    assert_eq!(status, 200, "seed upsert failed: {body}");

    let ids = |args: serde_json::Value| -> String {
        let (status, body) = call_related(&server, args);
        assert_eq!(status, 200, "{body}");
        super::text(&super::result(&body))
    };
    let plain = ids(json!({"collection": "sim", "id": "src", "top_k": 2}));
    assert!(
        !plain.contains("\"far\""),
        "the outlier should be crowded out: {plain}"
    );

    let spread = ids(json!({"collection": "sim", "id": "src", "top_k": 2, "diversity": 0.3}));
    assert!(
        spread.contains("\"far\""),
        "diversity should surface the outlier: {spread}"
    );
}

/// The tool schemas must advertise `diversity` wherever the handler honours it, and the
/// text-native rule still holds: no tool may take a raw vector.
#[test]
fn diversity_is_advertised_on_the_tools_that_honour_it() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    let (status, body) = super::mcp(
        &server,
        "tools/list",
        None,
        &super::rpc(1, "tools/list", json!({})),
    );
    assert_eq!(status, 200, "tools/list failed: {body}");

    let listed = super::result(&body);
    let tools = listed["tools"].as_array().expect("tools array");
    for want in ["recall", "text_search", "related"] {
        let tool = tools
            .iter()
            .find(|t| t["name"] == want)
            .unwrap_or_else(|| panic!("no `{want}` tool"));
        let props = &tool["inputSchema"]["properties"];
        assert!(
            props["diversity"].is_object(),
            "`{want}` should advertise diversity: {props}"
        );
        assert!(
            props["diversity"]["description"].is_string(),
            "`{want}`'s diversity needs a hand-written description: {props}"
        );
    }
    // hybrid_search fuses two legs through `HybridOpts`, which carries no diversity, so
    // advertising one there would promise a knob the handler cannot honour.
    let hybrid = tools
        .iter()
        .find(|t| t["name"] == "hybrid_search")
        .expect("no `hybrid_search` tool");
    assert!(
        hybrid["inputSchema"]["properties"]["diversity"].is_null(),
        "hybrid_search must not advertise diversity: {hybrid}"
    );
    for tool in tools {
        assert!(
            tool["inputSchema"]["properties"]["vector"].is_null(),
            "no tool may take a raw vector: {}",
            tool["name"]
        );
    }
}
