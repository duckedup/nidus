//! The `list_aliases`/`set_alias`/`drop_alias` MCP tools (nidus-klh.3). No embedder needed:
//! every case here upserts vectors directly over HTTP and uses `text_search`, like `hygiene.rs`.

use serde_json::json;

use super::{call, mcp, result, text};
use crate::harness::Server;

/// Setting an alias then querying through it must return hits whose `collection` names the
/// concrete target, not the alias — the resolution the whole feature exists to hide.
#[test]
fn set_alias_then_text_search_returns_the_concrete_collection() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
    let (status, _) = server.post(
        "/collections/docs/upsert",
        &json!({"records": [
            {"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "aliased entry"}}}
        ]}),
    );
    assert_eq!(status, 200);
    assert_eq!(
        server
            .post("/collections/docs/fts-schema", &json!({"fields": ["body"]}))
            .0,
        200
    );

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("set_alias"),
        &call(1, "set_alias", json!({"name": "d", "target": "docs"})),
    );
    assert_eq!(status, 200, "set_alias failed: {body}");
    let said = text(&result(&body));
    assert!(
        said.contains("`d`") && said.contains("`docs`"),
        "the confirmation should name both the alias and the concrete target: {said}"
    );

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("text_search"),
        &call(
            2,
            "text_search",
            json!({"collection": "d", "field": "body", "query": "aliased"}),
        ),
    );
    assert_eq!(status, 200, "text_search through the alias failed: {body}");
    let hits = text(&result(&body));
    assert!(
        hits.contains("\"a\""),
        "text_search through the alias should find the record: {hits}"
    );
    assert!(
        hits.contains("\"collection\": \"docs\""),
        "a hit reached through an alias must carry the concrete collection name, not the \
         alias: {hits}"
    );
}

/// An empty alias map answers in prose, for the same reason `list_collections` does on an
/// empty store: a model handed `{}` tends to retry the identical call.
#[test]
fn list_aliases_on_a_store_with_none_is_a_sentence() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("list_aliases"),
        &call(1, "list_aliases", json!({})),
    );
    assert_eq!(status, 200, "list_aliases failed: {body}");
    let said = text(&result(&body));
    assert!(
        said.contains("no aliases"),
        "an empty alias map should explain itself, not answer `{{}}`: {said}"
    );
}

/// Dropping an alias removes only the indirect name — the collection and its records stay
/// reachable under their concrete name.
#[test]
fn drop_alias_leaves_the_collection_reachable() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    assert_eq!(server.post("/collections/docs2", &json!({})).0, 200);
    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("set_alias"),
        &call(1, "set_alias", json!({"name": "d2", "target": "docs2"})),
    );
    assert_eq!(status, 200, "set_alias failed: {body}");

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("drop_alias"),
        &call(2, "drop_alias", json!({"name": "d2"})),
    );
    assert_eq!(status, 200, "drop_alias failed: {body}");
    let said = text(&result(&body));
    assert!(
        said.contains("`d2`") && said.contains("`docs2`"),
        "the confirmation should name both the alias and its concrete target: {said}"
    );

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("list_collections"),
        &call(3, "list_collections", json!({})),
    );
    assert_eq!(status, 200, "list_collections failed: {body}");
    assert!(
        text(&result(&body)).contains("docs2"),
        "the collection must remain after its alias is dropped: {body}"
    );

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("list_aliases"),
        &call(4, "list_aliases", json!({})),
    );
    assert_eq!(status, 200, "list_aliases failed: {body}");
    assert!(
        !text(&result(&body)).contains("d2"),
        "the dropped alias must not still be listed: {body}"
    );
}

/// Aliases resolve in one hop, never chained: pointing a new alias at an existing alias is
/// rejected with the pinned message.
#[test]
fn set_alias_at_an_alias_target_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    assert_eq!(server.post("/collections/docs3", &json!({})).0, 200);
    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("set_alias"),
        &call(1, "set_alias", json!({"name": "d3", "target": "docs3"})),
    );
    assert_eq!(status, 200, "set_alias failed: {body}");

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("set_alias"),
        &call(2, "set_alias", json!({"name": "d4", "target": "d3"})),
    );
    assert_eq!(
        status, 400,
        "chaining an alias onto another alias is a caller fault: {body}"
    );
    assert_eq!(body["error"]["code"].as_i64(), Some(-32602), "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("is itself an alias") && message.contains("never chained"),
        "the pinned chain message should be surfaced verbatim: {message}"
    );
}
