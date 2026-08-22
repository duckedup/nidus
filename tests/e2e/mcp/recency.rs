//! Recency stamps and TTL (nidus-k28.5). Needs an embedder, so this module compiles away
//! outside `embed-ollama` (`just ci-serve`), like `attrs.rs`.

#![cfg(feature = "embed-ollama")]

use serde_json::json;

use super::support::{DIM, per_text_embedder_server};
use super::{call, mcp, result, text};

/// `remember` a text, returning the raw tool-result text.
fn remember(server: &crate::harness::RunningServer, args: serde_json::Value) -> String {
    let (status, body) = mcp(
        server,
        "tools/call",
        Some("remember"),
        &call(1, "remember", args),
    );
    assert_eq!(status, 200, "remember failed: {body}");
    text(&result(&body))
}

/// The stored attrs of `id`, as `get` renders them.
fn get_text(server: &crate::harness::RunningServer, collection: &str, id: &str) -> String {
    let (status, body) = mcp(
        server,
        "tools/call",
        Some("get"),
        &call(2, "get", json!({"collection": collection, "id": id})),
    );
    assert_eq!(status, 200, "get failed: {body}");
    text(&result(&body))
}

/// `recall` a query, returning the raw tool-result text.
fn recall(server: &crate::harness::RunningServer, args: serde_json::Value) -> String {
    let (status, body) = mcp(
        server,
        "tools/call",
        Some("recall"),
        &call(5, "recall", args),
    );
    assert_eq!(status, 200, "recall failed: {body}");
    text(&result(&body))
}

/// Pull the `Value::Int` stored under `key` out of a rendered `get` result.
fn extract_count(rendered: &str, key: &str) -> i64 {
    let at = rendered
        .find(key)
        .unwrap_or_else(|| panic!("{key} missing from {rendered}"));
    let tail = &rendered[at..];
    let start = tail.find("Int").expect("Int tag");
    tail[start..]
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .expect("access count")
}

/// Both stamps appear without the caller passing anything.
#[test]
fn remember_stamps_created_and_updated_unprompted() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"collection": "notes", "text": "a fact", "id": "f"}),
    );
    let rendered = get_text(&server, "notes", "f");

    assert!(
        rendered.contains("nidus.created_at") && rendered.contains("nidus.updated_at"),
        "both recency stamps must be present without the caller passing them: {rendered}"
    );
    assert!(
        rendered.contains("DateTime"),
        "stamps are Value::DateTime, not Int: {rendered}"
    );
}

/// Re-remembering an id must not reset its birth date. `upsert` replaces a doc's attrs
/// wholesale, so this only holds because the write path reads the prior value back first.
#[test]
fn re_remembering_preserves_created_at() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"collection": "notes", "text": "first", "id": "same"}),
    );
    let first = get_text(&server, "notes", "same");
    let created = extract_stamp(&first, "nidus.created_at");

    remember(
        &server,
        json!({"collection": "notes", "text": "second", "id": "same"}),
    );
    let second = get_text(&server, "notes", "same");

    assert_eq!(
        created,
        extract_stamp(&second, "nidus.created_at"),
        "created_at must carry forward across a re-remember: {second}"
    );
}

/// Pull the epoch-ms integer stored under `key` out of a rendered `get` result.
fn extract_stamp(rendered: &str, key: &str) -> i64 {
    let at = rendered
        .find(key)
        .unwrap_or_else(|| panic!("{key} missing from {rendered}"));
    let tail = &rendered[at..];
    let start = tail.find("DateTime").expect("DateTime tag");
    tail[start..]
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .expect("epoch ms")
}

/// **The D5 case.** An entry that never received a TTL must still surface. A bare
/// `Gt`/`Ge` guard is false on an absent key, so a wrong implementation passes every other
/// test in this file while silently hiding the entire store.
#[test]
fn an_entry_with_no_ttl_still_surfaces() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"collection": "notes", "text": "no ttl here", "id": "plain"}),
    );

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("recall"),
        &call(
            3,
            "recall",
            json!({"collection": "notes", "query": "no ttl here"}),
        ),
    );
    assert_eq!(status, 200, "recall failed: {body}");
    let rendered = text(&result(&body));
    assert!(
        rendered.contains("plain"),
        "an entry with no expires_at must not be filtered out by the TTL guard: {rendered}"
    );

    assert!(
        get_text(&server, "notes", "plain").contains("no ttl here"),
        "get must also return a never-expiring entry"
    );
}

/// An already-expired entry disappears from every read surface. `ttl_seconds: 0` expires it
/// at the instant it is written, which needs no sleeping.
#[test]
fn an_expired_entry_is_hidden_from_every_read_tool() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({
            "collection": "notes",
            "text": "ephemeral scratch note",
            "id": "gone",
            "ttl_seconds": 0
        }),
    );
    // A live neighbour, so an empty result cannot be mistaken for an empty collection.
    remember(
        &server,
        json!({"collection": "notes", "text": "durable note", "id": "kept"}),
    );

    for (tool, args) in [
        (
            "recall",
            json!({"collection": "notes", "query": "ephemeral scratch note"}),
        ),
        (
            "text_search",
            json!({"collection": "notes", "field": "nidus.text", "query": "ephemeral"}),
        ),
        (
            "hybrid_search",
            json!({"collection": "notes", "field": "nidus.text", "query": "ephemeral"}),
        ),
        ("browse", json!({"collection": "notes"})),
    ] {
        let (status, body) = mcp(&server, "tools/call", Some(tool), &call(4, tool, args));
        assert_eq!(status, 200, "{tool} failed: {body}");
        let rendered = text(&result(&body));
        assert!(
            !rendered.contains("gone"),
            "{tool} must not surface an expired entry: {rendered}"
        );
    }

    // `get` bypasses Filter entirely, so it carries its own expiry check.
    let rendered = get_text(&server, "notes", "gone");
    assert!(
        !rendered.contains("ephemeral scratch note"),
        "get must report an expired entry as a miss: {rendered}"
    );
    assert!(
        get_text(&server, "notes", "kept").contains("durable note"),
        "the unexpired neighbour must still be gettable"
    );
}

/// A plain `recall` (no `reinforce`) must leave the store byte-for-byte as `run_read` would.
#[test]
fn a_default_recall_stamps_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"collection": "notes", "text": "a fact", "id": "f"}),
    );
    recall(&server, json!({"collection": "notes", "query": "a fact"}));

    let rendered = get_text(&server, "notes", "f");
    assert!(
        !rendered.contains("nidus.access_count"),
        "a default recall must not stamp access_count: {rendered}"
    );
}

/// `reinforce: true` stamps every call, and the count accumulates across calls.
#[test]
fn a_reinforced_recall_stamps_and_increments() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"collection": "notes", "text": "a fact", "id": "f"}),
    );

    recall(
        &server,
        json!({"collection": "notes", "query": "a fact", "reinforce": true}),
    );
    let first = get_text(&server, "notes", "f");
    assert_eq!(
        extract_count(&first, "nidus.access_count"),
        1,
        "first reinforced recall must stamp access_count 1: {first}"
    );

    recall(
        &server,
        json!({"collection": "notes", "query": "a fact", "reinforce": true}),
    );
    let second = get_text(&server, "notes", "f");
    assert_eq!(
        extract_count(&second, "nidus.access_count"),
        2,
        "second reinforced recall must increment access_count to 2: {second}"
    );
}

/// `extend_ttl_seconds` moves an existing expiry further out, and never mints one on an
/// entry that had none (D-series: a never-expiring memory must not turn mortal).
#[test]
fn extend_ttl_seconds_only_moves_an_existing_expiry() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"collection": "notes", "text": "expiring fact", "id": "e", "ttl_seconds": 3600}),
    );
    remember(
        &server,
        json!({"collection": "notes", "text": "eternal fact", "id": "n"}),
    );

    let before = extract_stamp(&get_text(&server, "notes", "e"), "nidus.expires_at");

    recall(
        &server,
        json!({
            "collection": "notes",
            "query": "fact",
            "top_k": 2,
            "reinforce": true,
            "extend_ttl_seconds": 7200
        }),
    );

    let after = extract_stamp(&get_text(&server, "notes", "e"), "nidus.expires_at");
    assert!(
        after > before,
        "extend_ttl_seconds must push an existing expiry further out: before {before}, after {after}"
    );
    assert!(
        !get_text(&server, "notes", "n").contains("nidus.expires_at"),
        "extend_ttl_seconds must never create an expiry on an entry that had none"
    );
}

/// The schema/impl drift check the blueprint calls for: cheap, and catches a rename that a
/// behavioural test would not see.
#[test]
fn the_recall_tool_schema_advertises_reinforce() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    let (status, body) = mcp(
        &server,
        "tools/list",
        None,
        &super::rpc(1, "tools/list", json!({})),
    );
    assert_eq!(status, 200, "tools/list failed: {body}");

    let listed = result(&body);
    let tools = listed["tools"].as_array().expect("tools array");
    let recall_tool = tools
        .iter()
        .find(|t| t["name"] == "recall")
        .expect("no `recall` tool");
    let reinforce = &recall_tool["inputSchema"]["properties"]["reinforce"];
    assert_eq!(
        reinforce["type"], "boolean",
        "`recall` must advertise `reinforce` as a boolean: {recall_tool}"
    );
}
