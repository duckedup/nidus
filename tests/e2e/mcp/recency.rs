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
