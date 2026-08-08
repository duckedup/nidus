//! Near-duplicate suppression on `remember` (#86). Every test uses the **per-text** mock
//! from `support`: under a fixed-vector mock every entry collides at 1.0, so these
//! assertions would pass even with the similarity computation entirely broken.

#![cfg(feature = "embed-ollama")]

use serde_json::json;

use super::support::{DIM, per_text_embedder_server};
use super::{call, mcp, result, text};

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

/// Every id currently in `collection`, via `browse`.
fn browse_text(server: &crate::harness::RunningServer, collection: &str) -> String {
    let (status, body) = mcp(
        server,
        "tools/call",
        Some("browse"),
        &call(2, "browse", json!({"collection": collection})),
    );
    assert_eq!(status, 200, "browse failed: {body}");
    text(&result(&body))
}

/// Two near-identical texts collapse to one entry, and the response says so — the ticket's
/// own acceptance criterion, so the model learns it already knew this.
#[test]
fn a_near_duplicate_updates_in_place_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    let first = remember(
        &server,
        json!({
            "collection": "notes",
            "text": "the ranking bug is in the upsert path",
            "dedupe_threshold": 0.99
        }),
    );
    assert!(
        first.contains("Remembered a new entry"),
        "the first write has nothing to match: {first}"
    );

    let second = remember(
        &server,
        json!({
            "collection": "notes",
            "text": "the ranking bug is in the upsert path",
            "dedupe_threshold": 0.99
        }),
    );
    assert!(
        second.contains("Updated an existing near-duplicate"),
        "a re-remember of the same text must report an update, not a fresh store: {second}"
    );

    let browsed = browse_text(&server, "notes");
    assert_eq!(
        browsed.matches("nidus.text").count(),
        1,
        "the near-duplicate must not have become a second competing entry: {browsed}"
    );
}

/// The other half, and the one the fixed-vector mock could never express: genuinely
/// different text must NOT be suppressed.
#[test]
fn different_text_is_not_suppressed() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"collection": "notes", "text": "alpha", "dedupe_threshold": 0.99}),
    );
    let second = remember(
        &server,
        json!({
            "collection": "notes",
            "text": "a completely unrelated observation about estuary birds",
            "dedupe_threshold": 0.99
        }),
    );

    assert!(
        second.contains("Remembered a new entry"),
        "distinct text must be stored, not folded into the earlier entry: {second}"
    );
    assert_eq!(
        browse_text(&server, "notes").matches("nidus.text").count(),
        2,
        "both distinct memories should exist"
    );
}

/// Dedupe is opt-in (D8): with no threshold, identical text stored under derived ids is
/// content-addressed to one id, and nothing reports an update.
#[test]
fn without_a_threshold_nothing_is_suppressed() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"collection": "notes", "text": "same words", "id": "one"}),
    );
    let second = remember(
        &server,
        json!({"collection": "notes", "text": "same words", "id": "two"}),
    );

    assert!(
        second.contains("Remembered a new entry"),
        "with no dedupe_threshold the check must not run at all: {second}"
    );
    assert_eq!(
        browse_text(&server, "notes").matches("nidus.text").count(),
        2,
        "both ids should exist when dedupe is off"
    );
}

/// An expired entry is dead to every read tool, so it must not be a dedupe candidate:
/// merging onto one inherits its already-past `nidus.expires_at`, landing a write that
/// reports success and is invisible from the instant it lands.
#[test]
fn dedupe_does_not_match_an_expired_entry() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({
            "collection": "notes",
            "text": "a thought worth keeping",
            "ttl_seconds": 0,
            "dedupe_threshold": 0.99
        }),
    );
    let second = remember(
        &server,
        json!({
            "collection": "notes",
            "text": "a thought worth keeping",
            "dedupe_threshold": 0.99
        }),
    );

    assert!(
        second.contains("Remembered a new entry"),
        "an expired entry must not be a dedupe candidate: {second}"
    );

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("recall"),
        &call(
            3,
            "recall",
            json!({"collection": "notes", "query": "a thought worth keeping"}),
        ),
    );
    assert_eq!(status, 200, "recall failed: {body}");
    assert!(
        text(&result(&body)).contains("a thought worth keeping"),
        "the freshly written entry must be visible, not born expired via an inherited TTL"
    );
}

/// **D6 — merge, not replace.** An attr set on the first write and omitted from the second
/// must survive the update-in-place. `upsert` replaces attrs wholesale, so this only holds
/// because the write path merges the matched entry's attrs back in.
#[test]
fn dedupe_merge_preserves_attrs_the_second_call_omitted() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({
            "collection": "notes",
            "text": "shared subject line",
            "dedupe_threshold": 0.99,
            "attrs": {"project": {"Str": "nidus"}, "kind": {"Str": "decision"}}
        }),
    );
    remember(
        &server,
        json!({
            "collection": "notes",
            "text": "shared subject line",
            "dedupe_threshold": 0.99,
            "attrs": {"kind": {"Str": "note"}}
        }),
    );

    // Match the rendered key, not the bare value: "nidus" alone also occurs inside every
    // reserved `nidus.*` key, so a looser assertion would hold even if `project` vanished.
    let browsed = browse_text(&server, "notes");
    assert!(
        browsed.contains("\"project\""),
        "an attr the second call omitted must survive the merge: {browsed}"
    );
    assert!(
        browsed.contains("note"),
        "a supplied attr must win the collision: {browsed}"
    );
    assert!(
        !browsed.contains("decision"),
        "the superseded value must be gone: {browsed}"
    );
}
