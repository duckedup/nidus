//! The HTTP memory routes (`/remember`, `/recall`) through the real binary (#107).
//! These handlers duplicate the MCP tool logic in a separate code path
//! (`src/server/mod.rs` vs `src/server/mcp/remember.rs`), so the dedupe, recency,
//! TTL, and provisioning behaviours the `mcp/*` suites pin need pinning here too.

#![cfg(feature = "embed-ollama")]

use serde_json::{Value, json};

use crate::harness::RunningServer;
use crate::mcp::support::{DIM, per_text_embedder_server};

/// `POST /collections/notes/remember`, asserting success and returning the response.
fn remember(server: &RunningServer, args: Value) -> Value {
    let (status, body) = server.post("/collections/notes/remember", &args);
    assert_eq!(status, 200, "remember failed: {body}");
    body
}

/// `POST /collections/notes/recall`, returning the ranked hits.
fn recall(server: &RunningServer, query: &str) -> Vec<Value> {
    let (status, body) = server.post(
        "/collections/notes/recall",
        &json!({"query": query, "top_k": 10}),
    );
    assert_eq!(status, 200, "recall failed: {body}");
    body.as_array().expect("recall returns an array").clone()
}

fn ids(hits: &[Value]) -> Vec<&str> {
    hits.iter().map(|h| h["id"].as_str().unwrap()).collect()
}

/// The epoch-ms behind a `{"DateTime": ms}` attr on a recall hit.
fn stamp(hit: &Value, key: &str) -> i64 {
    hit["attrs"][key]["DateTime"]
        .as_i64()
        .unwrap_or_else(|| panic!("{key} missing or not DateTime: {hit}"))
}

/// remember → recall round-trips with zero setup: the collection and its FTS schema
/// are provisioned on first write, and both recency stamps land as `DateTime`.
#[test]
fn remember_then_recall_round_trips_and_provisions() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"id": "bug", "text": "the ranking bug is in the upsert path"}),
    );
    remember(
        &server,
        json!({"id": "groceries", "text": "a completely unrelated grocery list"}),
    );

    let hits = recall(&server, "the ranking bug is in the upsert path");
    assert_eq!(
        hits[0]["id"], "bug",
        "matching text must rank first: {hits:?}"
    );
    assert!(stamp(&hits[0], "nidus.created_at") > 0);
    assert!(stamp(&hits[0], "nidus.updated_at") > 0);

    // First-write provisioning declared the default FTS schema over `nidus.text`.
    let (status, hits) = server.post(
        "/text-search",
        &json!({"field": "nidus.text", "query": "ranking", "top_k": 5}),
    );
    assert_eq!(
        status, 200,
        "text-search over the provisioned schema: {hits}"
    );
    assert_eq!(hits[0]["id"], "bug", "{hits}");
}

/// Re-remembering an id keeps its birth date — the handler reads the prior value
/// back before the wholesale-attr upsert, same as the MCP tool.
#[test]
fn re_remember_preserves_created_at_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(&server, json!({"id": "same", "text": "first version"}));
    let created = stamp(&recall(&server, "first version")[0], "nidus.created_at");

    remember(&server, json!({"id": "same", "text": "second version"}));
    let hit = &recall(&server, "second version")[0];
    assert_eq!(hit["id"], "same");
    assert_eq!(
        created,
        stamp(hit, "nidus.created_at"),
        "created_at must carry forward across a re-remember: {hit}"
    );
}

/// A near-duplicate write with `dedupe_threshold` updates the matched entry in place:
/// the response says so, redirects to the survivor's id, and merges attrs (supplied
/// keys win, omitted keys survive) instead of replacing them.
#[test]
fn dedupe_updates_in_place_and_merges_attrs_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);
    let text = "the deploy runbook lives in the ops repo";

    remember(
        &server,
        json!({"id": "first", "text": text,
               "attrs": {"keep": {"Str": "x"}, "k": {"Str": "v1"}}}),
    );
    let body = remember(
        &server,
        json!({"id": "second", "text": text, "dedupe_threshold": 0.95,
               "attrs": {"k": {"Str": "v2"}}}),
    );
    assert_eq!(body["deduped"], true, "{body}");
    assert_eq!(
        body["id"], "first",
        "write must redirect to the match: {body}"
    );

    let (status, listed) = server.post("/list", &json!({"limit": 100}));
    assert_eq!(status, 200);
    assert_eq!(
        listed.as_array().unwrap().len(),
        1,
        "one entry, not two: {listed}"
    );
    let attrs = &listed[0]["attrs"];
    assert_eq!(
        attrs["k"]["Str"], "v2",
        "supplied key wins the collision: {attrs}"
    );
    assert_eq!(
        attrs["keep"]["Str"], "x",
        "omitted key survives the merge: {attrs}"
    );
}

/// An expired entry is not a dedupe candidate — matching one would inherit its past
/// `expires_at` and land a write that reports success but is never visible.
#[test]
fn dedupe_does_not_match_an_expired_candidate() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);
    let text = "an ephemeral note about the flaky test";

    remember(
        &server,
        json!({"id": "old", "text": text, "ttl_seconds": 0}),
    );
    let body = remember(
        &server,
        json!({"id": "new", "text": text, "dedupe_threshold": 0.9}),
    );
    assert_eq!(body["deduped"], false, "{body}");
    assert_eq!(body["id"], "new", "{body}");

    let hits = recall(&server, text);
    assert!(
        ids(&hits).contains(&"new"),
        "the fresh entry must be live: {hits:?}"
    );
}

/// **The #106 regression.** `/recall` hides an expired entry and still surfaces one
/// that never got a TTL (D5) — while the raw `/list` route deliberately sees the
/// expired row: TTL is read-time memory semantics, not deletion.
#[test]
fn recall_hides_expired_entries_but_raw_list_still_sees_them() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"id": "gone", "text": "ephemeral scratch note", "ttl_seconds": 0}),
    );
    remember(&server, json!({"id": "kept", "text": "durable note"}));

    let hits = recall(&server, "ephemeral scratch note");
    assert!(
        !ids(&hits).contains(&"gone"),
        "expired entry leaked: {hits:?}"
    );
    let hits = recall(&server, "durable note");
    assert!(
        ids(&hits).contains(&"kept"),
        "no-TTL entry must surface: {hits:?}"
    );

    let (status, listed) = server.post("/list", &json!({"limit": 100}));
    assert_eq!(status, 200);
    let listed: Vec<&str> = listed
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["id"].as_str().unwrap())
        .collect();
    assert!(
        listed.contains(&"gone") && listed.contains(&"kept"),
        "raw list is unguarded by design and sees both: {listed:?}"
    );
}
