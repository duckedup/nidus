//! E2E tests for the record-level write hygiene tools — `forget`, `get`, `browse`
//! (nidus-k28.4). No embedder needed: every case here upserts vectors directly over HTTP
//! and drives the tools with plain metadata, the same pattern `filters.rs` uses.

use serde_json::json;

use crate::harness::{RunningServer, Server};

/// Seed a `mem` collection with three records, one (`c`) tagged `kind: "wip"` so a
/// filter-based `forget` has something to distinguish.
fn seed(server: &RunningServer) {
    let (status, body) = server.post(
        "/collections/mem/upsert",
        &json!({"records": [
            {"id": "a", "vector": [1, 0, 0], "attrs": {"kind": {"Str": "note"}}},
            {"id": "b", "vector": [0, 1, 0], "attrs": {"kind": {"Str": "note"}}},
            {"id": "c", "vector": [0, 0, 1], "attrs": {"kind": {"Str": "wip"}}}
        ]}),
    );
    assert_eq!(status, 200, "seed upsert failed: {body}");
}

/// The ids present in a collection right now, via the raw HTTP route — used as an
/// independent check that `forget` did (or did not) actually change the store.
fn ids_in(server: &RunningServer, collection: &str) -> Vec<String> {
    let (status, body) = server.post("/list", &json!({"scope": [collection], "limit": 100}));
    assert_eq!(status, 200, "list failed: {body}");
    body.as_array()
        .expect("list result array")
        .iter()
        .map(|h| h["id"].as_str().expect("hit id").to_string())
        .collect()
}

#[test]
fn forget_by_id_removes_only_that_entry() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("forget"),
        &super::call(1, "forget", json!({"collection": "mem", "ids": ["a"]})),
    );
    assert_eq!(status, 200, "{body}");
    let said = super::text(&super::result(&body));
    assert!(said.contains('1'), "should report one removed: {said}");

    let mut remaining = ids_in(&server, "mem");
    remaining.sort();
    assert_eq!(remaining, vec!["b".to_string(), "c".to_string()]);
}

#[test]
fn forget_by_filter_removes_every_match() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("forget"),
        &super::call(
            2,
            "forget",
            json!({"collection": "mem", "filter": [{"Eq": ["kind", {"Str": "note"}]}]}),
        ),
    );
    assert_eq!(status, 200, "{body}");

    let remaining = ids_in(&server, "mem");
    assert_eq!(remaining, vec!["c".to_string()], "{remaining:?}");
}

/// Filter wins when both are given: `ids` names `c`, but the filter matches `a`/`b`
/// instead, and only the filter's matches should be removed.
#[test]
fn filter_wins_over_ids_when_both_given() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("forget"),
        &super::call(
            3,
            "forget",
            json!({
                "collection": "mem",
                "ids": ["c"],
                "filter": [{"Eq": ["kind", {"Str": "note"}]}]
            }),
        ),
    );
    assert_eq!(status, 200, "{body}");

    let remaining = ids_in(&server, "mem");
    assert_eq!(
        remaining,
        vec!["c".to_string()],
        "filter should have won, sparing `c`: {remaining:?}"
    );
}

#[test]
fn forget_of_a_nonexistent_id_is_not_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("forget"),
        &super::call(4, "forget", json!({"collection": "mem", "ids": ["nope"]})),
    );
    assert_eq!(status, 200, "a missing id must not be an error: {body}");
    let said = super::text(&super::result(&body));
    assert!(said.contains('0'), "should report zero removed: {said}");

    let mut remaining = ids_in(&server, "mem");
    remaining.sort();
    assert_eq!(
        remaining,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}

/// The single most important behavior in this unit: neither `ids` nor `filter` must be
/// a caller fault, never a silent whole-collection wipe.
#[test]
fn forget_with_neither_ids_nor_filter_is_a_caller_fault_and_removes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("forget"),
        &super::call(5, "forget", json!({"collection": "mem"})),
    );
    assert_eq!(
        status, 400,
        "omitting both should be a caller fault: {body}"
    );
    assert_eq!(body["error"]["code"].as_i64(), Some(-32602), "{body}");

    let mut remaining = ids_in(&server, "mem");
    remaining.sort();
    assert_eq!(
        remaining,
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        "the collection must be untouched: {remaining:?}"
    );
}

#[test]
fn get_returns_id_and_attrs_but_never_a_vector() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("get"),
        &super::call(1, "get", json!({"collection": "mem", "id": "a"})),
    );
    assert_eq!(status, 200, "{body}");
    let rendered = super::text(&super::result(&body));
    assert!(rendered.contains("\"id\": \"a\""), "{rendered}");
    assert!(rendered.contains("note"), "{rendered}");
    assert!(
        !rendered.contains("\"vector\""),
        "get must never emit a vector: {rendered}"
    );
}

#[test]
fn get_of_a_missing_id_is_a_plain_sentence_not_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("get"),
        &super::call(2, "get", json!({"collection": "mem", "id": "nope"})),
    );
    assert_eq!(status, 200, "a miss must not be an error: {body}");
    let said = super::text(&super::result(&body));
    assert!(
        said.to_lowercase().contains("no entry"),
        "should say plainly that there is no such entry: {said}"
    );
}

#[test]
fn browse_is_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("browse"),
        &super::call(1, "browse", json!({"collection": "mem", "limit": 2})),
    );
    assert_eq!(status, 200, "{body}");
    let rendered = super::text(&super::result(&body));
    let hits: serde_json::Value = serde_json::from_str(&rendered).unwrap_or_else(|_| json!([]));
    assert_eq!(
        hits.as_array().expect("hits array").len(),
        2,
        "limit should cap the page: {rendered}"
    );

    // An absurd limit is a caller fault, not an unbounded scan.
    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("browse"),
        &super::call(2, "browse", json!({"collection": "mem", "limit": 50_000})),
    );
    assert_eq!(status, 400, "an over-cap limit should be refused: {body}");
    assert_eq!(body["error"]["code"].as_i64(), Some(-32602), "{body}");
}

#[test]
fn browse_honours_a_filter() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("browse"),
        &super::call(
            3,
            "browse",
            json!({"collection": "mem", "filter": [{"Eq": ["kind", {"Str": "wip"}]}]}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    let rendered = super::text(&super::result(&body));
    assert!(rendered.contains("\"c\""), "{rendered}");
    assert!(!rendered.contains("\"a\""), "{rendered}");
    assert!(!rendered.contains("\"b\""), "{rendered}");
}
