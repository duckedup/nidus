//! Collection alias end-to-end tests: a real `nidus serve` process over a real socket,
//! the real `nidus` binary driving the CLI subcommands, and the cross-process repoint
//! case only two separate processes can prove (nidus-klh).
//!
//! Every assertion below names which collection's records came back and which concrete
//! collection name the hit carries — never just that the machinery ran.

use serde_json::{Value, json};

use crate::harness::{self, RunningServer, Server, ok};

// ── HTTP helpers: PUT/DELETE reach only the alias admin routes, so these two verbs'
// wrappers live here beside their tests (the send they delegate to is shared). ──

fn put(server: &RunningServer, path: &str, body: &Value) -> (u16, Value) {
    harness::send(server, "PUT", path, Some(body))
}

fn delete(server: &RunningServer, path: &str) -> (u16, Value) {
    harness::send(server, "DELETE", path, None)
}

fn create(server: &RunningServer, name: &str) {
    let (status, body) = server.post(&format!("/collections/{name}"), &json!({}));
    assert_eq!(status, 200, "create {name}: {body}");
}

fn upsert_one(server: &RunningServer, collection: &str, id: &str, vector: [f32; 3]) {
    let (status, body) = server.post(
        &format!("/collections/{collection}/upsert"),
        &json!({"records": [{"id": id, "vector": vector, "attrs": {}}]}),
    );
    assert_eq!(status, 200, "upsert into {collection}: {body}");
}

/// The vector query every test below shares: it points straight at whichever
/// collection's sole record is `[1, 0, 0]`, so which id/collection comes back is the
/// whole assertion.
const QUERY: [f32; 3] = [1.0, 0.0, 0.0];

fn search_scope(server: &RunningServer, scope: &str) -> (u16, Value) {
    server.post(
        "/search",
        &json!({"query": QUERY, "top_k": 5, "scope": [scope]}),
    )
}

// ── 1. The HTTP round trip on a real socket ──────────────────────────────────

/// A repoint over a real socket changes which collection a scoped search returns, hits
/// carry the concrete name on the wire, and dropping the alias leaves its former
/// target's records reachable.
#[test]
fn alias_repoint_over_http_changes_which_collection_a_search_returns() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    create(&server, "docs");
    upsert_one(&server, "docs", "a", QUERY);
    create(&server, "docs_v2");
    upsert_one(&server, "docs_v2", "b", QUERY);

    let (status, body) = put(&server, "/aliases/docs_alias", &json!({"target": "docs"}));
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["alias"], "docs_alias");
    assert_eq!(body["target"], "docs");

    let (status, hits) = search_scope(&server, "docs_alias");
    assert_eq!(status, 200, "{hits}");
    assert_eq!(hits[0]["id"], "a", "{hits}");
    assert_eq!(
        hits[0]["collection"], "docs",
        "hit must carry the concrete collection, not the alias: {hits}"
    );

    let (status, aliases) = server.get("/aliases");
    assert_eq!(status, 200);
    assert_eq!(aliases, json!({"docs_alias": "docs"}));

    let (status, collections) = server.get("/collections");
    assert_eq!(status, 200);
    let names: Vec<String> = serde_json::from_value(collections).unwrap();
    assert!(names.contains(&"docs".to_string()));
    assert!(names.contains(&"docs_v2".to_string()));
    assert!(
        !names.contains(&"docs_alias".to_string()),
        "GET /collections must never list an alias: {names:?}"
    );

    // Repoint: the same alias, the same scope, a different collection's record comes back.
    let (status, body) = put(
        &server,
        "/aliases/docs_alias",
        &json!({"target": "docs_v2"}),
    );
    assert_eq!(status, 200, "{body}");

    let (status, hits) = search_scope(&server, "docs_alias");
    assert_eq!(status, 200, "{hits}");
    assert_eq!(hits[0]["id"], "b", "{hits}");
    assert_eq!(hits[0]["collection"], "docs_v2", "{hits}");

    // Drop the alias: it resolves to nothing, but docs_v2's own record is unaffected.
    let (status, body) = delete(&server, "/aliases/docs_alias");
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["dropped"], "docs_alias");

    let (status, hits) = search_scope(&server, "docs_alias");
    assert!(
        status >= 400 || hits.as_array().map(Vec::len) == Some(0),
        "a dropped alias must fail or return nothing, not stale data: {status} {hits}"
    );

    let (status, hits) = search_scope(&server, "docs_v2");
    assert_eq!(status, 200);
    assert_eq!(
        hits[0]["id"], "b",
        "docs_v2 must still be reachable under its concrete name: {hits}"
    );
}

// ── 2. The CLI, driving the real binary ──────────────────────────────────────

/// Deliberately NOT `harness::fails`: this one does not assert stdout is empty. Whether
/// that omission is intentional or copy drift is an open question (see nidus-178); until
/// it is settled, unifying would either add an assertion or drop one.
fn cli_fails(args: &[&str], stdin: &str) -> String {
    let out = harness::run(args, stdin);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success(),
        "nidus {args:?} unexpectedly succeeded\n--- stdout ---\n{stdout}"
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn cli_ids(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected a JSON array, got {v}"))
        .iter()
        .map(|h| h["id"].as_str().expect("an id").to_string())
        .collect()
}

/// `set-alias`, `aliases`, `drop-alias` against the real binary, plus a search scoped to
/// the alias resolving to the concrete target's records at each step. Also covers the
/// two rejection paths: a chained repoint, and `set-alias` at a non-existent target.
#[test]
fn cli_alias_lifecycle_against_a_real_store() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");

    let out = ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    assert_eq!(out["created"], "docs");
    ok(
        &["upsert", "--dir", dir, "docs"],
        &json!([{"id": "a", "vector": QUERY, "attrs": {}}]).to_string(),
    );

    let out = ok(&["create", "--dir", dir, "docs_v2"], "");
    assert_eq!(out["created"], "docs_v2");
    ok(
        &["upsert", "--dir", dir, "docs_v2"],
        &json!([{"id": "b", "vector": QUERY, "attrs": {}}]).to_string(),
    );

    let out = ok(&["set-alias", "--dir", dir, "docs_alias", "docs"], "");
    assert_eq!(out["alias"], "docs_alias");
    assert_eq!(out["target"], "docs");

    let out = ok(&["aliases", "--dir", dir], "");
    assert_eq!(out, json!({"docs_alias": "docs"}));

    let hits = ok(
        &["search", "--dir", dir, "-k", "5", "docs_alias"],
        &json!(QUERY).to_string(),
    );
    assert_eq!(cli_ids(&hits), ["a"], "{hits}");
    assert_eq!(hits[0]["collection"], "docs", "{hits}");

    // Repoint: same alias, same scope, the other collection's record comes back.
    ok(&["set-alias", "--dir", dir, "docs_alias", "docs_v2"], "");
    let hits = ok(
        &["search", "--dir", dir, "-k", "5", "docs_alias"],
        &json!(QUERY).to_string(),
    );
    assert_eq!(cli_ids(&hits), ["b"], "{hits}");
    assert_eq!(hits[0]["collection"], "docs_v2", "{hits}");

    let out = ok(&["drop-alias", "--dir", dir, "docs_alias"], "");
    assert_eq!(out["dropped"], "docs_alias");

    // Rejection: a chained repoint through an existing alias.
    ok(&["set-alias", "--dir", dir, "chain_alias", "docs"], "");
    let err = cli_fails(
        &["set-alias", "--dir", dir, "another_alias", "chain_alias"],
        "",
    );
    assert!(
        err.contains("is itself an alias"),
        "expected the chain rejection message: {err}"
    );

    // Rejection: a target that does not exist.
    let err = cli_fails(&["set-alias", "--dir", dir, "bogus_alias", "nope"], "");
    assert!(
        err.contains("no such collection"),
        "expected the missing-target message: {err}"
    );
}

// ── 3. Cross-process visibility ──────────────────────────────────────────────

/// A reader process sees a repoint only after `POST /refresh`, never a window resolving
/// to nothing. Goes red if the alias write path forgets the segment-version bump the
/// lock-free reader watches.
#[test]
fn a_reader_process_sees_the_repoint_only_after_refresh() {
    let dir = tempfile::tempdir().unwrap();

    let writer = Server::new(dir.path(), 3).start();
    create(&writer, "docs");
    upsert_one(&writer, "docs", "a", QUERY);
    create(&writer, "docs_v2");
    upsert_one(&writer, "docs_v2", "b", QUERY);
    let (status, body) = put(&writer, "/aliases/docs_alias", &json!({"target": "docs"}));
    assert_eq!(status, 200, "{body}");

    let reader = Server::new(dir.path(), 3).args(["--read-only"]).start();
    let (status, hits) = search_scope(&reader, "docs_alias");
    assert_eq!(status, 200, "{hits}");
    assert_eq!(
        hits[0]["id"], "a",
        "reader must see the alias before any repoint: {hits}"
    );
    assert_eq!(hits[0]["collection"], "docs", "{hits}");

    // The writer repoints...
    let (status, body) = put(
        &writer,
        "/aliases/docs_alias",
        &json!({"target": "docs_v2"}),
    );
    assert_eq!(status, 200, "{body}");

    // ...but the reader, having not refreshed, still serves the old target: never
    // empty, never a 5xx.
    let (status, hits) = search_scope(&reader, "docs_alias");
    assert_eq!(
        status, 200,
        "a repoint concurrent with a read must never 5xx the reader: {hits}"
    );
    assert_eq!(
        hits.as_array().map(Vec::len),
        Some(1),
        "and must never resolve to nothing: {hits}"
    );
    assert_eq!(
        hits[0]["collection"], "docs",
        "before /refresh the reader must still resolve the old target: {hits}"
    );
    assert_eq!(hits[0]["id"], "a", "{hits}");

    let (status, refreshed) = reader.post("/refresh", &json!({}));
    assert_eq!(status, 200, "{refreshed}");
    assert_eq!(refreshed["adopted"], true, "{refreshed}");

    let (status, hits) = search_scope(&reader, "docs_alias");
    assert_eq!(status, 200, "{hits}");
    assert_eq!(
        hits[0]["collection"], "docs_v2",
        "after /refresh the reader must resolve the new target: {hits}"
    );
    assert_eq!(hits[0]["id"], "b", "{hits}");
}

// ── 4. Restart ────────────────────────────────────────────────────────────────

/// An alias survives a server restart on the same directory — proof it round-tripped
/// through the manifest on real bytes, not just held in RAM.
#[test]
fn alias_survives_a_server_restart() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    create(&server, "docs");
    upsert_one(&server, "docs", "a", QUERY);
    let (status, body) = put(&server, "/aliases/docs_alias", &json!({"target": "docs"}));
    assert_eq!(status, 200, "{body}");

    assert!(server.shutdown(), "clean shutdown should exit successfully");

    let restarted = Server::new(dir.path(), 3).start();
    let (status, aliases) = restarted.get("/aliases");
    assert_eq!(status, 200);
    assert_eq!(
        aliases,
        json!({"docs_alias": "docs"}),
        "the alias must survive the manifest round trip: {aliases}"
    );

    let (status, hits) = search_scope(&restarted, "docs_alias");
    assert_eq!(status, 200, "{hits}");
    assert_eq!(hits[0]["id"], "a", "{hits}");
    assert_eq!(hits[0]["collection"], "docs", "{hits}");
}
