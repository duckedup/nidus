//! The MCP leg of SQL-shaped read syntax (nidus-yq9p.2/.3/.6): the `query` tool reads the
//! real store like every other tool here, and refuses a literal vector (U4's law — this
//! surface is text-native, `.claude/rules/cli-feature.md`). Shared parse-error-message
//! consistency across CLI/HTTP/MCP lives in `tests/e2e/sql.rs`; this file is MCP-specific.

use serde_json::json;

use crate::harness::Server;

/// `ORDER BY knn(...)` takes a literal vector no model can type here, so the `query` tool
/// refuses it before the SQL ever reaches the parser — a caller fault naming the offset.
#[test]
fn query_tool_refuses_a_vector_literal() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let sql = "SELECT * FROM notes ORDER BY knn([1,0,0])";
    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("query"),
        &super::call(1, "query", json!({"sql": sql})),
    );
    assert_eq!(
        status, 400,
        "a refused vector literal is a caller fault: {body}"
    );
    assert_eq!(
        body["error"]["code"].as_i64(),
        Some(-32602),
        "invalid_params, not a server fault: {body}"
    );
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("sql parse error"), "{message}");
    assert!(message.contains("at byte"), "{message}");
    assert!(message.contains("§7.6"), "{message}");
    assert!(
        message.contains("knn(...) needs a vector"),
        "the message should say why, not just that it failed: {message}"
    );
}

/// `knn` with no following `(` is not a ranking call — an ordinary field named `knnish`
/// must not be refused just for containing the substring.
#[test]
fn query_tool_does_not_refuse_knn_as_a_mere_substring() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    assert_eq!(server.post("/collections/notes", &json!({})).0, 200);

    // `knnish` is an identifier, not a call to `knn(...)` — the field need not exist for the
    // refusal check to run first, so this proves the scanner is word-boundary aware.
    let sql = "SELECT * FROM notes WHERE knnish = 'x' ORDER BY score";
    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("query"),
        &super::call(1, "query", json!({"sql": sql})),
    );
    assert_eq!(
        status, 200,
        "a bare-word `knnish` must not be refused as a vector literal: {body}"
    );
}

/// The tool reads the same store the HTTP routes write, exactly like every other tool here
/// (`tools_read_the_same_store_the_http_routes_write`): a WHERE filter plus a match(...)
/// ranking, both compiled from SQL, both answered from the real index.
#[test]
fn query_tool_runs_a_real_filter_and_match_query() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    assert_eq!(server.post("/collections/notes", &json!({})).0, 200);
    assert_eq!(
        server
            .post(
                "/collections/notes/fts-schema",
                &json!({"fields": ["body"]})
            )
            .0,
        200
    );
    let (status, _) = server.post(
        "/collections/notes/upsert",
        &json!({"records": [
            {"id": "a", "vector": [1, 0, 0], "attrs": {
                "lang": {"Str": "rust"}, "body": {"Str": "ranking bug in upsert"}}},
            {"id": "b", "vector": [0, 1, 0], "attrs": {
                "lang": {"Str": "go"}, "body": {"Str": "ranking bug in upsert"}}}
        ]}),
    );
    assert_eq!(status, 200);

    let sql = "SELECT * FROM notes WHERE lang = 'rust' ORDER BY match(body, 'ranking bug')";
    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("query"),
        &super::call(1, "query", json!({"sql": sql})),
    );
    assert_eq!(status, 200, "query failed: {body}");
    let hits = super::text(&super::result(&body));
    assert!(
        hits.contains("\"a\""),
        "the filter-matching record should be found: {hits}"
    );
    assert!(
        !hits.contains("\"b\""),
        "the other-language record must be excluded: {hits}"
    );
}

/// A parse error is a caller fault, `invalid_params`, carrying the same marker/offset/section
/// text every other surface produces (root blueprint's error model).
#[test]
fn query_tool_reports_a_parse_error_as_invalid_params() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let sql = "SELECT * FROM notes WHERE lang =";
    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("query"),
        &super::call(1, "query", json!({"sql": sql})),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["error"]["code"].as_i64(), Some(-32602), "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("sql parse error"), "{message}");
    assert!(message.contains("at byte"), "{message}");
    assert!(message.contains("§7.12"), "{message}");
}
