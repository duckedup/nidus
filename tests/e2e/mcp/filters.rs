//! E2E tests for the metadata `filter` on `recall`, `text_search`, and `hybrid_search`
//! (nidus-k28.3). Filtering happens *before* scoring, so a tied embedding still proves which
//! ids survive; only the two query-embedding tools need an embedder, and are gated below.

use std::collections::BTreeSet;

use serde_json::{Value, json};

use crate::harness::{RunningServer, Server};

/// Seed a `mem` collection: five records over `project`/`kind`/`tags`/`body`, one (`"e"`)
/// deliberately missing `kind` to exercise the absent-key asymmetry. `body` carries "note"
/// in every record so a keyword search over it matches the whole corpus before filtering.
fn seed(server: &RunningServer) {
    assert_eq!(
        server
            .post("/collections/mem/fts-schema", &json!({"fields": ["body"]}))
            .0,
        200
    );
    let (status, body) = server.post(
        "/collections/mem/upsert",
        &json!({"records": [
            {"id": "a", "vector": [1, 0, 0], "attrs": {
                "project": {"Str": "nidus"}, "kind": {"Str": "decision"},
                "tags": {"List": ["rust", "wip"]}, "body": {"Str": "a rust note"}
            }},
            {"id": "b", "vector": [0, 1, 0], "attrs": {
                "project": {"Str": "nidus"}, "kind": {"Str": "note"},
                "tags": {"List": ["rust"]}, "body": {"Str": "a rust note"}
            }},
            {"id": "c", "vector": [0, 0, 1], "attrs": {
                "project": {"Str": "beads"}, "kind": {"Str": "decision"},
                "tags": {"List": ["go"]}, "body": {"Str": "a go note"}
            }},
            {"id": "d", "vector": [1, 1, 0], "attrs": {
                "project": {"Str": "other"}, "kind": {"Str": "decision"},
                "tags": {"List": ["rust"]}, "body": {"Str": "another note"}
            }},
            {"id": "e", "vector": [1, 0, 1], "attrs": {
                "project": {"Str": "nidus"},
                "tags": {"List": []}, "body": {"Str": "a kindless note"}
            }}
        ]}),
    );
    assert_eq!(status, 200, "seed upsert failed: {body}");
}

/// The ids of every hit in a `recall`/`text_search`/`hybrid_search` result's rendered JSON.
fn hit_ids(rendered: &str) -> BTreeSet<String> {
    let hits: Value = serde_json::from_str(rendered).unwrap_or_else(|_| json!([]));
    hits.as_array()
        .expect("hits array")
        .iter()
        .map(|h| h["id"].as_str().expect("hit id").to_string())
        .collect()
}

fn ids(set: &[&str]) -> BTreeSet<String> {
    set.iter().map(|s| s.to_string()).collect()
}

/// (project = nidus OR project = beads) AND NOT (tags contains "wip"). Excludes `a` (wip)
/// and `d` (other project); keeps `b`, `c`, and `e` — `e` has no `wip` tag to contain.
fn project_or_no_wip() -> Value {
    json!([
        {"Any": [
            {"Eq": ["project", {"Str": "nidus"}]},
            {"Eq": ["project", {"Str": "beads"}]}
        ]},
        {"Not": {"Contains": ["tags", {"Str": "wip"}]}}
    ])
}

/// A filtered `text_search` returns exactly the filter-matching subset, not the whole
/// keyword match — the base case every other test in this file builds on.
#[test]
fn text_search_filter_excludes_non_matching() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("text_search"),
        &super::call(
            1,
            "text_search",
            json!({
                "collection": "mem", "field": "body", "query": "note", "top_k": 10,
                "filter": [{"Eq": ["project", {"Str": "nidus"}]}]
            }),
        ),
    );
    assert_eq!(status, 200, "{body}");
    let rendered = super::text(&super::result(&body));
    assert_eq!(hit_ids(&rendered), ids(&["a", "b", "e"]), "{rendered}");
}

/// `Any`/`Not` nested together — proves the recursive `$defs`-based schema round-trips a
/// realistic boolean shape, not just one flat predicate.
#[test]
fn nested_any_and_not_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("text_search"),
        &super::call(
            2,
            "text_search",
            json!({
                "collection": "mem", "field": "body", "query": "note", "top_k": 10,
                "filter": project_or_no_wip()
            }),
        ),
    );
    assert_eq!(status, 200, "{body}");
    let rendered = super::text(&super::result(&body));
    assert_eq!(hit_ids(&rendered), ids(&["b", "c", "e"]), "{rendered}");

    // `Not` wrapping an `Any` instead: NOT (project = other OR tags contains "wip").
    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("text_search"),
        &super::call(
            3,
            "text_search",
            json!({
                "collection": "mem", "field": "body", "query": "note", "top_k": 10,
                "filter": [{"Not": {"Any": [
                    {"Eq": ["project", {"Str": "other"}]},
                    {"Contains": ["tags", {"Str": "wip"}]}
                ]}}]
            }),
        ),
    );
    assert_eq!(status, 200, "{body}");
    let rendered = super::text(&super::result(&body));
    assert_eq!(hit_ids(&rendered), ids(&["b", "c", "e"]), "{rendered}");
}

/// `Ne` requires the key present and different; `Not(Eq(...))` is a true complement and
/// also matches a record where the key is simply absent (`e` has no `kind`). Same asymmetry
/// the description warns about must actually hold, not just be documented.
#[test]
fn ne_and_not_eq_disagree_on_an_absent_key() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("text_search"),
        &super::call(
            4,
            "text_search",
            json!({
                "collection": "mem", "field": "body", "query": "note", "top_k": 10,
                "filter": [{"Ne": ["kind", {"Str": "decision"}]}]
            }),
        ),
    );
    assert_eq!(status, 200, "{body}");
    let ne_ids = hit_ids(&super::text(&super::result(&body)));
    assert_eq!(
        ne_ids,
        ids(&["b"]),
        "Ne must require `kind` present: {ne_ids:?}"
    );

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("text_search"),
        &super::call(
            5,
            "text_search",
            json!({
                "collection": "mem", "field": "body", "query": "note", "top_k": 10,
                "filter": [{"Not": {"Eq": ["kind", {"Str": "decision"}]}}]
            }),
        ),
    );
    assert_eq!(status, 200, "{body}");
    let not_eq_ids = hit_ids(&super::text(&super::result(&body)));
    assert_eq!(
        not_eq_ids,
        ids(&["b", "e"]),
        "Not(Eq(..)) must also match a missing key: {not_eq_ids:?}"
    );
}

/// A filter that fails to deserialize into `Filter` is a caller fault (-32602), the same
/// shape as `missing_required_argument_names_the_argument` in the parent suite.
#[test]
fn malformed_filter_is_a_caller_fault() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(&server);

    let (status, body) = super::mcp(
        &server,
        "tools/call",
        Some("text_search"),
        &super::call(
            6,
            "text_search",
            json!({
                "collection": "mem", "field": "body", "query": "note",
                "filter": "not a predicate array"
            }),
        ),
    );
    assert_eq!(status, 400, "a malformed filter is a caller fault: {body}");
    assert_eq!(body["error"]["code"].as_i64(), Some(-32602), "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("filter"),
        "should name `filter`: {message}"
    );
}

// `recall` and `hybrid_search` embed the query server-side, so their `filter` needs a real
// (if trivial) embedder — `embed-openai-compat`, present under `just ci-serve` but not the
// base `cli,mcp` lane. `text_search` above already covers the filter logic without one.
#[cfg(feature = "embed-openai-compat")]
mod with_embedder {
    use super::*;

    /// A one-request-per-connection OpenAI-compatible `/v1/embeddings` mock that always
    /// answers the same `dim`-length vector. Filtering happens before scoring, so a tied
    /// embedding cannot change which ids a filter keeps — only their (irrelevant) order.
    fn start_embed_mock(dim: usize) -> String {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind embed mock");
        let addr = listener.local_addr().expect("embed mock addr");
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                thread::spawn(move || {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 4096];
                    // Enough to drain a small request before replying; the mock ignores
                    // the body's content, so it does not need to be parsed.
                    loop {
                        match stream.read(&mut tmp) {
                            Ok(0) => break,
                            Ok(n) => {
                                buf.extend_from_slice(&tmp[..n]);
                                if buf.windows(4).any(|w| w == b"\r\n\r\n") || n < tmp.len() {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    let embedding = vec!["0.1"; dim].join(",");
                    let resp_body =
                        format!(r#"{{"data":[{{"embedding":[{embedding}],"index":0}}]}}"#);
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp_body.len(),
                        resp_body
                    );
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.flush();
                });
            }
        });
        format!("http://{addr}")
    }

    fn server_with_embedder(dir: &std::path::Path) -> RunningServer {
        let base_url = start_embed_mock(3);
        Server::new(dir, 3)
            .args([
                "--embed-provider",
                "openai-compat",
                "--embed-model",
                "mock",
                "--embed-base-url",
                &base_url,
            ])
            .start()
    }

    #[test]
    fn recall_filter_excludes_non_matching() {
        let dir = tempfile::tempdir().unwrap();
        let server = server_with_embedder(dir.path());
        seed(&server);

        let (status, body) = super::super::mcp(
            &server,
            "tools/call",
            Some("recall"),
            &super::super::call(
                1,
                "recall",
                json!({
                    "collection": "mem", "query": "note", "top_k": 10,
                    "filter": project_or_no_wip()
                }),
            ),
        );
        assert_eq!(status, 200, "{body}");
        let rendered = super::super::text(&super::super::result(&body));
        assert_eq!(hit_ids(&rendered), ids(&["b", "c", "e"]), "{rendered}");
    }

    #[test]
    fn hybrid_search_filter_excludes_non_matching() {
        let dir = tempfile::tempdir().unwrap();
        let server = server_with_embedder(dir.path());
        seed(&server);

        let (status, body) = super::super::mcp(
            &server,
            "tools/call",
            Some("hybrid_search"),
            &super::super::call(
                1,
                "hybrid_search",
                json!({
                    "collection": "mem", "field": "body", "query": "note", "top_k": 10,
                    "filter": [{"Eq": ["kind", {"Str": "decision"}]}]
                }),
            ),
        );
        assert_eq!(status, 200, "{body}");
        let rendered = super::super::text(&super::super::result(&body));
        assert_eq!(hit_ids(&rendered), ids(&["a", "c", "d"]), "{rendered}");
    }
}
