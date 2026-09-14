//! Collection lifecycle from MCP (nidus-k28.7): a pure-MCP client must reach a working
//! `text_search`/`hybrid_search` with no CLI and no HTTP setup call. Needs an embedder, so
//! this module compiles away outside `embed-ollama` (`just ci-serve`), like `attrs.rs`.

#![cfg(feature = "embed-ollama")]

use serde_json::json;

use super::support::{DIM, per_text_embedder_server};
use super::{call, mcp, result, text};

/// The whole point of #87: remember into a collection that does not exist yet, then find
/// it by wording, with no CLI or HTTP setup. Non-summarize is load-bearing — summarize
/// mode stamps `nidus.summary` too, hiding whether the default path stores anything.
#[test]
fn remember_then_text_search_needs_no_cli_setup() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("remember"),
        &call(
            1,
            "remember",
            json!({
                "collection": "fresh",
                "text": "the peregrine falcon nests on the cathedral tower",
                "id": "falcon"
            }),
        ),
    );
    assert_eq!(
        status, 200,
        "remember into a fresh collection failed: {body}"
    );

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("text_search"),
        &call(
            2,
            "text_search",
            json!({"collection": "fresh", "field": "nidus.text", "query": "peregrine"}),
        ),
    );
    assert_eq!(status, 200, "text_search failed: {body}");
    let rendered = text(&result(&body));
    assert!(
        rendered.contains("falcon"),
        "a word from the remembered text must be findable with no CLI setup — \
         the collection's FTS schema should have been declared on first write: {rendered}"
    );
}

/// The raw text must land in `nidus.text` on the default path, not only under `summarize`.
/// Without this the auto-declared schema would index nothing and k28.7 would look complete
/// while staying functionally dead.
#[test]
fn plain_remember_stores_the_raw_text_attr() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    let (status, _) = mcp(
        &server,
        "tools/call",
        Some("remember"),
        &call(
            1,
            "remember",
            json!({"collection": "notes", "text": "kestrels hover", "id": "k"}),
        ),
    );
    assert_eq!(status, 200);

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("get"),
        &call(2, "get", json!({"collection": "notes", "id": "k"})),
    );
    assert_eq!(status, 200, "get failed: {body}");
    let rendered = text(&result(&body));
    assert!(
        rendered.contains("nidus.text") && rendered.contains("kestrels hover"),
        "the raw text must be stored under nidus.text: {rendered}"
    );
}

/// `hybrid_search` fuses the vector and text legs, so it needs the same auto-declared
/// schema. Covering it separately because it is the tool the epic names as dead on arrival.
#[test]
fn hybrid_search_works_on_an_auto_provisioned_collection() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    for (id, body_text) in [
        ("a", "migrating swifts over the estuary"),
        ("b", "unrelated"),
    ] {
        let (status, _) = mcp(
            &server,
            "tools/call",
            Some("remember"),
            &call(
                1,
                "remember",
                json!({"collection": "birds", "text": body_text, "id": id}),
            ),
        );
        assert_eq!(status, 200);
    }

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("hybrid_search"),
        &call(
            2,
            "hybrid_search",
            json!({"collection": "birds", "field": "nidus.text", "query": "swifts estuary"}),
        ),
    );
    assert_eq!(status, 200, "hybrid_search failed: {body}");
    let rendered = text(&result(&body));
    assert!(
        rendered.contains("\"a\""),
        "the text-matching entry should surface through the fused ranking: {rendered}"
    );
}

/// nidus-29ui: `hybrid_search`'s `rollup` must cap chunks per parent, the same as `recall`
/// and `text_search` already do — proving `HybridOpts.limit_per`/`expand` are wired through
/// the fused ranking, not silently dropped.
#[test]
fn hybrid_search_rollup_caps_chunks_per_parent() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    assert_eq!(
        server
            .post(
                "/collections/docs/fts-schema",
                &json!({"fields": ["nidus.text"]})
            )
            .0,
        200
    );
    let (status, body) = server.post(
        "/collections/docs/upsert",
        &json!({"records": [
            {"id": "p1-a", "vector": [1, 0, 0], "attrs": {
                "nidus.parent_id": {"Str": "p1"}, "nidus.chunk_index": {"Int": 0},
                "nidus.text": {"Str": "swifts over the estuary at dawn"}
            }},
            {"id": "p1-b", "vector": [0.9, 0.1, 0], "attrs": {
                "nidus.parent_id": {"Str": "p1"}, "nidus.chunk_index": {"Int": 1},
                "nidus.text": {"Str": "swifts and estuary birds past noon"}
            }},
            {"id": "p2-a", "vector": [0, 1, 0], "attrs": {
                "nidus.parent_id": {"Str": "p2"}, "nidus.chunk_index": {"Int": 0},
                "nidus.text": {"Str": "an unrelated grocery list"}
            }}
        ]}),
    );
    assert_eq!(status, 200, "seed upsert failed: {body}");

    let ids = |args: serde_json::Value| -> std::collections::BTreeSet<String> {
        let (status, body) = mcp(
            &server,
            "tools/call",
            Some("hybrid_search"),
            &call(1, "hybrid_search", args),
        );
        assert_eq!(status, 200, "{body}");
        let hits: serde_json::Value =
            serde_json::from_str(&text(&result(&body))).expect("hits JSON");
        hits.as_array()
            .expect("hits array")
            .iter()
            .map(|h| h["id"].as_str().expect("hit id").to_string())
            .collect()
    };

    let plain = ids(
        json!({"collection": "docs", "field": "nidus.text", "query": "swifts estuary", "top_k": 5}),
    );
    assert!(
        plain.contains("p1-a") && plain.contains("p1-b"),
        "both p1 chunks should surface without a rollup cap: {plain:?}"
    );

    let capped = ids(json!({
        "collection": "docs", "field": "nidus.text", "query": "swifts estuary",
        "top_k": 5, "rollup": {"per_parent": 1}
    }));
    assert_eq!(
        capped.iter().filter(|id| id.starts_with("p1")).count(),
        1,
        "rollup {{per_parent: 1}} must cap p1's chunks to one: {capped:?}"
    );
    assert_ne!(
        plain, capped,
        "rollup must change the hit set, not be a no-op: {capped:?}"
    );
}
