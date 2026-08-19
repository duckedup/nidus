//! The e2e proof that a hosted-reranker provider stage changes the returned order against
//! a real `nidus serve` (and `nidus search`) binary — nidus-4ss's headline acceptance
//! criterion, and the one thing nothing in-process can prove. `rerank_mock.rs`'s hand-rolled
//! TCP mock stands in for Voyage; the suite runs with no provider secrets.

#![cfg(all(feature = "cli", feature = "rerank-voyage", feature = "embed-ollama"))]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};

use serde_json::{Value, json};

use crate::harness::{RunningServer, Server};
use crate::rerank_mock::{mock_reranker_constant, mock_reranker_inverting};

/// The dimension every test in this suite shares.
const DIM: usize = 2;

/// A persistent mock embedder answering the same vector for every text, in Ollama's wire
/// shape — ties every memory's cosine score, so `recall`'s plain order is the store's own
/// `(collection, id)` tie-break, a deterministic input the inverting mock can prove it reversed.
fn mock_embedder_fixed(dim: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock embedder");
    let addr = listener.local_addr().expect("mock embedder addr");
    let vector: Vec<f64> = (0..dim).map(|i| (i + 1) as f64 * 0.1).collect();
    let body = json!({ "embeddings": [vector] }).to_string();
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            drain_request(&mut stream);
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
        }
    });
    format!("http://{addr}")
}

/// Drain one HTTP/1.1 request so the connection can be answered; the fixed embedder
/// mock never needs to look at the body.
fn drain_request(stream: &mut TcpStream) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    loop {
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            return;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).position(|w| w == b"\r\n\r\n").is_some() {
            return;
        }
    }
}

/// A server wired to the fixed embedder (only `recall` needs one) and a reranker pointed
/// at `rerank_url` — the shape `Server::new(...).args([...])` the blueprint specifies.
fn reranked_server(dir: &std::path::Path, rerank_url: &str) -> RunningServer {
    let embed_url = mock_embedder_fixed(DIM);
    Server::new(dir, DIM)
        .args([
            "--embed-provider",
            "ollama",
            "--embed-base-url",
            &embed_url,
            "--rerank-provider",
            "voyage",
            "--rerank-base-url",
            rerank_url,
            "--rerank-api-key",
            "x",
        ])
        .start()
}

fn create_collection(server: &RunningServer, name: &str) {
    let (status, body) = server.post(&format!("/collections/{name}"), &json!({}));
    assert_eq!(status, 200, "create {name}: {body}");
}

fn upsert(server: &RunningServer, name: &str, records: Value) {
    let (status, body) = server.post(
        &format!("/collections/{name}/upsert"),
        &json!({"records": records}),
    );
    assert_eq!(status, 200, "upsert {name}: {body}");
}

fn set_fts_schema(server: &RunningServer, name: &str, field: &str) {
    let (status, body) = server.post(
        &format!("/collections/{name}/fts-schema"),
        &json!({"fields": [field]}),
    );
    assert_eq!(status, 200, "fts-schema {name}: {body}");
}

/// The ids of a hits array, in the order printed — the only thing every test here asserts.
fn ids(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected a JSON array, got {v}"))
        .iter()
        .map(|h| h["id"].as_str().expect("an id").to_string())
        .collect()
}

/// Three records with a known, untied cosine order against `[1.0, 0.0]`: "a" exact match,
/// "b" partial, "c" orthogonal. Each carries `nidus.text` so the reranker has text to score.
fn seed_vectors(server: &RunningServer, collection: &str) {
    create_collection(server, collection);
    upsert(
        server,
        collection,
        json!([
            {"id": "a", "vector": [1.0, 0.0], "attrs": {"nidus.text": {"Str": "alpha document"}}},
            {"id": "b", "vector": [0.5, 0.5], "attrs": {"nidus.text": {"Str": "beta document"}}},
            {"id": "c", "vector": [0.0, 1.0], "attrs": {"nidus.text": {"Str": "gamma document"}}}
        ]),
    );
}

/// `/search`: with `top_k` covering every candidate, an inverting reranker must produce
/// the exact reverse of the un-reranked order. This is the test to disable the rerank stage
/// against and watch fail — an ordering assertion that passes either way proves nothing.
#[test]
fn search_order_inverts_with_a_rerank_stage() {
    let dir = tempfile::tempdir().unwrap();
    let mock = mock_reranker_inverting();
    let server = reranked_server(dir.path(), &mock);
    seed_vectors(&server, "docs");

    let (status, plain) = server.post("/search", &json!({"query": [1.0, 0.0], "top_k": 3}));
    assert_eq!(status, 200, "{plain}");
    let plain_ids = ids(&plain);

    let (status, reranked) = server.post(
        "/search",
        &json!({"query": [1.0, 0.0], "top_k": 3, "rerank": {"query": "q"}}),
    );
    assert_eq!(status, 200, "{reranked}");
    let mut expected = plain_ids.clone();
    expected.reverse();
    let reranked_ids = ids(&reranked);
    assert_eq!(
        reranked_ids, expected,
        "rerank must exactly invert the plain order"
    );
    assert_ne!(
        reranked_ids, plain_ids,
        "a no-op rerank must not pass this test"
    );
}

/// `/hybrid-search`: same inversion proof, over the fused (vector + BM25) ranking.
#[test]
fn hybrid_search_order_inverts_with_a_rerank_stage() {
    let dir = tempfile::tempdir().unwrap();
    let mock = mock_reranker_inverting();
    let server = reranked_server(dir.path(), &mock);
    create_collection(&server, "docs");
    set_fts_schema(&server, "docs", "nidus.text");
    // Term frequency descends with the same a/b/c order as the vector leg's cosine, so
    // both legs agree unambiguously and the fused rank never depends on a tie-break.
    upsert(
        &server,
        "docs",
        json!([
            {"id": "a", "vector": [1.0, 0.0], "attrs": {"nidus.text": {"Str": "keyword keyword keyword"}}},
            {"id": "b", "vector": [0.5, 0.5], "attrs": {"nidus.text": {"Str": "keyword keyword"}}},
            {"id": "c", "vector": [0.0, 1.0], "attrs": {"nidus.text": {"Str": "keyword"}}}
        ]),
    );

    let body = |rerank: Option<Value>| {
        let mut b = json!({
            "vector": [1.0, 0.0], "field": "nidus.text", "text": "keyword", "top_k": 3
        });
        if let Some(r) = rerank {
            b["rerank"] = r;
        }
        b
    };

    let (status, plain) = server.post("/hybrid-search", &body(None));
    assert_eq!(status, 200, "{plain}");
    let plain_ids = ids(&plain);

    let (status, reranked) = server.post("/hybrid-search", &body(Some(json!({"query": "q"}))));
    assert_eq!(status, 200, "{reranked}");
    let mut expected = plain_ids.clone();
    expected.reverse();
    let reranked_ids = ids(&reranked);
    assert_eq!(
        reranked_ids, expected,
        "rerank must exactly invert the fused order"
    );
    assert_ne!(
        reranked_ids, plain_ids,
        "a no-op rerank must not pass this test"
    );
}

/// `/text-search`: same inversion proof, over a plain BM25 ranking.
#[test]
fn text_search_order_inverts_with_a_rerank_stage() {
    let dir = tempfile::tempdir().unwrap();
    let mock = mock_reranker_inverting();
    let server = reranked_server(dir.path(), &mock);
    create_collection(&server, "docs");
    set_fts_schema(&server, "docs", "nidus.text");
    upsert(
        &server,
        "docs",
        json!([
            {"id": "a", "attrs": {"nidus.text": {"Str": "keyword keyword keyword"}}},
            {"id": "b", "attrs": {"nidus.text": {"Str": "keyword keyword"}}},
            {"id": "c", "attrs": {"nidus.text": {"Str": "keyword"}}}
        ]),
    );

    let body = |rerank: Option<Value>| {
        let mut b = json!({"field": "nidus.text", "query": "keyword", "top_k": 3});
        if let Some(r) = rerank {
            b["rerank"] = r;
        }
        b
    };

    let (status, plain) = server.post("/text-search", &body(None));
    assert_eq!(status, 200, "{plain}");
    let plain_ids = ids(&plain);

    let (status, reranked) = server.post("/text-search", &body(Some(json!({"query": "q"}))));
    assert_eq!(status, 200, "{reranked}");
    let mut expected = plain_ids.clone();
    expected.reverse();
    let reranked_ids = ids(&reranked);
    assert_eq!(
        reranked_ids, expected,
        "rerank must exactly invert the BM25 order"
    );
    assert_ne!(
        reranked_ids, plain_ids,
        "a no-op rerank must not pass this test"
    );
}

/// `/collections/{name}/recall`: the memory surface, where a rerank is most likely used
/// in practice. The fixed embedder ties every memory, so the plain order is the store's own
/// `(collection, id)` tie-break — a deterministic input the inverting mock proves it reversed.
#[test]
fn recall_order_inverts_with_a_rerank_stage() {
    let dir = tempfile::tempdir().unwrap();
    let mock = mock_reranker_inverting();
    let server = reranked_server(dir.path(), &mock);

    for (id, text) in [("a", "alpha"), ("b", "beta"), ("c", "gamma")] {
        let (status, body) = server.post(
            "/collections/notes/remember",
            &json!({"id": id, "text": text}),
        );
        assert_eq!(status, 200, "remember {id}: {body}");
    }

    let (status, plain) = server.post(
        "/collections/notes/recall",
        &json!({"query": "anything", "top_k": 3}),
    );
    assert_eq!(status, 200, "{plain}");
    let plain_ids = ids(&plain);

    let (status, reranked) = server.post(
        "/collections/notes/recall",
        &json!({"query": "anything", "top_k": 3, "rerank": {"query": "q"}}),
    );
    assert_eq!(status, 200, "{reranked}");
    let mut expected = plain_ids.clone();
    expected.reverse();
    let reranked_ids = ids(&reranked);
    assert_eq!(
        reranked_ids, expected,
        "rerank must exactly invert the recall order"
    );
    assert_ne!(
        reranked_ids, plain_ids,
        "a no-op rerank must not pass this test"
    );
}

/// `/search` with a `rerank` object but no `query` is refused: a raw-vector query has
/// no text of its own, so there is nothing to score against.
#[test]
fn search_rerank_without_a_query_is_a_400() {
    let dir = tempfile::tempdir().unwrap();
    let mock = mock_reranker_inverting();
    let server = reranked_server(dir.path(), &mock);
    seed_vectors(&server, "docs");

    let (status, body) = server.post(
        "/search",
        &json!({"query": [1.0, 0.0], "top_k": 3, "rerank": {}}),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"].as_str().unwrap_or_default().contains("query"),
        "should name the missing query: {body}"
    );
}

/// Passthrough: no candidate carries `nidus.text`, so a reranked request must come back
/// byte-identical in order to the un-reranked one — the mock's inversion would show up in
/// the order if the empty-candidates short-circuit were broken.
#[test]
fn search_passes_through_candidates_with_no_text() {
    let dir = tempfile::tempdir().unwrap();
    let mock = mock_reranker_inverting();
    let server = reranked_server(dir.path(), &mock);
    create_collection(&server, "docs");
    upsert(
        &server,
        "docs",
        json!([
            {"id": "a", "vector": [1.0, 0.0], "attrs": {}},
            {"id": "b", "vector": [0.5, 0.5], "attrs": {}},
            {"id": "c", "vector": [0.0, 1.0], "attrs": {}}
        ]),
    );

    let (status, plain) = server.post("/search", &json!({"query": [1.0, 0.0], "top_k": 3}));
    assert_eq!(status, 200, "{plain}");
    let plain_ids = ids(&plain);

    let (status, reranked) = server.post(
        "/search",
        &json!({"query": [1.0, 0.0], "top_k": 3, "rerank": {"query": "q"}}),
    );
    assert_eq!(status, 200, "{reranked}");
    assert_eq!(
        ids(&reranked),
        plain_ids,
        "no candidate carries text: the order must be unchanged"
    );
}

/// The tie-break: a constant-score reranker means every candidate ties, so the response
/// order must be `(collection, id)` ascending — the ids sort differently from the cosine
/// order, so this cannot pass by coincidentally matching either the plain or reversed order.
#[test]
fn tie_break_is_collection_then_id_ascending() {
    let dir = tempfile::tempdir().unwrap();
    let mock = mock_reranker_constant();
    let server = reranked_server(dir.path(), &mock);
    create_collection(&server, "docs");
    upsert(
        &server,
        "docs",
        json!([
            {"id": "z", "vector": [1.0, 0.0], "attrs": {"nidus.text": {"Str": "z text"}}},
            {"id": "y", "vector": [0.5, 0.5], "attrs": {"nidus.text": {"Str": "y text"}}},
            {"id": "x", "vector": [0.0, 1.0], "attrs": {"nidus.text": {"Str": "x text"}}}
        ]),
    );

    let (status, plain) = server.post("/search", &json!({"query": [1.0, 0.0], "top_k": 3}));
    assert_eq!(status, 200, "{plain}");
    assert_eq!(
        ids(&plain),
        vec!["z", "y", "x"],
        "sanity: plain cosine order: {plain}"
    );

    let (status, reranked) = server.post(
        "/search",
        &json!({"query": [1.0, 0.0], "top_k": 3, "rerank": {"query": "q"}}),
    );
    assert_eq!(status, 200, "{reranked}");
    assert_eq!(
        ids(&reranked),
        vec!["x", "y", "z"],
        "equal provider scores must tie-break (collection, id) ascending: {reranked}"
    );
}

/// A server started with no `--rerank-provider` refuses a `rerank` field outright,
/// rather than silently answering the un-reranked order.
#[test]
fn search_rerank_without_a_configured_reranker_is_a_400() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM).start();
    create_collection(&server, "docs");
    upsert(
        &server,
        "docs",
        json!([{"id": "a", "vector": [1.0, 0.0], "attrs": {}}]),
    );

    let (status, body) = server.post(
        "/search",
        &json!({"query": [1.0, 0.0], "top_k": 1, "rerank": {"query": "q"}}),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("reranker"),
        "should name the missing reranker: {body}"
    );
}

/// A projection excluding the text field must still rerank (the orchestrator
/// force-includes it over the fetch) and must not leak it back into the response attrs.
#[test]
fn projection_excluding_text_still_reranks_but_the_response_omits_it() {
    let dir = tempfile::tempdir().unwrap();
    let mock = mock_reranker_inverting();
    let server = reranked_server(dir.path(), &mock);
    seed_vectors(&server, "docs");

    let (status, plain) = server.post("/search", &json!({"query": [1.0, 0.0], "top_k": 3}));
    assert_eq!(status, 200, "{plain}");
    let mut expected = ids(&plain);
    expected.reverse();

    let (status, reranked) = server.post(
        "/search",
        &json!({
            "query": [1.0, 0.0], "top_k": 3,
            "include_attributes": [],
            "rerank": {"query": "q"}
        }),
    );
    assert_eq!(status, 200, "{reranked}");
    assert_eq!(
        ids(&reranked),
        expected,
        "the order still flips despite the narrow projection"
    );
    for hit in reranked.as_array().unwrap() {
        assert!(
            hit["attrs"].as_object().unwrap().is_empty(),
            "the forced-in text field must be trimmed back out: {hit}"
        );
    }
}

/// A minimal `nidus <args>` runner for the CLI leg. Deliberately not reusing `cli.rs`'s
/// `run`/`ok` helpers — those are private to that module — so this file stays disjoint.
fn run_nidus(args: &[&str], stdin: &str) -> Value {
    let mut child = Command::new(env!("CARGO_BIN_EXE_nidus"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear()
        .envs(std::env::vars().filter(|(k, _)| !k.starts_with("NIDUS_")))
        .spawn()
        .unwrap_or_else(|e| panic!("spawn nidus {args:?}: {e}"));
    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(stdin.as_bytes())
        .unwrap_or_else(|e| panic!("write stdin for {args:?}: {e}"));
    let out = child
        .wait_with_output()
        .unwrap_or_else(|e| panic!("wait for nidus {args:?}: {e}"));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "nidus {args:?} exited {:?}\n--- stderr ---\n{stderr}",
        out.status.code()
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!("nidus {args:?} printed non-JSON: {e}\n--- stdout ---\n{stdout}")
    })
}

/// `nidus search --rerank-provider voyage --rerank-base-url … --rerank-query …` over a
/// store directory, driven with no server at all — the CLI leg's only real proof.
#[test]
fn cli_search_rerank_inverts_the_printed_order() {
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().expect("utf-8 temp path");
    let mock = mock_reranker_inverting();

    run_nidus(&["create", "--dir", dir_str, "--dim", "2", "docs"], "");
    run_nidus(
        &["upsert", "--dir", dir_str, "docs"],
        &json!([
            {"id": "a", "vector": [1.0, 0.0], "attrs": {"nidus.text": {"Str": "alpha"}}},
            {"id": "b", "vector": [0.5, 0.5], "attrs": {"nidus.text": {"Str": "beta"}}},
            {"id": "c", "vector": [0.0, 1.0], "attrs": {"nidus.text": {"Str": "gamma"}}}
        ])
        .to_string(),
    );

    let plain = run_nidus(
        &["search", "--dir", dir_str, "--top-k", "3", "docs"],
        &json!([1.0, 0.0]).to_string(),
    );
    let plain_ids = ids(&plain);

    let reranked = run_nidus(
        &[
            "search",
            "--dir",
            dir_str,
            "--top-k",
            "3",
            "--rerank-provider",
            "voyage",
            "--rerank-base-url",
            &mock,
            "--rerank-api-key",
            "x",
            "--rerank-query",
            "q",
            "docs",
        ],
        &json!([1.0, 0.0]).to_string(),
    );
    let mut expected = plain_ids.clone();
    expected.reverse();
    let reranked_ids = ids(&reranked);
    assert_eq!(
        reranked_ids, expected,
        "the CLI search order must invert too"
    );
    assert_ne!(
        reranked_ids, plain_ids,
        "a no-op rerank must not pass this test"
    );
}
