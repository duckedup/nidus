//! A hand-rolled TCP mock standing in for a Voyage-shaped rerank API (nidus-4ss), so the
//! `rerank` suite proves against a real binary with no provider secrets. Same
//! persistent-listener shape as `tests/e2e/mcp/support.rs`'s embedder mocks
//! (`support.rs:103-116`), duplicated rather than shared so this unit stays file-disjoint.

#![cfg(all(feature = "cli", feature = "rerank-voyage", feature = "embed-ollama"))]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use serde_json::{Value, json};

/// Drain one HTTP/1.1 request (headers + `Content-Length` body) and return the body bytes.
fn read_request_body(stream: &mut TcpStream) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    let header_end = loop {
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            return Vec::new();
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let content_length: usize = head
        .lines()
        .find_map(|l| {
            let l = l.to_ascii_lowercase();
            l.strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    while buf.len() < header_end + content_length {
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    buf[header_end..].to_vec()
}

/// Answer one request with `body` as a `200 application/json` response.
fn write_json_response(mut stream: TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
}

/// The length of the `documents` array in a Voyage-shaped rerank request body. Reading it
/// rather than assuming a fixed count matters: a fixed-length response would silently
/// truncate the candidate set and look like a working rerank.
fn document_count(body: &[u8]) -> usize {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v["documents"].as_array().map(|a| a.len()))
        .unwrap_or(0)
}

/// A persistent mock reranker that scores each document by its position in the request:
/// the candidate the store ranked last gets the highest score, so re-sorting produces the
/// exact inverse of the store's own order — no rule could be mistaken for a no-op.
pub(crate) fn mock_reranker_inverting() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock reranker");
    let addr = listener.local_addr().expect("mock reranker addr");
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let body = read_request_body(&mut stream);
            let n = document_count(&body);
            let data: Vec<Value> = (0..n)
                .map(|i| json!({"index": i, "relevance_score": i as f32}))
                .collect();
            write_json_response(stream, &json!({"data": data}).to_string());
        }
    });
    format!("http://{addr}")
}

/// A persistent mock reranker that answers an identical score for every document, so the
/// resulting order is decided purely by the store's own `(collection, id)` tie-break —
/// used to prove that path independent of any provider ordering.
pub(crate) fn mock_reranker_constant() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock reranker");
    let addr = listener.local_addr().expect("mock reranker addr");
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let body = read_request_body(&mut stream);
            let n = document_count(&body);
            let data: Vec<Value> = (0..n)
                .map(|i| json!({"index": i, "relevance_score": 1.0}))
                .collect();
            write_json_response(stream, &json!({"data": data}).to_string());
        }
    });
    format!("http://{addr}")
}

/// A minimal blocking HTTP/1.1 POST, used only by this file's own self-test: connect,
/// send `body` with a `Content-Length`, then drain the response the same way the mock
/// itself drains a request.
fn post(url: &str, body: &str) -> Vec<u8> {
    let addr = url.strip_prefix("http://").expect("mock url has no scheme");
    let mut stream = TcpStream::connect(addr).expect("connect to mock reranker");
    let request = format!(
        "POST / HTTP/1.1\r\nHost: {addr}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream
        .write_all(request.as_bytes())
        .expect("send mock request");
    read_response_body(&mut stream)
}

/// The response-side mirror of [`read_request_body`]: same `Content-Length` framing, a
/// status line instead of a request line.
fn read_response_body(stream: &mut TcpStream) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    let header_end = loop {
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            return Vec::new();
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let content_length: usize = head
        .lines()
        .find_map(|l| {
            let l = l.to_ascii_lowercase();
            l.strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    while buf.len() < header_end + content_length {
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    buf[header_end..].to_vec()
}

/// The mock's own behaviour, pinned directly: a 3-document request must come back with
/// scores that strictly increase by index — the property [`mock_reranker_inverting`]
/// needs in order to invert the store's (best-first) candidate order.
#[test]
fn mock_reranker_inverting_scores_strictly_ascend_by_index() {
    let url = mock_reranker_inverting();
    let body = json!({
        "model": "rerank-2.5",
        "query": "q",
        "documents": ["d0", "d1", "d2"],
        "return_documents": false
    })
    .to_string();
    let resp: Value = serde_json::from_slice(&post(&url, &body)).expect("mock response JSON");
    let mut by_index: Vec<(usize, f32)> = resp["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|e| {
            (
                e["index"].as_u64().expect("index") as usize,
                e["relevance_score"].as_f64().expect("relevance_score") as f32,
            )
        })
        .collect();
    by_index.sort_by_key(|(i, _)| *i);
    let scores: Vec<f32> = by_index.into_iter().map(|(_, s)| s).collect();
    assert_eq!(scores.len(), 3, "{scores:?}");
    assert!(
        scores[0] < scores[1] && scores[1] < scores[2],
        "scores must strictly increase by index so a resort inverts the input order: {scores:?}"
    );
}
