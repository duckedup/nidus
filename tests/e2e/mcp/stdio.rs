//! MCP over stdio (nidus-k28.1): the real `nidus mcp` binary, piped stdin/stdout, framed as
//! newline-delimited JSON-RPC. `harness::StdioServer` speaks the framing; the builders here
//! are stdio's own, since `mod.rs`'s header constructs have no equivalent over a pipe.

use serde_json::{Value, json};

use crate::harness::{RunningStdioServer, StdioServer};

use super::{VERSION, result, tool_names};

/// The `initialize` request every client sends first, over a real session-based transport
/// (unlike the header-driven HTTP surface, stdio negotiates once via the classic handshake).
fn initialize_request(id: u32) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "initialize",
        "params": {
            "protocolVersion": VERSION,
            "capabilities": {},
            "clientInfo": { "name": "nidus-e2e-stdio", "version": "0" }
        }
    })
}

/// The `notifications/initialized` notification — no `id`, and no response to read.
fn initialized_notification() -> Value {
    json!({ "jsonrpc": "2.0", "method": "notifications/initialized", "params": {} })
}

fn request(id: u32, method: &str, params: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
}

/// `initialize` then `notifications/initialized`; returns the `initialize` response's `result`.
fn handshake(server: &mut RunningStdioServer) -> Value {
    let resp = server.request(&initialize_request(1));
    let init = result(&resp);
    server.send(&initialized_notification());
    init
}

/// `initialize` reports the negotiated protocol version and names nidus, not the SDK — the
/// same distinction `discover_advertises_protocol_and_tools` pins for HTTP.
#[test]
fn initialize_reports_protocol_and_server_info() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = StdioServer::new(dir.path(), 3).spawn();

    let init = handshake(&mut server);
    assert_eq!(
        init["protocolVersion"], VERSION,
        "should negotiate the requested version: {init}"
    );
    assert_eq!(
        init["serverInfo"]["name"], "nidus",
        "serverInfo must name nidus, not rmcp: {init}"
    );
    assert_eq!(
        init["serverInfo"]["version"],
        env!("CARGO_PKG_VERSION"),
        "serverInfo version should track the crate version: {init}"
    );
}

/// The ten tools, in the same order HTTP reports them — the split must not have forked
/// the list by transport.
#[test]
fn tools_list_matches_http_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = StdioServer::new(dir.path(), 3).spawn();
    handshake(&mut server);

    let resp = server.request(&request(2, "tools/list", json!({})));
    let listed = result(&resp);
    assert_eq!(
        tool_names(&listed),
        vec![
            "remember",
            "recall",
            "text_search",
            "hybrid_search",
            "list_collections",
            "stats",
            "forget",
            "get",
            "browse",
            "related",
        ],
        "stdio's tool list/order must match HTTP's: {listed}"
    );
}

/// The same `NidusMcp` answers both transports, so resources/prompts must not have forked
/// by transport either — the sibling of `tools_list_matches_http_order` above.
#[test]
fn resources_and_prompts_match_over_stdio() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = StdioServer::new(dir.path(), 3).spawn();
    handshake(&mut server);

    let resp = server.request(&request(2, "resources/templates/list", json!({})));
    let listed = result(&resp);
    let templates = listed["resourceTemplates"]
        .as_array()
        .expect("resourceTemplates array");
    assert_eq!(templates.len(), 1, "{listed}");
    assert_eq!(
        templates[0]["uriTemplate"], "nidus://collections/{collection}/entries/{id}",
        "stdio's entry template must match HTTP's: {listed}"
    );

    let resp = server.request(&request(3, "prompts/list", json!({})));
    let listed = result(&resp);
    let prompts = listed["prompts"].as_array().expect("prompts array");
    assert_eq!(prompts.len(), 1, "{listed}");
    assert_eq!(
        prompts[0]["name"], "recall_then_answer",
        "stdio's prompt name must match HTTP's: {listed}"
    );
}

/// A held writer lock is a multi-client problem stdio does not solve — the design is that a
/// second local session fails immediately rather than queueing behind the first.
#[test]
fn a_second_process_on_the_same_directory_fails_fast_naming_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    let mut first = StdioServer::new(dir.path(), 3).spawn();
    handshake(&mut first);

    let second = StdioServer::new(dir.path(), 3).spawn();
    let (status, stderr) = second.wait();
    assert!(
        !status.success(),
        "a second `nidus mcp` on a locked store must exit nonzero\n--- stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("locked") && stderr.contains("nidus serve"),
        "the error should name the lock conflict and point at `nidus serve`: {stderr}"
    );
}

/// `--read-only` must actually open read-only rather than being silently dropped: a reader
/// takes no writer lock, so it neither blocks nor is blocked by a concurrent writer.
#[test]
fn read_only_opens_without_taking_the_writer_lock() {
    let dir = tempfile::tempdir().unwrap();
    // Seed the store so there is something to open read-only.
    drop(StdioServer::new(dir.path(), 3).spawn());

    let mut reader = StdioServer::new(dir.path(), 3)
        .args(["--read-only"])
        .spawn();
    handshake(&mut reader);

    // The writer lock is still free, so a normal session starts alongside the reader.
    let mut writer = StdioServer::new(dir.path(), 3).spawn();
    handshake(&mut writer);
}

/// `remember` -> `recall`, needing a real (if fake) embedding provider. `test-e2e`'s default
/// `cli,mcp` build carries no `embed-*` feature, so this compiles and round-trips only once
/// `embed-ollama` is present — e.g. under `just ci-serve`.
#[cfg(feature = "embed-ollama")]
mod round_trip {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};

    use serde_json::json;

    use crate::harness::StdioServer;

    use super::super::{result, text};
    use super::{call, handshake};

    #[test]
    fn remember_then_recall_round_trips_over_the_pipe() {
        let dir = tempfile::tempdir().unwrap();
        let base_url = mock_embedder(3);
        let mut server = StdioServer::new(dir.path(), 3)
            .args(["--embed-provider", "ollama", "--embed-base-url", &base_url])
            .spawn();
        handshake(&mut server);

        let resp = server.request(&call(
            2,
            "remember",
            json!({
                "collection": "notes",
                "text": "the ranking bug is in the upsert path",
                "id": "ranking-bug",
            }),
        ));
        let remembered = text(&result(&resp));
        assert!(
            remembered.contains("id `ranking-bug`"),
            "remember should confirm the id it stored, over the pipe: {remembered}"
        );

        // The mock always returns the same embedding, so any query hits — the assertion is
        // that stdio's `recall` reaches the exact entry `remember` just wrote over the same pipe.
        let resp = server.request(&call(
            3,
            "recall",
            json!({"collection": "notes", "query": "ranking bug"}),
        ));
        let recalled = text(&result(&resp));
        assert!(
            recalled.contains("ranking-bug"),
            "recall should find the remembered entry over the pipe: {recalled}"
        );
    }

    /// A tiny persistent mock HTTP server answering every request with the same
    /// fixed-dimension embedding — Ollama's wire shape (`{"embeddings": [[...]]}`),
    /// picked because it needs no API key.
    fn mock_embedder(dim: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock embedder");
        let addr = listener.local_addr().expect("mock embedder addr");
        let vector: Vec<f64> = (0..dim).map(|i| (i + 1) as f64 * 0.1).collect();
        let body = json!({ "embeddings": [vector] }).to_string();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                respond_once(stream, &body);
            }
        });
        format!("http://{addr}")
    }

    /// Drain one HTTP/1.1 request (headers + `Content-Length` body) and answer it with
    /// `body` — enough of the protocol for `reqwest` to round-trip, nothing more.
    fn respond_once(mut stream: TcpStream, body: &str) {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        let header_end = loop {
            let n = stream.read(&mut tmp).unwrap_or(0);
            if n == 0 {
                return;
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
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
    }
}

/// `tools/call` for `tool` with `arguments`. Only [`round_trip`] needs it.
#[cfg(feature = "embed-ollama")]
fn call(id: u32, tool: &str, arguments: Value) -> Value {
    request(
        id,
        "tools/call",
        json!({ "name": tool, "arguments": arguments }),
    )
}
