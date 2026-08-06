//! End-to-end tests for the MCP `2026-07-28` surface at `/mcp` (nidus-zm2.3).
//!
//! These drive the real binary over a real socket, which is the only place several of
//! these properties are observable at all. The MCP endpoint is a nested `tower` service
//! rather than an axum handler, so an in-process `oneshot` on the router would not
//! exercise the nesting; and `2026-07-28` puts part of the protocol **in the headers**
//! (`Mcp-Method`/`Mcp-Name`), which only a real request carries.
//!
//! **No embedder is available in this lane** — the e2e services are an object store and a
//! memory tier, not an embedding provider — so `remember`/`recall`/`hybrid_search` cannot
//! be round-tripped here. That is covered deliberately rather than skipped: the tests
//! below assert those tools fail *honestly* (an operator-facing `internal_error`, never
//! `invalid_params`), which is the behaviour that keeps an agent from retrying forever
//! against a server that simply needs reconfiguring. The tools that need no embedder —
//! `list_collections`, `stats`, `text_search` — are round-tripped for real.

use serde_json::{Value, json};

use crate::harness::{RunningServer, Server};

/// The protocol version these tests speak.
const VERSION: &str = "2026-07-28";

/// Headers every MCP request needs.
///
/// `Accept` names both types because Streamable HTTP may answer either: nidus configures
/// `json_response`, so a tool call comes back as JSON, but rmcp falls back to an SSE
/// stream if a handler ever emits a notification first, and a client that did not accept
/// both would break on that fallback rather than on anything it did wrong.
fn headers<'a>(method: &'a str, name: Option<&'a str>) -> Vec<(&'a str, &'a str)> {
    let mut h = vec![
        ("accept", "application/json, text/event-stream"),
        ("mcp-protocol-version", VERSION),
        ("mcp-method", method),
    ];
    if let Some(name) = name {
        h.push(("mcp-name", name));
    }
    h
}

/// A JSON-RPC envelope carrying the `_meta` that `2026-07-28` requires per request.
///
/// The handshake is gone in this revision, so protocol version and client identity ride on
/// every call instead of being negotiated once. Omitting them is not a detail — the server
/// answers `MissingRequiredClientCapability`/`UnsupportedProtocolVersion` — so the helper
/// makes it impossible to forget in a test.
fn rpc(id: u32, method: &str, params: Value) -> Value {
    let mut params = params;
    params["_meta"] = json!({
        "io.modelcontextprotocol/protocolVersion": VERSION,
        "io.modelcontextprotocol/clientInfo": { "name": "nidus-e2e", "version": "0" },
        "io.modelcontextprotocol/clientCapabilities": {},
    });
    json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
}

/// `tools/call` for `tool` with `arguments`.
fn call(id: u32, tool: &str, arguments: Value) -> Value {
    rpc(
        id,
        "tools/call",
        json!({ "name": tool, "arguments": arguments }),
    )
}

/// POST an MCP request and return `(status, body)`.
fn mcp(server: &RunningServer, method: &str, name: Option<&str>, body: &Value) -> (u16, Value) {
    server.post_with_headers("/mcp", body, &headers(method, name))
}

/// The `result` of a successful JSON-RPC response, or a panic naming the error.
fn result(body: &Value) -> Value {
    assert!(
        body.get("error").is_none(),
        "expected a result, got JSON-RPC error: {body}"
    );
    body["result"].clone()
}

/// The concatenated text of a `tools/call` result's content blocks.
fn text(result: &Value) -> String {
    result["content"]
        .as_array()
        .expect("content array")
        .iter()
        .filter_map(|b| b["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The tool list, in the order the server returned it.
fn tool_names(result: &Value) -> Vec<String> {
    result["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|t| t["name"].as_str().expect("tool name").to_string())
        .collect()
}

/// `server/discover` is the one RPC `2026-07-28` requires of a server, and it must work as
/// the very first call — there is no handshake to precede it.
#[test]
fn discover_advertises_protocol_and_tools() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = mcp(
        &server,
        "server/discover",
        None,
        &rpc(1, "server/discover", json!({})),
    );
    assert_eq!(status, 200, "server/discover failed: {body}");

    let result = result(&body);

    // `2026-07-28` moved server identity into the result's `_meta` (SEP-2575) rather than a
    // top-level field, so that is where a conformant client looks for it.
    let info = &result["_meta"]["io.modelcontextprotocol/serverInfo"];
    // Must be THIS crate, not the SDK. `Implementation::from_build_env()` would report
    // `rmcp` here — a perfectly well-formed answer naming the wrong software, which is why
    // it is asserted rather than assumed.
    assert_eq!(
        info["name"], "nidus",
        "serverInfo must name nidus, not the SDK: {result}"
    );
    assert_eq!(
        info["version"],
        env!("CARGO_PKG_VERSION"),
        "serverInfo version should track the crate version: {result}"
    );

    // The whole reason this surface takes rmcp rather than hand-rolling `2026-07-28`:
    // older clients still work. Most MCP clients deployed today predate this revision, so
    // a server advertising only the newest version would talk to almost none of them.
    let versions: Vec<&str> = result["supportedVersions"]
        .as_array()
        .expect("supportedVersions array")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(
        versions.contains(&VERSION),
        "the current protocol version must be supported: {versions:?}"
    );
    assert!(
        versions.contains(&"2025-11-25"),
        "the previous revision must remain supported — dropping it would strand most \
         deployed clients: {versions:?}"
    );

    // Tools are the only capability this surface claims. If resources/prompts/logging ever
    // appear here it means something was enabled with no implementation behind it.
    assert!(
        result["capabilities"]["tools"].is_object(),
        "tools capability should be advertised: {result}"
    );
    for absent in ["resources", "prompts", "logging"] {
        assert!(
            result["capabilities"][absent].is_null(),
            "{absent} must not be advertised — nothing implements it: {result}"
        );
    }
}

/// `tools/list` must return every tool, in a stable order, with the cache hints
/// `2026-07-28` made mandatory.
///
/// The order assertion is not pedantry. SEP-2549 asks for a deterministic order precisely
/// so clients can cache the list and LLM prompt caches keep hitting; a reordering silently
/// invalidates every client's cached prefix, which is exactly the kind of regression no
/// one notices without a test.
#[test]
fn tools_list_is_complete_ordered_and_cacheable() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = mcp(
        &server,
        "tools/list",
        None,
        &rpc(1, "tools/list", json!({})),
    );
    assert_eq!(status, 200, "tools/list failed: {body}");
    let result = result(&body);

    assert_eq!(
        tool_names(&result),
        vec![
            "remember",
            "recall",
            "text_search",
            "hybrid_search",
            "list_collections",
            "stats",
        ],
        "tool list changed — if this is deliberate, append rather than reorder"
    );

    // Required by SEP-2549 on every list result.
    assert!(
        result["ttlMs"].as_u64().is_some_and(|t| t > 0),
        "ttlMs must be present and positive: {result}"
    );
    assert_eq!(
        result["cacheScope"], "public",
        "the tool list is caller-independent, so it is publicly cacheable: {result}"
    );

    // Every tool needs a description and an object schema: a tool a model cannot
    // understand is worse than one that is absent, because it will still get called.
    for tool in result["tools"].as_array().expect("tools array") {
        let name = tool["name"].as_str().unwrap_or("<unnamed>");
        assert!(
            tool["description"].as_str().is_some_and(|d| d.len() > 40),
            "tool `{name}` needs a substantive description: {tool}"
        );
        assert_eq!(
            tool["inputSchema"]["type"], "object",
            "tool `{name}` schema must be an object: {tool}"
        );
    }

    // The raw-vector routes must NOT have leaked onto this surface: no model can emit a
    // 384-float array as a tool argument, so a `vector` parameter is always a mistake.
    let rendered = result.to_string();
    assert!(
        !rendered.contains("\"vector\""),
        "no MCP tool may take a raw vector argument: {rendered}"
    );
}

/// The adapter reads the *same store* the HTTP routes write.
///
/// Written over HTTP with raw vectors (which needs no embedder) and read back over MCP —
/// so this also pins the claim that the MCP layer is an adapter over the shared
/// `run_read`/`run_write` plumbing rather than a second engine with its own view.
#[test]
fn tools_read_the_same_store_the_http_routes_write() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    assert_eq!(server.post("/collections/notes", &json!({})).0, 200);
    let (status, _) = server.post(
        "/collections/notes/upsert",
        &json!({"records": [
            {"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "ranking bug in upsert"}}},
            {"id": "b", "vector": [0, 1, 0], "attrs": {"body": {"Str": "unrelated note"}}}
        ]}),
    );
    assert_eq!(status, 200);

    // list_collections sees the collection created over HTTP.
    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("list_collections"),
        &call(1, "list_collections", json!({})),
    );
    assert_eq!(status, 200, "list_collections failed: {body}");
    assert!(
        text(&result(&body)).contains("notes"),
        "list_collections should see the HTTP-created collection: {body}"
    );

    // stats reports the store the HTTP routes populated.
    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("stats"),
        &call(2, "stats", json!({})),
    );
    assert_eq!(status, 200, "stats failed: {body}");
    let stats = text(&result(&body));
    assert!(
        stats.contains("\"dimension\": 3"),
        "stats should report the store dimension: {stats}"
    );

    // text_search is the one search tool needing no embedder, so it round-trips for real.
    assert_eq!(
        server
            .post(
                "/collections/notes/fts-schema",
                &json!({"fields": ["body"]})
            )
            .0,
        200
    );
    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("text_search"),
        &call(
            3,
            "text_search",
            json!({"collection": "notes", "field": "body", "query": "ranking"}),
        ),
    );
    assert_eq!(status, 200, "text_search failed: {body}");
    let hits = text(&result(&body));
    assert!(
        hits.contains("\"a\""),
        "text_search should find the matching record: {hits}"
    );
    assert!(
        !hits.contains("\"b\""),
        "text_search should not return the non-matching record: {hits}"
    );
}

/// An empty result must say so in words rather than returning `[]`.
///
/// A model handed a bare empty array frequently retries the identical query; told plainly
/// that nothing matched, it broadens or moves on. This is a behavioural contract with the
/// caller, not cosmetics.
#[test]
fn empty_results_are_stated_in_words() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("list_collections"),
        &call(1, "list_collections", json!({})),
    );
    assert_eq!(status, 200, "list_collections failed: {body}");
    let said = text(&result(&body));
    assert!(
        said.contains("no collections"),
        "an empty store should explain itself: {said}"
    );
}

/// A tool needing an embedder on a server started without one must fail as a *server*
/// problem, not a caller problem.
///
/// This is the regression guard for the error mapping. `invalid_params` here would tell the
/// agent its arguments were wrong, and a capable agent responds to that by rephrasing and
/// retrying — forever, because no phrasing can conjure an embedder. The message has to name
/// the fix an operator can actually apply.
#[test]
fn memory_tools_without_an_embedder_fail_as_a_server_fault() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("remember"),
        &call(
            1,
            "remember",
            json!({"collection": "notes", "text": "something worth keeping"}),
        ),
    );
    // `200` on purpose, and the counterpart to the `400` a caller fault gets: rmcp maps
    // only `invalid params` / version / capability errors onto HTTP status. A server fault
    // is a well-formed request that the server could not honour, so it stays `200` with the
    // error in the JSON-RPC envelope.
    assert_eq!(
        status, 200,
        "a server fault stays HTTP 200 with the error in the envelope: {body}"
    );

    let code = body["error"]["code"].as_i64();
    assert_eq!(
        code,
        Some(-32603),
        "a missing embedder is an internal error, not invalid params: {body}"
    );
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("--embed-provider"),
        "the error must name the flag that fixes it: {message}"
    );
}

/// An unknown tool is `invalid_params` (`-32602`) — a caller fault, and one a retry can fix.
#[test]
fn unknown_tool_is_a_caller_fault() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("teleport"),
        &call(1, "teleport", json!({})),
    );
    // `400`, not `200`: this revision maps `invalid params` onto a malformed *HTTP*
    // request, so a caller fault is visible to a gateway without parsing the body. Server
    // faults keep `200` and carry the error in the envelope — see the embedder test.
    assert_eq!(
        status, 400,
        "a caller fault should surface as HTTP 400: {body}"
    );
    assert_eq!(
        body["error"]["code"].as_i64(),
        Some(-32602),
        "unknown tool should be invalid params: {body}"
    );
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("teleport"),
        "the error should name the tool that does not exist: {body}"
    );
}

/// Missing a required argument is `invalid_params`, and the message names the argument.
#[test]
fn missing_required_argument_names_the_argument() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("text_search"),
        // `field` and `query` are required by the schema and absent here.
        &call(1, "text_search", json!({"collection": "notes"})),
    );
    assert_eq!(status, 400, "a missing argument is a caller fault: {body}");
    assert_eq!(body["error"]["code"].as_i64(), Some(-32602), "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("field") || message.contains("query"),
        "the error should name the missing argument: {message}"
    );
}

/// `Mcp-Name` disagreeing with the body's tool name is `HeaderMismatch` (`-32020`).
///
/// The header exists so gateways can route and authorize without parsing the body; if the
/// two could disagree silently, a gateway's decision would be about a different call than
/// the one the server runs. `2026-07-28` renumbered this code from `-32001` into the
/// MCP-reserved range, so the value is also a version marker.
#[test]
fn header_body_mismatch_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    // Body says `stats`; the header claims `remember`.
    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("remember"),
        &call(1, "stats", json!({})),
    );

    // Either a transport-level rejection or a JSON-RPC error carrying the code — both are
    // conformant, so accept either shape and assert only that it did not silently succeed.
    if status == 200 {
        assert_eq!(
            body["error"]["code"].as_i64(),
            Some(-32020),
            "a header/body mismatch must be rejected, not executed: {body}"
        );
    } else {
        assert!(
            (400..500).contains(&status),
            "expected a client-error status for a header/body mismatch, got {status}: {body}"
        );
    }
}

/// The `Host` header must survive `Router::nest`.
///
/// rmcp's own comment (`streamable_http_server/tower.rs:856`) warns that `axum::Router::nest`
/// can drop the `Host` header hyper synthesizes, and rmcp validates `Host` by default as
/// DNS-rebinding protection. Nesting is exactly how nidus mounts this service, so a change
/// in either crate that reintroduces the drop would break every MCP request — with a
/// failure that looks nothing like its cause. This pins it.
#[test]
fn host_header_survives_nesting() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    // An explicit, non-loopback Host: the value most likely to be rejected if validation
    // is active and most likely to reveal a dropped header if it is not.
    let mut h = headers("tools/list", None);
    h.push(("host", "nidus.example.com"));
    let (status, body) = server.post_with_headers("/mcp", &rpc(1, "tools/list", json!({})), &h);

    assert_eq!(
        status, 200,
        "a nested MCP service must accept a rewritten Host (an ingress will set one): {body}"
    );
    assert!(
        !tool_names(&result(&body)).is_empty(),
        "tools/list should answer normally through a rewritten Host: {body}"
    );
}

/// `/mcp` inherits the server's bearer-token auth.
///
/// The whole design rests on nesting the service *inside* the middleware stack rather than
/// beside it, so the endpoint gets the body limit, backpressure, metrics, and — most
/// importantly — the token check without reimplementing any of them. If this ever fails,
/// `/mcp` is an unauthenticated hole in an otherwise guarded server, which is the worst
/// possible way for a refactor to go wrong.
#[test]
fn mcp_requires_the_bearer_token() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).token("sekrit").start();

    // The harness attaches the token, so the authorized path proves the endpoint works.
    let (status, body) = mcp(
        &server,
        "tools/list",
        None,
        &rpc(1, "tools/list", json!({})),
    );
    assert_eq!(status, 200, "authorized MCP request should succeed: {body}");

    // Now the same request with no credential, using a bare agent so the harness cannot
    // helpfully add one back.
    let url = format!("{}/mcp", server.base_url());
    let sent = serde_json::to_vec(&rpc(2, "tools/list", json!({}))).unwrap();
    let res = ureq::Agent::new_with_defaults()
        .post(&url)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream")
        .header("mcp-protocol-version", VERSION)
        .header("mcp-method", "tools/list")
        .send(&sent);

    let status = match res {
        Ok(r) => r.status().as_u16(),
        Err(ureq::Error::StatusCode(code)) => code,
        Err(e) => panic!("unauthenticated MCP request: {e}"),
    };
    assert_eq!(
        status, 401,
        "an unauthenticated /mcp request must be rejected like any other route"
    );
}
