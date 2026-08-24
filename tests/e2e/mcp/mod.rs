//! E2E tests for the MCP `2026-07-28` surface at `/mcp` (nidus-zm2.3). A nested `tower`
//! service carrying part of its protocol in headers, so an in-process `oneshot` exercises
//! neither. This file stays HTTP-only; `stdio`/`attrs`/`filters`/`hygiene` are siblings.

mod aliases;
mod attrs;
mod dedupe;
mod filters;
mod hygiene;
mod lifecycle;
mod prompts;
mod recency;
mod related;
mod resources;
mod stdio;
pub(super) mod support;

use serde_json::{Value, json};

use crate::harness::{RunningServer, Server};

/// The protocol version these tests speak.
pub(super) const VERSION: &str = "2026-07-28";

/// Headers every MCP request needs. `Accept` names both types because Streamable HTTP may
/// answer either — JSON here, or SSE if a handler ever emits a notification first.
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

/// A JSON-RPC envelope with the `_meta` this revision requires per request. With no
/// handshake, version and identity ride on every call, so this makes them hard to forget.
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

/// The `result` of a successful JSON-RPC response, or a panic naming the error. Shared with
/// `stdio` — the envelope shape is transport-agnostic.
pub(super) fn result(body: &Value) -> Value {
    assert!(
        body.get("error").is_none(),
        "expected a result, got JSON-RPC error: {body}"
    );
    body["result"].clone()
}

/// The concatenated text of a `tools/call` result's content blocks. Shared with `stdio`.
pub(super) fn text(result: &Value) -> String {
    result["content"]
        .as_array()
        .expect("content array")
        .iter()
        .filter_map(|b| b["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The tool list, in the order the server returned it. Shared with `stdio`.
pub(super) fn tool_names(result: &Value) -> Vec<String> {
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

    // This revision moved server identity into the result's `_meta` (SEP-2575).
    let info = &result["_meta"]["io.modelcontextprotocol/serverInfo"];
    // Must be THIS crate: `Implementation::from_build_env()` reports `rmcp` — a well-formed
    // answer naming the wrong software.
    assert_eq!(
        info["name"], "nidus",
        "serverInfo must name nidus, not the SDK: {result}"
    );
    assert_eq!(
        info["version"],
        env!("CARGO_PKG_VERSION"),
        "serverInfo version should track the crate version: {result}"
    );

    // Why this takes rmcp rather than hand-rolling: most deployed clients predate this
    // revision, and a server advertising only the newest version would reach almost none.
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

    // Tools, resources, and prompts are all implemented now; logging has no implementation.
    for present in ["tools", "resources", "prompts"] {
        assert!(
            result["capabilities"][present].is_object(),
            "{present} capability should be advertised: {result}"
        );
    }
    assert!(
        result["capabilities"]["logging"].is_null(),
        "logging must not be advertised — nothing implements it: {result}"
    );

    // Advertising a notification nothing ever sends is worse than not offering it: every
    // nidus op is one fast synchronous call, so there is nothing to subscribe to and
    // nothing whose list ever changes out from under a cached client.
    assert_ne!(
        result["capabilities"]["resources"]["subscribe"], true,
        "resources/subscribe must not be advertised: {result}"
    );
    assert_ne!(
        result["capabilities"]["resources"]["listChanged"], true,
        "resources listChanged must not be advertised: {result}"
    );
    assert_ne!(
        result["capabilities"]["prompts"]["listChanged"], true,
        "prompts listChanged must not be advertised: {result}"
    );
}

/// `tools/list` returns every tool, in a stable order, with the mandatory cache hints.
/// Order matters: reordering silently invalidates every client's cached prompt prefix.
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
            "forget",
            "get",
            "browse",
            "related",
            "suggest",
            "list_aliases",
            "set_alias",
            "drop_alias",
        ],
        "tool list changed — if this is deliberate, append rather than reorder. The alias \
         tools must stay last so `related`'s position never shifts (SEP-2549)."
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

    // A tool a model cannot understand is worse than an absent one: it still gets called.
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

    // No model can emit a 384-float array, so a `vector` parameter is always a mistake.
    let rendered = result.to_string();
    assert!(
        !rendered.contains("\"vector\""),
        "no MCP tool may take a raw vector argument: {rendered}"
    );

    // Only the two BM25-clause tools take a typeahead `prefix` boolean (nidus-p1n).
    let tools = result["tools"].as_array().expect("tools array");
    for want in ["text_search", "hybrid_search"] {
        let tool = tools
            .iter()
            .find(|t| t["name"] == want)
            .unwrap_or_else(|| panic!("no `{want}` tool"));
        let prefix = &tool["inputSchema"]["properties"]["prefix"];
        assert_eq!(
            prefix["type"], "boolean",
            "`{want}` should advertise a boolean `prefix`: {tool}"
        );
        assert!(
            prefix["description"].as_str().is_some_and(|d| d.len() > 20),
            "`{want}`'s `prefix` needs a hand-written description: {tool}"
        );
    }
    let recall = tools
        .iter()
        .find(|t| t["name"] == "recall")
        .expect("no `recall` tool");
    assert!(
        recall["inputSchema"]["properties"]["prefix"].is_null(),
        "recall has no BM25 clause, so it must not advertise `prefix`: {recall}"
    );
}

/// Walk a schema and collect every `$ref` fragment and every `$defs` key path it declares.
fn refs_and_defs(node: &Value, at_root: bool, refs: &mut Vec<String>, defs: &mut Vec<String>) {
    match node {
        Value::Object(map) => {
            for (key, value) in map {
                if key == "$ref"
                    && let Some(r) = value.as_str()
                {
                    refs.push(r.to_string());
                } else if key == "$defs"
                    && at_root
                    && let Some(d) = value.as_object()
                {
                    defs.extend(d.keys().map(|k| format!("#/$defs/{k}")));
                }
                refs_and_defs(value, false, refs, defs);
            }
        }
        Value::Array(items) => {
            for item in items {
                refs_and_defs(item, false, refs, defs);
            }
        }
        _ => {}
    }
}

/// Every `$ref` a tool advertises must resolve against that tool's OWN schema root — a JSON
/// Pointer fragment is resolved from the document root, so `$defs` parked under a property
/// resolves nowhere and a strict client rejects the whole schema (nidus-k28.3).
#[test]
fn every_schema_ref_resolves_at_its_own_document_root() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = mcp(
        &server,
        "tools/list",
        None,
        &rpc(1, "tools/list", json!({})),
    );
    assert_eq!(status, 200, "tools/list failed: {body}");

    for tool in result(&body)["tools"].as_array().expect("tools array") {
        let name = tool["name"].as_str().unwrap_or("<unnamed>");
        let schema = &tool["inputSchema"];
        let (mut refs, mut defs) = (Vec::new(), Vec::new());
        refs_and_defs(schema, true, &mut refs, &mut defs);
        for r in &refs {
            assert!(
                defs.contains(r),
                "tool `{name}` emits `{r}` but its schema root declares only {defs:?} — \
                 a client resolving from the root cannot compile this: {schema}"
            );
        }
    }
}

/// The adapter reads the *same store* the HTTP routes write — written over HTTP with raw
/// vectors, read back over MCP, pinning that this is an adapter and not a second engine.
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

/// `text_search`'s `prefix` flag expands the final word as a typeahead match against the
/// indexed vocabulary; without it, a truncated word is not itself an indexed term and finds
/// nothing (nidus-p1n). Proves the flag actually reaches the store, not just the schema.
#[test]
fn text_search_prefix_matches_a_truncated_final_word() {
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
            {"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "banana bread"}}}
        ]}),
    );
    assert_eq!(status, 200);

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("text_search"),
        &call(
            1,
            "text_search",
            json!({"collection": "notes", "field": "body", "query": "ban", "prefix": true}),
        ),
    );
    assert_eq!(status, 200, "text_search failed: {body}");
    let hits = text(&result(&body));
    assert!(
        hits.contains("\"a\""),
        "prefix:true should match the truncated final word: {hits}"
    );

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("text_search"),
        &call(
            2,
            "text_search",
            json!({"collection": "notes", "field": "body", "query": "ban"}),
        ),
    );
    assert_eq!(status, 200, "text_search failed: {body}");
    let hits = text(&result(&body));
    assert!(
        !hits.contains("\"a\""),
        "without prefix, a truncated word must not match: {hits}"
    );
}

/// `suggest` ranks completions by live document frequency, commonest first — the opposite
/// of the idf a prefix *clause* would rank documents by (nidus-ux0). Asserting order and
/// counts, not just success, so a handler stubbed to an empty list would still fail this.
#[test]
fn suggest_tool_completes_a_prefix_ranked_by_document_frequency() {
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
            {"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "the banana bread"}}},
            {"id": "b", "vector": [0, 1, 0], "attrs": {"body": {"Str": "another banana muffin"}}},
            {"id": "c", "vector": [0, 0, 1], "attrs": {"body": {"Str": "bandit raid"}}}
        ]}),
    );
    assert_eq!(status, 200);

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("suggest"),
        &call(
            1,
            "suggest",
            json!({"collection": "notes", "field": "body", "prefix": "ban"}),
        ),
    );
    assert_eq!(status, 200, "suggest failed: {body}");
    let out = text(&result(&body));
    let banana_at = out.find("\"banana\"").expect("banana in response");
    let bandit_at = out.find("\"bandit\"").expect("bandit in response");
    assert!(
        banana_at < bandit_at,
        "the commonest term (df 2) must come before the rarer one (df 1): {out}"
    );
    assert!(
        out.contains("\"df\": 2") && out.contains("\"df\": 1"),
        "each completion's document frequency must be reported: {out}"
    );
}

/// nidus-3j8 over MCP: `filter` narrows each completion's `df`, and a completion no matching
/// entry carries is not offered at all. nidus-ucl comes with it, through `prefix` alone.
#[test]
fn suggest_tool_takes_a_filter_and_conditions_on_the_typed_phrase() {
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
            {"id": "a", "vector": [1, 0, 0],
             "attrs": {"body": {"Str": "quick banana"}, "kind": {"Str": "keep"}}},
            {"id": "b", "vector": [0, 1, 0],
             "attrs": {"body": {"Str": "bandit raid"}, "kind": {"Str": "drop"}}}
        ]}),
    );
    assert_eq!(status, 200);

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("suggest"),
        &call(
            1,
            "suggest",
            json!({
                "collection": "notes",
                "field": "body",
                "prefix": "ban",
                "filter": [{"Eq": ["kind", {"Str": "keep"}]}]
            }),
        ),
    );
    assert_eq!(status, 200, "suggest failed: {body}");
    let out = text(&result(&body));
    assert!(
        out.contains("\"banana\"") && !out.contains("\"bandit\""),
        "the filtered-out entry's only completion must be absent: {out}"
    );

    // The words before the fragment narrow it the same way, with no extra argument.
    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("suggest"),
        &call(
            2,
            "suggest",
            json!({"collection": "notes", "field": "body", "prefix": "quick ban"}),
        ),
    );
    assert_eq!(status, 200, "suggest failed: {body}");
    let out = text(&result(&body));
    assert!(
        out.contains("\"banana\"") && !out.contains("\"bandit\""),
        "bandit shares no entry with quick: {out}"
    );
}

/// A prefix matching no indexed term says so in a sentence, not `[]`, mirroring the same
/// empty-result rule the other search tools follow.
#[test]
fn suggest_tool_says_so_when_nothing_matches() {
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
            {"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "banana bread"}}}
        ]}),
    );
    assert_eq!(status, 200);

    let (status, body) = mcp(
        &server,
        "tools/call",
        Some("suggest"),
        &call(
            1,
            "suggest",
            json!({"collection": "notes", "field": "body", "prefix": "zzz"}),
        ),
    );
    assert_eq!(status, 200, "suggest failed: {body}");
    let said = text(&result(&body));
    assert!(
        said.contains("No indexed terms start with that prefix."),
        "an empty completion list should explain itself, not return `[]`: {said}"
    );
}

/// An empty result says so in words rather than returning `[]`: a model handed a bare empty
/// array tends to retry the identical query.
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

/// A tool needing an embedder on a server without one fails as a *server* problem.
/// `invalid_params` would have the agent rephrase and retry forever, since no phrasing
/// conjures an embedder — so the message must name the real fix.
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
    // `200` on purpose, unlike the `400` a caller fault gets: a server fault is a
    // well-formed request, so it stays `200` with the error in the envelope.
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

/// The prompt degrades exactly like `remember`/`recall`: needing an embedder on a server
/// without one is a server fault, not a caller mistake a retry can fix.
#[test]
fn prompts_without_an_embedder_fail_as_a_server_fault() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = mcp(
        &server,
        "prompts/get",
        Some("recall_then_answer"),
        &rpc(
            1,
            "prompts/get",
            json!({
                "name": "recall_then_answer",
                "arguments": {"question": "anything", "collection": "notes"}
            }),
        ),
    );
    assert_eq!(
        status, 200,
        "a server fault stays HTTP 200 with the error in the envelope: {body}"
    );
    assert_eq!(
        body["error"]["code"].as_i64(),
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
    // `400`, not `200`: `invalid params` maps onto a malformed HTTP request, so a gateway
    // sees a caller fault without parsing the body.
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

/// `Mcp-Name` disagreeing with the body's tool name is `HeaderMismatch` (`-32020`). Silent
/// disagreement would have a gateway authorizing a different call than the server runs.
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

    // Either shape is conformant, so assert only that it did not silently succeed.
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

/// The `Host` header must survive `Router::nest`. rmcp's own note (`tower.rs:856`) warns
/// nesting can drop the `Host` hyper synthesizes, and it validates `Host` by default — so a
/// regression breaks every MCP request with a failure that looks nothing like its cause.
#[test]
fn host_header_survives_nesting() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    // Non-loopback: the value most likely to be rejected if validation is somehow active.
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

/// `/mcp` inherits the server's bearer-token auth. The design rests on nesting inside the
/// middleware stack, so if this fails `/mcp` is a hole in an otherwise guarded server.
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
