//! E2E tests for the MCP resources surface (nidus-91): collections listed concretely,
//! entries reached by the `nidus://` template. No embedder needed — reads go through
//! `db.list`/`db.get` exactly like `browse`/`get`, so this module is ungated.

use serde_json::{Value, json};

use crate::harness::{RunningServer, Server};

use super::{mcp, result, rpc};

/// Percent-encode a URL path segment (the HTTP routes take one raw segment per
/// `{name}`, so a collection name containing `/` must be encoded to stay one segment).
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Create a collection and seed it with `records` (raw vectors, no embedder needed).
fn seed(server: &RunningServer, collection: &str, records: Value) {
    let path = url_encode(collection);
    assert_eq!(
        server.post(&format!("/collections/{path}"), &json!({})).0,
        200
    );
    let (status, body) = server.post(
        &format!("/collections/{path}/upsert"),
        &json!({ "records": records }),
    );
    assert_eq!(status, 200, "seed upsert into {collection} failed: {body}");
}

/// Milliseconds since the epoch, `offset_ms` away from now (negative = in the past).
fn epoch_ms(offset_ms: i64) -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
        + offset_ms
}

/// `resources/list`, unwrapped to its `resources` array.
fn list_resources(server: &RunningServer) -> Value {
    let (status, body) = mcp(
        server,
        "resources/list",
        None,
        &rpc(1, "resources/list", json!({})),
    );
    assert_eq!(status, 200, "resources/list failed: {body}");
    result(&body)
}

/// `resources/read` for `uri`, with the mandatory `Mcp-Name` header set to `uri` itself.
fn read_resource(server: &RunningServer, uri: &str) -> (u16, Value) {
    mcp(
        server,
        "resources/read",
        Some(uri),
        &rpc(2, "resources/read", json!({ "uri": uri })),
    )
}

#[test]
fn resources_list_names_every_collection() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(
        &server,
        "notes",
        json!([{"id": "a", "vector": [1, 0, 0], "attrs": {}}]),
    );
    seed(
        &server,
        "todos",
        json!([{"id": "b", "vector": [0, 1, 0], "attrs": {}}]),
    );

    let listed = list_resources(&server);
    assert!(
        listed["ttlMs"].as_u64().is_some_and(|t| t > 0),
        "ttlMs must be present and positive (SEP-2549): {listed}"
    );
    assert_eq!(
        listed["cacheScope"], "public",
        "the resource list is caller-independent, so it is publicly cacheable: {listed}"
    );

    let resources = listed["resources"].as_array().expect("resources array");
    let uris: Vec<&str> = resources.iter().filter_map(|r| r["uri"].as_str()).collect();
    assert!(
        uris.contains(&"nidus://collections/notes"),
        "notes must be listed by its exact URI: {listed}"
    );
    assert!(
        uris.contains(&"nidus://collections/todos"),
        "todos must be listed by its exact URI: {listed}"
    );
    for r in resources {
        assert!(
            r["description"].as_str().is_some_and(|d| !d.is_empty()),
            "every listed resource needs a non-empty description: {r}"
        );
        assert_eq!(
            r["mimeType"], "application/json",
            "a collection resource is JSON, never raw text: {r}"
        );
    }
}

#[test]
fn resource_templates_advertise_the_entry_uri() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = mcp(
        &server,
        "resources/templates/list",
        None,
        &rpc(1, "resources/templates/list", json!({})),
    );
    assert_eq!(status, 200, "resources/templates/list failed: {body}");
    let listed = result(&body);
    let templates = listed["resourceTemplates"]
        .as_array()
        .expect("resourceTemplates array");
    let uri_templates: Vec<&str> = templates
        .iter()
        .filter_map(|t| t["uriTemplate"].as_str())
        .collect();
    assert!(
        uri_templates.contains(&"nidus://collections/{collection}/entries/{id}"),
        "a client builds entry URIs from this exact string: {listed}"
    );
}

#[test]
fn reading_a_collection_returns_its_entries_and_no_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(
        &server,
        "notes",
        json!([
            {"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "first"}}},
            {"id": "b", "vector": [0, 1, 0], "attrs": {"body": {"Str": "second"}}}
        ]),
    );

    let (status, body) = read_resource(&server, "nidus://collections/notes");
    assert_eq!(status, 200, "read of a collection failed: {body}");
    let contents = result(&body)["contents"].clone();
    let entries: Value =
        serde_json::from_str(contents[0]["text"].as_str().expect("contents[0].text"))
            .expect("entries JSON");
    let ids: Vec<&str> = entries["entries"]
        .as_array()
        .expect("entries array")
        .iter()
        .filter_map(|e| e["id"].as_str())
        .collect();
    assert!(ids.contains(&"a") && ids.contains(&"b"), "{entries}");
    assert_eq!(
        entries["truncated"], false,
        "two entries is not a truncated page: {entries}"
    );
    let rendered = entries.to_string();
    assert!(
        rendered.contains("first") && rendered.contains("second"),
        "{rendered}"
    );
    assert!(
        !rendered.contains("\"vector\""),
        "a resource must never carry a vector: {rendered}"
    );
}

#[test]
fn reading_an_entry_returns_that_entry_only() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(
        &server,
        "notes",
        json!([
            {"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "first"}}},
            {"id": "b", "vector": [0, 1, 0], "attrs": {"body": {"Str": "second"}}}
        ]),
    );

    let (status, body) = read_resource(&server, "nidus://collections/notes/entries/a");
    assert_eq!(status, 200, "read of an entry failed: {body}");
    let contents = result(&body)["contents"].clone();
    let rendered = contents[0]["text"].as_str().expect("contents[0].text");
    assert!(rendered.contains("\"id\": \"a\""), "{rendered}");
    assert!(rendered.contains("first"), "{rendered}");
    assert!(
        !rendered.contains("\"b\""),
        "the other record must be absent: {rendered}"
    );
    assert!(
        !rendered.contains("\"vector\""),
        "an entry must never carry a vector: {rendered}"
    );
}

/// The test that earns the whole codec: a collection named `notes/2026` and an id containing
/// a space, driven through a real request rather than the URI encoder's own unit tests.
#[test]
fn a_percent_encoded_name_round_trips_through_a_real_request() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(
        &server,
        "notes/2026",
        json!([{"id": "an entry", "vector": [1, 0, 0], "attrs": {"body": {"Str": "hi"}}}]),
    );

    let listed = list_resources(&server);
    let uri = listed["resources"]
        .as_array()
        .expect("resources array")
        .iter()
        .find(|r| r["name"] == "notes/2026")
        .unwrap_or_else(|| panic!("notes/2026 must be listed: {listed}"))["uri"]
        .as_str()
        .expect("uri string")
        .to_string();
    assert!(
        uri.is_ascii(),
        "the advertised URI must be pure ASCII: {uri}"
    );
    assert!(
        !uri.contains(' '),
        "the advertised URI must contain no space: {uri}"
    );

    let entry_uri = format!("{uri}/entries/{}", url_encode("an entry"));
    let (status, body) = read_resource(&server, &entry_uri);
    assert_eq!(
        status, 200,
        "read of the percent-encoded entry failed: {body}"
    );
    let rendered = result(&body)["contents"][0]["text"]
        .as_str()
        .expect("contents[0].text")
        .to_string();
    assert!(rendered.contains("\"id\": \"an entry\""), "{rendered}");
}

#[test]
fn an_unknown_or_malformed_uri_is_a_caller_fault() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(
        &server,
        "notes",
        json!([{"id": "a", "vector": [1, 0, 0], "attrs": {}}]),
    );

    for uri in [
        "file:///etc/passwd",
        "nidus://collections",
        "nidus://collections/notes/entries/nope",
    ] {
        let (status, body) = read_resource(&server, uri);
        assert_eq!(status, 400, "`{uri}` should be a caller fault: {body}");
        assert_eq!(
            body["error"]["code"].as_i64(),
            Some(-32602),
            "`{uri}`: {body}"
        );
    }

    // The two malformed ones (not the well-formed-but-missing entry) must name a template so
    // a model can self-correct.
    for uri in ["file:///etc/passwd", "nidus://collections"] {
        let (_, body) = read_resource(&server, uri);
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("nidus://collections/{collection}")
                || message.contains("nidus://collections/{collection}/entries/{id}"),
            "`{uri}` error should name a template: {message}"
        );
    }
}

#[test]
fn an_expired_entry_is_unreadable_through_its_uri_and_its_collection() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(
        &server,
        "notes",
        json!([
            {
                "id": "gone",
                "vector": [1, 0, 0],
                "attrs": {"nidus.expires_at": {"DateTime": epoch_ms(-60_000)}}
            },
            {"id": "kept", "vector": [0, 1, 0], "attrs": {}}
        ]),
    );

    let (status, body) = read_resource(&server, "nidus://collections/notes");
    assert_eq!(status, 200, "{body}");
    let rendered = result(&body)["contents"][0]["text"]
        .as_str()
        .expect("contents[0].text")
        .to_string();
    assert!(
        !rendered.contains("\"gone\""),
        "an expired entry must not surface in its collection: {rendered}"
    );
    assert!(rendered.contains("\"kept\""), "{rendered}");

    let (status, body) = read_resource(&server, "nidus://collections/notes/entries/gone");
    assert_eq!(
        status, 400,
        "reading an expired entry directly must fail: {body}"
    );
    assert_eq!(body["error"]["code"].as_i64(), Some(-32602), "{body}");
}

#[test]
fn mcp_name_must_match_the_uri() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    seed(
        &server,
        "notes",
        json!([{"id": "a", "vector": [1, 0, 0], "attrs": {}}]),
    );

    // Body asks for `notes`; the header claims a different URI entirely.
    let (status, body) = mcp(
        &server,
        "resources/read",
        Some("nidus://collections/other"),
        &rpc(
            1,
            "resources/read",
            json!({ "uri": "nidus://collections/notes" }),
        ),
    );

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

/// A collection larger than one page must stay parseable as the `application/json` it is
/// advertised as. The truncation signal rides inside the JSON; appending prose to the body
/// would break exactly the large collections the notice exists for.
#[test]
fn a_truncated_collection_page_is_still_valid_json_and_says_it_is_truncated() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    // Comfortably past the default page size, so this does not silently stop testing
    // truncation if that default is raised a little.
    let records: Vec<Value> = (0..40)
        .map(|i| json!({"id": format!("r{i}"), "vector": [1, 0, 0], "attrs": {}}))
        .collect();
    seed(&server, "big", json!(records));

    let (status, body) = read_resource(&server, "nidus://collections/big");
    assert_eq!(status, 200, "read of a large collection failed: {body}");
    let contents = result(&body)["contents"].clone();
    assert_eq!(
        contents[0]["mimeType"], "application/json",
        "the collection read advertises JSON: {contents}"
    );
    let text = contents[0]["text"].as_str().expect("contents[0].text");

    // The whole point: parse it as the mime type promises, notice and all.
    let parsed: Value = serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("a truncated page must still be valid JSON ({e}): {text}"));
    assert_eq!(
        parsed["truncated"], true,
        "a page short of the collection must say so: {parsed}"
    );
    let n = parsed["entries"].as_array().expect("entries array").len();
    assert!(
        n > 0 && n < 40,
        "expected a partial page, got {n}: {parsed}"
    );
    assert!(
        parsed["note"]
            .as_str()
            .is_some_and(|s| s.contains("browse")),
        "the note should point at the tool that pages further: {parsed}"
    );
}

/// A collection holding exactly one page's worth is NOT truncated. Detecting truncation as
/// `len() == limit` gets this wrong and sends a client paging for entries that do not exist.
#[test]
fn a_collection_of_exactly_one_page_is_not_reported_as_truncated() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    // Learn the page size from the server rather than hard-coding it: seed one page's worth
    // exactly, which is the boundary this test exists for.
    let probe: Vec<Value> = (0..200)
        .map(|i| json!({"id": format!("p{i}"), "vector": [1, 0, 0], "attrs": {}}))
        .collect();
    seed(&server, "probe", json!(probe));
    let (_, body) = read_resource(&server, "nidus://collections/probe");
    let text = result(&body)["contents"][0]["text"]
        .as_str()
        .expect("text")
        .to_string();
    let parsed: Value = serde_json::from_str(&text).expect("probe JSON");
    let page = parsed["entries"].as_array().expect("entries").len();
    assert!(
        parsed["truncated"] == true,
        "probe should truncate: {parsed}"
    );

    let exact: Vec<Value> = (0..page)
        .map(|i| json!({"id": format!("e{i}"), "vector": [1, 0, 0], "attrs": {}}))
        .collect();
    seed(&server, "exact", json!(exact));
    let (status, body) = read_resource(&server, "nidus://collections/exact");
    assert_eq!(status, 200, "read failed: {body}");
    let parsed: Value =
        serde_json::from_str(result(&body)["contents"][0]["text"].as_str().expect("text"))
            .expect("exact JSON");
    assert_eq!(
        parsed["entries"].as_array().expect("entries").len(),
        page,
        "the whole collection should be on the page: {parsed}"
    );
    assert_eq!(
        parsed["truncated"], false,
        "a collection of exactly one page has nothing further to page to: {parsed}"
    );
    assert!(
        parsed["note"].is_null(),
        "no paging note when there is nothing to page to: {parsed}"
    );
}
