//! E2E tests for the MCP prompts surface (nidus-91): `recall_then_answer` runs a real
//! recall server-side. Needs an embedder, so this module compiles away outside
//! `embed-ollama` (`just ci-serve`), like `lifecycle.rs`.

#![cfg(feature = "embed-ollama")]

use serde_json::{Value, json};

use crate::harness::RunningServer;

use super::support::{DIM, per_text_embedder_server};
use super::{call, mcp, result, rpc};

/// `remember` a text, asserting the write succeeded.
fn remember(server: &RunningServer, args: Value) {
    let (status, body) = mcp(
        server,
        "tools/call",
        Some("remember"),
        &call(1, "remember", args),
    );
    assert_eq!(status, 200, "remember failed: {body}");
}

/// `prompts/get` for `name`, with the mandatory `Mcp-Name` header set to `name` itself.
fn get_prompt(server: &RunningServer, name: &str, arguments: Value) -> (u16, Value) {
    mcp(
        server,
        "prompts/get",
        Some(name),
        &rpc(
            2,
            "prompts/get",
            json!({ "name": name, "arguments": arguments }),
        ),
    )
}

/// The text of the first rendered message.
fn message_text(result: &Value) -> String {
    result["messages"][0]["content"]["text"]
        .as_str()
        .expect("messages[0].content.text")
        .to_string()
}

#[test]
fn prompts_list_advertises_recall_then_answer() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    let (status, body) = mcp(
        &server,
        "prompts/list",
        None,
        &rpc(1, "prompts/list", json!({})),
    );
    assert_eq!(status, 200, "prompts/list failed: {body}");
    let listed = result(&body);

    assert!(
        listed["ttlMs"].as_u64().is_some_and(|t| t > 0),
        "ttlMs must be present and positive (SEP-2549): {listed}"
    );
    assert_eq!(listed["cacheScope"], "public", "{listed}");

    let prompts = listed["prompts"].as_array().expect("prompts array");
    let prompt = prompts
        .iter()
        .find(|p| p["name"] == "recall_then_answer")
        .unwrap_or_else(|| panic!("recall_then_answer must be advertised: {listed}"));
    assert!(
        prompt["description"].as_str().is_some_and(|d| d.len() > 40),
        "the prompt needs a substantive description: {prompt}"
    );

    let args = prompt["arguments"].as_array().expect("arguments array");
    let names: Vec<&str> = args.iter().filter_map(|a| a["name"].as_str()).collect();
    assert_eq!(
        names,
        vec!["question", "collection", "top_k"],
        "argument set/order changed: {args:?}"
    );
    for (name, required) in [("question", true), ("collection", true), ("top_k", false)] {
        let arg = args
            .iter()
            .find(|a| a["name"] == name)
            .expect("argument present");
        assert_eq!(
            arg["required"].as_bool().unwrap_or(false),
            required,
            "`{name}` required-ness changed: {arg}"
        );
    }
}

/// The acceptance criterion: the returned message must actually contain the recalled
/// memory's text, not a static template — proof the recall really ran server-side.
#[test]
fn the_prompt_comes_back_with_the_recalled_memories_already_in_it() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"collection": "notes", "text": "the ranking bug is in the upsert path", "id": "bug"}),
    );
    remember(
        &server,
        json!({"collection": "notes", "text": "a completely unrelated grocery list", "id": "groceries"}),
    );

    // `top_k: 1` is load-bearing: `recall` returns the top k ranked hits, not only the
    // matches, so at the default k both memories come back and "the unrelated one is
    // absent" would assert nothing about ranking.
    let (status, body) = get_prompt(
        &server,
        "recall_then_answer",
        json!({"question": "the ranking bug is in the upsert path", "collection": "notes", "top_k": 1}),
    );
    assert_eq!(status, 200, "prompts/get failed: {body}");
    let rendered = message_text(&result(&body));
    assert!(
        rendered.contains("ranking bug"),
        "the matching memory's text must appear: {rendered}"
    );
    assert!(
        !rendered.contains("grocery"),
        "the better-matching memory must win the single slot, and `top_k` must be \
         honoured: {rendered}"
    );
    // Against the scaffold, not the bare question text: the question here is word-for-word
    // the matching memory, so a bare `contains` would pass on the memory alone.
    assert!(
        rendered.contains("Question: the ranking bug is in the upsert path"),
        "the question must be interpolated into the prompt: {rendered}"
    );
}

#[test]
fn the_prompt_carries_no_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"collection": "notes", "text": "the ranking bug is in the upsert path", "id": "bug"}),
    );

    let (status, body) = get_prompt(
        &server,
        "recall_then_answer",
        json!({"question": "ranking bug", "collection": "notes"}),
    );
    assert_eq!(status, 200, "{body}");
    let rendered = message_text(&result(&body));
    assert!(
        !rendered.contains("\"vector\""),
        "the handler embedded the question, but the rendered prompt must never carry a vector: {rendered}"
    );
}

/// The TTL guard must reach the prompt path too, not only the tools.
#[test]
fn an_expired_memory_is_not_recalled_into_the_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({
            "collection": "notes",
            "text": "ephemeral scratch note",
            "id": "gone",
            "ttl_seconds": 0
        }),
    );
    remember(
        &server,
        json!({"collection": "notes", "text": "durable note", "id": "kept"}),
    );

    let (status, body) = get_prompt(
        &server,
        "recall_then_answer",
        json!({"question": "ephemeral scratch note", "collection": "notes"}),
    );
    assert_eq!(status, 200, "{body}");
    let rendered = message_text(&result(&body));
    assert!(
        !rendered.contains("\"gone\""),
        "an expired memory must not be recalled into the prompt: {rendered}"
    );
}

#[test]
fn a_missing_required_argument_is_a_caller_fault() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    let (status, body) = get_prompt(
        &server,
        "recall_then_answer",
        json!({"collection": "notes"}),
    );
    assert_eq!(status, 400, "a missing argument is a caller fault: {body}");
    assert_eq!(body["error"]["code"].as_i64(), Some(-32602), "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("question"),
        "the error should name the missing argument: {body}"
    );
}

#[test]
fn an_unknown_prompt_name_is_a_caller_fault() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    let (status, body) = get_prompt(&server, "teleport", json!({}));
    assert_eq!(
        status, 400,
        "an unknown prompt should be a caller fault: {body}"
    );
    assert_eq!(body["error"]["code"].as_i64(), Some(-32602), "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("teleport"),
        "the error should name the prompt that does not exist: {body}"
    );
}

/// `PromptArgument` carries no type on the wire, so a client filling in a template cannot
/// know `top_k` is a number and will often send `"1"`. Rejecting that rejects a compliant
/// caller, so both forms must work and must mean the same thing.
#[test]
fn top_k_is_accepted_as_a_string_the_way_a_prompt_client_sends_it() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    remember(
        &server,
        json!({"collection": "notes", "text": "the ranking bug is in the upsert path", "id": "bug"}),
    );
    remember(
        &server,
        json!({"collection": "notes", "text": "a completely unrelated grocery list", "id": "groceries"}),
    );

    let ask = |top_k: Value| {
        let (status, body) = get_prompt(
            &server,
            "recall_then_answer",
            json!({
                "question": "the ranking bug is in the upsert path",
                "collection": "notes",
                "top_k": top_k,
            }),
        );
        assert_eq!(status, 200, "prompts/get failed for {top_k}: {body}");
        message_text(&result(&body))
    };

    let from_number = ask(json!(1));
    let from_string = ask(json!("1"));
    assert!(
        !from_string.contains("grocery"),
        "a string `top_k` must bound the recall exactly as the number does: {from_string}"
    );
    assert_eq!(
        from_number, from_string,
        "`1` and \"1\" must produce the same prompt"
    );
}

/// A `top_k` that is neither a number nor a numeric string is still a caller fault, so the
/// leniency above does not become "accept anything and silently default".
#[test]
fn a_non_numeric_top_k_is_still_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let server = per_text_embedder_server(dir.path(), DIM);

    let (status, body) = get_prompt(
        &server,
        "recall_then_answer",
        json!({"question": "anything", "collection": "notes", "top_k": "lots"}),
    );
    assert_eq!(status, 400, "a bad argument is a caller fault: {body}");
    assert_eq!(body["error"]["code"].as_i64(), Some(-32602), "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("top_k"),
        "the error should name the argument: {body}"
    );
}
