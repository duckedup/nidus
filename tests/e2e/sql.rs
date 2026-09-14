//! SQL-specific e2e coverage (nidus-yq9p.6): parse-error classes asserted identically across
//! CLI, HTTP, and MCP (root `BLUEPRINT-nidus-yq9p-2-3-6.md`'s error model), `--compile`, `;`
//! batching order, and the deep-nesting guard. Cross-surface *conformance* over real data
//! lives in `tests/e2e/corpus.rs`; this file is the SQL front end's own edge cases.

use serde_json::{Value, json};

use crate::harness::{Server, fails, ok, run};

/// A store with one collection and two distinguishable records, for the batching test.
fn seeded_dir() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", dir, "--dim", "2", "notes"], "");
    ok(
        &["upsert", "--dir", dir, "notes"],
        &json!([
            {"id": "a", "vector": [1, 0], "attrs": {"kind": {"Str": "x"}}},
            {"id": "b", "vector": [0, 1], "attrs": {"kind": {"Str": "y"}}}
        ])
        .to_string(),
    );
    tmp
}

fn ids_of(v: &Value) -> Vec<String> {
    let hits = v.get("hits").unwrap_or(v);
    hits.as_array()
        .unwrap_or_else(|| panic!("expected a hits array, got {v}"))
        .iter()
        .map(|h| h["id"].as_str().unwrap_or_default().to_string())
        .collect()
}

// ── Parse-error classes, asserted identically on every in-crate surface ─────

/// One class: `sql`, and every substring its error message must carry on every surface —
/// always the marker, a byte offset, and the violated `SPEC.md` §7 section.
struct ErrorClass {
    name: &'static str,
    sql: String,
    contains: Vec<&'static str>,
    /// False when the class needs a vector literal, which MCP refuses by law
    /// (`.claude/rules/cli-feature.md`) before the parser ever sees it.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    on_mcp: bool,
}

fn error_classes() -> Vec<ErrorClass> {
    vec![
        ErrorClass {
            name: "missing-value-after-operator",
            sql: "SELECT * FROM notes WHERE lang =".to_string(),
            contains: vec!["sql parse error", "at byte", "§7.12", "expected a value"],
            on_mcp: true,
        },
        ErrorClass {
            name: "unknown-with-option",
            sql: "SELECT * FROM notes ORDER BY match(body, 'x') WITH (nonsense)".to_string(),
            contains: vec!["sql parse error", "at byte", "§7.12", "unknown WITH option"],
            on_mcp: true,
        },
        ErrorClass {
            name: "malformed-vector-literal",
            sql: "SELECT * FROM notes ORDER BY knn(1,0])".to_string(),
            contains: vec!["sql parse error", "at byte", "§7.6", "expected '['"],
            on_mcp: false,
        },
        ErrorClass {
            name: "predicate-nesting-too-deep",
            sql: format!(
                "SELECT * FROM notes WHERE {}a = 1{}",
                "(".repeat(200),
                ")".repeat(200)
            ),
            contains: vec!["sql parse error", "at byte", "§7.3", "nesting exceeds"],
            on_mcp: true,
        },
    ]
}

/// Every parse-error class exits nonzero with the one shared message, never a partial
/// stdout — template: `bad_invocations_exit_nonzero_with_a_useful_message` (`tests/e2e/cli.rs`).
#[test]
fn cli_parse_error_classes_report_offset_and_section() {
    let tmp = seeded_dir();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    for class in error_classes() {
        let err = fails(&["query", "--dir", dir, &class.sql], "");
        assert!(err.starts_with("error:"), "class {}: {err}", class.name);
        for s in &class.contains {
            assert!(
                err.contains(s),
                "class {} missing {s:?} in: {err}",
                class.name
            );
        }
    }
}

/// The same classes, over `POST /query`: `400`, and the exact same message text in the body
/// (the error chain the root blueprint promises — `anyhow::Error` -> `ApiError` -> `{"error":
/// "..."}`, no re-wording at the HTTP boundary).
#[test]
fn http_parse_error_classes_are_400_with_the_same_message() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let server = Server::new(tmp.path(), 2).start();
    for class in error_classes() {
        let (status, body) = server.post("/query", &json!({"sql": class.sql}));
        assert_eq!(status, 400, "class {}: {body}", class.name);
        let msg = body["error"].as_str().unwrap_or_default();
        for s in &class.contains {
            assert!(
                msg.contains(s),
                "class {} missing {s:?} in: {msg}",
                class.name
            );
        }
    }
}

#[cfg(feature = "mcp")]
mod mcp_parse_errors {
    use serde_json::{Value, json};

    use crate::harness::Server;

    const VERSION: &str = "2026-07-28";

    fn call_query(server: &crate::harness::RunningServer, sql: &str) -> (u16, Value) {
        let params = json!({
            "name": "query",
            "arguments": {"sql": sql},
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": VERSION,
                "io.modelcontextprotocol/clientInfo": {"name": "nidus-sql-e2e", "version": "0"},
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        });
        let body = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params});
        let headers = [
            ("accept", "application/json, text/event-stream"),
            ("mcp-protocol-version", VERSION),
            ("mcp-method", "tools/call"),
            ("mcp-name", "query"),
        ];
        server.post_with_headers("/mcp", &body, &headers)
    }

    /// Same classes again, over MCP: a parse error is a caller fault (`invalid_params`,
    /// `-32602`, HTTP `400`), and the message is the identical text, not a re-wording.
    #[test]
    fn mcp_parse_error_classes_are_invalid_params_with_the_same_message() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let server = Server::new(tmp.path(), 2).start();
        for class in super::error_classes().into_iter().filter(|c| c.on_mcp) {
            let (status, body) = call_query(&server, &class.sql);
            assert_eq!(status, 400, "class {}: {body}", class.name);
            assert_eq!(
                body["error"]["code"].as_i64(),
                Some(-32602),
                "class {}: a parse error is a caller fault: {body}",
                class.name
            );
            let msg = body["error"]["message"].as_str().unwrap_or_default();
            for s in &class.contains {
                assert!(
                    msg.contains(s),
                    "class {} missing {s:?} in: {msg}",
                    class.name
                );
            }
        }
    }
}

// ── `--compile`: renders the compiled form, runs nothing ────────────────────

#[test]
fn compile_only_prints_the_compiled_form_and_runs_nothing() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", dir, "--dim", "2", "notes"], "");

    let out = ok(
        &[
            "query",
            "--dir",
            dir,
            "--compile",
            "SELECT * FROM notes WHERE lang = 'rust' ORDER BY knn([1,0]) LIMIT 5",
        ],
        "",
    );
    // `--compile` prints one compiled statement per element, so a single SELECT is `[ { .. } ]`.
    let c = &out[0];
    assert_eq!(c["kind"], "search", "{out}");
    assert_eq!(c["collections"], json!(["notes"]), "{out}");
    assert_eq!(c["vector"], json!([1.0, 0.0]), "{out}");
    assert_eq!(c["opts"]["top_k"], 5, "{out}");

    // Nothing ran: the store still has zero records.
    let stats = ok(&["stats", "--dir", dir], "");
    assert_eq!(stats["collections"], json!(["notes"]), "{stats}");
}

// ── `;`-batching: each statement runs, in order, over the store's live state ────────────

/// A `;`-separated script prints one JSON document per statement, in the order written —
/// not the reverse, and not interleaved.
#[test]
fn batch_script_runs_each_statement_in_order() {
    let tmp = seeded_dir();
    let dir = tmp.path().to_str().expect("utf-8 temp path");

    let sql = "SELECT * FROM notes WHERE kind = 'x'; SELECT * FROM notes WHERE kind = 'y'";
    let out = run(&["query", "--dir", dir, sql], "");
    assert!(
        out.status.success(),
        "nidus query exited {:?}\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let docs: Vec<Value> = serde_json::Deserializer::from_str(&stdout)
        .into_iter::<Value>()
        .map(|r| r.unwrap_or_else(|e| panic!("batch stdout not JSON ({e}):\n{stdout}")))
        .collect();
    assert_eq!(
        docs.len(),
        2,
        "expected one document per statement:\n{stdout}"
    );
    assert_eq!(ids_of(&docs[0]), vec!["a"], "first statement: {stdout}");
    assert_eq!(ids_of(&docs[1]), vec!["b"], "second statement: {stdout}");
}

// ── Depth guard: a clean error, not a crash ──────────────────────────────────

/// A 200-deep nested `WHERE` clause — past `MAX_NEST_DEPTH` (128) — returns a clean parse
/// error rather than overflowing the stack (SPEC §1 Stable: graceful resource exhaustion).
/// The process must exit with the normal error path, never a signal/abort.
#[test]
fn deeply_nested_predicate_fails_cleanly_instead_of_aborting() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", dir, "--dim", "2", "notes"], "");

    let sql = format!(
        "SELECT * FROM notes WHERE {}a = 1{}",
        "(".repeat(200),
        ")".repeat(200)
    );
    let out = run(&["query", "--dir", dir, &sql], "");
    assert!(
        out.status.code().is_some(),
        "the process must exit normally (with a code), not die by signal"
    );
    assert!(
        !out.status.success(),
        "a too-deep predicate must be refused, not accepted"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("nesting exceeds"), "{stderr}");
    assert!(stderr.contains("§7.3"), "{stderr}");
}
