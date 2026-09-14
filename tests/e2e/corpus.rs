//! The conformance corpus (nidus-yq9p.2): replays `tests/corpus/queries.json` across every
//! in-crate surface, asserting the SQL spelling and its typed `dsl` twin return identical
//! ordered ids. Format and the "adding a case" workflow: `tests/corpus/README.md`.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Value, json};

use nidus::server::dto::{
    AggregateRequest, HybridSearchRequest, ListRequest, SearchRequest, TextSearchRequest,
};
use nidus::{
    AggregateOpts, Aggregation, Config, FtsClause, FtsQuery, Hit, HybridOpts, ListOpts, Nidus,
    OpenMode, Projection, QueryAnswer, Scope, SearchOpts, Value as NValue,
};

use crate::harness::{RunningServer, Server};

const RAW_CORPUS: &str = include_str!("../corpus/queries.json");

#[derive(Debug, Deserialize)]
struct Corpus {
    fixture: Fixture,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    dim: usize,
    #[serde(default)]
    fts: BTreeMap<String, Vec<String>>,
    collections: BTreeMap<String, Vec<FixtureRecord>>,
}

#[derive(Debug, Deserialize)]
struct FixtureRecord {
    id: String,
    vector: Vec<f32>,
    attrs: BTreeMap<String, NValue>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    section: String,
    sql: String,
    #[serde(default)]
    dsl: Option<Dsl>,
    expect: Expect,
    surfaces: Vec<String>,
    #[serde(default)]
    #[allow(dead_code)] // documentation for a human reading the corpus, not read by the runner
    note: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Dsl {
    endpoint: String,
    body: Value,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Expect {
    Ids { ids: Vec<String> },
    Aggregation { aggregation: Value },
    Batch { batch_ids: Vec<Vec<String>> },
}

/// One statement's answer, normalized so every surface (typed JSON, HTTP JSON, or a directly
/// typed library call) reduces to the same comparable shape.
#[derive(Debug, Clone, PartialEq)]
enum CaseResult {
    Ids(Vec<String>),
    Aggregation(AggSummary),
}

/// `(count, sums, groups)` — `groups_truncated` is deliberately not part of the comparison
/// (every corpus case keeps it `false`, and the DTO surfaces omit the field entirely then).
type AggSummary = (u64, Value, Vec<(Value, u64, Value)>);

fn load_corpus() -> Corpus {
    serde_json::from_str(RAW_CORPUS).expect("tests/corpus/queries.json must parse")
}

/// Seed every fixture collection over HTTP: this is the ONE write path the whole corpus uses,
/// so the lib/cli/mcp legs below all read back exactly what it wrote (D0010-in-spirit: one
/// seed, no per-surface fixture drift).
fn seed(server: &RunningServer, fixture: &Fixture) {
    for (name, records) in &fixture.collections {
        let (status, body) = server.post(&format!("/collections/{name}"), &json!({}));
        assert_eq!(status, 200, "create collection {name}: {body}");
        if let Some(fields) = fixture.fts.get(name) {
            let (status, body) = server.post(
                &format!("/collections/{name}/fts-schema"),
                &json!({"fields": fields}),
            );
            assert_eq!(status, 200, "fts-schema for {name}: {body}");
        }
        let wire: Vec<Value> = records
            .iter()
            .map(|r| {
                let attrs = serde_json::to_value(&r.attrs).expect("serialize fixture attrs");
                json!({"id": r.id, "vector": r.vector, "attrs": attrs})
            })
            .collect();
        let (status, body) = server.post(
            &format!("/collections/{name}/upsert"),
            &json!({"records": wire}),
        );
        assert_eq!(status, 200, "seeding {name}: {body}");
    }
}

fn ids(hits: &[Hit]) -> Vec<String> {
    hits.iter().map(|h| h.id.clone()).collect()
}

fn agg_summary_from_typed(agg: &Aggregation) -> AggSummary {
    let sums = serde_json::to_value(&agg.sums).expect("serialize sums");
    let groups = agg
        .groups
        .iter()
        .map(|g| {
            let value = g
                .value
                .as_ref()
                .map(|v| serde_json::to_value(v).expect("serialize group value"))
                .unwrap_or(Value::Null);
            let sums = serde_json::to_value(&g.sums).expect("serialize group sums");
            (value, g.count, sums)
        })
        .collect();
    (agg.count, sums, groups)
}

fn agg_summary_from_json(v: &Value) -> AggSummary {
    let count = v["count"].as_u64().unwrap_or(0);
    let sums = v["sums"].clone();
    let groups = v["groups"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|g| {
            (
                g["value"].clone(),
                g["count"].as_u64().unwrap_or(0),
                g["sums"].clone(),
            )
        })
        .collect();
    (count, sums, groups)
}

/// A `Hits`-dispatch JSON body is either a bare array or `{"hits": [...], "plan": {...}}`.
fn ids_from_response_json(v: &Value) -> Vec<String> {
    let hits = v.get("hits").unwrap_or(v);
    hits.as_array()
        .unwrap_or_else(|| panic!("expected a hits array, got {v}"))
        .iter()
        .map(|h| h["id"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn result_from_json(v: &Value, is_aggregate: bool) -> CaseResult {
    if is_aggregate {
        CaseResult::Aggregation(agg_summary_from_json(v))
    } else {
        CaseResult::Ids(ids_from_response_json(v))
    }
}

fn scope_of(names: &[String]) -> Vec<&str> {
    names.iter().map(String::as_str).collect()
}

/// Run one statement's `dsl` twin directly through the library: deserialize the wire body into
/// the same DTO the HTTP handler uses, build the typed opts, and call the matching `Nidus`
/// method — the "typed caller" spelling every corpus case's `sql` compiles down to.
fn run_dsl_lib(db: &Nidus, dsl: &Dsl) -> anyhow::Result<CaseResult> {
    match dsl.endpoint.as_str() {
        "/search" => {
            let req: SearchRequest = serde_json::from_value(dsl.body.clone())?;
            let opts = SearchOpts {
                top_k: req.top_k,
                offset: req.offset,
                filter: req.filter,
                min_score: req.min_score,
                exact: req.exact,
                projection: Projection::All,
                explain: false,
                plan: req.plan,
                rank_by: req.rank_by,
                limit_per: req.limit_per,
                diversity: req.diversity,
                rerank: None,
                expand: req.expand.map(Into::into),
            };
            let refs = scope_of(&req.scope);
            let scope = if refs.is_empty() {
                Scope::All
            } else {
                Scope::Collections(&refs)
            };
            let hits = if opts.plan {
                db.search_with_plan(scope, &req.query, &opts)?.0
            } else {
                db.search(scope, &req.query, &opts)?
            };
            Ok(CaseResult::Ids(ids(&hits)))
        }
        "/list" => {
            let req: ListRequest = serde_json::from_value(dsl.body.clone())?;
            let opts = ListOpts {
                offset: req.offset,
                limit: req.limit,
                filter: req.filter,
                projection: Projection::All,
                order_by: req.order_by,
            };
            let refs = scope_of(&req.scope);
            let scope = if refs.is_empty() {
                Scope::All
            } else {
                Scope::Collections(&refs)
            };
            Ok(CaseResult::Ids(ids(&db.list(scope, &opts)?)))
        }
        "/aggregate" => {
            let req: AggregateRequest = serde_json::from_value(dsl.body.clone())?;
            let opts = AggregateOpts {
                filter: req.filter,
                sum: req.sum,
                group_by: req.group_by,
            };
            let refs = scope_of(&req.scope);
            let scope = if refs.is_empty() {
                Scope::All
            } else {
                Scope::Collections(&refs)
            };
            let agg = db.aggregate(scope, &opts)?;
            Ok(CaseResult::Aggregation(agg_summary_from_typed(&agg)))
        }
        "/text-search" => {
            let req: TextSearchRequest = serde_json::from_value(dsl.body.clone())?;
            let opts = SearchOpts {
                top_k: req.top_k,
                offset: req.offset,
                filter: req.filter,
                min_score: req.min_score,
                exact: false,
                projection: Projection::All,
                explain: req.explain,
                plan: false,
                rank_by: req.rank_by,
                limit_per: req.limit_per,
                diversity: req.diversity,
                rerank: None,
                expand: req.expand.map(Into::into),
            };
            let field = req.field.expect("corpus /text-search dsl needs `field`");
            let text = req.query.expect("corpus /text-search dsl needs `query`");
            let clause = if req.prefix {
                FtsClause::new(field, text).prefix()
            } else {
                FtsClause::new(field, text)
            };
            let query = FtsQuery {
                clauses: vec![clause],
                combine: req.combine,
                highlight: None,
            };
            let refs = scope_of(&req.scope);
            let scope = if refs.is_empty() {
                Scope::All
            } else {
                Scope::Collections(&refs)
            };
            Ok(CaseResult::Ids(ids(&db.text_search(scope, &query, &opts)?)))
        }
        "/hybrid-search" => {
            let req: HybridSearchRequest = serde_json::from_value(dsl.body.clone())?;
            let opts = HybridOpts {
                top_k: req.top_k,
                offset: req.offset,
                filter: req.filter,
                rrf_k: req.rrf_k,
                candidates: req.candidates,
                explain: req.explain,
                plan: req.plan,
                vector_weight: req.vector_weight,
                text_weight: req.text_weight,
                expand: req.expand.map(Into::into),
                rerank: None,
            };
            let field = req.field.expect("corpus /hybrid-search dsl needs `field`");
            let text = req.text.expect("corpus /hybrid-search dsl needs `text`");
            let clause = if req.prefix {
                FtsClause::new(field, text).prefix()
            } else {
                FtsClause::new(field, text)
            };
            let query = FtsQuery {
                clauses: vec![clause],
                combine: req.combine,
                highlight: None,
            };
            let refs = scope_of(&req.scope);
            let scope = if refs.is_empty() {
                Scope::All
            } else {
                Scope::Collections(&refs)
            };
            let hits = if opts.plan {
                db.hybrid_search_with_plan(scope, &req.vector, &query, &opts)?
                    .0
            } else {
                db.hybrid_search(scope, &req.vector, &query, &opts)?
            };
            Ok(CaseResult::Ids(ids(&hits)))
        }
        other => anyhow::bail!("corpus runner: no lib mapping for dsl endpoint {other}"),
    }
}

fn case_result_from_answer(a: QueryAnswer) -> CaseResult {
    match a {
        QueryAnswer::Hits { hits, .. } => CaseResult::Ids(ids(&hits)),
        QueryAnswer::Aggregation(agg) => CaseResult::Aggregation(agg_summary_from_typed(&agg)),
    }
}

fn run_sql_lib(db: &Nidus, sql: &str) -> Vec<CaseResult> {
    db.query_batch(sql)
        .unwrap_or_else(|e| panic!("db.query_batch({sql:?}): {e:#}"))
        .into_iter()
        .map(case_result_from_answer)
        .collect()
}

fn run_dsl_http(server: &RunningServer, dsl: &Dsl) -> CaseResult {
    let (status, body) = server.post(&dsl.endpoint, &dsl.body);
    assert_eq!(status, 200, "{} failed: {body}", dsl.endpoint);
    result_from_json(&body, dsl.endpoint == "/aggregate")
}

fn run_sql_http(server: &RunningServer, sql: &str, is_aggregate: bool) -> Vec<CaseResult> {
    let (status, body) = server.post("/query", &json!({"sql": sql}));
    assert_eq!(status, 200, "/query failed for {sql:?}: {body}");
    split_answers(&body, is_aggregate)
}

/// One statement's hits answer is itself a bare array of hits, so a top-level array is only a
/// `;`-batch when its elements are answer-shaped rather than hit-shaped (a hit always carries
/// `id`). Mirrors the Go SDK's `looksLikeQueryAnswer`.
fn split_answers(body: &Value, is_aggregate: bool) -> Vec<CaseResult> {
    match body {
        Value::Array(items) => {
            let is_batch = items
                .first()
                .is_some_and(|e| e.is_array() || e.get("id").is_none());
            if is_batch {
                items
                    .iter()
                    .map(|v| result_from_json(v, is_aggregate))
                    .collect()
            } else {
                vec![result_from_json(body, is_aggregate)]
            }
        }
        other => vec![result_from_json(other, is_aggregate)],
    }
}

/// Every JSON document `nidus`'s stdout carries, in order — `nidus query`'s batch mode prints
/// one pretty-printed document per statement with no separator, so a plain `from_str` (which
/// expects exactly one root value) cannot read it back.
fn cli_json_docs(args: &[&str]) -> Vec<Value> {
    let out = crate::harness::run(args, "");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "nidus {args:?} exited {:?}\n--- stderr ---\n{stderr}",
        out.status.code()
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    serde_json::Deserializer::from_str(&stdout)
        .into_iter::<Value>()
        .map(|r| r.unwrap_or_else(|e| panic!("nidus {args:?} stdout not JSON ({e}):\n{stdout}")))
        .collect()
}

/// The one JSON document a stdin-fed invocation carries.
fn cli_json_doc_stdin(args: &[&str], stdin: &str) -> Value {
    let out = crate::harness::run(args, stdin);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "nidus {args:?} exited {:?}\n--- stderr ---\n{stderr}",
        out.status.code()
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("nidus {args:?} stdout not JSON ({e}):\n{stdout}"))
}

fn scope_args(body: &Value) -> Vec<String> {
    body.get("scope")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|c| c.as_str().unwrap_or_default().to_string())
        .collect()
}

/// Build the `nidus` CLI invocation equivalent to one `dsl` body, as `(args, stdin)`. Handles
/// exactly the endpoint/field combinations the corpus uses today — see `tests/corpus/README.md`
/// before adding a case that needs a flag this does not yet map.
fn cli_invocation(dir: &str, dsl: &Dsl) -> (Vec<String>, String) {
    let body = &dsl.body;
    let mut args = vec!["--dir".to_string(), dir.to_string()];
    match dsl.endpoint.as_str() {
        "/search" => {
            args.insert(0, "search".to_string());
            if let Some(k) = body.get("top_k").and_then(Value::as_u64) {
                args.push("-k".into());
                args.push(k.to_string());
            }
            if let Some(f) = body.get("filter") {
                args.push("--where".into());
                args.push(f.to_string());
            }
            if body.get("plan").and_then(Value::as_bool) == Some(true) {
                args.push("--plan".into());
            }
            args.extend(scope_args(body));
            (args, body["query"].to_string())
        }
        "/list" => {
            args.insert(0, "list".to_string());
            if let Some(f) = body.get("filter") {
                args.push("--where".into());
                args.push(f.to_string());
            }
            if let Some(ob) = body.get("order_by") {
                args.push("--order-by".into());
                args.push(ob["field"].as_str().unwrap_or_default().to_string());
                if ob["descending"].as_bool() == Some(true) {
                    args.push("--desc".into());
                }
            }
            args.extend(scope_args(body));
            (args, String::new())
        }
        "/aggregate" => {
            args.insert(0, "aggregate".to_string());
            if let Some(f) = body.get("filter") {
                args.push("--where".into());
                args.push(f.to_string());
            }
            for s in body
                .get("sum")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
            {
                args.push("--sum".into());
                args.push(s.as_str().unwrap_or_default().to_string());
            }
            if let Some(g) = body.get("group_by").and_then(Value::as_str) {
                args.push("--group-by".into());
                args.push(g.to_string());
            }
            args.extend(scope_args(body));
            (args, String::new())
        }
        "/text-search" => {
            args.insert(0, "text-search".to_string());
            args.push(body["field"].as_str().unwrap_or_default().to_string());
            args.push(body["query"].as_str().unwrap_or_default().to_string());
            for c in scope_args(body) {
                args.push("--in".into());
                args.push(c);
            }
            if let Some(k) = body.get("top_k").and_then(Value::as_u64) {
                args.push("-k".into());
                args.push(k.to_string());
            }
            if let Some(f) = body.get("filter") {
                args.push("--where".into());
                args.push(f.to_string());
            }
            if body.get("explain").and_then(Value::as_bool) == Some(true) {
                args.push("--explain".into());
            }
            if let Some(lp) = body.get("limit_per") {
                args.push("--limit-per".into());
                args.push(lp["field"].as_str().unwrap_or_default().to_string());
                args.push("--limit-per-max".into());
                args.push(lp["max"].as_u64().unwrap_or_default().to_string());
            }
            if let Some(ex) = body.get("expand") {
                args.push("--expand-radius".into());
                args.push(ex["radius"].as_u64().unwrap_or_default().to_string());
                args.push("--expand-parent-field".into());
                args.push(ex["parent_field"].as_str().unwrap_or_default().to_string());
                args.push("--expand-index-field".into());
                args.push(ex["index_field"].as_str().unwrap_or_default().to_string());
                args.push("--expand-text-field".into());
                args.push(ex["text_field"].as_str().unwrap_or_default().to_string());
            }
            (args, String::new())
        }
        "/hybrid-search" => {
            args.insert(0, "hybrid-search".to_string());
            args.push(body["field"].as_str().unwrap_or_default().to_string());
            args.push(body["text"].as_str().unwrap_or_default().to_string());
            for c in scope_args(body) {
                args.push("--in".into());
                args.push(c);
            }
            if let Some(k) = body.get("top_k").and_then(Value::as_u64) {
                args.push("-k".into());
                args.push(k.to_string());
            }
            (args, body["vector"].to_string())
        }
        other => panic!("corpus runner: no CLI mapping for dsl endpoint {other}"),
    }
}

fn run_dsl_cli(dir: &str, dsl: &Dsl) -> CaseResult {
    let (args, stdin) = cli_invocation(dir, dsl);
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let doc = cli_json_doc_stdin(&arg_refs, &stdin);
    result_from_json(&doc, dsl.endpoint == "/aggregate")
}

fn run_sql_cli(dir: &str, sql: &str, is_aggregate: bool) -> Vec<CaseResult> {
    cli_json_docs(&["query", "--dir", dir, sql])
        .iter()
        .map(|d| result_from_json(d, is_aggregate))
        .collect()
}

// ── MCP leg: a minimal JSON-RPC client, self-contained so this file needs no visibility
// changes to the `mcp` test module (a sibling, not an ancestor, of this one). ──────────

#[cfg(feature = "mcp")]
mod mcp_client {
    use serde_json::{Value, json};

    use crate::harness::RunningServer;

    const VERSION: &str = "2026-07-28";

    /// Call the `query` tool with `sql` and return the raw JSON-RPC response envelope.
    pub(super) fn call_query(server: &RunningServer, sql: &str) -> Value {
        let params = json!({
            "name": "query",
            "arguments": {"sql": sql},
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": VERSION,
                "io.modelcontextprotocol/clientInfo": {"name": "nidus-corpus", "version": "0"},
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
        let (status, resp) = server.post_with_headers("/mcp", &body, &headers);
        assert_eq!(status, 200, "mcp query({sql:?}) transport failed: {resp}");
        resp
    }

    /// The concatenated text of a successful result's content blocks, or a panic naming the
    /// JSON-RPC error for a failed call.
    pub(super) fn text_or_panic(resp: &Value, sql: &str) -> String {
        if let Some(err) = resp.get("error") {
            panic!("mcp query({sql:?}) returned a JSON-RPC error: {err}");
        }
        resp["result"]["content"]
            .as_array()
            .unwrap_or_else(|| panic!("mcp query({sql:?}): no content array in {resp}"))
            .iter()
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(feature = "mcp")]
fn run_sql_mcp(server: &RunningServer, sql: &str, is_aggregate: bool) -> Vec<CaseResult> {
    let resp = mcp_client::call_query(server, sql);
    let text = mcp_client::text_or_panic(&resp, sql);
    let v: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("mcp query({sql:?}) content not JSON ({e}):\n{text}"));
    vec![result_from_json(&v, is_aggregate)]
}

// ── Comparison and reporting ─────────────────────────────────────────────────

/// Compare one surface's SQL-leg answer(s) against the case's `expect`, reporting a mismatch
/// as a per-surface diff of ordered ids — never a bare `assert_eq!` (nidus-yq9p.2).
fn check(case: &Case, surface: &str, got: &[CaseResult]) {
    let label = format!("case `{}` (§{}) on {surface}", case.name, case.section);
    if let Expect::Batch { batch_ids } = &case.expect {
        assert_eq!(
            got.len(),
            batch_ids.len(),
            "{label}: expected {} statements, got {}",
            batch_ids.len(),
            got.len()
        );
        for (i, (g, want)) in got.iter().zip(batch_ids.iter()).enumerate() {
            let CaseResult::Ids(got_ids) = g else {
                panic!("{label}: statement {i} was not a Hits answer")
            };
            assert_eq!(
                got_ids, want,
                "{label}, statement {i} diverged:\n  got:  {got_ids:?}\n  want: {want:?}"
            );
        }
        return;
    }
    assert_eq!(
        got.len(),
        1,
        "{label}: expected one statement, got {}",
        got.len()
    );
    match (&got[0], &case.expect) {
        (CaseResult::Ids(got_ids), Expect::Ids { ids: want }) => {
            assert_eq!(
                got_ids, want,
                "{label} diverged:\n  got:  {got_ids:?}\n  want: {want:?}"
            );
        }
        (CaseResult::Aggregation(got_agg), Expect::Aggregation { aggregation }) => {
            let want = agg_summary_from_json(aggregation);
            assert_eq!(
                *got_agg, want,
                "{label} diverged:\n  got:  {got_agg:?}\n  want: {want:?}"
            );
        }
        _ => panic!("{label}: the answer's shape did not match `expect`"),
    }
}

/// The `sql` and `dsl` spellings must be byte-identical in the ids they return — the property
/// that proves SQL is a front end over the existing entry points, not a second engine.
fn check_dsl_matches(
    case: &Case,
    surface: &str,
    sql_results: &[CaseResult],
    dsl_result: &CaseResult,
) {
    assert_eq!(
        sql_results.len(),
        1,
        "case `{}` on {surface}: a dsl comparison needs exactly one sql statement",
        case.name
    );
    assert_eq!(
        &sql_results[0], dsl_result,
        "case `{}` (§{}) diverged on {surface}: the sql spelling and its dsl twin disagree\n  sql: {:?}\n  dsl: {:?}",
        case.name, case.section, sql_results[0], dsl_result
    );
}

fn is_aggregate(case: &Case) -> bool {
    matches!(case.expect, Expect::Aggregation { .. })
}

fn run_case(case: &Case, server: &RunningServer, db: &Nidus, dir: &str) {
    let agg = is_aggregate(case);
    for surface in &case.surfaces {
        match surface.as_str() {
            "lib" => {
                let sql_results = run_sql_lib(db, &case.sql);
                check(case, "lib", &sql_results);
                if let Some(dsl) = &case.dsl {
                    let dsl_result = run_dsl_lib(db, dsl)
                        .unwrap_or_else(|e| panic!("case `{}` lib dsl leg: {e:#}", case.name));
                    check_dsl_matches(case, "lib", &sql_results, &dsl_result);
                }
            }
            "http" => {
                let sql_results = run_sql_http(server, &case.sql, agg);
                check(case, "http", &sql_results);
                if let Some(dsl) = &case.dsl {
                    let dsl_result = run_dsl_http(server, dsl);
                    check_dsl_matches(case, "http", &sql_results, &dsl_result);
                }
            }
            "cli" => {
                let sql_results = run_sql_cli(dir, &case.sql, agg);
                check(case, "cli", &sql_results);
                if let Some(dsl) = &case.dsl {
                    let dsl_result = run_dsl_cli(dir, dsl);
                    check_dsl_matches(case, "cli", &sql_results, &dsl_result);
                }
            }
            "mcp" => {
                #[cfg(feature = "mcp")]
                {
                    let sql_results = run_sql_mcp(server, &case.sql, agg);
                    check(case, "mcp", &sql_results);
                    // No generic dsl runner here: MCP's typed tools (`text_search`,
                    // `hybrid_search`, ...) do not accept an arbitrary endpoint body, so the
                    // sql-vs-`expect` check above is this surface's guard.
                }
            }
            "python" | "js" | "go" => {
                // Read by the SDK integration suites (a later unit); this Rust runner has no
                // such surfaces of its own.
            }
            other => panic!("case `{}`: unknown surface `{other}`", case.name),
        }
    }
}

/// Replay every corpus case across every surface it names. See `tests/corpus/README.md`.
#[test]
fn corpus_conformance() {
    let corpus = load_corpus();
    let tmp = tempfile::tempdir().expect("temp dir");
    let dim = corpus.fixture.dim;
    let server = Server::new(tmp.path(), dim).start();
    seed(&server, &corpus.fixture);

    let cfg = Config::new(tmp.path(), dim).open_mode(OpenMode::ReadOnly);
    let db = Nidus::open(cfg).expect("open the seeded store read-only for the lib leg");
    let dir = tmp.path().to_str().expect("utf-8 temp path").to_string();

    for case in &corpus.cases {
        run_case(case, &server, &db, &dir);
    }
}
