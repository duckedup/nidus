//! Command-line end-to-end tests: the real `nidus` binary, run over a temp store dir.
//! Everything here is what the parse-level tests in `src/cli/` structurally cannot see —
//! stdout JSON shapes, stdin handling, exit codes, and the archive on disk.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};

/// Run the binary with `args`, feeding `stdin` and capturing both streams. `NIDUS_*` is
/// stripped for the same reason the HTTP harness strips it: an inherited env var is a
/// flag default here, so it would silently override the flag under test.
fn run(args: &[&str], stdin: &str) -> Output {
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
    child
        .wait_with_output()
        .unwrap_or_else(|e| panic!("wait for nidus {args:?}: {e}"))
}

/// Run a command that must succeed, returning its stdout parsed as JSON. Parsing rather
/// than substring-matching is the point: it fails on a malformed document, not just a
/// changed one.
fn ok(args: &[&str], stdin: &str) -> Value {
    let out = run(args, stdin);
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

/// Run a command that must fail, returning its stderr. Also asserts nothing was printed
/// to stdout — a failing command that emits half a JSON document would poison a pipeline.
fn fails(args: &[&str], stdin: &str) -> String {
    let out = run(args, stdin);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success(),
        "nidus {args:?} unexpectedly succeeded\n--- stdout ---\n{stdout}"
    );
    assert!(stdout.trim().is_empty(), "wrote to stdout: {stdout}");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The ids of a `Vec<HitDto>` / `Vec<Record>` response, in the order printed.
fn ids(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected a JSON array, got {v}"))
        .iter()
        .map(|h| h["id"].as_str().expect("an id").to_string())
        .collect()
}

/// Three records spanning the attr types the query subcommands below filter, sum, sort
/// and full-text search on.
fn seed() -> String {
    json!([
        {"id": "a", "vector": [1, 0, 0], "attrs": {
            "lang": {"Str": "rust"}, "bytes": {"Int": 10},
            "body": {"Str": "the fox runs quickly"}}},
        {"id": "b", "vector": [0, 1, 0], "attrs": {
            "lang": {"Str": "go"}, "bytes": {"Int": 32},
            "body": {"Str": "goroutines are cheap"}}},
        {"id": "c", "vector": [0, 0, 1], "attrs": {
            "lang": {"Str": "rust"}, "bytes": {"Int": 5},
            "body": {"Str": "unrelated prose"}}}
    ])
    .to_string()
}

/// The whole documented life of a store, one subcommand at a time through the real
/// binary: create → upsert from stdin → search → fts → get/list/aggregate → delete →
/// compact → stats → drop.
#[test]
fn full_store_lifecycle_through_the_binary() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");

    // `--dim` is only needed to create; every later command reads it from the header.
    let out = ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    assert_eq!(out["created"], "docs");
    assert_eq!(ok(&["collections", "--dir", dir], ""), json!(["docs"]));

    let out = ok(&["upsert", "--dir", dir, "docs"], &seed());
    assert_eq!(out["upserted"], 3, "records must come from stdin: {out}");

    // Search reads the query vector from stdin, and ranks by cosine.
    let hits = ok(&["search", "--dir", dir, "-k", "2", "docs"], "[1,0,0]");
    assert_eq!(ids(&hits), ["a", "b"], "{hits}");
    assert_eq!(hits[0]["collection"], "docs");
    assert!(hits[0]["score"].as_f64().expect("a score") > 0.99, "{hits}");
    assert_eq!(hits[0]["attrs"]["lang"], json!({"Str": "rust"}));

    // A filter and a projection, both parsed from their JSON/flag forms.
    let hits = ok(
        &[
            "search",
            "--dir",
            dir,
            "-k",
            "5",
            "--where",
            r#"[{"Eq": ["lang", {"Str": "go"}]}]"#,
            "--include-attr",
            "lang",
            "docs",
        ],
        "[1,0,0]",
    );
    assert_eq!(ids(&hits), ["b"], "{hits}");
    assert_eq!(hits[0]["attrs"], json!({"lang": {"Str": "go"}}), "{hits}");

    // Declaring an FTS field reports the tuning it actually applied, per field.
    let out = ok(
        &["set-fts-schema", "--dir", dir, "--field", "body", "docs"],
        "",
    );
    assert_eq!(out["collection"], "docs");
    assert_eq!(out["fts_fields"][0]["field"], "body", "{out}");
    assert_eq!(out["fts_fields"][0]["b"], 0.75, "BM25 default: {out}");

    // "running" matches a document spelling it "runs" — a shared stem, which no
    // substring search for "running" could have found.
    let hits = ok(
        &["text-search", "--dir", dir, "-k", "5", "body", "running"],
        "",
    );
    assert_eq!(ids(&hits), ["a"], "stemming should reach the CLI: {hits}");

    // Hybrid fuses the two legs: the vector leg favours b, the text leg a.
    let hits = ok(
        &["hybrid-search", "--dir", dir, "-k", "5", "body", "running"],
        "[0,1,0]",
    );
    let fused = ids(&hits);
    assert!(
        fused.contains(&"a".to_string()) && fused.contains(&"b".to_string()),
        "both legs should contribute: {hits}"
    );

    // `get` prints whole records — vectors included, unlike a hit.
    let recs = ok(&["get", "--dir", dir, "docs"], "");
    let mut got = ids(&recs);
    got.sort();
    assert_eq!(got, ["a", "b", "c"], "{recs}");
    assert_eq!(
        recs.as_array().expect("records")[0]["vector"]
            .as_array()
            .map(Vec::len),
        Some(3)
    );

    // `list` sorts by an attribute rather than storage order.
    let listed = ok(
        &[
            "list",
            "--dir",
            dir,
            "-n",
            "10",
            "--order-by",
            "bytes",
            "--desc",
            "docs",
        ],
        "",
    );
    assert_eq!(
        ids(&listed),
        ["b", "a", "c"],
        "descending by bytes: {listed}"
    );

    let agg = ok(
        &[
            "aggregate",
            "--dir",
            dir,
            "--sum",
            "bytes",
            "--group-by",
            "lang",
            "docs",
        ],
        "",
    );
    assert_eq!(agg["count"], 3, "{agg}");
    assert_eq!(agg["sums"]["bytes"], json!({"Int": 47}), "{agg}");
    let groups = agg["groups"].as_array().expect("groups");
    assert_eq!(groups.len(), 2, "{agg}");
    assert_eq!(groups[0]["value"], json!({"Str": "rust"}), "{agg}");
    assert_eq!(groups[0]["count"], 2, "{agg}");

    // Delete by id, then by filter — the two arms of the same subcommand. Each is
    // compacted away before the next, keeping the dead-row ratio under the auto-compact
    // threshold: a read-only open past it currently fails outright (#116).
    assert_eq!(ok(&["delete", "--dir", dir, "docs", "c"], "")["deleted"], 1);
    let stats = ok(&["stats", "--dir", dir], "");
    assert_eq!(stats["footprint"]["dead_rows"], 1, "{stats}");
    assert_eq!(ok(&["compact", "--dir", dir], "")["ok"], true);

    let out = ok(
        &[
            "delete",
            "--dir",
            dir,
            "--where",
            r#"[{"Eq": ["lang", {"Str": "go"}]}]"#,
            "docs",
        ],
        "",
    );
    assert_eq!(out["deleted"], 1, "{out}");
    assert_eq!(ok(&["compact", "--dir", dir], "")["ok"], true);

    let stats = ok(&["stats", "--dir", dir], "");
    assert_eq!(stats["dimension"], 3, "{stats}");
    assert_eq!(stats["distance"], "Cosine", "{stats}");
    assert_eq!(stats["ann"], Value::Null, "exact by default: {stats}");
    assert_eq!(stats["collections"], json!(["docs"]), "{stats}");
    assert_eq!(stats["footprint"]["doc_count"], 1, "{stats}");
    assert_eq!(stats["footprint"]["dead_rows"], 0, "compacted: {stats}");

    // Dropping the last collection leaves every row dead, so compact once more before
    // the closing read-only open — again #116.
    assert_eq!(ok(&["drop", "--dir", dir, "docs"], "")["dropped"], "docs");
    assert_eq!(ok(&["compact", "--dir", dir], "")["ok"], true);
    assert_eq!(ok(&["collections", "--dir", dir], ""), json!([]));
}

/// Every way a caller gets this wrong must exit nonzero and say what to do — not exit 0
/// with an empty result, and not print half a JSON document.
#[test]
fn bad_invocations_exit_nonzero_with_a_useful_message() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");

    // A filter that is JSON but not a `Filter`.
    let err = fails(
        &["search", "--dir", dir, "--where", r#"{"Eq": 1}"#, "docs"],
        "[1,0,0]",
    );
    assert!(err.starts_with("error:"), "{err}");

    // A store that does not exist, with no `--dim` to create one.
    let missing = tmp.path().join("nowhere");
    let err = fails(&["stats", "--dir", missing.to_str().unwrap()], "");
    assert!(
        err.contains("no store at") && err.contains("--dim"),
        "{err}"
    );

    // Deleting by id *and* by filter at once: clap refuses the pair outright.
    let err = fails(&["delete", "--dir", dir, "docs", "a", "--where", "[]"], "");
    assert!(
        err.contains("--where"),
        "the refusal should name the flag: {err}"
    );

    // Two projections at once is a refusal, not a precedence rule.
    let err = fails(
        &[
            "search",
            "--dir",
            dir,
            "--include-attr",
            "lang",
            "--exclude-attr",
            "bytes",
            "docs",
        ],
        "[1,0,0]",
    );
    assert!(err.contains("mutually exclusive"), "{err}");

    // A schema declaring nothing would silently index nothing.
    let err = fails(&["set-fts-schema", "--dir", dir, "docs"], "");
    assert!(err.contains("--field"), "{err}");

    // A query vector of the wrong length is caught, not scored against garbage.
    let err = fails(&["search", "--dir", dir, "docs"], "[1,0]");
    assert!(err.contains("dimension"), "{err}");
}

/// Read subcommands open read-only, so they never contend with a running `nidus serve`
/// writer. Only two real processes over one directory can prove that.
#[test]
fn read_subcommands_run_against_a_dir_a_server_holds() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    let server = crate::harness::Server::new(tmp.path(), 3).start();
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
    assert_eq!(
        server
            .post(
                "/collections/docs/upsert",
                &json!({"records": [{"id": "a", "vector": [1, 0, 0], "attrs": {}}]}),
            )
            .0,
        200
    );

    let stats = ok(&["stats", "--dir", dir], "");
    assert_eq!(stats["footprint"]["doc_count"], 1, "{stats}");
    assert_eq!(ids(&ok(&["get", "--dir", dir, "docs"], "")), ["a"]);

    // A mutating subcommand, by contrast, wants the lock the server holds and is refused.
    let err = fails(&["upsert", "--dir", dir, "docs"], "[]");
    assert!(err.contains("lock"), "{err}");
}

/// `backup` writes a real archive to disk and `restore` reads it back into a fresh
/// store — the binary path, including the guard that refuses to overwrite silently.
#[test]
fn backup_and_restore_round_trip_through_the_binary() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let src = src.to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", src, "--dim", "3", "docs"], "");
    ok(&["upsert", "--dir", src, "docs"], &seed());

    let archive = tmp.path().join("snapshot.tar.gz");
    let archive = archive.to_str().expect("utf-8 temp path");
    let report = ok(&["backup", "--dir", src, "-o", archive], "");
    assert_eq!(report["dimension"], 3, "{report}");
    assert_eq!(report["distance"], "Cosine", "{report}");
    assert_eq!(report["backup"], archive, "{report}");
    assert!(
        report["archive_bytes"].as_u64().unwrap_or(0) > 0,
        "{report}"
    );
    assert!(Path::new(archive).is_file(), "no archive on disk");

    // Into a fresh (empty) location: no store there yet, so no confirmation is needed.
    let dst = tmp.path().join("dst");
    std::fs::create_dir(&dst).unwrap();
    let dst = dst.to_str().expect("utf-8 temp path");
    let report = ok(&["restore", "-i", archive, "--dir", dst], "");
    assert_eq!(report["records"], 3, "{report}");
    assert_eq!(report["collections"], json!(["docs"]), "{report}");
    assert_eq!(report["dimension"], 3, "{report}");

    // The records really are queryable in the restored store, not merely counted.
    let mut got = ids(&ok(&["get", "--dir", dst, "docs"], ""));
    got.sort();
    assert_eq!(got, ["a", "b", "c"]);
    assert_eq!(
        ids(&ok(&["search", "--dir", dst, "-k", "1"], "[0,1,0]")),
        ["b"]
    );

    // Restoring over a store that already exists must refuse: stdin is a pipe, so the
    // prompt reads EOF and takes the safe default rather than overwriting.
    let err = fails(&["restore", "-i", archive, "--dir", dst], "");
    assert!(err.contains("aborted") && err.contains("--yes"), "{err}");
    assert_eq!(ids(&ok(&["get", "--dir", dst, "docs"], "")).len(), 3);

    // `--yes` is the scripted path, and it goes through.
    let report = ok(&["restore", "-i", archive, "--dir", dst, "--yes"], "");
    assert_eq!(report["records"], 3, "{report}");
}
