//! Command-line end-to-end tests: the real `nidus` binary, run over a temp store dir.
//! Everything here is what the parse-level tests in `src/cli/` structurally cannot see —
//! stdout JSON shapes, stdin handling, exit codes, and the archive on disk.

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Output, Stdio};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
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

/// Flip one byte of `path` at `offset`, in place. Used to corrupt an archive at a
/// chosen position rather than a random one, so a failing assertion reproduces.
fn flip_byte(path: &str, offset: usize) {
    let mut bytes = std::fs::read(path).unwrap();
    assert!(offset < bytes.len(), "offset {offset} out of range");
    bytes[offset] ^= 0xFF;
    std::fs::write(path, &bytes).unwrap();
}

/// Flip one byte of the first tar entry's content, then re-gzip losslessly. Every
/// structural layer stays valid, so only the CRC baseline can catch it (#152).
fn corrupt_first_entry_content(archive: &str, content_offset: usize) {
    let mut tar_bytes = Vec::new();
    GzDecoder::new(&std::fs::File::open(archive).unwrap())
        .read_to_end(&mut tar_bytes)
        .unwrap();
    let offset = 512 + content_offset;
    assert!(offset < tar_bytes.len(), "content offset out of range");
    tar_bytes[offset] ^= 0xFF;

    let mut regzipped = Vec::new();
    let mut enc = GzEncoder::new(&mut regzipped, Compression::default());
    enc.write_all(&tar_bytes).unwrap();
    enc.finish().unwrap();
    std::fs::write(archive, regzipped).unwrap();
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

    // Hybrid fuses two legs that disagree, so the fused winner is one neither leg picks
    // alone. A `contains` check would prove nothing: at k=5 over three docs the vector
    // leg alone returns all of them, text leg or no text leg.
    let hits = ok(&["search", "--dir", dir, "-k", "1", "docs"], "[0,1,0]");
    assert_eq!(
        ids(&hits),
        ["b"],
        "the vector leg alone ranks b first: {hits}"
    );
    let hits = ok(
        &["hybrid-search", "--dir", dir, "-k", "1", "body", "running"],
        "[0,1,0]",
    );
    assert_eq!(
        ids(&hits),
        ["a"],
        "the text leg must outweigh the vector leg's own top hit: {hits}"
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

/// `similar` searches the vector already stored at COLLECTION/ID, defaulting scope to
/// that same collection and dropping the source record from the results.
#[test]
fn similar_finds_the_nearest_neighbour_and_excludes_the_source() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    ok(
        &["upsert", "--dir", dir, "docs"],
        &json!([
            {"id": "a", "vector": [1.0, 0.0, 0.0], "attrs": {}},
            {"id": "b", "vector": [0.8, 0.6, 0.0], "attrs": {}},
            {"id": "c", "vector": [0.0, 0.0, 1.0], "attrs": {}}
        ])
        .to_string(),
    );

    let hits = ok(&["similar", "--dir", dir, "docs", "a"], "");
    let hit_ids = ids(&hits);
    assert!(!hit_ids.contains(&"a".to_string()), "{hits}");
    assert!(hit_ids.contains(&"b".to_string()), "{hits}");
}

/// A text-only source record has no vector to search with, so `similar` refuses rather
/// than scanning garbage, naming the id and the reason on stderr.
#[test]
fn similar_on_a_text_only_source_exits_nonzero_with_the_reason() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    ok(
        &["upsert", "--dir", dir, "docs"],
        &json!([{"id": "t1", "attrs": {"kind": {"Str": "note"}}}]).to_string(),
    );

    let err = fails(&["similar", "--dir", dir, "docs", "t1"], "");
    assert!(err.contains("t1"), "should name the record: {err}");
    assert!(err.contains("text-only"), "should say why: {err}");
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

/// A byte flipped in the compressed body must fail `verify` and name the corruption,
/// and a byte flipped in either edge case (gzip trailer, tar's zero padding) must fail
/// too, since #152 showed a corrupted archive can otherwise restore looking clean.
#[test]
fn verify_accepts_a_good_archive_and_rejects_a_corrupted_one() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let src = src.to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", src, "--dim", "3", "docs"], "");
    ok(&["upsert", "--dir", src, "docs"], &seed());

    let archive = tmp.path().join("snapshot.tar.gz");
    let archive = archive.to_str().expect("utf-8 temp path");
    ok(&["backup", "--dir", src, "-o", archive], "");

    let report = ok(&["verify", "-i", archive], "");
    assert_eq!(report["dimension"], 3, "{report}");
    assert_eq!(report["collections"], json!(["docs"]), "{report}");
    assert_eq!(report["records"], 3, "{report}");
    assert!(
        report["objects_checked"].as_u64().unwrap_or(0) > 0,
        "{report}"
    );

    // Corrupted vector content that every structural layer still accepts: only the
    // CRC baseline can catch it, so the message must name the object and its crc32.
    corrupt_first_entry_content(archive, 70);
    let err = fails(&["verify", "-i", archive], "");
    assert!(
        err.contains("data") && err.contains("crc32"),
        "should name the corrupted object: {err}"
    );

    // A raw flip breaks a structural layer instead. The message varies by which one,
    // so assert only that it fails loudly rather than pinning tar/gzip wording.
    ok(&["backup", "--dir", src, "-o", archive], "");
    let len = std::fs::metadata(archive).unwrap().len() as usize;
    assert!(len > 100, "archive too small to pick a body offset");
    flip_byte(archive, len / 4);
    assert!(!fails(&["verify", "-i", archive], "").is_empty());
}

/// The #152 regression: gzip's trailer CRC is never reached and tar checksums only
/// headers, so a payload-content flip restores with exit 0 and a correct-looking
/// report (measured 114/204 single-bit flips). Must fail pre-fix, or it proves nothing.
#[test]
fn restore_rejects_a_corrupted_archive() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let src = src.to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", src, "--dim", "3", "docs"], "");
    ok(&["upsert", "--dir", src, "docs"], &seed());

    let archive = tmp.path().join("snapshot.tar.gz");
    let archive = archive.to_str().expect("utf-8 temp path");
    ok(&["backup", "--dir", src, "-o", archive], "");

    // `data`'s content starts right after its 512-byte tar header; offset 70 lands
    // past the segment's own 64-byte header, inside real vector bytes (#152's example).
    corrupt_first_entry_content(archive, 70);

    let dst = tmp.path().join("dst");
    std::fs::create_dir(&dst).unwrap();
    let dst = dst.to_str().expect("utf-8 temp path");
    fails(&["restore", "-i", archive, "--dir", dst, "--yes"], "");

    // Left unusable-or-absent: either no readable store at all, or one that itself
    // fails `stats` — never a silently-wrong-but-successful-looking store.
    let stats = run(&["stats", "--dir", dst], "");
    assert!(
        !stats.status.success(),
        "a corrupted restore must not leave a usable store: {:?}",
        String::from_utf8_lossy(&stats.stdout)
    );
}

/// Chopping the gzip trailer is a different failure mode than body corruption
/// (#138) — cover it separately so one fix cannot regress the other silently.
#[test]
fn verify_rejects_a_truncated_archive() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let src = src.to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", src, "--dim", "3", "docs"], "");
    ok(&["upsert", "--dir", src, "docs"], &seed());

    let archive = tmp.path().join("snapshot.tar.gz");
    let archive = archive.to_str().expect("utf-8 temp path");
    ok(&["backup", "--dir", src, "-o", archive], "");

    let mut bytes = std::fs::read(archive).unwrap();
    assert!(bytes.len() > 8, "archive too small to truncate");
    bytes.truncate(bytes.len() - 8);
    std::fs::write(archive, &bytes).unwrap();

    fails(&["verify", "-i", archive], "");
}

/// `backup --verify` on the happy path must still print a `BackupReport`, not switch
/// its output shape to a verify report just because the flag was passed.
#[test]
fn backup_verify_flag_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let src = src.to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", src, "--dim", "3", "docs"], "");
    ok(&["upsert", "--dir", src, "docs"], &seed());

    let archive = tmp.path().join("snapshot.tar.gz");
    let archive = archive.to_str().expect("utf-8 temp path");
    let report = ok(&["backup", "--dir", src, "-o", archive, "--verify"], "");
    assert_eq!(report["backup"], archive, "still a BackupReport: {report}");
    assert_eq!(report["dimension"], 3, "{report}");
    assert!(
        report["archive_bytes"].as_u64().unwrap_or(0) > 0,
        "{report}"
    );
}

/// #130's regression class was a sealed segment silently dropped from a backup;
/// `verify` must actually open a segmented store, not just a single-segment one.
#[test]
fn verify_accepts_a_segmented_store() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let src = src.to_str().expect("utf-8 temp path");
    ok(
        &[
            "create",
            "--dir",
            src,
            "--dim",
            "3",
            "--segment-max-rows",
            "2",
            "docs",
        ],
        "",
    );
    // A seal is only checked at the *start* of the next append (`maybe_seal`), so it
    // takes two calls past the threshold, not one call of 3 rows, to freeze a segment.
    let two = json!([
        {"id": "a", "vector": [1, 0, 0], "attrs": {}},
        {"id": "b", "vector": [0, 1, 0], "attrs": {}},
    ])
    .to_string();
    let one = json!([{"id": "c", "vector": [0, 0, 1], "attrs": {}}]).to_string();
    ok(
        &["upsert", "--dir", src, "--segment-max-rows", "2", "docs"],
        &two,
    );
    ok(
        &["upsert", "--dir", src, "--segment-max-rows", "2", "docs"],
        &one,
    );

    let archive = tmp.path().join("snapshot.tar.gz");
    let archive = archive.to_str().expect("utf-8 temp path");
    let backup_report = ok(&["backup", "--dir", src, "-o", archive], "");
    assert!(
        backup_report["segments"].as_u64().unwrap_or(0) > 0,
        "expected at least one sealed segment: {backup_report}"
    );

    let report = ok(&["verify", "-i", archive], "");
    assert_eq!(report["records"], 3, "{report}");
    assert_eq!(report["collections"], json!(["docs"]), "{report}");
}

// ── `nidus remember` / `nidus recall` (#134) ────────────────────────────────
// Gated like `memory_http.rs`: the mock embedder lives under the `mcp` module and
// needs `embed-ollama`, so these compile under `--features serve`, not plain `cli`.

/// The `--embed-*` flags pointing every memory subcommand at a mock embedder.
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
fn embed_args(url: &str) -> [String; 4] {
    [
        "--embed-provider".into(),
        "ollama".into(),
        "--embed-base-url".into(),
        url.into(),
    ]
}

/// Remember then recall, one process each, no server anywhere. The point of #134: a shell
/// hook can write a fact and read it back without a long-lived `nidus serve`.
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
#[test]
fn remember_and_recall_round_trip_through_the_binary() {
    use crate::mcp::support::{DIM, mock_embedder_per_text};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    let url = mock_embedder_per_text(DIM);
    let e = embed_args(&url);
    let (p, b) = (e[0].as_str(), e[1].as_str());
    let (u, v) = (e[2].as_str(), e[3].as_str());

    // No `--dim`: the store is created at the embedder's own dimension.
    let out = ok(
        &[
            "remember",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "--attrs",
            r#"{"tag": {"Str": "ops"}}"#,
            "notes",
            "the ranking bug is in the upsert path",
        ],
        "",
    );
    assert_eq!(out["collection"], "notes", "{out}");
    assert_eq!(
        out["dimension"], DIM,
        "dimension comes from the embedder: {out}"
    );
    assert_eq!(out["mode"], "Raw", "{out}");
    let derived = out["id"].as_str().expect("an id").to_string();
    assert!(derived.starts_with("mem-"), "derived id: {out}");

    let hits = ok(
        &[
            "recall",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "-k",
            "5",
            "notes",
            "the ranking bug",
        ],
        "",
    );
    assert_eq!(ids(&hits), [derived.as_str()], "{hits}");
    assert_eq!(hits[0]["attrs"]["tag"], json!({"Str": "ops"}), "{hits}");
    // The raw text is stamped, so a recall hit says what was remembered — and the FTS
    // schema `remember` declares over that field has something to index.
    assert_eq!(
        hits[0]["attrs"]["nidus.text"],
        json!({"Str": "the ranking bug is in the upsert path"}),
        "{hits}"
    );

    // Re-remembering the same text is content-addressed to the same id, so it replaces
    // that record rather than accumulating a near-duplicate beside it.
    let again = ok(
        &[
            "remember",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "notes",
            "the ranking bug is in the upsert path",
        ],
        "",
    );
    assert_eq!(again["id"], derived.as_str(), "content-addressed: {again}");
    let hits = ok(
        &[
            "recall",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "-k",
            "5",
            "notes",
            "the ranking bug",
        ],
        "",
    );
    assert_eq!(
        ids(&hits),
        [derived.as_str()],
        "one record, not two: {hits}"
    );

    let found = ok(
        &[
            "text-search",
            "--dir",
            dir,
            "-k",
            "5",
            "nidus.text",
            "ranking",
        ],
        "",
    );
    assert_eq!(
        ids(&found),
        [derived.as_str()],
        "remembered text is searchable: {found}"
    );

    // An explicit id wins over the derived one, and `--where` filters the recall.
    ok(
        &[
            "remember",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "--id",
            "manual",
            "--attrs",
            r#"{"tag": {"Str": "docs"}}"#,
            "notes",
            "the changelog lives in docs",
        ],
        "",
    );
    let hits = ok(
        &[
            "recall",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "-k",
            "5",
            "--where",
            r#"[{"Eq": ["tag", {"Str": "docs"}]}]"#,
            "notes",
            "changelog",
        ],
        "",
    );
    assert_eq!(ids(&hits), ["manual"], "filtered recall: {hits}");
}

/// The #133 knobs, through the binary: an expired write never surfaces on recall, and a
/// near-duplicate write redirects to the matched entry and says so in the output JSON.
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
#[test]
fn ttl_and_dedupe_flags_reach_the_memory_layer() {
    use crate::mcp::support::{DIM, mock_embedder_per_text};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    let url = mock_embedder_per_text(DIM);
    let e = embed_args(&url);
    let (p, b) = (e[0].as_str(), e[1].as_str());
    let (u, v) = (e[2].as_str(), e[3].as_str());

    // Expired the instant it is written; a live neighbour proves the store still answers.
    #[rustfmt::skip]
    ok(&["remember", "--dir", dir, p, b, u, v, "--id", "gone", "--ttl-seconds", "0",
         "notes", "ephemeral scratch note"], "");
    #[rustfmt::skip]
    ok(&["remember", "--dir", dir, p, b, u, v, "--id", "kept",
         "notes", "durable note"], "");

    #[rustfmt::skip]
    let hits = ok(&["recall", "--dir", dir, p, b, u, v, "-k", "5",
                    "notes", "ephemeral scratch note"], "");
    assert!(
        !ids(&hits).contains(&"gone".to_string()),
        "expired entry leaked: {hits}"
    );
    #[rustfmt::skip]
    let hits = ok(&["recall", "--dir", dir, p, b, u, v, "-k", "5",
                    "notes", "durable note"], "");
    assert!(
        ids(&hits).contains(&"kept".to_string()),
        "no-TTL entry must surface: {hits}"
    );

    // A near-duplicate with the flag lands on the existing entry instead of a rival.
    #[rustfmt::skip]
    let out = ok(&["remember", "--dir", dir, p, b, u, v, "--id", "rival",
                   "--dedupe-threshold", "0.95", "notes", "durable note"], "");
    assert_eq!(out["deduped"], true, "{out}");
    assert_eq!(out["id"], "kept", "write must redirect to the match: {out}");
}

/// `recall` opens read-only, so it answers against a directory a live server holds — the
/// property that makes it usable from a shell hook on a machine already running nidus.
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
#[test]
fn recall_runs_against_a_dir_a_server_holds() {
    use crate::mcp::support::{DIM, per_text_embedder_server};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    let url = crate::mcp::support::mock_embedder_per_text(DIM);
    let e = embed_args(&url);
    let (p, b) = (e[0].as_str(), e[1].as_str());
    let (u, v) = (e[2].as_str(), e[3].as_str());

    // Seed before the server takes the writer lock.
    ok(
        &[
            "remember",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "--id",
            "seeded",
            "notes",
            "deploys run at noon",
        ],
        "",
    );

    let server = per_text_embedder_server(tmp.path(), DIM);
    assert_eq!(server.get("/health").0, 200);

    let hits = ok(
        &[
            "recall", "--dir", dir, p, b, u, v, "-k", "5", "notes", "deploys",
        ],
        "",
    );
    assert_eq!(
        ids(&hits),
        ["seeded"],
        "read-only recall beside a writer: {hits}"
    );

    // `remember`, by contrast, wants the lock the server is holding.
    let err = fails(
        &["remember", "--dir", dir, p, b, u, v, "notes", "a new fact"],
        "",
    );
    assert!(err.contains("locked"), "{err}");
}

/// `--reinforce` opens ReadWrite, so unlike a plain recall it collides with a server
/// holding the writer lock — the new failure mode this unit introduces. It must fail
/// loudly, naming the lock, not silently skip the stamp.
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
#[test]
fn reinforce_recall_refuses_while_a_server_holds_the_writer_lock() {
    use crate::mcp::support::{DIM, per_text_embedder_server};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    let url = crate::mcp::support::mock_embedder_per_text(DIM);
    let e = embed_args(&url);
    let (p, b) = (e[0].as_str(), e[1].as_str());
    let (u, v) = (e[2].as_str(), e[3].as_str());

    ok(
        &[
            "remember",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "--id",
            "seeded",
            "notes",
            "deploys run at noon",
        ],
        "",
    );

    let server = per_text_embedder_server(tmp.path(), DIM);
    assert_eq!(server.get("/health").0, 200);

    let err = fails(
        &[
            "recall",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "--reinforce",
            "-k",
            "5",
            "notes",
            "deploys",
        ],
        "",
    );
    assert!(err.contains("locked"), "{err}");
}

/// With nothing holding the writer lock, `--reinforce` stamps `nidus.access_count`
/// durably; a second reinforced recall bumps it again, proving the stamp lands on disk
/// rather than only in the printed hits (which `recall` would show either way).
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
#[test]
fn reinforce_recall_stamps_when_nothing_holds_the_lock() {
    use crate::mcp::support::DIM;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    let url = crate::mcp::support::mock_embedder_per_text(DIM);
    let e = embed_args(&url);
    let (p, b) = (e[0].as_str(), e[1].as_str());
    let (u, v) = (e[2].as_str(), e[3].as_str());

    ok(
        &[
            "remember",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "--id",
            "seeded",
            "notes",
            "deploys run at noon",
        ],
        "",
    );

    ok(
        &[
            "recall",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "--reinforce",
            "-k",
            "5",
            "notes",
            "deploys",
        ],
        "",
    );
    let recs = ok(&["get", "--dir", dir, "notes"], "");
    let seeded = recs
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == "seeded")
        .expect("seeded record");
    assert_eq!(
        seeded["attrs"]["nidus.access_count"],
        json!({"Int": 1}),
        "{recs}"
    );

    ok(
        &[
            "recall",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "--reinforce",
            "-k",
            "5",
            "notes",
            "deploys",
        ],
        "",
    );
    let recs = ok(&["get", "--dir", dir, "notes"], "");
    let seeded = recs
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == "seeded")
        .expect("seeded record");
    assert_eq!(
        seeded["attrs"]["nidus.access_count"],
        json!({"Int": 2}),
        "{recs}"
    );
}

/// `--read-only` and `--reinforce` together are refused before the store is even opened
/// (the existing mutating-command guard), not left to fail obscurely once reinforcement
/// tries to write.
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
#[test]
fn read_only_plus_reinforce_is_refused_before_opening() {
    use crate::mcp::support::DIM;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    let url = crate::mcp::support::mock_embedder_per_text(DIM);
    let e = embed_args(&url);
    let (p, b) = (e[0].as_str(), e[1].as_str());
    let (u, v) = (e[2].as_str(), e[3].as_str());

    ok(
        &[
            "remember",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "--id",
            "seeded",
            "notes",
            "deploys run at noon",
        ],
        "",
    );

    let err = fails(
        &[
            "recall",
            "--dir",
            dir,
            "--read-only",
            p,
            b,
            u,
            v,
            "--reinforce",
            "-k",
            "5",
            "notes",
            "deploys",
        ],
        "",
    );
    assert!(
        err.contains("--read-only was set, but this command mutates the store"),
        "{err}"
    );
}

/// A plain recall (no `--reinforce`) stamps nothing — the unreinforced behaviour this
/// unit must leave unchanged.
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
#[test]
fn a_plain_recall_still_stamps_nothing() {
    use crate::mcp::support::DIM;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    let url = crate::mcp::support::mock_embedder_per_text(DIM);
    let e = embed_args(&url);
    let (p, b) = (e[0].as_str(), e[1].as_str());
    let (u, v) = (e[2].as_str(), e[3].as_str());

    ok(
        &[
            "remember",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "--id",
            "seeded",
            "notes",
            "deploys run at noon",
        ],
        "",
    );
    ok(
        &[
            "recall", "--dir", dir, p, b, u, v, "-k", "5", "notes", "deploys",
        ],
        "",
    );

    let recs = ok(&["get", "--dir", dir, "notes"], "");
    let seeded = recs
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == "seeded")
        .expect("seeded record");
    assert!(
        seeded["attrs"].get("nidus.access_count").is_none(),
        "{recs}"
    );
}

/// Both subcommands need an embedder, and the refusal must name the flag that supplies
/// one — the store is otherwise perfectly openable, so the error is the only signal.
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
#[test]
fn memory_subcommands_without_an_embedder_name_the_flag() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");

    for args in [
        vec!["remember", "--dir", dir, "--dim", "3", "notes", "a fact"],
        vec!["recall", "--dir", dir, "--dim", "3", "notes", "a fact"],
    ] {
        let err = fails(&args, "");
        assert!(
            err.contains("--embed-provider"),
            "{:?} should name the flag: {err}",
            args[0]
        );
    }
}

/// `compact --expired` through the real binary. No embedder needed: `nidus.expires_at`
/// is a plain attr an ordinary upsert can write, which is why the sweep is not behind
/// the `memory` feature.
#[test]
fn compact_expired_reclaims_only_past_entries() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");

    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    let records = json!([
        {"id": "past", "vector": [1, 0, 0],
         "attrs": {"nidus.expires_at": {"DateTime": 1_000}}},
        {"id": "future", "vector": [0, 1, 0],
         "attrs": {"nidus.expires_at": {"DateTime": 32_503_680_000_000i64}}},
        {"id": "never", "vector": [0, 0, 1], "attrs": {}}
    ])
    .to_string();
    assert_eq!(
        ok(&["upsert", "--dir", dir, "docs"], &records)["upserted"],
        3
    );

    let out = ok(&["compact", "--dir", dir, "--expired"], "");
    assert_eq!(out["swept"], 1, "only the past-expiry entry: {out}");

    let remaining = ok(&["list", "--dir", dir, "docs"], "");
    let mut ids: Vec<&str> = remaining
        .as_array()
        .expect("a list")
        .iter()
        .map(|h| h["id"].as_str().expect("an id"))
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, ["future", "never"], "{remaining}");

    let stats = ok(&["stats", "--dir", dir], "");
    assert_eq!(stats["footprint"]["dead_rows"], 0, "swept: {stats}");
}

// ── `--at-version` / `--history-versions` / `nidus versions` (nidus-bnf) ───

/// `--at-version` serves the older snapshot: it sees what existed at the pinned commit
/// and nothing committed after, even a since-deleted record that was still live then.
#[test]
fn at_version_serves_older_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");

    ok(
        &[
            "create",
            "--dir",
            dir,
            "--dim",
            "3",
            "--history-versions",
            "5",
            "docs",
        ],
        "",
    );
    let ab = json!([
        {"id": "a", "vector": [1, 0, 0], "attrs": {}},
        {"id": "b", "vector": [0, 1, 0], "attrs": {}}
    ])
    .to_string();
    ok(
        &["upsert", "--dir", dir, "--history-versions", "5", "docs"],
        &ab,
    );

    let v = ok(&["versions", "--dir", dir], "")["commit_version"]
        .as_u64()
        .expect("commit_version");

    let c = json!([{"id": "c", "vector": [0, 0, 1], "attrs": {}}]).to_string();
    ok(
        &["upsert", "--dir", dir, "--history-versions", "5", "docs"],
        &c,
    );
    ok(&["delete", "--dir", dir, "docs", "a"], "");

    let hits = ok(
        &[
            "search",
            "--dir",
            dir,
            "--at-version",
            &v.to_string(),
            "docs",
        ],
        "[1, 1, 1]",
    );
    let mut got = ids(&hits);
    got.sort_unstable();
    assert_eq!(got, ["a", "b"], "{hits}");
}

/// A write subcommand against `--at-version` fails at the mutation itself, naming
/// read-only — never a silent downgrade to a read.
#[test]
fn at_version_on_write_subcommand_fails_read_only() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");

    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    let v = ok(&["versions", "--dir", dir], "")["commit_version"]
        .as_u64()
        .expect("commit_version");

    let record = json!([{"id": "a", "vector": [1, 0, 0], "attrs": {}}]).to_string();
    let err = fails(
        &[
            "upsert",
            "--dir",
            dir,
            "--at-version",
            &v.to_string(),
            "docs",
        ],
        &record,
    );
    assert!(
        err.to_lowercase().contains("read-only"),
        "expected a read-only message: {err}"
    );
}

/// `nidus versions` reports the current commit version, and `null` for `pinned` on an
/// unpinned handle, and the pin itself once `--at-version` names a recorded one.
#[test]
fn versions_reports_commit_version_and_pinned() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");

    ok(
        &[
            "create",
            "--dir",
            dir,
            "--dim",
            "3",
            "--history-versions",
            "5",
            "docs",
        ],
        "",
    );
    let out = ok(&["versions", "--dir", dir], "");
    let v = out["commit_version"].as_u64().expect("commit_version");
    assert!(out["pinned"].is_null(), "{out}");

    let pinned = ok(
        &["versions", "--dir", dir, "--at-version", &v.to_string()],
        "",
    );
    assert_eq!(pinned["pinned"], v, "{pinned}");
}

/// Asking for a version whose segments a compaction already reclaimed fails loudly,
/// naming the oldest version still readable — not just a bare non-zero exit.
#[test]
fn at_version_after_compaction_names_oldest_readable() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");

    ok(
        &[
            "create",
            "--dir",
            dir,
            "--dim",
            "3",
            "--history-versions",
            "5",
            "docs",
        ],
        "",
    );
    let record = json!([{"id": "a", "vector": [1, 0, 0], "attrs": {}}]).to_string();
    ok(
        &["upsert", "--dir", dir, "--history-versions", "5", "docs"],
        &record,
    );
    let v = ok(&["versions", "--dir", dir], "")["commit_version"]
        .as_u64()
        .expect("commit_version");

    ok(&["compact", "--dir", dir], "");
    let oldest = ok(&["versions", "--dir", dir], "")["oldest_readable"]
        .as_u64()
        .expect("oldest_readable after a compaction that reclaimed history");

    let err = fails(
        &[
            "search",
            "--dir",
            dir,
            "--at-version",
            &v.to_string(),
            "docs",
        ],
        "[1, 0, 0]",
    );
    assert!(
        err.contains(&format!("oldest readable version is {oldest}")),
        "expected the oldest readable version named: {err}"
    );
}

// ── `nidus text-search --rerank` (nidus-d42) ────────────────────────────────

/// `--rerank` reorders `text-search`'s BM25 ranking through the real `--rerank-provider`
/// wiring (the same mock `tests/e2e/rerank.rs` uses): identical bodies tie the baseline,
/// so the inverting mock's reversal cannot coincide with it by chance.
#[cfg(all(feature = "mcp", feature = "embed-ollama", feature = "rerank-cohere"))]
#[test]
fn text_search_rerank_reorders_against_the_baseline() {
    use crate::mcp::support::mock_reranker_inverting;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");

    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    let records = json!([
        {"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "cat"}}},
        {"id": "b", "vector": [0, 1, 0], "attrs": {"body": {"Str": "cat"}}},
        {"id": "c", "vector": [0, 0, 1], "attrs": {"body": {"Str": "cat"}}}
    ])
    .to_string();
    assert_eq!(
        ok(&["upsert", "--dir", dir, "docs"], &records)["upserted"],
        3
    );
    ok(
        &["set-fts-schema", "--dir", dir, "--field", "body", "docs"],
        "",
    );

    let baseline = ids(&ok(
        &["text-search", "--dir", dir, "-k", "5", "body", "cat"],
        "",
    ));
    assert_eq!(baseline.len(), 3, "{baseline:?}");

    let rerank_url = mock_reranker_inverting();
    let reranked = ids(&ok(
        &[
            "text-search",
            "--dir",
            dir,
            "-k",
            "5",
            "body",
            "cat",
            "--rerank",
            "--rerank-provider",
            "cohere",
            "--rerank-api-key",
            "mock-key",
            "--rerank-base-url",
            &rerank_url,
            "--rerank-text-attr",
            "body",
        ],
        "",
    ));
    assert_eq!(reranked.len(), 3, "{reranked:?}");
    assert_ne!(
        reranked, baseline,
        "--rerank must reorder against the no-rerank baseline"
    );
}

/// `recall --rerank` is a wholly separate ~95-line path (embed, cross-model identity guard,
/// TTL filter, rerank), so it needs its own proof: it must reorder AND still hide an expired
/// memory, which is the guard most easily lost when a path is re-implemented.
#[cfg(all(feature = "mcp", feature = "embed-ollama", feature = "rerank-cohere"))]
#[test]
fn recall_rerank_reorders_and_still_hides_expired_memories() {
    use crate::mcp::support::{DIM, mock_embedder_per_text, mock_reranker_inverting};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    let url = mock_embedder_per_text(DIM);
    let e = embed_args(&url);
    let (p, b) = (e[0].as_str(), e[1].as_str());
    let (u, v) = (e[2].as_str(), e[3].as_str());

    for text in [
        "the ranking bug is in the upsert path",
        "ranking is documented in the guide",
        "ranking performance over a large corpus",
    ] {
        ok(&["remember", "--dir", dir, p, b, u, v, "notes", text], "");
    }
    // Already expired: a negative TTL puts the deadline in the past.
    let expired = ok(
        &[
            "remember",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "--ttl-seconds=-3600",
            "notes",
            "ranking secret that must never surface",
        ],
        "",
    );
    let expired_id = expired["id"].as_str().expect("an id").to_string();

    let baseline = ids(&ok(
        &[
            "recall", "--dir", dir, p, b, u, v, "-k", "5", "notes", "ranking",
        ],
        "",
    ));
    assert_eq!(
        baseline.len(),
        3,
        "the expired memory must be hidden: {baseline:?}"
    );

    let rerank_url = mock_reranker_inverting();
    let reranked = ids(&ok(
        &[
            "recall",
            "--dir",
            dir,
            p,
            b,
            u,
            v,
            "-k",
            "5",
            "--rerank",
            "--rerank-provider",
            "cohere",
            "--rerank-api-key",
            "mock-key",
            "--rerank-base-url",
            &rerank_url,
            "notes",
            "ranking",
        ],
        "",
    ));
    assert_ne!(
        reranked, baseline,
        "recall --rerank must reorder against its baseline"
    );
    assert!(
        !reranked.contains(&expired_id),
        "the TTL guard must survive the reranked path: {reranked:?}"
    );
}

/// The vector-query legs of the same wiring. `search` and `hybrid-search` carry no text of
/// their own, so both require an explicit `--rerank-query`; each must reorder its own
/// baseline through the real flags, which no other test covers.
#[cfg(all(feature = "mcp", feature = "embed-ollama", feature = "rerank-cohere"))]
#[test]
fn search_and_hybrid_search_rerank_reorder_against_their_baselines() {
    use crate::mcp::support::mock_reranker_inverting;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    let records = json!([
        {"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "cat"}}},
        {"id": "b", "vector": [0, 1, 0], "attrs": {"body": {"Str": "cat"}}},
        {"id": "c", "vector": [0, 0, 1], "attrs": {"body": {"Str": "cat"}}}
    ])
    .to_string();
    assert_eq!(
        ok(&["upsert", "--dir", dir, "docs"], &records)["upserted"],
        3
    );
    ok(
        &["set-fts-schema", "--dir", dir, "--field", "body", "docs"],
        "",
    );
    let rerank_url = mock_reranker_inverting();
    let vector = "[1, 1, 1]";

    let baseline = ids(&ok(&["search", "--dir", dir, "-k", "5"], vector));
    assert_eq!(baseline.len(), 3, "{baseline:?}");
    let reranked = ids(&ok(
        &[
            "search",
            "--dir",
            dir,
            "-k",
            "5",
            "--rerank",
            "--rerank-query",
            "cat",
            "--rerank-provider",
            "cohere",
            "--rerank-api-key",
            "mock-key",
            "--rerank-base-url",
            &rerank_url,
            "--rerank-text-attr",
            "body",
        ],
        vector,
    ));
    assert_ne!(
        reranked, baseline,
        "search --rerank must reorder against its baseline"
    );

    let hybrid_baseline = ids(&ok(
        &["hybrid-search", "--dir", dir, "-k", "5", "body", "cat"],
        vector,
    ));
    assert_eq!(hybrid_baseline.len(), 3, "{hybrid_baseline:?}");
    let hybrid_reranked = ids(&ok(
        &[
            "hybrid-search",
            "--dir",
            dir,
            "-k",
            "5",
            "body",
            "cat",
            "--rerank",
            "--rerank-query",
            "cat",
            "--rerank-provider",
            "cohere",
            "--rerank-api-key",
            "mock-key",
            "--rerank-base-url",
            &rerank_url,
            "--rerank-text-attr",
            "body",
        ],
        vector,
    ));
    assert_ne!(
        hybrid_reranked, hybrid_baseline,
        "hybrid-search --rerank must reorder against its baseline"
    );
}

/// A raw vector query has no text to fall back on, so `--rerank` without `--rerank-query`
/// must refuse rather than rerank against an empty string.
#[cfg(feature = "rerank")]
#[test]
fn search_rerank_without_a_query_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    let err = fails(&["search", "--dir", dir, "--rerank"], "[1, 0, 0]");
    assert!(err.contains("--rerank-query"), "{err}");
}

/// `--rerank` with no `--rerank-provider` is a clear, nonzero-exit error naming the flag —
/// never a silent, un-reranked success.
#[cfg(feature = "rerank")]
#[test]
fn text_search_rerank_without_a_provider_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    ok(
        &["upsert", "--dir", dir, "docs"],
        &json!([{"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "cat"}}}]).to_string(),
    );
    ok(
        &["set-fts-schema", "--dir", dir, "--field", "body", "docs"],
        "",
    );

    let err = fails(
        &["text-search", "--dir", dir, "body", "cat", "--rerank"],
        "",
    );
    assert!(err.contains("--rerank-provider"), "{err}");
}

/// A crowded corpus: three near-copies of the query direction plus one outlier that scores
/// lower. Written through the real binary's stdin, the shape `--diversity` has to reshape.
fn crowded_seed() -> String {
    json!([
        {"id": "dup0", "vector": [1, 0.02, 0], "attrs": {}},
        {"id": "dup1", "vector": [1, 0.03, 0], "attrs": {}},
        {"id": "dup2", "vector": [1, 0.04, 0], "attrs": {}},
        {"id": "novel", "vector": [0.6, 0.8, 0], "attrs": {}}
    ])
    .to_string()
}

/// `--diversity` must change which ids come back, not merely be accepted. A flag that
/// parses and does nothing passes any test that only checks the command succeeded.
#[test]
fn cli_diversity_changes_the_returned_ids() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().unwrap();
    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    ok(&["upsert", "--dir", dir, "docs"], &crowded_seed());

    let ids = |args: &[&str]| -> Vec<String> {
        ok(args, "[1, 0, 0]")
            .as_array()
            .expect("search prints an array")
            .iter()
            .map(|h| h["id"].as_str().expect("id").to_string())
            .collect()
    };
    assert_eq!(
        ids(&["search", "--dir", dir, "-k", "2", "docs"]),
        ["dup0", "dup1"]
    );
    assert_eq!(
        ids(&[
            "search",
            "--dir",
            dir,
            "-k",
            "2",
            "--diversity",
            "0.3",
            "docs"
        ]),
        ["dup0", "novel"],
        "--diversity did not reshape the page"
    );

    // Out of range is a caller error, refused rather than clamped.
    let err = fails(
        &[
            "search",
            "--dir",
            dir,
            "-k",
            "2",
            "--diversity",
            "2.0",
            "docs",
        ],
        "[1, 0, 0]",
    );
    assert!(err.contains("diversity"), "{err}");
}

/// `--diversity` on `similar`, through the binary. The source is excluded, so the crowding
/// this has to break up is among the remaining near-copies.
#[test]
fn cli_diversity_reshapes_a_similar_search() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().unwrap();
    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    ok(&["upsert", "--dir", dir, "docs"], &crowded_seed());

    let ids = |args: &[&str]| -> Vec<String> {
        ok(args, "")
            .as_array()
            .expect("similar prints an array")
            .iter()
            .map(|h| h["id"].as_str().expect("id").to_string())
            .collect()
    };
    assert_eq!(
        ids(&["similar", "--dir", dir, "-k", "2", "docs", "dup0"]),
        ["dup1", "dup2"]
    );
    assert_eq!(
        ids(&[
            "similar",
            "--dir",
            dir,
            "-k",
            "2",
            "--diversity",
            "0.3",
            "docs",
            "dup0",
        ]),
        ["dup1", "novel"],
        "--diversity did not reshape the similar page"
    );
}

/// `--diversity` on `text-search`, through the binary. The ranking is BM25 here, not cosine,
/// but redundancy is still measured in vector space, which is the whole point of the knob.
#[test]
fn cli_diversity_reshapes_a_text_search() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().unwrap();
    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    ok(
        &["set-fts-schema", "--dir", dir, "docs", "--field", "body"],
        "",
    );
    // `t0`/`t1` share a vector, so they are redundant with each other; `t2` is orthogonal and
    // ranks last on BM25 alone (one term in the longest body).
    let seed = json!([
        {"id": "t0", "vector": [1, 0, 0], "attrs": {"body": {"Str": "alpha alpha alpha"}}},
        {"id": "t1", "vector": [1, 0, 0], "attrs": {"body": {"Str": "alpha alpha"}}},
        {"id": "t2", "vector": [0, 1, 0], "attrs": {"body": {"Str": "alpha beta gamma"}}}
    ])
    .to_string();
    ok(&["upsert", "--dir", dir, "docs"], &seed);

    let ids = |args: &[&str]| -> Vec<String> {
        ok(args, "")
            .as_array()
            .expect("text-search prints an array")
            .iter()
            .map(|h| h["id"].as_str().expect("id").to_string())
            .collect()
    };
    assert_eq!(
        ids(&[
            "text-search",
            "--dir",
            dir,
            "--in",
            "docs",
            "-k",
            "2",
            "body",
            "alpha",
        ]),
        ["t0", "t1"]
    );
    assert_eq!(
        ids(&[
            "text-search",
            "--dir",
            dir,
            "--in",
            "docs",
            "-k",
            "2",
            "--diversity",
            "0.3",
            "body",
            "alpha",
        ]),
        ["t0", "t2"],
        "--diversity did not reshape the text-search page"
    );
}

/// `text-search`'s ranking and projection knobs through the real binary: the same
/// `--limit-per`/`--rank-by`/`--include-attr` the route has always carried (nidus-33g).
#[test]
fn cli_text_search_ranking_knobs_reshape_the_page() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().unwrap();
    ok(&["create", "--dir", dir, "--dim", "3", "docs"], "");
    ok(
        &["set-fts-schema", "--dir", dir, "docs", "--field", "body"],
        "",
    );
    // BM25 orders these d1 > d2 > d3 > d4 on "alpha" (falling tf, rising length); three of
    // the four share one `file`, and only d1 is old enough for the decay below to bury.
    let seed = json!([
        {"id": "d1", "vector": [1, 0, 0], "attrs": {
            "body": {"Str": "alpha alpha alpha"}, "file": {"Str": "a.md"}, "ts": {"Int": 0}}},
        {"id": "d2", "vector": [0, 1, 0], "attrs": {
            "body": {"Str": "alpha alpha"}, "file": {"Str": "a.md"}, "ts": {"Int": 1000000}}},
        {"id": "d3", "vector": [0, 0, 1], "attrs": {
            "body": {"Str": "alpha"}, "file": {"Str": "a.md"}, "ts": {"Int": 1000000}}},
        {"id": "d4", "vector": [1, 1, 0], "attrs": {
            "body": {"Str": "alpha beta"}, "file": {"Str": "b.md"}, "ts": {"Int": 1000000}}}
    ])
    .to_string();
    ok(&["upsert", "--dir", dir, "docs"], &seed);

    let base = ["text-search", "--dir", dir, "--in", "docs", "-k", "4"];
    let query = ["body", "alpha"];
    let hits = |extra: &[&str]| -> Value {
        let mut args: Vec<&str> = base.to_vec();
        args.extend_from_slice(extra);
        args.extend_from_slice(&query);
        ok(&args, "")
    };

    assert_eq!(ids(&hits(&[])), ["d1", "d2", "d3", "d4"]);

    // One hit per `file`, so two of the three a.md docs are dropped even though k=4.
    assert_eq!(
        ids(&hits(&["--limit-per", "file", "--limit-per-max", "1"])),
        ["d1", "d4"],
        "--limit-per did not cap the text-search page"
    );
    assert_eq!(
        ids(&hits(&["--limit-per", "file", "--limit-per-max", "2"])),
        ["d1", "d2", "d4"],
        "--limit-per-max 2 kept the wrong number per file"
    );

    // A half-life of 1s with a lambda far above any BM25 gap here buries d1 outright.
    let decayed = hits(&[
        "--rank-by",
        r#"{"Decay":{"field":"ts","origin":1000000,"scale":1000,"lambda":100.0}}"#,
    ]);
    assert_eq!(
        ids(&decayed).last().map(String::as_str),
        Some("d1"),
        "--rank-by did not apply the recency penalty"
    );

    let projected = hits(&["--include-attr", "file"]);
    let attrs = &projected[0]["attrs"];
    assert_eq!(attrs["file"], json!({"Str": "a.md"}));
    assert!(
        attrs.get("body").is_none() && attrs.get("ts").is_none(),
        "--include-attr did not drop the other attrs: {attrs}"
    );

    // Both projection sides at once is an error, not a precedence rule.
    let err = fails(
        &[
            "text-search",
            "--dir",
            dir,
            "--in",
            "docs",
            "--include-attr",
            "file",
            "--exclude-attr",
            "body",
            "body",
            "alpha",
        ],
        "",
    );
    assert!(
        err.contains("include") || err.contains("exclude"),
        "unhelpful projection error: {err}"
    );
}

/// `recall --diversity` through the binary. `remember` pins the collection and provisions the
/// store at the embedder's dimension; the crowded corpus is then written as raw vectors built
/// in the query's own embedding space, so which hits are redundant is computed, not guessed.
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
#[test]
fn cli_diversity_reshapes_a_recall() {
    use crate::mcp::support::{DIM, mock_embedder_per_text, vector_for};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    let url = mock_embedder_per_text(DIM);
    let e = embed_args(&url);
    let (p, b) = (e[0].as_str(), e[1].as_str());
    let (u, v) = (e[2].as_str(), e[3].as_str());

    // `remember` provisions the collection at the embedder's dimension and pins its identity,
    // so the raw-vector upsert below lands in a store `recall` will accept.
    let seeded = ok(&["remember", "--dir", dir, p, b, u, v, "notes", "seed"], "");
    // Remove it by its derived id, not a glob: `delete` takes literal ids, so a pattern would
    // silently delete nothing and leave the seed competing in the ranking below.
    let seeded_id = seeded["id"].as_str().expect("a derived id").to_string();
    let removed = ok(&["delete", "--dir", dir, "notes", &seeded_id], "");
    // Two assertions on purpose: the count catches a delete that matched nothing at the line
    // it happened, and the emptiness check catches the state rather than the operation.
    assert_eq!(removed["deleted"], 1, "{removed}");
    assert!(
        ids(&ok(&["list", "--dir", dir, "notes"], "")).is_empty(),
        "the seed must be gone before the corpus is written"
    );

    let dir_vec = unit_vec(vector_for("query", DIM));
    let side = orthogonal_unit_vec(&dir_vec);
    let mix =
        |a: f32, c: f32| -> Vec<f32> { (0..DIM).map(|i| a * dir_vec[i] + c * side[i]).collect() };
    let seed = json!([
        {"id": "dup0", "vector": mix(1.0, 0.0), "attrs": {"nidus.text": {"Str": "alpha"}}},
        {"id": "dup1", "vector": mix(0.9999, 0.0141), "attrs": {"nidus.text": {"Str": "alpha again"}}},
        {"id": "novel", "vector": mix(0.6, 0.8), "attrs": {"nidus.text": {"Str": "different"}}}
    ])
    .to_string();
    ok(&["upsert", "--dir", dir, "notes"], &seed);

    let recalled = |extra: &[&str]| -> Vec<String> {
        let mut args = vec!["recall", "--dir", dir, p, b, u, v, "-k", "2"];
        args.extend_from_slice(extra);
        args.extend_from_slice(&["notes", "query"]);
        ids(&ok(&args, ""))
    };
    assert_eq!(recalled(&[]), ["dup0", "dup1"]);
    assert_eq!(
        recalled(&["--diversity", "0.3"]),
        ["dup0", "novel"],
        "--diversity did not reshape the recall window"
    );
    // `RecallOpts` uses zero as its "unset" sentinel for `top_k`/`min_score`; a real zero
    // lambda must not be swallowed by that convention.
    assert_eq!(recalled(&["--diversity", "0"]), ["dup0", "novel"]);
}

/// `v` scaled to unit length; a zero vector is returned unchanged, as the store does.
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
fn unit_vec(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

/// A unit vector orthogonal to `u`, for building a corpus at a known cosine.
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
fn orthogonal_unit_vec(u: &[f32]) -> Vec<f32> {
    let pick = if u[0].abs() < 0.9 { 0 } else { 1 };
    let mut e = vec![0.0f32; u.len()];
    e[pick] = 1.0;
    let dot: f32 = u.iter().zip(&e).map(|(a, b)| a * b).sum();
    for (i, x) in e.iter_mut().enumerate() {
        *x -= dot * u[i];
    }
    unit_vec(e)
}
