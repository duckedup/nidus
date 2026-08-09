//! #141: a store's open profile (ann/quantization/query_threads/mmap), once recorded by
//! `nidus configure`, is inherited by every later open — `nidus serve` included, with no
//! flags repeated. Only a real spawn proves this: the profile lives in the on-disk manifest
//! and is merged into `Config` inside `Store::open`, which an in-process `tower::oneshot`
//! test never drives — it builds its `Config` directly and skips that merge entirely.

use std::process::{Command, Stdio};

use crate::harness::Server;

/// Run `nidus <args>` to completion (no stdin), returning `(stdout, stderr)` on success and
/// panicking with both streams on failure — the CLI counterpart to `Server` for subcommands
/// that don't bind a port.
fn nidus_ok(args: &[&str]) -> (String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_nidus"))
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear()
        .envs(std::env::vars().filter(|(k, _)| !k.starts_with("NIDUS_")))
        .output()
        .unwrap_or_else(|e| panic!("spawn nidus {args:?}: {e}"));
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "nidus {args:?} exited {:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        out.status.code()
    );
    (stdout, stderr)
}

/// `nidus configure --dir <dir> --dim <dim> <extra>`, e.g. `["--ann", "hnsw"]`.
fn configure(dir: &std::path::Path, dim: usize, extra: &[&str]) {
    let dir = dir.to_str().expect("utf-8 temp path");
    let mut args = vec!["configure", "--dir", dir, "--dim"];
    let dim_s = dim.to_string();
    args.push(&dim_s);
    args.extend_from_slice(extra);
    nidus_ok(&args);
}

/// The headline claim: `configure --ann hnsw` once, then `serve` with no `--ann` at all,
/// still comes up with HNSW active. An in-process test cannot show it, having built its
/// `Config` directly rather than through the merge in a real `Store::open`.
#[test]
fn configured_ann_is_inherited_by_a_bare_serve() {
    let dir = tempfile::tempdir().unwrap();
    configure(dir.path(), 3, &["--ann", "hnsw"]);

    let server = Server::new(dir.path(), 3).start();
    let (status, stats) = server.get("/stats");
    assert_eq!(status, 200, "stats: {stats}");
    assert_eq!(
        stats["ann"]["kind"], "Hnsw",
        "a recorded profile should reach a bare serve with no --ann: {stats}"
    );
}

/// An explicit flag on the later command still wins over the recorded default for that
/// knob — the profile fills gaps, it does not shadow what the caller actually typed.
#[test]
fn an_explicit_flag_still_overrides_the_recorded_profile() {
    let dir = tempfile::tempdir().unwrap();
    configure(dir.path(), 3, &["--ann", "hnsw"]);

    let server = Server::new(dir.path(), 3).args(["--ann", "ivf"]).start();
    let (status, stats) = server.get("/stats");
    assert_eq!(status, 200, "stats: {stats}");
    assert_eq!(
        stats["ann"]["kind"], "Ivf",
        "an explicit --ann should beat the recorded hnsw default: {stats}"
    );
}

/// `configure --clear` removes the recorded profile, so a later bare open falls all the way
/// back to exact brute-force search, exactly as if nothing had ever been configured.
#[test]
fn a_cleared_profile_falls_back_to_exact_search() {
    let dir = tempfile::tempdir().unwrap();
    configure(dir.path(), 3, &["--ann", "hnsw"]);
    configure(dir.path(), 3, &["--clear"]);

    let server = Server::new(dir.path(), 3).start();
    let (status, stats) = server.get("/stats");
    assert_eq!(status, 200, "stats: {stats}");
    assert_eq!(
        stats["ann"],
        serde_json::Value::Null,
        "a cleared profile should leave exact search active: {stats}"
    );
}

/// The ticket title names "forgetting the flag on one command" — not just `serve`. A plain
/// `nidus search` in a second process, with no `--ann` anywhere on its command line, must
/// still open against the configured profile and succeed.
#[test]
fn a_plain_search_inherits_the_configured_profile_too() {
    let dir = tempfile::tempdir().unwrap();
    configure(dir.path(), 3, &["--ann", "hnsw"]);

    let dir_s = dir.path().to_str().expect("utf-8 temp path");
    let out = Command::new(env!("CARGO_BIN_EXE_nidus"))
        .args(["search", "--dir", dir_s])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear()
        .envs(std::env::vars().filter(|(k, _)| !k.starts_with("NIDUS_")))
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .expect("stdin piped")
                .write_all(b"[1, 0, 0]")?;
            child.wait_with_output()
        })
        .expect("run nidus search");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "a plain `nidus search` against a configured store should succeed, got {:?}\n\
         --- stderr ---\n{stderr}",
        out.status.code()
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let _: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("nidus search printed non-JSON: {e}\n--- stdout ---\n{stdout}"));

    // Succeeding proves the store opened, not that the profile applied: `nidus search` has no
    // profile-dependent output, so this half would stay green with the merge ripped out.
    let (stats_out, _) = nidus_ok(&["stats", "--dir", dir_s]);
    let stats: serde_json::Value = serde_json::from_str(&stats_out).expect("stats json");
    assert_eq!(
        stats["ann"]["kind"], "Hnsw",
        "a second process with no flags must resolve the recorded profile: {stats}"
    );
}

/// `ann` alone is the only knob `/stats` used to expose, so a test asserting only it would
/// prove a quarter of the ticket. Record all four, open bare, and read every one back.
#[test]
fn all_four_recorded_knobs_reach_a_bare_serve() {
    let dir = tempfile::tempdir().unwrap();
    configure(
        dir.path(),
        3,
        &[
            "--ann",
            "hnsw",
            "--quantization",
            "int8",
            "--query-threads",
            "4",
            "--mmap",
            "--segment-max-rows",
            "1000",
        ],
    );

    let server = Server::new(dir.path(), 3).start();
    let (status, stats) = server.get("/stats");
    assert_eq!(status, 200, "stats: {stats}");
    assert_eq!(stats["ann"]["kind"], "Hnsw", "ann: {stats}");
    assert_eq!(
        stats["quantization"]["kind"], "Int8",
        "quantization: {stats}"
    );
    assert_eq!(stats["query_threads"], 4, "query_threads: {stats}");
    assert_eq!(stats["mmap"], true, "mmap: {stats}");
}

/// The three-state interaction (unset / recorded / explicitly overridden) for the one knob
/// that could not express "off" before this change. `--mmap` was a bare bool, so a recorded
/// `mmap = true` would have been unturnoffable without the new `--no-mmap`.
#[test]
fn no_mmap_overrides_a_recorded_mmap() {
    let dir = tempfile::tempdir().unwrap();
    configure(dir.path(), 3, &["--mmap", "--segment-max-rows", "1000"]);

    // `nidus stats` rather than a first `serve`: it exits cleanly and releases the writer
    // lock, where a killed server leaves its advisory lock behind for the whole TTL.
    let dir_s = dir.path().to_str().expect("utf-8 temp path");
    let (stdout, _) = nidus_ok(&["stats", "--dir", dir_s]);
    let stats: serde_json::Value = serde_json::from_str(&stdout).expect("stats json");
    assert_eq!(stats["mmap"], true, "recorded mmap should apply: {stats}");

    let server = Server::new(dir.path(), 3).args(["--no-mmap"]).start();
    let (status, stats) = server.get("/stats");
    assert_eq!(status, 200, "stats: {stats}");
    assert_eq!(
        stats["mmap"], false,
        "--no-mmap must beat a recorded mmap = true: {stats}"
    );
}

/// The v1 manifest shape, redeclared here on purpose: this test must not share a definition
/// with the one `Manifest::decode` dispatches on, or a drift in that struct would move both
/// sides together and the migration would look tested when it is not.
#[derive(serde::Serialize)]
struct ManifestV1Wire {
    format_version: u16,
    dimension: u64,
    distance: nidus::Distance,
    segments: Vec<String>,
    next_id: u64,
    version: u64,
}

/// The claim #174 rests on: a store written before this change still opens. The blob is built
/// byte-wise because the new `encode` only ever emits v2 and cannot produce a v1 manifest.
#[test]
fn a_v1_manifest_is_lifted_by_the_real_binary() {
    let dir = tempfile::tempdir().unwrap();
    configure(dir.path(), 3, &["--ann", "hnsw"]);

    let v1 = ManifestV1Wire {
        format_version: 1,
        dimension: 3,
        distance: nidus::Distance::Cosine,
        segments: vec!["data".to_string()],
        next_id: 1,
        version: 1,
    };
    let payload = bincode::serialize(&v1).expect("serialize v1 manifest");
    let mut bytes = crc32fast::hash(&payload).to_le_bytes().to_vec();
    bytes.extend_from_slice(&payload);
    std::fs::write(dir.path().join("manifest"), &bytes).expect("overwrite manifest as v1");

    let server = Server::new(dir.path(), 3).start();
    let (status, stats) = server.get("/stats");
    assert_eq!(status, 200, "a v1 manifest must still open: {stats}");
    assert_eq!(stats["dimension"], 3, "{stats}");
    assert_eq!(
        stats["ann"],
        serde_json::Value::Null,
        "a v1 manifest carries no profile, so it lifts to built-in defaults: {stats}"
    );
}

/// Regression, review finding: `configure` recorded only the knobs on that one command line,
/// wiping anything an earlier call had set. Reproduced against the real binary before the fix.
#[test]
fn a_second_configure_adds_a_knob_without_erasing_the_first() {
    let dir = tempfile::tempdir().unwrap();
    configure(dir.path(), 3, &["--ann", "hnsw"]);
    configure(dir.path(), 3, &["--query-threads", "4"]);

    let dir_s = dir.path().to_str().expect("utf-8 temp path");
    let (stdout, _) = nidus_ok(&["stats", "--dir", dir_s]);
    let stats: serde_json::Value = serde_json::from_str(&stdout).expect("stats json");
    assert_eq!(
        stats["ann"]["kind"], "Hnsw",
        "the first configure's ann must survive the second call: {stats}"
    );
    assert_eq!(stats["query_threads"], 4, "{stats}");
}
