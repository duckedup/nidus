//! `nidus tune` end-to-end: a real store, a real sweep, checked against an independently
//! reasoned-about correctness anchor (IVF with `n_probe == n_lists` visits every row) rather
//! than "the command exited 0 and printed JSON" (nidus-sk9).

use std::io::Write;
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};

/// Mirrors `cli.rs`'s helper (copied, not shared — each e2e suite stands alone). Strips
/// `NIDUS_*` so an inherited env var can't silently override the flag under test.
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

/// Run a command that must succeed, returning its stdout parsed as JSON.
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

/// Run a command that must fail, returning its stderr.
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

/// Deterministic vectors without a PRNG dependency: SplitMix64, same approach as
/// `scale.rs`'s `Rng` (kept local — e2e suites don't reach into the library's internals).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float in `[-1, 1)`.
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 8_388_608.0 - 1.0
    }

    fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_unit()).collect()
    }
}

/// `n` random `dim`-wide records as one JSON array, ready for `upsert` stdin.
fn corpus(seed: u64, n: usize, dim: usize) -> String {
    let mut rng = Rng(seed);
    let records: Vec<Value> = (0..n)
        .map(|i| json!({"id": format!("d{i}"), "vector": rng.vector(dim), "attrs": {}}))
        .collect();
    json!(records).to_string()
}

/// Create `collection` at `dim` in `dir` and upsert `n` random vectors into it — few
/// hundred, not scale.rs's 10k: this is a debug build and the sweep grid runs many cells.
fn seed_store(dir: &str, collection: &str, dim: usize, n: usize, seed: u64) {
    ok(
        &[
            "create",
            "--dir",
            dir,
            "--dim",
            &dim.to_string(),
            collection,
        ],
        "",
    );
    let out = ok(&["upsert", "--dir", dir, collection], &corpus(seed, n, dim));
    assert_eq!(out["upserted"], n as u64, "{out}");
}

/// `nidus tune --dir <dir> <extra...>`, parsed as JSON. Every test below passes `--ann ivf`
/// explicitly (required by the CLI) plus its own knob lists.
fn run_tune(dir: &str, extra: &[&str]) -> Value {
    let mut args = vec!["tune", "--dir", dir];
    args.extend_from_slice(extra);
    ok(&args, "")
}

fn recall_of(cell: &Value) -> f64 {
    cell["recall_at_k"]
        .as_f64()
        .unwrap_or_else(|| panic!("cell missing recall_at_k: {cell}"))
}

/// The one swept cell matching `(n_probe, overscan)`, or a panic naming what was missing.
fn find_cell(cells: &[Value], n_probe: u64, overscan: u64) -> &Value {
    cells
        .iter()
        .find(|c| {
            c["ann"]["n_probe"].as_u64() == Some(n_probe)
                && c["ann"]["overscan"].as_u64() == Some(overscan)
        })
        .unwrap_or_else(|| panic!("no cell for n_probe={n_probe} overscan={overscan}"))
}

/// (1) The sweep works against a real store: every cell's recall@k is present and in
/// `[0, 1]`, and the recommended block names knobs that were actually part of the sweep —
/// this would fail if `tune` faked a fixed recall or a fixed recommendation.
#[test]
fn sweep_against_a_real_store_reports_recall_in_range_and_recommends_a_swept_cell() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    seed_store(dir, "col", 16, 300, 1);

    let report = run_tune(
        dir,
        &[
            "--ann",
            "ivf",
            "--ann-n-lists",
            "10",
            "--n-probe",
            "1,5,10",
            "--overscan",
            "1,4,8",
            "--sample",
            "40",
            "--top-k",
            "5",
        ],
    );

    let cells = report["cells"].as_array().expect("cells array");
    assert_eq!(cells.len(), 9, "expected the full 3x3 grid: {report}");
    for cell in cells {
        let recall = recall_of(cell);
        assert!(
            (0.0..=1.0).contains(&recall),
            "recall_at_k out of [0,1]: {cell}"
        );
    }

    let rec_n_probe = report["recommended"]["config"]["ann"]["n_probe"]
        .as_u64()
        .expect("recommended n_probe");
    let rec_overscan = report["recommended"]["config"]["ann"]["overscan"]
        .as_u64()
        .expect("recommended overscan");
    assert!(
        [1, 5, 10].contains(&rec_n_probe),
        "recommended n_probe {rec_n_probe} was not among the swept values: {report}"
    );
    assert!(
        [1, 4, 8].contains(&rec_overscan),
        "recommended overscan {rec_overscan} was not among the swept values: {report}"
    );
}

/// Correctness anchor (2): IVF with `n_probe == n_lists` scans every row, so with no
/// quantization it must match the `exact: true` ground truth exactly (recall 1.0). A
/// starved cell (3) (`n_probe=1, overscan=1`) must then score strictly below it.
#[test]
fn generous_ivf_cell_matches_exact_ground_truth_and_starved_cell_scores_lower() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    seed_store(dir, "col", 16, 300, 2);

    let report = run_tune(
        dir,
        &[
            "--ann",
            "ivf",
            "--ann-n-lists",
            "10",
            "--n-probe",
            "1,10",
            "--overscan",
            "1,8",
            "--sample",
            "60",
            "--top-k",
            "5",
        ],
    );
    let cells = report["cells"].as_array().expect("cells array");
    let generous = find_cell(cells, 10, 8);
    let starved = find_cell(cells, 1, 1);
    let (g, s) = (recall_of(generous), recall_of(starved));
    assert_eq!(
        g, 1.0,
        "probing every IVF list with no quantization must match exact ground truth: {generous}"
    );
    assert!(
        s < g,
        "starved cell ({s}) should score below generous ({g}): {report}"
    );
}

/// (4) The self-hit policy appears in the output, per the ticket's "say which" requirement.
#[test]
fn self_hit_policy_is_reported_in_the_output() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    seed_store(dir, "col", 8, 40, 3);

    let report = run_tune(
        dir,
        &[
            "--ann",
            "ivf",
            "--ann-n-lists",
            "4",
            "--n-probe",
            "4",
            "--overscan",
            "4",
        ],
    );
    let policy = report["sample"]["self_hit_policy"]
        .as_str()
        .expect("self_hit_policy string");
    assert!(
        !policy.trim().is_empty(),
        "self_hit_policy must not be empty: {report}"
    );
    assert!(
        policy.contains("own"),
        "self_hit_policy should describe excluding the query's own hit, got: {policy}"
    );
}

/// (5) `tune` opens read-only, so it runs alongside a live `nidus serve` holding the writer
/// lock — the same coexistence `cli.rs`'s `read_subcommands_run_against_a_dir_a_server_holds`
/// proves for other read-only subcommands, and the ticket's "self-check after a reindex".
#[test]
fn tune_runs_read_only_alongside_a_live_server() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    seed_store(dir, "col", 8, 60, 4);

    let server = crate::harness::Server::new(tmp.path(), 8).start();
    assert_eq!(server.post("/collections/other", &json!({})).0, 200);

    let report = run_tune(
        dir,
        &[
            "--ann",
            "ivf",
            "--ann-n-lists",
            "4",
            "--n-probe",
            "4",
            "--overscan",
            "4",
        ],
    );
    let cells = report["cells"].as_array().expect("cells array");
    assert!(!cells.is_empty(), "{report}");
    for cell in cells {
        let recall = recall_of(cell);
        assert!((0.0..=1.0).contains(&recall), "{cell}");
    }

    // The lock the server holds is real: a mutating subcommand is refused while it runs.
    let err = fails(&["upsert", "--dir", dir, "col"], "[]");
    assert!(err.contains("lock"), "{err}");
}

/// (6) A many-cell sweep — 3 quantizations x 3 params x 3 overscans, all in one
/// invocation — leaves no lock file behind: `OpenMode::ReadOnly` never takes one.
#[test]
fn many_cell_sweep_leaves_no_lock_file_behind() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    seed_store(dir, "col", 8, 60, 5);

    let report = run_tune(
        dir,
        &[
            "--ann",
            "ivf",
            "--ann-n-lists",
            "4",
            "--n-probe",
            "1,2,4",
            "--overscan",
            "1,2,4",
            "--sweep-quantization",
            "--sample",
            "20",
        ],
    );
    let cells = report["cells"].as_array().expect("cells array");
    assert_eq!(
        cells.len(),
        27,
        "expected the full quantization x param x overscan grid: {report}"
    );

    assert!(
        !tmp.path().join("lock").exists(),
        "tune must not leave a lock file behind after a many-cell sweep"
    );
}
