//! nidus-ixw: an on/off flag pair where the "on" side has an env var. Only a real spawn can
//! show it — `clap` resolves `NIDUS_*` from the *process* environment, which an in-process
//! parse test cannot set (`std::env::set_var` is unsafe, and the crate denies unsafe code).

use std::process::{Command, Output, Stdio};

/// Run the binary with `args` and exactly the `NIDUS_*` variables in `env` — the inherited
/// ones are stripped, so a variable set in the developer's shell cannot decide the outcome.
fn run_with_env(args: &[&str], env: &[(&str, &str)]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nidus"))
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear()
        .envs(std::env::vars().filter(|(k, _)| !k.starts_with("NIDUS_")))
        .envs(env.iter().copied())
        .output()
        .unwrap_or_else(|e| panic!("spawn nidus {args:?}: {e}"))
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The premise the override test rests on: `NIDUS_CLUSTER` alone really does turn cluster
/// mode on, which a local store then rejects. Without this, the next test proves nothing.
#[test]
fn a_cluster_env_default_reaches_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let dir_s = dir.path().to_str().expect("utf-8 temp path");
    let out = run_with_env(
        &["stats", "--dir", dir_s, "--dim", "3"],
        &[("NIDUS_CLUSTER", "true")],
    );

    let stderr = stderr_of(&out);
    assert!(
        !out.status.success(),
        "NIDUS_CLUSTER=true should have been honoured against a local store"
    );
    assert!(
        stderr.contains("cluster mode requires"),
        "it should fail on the cluster requirements, not something else: {stderr}"
    );
}

/// The bug: `--no-cluster` could not turn off an env-set default, because `conflicts_with`
/// counted the env var as the flag being present and rejected the pair outright.
#[test]
fn no_cluster_overrides_the_cluster_env_default() {
    let dir = tempfile::tempdir().unwrap();
    let dir_s = dir.path().to_str().expect("utf-8 temp path");
    let out = run_with_env(
        &["stats", "--dir", dir_s, "--dim", "3", "--no-cluster"],
        &[("NIDUS_CLUSTER", "true")],
    );

    let stderr = stderr_of(&out);
    assert!(
        out.status.success(),
        "--no-cluster must beat NIDUS_CLUSTER, got {:?}\n--- stderr ---\n{stderr}",
        out.status.code()
    );
}

/// The same shape for the other pair. A local store is happy either way here, so what this
/// asserts is that the pair parses at all — the conflict error is what used to stop it.
#[test]
fn no_mmap_overrides_the_mmap_env_default() {
    let dir = tempfile::tempdir().unwrap();
    let dir_s = dir.path().to_str().expect("utf-8 temp path");
    let out = run_with_env(
        &["stats", "--dir", dir_s, "--dim", "3", "--no-mmap"],
        &[("NIDUS_MMAP", "true")],
    );

    let stderr = stderr_of(&out);
    assert!(
        out.status.success() && !stderr.contains("cannot be used with"),
        "--no-mmap must beat NIDUS_MMAP, got {:?}\n--- stderr ---\n{stderr}",
        out.status.code()
    );
}

/// Both sides typed on one command line is still a contradiction, and still refused —
/// dropping `conflicts_with` moved that check, it did not remove it.
#[test]
fn both_sides_typed_together_are_still_refused() {
    let dir = tempfile::tempdir().unwrap();
    let dir_s = dir.path().to_str().expect("utf-8 temp path");
    let out = run_with_env(
        &["stats", "--dir", dir_s, "--dim", "3", "--mmap", "--no-mmap"],
        &[],
    );

    let stderr = stderr_of(&out);
    assert!(
        !out.status.success(),
        "--mmap --no-mmap together should fail"
    );
    assert!(
        stderr.contains("cannot be used with"),
        "it should read as a flag conflict: {stderr}"
    );
}
