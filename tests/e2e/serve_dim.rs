//! #139: `nidus serve --embed-provider …` into a fresh directory needs no `--dim`.
//! Only a real spawn proves this — the flag-to-`ServeConfig` wiring and the order in
//! which the embedder and the store config are resolved are invisible in-process.

use std::process::{Command, Stdio};

use crate::harness::Server;
use crate::mcp::support::{DIM, mock_embedder_per_text};

/// The `--embed-*` flags pointing `serve` at a mock embedder, as owned strings.
fn embed_args(url: &str) -> [String; 4] {
    [
        "--embed-provider".into(),
        "ollama".into(),
        "--embed-base-url".into(),
        url.into(),
    ]
}

/// Run `nidus serve` to completion and return its stderr. Only useful for the invocations
/// that are meant to fail before binding, which the harness cannot represent.
fn serve_stderr(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_nidus"))
        .arg("serve")
        .args(args)
        .arg("--addr")
        .arg("127.0.0.1:0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear()
        .envs(std::env::vars().filter(|(k, _)| !k.starts_with("NIDUS_")))
        .output()
        .expect("spawn nidus serve");
    assert!(!out.status.success(), "this invocation should have failed");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The ticket itself: no `--dim` anywhere on the command line, and the store comes up at
/// the embedder's own dimension.
#[test]
fn serve_infers_dim_from_the_embedder_on_a_fresh_store() {
    let tmp = tempfile::tempdir().unwrap();
    let url = mock_embedder_per_text(DIM);

    let server = Server::without_dim(tmp.path())
        .args(embed_args(&url))
        .start();

    let (status, stats) = server.get("/stats");
    assert_eq!(status, 200, "stats: {stats}");
    assert_eq!(
        stats["dimension"],
        serde_json::json!(DIM),
        "the embedder's dimension should have been adopted: {stats}"
    );
}

/// The fallback must not swallow the real error: with no embedder to ask, a fresh store
/// and no `--dim` is still the same hard failure it always was.
#[test]
fn serve_without_dim_or_embedder_still_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let err = serve_stderr(&["--dir", tmp.path().to_str().expect("utf-8 temp path")]);
    assert!(
        err.contains("--dim"),
        "the error should still name --dim, got: {err}"
    );
}

/// The inference is gated on there being no store yet, not merely on `--dim` being
/// absent. Copying `cli/memory.rs::open_with` verbatim would override the header here
/// and break every raw vector endpoint.
#[test]
fn an_existing_store_keeps_its_own_dimension_over_the_embedder() {
    let tmp = tempfile::tempdir().unwrap();
    let seeded = Server::new(tmp.path(), DIM + 1).start();
    assert_eq!(
        seeded.get("/stats").1["dimension"],
        serde_json::json!(DIM + 1)
    );
    assert!(seeded.shutdown(), "seed server should stop cleanly");

    let url = mock_embedder_per_text(DIM);
    let server = Server::without_dim(tmp.path())
        .args(embed_args(&url))
        .start();

    let (status, stats) = server.get("/stats");
    assert_eq!(status, 200, "stats: {stats}");
    assert_eq!(
        stats["dimension"],
        serde_json::json!(DIM + 1),
        "the header must win over the embedder: {stats}"
    );
}

/// An explicit `--dim` that disagrees with an existing store is still refused. The
/// inference changed which value is chosen when there is none, not the check.
#[test]
fn an_explicit_dim_still_has_to_match_an_existing_store() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_str().expect("utf-8 temp path");
    assert!(
        Server::new(tmp.path(), DIM).start().shutdown(),
        "seed server should stop cleanly"
    );

    let err = serve_stderr(&["--dir", dir, "--dim", &(DIM + 1).to_string()]);
    assert!(
        err.to_lowercase().contains("dimension"),
        "a mismatch should still be refused, got: {err}"
    );
}
