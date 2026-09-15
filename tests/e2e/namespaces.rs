//! End-to-end tests for `nidus serve --namespaced` (nidus-pcpc.2): many stores behind one
//! byte-bounded warm set, driven over a real socket against the real binary. `harness.rs`'s
//! `Server::over_base` builds these; every route here is `/ns/{namespace}/...` or the
//! process-level `/ready` and `/namespaces`.

use serde_json::{Value, json};

use crate::harness::{RunningServer, Server};

/// Upsert one record into `ns`'s `c` collection — the cheapest way to warm a namespace
/// and give it a nonzero byte footprint (an empty store sizes at 0, so it can never be
/// evicted; see `src/namespaces/warm.rs`).
fn upsert(server: &RunningServer, ns: &str, id: &str) -> (u16, Value) {
    server.post(
        &format!("/ns/{ns}/collections/c/upsert"),
        &json!({"records": [{"id": id, "vector": [1.0, 0.0, 0.0], "attrs": {}}]}),
    )
}

fn records(server: &RunningServer, ns: &str) -> (u16, Value) {
    server.get(&format!("/ns/{ns}/collections/c/records"))
}

/// 1. A second request reuses the warm store rather than reopening it: a corrupt on-disk
/// manifest stays invisible until the store is actually evicted. The eviction half is
/// load-bearing — without it, the reuse above would pass even if corruption were harmless.
#[test]
fn warm_store_survives_a_corrupt_manifest_until_evicted() {
    let base = tempfile::tempdir().expect("temp dir");
    let server = Server::over_base(base.path(), 3)
        .args(["--warm-budget-bytes", "1"])
        .start();

    let (status, body) = upsert(&server, "a", "1");
    assert_eq!(status, 200, "warm namespace a: {body}");

    let manifest = base.path().join("a").join("manifest");
    assert!(
        manifest.exists(),
        "namespace a must have opened on disk at {manifest:?}"
    );
    std::fs::write(&manifest, b"not a manifest").expect("corrupt a's on-disk manifest");

    // Still warm: reads never touch the manifest file, so the corruption is invisible.
    let (status, body) = records(&server, "a");
    assert_eq!(
        status, 200,
        "a warm store must keep serving from RAM, not reopen: {body}"
    );

    // Evict "a" by admitting "b" under a 1-byte budget — the same mechanism
    // `src/namespaces/tests.rs::evicting_a_namespace_releases_its_writer_lock` exercises
    // in-process, driven here through the real server instead.
    let (status, body) = upsert(&server, "b", "1");
    assert_eq!(status, 200, "warm namespace b, evicting a: {body}");

    // Reopening "a" now reads the corrupted manifest and must fail — the counterfactual
    // that makes the reuse above meaningful.
    let (status, body) = records(&server, "a");
    assert_ne!(
        status, 200,
        "a's corrupted manifest must surface once eviction forces a real reopen: {body}"
    );
}

/// 2. Eviction under the byte budget releases the writer lock, so a later request
/// re-opens cleanly. Counterfactual: marking a namespace cold without dropping its store
/// would leave the writer lock held, and the re-open below would fail.
#[test]
fn eviction_releases_the_writer_lock_and_a_later_request_reopens_cleanly() {
    let base = tempfile::tempdir().expect("temp dir");
    let server = Server::over_base(base.path(), 3)
        .args(["--warm-budget-bytes", "1"])
        .start();

    let (status, body) = upsert(&server, "a", "1");
    assert_eq!(status, 200, "warm namespace a: {body}");

    // Force eviction of "a" the same way as above, with nothing corrupted this time.
    let (status, body) = upsert(&server, "b", "1");
    assert_eq!(status, 200, "warm namespace b, evicting a: {body}");

    let (status, body) = records(&server, "a");
    assert_eq!(
        status, 200,
        "a must re-open cleanly once evicted — a held lock would 409 here: {body}"
    );
    let ids: Vec<&str> = body
        .as_array()
        .expect("records array")
        .iter()
        .filter_map(|r| r["id"].as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["1"],
        "the flush on eviction must not have lost a's write: {body}"
    );
}

/// 3. Per-namespace readiness answers per namespace, and `/ready` still answers with no
/// namespace ever touched. Counterfactual: an aggregate readiness implementation has
/// nothing per-name to report, and answering `/ready` from it needs a namespace opened.
#[test]
fn ready_is_process_level_and_namespaces_reports_per_namespace_readiness() {
    let base = tempfile::tempdir().expect("temp dir");
    let server = Server::over_base(base.path(), 3).start();

    // `/ready` answers before any namespace has ever been requested.
    let (status, body) = server.get("/ready");
    assert_eq!(status, 200, "/ready must answer in namespaced mode: {body}");
    assert_eq!(body["ready"], Value::Bool(true));

    // And it must not have opened anything to answer: nothing is warm yet.
    let (status, body) = server.get("/namespaces");
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body.as_array().map(Vec::len),
        Some(0),
        "/ready must not have opened a namespace: {body}"
    );

    // Now warm two namespaces and confirm the listing answers each of them by name,
    // which an aggregate (one shared readiness value) has no way to do.
    assert_eq!(upsert(&server, "a", "1").0, 200);
    assert_eq!(upsert(&server, "b", "1").0, 200);

    let (status, body) = server.get("/namespaces");
    assert_eq!(status, 200, "{body}");
    let entries = body.as_array().expect("namespaces array");
    assert_eq!(entries.len(), 2, "both warm namespaces: {body}");
    for name in ["a", "b"] {
        let entry = entries
            .iter()
            .find(|e| e["name"] == name)
            .unwrap_or_else(|| panic!("no entry named {name:?} in {body}"));
        assert!(
            entry["role"].is_string(),
            "{name}'s own readiness must be reported: {entry}"
        );
        assert_eq!(
            entry["fenced"],
            Value::Bool(false),
            "{name}'s own readiness must be reported: {entry}"
        );
    }
}

/// 4. The listing route reports only warm namespaces and opens nothing. Counterfactual:
/// an implementation that enumerates the base directory would report "cold" below too —
/// it was never touched through the server, only created on disk beside it.
#[test]
fn namespaces_listing_reports_only_warm_namespaces() {
    let base = tempfile::tempdir().expect("temp dir");
    // A namespace-shaped directory that exists on disk but was never opened through the
    // running server — the listing must not notice it.
    std::fs::create_dir_all(base.path().join("cold")).expect("create cold dir");

    let server = Server::over_base(base.path(), 3).start();
    assert_eq!(upsert(&server, "hot", "1").0, 200, "warm namespace hot");

    let (status, body) = server.get("/namespaces");
    assert_eq!(status, 200, "{body}");
    let names: Vec<&str> = body
        .as_array()
        .expect("namespaces array")
        .iter()
        .filter_map(|e| e["name"].as_str())
        .collect();
    assert_eq!(
        names,
        vec!["hot"],
        "listing must report only what was actually opened: {body}"
    );
}
