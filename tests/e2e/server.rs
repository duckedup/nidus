//! Standalone-server end-to-end tests: one real `nidus serve` process, driven over a
//! real socket. Each of these covers something `oneshot()` structurally cannot —
//! the bind, the flag wiring, socket-level framing, cross-process locking, or restart.

use serde_json::{Value, json};

use crate::harness::Server;

/// Two orthogonal unit vectors plus attrs, the shape the other suites reuse.
fn records() -> Value {
    json!({"records": [
        {"id": "a", "vector": [1, 0, 0], "attrs": {"lang": {"Str": "rust"}}},
        {"id": "b", "vector": [0, 1, 0], "attrs": {"lang": {"Str": "go"}}}
    ]})
}

/// The documented lifecycle, over the wire: create → upsert → search → text-search →
/// hybrid → stats. Mirrors `server::tests::full_lifecycle_over_http`, but through the
/// binary, a socket, and real HTTP rather than an in-process `Router` call.
#[test]
fn full_lifecycle_over_real_http() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);

    let (status, body) = server.post("/collections/docs/upsert", &records());
    assert_eq!(status, 200, "upsert failed: {body}");
    assert_eq!(body["upserted"], 2);

    let (status, hits) = server.post("/search", &json!({"query": [1, 0, 0], "top_k": 1}));
    assert_eq!(status, 200);
    assert_eq!(hits[0]["id"], "a");

    // Filters survive the JSON round trip through a real request body.
    let (status, hits) = server.post(
        "/search",
        &json!({"query": [1, 0, 0], "top_k": 5, "filter": [{"Eq": ["lang", {"Str": "go"}]}]}),
    );
    assert_eq!(status, 200);
    assert_eq!(hits.as_array().map(Vec::len), Some(1));
    assert_eq!(hits[0]["id"], "b");

    // Full-text and hybrid: declare a schema, add a body field, query by text.
    assert_eq!(
        server
            .post("/collections/docs/fts-schema", &json!({"fields": ["body"]}))
            .0,
        200
    );
    assert_eq!(
        server
            .post(
                "/collections/docs/upsert",
                &json!({"records": [
                    {"id": "c", "vector": [0, 0, 1], "attrs": {"body": {"Str": "foxes are running quickly"}}}
                ]}),
            )
            .0,
        200
    );
    let (status, hits) = server.post(
        "/text-search",
        &json!({"field": "body", "query": "run", "top_k": 5}),
    );
    assert_eq!(status, 200);
    assert_eq!(hits[0]["id"], "c", "stemmed text search should match c");

    let (status, hits) = server.post(
        "/hybrid-search",
        &json!({"vector": [1, 0, 0], "field": "body", "text": "fox", "top_k": 5}),
    );
    assert_eq!(status, 200);
    let ids: Vec<&str> = hits
        .as_array()
        .expect("hits array")
        .iter()
        .filter_map(|h| h["id"].as_str())
        .collect();
    assert!(ids.contains(&"a") && ids.contains(&"c"), "got {ids:?}");

    let (status, stats) = server.get("/stats");
    assert_eq!(status, 200);
    assert_eq!(stats["dimension"], 3);
    assert_eq!(stats["collections"], json!(["docs"]));
    assert_eq!(stats["footprint"]["doc_count"], 3);
}

/// `--token` is enforced on real requests, and `/health` stays exempt so a load
/// balancer can probe an authenticated server.
#[test]
fn token_auth_is_enforced_over_the_wire() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).token("s3cret").start();

    // The harness sends the bearer token, so authorised requests work.
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
    assert_eq!(server.get("/stats").0, 200);

    // A bare client without the header is refused.
    let anon = ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .http_status_as_error(false)
            .build(),
    );
    let unauthorised = anon
        .get(format!("{}/stats", server.base_url()))
        .call()
        .expect("request completes");
    assert_eq!(unauthorised.status().as_u16(), 401);

    // …except on /health, which must answer unauthenticated.
    let health = anon
        .get(format!("{}/health", server.base_url()))
        .call()
        .expect("request completes");
    assert_eq!(health.status().as_u16(), 200);
}

/// `--max-body-bytes` rejects an oversize upsert with 413 rather than hanging, OOMing,
/// or dropping the connection. Only a real socket exercises the body-limit layer.
#[test]
fn oversize_body_is_rejected_with_413() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3)
        .args(["--max-body-bytes", "2048"])
        .start();

    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);

    // Comfortably over the 2 KiB cap: one padded attr string.
    let big = json!({"records": [
        {"id": "big", "vector": [1, 0, 0], "attrs": {"pad": {"Str": "x".repeat(8192)}}}
    ]});
    let (status, _) = server.post_bytes(
        "/collections/docs/upsert",
        serde_json::to_vec(&big).unwrap().as_slice(),
    );
    assert_eq!(status, 413, "oversize upsert should be 413");

    // The server is still healthy and still serving after the rejection.
    assert_eq!(server.get("/health").0, 200);
    let (status, body) = server.post("/collections/docs/upsert", &records());
    assert_eq!(status, 200, "server still usable: {body}");
}

/// Concurrent clients against one server. The handlers take a write/read lock on a shared `Nidus`
/// from within `spawn_blocking`, so a lock-ordering mistake would deadlock here — and `oneshot()`
/// tests, being sequential and single-task, could not surface it.
#[test]
fn concurrent_clients_are_served() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
    assert_eq!(server.post("/collections/docs/upsert", &records()).0, 200);

    // Readers and writers interleaved, so searches contend with upserts for the lock.
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let server = &server;
                scope.spawn(move || {
                    if i % 4 == 0 {
                        let rec = json!({"records": [
                            {"id": format!("w{i}"), "vector": [0, 0, 1], "attrs": {}}
                        ]});
                        let (status, body) = server.post("/collections/docs/upsert", &rec);
                        assert_eq!(status, 200, "concurrent upsert {i}: {body}");
                    } else {
                        let (status, hits) =
                            server.post("/search", &json!({"query": [1, 0, 0], "top_k": 2}));
                        assert_eq!(status, 200, "concurrent search {i}: {hits}");
                        assert_eq!(hits[0]["id"], "a");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("no client thread panicked");
        }
    });

    // All four writers landed: the two seed records plus w0, w4, w8, w12.
    let (_, stats) = server.get("/stats");
    assert_eq!(stats["footprint"]["doc_count"], 6);
}

/// A second server over the same directory is refused, with the message that explains
/// what to do about it (nidus-32y) — asserted for the first time against two real
/// processes rather than two `Store` values in one.
#[test]
fn second_server_on_the_same_dir_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let _held = Server::new(dir.path(), 3).start();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_nidus"))
        .args(["serve", "--dir"])
        .arg(dir.path())
        .args(["--dim", "3", "--addr", "127.0.0.1:0"])
        .output()
        .expect("run second server");

    assert!(!output.status.success(), "second serve must fail");
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(
        err.contains("locked") || err.contains("already"),
        "unhelpful lock error: {err}"
    );
}

/// Cluster mode is refused on a backend without compare-and-swap (nidus-lp4.2).
#[test]
fn cluster_mode_is_refused_without_the_required_backends() {
    let dir = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_nidus"))
        .args(["serve", "--dir"])
        .arg(dir.path())
        .args(["--dim", "3", "--addr", "127.0.0.1:0", "--cluster"])
        .output()
        .expect("run cluster server");

    assert!(
        !output.status.success(),
        "--cluster on a local store must not start"
    );
    let err = String::from_utf8_lossy(&output.stderr).to_lowercase();
    assert!(
        err.contains("shared object-store") || err.contains("compare-and-swap"),
        "the refusal should name the missing capability: {err}"
    );
}

/// `/ready` and `/cluster` work on an ordinary single-node store too — the endpoints must
/// not be cluster-only, or a non-cluster deployment could not use the probes at all.
#[test]
fn probes_work_on_a_single_node_store() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();

    let (status, body) = server.get("/ready");
    assert_eq!(status, 200, "{body}");

    let (status, body) = server.get("/cluster");
    assert_eq!(status, 200);
    assert_eq!(body["role"], "Writer", "single-node writer role: {body}");
    assert_eq!(body["cluster"], false);
    assert_eq!(body["fenced"], false);
    assert_eq!(
        body["holds_writer_handle"], true,
        "a single-node writer holds the plain writer lock: {body}"
    );
}

/// SIGTERM is the graceful path: it flushes and releases the writer lock, so a
/// replacement server starts immediately over the same directory and sees the data.
/// Covers `serve()`'s shutdown handler, which no in-process test reaches.
#[cfg(unix)]
#[test]
fn sigterm_flushes_and_releases_the_lock() {
    let dir = tempfile::tempdir().unwrap();

    let server = Server::new(dir.path(), 3).start();
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
    assert_eq!(server.post("/collections/docs/upsert", &records()).0, 200);
    assert!(server.shutdown(), "clean shutdown should exit successfully");

    // No lock-reclaim wait: a graceful exit released it, so this start proves both that
    // the lock is gone and that the committed records replayed.
    let restarted = Server::new(dir.path(), 3).start();
    let (status, stats) = restarted.get("/stats");
    assert_eq!(status, 200);
    assert_eq!(stats["footprint"]["doc_count"], 2);
    let (_, hits) = restarted.post("/search", &json!({"query": [0, 1, 0], "top_k": 1}));
    assert_eq!(hits[0]["id"], "b", "attrs and vectors survived the restart");
}

/// SIGKILL is the crash path: nothing flushes and the lock file is left behind. The
/// per-batch fsync default means acknowledged writes still survive, and `--lock-ttl`
/// governs when the stale lock may be reclaimed.
#[test]
fn killed_server_leaves_data_intact_and_lock_reclaimable() {
    let dir = tempfile::tempdir().unwrap();

    // A 1s TTL keeps the reclaim wait short enough to assert on.
    let server = Server::new(dir.path(), 3).args(["--lock-ttl", "1"]).start();
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
    assert_eq!(server.post("/collections/docs/upsert", &records()).0, 200);
    server.kill();

    // Past the TTL the abandoned lock is reclaimable and the fsynced batch is intact.
    std::thread::sleep(std::time::Duration::from_secs(2));
    let restarted = Server::new(dir.path(), 3).args(["--lock-ttl", "1"]).start();
    let (status, stats) = restarted.get("/stats");
    assert_eq!(status, 200);
    assert_eq!(
        stats["footprint"]["doc_count"], 2,
        "per-batch fsync should have preserved the acknowledged upsert"
    );
}

/// The flags phase 0 added must actually change how the *server* opens the store, not
/// just how `Config` parses. `/stats` echoes the ANN config, so it can prove the flag
/// reached the running store; quantization is asserted behaviourally (search still
/// ranks correctly through the quantized first pass + exact rerank).
#[test]
fn opt_in_flags_reach_the_running_store() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3)
        .args([
            "--ann",
            "hnsw",
            "--quantization",
            "int8",
            "--query-threads",
            "4",
        ])
        .start();

    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
    assert_eq!(server.post("/collections/docs/upsert", &records()).0, 200);

    let (status, stats) = server.get("/stats");
    assert_eq!(status, 200);
    assert_eq!(
        stats["ann"]["kind"], "Hnsw",
        "--ann should reach the served store, got {stats}"
    );

    // Ranking stays correct through the ANN walk + quantized first pass + f32 rerank.
    let (status, hits) = server.post("/search", &json!({"query": [1, 0, 0], "top_k": 1}));
    assert_eq!(status, 200);
    assert_eq!(hits[0]["id"], "a");
}

/// **A long write must not make the instance report NOT ready** (nidus-abx.3).
#[test]
fn a_long_write_does_not_make_the_instance_report_unready() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 64).start();
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);

    // Big enough that the write guard is held across many probe intervals in a debug build,
    // small enough not to dominate the fast lane's runtime.
    let records: Vec<Value> = (0..10_000)
        .map(|i| {
            json!({
                "id": format!("d{i}"),
                "vector": (0..64).map(|d| ((i + d) % 17) as f64).collect::<Vec<_>>(),
                "attrs": {},
            })
        })
        .collect();
    let body = json!({ "records": records });

    let mut probes = 0usize;
    let mut unready: Vec<usize> = Vec::new();
    std::thread::scope(|s| {
        let writing = s.spawn(|| server.post("/collections/docs/upsert", &body));
        while !writing.is_finished() {
            if !server.is_ready() {
                unready.push(probes);
            }
            probes += 1;
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(writing.join().expect("upsert thread").0, 200);
    });

    assert!(
        unready.is_empty(),
        "readiness dropped during a healthy write at probes {unready:?} of {probes} — busy is \
         not unhealthy, and flapping here takes the writer out of rotation mid-batch"
    );
    // Guard against the test silently proving nothing: if the write finished before any
    // probe overlapped it, the assertion above never actually observed a busy store.
    assert!(
        probes >= 5,
        "the upsert completed too quickly to exercise the race (only {probes} probes overlapped)"
    );
}

/// **Group commit does not weaken durability: every acknowledged concurrent write survives
/// SIGKILL** (nidus-xb9.1).
#[test]
fn every_acknowledged_concurrent_write_survives_sigkill() {
    let dir = tempfile::tempdir().unwrap();
    // A short TTL so the abandoned lock from the kill is reclaimable without a long wait.
    let server = Server::new(dir.path(), 3).args(["--lock-ttl", "1"]).start();
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);

    const WRITERS: usize = 8;
    const PER_WRITER: usize = 12;

    // Only ids the server actually answered `200` for. An id whose request failed, or whose
    // response never arrived, proves nothing either way and must not be asserted on.
    let acked: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..WRITERS)
            .map(|w| {
                let server = &server;
                scope.spawn(move || {
                    let mut ok = Vec::new();
                    for r in 0..PER_WRITER {
                        let id = format!("w{w}-{r}");
                        let body = json!({"records": [
                            {"id": id, "vector": [1, 0, 0], "attrs": {}}
                        ]});
                        let (status, resp) = server.post("/collections/docs/upsert", &body);
                        assert_eq!(status, 200, "upsert {id} failed: {resp}");
                        ok.push(id);
                    }
                    ok
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("no writer thread panicked"))
            .collect()
    });
    assert_eq!(acked.len(), WRITERS * PER_WRITER);

    // Before the crash, confirm the writes really did share barriers — otherwise this test
    // would pass just as well with group commit removed, and would be measuring nothing.
    let text = crate::harness::scrape(&server);
    let groups = crate::harness::metric(&text, "nidus_write_groups_total").expect("groups");
    let members =
        crate::harness::metric(&text, "nidus_write_group_members_total").expect("group members");
    assert!(
        members > groups,
        "the writes never overlapped ({members} writes in {groups} groups), so this run did \
         not exercise a shared barrier"
    );

    // The crash. Not a shutdown: `kill` is SIGKILL, so nothing flushes and nothing runs on
    // the way out. Whatever is on disk now is whatever the barriers put there.
    server.kill();

    std::thread::sleep(std::time::Duration::from_secs(2));
    let restarted = Server::new(dir.path(), 3).args(["--lock-ttl", "1"]).start();
    let (status, records) = restarted.post("/list", &json!({"limit": 10_000}));
    assert_eq!(status, 200);
    let survivors: std::collections::HashSet<String> = records
        .as_array()
        .expect("a list response")
        .iter()
        .map(|r| r["id"].as_str().expect("an id").to_string())
        .collect();

    let lost: Vec<&String> = acked.iter().filter(|id| !survivors.contains(*id)).collect();
    assert!(
        lost.is_empty(),
        "{} of {} acknowledged writes did not survive the crash: {lost:?} — a 200 was returned \
         before its bytes were durable",
        lost.len(),
        acked.len()
    );
}
