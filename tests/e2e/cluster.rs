//! Cluster-mode end-to-end tests: **several real `nidus serve` processes** over a shared
//! object store and a shared memory tier.
//!
//! The cluster tests in `src/store/tests.rs` are thorough about the *logic* but run
//! entirely in one process — `cluster_writer()` and `cluster_reader()` are two `Store`
//! values in the same test function over `InMemObjectStore` + `LocalRam`, and a lost lease
//! is simulated with `drop()`. That leaves three things unproven, all of which are what
//! actually breaks in production:
//!
//! * **Real S3 semantics.** The compare-and-swap fencing relies on conditional-write
//!   behaviour (`If-Match` on an ETag). `InMemObjectStore` models what we *believe* S3
//!   does; only a real server proves it.
//! * **Process boundaries.** A writer that is `drop()`ped releases its lease cleanly. A
//!   writer that is killed, or merely *stalled*, does not — and that is the case the
//!   fencing exists for.
//! * **The server.** Nothing proved the HTTP layer drives any of this. It did not:
//!   see `nidus-6bb`, found by exactly this suite.
//!
//! **These tests need Docker services and so are `#[ignore]`d** — `just test-cli` stays
//! service-free. Run them with `just test-e2e-cluster` (see that recipe for the
//! `docker run` lines), or point them at your own services with `NIDUS_E2E_S3_ENDPOINT`,
//! `NIDUS_E2E_S3_BUCKET`, `NIDUS_E2E_S3_KEY`, `NIDUS_E2E_S3_SECRET`, and
//! `NIDUS_E2E_REDIS_URL`.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::harness::{RunningServer, Server};

/// Stale-lease window. Short so takeover tests do not dominate the suite's runtime, but
/// well clear of the per-request latency to a local container.
const LOCK_TTL_SECS: u32 = 2;

/// How long to wait past `LOCK_TTL_SECS` before expecting a lease to be reclaimable.
fn past_the_lease() -> Duration {
    Duration::from_secs(LOCK_TTL_SECS as u64 + 2)
}

fn service(var: &str, default: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| default.to_string())
}

/// Fail fast, once per process, if the backing services are not reachable.
///
/// Without this the first symptom is a child process dying with `Connection refused`
/// buried in its captured stderr — technically diagnosable, but it does not tell you the
/// one thing you need to know, which is that you forgot to start the services.
fn require_services() {
    static CHECKED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CHECKED.get_or_init(|| {
        let endpoint = service("NIDUS_E2E_S3_ENDPOINT", "http://127.0.0.1:9100");
        let redis = service("NIDUS_E2E_REDIS_URL", "redis://127.0.0.1:6479");
        let hint = "start them with `just e2e-services-up` (or point the tests elsewhere \
                    with NIDUS_E2E_S3_ENDPOINT / NIDUS_E2E_REDIS_URL)";

        let health = format!("{}/minio/health/live", endpoint.trim_end_matches('/'));
        assert!(
            ureq::get(&health).call().is_ok(),
            "S3 endpoint {endpoint} is not reachable — {hint}"
        );

        // No Redis round trip needed: a TCP connect distinguishes "nothing listening"
        // (the mistake this guards) from a protocol-level problem, which the store's own
        // error would explain better than we could here. Strip the scheme, then any
        // `/db` or `?prefix=…` tail, so a fully-specified URL still yields host:port.
        let addr = redis
            .split_once("://")
            .map_or(redis.as_str(), |(_, rest)| rest)
            .split(['/', '?'])
            .next()
            .unwrap_or_default();
        assert!(
            std::net::TcpStream::connect(addr).is_ok(),
            "memory tier {redis} is not reachable at {addr} — {hint}"
        );
    });
}

/// A store prefix unique to this process *and* this call, so tests neither collide with
/// each other nor inherit objects left by an earlier run of the same test.
fn unique_prefix(name: &str) -> String {
    static N: AtomicU32 = AtomicU32::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after 1970")
        .as_nanos();
    let seq = N.fetch_add(1, Ordering::Relaxed);
    format!("e2e/{name}-{nanos}-{seq}")
}

/// One instance in a cluster over `prefix`. `read_only` picks a lock-free reader.
///
/// `--dir` still gets a temp directory: in cluster mode the durable bytes live in the
/// object store, but the flag is still required, so each instance gets its own scratch
/// path exactly as separate machines would.
fn instance(prefix: &str, read_only: bool, extra: &[&str]) -> (tempfile::TempDir, RunningServer) {
    require_services();
    let dir = tempfile::tempdir().expect("temp dir");
    let bucket = service("NIDUS_E2E_S3_BUCKET", "nidus-test");
    let mut args = vec![
        "--cluster".to_string(),
        "--persistence".to_string(),
        format!("s3://{bucket}/{prefix}"),
        "--memory".to_string(),
        service("NIDUS_E2E_REDIS_URL", "redis://127.0.0.1:6479"),
        "--lock-ttl".to_string(),
        LOCK_TTL_SECS.to_string(),
    ];
    if read_only {
        args.push("--read-only".to_string());
    }
    args.extend(extra.iter().map(|s| s.to_string()));

    let server = Server::new(dir.path(), 3)
        .args(&args)
        .env(
            "AWS_ENDPOINT_URL",
            &service("NIDUS_E2E_S3_ENDPOINT", "http://127.0.0.1:9100"),
        )
        .env(
            "AWS_ACCESS_KEY_ID",
            &service("NIDUS_E2E_S3_KEY", "minioadmin"),
        )
        .env(
            "AWS_SECRET_ACCESS_KEY",
            &service("NIDUS_E2E_S3_SECRET", "minioadmin"),
        )
        .env("AWS_REGION", &service("NIDUS_E2E_S3_REGION", "us-east-1"))
        .start();
    // The TempDir must outlive the server, so hand both back together.
    (dir, server)
}

/// Spawn a cluster writer, expecting it to fail, and return its stderr — for the tests
/// that assert an instance is *refused*.
fn expect_start_failure(prefix: &str) -> String {
    let dir = tempfile::tempdir().expect("temp dir");
    let bucket = service("NIDUS_E2E_S3_BUCKET", "nidus-test");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_nidus"))
        .args(["serve", "--dir"])
        .arg(dir.path())
        .args(["--dim", "3", "--addr", "127.0.0.1:0", "--cluster"])
        .args(["--persistence", &format!("s3://{bucket}/{prefix}")])
        .args([
            "--memory",
            &service("NIDUS_E2E_REDIS_URL", "redis://127.0.0.1:6479"),
        ])
        .args(["--lock-ttl", &LOCK_TTL_SECS.to_string()])
        .env(
            "AWS_ENDPOINT_URL",
            service("NIDUS_E2E_S3_ENDPOINT", "http://127.0.0.1:9100"),
        )
        .env(
            "AWS_ACCESS_KEY_ID",
            service("NIDUS_E2E_S3_KEY", "minioadmin"),
        )
        .env(
            "AWS_SECRET_ACCESS_KEY",
            service("NIDUS_E2E_S3_SECRET", "minioadmin"),
        )
        .env("AWS_REGION", service("NIDUS_E2E_S3_REGION", "us-east-1"))
        .output()
        .expect("run second writer");
    assert!(
        !out.status.success(),
        "a second cluster writer must not start"
    );
    String::from_utf8_lossy(&out.stderr).to_string()
}

fn seed(server: &RunningServer) {
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
}

fn upsert(server: &RunningServer, id: &str, vector: [i32; 3]) -> (u16, Value) {
    server.post(
        "/collections/docs/upsert",
        &json!({"records": [{"id": id, "vector": vector, "attrs": {}}]}),
    )
}

fn doc_count(server: &RunningServer) -> u64 {
    let (status, stats) = server.get("/stats");
    assert_eq!(status, 200, "stats failed: {stats}");
    stats["footprint"]["doc_count"]
        .as_u64()
        .unwrap_or_else(|| panic!("no doc_count in {stats}"))
}

fn ids(server: &RunningServer) -> Vec<String> {
    let (status, hits) = server.post("/search", &json!({"query": [1, 0, 0], "top_k": 100}));
    assert_eq!(status, 200, "search failed: {hits}");
    let mut ids: Vec<String> = hits
        .as_array()
        .expect("hits array")
        .iter()
        .filter_map(|h| h["id"].as_str().map(str::to_string))
        .collect();
    ids.sort();
    ids
}

/// The headline behaviour: a reader instance picks up a writer instance's commits.
///
/// This is what `nidus-6bb` was: every underlying piece worked, but no server code path
/// called `refresh()`, so a reader served its open-time snapshot forever. `POST /refresh`
/// is the fix, and `adopted` distinguishes "advanced" from "nothing new".
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn reader_adopts_writer_commits_on_refresh() {
    let prefix = unique_prefix("refresh");
    let (_wdir, writer) = instance(&prefix, false, &[]);
    seed(&writer);
    assert_eq!(upsert(&writer, "first", [1, 0, 0]).0, 200);

    // The reader adopts the committed state at open — this much already worked.
    let (_rdir, reader) = instance(&prefix, true, &[]);
    assert_eq!(doc_count(&reader), 1, "reader should adopt state at open");

    // Nothing has changed since, so a refresh is a no-op and says so.
    let (status, body) = reader.post("/refresh", &json!({}));
    assert_eq!(status, 200);
    assert_eq!(body["adopted"], false, "no new commits to adopt");

    // The writer commits again.
    assert_eq!(upsert(&writer, "second", [0, 1, 0]).0, 200);
    assert_eq!(doc_count(&writer), 2);

    // Pin down the actual contract: a reader is stale *until* it refreshes. This is the
    // documented cost of not putting a manifest fetch on every read, and asserting it
    // here means the test would also catch someone silently making reads auto-refresh.
    assert_eq!(
        doc_count(&reader),
        1,
        "a reader should not see new commits until it refreshes"
    );

    let (status, body) = reader.post("/refresh", &json!({}));
    assert_eq!(status, 200);
    assert_eq!(body["adopted"], true, "a new commit should be adopted");
    assert_eq!(doc_count(&reader), 2, "reader must see the writer's commit");
    assert_eq!(ids(&reader), vec!["first", "second"]);
}

/// A read-only instance takes no lease, so several may run alongside the writer — the
/// fan-out this mode exists for.
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn many_readers_coexist_with_one_writer() {
    let prefix = unique_prefix("fanout");
    let (_wdir, writer) = instance(&prefix, false, &[]);
    seed(&writer);
    assert_eq!(upsert(&writer, "a", [1, 0, 0]).0, 200);

    let readers: Vec<_> = (0..3).map(|_| instance(&prefix, true, &[])).collect();
    for (_dir, reader) in &readers {
        assert_eq!(doc_count(reader), 1);
    }

    assert_eq!(upsert(&writer, "b", [0, 1, 0]).0, 200);
    for (_dir, reader) in &readers {
        assert_eq!(reader.post("/refresh", &json!({})).1["adopted"], true);
        assert_eq!(doc_count(reader), 2);
        assert_eq!(ids(reader), vec!["a", "b"]);
    }
}

/// The writer lease is exclusive across processes: while one instance holds it, a second
/// writer over the same store is refused. Previously only asserted between two `Store`
/// values in one process.
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn second_writer_is_excluded_across_processes() {
    let prefix = unique_prefix("exclusion");
    let (_wdir, writer) = instance(&prefix, false, &[]);
    seed(&writer);

    let err = expect_start_failure(&prefix);
    assert!(
        err.contains("lock") || err.contains("lease") || err.contains("locked"),
        "unhelpful exclusion error: {err}"
    );

    // The incumbent is unaffected by the rejected challenger.
    assert_eq!(upsert(&writer, "a", [1, 0, 0]).0, 200);
    assert_eq!(doc_count(&writer), 1);
}

/// A writer that *dies* — SIGKILL, no clean release, lease left behind — must not wedge
/// the store: past the TTL another instance takes over, and everything the dead writer
/// acknowledged is still there.
///
/// `drop()` cannot express this, because dropping releases the lease properly.
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn killed_writer_lease_is_taken_over_with_data_intact() {
    let prefix = unique_prefix("takeover");
    let (_wdir, writer) = instance(&prefix, false, &[]);
    seed(&writer);
    assert_eq!(upsert(&writer, "committed", [1, 0, 0]).0, 200);
    writer.kill();

    std::thread::sleep(past_the_lease());

    let (_ndir, successor) = instance(&prefix, false, &[]);
    assert_eq!(
        doc_count(&successor),
        1,
        "the killed writer's acknowledged commit must survive"
    );
    assert_eq!(upsert(&successor, "after", [0, 1, 0]).0, 200);
    assert_eq!(ids(&successor), vec!["after", "committed"]);
}

/// **The split-brain fence.** A writer that stalls after its lease check — a long GC
/// pause, a descheduled host — can wake up already superseded and still believe it holds
/// the lease. SIGSTOP manufactures exactly that stall, which `drop()` cannot.
///
/// nidus fences this twice: the lease is re-verified at the start of each batch, and every
/// durable write is a compare-and-swap on the version last seen (`If-Match` on the ETag,
/// from nidus-ahw). **Observed here: the lease re-check fires first**, so this test proves
/// the outer fence and asserts on its message; the CAS backstop behind it is covered by
/// the in-RAM tests, which can force that exact interleaving.
///
/// Real-S3 CAS semantics are not untested, though — in cluster mode *every* durable write
/// goes through `put_cas`, so all six tests in this file would fail if minio's `If-Match`
/// handling disagreed with what `InMemObjectStore` models.
///
/// What must never happen, by either fence: a successful write that discards the
/// successor's committed data.
#[cfg(unix)]
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn stalled_writer_is_fenced_and_cannot_clobber() {
    let prefix = unique_prefix("fence");
    let (_adir, writer_a) = instance(&prefix, false, &[]);
    seed(&writer_a);
    assert_eq!(upsert(&writer_a, "from-a", [1, 0, 0]).0, 200);

    // A stalls long enough for its lease to lapse.
    writer_a.pause();
    std::thread::sleep(past_the_lease());

    // B takes over and commits.
    let (_bdir, writer_b) = instance(&prefix, false, &[]);
    assert_eq!(
        doc_count(&writer_b),
        1,
        "successor should load A's committed state"
    );
    assert_eq!(upsert(&writer_b, "from-b", [0, 1, 0]).0, 200);

    // A wakes up superseded and tries to write. A bare `!= 200` would also pass if the
    // write failed for some unrelated reason, so assert on *why* it was refused.
    writer_a.resume();
    let (status, body) = upsert(&writer_a, "from-a-stale", [0, 0, 1]);
    assert_ne!(
        status, 200,
        "a superseded writer's write must be refused, got 200 with {body}"
    );
    let err = body["error"].as_str().unwrap_or_default();
    assert!(
        err.contains("lease"),
        "expected a lease-fencing error, got {status} {body}"
    );

    // The decisive assertion: B's data is intact and A's stale row never landed.
    assert_eq!(
        ids(&writer_b),
        vec!["from-a", "from-b"],
        "the fenced writer must not have clobbered committed data"
    );

    // And a fresh reader agrees — the refusal held in the durable store, not just in B's
    // memory.
    let (_rdir, reader) = instance(&prefix, true, &[]);
    assert_eq!(ids(&reader), vec!["from-a", "from-b"]);
}

/// A reader restarted from scratch reconstructs the same state, so the shared backend and
/// tier are a complete record of the store — not just a delta over some local file.
///
/// Note this asserts the *result*, not the mechanism: whether the reader adopted the
/// valkey working-set snapshot or replayed the log is not observable over HTTP. The
/// adopt-vs-replay path itself is covered by the in-RAM tests in `src/store/tests.rs`.
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn fresh_reader_reconstructs_state_from_shared_backend() {
    let prefix = unique_prefix("adopt");
    let (_wdir, writer) = instance(&prefix, false, &[]);
    seed(&writer);
    for (i, id) in ["x", "y", "z"].iter().enumerate() {
        let v = [i as i32 % 2, (i as i32 + 1) % 2, 0];
        assert_eq!(upsert(&writer, id, v).0, 200);
    }
    let expected = ids(&writer);

    // A brand-new reader process with an empty --dir: everything must come from the
    // shared object store + tier.
    let (_rdir, reader) = instance(&prefix, true, &[]);
    assert_eq!(doc_count(&reader), 3);
    assert_eq!(ids(&reader), expected);
}
