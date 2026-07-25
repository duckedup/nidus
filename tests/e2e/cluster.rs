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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
        // error would explain better than we could here.
        //
        // Reduce the URL to one reachable `host:port`: drop the scheme, then any `/db` or
        // `?cluster=true` tail, then take the FIRST of a comma-separated seed list — a
        // Valkey Cluster URL carries every seed, and handing the whole list to
        // `TcpStream::connect` fails on a perfectly healthy cluster. Reaching one seed is
        // all this guard needs to prove.
        let addr = redis
            .split_once("://")
            .map_or(redis.as_str(), |(_, rest)| rest)
            .split(['/', '?'])
            .next()
            .unwrap_or_default()
            .split(',')
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
    instance_with_ttl(prefix, read_only, LOCK_TTL_SECS, extra)
}

/// [`instance`] with an explicit lease TTL, for the tests whose whole point is the TTL.
fn instance_with_ttl(
    prefix: &str,
    read_only: bool,
    ttl: u32,
    extra: &[&str],
) -> (tempfile::TempDir, RunningServer) {
    let (dir, server) = build_instance(prefix, read_only, ttl, extra);
    server.await_ready_or_panic();
    (dir, server)
}

/// Like [`instance`] but does **not** wait for the store to open — for a standby writer,
/// which stays unready on purpose until the incumbent's lease lapses.
fn instance_unready(prefix: &str, extra: &[&str]) -> (tempfile::TempDir, RunningServer) {
    build_instance(prefix, false, LOCK_TTL_SECS, extra)
}

/// [`instance_unready`] with an explicit lease TTL.
fn instance_unready_with_ttl(
    prefix: &str,
    ttl: u32,
    extra: &[&str],
) -> (tempfile::TempDir, RunningServer) {
    build_instance(prefix, false, ttl, extra)
}

fn build_instance(
    prefix: &str,
    read_only: bool,
    ttl: u32,
    extra: &[&str],
) -> (tempfile::TempDir, RunningServer) {
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
        ttl.to_string(),
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
        // Lease tracing on for every cluster instance. A lease bug is a multi-process race
        // that reproduces here and essentially nowhere else, and the harness surfaces a
        // child's stderr only when a test fails — so this costs nothing on a green run and is
        // the whole diagnosis on a red one. nidus-lp4.7 took a second pass precisely because
        // these lines (with their pids) were not there the first time.
        .env("NIDUS_LEASE_DEBUG", "1")
        .start_unready();
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

/// **Automatic promotion — the availability property, with no external restart.**
///
/// Before `--wait-for-lease`, a second writer exited the instant it found the lease held
/// (asserted above), so a "hot standby" could only be approximated by a supervisor
/// restarting a crash-looping pod: failover took lease TTL *plus* whatever backoff the
/// supervisor had reached, and `CrashLoopBackOff` is an alert rather than a design.
///
/// Here the standby stays up and is promoted on its own. Note what it asserts along the
/// way, because each part was a separate way to get this wrong:
///
/// * while waiting, the standby is **live but not ready** — a failing liveness probe would
///   have a supervisor kill the very instance meant to be waiting, and a passing readiness
///   probe would have a load balancer send it traffic it cannot serve;
/// * its data routes answer `503`, not `500` — nothing is broken, there is just no store;
/// * promotion happens within roughly the lease TTL of the incumbent dying, **without any
///   external restart** — the assertion that was impossible to write before;
/// * and the promoted instance can write, and sees everything its predecessor committed.
#[cfg(unix)]
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn standby_is_promoted_after_the_writer_dies() {
    let prefix = unique_prefix("promote");
    let (_wdir, writer) = instance(&prefix, false, &[]);
    seed(&writer);
    assert_eq!(upsert(&writer, "before", [1, 0, 0]).0, 200);

    // The standby: a cluster writer that waits instead of exiting.
    let (_sdir, standby) = instance_unready(&prefix, &["--wait-for-lease"]);

    // Waiting, not broken: alive, deliberately unready, and honest about why.
    assert!(standby.is_live(), "a waiting standby must answer liveness");
    assert!(
        !standby.is_ready(),
        "a standby must not report ready while the incumbent holds the lease"
    );
    let (status, body) = standby.get("/stats");
    assert_eq!(
        status, 503,
        "an unpromoted standby should answer 503, got {status} {body}"
    );

    // The incumbent dies outright — no clean release, no lease handback.
    writer.kill();

    // Promotion is the standby's own doing. Allow generous slack over the TTL for a loaded
    // CI runner; the point is that it happens at all, unattended.
    let promoted = standby
        .ready_within(past_the_lease() + Duration::from_secs(10))
        .expect("standby must be promoted after the writer dies, with no external restart");
    println!("standby promoted {promoted:.2?} after the writer was killed");

    // A promoted standby is a full writer: it can write, and it inherited the state.
    assert_eq!(upsert(&standby, "after", [0, 1, 0]).0, 200);
    assert_eq!(
        ids(&standby),
        vec!["after", "before"],
        "the promoted standby must see its predecessor's committed data"
    );
}

/// **An idle writer keeps its lease, so a standby does not steal it** (nidus-lp4.6).
///
/// The lease used to be renewed *only* at the start of a write batch. That was fine before
/// standbys existed: nothing was waiting to take over. Once `--wait-for-lease` shipped, a
/// writer that simply had nothing to do for longer than `lock_ttl` — or was mid-way through
/// one very large batch — would have its lease judged stale and be replaced, discarding a
/// perfectly healthy writer (and, mid-batch, its work).
///
/// `nidus serve` now renews out of band on a timer using a `LeaseRenewer` that does **not**
/// need the store lock, so renewal continues even while a long write holds the guard.
///
/// **This was the nidus-lp4.7 reproduction** — a standby acquiring the lease while the
/// incumbent was alive and well within its TTL, leaving two instances both reporting
/// `holds_writer_handle` (a mutual-exclusion failure, not merely an availability one). It
/// found two real bugs and now passes, so it is part of the normal suite. Its assertions were
/// never weakened to get there; see the git history of `backend/object.rs` for the fixes.
#[cfg(unix)]
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn idle_writer_keeps_its_lease_against_a_waiting_standby() {
    // A longer TTL than the suite default ON PURPOSE: lease timestamps are second
    // granularity, so a 2s TTL cannot distinguish "renewed 700ms ago" from "expired".
    const TTL: u32 = 8;
    let prefix = unique_prefix("idle-lease");
    let (_wdir, writer) = instance_with_ttl(&prefix, false, TTL, &[]);
    seed(&writer);
    assert_eq!(upsert(&writer, "a", [1, 0, 0]).0, 200);

    let (_sdir, standby) = instance_unready_with_ttl(&prefix, TTL, &["--wait-for-lease"]);
    assert!(
        !standby.is_ready(),
        "standby waits while the writer is alive"
    );

    // Idle for well past the lease TTL, issuing no writes at all — so nothing in the write
    // path renews. Only the out-of-band renewer can keep the lease alive here.
    std::thread::sleep(Duration::from_secs(20));

    assert!(
        !standby.is_ready(),
        "the standby must NOT have been promoted: an idle writer is still a live writer, \
         and its lease should have been renewed out of band"
    );
    assert!(writer.is_ready(), "the idle writer is still healthy");
    assert_eq!(
        writer.get("/cluster").1["fenced"],
        false,
        "an idle writer must not have been fenced"
    );

    // The mutual-exclusion property stated directly: whatever the standby thinks its role is,
    // it must not claim the writer handle. (While it waits it has no store open, so `/cluster`
    // answers 503 — - that is itself proof it was not promoted. If it ever answers, the claim
    // must still be false.)
    let (status, body) = standby.get("/cluster");
    if status == 200 {
        assert_eq!(
            body["holds_writer_handle"], false,
            "two instances must never both hold the writer handle: {body}"
        );
    }

    // And the writer can still write — proof it genuinely still holds the lease, not merely
    // that the standby was slow to notice.
    assert_eq!(upsert(&writer, "b", [0, 1, 0]).0, 200);
    assert_eq!(ids(&writer), vec!["a", "b"]);
}

/// **A fenced writer reports unready without waiting for a write to find out** (nidus-lp4.7).
///
/// `fenced` used to be latched *only* by a failing write. The background renewer — which runs
/// every `lock_ttl/3` and is therefore the first thing to learn the lease is gone — printed
/// the loss to stderr and dropped it. So a superseded writer with no traffic kept answering
/// `/ready` with 200 and reporting `holds_writer_handle: true` indefinitely, and an
/// orchestrator had no way to see it until a write happened to arrive and fail. On a
/// low-traffic store that could be a long time, and it is precisely the interval during which
/// two instances both look like the writer.
///
/// Distinct from `fenced_writer_reports_unready`, which latches the state *via* a rejected
/// write: this one asserts the discovery happens on its own, with no write attempted at all.
#[cfg(unix)]
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn a_superseded_writer_discovers_it_is_fenced_without_any_write() {
    let prefix = unique_prefix("bg-fence");
    let (_adir, writer_a) = instance(&prefix, false, &[]);
    seed(&writer_a);
    assert_eq!(upsert(&writer_a, "from-a", [1, 0, 0]).0, 200);
    assert!(writer_a.is_ready(), "a healthy writer is ready");

    // A stalls past its lease; B takes it over and commits.
    writer_a.pause();
    std::thread::sleep(past_the_lease());
    let (_bdir, writer_b) = instance(&prefix, false, &[]);
    assert_eq!(upsert(&writer_b, "from-b", [0, 1, 0]).0, 200);

    // A wakes up. Nothing writes to it — the only thing that can notice the takeover is its
    // own background renewal tick. Give it a few of those (`lock_ttl/3`) plus slack for a
    // loaded debug build.
    writer_a.resume();
    let deadline = Instant::now() + Duration::from_secs(15);
    while writer_a.is_ready() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(200));
    }

    assert!(
        !writer_a.is_ready(),
        "a superseded writer must discover it is fenced on its own renewal timer, without \
         needing a write to fail first: {}",
        writer_a.stderr()
    );
    let (status, body) = writer_a.get("/cluster");
    assert_eq!(status, 200);
    assert_eq!(body["fenced"], true, "/cluster must report it: {body}");
    assert_eq!(
        body["holds_writer_handle"], false,
        "and it must stop claiming the handle B now holds: {body}"
    );

    // B is unaffected — exactly one instance holds the handle.
    assert!(writer_b.is_ready());
    assert_eq!(writer_b.get("/cluster").1["holds_writer_handle"], true);
}

/// **A fenced writer must stop reporting ready** (nidus-lp4.1).
///
/// This was the sharpest production gap: `/health` was a hardcoded `"ok"`, and the shipped
/// chart probed it for *readiness*. So a writer that had been superseded — every write
/// failing with "writer lease lost" — stayed in the Service and kept receiving writes it
/// could not perform. The one signal Kubernetes could act on was inverted.
///
/// Reproduces it the way it actually happens: stall the writer past its lease, let a peer
/// take over, then let the stalled writer wake up and try to write.
#[cfg(unix)]
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn fenced_writer_reports_unready() {
    let prefix = unique_prefix("fenced-ready");
    let (_adir, writer_a) = instance(&prefix, false, &[]);
    seed(&writer_a);
    assert_eq!(upsert(&writer_a, "from-a", [1, 0, 0]).0, 200);
    assert!(writer_a.is_ready(), "a healthy writer is ready");

    // A stalls past its lease; B takes over and commits.
    writer_a.pause();
    std::thread::sleep(past_the_lease());
    let (_bdir, writer_b) = instance(&prefix, false, &[]);
    assert_eq!(upsert(&writer_b, "from-b", [0, 1, 0]).0, 200);

    // A wakes and discovers it is fenced. The failed write is what latches the state.
    writer_a.resume();
    let (status, _) = upsert(&writer_a, "from-a-stale", [0, 0, 1]);
    assert_ne!(status, 200, "a superseded writer's write must be refused");

    // The point of the test: that fact is now visible to an orchestrator.
    assert!(
        !writer_a.is_ready(),
        "a fenced writer must report NOT ready so the load balancer stops sending it writes"
    );
    // …but it is still alive, so a supervisor restarts it rather than reporting it hung.
    assert!(
        writer_a.is_live(),
        "a fenced writer is still a live process"
    );

    let (status, body) = writer_a.get("/cluster");
    assert_eq!(status, 200);
    assert_eq!(
        body["fenced"], true,
        "/cluster must report the fencing: {body}"
    );
    assert_eq!(
        body["holds_writer_handle"], false,
        "a fenced writer no longer holds the handle: {body}"
    );

    // The peer that took over is unaffected and still ready.
    assert!(writer_b.is_ready());
    assert_eq!(writer_b.get("/cluster").1["fenced"], false);
}

/// **A reader past its staleness bound reports unready** (nidus-lp4.4).
///
/// A cluster reader only advances when something refreshes it, so a reader whose refresher
/// died would otherwise serve ever-older results while looking perfectly healthy. With
/// `--max-staleness` set, readiness is what escalates that.
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn stale_reader_reports_unready() {
    let prefix = unique_prefix("stale-ready");
    let (_wdir, writer) = instance(&prefix, false, &[]);
    seed(&writer);
    assert_eq!(upsert(&writer, "a", [1, 0, 0]).0, 200);

    // A reader that must verify itself current every 2s, with nothing refreshing it.
    let (_rdir, reader) = instance(&prefix, true, &["--max-staleness", "2"]);
    assert!(
        reader.is_ready(),
        "a freshly opened reader has just verified itself current"
    );

    // Let the bound lapse without refreshing.
    std::thread::sleep(Duration::from_secs(4));
    assert!(
        !reader.is_ready(),
        "past --max-staleness a reader must report NOT ready"
    );

    // Reads still work — the bound governs *routing*, not correctness. An operator who
    // ignores readiness still gets (stale) answers rather than errors.
    assert_eq!(reader.get("/stats").0, 200);

    // A refresh re-verifies it and readiness returns, so this is recoverable rather than
    // a one-way trip.
    assert_eq!(reader.post("/refresh", &json!({})).0, 200);
    assert!(
        reader.is_ready(),
        "after refreshing, the reader is current again and ready"
    );
}

/// `--refresh-interval` keeps a reader current with no sidecar, so a staleness bound
/// tighter than the interval never trips (nidus-lp4.4).
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn self_refreshing_reader_stays_fresh_and_current() {
    let prefix = unique_prefix("self-refresh");
    let (_wdir, writer) = instance(&prefix, false, &[]);
    seed(&writer);
    assert_eq!(upsert(&writer, "first", [1, 0, 0]).0, 200);

    // Refreshes itself every second; must stay inside a 5s staleness bound unattended.
    let (_rdir, reader) = instance(
        &prefix,
        true,
        &["--refresh-interval", "1", "--max-staleness", "5"],
    );
    assert_eq!(doc_count(&reader), 1);

    // Commit more, then wait for the interval to pick it up — no POST /refresh here.
    assert_eq!(upsert(&writer, "second", [0, 1, 0]).0, 200);
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    while std::time::Instant::now() < deadline && doc_count(&reader) < 2 {
        std::thread::sleep(Duration::from_millis(200));
    }
    assert_eq!(
        doc_count(&reader),
        2,
        "the interval refresher should have adopted the writer's commit unattended"
    );
    assert_eq!(ids(&reader), vec!["first", "second"]);
    assert!(
        reader.is_ready(),
        "a self-refreshing reader stays inside its staleness bound"
    );
}

/// `/cluster` gives an operator the state they need mid-incident, and the writer and reader
/// disagree in exactly the ways they should (nidus-lp4.5).
#[test]
#[ignore = "needs minio + valkey (just test-e2e-cluster)"]
fn cluster_endpoint_distinguishes_writer_from_reader() {
    let prefix = unique_prefix("observability");
    let (_wdir, writer) = instance(&prefix, false, &[]);
    seed(&writer);
    assert_eq!(upsert(&writer, "a", [1, 0, 0]).0, 200);
    let (_rdir, reader) = instance(&prefix, true, &[]);

    let (status, w) = writer.get("/cluster");
    assert_eq!(status, 200);
    assert_eq!(w["role"], "ClusterWriter");
    assert_eq!(w["cluster"], true);
    assert_eq!(w["holds_writer_handle"], true);
    assert_eq!(w["staleness_secs"], 0, "a writer is never stale");
    let owner = w["lease_owner"]
        .as_str()
        .expect("a cluster writer exposes its fencing token");
    assert!(!owner.is_empty());

    let (status, r) = reader.get("/cluster");
    assert_eq!(status, 200);
    assert_eq!(r["role"], "ClusterReader");
    assert_eq!(r["holds_writer_handle"], false, "a reader takes no handle");
    assert_eq!(
        r["lease_owner"],
        Value::Null,
        "a reader has no fencing token"
    );

    // Both are serving the same commit, so the counter agrees — which is what makes it
    // usable as a lag measure when they disagree.
    assert_eq!(w["commit_version"], r["commit_version"]);
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
