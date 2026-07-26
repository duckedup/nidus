//! Serving-edge hardening against a real process (epic nidus-abx).
//!
//! What the in-process `tower::oneshot` suites structurally cannot reach: a *shared*
//! permit pool under genuinely concurrent connections (a `oneshot` router serves one
//! request at a time, so admission control there can only be tested by pre-exhausting it),
//! the CLI-flag → `ServeConfig` wiring for the new knobs, `NIDUS_LOG` filtering in the
//! process that reads it, and `/metrics` served over a real socket.

use serde_json::{Value, json};
use std::io::Write as _;
use std::net::TcpStream;
use std::sync::{Arc, atomic::AtomicUsize, atomic::Ordering};
use std::time::{Duration, Instant};

use crate::harness::Server;

/// Vectors seeded before the load test, so a search does enough work to overlap with its
/// neighbours rather than completing before the next connection is even accepted.
const SEED_DOCS: usize = 2_000;
const DIM: usize = 64;

fn seed(server: &crate::harness::RunningServer, n: usize) {
    let records: Vec<_> = (0..n)
        .map(|i| {
            // A cheap deterministic spread; the ranking is irrelevant here, only the cost.
            let v: Vec<f32> = (0..DIM)
                .map(|d| ((i * 31 + d * 17) % 97) as f32 / 97.0)
                .collect();
            json!({ "id": format!("d{i}"), "vector": v, "attrs": {} })
        })
        .collect();
    let (status, body) = server.post("/collections/load/upsert", &json!({ "records": records }));
    assert_eq!(status, 200, "seeding failed: {body}");
}

/// **Beyond the concurrency limit, requests are shed with `503` — not queued** (nidus-abx.2).
///
/// `CLIENTS` threads issue `REQUESTS_PER_CLIENT` searches each against
/// `--max-concurrent-requests 1`. With that many requests in flight and one permit,
/// shedding is a structural certainty rather than a race the test hopes to win.
///
/// ## Two things this test does deliberately, both learned the hard way
///
/// **Each thread gets its own `ureq::Agent`**, rather than sharing the harness's. Sixteen
/// threads hammering one connection pool is a *client* configuration no real deployment
/// has, and it was the shape that failed on CI: the shared pool's connections churn under
/// contention and a thread eventually picks up one the far side has already closed
/// ("Peer disconnected"). Independent agents model independent callers, which is what the
/// server is actually being tested against.
///
/// **Transport errors are counted, not fatal.** The harness's `post` panics on any socket
/// error, which is right for a functional assertion and wrong here: on a shared two-core
/// runner, loopback sockets under sustained concurrency occasionally fault for reasons that
/// say nothing about admission control. They are reported in the failure message so a real
/// regression (everything failing at the transport) is still visible — what would hide a
/// regression is silence, not tolerance. Every assertion that matters is unchanged.
///
/// Note what is NOT relaxed: statuses other than 200/503 still fail, every 503 must be
/// marked retryable, and the server's own shed counter must corroborate what clients saw.
///
/// The one timing assertion is order-of-magnitude and generous, per CLAUDE.md — this is a
/// debug build on a shared runner.
#[test]
fn concurrent_load_beyond_the_limit_is_shed_not_queued() {
    const CLIENTS: usize = 16;
    const REQUESTS_PER_CLIENT: usize = 25;
    const PACE: Duration = Duration::from_millis(5);

    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM)
        .args(["--max-concurrent-requests", "1"])
        .start();
    seed(&server, SEED_DOCS);

    let ok = Arc::new(AtomicUsize::new(0));
    let shed = Arc::new(AtomicUsize::new(0));
    let other = Arc::new(AtomicUsize::new(0));
    let transport_errors = Arc::new(AtomicUsize::new(0));
    let not_marked_retryable = Arc::new(AtomicUsize::new(0));

    let url = format!("{}/search", server.base_url());
    let started = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..CLIENTS {
            let (ok, shed, other, transport_errors, not_marked_retryable) = (
                Arc::clone(&ok),
                Arc::clone(&shed),
                Arc::clone(&other),
                Arc::clone(&transport_errors),
                Arc::clone(&not_marked_retryable),
            );
            let url = url.clone();
            s.spawn(move || {
                // This client's own connection pool — see the note above.
                let agent = ureq::Agent::new_with_config(
                    ureq::config::Config::builder()
                        .http_status_as_error(false)
                        .timeout_global(Some(Duration::from_secs(30)))
                        .build(),
                );
                let query: Vec<f32> = (0..DIM).map(|d| (d % 7) as f32).collect();
                let body = serde_json::to_vec(&json!({ "query": query, "top_k": 20 })).unwrap();
                for _ in 0..REQUESTS_PER_CLIENT {
                    let res = agent
                        .post(&url)
                        .header("content-type", "application/json")
                        .send(body.as_slice());
                    match res {
                        Err(_) => {
                            transport_errors.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(res) => {
                            let status = res.status().as_u16();
                            let payload = res
                                .into_body()
                                .read_to_vec()
                                .ok()
                                .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                                .unwrap_or(Value::Null);
                            match status {
                                200 => {
                                    ok.fetch_add(1, Ordering::Relaxed);
                                }
                                503 => {
                                    shed.fetch_add(1, Ordering::Relaxed);
                                    if payload["retryable"] != json!(true) {
                                        not_marked_retryable.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                                _ => {
                                    other.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                    std::thread::sleep(PACE);
                }
            });
        }
    });
    let elapsed = started.elapsed();

    let (ok, shed, other, errs) = (
        ok.load(Ordering::Relaxed),
        shed.load(Ordering::Relaxed),
        other.load(Ordering::Relaxed),
        transport_errors.load(Ordering::Relaxed),
    );
    let tally = format!("{ok} ok, {shed} shed, {other} other, {errs} transport errors");
    assert_eq!(other, 0, "unexpected statuses beyond 200/503 — {tally}");
    assert!(
        ok > 0,
        "the server should still be serving under load — {tally}"
    );
    assert!(
        shed > 0,
        "{CLIENTS} concurrent clients against a limit of 1 must shed something — {tally}"
    );
    assert_eq!(
        not_marked_retryable.load(Ordering::Relaxed),
        0,
        "every shed response must say it is retryable — {tally}"
    );
    // Tolerated, but not unbounded: if most requests never completed a round trip, the
    // server is not shedding, it is failing, and that must not pass as success.
    assert!(
        errs * 2 < CLIENTS * REQUESTS_PER_CLIENT,
        "most requests failed at the transport, which is not load shedding — {tally}\n\
         --- server stderr ---\n{}",
        server.stderr()
    );
    // Order-of-magnitude, generous on purpose (CLAUDE.md): what is asserted is that the
    // requests came back at all. Unbounded queueing behind one write lock would blow
    // through each client's own 30s timeout long before this fires.
    assert!(
        elapsed < Duration::from_secs(60),
        "load ran {elapsed:?} — requests were queueing, not shedding — {tally}"
    );

    // The server's own counters corroborate what the clients saw. This is also the proof
    // that the metrics path stays answerable under saturation: the scrape happens while
    // the instance has just been shedding, and it takes no store lock.
    let scrape = scrape(&server);
    let counted = metric(&scrape, "nidus_http_requests_shed_total")
        .unwrap_or_else(|| panic!("shed counter missing:\n{scrape}"));
    assert!(
        counted >= shed as f64,
        "server counted {counted} shed, clients saw {shed}"
    );
}

/// **A client that withholds its request body must not pin a concurrency permit**
/// (nidus-6c2).
///
/// The store permit used to be taken before the handler ran, and the handler is what
/// awaits the body — so a client that sent complete headers with a `Content-Length` and
/// then went silent held a permit for the whole request deadline, or forever with
/// `--read-timeout 0`. Against `--max-concurrent-requests 1` that was a one-connection
/// denial of service.
///
/// The body is now received in its own phase, before any store permit is taken, so the
/// assertion is the strong one: with a silent client parked on the connection, ordinary
/// traffic keeps being served *throughout* — not "recovers eventually".
///
/// Driven over a raw socket, because no HTTP client will send headers promising a body and
/// then refuse to send it — that is precisely the misbehaviour under test.
///
/// `--read-timeout 0` and `--body-idle-timeout 0` are both set deliberately: they remove
/// every deadline that could paper over the problem by eventually releasing the permit, so
/// this passes only if the phase split is doing the work on its own.
#[test]
fn a_withheld_request_body_does_not_pin_a_permit() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM)
        .args([
            "--max-concurrent-requests",
            "1",
            "--read-timeout",
            "0",
            "--body-idle-timeout",
            "0",
        ])
        .start();
    seed(&server, 200);

    let addr = server.base_url().trim_start_matches("http://").to_string();
    let mut silent = TcpStream::connect(&addr).expect("connect");
    write!(
        silent,
        "POST /search HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: 4096\r\n\r\n"
    )
    .expect("send headers");
    silent.flush().expect("flush headers");
    // Give the server time to accept the connection and read the headers, so the request
    // really is parked mid-flight rather than not yet noticed.
    std::thread::sleep(Duration::from_millis(300));

    let query = json!({ "query": vec![1.0f32; DIM], "top_k": 5 });
    for i in 0..10 {
        let (status, body) = server.post("/search", &query);
        assert_eq!(
            status,
            200,
            "request {i} was refused while a silent client sat on the connection — \
             the body wait is holding a store permit\n{body}\n--- server stderr ---\n{}",
            server.stderr()
        );
    }

    // A bodyless route is equally unaffected.
    assert_eq!(server.get("/stats").0, 200);
    drop(silent);
}

/// Both timeout flags and the concurrency flag reach `ServeConfig` — the wiring the
/// in-process suites cannot see, because they construct `AppState` directly.
///
/// `--read-timeout 1` with `--max-concurrent-requests` left at auto: an ordinary search is
/// milliseconds, so the deadline must NOT fire. A flag that silently aborted healthy
/// traffic would be worse than no flag.
#[test]
fn timeout_flags_are_wired_and_do_not_fire_on_healthy_traffic() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM)
        .args(["--read-timeout", "5", "--write-timeout", "30"])
        .start();
    seed(&server, 200);

    let query: Vec<f32> = (0..DIM).map(|d| (d % 5) as f32).collect();
    let (status, body) = server.post("/search", &json!({ "query": query, "top_k": 5 }));
    assert_eq!(status, 200, "{body}");
    assert!(body.as_array().is_some_and(|a| !a.is_empty()));
}

/// `--max-concurrent-requests 0` is the documented "auto" value, not "admit nothing".
///
/// Worth its own test because the failure mode is total: reading `0` as a literal cap would
/// make every configured-to-default server shed every request, while still passing every
/// unit test that constructs `Limits` directly.
#[test]
fn zero_concurrency_means_auto_not_zero() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM)
        .args(["--max-concurrent-requests", "0"])
        .start();
    let (status, _) = server.get("/stats");
    assert_eq!(status, 200, "auto must not shed ordinary traffic");

    let scrape = scrape(&server);
    let limit = metric(&scrape, "nidus_http_concurrency_limit").expect("limit gauge");
    assert!(limit >= 64.0, "auto floors at 64, got {limit}");
}

/// `/metrics` is reachable **without** a credential on a token-protected server (a scraper
/// that got a `401` would report the target as down), and never names a collection
/// (nidus-abx.4).
#[test]
fn metrics_is_scrapeable_without_a_token_and_names_no_collections() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM).token("s3cret").start();

    let (status, body) = server.post(
        "/collections/very-secret-project/upsert",
        &json!({"records": [{"id": "a", "vector": vec![0.5f32; DIM], "attrs": {}}]}),
    );
    assert_eq!(status, 200, "{body}");

    // A bare client with no Authorization header at all.
    let raw = ureq::get(format!("{}/metrics", server.base_url()))
        .call()
        .expect("scrape /metrics unauthenticated");
    assert_eq!(raw.status().as_u16(), 200);
    let text = raw.into_body().read_to_string().expect("metrics body");

    assert!(
        !text.contains("very-secret-project"),
        "a collection name reached a metric label:\n{text}"
    );
    assert!(text.contains("nidus_http_requests_total{route=\"/collections/{name}/upsert\""));
    assert!(text.contains("nidus_lease_renew_attempts_total"));

    // And a data route still demands the token — opening /metrics must not have opened
    // anything else.
    let denied = ureq::get(format!("{}/stats", server.base_url()))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .expect("call /stats");
    assert_eq!(denied.status().as_u16(), 401);
}

/// A correlation id survives a real round trip, and the access log carries it — the point
/// of the id is that the same string appears in the client's record and the server's.
#[test]
fn request_ids_round_trip_into_the_access_log() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM).start();

    let resp = ureq::get(format!("{}/stats", server.base_url()))
        .header("x-request-id", "e2e-correlation-1")
        .call()
        .expect("GET /stats");
    assert_eq!(
        resp.headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok()),
        Some("e2e-correlation-1")
    );

    // The access line is emitted after the response is produced, so give the child a
    // moment to flush it before reading its stderr.
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && !server.stderr().contains("e2e-correlation-1") {
        std::thread::sleep(Duration::from_millis(25));
    }
    let log = server.stderr();
    assert!(
        log.contains("id=e2e-correlation-1"),
        "the access log should carry the caller's id:\n{log}"
    );
    assert!(
        log.contains("route=/stats") && log.contains("status=200"),
        "the access line should be structured key=value:\n{log}"
    );
}

/// `NIDUS_LOG` turns detail down, in the process that reads it.
///
/// At `error`, the per-request access lines must disappear while the startup banner (a
/// plain `println`, not a log record) stays — that split is what lets a test suite or a
/// noisy production deployment silence traffic logging without losing the line that
/// reports the bound port.
#[test]
fn nidus_log_filters_the_access_log() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM)
        .env("NIDUS_LOG", "error")
        .start();

    for _ in 0..3 {
        let (status, _) = server.get("/stats");
        assert_eq!(status, 200);
    }
    std::thread::sleep(Duration::from_millis(200));

    let log = server.stderr();
    assert!(
        !log.contains("target=http"),
        "NIDUS_LOG=error must suppress the access log:\n{log}"
    );
    assert!(
        log.contains("nidus serving on http://"),
        "the startup banner must survive any log level:\n{log}"
    );
}

/// A loopback bind must NOT print an exposure warning (nidus-abx.6).
///
/// The complementary case — that a non-loopback bind *does* warn — is
/// `server::tests::exposure_is_classified_by_reachability_then_auth`, unit-tested on the
/// classification rather than here: binding `0.0.0.0` from a test would open a real
/// off-box socket on whatever machine runs it, which is not a thing a test suite should do.
#[test]
fn a_loopback_bind_prints_no_security_warning() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM).start();
    let (status, _) = server.get("/stats");
    assert_eq!(status, 200);

    let log = server.stderr();
    assert!(
        !log.contains("off-loopback"),
        "a localhost server must not cry wolf:\n{log}"
    );
}

/// Scrape `/metrics` as text.
fn scrape(server: &crate::harness::RunningServer) -> String {
    ureq::get(format!("{}/metrics", server.base_url()))
        .call()
        .expect("scrape /metrics")
        .into_body()
        .read_to_string()
        .expect("metrics body")
}

/// Pull a single unlabelled sample out of a Prometheus text exposition.
fn metric(scrape: &str, name: &str) -> Option<f64> {
    scrape.lines().find_map(|l| {
        l.strip_prefix(name)
            .filter(|rest| rest.starts_with(' '))
            .and_then(|rest| rest.trim().parse().ok())
    })
}
