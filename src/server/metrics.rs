//! Traffic-level observability: `GET /metrics`, and the access log behind it (nidus-abx.4).
//!
//! `crate::metrics` holds the counters the *library* maintains (lease renewals, backend
//! retries, which search path served a query). This module adds the ones only the *server*
//! can know — request rate, latency distribution, and error rate by route — and renders
//! both as Prometheus text.
//!
//! ## Bounded cardinality, by construction
//!
//! Every request is labelled with a route **template**, not its path, and
//! [`route_label`] returns `&'static str` — so a collection name physically cannot reach a
//! metric label. That is a correctness property (an unbounded label set is the classic way
//! to take a Prometheus server down with a metrics endpoint) and a disclosure one: the
//! scrape reveals traffic shape, never what is stored. It is what makes leaving `/metrics`
//! unauthenticated a defensible default.
//!
//! Deliberately *not* `axum::extract::MatchedPath`: whether that extension is populated
//! depends on where in the layer stack a middleware sits, and a metric that silently
//! degrades to `other` because a layer moved is worse than one that never depended on it.
//!
//! ## Cheapness is a hard constraint
//!
//! Recording is `fetch_add(Relaxed)` on a fixed, preallocated array — no map, no
//! allocation, and above all no lock. The store is already serialised behind one `RwLock`;
//! an observability layer that contended with it would become the problem it exists to
//! diagnose.
//!
//! ## The two in-flight gauges measure different things, on purpose (nidus-bcg)
//!
//! * `nidus_http_requests_in_flight` — handler futures alive, probes included. Counted by
//!   the [`InFlight`] guard, so a client that disconnects mid-request decrements it.
//! * `nidus_http_admitted_in_flight` — **concurrency permits held**, read straight off the
//!   semaphore in [`super::limits`]. This is the one to correlate with
//!   `nidus_http_requests_shed_total`: shedding happens exactly when it reaches
//!   `nidus_http_concurrency_limit`.
//!
//! The permit gauge is named after the admission decision, and it reports that decision
//! exactly. It is therefore **not** a count of work executing: when a request hits its
//! deadline the permit is released immediately — holding it would shed live traffic on
//! behalf of work nobody is waiting for — while the blocking task keeps running until it
//! observes the cancellation token. For that window the gauge is low.
//!
//! The window is bounded by cooperative cancellation: the scan kernels check the token
//! every few thousand rows, so it is milliseconds, not a whole scan. It is entered exactly
//! `nidus_http_requests_timed_out_total` times — if that counter is flat, the discrepancy
//! has never occurred on this instance, which is the measurement to make before wanting a
//! second gauge for it.
//!
//! Folding orphaned work into the permit gauge was considered and rejected: the gauge
//! would then no longer correspond to the admission decision it is named after, which is
//! the only reason to graph it against the shed count.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use axum::{
    extract::{Request, State},
    http::{HeaderName, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

use super::AppState;

/// The route templates traffic is bucketed by. Fixed at compile time, so the label set is
/// too. `other` catches 404s and anything a future route forgets to name — a visible
/// bucket rather than silence.
const ROUTES: &[&str] = &[
    "/health",
    "/ready",
    "/metrics",
    "/cluster",
    "/stats",
    "/collections",
    "/collections/{name}",
    "/collections/{name}/meta",
    "/collections/{name}/upsert",
    "/collections/{name}/delete",
    "/collections/{name}/records",
    "/collections/{name}/fts-schema",
    "/collections/{name}/remember",
    "/collections/{name}/recall",
    "/search",
    "/text-search",
    "/hybrid-search",
    "/list",
    "/flush",
    "/compact",
    "/refresh",
    "other",
];

/// Upper bounds, in seconds, for the latency histogram (plus an implicit `+Inf`).
///
/// Spread wide on purpose: nidus-8fn measured the HTTP path at ~100–140µs of constant
/// overhead, and a large upsert legitimately runs for minutes, so the honest range this has
/// to resolve spans five orders of magnitude. Buckets bunch where the interesting
/// transitions are — sub-millisecond for a small search, seconds for a scan over a big
/// store, tens of seconds for a batch write.
const BUCKETS: &[f64] = &[
    0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// Per-route request counts, split by status class, plus a latency histogram.
struct RouteStats {
    /// Index 0..=4 → 1xx..5xx. A class rather than the exact code: it answers "what is the
    /// error rate" without multiplying the label set by every status the server can return,
    /// and the two statuses an operator actually needs to tell apart under load — a shed
    /// `503` and a deadline `504` — have their own counters below.
    by_class: [AtomicU64; 5],
    /// Cumulative bucket counts (`le`), Prometheus-style: each is "requests at or under
    /// this bound", filled in at render time from the per-bucket tallies here.
    buckets: [AtomicU64; BUCKETS.len()],
    /// Requests slower than every bound — the `+Inf` bucket's exclusive tail.
    over: AtomicU64,
    /// Summed latency in microseconds. Integer, so the sum never drifts the way a
    /// concurrently-added float would.
    sum_micros: AtomicU64,
    count: AtomicU64,
}

impl RouteStats {
    const fn new() -> RouteStats {
        #[allow(clippy::declare_interior_mutable_const)]
        const Z: AtomicU64 = AtomicU64::new(0);
        RouteStats {
            by_class: [Z; 5],
            buckets: [Z; BUCKETS.len()],
            over: Z,
            sum_micros: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    fn record(&self, status: StatusCode, micros: u64) {
        let class = (status.as_u16() / 100).clamp(1, 5) as usize - 1;
        self.by_class[class].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_micros.fetch_add(micros, Ordering::Relaxed);
        let secs = micros as f64 / 1_000_000.0;
        match BUCKETS.iter().position(|&b| secs <= b) {
            Some(i) => self.buckets[i].fetch_add(1, Ordering::Relaxed),
            None => self.over.fetch_add(1, Ordering::Relaxed),
        };
    }
}

/// Server-side traffic metrics, one entry per route template.
pub(super) struct HttpMetrics {
    routes: [RouteStats; ROUTES.len()],
    /// Requests currently being handled, probes included.
    in_flight: crate::metrics::Gauge,
    /// Requests whose client vanished before a response was produced — axum drops the
    /// handler future when the connection closes, so these never reach a status. Worth its
    /// own counter: a rising count is clients giving up, which looks like nothing at all in
    /// a request/status breakdown that only ever sees completed requests.
    cancelled: crate::metrics::Counter,
    /// Requests refused with `503` because the concurrency cap was reached (nidus-abx.2).
    pub(super) shed: crate::metrics::Counter,
    /// Requests abandoned with `504` at their deadline. Note this counts *clients freed*,
    /// not work stopped — see `limits.rs`.
    pub(super) timed_out: crate::metrics::Counter,
}

impl HttpMetrics {
    const fn new() -> HttpMetrics {
        #[allow(clippy::declare_interior_mutable_const)]
        const S: RouteStats = RouteStats::new();
        HttpMetrics {
            routes: [S; ROUTES.len()],
            in_flight: crate::metrics::Gauge::new(),
            cancelled: crate::metrics::Counter::new(),
            shed: crate::metrics::Counter::new(),
            timed_out: crate::metrics::Counter::new(),
        }
    }
}

/// RAII for the in-flight gauge.
///
/// **Not** a matching `inc()`/`dec()` pair around the await: axum drops the handler future
/// when a client disconnects mid-request, so the `dec()` would simply never run and the
/// gauge would ratchet upward forever — a metric that silently becomes a lie, on exactly
/// the traffic pattern (impatient clients under load) you would be watching it for. `Drop`
/// runs on the cancellation path too.
struct InFlight {
    /// Set once a response was produced, so `Drop` can tell "finished" from "abandoned".
    completed: bool,
}

impl InFlight {
    fn enter() -> InFlight {
        HTTP.in_flight.inc();
        InFlight { completed: false }
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        HTTP.in_flight.dec();
        if !self.completed {
            HTTP.cancelled.inc();
        }
    }
}

static HTTP: HttpMetrics = HttpMetrics::new();

pub(super) fn http() -> &'static HttpMetrics {
    &HTTP
}

/// Map a request path to its slot in [`ROUTES`].
///
/// An **index**, not a label: the index is what `HTTP.routes` is addressed by, and deriving
/// a `&'static str` only to search for it again would be two scans and an unreachable
/// fallback. `ROUTES[route_index(path)]` is the name.
///
/// Collection names are collapsed to `{name}`, so `/collections/secret-project/upsert` and
/// `/collections/notes/upsert` share one series — see the module docs on why that matters.
fn route_index(path: &str) -> usize {
    let slot = |name: &str| ROUTES.iter().position(|r| *r == name).expect("known route");
    // Exact routes first — the common case, and unambiguous.
    if let Some(i) = ROUTES.iter().position(|r| *r == path && !r.contains('{')) {
        return i;
    }
    // `/collections/<name>[/<verb>]`. Anything else is `other`.
    let Some(rest) = path.strip_prefix("/collections/") else {
        return slot("other");
    };
    slot(match rest.split_once('/') {
        // `/collections/<name>` — create and drop.
        None => "/collections/{name}",
        Some((_name, verb)) => match verb {
            "meta" => "/collections/{name}/meta",
            "upsert" => "/collections/{name}/upsert",
            "delete" => "/collections/{name}/delete",
            "records" => "/collections/{name}/records",
            "fts-schema" => "/collections/{name}/fts-schema",
            "remember" => "/collections/{name}/remember",
            "recall" => "/collections/{name}/recall",
            _ => "other",
        },
    })
}

#[cfg(test)]
fn route_label(path: &str) -> &'static str {
    ROUTES[route_index(path)]
}

/// Header carrying the correlation id, in and out.
const REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// The outermost middleware: time the request, label it, log it, echo its id.
///
/// Sits outside auth and backpressure so a `401` and a shed `503` are both counted — an
/// error rate that excludes the errors a client is actually seeing is worse than none.
pub(super) async fn observe(req: Request, next: Next) -> Response {
    let started = Instant::now();
    let slot = route_index(req.uri().path());
    let route = ROUTES[slot];
    let method = req.method().clone();
    let request_id = request_id(&req);

    let mut guard = InFlight::enter();
    let mut resp = next.run(req).await;
    guard.completed = true;

    let elapsed = started.elapsed();
    let status = resp.status();
    HTTP.routes[slot].record(status, elapsed.as_micros() as u64);

    // Echoed so a caller can quote the id in a bug report and an operator can grep for it —
    // that correlation is the whole reason the id exists.
    if let Ok(v) = HeaderValue::from_str(&request_id) {
        resp.headers_mut().insert(REQUEST_ID, v);
    }

    // One access line per request. `info` for ordinary traffic, `warn` for a server fault:
    // a 5xx is the thing worth seeing when the level is turned down, and a 4xx is the
    // client's problem, not the instance's.
    let level = if status.is_server_error() {
        crate::diag::Level::Warn
    } else {
        crate::diag::Level::Info
    };
    crate::diag::diag!(
        level,
        "http",
        "request",
        "id" => request_id,
        "method" => method,
        "route" => route,
        "status" => status.as_u16(),
        // Three decimals: microsecond resolution, without the seventeen-digit float tail
        // that makes a log line unreadable and a naive parser's life harder.
        "duration_ms" => format_args!("{:.3}", elapsed.as_secs_f64() * 1000.0),
    );
    resp
}

/// Take the caller's `x-request-id` when it is present and sane, otherwise mint one.
///
/// Honouring an inbound id is what makes nidus a link in someone else's trace rather than
/// the place correlation stops. It is bounded and filtered before use: the value ends up in
/// a log line and a response header, and an unvalidated header is how a caller injects a
/// newline into your logs.
fn request_id(req: &Request) -> String {
    if let Some(given) = req
        .headers()
        .get(&REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| {
            !v.is_empty()
                && v.len() <= 64
                && v.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
        })
    {
        return given.to_string();
    }
    mint_request_id()
}

/// A fresh id: process id and a monotonic counter, hex.
///
/// Not a UUID and not random — nothing here needs unguessability, only uniqueness within a
/// deployment's log stream, and pulling `uuid`/`rand` into the tree for that would be a
/// dependency decision (CLAUDE.md) in exchange for nothing. The pid disambiguates instances
/// on one host; the counter disambiguates requests within one process.
fn mint_request_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("{:x}-{n:x}", std::process::id())
}

/// `GET /metrics` — the Prometheus text exposition, hand-rolled.
///
/// Unauthenticated on purpose (see `auth::is_public`), and lock-free: every number read
/// here is an atomic, so a scrape during a multi-minute upsert answers instantly instead of
/// queueing behind the write guard. That is the same discipline `/ready` follows and for
/// the same reason — the endpoints you consult during an incident must not be the ones the
/// incident blocks.
pub(super) async fn metrics_endpoint(State(st): State<AppState>) -> Response {
    let mut out = String::with_capacity(8 * 1024);
    render(&mut out, &st);
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        out,
    )
        .into_response()
}

fn render(out: &mut String, st: &AppState) {
    use std::fmt::Write as _;

    // ── Library counters ────────────────────────────────────────────────────
    for (name, help, value) in crate::metrics::metrics().counters() {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} counter");
        let _ = writeln!(out, "{name} {value}");
    }

    // ── Traffic ─────────────────────────────────────────────────────────────
    let _ = writeln!(
        out,
        "# HELP nidus_http_requests_total HTTP requests served, by route and status class"
    );
    let _ = writeln!(out, "# TYPE nidus_http_requests_total counter");
    for (i, route) in ROUTES.iter().enumerate() {
        for (c, count) in HTTP.routes[i].by_class.iter().enumerate() {
            let n = count.load(Ordering::Relaxed);
            if n == 0 {
                continue; // skip empty series — a scrape should not be mostly zeroes
            }
            let _ = writeln!(
                out,
                "nidus_http_requests_total{{route=\"{route}\",status=\"{}xx\"}} {n}",
                c + 1
            );
        }
    }

    let _ = writeln!(
        out,
        "# HELP nidus_http_request_duration_seconds Request latency by route"
    );
    let _ = writeln!(out, "# TYPE nidus_http_request_duration_seconds histogram");
    for (i, route) in ROUTES.iter().enumerate() {
        let s = &HTTP.routes[i];
        let count = s.count.load(Ordering::Relaxed);
        if count == 0 {
            continue;
        }
        // Prometheus buckets are cumulative: each `le` is "at or under this bound".
        let mut cumulative = 0u64;
        for (b, bound) in BUCKETS.iter().enumerate() {
            cumulative += s.buckets[b].load(Ordering::Relaxed);
            let _ = writeln!(
                out,
                "nidus_http_request_duration_seconds_bucket{{route=\"{route}\",le=\"{bound}\"}} {cumulative}"
            );
        }
        let _ = writeln!(
            out,
            "nidus_http_request_duration_seconds_bucket{{route=\"{route}\",le=\"+Inf\"}} {count}"
        );
        let sum = s.sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let _ = writeln!(
            out,
            "nidus_http_request_duration_seconds_sum{{route=\"{route}\"}} {sum}"
        );
        let _ = writeln!(
            out,
            "nidus_http_request_duration_seconds_count{{route=\"{route}\"}} {count}"
        );
    }

    for (name, help, value) in [
        (
            "nidus_http_requests_shed_total",
            "Requests refused with 503 at the concurrency limit",
            HTTP.shed.get(),
        ),
        (
            "nidus_http_requests_timed_out_total",
            "Requests abandoned with 504 at their deadline: clients freed and the scan \
             signalled to stop",
            HTTP.timed_out.get(),
        ),
        (
            "nidus_http_requests_cancelled_total",
            "Requests whose client disconnected before a response was produced",
            HTTP.cancelled.get(),
        ),
    ] {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} counter");
        let _ = writeln!(out, "{name} {value}");
    }

    for (name, help, value) in [
        (
            "nidus_http_requests_in_flight",
            "Requests currently being handled, probes included",
            HTTP.in_flight.get(),
        ),
        (
            "nidus_http_admitted_in_flight",
            "Concurrency permits held: store-touching requests admitted and not yet \
             released, excluding work whose deadline already fired",
            st.limits.in_flight() as u64,
        ),
        (
            "nidus_http_concurrency_limit",
            "Configured --max-concurrent-requests",
            st.limits.limit() as u64,
        ),
    ] {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} gauge");
        let _ = writeln!(out, "{name} {value}");
    }

    // ── Instance state ──────────────────────────────────────────────────────
    //
    // The *same* decision `GET /ready` answers with, not a re-derivation of it — a
    // dashboard that disagreed with the load balancer about whether an instance is serving
    // would be worse than no dashboard. Lock-free like the probe, so a scrape during a long
    // write answers instantly.
    let _ = writeln!(
        out,
        "# HELP nidus_ready Whether this instance reports ready (1) or not (0)"
    );
    let _ = writeln!(out, "# TYPE nidus_ready gauge");
    let _ = writeln!(
        out,
        "nidus_ready {}",
        u8::from(super::readiness_check(st).is_ok())
    );

    if let Some(r) = st.readiness.get() {
        let _ = writeln!(
            out,
            "# HELP nidus_staleness_seconds Seconds since this instance last verified it was current"
        );
        let _ = writeln!(out, "# TYPE nidus_staleness_seconds gauge");
        let _ = writeln!(out, "nidus_staleness_seconds {}", r.staleness_secs());
        let _ = writeln!(
            out,
            "# HELP nidus_writer_fenced Whether this writer has been superseded (1) or not (0)"
        );
        let _ = writeln!(out, "# TYPE nidus_writer_fenced gauge");
        let _ = writeln!(out, "nidus_writer_fenced {}", u8::from(r.fenced()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every label this can produce must be one of `ROUTES` — that invariant is what bounds
    /// the cardinality, and `route_slot` silently falls back to `other` if it is ever
    /// broken, so it needs asserting rather than assuming.
    #[test]
    fn every_label_is_a_known_route() {
        let paths = [
            "/health",
            "/ready",
            "/metrics",
            "/cluster",
            "/stats",
            "/collections",
            "/collections/docs",
            "/collections/docs/meta",
            "/collections/docs/upsert",
            "/collections/docs/delete",
            "/collections/docs/records",
            "/collections/docs/fts-schema",
            "/collections/docs/remember",
            "/collections/docs/recall",
            "/search",
            "/text-search",
            "/hybrid-search",
            "/list",
            "/flush",
            "/compact",
            "/refresh",
            "/nope",
            "/collections/docs/nope",
            "/collections/a/b/c",
            "",
        ];
        for p in paths {
            let label = route_label(p);
            assert!(
                ROUTES.contains(&label),
                "{p} produced {label}, which is not a known route"
            );
        }
    }

    /// A collection name must never reach a label — that is both the cardinality bound and
    /// the reason `/metrics` can be left unauthenticated.
    #[test]
    fn collection_names_are_collapsed() {
        assert_eq!(
            route_label("/collections/very-secret-project/upsert"),
            "/collections/{name}/upsert"
        );
        assert_eq!(
            route_label("/collections/other-name/upsert"),
            route_label("/collections/notes/upsert")
        );
        // The bare create/drop route, whose name segment is the last one.
        assert_eq!(route_label("/collections/notes"), "/collections/{name}");
    }

    #[test]
    fn histogram_buckets_are_ascending() {
        assert!(BUCKETS.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn stats_land_in_the_right_bucket_and_class() {
        let s = RouteStats::new();
        s.record(StatusCode::OK, 700); // 0.7ms → the 0.001 bucket
        s.record(StatusCode::INTERNAL_SERVER_ERROR, 2_000_000); // 2s → the 2.5 bucket
        s.record(StatusCode::OK, 120_000_000); // 120s → over every bound

        assert_eq!(s.count.load(Ordering::Relaxed), 3);
        assert_eq!(s.by_class[1].load(Ordering::Relaxed), 2, "2xx");
        assert_eq!(s.by_class[4].load(Ordering::Relaxed), 1, "5xx");
        assert_eq!(s.over.load(Ordering::Relaxed), 1);
        let i = BUCKETS.iter().position(|&b| b == 0.001).unwrap();
        assert_eq!(s.buckets[i].load(Ordering::Relaxed), 1);
        let i = BUCKETS.iter().position(|&b| b == 2.5).unwrap();
        assert_eq!(s.buckets[i].load(Ordering::Relaxed), 1);
    }

    /// An inbound correlation id is honoured, but only when it cannot corrupt a log line or
    /// a response header. A newline in a header value is a log-injection primitive.
    #[test]
    fn inbound_request_ids_are_validated() {
        let req = |v: &str| {
            Request::builder()
                .uri("/health")
                .header("x-request-id", v)
                .body(axum::body::Body::empty())
                .unwrap()
        };
        assert_eq!(request_id(&req("trace-abc.123:9")), "trace-abc.123:9");
        // Rejected → a minted id, which never contains these.
        assert!(!request_id(&req("has space")).contains(' '));
        assert!(!request_id(&req(&"x".repeat(65))).contains('x'));
        let injected = request_id(&req("ok\tvalue"));
        assert!(!injected.contains('\t'));

        // No header at all → still an id.
        let bare = Request::builder()
            .uri("/health")
            .body(axum::body::Body::empty())
            .unwrap();
        assert!(!request_id(&bare).is_empty());
    }

    #[test]
    fn minted_ids_are_unique() {
        let a = mint_request_id();
        let b = mint_request_id();
        assert_ne!(a, b);
    }
}
