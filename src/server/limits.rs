//! Backpressure: a concurrency cap, load shedding, and per-request timeouts (nidus-abx.2).
//!
//! Before this the entire middleware stack was a body limit and auth — which bounds how
//! big **one** request can be, and nothing about how **many** or how **long**. Both gaps
//! are reachable from ordinary client behaviour rather than malice:
//!
//! * Every accepted request that needs the store queues on one `RwLock`. In-flight bodies
//!   accumulate in RAM on a store whose whole design is "the working set is in RAM", so
//!   the queue competes for memory with the data, and there was no point at which the
//!   server said *too much* — it simply degraded until the allocator or the OOM killer
//!   intervened. Note the library defends its own allocations with `try_reserve`
//!   (SPEC §6.6); unbounded request queueing above it bypassed that guard entirely.
//! * A long write takes the exclusive guard for the whole batch, so readers queued behind
//!   it indefinitely rather than failing fast. From a client's perspective the server hung.
//!
//! ## Why hand-rolled rather than tower's layers
//!
//! `tower` is already a dependency and ships `ConcurrencyLimitLayer` / `LoadShedLayer` /
//! `TimeoutLayer`, so the obvious move is to compose those. Three things decided against it:
//!
//! 1. **Probes must never be shed.** `/health`, `/ready` and `/metrics` have to answer while
//!    the instance is saturated — that is exactly when someone is looking. A tower layer
//!    wrapping the router sheds everything uniformly, so a load spike would fail liveness
//!    and get a busy-but-healthy instance *restarted*: the same availability trap
//!    nidus-abx.1 and .3 just closed, reopened one layer up. Exempting paths needs a filter
//!    on the request, which is what this middleware is.
//! 2. **`ConcurrencyLimitLayer`'s permit pool is per-clone**, and axum clones the router per
//!    connection, so the honest composition is `GlobalConcurrencyLimitLayer` — at which
//!    point the shared `Semaphore` here is the same object with less indirection.
//! 3. **The response is the product.** A shed request should carry `Retry-After` and a JSON
//!    body the SDKs already know how to read; tower's layers surface `Overloaded`/`Elapsed`
//!    as errors that `HandleErrorLayer` must downcast and re-dress. Same result, more
//!    moving parts.
//!
//! ## A deadline frees the CPU too, not just the client
//!
//! Dropping the response future returns a `504` to the caller but cannot stop the work:
//! the search is on a `spawn_blocking` task, and blocking tasks are not cancellable. So the
//! deadline arm also **signals the scan to stop**, through the cooperative token in
//! [`crate::cancel`] — the scan kernels check it every a few thousand rows and bail. Under
//! load, finishing a scan nobody is waiting for is the worst possible use of a core.
//!
//! It is cooperative, so it is prompt rather than instant: work already inside a chunk
//! finishes. That is the right trade — a per-row check would tax every query to help the
//! rare abandoned one.
//!
//! **No per-IP rate limiting.** That belongs at the proxy that already terminates TLS in
//! front of nidus (see the deployment guide); duplicating it here would be the wrong layer.

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::{
    Json,
    body::{Body, Bytes},
    extract::{Request, State},
    http::{Method, StatusCode, header::RETRY_AFTER},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;
use tokio::sync::Semaphore;

use super::AppState;
use super::auth::is_public;

/// The admission-control state, shared by every request through [`AppState`].
pub(super) struct Limits {
    /// Permits for in-flight, store-touching requests. `try_acquire` rather than `acquire`
    /// is what makes this load *shedding* rather than queueing: past the cap the answer is
    /// an immediate `503`, not a place in a line that has no bound.
    permits: Semaphore,
    /// The cap, kept alongside because `Semaphore` only reports what is *available*.
    limit: usize,
    /// Deadline for a read (`GET`, and the POST-shaped search routes).
    read_timeout: Option<Duration>,
    /// Deadline for a mutation. Separate because the honest latencies differ by orders of
    /// magnitude: a search is milliseconds, a large upsert legitimately takes minutes, and
    /// one bound tight enough for the first would abort the second mid-batch.
    write_timeout: Option<Duration>,
    /// How long a request body may go without delivering a frame before it is abandoned
    /// (nidus-6c2). See [`IdleTimeoutBody`].
    body_idle_timeout: Option<Duration>,
    /// Permits for the **body-reception** phase, which happens before a store permit is
    /// taken. See [`backpressure`] for why the two phases are separate.
    ///
    /// Sized [`BODY_SLOT_FACTOR`]× the store limit rather than given its own flag: it
    /// bounds memory, not CPU, and the two are not independent in any way an operator
    /// would want to tune separately. A derived number that is right is better than a
    /// knob nobody knows how to set.
    body_slots: Semaphore,
    /// Largest body this server will receive, enforced here because the body is consumed
    /// before the extractors (and therefore before `DefaultBodyLimit`) ever see it.
    max_body_bytes: usize,
}

/// How many bodies may be arriving at once, as a multiple of the store-work limit.
///
/// Bigger than 1× on purpose: body reception is IO-bound and usually over in microseconds,
/// so sizing it to the CPU-bound store limit would shed real traffic to protect against a
/// phase that is not the bottleneck. Bounded at all because an unbounded count of
/// concurrent bodies is the exact memory problem nidus-abx.2 exists to prevent.
const BODY_SLOT_FACTOR: usize = 4;

impl Limits {
    pub(super) fn new(
        limit: usize,
        read_timeout: Option<Duration>,
        write_timeout: Option<Duration>,
        body_idle_timeout: Option<Duration>,
        max_body_bytes: usize,
    ) -> Limits {
        Limits {
            permits: Semaphore::new(limit),
            limit,
            read_timeout,
            write_timeout,
            body_idle_timeout,
            body_slots: Semaphore::new(limit.saturating_mul(BODY_SLOT_FACTOR)),
            max_body_bytes,
        }
    }

    /// In-flight store-touching requests right now.
    pub(super) fn in_flight(&self) -> usize {
        self.limit.saturating_sub(self.permits.available_permits())
    }

    pub(super) fn limit(&self) -> usize {
        self.limit
    }
}

/// Resolve `--max-concurrent-requests`, where `0` means "auto".
///
/// Auto is `8 ×` available parallelism, floored at 64. The multiplier is not arbitrary:
/// search is CPU-bound brute force, so admitting far more concurrent scans than cores buys
/// no throughput and costs memory — but a cap at core count would shed on a modest burst of
/// cheap requests (`/stats`, a small `get`) that never touch a core for long, so it is a
/// small multiple rather than 1×. The floor keeps a one- or two-core container from shedding
/// under trivial load, where the limit would be protecting nothing.
pub(super) fn resolve_concurrency(configured: usize) -> usize {
    if configured > 0 {
        return configured;
    }
    let cores = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    (cores * 8).max(64)
}

/// Admission control + deadline, as one middleware.
///
/// Probes are exempt from both: they take no store lock (nidus-abx.1/.3 made sure of that),
/// so they cost nothing to admit, and they are the one thing that must still answer when
/// everything else is shedding.
pub(super) async fn backpressure(
    State(st): State<AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    if is_public(req.uri().path()) {
        return next.run(req).await;
    }
    let limits = &st.limits;

    // ── Phase 1: receive the body, WITHOUT a store permit (nidus-6c2) ───────────────
    //
    // The store permit used to be taken first, and the handler is what awaits the body —
    // so a client that sent headers and then went silent pinned a permit for the whole
    // request deadline, denying service to work that was ready to run. Receiving the body
    // first means a stalled client can no longer touch the store's admission pool at all.
    //
    // It is still bounded, by its own larger pool: an unbounded number of bodies arriving
    // at once is the memory problem this epic exists to prevent, just relocated.
    match receive_body(limits, req).await {
        Ok(received) => req = received,
        Err(response) => return response,
    }

    // ── Phase 2: do the work, holding a store permit ────────────────────────────────
    let Ok(_permit) = limits.permits.try_acquire() else {
        super::metrics::http().shed.inc();
        crate::diag::diag!(
            crate::diag::Level::Warn,
            "http",
            "shedding request: concurrency limit reached",
            "limit" => limits.limit,
            "path" => req.uri().path(),
        );
        return overloaded(limits.limit);
    };

    let timeout = if is_mutation(req.method(), req.uri().path()) {
        limits.write_timeout
    } else {
        limits.read_timeout
    };
    let Some(timeout) = timeout else {
        return next.run(req).await;
    };

    let path = req.uri().path().to_string();
    // The token this request's store work will run under. Dropping the response future
    // frees the client but cannot stop a `spawn_blocking` scan, so the deadline arm below
    // signals the scan itself to stop (`crate::cancel`).
    let cancel = crate::Cancel::new();
    let outcome = tokio::time::timeout(timeout, CANCEL.scope(cancel.clone(), next.run(req))).await;
    match outcome {
        Ok(resp) => resp,
        Err(_) => {
            cancel.cancel();
            super::metrics::http().timed_out.inc();
            crate::diag::diag!(
                crate::diag::Level::Warn,
                "http",
                "request exceeded its deadline; the client is freed and the scan is asked \
                 to stop",
                "timeout_secs" => timeout.as_secs(),
                "path" => path,
            );
            timed_out(timeout)
        }
    }
}

tokio::task_local! {
    /// The cancellation token for the request being handled on this task.
    ///
    /// A task-local rather than a request extension or a handler argument: `run_read` /
    /// `run_write` are where the token has to be picked up, and they are called from every
    /// handler but receive neither the `Request` nor an extractor. A task-local reaches
    /// them without threading a parameter through a dozen signatures that have no other
    /// reason to know about cancellation.
    pub(super) static CANCEL: crate::Cancel;
}

/// The current request's cancellation token, or `None` outside a request (the in-process
/// router tests, and the background lease/refresh tasks).
pub(super) fn current_cancel() -> Option<crate::Cancel> {
    CANCEL.try_with(Clone::clone).ok()
}

/// `503` with `Retry-After: 1`. A shed request is **retryable**: nothing was attempted, the
/// store is untouched, and the same request a moment later will very likely succeed. That
/// is exactly the contract `503` carries, and it composes with the readiness signal an
/// orchestrator already reads.
fn overloaded(limit: usize) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(RETRY_AFTER, "1")],
        Json(json!({
            "error": format!(
                "server overloaded: {limit} requests already in flight (the configured \
                 --max-concurrent-requests) — retry shortly"
            ),
            "retryable": true,
        })),
    )
        .into_response()
}

/// `504`, not `503`: the request *was* admitted and *is* being worked on, so an immediate
/// retry would pile a second copy of the same work onto an instance that is already behind.
/// The distinction matters to a client deciding whether to retry — which is why the body
/// says so rather than leaving it to be inferred from the status.
fn timed_out(after: Duration) -> Response {
    (
        StatusCode::GATEWAY_TIMEOUT,
        Json(json!({
            "error": format!(
                "request exceeded the {}s server deadline; the work may still be running, \
                 so retry only after backing off",
                after.as_secs()
            ),
            "retryable": false,
        })),
    )
        .into_response()
}

/// Read the request body to completion under the idle timeout and the size cap, holding a
/// body slot rather than a store permit.
///
/// Buffering here rather than leaving it to the extractors is what moves the wait off the
/// store's admission pool. It costs nothing extra in memory: every body was already
/// buffered whole by the `Json` extractor a moment later — this only moves *where*.
///
/// Consuming the body means `DefaultBodyLimit` never sees it, so the size cap is enforced
/// here too. `Content-Length` is checked first when the client sent one (every real client
/// does), which rejects an oversized upload before a single byte of it is read.
async fn receive_body(limits: &Limits, req: Request) -> Result<Request, Response> {
    let declared = req
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|n| n > limits.max_body_bytes) {
        return Err(too_large(limits.max_body_bytes));
    }
    // Nothing to receive: skip the slot entirely so a bodyless GET never queues behind
    // uploads.
    if declared == Some(0) {
        return Ok(req);
    }

    let Ok(_slot) = limits.body_slots.try_acquire() else {
        super::metrics::http().shed.inc();
        crate::diag::diag!(
            crate::diag::Level::Warn,
            "http",
            "shedding request: too many bodies already arriving",
            "limit" => limits.body_slots.available_permits(),
            "path" => req.uri().path(),
        );
        return Err(overloaded(limits.limit));
    };

    let (parts, body) = req.into_parts();
    let body = match limits.body_idle_timeout {
        Some(idle) => Body::new(IdleTimeoutBody::new(body, idle)),
        None => body,
    };
    match axum::body::to_bytes(body, limits.max_body_bytes).await {
        Ok(bytes) => Ok(Request::from_parts(parts, Body::from(bytes))),
        // `to_bytes` reports "too large" and "the connection died" as the same error type.
        // Reporting 413 is right for the case a client can act on, and for a dead
        // connection the response goes nowhere anyway.
        Err(e) => {
            crate::diag::diag!(
                crate::diag::Level::Debug,
                "http",
                "request body was not received",
                "err" => e,
            );
            Err(too_large(limits.max_body_bytes))
        }
    }
}

/// `413`, matching what `DefaultBodyLimit` would have produced — kept in the same JSON
/// shape as every other error so a client parses one thing.
fn too_large(max: usize) -> Response {
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        Json(json!({
            "error": format!("request body exceeds the {max}-byte limit (--max-body-bytes)"),
            "retryable": false,
        })),
    )
        .into_response()
}

/// A request body that gives up if it goes quiet (nidus-6c2).
///
/// ## The hole this closes
///
/// A permit is acquired before the handler runs, and the handler is what awaits the body.
/// So a client that sends complete headers with a `Content-Length` and then sends nothing
/// pins a permit — reproduced with a raw socket: against `--max-concurrent-requests 1` and
/// `--read-timeout 0`, one silent connection sheds every other request indefinitely. It is
/// a slow-loris, aimed at admission control rather than at memory, and it is partly
/// self-inflicted: before the concurrency cap existed there were no permits to exhaust.
///
/// ## Why *idle*, not total
///
/// The obvious fix — one deadline on the whole body — is wrong here. A legitimate upsert
/// can be a 256 MiB payload over a slow link, so any total bound tight enough to be useful
/// against a silent client would abort real uploads. What actually distinguishes an attack
/// from a slow upload is **progress**: this resets its clock on every frame, so a body that
/// keeps arriving is never cut off however long it takes, and one that stalls dies quickly.
/// (Same semantic as nginx's `client_body_timeout`, and for the same reason.)
///
/// ## What it does not do
///
/// The permit is still held while the body arrives, which is deliberate — that is what
/// bounds concurrent body buffering against a store whose working set is in RAM. So this
/// shortens the window an attacker can pin a permit for; it does not remove it. Genuine
/// slow-client defence belongs at the proxy, alongside the rate limiting and TLS the
/// deployment guide already puts there.
///
/// A stalled body surfaces as a body-read error, which axum's extractors report as `400`
/// rather than `408`. The status is cosmetic; releasing the permit is the point.
struct IdleTimeoutBody {
    inner: Body,
    idle: Duration,
    /// Armed on the first `Pending`, dropped whenever a frame lands — so the timer measures
    /// the gap since the last byte, not the age of the request.
    sleep: Option<Pin<Box<tokio::time::Sleep>>>,
}

impl IdleTimeoutBody {
    fn new(inner: Body, idle: Duration) -> IdleTimeoutBody {
        IdleTimeoutBody {
            inner,
            idle,
            sleep: None,
        }
    }
}

impl http_body::Body for IdleTimeoutBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Bytes>, axum::Error>>> {
        // Every field is `Unpin` (`axum::body::Body` is, and `Sleep` is behind a `Box`), so
        // the pin can be projected away rather than carried through by hand.
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(frame) => {
                // Progress: disarm, so the next gap is timed from here.
                this.sleep = None;
                Poll::Ready(frame)
            }
            Poll::Pending => {
                let idle = this.idle;
                let sleep = this
                    .sleep
                    .get_or_insert_with(|| Box::pin(tokio::time::sleep(idle)));
                match sleep.as_mut().poll(cx) {
                    Poll::Ready(()) => Poll::Ready(Some(Err(axum::Error::new(format!(
                        "request body stalled: no data for {}s",
                        idle.as_secs()
                    ))))),
                    Poll::Pending => Poll::Pending,
                }
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// Whether a request mutates the store, and so gets the longer deadline.
///
/// Method alone is not enough: the search routes are `POST` because a query vector does not
/// belong in a URL, so they would be misclassified as writes and inherit a deadline meant
/// for a multi-minute upsert. Naming the read-shaped POSTs explicitly and treating every
/// other mutating method as a write is the safe direction to be wrong in — a *new* mutating
/// route added later defaults to the generous bound rather than being cut off mid-batch.
fn is_mutation(method: &Method, path: &str) -> bool {
    if method == Method::GET || method == Method::HEAD {
        return false;
    }
    // Read-shaped POSTs, and the per-collection `recall` (a search over text).
    !(matches!(
        path,
        "/search" | "/text-search" | "/hybrid-search" | "/list"
    ) || path.ends_with("/recall"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_shaped_posts_are_reads() {
        for p in ["/search", "/text-search", "/hybrid-search", "/list"] {
            assert!(!is_mutation(&Method::POST, p), "{p} is a read");
        }
        assert!(!is_mutation(&Method::POST, "/collections/notes/recall"));
        assert!(!is_mutation(&Method::GET, "/stats"));
    }

    #[test]
    fn mutating_routes_get_the_write_deadline() {
        assert!(is_mutation(&Method::POST, "/collections/docs/upsert"));
        assert!(is_mutation(&Method::POST, "/collections/docs/delete"));
        assert!(is_mutation(&Method::POST, "/compact"));
        assert!(is_mutation(&Method::POST, "/flush"));
        assert!(is_mutation(&Method::POST, "/refresh"));
        assert!(is_mutation(&Method::DELETE, "/collections/docs"));
        assert!(is_mutation(&Method::PUT, "/collections/docs/meta"));
        assert!(is_mutation(&Method::POST, "/collections/notes/remember"));
        // An unrecognised mutating route defaults to "write" — the safe direction.
        assert!(is_mutation(&Method::POST, "/something-new"));
    }

    #[test]
    fn auto_concurrency_has_a_floor() {
        assert!(resolve_concurrency(0) >= 64);
        // An explicit value is honoured exactly, however small — an operator who asks for
        // 1 is testing shedding and must get it.
        assert_eq!(resolve_concurrency(1), 1);
        assert_eq!(resolve_concurrency(500), 500);
    }

    #[test]
    fn in_flight_tracks_outstanding_permits() {
        let l = Limits::new(2, None, None, None, 1024);
        assert_eq!(l.in_flight(), 0);
        let a = l.permits.try_acquire().unwrap();
        assert_eq!(l.in_flight(), 1);
        let _b = l.permits.try_acquire().unwrap();
        assert_eq!(l.in_flight(), 2);
        assert!(l.permits.try_acquire().is_err(), "cap reached, shed");
        drop(a);
        assert_eq!(l.in_flight(), 1);
        assert!(l.permits.try_acquire().is_ok(), "and it recovers");
    }
}
