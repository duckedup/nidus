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
//! ## What this does NOT do
//!
//! **A timeout frees the client, not the CPU.** When the deadline fires, the response
//! future is dropped and the caller gets a `504` — but a search already running on a
//! blocking task runs to completion regardless, because `spawn_blocking` work is not
//! cancellable. Abandoned work still costs a full scan. Genuinely cancelling a running scan
//! needs a cooperative check in the scan loop and is a separate, larger decision; it is
//! recorded here so nobody reads "we have timeouts" as "we stop the work".
//!
//! One consequence worth stating outright: the permit is released when the deadline fires,
//! so during the tail of an abandoned request the admission count *understates* the work
//! actually in flight. Holding the permit until the orphaned task finished would be no
//! better — the server would then shed live traffic on behalf of work nobody is waiting
//! for. Both are wrong; this is the less harmful wrong, and it is the right shape only once
//! a scan can actually be cancelled.
//!
//! **No per-IP rate limiting.** That belongs at the proxy that already terminates TLS in
//! front of nidus (see the deployment guide); duplicating it here would be the wrong layer.

use std::time::Duration;

use axum::{
    Json,
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
}

impl Limits {
    pub(super) fn new(
        limit: usize,
        read_timeout: Option<Duration>,
        write_timeout: Option<Duration>,
    ) -> Limits {
        Limits {
            permits: Semaphore::new(limit),
            limit,
            read_timeout,
            write_timeout,
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
pub(super) async fn backpressure(State(st): State<AppState>, req: Request, next: Next) -> Response {
    if is_public(req.uri().path()) {
        return next.run(req).await;
    }
    let limits = &st.limits;

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
    match tokio::time::timeout(timeout, next.run(req)).await {
        Ok(resp) => resp,
        Err(_) => {
            super::metrics::http().timed_out.inc();
            crate::diag::diag!(
                crate::diag::Level::Warn,
                "http",
                "request exceeded its deadline; the client is freed but the work continues",
                "timeout_secs" => timeout.as_secs(),
                "path" => path,
            );
            timed_out(timeout)
        }
    }
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
        let l = Limits::new(2, None, None);
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
