//! `nidus serve` — a thin HTTP wrapper over one open [`Nidus`] (SPEC.md §9).
//!
//! The core stays an in-process, synchronous library; this module is the optional
//! server seam the SPEC anticipates — a separate wrapper, not a change to the core.
//! The store is held behind `Arc<RwLock<Nidus>>` and every operation runs on a
//! blocking task (`spawn_blocking`), the exact pattern the README/CLAUDE.md
//! prescribe for driving the synchronous store from async code: take the lock
//! (shared for reads, exclusive for writes), run the CPU/IO-bound op off the async
//! executor, drop the lock — never held across an `.await`. Endpoints map 1:1 to
//! the public API.

mod auth;
mod commit;
pub mod dto;
mod limits;
mod metrics;

use std::sync::{Arc, RwLock};

use anyhow::Context;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value as JsonValue, json};
use tokio::net::TcpListener;

use crate::{FtsQuery, HybridOpts, Language, Nidus, Record, Scope, SearchOpts};
use dto::{
    AnnDto, DeleteRequest, FootprintDto, FtsSchemaRequest, HitDto, HybridSearchRequest,
    ListRequest, SearchRequest, TextSearchRequest, UpsertRequest,
};

// ── AI-ingest (memory) imports: only under the `memory` feature (pulled by the
// `serve` umbrella). Plain `cli` builds a lean server without these. ──
#[cfg(feature = "memory")]
use crate::embed::{AnyEmbedder, Embedder};
#[cfg(all(feature = "memory", feature = "summarize"))]
use crate::summarize::{AnySummarizer, SummarizeOpts, Summarizer};
#[cfg(feature = "memory")]
use dto::{RecallRequest, RememberRequest};

/// How `nidus serve` is configured beyond the store itself.
pub struct ServeConfig {
    /// Bind address.
    pub addr: String,
    /// When `Some`, every request except `/health` must carry
    /// `Authorization: Bearer <token>`. `None` leaves the server unauthenticated
    /// (the frictionless localhost default).
    pub token: Option<String>,
    /// Maximum request body size in bytes. The store buffers each body in memory,
    /// so this is also the largest single upsert payload.
    pub max_body_bytes: usize,
    /// Cap on store-touching requests in flight; beyond it, requests are **shed** with
    /// `503` rather than queued (nidus-abx.2). `0` resolves to `8 ×` CPU cores, floored at
    /// 64 — see [`limits::resolve_concurrency`]. Probe endpoints are never shed.
    pub max_concurrent_requests: usize,
    /// Wall-clock deadline for a read request. `None` disables it.
    pub read_timeout: Option<std::time::Duration>,
    /// Wall-clock deadline for a mutating request — separate from `read_timeout` because a
    /// large upsert legitimately runs for minutes while a search is milliseconds. `None`
    /// disables it.
    pub write_timeout: Option<std::time::Duration>,
    /// How long a request body may go without delivering a frame before it is abandoned
    /// (nidus-6c2). An **idle** bound, not a total one: a body that keeps arriving is never
    /// cut off however large it is. `None` disables it, which also removes the only thing
    /// stopping a silent client from pinning a concurrency permit.
    pub body_idle_timeout: Option<std::time::Duration>,
    /// Fail readiness once a read-only instance is staler than this (mirrors
    /// [`Config::max_staleness`](crate::Config::max_staleness)). `None` = no bound.
    pub max_staleness: Option<std::time::Duration>,
    /// Refresh a read-only instance on this interval so it stays current without a sidecar
    /// or cron calling `POST /refresh`. `None` (the default) leaves refreshing entirely to
    /// the caller.
    ///
    /// A server-side tokio task, deliberately NOT a library background thread — "no
    /// background threads" is a property of the sync core, and the server is already async.
    /// Also deliberately not a refresh-per-read: that would put a manifest fetch on the hot
    /// path of exactly the read-heavy fan-out cluster mode exists for.
    pub refresh_interval: Option<std::time::Duration>,
    /// How often to renew the cluster writer lease out of band. Should be well under
    /// `Config::lock_ttl` — a third of it is a reasonable default — so a long write cannot
    /// let the lease lapse. Ignored unless this instance is a cluster writer.
    pub lease_renew_interval: std::time::Duration,
    /// Embedder that backs the text-native `/remember` and `/recall` routes. When
    /// `None`, those routes answer `400` (the server was started without an
    /// embedder). Built by the CLI from `--embed-provider …`.
    #[cfg(feature = "memory")]
    pub embedder: Option<Arc<AnyEmbedder>>,
    /// Optional summarizer enabling `mode: "summarize"` on `/remember`. When
    /// `None`, a summarize request answers `400`.
    #[cfg(all(feature = "memory", feature = "summarize"))]
    pub summarizer: Option<Arc<AnySummarizer>>,
}

/// Shared, cloneable handle to the one open store.
///
/// The store sits behind an `RwLock`, not a `Mutex`: read endpoints (search,
/// list, get) take `&Nidus` and run **concurrently**, while writes take the
/// exclusive guard. Brute-force search is CPU-bound, so letting parallel queries
/// use multiple cores is the whole point at this scale.
#[derive(Clone)]
struct AppState {
    /// `None` until the store finishes opening — a standby writer waiting for promotion
    /// sits here indefinitely by design. Data routes answer `503` while it is empty; see
    /// [`serve`] for why the listener comes up first.
    db: Arc<RwLock<Option<Nidus>>>,
    /// Mirrors `db.is_some()` for [`ready`] to read.
    ///
    /// Not just `db.read().is_some()`: that takes a **blocking** lock, and `ready` runs on
    /// the async executor rather than a blocking task. A long write (a large upsert holds
    /// the write guard for seconds) would then stall every readiness probe behind it — and
    /// with enough concurrent probes, stall executor threads themselves. A probe must
    /// answer in constant time no matter what the store is doing, so it reads an atomic.
    open: Arc<std::sync::atomic::AtomicBool>,
    /// The lock-free readiness handle, published once the store opens (see
    /// [`crate::Readiness`]). A `OnceLock` because a store opens exactly once per process,
    /// so this needs publication but never mutation — and reading it costs an atomic load,
    /// which is what keeps [`ready`] off the store lock entirely (nidus-abx.3).
    readiness: Arc<std::sync::OnceLock<crate::Readiness>>,
    /// Readiness fails past this much reader staleness (`Config::max_staleness`), copied
    /// here so a probe never has to reach into the store's config behind the lock.
    max_staleness: Option<std::time::Duration>,
    token: Option<auth::Token>,
    /// Admission control: the concurrency permits and the per-request deadlines
    /// (nidus-abx.2). `Arc` because `AppState` is cloned per request and the permit pool
    /// must be the *same* pool for all of them — a per-clone semaphore would cap nothing.
    limits: Arc<limits::Limits>,
    /// Group commit for the write path (nidus-xb9.1): concurrent writes are applied together
    /// under one store guard and share one disk barrier instead of taking one each. `Arc` for
    /// the same reason as `limits` — a per-clone queue would coalesce nothing.
    commit: Arc<commit::Committer>,
    /// Shared embedder for the `memory` routes; `None` disables them (→ `400`).
    #[cfg(feature = "memory")]
    embedder: Option<Arc<AnyEmbedder>>,
    /// Shared summarizer for `mode: "summarize"`; `None` disables it (→ `400`).
    #[cfg(all(feature = "memory", feature = "summarize"))]
    summarizer: Option<Arc<AnySummarizer>>,
}

/// Bind the address, open the store, and serve until a shutdown signal (Ctrl-C /
/// SIGTERM); flush and release the writer handle on shutdown.
///
/// `open` is a closure rather than an already-open [`Nidus`] because **binding happens
/// first**. Opening can block for a long time by design: a standby writer
/// ([`LeaseWait::Forever`](crate::LeaseWait)) waits for the incumbent to die before it
/// gets a handle. If nothing were listening during that wait, a supervisor's liveness
/// probe would fail and kill the very standby that is meant to be waiting — turning the
/// feature into the crash-loop it exists to remove.
///
/// So the listener comes up immediately and the store is opened on a blocking task.
/// Until it succeeds, `/health` answers (the process is alive) while `/ready` and every
/// data route answer `503` (there is no store yet). An open *failure* — as opposed to
/// waiting — shuts the server down and is returned from here.
pub async fn serve<F>(open: F, cfg: ServeConfig) -> anyhow::Result<()>
where
    F: FnOnce() -> anyhow::Result<Nidus> + Send + 'static,
{
    let concurrency = limits::resolve_concurrency(cfg.max_concurrent_requests);
    let state = AppState {
        db: Arc::new(RwLock::new(None)),
        open: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        readiness: Arc::new(std::sync::OnceLock::new()),
        max_staleness: cfg.max_staleness,
        token: cfg.token.map(auth::Token::new),
        limits: Arc::new(limits::Limits::new(
            concurrency,
            cfg.read_timeout,
            cfg.write_timeout,
            cfg.body_idle_timeout,
            cfg.max_body_bytes,
        )),
        commit: commit::Committer::new(),
        #[cfg(feature = "memory")]
        embedder: cfg.embedder,
        #[cfg(all(feature = "memory", feature = "summarize"))]
        summarizer: cfg.summarizer,
    };
    let renew_every = cfg.lease_renew_interval;
    let app = router(state.clone(), cfg.max_body_bytes);

    let listener = TcpListener::bind(&cfg.addr)
        .await
        .with_context(|| format!("binding {}", cfg.addr))?;
    let auth_note = if state.token.is_some() {
        " (bearer-token auth required)"
    } else {
        ""
    };
    // Report the address actually bound, not the one requested: with a `:0` port the
    // kernel picks it, so `cfg.addr` would print a useless `:0` and leave the caller
    // (a test harness, or anyone avoiding a port collision) no way to learn the port.
    let bound = listener
        .local_addr()
        .map_or_else(|_| cfg.addr.clone(), |a| a.to_string());
    // Kept as a plain line rather than a `diag!` event: this is the startup banner a human
    // reads (and the e2e harness parses to learn the bound port), not a log record.
    eprintln!("nidus serving on http://{bound} (Ctrl-C / SIGTERM to stop){auth_note}");
    warn_on_exposure(listener.local_addr().ok(), state.token.is_some());

    // Open on a blocking task; a failure (not a wait) asks the server to stop, and is
    // re-raised after `axum::serve` returns so the process exits non-zero.
    let open_failed = Arc::new(RwLock::new(None::<anyhow::Error>));
    let abort = Arc::new(tokio::sync::Notify::new());
    let slot = state.db.clone();
    let open_flag = state.open.clone();
    let readiness_slot = state.readiness.clone();
    let failure_slot = open_failed.clone();
    let abort_tx = abort.clone();
    tokio::task::spawn_blocking(move || match open() {
        Ok(db) => {
            if let Ok(mut slot) = slot.write() {
                // Take the lock-free readiness handle before publishing, so that whenever
                // `open` reads true the handle is guaranteed to be there (nidus-abx.3).
                let _ = readiness_slot.set(db.readiness());
                *slot = Some(db);
                // Publish only after the store is in place, so a probe never sees
                // `ready` before a request could actually be served.
                open_flag.store(true, std::sync::atomic::Ordering::Release);
                crate::diag::diag!(
                    crate::diag::Level::Info,
                    "server",
                    "store open — serving requests"
                );
            }
        }
        Err(e) => {
            if let Ok(mut failure) = failure_slot.write() {
                *failure = Some(e);
            }
            abort_tx.notify_one();
        }
    });

    // Keep the writer lease warm on a timer.
    //
    // The lease is otherwise renewed only at the START of each write batch, which is fine
    // while batches are short. A batch longer than `lock_ttl` — a very large upsert, or a
    // slow object-store PUT — would let a standby (nidus-lp4.3) conclude the writer died and
    // take over, fencing a writer that was perfectly healthy and discarding its work. That
    // became a live risk the moment standbys shipped.
    //
    // Renewing needs its own lease handle, NOT the store lock: a long write holds the write
    // guard for its whole duration, so a renewer that waited on the lock would be blocked
    // exactly when it matters. `Nidus::lease_handle` exists for this.
    {
        let db = state.db.clone();
        let ttl = renew_every;
        tokio::spawn(async move {
            // Wait for the store to open (a standby may take a long time), then grab the
            // lease handle once. `None` means this instance is not a cluster writer.
            let lease = loop {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                let handle = db
                    .try_read()
                    .ok()
                    .and_then(|g| g.as_ref().and_then(|db| db.lease_renewer()));
                if let Some(handle) = handle {
                    break handle;
                }
                // Opened but holds no lease → not a cluster writer, nothing to renew.
                if db
                    .try_read()
                    .ok()
                    .is_some_and(|g| g.as_ref().is_some_and(|db| db.lease_renewer().is_none()))
                {
                    return;
                }
            };
            let mut ticker = tokio::time::interval(ttl);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                let lease = lease.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    if let Err(e) = lease.renew() {
                        // A definitive loss latches the store's `fenced` flag through the
                        // renewer's shared handle, so `/ready` starts failing and `/cluster`
                        // reports it immediately — without waiting for a write to discover it
                        // (nidus-lp4.7). A transient backend error latches nothing: this tick
                        // simply failed, and the next one will try again.
                        if crate::backend::is_lease_lost(&e) {
                            crate::diag::diag!(
                                crate::diag::Level::Error,
                                "lease",
                                "writer lease LOST on background renewal — this instance is \
                                 fenced and now reports NOT ready",
                                "err" => format!("{e:#}"),
                            );
                        } else {
                            crate::diag::diag!(
                                crate::diag::Level::Warn,
                                "lease",
                                "background lease renewal failed transiently, will retry",
                                "err" => format!("{e:#}"),
                            );
                        }
                    }
                })
                .await;
            }
        });
    }

    // Optional self-refresh, so a read-only instance stays current without a sidecar
    // calling POST /refresh. A tokio task rather than a library background thread: "no
    // background threads" is a property of the sync core, and the server is already async.
    // `refresh()` is a no-op on a writer, so this is harmless whatever the role.
    if let Some(interval) = cfg.refresh_interval {
        let db = state.db.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            // Skip missed ticks rather than firing a burst to catch up — a backlog of
            // manifest fetches is the opposite of what an interval refresher is for.
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                let db = db.clone();
                // On a blocking task: refresh does object-store IO.
                let _ = tokio::task::spawn_blocking(move || {
                    if let Ok(mut guard) = db.write()
                        && let Some(db) = guard.as_mut()
                        && let Err(e) = db.refresh()
                    {
                        // Not fatal: the staleness clock keeps running, so readiness
                        // (with --max-staleness) is what escalates a persistent failure.
                        crate::metrics::metrics().refresh_failures.inc();
                        crate::diag::diag!(
                            crate::diag::Level::Warn,
                            "refresh",
                            "scheduled refresh failed",
                            "err" => format!("{e:#}"),
                        );
                    }
                })
                .await;
            }
        });
    }

    let shutdown = async move {
        tokio::select! {
            _ = shutdown_signal() => {}
            _ = abort.notified() => {}
        }
    };
    let served = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .context("server error");

    // Best-effort durability flush on a clean shutdown (no-op if never opened).
    if let Ok(mut db) = state.db.write()
        && let Some(db) = db.as_mut()
    {
        let _ = db.flush();
    }

    // A failed open outranks the serve result: it is the actual cause.
    if let Some(e) = open_failed.write().ok().and_then(|mut f| f.take()) {
        return Err(e);
    }
    served
}

/// Warn at startup when the bind address makes the security posture worse than the
/// configuration suggests (nidus-abx.6).
///
/// nidus serves **plain HTTP by design** — TLS is expected to be terminated in front of it
/// by an ingress, sidecar, or mesh, which does that job better than a TLS stack compiled
/// into the store would. The defect this closes is not the absence of TLS but the silence
/// about it: a reader who follows the deployment guide, sets `--token`, and binds
/// `0.0.0.0` has no indication that the credential they just configured crosses the network
/// in cleartext on every request, alongside every vector and metadata value.
///
/// **Warn, never refuse.** Refusing a non-loopback bind would break every legitimate
/// deployment that terminates TLS at a proxy — which is precisely the architecture being
/// recommended — so this has to stay advisory. It is emitted at `warn`, which is on by
/// default, and it names the concrete consequence rather than gesturing at "security".
fn warn_on_exposure(bound: Option<std::net::SocketAddr>, has_token: bool) {
    let Some(addr) = bound else { return };
    match exposure(addr, has_token) {
        Exposure::Contained => {}
        Exposure::CleartextCredential => crate::diag::diag!(
            crate::diag::Level::Warn,
            "server",
            "bound off-loopback over plain HTTP: --token authenticates callers but confers \
             NO confidentiality, so the token and every vector, id, and metadata value \
             cross the network in cleartext — terminate TLS in front of nidus (ingress, \
             sidecar, or mesh)",
            "addr" => addr,
        ),
        Exposure::Unauthenticated => crate::diag::diag!(
            crate::diag::Level::Warn,
            "server",
            "bound off-loopback with NO --token: this store is readable and writable by \
             anyone who can reach the address — pass --token and terminate TLS in front \
             of nidus",
            "addr" => addr,
        ),
    }
}

/// What a bind address plus the auth setting says about exposure.
///
/// Split from the warning itself so the *decision* can be tested exactly, without a test
/// having to scrape stderr or bind a public socket on a developer's machine.
#[derive(Debug, PartialEq, Eq)]
enum Exposure {
    /// Loopback: nothing leaves the box, so there is nothing to warn about.
    Contained,
    /// Off-box with a token — authenticated, but the credential and the data are in
    /// cleartext on the wire.
    CleartextCredential,
    /// Off-box with no token — an open, writable vector store.
    Unauthenticated,
}

fn exposure(addr: std::net::SocketAddr, has_token: bool) -> Exposure {
    if addr.ip().is_loopback() {
        Exposure::Contained
    } else if has_token {
        Exposure::CleartextCredential
    } else {
        Exposure::Unauthenticated
    }
}

fn router(state: AppState, max_body_bytes: usize) -> Router {
    let router = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics::metrics_endpoint))
        .route("/cluster", get(cluster))
        .route("/stats", get(stats))
        .route("/collections", get(list_collections))
        .route(
            "/collections/{name}",
            post(create_collection).delete(drop_collection),
        )
        .route("/collections/{name}/meta", get(get_meta).put(set_meta))
        .route("/collections/{name}/upsert", post(upsert))
        .route("/collections/{name}/delete", post(delete_records))
        .route("/collections/{name}/records", get(records))
        .route("/collections/{name}/fts-schema", post(set_fts_schema))
        .route("/search", post(search))
        .route("/text-search", post(text_search))
        .route("/hybrid-search", post(hybrid_search))
        .route("/list", post(list))
        .route("/flush", post(flush))
        .route("/compact", post(compact))
        .route("/refresh", post(refresh));

    // Text-native memory routes: the SDKs send TEXT and the server embeds /
    // summarizes. Present only when the `memory` feature is compiled in (the
    // `serve` umbrella); a plain `cli` build ships the raw endpoints above only.
    #[cfg(feature = "memory")]
    let router = router
        .route("/collections/{name}/remember", post(remember))
        .route("/collections/{name}/recall", post(recall));

    // Layer order matters, and `.layer()` applies **outermost last**. Reading inside-out:
    //
    //   body limit  ← per-extractor, closest to the handler
    //   backpressure ← admit or shed; holds a permit for the handler's whole lifetime
    //   auth         ← outside backpressure so an unauthenticated request is rejected
    //                  without ever consuming a permit
    //   observe      ← outermost, so a 401 and a shed 503 are both counted and logged;
    //                  an error rate that excludes the errors clients actually see is
    //                  worse than no error rate at all
    router
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            limits::backpressure,
        ))
        .layer(middleware::from_fn_with_state(state.clone(), auth::auth))
        .layer(middleware::from_fn(metrics::observe))
        .with_state(state)
}

/// Resolve on the first shutdown signal: Ctrl-C (SIGINT) everywhere, plus SIGTERM on
/// Unix — the signal Docker/Kubernetes send to stop a container. Catching SIGTERM is
/// what lets the graceful path below run (flush + writer-lock release on `Nidus` drop)
/// before exit; without it the process is eventually SIGKILLed, the writer lock is
/// never released, and a restarted pod must wait out the full lock TTL before it can
/// re-acquire it. With it, a rolling restart hands the lock over immediately.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                term.recv().await;
            }
            // If the handler can't be installed, just never fire this arm.
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

// ── Handlers ──────────────────────────────────────────────────────────────

/// Liveness: the process is up and the HTTP stack is answering.
///
/// Deliberately says nothing about the store. A standby writer waiting for promotion is
/// *alive* — killing it is precisely the wrong response, since waiting is its job — so this
/// must keep answering while [`AppState::db`] is still empty. Whether the instance can
/// actually serve traffic is [`ready`]'s question.
async fn health(State(st): State<AppState>) -> Response {
    // A **poisoned** store lock is the one condition under which this process is broken
    // beyond recovery, and it must be escalated rather than papered over (nidus-abx.1).
    //
    // std only poisons an `RwLock` when a panic unwinds while it is held for *writing* —
    // verified, not assumed — so a poisoned store is by construction one whose in-RAM index
    // was mid-mutation when the panic hit and may no longer match the durable bytes. Every
    // request from here on fails, permanently, because poisoning never clears. Suppressing
    // it (`clear_poison`, or catching the panic) would resume serving from that suspect
    // index while discarding the only evidence — a loud correct failure traded for a quiet
    // wrong one. So the poison flag is treated as the useful signal it is: report unhealthy,
    // let liveness restart the process, and let a fresh instance rebuild from disk.
    //
    // In a cluster that restart is also what unblocks failover: a poisoned writer goes on
    // renewing its lease from the background renewer (which deliberately holds no store
    // lock), so no standby can be promoted until this process actually dies.
    if st.db.is_poisoned() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unhealthy",
                "error": "store lock poisoned: a panic left this instance's in-RAM state \
                          untrustworthy — it must be restarted",
            })),
        )
            .into_response();
    }
    // Note what is deliberately NOT checked: whether the lock is currently *held*. A long
    // write makes the store busy, not broken, and restarting an instance mid-batch would be
    // far worse than the bug this guards. `is_poisoned` reads a flag and never acquires, so
    // busy-ness is invisible here — the same distinction `ready` makes (nidus-abx.3).
    "ok".into_response()
}

/// Readiness: this instance has a store open and can serve requests.
///
/// `503` while a standby waits for the writer handle, so a load balancer routes around it
/// instead of sending requests that would all answer `503` anyway. Split from
/// [`health`] because the two genuinely differ for a standby: live, but not ready.
///
/// Beyond "is a store open", readiness asks whether this instance can serve *usefully*, so
/// it also fails for a **fenced** writer (superseded — every write will fail) and for a
/// reader past its `--max-staleness` bound.
///
/// **Answered entirely from atomics, with no store lock** (nidus-abx.3). This used to route
/// through `cluster_status`, which needs the lock: `try_read` returned `WouldBlock` while a
/// large upsert held the write guard, and the probe reported `503` — so the single writer
/// dropped out of the load balancer in the middle of the very batch it existed to perform.
/// Busy is not unhealthy. Reading through the lock-free [`crate::Readiness`] handle removes
/// the `WouldBlock` case from this path altogether rather than merely tolerating it.
async fn ready(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    let (role, staleness_secs) = readiness_check(&st)?;
    Ok(Json(json!({
        "ready": true,
        "role": role,
        "staleness_secs": staleness_secs,
    })))
}

/// The readiness decision, in one place.
///
/// `GET /ready` answers with it and `/metrics` exports it as `nidus_ready`. Factored
/// because they had already drifted once in draft — the gauge omitted the staleness bound,
/// so a reader that had stopped refreshing would report `nidus_ready 1` on the dashboard
/// while `/ready` was `503`ing it out of the load balancer. Two answers to "is this
/// instance serving" is worse than either one alone.
///
/// Returns `(role, staleness_secs)` when ready. Every check reads an atomic and acquires
/// nothing (nidus-abx.3), so both callers stay off the store lock.
fn readiness_check(st: &AppState) -> Result<(String, u64), ApiError> {
    if !st.open.load(std::sync::atomic::Ordering::Acquire) {
        return Err(ApiError::from(not_open()));
    }
    // A poisoned lock means every data route now fails permanently (nidus-abx.1), so this
    // instance must leave the Service as well as being restarted by liveness — waiting for
    // the restart would keep traffic arriving at something that can only 500.
    //
    // Checked here rather than inherited from `read_status`, which this handler no longer
    // calls: `is_poisoned` reads a flag and acquires nothing, so it keeps readiness entirely
    // off the lock. Making readiness lock-free must not make it blind.
    if st.db.is_poisoned() {
        return Err(ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            err: anyhow::anyhow!(
                "store lock poisoned: a panic left this instance's in-RAM state \
                 untrustworthy — it must be restarted"
            ),
        });
    }
    // Published before the `open` flag above, so this is always present once open is true.
    let Some(status) = st.readiness.get() else {
        return Err(ApiError::from(not_open()));
    };
    if status.fenced() {
        return Err(ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            err: anyhow::anyhow!(
                "writer fenced: this instance was superseded and every write will fail — \
                 it must be replaced"
            ),
        });
    }
    let staleness_secs = status.staleness_secs();
    if let Some(max) = st.max_staleness
        && staleness_secs > max.as_secs()
    {
        return Err(ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            err: anyhow::anyhow!(
                "stale: last verified current {}s ago, beyond the {}s bound — this reader \
                 is not being refreshed",
                staleness_secs,
                max.as_secs()
            ),
        });
    }
    Ok((format!("{:?}", status.role()), staleness_secs))
}

/// `GET /cluster` — role, writer-handle state, fencing token, commit counter, staleness.
///
/// The introspection an operator needs mid-incident: which instance holds the writer
/// handle, whether this one has been fenced, and how far behind a reader is. Reads only
/// in-RAM state, so it is safe to scrape.
async fn cluster(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    let s = read_status(&st)?;
    Ok(Json(json!({
        "role": format!("{:?}", s.role),
        "cluster": s.cluster,
        "holds_writer_handle": s.holds_writer_handle,
        "fenced": s.fenced,
        "lease_owner": s.lease_owner,
        "commit_version": s.commit_version,
        "staleness_secs": s.staleness_secs,
        "max_staleness_secs": st.max_staleness.map(|d| d.as_secs()),
    })))
}

/// Read [`ClusterStatus`] without blocking the async executor.
///
/// `try_read` rather than `read`: a probe must answer in constant time, and a blocking
/// acquisition would queue behind a long write (a large upsert holds the guard for
/// seconds). Contention is reported as `503` — momentarily unable to answer is the honest
/// response, and far better than stalling executor threads under a probe storm.
fn read_status(st: &AppState) -> Result<crate::ClusterStatus, ApiError> {
    match st.db.try_read() {
        Ok(guard) => match guard.as_ref() {
            Some(db) => Ok(db.cluster_status()),
            None => Err(ApiError::from(not_open())),
        },
        Err(std::sync::TryLockError::WouldBlock) => Err(ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            err: anyhow::anyhow!("store busy: could not read status without blocking"),
        }),
        Err(std::sync::TryLockError::Poisoned(_)) => {
            Err(ApiError::internal(anyhow::anyhow!("store lock poisoned")))
        }
    }
}

/// Store-wide introspection: pinned dimension, distance metric, the collection
/// list, and the on-disk footprint. Mirrors the CLI `stats` command so a
/// network-only client can inspect the store without the binary.
async fn stats(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    let body = run_read(st, |db| {
        Ok(json!({
            "dimension": db.dimension(),
            "distance": format!("{:?}", db.config().distance),
            "ann": db.config().ann.map(AnnDto::from),
            "collections": db.collections(),
            "footprint": FootprintDto::from(db.footprint()),
        }))
    })
    .await?;
    Ok(Json(body))
}

async fn list_collections(State(st): State<AppState>) -> Result<Json<Vec<String>>, ApiError> {
    let names = run_read(st, |db| Ok(db.collections())).await?;
    Ok(Json(names))
}

async fn create_collection(
    State(st): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<JsonValue>, ApiError> {
    let created = run_write(st, move |db| {
        db.create_collection(&name)?;
        Ok(name)
    })
    .await?;
    Ok(Json(json!({ "created": created })))
}

async fn drop_collection(
    State(st): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<JsonValue>, ApiError> {
    let dropped = run_write(st, move |db| {
        db.drop_collection(&name)?;
        Ok(name)
    })
    .await?;
    Ok(Json(json!({ "dropped": dropped })))
}

async fn get_meta(
    State(st): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<std::collections::BTreeMap<String, String>>, ApiError> {
    let meta = run_read(st, move |db| Ok(db.get_meta(&name))).await?;
    Ok(Json(meta))
}

async fn set_meta(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(meta): Json<std::collections::BTreeMap<String, String>>,
) -> Result<Json<JsonValue>, ApiError> {
    run_write(st, move |db| db.set_meta(&name, meta)).await?;
    Ok(Json(json!({ "ok": true })))
}

async fn upsert(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let n = run_write(st, move |db| db.upsert(&name, &req.records)).await?;
    Ok(Json(json!({ "upserted": n })))
}

async fn delete_records(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<DeleteRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let n = run_write(st, move |db| match req.filter {
        Some(f) => db.delete_where(&name, &f),
        None => {
            let ids: Vec<&str> = req.ids.iter().map(String::as_str).collect();
            db.delete(&name, &ids)
        }
    })
    .await?;
    Ok(Json(json!({ "deleted": n })))
}

async fn records(
    State(st): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<Vec<Record>>, ApiError> {
    let recs = run_read(st, move |db| Ok(db.get_all(&name))).await?;
    Ok(Json(recs))
}

async fn search(
    State(st): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Vec<HitDto>>, ApiError> {
    let hits = run_read(st, move |db| {
        let SearchRequest {
            query,
            scope,
            top_k,
            min_score,
            filter,
        } = req;
        let opts = SearchOpts {
            top_k,
            min_score,
            filter,
        };
        scoped(&scope, |s| db.search(s, &query, &opts))
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

/// Resolve a wire `scope` (an empty list means "every collection") and run `f` with the
/// corresponding [`Scope`]. Shared by the `/search`, `/text-search`, `/hybrid-search`,
/// and `/list` handlers so the empty-means-all rule lives in one place.
fn scoped<T>(scope: &[String], f: impl FnOnce(Scope) -> anyhow::Result<T>) -> anyhow::Result<T> {
    let refs: Vec<&str> = scope.iter().map(String::as_str).collect();
    if refs.is_empty() {
        f(Scope::All)
    } else {
        f(Scope::Collections(&refs))
    }
}

async fn list(
    State(st): State<AppState>,
    Json(req): Json<ListRequest>,
) -> Result<Json<Vec<HitDto>>, ApiError> {
    let hits = run_read(st, move |db| {
        let ListRequest {
            scope,
            offset,
            limit,
            filter,
        } = req;
        scoped(&scope, |s| db.list(s, &filter, offset, limit))
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

async fn set_fts_schema(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<FtsSchemaRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    run_write(st, move |db| {
        let decl: Vec<(String, Language)> = req
            .fields
            .iter()
            .map(|f| (f.clone(), Language::English))
            .collect();
        db.set_fts_schema(&name, &decl)
    })
    .await?;
    Ok(Json(json!({ "ok": true })))
}

async fn text_search(
    State(st): State<AppState>,
    Json(req): Json<TextSearchRequest>,
) -> Result<Json<Vec<HitDto>>, ApiError> {
    let hits = run_read(st, move |db| {
        let TextSearchRequest {
            field,
            query,
            scope,
            top_k,
            min_score,
            filter,
        } = req;
        let opts = SearchOpts {
            top_k,
            min_score,
            filter,
        };
        let q = FtsQuery::new(field, query);
        scoped(&scope, |s| db.text_search(s, &q, &opts))
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

async fn hybrid_search(
    State(st): State<AppState>,
    Json(req): Json<HybridSearchRequest>,
) -> Result<Json<Vec<HitDto>>, ApiError> {
    let hits = run_read(st, move |db| {
        let HybridSearchRequest {
            vector,
            field,
            text,
            scope,
            top_k,
            filter,
            rrf_k,
            candidates,
        } = req;
        let opts = HybridOpts {
            top_k,
            filter,
            rrf_k,
            candidates,
        };
        let q = FtsQuery::new(field, text);
        scoped(&scope, |s| db.hybrid_search(s, &vector, &q, &opts))
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

async fn flush(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    run_write(st, |db| db.flush()).await?;
    Ok(Json(json!({ "ok": true })))
}

async fn compact(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    run_write(st, |db| db.compact()).await?;
    Ok(Json(json!({ "ok": true })))
}

/// `POST /refresh` — adopt a writer's newer committed state (SPEC §14.6).
///
/// A read-only instance over a shared store loads a snapshot at open and would otherwise
/// serve it forever; this is how a caller advances it. Explicit rather than automatic
/// because the alternative — refreshing before every read — puts a manifest fetch on the
/// hot path of exactly the read-heavy fan-out this mode exists for. Callers that want
/// near-live reads can poll it; those that write through one instance need never call it.
///
/// `adopted` reports whether newer state was actually taken up, so a poller can tell "no
/// change" from "advanced". Harmless in every other configuration: a writer already holds
/// the only mutating handle and an in-memory store has no backend, so both answer `false`.
async fn refresh(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    let adopted = run_write(st, |db| db.refresh()).await?;
    Ok(Json(json!({ "adopted": adopted })))
}

// ── Memory handlers (the `memory` feature) ───────────────────────────────────
//
// CRITICAL async/lock discipline (see the module docs): embedding and
// summarizing are async network IO and MUST happen OUTSIDE the store `RwLock`
// — the guard is never held across an `.await`. So each handler does the
// network work lock-free first, then takes the lock only for the synchronous
// store step. The pin/identity/search logic is REUSED from `crate::memory`
// (the same code the in-process `Memory` uses), not reimplemented here.

/// `POST /collections/{name}/remember` — text in. Optionally summarize, then
/// embed (both lock-free), then upsert under the write lock. `mode` is `"raw"`
/// (default) or `"summarize"`.
#[cfg(feature = "memory")]
async fn remember(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let embedder = st.embedder.clone().ok_or_else(missing_embedder_error)?;

    let RememberRequest {
        id,
        text,
        mode,
        attrs,
    } = req;
    // `mut` only when a summarizer can stamp META_SUMMARY/META_SOURCE into it.
    #[cfg_attr(not(all(feature = "memory", feature = "summarize")), allow(unused_mut))]
    let mut attrs = attrs;

    // 1) (Optional) summarize + 2) embed — all lock-free network IO.
    let embed_text: String = match mode.as_deref() {
        Some("summarize") => {
            #[cfg(all(feature = "memory", feature = "summarize"))]
            {
                let summarizer = st.summarizer.clone().ok_or_else(|| {
                    ApiError::bad_request(anyhow::anyhow!(
                        "nidus serve was started without a summarizer; pass --summarize-provider …"
                    ))
                })?;
                let summary = summarizer
                    .summarize(&text, &SummarizeOpts::default())
                    .await
                    .map_err(anyhow::Error::new)?;
                // Stamp the same attr keys the in-process `Memory` uses so a
                // recall hit is explainable back to the source text.
                attrs.insert(
                    crate::memory::META_SUMMARY.to_string(),
                    crate::Value::Str(summary.clone()),
                );
                attrs.insert(
                    crate::memory::META_SOURCE.to_string(),
                    crate::Value::Str(text.clone()),
                );
                summary
            }
            #[cfg(all(feature = "memory", not(feature = "summarize")))]
            {
                return Err(ApiError::bad_request(anyhow::anyhow!(
                    "this build has no summarizer support; rebuild with --features serve"
                )));
            }
        }
        Some("raw") | None => text,
        Some(other) => {
            return Err(ApiError::bad_request(anyhow::anyhow!(
                "unknown remember mode '{other}'; use 'raw' or 'summarize'"
            )));
        }
    };
    let vector = embedder
        .embed(&embed_text)
        .await
        .map_err(anyhow::Error::new)?;

    // 3) Store: pin the embedding-space identity (reused from `crate::memory`)
    //    and upsert — the only step that takes the write lock.
    let n = run_write(st, move |db| {
        crate::memory::ensure_collection_and_pin(db, embedder.as_ref(), &name)?;
        db.upsert(&name, &[Record::new(id, vector, attrs)])
    })
    .await?;
    Ok(Json(json!({ "ok": true, "upserted": n })))
}

/// `POST /collections/{name}/recall` — query text in, ranked hits out. Embeds
/// the query lock-free, then reads under the shared lock. Refuses a cross-model
/// recall via the same identity guard the in-process `Memory` uses.
#[cfg(feature = "memory")]
async fn recall(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<Vec<HitDto>>, ApiError> {
    let embedder = st.embedder.clone().ok_or_else(missing_embedder_error)?;

    let RecallRequest {
        query,
        top_k,
        min_score,
        filter,
    } = req;

    // Embed the query off-lock (network IO), then search under the read lock.
    let vector = embedder
        .embed_query(&query)
        .await
        .map_err(anyhow::Error::new)?;
    let opts = SearchOpts {
        top_k,
        min_score,
        filter,
    };
    let hits = run_read(st, move |db| {
        crate::memory::guard_recall_identity(db, embedder.as_ref(), &name)?;
        db.search(name.as_str(), &vector, &opts)
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

/// The `400` returned when a memory route is hit but no embedder was configured
/// at serve time.
#[cfg(feature = "memory")]
fn missing_embedder_error() -> ApiError {
    ApiError::bad_request(anyhow::anyhow!(
        "nidus serve was started without an embedder; pass --embed-provider … to enable /remember and /recall"
    ))
}

/// Run a **read** operation on a blocking task under a shared lock — concurrent
/// reads proceed in parallel.
async fn run_read<F, T>(st: AppState, f: F) -> Result<T, ApiError>
where
    F: FnOnce(&Nidus) -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    // Picked up here, on the async side, because a task-local does not follow work onto a
    // blocking thread — this is the handoff, and doing it in the two `run_*` helpers means
    // every handler gets cancellation without knowing the concept exists.
    let cancel = limits::current_cancel();
    tokio::task::spawn_blocking(move || {
        let db = st
            .db
            .read()
            .map_err(|_| anyhow::anyhow!("store lock poisoned"))?;
        let db = db.as_ref().ok_or_else(not_open)?;
        match cancel {
            Some(cancel) => cancel.scope(|| f(db)),
            None => f(db),
        }
    })
    .await
    .map_err(|e| ApiError::internal(anyhow::anyhow!("task join error: {e}")))?
    .map_err(ApiError::from)
}

/// Run a **write** operation under the exclusive lock, **group-committed**: it is applied
/// together with whatever other writes are queued at that moment, and the group shares one
/// disk barrier (see [`commit`], nidus-xb9.1).
///
/// The result is returned only after that barrier succeeds, so this is exactly as durable as
/// the fsync-per-call path it replaced — a `200` from here still means the bytes are on disk.
/// A single writer with nothing queued beside it forms a group of one and pays the same
/// append-then-barrier it always did.
async fn run_write<F, T>(st: AppState, f: F) -> Result<T, ApiError>
where
    F: FnOnce(&mut Nidus) -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    // Refuse before queueing when there is no store to write to, so the honest `503` comes
    // from here rather than from the committer's cannot-answer fallback. Reads an atomic, not
    // the store lock — a queue of writes must not be able to delay this.
    if !st.open.load(std::sync::atomic::Ordering::Acquire) {
        return Err(ApiError::from(not_open()));
    }
    // Picked up on the async side for the same reason as in `run_read`: a task-local does not
    // follow work onto the blocking thread that ends up applying it.
    let cancel = limits::current_cancel();
    st.commit
        .submit(st.db.clone(), cancel, f)
        .await
        .map_err(ApiError::from)
}

// ── Error response ──────────────────────────────────────────────────────────

/// A handler error carrying the HTTP status to report. The body is always
/// `{ "error": … }`. Status is classified from the error so clients can tell a
/// bad request from a genuine server fault (the library uses `anyhow`, so the
/// classification is by message — the few client-fault errors the store raises
/// have stable, distinctive wording).
struct ApiError {
    status: StatusCode,
    err: anyhow::Error,
}

impl ApiError {
    fn internal(err: anyhow::Error) -> Self {
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            err,
        }
    }

    /// A `400 Bad Request` with a caller-facing message (e.g. a memory route hit
    /// on a server started without an embedder, or an unknown `remember` mode).
    #[cfg(feature = "memory")]
    fn bad_request(err: anyhow::Error) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            err,
        }
    }
}

/// Map a store error to an HTTP status. Defaults to `500`; recognises the
/// store's client-fault messages and the writer-lock conflict.
/// The error every data route returns before the store is open — a standby writer still
/// waiting for promotion, or the brief window during a normal open.
///
/// `503` with `Retry-After` semantics is the honest answer: the request is valid and the
/// process is healthy, it simply has no store to serve yet.
fn not_open() -> anyhow::Error {
    anyhow::anyhow!(
        "store is not open yet: this instance is waiting for the writer handle \
         (standby) or still starting up"
    )
}

fn classify(err: &anyhow::Error) -> StatusCode {
    let msg = format!("{err:#}").to_lowercase();
    if msg.contains("store is not open yet") {
        StatusCode::SERVICE_UNAVAILABLE
    } else if msg.contains("does not match store dimension") {
        StatusCode::BAD_REQUEST
    } else if msg.contains("read-only store") {
        StatusCode::FORBIDDEN
    } else if msg.contains("store is locked") {
        StatusCode::CONFLICT
    } else if msg.contains("different embedding models") {
        // remember/recall into a collection already pinned to another embedder:
        // the request conflicts with the collection's committed embedding space.
        StatusCode::CONFLICT
    } else if msg.contains("max_vector_bytes") || msg.contains("out of memory") {
        StatusCode::INSUFFICIENT_STORAGE
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError {
            status: classify(&err),
            err,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({ "error": format!("{:#}", self.err) })),
        )
            .into_response()
    }
}

/// The single place tests build [`AppState`], so adding a field updates one site instead of
/// every helper. Lives at module level, not inside `mod tests`, so the `memory`-gated
/// `memory_tests` module sees it too via `use super::*` — those helpers compile only on the
/// `serve` lane, which is exactly how they drifted out of sync unnoticed.
#[cfg(test)]
fn test_state(db: Option<Nidus>) -> AppState {
    let open = db.is_some();
    // Publish the readiness handle exactly as `serve` does, so the tests exercise the same
    // lock-free path a real probe takes rather than a special case.
    let readiness = std::sync::OnceLock::new();
    if let Some(db) = &db {
        let _ = readiness.set(db.readiness());
    }
    AppState {
        db: Arc::new(RwLock::new(db)),
        open: Arc::new(std::sync::atomic::AtomicBool::new(open)),
        readiness: Arc::new(readiness),
        max_staleness: None,
        token: None,
        // Generous by default so an ordinary test never trips admission control; the
        // backpressure tests build their own tight `Limits`.
        limits: Arc::new(limits::Limits::new(
            1024,
            None,
            None,
            None,
            16 * 1024 * 1024,
        )),
        commit: commit::Committer::new(),
        #[cfg(feature = "memory")]
        embedder: None,
        #[cfg(all(feature = "memory", feature = "summarize"))]
        summarizer: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, header::AUTHORIZATION};
    use tower::ServiceExt; // for `oneshot`

    /// Build a router over a fresh in-memory store of the given dimension.
    fn test_router(dim: usize) -> Router {
        let db = Nidus::open_in_memory(dim).unwrap();
        router_over(Some(db))
    }

    /// Build a router over an optional store — `None` models an instance whose store is
    /// not open yet (a standby waiting for promotion).
    fn router_over(db: Option<Nidus>) -> Router {
        let state = test_state(db);
        router(state, 16 * 1024 * 1024)
    }

    /// Build a router **and keep the state**, for the tests that manipulate the store lock
    /// itself — holding it to model a long write, or poisoning it to model a panic.
    fn router_and_state(dim: usize) -> (Router, AppState) {
        let db = Nidus::open_in_memory(dim).unwrap();
        let state = test_state(Some(db));
        (router(state.clone(), 16 * 1024 * 1024), state)
    }

    async fn json_body(resp: Response) -> JsonValue {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn post(path: &str, body: JsonValue) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn get(path: &str) -> Request<Body> {
        Request::builder().uri(path).body(Body::empty()).unwrap()
    }

    /// A client that never links the library can drive the whole lifecycle over
    /// HTTP: create → upsert → search → stats. Exercises the network-only surface
    /// the docs promise.
    #[tokio::test]
    async fn full_lifecycle_over_http() {
        let app = test_router(3);

        // Create a collection.
        let resp = app
            .clone()
            .oneshot(post("/collections/docs", json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Upsert two records.
        let resp = app
            .clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "a", "vector": [1, 0, 0], "attrs": {"lang": {"Str": "rust"}}},
                    {"id": "b", "vector": [0, 1, 0], "attrs": {"lang": {"Str": "go"}}}
                ]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["upserted"], 2);

        // Search.
        let resp = app
            .clone()
            .oneshot(post("/search", json!({"query": [1, 0, 0], "top_k": 1})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        assert_eq!(hits[0]["id"], "a");

        // Stats reflects the store: dimension, collection list, and footprint.
        let resp = app.clone().oneshot(get("/stats")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let stats = json_body(resp).await;
        assert_eq!(stats["dimension"], 3);
        assert_eq!(stats["distance"], "Cosine");
        assert_eq!(stats["ann"], JsonValue::Null); // exact search by default
        assert_eq!(stats["collections"], json!(["docs"]));
        assert_eq!(stats["footprint"]["doc_count"], 2);
    }

    /// Before the store is open — a standby waiting for promotion — liveness must still
    /// answer while readiness and every data route say `503`. Getting this backwards is
    /// what makes a standby unusable: a failing liveness probe has a supervisor kill the
    /// very instance that is meant to be waiting, and a passing readiness probe has a load
    /// balancer send it traffic it cannot serve.
    #[tokio::test]
    async fn not_open_is_live_but_not_ready() {
        let app = router_over(None);

        // Liveness: the process is up.
        let resp = app.clone().oneshot(get("/health")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Readiness: no store, so explicitly not ready.
        let resp = app.clone().oneshot(get("/ready")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        // Data routes: 503, not 500 — the request is fine, the instance just has no store.
        let resp = app.clone().oneshot(get("/stats")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let resp = app
            .clone()
            .oneshot(post("/search", json!({"query": [1, 0, 0], "top_k": 1})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// **A busy store is still ready** (nidus-abx.3).
    ///
    /// Readiness used to be answered through `cluster_status`, which needs the store lock, so
    /// a large upsert holding the write guard turned into `WouldBlock` and then a `503`. In a
    /// cluster that pulled the single writer out of the load balancer in the middle of the
    /// very batch it existed to perform. Busy is not unhealthy.
    // Holding the guard across the awaits is the whole point: it models a write batch in
    // flight while probes arrive. It cannot deadlock — the handlers under test are precisely
    // the ones that must never take this lock, which is what the assertions verify. If a
    // future change made `/ready` or `/health` acquire it, this test would hang rather than
    // fail, which is itself a loud signal.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn a_busy_store_is_still_ready() {
        let (app, state) = router_and_state(3);

        // Hold the exclusive guard for the length of the test — exactly what a long upsert
        // does. Nothing below may block on it.
        let guard = state.db.write().unwrap();

        let resp = app.clone().oneshot(get("/ready")).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a write in flight must not make a healthy instance report NOT ready"
        );

        // Liveness likewise: the lock is held, not poisoned.
        let resp = app.clone().oneshot(get("/health")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // By deliberate contrast, `/cluster` still answers 503 while it cannot read without
        // blocking. Nothing routes on it, so "momentarily unable to answer" is the honest
        // response there — the distinction is the point, not an oversight.
        let resp = app.clone().oneshot(get("/cluster")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        drop(guard);
        let resp = app.clone().oneshot(get("/cluster")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "and it recovers once free");
    }

    /// **Concurrent writes are applied as one group sharing one barrier** (nidus-xb9.1).
    ///
    /// The measured ceiling this exists to move is the per-call disk barrier: `~7.6ms` paid
    /// once per `upsert` *call*, so eight concurrent clients used to take eight barriers to
    /// commit eight batches. What has to be true afterwards is that the group actually forms —
    /// `writes > groups` — and that every request in it still gets its `200`.
    ///
    /// Forming a group deterministically needs the writes to genuinely overlap, so the store
    /// guard is held while they queue. That is not a contrivance: it is exactly the state a
    /// server is in whenever a write is already running when the next arrives, which the
    /// concurrency sweep showed is most of the time under load.
    // The guard is held across awaits on purpose (see `a_busy_store_is_still_ready`): it is
    // what puts several writes in the queue at once. The tasks below are spawned, not awaited,
    // so nothing here waits on the lock it holds.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_writes_share_one_barrier_and_all_still_get_their_200() {
        let (app, state) = router_and_state(3);

        const N: usize = 8;
        let guard = state.db.write().unwrap();
        let mut tasks = tokio::task::JoinSet::new();
        for i in 0..N {
            let app = app.clone();
            tasks.spawn(async move {
                app.oneshot(post(
                    "/collections/docs/upsert",
                    json!({"records": [{"id": format!("d{i}"), "vector": [1, 0, 0], "attrs": {}}]}),
                ))
                .await
                .unwrap()
                .status()
            });
        }
        // Wait until every write has reached the queue. The leader is parked on the guard held
        // below, so none of them can complete — but it may already have *drained* them, which
        // is why this waits on the monotonic submitted count rather than the queue's length:
        // the length is 0 in exactly the case where the coalescing worked best.
        //
        // `sleep`, not `yield_now`: this thread has to genuinely stand aside, and the test
        // binary runs hundreds of tests at once, so a spin loop just keeps the CPU it is
        // waiting for the workers to have.
        for _ in 0..2_000 {
            if state.commit.submitted() >= N as u64 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        assert_eq!(
            state.commit.submitted(),
            N as u64,
            "not every write reached the queue, so this test would not be measuring group commit"
        );
        drop(guard);

        while let Some(status) = tasks.join_next().await {
            assert_eq!(
                status.unwrap(),
                StatusCode::OK,
                "sharing a barrier must not cost any request its acknowledgement"
            );
        }

        let (groups, writes) = state.commit.stats();
        assert_eq!(writes, N as u64, "every write was applied exactly once");
        assert!(
            groups < writes,
            "the point of group commit: {writes} writes committed in {groups} groups"
        );

        // And all of it is in the store — coalescing barriers must not lose writes.
        let resp = app
            .oneshot(post("/list", json!({"limit": 100})))
            .await
            .unwrap();
        assert_eq!(json_body(resp).await.as_array().unwrap().len(), N);
    }

    /// **A single write forms a group of one and is never made to wait for company.**
    ///
    /// The classic way to get group commit wrong is a timed window: throughput rises under load
    /// and every uncontended write pays a delay it gains nothing from. There is no window here,
    /// and this is the guard against one appearing — with nothing else queued, one request must
    /// produce exactly one group.
    #[tokio::test]
    async fn a_lone_write_commits_immediately_as_a_group_of_one() {
        let (app, state) = router_and_state(3);
        for i in 0..3 {
            let resp = app
                .clone()
                .oneshot(post(
                    "/collections/docs/upsert",
                    json!({"records": [{"id": format!("d{i}"), "vector": [1, 0, 0], "attrs": {}}]}),
                ))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
        assert_eq!(
            state.commit.stats(),
            (3, 3),
            "three sequential writes must be three groups of one, with no waiting"
        );
    }

    /// **A poisoned store lock must report UNHEALTHY** so liveness restarts the process
    /// (nidus-abx.1).
    ///
    /// `std` poisons an `RwLock` only when a panic unwinds while it is held for *writing*, so
    /// a poisoned store is by construction one whose in-RAM index was mid-mutation and may no
    /// longer match the durable bytes. Poisoning never clears, so every later request fails
    /// forever — while `/health` used to return a hardcoded `"ok"`, meaning liveness never
    /// fired and the pod was never recycled. In a cluster that is worse than a crash: the
    /// background lease renewer holds no store lock, so the bricked writer keeps renewing its
    /// lease and no standby can be promoted.
    #[tokio::test]
    async fn a_poisoned_store_lock_reports_unhealthy_so_liveness_restarts_it() {
        let (app, state) = router_and_state(3);
        let resp = app.clone().oneshot(get("/health")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "healthy to begin with");

        // Poison it the way a panicking write handler would: unwind while holding the
        // exclusive guard. The panic hook is silenced for the moment it takes, so a
        // deliberate panic does not look like a failure in the test log. (Worst case a
        // *concurrent* test's panic message is suppressed; that test still fails.)
        let db = state.db.clone();
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let _ = std::thread::spawn(move || {
            let _guard = db.write().unwrap();
            panic!("a handler panicked mid-write");
        })
        .join();
        std::panic::set_hook(hook);
        assert!(
            state.db.is_poisoned(),
            "a panic on the WRITE path must poison the lock"
        );

        let resp = app.clone().oneshot(get("/health")).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "a poisoned store is unrecoverable — liveness must fail so the process restarts"
        );
        let body = json_body(resp).await;
        assert_eq!(body["status"], "unhealthy");

        // Readiness must agree, so the instance leaves the Service as well as getting
        // restarted — otherwise traffic keeps arriving at something that can only 500 for
        // however long the liveness probe takes to fire. (This assertion caught exactly that
        // gap: making readiness lock-free initially made it blind to the poison flag.)
        let resp = app.clone().oneshot(get("/ready")).await.unwrap();
        assert_ne!(resp.status(), StatusCode::OK);
    }

    /// A panic on the **read** path must NOT brick the instance — it does not poison the
    /// lock, so this whole failure mode is writer-only. Guards the reasoning in
    /// `a_poisoned_store_lock_reports_unhealthy_so_liveness_restarts_it`: if std ever changed
    /// here, the health check above would start firing on harmless search panics.
    #[tokio::test]
    async fn a_panic_on_the_read_path_leaves_the_instance_healthy() {
        let (app, state) = router_and_state(3);

        let db = state.db.clone();
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let _ = std::thread::spawn(move || {
            let _guard = db.read().unwrap();
            panic!("a search panicked");
        })
        .join();
        std::panic::set_hook(hook);

        assert!(
            !state.db.is_poisoned(),
            "a read-path panic must not poison the lock"
        );
        let resp = app.clone().oneshot(get("/health")).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "still healthy, still serving"
        );
        let resp = app.clone().oneshot(get("/ready")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Once the store is open, readiness flips.
    #[tokio::test]
    async fn open_store_is_ready() {
        let resp = test_router(3).oneshot(get("/ready")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// `/cluster` reports the instance's role and handle state. An in-memory store is the
    /// degenerate case — no cluster, no lease — and must say so rather than erroring.
    #[tokio::test]
    async fn cluster_endpoint_reports_role() {
        let resp = test_router(3).oneshot(get("/cluster")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["role"], "InMemory");
        assert_eq!(body["cluster"], false);
        assert_eq!(body["fenced"], false);
        assert_eq!(body["lease_owner"], JsonValue::Null);
        assert_eq!(body["staleness_secs"], 0);
    }

    /// A staleness bound only ever fails a *reader*: a writer is the current state by
    /// definition and reports zero staleness, so an aggressive bound must not take the
    /// writer out of rotation.
    #[tokio::test]
    async fn staleness_bound_does_not_fail_a_writer() {
        let db = Nidus::open_in_memory(3).unwrap();
        let state = AppState {
            // Zero tolerance: anything with nonzero staleness would fail.
            max_staleness: Some(std::time::Duration::ZERO),
            ..test_state(Some(db))
        };
        let app = router(state, 16 * 1024 * 1024);
        let resp = app.oneshot(get("/ready")).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a writer reports 0 staleness and must stay ready"
        );
    }

    /// Both probe endpoints stay reachable without a token: an orchestrator would read a
    /// `401` as "not ready" and never route to a healthy instance.
    #[tokio::test]
    async fn probes_are_exempt_from_auth() {
        let db = Nidus::open_in_memory(3).unwrap();
        let state = AppState {
            token: Some(auth::Token::new("s3cret")),
            ..test_state(Some(db))
        };
        let app = router(state, 16 * 1024 * 1024);

        for path in ["/health", "/ready"] {
            let resp = app.clone().oneshot(get(path)).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{path} should skip auth");
        }
        // A real route still requires the token.
        let resp = app.clone().oneshot(get("/stats")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    /// `POST /refresh` is routed and answers `adopted: false` where there is nothing to
    /// adopt — an in-memory store tracks no separate writer. Whether a cluster *reader*
    /// actually takes up a writer's commits needs two processes and a shared backend, so
    /// that lives in `tests/e2e/cluster.rs`.
    #[tokio::test]
    async fn refresh_is_a_no_op_without_a_shared_writer() {
        let app = test_router(3);
        let resp = app
            .clone()
            .oneshot(post("/refresh", json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["adopted"], false);
    }

    /// Full-text + hybrid search over HTTP: declare schema, upsert (incl. a text-only
    /// doc), then text-search and hybrid-search.
    #[tokio::test]
    async fn fts_and_hybrid_over_http() {
        let app = test_router(3);

        // Declare the FTS schema for `docs`.`body`.
        let resp = app
            .clone()
            .oneshot(post(
                "/collections/docs/fts-schema",
                json!({"fields": ["body"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Upsert a vector doc and a text-only doc (vector omitted).
        let resp = app
            .clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "the quick brown fox"}}},
                    {"id": "b", "attrs": {"body": {"Str": "foxes are running quickly"}}}
                ]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["upserted"], 2);

        // Text search: "running" stems to match doc b.
        let resp = app
            .clone()
            .oneshot(post(
                "/text-search",
                json!({"field": "body", "query": "run", "top_k": 5}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        assert_eq!(hits[0]["id"], "b");

        // Hybrid: vector favours a, text favours b — both surface.
        let resp = app
            .clone()
            .oneshot(post(
                "/hybrid-search",
                json!({"vector": [1, 0, 0], "field": "body", "text": "fox", "top_k": 5}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ids: Vec<String> = json_body(resp)
            .await
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h["id"].as_str().unwrap().to_string())
            .collect();
        assert!(ids.contains(&"a".to_string()) && ids.contains(&"b".to_string()));
    }

    // ── Backpressure (nidus-abx.2) ──────────────────────────────────────────

    /// A router whose admission control is already exhausted. `Limits::new(0, …)` hands out
    /// no permits at all, so **every** non-exempt request sheds — a deterministic stand-in
    /// for saturation that needs no concurrency and cannot flake. (Not reachable through
    /// configuration: `resolve_concurrency` reads `0` as "auto".) That the permit pool
    /// itself fills and drains correctly is `limits::tests::in_flight_tracks_outstanding_permits`.
    fn saturated_router() -> Router {
        let db = Nidus::open_in_memory(3).unwrap();
        let state = AppState {
            limits: Arc::new(limits::Limits::new(0, None, None, None, 16 * 1024 * 1024)),
            ..test_state(Some(db))
        };
        router(state, 16 * 1024 * 1024)
    }

    /// Past the cap, a request is **shed** with a retryable `503` + `Retry-After` — not
    /// queued behind a lock with no bound on how long it waits, which is what the server
    /// did before.
    #[tokio::test]
    async fn requests_past_the_concurrency_limit_are_shed() {
        let app = saturated_router();

        let resp = app
            .clone()
            .oneshot(post("/search", json!({"query": [1, 0, 0], "top_k": 1})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get("retry-after").unwrap(),
            "1",
            "a shed request must tell the client to come back"
        );
        let body = json_body(resp).await;
        assert_eq!(
            body["retryable"], true,
            "nothing was attempted, so retrying is safe — say so in the body, not just \
             the status"
        );
        assert!(body["error"].as_str().unwrap().contains("overloaded"));
    }

    /// **Probes are never shed.** They take no store lock, so they cost nothing to admit —
    /// and shedding them under load would fail liveness and get a busy-but-healthy instance
    /// restarted, which is exactly the availability trap nidus-abx.1/.3 closed one layer
    /// down. `/metrics` too: an incident is when someone is looking.
    #[tokio::test]
    async fn probes_and_metrics_survive_saturation() {
        let app = saturated_router();
        for path in ["/health", "/ready", "/metrics"] {
            let resp = app.clone().oneshot(get(path)).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "{path} must answer while the server is shedding"
            );
        }
    }

    /// A request cannot occupy a connection indefinitely. The deadline answers `504` —
    /// **not** `503`: this request *was* admitted and its work is still running, so an
    /// immediate retry would pile a second copy onto an instance already behind.
    // Holding the guard across awaits is the point: it is a write batch in flight that the
    // read below can never get past, so the deadline is the only thing that can end it.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn a_request_that_outlives_its_deadline_gets_504() {
        let db = Nidus::open_in_memory(3).unwrap();
        let state = AppState {
            limits: Arc::new(limits::Limits::new(
                8,
                Some(std::time::Duration::from_millis(50)),
                None,
                None,
                16 * 1024 * 1024,
            )),
            ..test_state(Some(db))
        };
        let app = router(state.clone(), 16 * 1024 * 1024);

        let guard = state.db.write().unwrap();
        let resp = app.clone().oneshot(get("/stats")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT);
        let body = json_body(resp).await;
        assert_eq!(
            body["retryable"], false,
            "the work may still be running; an immediate retry doubles it"
        );

        // Releasing the guard lets the abandoned blocking task finish, and the next request
        // succeeds — the deadline freed the client without breaking the instance.
        drop(guard);
        let resp = app.oneshot(get("/stats")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── Observability (nidus-abx.4) ─────────────────────────────────────────

    /// `/metrics` renders Prometheus text, needs no token, and — the property that makes
    /// leaving it unauthenticated defensible — never names a collection.
    #[tokio::test]
    async fn metrics_scrape_is_open_and_leaks_no_collection_names() {
        let db = Nidus::open_in_memory(3).unwrap();
        let state = AppState {
            token: Some(auth::Token::new("s3cret")),
            ..test_state(Some(db))
        };
        let app = router(state, 16 * 1024 * 1024);

        // Drive traffic through a collection whose name would be unmistakable in a label.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections/very-secret-project/upsert")
                    .header("content-type", "application/json")
                    .header(AUTHORIZATION, "Bearer s3cret")
                    .body(Body::from(
                        json!({"records": [{"id": "a", "vector": [1, 0, 0], "attrs": {}}]})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // No credential — a scraper that got a 401 would report the target as down.
        let resp = app.clone().oneshot(get("/metrics")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            !text.contains("very-secret-project"),
            "a collection name reached a metric label:\n{text}"
        );
        assert!(
            text.contains("nidus_http_requests_total{route=\"/collections/{name}/upsert\""),
            "the upsert should be counted under its template:\n{text}"
        );
        // Library counters and the histogram both render.
        assert!(text.contains("nidus_search_queries_total"));
        assert!(text.contains("nidus_http_request_duration_seconds_bucket"));
        assert!(text.contains("nidus_http_concurrency_limit"));
        // Every series must be preceded by its TYPE line, or Prometheus rejects the scrape.
        assert!(text.contains("# TYPE nidus_http_requests_total counter"));
    }

    /// The two in-flight gauges are separate series with separate meanings, and the
    /// permit gauge says out loud that it excludes work whose deadline already fired
    /// (nidus-bcg).
    ///
    /// Asserted rather than left to the docs because the discrepancy is invisible in a
    /// scrape: the gauge reads plausibly low, and an operator correlating it with the shed
    /// count has no way to tell "nothing is running" from "the permit was handed back
    /// while the scan finishes noticing". The HELP text is the only place that distinction
    /// is delivered to the person reading it.
    #[tokio::test]
    async fn the_two_in_flight_gauges_are_distinct_and_self_describing() {
        let app = test_router(3);
        let resp = app.oneshot(get("/metrics")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();

        for name in [
            "nidus_http_requests_in_flight",
            "nidus_http_admitted_in_flight",
        ] {
            assert!(
                text.contains(&format!("# HELP {name} ")),
                "{name} has no HELP line:\n{text}"
            );
            assert!(
                text.contains(&format!("# TYPE {name} gauge")),
                "{name} has no TYPE line:\n{text}"
            );
        }

        let help = text
            .lines()
            .find(|l| l.starts_with("# HELP nidus_http_admitted_in_flight "))
            .expect("checked above");
        assert!(
            help.contains("permits") && help.contains("deadline"),
            "the permit gauge must name what it counts and what it excludes, got: {help}"
        );
    }

    /// Search-path counters move, so a scrape can distinguish "queries are slow" from
    /// "queries are slow because the index is not being used".
    #[tokio::test]
    async fn a_search_advances_the_search_counters() {
        let app = test_router(3);
        let before = crate::metrics::metrics().search_queries.get();
        let resp = app
            .oneshot(post("/search", json!({"query": [1, 0, 0], "top_k": 1})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            crate::metrics::metrics().search_queries.get() > before,
            "a served search must be counted"
        );
    }

    /// `nidus_ready` must always agree with `GET /ready` — they share one decision
    /// (`readiness_check`), and this is the regression guard on that.
    ///
    /// A dashboard that disagrees with the load balancer about whether an instance is
    /// serving is worse than no dashboard: the poisoned case below is exactly when someone
    /// is looking at both.
    #[tokio::test]
    async fn the_ready_gauge_agrees_with_the_ready_probe() {
        async fn ready_pair(app: &Router) -> (bool, bool) {
            let probe =
                app.clone().oneshot(get("/ready")).await.unwrap().status() == StatusCode::OK;
            let scrape = app.clone().oneshot(get("/metrics")).await.unwrap();
            assert_eq!(
                scrape.status(),
                StatusCode::OK,
                "/metrics must answer in every state — it is what you read during one"
            );
            let bytes = to_bytes(scrape.into_body(), usize::MAX).await.unwrap();
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            let gauge = text
                .lines()
                .find_map(|l| l.strip_prefix("nidus_ready "))
                .expect("nidus_ready gauge")
                == "1";
            (probe, gauge)
        }

        // No store yet (a standby waiting for promotion): neither says ready.
        let (probe, gauge) = ready_pair(&router_over(None)).await;
        assert!(!probe && !gauge, "not open: probe={probe} gauge={gauge}");

        // Open and healthy: both say ready.
        let (app, state) = router_and_state(3);
        let (probe, gauge) = ready_pair(&app).await;
        assert!(probe && gauge, "open: probe={probe} gauge={gauge}");

        // Poisoned by a panic on the write path: both must flip.
        let db = state.db.clone();
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let _ = std::thread::spawn(move || {
            let _guard = db.write().unwrap();
            panic!("a handler panicked mid-write");
        })
        .join();
        std::panic::set_hook(hook);

        let (probe, gauge) = ready_pair(&app).await;
        assert!(!probe && !gauge, "poisoned: probe={probe} gauge={gauge}");
    }

    /// A caller's correlation id is echoed back, so a client can quote it in a bug report
    /// and an operator can grep for the same string in the access log.
    #[tokio::test]
    async fn request_ids_are_echoed_and_minted() {
        let app = test_router(3);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header("x-request-id", "trace-42")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.headers().get("x-request-id").unwrap(), "trace-42");

        // Absent one, the server mints its own rather than leaving the response unlabelled.
        let resp = app.oneshot(get("/health")).await.unwrap();
        assert!(resp.headers().get("x-request-id").is_some());
    }

    // ── Exposure warnings (nidus-abx.6) ─────────────────────────────────────

    /// The startup warning is **advisory** — it must never refuse. Refusing a non-loopback
    /// bind would break every deployment that terminates TLS at a proxy, which is the
    /// architecture the docs recommend. This asserts the function is total: no panic, no
    /// error, on every combination including an unknown address.
    #[test]
    fn exposure_warnings_never_refuse() {
        let loopback: std::net::SocketAddr = "127.0.0.1:7700".parse().unwrap();
        let public: std::net::SocketAddr = "0.0.0.0:7700".parse().unwrap();
        for addr in [None, Some(loopback), Some(public)] {
            for has_token in [false, true] {
                warn_on_exposure(addr, has_token);
            }
        }
    }

    /// The two configurations that earn a warning, and the one that does not.
    ///
    /// A loopback bind must stay silent: `nidus serve` on a laptop is the frictionless
    /// default the docs lead with, and a security warning printed on every ordinary run is
    /// one an operator learns to scroll past — which costs exactly the case it exists for.
    #[test]
    fn exposure_is_classified_by_reachability_then_auth() {
        let cases = [
            ("127.0.0.1:7700", true, Exposure::Contained),
            ("127.0.0.1:7700", false, Exposure::Contained),
            ("[::1]:7700", false, Exposure::Contained),
            ("0.0.0.0:7700", true, Exposure::CleartextCredential),
            ("0.0.0.0:7700", false, Exposure::Unauthenticated),
            ("10.1.2.3:7700", true, Exposure::CleartextCredential),
            ("10.1.2.3:7700", false, Exposure::Unauthenticated),
        ];
        for (addr, has_token, want) in cases {
            assert_eq!(
                exposure(addr.parse().unwrap(), has_token),
                want,
                "{addr} (token: {has_token})"
            );
        }
    }

    #[test]
    fn classify_maps_client_faults() {
        let cases = [
            (
                "vector length 4 does not match store dimension 8",
                StatusCode::BAD_REQUEST,
            ),
            (
                "read-only store: mutations are not allowed",
                StatusCode::FORBIDDEN,
            ),
            ("store is locked: /tmp/s/lock", StatusCode::CONFLICT),
            (
                "collection 'x' was written with embedder 'a/b', but this Memory uses 'c/d'; \
                 vectors from different embedding models are not comparable",
                StatusCode::CONFLICT,
            ),
            (
                "upsert would grow the vector matrix to 9 bytes, exceeding max_vector_bytes (8 bytes)",
                StatusCode::INSUFFICIENT_STORAGE,
            ),
            (
                "out of memory reserving capacity for 3 rows",
                StatusCode::INSUFFICIENT_STORAGE,
            ),
            (
                "something unexpected blew up",
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ];
        for (msg, want) in cases {
            assert_eq!(classify(&anyhow!("{msg}")), want, "message: {msg}");
        }
    }

    #[test]
    fn classify_sees_through_context_chains() {
        // The store wraps errors with .context(); classify reads the full chain.
        let err = anyhow!("vector length 4 does not match store dimension 8")
            .context("while upserting into 'docs'");
        assert_eq!(classify(&err), StatusCode::BAD_REQUEST);
    }
}

// ── Memory-route tests (the `memory` feature) ────────────────────────────────
//
// These drive the `/remember` + `/recall` handlers **offline**: the server's
// embedder is an `OpenAiCompat` adapter pointed at a tiny in-process TCP mock
// that always answers with a fixed `{"data":[{"embedding":[…],"index":0}]}`.
// No real provider network is touched (mirrors the mock in `src/embed/*`).
//
// Requires `embed-openai-compat`: the mock is driven through the OpenAI-compatible
// embedder, so a `memory` build with no provider adapter compiled (an `AnyEmbedder`
// with zero variants) has nothing to exercise here. Every CI test lane that turns
// on `memory` also enables `embed-all`, so coverage is unchanged.
#[cfg(all(test, feature = "memory", feature = "embed-openai-compat"))]
mod memory_tests {
    use super::*;
    use crate::embed::{EmbedConfig, EmbedProvider};
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;
    use std::thread;
    use tower::ServiceExt; // for `oneshot`

    /// The fixed embedding every mock response carries — a 3-dim vector, so the
    /// backing store is opened at dimension 3. A stored doc and any query embed
    /// to the *same* vector, so recall scores it ~1.0.
    const EMBED_BODY: &str = r#"{"data":[{"embedding":[0.1,0.2,0.3],"index":0}]}"#;
    const DIM: usize = 3;

    /// A multi-connection HTTP/1.1 mock: accepts connections forever on a
    /// background thread, drains each request (headers + Content-Length body),
    /// and replies with `EMBED_BODY`. Unlike the one-shot `embed::testutil`
    /// mock, this survives the several calls a remember→recall flow makes
    /// (dimension probe on build, then embed, then embed_query).
    fn spawn_embed_mock() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
        let addr = listener.local_addr().expect("mock addr");
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buf = Vec::new();
                let mut tmp = [0u8; 1024];
                let header_end = loop {
                    match stream.read(&mut tmp) {
                        Ok(0) => break buf.len(),
                        Ok(n) => {
                            buf.extend_from_slice(&tmp[..n]);
                            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                                break pos + 4;
                            }
                        }
                        Err(_) => break buf.len(),
                    }
                };
                let head = String::from_utf8_lossy(&buf[..header_end.min(buf.len())]).to_string();
                let content_length = head
                    .lines()
                    .find_map(|l| {
                        l.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                    })
                    .unwrap_or(0);
                while buf.len() < header_end + content_length {
                    match stream.read(&mut tmp) {
                        Ok(0) => break,
                        Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        Err(_) => break,
                    }
                }
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    EMBED_BODY.len(),
                    EMBED_BODY
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{addr}")
    }

    /// A router whose memory routes are backed by the offline mock embedder.
    async fn router_with_mock_embedder() -> Router {
        let base = spawn_embed_mock();
        let embedder = AnyEmbedder::build(
            EmbedProvider::OpenAiCompat,
            EmbedConfig::new("mock-model").base_url(base),
        )
        .await
        .expect("build mock embedder");
        let db = Nidus::open_in_memory(DIM).unwrap();
        let state = AppState {
            embedder: Some(Arc::new(embedder)),
            ..test_state(Some(db))
        };
        router(state, 16 * 1024 * 1024)
    }

    /// A router with NO embedder configured — memory routes must answer `400`.
    fn router_without_embedder() -> Router {
        let state = test_state(Some(Nidus::open_in_memory(DIM).unwrap()));
        router(state, 16 * 1024 * 1024)
    }

    async fn json_body(resp: Response) -> JsonValue {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn post(path: &str, body: JsonValue) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// Remember text (server embeds via the mock), then recall it back.
    #[tokio::test]
    async fn remember_then_recall_over_http() {
        let app = router_with_mock_embedder().await;

        let resp = app
            .clone()
            .oneshot(post(
                "/collections/notes/remember",
                json!({"id": "a", "text": "the quick brown fox", "attrs": {"tag": {"Str": "x"}}}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["ok"], true);
        assert_eq!(body["upserted"], 1);

        let resp = app
            .clone()
            .oneshot(post(
                "/collections/notes/recall",
                json!({"query": "quick fox", "top_k": 5}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        assert_eq!(hits[0]["id"], "a");
        assert_eq!(hits[0]["attrs"]["tag"]["Str"], "x");
    }

    /// A server started without an embedder rejects the memory routes with `400`
    /// and a message pointing at `--embed-provider`.
    #[tokio::test]
    async fn memory_routes_400_without_embedder() {
        let app = router_without_embedder();

        let resp = app
            .clone()
            .oneshot(post(
                "/collections/notes/remember",
                json!({"id": "a", "text": "hi"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = json_body(resp).await;
        assert!(
            body["error"].as_str().unwrap().contains("--embed-provider"),
            "message names the flag: {body}"
        );

        let resp = app
            .oneshot(post("/collections/notes/recall", json!({"query": "hi"})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// An unknown `mode` on `/remember` is a `400`, not a silent raw embed.
    #[tokio::test]
    async fn remember_rejects_unknown_mode() {
        let app = router_with_mock_embedder().await;
        let resp = app
            .oneshot(post(
                "/collections/notes/remember",
                json!({"id": "a", "text": "hi", "mode": "bogus"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// The wire DTOs deserialize from their documented shapes, defaults included.
    #[test]
    fn dto_serde() {
        let r: RememberRequest =
            serde_json::from_value(json!({"id": "a", "text": "hello", "attrs": {"k": {"Int": 1}}}))
                .unwrap();
        assert_eq!(r.id, "a");
        assert_eq!(r.text, "hello");
        assert!(r.mode.is_none());
        assert_eq!(r.attrs.len(), 1);

        // mode + minimal recall body (top_k defaults to 10).
        let r: RememberRequest =
            serde_json::from_value(json!({"id": "b", "text": "t", "mode": "summarize"})).unwrap();
        assert_eq!(r.mode.as_deref(), Some("summarize"));
        assert!(r.attrs.is_empty());

        let q: RecallRequest = serde_json::from_value(json!({"query": "find me"})).unwrap();
        assert_eq!(q.query, "find me");
        assert_eq!(q.top_k, 10);
        assert!(q.min_score.is_none());
    }
}
