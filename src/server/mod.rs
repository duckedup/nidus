//! `nidus serve` — a thin HTTP wrapper over one open [`Nidus`] (SPEC.md §9).

mod auth;
mod commit;
pub mod dto;
mod limits;
#[cfg(feature = "mcp")]
mod mcp;
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

use crate::{
    FilterIndexField, FtsField, FtsQuery, HybridOpts, ListOpts, Nidus, Record, RerankOpts, Scope,
    SearchOpts,
};
use dto::{
    AggregateRequest, AggregationDto, AnnDto, BatchFuse, BatchSearchRequest, BatchSearchResponse,
    CompactRequest, DeleteRequest, FilterIndexRequest, FootprintDto, FtsSchemaRequest, HitDto,
    HybridSearchRequest, ListRequest, MAX_BATCH_QUERIES, MAX_TOP_K, SearchRequest, SimilarRequest,
    TextSearchRequest, UpsertRequest, VersionsDto,
};

// ── AI-ingest (memory) imports: only under the `memory` feature (pulled by the
// `serve` umbrella). Plain `cli` builds a lean server without these. ──
#[cfg(feature = "memory")]
use crate::embed::{AnyEmbedder, Embedder};
#[cfg(all(feature = "memory", feature = "summarize"))]
use crate::summarize::{AnySummarizer, SummarizeOpts, Summarizer};
#[cfg(feature = "memory")]
use dto::{RecallRequest, RememberRequest};

// ── Rerank (opt-in cross-encoder stage) imports: applies to `/search`/`/hybrid-search`
// too, unlike `summarize`. Only `rerank_hits` is used — the `&Nidus`-holding
// `search_reranked`/`hybrid_reranked` wrappers would hold the store lock across the await.
#[cfg(feature = "rerank")]
use crate::Hit;
#[cfg(feature = "rerank")]
use crate::rerank::{AnyReranker, rerank_hits};
#[cfg(feature = "rerank")]
use dto::RerankRequest;

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
    /// How long a request body may stall between frames before it is abandoned (nidus-6c2). An
    /// *idle* bound, not a total one. `None` disables it, removing the only thing stopping a
    /// silent client from pinning a concurrency permit.
    pub body_idle_timeout: Option<std::time::Duration>,
    /// Fail readiness once a read-only instance is staler than this (mirrors
    /// [`Config::max_staleness`](crate::Config::max_staleness)). `None` = no bound.
    pub max_staleness: Option<std::time::Duration>,
    /// Refresh a read-only instance on this interval so it stays current without a sidecar
    /// or cron calling `POST /refresh`. `None` (the default) leaves refreshing entirely to
    /// the caller.
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
    /// Optional reranker for the opt-in cross-encoder stage on `/search`, `/hybrid-search`,
    /// and `/collections/{name}/recall`. `None` makes a request asking for it answer `400`
    /// naming `--rerank-provider`, never a silent pass-through.
    #[cfg(feature = "rerank")]
    pub reranker: Option<Arc<AnyReranker>>,
}

/// How `nidus mcp` (the stdio transport) is configured beyond the store itself. No
/// `addr`/`token`/limit knobs like [`ServeConfig`]'s — none of those mean anything over a pipe
/// with exactly one client.
#[cfg(feature = "mcp")]
pub struct StdioConfig {
    /// Embedder backing `remember`/`recall`. `None` disables them (the tool then answers
    /// an `internal_error` naming `--embed-provider`, same as `nidus serve`).
    #[cfg(feature = "memory")]
    pub embedder: Option<Arc<AnyEmbedder>>,
    /// Optional summarizer enabling `summarize: true` on `remember`.
    #[cfg(all(feature = "memory", feature = "summarize"))]
    pub summarizer: Option<Arc<AnySummarizer>>,
    /// Optional reranker, as in [`ServeConfig`]. `None` makes a rerank request answer `400`.
    #[cfg(feature = "rerank")]
    pub reranker: Option<Arc<AnyReranker>>,
    /// How often to renew the cluster writer lease, as in [`ServeConfig`]. An interactive
    /// session idles far longer than `lock_ttl` between writes, so without this a peer
    /// reclaims the lease and the next write finds itself fenced.
    pub lease_renew_interval: std::time::Duration,
}

/// Shared, cloneable handle to the one open store.
#[derive(Clone)]
struct AppState {
    /// `None` until the store finishes opening — a standby writer waiting for promotion
    /// sits here indefinitely by design. Data routes answer `503` while it is empty; see
    /// [`serve`] for why the listener comes up first.
    db: Arc<RwLock<Option<Nidus>>>,
    /// Mirrors `db.is_some()` for [`ready`] to read.
    open: Arc<std::sync::atomic::AtomicBool>,
    /// The lock-free readiness handle, published once the store opens. A `OnceLock` because a
    /// store opens exactly once per process; reading it costs an atomic load, which keeps
    /// [`ready`] off the store lock entirely (nidus-abx.3).
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
    /// Shared reranker for the opt-in cross-encoder stage; `None` disables it (→ `400`).
    #[cfg(feature = "rerank")]
    reranker: Option<Arc<AnyReranker>>,
}

/// Bind the address, open the store, and serve until a shutdown signal (Ctrl-C /
/// SIGTERM); flush and release the writer handle on shutdown.
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
        #[cfg(feature = "rerank")]
        reranker: cfg.reranker,
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
    spawn_lease_renewal(state.db.clone(), renew_every);

    // Optional self-refresh, so a read-only instance stays current without a sidecar
    // calling POST /refresh. A tokio task rather than a library background thread: "no
    // background threads" is a property of the sync core, and the server is already async.
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

    shutdown_store(&state.db);

    // A failed open outranks the serve result: it is the actual cause.
    if let Some(e) = open_failed.write().ok().and_then(|mut f| f.take()) {
        return Err(e);
    }
    served
}

/// Speak MCP over stdio, for a local client that spawns its own `nidus mcp --dir …`. Unlike
/// [`serve`], the store opens EAGERLY and FAILS FAST — a second process on the same
/// directory exits immediately with an error naming the lock conflict.
#[cfg(feature = "mcp")]
pub async fn serve_stdio<F>(open: F, cfg: StdioConfig) -> anyhow::Result<()>
where
    F: FnOnce() -> anyhow::Result<Nidus> + Send + 'static,
{
    let db = tokio::task::spawn_blocking(open)
        .await
        .map_err(|e| anyhow::anyhow!("task join error: {e}"))?
        .context(
            "opening the store for `nidus mcp` — if another process already holds the writer \
             lock, run `nidus serve` instead so every client shares one writer",
        )?;
    let readiness = std::sync::OnceLock::new();
    let _ = readiness.set(db.readiness());
    let slot = Arc::new(RwLock::new(Some(db)));

    // A slim, inert `AppState`: no listener, so nothing sheds or bounds, and
    // `limits::current_cancel()` correctly returns `None` off the axum path. "Large", not
    // `usize::MAX`: tokio's `Semaphore` panics past 2^61-1 and `Limits` multiplies by 4.
    const EFFECTIVELY_UNBOUNDED: usize = 1 << 20;
    let state = AppState {
        db: slot.clone(),
        open: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        readiness: Arc::new(readiness),
        max_staleness: None,
        token: None,
        limits: Arc::new(limits::Limits::new(
            EFFECTIVELY_UNBOUNDED,
            None,
            None,
            None,
            usize::MAX,
        )),
        commit: commit::Committer::new(),
        #[cfg(feature = "memory")]
        embedder: cfg.embedder,
        #[cfg(all(feature = "memory", feature = "summarize"))]
        summarizer: cfg.summarizer,
        #[cfg(feature = "rerank")]
        reranker: cfg.reranker,
    };
    spawn_lease_renewal(slot.clone(), cfg.lease_renew_interval);

    // STDERR, NEVER STDOUT: stdout is the JSON-RPC framing channel, and a stray print there
    // corrupts the stream.
    eprintln!("nidus speaking MCP 2026-07-28 over stdio (EOF / Ctrl-C / SIGTERM to stop)");

    // A supervisor stops this with a signal rather than by closing stdin, and the flush below
    // (plus the writer-lock release on drop) is only reachable if we return instead of dying.
    let served = tokio::select! {
        r = mcp::serve_stdio(state) => r,
        _ = shutdown_signal() => Ok(()),
    };

    shutdown_store(&slot);
    served
}

/// Flush, then persist the derived ann/fts caches, on a clean shutdown. Both are
/// best-effort: a failure warns and the next open rebuilds, and an interrupted persist is
/// harmless because a torn cache fails its CRC and is discarded rather than adopted.
fn shutdown_store(slot: &Arc<RwLock<Option<Nidus>>>) {
    let Ok(mut guard) = slot.write() else { return };
    let Some(db) = guard.as_mut() else { return };
    if let Err(e) = db.flush() {
        crate::diag::diag!(
            crate::diag::Level::Warn,
            "shutdown",
            "flush failed",
            "err" => format!("{e:#}"),
        );
    }
    // Without this the store reopens correct but cold, silently paying a full index
    // rebuild — the ~0.05s warm open in performance.md needs no out-of-band call (#142).
    if let Err(e) = db.persist_index() {
        crate::diag::diag!(
            crate::diag::Level::Warn,
            "shutdown",
            "persisting the index cache failed",
            "err" => format!("{e:#}"),
        );
    }
}

/// Warn at startup when the bind address makes the security posture worse than the
/// configuration suggests (nidus-abx.6).
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
        .route("/versions", get(versions))
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
        .route("/collections/{name}/filter-index", post(set_filter_index))
        .route("/search", post(search))
        .route("/search/batch", post(search_batch))
        .route("/search/similar", post(search_similar))
        .route("/text-search", post(text_search))
        .route("/hybrid-search", post(hybrid_search))
        .route("/list", post(list))
        .route("/aggregate", post(aggregate))
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

    // The MCP surface (nidus-zm2), nested *before* the `.layer()` calls below so it inherits
    // the whole middleware stack instead of growing its own copy of each layer.
    #[cfg(feature = "mcp")]
    let router = router.nest_service("/mcp", mcp::service(state.clone(), max_body_bytes));

    // `.layer()` applies outermost last, so inside-out this reads: body limit, backpressure,
    // auth (outside backpressure, so an unauthenticated request never consumes a permit), then
    // observe outermost, so a 401 and a shed 503 are both counted.
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

/// Renew the cluster writer lease on a timer, independent of write traffic. Without it an
/// idle writer's lease goes stale and a standby peer legitimately reclaims it, fencing a
/// process that is alive and about to write. No-op when this instance holds no lease.
fn spawn_lease_renewal(db: Arc<RwLock<Option<Nidus>>>, ttl: std::time::Duration) {
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
                    // A definitive loss latches the store's `fenced` flag, so `/ready` and
                    // `/cluster` report it without waiting for a write to discover it
                    // (nidus-lp4.7). A transient error latches nothing; the next tick retries.
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

/// Resolve on the first shutdown signal: Ctrl-C everywhere, plus SIGTERM on Unix. Catching
/// SIGTERM is what lets the graceful path run (flush + writer-lock release on drop); without it
/// the process is SIGKILLed and a restarted pod waits out the full lock TTL.
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
async fn health(State(st): State<AppState>) -> Response {
    // A **poisoned** store lock is the one condition under which this process is broken
    // beyond recovery, and it must be escalated rather than papered over (nidus-abx.1).
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
    // Deliberately NOT checked: whether the lock is *held*. A long write is busy, not broken,
    // and restarting mid-batch would be worse than the bug this guards. `is_poisoned` reads a
    // flag without acquiring, so busy-ness is invisible here (nidus-abx.3).
    "ok".into_response()
}

/// Readiness: this instance has a store open and can serve requests.
async fn ready(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    let (role, staleness_secs) = readiness_check(&st)?;
    Ok(Json(json!({
        "ready": true,
        "role": role,
        "staleness_secs": staleness_secs,
    })))
}

/// The readiness decision, in one place.
fn readiness_check(st: &AppState) -> Result<(String, u64), ApiError> {
    if !st.open.load(std::sync::atomic::Ordering::Acquire) {
        return Err(ApiError::from(not_open()));
    }
    // A poisoned lock means every data route now fails permanently (nidus-abx.1), so this
    // instance must leave the Service as well as being restarted by liveness — waiting for
    // the restart would keep traffic arriving at something that can only 500.
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

/// `GET /versions` — the readable commit points and this instance's pin (SPEC §14.2).
async fn versions(State(st): State<AppState>) -> Result<Json<VersionsDto>, ApiError> {
    let v = run_read(st, |db| db.versions()).await?;
    Ok(Json(VersionsDto::from(v)))
}

/// Read [`ClusterStatus`] without blocking the async executor.
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
            "quantization": db.config().quantization,
            "query_threads": db.config().query_threads,
            "mmap": db.config().mmap,
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
    let SearchPlan {
        scope,
        query,
        opts,
        rerank_query,
    } = plan_search(req)?;
    #[cfg(feature = "rerank")]
    if let Some(rerank_query) = rerank_query {
        let reranker = st.reranker.clone().ok_or_else(missing_reranker_error)?;
        let hits = rerank_search_and_finish(st, reranker, scope, query, rerank_query, opts).await?;
        return Ok(Json(hits.into_iter().map(HitDto::from).collect()));
    }
    #[cfg(not(feature = "rerank"))]
    let _ = rerank_query;
    let hits = run_read(st, move |db| {
        scoped(&scope, |s| db.search(s, &query, &opts))
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

/// Widen the search to [`store::rerank::rerank_depth`] inside `run_read`, await the reranker
/// **outside** it (network IO must never be held under the store lock), then cut the caller's
/// real page via the promoted `Store::finish` in a second, lock-brief `run_read`.
#[cfg(feature = "rerank")]
async fn rerank_search_and_finish(
    st: AppState,
    reranker: Arc<AnyReranker>,
    scope: Vec<String>,
    query: Vec<f32>,
    rerank_query: String,
    opts: SearchOpts,
) -> Result<Vec<Hit>, ApiError> {
    let rerank_opts = opts.rerank.clone().unwrap_or_default();
    let (widened, kept) = crate::store::rerank::widened_opts(&opts);
    let hits = run_read(st.clone(), move |db| {
        scoped(&scope, |s| db.search(s, &query, &widened))
    })
    .await?;
    let mut reranked = rerank_hits(reranker.as_ref(), &rerank_query, hits, &rerank_opts).await?;
    crate::store::rerank::retrim(&mut reranked, &opts, kept);
    run_read(st, move |db| Ok(db.store().finish(reranked, &opts))).await
}

/// One wire `SearchRequest` resolved into what `db.search` needs. `rerank_query` is carried
/// separately because it is only used outside `run_read`, and is `None` when unreranked.
struct SearchPlan {
    scope: Vec<String>,
    query: Vec<f32>,
    opts: SearchOpts,
    rerank_query: Option<String>,
}

/// Turn one wire `SearchRequest` into a [`SearchPlan`], running every check that can 400
/// **before** any query executes — a batch must not half-run and then reject leg 7.
fn plan_search(req: SearchRequest) -> Result<SearchPlan, ApiError> {
    check_page(req.offset, req.top_k)?;
    let projection = check_projection(req.include_attributes, req.exclude_attributes)?;
    #[cfg(feature = "rerank")]
    let (rerank, rerank_query) = match check_rerank(req.rerank, None)? {
        Some((ro, q)) => (Some(ro), Some(q)),
        None => (None, None),
    };
    #[cfg(not(feature = "rerank"))]
    let (rerank, rerank_query): (Option<RerankOpts>, Option<String>) = (None, None);
    let opts = SearchOpts {
        top_k: req.top_k,
        offset: req.offset,
        min_score: req.min_score,
        filter: req.filter,
        exact: req.exact,
        projection,
        // Vector search has one score to report; annotations are a text/hybrid surface.
        explain: false,
        rank_by: req.rank_by,
        limit_per: req.limit_per,
        diversity: req.diversity,
        expand: req.expand.map(Into::into),
        rerank,
    };
    #[cfg(feature = "rerank")]
    check_rerank_depth(&opts)?;
    Ok(SearchPlan {
        scope: req.scope,
        query: req.query,
        opts,
        rerank_query,
    })
}

/// `POST /search/similar`: "more like this" over the vector already stored at
/// `collection`/`id`. An empty `scope` searches only the source's own collection, unlike plain
/// `search` where an empty scope means every collection.
async fn search_similar(
    State(st): State<AppState>,
    Json(req): Json<SimilarRequest>,
) -> Result<Json<Vec<HitDto>>, ApiError> {
    check_page(req.offset, req.top_k)?;
    let projection = check_projection(req.include_attributes, req.exclude_attributes)?;
    let opts = SearchOpts {
        top_k: req.top_k,
        offset: req.offset,
        min_score: req.min_score,
        filter: req.filter,
        exact: req.exact,
        projection,
        explain: false,
        rank_by: req.rank_by,
        limit_per: req.limit_per,
        diversity: req.diversity,
        expand: req.expand.map(Into::into),
        // No rerank here: a cross-encoder scores (query text, candidate) pairs, and
        // more-like-this starts from a stored vector with no query text to score against.
        rerank: None,
    };
    let scope = if req.scope.is_empty() {
        vec![req.collection.clone()]
    } else {
        req.scope
    };
    let (collection, id) = (req.collection, req.id);
    let hits = run_read(st, move |db| {
        scoped(&scope, |s| db.search_similar(s, &collection, &id, &opts))
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

/// `POST /search/batch`: answer up to [`MAX_BATCH_QUERIES`] vector queries in one round-trip,
/// optionally fusing them into a single ranking with RRF (nidus-m50.11).
async fn search_batch(
    State(st): State<AppState>,
    Json(req): Json<BatchSearchRequest>,
) -> Result<Json<BatchSearchResponse>, ApiError> {
    let BatchSearchRequest { queries, fuse } = req;
    if queries.is_empty() {
        return Err(ApiError::bad_request(anyhow::anyhow!(
            "queries must not be empty"
        )));
    }
    if queries.len() > MAX_BATCH_QUERIES {
        return Err(ApiError::bad_request(anyhow::anyhow!(
            "{} queries exceeds the maximum batch of {MAX_BATCH_QUERIES}",
            queries.len()
        )));
    }
    if let Some(f) = &fuse {
        check_fuse(f, queries.len())?;
    }
    // Rerank is not supported on `/search/batch` in v1 (root blueprint, decision 5): reject
    // rather than silently ignore, so a caller asking for it never mistakes plain metric
    // order for a reranked one.
    #[cfg(feature = "rerank")]
    if queries.iter().any(|q| q.rerank.is_some()) {
        return Err(ApiError::bad_request(anyhow::anyhow!(
            "rerank is not supported on /search/batch queries"
        )));
    }
    let plans = queries
        .into_iter()
        .map(plan_search)
        .collect::<Result<Vec<_>, ApiError>>()?;

    // One `run_read`, so the whole batch takes one lock, one blocking task, and lives under
    // the one request deadline — the batch cap above is what bounds the work that buys.
    let legs = run_read(st, move |db| {
        plans
            .iter()
            .map(|p| scoped(&p.scope, |s| db.search(s, &p.query, &p.opts)))
            .collect::<anyhow::Result<Vec<_>>>()
    })
    .await?;

    let Some(f) = fuse else {
        let results = legs
            .into_iter()
            .map(|hits| hits.into_iter().map(HitDto::from).collect())
            .collect();
        return Ok(Json(BatchSearchResponse {
            results: Some(results),
            fused: None,
        }));
    };
    let weighted = legs
        .into_iter()
        .enumerate()
        .map(|(i, hits)| {
            let leg = crate::fuse::FusionLeg::new(hits);
            match f.weights.get(i) {
                Some(&w) => leg.weight(w),
                None => leg,
            }
        })
        .collect();
    let fused = crate::fuse::rrf_fuse(weighted, f.rrf_k)
        .into_iter()
        .take(f.top_k)
        .map(|(hit, _)| HitDto::from(hit))
        .collect();
    Ok(Json(BatchSearchResponse {
        results: None,
        fused: Some(fused),
    }))
}

/// Refuse a fusion the server cannot honour as written. A weights list that does not line up
/// with the queries would re-weight the wrong leg, which is worse than refusing.
fn check_fuse(f: &BatchFuse, queries: usize) -> Result<(), ApiError> {
    if f.top_k > MAX_TOP_K {
        return Err(ApiError::bad_request(anyhow::anyhow!(
            "fuse.top_k {} exceeds the maximum of {MAX_TOP_K}",
            f.top_k
        )));
    }
    if !f.weights.is_empty() && f.weights.len() != queries {
        return Err(ApiError::bad_request(anyhow::anyhow!(
            "fuse.weights has {} entries but there are {queries} queries",
            f.weights.len()
        )));
    }
    if let Some(w) = f.weights.iter().find(|w| !w.is_finite() || **w < 0.0) {
        return Err(ApiError::bad_request(anyhow::anyhow!(
            "fuse.weights must be finite and non-negative, got {w}"
        )));
    }
    if !f.rrf_k.is_finite() || f.rrf_k < 0.0 {
        return Err(ApiError::bad_request(anyhow::anyhow!(
            "fuse.rrf_k must be finite and non-negative, got {}",
            f.rrf_k
        )));
    }
    Ok(())
}

/// `POST /aggregate`: count the filter-matching records and sum the named attributes,
/// without materializing any of them.
async fn aggregate(
    State(st): State<AppState>,
    Json(req): Json<AggregateRequest>,
) -> Result<Json<AggregationDto>, ApiError> {
    let AggregateRequest {
        scope,
        filter,
        sum,
        group_by,
    } = req;
    let opts = crate::AggregateOpts {
        filter,
        sum,
        group_by,
    };
    let out = run_read(st, move |db| scoped(&scope, |s| db.aggregate(s, &opts))).await?;
    Ok(Json(AggregationDto::from(out)))
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
    let ListRequest {
        scope,
        offset,
        limit,
        filter,
        include_attributes,
        exclude_attributes,
        order_by,
    } = req;
    let projection = check_projection(include_attributes, exclude_attributes)?;
    let hits = run_read(st, move |db| {
        let opts = ListOpts {
            offset,
            limit,
            filter,
            projection,
            order_by,
        };
        scoped(&scope, |s| db.list(s, &opts))
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
        let decl: Vec<FtsField> = req.fields.into_iter().map(FtsField::from).collect();
        db.set_fts_schema(&name, &decl)
    })
    .await?;
    Ok(Json(json!({ "ok": true })))
}

/// `POST /collections/{name}/filter-index` — declare the fields indexed for the text
/// predicates. Speed only: results are identical with or without it.
async fn set_filter_index(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<FilterIndexRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    run_write(st, move |db| {
        let decl: Vec<FilterIndexField> =
            req.fields.into_iter().map(FilterIndexField::from).collect();
        db.set_filter_index(&name, &decl)
    })
    .await?;
    Ok(Json(json!({ "ok": true })))
}

async fn text_search(
    State(st): State<AppState>,
    Json(req): Json<TextSearchRequest>,
) -> Result<Json<Vec<HitDto>>, ApiError> {
    check_page(req.offset, req.top_k)?;
    let TextSearchRequest {
        field,
        query,
        clauses,
        combine,
        scope,
        top_k,
        offset,
        min_score,
        filter,
        explain,
        highlight,
        include_attributes,
        exclude_attributes,
        rank_by,
        limit_per,
        diversity,
        expand,
        #[cfg(feature = "rerank")]
        rerank,
    } = req;
    #[cfg(feature = "rerank")]
    let default_rerank_query = query.clone();
    let clauses = check_clauses(field, query, clauses)?;
    let projection = check_projection(include_attributes, exclude_attributes)?;
    #[cfg(feature = "rerank")]
    let (rerank, rerank_query) = match check_rerank(rerank, default_rerank_query.as_deref())? {
        Some((ro, q)) => (Some(ro), Some(q)),
        None => (None, None),
    };
    #[cfg(not(feature = "rerank"))]
    let (rerank, rerank_query): (Option<RerankOpts>, Option<String>) = (None, None);

    let opts = SearchOpts {
        top_k,
        offset,
        min_score,
        filter,
        explain,
        projection,
        rank_by,
        limit_per,
        diversity,
        expand: expand.map(Into::into),
        rerank,
        ..Default::default()
    };
    #[cfg(feature = "rerank")]
    check_rerank_depth(&opts)?;
    let q = FtsQuery {
        clauses,
        combine,
        highlight,
    };

    #[cfg(feature = "rerank")]
    if let Some(rerank_query) = rerank_query {
        let reranker = st.reranker.clone().ok_or_else(missing_reranker_error)?;
        let hits =
            rerank_text_search_and_finish(st, reranker, scope, q, rerank_query, opts).await?;
        return Ok(Json(hits.into_iter().map(HitDto::from).collect()));
    }
    #[cfg(not(feature = "rerank"))]
    let _ = rerank_query;

    let hits = run_read(st, move |db| {
        scoped(&scope, |s| db.text_search(s, &q, &opts))
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

async fn hybrid_search(
    State(st): State<AppState>,
    Json(req): Json<HybridSearchRequest>,
) -> Result<Json<Vec<HitDto>>, ApiError> {
    check_page(req.offset, req.top_k)?;
    let HybridSearchRequest {
        vector,
        field,
        text,
        clauses,
        combine,
        scope,
        top_k,
        offset,
        filter,
        rrf_k,
        candidates,
        explain,
        highlight,
        vector_weight,
        text_weight,
        expand,
        #[cfg(feature = "rerank")]
        rerank,
    } = req;
    let clauses = check_clauses(field, text, clauses)?;
    #[cfg(feature = "rerank")]
    let (rerank, rerank_query) = match check_rerank(rerank, None)? {
        Some((ro, q)) => (Some(ro), Some(q)),
        None => (None, None),
    };
    #[cfg(not(feature = "rerank"))]
    let (rerank, rerank_query): (Option<RerankOpts>, Option<String>) = (None, None);
    let opts = HybridOpts {
        top_k,
        offset,
        filter,
        rrf_k,
        candidates,
        explain,
        vector_weight,
        text_weight,
        expand: expand.map(Into::into),
        rerank,
    };
    #[cfg(feature = "rerank")]
    check_rerank_hybrid_depth(&opts)?;
    let q = FtsQuery {
        clauses,
        combine,
        highlight,
    };

    #[cfg(feature = "rerank")]
    if let Some(rerank_query) = rerank_query {
        let reranker = st.reranker.clone().ok_or_else(missing_reranker_error)?;
        let hits =
            rerank_hybrid_and_finish(st, reranker, scope, vector, q, rerank_query, opts).await?;
        return Ok(Json(hits.into_iter().map(HitDto::from).collect()));
    }
    #[cfg(not(feature = "rerank"))]
    let _ = rerank_query;

    let hits = run_read(st, move |db| {
        scoped(&scope, |s| db.hybrid_search(s, &vector, &q, &opts))
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

/// Hybrid analogue of [`rerank_search_and_finish`]: widen inside `run_read`, rerank outside
/// it, then cut the page via the promoted `Store::finish_hybrid`.
#[cfg(feature = "rerank")]
async fn rerank_hybrid_and_finish(
    st: AppState,
    reranker: Arc<AnyReranker>,
    scope: Vec<String>,
    vector: Vec<f32>,
    q: FtsQuery,
    rerank_query: String,
    opts: HybridOpts,
) -> Result<Vec<Hit>, ApiError> {
    let rerank_opts = opts.rerank.clone().unwrap_or_default();
    let overscan = rerank_opts.overscan.max(1);
    let widened = HybridOpts {
        top_k: opts
            .offset
            .saturating_add(opts.top_k)
            .saturating_mul(overscan),
        offset: 0,
        candidates: opts.candidates.saturating_mul(overscan),
        ..opts.clone()
    };
    let hits = run_read(st.clone(), move |db| {
        scoped(&scope, |s| db.hybrid_search(s, &vector, &q, &widened))
    })
    .await?;
    let reranked = rerank_hits(reranker.as_ref(), &rerank_query, hits, &rerank_opts).await?;
    run_read(st, move |db| Ok(db.store().finish_hybrid(reranked, &opts))).await
}

/// Text analogue of [`rerank_search_and_finish`]: widen inside `run_read`, rerank outside
/// it, then cut the page via the promoted `Store::finish`.
#[cfg(feature = "rerank")]
async fn rerank_text_search_and_finish(
    st: AppState,
    reranker: Arc<AnyReranker>,
    scope: Vec<String>,
    q: FtsQuery,
    rerank_query: String,
    opts: SearchOpts,
) -> Result<Vec<Hit>, ApiError> {
    let rerank_opts = opts.rerank.clone().unwrap_or_default();
    let (widened, kept) = crate::store::rerank::widened_opts(&opts);
    let hits = run_read(st.clone(), move |db| {
        scoped(&scope, |s| db.text_search(s, &q, &widened))
    })
    .await?;
    let mut reranked = rerank_hits(reranker.as_ref(), &rerank_query, hits, &rerank_opts).await?;
    crate::store::rerank::retrim(&mut reranked, &opts, kept);
    run_read(st, move |db| Ok(db.store().finish(reranked, &opts))).await
}

async fn flush(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    run_write(st, |db| db.flush()).await?;
    Ok(Json(json!({ "ok": true })))
}

/// `POST /compact` — bodyless, `{}`, or `{"expired": true}` all accepted; a caller that never
/// sends a body (every shipped SDK) must keep compacting, so the body is optional, not required.
async fn compact(
    State(st): State<AppState>,
    body: Option<Json<CompactRequest>>,
) -> Result<Json<JsonValue>, ApiError> {
    let req = body.map(|Json(r)| r).unwrap_or_default();
    if req.expired {
        run_write(st, |db| db.sweep_expired().map(|_| ())).await?;
    } else {
        run_write(st, |db| db.compact()).await?;
    }
    Ok(Json(json!({ "ok": true })))
}

/// `POST /refresh` — adopt a writer's newer committed state (SPEC §14.6).
async fn refresh(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    let adopted = run_write(st, |db| db.refresh()).await?;
    Ok(Json(json!({ "adopted": adopted })))
}

// ── Memory handlers (the `memory` feature) ───────────────────────────────────
// CRITICAL: embedding and summarizing are async network IO and MUST happen OUTSIDE the store
// `RwLock` — never hold the guard across an `.await`. Logic is reused from `crate::memory`.

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
        ttl_seconds,
        dedupe_threshold,
    } = req;
    // `mut` only when a summarizer can stamp META_SUMMARY into it.
    #[cfg_attr(not(all(feature = "memory", feature = "summarize")), allow(unused_mut))]
    let mut attrs = attrs;
    let raw_text = text.clone();

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
                // Stamp the same attr key the in-process `Memory` uses so a
                // recall hit is explainable back to the source text.
                attrs.insert(
                    crate::memory::META_SUMMARY.to_string(),
                    crate::Value::Str(summary.clone()),
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

    // 3) Store: stamping, dedup, recency, and the upsert all live in `crate::memory` so
    // this surface cannot drift from the MCP tool or the in-process `Memory`. The whole
    // body runs inside one write closure, which is what makes the read-modify-write atomic.
    let write = crate::memory::RememberWrite {
        id,
        text: raw_text,
        attrs,
        ttl_seconds,
        dedupe_threshold,
    };
    let written = run_write(st, move |db| {
        crate::memory::commit_remember(db, embedder.as_ref(), &name, write, vector)
    })
    .await?;
    // `id` and `deduped` are echoed because a dedupe match redirects the write to another
    // entry: without them the caller cannot tell which record it just changed.
    Ok(Json(json!({
        "ok": true,
        "upserted": written.upserted,
        "id": written.id,
        "deduped": written.deduped,
    })))
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
    check_page(0, req.top_k)?;
    let embedder = st.embedder.clone().ok_or_else(missing_embedder_error)?;

    let RecallRequest {
        query,
        top_k,
        min_score,
        filter,
        diversity,
        rollup,
        #[cfg(feature = "rerank")]
        rerank,
    } = req;
    // The rerank query defaults to the recall query itself (decision: `rerank: {}` is a
    // valid minimal form here, unlike `/search`, which has no text of its own to fall back to).
    #[cfg(feature = "rerank")]
    let (rerank_opts, rerank_query) = match check_rerank(rerank, Some(query.as_str()))? {
        Some((ro, q)) => (Some(ro), Some(q)),
        None => (None, None),
    };
    #[cfg(not(feature = "rerank"))]
    let (rerank_opts, rerank_query): (Option<RerankOpts>, Option<String>) = (None, None);

    // Embed the query off-lock (network IO), then search under the read lock.
    let vector = embedder
        .embed_query(&query)
        .await
        .map_err(anyhow::Error::new)?;
    // Same TTL guard as every MCP read tool: AND-ed into the caller's filter so an
    // expired memory is invisible here too, not just over MCP (#106).
    let mut filter = filter;
    filter
        .0
        .push(crate::memory::not_expired_predicate(crate::meta::now_ms()));
    // Through `Rollup::as_opts`, the same mapping `Memory::recall` uses — the two surfaces
    // must not drift on what "read this as a chunked corpus" means.
    let rollup = rollup.map(crate::memory::Rollup::from).map(|r| r.as_opts());
    let opts = SearchOpts {
        top_k,
        min_score,
        filter,
        diversity,
        limit_per: rollup.as_ref().map(|(cap, _)| cap.clone()),
        expand: rollup.map(|(_, e)| e),
        rerank: rerank_opts,
        ..Default::default()
    };
    #[cfg(feature = "rerank")]
    check_rerank_depth(&opts)?;

    #[cfg(feature = "rerank")]
    if let Some(rerank_query) = rerank_query {
        let reranker = st.reranker.clone().ok_or_else(missing_reranker_error)?;
        let hits =
            rerank_recall_and_finish(st, reranker, name, embedder, vector, rerank_query, opts)
                .await?;
        return Ok(Json(hits.into_iter().map(HitDto::from).collect()));
    }
    #[cfg(not(feature = "rerank"))]
    let _ = rerank_query;

    let hits = run_read(st, move |db| {
        crate::memory::guard_recall_identity(db, embedder.as_ref(), &name)?;
        db.search(name.as_str(), &vector, &opts)
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

/// Recall analogue of [`rerank_search_and_finish`]: the identity guard runs inside both
/// `run_read`s (widen and tail), same as the unreranked path above.
#[cfg(all(feature = "memory", feature = "rerank"))]
async fn rerank_recall_and_finish(
    st: AppState,
    reranker: Arc<AnyReranker>,
    name: String,
    embedder: Arc<AnyEmbedder>,
    vector: Vec<f32>,
    rerank_query: String,
    opts: SearchOpts,
) -> Result<Vec<Hit>, ApiError> {
    let rerank_opts = opts.rerank.clone().unwrap_or_default();
    let (widened, kept) = crate::store::rerank::widened_opts(&opts);
    let name_for_widen = name.clone();
    let hits = run_read(st.clone(), move |db| {
        crate::memory::guard_recall_identity(db, embedder.as_ref(), &name_for_widen)?;
        db.search(name_for_widen.as_str(), &vector, &widened)
    })
    .await?;
    let mut reranked = rerank_hits(reranker.as_ref(), &rerank_query, hits, &rerank_opts).await?;
    crate::store::rerank::retrim(&mut reranked, &opts, kept);
    run_read(st, move |db| Ok(db.store().finish(reranked, &opts))).await
}

/// The `400` returned when a memory route is hit but no embedder was configured
/// at serve time.
#[cfg(feature = "memory")]
fn missing_embedder_error() -> ApiError {
    ApiError::bad_request(anyhow::anyhow!(
        "nidus serve was started without an embedder; pass --embed-provider … to enable /remember and /recall"
    ))
}

/// The `400` returned when a request asks for reranking but the server was started
/// without a reranker — never a `500`, and never a silent unreranked pass-through.
#[cfg(feature = "rerank")]
fn missing_reranker_error() -> ApiError {
    ApiError::bad_request(anyhow::anyhow!(
        "nidus serve was started without a reranker; pass --rerank-provider … to enable reranking"
    ))
}

/// Validate an optional [`RerankRequest`] into ([`RerankOpts`], query text), or a `400`.
/// `rerank: None` is `Ok(None)` unconditionally. `default_query` back-fills an empty
/// `rerank.query` (recall's own text); `/search` passes `None`, having no text of its own.
#[cfg(feature = "rerank")]
fn check_rerank(
    rerank: Option<RerankRequest>,
    default_query: Option<&str>,
) -> Result<Option<(RerankOpts, String)>, ApiError> {
    let Some(r) = rerank else { return Ok(None) };
    let query = if !r.query.is_empty() {
        r.query
    } else {
        default_query.unwrap_or_default().to_string()
    };
    if query.is_empty() {
        return Err(ApiError::bad_request(anyhow::anyhow!(
            "rerank.query must not be empty"
        )));
    }
    let mut opts = RerankOpts::default();
    if let Some(overscan) = r.overscan {
        if overscan == 0 {
            return Err(ApiError::bad_request(anyhow::anyhow!(
                "rerank.overscan must be at least 1"
            )));
        }
        opts.overscan = overscan;
    }
    if let Some(attr) = r.text_attr {
        opts.text_attr = attr;
    }
    Ok(Some((opts, query)))
}

/// Refuse a rerank candidate window past [`MAX_TOP_K`]: past this the store would have to rank,
/// and the provider score, an unreasonable depth. Reuses `rerank_depth`'s real formula
/// (`limit_per`'s overfetch included) rather than approximating it a second time.
#[cfg(feature = "rerank")]
fn check_rerank_depth(opts: &SearchOpts) -> Result<(), ApiError> {
    if opts.rerank.is_none() {
        return Ok(());
    }
    let depth = crate::store::rerank::rerank_depth(opts);
    if depth > MAX_TOP_K {
        return Err(ApiError::bad_request(anyhow::anyhow!(
            "rerank over top_k {} (offset {}) asks for a candidate depth of {depth}, exceeding \
             the maximum of {MAX_TOP_K}",
            opts.top_k,
            opts.offset
        )));
    }
    Ok(())
}

/// Hybrid analogue of [`check_rerank_depth`]: `HybridOpts` has no `limit_per`, so the depth is
/// just `(offset + top_k) * overscan`.
#[cfg(feature = "rerank")]
fn check_rerank_hybrid_depth(opts: &HybridOpts) -> Result<(), ApiError> {
    let Some(r) = &opts.rerank else { return Ok(()) };
    let depth = opts
        .offset
        .saturating_add(opts.top_k)
        .saturating_mul(r.overscan.max(1));
    if depth > MAX_TOP_K {
        return Err(ApiError::bad_request(anyhow::anyhow!(
            "rerank over top_k {} (offset {}) asks for a candidate depth of {depth}, exceeding \
             the maximum of {MAX_TOP_K}",
            opts.top_k,
            opts.offset
        )));
    }
    Ok(())
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

/// A handler error carrying the HTTP status to report; the body is always `{ "error": … }`.
/// Status is classified from the error so clients can tell a bad request from a server fault —
/// by message, since the library uses `anyhow`.
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
    fn bad_request(err: anyhow::Error) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            err,
        }
    }
}

/// Refuse an absurd page at the edge (nidus-m50.17, nidus-m50.8). The bound is on
/// `offset + top_k`, since that is how deep the ranking is actually computed — and it is a
/// refusal, not a clamp: a silently shortened page would look like the end of the results.
fn check_page(offset: usize, top_k: usize) -> Result<(), ApiError> {
    let depth = offset.saturating_add(top_k);
    if depth > MAX_TOP_K {
        return Err(ApiError::bad_request(anyhow::anyhow!(
            "offset {offset} + top_k {top_k} exceeds the maximum of {MAX_TOP_K}"
        )));
    }
    Ok(())
}

/// Resolve a request's projection, refusing a body that names both an include and an exclude
/// list. A `400`, not a precedence rule: silently honouring one of two contradictory
/// instructions returns a payload the caller did not ask for (nidus-m50.15).
fn check_projection(
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
) -> Result<crate::Projection, ApiError> {
    dto::resolve_projection(include, exclude)
        .map_err(|msg| ApiError::bad_request(anyhow::anyhow!(msg)))
}

/// Resolve a text query's clauses, refusing an unusable body — both spellings at once, neither,
/// or an empty `clauses` list — with a `400` (nidus-m50.10).
fn check_clauses(
    field: Option<String>,
    text: Option<String>,
    clauses: Option<Vec<dto::FtsClauseDto>>,
) -> Result<Vec<crate::FtsClause>, ApiError> {
    dto::resolve_clauses(field, text, clauses)
        .map_err(|msg| ApiError::bad_request(anyhow::anyhow!(msg)))
}

/// Map a store error to an HTTP status. Defaults to `500`; recognises the
/// store's client-fault messages and the writer-lock conflict.
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
    } else if msg.contains("does not match store dimension")
        || msg.contains("fts field")
        || msg.contains("filter index")
        || msg.contains("full-text query")
        || msg.contains(crate::store::BAD_QUERY)
    {
        // A rejected FTS or filter-index declaration, a clause-less text query, or a
        // malformed ranking knob: bad request bodies, not server faults.
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

/// The single place tests build [`AppState`], so a new field updates one site. At module level
/// rather than inside `mod tests` so the `memory`-gated `memory_tests` sees it too — those
/// compile only on the `serve` lane, which is how they drifted out of sync unnoticed.
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
        #[cfg(feature = "rerank")]
        reranker: None,
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

    /// A genuinely bodyless POST — no `content-type`, no body — the shape every shipped SDK
    /// and the documented bodyless `curl -X POST /compact` send. Distinct from `post(path,
    /// json!({}))`, which sets `content-type: application/json` over an empty object.
    fn post_no_body(path: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .body(Body::empty())
            .unwrap()
    }

    /// `expand` over the wire: the DTO reaches `SearchOpts`, the store stitches the window,
    /// and `context` reaches the response. The ids and scores are identical with it off, so
    /// the payload is all it can change.
    #[tokio::test]
    async fn expand_widens_a_hit_over_http_without_reordering() {
        let app = test_router(2);
        // Three overlapping chunks of "one two three four five six", as `ingest` would write.
        let chunk = |i: usize, start: usize, text: &str, v: f64| {
            json!({"id": format!("d#{i}"), "vector": [1.0, v], "attrs": {
                "nidus.parent_id": {"Str": "d"},
                "nidus.chunk_index": {"Int": i},
                "nidus.char_start": {"Int": start},
                "nidus.text": {"Str": text}
            }})
        };
        let resp = app
            .clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    chunk(0, 0, "one two three", 0.0),
                    chunk(1, 8, "three four five", 0.1),
                    chunk(2, 19, "five six", 0.2)
                ]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let plain = json_body(
            app.clone()
                .oneshot(post("/search", json!({"query": [1, 0.1], "top_k": 3})))
                .await
                .unwrap(),
        )
        .await;
        let expanded = json_body(
            app.clone()
                .oneshot(post(
                    "/search",
                    json!({"query": [1, 0.1], "top_k": 3, "expand": {"radius": 1}}),
                ))
                .await
                .unwrap(),
        )
        .await;

        let ranking = |hits: &JsonValue| -> Vec<JsonValue> {
            hits.as_array()
                .unwrap()
                .iter()
                .map(|h| json!([h["id"], h["score"]]))
                .collect()
        };
        assert_eq!(ranking(&plain), ranking(&expanded), "payload only");
        assert!(plain[0].get("context").is_none(), "{}", plain[0]);
        // The whole document, with neither seam repeating the shared overlap.
        assert_eq!(expanded[0]["context"], "one two three four five six");
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

    /// A bodyless `POST /compact` — what every shipped SDK and the documented curl send —
    /// must still reclaim. Asserts the `dead_rows` delta, since an envelope-only check
    /// would pass against a handler that does nothing.
    #[tokio::test]
    async fn compact_with_no_body_still_reclaims_dead_rows() {
        let app = test_router(3);
        app.clone()
            .oneshot(post("/collections/docs", json!({})))
            .await
            .unwrap();
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [{"id": "a", "vector": [1, 0, 0], "attrs": {}}]}),
            ))
            .await
            .unwrap();
        // Overwriting "a" leaves its old row dead.
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [{"id": "a", "vector": [0, 1, 0], "attrs": {}}]}),
            ))
            .await
            .unwrap();

        let stats = json_body(app.clone().oneshot(get("/stats")).await.unwrap()).await;
        let dead_before = stats["footprint"]["dead_rows"].as_u64().unwrap();
        assert!(dead_before > 0, "overwrite should have left a dead row");

        let resp = app.clone().oneshot(post_no_body("/compact")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await, json!({"ok": true}));

        let stats = json_body(app.oneshot(get("/stats")).await.unwrap()).await;
        assert_eq!(stats["footprint"]["dead_rows"], 0);
    }

    /// `{}` (what none of the SDKs send today, but a client could) must behave identically
    /// to a bodyless request — `expired` defaults to `false` either way.
    #[tokio::test]
    async fn compact_with_empty_json_body_still_reclaims_dead_rows() {
        let app = test_router(3);
        app.clone()
            .oneshot(post("/collections/docs", json!({})))
            .await
            .unwrap();
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [{"id": "a", "vector": [1, 0, 0], "attrs": {}}]}),
            ))
            .await
            .unwrap();
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [{"id": "a", "vector": [0, 1, 0], "attrs": {}}]}),
            ))
            .await
            .unwrap();

        let stats = json_body(app.clone().oneshot(get("/stats")).await.unwrap()).await;
        assert!(stats["footprint"]["dead_rows"].as_u64().unwrap() > 0);

        let resp = app
            .clone()
            .oneshot(post("/compact", json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await, json!({"ok": true}));

        let stats = json_body(app.oneshot(get("/stats")).await.unwrap()).await;
        assert_eq!(stats["footprint"]["dead_rows"], 0);
    }

    /// `{"expired": true}` sweeps entries whose `nidus.expires_at` has passed, on top of the
    /// ordinary reclaim — asserted via the `doc_count` drop and a follow-up search, not just
    /// the `{"ok": true}` envelope (there is otherwise zero coverage this actually deletes).
    #[tokio::test]
    async fn compact_with_expired_true_sweeps_past_ttl_entries() {
        let app = test_router(3);
        app.clone()
            .oneshot(post("/collections/docs", json!({})))
            .await
            .unwrap();
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "stale", "vector": [1, 0, 0],
                     "attrs": {"nidus.expires_at": {"DateTime": 1}}},
                    {"id": "fresh", "vector": [0, 1, 0],
                     "attrs": {"nidus.expires_at": {"DateTime": 99_999_999_999_999_i64}}}
                ]}),
            ))
            .await
            .unwrap();

        let stats = json_body(app.clone().oneshot(get("/stats")).await.unwrap()).await;
        assert_eq!(stats["footprint"]["doc_count"], 2);

        let resp = app
            .clone()
            .oneshot(post("/compact", json!({"expired": true})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await, json!({"ok": true}));

        let stats = json_body(app.clone().oneshot(get("/stats")).await.unwrap()).await;
        assert_eq!(stats["footprint"]["doc_count"], 1);

        let resp = app
            .clone()
            .oneshot(post("/search", json!({"query": [0, 1, 0], "top_k": 10})))
            .await
            .unwrap();
        let hits = json_body(resp).await;
        assert_eq!(hits.as_array().unwrap().len(), 1);
        assert_eq!(hits[0]["id"], "fresh");
    }

    /// A filter the caller wrote wrong is a bad request, not a server fault (nidus-oih).
    /// Only the library `Err` was pinned before, so these fell through `classify` to a 500 —
    /// which also books a client mistake against the 5xx health metric.
    #[tokio::test]
    async fn an_unusable_filter_is_a_bad_request_not_a_server_error() {
        let app = test_router(3);
        for (path, body) in [
            (
                "/search",
                json!({"query": [1, 0, 0], "filter": [{"Regex": ["k", "("]}]}),
            ),
            (
                "/search",
                json!({"query": [1, 0, 0], "filter": [{"Fuzzy": ["k", "x", 99]}]}),
            ),
            // Nested inside a group: `validate` recurses, so the tag must survive the nesting.
            (
                "/search",
                json!({"query": [1, 0, 0], "filter": [{"Not": {"Regex": ["k", "("]}}]}),
            ),
            ("/list", json!({"filter": [{"Regex": ["k", "("]}]})),
        ] {
            let resp = app.clone().oneshot(post(path, body.clone())).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::BAD_REQUEST,
                "{path} {body}: a caller's bad filter must not be a 5xx"
            );
            // The 400 must still say WHICH predicate was wrong. Tagging the error with
            // `.context` instead of inline once reduced this to "invalid query option".
            let err = json_body(resp).await["error"].as_str().unwrap().to_string();
            assert!(err.contains('`'), "{path}: error names no predicate: {err}");
        }
    }

    /// A `top_k` nothing clamps used to reach the bounded top-k heap and abort the process
    /// on "capacity overflow" (nidus-m50.17). The edge must call it a bad request instead.
    #[tokio::test]
    async fn an_absurd_top_k_is_a_bad_request_not_a_panic() {
        let app = test_router(3);
        for (path, body) in [
            (
                "/search",
                json!({"query": [1, 0, 0], "top_k": usize::MAX / 2}),
            ),
            (
                "/text-search",
                json!({"field": "body", "query": "x", "top_k": MAX_TOP_K + 1}),
            ),
            (
                "/hybrid-search",
                json!({"vector": [1, 0, 0], "field": "body", "text": "x", "top_k": MAX_TOP_K + 1}),
            ),
        ] {
            let resp = app.clone().oneshot(post(path, body)).await.unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{path}");
            let err = json_body(resp).await["error"].as_str().unwrap().to_string();
            assert!(err.contains("top_k"), "{path}: {err}");
        }
    }

    /// The bound is a ceiling, not a narrowing: everything up to it still searches.
    #[tokio::test]
    async fn a_top_k_at_the_maximum_still_searches() {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [{"id": "a", "vector": [1, 0, 0], "attrs": {}}]}),
            ))
            .await
            .unwrap();
        let resp = app
            .clone()
            .oneshot(post(
                "/search",
                json!({"query": [1, 0, 0], "top_k": MAX_TOP_K}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await[0]["id"], "a");
    }

    /// The bound is on how deep the ranking is computed, so a legal `top_k` with an offset
    /// that pushes past the ceiling is refused — and refused, not clamped, because a
    /// silently shortened page is indistinguishable from the end of the results.
    #[tokio::test]
    async fn an_offset_that_pushes_past_the_maximum_is_a_bad_request() {
        let app = test_router(3);
        for (path, body) in [
            (
                "/search",
                json!({"query": [1, 0, 0], "top_k": 10, "offset": MAX_TOP_K}),
            ),
            (
                "/text-search",
                json!({"field": "body", "query": "x", "top_k": 1, "offset": MAX_TOP_K}),
            ),
            (
                "/hybrid-search",
                json!({"vector": [1, 0, 0], "field": "body", "text": "x", "offset": MAX_TOP_K}),
            ),
        ] {
            let resp = app.clone().oneshot(post(path, body)).await.unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{path}");
            let err = json_body(resp).await["error"].as_str().unwrap().to_string();
            assert!(err.contains("offset"), "{path}: {err}");
        }
        // Exactly at the ceiling still searches.
        let resp = app
            .oneshot(post(
                "/search",
                json!({"query": [1, 0, 0], "top_k": 10, "offset": MAX_TOP_K - 10}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// `offset` is additive: omitting it must behave exactly as before it existed, and
    /// supplying it must cut the same ranking into non-overlapping pages.
    #[tokio::test]
    async fn search_offset_paginates_over_http() {
        let app = test_router(3);
        for (i, id) in ["a", "b", "c", "d"].iter().enumerate() {
            app.clone()
                .oneshot(post(
                    "/collections/docs/upsert",
                    json!({"records": [{"id": id, "vector": [1.0, i as f32 * 0.1, 0.0], "attrs": {}}]}),
                ))
                .await
                .unwrap();
        }
        let hits = |body: JsonValue| {
            let app = app.clone();
            async move {
                let resp = app.oneshot(post("/search", body)).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                json_body(resp).await
            }
        };

        let implicit = hits(json!({"query": [1, 0, 0], "top_k": 2})).await;
        let explicit = hits(json!({"query": [1, 0, 0], "top_k": 2, "offset": 0})).await;
        assert_eq!(implicit, explicit, "an omitted offset is offset 0");
        assert_eq!(implicit[0]["id"], "a");

        let second = hits(json!({"query": [1, 0, 0], "top_k": 2, "offset": 2})).await;
        assert_eq!(second[0]["id"], "c");
        assert_eq!(second[1]["id"], "d");

        let past_the_end = hits(json!({"query": [1, 0, 0], "top_k": 2, "offset": 99})).await;
        assert_eq!(
            past_the_end,
            json!([]),
            "an offset past the end is an empty page"
        );
    }

    /// "More like this": the source record must not reappear in its own results, and the
    /// nearest real neighbour must rank first.
    #[tokio::test]
    async fn search_similar_over_http_excludes_source_and_ranks_neighbour_first() {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "src", "vector": [1, 0, 0], "attrs": {}},
                    {"id": "near", "vector": [0.9, 0.1, 0.0], "attrs": {}},
                    {"id": "far", "vector": [0, 1, 0], "attrs": {}}
                ]}),
            ))
            .await
            .unwrap();
        let resp = app
            .oneshot(post(
                "/search/similar",
                json!({"collection": "docs", "id": "src", "top_k": 10}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        let ids: Vec<&str> = hits
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h["id"].as_str().unwrap())
            .collect();
        assert!(!ids.contains(&"src"), "source must not self-match: {ids:?}");
        assert_eq!(hits[0]["id"], "near");
    }

    /// A byte-identical duplicate is a real neighbour, not the source — it must still come
    /// back, and score ~1.0.
    #[tokio::test]
    async fn search_similar_over_http_keeps_a_true_duplicate() {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "src", "vector": [1, 0, 0], "attrs": {}},
                    {"id": "dup", "vector": [1, 0, 0], "attrs": {}}
                ]}),
            ))
            .await
            .unwrap();
        let resp = app
            .oneshot(post(
                "/search/similar",
                json!({"collection": "docs", "id": "src", "top_k": 10}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        assert_eq!(hits.as_array().unwrap().len(), 1);
        assert_eq!(hits[0]["id"], "dup");
        assert!((hits[0]["score"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    }

    /// A text-only source has no vector to search from; the store's `BAD_QUERY`-marked
    /// error must classify as a client fault, and the body must carry the reason.
    #[tokio::test]
    async fn search_similar_over_http_on_text_only_source_is_bad_request() {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [{"id": "t1", "attrs": {"kind": {"Str": "note"}}}]}),
            ))
            .await
            .unwrap();
        let resp = app
            .oneshot(post(
                "/search/similar",
                json!({"collection": "docs", "id": "t1"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let err = json_body(resp).await["error"].as_str().unwrap().to_string();
        assert!(err.contains("t1"), "{err}");
    }

    /// An id that names no record at all is a client fault too, and distinct in wording from
    /// the text-only case above.
    #[tokio::test]
    async fn search_similar_over_http_on_unknown_id_is_bad_request() {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [{"id": "src", "vector": [1, 0, 0], "attrs": {}}]}),
            ))
            .await
            .unwrap();
        let resp = app
            .oneshot(post(
                "/search/similar",
                json!({"collection": "docs", "id": "ghost"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let err = json_body(resp).await["error"].as_str().unwrap().to_string();
        assert!(err.contains("ghost"), "{err}");
    }

    /// Same ceiling as `/search`: `offset + top_k` past `MAX_TOP_K` is refused, not clamped.
    #[tokio::test]
    async fn search_similar_over_http_offset_past_max_top_k_is_bad_request() {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [{"id": "src", "vector": [1, 0, 0], "attrs": {}}]}),
            ))
            .await
            .unwrap();
        let resp = app
            .oneshot(post(
                "/search/similar",
                json!({"collection": "docs", "id": "src", "top_k": 10, "offset": MAX_TOP_K}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let err = json_body(resp).await["error"].as_str().unwrap().to_string();
        assert!(err.contains("offset"), "{err}");
    }

    /// An omitted `scope` must search only the source's own collection — the one place this
    /// route differs from `/search`, where an empty scope means every collection. Without this
    /// test the default could regress to `All` without anything going red.
    #[tokio::test]
    async fn search_similar_over_http_scope_defaults_to_source_collection() {
        let app = test_router(3);
        app.clone()
            .oneshot(post("/collections/a", json!({})))
            .await
            .unwrap();
        app.clone()
            .oneshot(post("/collections/b", json!({})))
            .await
            .unwrap();
        app.clone()
            .oneshot(post(
                "/collections/a/upsert",
                json!({"records": [{"id": "src", "vector": [1, 0, 0], "attrs": {}}]}),
            ))
            .await
            .unwrap();
        app.clone()
            .oneshot(post(
                "/collections/b/upsert",
                // Nearer than any neighbour in "a" would be, so a wrongly-`All` default
                // would surface it and this test would fail.
                json!({"records": [{"id": "nearer", "vector": [1, 0, 0], "attrs": {}}]}),
            ))
            .await
            .unwrap();

        let resp = app
            .clone()
            .oneshot(post(
                "/search/similar",
                json!({"collection": "a", "id": "src", "top_k": 10}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        assert_eq!(
            hits.as_array().unwrap().len(),
            0,
            "default scope must stay within \"a\", which has no other record: {hits:?}"
        );

        let resp = app
            .oneshot(post(
                "/search/similar",
                json!({"collection": "a", "id": "src", "top_k": 10, "scope": ["a", "b"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        assert_eq!(hits.as_array().unwrap().len(), 1);
        assert_eq!(hits[0]["collection"], "b");
        assert_eq!(hits[0]["id"], "nearer");
    }

    /// Projection is opt-in over the wire: an omitted pair is every attr, `include_attributes`
    /// narrows to the named ones, `exclude_attributes` drops them — on `/search` and `/list` alike.
    #[tokio::test]
    async fn projection_narrows_the_attrs_over_http() {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [{"id": "a", "vector": [1, 0, 0], "attrs": {
                    "title": {"Str": "t"}, "body": {"Str": "long"}, "lang": {"Str": "rust"}
                }}]}),
            ))
            .await
            .unwrap();
        let attrs = |path: &'static str, body: JsonValue| {
            let app = app.clone();
            async move {
                let resp = app.oneshot(post(path, body)).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                let hits = json_body(resp).await;
                hits[0]["attrs"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .cloned()
                    .collect::<Vec<String>>()
            }
        };
        let query = json!({"query": [1, 0, 0], "top_k": 1});
        assert_eq!(
            attrs("/search", query.clone()).await,
            vec!["body", "lang", "title"],
            "an omitted projection is every attr"
        );
        assert_eq!(
            attrs(
                "/search",
                json!({"query": [1, 0, 0], "top_k": 1, "include_attributes": ["title"]})
            )
            .await,
            vec!["title"]
        );
        assert_eq!(
            attrs(
                "/search",
                json!({"query": [1, 0, 0], "top_k": 1, "exclude_attributes": ["body"]})
            )
            .await,
            vec!["lang", "title"]
        );
        assert_eq!(
            attrs("/list", json!({"include_attributes": ["lang"]})).await,
            vec!["lang"]
        );
        assert_eq!(
            attrs("/list", json!({"exclude_attributes": ["body", "lang"]})).await,
            vec!["title"]
        );
    }

    /// Both projection lists in one body is a `400`, not a precedence rule (nidus-m50.15):
    /// honouring one of two contradictory instructions ships a payload nobody asked for.
    #[tokio::test]
    async fn both_projection_lists_at_once_is_a_bad_request() {
        let app = test_router(3);
        for (path, body) in [
            (
                "/search",
                json!({"query": [1, 0, 0], "include_attributes": ["a"], "exclude_attributes": ["b"]}),
            ),
            (
                "/list",
                json!({"include_attributes": ["a"], "exclude_attributes": ["b"]}),
            ),
        ] {
            let resp = app.clone().oneshot(post(path, body)).await.unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{path}");
            let err = json_body(resp).await["error"].as_str().unwrap().to_string();
            assert!(err.contains("mutually exclusive"), "{path}: {err}");
        }
    }

    /// `exact` is additive: omitting it is the store's configured path, and asking for it
    /// answers the same ranking (this store has no index, so both are the brute-force scan).
    #[tokio::test]
    async fn exact_is_an_additive_search_knob() {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "a", "vector": [1, 0, 0], "attrs": {}},
                    {"id": "b", "vector": [0, 1, 0], "attrs": {}}
                ]}),
            ))
            .await
            .unwrap();
        let hits = |body: JsonValue| {
            let app = app.clone();
            async move {
                let resp = app.oneshot(post("/search", body)).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                json_body(resp).await
            }
        };
        let implicit = hits(json!({"query": [1, 0, 0], "top_k": 2})).await;
        let forced = hits(json!({"query": [1, 0, 0], "top_k": 2, "exact": true})).await;
        assert_eq!(implicit, forced);
        assert_eq!(implicit[0]["id"], "a");
    }

    /// Two docs at the same base score, one a week older, in one file. Enough to drive decay,
    /// `limit_per`, `order_by`, and `/aggregate` over the wire.
    async fn ranked_router() -> Router {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "fresh", "vector": [1, 0, 0],
                     "attrs": {"ts": {"DateTime": 1_000_000_000_000i64},
                               "file": {"Str": "a.rs"}, "bytes": {"Int": 10}}},
                    {"id": "stale", "vector": [1, 0, 0],
                     "attrs": {"ts": {"DateTime": 999_395_200_000i64},
                               "file": {"Str": "a.rs"}, "bytes": {"Int": 32}}}
                ]}),
            ))
            .await
            .unwrap();
        app
    }

    #[tokio::test]
    async fn rank_by_decay_reorders_over_http() {
        let app = ranked_router().await;
        let body = json!({
            "query": [1, 0, 0], "top_k": 2,
            "rank_by": {"Decay": {"field": "ts", "origin": 1_000_000_000_000i64,
                                  "scale": 604_800_000i64, "lambda": 0.4}}
        });
        let resp = app.clone().oneshot(post("/search", body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        assert_eq!(hits[0]["id"], "fresh");
        assert_eq!(hits[1]["id"], "stale");
        // base 1.0 − 0.4 × (1 − 0.5) at exactly one half-life.
        let stale = hits[1]["score"].as_f64().unwrap();
        assert!((stale - 0.8).abs() < 1e-5, "{stale}");
    }

    #[tokio::test]
    async fn a_malformed_rank_by_is_a_bad_request() {
        let app = ranked_router().await;
        let body = json!({
            "query": [1, 0, 0],
            "rank_by": {"Decay": {"field": "ts", "origin": 0, "scale": 0}}
        });
        let resp = app.oneshot(post("/search", body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn limit_per_caps_hits_over_http() {
        let app = ranked_router().await;
        let body =
            json!({"query": [1, 0, 0], "top_k": 2, "limit_per": {"field": "file", "max": 1}});
        let resp = app.oneshot(post("/search", body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        assert_eq!(hits.as_array().unwrap().len(), 1, "both share one file");
    }

    /// A crowded corpus over the wire: three near-copies plus one outlier. Without the knob
    /// the page is two copies; with it the outlier takes slot 2.
    async fn crowded_router() -> Router {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "dup0", "vector": [1, 0.02, 0], "attrs": {"file": {"Str": "a.rs"}}},
                    {"id": "dup1", "vector": [1, 0.03, 0], "attrs": {"file": {"Str": "a.rs"}}},
                    {"id": "dup2", "vector": [1, 0.04, 0], "attrs": {"file": {"Str": "a.rs"}}},
                    {"id": "novel", "vector": [0.6, 0.8, 0], "attrs": {"file": {"Str": "b.rs"}}}
                ]}),
            ))
            .await
            .unwrap();
        app
    }

    #[tokio::test]
    async fn diversity_changes_the_page_over_http() {
        let app = crowded_router().await;
        let ids = |diversity: Option<f32>| {
            let app = app.clone();
            async move {
                let mut body = json!({"query": [1, 0, 0], "top_k": 2});
                if let Some(d) = diversity {
                    body["diversity"] = json!(d);
                }
                let resp = app.oneshot(post("/search", body)).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                let hits = json_body(resp).await;
                hits.as_array()
                    .unwrap()
                    .iter()
                    .map(|h| h["id"].as_str().unwrap().to_string())
                    .collect::<Vec<String>>()
            }
        };
        assert_eq!(ids(None).await, ["dup0", "dup1"]);
        assert_eq!(ids(Some(0.3)).await, ["dup0", "novel"]);
    }

    /// A lambda outside `[0, 1]` is a caller fault, so it must be a 400 rather than a 500.
    #[tokio::test]
    async fn a_malformed_diversity_is_a_bad_request() {
        let app = crowded_router().await;
        for bad in [-0.1, 1.5] {
            let body = json!({"query": [1, 0, 0], "diversity": bad});
            let resp = app.clone().oneshot(post("/search", body)).await.unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{bad}");
        }
    }

    #[tokio::test]
    async fn order_by_sorts_a_list_over_http() {
        let app = ranked_router().await;
        let ids = |descending: bool| {
            let app = app.clone();
            async move {
                let body = json!({"order_by": {"field": "bytes", "descending": descending}});
                let resp = app.oneshot(post("/list", body)).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                json_body(resp).await
            }
        };
        assert_eq!(ids(false).await[0]["id"], "fresh");
        assert_eq!(ids(true).await[0]["id"], "stale");
    }

    #[tokio::test]
    async fn aggregate_counts_and_sums_over_http() {
        let app = ranked_router().await;
        let resp = app
            .clone()
            .oneshot(post("/aggregate", json!({"sum": ["bytes"]})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = json_body(resp).await;
        assert_eq!(out["count"], 2);
        assert_eq!(out["sums"]["bytes"], json!({"Int": 42}));

        // A filter that matches nothing still answers, with a zero count.
        let body = json!({"filter": [{"Eq": ["file", {"Str": "nope"}]}], "sum": ["bytes"]});
        let resp = app.oneshot(post("/aggregate", body)).await.unwrap();
        let out = json_body(resp).await;
        assert_eq!(out["count"], 0);
        assert_eq!(out["sums"]["bytes"], json!({"Int": 0}));
    }

    /// A batch answers each query independently and in request order — the point of the
    /// endpoint is saving round-trips, not changing any single query's answer (nidus-m50.11).
    #[tokio::test]
    async fn a_batch_answers_every_query_in_order() {
        let app = ranked_router().await;
        let body = json!({"queries": [
            {"query": [1, 0, 0], "top_k": 1},
            {"query": [1, 0, 0], "top_k": 2, "filter": [{"Eq": ["bytes", {"Int": 32}]}]}
        ]});
        let resp = app
            .clone()
            .oneshot(post("/search/batch", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = json_body(resp).await;
        assert!(out.get("fused").is_none(), "unfused batch must not fuse");
        let results = out["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].as_array().unwrap().len(), 1);
        assert_eq!(results[1].as_array().unwrap().len(), 1);
        assert_eq!(results[1][0]["id"], "stale", "leg 2's filter still applies");

        // Each leg must match what the same query returns on its own.
        let solo = app
            .oneshot(post("/search", json!({"query": [1, 0, 0], "top_k": 1})))
            .await
            .unwrap();
        assert_eq!(json_body(solo).await[0]["id"], results[0][0]["id"]);
    }

    /// Fusing collapses the legs into one ranking, so a document both legs return appears once.
    #[tokio::test]
    async fn a_fused_batch_returns_one_merged_ranking() {
        let app = ranked_router().await;
        let body = json!({
            "queries": [{"query": [1, 0, 0], "top_k": 2}, {"query": [1, 0, 0], "top_k": 2}],
            "fuse": {"top_k": 10}
        });
        let resp = app.oneshot(post("/search/batch", body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = json_body(resp).await;
        assert!(out.get("results").is_none(), "fused batch returns one list");
        let fused = out["fused"].as_array().unwrap();
        assert_eq!(fused.len(), 2, "two documents, deduplicated across legs");
        let ids: Vec<&str> = fused.iter().map(|h| h["id"].as_str().unwrap()).collect();
        assert_eq!(
            ids.len(),
            std::collections::HashSet::<&&str>::from_iter(ids.iter()).len()
        );
    }

    /// The bounds that keep one request from buying unbounded scan, and the weights list that
    /// would otherwise silently re-weight the wrong leg. Each is a caller mistake, so each 400s.
    #[tokio::test]
    async fn a_batch_refuses_what_it_cannot_honour() {
        let app = ranked_router().await;
        let one = json!({"query": [1, 0, 0], "top_k": 1});
        let cases = [
            json!({"queries": []}),
            json!({"queries": (0..MAX_BATCH_QUERIES + 1).map(|_| one.clone()).collect::<Vec<_>>()}),
            json!({"queries": [one.clone(), one.clone()], "fuse": {"weights": [1.0]}}),
            json!({"queries": [one.clone()], "fuse": {"weights": [-1.0]}}),
            json!({"queries": [one.clone()], "fuse": {"top_k": MAX_TOP_K + 1}}),
            // A per-query page cap still applies inside a batch.
            json!({"queries": [{"query": [1, 0, 0], "top_k": MAX_TOP_K + 1}]}),
        ];
        for body in cases {
            let resp = app
                .clone()
                .oneshot(post("/search/batch", body.clone()))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "accepted {body}");
        }
    }

    /// A batch is rejected as a whole before ANY leg runs, so a bad leg cannot leave the store
    /// half-queried and the caller holding a partial answer they cannot tell apart.
    #[tokio::test]
    async fn one_bad_leg_rejects_the_whole_batch() {
        let app = ranked_router().await;
        let body = json!({"queries": [
            {"query": [1, 0, 0], "top_k": 1},
            {"query": [1, 0, 0], "include_attributes": ["a"], "exclude_attributes": ["b"]}
        ]});
        let resp = app.oneshot(post("/search/batch", body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// `group_by` adds per-value rows without disturbing the totals, and the two records here
    /// share one `file` — so the grouped answer is one row, not two (nidus-bmh).
    #[tokio::test]
    async fn aggregate_group_by_returns_per_value_rows() {
        let app = ranked_router().await;
        let body = json!({"sum": ["bytes"], "group_by": "file"});
        let resp = app.clone().oneshot(post("/aggregate", body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = json_body(resp).await;
        assert_eq!(out["count"], 2);
        assert_eq!(out["sums"]["bytes"], json!({"Int": 42}));
        assert_eq!(out["groups"].as_array().unwrap().len(), 1);
        assert_eq!(out["groups"][0]["value"], json!({"Str": "a.rs"}));
        assert_eq!(out["groups"][0]["count"], 2);
        assert_eq!(out["groups"][0]["sums"]["bytes"], json!({"Int": 42}));

        // An ungrouped request keeps the exact shape it had before grouping existed, so a
        // client written against the old response cannot trip over an empty `groups`.
        let resp = app
            .oneshot(post("/aggregate", json!({"sum": ["bytes"]})))
            .await
            .unwrap();
        let out = json_body(resp).await;
        assert!(out.get("groups").is_none(), "got {out}");
        assert!(out.get("groups_truncated").is_none(), "got {out}");
    }

    /// An empty `group_by` would put every record in the "missing" group and read as a working
    /// query, so it is a caller mistake — and a caller mistake is a 400, not a 500.
    #[tokio::test]
    async fn an_empty_group_by_is_a_bad_request() {
        let app = ranked_router().await;
        let body = json!({"sum": ["bytes"], "group_by": ""});
        let resp = app.oneshot(post("/aggregate", body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Declaring the filter index must actually take effect, so this asserts a subsequent
    /// query's results rather than the 200 — the route responding says nothing.
    #[tokio::test]
    async fn declaring_a_filter_index_takes_effect_and_changes_no_results() {
        for fields in [
            json!(["body"]),
            json!([{"field": "body", "trigrams": false}]),
        ] {
            let app = test_router(3);
            let resp = app
                .clone()
                .oneshot(post(
                    "/collections/docs/filter-index",
                    json!({ "fields": fields }),
                ))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);

            app.clone()
                .oneshot(post(
                    "/collections/docs/upsert",
                    json!({"records": [
                        {"id": "a", "vector": [1, 0, 0], "attrs": {"body": {"Str": "quantum physics"}}},
                        {"id": "b", "vector": [0, 1, 0], "attrs": {"body": {"Str": "classical optics"}}}
                    ]}),
                ))
                .await
                .unwrap();

            let resp = app
                .oneshot(post(
                    "/list",
                    json!({"scope": ["docs"], "filter": [
                        {"ContainsAllTokens": ["body", "quantum"]}
                    ]}),
                ))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let hits = json_body(resp).await;
            let ids: Vec<&str> = hits
                .as_array()
                .unwrap()
                .iter()
                .map(|h| h["id"].as_str().unwrap())
                .collect();
            assert_eq!(ids, ["a"], "fields = {fields}");
        }
    }

    /// Both structures off would index nothing while reading as indexed — a caller mistake,
    /// so a 400 rather than a silently inert declaration.
    #[tokio::test]
    async fn a_filter_index_field_indexing_nothing_is_a_bad_request() {
        let app = test_router(3);
        let resp = app
            .oneshot(post(
                "/collections/docs/filter-index",
                json!({"fields": [{"field": "body", "tokens": false, "trigrams": false}]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn hybrid_leg_weights_default_to_the_unweighted_fusion() {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/fts-schema",
                json!({"fields": ["body"]}),
            ))
            .await
            .unwrap();
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "v", "vector": [1, 0, 0], "attrs": {"body": {"Str": "nothing"}}},
                    {"id": "t", "attrs": {"body": {"Str": "quantum physics"}}}
                ]}),
            ))
            .await
            .unwrap();
        let fused = |body: JsonValue| {
            let app = app.clone();
            async move {
                let resp = app.oneshot(post("/hybrid-search", body)).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                json_body(resp).await
            }
        };
        let base = json!({"vector": [1, 0, 0], "field": "body", "text": "quantum", "top_k": 2});
        let mut explicit = base.clone();
        explicit["vector_weight"] = json!(1.0);
        explicit["text_weight"] = json!(1.0);
        assert_eq!(fused(base.clone()).await, fused(explicit).await);

        let mut heavy = base;
        heavy["vector_weight"] = json!(8.0);
        assert_eq!(fused(heavy).await[0]["id"], "v");
    }

    /// Before the store opens — a standby awaiting promotion — liveness must answer while
    /// readiness and every data route say `503`. Backwards, a failing liveness probe kills the
    /// instance meant to be waiting and a passing readiness one sends it unservable traffic.
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
        // Wait until every write reaches the queue, counted by the monotonic submitted count
        // rather than queue length — the length is 0 in exactly the case where coalescing worked
        // best. `sleep`, not `yield_now`: a spin loop keeps the CPU the workers need.
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
    #[tokio::test]
    async fn a_poisoned_store_lock_reports_unhealthy_so_liveness_restarts_it() {
        let (app, state) = router_and_state(3);
        let resp = app.clone().oneshot(get("/health")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "healthy to begin with");

        // Poison it the way a panicking write handler would: unwind holding the exclusive guard.
        // The panic hook is silenced meanwhile so a deliberate panic does not look like a test
        // failure; worst case a concurrent test's message is suppressed, and it still fails.
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

        // Readiness must agree, so the instance leaves the Service as well as being restarted;
        // otherwise traffic keeps arriving at something that can only 500. This assertion caught
        // that gap — making readiness lock-free initially made it blind to the poison flag.
        let resp = app.clone().oneshot(get("/ready")).await.unwrap();
        assert_ne!(resp.status(), StatusCode::OK);
    }

    /// A panic on the *read* path must NOT brick the instance — it does not poison the lock, so
    /// the failure mode is writer-only. If std ever changed here, the health check above would
    /// start firing on harmless search panics.
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

    /// `/versions` on a fresh in-memory store: a live commit, no pin, and the two
    /// nullable fields present as JSON `null` rather than absent.
    #[tokio::test]
    async fn versions_endpoint_reports_commit_and_no_pin() {
        let resp = test_router(3).oneshot(get("/versions")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert!(body["commit_version"].is_u64());
        assert!(body["pinned"].is_null());
        assert!(body["oldest_readable"].is_null());
        assert!(body["readable"].is_array());
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

    /// `POST /refresh` is routed and answers `adopted: false` when there is nothing to adopt —
    /// an in-memory store tracks no separate writer. Whether a cluster reader really takes up a
    /// writer's commits needs two processes, so that lives in `tests/e2e/cluster.rs`.
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

    /// The tuned form of the FTS schema body: per-field `k1`/`b`/analyzer, mixed with the
    /// bare-name form in one request, and visible in the ranking it produces.
    #[tokio::test]
    async fn fts_schema_accepts_per_field_tuning() {
        let app = test_router(3);
        let resp = app
            .clone()
            .oneshot(post(
                "/collections/docs/fts-schema",
                json!({"fields": ["title", {"field": "body", "b": 0.0, "ascii_folding": true}]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "short", "attrs": {"body": {"Str": "café"}}},
                    {"id": "long", "attrs": {"body": {"Str": "cafe cafe plus assorted padding words"}}}
                ]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Folding made both spellings one term, and `b = 0` lets the longer doc win.
        let resp = app
            .clone()
            .oneshot(post(
                "/text-search",
                json!({"field": "body", "query": "cafe", "top_k": 5}),
            ))
            .await
            .unwrap();
        let hits = json_body(resp).await;
        assert_eq!(hits.as_array().unwrap().len(), 2);
        assert_eq!(hits[0]["id"], "long");
    }

    /// An out-of-range BM25 parameter is a 400, not a store that scores nonsense forever.
    #[tokio::test]
    async fn fts_schema_rejects_an_impossible_parameter() {
        let resp = test_router(3)
            .oneshot(post(
                "/collections/docs/fts-schema",
                json!({"fields": [{"field": "body", "b": 4.0}]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// A store with `title` + `body` indexed and three docs, for the multi-clause tests.
    async fn two_field_router() -> Router {
        let app = test_router(3);
        assert_eq!(
            app.clone()
                .oneshot(post(
                    "/collections/docs/fts-schema",
                    json!({"fields": [{"field": "title", "b": 0.0}, {"field": "body", "b": 0.0}]}),
                ))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        let resp = app
            .clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "spread", "attrs": {"title": {"Str": "needle"}, "body": {"Str": "needle"}}},
                    {"id": "focused", "attrs": {"title": {"Str": "alpha"}, "body": {"Str": "needle needle needle needle"}}},
                    {"id": "filler", "attrs": {"title": {"Str": "needle"}, "body": {"Str": "gamma"}}}
                ]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        app
    }

    /// Several BM25 clauses over the wire, and `combine` changing the ranking (nidus-m50.10).
    #[tokio::test]
    async fn multi_clause_text_search_over_http() {
        let app = two_field_router().await;
        let query = |combine: &str| {
            json!({"clauses": [
                {"field": "title", "query": "needle"},
                {"field": "body", "query": "needle"}
            ], "combine": combine, "top_k": 5})
        };
        let top = |hits: &JsonValue| hits[0]["id"].as_str().unwrap().to_string();

        let resp = app
            .clone()
            .oneshot(post("/text-search", query("Sum")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(top(&json_body(resp).await), "spread");

        let resp = app
            .clone()
            .oneshot(post("/text-search", query("Max")))
            .await
            .unwrap();
        assert_eq!(top(&json_body(resp).await), "focused");
    }

    /// A body with no usable clause spelling is a `400`, never an empty result set.
    #[tokio::test]
    async fn an_unusable_text_query_body_is_a_400() {
        let app = two_field_router().await;
        for body in [
            json!({"clauses": []}),
            json!({"top_k": 5}),
            json!({"field": "body", "query": "x", "clauses": [{"field": "title", "query": "y"}]}),
        ] {
            for route in ["/text-search", "/hybrid-search"] {
                let mut body = body.clone();
                if route == "/hybrid-search" {
                    // The hybrid body spells the single form `field` + `text`.
                    let obj = body.as_object_mut().unwrap();
                    if let Some(q) = obj.remove("query") {
                        obj.insert("text".into(), q);
                    }
                    obj.insert("vector".into(), json!([1, 0, 0]));
                }
                let resp = app
                    .clone()
                    .oneshot(post(route, body.clone()))
                    .await
                    .unwrap();
                assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{route} {body}");
            }
        }
    }

    /// `explain` and `highlight` are opt-in, and the fragments index the stored text
    /// even when the projection dropped the field (nidus-m50.5).
    #[tokio::test]
    async fn annotations_are_opt_in_over_http() {
        let app = two_field_router().await;

        // Default: no `annotations` key at all.
        let resp = app
            .clone()
            .oneshot(post(
                "/text-search",
                json!({"field": "body", "query": "needle", "top_k": 5}),
            ))
            .await
            .unwrap();
        let hits = json_body(resp).await;
        assert!(hits[0].get("annotations").is_none());

        let resp = app
            .clone()
            .oneshot(post(
                "/text-search",
                json!({
                    "clauses": [{"field": "title", "query": "needle"}, {"field": "body", "query": "needle"}],
                    "top_k": 5, "explain": true, "highlight": {"fragment_chars": 40}
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        let a = &hits[0]["annotations"];
        assert_eq!(a["clauses"][0]["field"], "title");
        assert_eq!(a["highlights"][0]["field"], "title");
        assert_eq!(a["highlights"][0]["fragments"][0]["spans"][0][0], 0);

        // Hybrid reports each leg's own rank and score.
        let resp = app
            .clone()
            .oneshot(post(
                "/hybrid-search",
                json!({"vector": [1, 0, 0], "field": "body", "text": "needle",
                       "top_k": 5, "explain": true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        assert!(hits[0]["annotations"]["text"]["rank"].is_number());
    }

    // ── Backpressure (nidus-abx.2) ──────────────────────────────────────────

    /// A router whose admission control is already exhausted: `Limits::new(0, …)` hands out no
    /// permits, so every non-exempt request sheds — deterministic saturation that cannot flake.
    /// Not reachable by config, where `resolve_concurrency` reads `0` as "auto".
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

    /// Probes are never shed: they take no store lock, and shedding them under load would fail
    /// liveness and restart a busy-but-healthy instance — the trap nidus-abx.1/.3 closed one
    /// layer down. `/metrics` too, since an incident is when someone is looking.
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

    /// The startup warning is advisory and must never refuse: refusing a non-loopback bind would
    /// break every deployment terminating TLS at a proxy, the architecture the docs recommend.
    /// Asserts the function is total over every combination.
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
            // The read-path counterpart (nidus-c5v). Same substring on purpose, so the
            // search guard needed no new mapping here — which is also why it must keep
            // that wording: rephrase it and the status silently falls back to 500.
            (
                "query length 2 does not match store dimension 3",
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

    /// A wrong-dimension query is a `400` with a message that names both lengths, on every
    /// route that takes a query vector (nidus-c5v).
    #[tokio::test]
    async fn wrong_dimension_query_is_a_400_on_every_vector_route() {
        let app = test_router(3);
        app.clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [{"id": "a", "vector": [1, 0, 0], "attrs": {}}]}),
            ))
            .await
            .unwrap();

        // Too short, too long, and empty — over `/search` and the `/hybrid-search`
        // vector leg. `/list` and `/text-search` take no vector and are unaffected.
        let cases = [
            ("/search", json!({"query": [1, 0], "top_k": 5})),
            ("/search", json!({"query": [1, 0, 0, 9], "top_k": 5})),
            ("/search", json!({"query": [], "top_k": 5})),
            (
                "/hybrid-search",
                json!({"vector": [1, 0], "field": "body", "text": "x", "top_k": 5}),
            ),
            // top_k 0 short-circuits the fusion, but the verdict must not depend on it.
            (
                "/hybrid-search",
                json!({"vector": [1, 0], "field": "body", "text": "x", "top_k": 0}),
            ),
        ];
        for (path, body) in cases {
            let resp = app.clone().oneshot(post(path, body.clone())).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::BAD_REQUEST,
                "{path} with {body} should be refused, not answered"
            );
            let msg = json_body(resp).await["error"].as_str().unwrap().to_string();
            assert!(
                msg.contains("does not match store dimension"),
                "the error should say what is wrong: {msg}"
            );
        }

        // The right length is still served — the guard must not have broken the happy path.
        let resp = app
            .clone()
            .oneshot(post("/search", json!({"query": [1, 0, 0], "top_k": 5})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await.as_array().unwrap().len(), 1);
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
// Drive `/remember` + `/recall` offline against an in-process TCP mock — no provider network.
// Requires `embed-openai-compat`; every CI lane enabling `memory` also enables `embed-all`.
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
    pub(super) const EMBED_BODY: &str = r#"{"data":[{"embedding":[0.1,0.2,0.3],"index":0}]}"#;
    pub(super) const DIM: usize = 3;

    /// A multi-connection HTTP/1.1 mock: accepts forever on a background thread, drains each
    /// request, and replies with `EMBED_BODY`. Unlike the one-shot `embed::testutil` mock, it
    /// survives the several calls a remember→recall flow makes; `pub(super)` for `rerank_tests`.
    pub(super) fn spawn_embed_mock() -> String {
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

    /// `rollup` over `/recall`: the text-native knob collapses a chunked corpus to one hit
    /// per document and widens it, through the same `Rollup::as_opts` the library uses.
    #[tokio::test]
    async fn recall_with_rollup_collapses_and_widens_over_http() {
        let app = router_with_mock_embedder().await;
        // Seeded raw, because chunked writes have no HTTP route: `nidus ingest` and the Rust
        // `Memory::remember_chunked` are what produce these attrs.
        let chunk = |doc: &str, i: usize, start: usize, text: &str| {
            json!({"id": format!("{doc}#{i}"), "vector": vec![0.5f32; DIM], "attrs": {
                "nidus.parent_id": {"Str": doc},
                "nidus.chunk_index": {"Int": i},
                "nidus.char_start": {"Int": start},
                "nidus.text": {"Str": text}
            }})
        };
        let resp = app
            .clone()
            .oneshot(post(
                "/collections/notes/upsert",
                json!({"records": [
                    chunk("d1", 0, 0, "alpha beta"),
                    chunk("d1", 1, 6, "beta gamma"),
                    chunk("d2", 0, 0, "delta epsilon")
                ]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let hits = json_body(
            app.clone()
                .oneshot(post(
                    "/collections/notes/recall",
                    json!({"query": "beta", "top_k": 10, "rollup": {"neighbours": 1}}),
                ))
                .await
                .unwrap(),
        )
        .await;
        let hits = hits.as_array().unwrap();
        let parents: std::collections::BTreeSet<&str> = hits
            .iter()
            .map(|h| h["attrs"]["nidus.parent_id"]["Str"].as_str().unwrap())
            .collect();
        assert_eq!(hits.len(), parents.len(), "one hit per document: {hits:?}");
        let d1 = hits
            .iter()
            .find(|h| h["attrs"]["nidus.parent_id"]["Str"] == "d1")
            .expect("d1 present");
        assert_eq!(d1["context"], "alpha beta gamma");
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

    /// `/recall` carries the same TTL guard as the MCP read tools (#106): an expired
    /// entry is hidden, an entry that never got a TTL still surfaces (D5).
    #[tokio::test]
    async fn recall_hides_expired_entries_over_http() {
        let app = router_with_mock_embedder().await;

        for (id, ttl) in [("kept", json!(null)), ("gone", json!(0))] {
            let resp = app
                .clone()
                .oneshot(post(
                    "/collections/notes/remember",
                    json!({"id": id, "text": "same text", "ttl_seconds": ttl}),
                ))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }

        let resp = app
            .oneshot(post(
                "/collections/notes/recall",
                json!({"query": "same text", "top_k": 5}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let hits = json_body(resp).await;
        let ids: Vec<&str> = hits
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h["id"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&"kept"), "no-TTL entry must surface: {ids:?}");
        assert!(!ids.contains(&"gone"), "expired entry leaked: {ids:?}");
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

// ── Rerank-route tests (the `rerank` feature) ────────────────────────────────
// Drive `/search`, `/hybrid-search`, and `/collections/{name}/recall` against a real
// `AnyReranker::Voyage` over an in-process mock; gated like `memory_tests`, on one concrete provider.
#[cfg(all(test, feature = "rerank-voyage"))]
mod rerank_tests {
    use super::*;
    use crate::rerank::testutil::mock_once;
    use crate::rerank::{AnyReranker, RerankConfig, RerankProvider};
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    const DIM: usize = 3;

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

    /// A reranker mock that reports these three scores back to whatever candidate
    /// documents it is called with, in input order: metric-worst first.
    const INVERTING_SCORES: &str = r#"{"data":[{"index":0,"relevance_score":0.1},{"index":1,"relevance_score":0.5},{"index":2,"relevance_score":0.9}]}"#;

    /// A router whose `reranker` is a real `AnyReranker::Voyage` pointed at a one-shot
    /// mock that answers `resp_body` to the single rerank call each test issues.
    fn router_with_mock_reranker(resp_body: &str) -> Router {
        let server = mock_once(200, resp_body);
        let reranker = AnyReranker::build(
            RerankProvider::Voyage,
            RerankConfig::new("mock-rerank")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let db = Nidus::open_in_memory(DIM).unwrap();
        let state = AppState {
            reranker: Some(Arc::new(reranker)),
            ..test_state(Some(db))
        };
        router(state, 16 * 1024 * 1024)
    }

    /// A router with NO reranker configured — a rerank request must answer `400`.
    fn router_without_reranker() -> Router {
        let state = test_state(Some(Nidus::open_in_memory(DIM).unwrap()));
        router(state, 16 * 1024 * 1024)
    }

    /// Three orthogonal-vector docs so a query of `[1,0,0]` scores `a` highest and ties
    /// `b`/`c` at zero — the plain metric order is `a, b, c` (ties break on id).
    async fn upsert_three(app: &Router) {
        let resp = app
            .clone()
            .oneshot(post(
                "/collections/docs/upsert",
                json!({"records": [
                    {"id": "a", "vector": [1, 0, 0], "attrs": {"nidus.text": {"Str": "doc-a"}}},
                    {"id": "b", "vector": [0, 1, 0], "attrs": {"nidus.text": {"Str": "doc-b"}}},
                    {"id": "c", "vector": [0, 0, 1], "attrs": {"nidus.text": {"Str": "doc-c"}}}
                ]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    fn ids(hits: &JsonValue) -> Vec<String> {
        hits.as_array()
            .unwrap()
            .iter()
            .map(|h| h["id"].as_str().unwrap().to_string())
            .collect()
    }

    /// A rerank request with no reranker configured is a `400` naming the flag, not a
    /// `500` and not a silent unreranked pass-through.
    #[tokio::test]
    async fn rerank_without_a_configured_reranker_is_400() {
        let app = router_without_reranker();
        upsert_three(&app).await;
        let resp = app
            .oneshot(post(
                "/search",
                json!({"query": [1, 0, 0], "top_k": 3, "rerank": {"query": "q"}}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = json_body(resp).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("--rerank-provider"),
            "message names the flag: {body}"
        );
    }

    /// The reranked order must actually flip the metric baseline, not merely succeed —
    /// a `200` is true whether or not reranking ran.
    #[tokio::test]
    async fn rerank_changes_the_returned_order_over_http() {
        let baseline_app = router_without_reranker();
        upsert_three(&baseline_app).await;
        let resp = baseline_app
            .oneshot(post("/search", json!({"query": [1, 0, 0], "top_k": 3})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let baseline = ids(&json_body(resp).await);
        assert_eq!(baseline, vec!["a", "b", "c"]);

        let app = router_with_mock_reranker(INVERTING_SCORES);
        upsert_three(&app).await;
        let resp = app
            .oneshot(post(
                "/search",
                json!({"query": [1, 0, 0], "top_k": 3, "rerank": {"query": "best doc"}}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let reranked = ids(&json_body(resp).await);
        assert_ne!(
            reranked, baseline,
            "rerank must flip the order: {reranked:?}"
        );
        assert_eq!(reranked, vec!["c", "b", "a"]);
    }

    /// `include_attributes` that drops the text attr must not silently disable the rerank
    /// (nidus-d6z). Both halves: the order still flips, and the response still honours the
    /// projection. Asserting only one would pass a half-fix.
    #[tokio::test]
    async fn a_narrow_projection_still_reranks_over_http() {
        let app = router_with_mock_reranker(INVERTING_SCORES);
        upsert_three(&app).await;
        let resp = app
            .oneshot(post(
                "/search",
                json!({"query": [1, 0, 0], "top_k": 3,
                       "include_attributes": ["kind"],
                       "rerank": {"query": "best doc"}}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(ids(&body), vec!["c", "b", "a"], "rerank must still run");
        for hit in body.as_array().expect("hit array") {
            let attrs = hit["attrs"].as_object().expect("attrs object");
            assert!(
                !attrs.contains_key(crate::model::META_TEXT),
                "the forced text attr must not leak into the response: {attrs:?}"
            );
        }
    }

    /// No `rerank` field on the request body is byte-identical to the pre-rerank shape —
    /// the additive-wire claim.
    #[tokio::test]
    async fn a_request_without_a_rerank_field_is_unchanged() {
        let app = router_without_reranker();
        upsert_three(&app).await;
        let resp = app
            .oneshot(post("/search", json!({"query": [1, 0, 0], "top_k": 3})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(ids(&json_body(resp).await), vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn rerank_with_an_empty_query_is_400() {
        let app = router_without_reranker();
        upsert_three(&app).await;
        let resp = app
            .oneshot(post(
                "/search",
                json!({"query": [1, 0, 0], "top_k": 3, "rerank": {"query": ""}}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn overscan_zero_is_400() {
        let app = router_without_reranker();
        upsert_three(&app).await;
        let resp = app
            .oneshot(post(
                "/search",
                json!({"query": [1, 0, 0], "top_k": 3, "rerank": {"query": "q", "overscan": 0}}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Declare the `nidus.text` FTS field then upsert the three docs (mirrors
    /// `fts_and_hybrid_over_http`: the text leg needs a declared field to match against).
    async fn setup_hybrid_docs(app: &Router) {
        let resp = app
            .clone()
            .oneshot(post(
                "/collections/docs/fts-schema",
                json!({"fields": ["nidus.text"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        upsert_three(app).await;
    }

    fn hybrid_search_body(rerank: Option<JsonValue>) -> JsonValue {
        let mut body = json!({
            "vector": [1, 0, 0],
            "field": "nidus.text",
            "text": "doc",
            "top_k": 3,
        });
        if let Some(r) = rerank {
            body["rerank"] = r;
        }
        body
    }

    /// The same order-flip as `/search`, over `/hybrid-search`'s fused ranking. The mock always
    /// scores candidate position 0 lowest and position 2 highest, so — whatever the fused
    /// pre-rerank order actually is — the reranked order must be its exact reverse.
    #[tokio::test]
    async fn rerank_changes_the_returned_order_over_hybrid_search() {
        let baseline_app = router_without_reranker();
        setup_hybrid_docs(&baseline_app).await;
        let resp = baseline_app
            .oneshot(post("/hybrid-search", hybrid_search_body(None)))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let mut baseline = ids(&json_body(resp).await);
        assert_eq!(baseline.len(), 3);

        let app = router_with_mock_reranker(INVERTING_SCORES);
        setup_hybrid_docs(&app).await;
        let resp = app
            .oneshot(post(
                "/hybrid-search",
                hybrid_search_body(Some(json!({"query": "best doc"}))),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let reranked = ids(&json_body(resp).await);
        baseline.reverse();
        assert_eq!(
            reranked, baseline,
            "rerank must reverse the pre-rerank fused order"
        );
    }

    /// The same order-flip, over `/collections/{name}/recall` — the `memory` route whose
    /// `rerank.query` defaults to the recall query itself.
    #[cfg(all(feature = "memory", feature = "embed-openai-compat"))]
    #[tokio::test]
    async fn rerank_changes_the_returned_order_over_recall() {
        use crate::embed::{EmbedConfig, EmbedProvider};

        // Every text embeds to the same fixed vector (`super::memory_tests::EMBED_BODY`), so
        // the plain metric order is a dead tie broken on id: `a, b, c`.
        let embed_base = super::memory_tests::spawn_embed_mock();
        let embedder = AnyEmbedder::build(
            EmbedProvider::OpenAiCompat,
            EmbedConfig::new("mock-model").base_url(embed_base),
        )
        .await
        .expect("build mock embedder");
        let server = mock_once(200, INVERTING_SCORES);
        let reranker = AnyReranker::build(
            RerankProvider::Voyage,
            RerankConfig::new("mock-rerank")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let db = Nidus::open_in_memory(super::memory_tests::DIM).unwrap();
        let state = AppState {
            embedder: Some(Arc::new(embedder)),
            reranker: Some(Arc::new(reranker)),
            ..test_state(Some(db))
        };
        let app = router(state, 16 * 1024 * 1024);

        for (id, text) in [("a", "doc-a"), ("b", "doc-b"), ("c", "doc-c")] {
            let resp = app
                .clone()
                .oneshot(post(
                    "/collections/notes/remember",
                    json!({"id": id, "text": text}),
                ))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }

        let resp = app
            .oneshot(post(
                "/collections/notes/recall",
                json!({"query": "anything", "top_k": 3, "rerank": {}}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(ids(&json_body(resp).await), vec!["c", "b", "a"]);
    }

    fn text_search_body(query: JsonValue, rerank: Option<JsonValue>) -> JsonValue {
        let mut body = query;
        body["top_k"] = json!(3);
        if let Some(r) = rerank {
            body["rerank"] = r;
        }
        body
    }

    /// The same order-flip as `/search`, over `/text-search`'s BM25 ranking. The mock always
    /// scores candidate position 0 lowest and position 2 highest, so the reranked order must
    /// be the exact reverse of the baseline BM25 order.
    #[tokio::test]
    async fn rerank_changes_the_returned_order_over_text_search() {
        let baseline_app = router_without_reranker();
        setup_hybrid_docs(&baseline_app).await;
        let resp = baseline_app
            .oneshot(post(
                "/text-search",
                text_search_body(json!({"field": "nidus.text", "query": "doc"}), None),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let mut baseline = ids(&json_body(resp).await);
        assert_eq!(baseline.len(), 3);

        let app = router_with_mock_reranker(INVERTING_SCORES);
        setup_hybrid_docs(&app).await;
        let resp = app
            .oneshot(post(
                "/text-search",
                text_search_body(
                    json!({"field": "nidus.text", "query": "doc"}),
                    Some(json!({"query": "best doc"})),
                ),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let reranked = ids(&json_body(resp).await);
        baseline.reverse();
        assert_eq!(
            reranked, baseline,
            "rerank must reverse the baseline BM25 order"
        );
    }

    /// `{"rerank": {}}` on the single-field spelling back-fills `rerank.query` from the
    /// text `query` — asserted both by the order flip and by the mock actually receiving it.
    #[tokio::test]
    async fn text_search_rerank_query_defaults_to_the_text_query() {
        let server = mock_once(200, INVERTING_SCORES);
        let reranker = AnyReranker::build(
            RerankProvider::Voyage,
            RerankConfig::new("mock-rerank")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let db = Nidus::open_in_memory(DIM).unwrap();
        let state = AppState {
            reranker: Some(Arc::new(reranker)),
            ..test_state(Some(db))
        };
        let app = router(state, 16 * 1024 * 1024);
        setup_hybrid_docs(&app).await;

        let resp = app
            .oneshot(post(
                "/text-search",
                text_search_body(
                    json!({"field": "nidus.text", "query": "doc"}),
                    Some(json!({})),
                ),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(ids(&json_body(resp).await), vec!["c", "b", "a"]);
        let captured = server.captured();
        assert!(
            captured.body.contains("doc"),
            "the defaulted rerank query must be the text query, not empty: {}",
            captured.body
        );
    }

    /// The `clauses` spelling has no single natural text, so `{"rerank": {}}` there must
    /// `400` rather than silently return an un-reranked `200` (decision 1).
    #[tokio::test]
    async fn text_search_rerank_without_query_on_clauses_is_400() {
        let app = router_without_reranker();
        setup_hybrid_docs(&app).await;
        let resp = app
            .oneshot(post(
                "/text-search",
                text_search_body(
                    json!({"clauses": [{"field": "nidus.text", "query": "doc"}]}),
                    Some(json!({})),
                ),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// A `/text-search` rerank request with no reranker configured is a `400` naming the
    /// flag, mirroring the other three rerankable routes.
    #[tokio::test]
    async fn text_search_rerank_without_a_configured_reranker_is_400() {
        let app = router_without_reranker();
        setup_hybrid_docs(&app).await;
        let resp = app
            .oneshot(post(
                "/text-search",
                text_search_body(
                    json!({"field": "nidus.text", "query": "doc"}),
                    Some(json!({"query": "q"})),
                ),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = json_body(resp).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("--rerank-provider"),
            "message names the flag: {body}"
        );
    }
}
