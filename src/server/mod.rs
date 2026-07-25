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

pub mod dto;

use std::sync::{Arc, RwLock};

use anyhow::Context;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Request, State},
    http::{StatusCode, header::AUTHORIZATION},
    middleware::{self, Next},
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
    /// Readiness fails past this much reader staleness (`Config::max_staleness`), copied
    /// here so a probe never has to reach into the store's config behind the lock.
    max_staleness: Option<std::time::Duration>,
    token: Option<Arc<str>>,
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
    let state = AppState {
        db: Arc::new(RwLock::new(None)),
        open: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        max_staleness: cfg.max_staleness,
        token: cfg.token.map(Arc::from),
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
    eprintln!("nidus serving on http://{bound} (Ctrl-C / SIGTERM to stop){auth_note}");

    // Open on a blocking task; a failure (not a wait) asks the server to stop, and is
    // re-raised after `axum::serve` returns so the process exits non-zero.
    let open_failed = Arc::new(RwLock::new(None::<anyhow::Error>));
    let abort = Arc::new(tokio::sync::Notify::new());
    let slot = state.db.clone();
    let open_flag = state.open.clone();
    let failure_slot = open_failed.clone();
    let abort_tx = abort.clone();
    tokio::task::spawn_blocking(move || match open() {
        Ok(db) => {
            if let Ok(mut slot) = slot.write() {
                *slot = Some(db);
                // Publish only after the store is in place, so a probe never sees
                // `ready` before a request could actually be served.
                open_flag.store(true, std::sync::atomic::Ordering::Release);
                eprintln!("nidus store open — serving requests");
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
                        // Losing the lease here is the same fencing signal a write would
                        // hit; the write path latches it, and readiness reports it.
                        eprintln!("nidus: background lease renewal failed: {e:#}");
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
                        eprintln!("nidus: scheduled refresh failed: {e:#}");
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

fn router(state: AppState, max_body_bytes: usize) -> Router {
    let router = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
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

    router
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .with_state(state)
}

/// Reject any request lacking a valid `Authorization: Bearer <token>` when a
/// token is configured. The probe endpoints are always open so liveness and readiness
/// checks need no credential — an orchestrator would otherwise read `401` as "not ready"
/// and never route to a perfectly healthy instance. A no-op when the server is
/// unauthenticated.
async fn auth(State(st): State<AppState>, req: Request, next: Next) -> Response {
    if let Some(expected) = &st.token
        && !matches!(req.uri().path(), "/health" | "/ready")
    {
        let presented = req
            .headers()
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        if presented != Some(expected.as_ref()) {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "missing or invalid bearer token" })),
            )
                .into_response();
        }
    }
    next.run(req).await
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
async fn health() -> &'static str {
    "ok"
}

/// Readiness: this instance has a store open and can serve requests.
///
/// `503` while a standby waits for the writer handle, so a load balancer routes around it
/// instead of sending requests that would all answer `503` anyway. Split from
/// [`health`] because the two genuinely differ for a standby: live, but not ready.
///
/// Note this currently reports only whether the store is *open*. A writer that has been
/// **fenced**, or a reader that is arbitrarily **stale**, still reports ready — see
/// `nidus-lp4.1`, which owns those.
async fn ready(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    if !st.open.load(std::sync::atomic::Ordering::Acquire) {
        return Err(ApiError::from(not_open()));
    }
    // Beyond "is a store open", readiness asks whether this instance can serve *usefully*.
    // Both of these are cheap in-RAM reads (see `ClusterStatus`), so a probe every few
    // seconds costs nothing and never touches the object store.
    let status = read_status(&st)?;
    if status.fenced {
        return Err(ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            err: anyhow::anyhow!(
                "writer fenced: this instance was superseded and every write will fail — \
                 it must be replaced"
            ),
        });
    }
    if let Some(max) = st.max_staleness
        && status.staleness_secs > max.as_secs()
    {
        return Err(ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            err: anyhow::anyhow!(
                "stale: last verified current {}s ago, beyond the {}s bound — this reader \
                 is not being refreshed",
                status.staleness_secs,
                max.as_secs()
            ),
        });
    }
    Ok(Json(json!({
        "ready": true,
        "role": format!("{:?}", status.role),
        "staleness_secs": status.staleness_secs,
    })))
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
    tokio::task::spawn_blocking(move || {
        let db = st
            .db
            .read()
            .map_err(|_| anyhow::anyhow!("store lock poisoned"))?;
        let db = db.as_ref().ok_or_else(not_open)?;
        f(db)
    })
    .await
    .map_err(|e| ApiError::internal(anyhow::anyhow!("task join error: {e}")))?
    .map_err(ApiError::from)
}

/// Run a **write** operation on a blocking task under the exclusive lock.
async fn run_write<F, T>(st: AppState, f: F) -> Result<T, ApiError>
where
    F: FnOnce(&mut Nidus) -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let mut db = st
            .db
            .write()
            .map_err(|_| anyhow::anyhow!("store lock poisoned"))?;
        let db = db.as_mut().ok_or_else(not_open)?;
        f(db)
    })
    .await
    .map_err(|e| ApiError::internal(anyhow::anyhow!("task join error: {e}")))?
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

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    /// Build a router over a fresh in-memory store of the given dimension.
    fn test_router(dim: usize) -> Router {
        let db = Nidus::open_in_memory(dim).unwrap();
        router_over(Some(db))
    }

    /// Build a router over an optional store — `None` models an instance whose store is
    /// not open yet (a standby waiting for promotion).
    fn router_over(db: Option<Nidus>) -> Router {
        let open = db.is_some();
        let state = AppState {
            db: Arc::new(RwLock::new(db)),
            open: Arc::new(std::sync::atomic::AtomicBool::new(open)),
            max_staleness: None,
            token: None,
            #[cfg(feature = "memory")]
            embedder: None,
            #[cfg(all(feature = "memory", feature = "summarize"))]
            summarizer: None,
        };
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
            db: Arc::new(RwLock::new(Some(db))),
            open: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            // Zero tolerance: anything with nonzero staleness would fail.
            max_staleness: Some(std::time::Duration::ZERO),
            token: None,
            #[cfg(feature = "memory")]
            embedder: None,
            #[cfg(all(feature = "memory", feature = "summarize"))]
            summarizer: None,
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
            db: Arc::new(RwLock::new(Some(db))),
            open: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            max_staleness: None,
            token: Some(Arc::from("s3cret")),
            #[cfg(feature = "memory")]
            embedder: None,
            #[cfg(all(feature = "memory", feature = "summarize"))]
            summarizer: None,
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
            db: Arc::new(RwLock::new(db)),
            token: None,
            embedder: Some(Arc::new(embedder)),
            #[cfg(all(feature = "memory", feature = "summarize"))]
            summarizer: None,
        };
        router(state, 16 * 1024 * 1024)
    }

    /// A router with NO embedder configured — memory routes must answer `400`.
    fn router_without_embedder() -> Router {
        let db = Nidus::open_in_memory(DIM).unwrap();
        let state = AppState {
            db: Arc::new(RwLock::new(db)),
            token: None,
            embedder: None,
            #[cfg(all(feature = "memory", feature = "summarize"))]
            summarizer: None,
        };
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
