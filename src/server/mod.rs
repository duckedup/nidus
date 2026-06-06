//! `nidus serve` — a thin HTTP wrapper over one open [`Nidus`] (SPEC.md §9).
//!
//! The core stays an in-process, synchronous library; this module is the optional
//! server seam the SPEC anticipates — a separate wrapper, not a change to the core.
//! The store is held behind `Arc<Mutex<Nidus>>` and every operation runs on a
//! blocking task (`spawn_blocking`), the exact pattern the README/CLAUDE.md
//! prescribe for driving the synchronous store from async code: lock, run the
//! CPU/IO-bound op off the async executor, drop the lock — never held across an
//! `.await`. Endpoints map 1:1 to the public API.

pub mod dto;

use std::sync::{Arc, Mutex};

use anyhow::Context;
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value as JsonValue, json};
use tokio::net::TcpListener;

use crate::{Nidus, Record, Scope, SearchOpts};
use dto::{DeleteRequest, HitDto, ListRequest, SearchRequest, UpsertRequest};

/// Shared, cloneable handle to the one open store.
#[derive(Clone)]
struct AppState {
    db: Arc<Mutex<Nidus>>,
}

/// Open the store, bind `addr`, and serve until Ctrl-C; flush on shutdown.
pub async fn serve(db: Nidus, addr: &str) -> anyhow::Result<()> {
    let state = AppState {
        db: Arc::new(Mutex::new(db)),
    };
    let app = router(state.clone());

    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    eprintln!("nidus serving on http://{addr} (Ctrl-C to stop)");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;

    // Best-effort durability flush on a clean shutdown.
    if let Ok(mut db) = state.db.lock() {
        let _ = db.flush();
    }
    Ok(())
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/collections", get(list_collections))
        .route(
            "/collections/{name}",
            post(create_collection).delete(drop_collection),
        )
        .route("/collections/{name}/meta", get(get_meta).put(set_meta))
        .route("/collections/{name}/upsert", post(upsert))
        .route("/collections/{name}/delete", post(delete_records))
        .route("/collections/{name}/records", get(records))
        .route("/search", post(search))
        .route("/list", post(list))
        .route("/flush", post(flush))
        .route("/compact", post(compact))
        .with_state(state)
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

// ── Handlers ──────────────────────────────────────────────────────────────

async fn health() -> &'static str {
    "ok"
}

async fn list_collections(State(st): State<AppState>) -> Result<Json<Vec<String>>, ApiError> {
    let names = run(st, |db| Ok(db.collections())).await?;
    Ok(Json(names))
}

async fn create_collection(
    State(st): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<JsonValue>, ApiError> {
    let created = run(st, move |db| {
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
    let dropped = run(st, move |db| {
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
    let meta = run(st, move |db| Ok(db.get_meta(&name))).await?;
    Ok(Json(meta))
}

async fn set_meta(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(meta): Json<std::collections::BTreeMap<String, String>>,
) -> Result<Json<JsonValue>, ApiError> {
    run(st, move |db| db.set_meta(&name, meta)).await?;
    Ok(Json(json!({ "ok": true })))
}

async fn upsert(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let n = run(st, move |db| db.upsert(&name, &req.records)).await?;
    Ok(Json(json!({ "upserted": n })))
}

async fn delete_records(
    State(st): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<DeleteRequest>,
) -> Result<Json<JsonValue>, ApiError> {
    let n = run(st, move |db| match req.filter {
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
    let recs = run(st, move |db| Ok(db.get_all(&name))).await?;
    Ok(Json(recs))
}

async fn search(
    State(st): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Vec<HitDto>>, ApiError> {
    let hits = run(st, move |db| {
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
        let refs: Vec<&str> = scope.iter().map(String::as_str).collect();
        if refs.is_empty() {
            db.search(Scope::All, &query, &opts)
        } else {
            db.search(Scope::Collections(&refs), &query, &opts)
        }
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

async fn list(
    State(st): State<AppState>,
    Json(req): Json<ListRequest>,
) -> Result<Json<Vec<HitDto>>, ApiError> {
    let hits = run(st, move |db| {
        let ListRequest {
            scope,
            limit,
            filter,
        } = req;
        let refs: Vec<&str> = scope.iter().map(String::as_str).collect();
        if refs.is_empty() {
            db.list(Scope::All, &filter, limit)
        } else {
            db.list(Scope::Collections(&refs), &filter, limit)
        }
    })
    .await?;
    Ok(Json(hits.into_iter().map(HitDto::from).collect()))
}

async fn flush(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    run(st, |db| db.flush()).await?;
    Ok(Json(json!({ "ok": true })))
}

async fn compact(State(st): State<AppState>) -> Result<Json<JsonValue>, ApiError> {
    run(st, |db| db.compact()).await?;
    Ok(Json(json!({ "ok": true })))
}

/// Run a store operation on a blocking task, holding the lock only for its
/// duration. `&mut Nidus` covers both reads and writes (read methods take `&self`).
async fn run<F, T>(st: AppState, f: F) -> Result<T, ApiError>
where
    F: FnOnce(&mut Nidus) -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let mut db = st
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("store lock poisoned"))?;
        f(&mut db)
    })
    .await
    .map_err(|e| ApiError(anyhow::anyhow!("task join error: {e}")))?
    .map_err(ApiError)
}

// ── Error response ──────────────────────────────────────────────────────────

/// Any error from a handler becomes a `500` with a JSON `{ "error": … }` body.
struct ApiError(anyhow::Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("{:#}", self.0) })),
        )
            .into_response()
    }
}

impl<E> From<E> for ApiError
where
    E: Into<anyhow::Error>,
{
    fn from(e: E) -> Self {
        ApiError(e.into())
    }
}
