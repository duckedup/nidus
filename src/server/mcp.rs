//! The MCP `2026-07-28` server surface (nidus-zm2), mounted at `/mcp`.
//!
//! nidus already ships the memory layer — `remember` text, `recall` the relevant bits,
//! embedding server-side — and names agent memory as a target use case. This module is
//! the front door for it: any MCP client (Claude Code, Claude Desktop, …) can use a
//! nidus store as agent memory with no glue code.
//!
//! **Why this arrives with `2026-07-28` and not earlier.** That revision removed
//! protocol-level sessions, the `initialize`/`initialized` handshake, the held-open HTTP
//! GET stream, and per-connection list results, and moved method/tool names into the
//! `Mcp-Method`/`Mcp-Name` headers. The result is stateless request/response, which is
//! what [`super::router`] already is. Pre-2.0 MCP wanted session affinity, and that would
//! have fought the cluster mode head-on: a standby writer answers `503` precisely so a
//! load balancer routes *around* it, and pinning clients to an instance is the opposite
//! of that. Because nidus had no MCP before this, it also starts clean — every feature
//! deprecated in the release (Roots, Sampling, Logging, HTTP+SSE, DCR) is one this
//! surface would never have implemented.
//!
//! **This is an adapter, never a second engine** (the binary-adapts-to-the-library rule).
//! Every tool below destructures its arguments and hands them to the same
//! [`run_read`](super::run_read)/[`run_write`](super::run_write) helpers the HTTP handlers
//! use, so locking, admission control, per-request deadlines, group commit, and error
//! mapping are shared rather than reimplemented. A tool that needed its own store access
//! path would be a bug.
//!
//! **The tool surface is text-native on purpose.** No model can emit a 384-float array as
//! a tool argument, so `POST /search` and `POST /hybrid-search`'s raw `vector` field have
//! no MCP equivalent. Instead `recall` and `hybrid_search` here take *text* and embed it
//! server-side with the configured embedder — the same thing the memory routes do. That
//! is why the `mcp` feature implies `memory`.
//!
//! **Schemas are hand-written JSON, not derived.** `schemars` rides along as an `rmcp`
//! dependency, but deriving would put the *field names* in charge of what the model sees.
//! Tool and parameter descriptions drive tool-selection quality more than anything else
//! in this file, so they are written by hand and reviewed as prose.

use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    model::{
        CacheScope, CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock,
        Implementation, ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerInfo,
        Tool,
    },
    service::RequestContext,
    transport::{
        StreamableHttpService, streamable_http_server::session::never::NeverSessionManager,
    },
};
use serde_json::{Map, Value as JsonValue, json};

// Both traits are imported for their methods on the `Any*` enums — `embed`/`embed_query`
// and `summarize` are trait methods, not inherent ones.
use crate::embed::Embedder;
#[cfg(feature = "summarize")]
use crate::summarize::Summarizer;

use super::{AppState, dto::HitDto};

/// How long a client may cache `tools/list` (SEP-2549).
///
/// The tool list is a compile-time constant — it changes when someone edits this file and
/// ships a new binary, never at runtime — so this is deliberately long. The matching
/// `cacheScope` is [`CacheScope::Public`] for the same reason: the list carries no
/// per-caller or per-store detail, so a shared intermediary caching one copy for every
/// client is correct rather than a leak.
const TOOLS_TTL_MS: u64 = 3_600_000;

/// Build the `/mcp` service.
///
/// Returned as a plain `tower` service so [`super::router`] can `nest_service` it *inside*
/// the middleware stack. That placement is the whole design: the MCP endpoint inherits the
/// body limit, backpressure, bearer-token auth, and metrics layers rather than growing its
/// own copy of each. In particular there is no MCP-specific authorization here — nidus's
/// existing constant-time bearer token guards this route exactly as it guards the others.
/// The spec does not require OAuth of a server, and for a store positioned at development
/// and small-scale use, bearer-plus-loopback is the honest security model; implementing
/// OAuth/CIMD would be a large surface bought for no one.
pub(super) fn service(
    state: AppState,
    max_body_bytes: usize,
) -> StreamableHttpService<NidusMcp, NeverSessionManager> {
    let mut config = rmcp::transport::streamable_http_server::StreamableHttpServerConfig::default();
    // `#[non_exhaustive]`, so these are assignments rather than a struct literal.
    //
    // Stateless. `2026-07-28` peers are served statelessly regardless of this flag
    // (SEP-2567 removed sessions); setting it false means we decline to keep server-side
    // state for *older* peers too, rather than quietly acquiring a session table this
    // store has no way to share across a cluster.
    config.legacy_session_mode = false;
    // Every tool here is a single request/response — the slowest is a brute-force scan, and
    // `compact` is deliberately not exposed — so none of them stream. Plain JSON keeps the
    // wire legible; rmcp still falls back to SSE if a handler ever emits a notification
    // first, so this is an optimisation, not a constraint.
    config.json_response = true;
    // Share the server's body limit rather than rmcp's own 4 MiB default, so one flag
    // (`--max-body-bytes`) governs every surface.
    config.max_request_body_bytes = max_body_bytes;
    // Empty = accept any `Host`. rmcp defaults to loopback-only as DNS-rebinding protection
    // for locally-run servers, which would reject every request to a nidus bound on a real
    // interface — i.e. the deployed case — and every request behind an ingress that rewrites
    // the header. What actually guards an exposed store here is the bearer token, and
    // `warn_on_exposure` already tells an operator when they have published one without it.
    // The e2e suite pins this (`host_header_survives_nesting`), because rmcp's own note at
    // `tower.rs:856` warns that `Router::nest` can drop the `Host` hyper synthesizes — so
    // leaving validation on would couple this endpoint to an axum nesting detail.
    config.allowed_hosts = Vec::new();

    StreamableHttpService::new(
        move || Ok(NidusMcp::new(state.clone())),
        Arc::new(NeverSessionManager::default()),
        config,
    )
}

/// The MCP handler over one open store.
///
/// Holds [`AppState`] and nothing else — no store handle, no lock, no cached tool list.
/// Cloning it is cloning the same `Arc`s every HTTP handler shares, which is what lets
/// `service` above mint one per request without cost.
#[derive(Clone)]
pub struct NidusMcp {
    state: AppState,
}

impl NidusMcp {
    fn new(state: AppState) -> Self {
        Self { state }
    }
}

/// A required string argument, or [`McpError::invalid_params`].
///
/// Note `-32602` rather than a bespoke code: `2026-07-28` renumbered the MCP-specific
/// errors into `-32020..=-32099` and left `invalid params` as plain JSON-RPC, so argument
/// validation belongs on the standard code.
fn required_str(args: &Map<String, JsonValue>, key: &str) -> Result<String, McpError> {
    match args.get(key) {
        Some(JsonValue::String(s)) if !s.trim().is_empty() => Ok(s.clone()),
        Some(JsonValue::String(_)) => Err(McpError::invalid_params(
            format!("`{key}` must not be empty"),
            None,
        )),
        Some(_) => Err(McpError::invalid_params(
            format!("`{key}` must be a string"),
            None,
        )),
        None => Err(McpError::invalid_params(
            format!("missing required argument `{key}`"),
            None,
        )),
    }
}

/// An optional positive integer argument.
fn optional_usize(args: &Map<String, JsonValue>, key: &str) -> Result<Option<usize>, McpError> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Number(n)) => n.as_u64().map(|v| Some(v as usize)).ok_or_else(|| {
            McpError::invalid_params(format!("`{key}` must be a positive integer"), None)
        }),
        Some(_) => Err(McpError::invalid_params(
            format!("`{key}` must be a number"),
            None,
        )),
    }
}

/// An optional float argument.
fn optional_f32(args: &Map<String, JsonValue>, key: &str) -> Result<Option<f32>, McpError> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Number(n)) => n
            .as_f64()
            .map(|v| Some(v as f32))
            .ok_or_else(|| McpError::invalid_params(format!("`{key}` must be a number"), None)),
        Some(_) => Err(McpError::invalid_params(
            format!("`{key}` must be a number"),
            None,
        )),
    }
}

/// Map a [`super::ApiError`] onto an MCP error.
///
/// Split by status rather than collapsing everything to one code, because the two halves
/// mean different things to a *model*. A `4xx` is something a retry can plausibly fix —
/// wrong collection name, a filter that does not parse — so it maps to `invalid_params`
/// and the agent gets to correct itself. A `5xx` is not, so it maps to `internal_error`;
/// reporting a server fault as bad arguments is what sends an agent into a retry loop,
/// rephrasing a request that was never the problem.
///
/// The one `4xx` that is genuinely *not* the model's fault — a memory tool on a server
/// started with no embedder — never reaches here: the tools below check for the embedder
/// up front and return `internal_error` themselves.
fn api_error(err: super::ApiError) -> McpError {
    let message = format!("{:#}", err.err);
    if err.status.is_client_error() {
        McpError::invalid_params(message, None)
    } else {
        McpError::internal_error(message, None)
    }
}

/// One tool definition. Hand-written schema — see the module note on why not derived.
fn tool(name: &'static str, description: &'static str, schema: JsonValue) -> Tool {
    let JsonValue::Object(schema) = schema else {
        // Every call site below is a `json!({...})` literal, so this is unreachable short
        // of an edit that changes one to a non-object.
        unreachable!("tool schema must be a JSON object");
    };
    Tool::new(name, description, Arc::new(schema))
}

/// The tool list, in a fixed order.
///
/// Order is deliberate and must stay stable: `2026-07-28` asks servers to return tools
/// deterministically so clients can cache the list and, more to the point, so an LLM's
/// prompt cache keeps hitting. Reordering this array invalidates every client's cached
/// prompt prefix, so add new tools at the end.
fn tools() -> Vec<Tool> {
    vec![
        tool(
            "remember",
            "Store a piece of text in long-term memory so it can be recalled later by \
             meaning. The text is embedded server-side — pass natural language, not \
             vectors. Use this to persist facts, decisions, preferences, and context \
             worth surviving beyond the current conversation.",
            json!({
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Which collection to store into, e.g. \"notes\". Collections are separate namespaces over one shared embedding space."
                    },
                    "text": {
                        "type": "string",
                        "description": "The natural-language text to remember."
                    },
                    "id": {
                        "type": "string",
                        "description": "Stable identifier. Remembering the same id again replaces the earlier entry — pass one to update, omit it for a new memory."
                    },
                    "summarize": {
                        "type": "boolean",
                        "description": "Summarize the text before embedding, storing the summary as what gets matched and keeping the original alongside it. Useful for long documents; needs the server to have been started with a summarizer."
                    }
                },
                "required": ["collection", "text"],
                "additionalProperties": false
            }),
        ),
        tool(
            "recall",
            "Search long-term memory by meaning and return the closest entries with \
             relevance scores. The query is embedded server-side — pass a natural-language \
             question, not vectors. This is semantic search: it finds entries that mean the \
             same thing as the query even when they share no words with it.",
            json!({
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Which collection to search."
                    },
                    "query": {
                        "type": "string",
                        "description": "A natural-language question or description of what you are looking for."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many results to return. Defaults to the server's configured value.",
                        "minimum": 1
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Drop results scoring below this. Scores are cosine similarity in [-1, 1]; around 0.7 is a reasonable floor for \"actually relevant\"."
                    }
                },
                "required": ["collection", "query"],
                "additionalProperties": false
            }),
        ),
        tool(
            "text_search",
            "Search memory by keyword (BM25 full-text), not by meaning. Use this when the \
             exact wording matters — an error string, an identifier, a proper noun — and \
             `recall` when the meaning matters. Requires a full-text schema on the field.",
            json!({
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Which collection to search."
                    },
                    "field": {
                        "type": "string",
                        "description": "The indexed attribute to search within."
                    },
                    "query": {
                        "type": "string",
                        "description": "Keywords to match."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many results to return.",
                        "minimum": 1
                    }
                },
                "required": ["collection", "field", "query"],
                "additionalProperties": false
            }),
        ),
        tool(
            "hybrid_search",
            "Search memory by meaning and keyword at once, fusing both rankings. Use this \
             when a query has both a semantic intent and a term that must appear — \
             \"the retry bug in the upsert path\". The text is embedded server-side.",
            json!({
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Which collection to search."
                    },
                    "field": {
                        "type": "string",
                        "description": "The indexed attribute to keyword-search within."
                    },
                    "query": {
                        "type": "string",
                        "description": "Natural language, used for BOTH the semantic and the keyword half of the search."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many results to return.",
                        "minimum": 1
                    }
                },
                "required": ["collection", "field", "query"],
                "additionalProperties": false
            }),
        ),
        tool(
            "list_collections",
            "List the collections in this store. Call this first if you do not already \
             know which collection to read from or write to.",
            json!({ "type": "object", "properties": {}, "additionalProperties": false }),
        ),
        tool(
            "stats",
            "Report the store's dimension, distance metric, collections, and memory \
             footprint. Diagnostic — useful for answering \"how much is in here\" and for \
             confirming the store is configured as expected.",
            json!({ "type": "object", "properties": {}, "additionalProperties": false }),
        ),
    ]
}

/// Render hits as compact JSON text.
///
/// Text rather than `structured_content`: the caller is a language model, and one readable
/// JSON block per result costs fewer tokens than the envelope a structured result implies
/// while staying just as parseable.
fn hits_content(hits: Vec<HitDto>) -> CallToolResult {
    if hits.is_empty() {
        // An explicit sentence, not `[]`. A model handed an empty array frequently retries
        // the identical query; told plainly that nothing matched, it moves on or broadens.
        return CallToolResult::success(vec![ContentBlock::text(
            "No matching entries in memory.".to_string(),
        )]);
    }
    let rendered = serde_json::to_string_pretty(&hits).unwrap_or_else(|_| "[]".to_string());
    CallToolResult::success(vec![ContentBlock::text(rendered)])
}

/// Derive a stable id for a memory whose caller did not supply one.
///
/// Content-addressed rather than random or clock-based, which makes `remember` idempotent:
/// remembering the same sentence twice replaces one entry instead of accumulating
/// near-duplicates that then compete for the same top-k slots. That is the right default
/// for an agent that cannot easily track which ids it has already used — and a caller who
/// *wants* two copies, or wants to update one later, passes an explicit `id`.
///
/// `DefaultHasher` is fixed-key (unlike `RandomState`), so the same text yields the same id
/// across processes and restarts — the property this depends on.
fn content_id(text: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    format!("mem-{:016x}", h.finish())
}

impl NidusMcp {
    /// The configured embedder, or the operator-facing error explaining its absence.
    ///
    /// `internal_error`, not `invalid_params`: nothing the model sends can conjure an
    /// embedder, so this must not read as a correctable mistake.
    fn embedder(&self) -> Result<Arc<crate::embed::AnyEmbedder>, McpError> {
        self.state.embedder.clone().ok_or_else(|| {
            McpError::internal_error(
                "this nidus server was started without an embedder, so it cannot embed text; \
                 restart it with --embed-provider … to enable remember/recall",
                None,
            )
        })
    }

    async fn remember(&self, args: &Map<String, JsonValue>) -> Result<CallToolResult, McpError> {
        let embedder = self.embedder()?;
        let collection = required_str(args, "collection")?;
        let text = required_str(args, "text")?;
        let id = match args.get("id") {
            Some(JsonValue::String(s)) if !s.trim().is_empty() => s.clone(),
            _ => content_id(&text),
        };
        let summarize = matches!(args.get("summarize"), Some(JsonValue::Bool(true)));

        // Mirrors the `/remember` handler: summarize (optional) then embed, both off-lock
        // network IO, and only the upsert takes the write guard.
        // `mut` only when a summarizer can stamp META_SUMMARY/META_SOURCE into it — the
        // same cfg dance the `/remember` handler does.
        #[cfg_attr(not(feature = "summarize"), allow(unused_mut))]
        let mut attrs = std::collections::BTreeMap::new();
        let embed_text = if summarize {
            #[cfg(feature = "summarize")]
            {
                let summarizer = self.state.summarizer.clone().ok_or_else(|| {
                    McpError::internal_error(
                        "this nidus server was started without a summarizer; restart it with \
                         --summarize-provider … or call remember without `summarize`",
                        None,
                    )
                })?;
                let summary = summarizer
                    .summarize(&text, &crate::summarize::SummarizeOpts::default())
                    .await
                    .map_err(|e| McpError::internal_error(e.to_string(), None))?;
                // Same attr keys the in-process `Memory` and the HTTP route stamp, so a
                // recall hit stays explainable back to its source text regardless of which
                // surface wrote it.
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
            #[cfg(not(feature = "summarize"))]
            {
                return Err(McpError::internal_error(
                    "this build has no summarizer support; call remember without `summarize`",
                    None,
                ));
            }
        } else {
            text
        };

        let vector = embedder
            .embed(&embed_text)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let name = collection.clone();
        let stored_id = id.clone();
        let n = super::run_write(self.state.clone(), move |db| {
            crate::memory::ensure_collection_and_pin(db, embedder.as_ref(), &name)?;
            db.upsert(&name, &[crate::Record::new(id, vector, attrs)])
        })
        .await
        .map_err(api_error)?;

        // Echo the id back: with a derived id the caller has no other way to learn it, and
        // it is what they need in order to update or delete this memory later.
        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Remembered {n} entry in `{collection}` with id `{stored_id}`."
        ))]))
    }

    async fn recall(&self, args: &Map<String, JsonValue>) -> Result<CallToolResult, McpError> {
        let embedder = self.embedder()?;
        let collection = required_str(args, "collection")?;
        let query = required_str(args, "query")?;
        let top_k = optional_usize(args, "top_k")?;
        let min_score = optional_f32(args, "min_score")?;

        let vector = embedder
            .embed_query(&query)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let opts = crate::SearchOpts {
            top_k: top_k.unwrap_or_else(super::dto::default_top_k),
            min_score,
            ..Default::default()
        };
        let hits = super::run_read(self.state.clone(), move |db| {
            // Same cross-model guard the HTTP route and the in-process `Memory` use:
            // recalling with a different embedder than the collection was written with
            // silently returns nonsense, so it is refused instead.
            crate::memory::guard_recall_identity(db, embedder.as_ref(), &collection)?;
            db.search(collection.as_str(), &vector, &opts)
        })
        .await
        .map_err(api_error)?;

        Ok(hits_content(hits.into_iter().map(HitDto::from).collect()))
    }

    async fn text_search(&self, args: &Map<String, JsonValue>) -> Result<CallToolResult, McpError> {
        let collection = required_str(args, "collection")?;
        let field = required_str(args, "field")?;
        let query = required_str(args, "query")?;
        let top_k = optional_usize(args, "top_k")?;

        let hits = super::run_read(self.state.clone(), move |db| {
            let opts = crate::SearchOpts {
                top_k: top_k.unwrap_or_else(super::dto::default_top_k),
                ..Default::default()
            };
            let q = crate::FtsQuery::new(field, query);
            db.text_search(crate::Scope::Collections(&[collection.as_str()]), &q, &opts)
        })
        .await
        .map_err(api_error)?;

        Ok(hits_content(hits.into_iter().map(HitDto::from).collect()))
    }

    async fn hybrid_search(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let embedder = self.embedder()?;
        let collection = required_str(args, "collection")?;
        let field = required_str(args, "field")?;
        let query = required_str(args, "query")?;
        let top_k = optional_usize(args, "top_k")?;

        // The one place this surface diverges from its HTTP counterpart: `POST
        // /hybrid-search` takes a caller-supplied `vector` alongside the text, which no
        // model can produce. Embedding the query text server-side gives the same fusion
        // from an argument an agent can actually write.
        let vector = embedder
            .embed_query(&query)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let hits = super::run_read(self.state.clone(), move |db| {
            // `rrf_k` and `candidates` stay at their defaults (60 / 100): they are fusion
            // tuning knobs with no meaning to a model, so exposing them as tool arguments
            // would add two ways to get worse results and none to get better ones.
            let opts = crate::HybridOpts {
                top_k: top_k.unwrap_or_else(super::dto::default_top_k),
                ..Default::default()
            };
            let q = crate::FtsQuery::new(field, query);
            db.hybrid_search(
                crate::Scope::Collections(&[collection.as_str()]),
                &vector,
                &q,
                &opts,
            )
        })
        .await
        .map_err(api_error)?;

        Ok(hits_content(hits.into_iter().map(HitDto::from).collect()))
    }

    async fn list_collections(&self) -> Result<CallToolResult, McpError> {
        let names = super::run_read(self.state.clone(), |db| Ok(db.collections()))
            .await
            .map_err(api_error)?;
        if names.is_empty() {
            return Ok(CallToolResult::success(vec![ContentBlock::text(
                "This store has no collections yet. `remember` creates one on first use."
                    .to_string(),
            )]));
        }
        Ok(CallToolResult::success(vec![ContentBlock::text(
            names.join("\n"),
        )]))
    }

    async fn stats(&self) -> Result<CallToolResult, McpError> {
        let body = super::run_read(self.state.clone(), |db| {
            Ok(json!({
                "dimension": db.dimension(),
                "distance": format!("{:?}", db.config().distance),
                "collections": db.collections(),
                "footprint": super::dto::FootprintDto::from(db.footprint()),
            }))
        })
        .await
        .map_err(api_error)?;
        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".to_string()),
        )]))
    }
}

impl ServerHandler for NidusMcp {
    fn get_info(&self) -> ServerInfo {
        // `#[non_exhaustive]`, so this is `new` plus builders rather than a struct literal.
        // Tools only — no resources, prompts, subscriptions, or tasks: every nidus
        // operation is a fast synchronous call, so there is nothing to subscribe to and
        // nothing long-running to hand back a task handle for.
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            // Named explicitly, NOT via `Implementation::from_build_env()`. That helper
            // reads `CARGO_PKG_*` as expanded inside *rmcp*, so it reports `rmcp 3.1.1` —
            // every client would think it had reached the SDK rather than this store. The
            // e2e suite pins this, because the wrong answer is perfectly well-formed and
            // would otherwise go unnoticed.
            .with_server_info(Implementation::new(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "nidus is a vector store used here as long-term memory. Use `remember` to \
                 store text worth keeping and `recall` to find it again by meaning. \
                 `text_search` matches exact wording instead, and `hybrid_search` does both \
                 at once. Call `list_collections` if you do not know which collection to \
                 use. Pass natural language throughout — never vectors.",
            )
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<rmcp::RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        // No pagination: six tools fit any client's budget, so a cursor would be pure
        // ceremony. `ttlMs`/`cacheScope` are required by SEP-2549 — see `TOOLS_TTL_MS`.
        Ok(ListToolsResult::with_all_items(tools())
            .with_ttl_ms(TOOLS_TTL_MS)
            .with_cache_scope(CacheScope::Public))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let args = request.arguments.unwrap_or_default();
        let result = match request.name.as_ref() {
            "remember" => self.remember(&args).await,
            "recall" => self.recall(&args).await,
            "text_search" => self.text_search(&args).await,
            "hybrid_search" => self.hybrid_search(&args).await,
            "list_collections" => self.list_collections().await,
            "stats" => self.stats().await,
            other => Err(McpError::invalid_params(
                format!("unknown tool `{other}`"),
                None,
            )),
        }?;
        // Always `Complete`. The other two variants of the MRTR envelope are
        // `InputRequired` — for a server that needs to ask the client something mid-call —
        // and `Task`, for work too long to hold a request open. Neither applies: every tool
        // here has all it needs in its arguments and returns in one round trip.
        Ok(CallToolResponse::Complete(result))
    }
}
