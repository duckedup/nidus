//! The MCP `2026-07-28` surface (nidus-zm2) at `/mcp` — an adapter over the memory layer,
//! routing through the same `run_read`/`run_write` helpers as the HTTP handlers. Two
//! constraints: no tool takes a raw `vector`, and schemas are hand-written, not derived.

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

// Imported for their methods on the `Any*` enums — both are trait methods, not inherent.
use crate::embed::Embedder;
#[cfg(feature = "summarize")]
use crate::summarize::Summarizer;

use super::{AppState, dto::HitDto};

/// How long a client may cache `tools/list` (SEP-2549). Long, and `Public` below, because
/// the list is a compile-time constant carrying no per-caller detail.
const TOOLS_TTL_MS: u64 = 3_600_000;

/// Build the `/mcp` service — a plain `tower` service so [`super::router`] can
/// `nest_service` it *inside* the middleware stack, inheriting auth and the rest. No
/// MCP-specific authorization: the spec does not require OAuth of a server.
pub(super) fn service(
    state: AppState,
    max_body_bytes: usize,
) -> StreamableHttpService<NidusMcp, NeverSessionManager> {
    // `#[non_exhaustive]`, hence assignments rather than a struct literal.
    let mut config = rmcp::transport::streamable_http_server::StreamableHttpServerConfig::default();
    // 2026-07-28 peers are stateless regardless; false declines a session table for older
    // peers too, which this store could not share across a cluster anyway.
    config.legacy_session_mode = false;
    // Every tool is one request/response, so nothing streams. rmcp still falls back to SSE
    // if a handler emits a notification first.
    config.json_response = true;
    // One `--max-body-bytes` governs every surface, rather than rmcp's own 4 MiB default.
    config.max_request_body_bytes = max_body_bytes;
    // Empty = accept any `Host`. rmcp defaults to loopback-only (DNS-rebinding protection),
    // which would reject a store bound on a real interface or behind a Host-rewriting
    // ingress; the bearer token is what actually guards those. Pinned by the e2e suite.
    config.allowed_hosts = Vec::new();

    StreamableHttpService::new(
        move || Ok(NidusMcp::new(state.clone())),
        Arc::new(NeverSessionManager::default()),
        config,
    )
}

/// The MCP handler over one open store. Holds [`AppState`] and nothing else, so minting one
/// per request costs only an `Arc` clone.
#[derive(Clone)]
pub struct NidusMcp {
    state: AppState,
}

impl NidusMcp {
    fn new(state: AppState) -> Self {
        Self { state }
    }
}

/// A required non-empty string argument.
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

/// `top_k`, defaulted and bounded by [`super::dto::MAX_TOP_K`], so an absurd value is a
/// tool-argument error here rather than an allocation the store has to survive.
fn optional_top_k(args: &Map<String, JsonValue>) -> Result<usize, McpError> {
    let k = optional_usize(args, "top_k")?.unwrap_or_else(super::dto::default_top_k);
    if k > super::dto::MAX_TOP_K {
        return Err(McpError::invalid_params(
            format!("`top_k` must not exceed {}", super::dto::MAX_TOP_K),
            None,
        ));
    }
    Ok(k)
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

/// Map a [`super::ApiError`] onto an MCP error, split by status: a `4xx` is worth a retry, a
/// `5xx` is not. Reporting a server fault as bad arguments causes rephrase-and-retry loops.
fn api_error(err: super::ApiError) -> McpError {
    let message = format!("{:#}", err.err);
    if err.status.is_client_error() {
        McpError::invalid_params(message, None)
    } else {
        McpError::internal_error(message, None)
    }
}

/// One tool definition, with a hand-written schema.
fn tool(name: &'static str, description: &'static str, schema: JsonValue) -> Tool {
    let JsonValue::Object(schema) = schema else {
        unreachable!("tool schema must be a JSON object");
    };
    Tool::new(name, description, Arc::new(schema))
}

/// The tool list. Order must stay stable — reordering invalidates every client's cached
/// prompt prefix (SEP-2549) — so add new tools at the end.
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

/// Render hits as JSON text.
fn hits_content(hits: Vec<HitDto>) -> CallToolResult {
    if hits.is_empty() {
        // A sentence, not `[]`: a model handed an empty array tends to retry the identical
        // query, where a plain statement makes it broaden or move on.
        return CallToolResult::success(vec![ContentBlock::text(
            "No matching entries in memory.".to_string(),
        )]);
    }
    let rendered = serde_json::to_string_pretty(&hits).unwrap_or_else(|_| "[]".to_string());
    CallToolResult::success(vec![ContentBlock::text(rendered)])
}

/// A stable id for a memory whose caller supplied none. Content-addressing makes `remember`
/// idempotent rather than accumulating near-duplicates that compete for the same top-k
/// slots; `DefaultHasher` is fixed-key, so the id survives restarts.
fn content_id(text: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    format!("mem-{:016x}", h.finish())
}

impl NidusMcp {
    /// The configured embedder, or an `internal_error` — nothing the model sends can
    /// conjure one, so this must not read as a correctable mistake.
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

        // As in `/remember`: summarize then embed off-lock, and only the upsert takes the
        // write guard. `mut` only when a summarizer can stamp attrs into it.
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
                // The same keys `Memory` and the HTTP route stamp, so a hit stays traceable
                // to its source text whichever surface wrote it.
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

        // Echoed back because a derived id is otherwise unknowable to the caller, who needs
        // it to update or delete this memory later.
        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Remembered {n} entry in `{collection}` with id `{stored_id}`."
        ))]))
    }

    async fn recall(&self, args: &Map<String, JsonValue>) -> Result<CallToolResult, McpError> {
        let embedder = self.embedder()?;
        let collection = required_str(args, "collection")?;
        let query = required_str(args, "query")?;
        let top_k = optional_top_k(args)?;
        let min_score = optional_f32(args, "min_score")?;

        let vector = embedder
            .embed_query(&query)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let opts = crate::SearchOpts {
            top_k,
            min_score,
            ..Default::default()
        };
        let hits = super::run_read(self.state.clone(), move |db| {
            // Recalling with a different embedder than wrote the collection returns
            // nonsense, so the same guard the HTTP route uses refuses it.
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
        let top_k = optional_top_k(args)?;

        let hits = super::run_read(self.state.clone(), move |db| {
            let opts = crate::SearchOpts {
                top_k,
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
        let top_k = optional_top_k(args)?;

        // The one divergence from `POST /hybrid-search`, which takes a caller-supplied
        // `vector`: embedding the query text gives the same fusion from an argument a model
        // can actually write.
        let vector = embedder
            .embed_query(&query)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let hits = super::run_read(self.state.clone(), move |db| {
            // `rrf_k`/`candidates` stay default: fusion knobs mean nothing to a model, so
            // exposing them adds ways to get worse results and none to get better ones.
            let opts = crate::HybridOpts {
                top_k,
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
        // Tools only: every nidus op is one fast synchronous call, so there is nothing to
        // subscribe to and nothing long-running to hand back a task handle for.
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            // NOT `Implementation::from_build_env()` — that reads `CARGO_PKG_*` as expanded
            // inside rmcp, reporting `rmcp 3.1.1` instead of this store. Pinned by e2e.
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
        // Unpaginated: six tools fit any client's budget.
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
        // Always `Complete`: the `InputRequired` and `Task` variants of the MRTR envelope
        // never apply, since every tool has what it needs and returns in one round trip.
        Ok(CallToolResponse::Complete(result))
    }
}
