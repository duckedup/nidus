//! The MCP `2026-07-28` surface (nidus-zm2) at `/mcp` — an adapter over the memory layer,
//! routing through the same `run_read`/`run_write` helpers as the HTTP handlers. Two
//! constraints: no tool takes a raw `vector`, and schemas are hand-written, not derived.

use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    model::{
        CacheScope, CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock,
        GetPromptRequestParams, GetPromptResponse, Implementation, ListPromptsResult,
        ListResourceTemplatesResult, ListResourcesResult, ListToolsResult, PaginatedRequestParams,
        ReadResourceRequestParams, ReadResourceResponse, ReadResourceResult, ServerCapabilities,
        ServerInfo, Tool,
    },
    service::RequestContext,
    transport::{
        StreamableHttpService, streamable_http_server::session::never::NeverSessionManager,
    },
};

use super::{AppState, dto::HitDto};

mod admin;
mod args;
mod hygiene;
mod prompts;
mod remember;
mod resources;
mod search;
mod stdio;
mod uri;

use args::TOOLS_TTL_MS;

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

/// Speak MCP over stdio instead of nesting inside the HTTP stack — the seam
/// [`super::serve_stdio`] hands an already-open [`AppState`] to.
pub(super) async fn serve_stdio(state: AppState) -> anyhow::Result<()> {
    stdio::serve(state).await
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
}

/// The tool list. Order must stay stable — reordering invalidates every client's cached
/// prompt prefix (SEP-2549) — so add new tools at the end.
fn tools() -> Vec<Tool> {
    let mut v = remember::tools();
    v.extend(search::tools());
    v.extend(admin::tools());
    v.extend(hygiene::tools());
    v
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

impl ServerHandler for NidusMcp {
    fn get_info(&self) -> ServerInfo {
        // Tools, resources, and prompts: every nidus op is one fast synchronous call, so
        // there is nothing to subscribe to and nothing long-running to hand back a task
        // handle for — subscriptions and tasks stay out for that same reason.
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .enable_prompts()
                .build(),
        )
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
                 use, or `browse` to see what a collection already holds before adding to \
                 it. Use `get` to check a specific id, and `forget` to correct or remove a \
                 memory that turned out wrong. Pass natural language throughout — never \
                 vectors. Memories are also addressable as `nidus://` resources, and \
                 `recall_then_answer` is a prompt that runs the recall for you.",
        )
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<rmcp::RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        // Unpaginated: nine tools fit any client's budget.
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
            "forget" => self.forget(&args).await,
            "get" => self.get(&args).await,
            "browse" => self.browse(&args).await,
            other => Err(McpError::invalid_params(
                format!("unknown tool `{other}`"),
                None,
            )),
        }?;
        // Always `Complete`: the `InputRequired` and `Task` variants of the MRTR envelope
        // never apply, since every tool has what it needs and returns in one round trip.
        Ok(CallToolResponse::Complete(result))
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<rmcp::RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(
            ListResourcesResult::with_all_items(self.list_resources().await?)
                .with_ttl_ms(resources::RESOURCES_TTL_MS)
                .with_cache_scope(CacheScope::Public),
        )
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<rmcp::RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(
            ListResourceTemplatesResult::with_all_items(resources::templates())
                .with_ttl_ms(resources::RESOURCES_TTL_MS)
                .with_cache_scope(CacheScope::Public),
        )
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<rmcp::RoleServer>,
    ) -> Result<ReadResourceResponse, McpError> {
        let contents = self.read_resource(&request.uri).await?;
        // Always `Complete`, for the same reason `call_tool` is: a resource read here
        // never needs a client round trip to finish.
        Ok(ReadResourceResponse::Complete(ReadResourceResult::new(
            contents,
        )))
    }

    async fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<rmcp::RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        Ok(ListPromptsResult::with_all_items(prompts::prompts())
            .with_ttl_ms(prompts::PROMPTS_TTL_MS)
            .with_cache_scope(CacheScope::Public))
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        _context: RequestContext<rmcp::RoleServer>,
    ) -> Result<GetPromptResponse, McpError> {
        let args = request.arguments.unwrap_or_default();
        let result = match request.name.as_ref() {
            "recall_then_answer" => self.get_prompt(&args).await,
            other => Err(McpError::invalid_params(
                format!("unknown prompt `{other}`"),
                None,
            )),
        }?;
        // Always `Complete`, for the same reason `call_tool` is.
        Ok(GetPromptResponse::Complete(result))
    }
}
