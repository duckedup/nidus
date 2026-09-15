//! Argument parsing and error-mapping helpers shared by every tool handler. FROZEN after
//! nidus-k28 unit A: later units add their own parsers to their own files, not here — except
//! `namespace` (nidus-pcpc.2) below, genuinely shared by every tool that touches a store.

use std::sync::Arc;

use rmcp::{ErrorData as McpError, model::Tool};
use serde_json::{Map, Value as JsonValue, json};

/// How long a client may cache `tools/list` (SEP-2549). Long, and `Public` in `mod.rs`,
/// because the list is a compile-time constant carrying no per-caller detail.
pub(super) const TOOLS_TTL_MS: u64 = 3_600_000;

/// A required non-empty string argument.
pub(super) fn required_str(args: &Map<String, JsonValue>, key: &str) -> Result<String, McpError> {
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
pub(super) fn optional_usize(
    args: &Map<String, JsonValue>,
    key: &str,
) -> Result<Option<usize>, McpError> {
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

/// `top_k`, defaulted and bounded by [`crate::server::dto::MAX_TOP_K`], so an absurd value is a
/// tool-argument error here rather than an allocation the store has to survive.
pub(super) fn optional_top_k(args: &Map<String, JsonValue>) -> Result<usize, McpError> {
    let k = optional_usize(args, "top_k")?.unwrap_or_else(crate::server::dto::default_top_k);
    if k > crate::server::dto::MAX_TOP_K {
        return Err(McpError::invalid_params(
            format!("`top_k` must not exceed {}", crate::server::dto::MAX_TOP_K),
            None,
        ));
    }
    Ok(k)
}

/// An optional boolean argument, defaulting to `false` when absent.
pub(super) fn optional_bool(args: &Map<String, JsonValue>, key: &str) -> Result<bool, McpError> {
    optional_bool_or(args, key, false)
}

/// An optional boolean argument with a caller-chosen default, for gates that are on by default.
pub(super) fn optional_bool_or(
    args: &Map<String, JsonValue>,
    key: &str,
    default: bool,
) -> Result<bool, McpError> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(default),
        Some(JsonValue::Bool(b)) => Ok(*b),
        Some(_) => Err(McpError::invalid_params(
            format!("`{key}` must be a boolean"),
            None,
        )),
    }
}

/// An optional float argument.
pub(super) fn optional_f32(
    args: &Map<String, JsonValue>,
    key: &str,
) -> Result<Option<f32>, McpError> {
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

/// An optional array of strings, e.g. named-vector selection (nidus-85t). Type-checked only;
/// semantic checks (empty, duplicate) are the store's job, so the error names the same rule
/// regardless of which surface sent the query.
pub(super) fn optional_string_array(
    args: &Map<String, JsonValue>,
    key: &str,
) -> Result<Vec<String>, McpError> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(Vec::new()),
        Some(JsonValue::Array(items)) => items
            .iter()
            .map(|v| match v {
                JsonValue::String(s) => Ok(s.clone()),
                _ => Err(McpError::invalid_params(
                    format!("`{key}` items must be strings"),
                    None,
                )),
            })
            .collect(),
        Some(_) => Err(McpError::invalid_params(
            format!("`{key}` must be an array of strings"),
            None,
        )),
    }
}

/// An optional `namespace` argument (nidus-pcpc.2): which store this one call addresses,
/// when the server is running in namespaced mode. Type-checked only; [`super::NidusMcp::
/// with_namespace`] resolves it against the server's actual addressing mode.
pub(super) fn optional_namespace(
    args: &Map<String, JsonValue>,
) -> Result<Option<String>, McpError> {
    match args.get("namespace") {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::String(s)) if !s.trim().is_empty() => Ok(Some(s.clone())),
        Some(JsonValue::String(_)) => Err(McpError::invalid_params(
            "`namespace` must not be empty",
            None,
        )),
        Some(_) => Err(McpError::invalid_params(
            "`namespace` must be a string",
            None,
        )),
    }
}

/// The `namespace` property, spliced into every tool schema that touches a store
/// (nidus-pcpc.2). Meaningless — refused, not ignored — outside namespaced mode.
pub(super) fn namespace_schema() -> JsonValue {
    json!({
        "type": "string",
        "description": "Which namespace this call addresses. Only applies to a server \
            started in namespaced mode (many independent stores behind one process); \
            omitting it there falls back to the namespace this connection was opened \
            against (a client connected at `/ns/{namespace}/mcp` rather than bare `/mcp`), \
            and an explicit value here overrides that. Supplying it against a server \
            started with a single `--dir` (one store, no namespaces) is refused rather \
            than silently ignored."
    })
}

/// Map a [`crate::server::ApiError`] onto an MCP error, split by status: a `4xx` is worth a
/// retry, a `5xx` is not. Reporting a server fault as bad arguments causes rephrase-and-retry loops.
pub(super) fn api_error(err: crate::server::ApiError) -> McpError {
    let message = format!("{:#}", err.err);
    if err.status.is_client_error() {
        McpError::invalid_params(message, None)
    } else {
        McpError::internal_error(message, None)
    }
}

/// One tool definition, with a hand-written schema.
pub(super) fn tool(name: &'static str, description: &'static str, schema: JsonValue) -> Tool {
    let JsonValue::Object(schema) = schema else {
        unreachable!("tool schema must be a JSON object");
    };
    Tool::new(name, description, Arc::new(schema))
}
