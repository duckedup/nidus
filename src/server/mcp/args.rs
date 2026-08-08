//! Argument parsing and error-mapping helpers shared by every tool handler. FROZEN after
//! nidus-k28 unit A: later units add their own parsers to their own files, not here.

use std::sync::Arc;

use rmcp::{ErrorData as McpError, model::Tool};
use serde_json::{Map, Value as JsonValue};

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
