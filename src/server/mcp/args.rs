//! Argument parsing and error-mapping helpers shared by every tool handler. Frozen after
//! nidus-k28 unit A for those original helpers; `optional_rerank` (nidus-4ss) is the one
//! addition since the `rerank` arg is shared by all three search tools same as these.

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

/// The optional `rerank` argument: `{ text_field?, overscan?, model? }`. No `query` — the
/// tool's own query/text argument stands in for it (see `search.rs`'s `rerank_schema`).
#[cfg(feature = "rerank")]
pub(super) fn optional_rerank(
    args: &Map<String, JsonValue>,
) -> Result<Option<crate::RerankOpts>, McpError> {
    let obj = match args.get("rerank") {
        None | Some(JsonValue::Null) => return Ok(None),
        Some(JsonValue::Object(obj)) => obj,
        Some(_) => return Err(McpError::invalid_params("`rerank` must be an object", None)),
    };
    let text_field = match obj.get("text_field") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::String(s)) => Some(s.clone()),
        Some(_) => {
            return Err(McpError::invalid_params(
                "`rerank.text_field` must be a string",
                None,
            ));
        }
    };
    let overscan = match obj.get("overscan") {
        None | Some(JsonValue::Null) => crate::DEFAULT_OVERSCAN,
        Some(JsonValue::Number(n)) => n.as_u64().ok_or_else(|| {
            McpError::invalid_params("`rerank.overscan` must be a positive integer", None)
        })? as usize,
        Some(_) => {
            return Err(McpError::invalid_params(
                "`rerank.overscan` must be an integer",
                None,
            ));
        }
    };
    // Refused, not clamped: the over-fetch drives one paid provider call per candidate, so an
    // absurd value is a tool-argument error rather than a bill the caller never asked for.
    if overscan > crate::MAX_OVERSCAN {
        return Err(McpError::invalid_params(
            format!(
                "`rerank.overscan` {overscan} exceeds the maximum of {}",
                crate::MAX_OVERSCAN
            ),
            None,
        ));
    }
    let model = match obj.get("model") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::String(s)) => Some(s.clone()),
        Some(_) => {
            return Err(McpError::invalid_params(
                "`rerank.model` must be a string",
                None,
            ));
        }
    };
    Ok(Some(crate::RerankOpts {
        text_field,
        overscan,
        model,
        query: None,
    }))
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

#[cfg(all(test, feature = "rerank"))]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: JsonValue) -> Map<String, JsonValue> {
        match v {
            JsonValue::Object(m) => m,
            other => panic!("expected a JSON object, got {other:?}"),
        }
    }

    #[test]
    fn optional_rerank_absent_or_null_is_none() {
        assert!(optional_rerank(&obj(json!({}))).unwrap().is_none());
        assert!(
            optional_rerank(&obj(json!({"rerank": null})))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn optional_rerank_defaults_overscan_and_carries_no_query() {
        let rr = optional_rerank(&obj(json!({"rerank": {"text_field": "body"}})))
            .unwrap()
            .unwrap();
        assert_eq!(rr.text_field.as_deref(), Some("body"));
        assert_eq!(rr.overscan, crate::DEFAULT_OVERSCAN);
        assert!(rr.model.is_none());
        assert!(rr.query.is_none());
    }

    #[test]
    fn optional_rerank_honours_overscan_and_model() {
        let rr = optional_rerank(&obj(json!({"rerank": {"overscan": 8, "model": "m"}})))
            .unwrap()
            .unwrap();
        assert_eq!(rr.overscan, 8);
        assert_eq!(rr.model.as_deref(), Some("m"));
    }

    #[test]
    fn optional_rerank_rejects_a_non_object() {
        assert!(optional_rerank(&obj(json!({"rerank": "nope"}))).is_err());
    }

    #[test]
    fn optional_rerank_rejects_a_non_integer_overscan() {
        assert!(optional_rerank(&obj(json!({"rerank": {"overscan": "many"}}))).is_err());
    }

    /// Refused, not clamped — the schema's `maximum` is advisory to a well-behaved client, so
    /// the parser has to enforce it against one that ignores the schema.
    #[test]
    fn optional_rerank_rejects_an_overscan_past_the_ceiling() {
        let over = json!({"rerank": {"overscan": crate::MAX_OVERSCAN + 1}});
        let err = optional_rerank(&obj(over)).unwrap_err();
        assert!(format!("{err:?}").contains("overscan"), "{err:?}");
        assert!(
            optional_rerank(&obj(json!({"rerank": {"overscan": crate::MAX_OVERSCAN}}))).is_ok()
        );
    }
}
