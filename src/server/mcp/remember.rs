//! The `remember` tool: embed text (optionally summarizing first), then upsert it.

use std::collections::BTreeMap;

use rmcp::{
    ErrorData as McpError,
    model::{CallToolResult, ContentBlock, Tool},
};
use serde_json::{Map, Value as JsonValue, json};

// Imported for its methods on `AnyEmbedder` — a trait method, not inherent.
use crate::embed::Embedder;
#[cfg(feature = "summarize")]
use crate::summarize::Summarizer;

use super::NidusMcp;
use super::args::{api_error, required_str, tool};

/// The `remember` tool definition.
pub(super) fn tools() -> Vec<Tool> {
    vec![tool(
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
                },
                "attrs": {
                    "type": "object",
                    "description": "Structured metadata stored alongside the text, so this memory can \
                                    later be found by filter as well as by meaning. Each value is \
                                    tagged by type, e.g. {\"project\": {\"Str\": \"nidus\"}, \"kind\": \
                                    {\"Str\": \"decision\"}, \"tags\": {\"List\": [\"mcp\", \"memory\"]}}. \
                                    Prefer stable keys you will filter on later — project, kind, path, \
                                    session.",
                    "additionalProperties": {
                        "oneOf": [
                            { "const": "Null" },
                            {
                                "type": "object",
                                "properties": { "Str": { "type": "string" } },
                                "required": ["Str"],
                                "additionalProperties": false
                            },
                            {
                                "type": "object",
                                "properties": { "Int": { "type": "integer" } },
                                "required": ["Int"],
                                "additionalProperties": false
                            },
                            {
                                "type": "object",
                                "properties": { "Bool": { "type": "boolean" } },
                                "required": ["Bool"],
                                "additionalProperties": false
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "List": { "type": "array", "items": { "type": "string" } }
                                },
                                "required": ["List"],
                                "additionalProperties": false
                            },
                            {
                                "type": "object",
                                "properties": { "Float": { "type": "number" } },
                                "required": ["Float"],
                                "additionalProperties": false
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "DateTime": {
                                        "type": "integer",
                                        "description": "UTC instant as epoch milliseconds."
                                    }
                                },
                                "required": ["DateTime"],
                                "additionalProperties": false
                            }
                        ]
                    }
                }
            },
            "required": ["collection", "text"],
            "additionalProperties": false
        }),
    )]
}

/// Parse the caller's `attrs` object into typed [`crate::Value`]s, via `Value`'s own
/// `Deserialize` — the wire shapes stay in lockstep with `value_json_spelling_is_stable`
/// rather than a hand-rolled variant match. A malformed map is a caller fault, not a server one.
fn parse_attrs(args: &Map<String, JsonValue>) -> Result<BTreeMap<String, crate::Value>, McpError> {
    match args.get("attrs") {
        None | Some(JsonValue::Null) => Ok(BTreeMap::new()),
        Some(JsonValue::Object(map)) => map
            .iter()
            .map(|(key, value)| {
                serde_json::from_value::<crate::Value>(value.clone())
                    .map(|v| (key.clone(), v))
                    .map_err(|e| {
                        McpError::invalid_params(
                            format!("`attrs.{key}` is not a valid value: {e}"),
                            None,
                        )
                    })
            })
            .collect(),
        Some(_) => Err(McpError::invalid_params("`attrs` must be an object", None)),
    }
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
    pub(super) async fn remember(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let embedder = self.embedder()?;
        let collection = required_str(args, "collection")?;
        let text = required_str(args, "text")?;
        let id = match args.get("id") {
            Some(JsonValue::String(s)) if !s.trim().is_empty() => s.clone(),
            _ => content_id(&text),
        };
        let summarize = matches!(args.get("summarize"), Some(JsonValue::Bool(true)));

        // As in `/remember`: caller attrs first, then (in summarize mode) `META_SUMMARY`/
        // `META_SOURCE` stamped in after, so the reserved `nidus.*` keys win a collision —
        // same merge order as the HTTP handler (`src/server/mod.rs`).
        #[cfg_attr(not(feature = "summarize"), allow(unused_mut))]
        let mut attrs = parse_attrs(args)?;
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
        let n = crate::server::run_write(self.state.clone(), move |db| {
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
}
