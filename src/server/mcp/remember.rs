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
use super::args::{api_error, optional_f32, optional_usize, required_str, tool};

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
                },
                "ttl_seconds": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional time-to-live. Once this many seconds have elapsed, the entry is excluded from recall/search/browse results (checked at read time, not reclaimed from disk immediately). Omit for a memory that never expires."
                },
                "dedupe_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Opt-in near-duplicate suppression: if an existing entry in this collection scores at or above this cosine similarity to the new text, update that entry in place instead of inserting a competing one. Attrs are merged, not replaced — fields already on the matched entry that this call omits survive, and its created_at carries forward; only overlapping keys are overwritten. Omit to always insert/replace by id as usual."
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
        let ttl_seconds = optional_usize(args, "ttl_seconds")?.map(|s| s as i64);
        let dedupe_threshold = optional_f32(args, "dedupe_threshold")?;

        // `META_SUMMARY` is the only reserved key this surface stamps; `META_TEXT` and the
        // recency attrs are stamped by `commit_remember`, so every surface gets them.
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
                attrs.insert(
                    crate::memory::META_SUMMARY.to_string(),
                    crate::Value::Str(summary.clone()),
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
            text.clone()
        };

        let vector = embedder
            .embed(&embed_text)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        // The dedup search, the read-back, and the upsert all run inside this one write
        // closure — atomic against every other queued write with no new plumbing (D8).
        let name = collection.clone();
        let write = crate::memory::RememberWrite {
            id,
            text,
            attrs,
            ttl_seconds,
            dedupe_threshold,
        };
        let written = crate::server::run_write(self.state.clone(), move |db| {
            crate::memory::commit_remember(db, embedder.as_ref(), &name, write, vector)
        })
        .await
        .map_err(api_error)?;

        // The model reads this back and acts on it: it must say which happened and the
        // resolved id, because a derived id (or a dedupe match) is otherwise unknowable to
        // the caller who needs it to update or forget this memory later.
        let verb = if written.deduped {
            "Updated an existing near-duplicate entry"
        } else {
            "Remembered a new entry"
        };
        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "{verb} ({} write) in `{collection}` with id `{}`.",
            written.upserted, written.id
        ))]))
    }
}
