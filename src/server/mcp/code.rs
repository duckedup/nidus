//! `code_search` (nidus-3gm unit 5): grouped hits over an AST-chunked corpus, driven
//! through the same `run_read` helper every other read tool uses. Text-native like its
//! siblings — no vector argument in, and the result is a file/symbol grouping that
//! structurally carries no source body either (mirrors `POST /code-search`).

use rmcp::{
    ErrorData as McpError,
    model::{CallToolResult, ContentBlock, Tool},
};
use serde_json::{Map, Value as JsonValue, json};

use crate::embed::Embedder;

use super::NidusMcp;
use super::args::{api_error, optional_usize, required_str, tool};
use super::search::{filter_defs, filter_schema, parse_filter};

/// The optional `semantic` argument (named to avoid `vector`, a boolean flag being no such
/// thing): `Some(true/false)` forces a ranking, `None` defers to the store — the same
/// choice `CodeSearchRequest::vector` makes over HTTP.
fn optional_tri_bool(args: &Map<String, JsonValue>, key: &str) -> Result<Option<bool>, McpError> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(McpError::invalid_params(
            format!("`{key}` must be a boolean"),
            None,
        )),
    }
}

pub(super) fn tools() -> Vec<Tool> {
    vec![tool(
        "code_search",
        "Search an AST-chunked code or docs corpus and get results grouped by file, each \
         with its matching symbols — name, kind, and line span. Never returns a raw \
         embedding or the source body; read the file at the given lines for the real code. \
         `semantic` picks meaning-based vs keyword ranking; omit it to let the server \
         decide from what the collection holds (keyword-only when it holds no embeddings).",
        json!({
            "$defs": filter_defs(),
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Which collection to search."
                },
                "query": {
                    "type": "string",
                    "description": "What to search for — a natural-language description for semantic search, or exact keywords for keyword search."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum files to return, ranked by their best-matching symbol.",
                    "minimum": 1
                },
                "filter": filter_schema(),
                "semantic": {
                    "type": "boolean",
                    "description": "Force meaning-based (true) or keyword (false) ranking. Omit to let the server decide."
                }
            },
            "required": ["collection", "query"],
            "additionalProperties": false
        }),
    )]
}

impl NidusMcp {
    pub(super) async fn code_search(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let collection = required_str(args, "collection")?;
        let query = required_str(args, "query")?;
        let limit =
            optional_usize(args, "limit")?.unwrap_or_else(crate::server::dto::default_top_k);
        if limit > crate::server::dto::MAX_TOP_K {
            return Err(McpError::invalid_params(
                format!("`limit` must not exceed {}", crate::server::dto::MAX_TOP_K),
                None,
            ));
        }
        let filter = parse_filter(args)?.unwrap_or_default();
        let want_semantic = optional_tri_bool(args, "semantic")?;

        let use_vector = match want_semantic {
            Some(v) => v,
            None => crate::server::run_read(self.state.clone(), |db| Ok(db.dimension() > 0))
                .await
                .map_err(api_error)?,
        };

        let hits = if use_vector {
            let embedder = self.embedder()?;
            let vector = embedder
                .embed_query(&query)
                .await
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;
            let opts = crate::SearchOpts {
                top_k: limit,
                filter,
                ..Default::default()
            };
            crate::server::run_read(self.state.clone(), move |db| {
                db.search(collection.as_str(), &vector, &opts)
            })
            .await
            .map_err(api_error)?
        } else {
            let opts = crate::SearchOpts {
                top_k: limit,
                filter,
                ..Default::default()
            };
            let q = crate::FtsQuery {
                clauses: vec![crate::FtsClause::new(crate::model::META_TEXT, query)],
                combine: crate::FtsCombine::default(),
                highlight: None,
            };
            crate::server::run_read(self.state.clone(), move |db| {
                db.text_search(collection.as_str(), &q, &opts)
            })
            .await
            .map_err(api_error)?
        };

        Ok(code_search_content(&hits))
    }
}

/// Render code-search hits as pretty JSON text: file-grouped, symbol/kind/line span only.
/// Reuses [`crate::server::code_search_response`] so this tool and `POST /code-search`
/// cannot drift on what a code hit looks like.
fn code_search_content(hits: &[crate::Hit]) -> CallToolResult {
    let response = crate::server::code_search_response(hits);
    if response.files.is_empty() {
        // A sentence, not `[]`: a model handed an empty array tends to retry the identical
        // query, where a plain statement makes it broaden or move on.
        return CallToolResult::success(vec![ContentBlock::text(
            "No matching code found.".to_string(),
        )]);
    }
    let rendered =
        serde_json::to_string_pretty(&response).unwrap_or_else(|_| "{\"files\":[]}".to_string());
    CallToolResult::success(vec![ContentBlock::text(rendered)])
}
