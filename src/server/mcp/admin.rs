//! `list_collections` and `stats` — store-wide introspection, no arguments.

use rmcp::{
    ErrorData as McpError,
    model::{CallToolResult, ContentBlock, Tool},
};
use serde_json::json;

use super::NidusMcp;
use super::args::{api_error, tool};

pub(super) fn tools() -> Vec<Tool> {
    vec![
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

impl NidusMcp {
    pub(super) async fn list_collections(&self) -> Result<CallToolResult, McpError> {
        let names = crate::server::run_read(self.state.clone(), |db| Ok(db.collections()))
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

    pub(super) async fn stats(&self) -> Result<CallToolResult, McpError> {
        let body = crate::server::run_read(self.state.clone(), |db| {
            Ok(json!({
                "dimension": db.dimension(),
                "distance": format!("{:?}", db.config().distance),
                "collections": db.collections(),
                "footprint": crate::server::dto::FootprintDto::from(db.footprint()),
            }))
        })
        .await
        .map_err(api_error)?;
        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".to_string()),
        )]))
    }
}
