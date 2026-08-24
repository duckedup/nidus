//! `list_collections` and `stats` — store-wide introspection, no arguments.

use rmcp::{
    ErrorData as McpError,
    model::{CallToolResult, ContentBlock, Tool},
};
use serde_json::{Map, Value as JsonValue, json};

use super::NidusMcp;
use super::args::{api_error, required_str, tool};

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

/// The alias lifecycle tools — registered **last** in [`super::tools`], after
/// `related`, so they never shift the position of a tool that predates them (SEP-2549).
pub(super) fn alias_tools() -> Vec<Tool> {
    vec![
        tool(
            "list_aliases",
            "List every collection alias in this store, as alias -> concrete collection. A \
             collection name you were given elsewhere may actually be an indirect alias; \
             call this to see what it really resolves to.",
            json!({ "type": "object", "properties": {}, "additionalProperties": false }),
        ),
        tool(
            "set_alias",
            "Create a new alias or repoint an existing one to a different collection, in one \
             atomic call. `target` must already exist as a concrete collection — aliases \
             resolve in a single hop and never chain, so `target` may not itself be an \
             alias.",
            json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "The alias name to create or repoint." },
                    "target": { "type": "string", "description": "The existing concrete collection the alias should point at." }
                },
                "required": ["name", "target"],
                "additionalProperties": false
            }),
        ),
        tool(
            "drop_alias",
            "Remove an alias by name. This only removes the indirect name — it does not \
             delete the collection it pointed at or any of its records.",
            json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "The alias name to remove." }
                },
                "required": ["name"],
                "additionalProperties": false
            }),
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

    pub(super) async fn list_aliases(&self) -> Result<CallToolResult, McpError> {
        let aliases = crate::server::run_read(self.state.clone(), |db| Ok(db.aliases()))
            .await
            .map_err(api_error)?;
        if aliases.is_empty() {
            return Ok(CallToolResult::success(vec![ContentBlock::text(
                "This store has no aliases. Every collection name resolves to itself.".to_string(),
            )]));
        }
        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&aliases).unwrap_or_else(|_| "{}".to_string()),
        )]))
    }

    pub(super) async fn set_alias(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let name = required_str(args, "name")?;
        let target = required_str(args, "target")?;
        let (name, target) = crate::server::run_write(self.state.clone(), move |db| {
            db.set_alias(&name, &target)?;
            Ok((name, target))
        })
        .await
        .map_err(api_error)?;
        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Alias `{name}` now points at collection `{target}`."
        ))]))
    }

    pub(super) async fn drop_alias(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let name = required_str(args, "name")?;
        let (name, dropped, target) = crate::server::run_write(self.state.clone(), move |db| {
            let target = db.resolve_alias(&name);
            let dropped = db.drop_alias(&name)?;
            Ok((name, dropped, target))
        })
        .await
        .map_err(api_error)?;
        let message = if dropped {
            let target = target.unwrap_or_default();
            format!("Dropped alias `{name}`. Collection `{target}` and its records are unaffected.")
        } else {
            format!("No alias named `{name}` existed; nothing to drop.")
        };
        Ok(CallToolResult::success(vec![ContentBlock::text(message)]))
    }
}
