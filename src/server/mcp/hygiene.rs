//! Record-level write hygiene (nidus-k28.4): `forget` to correct or remove a memory, `get`
//! to fetch one by id, and `browse` to list what a collection already holds — so an agent
//! can check for a near-duplicate before writing one, or fix a memory that turned out wrong.

use rmcp::{
    ErrorData as McpError,
    model::{CallToolResult, ContentBlock, Tool},
};
use serde_json::{Map, Value as JsonValue, json};

use crate::{ListOpts, Projection, Scope};

use super::NidusMcp;
use super::args::{api_error, optional_usize, required_str, tool};
use super::search::{filter_defs, filter_schema, parse_filter, with_ttl_guard};
use super::{HitDto, hits_content};

/// An optional string argument (as opposed to [`super::args::required_str`]'s required one).
fn optional_str(args: &Map<String, JsonValue>, key: &str) -> Result<Option<String>, McpError> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(McpError::invalid_params(
            format!("`{key}` must be a string"),
            None,
        )),
    }
}

/// An optional array of strings, e.g. `forget`'s `ids`.
fn optional_string_array(
    args: &Map<String, JsonValue>,
    key: &str,
) -> Result<Option<Vec<String>>, McpError> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Array(items)) => items
            .iter()
            .map(|v| match v {
                JsonValue::String(s) => Ok(s.clone()),
                _ => Err(McpError::invalid_params(
                    format!("`{key}` must be an array of strings"),
                    None,
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(McpError::invalid_params(
            format!("`{key}` must be an array of strings"),
            None,
        )),
    }
}

pub(super) fn tools() -> Vec<Tool> {
    vec![
        tool(
            "forget",
            "Permanently remove memories from a collection. Provide `ids` to remove specific \
             entries by id, or `filter` to remove every entry in the collection that matches \
             the filter — a filter can match many or all records, so review it before calling. \
             If both are given, `filter` wins and `ids` is ignored. You must provide at least \
             one of them; calling with neither is refused rather than treated as \"remove \
             everything\". Removing an id that does not exist is not an error — it just removes \
             nothing. This cannot be undone.",
            json!({
                "$defs": filter_defs(),
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Which collection to remove from."
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific memory ids to remove. Ignored if `filter` is also given."
                    },
                    "filter": filter_schema()
                },
                "required": ["collection"],
                "anyOf": [
                    {"required": ["ids"]},
                    {"required": ["filter"]}
                ],
                "additionalProperties": false
            }),
        ),
        tool(
            "get",
            "Fetch one memory by its exact id — a direct lookup, not a search. Use this to \
             check whether something is already remembered (to avoid writing a \
             near-duplicate) or to inspect an entry before deciding to `forget` or replace \
             it. A miss is reported in words, not as an error.",
            json!({
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Which collection to read from."
                    },
                    "id": {
                        "type": "string",
                        "description": "The memory's id."
                    }
                },
                "required": ["collection", "id"],
                "additionalProperties": false
            }),
        ),
        tool(
            "browse",
            "List entries in memory without a search query — useful for seeing what is \
             already stored before adding to it. Returns entries in storage order, not \
             ranked by relevance. Bounded and paginated: at most 10,000 per call, defaulting \
             to a smaller page; use `offset` to page through more.",
            json!({
                "$defs": filter_defs(),
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Which collection to browse. Omit to browse every collection."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum entries to return.",
                        "minimum": 1
                    },
                    "offset": {
                        "type": "integer",
                        "description": "How many matching entries to skip, for paging through a larger collection.",
                        "minimum": 0
                    },
                    "filter": filter_schema()
                },
                "required": [],
                "additionalProperties": false
            }),
        ),
    ]
}

impl NidusMcp {
    pub(super) async fn forget(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let collection = required_str(args, "collection")?;
        let ids = optional_string_array(args, "ids")?;
        let filter = parse_filter(args)?;

        // The single most important check in this tool: neither scoping argument given must
        // be a caller fault, never a silent "remove everything in the collection".
        if filter.is_none() && ids.is_none() {
            return Err(McpError::invalid_params(
                "forget requires `ids` or `filter` to scope what gets removed",
                None,
            ));
        }

        let name = collection.clone();
        let ids = ids.unwrap_or_default();
        let n = crate::server::run_write(self.state.clone(), move |db| match filter {
            Some(f) => db.delete_where(&name, &f),
            None => {
                let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
                db.delete(&name, &refs)
            }
        })
        .await
        .map_err(api_error)?;

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Forgot {n} entr{} from `{collection}`.",
            if n == 1 { "y" } else { "ies" }
        ))]))
    }

    pub(super) async fn get(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let collection = required_str(args, "collection")?;
        let id = required_str(args, "id")?;

        let name = collection.clone();
        let lookup_id = id.clone();
        let record =
            crate::server::run_read(self.state.clone(), move |db| Ok(db.get(&name, &lookup_id)))
                .await
                .map_err(api_error)?;

        // `get` is a direct map lookup that bypasses `Filter`, so it cannot inherit the
        // guard via `with_ttl_guard`; reusing `filter::matches` keeps the absent-key
        // semantics in one place.
        let guard = crate::Filter(vec![crate::memory::not_expired_predicate(
            crate::meta::now_ms(),
        )]);
        let record = record.filter(|r| crate::filter::matches(&guard, &r.attrs));

        // Only `id`/`attrs`: `Record::vector` would flood the model's context with floats
        // and break the text-native contract every other tool here holds to.
        match record {
            None => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "No entry with id `{id}` in `{collection}`."
            ))])),
            Some(r) => {
                let body = json!({ "id": r.id, "attrs": r.attrs });
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".to_string()),
                )]))
            }
        }
    }

    pub(super) async fn browse(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let collection = optional_str(args, "collection")?;
        let limit =
            optional_usize(args, "limit")?.unwrap_or_else(crate::server::dto::default_top_k);
        if limit > crate::server::dto::MAX_TOP_K {
            return Err(McpError::invalid_params(
                format!("`limit` must not exceed {}", crate::server::dto::MAX_TOP_K),
                None,
            ));
        }
        let offset = optional_usize(args, "offset")?.unwrap_or(0);
        let filter = with_ttl_guard(parse_filter(args)?);

        let hits = crate::server::run_read(self.state.clone(), move |db| {
            let opts = ListOpts {
                offset,
                limit,
                filter,
                projection: Projection::default(),
                order_by: None,
            };
            match &collection {
                Some(name) => db.list(name.as_str(), &opts),
                None => db.list(Scope::All, &opts),
            }
        })
        .await
        .map_err(api_error)?;

        Ok(hits_content(hits.into_iter().map(HitDto::from).collect()))
    }
}
