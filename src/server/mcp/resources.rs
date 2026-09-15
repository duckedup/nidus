//! MCP resources: collections listed concretely, entries reached by template. Read-only,
//! and the same `run_read` + TTL guard path the `browse`/`get` tools use.

use rmcp::{
    ErrorData as McpError,
    model::{Resource, ResourceContents, ResourceTemplate},
};
use serde_json::json;

use crate::{ListOpts, Projection};

use super::HitDto;
use super::NidusMcp;
use super::args::api_error;
use super::search::with_ttl_guard;
use super::uri::{self, Target};

/// Short, unlike the tool list's hour: this reflects live collections, which `remember`
/// creates on first write.
pub(super) const RESOURCES_TTL_MS: u64 = 60_000;

pub(super) fn templates() -> Vec<ResourceTemplate> {
    vec![
        ResourceTemplate::new(uri::ENTRY_TEMPLATE, "entry")
            .with_title("Memory entry")
            .with_description(
                "One memory by id. `collection` and `id` are percent-encoded path segments. \
                 The content is the entry's id and attributes as JSON, never its vector.",
            )
            .with_mime_type("application/json"),
        ResourceTemplate::new(uri::NS_ENTRY_TEMPLATE, "namespaced entry")
            .with_title("Memory entry (explicit namespace)")
            .with_description(
                "Same as `entry`, but names its namespace explicitly (nidus-pcpc.2) rather \
                 than relying on the reading connection's own `/ns/{namespace}/mcp` scoping. \
                 Only meaningful against a server started in namespaced mode.",
            )
            .with_mime_type("application/json"),
    ]
}

impl NidusMcp {
    /// This connection's ambient namespace, if `/ns/{name}/mcp` set one — embedded into
    /// generated URIs so they stay self-addressing (nidus-pcpc.2). Never fails, unlike
    /// `run_read`'s own namespace resolution.
    fn active_namespace() -> Option<String> {
        crate::server::NAMESPACE.try_with(Clone::clone).ok()
    }

    pub(super) async fn list_resources(&self) -> Result<Vec<Resource>, McpError> {
        let namespace = Self::active_namespace();
        let names = crate::server::run_read(self.state.clone(), |db| Ok(db.collections()))
            .await
            .map_err(api_error)?;
        Ok(names
            .into_iter()
            .map(|name| {
                Resource::new(
                    uri::collection_uri(namespace.as_deref(), &name),
                    name.clone(),
                )
                .with_title(name)
                .with_description(
                    "A bounded page of this collection's entries, as JSON. Page further \
                         with the `browse` tool.",
                )
                .with_mime_type("application/json")
            })
            .collect())
    }

    pub(super) async fn read_resource(&self, uri: &str) -> Result<Vec<ResourceContents>, McpError> {
        let target = uri::parse(uri).ok_or_else(|| {
            McpError::invalid_params(
                format!(
                    "`{uri}` is not a valid nidus resource URI; expected `{}` or `{}`",
                    uri::COLLECTION_TEMPLATE,
                    uri::ENTRY_TEMPLATE
                ),
                None,
            )
        })?;

        let body = match target {
            Target::Collection {
                namespace,
                collection: name,
            } => {
                let limit = crate::server::dto::default_top_k();
                let read_name = name.clone();
                // Ask for one more than we will show: `list` reports no has-more signal, so
                // `len() == limit` cannot tell a full page from an exactly-full collection.
                let hits = self
                    .with_namespace(namespace.clone(), async {
                        crate::server::run_read(self.state.clone(), move |db| {
                            let opts = ListOpts {
                                offset: 0,
                                limit: limit.saturating_add(1),
                                filter: with_ttl_guard(None),
                                projection: Projection::default(),
                                order_by: None,
                            };
                            db.list(read_name.as_str(), &opts)
                        })
                        .await
                        .map_err(api_error)
                    })
                    .await?;
                let truncated = hits.len() > limit;
                // Each entry carries its own URI so a client can navigate to it rather than
                // re-deriving the percent-encoding; `score` is dropped, being meaningless
                // for a listing with no query. Still `{id, attrs}`, so still no vector.
                let ns_for_entries = namespace.clone();
                let listed: Vec<_> = hits
                    .into_iter()
                    .take(limit)
                    .map(HitDto::from)
                    .map(|h| {
                        json!({
                            "uri": uri::entry_uri(ns_for_entries.as_deref(), &h.collection, &h.id),
                            "id": h.id,
                            "attrs": h.attrs,
                        })
                    })
                    .collect();
                // An object, not a bare array, so the truncation signal rides *inside* the
                // JSON: this is served as `application/json`, and appending prose to the
                // array would make the body unparseable exactly when a collection is large.
                let mut body = json!({ "entries": listed, "truncated": truncated });
                if truncated {
                    body["note"] = json!(format!(
                        "Showing the first {limit} entries; use the `browse` tool to page further."
                    ));
                }
                serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".to_string())
            }
            Target::Entry {
                namespace,
                collection,
                id,
            } => {
                let name = collection.clone();
                let lookup_id = id.clone();
                let record = self
                    .with_namespace(namespace, async {
                        crate::server::run_read(self.state.clone(), move |db| {
                            Ok(db.get(&name, &lookup_id))
                        })
                        .await
                        .map_err(api_error)
                    })
                    .await?;

                // `get` bypasses `Filter`, so it cannot inherit `with_ttl_guard`; reusing
                // `filter::matches` keeps the absent-key semantics in one place (hygiene.rs::get).
                let guard = crate::Filter(vec![crate::memory::not_expired_predicate(
                    crate::meta::now_ms(),
                )]);
                let record = record.filter(|r| crate::filter::matches(&guard, &r.attrs));

                match record {
                    None => {
                        return Err(McpError::invalid_params(
                            format!("no entry `{id}` in `{collection}` (missing or expired)"),
                            None,
                        ));
                    }
                    Some(r) => {
                        serde_json::to_string_pretty(&json!({ "id": r.id, "attrs": r.attrs }))
                            .unwrap_or_else(|_| "{}".to_string())
                    }
                }
            }
        };

        Ok(vec![
            ResourceContents::text(body, uri).with_mime_type("application/json"),
        ])
    }
}
