//! The four read-only search tools — `recall`, `text_search`, `hybrid_search`, `related` —
//! and the metadata `filter` they share (nidus-k28.3).

use rmcp::{
    ErrorData as McpError,
    model::{CallToolResult, Tool},
};
use serde_json::{Map, Value as JsonValue, json};

// Imported for its methods on `AnyEmbedder` — a trait method, not inherent.
use crate::embed::Embedder;
use crate::memory::not_expired_predicate;
use crate::meta::now_ms;
use crate::{Filter, HybridOpts, SearchOpts};

use super::NidusMcp;
#[cfg(feature = "rerank")]
use super::args::optional_rerank;
use super::args::{api_error, optional_f32, optional_top_k, required_str, tool};
use super::{HitDto, hits_content};

/// A tagged [`crate::Value`]: `{"Str": "x"}`, `{"Int": 5}`, `{"Float": 1.5}`, `{"Bool":
/// true}`, or `{"List": ["a", "b"]}` for a list attribute (what `Contains`/`ContainsAny`
/// test against). `Null`/`DateTime` are left out of this curated surface.
fn value_schema() -> JsonValue {
    json!({
        "oneOf": [
            {"type": "object", "properties": {"Str": {"type": "string"}}, "required": ["Str"], "additionalProperties": false},
            {"type": "object", "properties": {"Int": {"type": "integer"}}, "required": ["Int"], "additionalProperties": false},
            {"type": "object", "properties": {"Float": {"type": "number"}}, "required": ["Float"], "additionalProperties": false},
            {"type": "object", "properties": {"Bool": {"type": "boolean"}}, "required": ["Bool"], "additionalProperties": false},
            {"type": "object", "properties": {"List": {"type": "array", "items": {"type": "string"}}}, "required": ["List"], "additionalProperties": false}
        ]
    })
}

/// A single-key `{"Name": payload}` object — the externally-tagged shape every
/// [`crate::Predicate`] variant serializes as.
fn tagged(name: &'static str, payload: JsonValue) -> JsonValue {
    let mut properties = Map::new();
    properties.insert(name.to_string(), payload);
    json!({
        "type": "object",
        "properties": JsonValue::Object(properties),
        "required": [name],
        "additionalProperties": false
    })
}

/// `{"Name": [key, value]}` — `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`, `Contains`.
fn key_value(name: &'static str) -> JsonValue {
    tagged(
        name,
        json!({
            "type": "array",
            "items": [{"type": "string"}, {"$ref": "#/$defs/value"}],
            "minItems": 2,
            "maxItems": 2
        }),
    )
}

/// `{"Name": [key, [value, ...]]}` — `In`, `NotIn`, `ContainsAny`.
fn key_values(name: &'static str) -> JsonValue {
    tagged(
        name,
        json!({
            "type": "array",
            "items": [{"type": "string"}, {"type": "array", "items": {"$ref": "#/$defs/value"}}],
            "minItems": 2,
            "maxItems": 2
        }),
    )
}

/// The recursive `Predicate` definition — a curated 13 of its 21 variants (the rest are
/// reachable over HTTP, not from MCP yet): `Eq Ne In NotIn Glob Lt Le Gt Ge Contains
/// ContainsAny Any Not`. The full set is too large to be a usable tool-selection prompt.
fn predicate_schema() -> JsonValue {
    json!({
        "oneOf": [
            key_value("Eq"),
            key_value("Ne"),
            key_value("Lt"),
            key_value("Le"),
            key_value("Gt"),
            key_value("Ge"),
            key_value("Contains"),
            tagged(
                "Glob",
                json!({
                    "type": "array",
                    "items": [{"type": "string"}, {"type": "string"}],
                    "minItems": 2,
                    "maxItems": 2
                }),
            ),
            key_values("In"),
            key_values("NotIn"),
            key_values("ContainsAny"),
            tagged("Any", json!({"type": "array", "items": {"$ref": "#/$defs/predicate"}})),
            tagged("Not", json!({"$ref": "#/$defs/predicate"})),
        ]
    })
}

/// The prose that carries the real work: a worked example, the AND/nesting rule, and the
/// absent-key asymmetry a model reaches for wrong by default.
const FILTER_DESCRIPTION: &str = "Optional metadata filter: a JSON array of predicates, \
AND-combined (a record must satisfy every one). `Any` OR-groups a list of predicates, and \
`Not` negates one predicate; both take predicates, not values, and nest arbitrarily. Example \
— entries where project is \"nidus\" or \"beads\", of kind \"decision\", excluding anything \
tagged \"wip\": [{\"Any\": [{\"Eq\": [\"project\", {\"Str\": \"nidus\"}]}, {\"Eq\": \
[\"project\", {\"Str\": \"beads\"}]}]}, {\"Eq\": [\"kind\", {\"Str\": \"decision\"}]}, \
{\"Not\": {\"Contains\": [\"tags\", {\"Str\": \"wip\"}]}}]. Values are tagged by type — see \
the `value` schema. Expired entries (a `remember` call's optional TTL) are already excluded \
from every result — no need to filter on `nidus.expires_at` yourself. ABSENT-KEY TRAP: \
`Ne`, `NotIn`, and `ContainsAny`'s sibling `NotContains` \
(not exposed here, but the rule carries) all require the key present with a different value, \
so they are FALSE when the key is simply missing. `Not` is a true complement instead: \
`Not({\"Eq\": [\"kind\", {\"Str\": \"decision\"}]})` is TRUE for a record with no `kind` at \
all, where `Ne` on the same key would be false. Reach for `Not(Eq(...))`, not `Ne`, when you \
mean \"missing or different\".";

/// The `$defs` the `filter` schema's `$ref`s point at. Every tool embedding [`filter_schema`]
/// must splice this in at its OWN schema root: a `$ref` fragment resolves from the document
/// root, so `$defs` nested under the `filter` property would never resolve.
pub(super) fn filter_defs() -> JsonValue {
    let mut defs = Map::new();
    defs.insert("value".to_string(), value_schema());
    defs.insert("predicate".to_string(), predicate_schema());
    JsonValue::Object(defs)
}

/// The `filter` property schema, shared by `recall`, `text_search`, and `hybrid_search`.
/// Pair every use with [`filter_defs`] at the enclosing schema's root.
pub(super) fn filter_schema() -> JsonValue {
    json!({
        "type": "array",
        "items": {"$ref": "#/$defs/predicate"},
        "description": FILTER_DESCRIPTION
    })
}

/// The `rerank` request option shared by `recall`, `text_search`, and `hybrid_search`. No
/// `query` property: each tool's own query/text argument stands in for it, so a model
/// cannot send two contradictory ones.
#[cfg(feature = "rerank")]
fn rerank_schema() -> JsonValue {
    json!({
        "type": "object",
        "description": "Rerank the results with a cross-encoder before returning them. Much more accurate than the default similarity ranking on the same corpus, at the cost of one extra API call. Use it when precision matters more than latency.",
        "properties": {
            "text_field": { "type": "string", "description": "Attr holding the text to score. Defaults to nidus.text, which is what remember writes." },
            "overscan":   { "type": "integer", "minimum": 1, "maximum": 64, "description": "How many times deeper than top_k to search before reranking. Defaults to 4. Higher finds more but costs more." },
            "model":      { "type": "string", "description": "Override the server's configured rerank model." }
        },
        "additionalProperties": false
    })
}

/// Splice [`rerank_schema`] into a tool's `properties`. Feature-gated at the function level
/// (not just the property) so `--features mcp` alone advertises no such argument at all,
/// mirroring `RerankDto`'s absence from the wire DTOs in that build.
#[cfg(feature = "rerank")]
fn with_rerank(mut schema: JsonValue) -> JsonValue {
    schema["properties"]["rerank"] = rerank_schema();
    schema
}

#[cfg(not(feature = "rerank"))]
fn with_rerank(schema: JsonValue) -> JsonValue {
    schema
}

/// Parse the optional `filter` argument. A present-but-malformed filter is a caller fault
/// (`invalid_params`), matching every other argument error in this module.
pub(super) fn parse_filter(args: &Map<String, JsonValue>) -> Result<Option<Filter>, McpError> {
    match args.get("filter") {
        None | Some(JsonValue::Null) => Ok(None),
        Some(v) => serde_json::from_value::<Filter>(v.clone())
            .map(Some)
            .map_err(|e| McpError::invalid_params(format!("`filter` is invalid: {e}"), None)),
    }
}

/// AND unit 1's not-expired guard into a caller's filter — never replacing it. Dropping
/// either half while merging is an easy, silent bug: the caller's own predicates could
/// vanish, or an expired entry could leak back into results (D4/D5, `nidus-k28`).
pub(super) fn with_ttl_guard(filter: Option<Filter>) -> Filter {
    let mut filter = filter.unwrap_or_default();
    filter.0.push(not_expired_predicate(now_ms()));
    filter
}

pub(super) fn tools() -> Vec<Tool> {
    vec![
        tool(
            "recall",
            "Search long-term memory by meaning and return the closest entries with \
             relevance scores. The query is embedded server-side — pass a natural-language \
             question, not vectors. This is semantic search: it finds entries that mean the \
             same thing as the query even when they share no words with it.",
            with_rerank(json!({
                "$defs": filter_defs(),
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Which collection to search."
                    },
                    "query": {
                        "type": "string",
                        "description": "A natural-language question or description of what you are looking for."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many results to return. Defaults to the server's configured value.",
                        "minimum": 1
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Drop results scoring below this. Scores are cosine similarity in [-1, 1]; around 0.7 is a reasonable floor for \"actually relevant\"."
                    },
                    "filter": filter_schema()
                },
                "required": ["collection", "query"],
                "additionalProperties": false
            })),
        ),
        tool(
            "text_search",
            "Search memory by keyword (BM25 full-text), not by meaning. Use this when the \
             exact wording matters — an error string, an identifier, a proper noun — and \
             `recall` when the meaning matters. Requires a full-text schema on the field.",
            with_rerank(json!({
                "$defs": filter_defs(),
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Which collection to search."
                    },
                    "field": {
                        "type": "string",
                        "description": "The indexed attribute to search within."
                    },
                    "query": {
                        "type": "string",
                        "description": "Keywords to match."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many results to return.",
                        "minimum": 1
                    },
                    "filter": filter_schema()
                },
                "required": ["collection", "field", "query"],
                "additionalProperties": false
            })),
        ),
        tool(
            "hybrid_search",
            "Search memory by meaning and keyword at once, fusing both rankings. Use this \
             when a query has both a semantic intent and a term that must appear — \
             \"the retry bug in the upsert path\". The text is embedded server-side.",
            with_rerank(json!({
                "$defs": filter_defs(),
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Which collection to search."
                    },
                    "field": {
                        "type": "string",
                        "description": "The indexed attribute to keyword-search within."
                    },
                    "query": {
                        "type": "string",
                        "description": "Natural language, used for BOTH the semantic and the keyword half of the search."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many results to return.",
                        "minimum": 1
                    },
                    "filter": filter_schema()
                },
                "required": ["collection", "field", "query"],
                "additionalProperties": false
            })),
        ),
    ]
}

/// `related`'s schema, kept separate from [`tools`] so [`super::tools`] can register it
/// last in the overall list without disturbing this module's own three-tool order.
pub(super) fn related_tool() -> Tool {
    tool(
        "related",
        "Find entries related to one you already have, using an existing entry as the \
         query instead of new text. Use this to follow a thread from a known memory, or \
         to check for a near-duplicate before writing a new one — an entry with nearly \
         identical content to the source will come back as a top hit. The source is \
         looked up by id and is never included in its own results.",
        json!({
            "$defs": filter_defs(),
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Which collection the source entry lives in; also where it searches."
                },
                "id": {
                    "type": "string",
                    "description": "The id of the entry to find things related to."
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many results to return. Defaults to the server's configured value.",
                    "minimum": 1
                },
                "min_score": {
                    "type": "number",
                    "description": "Drop results scoring below this. Scores are cosine similarity in [-1, 1]; around 0.7 is a reasonable floor for \"actually relevant\"."
                },
                "filter": filter_schema()
            },
            "required": ["collection", "id"],
            "additionalProperties": false
        }),
    )
}

#[cfg(all(test, feature = "rerank"))]
mod rerank_schema_tests {
    use super::*;

    /// All three tools list `rerank` (so `additionalProperties: false` honours rather than
    /// drops it), and no property of it has an array shape — the MCP text-native invariant.
    #[test]
    fn rerank_arg_is_listed_and_carries_no_vector_shaped_property() {
        for t in tools() {
            let rerank = t
                .input_schema
                .get("properties")
                .and_then(|p| p.get("rerank"))
                .unwrap_or_else(|| panic!("tool `{}` must list `rerank`", t.name));
            let props = rerank
                .get("properties")
                .and_then(|p| p.as_object())
                .expect("rerank schema carries its own properties");
            assert!(
                !props.contains_key("query"),
                "tool `{}`'s rerank arg must not carry its own query",
                t.name
            );
            for (name, prop) in props {
                assert_ne!(
                    prop.get("type").and_then(|v| v.as_str()),
                    Some("array"),
                    "tool `{}`'s rerank.{name} must not be array-shaped (no vectors over MCP)",
                    t.name
                );
            }
        }
    }
}

/// Route a plain `anyhow::Result` failure through [`crate::server::ApiError`]'s
/// message-based client/server-fault split, so the rerank branches report faults exactly
/// like the plain path does via [`api_error`].
#[cfg(feature = "rerank")]
fn anyhow_error(e: anyhow::Error) -> McpError {
    api_error(crate::server::ApiError::from(e))
}

impl NidusMcp {
    pub(super) async fn recall(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let embedder = self.embedder()?;
        let collection = required_str(args, "collection")?;
        let query = required_str(args, "query")?;
        let top_k = optional_top_k(args)?;
        let min_score = optional_f32(args, "min_score")?;
        let filter = parse_filter(args)?;
        #[cfg(feature = "rerank")]
        let rerank = optional_rerank(args)?;

        let vector = embedder
            .embed_query(&query)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let opts = SearchOpts {
            top_k,
            min_score,
            filter: with_ttl_guard(filter),
            ..Default::default()
        };

        // Fetch the deep window through `run_read` so its lock guard drops, then call the
        // provider: a guard held across the `.await` makes `call_tool`'s future non-`Send`.
        #[cfg(feature = "rerank")]
        if let Some(mut rr) = rerank {
            let reranker = self.reranker()?;
            if rr.query.is_none() {
                rr.query = Some(query.clone());
            }
            let (deep, plan) =
                crate::rerank::apply::plan_search(&opts, &rr).map_err(anyhow_error)?;
            let hits = crate::server::run_read(self.state.clone(), move |db| {
                crate::memory::guard_recall_identity(db, embedder.as_ref(), &collection)?;
                db.search(collection.as_str(), &vector, &deep)
            })
            .await
            .map_err(api_error)?;
            let hits = crate::rerank::apply::finish(hits, reranker.as_ref(), &plan)
                .await
                .map_err(anyhow_error)?;
            return Ok(hits_content(hits.into_iter().map(HitDto::from).collect()));
        }

        let hits = crate::server::run_read(self.state.clone(), move |db| {
            // Recalling with a different embedder than wrote the collection returns
            // nonsense, so the same guard the HTTP route uses refuses it.
            crate::memory::guard_recall_identity(db, embedder.as_ref(), &collection)?;
            db.search(collection.as_str(), &vector, &opts)
        })
        .await
        .map_err(api_error)?;

        Ok(hits_content(hits.into_iter().map(HitDto::from).collect()))
    }

    pub(super) async fn text_search(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let collection = required_str(args, "collection")?;
        let field = required_str(args, "field")?;
        let query = required_str(args, "query")?;
        let top_k = optional_top_k(args)?;
        let filter = parse_filter(args)?;
        #[cfg(feature = "rerank")]
        let rerank = optional_rerank(args)?;

        let opts = SearchOpts {
            top_k,
            filter: with_ttl_guard(filter),
            ..Default::default()
        };
        let q = crate::FtsQuery::new(field, query);

        // `rr.query` defaults to `q`'s own clause text inside `text_search_reranked`, so
        // there is nothing to fill in here — unlike `recall`'s raw-vector path.
        #[cfg(feature = "rerank")]
        if let Some(rr) = rerank {
            let reranker = self.reranker()?;
            let (deep, plan) = crate::rerank::apply::plan_text_search(&q, &opts, &rr);
            let hits = crate::server::run_read(self.state.clone(), move |db| {
                db.text_search(crate::Scope::Collections(&[collection.as_str()]), &q, &deep)
            })
            .await
            .map_err(api_error)?;
            let hits = crate::rerank::apply::finish(hits, reranker.as_ref(), &plan)
                .await
                .map_err(anyhow_error)?;
            return Ok(hits_content(hits.into_iter().map(HitDto::from).collect()));
        }

        let hits = crate::server::run_read(self.state.clone(), move |db| {
            db.text_search(crate::Scope::Collections(&[collection.as_str()]), &q, &opts)
        })
        .await
        .map_err(api_error)?;

        Ok(hits_content(hits.into_iter().map(HitDto::from).collect()))
    }

    pub(super) async fn hybrid_search(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let embedder = self.embedder()?;
        let collection = required_str(args, "collection")?;
        let field = required_str(args, "field")?;
        let query = required_str(args, "query")?;
        let top_k = optional_top_k(args)?;
        let filter = parse_filter(args)?;
        #[cfg(feature = "rerank")]
        let rerank = optional_rerank(args)?;

        // The one divergence from `POST /hybrid-search`, which takes a caller-supplied
        // `vector`: embedding the query text gives the same fusion from an argument a model
        // can actually write.
        let vector = embedder
            .embed_query(&query)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        // `rrf_k`/`candidates` stay default: fusion knobs mean nothing to a model, so
        // exposing them adds ways to get worse results and none to get better ones.
        let opts = HybridOpts {
            top_k,
            filter: with_ttl_guard(filter),
            ..Default::default()
        };
        let q = crate::FtsQuery::new(field, query);

        // `rr.query` defaults to `q`'s own clause text inside `hybrid_search_reranked`,
        // which is `query` again here, so there is nothing to fill in.
        #[cfg(feature = "rerank")]
        if let Some(rr) = rerank {
            let reranker = self.reranker()?;
            let (deep, plan) = crate::rerank::apply::plan_hybrid_search(&q, &opts, &rr);
            let hits = crate::server::run_read(self.state.clone(), move |db| {
                db.hybrid_search(
                    crate::Scope::Collections(&[collection.as_str()]),
                    &vector,
                    &q,
                    &deep,
                )
            })
            .await
            .map_err(api_error)?;
            let hits = crate::rerank::apply::finish(hits, reranker.as_ref(), &plan)
                .await
                .map_err(anyhow_error)?;
            return Ok(hits_content(hits.into_iter().map(HitDto::from).collect()));
        }

        let hits = crate::server::run_read(self.state.clone(), move |db| {
            db.hybrid_search(
                crate::Scope::Collections(&[collection.as_str()]),
                &vector,
                &q,
                &opts,
            )
        })
        .await
        .map_err(api_error)?;

        Ok(hits_content(hits.into_iter().map(HitDto::from).collect()))
    }

    /// "More like this": search using an already-stored entry instead of embedding new
    /// text. `search_similar` already names a missing/text-only source; the not-expired
    /// check here is the one thing a direct id lookup skips, the way `get`'s does.
    pub(super) async fn related(
        &self,
        args: &Map<String, JsonValue>,
    ) -> Result<CallToolResult, McpError> {
        let collection = required_str(args, "collection")?;
        let id = required_str(args, "id")?;
        let top_k = optional_top_k(args)?;
        let min_score = optional_f32(args, "min_score")?;
        let filter = parse_filter(args)?;

        let hits = crate::server::run_read(self.state.clone(), move |db| {
            if let Some(source) = db.get(&collection, &id) {
                let guard = Filter(vec![not_expired_predicate(now_ms())]);
                if !crate::filter::matches(&guard, &source.attrs) {
                    anyhow::bail!(
                        "{}: record `{collection}/{id}` has expired and cannot be used as a query",
                        crate::store::BAD_QUERY
                    );
                }
            }
            let opts = SearchOpts {
                top_k,
                min_score,
                filter: with_ttl_guard(filter),
                ..Default::default()
            };
            db.search_similar(collection.as_str(), collection.as_str(), id.as_str(), &opts)
        })
        .await
        .map_err(api_error)?;

        Ok(hits_content(hits.into_iter().map(HitDto::from).collect()))
    }
}
