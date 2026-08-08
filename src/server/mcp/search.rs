//! The three read-only search tools — `recall`, `text_search`, `hybrid_search` — and the
//! metadata `filter` they share (nidus-k28.3).

use rmcp::{
    ErrorData as McpError,
    model::{CallToolResult, Tool},
};
use serde_json::{Map, Value as JsonValue, json};

// Imported for its methods on `AnyEmbedder` — a trait method, not inherent.
use crate::embed::Embedder;
use crate::memory::{not_expired_predicate, now_ms};
use crate::{Filter, HybridOpts, SearchOpts};

use super::NidusMcp;
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
            }),
        ),
        tool(
            "text_search",
            "Search memory by keyword (BM25 full-text), not by meaning. Use this when the \
             exact wording matters — an error string, an identifier, a proper noun — and \
             `recall` when the meaning matters. Requires a full-text schema on the field.",
            json!({
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
            }),
        ),
        tool(
            "hybrid_search",
            "Search memory by meaning and keyword at once, fusing both rankings. Use this \
             when a query has both a semantic intent and a term that must appear — \
             \"the retry bug in the upsert path\". The text is embedded server-side.",
            json!({
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
            }),
        ),
    ]
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

        let hits = crate::server::run_read(self.state.clone(), move |db| {
            let opts = SearchOpts {
                top_k,
                filter: with_ttl_guard(filter),
                ..Default::default()
            };
            let q = crate::FtsQuery::new(field, query);
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

        // The one divergence from `POST /hybrid-search`, which takes a caller-supplied
        // `vector`: embedding the query text gives the same fusion from an argument a model
        // can actually write.
        let vector = embedder
            .embed_query(&query)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let hits = crate::server::run_read(self.state.clone(), move |db| {
            // `rrf_k`/`candidates` stay default: fusion knobs mean nothing to a model, so
            // exposing them adds ways to get worse results and none to get better ones.
            let opts = HybridOpts {
                top_k,
                filter: with_ttl_guard(filter),
                ..Default::default()
            };
            let q = crate::FtsQuery::new(field, query);
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
}
