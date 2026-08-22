//! The four search tools — `recall`, `text_search`, `hybrid_search`, `related` — and the
//! metadata `filter` they share (nidus-k28.3). `recall`'s `reinforce` flag also writes.

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
use super::args::{
    api_error, optional_bool, optional_f32, optional_top_k, optional_usize, required_str, tool,
};
use super::{HitDto, hits_content, hits_with_plan_content};

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

/// Parse `rerank`/`rerank_overscan` into a [`crate::RerankOpts`], or `None` when unset.
/// `rerank_overscan` alone (without `rerank: true`) is ignored. Unconditional: `RerankOpts`
/// is plain data, needing no rerank feature to compile or run.
pub(super) fn parse_rerank(
    args: &Map<String, JsonValue>,
) -> Result<Option<crate::RerankOpts>, McpError> {
    if !optional_bool(args, "rerank")? {
        return Ok(None);
    }
    let mut opts = crate::RerankOpts::default();
    if let Some(overscan) = optional_usize(args, "rerank_overscan")? {
        if overscan == 0 {
            return Err(McpError::invalid_params(
                "`rerank_overscan` must be at least 1",
                None,
            ));
        }
        opts.overscan = overscan;
    }
    Ok(Some(opts))
}

/// Refuse a rerank candidate window past [`crate::server::dto::MAX_TOP_K`] — past this the
/// store would rank, and the provider score, an unreasonable depth. Reuses `rerank_depth`'s
/// real formula rather than approximating it a second time.
fn check_rerank_search_depth(opts: &SearchOpts) -> Result<(), McpError> {
    if opts.rerank.is_none() {
        return Ok(());
    }
    let depth = crate::store::rerank::rerank_depth(opts);
    if depth > crate::server::dto::MAX_TOP_K {
        return Err(McpError::invalid_params(
            format!(
                "rerank over top_k {} asks for a candidate depth of {depth}, exceeding the maximum of {}",
                opts.top_k,
                crate::server::dto::MAX_TOP_K
            ),
            None,
        ));
    }
    Ok(())
}

/// Hybrid analogue of [`check_rerank_search_depth`]: `HybridOpts` has no `limit_per`, so the
/// depth is just `(offset + top_k) * overscan`.
fn check_rerank_hybrid_depth(opts: &HybridOpts) -> Result<(), McpError> {
    let Some(r) = &opts.rerank else {
        return Ok(());
    };
    let depth = opts
        .offset
        .saturating_add(opts.top_k)
        .saturating_mul(r.overscan.max(1));
    if depth > crate::server::dto::MAX_TOP_K {
        return Err(McpError::invalid_params(
            format!(
                "rerank over top_k {} asks for a candidate depth of {depth}, exceeding the maximum of {}",
                opts.top_k,
                crate::server::dto::MAX_TOP_K
            ),
            None,
        ));
    }
    Ok(())
}

/// AND unit 1's not-expired guard into a caller's filter — never replacing it. Dropping
/// either half while merging is an easy, silent bug: the caller's own predicates could
/// vanish, or an expired entry could leak back into results (D4/D5, `nidus-k28`).
pub(super) fn with_ttl_guard(filter: Option<Filter>) -> Filter {
    let mut filter = filter.unwrap_or_default();
    filter.0.push(not_expired_predicate(now_ms()));
    filter
}

/// `plan`: opt into a [`crate::QueryPlan`] alongside the hits, shared by `recall` and
/// `related` (the two vector-path tools; `text_search` has no plan).
fn plan_schema() -> JsonValue {
    json!({
        "type": "boolean",
        "description": "Also report how the search ran: which index path answered it, how \
            many rows were scanned, and where the time went. Useful when results look thin \
            or a search is slow. Off by default."
    })
}

/// `diversity`: the MMR knob, shared by every tool whose ranking `Store::finish` shapes.
fn diversity_schema() -> JsonValue {
    json!({
        "type": "number",
        "description": "Spread near-duplicate results apart. 1.0 is pure relevance (the default behaviour when omitted), 0.5 balances, 0.0 maximises variety. Use it when several results say the same thing.",
        "minimum": 0.0,
        "maximum": 1.0
    })
}

/// Parse the `rollup` object into the `limit_per`/`expand` pair, through
/// [`Rollup::as_opts`] — the same mapping the HTTP and in-process recalls use.
pub(super) fn parse_rollup(
    args: &Map<String, JsonValue>,
) -> Result<Option<(crate::LimitPer, crate::Expand)>, McpError> {
    let Some(value) = args.get("rollup").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let obj = value
        .as_object()
        .ok_or_else(|| McpError::invalid_params("`rollup` must be an object".to_string(), None))?;
    let rollup = crate::Rollup {
        per_parent: optional_usize(obj, "per_parent")?.unwrap_or(1),
        neighbours: optional_usize(obj, "neighbours")?.unwrap_or(0),
    };
    Ok(Some(rollup.as_opts()))
}

/// `rollup`: read a chunked corpus as documents. The text-native form of `Expand` — a model
/// means "one result per document, widened", never a set of attr names, so the four-field
/// wire form stays on the HTTP/CLI/SDK surfaces.
fn rollup_schema() -> JsonValue {
    json!({
        "type": "object",
        "description": "Read a chunked corpus as documents rather than fragments: collapse each document's chunks to its best match, then widen that match with the chunks around it so you read a passage instead of a sentence fragment. Use it on any collection written by `nidus ingest` or `remember_chunked`.",
        "properties": {
            "per_parent": {
                "type": "integer",
                "description": "Chunks kept per document. Defaults to 1, the best-matching chunk.",
                "minimum": 1
            },
            "neighbours": {
                "type": "integer",
                "description": "Chunks stitched either side of each kept chunk, returned as the result's `context`. 1 or 2 is usually enough.",
                "minimum": 0
            }
        },
        "additionalProperties": false
    })
}

/// `rerank`: opt into the cross-encoder post-ranking stage, shared by `recall`,
/// `text_search`, and `hybrid_search`. A plain boolean, not an object: the query text is
/// already a required argument on every one of these tools.
fn rerank_bool_schema() -> JsonValue {
    json!({
        "type": "boolean",
        "description": "Re-score the top candidates with a cross-encoder, which reads the query and each candidate together and is markedly more accurate than embedding similarity alone. Costs one extra provider call. Use it when precision matters more than latency."
    })
}

/// `rerank_overscan`: how many times `top_k` candidates to widen the search to before
/// reranking. Ignored unless `rerank: true` is also set.
fn rerank_overscan_schema() -> JsonValue {
    json!({
        "type": "integer",
        "description": "How many times `top_k` candidates to retrieve before reranking. Higher finds more, costs more. Defaults to 10.",
        "minimum": 1
    })
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
                    "filter": filter_schema(),
                    "diversity": diversity_schema(),
                    "rollup": rollup_schema(),
                    "rerank": rerank_bool_schema(),
                    "rerank_overscan": rerank_overscan_schema(),
                    "plan": plan_schema(),
                    "reinforce": {
                        "type": "boolean",
                        "description": "Record that these entries were useful. Entries you recall with this set float up in later searches that rank on reinforcement, and entries nothing ever recalls sink. Leave it off for a plain lookup."
                    },
                    "extend_ttl_seconds": {
                        "type": "integer",
                        "description": "Also push the expiry of every returned entry out to this many seconds from now. Only applies with `reinforce`, and only to entries that already expire.",
                        "minimum": 1
                    }
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
                    "filter": filter_schema(),
                    "diversity": diversity_schema(),
                    "rollup": rollup_schema(),
                    "rerank": rerank_bool_schema(),
                    "rerank_overscan": rerank_overscan_schema()
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
                    "filter": filter_schema(),
                    "rerank": rerank_bool_schema(),
                    "rerank_overscan": rerank_overscan_schema()
                },
                "required": ["collection", "field", "query"],
                "additionalProperties": false
            }),
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
                "filter": filter_schema(),
                "diversity": diversity_schema(),
                "rollup": rollup_schema(),
                "plan": plan_schema()
            },
            "required": ["collection", "id"],
            "additionalProperties": false
        }),
    )
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
        let diversity = optional_f32(args, "diversity")?;
        let rollup = parse_rollup(args)?;
        let rerank = parse_rerank(args)?;
        let reinforce = optional_bool(args, "reinforce")?;
        let extend_ttl_seconds = optional_usize(args, "extend_ttl_seconds")?.map(|s| s as i64);
        let plan = optional_bool(args, "plan")?;

        let vector = embedder
            .embed_query(&query)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let opts = SearchOpts {
            top_k,
            min_score,
            filter: with_ttl_guard(filter),
            diversity,
            limit_per: rollup.as_ref().map(|(cap, _)| cap.clone()),
            expand: rollup.map(|(_, e)| e),
            rerank,
            plan,
            ..Default::default()
        };
        check_rerank_search_depth(&opts)?;

        #[cfg(feature = "rerank")]
        if opts.rerank.is_some() {
            let reranker = self.reranker()?;
            let (hits, plan) = self
                .rerank_recall_and_finish(
                    reranker,
                    embedder,
                    collection,
                    vector,
                    (query, reinforce.then_some(extend_ttl_seconds)),
                    opts,
                )
                .await?;
            return Ok(hits_with_plan_content(
                hits.into_iter().map(HitDto::from).collect(),
                plan,
            ));
        }
        #[cfg(not(feature = "rerank"))]
        if opts.rerank.is_some() {
            return Err(McpError::invalid_params(
                "this nidus server was built without rerank support (the `rerank` feature); \
                 `rerank` is unavailable",
                None,
            ));
        }

        let (hits, plan) = if reinforce {
            crate::server::run_write(self.state.clone(), move |db| {
                crate::memory::guard_recall_identity(db, embedder.as_ref(), &collection)?;
                if opts.plan {
                    let (hits, plan) = db.search_with_plan(collection.as_str(), &vector, &opts)?;
                    crate::memory::reinforce_hits(db, &collection, &hits, extend_ttl_seconds)?;
                    Ok((hits, Some(plan)))
                } else {
                    let hits = crate::memory::commit_recall(
                        db,
                        &collection,
                        &vector,
                        &opts,
                        extend_ttl_seconds,
                    )?;
                    Ok((hits, None))
                }
            })
            .await
            .map_err(api_error)?
        } else {
            crate::server::run_read(self.state.clone(), move |db| {
                // Recalling with a different embedder than wrote the collection returns
                // nonsense, so the same guard the HTTP route uses refuses it.
                crate::memory::guard_recall_identity(db, embedder.as_ref(), &collection)?;
                if opts.plan {
                    let (hits, plan) = db.search_with_plan(collection.as_str(), &vector, &opts)?;
                    Ok((hits, Some(plan)))
                } else {
                    Ok((db.search(collection.as_str(), &vector, &opts)?, None))
                }
            })
            .await
            .map_err(api_error)?
        };

        Ok(hits_with_plan_content(
            hits.into_iter().map(HitDto::from).collect(),
            plan,
        ))
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
        let diversity = optional_f32(args, "diversity")?;
        let rollup = parse_rollup(args)?;
        let rerank = parse_rerank(args)?;

        let opts = SearchOpts {
            top_k,
            filter: with_ttl_guard(filter),
            diversity,
            limit_per: rollup.as_ref().map(|(cap, _)| cap.clone()),
            expand: rollup.map(|(_, e)| e),
            rerank,
            ..Default::default()
        };
        check_rerank_search_depth(&opts)?;
        let q = crate::FtsQuery::new(field, query.clone());

        #[cfg(feature = "rerank")]
        if opts.rerank.is_some() {
            let reranker = self.reranker()?;
            let hits = self
                .rerank_text_search_and_finish(reranker, collection, q, query, opts)
                .await?;
            return Ok(hits_content(hits.into_iter().map(HitDto::from).collect()));
        }
        #[cfg(not(feature = "rerank"))]
        if opts.rerank.is_some() {
            return Err(McpError::invalid_params(
                "this nidus server was built without rerank support (the `rerank` feature); \
                 `rerank` is unavailable",
                None,
            ));
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
        let rerank = parse_rerank(args)?;

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
            rerank,
            ..Default::default()
        };
        check_rerank_hybrid_depth(&opts)?;
        let q = crate::FtsQuery::new(field, query.clone());

        #[cfg(feature = "rerank")]
        if opts.rerank.is_some() {
            let reranker = self.reranker()?;
            let hits = self
                .rerank_hybrid_and_finish(reranker, collection, vector, q, query, opts)
                .await?;
            return Ok(hits_content(hits.into_iter().map(HitDto::from).collect()));
        }
        #[cfg(not(feature = "rerank"))]
        if opts.rerank.is_some() {
            return Err(McpError::invalid_params(
                "this nidus server was built without rerank support (the `rerank` feature); \
                 `rerank` is unavailable",
                None,
            ));
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
        let diversity = optional_f32(args, "diversity")?;
        let rollup = parse_rollup(args)?;
        let plan = optional_bool(args, "plan")?;

        let (hits, plan) = crate::server::run_read(self.state.clone(), move |db| {
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
                diversity,
                limit_per: rollup.as_ref().map(|(cap, _)| cap.clone()),
                expand: rollup.map(|(_, e)| e),
                plan,
                ..Default::default()
            };
            if opts.plan {
                let (hits, plan) = db.search_similar_with_plan(
                    collection.as_str(),
                    collection.as_str(),
                    id.as_str(),
                    &opts,
                )?;
                Ok((hits, Some(plan)))
            } else {
                let hits = db.search_similar(
                    collection.as_str(),
                    collection.as_str(),
                    id.as_str(),
                    &opts,
                )?;
                Ok((hits, None))
            }
        })
        .await
        .map_err(api_error)?;

        Ok(hits_with_plan_content(
            hits.into_iter().map(HitDto::from).collect(),
            plan,
        ))
    }
}

/// The async half of MCP rerank: widen inside `run_read`, rerank outside it (network IO
/// must never sit under the store lock), then cut the page via the promoted
/// `Store::finish`/`finish_hybrid` — mirrors `crate::server`'s HTTP analogue, inlined here.
#[cfg(feature = "rerank")]
impl NidusMcp {
    async fn rerank_recall_and_finish(
        &self,
        reranker: std::sync::Arc<crate::rerank::AnyReranker>,
        embedder: std::sync::Arc<crate::embed::AnyEmbedder>,
        collection: String,
        vector: Vec<f32>,
        query_and_reinforce: (String, Option<Option<i64>>),
        opts: SearchOpts,
    ) -> Result<(Vec<crate::Hit>, Option<crate::QueryPlan>), McpError> {
        let (rerank_query, reinforce) = query_and_reinforce;
        let rerank_opts = opts.rerank.clone().unwrap_or_default();
        let (widened, kept) = crate::store::rerank::widened_opts(&opts);
        let name = collection.clone();
        // The plan (when asked for) describes this widened pre-rerank scan, not the
        // caller's page — rerank and retrim below are metadata-only and never rescan.
        let (hits, plan) = crate::server::run_read(self.state.clone(), move |db| {
            crate::memory::guard_recall_identity(db, embedder.as_ref(), &collection)?;
            if widened.plan {
                let (hits, plan) = db.search_with_plan(collection.as_str(), &vector, &widened)?;
                Ok((hits, Some(plan)))
            } else {
                Ok((db.search(collection.as_str(), &vector, &widened)?, None))
            }
        })
        .await
        .map_err(api_error)?;
        let mut reranked =
            crate::rerank::rerank_hits(reranker.as_ref(), &rerank_query, hits, &rerank_opts)
                .await
                .map_err(|e| McpError::internal_error(format!("{e:#}"), None))?;
        crate::store::rerank::retrim(&mut reranked, &opts, kept);
        // Stamp the **finished** page, never the overscanned candidate set — same rule the
        // HTTP handler follows, and the reason this cannot just call `finish` under a read.
        let hits = match reinforce {
            None => crate::server::run_read(self.state.clone(), move |db| {
                Ok(db.store().finish(reranked, &opts))
            })
            .await
            .map_err(api_error)?,
            Some(extend_ttl_seconds) => crate::server::run_write(self.state.clone(), move |db| {
                let finished = db.store().finish(reranked, &opts);
                crate::memory::reinforce_hits(db, &name, &finished, extend_ttl_seconds)?;
                Ok(finished)
            })
            .await
            .map_err(api_error)?,
        };
        Ok((hits, plan))
    }

    async fn rerank_text_search_and_finish(
        &self,
        reranker: std::sync::Arc<crate::rerank::AnyReranker>,
        collection: String,
        q: crate::FtsQuery,
        rerank_query: String,
        opts: SearchOpts,
    ) -> Result<Vec<crate::Hit>, McpError> {
        let rerank_opts = opts.rerank.clone().unwrap_or_default();
        let (widened, kept) = crate::store::rerank::widened_opts(&opts);
        let hits = crate::server::run_read(self.state.clone(), move |db| {
            db.text_search(
                crate::Scope::Collections(&[collection.as_str()]),
                &q,
                &widened,
            )
        })
        .await
        .map_err(api_error)?;
        let mut reranked =
            crate::rerank::rerank_hits(reranker.as_ref(), &rerank_query, hits, &rerank_opts)
                .await
                .map_err(|e| McpError::internal_error(format!("{e:#}"), None))?;
        crate::store::rerank::retrim(&mut reranked, &opts, kept);
        crate::server::run_read(self.state.clone(), move |db| {
            Ok(db.store().finish(reranked, &opts))
        })
        .await
        .map_err(api_error)
    }

    async fn rerank_hybrid_and_finish(
        &self,
        reranker: std::sync::Arc<crate::rerank::AnyReranker>,
        collection: String,
        vector: Vec<f32>,
        q: crate::FtsQuery,
        rerank_query: String,
        opts: HybridOpts,
    ) -> Result<Vec<crate::Hit>, McpError> {
        let rerank_opts = opts.rerank.clone().unwrap_or_default();
        let overscan = rerank_opts.overscan.max(1);
        let widened = HybridOpts {
            top_k: opts
                .offset
                .saturating_add(opts.top_k)
                .saturating_mul(overscan),
            offset: 0,
            candidates: opts.candidates.saturating_mul(overscan),
            ..opts.clone()
        };
        let hits = crate::server::run_read(self.state.clone(), move |db| {
            db.hybrid_search(
                crate::Scope::Collections(&[collection.as_str()]),
                &vector,
                &q,
                &widened,
            )
        })
        .await
        .map_err(api_error)?;
        let reranked =
            crate::rerank::rerank_hits(reranker.as_ref(), &rerank_query, hits, &rerank_opts)
                .await
                .map_err(|e| McpError::internal_error(format!("{e:#}"), None))?;
        crate::server::run_read(self.state.clone(), move |db| {
            Ok(db.store().finish_hybrid(reranked, &opts))
        })
        .await
        .map_err(api_error)
    }
}

// Every test here drives the rerank stage, so the module as a whole is rerank-gated: under
// a plain `mcp` build its imports would be unused rather than merely untested.
#[cfg(all(test, feature = "rerank"))]
mod tests {
    use super::*;

    /// Both rerank tests below need this; gated the same as the narrower of the two
    /// (`rerank-voyage` implies `rerank`) so it is never dead code on its own.
    fn obj(v: JsonValue) -> Map<String, JsonValue> {
        match v {
            JsonValue::Object(m) => m,
            _ => panic!("expected a JSON object"),
        }
    }

    /// A rerank request against a server with no reranker configured is a typed error
    /// naming the flag, never a silent unreranked pass-through. Uses `text_search`, which
    /// needs no embedder, to isolate the reranker-missing path.
    #[tokio::test]
    async fn rerank_without_a_configured_reranker_is_an_error() {
        let mut db = crate::Nidus::open_in_memory(3).unwrap();
        db.set_fts_schema("notes", &[crate::FtsField::new("body")])
            .unwrap();
        let mcp = NidusMcp::new(crate::server::test_state(Some(db)));

        let err = mcp
            .text_search(&obj(json!({
                "collection": "notes",
                "field": "body",
                "query": "anything",
                "rerank": true
            })))
            .await
            .unwrap_err();
        assert!(
            err.message.contains("--rerank-provider"),
            "error must name the flag: {}",
            err.message
        );
    }

    /// `recall`'s rerank stage over an in-process mock cross-encoder: three docs that tie
    /// on cosine (mocked embedder returns the same vector for everything) come back
    /// reordered by the mock's scores, proving the stage actually ran.
    #[cfg(all(
        feature = "memory",
        feature = "embed-openai-compat",
        feature = "rerank-voyage"
    ))]
    #[tokio::test]
    async fn recall_with_rerank_changes_the_order() {
        use crate::embed::{AnyEmbedder, EmbedConfig, EmbedProvider};
        use crate::rerank::testutil::mock_once;
        use crate::rerank::{AnyReranker, RerankConfig, RerankProvider};

        const INVERTING_SCORES: &str = r#"{"data":[{"index":0,"relevance_score":0.1},
            {"index":1,"relevance_score":0.5},{"index":2,"relevance_score":0.9}]}"#;

        let embed_base = crate::server::memory_tests::spawn_embed_mock();
        let embedder = AnyEmbedder::build(
            EmbedProvider::OpenAiCompat,
            EmbedConfig::new("mock-model").base_url(embed_base),
        )
        .await
        .expect("build mock embedder");
        let mock = mock_once(200, INVERTING_SCORES);
        let reranker = AnyReranker::build(
            RerankProvider::Voyage,
            RerankConfig::new("mock-rerank")
                .api_key("k")
                .base_url(&mock.base_url),
        )
        .unwrap();

        let mut db = crate::Nidus::open_in_memory(crate::server::memory_tests::DIM).unwrap();
        // Every doc gets the same vector (a dead cosine tie, broken on id: a, b, c) so any
        // reordering in the result can only be the rerank stage's doing.
        for id in ["a", "b", "c"] {
            let mut attrs = std::collections::BTreeMap::new();
            attrs.insert(
                crate::META_TEXT.to_string(),
                crate::Value::Str(format!("doc-{id}")),
            );
            db.upsert(
                "notes",
                &[crate::Record::new(id, vec![0.1, 0.2, 0.3], attrs)],
            )
            .unwrap();
        }
        let state = crate::server::AppState {
            embedder: Some(std::sync::Arc::new(embedder)),
            reranker: Some(std::sync::Arc::new(reranker)),
            ..crate::server::test_state(Some(db))
        };
        let mcp = NidusMcp::new(state);

        let baseline = mcp
            .recall(&obj(
                json!({"collection": "notes", "query": "anything", "top_k": 3}),
            ))
            .await
            .unwrap();
        assert_eq!(
            ids_of(baseline),
            vec!["a", "b", "c"],
            "pre-rerank order is the id tie-break"
        );

        let reranked = mcp
            .recall(&obj(
                json!({"collection": "notes", "query": "anything", "top_k": 3, "rerank": true}),
            ))
            .await
            .unwrap();
        assert_eq!(
            ids_of(reranked),
            vec!["c", "b", "a"],
            "rerank must reverse the tied order"
        );
    }

    /// `reinforce` must survive the rerank fork. It used to be dropped silently there: the
    /// rerank branch returns early, so a caller asking for both got the reranked page and no
    /// stamp, with nothing to say so.
    #[cfg(all(
        feature = "memory",
        feature = "embed-openai-compat",
        feature = "rerank-voyage"
    ))]
    #[tokio::test]
    async fn a_reranked_recall_still_reinforces() {
        use crate::embed::{AnyEmbedder, EmbedConfig, EmbedProvider};
        use crate::rerank::testutil::mock_once;
        use crate::rerank::{AnyReranker, RerankConfig, RerankProvider};

        const SCORES: &str = r#"{"data":[{"index":0,"relevance_score":0.9},
            {"index":1,"relevance_score":0.5}]}"#;

        let embed_base = crate::server::memory_tests::spawn_embed_mock();
        let embedder = AnyEmbedder::build(
            EmbedProvider::OpenAiCompat,
            EmbedConfig::new("mock-model").base_url(embed_base),
        )
        .await
        .expect("build mock embedder");
        let mock = mock_once(200, SCORES);
        let reranker = AnyReranker::build(
            RerankProvider::Voyage,
            RerankConfig::new("mock-rerank")
                .api_key("k")
                .base_url(&mock.base_url),
        )
        .unwrap();

        let mut db = crate::Nidus::open_in_memory(crate::server::memory_tests::DIM).unwrap();
        for id in ["a", "b"] {
            let mut attrs = std::collections::BTreeMap::new();
            attrs.insert(
                crate::META_TEXT.to_string(),
                crate::Value::Str(format!("doc-{id}")),
            );
            db.upsert(
                "notes",
                &[crate::Record::new(id, vec![0.1, 0.2, 0.3], attrs)],
            )
            .unwrap();
        }
        let state = crate::server::AppState {
            embedder: Some(std::sync::Arc::new(embedder)),
            reranker: Some(std::sync::Arc::new(reranker)),
            ..crate::server::test_state(Some(db))
        };
        let mcp = NidusMcp::new(state);

        mcp.recall(&obj(json!({
            "collection": "notes", "query": "anything", "top_k": 2,
            "rerank": true, "reinforce": true
        })))
        .await
        .unwrap();

        let guard = mcp.state.db.read().expect("lock");
        let db = guard.as_ref().expect("store");
        for id in ["a", "b"] {
            let rec = db
                .get("notes", id)
                .unwrap_or_else(|| panic!("{id} must still exist"));
            assert_eq!(
                rec.attrs.get(crate::meta::META_ACCESS_COUNT),
                Some(&crate::Value::Int(1)),
                "a reranked recall must stamp the page it returned ({id})"
            );
        }
    }

    /// Pull the `id` list out of a search tool's rendered JSON content.
    #[cfg(all(
        feature = "memory",
        feature = "embed-openai-compat",
        feature = "rerank-voyage"
    ))]
    fn ids_of(result: CallToolResult) -> Vec<String> {
        let rmcp::model::ContentBlock::Text(text) =
            result.content.into_iter().next().expect("content block")
        else {
            panic!("expected text content");
        };
        let hits: JsonValue = serde_json::from_str(&text.text).expect("hits JSON");
        hits.as_array()
            .expect("hits array")
            .iter()
            .map(|h| h["id"].as_str().expect("id").to_string())
            .collect()
    }
}
