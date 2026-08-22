//! Wire types for the HTTP API and the CLI's JSON I/O.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    Aggregation, AnnConfig, AnnKind, Annotations, Expand, Filter, FilterIndexField, Footprint,
    FtsClause, FtsCombine, FtsField, HighlightOpts, Hit, Language, LimitPer, ListOpts, OrderBy,
    Projection, QueryPlan, RankBy, Record, StoreVersions, Value,
};

/// Body of `POST /collections/{name}/upsert`.
#[derive(Debug, Deserialize)]
pub struct UpsertRequest {
    pub records: Vec<Record>,
}

/// Body of `POST /collections/{name}/delete`. Supply `ids` to delete by id, or
/// `filter` to delete every matching record; `filter` wins if both are present.
#[derive(Debug, Default, Deserialize)]
pub struct DeleteRequest {
    #[serde(default)]
    pub ids: Vec<String>,
    #[serde(default)]
    pub filter: Option<Filter>,
}

/// Body of `POST /compact`. Bodyless (or `{}`) keeps the plain reclaim; `expired: true`
/// sweeps `nidus.expires_at`-past entries first (nidus-140).
#[derive(Debug, Default, Deserialize)]
pub struct CompactRequest {
    #[serde(default)]
    pub expired: bool,
}

/// The default result count. `pub(super)` so the MCP tools share it rather than picking
/// their own, which would answer one query two different ways depending on the surface.
pub(super) fn default_top_k() -> usize {
    10
}

/// The largest `offset + top_k` any request surface accepts. Past this a request is an
/// allocation demand rather than a query — no store returns ten thousand hits usefully — and
/// the bounded top-k kernel would otherwise be handed a `k` it must defend against itself.
pub(super) const MAX_TOP_K: usize = 10_000;

/// Wire form of [`crate::Expand`]: widen each hit with its document's neighbouring chunks.
/// Every field but `radius` defaults to the reserved chunk attrs, so `{"radius": 1}` is the
/// whole body a chunked corpus needs.
#[derive(Debug, Deserialize)]
pub struct ExpandRequest {
    #[serde(default = "default_parent_field")]
    pub parent_field: String,
    #[serde(default = "default_index_field")]
    pub index_field: String,
    #[serde(default = "default_text_field")]
    pub text_field: String,
    #[serde(default)]
    pub radius: usize,
}

impl From<ExpandRequest> for Expand {
    fn from(r: ExpandRequest) -> Self {
        Self {
            parent_field: r.parent_field,
            index_field: r.index_field,
            text_field: r.text_field,
            radius: r.radius,
        }
    }
}

fn default_parent_field() -> String {
    crate::META_PARENT_ID.to_string()
}

fn default_index_field() -> String {
    crate::META_CHUNK_INDEX.to_string()
}

fn default_text_field() -> String {
    crate::model::META_TEXT.to_string()
}

/// Body of `POST /search`. An empty `scope` searches every collection; `offset` skips
/// that many top-ranked hits, for pagination.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
    #[serde(default)]
    pub filter: Filter,
    /// Force the exact scan for this query, bypassing any index and the quantized first pass.
    #[serde(default)]
    pub exact: bool,
    #[serde(default)]
    pub include_attributes: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_attributes: Option<Vec<String>>,
    /// A ranking expression over the metric, e.g. `{"Decay": {"field": "ts", "origin": …}}`.
    #[serde(default)]
    pub rank_by: Option<RankBy>,
    /// Cap hits per distinct value of an attribute: `{"field": "path", "max": 2}`.
    #[serde(default)]
    pub limit_per: Option<LimitPer>,
    /// Widen each hit with its document's neighbouring chunks: `{"radius": 1}`.
    #[serde(default)]
    pub expand: Option<ExpandRequest>,
    /// MMR lambda spreading the page in vector space: `1.0` pure relevance, `0.0` pure spread.
    #[serde(default)]
    pub diversity: Option<f32>,
    /// Opt into the cross-encoder rerank stage. `None` (the default, and the only shape an
    /// old client ever sends) leaves the metric ranking untouched.
    #[cfg(feature = "rerank")]
    #[serde(default)]
    pub rerank: Option<RerankRequest>,
    /// Report how the query ran alongside the hits: `{"hits": [...], "plan": {...}}` instead
    /// of the bare array. Default `false` keeps today's response byte-identical.
    #[serde(default)]
    pub plan: bool,
}

/// The opt-in cross-encoder stage on a search request. `query` back-fills from the request's
/// own text where one exists (`/collections/{name}/recall`); `/search` and `/hybrid-search`
/// carry no text of their own, so an empty `query` there is a `400`.
#[cfg(feature = "rerank")]
#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub overscan: Option<usize>,
    #[serde(default)]
    pub text_attr: Option<String>,
}

/// Body of `POST /search/similar`: "more like this" over the vector already stored at
/// `collection`/`id`. Unlike [`SearchRequest`], an empty `scope` means the source's own
/// collection rather than every collection.
#[derive(Debug, Deserialize)]
pub struct SimilarRequest {
    pub collection: String,
    pub id: String,
    /// Which collections to search. Empty defaults to the source's own collection.
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
    #[serde(default)]
    pub filter: Filter,
    #[serde(default)]
    pub exact: bool,
    #[serde(default)]
    pub include_attributes: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_attributes: Option<Vec<String>>,
    #[serde(default)]
    pub rank_by: Option<RankBy>,
    #[serde(default)]
    pub limit_per: Option<LimitPer>,
    /// Widen each hit with its document's neighbouring chunks: `{"radius": 1}`.
    #[serde(default)]
    pub expand: Option<ExpandRequest>,
    /// MMR lambda spreading the page in vector space: `1.0` pure relevance, `0.0` pure spread.
    #[serde(default)]
    pub diversity: Option<f32>,
    /// Report how the query ran alongside the hits: `{"hits": [...], "plan": {...}}` instead
    /// of the bare array. Default `false` keeps today's response byte-identical.
    #[serde(default)]
    pub plan: bool,
}

/// Most queries one batch may carry. Matches turbopuffer's documented cap; the point is that
/// ONE request, holding one concurrency permit under one deadline, cannot buy unbounded scan.
pub(super) const MAX_BATCH_QUERIES: usize = 16;

/// Body of `POST /search/batch` (nidus-m50.11): several vector queries answered in one
/// round-trip. Each entry is an ordinary [`SearchRequest`], with its own scope and filter.
#[derive(Debug, Deserialize)]
pub struct BatchSearchRequest {
    pub queries: Vec<SearchRequest>,
    /// Merge the per-query rankings into ONE list instead of returning them side by side.
    #[serde(default)]
    pub fuse: Option<BatchFuse>,
}

/// Cross-query RRF: the same fusion `/hybrid-search` runs, over N query legs rather than a
/// vector leg and a text leg.
#[derive(Debug, Deserialize)]
pub struct BatchFuse {
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
    /// Per-query weights in request order. Empty leaves every leg neutral; otherwise the
    /// length must equal `queries`, since a short list would silently re-weight the wrong leg.
    #[serde(default)]
    pub weights: Vec<f32>,
    /// How many fused hits to return. Each leg is still ranked to its own `top_k`.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

/// The answer to a [`BatchSearchRequest`]: one ranking per query, or the single fused ranking
/// when `fuse` was asked for. Exactly one of the two fields is present.
#[derive(Debug, Serialize)]
pub struct BatchSearchResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<Vec<HitDto>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fused: Option<Vec<HitDto>>,
}

/// Either the bare hit array (no plan asked for) or `{hits, plan}`. Untagged, so an
/// unasked response is byte-identical to what every existing client already parses.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum SearchResponse {
    Hits(Vec<HitDto>),
    Explained {
        hits: Vec<HitDto>,
        /// Boxed: the plan dwarfs the bare-array variant, and serde flattens it either way.
        plan: Box<QueryPlan>,
    },
}

impl SearchResponse {
    /// `plan` present iff the caller asked for one; kept as a constructor so the three
    /// handlers that can return a plan cannot each re-derive this branch differently.
    pub fn new(hits: Vec<Hit>, plan: Option<QueryPlan>) -> Self {
        let hits = hits.into_iter().map(HitDto::from).collect();
        match plan {
            Some(plan) => Self::Explained {
                hits,
                plan: Box::new(plan),
            },
            None => Self::Hits(hits),
        }
    }
}

/// Resolve a request's projection fields. The two are spelled out on each request rather than
/// `#[serde(flatten)]`ed from a shared struct, because flatten buffers the *whole* body — the
/// query vector included — through `Content` on every search.
pub(super) fn resolve_projection(
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
) -> Result<Projection, &'static str> {
    match (include, exclude) {
        (None, None) => Ok(Projection::All),
        (Some(keys), None) => Ok(Projection::Include(keys)),
        (None, Some(keys)) => Ok(Projection::Exclude(keys)),
        (Some(_), Some(_)) => Err(
            "include_attributes and exclude_attributes are mutually exclusive; send at most one",
        ),
    }
}

/// Derived from [`ListOpts`] so the wire default and the library default cannot drift.
fn default_limit() -> usize {
    ListOpts::default().limit
}

fn default_rrf_k() -> f32 {
    60.0
}

fn default_candidates() -> usize {
    100
}

/// One clause of a multi-field text query: `{"field": "title", "query": "rust"}`.
#[derive(Debug, Deserialize)]
pub struct FtsClauseDto {
    pub field: String,
    pub query: String,
}

/// Body of `POST /text-search` (BM25). An empty `scope` searches every collection. Name the
/// fields either as one `field` + `query` pair or as a `clauses` list — never both.
#[derive(Debug, Deserialize)]
pub struct TextSearchRequest {
    #[serde(default)]
    pub field: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub clauses: Option<Vec<FtsClauseDto>>,
    /// How several clauses fold into one score: `"Sum"` (default) or `"Max"`.
    #[serde(default)]
    pub combine: FtsCombine,
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub offset: usize,
    /// A raw BM25 score floor (not cosine).
    #[serde(default)]
    pub min_score: Option<f32>,
    #[serde(default)]
    pub filter: Filter,
    /// Report each matched clause's own BM25 score on every hit.
    #[serde(default)]
    pub explain: bool,
    /// Return highlighted fragments; `{}` takes the defaults. Highlighting reads the stored
    /// text, so it still works on a field the projection below dropped.
    #[serde(default)]
    pub highlight: Option<HighlightOpts>,
    #[serde(default)]
    pub include_attributes: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_attributes: Option<Vec<String>>,
    #[serde(default)]
    pub rank_by: Option<RankBy>,
    #[serde(default)]
    pub limit_per: Option<LimitPer>,
    /// Widen each hit with its document's neighbouring chunks: `{"radius": 1}`.
    #[serde(default)]
    pub expand: Option<ExpandRequest>,
    /// MMR lambda spreading the page in vector space: `1.0` pure relevance, `0.0` pure spread.
    #[serde(default)]
    pub diversity: Option<f32>,
    /// Opt into the cross-encoder rerank stage. An omitted or empty `rerank.query` falls
    /// back to the single-field `query` above; the `clauses` spelling has no single text,
    /// so it must name `rerank.query` itself.
    #[cfg(feature = "rerank")]
    #[serde(default)]
    pub rerank: Option<RerankRequest>,
}

/// Body of `POST /hybrid-search`: fuse a vector query and a BM25 text query (RRF). The text
/// leg takes the same `field`+`text` / `clauses` choice as `/text-search`.
#[derive(Debug, Deserialize)]
pub struct HybridSearchRequest {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub field: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub clauses: Option<Vec<FtsClauseDto>>,
    #[serde(default)]
    pub combine: FtsCombine,
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub filter: Filter,
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
    #[serde(default = "default_candidates")]
    pub candidates: usize,
    /// Report each leg's own rank and score, plus every matched clause's BM25 score.
    #[serde(default)]
    pub explain: bool,
    #[serde(default)]
    pub highlight: Option<HighlightOpts>,
    /// Weight on the vector leg's RRF contribution. Both weights at `1.0` (the default)
    /// reproduces the unweighted fusion exactly.
    #[serde(default = "default_weight")]
    pub vector_weight: f32,
    /// Weight on the BM25 leg's RRF contribution.
    #[serde(default = "default_weight")]
    pub text_weight: f32,
    /// Widen each fused hit with its document's neighbouring chunks: `{"radius": 1}`.
    #[serde(default)]
    pub expand: Option<ExpandRequest>,
    /// Opt into the cross-encoder rerank stage over the fused ranking.
    #[cfg(feature = "rerank")]
    #[serde(default)]
    pub rerank: Option<RerankRequest>,
    /// Report how the query ran alongside the hits: `{"hits": [...], "plan": {...}}` instead
    /// of the bare array. Default `false` keeps today's response byte-identical.
    #[serde(default)]
    pub plan: bool,
}

fn default_weight() -> f32 {
    1.0
}

/// Resolve a text query's clauses from the two accepted spellings: the single `field` +
/// `text` pair, or a non-empty `clauses` list. Both, neither, or an empty list is a caller
/// error — an empty result would otherwise read as "no matches" rather than "no query".
pub(super) fn resolve_clauses(
    field: Option<String>,
    text: Option<String>,
    clauses: Option<Vec<FtsClauseDto>>,
) -> Result<Vec<FtsClause>, &'static str> {
    match (field, text, clauses) {
        (Some(f), Some(t), None) => Ok(vec![FtsClause::new(f, t)]),
        (None, None, Some(list)) if !list.is_empty() => Ok(list
            .into_iter()
            .map(|c| FtsClause::new(c.field, c.query))
            .collect()),
        (None, None, Some(_)) => Err("clauses must not be empty"),
        (None, None, None) => Err("a text query needs a field plus its text, or a clauses list"),
        (_, _, Some(_)) => Err("field/text and clauses are mutually exclusive; send one form"),
        _ => Err("field and its query text must be sent together"),
    }
}

/// One entry of [`FtsSchemaRequest::fields`]: either a bare field name (everything
/// defaulted) or an object naming the field plus the params to override. The two forms
/// are why this is `untagged` — `{"fields": ["body"]}` must keep working verbatim.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum FtsFieldDto {
    /// `"body"` — default BM25 params, default analyzer.
    Name(String),
    /// `{"field": "body", "k1": 1.5, "ascii_folding": true}` — any subset of the knobs.
    Spec {
        field: String,
        #[serde(default)]
        k1: Option<f32>,
        #[serde(default)]
        b: Option<f32>,
        #[serde(default)]
        language: Option<Language>,
        #[serde(default)]
        ascii_folding: Option<bool>,
        #[serde(default)]
        max_token_len: Option<usize>,
    },
}

impl From<FtsFieldDto> for FtsField {
    fn from(dto: FtsFieldDto) -> Self {
        match dto {
            FtsFieldDto::Name(field) => FtsField::new(field),
            FtsFieldDto::Spec {
                field,
                k1,
                b,
                language,
                ascii_folding,
                max_token_len,
            } => {
                let mut f = FtsField::new(field);
                // Each knob is independent: an absent one keeps the default rather than
                // resetting a sibling, so a partial body means "change only this".
                f.k1 = k1.unwrap_or(f.k1);
                f.b = b.unwrap_or(f.b);
                f.analyzer.language = language.unwrap_or(f.analyzer.language);
                f.analyzer.ascii_folding = ascii_folding.unwrap_or(f.analyzer.ascii_folding);
                f.analyzer.max_token_len = max_token_len.or(f.analyzer.max_token_len);
                f
            }
        }
    }
}

/// Body of `POST /collections/{name}/fts-schema`: the attribute fields to full-text
/// index, each a name or a `{field, k1, b, language, ascii_folding, max_token_len}` object.
#[derive(Debug, Deserialize)]
pub struct FtsSchemaRequest {
    pub fields: Vec<FtsFieldDto>,
}

/// One entry of [`FilterIndexRequest::fields`]: a bare field name, or an object naming the
/// field plus which structures to build. `untagged` for the same reason as [`FtsFieldDto`]:
/// `{"fields": ["text"]}` is the common form and must stay valid.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum FilterIndexFieldDto {
    /// `"text"` — both structures.
    Name(String),
    /// `{"field": "text", "trigrams": false}` — either knob, independently.
    Spec {
        field: String,
        #[serde(default)]
        tokens: Option<bool>,
        #[serde(default)]
        trigrams: Option<bool>,
    },
}

impl From<FilterIndexFieldDto> for FilterIndexField {
    fn from(dto: FilterIndexFieldDto) -> Self {
        match dto {
            FilterIndexFieldDto::Name(field) => FilterIndexField::new(field),
            FilterIndexFieldDto::Spec {
                field,
                tokens,
                trigrams,
            } => {
                let f = FilterIndexField::new(field);
                // Absent means "leave this one alone", matching `FtsFieldDto`.
                let f = match tokens {
                    Some(on) => f.tokens(on),
                    None => f,
                };
                match trigrams {
                    Some(on) => f.trigrams(on),
                    None => f,
                }
            }
        }
    }
}

/// Body of `POST /collections/{name}/filter-index`: the attribute fields to index for the
/// text predicates, each a name or a `{field, tokens, trigrams}` object. An empty `fields`
/// drops the declaration.
#[derive(Debug, Deserialize)]
pub struct FilterIndexRequest {
    pub fields: Vec<FilterIndexFieldDto>,
}

/// Body of `POST /list`. Metadata-only query (no vector). An empty `scope`
/// lists from every collection. `offset` skips matches for pagination.
#[derive(Debug, Deserialize)]
pub struct ListRequest {
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub filter: Filter,
    #[serde(default)]
    pub include_attributes: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_attributes: Option<Vec<String>>,
    /// Sort by an attribute: `{"field": "updated_at", "descending": true}`.
    #[serde(default)]
    pub order_by: Option<OrderBy>,
}

/// Body of `POST /aggregate`. An empty `scope` aggregates over every collection.
#[derive(Debug, Default, Deserialize)]
pub struct AggregateRequest {
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default)]
    pub filter: Filter,
    /// Attributes to sum alongside the always-reported count.
    #[serde(default)]
    pub sum: Vec<String>,
    /// Split the answer into one row per distinct value of this attribute.
    #[serde(default)]
    pub group_by: Option<String>,
}

/// Serializable mirror of [`crate::Aggregation`]: `{"count": 12, "sums": {"bytes": {"Int": 40}}}`,
/// plus `groups` when the request asked for a `group_by`.
#[derive(Debug, Serialize)]
pub struct AggregationDto {
    pub count: u64,
    pub sums: BTreeMap<String, Value>,
    /// Omitted entirely when no `group_by` was requested, so an ungrouped answer keeps the
    /// shape it had before grouping existed.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupDto>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub groups_truncated: bool,
}

/// One `group_by` row. `value` is `null` for the records missing the attribute — distinct from
/// a present `{"Null": null}`, which groups on its own.
#[derive(Debug, Serialize)]
pub struct GroupDto {
    pub value: Option<Value>,
    pub count: u64,
    pub sums: BTreeMap<String, Value>,
}

impl From<Aggregation> for AggregationDto {
    fn from(a: Aggregation) -> Self {
        Self {
            count: a.count,
            sums: a.sums,
            groups: a
                .groups
                .into_iter()
                .map(|g| GroupDto {
                    value: g.value,
                    count: g.count,
                    sums: g.sums,
                })
                .collect(),
            groups_truncated: a.groups_truncated,
        }
    }
}

/// Body of `POST /collections/{name}/remember` (the `memory` feature).
#[cfg(feature = "memory")]
#[derive(Debug, Deserialize)]
pub struct RememberRequest {
    pub id: String,
    pub text: String,
    /// `"raw"` (embed the text as given, the default) or `"summarize"`
    /// (summarize first, embed the summary — requires a summarizer).
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub attrs: BTreeMap<String, Value>,
    /// Seconds until this memory expires, from the moment it is written. `None`
    /// (the default) never expires.
    #[serde(default)]
    pub ttl_seconds: Option<i64>,
    /// Cosine-similarity floor above which this write updates the nearest existing
    /// entry instead of inserting a competing near-duplicate. `None` (the default)
    /// disables dedup — every write is a plain upsert by `id`.
    #[serde(default)]
    pub dedupe_threshold: Option<f32>,
}

/// Wire form of [`crate::memory::Rollup`]: read a chunked corpus as documents rather than
/// fragments. `{"neighbours": 1}` keeps the best chunk per document and widens it.
#[cfg(feature = "memory")]
#[derive(Debug, Deserialize)]
pub struct RollupRequest {
    #[serde(default = "default_per_parent")]
    pub per_parent: usize,
    #[serde(default)]
    pub neighbours: usize,
}

#[cfg(feature = "memory")]
impl From<RollupRequest> for crate::memory::Rollup {
    fn from(r: RollupRequest) -> Self {
        Self {
            per_parent: r.per_parent,
            neighbours: r.neighbours,
        }
    }
}

#[cfg(feature = "memory")]
fn default_per_parent() -> usize {
    1
}

/// Body of `POST /collections/{name}/recall` (the `memory` feature).
#[cfg(feature = "memory")]
#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
    #[serde(default)]
    pub filter: Filter,
    /// MMR lambda spreading the recalled window in vector space, so one verbose document's
    /// near-identical chunks stop filling it.
    #[serde(default)]
    pub diversity: Option<f32>,
    /// Read the collection as a chunked corpus: `{"neighbours": 1}`.
    #[serde(default)]
    pub rollup: Option<RollupRequest>,
    /// Opt into the cross-encoder rerank stage. An omitted or empty `rerank.query` falls
    /// back to `query` above, so `{"rerank": {}}` is a valid minimal form here.
    #[cfg(feature = "rerank")]
    #[serde(default)]
    pub rerank: Option<RerankRequest>,
    /// Stamp `nidus.access_count` / `nidus.last_accessed` on every returned entry. This
    /// makes the recall a **write**: it takes the writer lock, and is refused on a
    /// read-only store rather than answered as though the stamp happened.
    #[serde(default)]
    pub reinforce: bool,
    /// Push an existing `nidus.expires_at` forward to `now + this`. Only honoured with
    /// `reinforce`; never creates an expiry on an entry that had none.
    #[serde(default)]
    pub extend_ttl_seconds: Option<i64>,
    /// Ranking expression layered over cosine, the same shape `/search` takes: decay over
    /// `nidus.last_accessed`, a reinforcement term over `nidus.access_count`, or both.
    #[serde(default)]
    pub rank_by: Option<RankBy>,
}

/// Serializable mirror of [`crate::Hit`] (which carries no serde derive).
#[derive(Debug, Serialize)]
pub struct HitDto {
    pub collection: String,
    pub id: String,
    pub score: f32,
    pub attrs: BTreeMap<String, Value>,
    /// Present only when the query asked to `explain` or to highlight, so an unannotated
    /// response is byte-identical to a nidus without annotations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Annotations>,
    /// The hit's chunk widened with its neighbours. Present only when the query asked to
    /// `expand`/`rollup`, so an unexpanded response is byte-identical to a nidus without it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

impl From<Hit> for HitDto {
    fn from(h: Hit) -> Self {
        Self {
            collection: h.collection,
            id: h.id,
            score: h.score,
            attrs: h.attrs,
            annotations: h.annotations,
            context: h.context,
        }
    }
}

/// Serializable mirror of [`crate::Footprint`].
#[derive(Debug, Serialize)]
pub struct FootprintDto {
    pub rows: u64,
    pub dead_rows: u64,
    pub dimension: usize,
    pub vector_bytes: u64,
    pub doc_count: usize,
    /// Memory held by the opt-in filter index; `0` when no collection declares one. The
    /// only externally observable evidence that a declaration is live, since the index
    /// deliberately changes no query's results.
    pub filter_index_bytes: u64,
}

impl From<Footprint> for FootprintDto {
    fn from(f: Footprint) -> Self {
        Self {
            rows: f.rows,
            dead_rows: f.dead_rows,
            dimension: f.dimension,
            vector_bytes: f.vector_bytes,
            doc_count: f.doc_count,
            filter_index_bytes: f.filter_index_bytes,
        }
    }
}

/// Serializable mirror of [`crate::StoreVersions`] for `GET /versions`.
#[derive(Debug, Serialize)]
pub struct VersionsDto {
    pub commit_version: u64,
    pub oldest_readable: Option<u64>,
    pub pinned: Option<u64>,
    pub readable: Vec<u64>,
}

impl From<StoreVersions> for VersionsDto {
    fn from(v: StoreVersions) -> Self {
        Self {
            commit_version: v.commit_version,
            oldest_readable: v.oldest_readable,
            pinned: v.pinned,
            readable: v.readable,
        }
    }
}

/// Serializable mirror of [`crate::AnnConfig`] for the `stats` surface. Only the
/// knobs that apply to the active [`AnnKind`] are emitted; the inert ones are
/// omitted. `stats` reports `null` when no ANN index is configured (exact search).
#[derive(Debug, Serialize)]
pub struct AnnDto {
    pub kind: String,
    pub overscan: usize,
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub m: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_construction: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_search: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_lists: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_probe: Option<usize>,
}

impl From<AnnConfig> for AnnDto {
    fn from(a: AnnConfig) -> Self {
        let (hnsw, ivf) = match a.kind {
            AnnKind::Hnsw => (true, false),
            AnnKind::Ivf => (false, true),
        };
        Self {
            kind: format!("{:?}", a.kind),
            overscan: a.overscan,
            seed: a.seed,
            m: hnsw.then_some(a.m),
            ef_construction: hnsw.then_some(a.ef_construction),
            ef_search: hnsw.then_some(a.ef_search),
            n_lists: ivf.then_some(a.n_lists),
            n_probe: ivf.then_some(a.n_probe),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::Predicate;

    /// Decode one `fields` entry exactly as the handler does.
    fn field(json: serde_json::Value) -> FtsField {
        serde_json::from_value::<FtsFieldDto>(json).unwrap().into()
    }

    #[test]
    fn a_bare_field_name_still_means_todays_defaults() {
        // The compatibility contract: `{"fields": ["body"]}` must not change meaning.
        let req: FtsSchemaRequest = serde_json::from_str(r#"{"fields":["body"]}"#).unwrap();
        let decoded: Vec<FtsField> = req.fields.into_iter().map(FtsField::from).collect();
        assert_eq!(decoded, vec![FtsField::new("body")]);
    }

    #[test]
    fn a_field_object_overrides_only_what_it_names() {
        let f = field(json!({"field": "body", "k1": 1.5}));
        assert_eq!(f.k1, 1.5);
        assert_eq!(f.b, 0.75, "an absent knob keeps its default");
        assert_eq!(f.analyzer, crate::Analyzer::default());

        let f = field(json!({"field": "body", "ascii_folding": true, "max_token_len": 40}));
        assert_eq!((f.k1, f.b), (1.2, 0.75));
        assert!(f.analyzer.ascii_folding);
        assert_eq!(f.analyzer.max_token_len, Some(40));

        // `{"field": "body"}` is the object spelling of the bare name.
        assert_eq!(field(json!({"field": "body"})), FtsField::new("body"));
    }

    #[test]
    fn the_language_accepts_both_json_spellings() {
        for form in ["English", "english", "en"] {
            let f = field(json!({"field": "body", "language": form}));
            assert_eq!(f.analyzer.language, Language::English);
        }
    }

    #[test]
    fn the_two_field_forms_mix_within_one_request() {
        let req: FtsSchemaRequest =
            serde_json::from_str(r#"{"fields":["title",{"field":"body","b":0.4}]}"#).unwrap();
        let decoded: Vec<FtsField> = req.fields.into_iter().map(FtsField::from).collect();
        assert_eq!(
            decoded,
            vec![FtsField::new("title"), FtsField::new("body").b(0.4)]
        );
    }

    /// Decode a `/text-search` body and resolve its clauses exactly as the handler does.
    fn clauses(json: serde_json::Value) -> Result<Vec<FtsClause>, &'static str> {
        let req: TextSearchRequest = serde_json::from_value(json).unwrap();
        resolve_clauses(req.field, req.query, req.clauses)
    }

    #[test]
    fn the_single_field_body_still_means_one_clause() {
        // The compatibility contract: `{"field": …, "query": …}` must not change meaning.
        assert_eq!(
            clauses(json!({"field": "body", "query": "run"})).unwrap(),
            vec![FtsClause::new("body", "run")]
        );
    }

    #[test]
    fn a_clauses_list_carries_distinct_text_per_field() {
        let got = clauses(json!({"clauses": [
            {"field": "title", "query": "rust"},
            {"field": "body", "query": "async runtime"}
        ]}))
        .unwrap();
        assert_eq!(
            got,
            vec![
                FtsClause::new("title", "rust"),
                FtsClause::new("body", "async runtime")
            ]
        );
    }

    #[test]
    fn an_unusable_clause_spelling_is_refused_rather_than_guessed() {
        for body in [
            json!({"clauses": []}),
            json!({}),
            json!({"field": "body"}),
            json!({"query": "run"}),
            json!({"field": "body", "query": "run", "clauses": [{"field": "t", "query": "x"}]}),
        ] {
            assert!(clauses(body.clone()).is_err(), "{body}");
        }
    }

    #[test]
    fn combine_accepts_both_json_spellings_and_defaults_to_sum() {
        let req: TextSearchRequest = serde_json::from_value(json!({"clauses": []})).unwrap();
        assert_eq!(req.combine, FtsCombine::Sum);
        for (form, want) in [
            ("Sum", FtsCombine::Sum),
            ("sum", FtsCombine::Sum),
            ("Max", FtsCombine::Max),
            ("max", FtsCombine::Max),
        ] {
            let req: TextSearchRequest =
                serde_json::from_value(json!({"clauses": [], "combine": form})).unwrap();
            assert_eq!(req.combine, want, "{form}");
        }
    }

    #[test]
    fn a_hit_without_annotations_serializes_exactly_as_it_always_did() {
        let hit = HitDto::from(Hit::new("docs", "a", 0.5, BTreeMap::new()));
        assert_eq!(
            serde_json::to_value(&hit).unwrap(),
            json!({"collection": "docs", "id": "a", "score": 0.5, "attrs": {}})
        );
    }

    /// The annotation wire spelling every SDK will mirror in nidus-m50.18. Asserted here
    /// because the crate's own round trip is bincode, which would not catch a rename.
    #[test]
    fn annotation_json_spelling_is_stable() {
        let mut hit = Hit::new("docs", "a", 0.5, BTreeMap::new());
        hit.annotations = Some(crate::Annotations {
            vector: Some(crate::LegScore {
                rank: 0,
                score: 0.5,
            }),
            text: Some(crate::LegScore {
                rank: 2,
                score: 1.5,
            }),
            clauses: vec![crate::ClauseScore {
                field: "title".into(),
                score: 1.5,
            }],
            highlights: vec![crate::Highlight {
                field: "body".into(),
                fragments: vec![crate::Fragment {
                    text: "we were running".into(),
                    spans: vec![(8, 15)],
                }],
            }],
        });
        assert_eq!(
            serde_json::to_value(HitDto::from(hit)).unwrap()["annotations"],
            json!({
                "vector": {"rank": 0, "score": 0.5},
                "text": {"rank": 2, "score": 1.5},
                "clauses": [{"field": "title", "score": 1.5}],
                "highlights": [{
                    "field": "body",
                    "fragments": [{"text": "we were running", "spans": [[8, 15]]}]
                }]
            })
        );
    }

    #[test]
    fn highlight_options_come_off_the_wire_with_their_defaults() {
        let req: TextSearchRequest =
            serde_json::from_value(json!({"clauses": [], "highlight": {}})).unwrap();
        assert_eq!(req.highlight, Some(HighlightOpts::default()));
        let req: TextSearchRequest =
            serde_json::from_value(json!({"clauses": [], "highlight": {"max_fragments": 3}}))
                .unwrap();
        assert_eq!(req.highlight.unwrap().max_fragments, 3);
        // Absent means no highlighting, and `explain` is off unless asked.
        let req: TextSearchRequest = serde_json::from_value(json!({"clauses": []})).unwrap();
        assert!(req.highlight.is_none() && !req.explain);
    }

    #[test]
    fn ann_dto_hnsw_emits_only_hnsw_knobs() {
        let v = serde_json::to_value(AnnDto::from(AnnConfig::hnsw())).unwrap();
        assert_eq!(v["kind"], "Hnsw");
        assert!(v.get("m").is_some());
        assert!(v.get("ef_search").is_some());
        // IVF-only knobs are skipped for an HNSW index.
        assert!(v.get("n_lists").is_none());
        assert!(v.get("n_probe").is_none());
    }

    #[test]
    fn ann_dto_ivf_emits_only_ivf_knobs() {
        let v = serde_json::to_value(AnnDto::from(AnnConfig::ivf())).unwrap();
        assert_eq!(v["kind"], "Ivf");
        assert!(v.get("n_probe").is_some());
        // HNSW-only knobs are skipped for an IVF index.
        assert!(v.get("m").is_none());
        assert!(v.get("ef_search").is_none());
    }

    /// The externally-tagged JSON every SDK mirrors by hand. Asserted here because the
    /// crate's own round-trip is bincode, which would not catch a change in the JSON
    /// spelling — and a silent change there breaks three clients at once.
    #[test]
    fn value_json_spelling_is_stable() {
        let cases = [
            (Value::Null, serde_json::json!("Null")),
            (Value::Str("x".into()), serde_json::json!({ "Str": "x" })),
            (Value::Int(42), serde_json::json!({ "Int": 42 })),
            (Value::Bool(true), serde_json::json!({ "Bool": true })),
            (
                Value::List(vec!["a".into()]),
                serde_json::json!({ "List": ["a"] }),
            ),
            (Value::Float(1.5), serde_json::json!({ "Float": 1.5 })),
            (
                Value::DateTime(1_700_000_000_000),
                serde_json::json!({ "DateTime": 1_700_000_000_000i64 }),
            ),
        ];
        for (value, want) in cases {
            let got = serde_json::to_value(&value).unwrap();
            assert_eq!(got, want, "wire spelling changed for {value:?}");
            let back: Value = serde_json::from_value(got).unwrap();
            assert_eq!(back, value);
        }
    }

    /// A whole number sent as `Float` must stay a `Float` — JSON writes 1.0 as `1.0`,
    /// but a client that reads it back as `Int` would break same-type comparison.
    #[test]
    fn a_whole_float_does_not_decode_as_an_int() {
        let json = serde_json::to_string(&Value::Float(1.0)).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&json).unwrap(),
            Value::Float(1.0)
        );
    }

    /// A `rank_by` body names only what it changes; the rest of `Decay` defaults, and
    /// `missing` defaults to `1.0` so an undated record is not penalized (nidus-m50.15 #8).
    #[test]
    fn a_rank_by_body_defaults_every_knob_it_omits() {
        let req: SearchRequest = serde_json::from_value(json!({
            "query": [1.0],
            "rank_by": {"Decay": {"field": "ts", "origin": 1_700_000_000_000i64}}
        }))
        .unwrap();
        let Some(crate::RankBy::Decay(d)) = req.rank_by else {
            panic!("expected a Decay expression");
        };
        assert_eq!(d.field, "ts");
        assert_eq!(d.scale, 7 * 86_400_000);
        assert_eq!((d.decay, d.lambda, d.missing), (0.5, 1.0, 1.0));
    }

    #[test]
    fn the_new_ranking_knobs_are_absent_by_default() {
        let req: SearchRequest = serde_json::from_value(json!({"query": [1.0]})).unwrap();
        assert!(req.rank_by.is_none());
        assert!(req.limit_per.is_none());
        let req: ListRequest = serde_json::from_value(json!({})).unwrap();
        assert!(req.order_by.is_none());
        let req: HybridSearchRequest =
            serde_json::from_value(json!({"vector": [1.0], "field": "b", "text": "x"})).unwrap();
        assert_eq!((req.vector_weight, req.text_weight), (1.0, 1.0));
    }

    #[test]
    fn diversity_is_absent_by_default_and_snake_case_on_the_wire() {
        let req: SearchRequest = serde_json::from_value(json!({"query": [1.0]})).unwrap();
        assert!(req.diversity.is_none());
        let req: SearchRequest =
            serde_json::from_value(json!({"query": [1.0], "diversity": 0.0})).unwrap();
        // `0.0` is a meaningful lambda (pure spread), not an absent one.
        assert_eq!(req.diversity, Some(0.0));
        let req: TextSearchRequest =
            serde_json::from_value(json!({"field": "body", "query": "x", "diversity": 0.5}))
                .unwrap();
        assert_eq!(req.diversity, Some(0.5));
        let req: SimilarRequest =
            serde_json::from_value(json!({"collection": "c", "id": "i", "diversity": 1.0}))
                .unwrap();
        assert_eq!(req.diversity, Some(1.0));
    }

    #[test]
    fn expand_fills_the_reserved_chunk_attrs_from_a_bare_radius() {
        let req: SearchRequest =
            serde_json::from_value(json!({"query": [1.0], "expand": {"radius": 2}})).unwrap();
        let e: Expand = req.expand.unwrap().into();
        assert_eq!(e, Expand::new(2));

        // A caller with its own chunk attrs overrides field by field.
        let req: SearchRequest = serde_json::from_value(json!({
            "query": [1.0],
            "expand": {"radius": 1, "parent_field": "doc", "text_field": "body"}
        }))
        .unwrap();
        let e: Expand = req.expand.unwrap().into();
        assert_eq!(e.parent_field, "doc");
        assert_eq!(e.text_field, "body");
        assert_eq!(e.index_field, crate::META_CHUNK_INDEX);
    }

    #[cfg(feature = "memory")]
    #[test]
    fn rollup_defaults_to_the_best_chunk_per_document() {
        let req: RecallRequest =
            serde_json::from_value(json!({"query": "hi", "rollup": {"neighbours": 1}})).unwrap();
        let r: crate::Rollup = req.rollup.unwrap().into();
        assert_eq!(r, crate::Rollup::new(1));
    }

    /// An unexpanded hit must serialize byte-identically to a nidus without expansion, so an
    /// old client never sees a key it does not know.
    #[test]
    fn context_is_absent_unless_the_query_expanded() {
        let plain = HitDto::from(Hit::new("c", "a", 1.0, BTreeMap::new()));
        let json = serde_json::to_value(&plain).unwrap();
        assert!(json.get("context").is_none(), "{json}");

        let mut hit = Hit::new("c", "a", 1.0, BTreeMap::new());
        hit.context = Some("widened".to_string());
        let json = serde_json::to_value(HitDto::from(hit)).unwrap();
        assert_eq!(json["context"], json!("widened"));
    }

    #[test]
    fn limit_per_and_order_by_wire_spellings_are_stable() {
        let req: SearchRequest = serde_json::from_value(json!({
            "query": [1.0], "limit_per": {"field": "path", "max": 2}
        }))
        .unwrap();
        assert_eq!(req.limit_per, Some(crate::LimitPer::new("path", 2)));

        let req: ListRequest =
            serde_json::from_value(json!({"order_by": {"field": "ts", "descending": true}}))
                .unwrap();
        assert_eq!(req.order_by, Some(crate::OrderBy::desc("ts")));
        // `descending` defaults to ascending.
        let req: ListRequest =
            serde_json::from_value(json!({"order_by": {"field": "ts"}})).unwrap();
        assert_eq!(req.order_by, Some(crate::OrderBy::asc("ts")));
    }

    #[test]
    fn an_aggregation_serializes_count_and_tagged_sums() {
        let out = AggregationDto::from(crate::Aggregation {
            count: 3,
            sums: BTreeMap::from([("bytes".to_string(), Value::Int(42))]),
            groups: Vec::new(),
            groups_truncated: false,
        });
        assert_eq!(
            serde_json::to_value(&out).unwrap(),
            json!({"count": 3, "sums": {"bytes": {"Int": 42}}})
        );
    }

    #[test]
    fn group_predicate_json_spelling_is_stable() {
        let p = Predicate::Not(Box::new(Predicate::Any(vec![Predicate::Contains(
            "tags".into(),
            Value::Str("wip".into()),
        )])));
        assert_eq!(
            serde_json::to_value(&p).unwrap(),
            serde_json::json!({ "Not": { "Any": [{ "Contains": ["tags", { "Str": "wip" }] }] } })
        );
    }

    fn dummy_plan() -> QueryPlan {
        QueryPlan {
            path: crate::QueryPath::Exact,
            rows_scanned: Some(3),
            candidates: None,
            narrowing: crate::Narrowing::Inactive,
            timings: crate::Timings::default(),
        }
    }

    #[test]
    fn a_search_response_with_no_plan_is_a_bare_array() {
        // The compatibility contract: an old client parsing a plain `Vec<HitDto>` must see
        // no change when `plan` was never asked for.
        let out = serde_json::to_value(SearchResponse::new(vec![], None)).unwrap();
        assert!(out.is_array());
        assert_eq!(out, json!([]));
    }

    #[test]
    fn a_search_response_with_a_plan_is_an_object_carrying_both_keys() {
        let out = serde_json::to_value(SearchResponse::new(vec![], Some(dummy_plan()))).unwrap();
        assert!(out.is_object());
        assert!(out.get("hits").unwrap().is_array());
        assert_eq!(out.get("plan").unwrap().get("path").unwrap(), "exact");
    }
}
