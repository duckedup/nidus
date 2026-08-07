//! Wire types for the HTTP API and the CLI's JSON I/O.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{AnnConfig, AnnKind, Filter, Footprint, Hit, ListOpts, Record, Value};

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

/// The default result count. `pub(super)` so the MCP tools share it rather than picking
/// their own, which would answer one query two different ways depending on the surface.
pub(super) fn default_top_k() -> usize {
    10
}

/// The largest `offset + top_k` any request surface accepts. Past this a request is an
/// allocation demand rather than a query — no store returns ten thousand hits usefully — and
/// the bounded top-k kernel would otherwise be handed a `k` it must defend against itself.
pub(super) const MAX_TOP_K: usize = 10_000;

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

/// Body of `POST /text-search` (BM25). An empty `scope` searches every collection.
#[derive(Debug, Deserialize)]
pub struct TextSearchRequest {
    pub field: String,
    pub query: String,
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
}

/// Body of `POST /hybrid-search`: fuse a vector query and a BM25 text query (RRF).
#[derive(Debug, Deserialize)]
pub struct HybridSearchRequest {
    pub vector: Vec<f32>,
    pub field: String,
    pub text: String,
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
}

/// Body of `POST /collections/{name}/fts-schema`: the attribute fields to full-text
/// index. Every field uses the US English analyzer — the only [`Language`] today.
#[derive(Debug, Deserialize)]
pub struct FtsSchemaRequest {
    pub fields: Vec<String>,
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
}

/// Serializable mirror of [`crate::Hit`] (which carries no serde derive).
#[derive(Debug, Serialize)]
pub struct HitDto {
    pub collection: String,
    pub id: String,
    pub score: f32,
    pub attrs: BTreeMap<String, Value>,
}

impl From<Hit> for HitDto {
    fn from(h: Hit) -> Self {
        Self {
            collection: h.collection,
            id: h.id,
            score: h.score,
            attrs: h.attrs,
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
}

impl From<Footprint> for FootprintDto {
    fn from(f: Footprint) -> Self {
        Self {
            rows: f.rows,
            dead_rows: f.dead_rows,
            dimension: f.dimension,
            vector_bytes: f.vector_bytes,
            doc_count: f.doc_count,
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
    use super::*;
    use crate::Predicate;

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
}
