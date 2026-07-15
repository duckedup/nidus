//! Wire types for the HTTP API and the CLI's JSON I/O.
//!
//! Request bodies deserialize into these; responses serialize out of them. The
//! core library types `Record`/`Value`/`Filter` already derive serde, so they ride
//! the wire directly. `Hit`/`Footprint` deliberately do *not* (the library keeps
//! its serde surface intentional), so we mirror them here and convert at the edge —
//! the binary layer adapts to the library, never the reverse.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{AnnConfig, AnnKind, Filter, Footprint, Hit, Record, Value};

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

fn default_top_k() -> usize {
    10
}

/// Body of `POST /search`. An empty `scope` searches every collection.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
    #[serde(default)]
    pub filter: Filter,
}

fn default_limit() -> usize {
    100
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
    pub filter: Filter,
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
    #[serde(default = "default_candidates")]
    pub candidates: usize,
}

/// Body of `POST /collections/{name}/fts-schema`: the attribute fields to full-text
/// index. Every field uses the US English analyzer — the only [`Language`] today.
///
/// Forward-compat note: when a second language ships, expressing per-field language
/// over the wire means evolving `fields` from `Vec<String>` to a typed
/// `Vec<{ field, language }>`. That is a breaking wire change, deliberately deferred
/// until there is a second language to choose (the library API already takes
/// `(field, Language)` pairs, so only this DTO + handler need to grow).
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
///
/// Text in: the server embeds `text` (optionally summarizing it first when
/// `mode` is `"summarize"`) and upserts a record under `id` with `attrs`. The
/// embedder is chosen at serve time (`--embed-provider …`); a request to this
/// route on a server started without one is a `400`.
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
///
/// Query text in, ranked hits out: the server embeds `query` and runs a vector
/// search over `collection`. Mirrors [`SearchRequest`] minus the caller-supplied
/// vector (which the server produces from `query`).
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
}
