//! Wire types for the HTTP API and the CLI's JSON I/O.
//!
//! Request bodies deserialize into these; responses serialize out of them. The
//! core library types `Record`/`Value`/`Filter` already derive serde, so they ride
//! the wire directly. `Hit`/`Footprint` deliberately do *not* (the library keeps
//! its serde surface intentional), so we mirror them here and convert at the edge —
//! the binary layer adapts to the library, never the reverse.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Filter, Footprint, Hit, Record, Value};

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

/// Body of `POST /list`. Metadata-only query (no vector). An empty `scope`
/// lists from every collection.
#[derive(Debug, Deserialize)]
pub struct ListRequest {
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
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
