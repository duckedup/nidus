//! Shared data vocabulary used across nidus modules.
//!
//! Pure type definitions plus serde derives. *Behavior* lives in the modules that
//! own it — `filter` evaluates a [`Filter`], `log` (de)serializes an [`Op`], etc.
//! This module is the single source of truth for the types those modules share, so
//! they can be built independently and still agree on shapes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The similarity / distance metric used for scoring. Pinned at store creation
/// (stored in the data header) — reopening with a different metric is an error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Distance {
    /// Cosine similarity: vectors are unit-normalized on insert, score = dot(q, v).
    /// Range \[−1, 1\]; 1 = identical direction.
    #[default]
    Cosine,
    /// Negative squared Euclidean distance: vectors stored as-is,
    /// score = −‖q − v‖². Range (−∞, 0\]; 0 = identical.
    Euclidean,
    /// Raw dot product: vectors stored as-is, score = dot(q, v).
    /// Range (−∞, ∞); magnitude carries signal.
    DotProduct,
}

/// Configuration for int8 scalar quantization. When enabled, the store maintains
/// an in-memory int8 vector matrix for faster first-pass scoring, then re-ranks
/// the top candidates using the original f32 vectors for accuracy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quantization {
    /// Overscan factor: the int8 first-pass selects `top_k * rescore` candidates,
    /// then the f32 rerank picks the true top-k. Higher = better recall, slower.
    /// Default: 4.
    pub rescore: usize,
}

impl Default for Quantization {
    fn default() -> Self {
        Self { rescore: 4 }
    }
}

/// A typed metadata value attached to a [`Record`].
///
/// `Null` is **distinct from an absent key**: absence means "not set / not indexed",
/// while `Null` means "set, and empty/none". Callers rely on this to tell
/// not-computed apart from computed-empty (e.g. optional relation lists).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Str(String),
    Int(i64),
    Bool(bool),
    List(Vec<String>),
}

/// A document: a caller-supplied id, its embedding, and typed metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Caller-supplied identity; the upsert key (idempotent within a collection).
    pub id: String,
    /// The embedding. Length must equal the store dimension.
    pub vector: Vec<f32>,
    /// Arbitrary typed metadata.
    pub attrs: BTreeMap<String, Value>,
}

/// A single attribute predicate. Predicates are AND-combined inside a [`Filter`].
///
/// Every predicate is a *positive assertion about a present attribute*: if `key` is
/// absent from a record's attrs, **no** predicate matches it — including the negative
/// ones (`Ne`/`NotIn`) and the range ones. The comparison variants are same-type only
/// (`Int`↔`Int` numeric, `Str`↔`Str` lexical, `Bool`↔`Bool` with `false < true`);
/// a cross-type or non-orderable comparison (`Null`, `List`) never matches a range.
/// See the root `SPEC.md` §7.1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Predicate {
    /// `attrs[key] == value`.
    Eq(String, Value),
    /// `attrs[key]` is present and `!= value`.
    Ne(String, Value),
    /// `attrs[key]` is a [`Value::Str`] matching the glob pattern.
    Glob(String, String),
    /// `attrs[key]` is equal to one of the values in the set.
    In(String, Vec<Value>),
    /// `attrs[key]` is present and *not* equal to any value in the set.
    NotIn(String, Vec<Value>),
    /// `attrs[key] < value` (same-type, orderable).
    Lt(String, Value),
    /// `attrs[key] <= value` (same-type, orderable).
    Le(String, Value),
    /// `attrs[key] > value` (same-type, orderable).
    Gt(String, Value),
    /// `attrs[key] >= value` (same-type, orderable).
    Ge(String, Value),
}

/// A conjunction (AND) of predicates. An empty filter matches everything.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Filter(pub Vec<Predicate>);

/// Query parameters for a search.
#[derive(Clone, Debug, Default)]
pub struct SearchOpts {
    /// Maximum number of results.
    pub top_k: usize,
    /// Pre-scoring metadata filter (applied before the dot product).
    pub filter: Filter,
    /// Drop results scoring below this cosine similarity.
    pub min_score: Option<f32>,
}

/// One search result. Carries its source `collection` (ids are unique only within a
/// collection) and the matched record's `attrs`, but deliberately not its vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub collection: String,
    pub id: String,
    pub score: f32,
    pub attrs: BTreeMap<String, Value>,
}

/// A cheap, allocation-free snapshot of a store's RAM/disk footprint — the
/// introspection hook a host uses to decide whether it can afford more data before
/// hitting a memory ceiling. `vector_bytes` is the dominant, predictable cost; the
/// in-RAM index (ids + attrs) is extra on top and not counted here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footprint {
    /// Physical rows in the data matrix (live + not-yet-compacted dead rows).
    pub rows: u64,
    /// Rows no longer referenced (reclaimable by `compact`).
    pub dead_rows: u64,
    /// The pinned embedding dimension.
    pub dimension: usize,
    /// Bytes occupied by the vector matrix: `rows * dimension * 4`. This is what
    /// `Config::max_vector_bytes` caps.
    pub vector_bytes: u64,
    /// Live documents across all collections.
    pub doc_count: usize,
}

/// A mutating operation recorded in the op log (the commit stream). `row` indexes
/// into the data segment. The on-disk log is a sequence of framed, checksummed,
/// bincode-encoded `Op`s (see `log` module + SPEC.md §5.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Op {
    CreateCollection {
        collection: String,
    },
    DropCollection {
        collection: String,
    },
    SetMeta {
        collection: String,
        meta: BTreeMap<String, String>,
    },
    Upsert {
        collection: String,
        id: String,
        row: u64,
        attrs: BTreeMap<String, Value>,
    },
    Delete {
        collection: String,
        id: String,
    },
}
