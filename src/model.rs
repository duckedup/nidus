//! Shared data vocabulary used across nidus modules.
//!
//! Pure type definitions plus serde derives. *Behavior* lives in the modules that
//! own it — `filter` evaluates a [`Filter`], `log` (de)serializes an [`Op`], etc.
//! This module is the single source of truth for the types those modules share, so
//! they can be built independently and still agree on shapes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::fts::Language;

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

/// Which quantization scheme the store maintains for the search first pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantKind {
    /// int8 scalar quantization — 4× smaller than f32, valid for any distance metric.
    Int8,
    /// Binary sign-bit quantization — 32× smaller than f32, with a Hamming first pass.
    /// **Cosine only:** sign codes approximate *angular* similarity and discard
    /// magnitude, so they are not a sound ranking proxy for dot-product or Euclidean.
    Binary,
}

/// Configuration for vector quantization. When enabled, the store maintains an
/// in-memory quantized matrix for faster first-pass scoring, then re-ranks the top
/// candidates using the original f32 vectors for accuracy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quantization {
    /// Which quantization scheme drives the first pass.
    pub kind: QuantKind,
    /// Overscan factor: the first pass selects `top_k * rescore` candidates, then the
    /// f32 rerank picks the true top-k. Higher = better recall, slower. Binary codes
    /// are coarser than int8, so [`Quantization::binary`] defaults to a larger factor.
    pub rescore: usize,
}

impl Quantization {
    /// int8 scalar quantization (overscan 4). Valid for any distance metric.
    pub fn int8() -> Self {
        Self {
            kind: QuantKind::Int8,
            rescore: 4,
        }
    }

    /// Binary sign-bit quantization (overscan 16). **Cosine only.** The coarser proxy
    /// warrants a larger default overscan than int8.
    pub fn binary() -> Self {
        Self {
            kind: QuantKind::Binary,
            rescore: 16,
        }
    }

    /// Set the overscan factor (clamped to at least 1).
    pub fn rescore(mut self, n: usize) -> Self {
        self.rescore = n.max(1);
        self
    }
}

impl Default for Quantization {
    /// int8 scalar quantization — the original default, unchanged.
    fn default() -> Self {
        Self::int8()
    }
}

/// Which approximate-nearest-neighbour index the store builds when ANN search is
/// enabled via [`crate::Config::ann`]. ANN is an **opt-in** mode: with no `ann`
/// configured, search is exact brute-force (the default), and none of this applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnnKind {
    /// Hierarchical Navigable Small World graph. Native incremental insert (matches
    /// nidus's append-only upsert), high recall, no training pass. The default.
    Hnsw,
    /// Inverted-file index: k-means centroids partition the space into lists; a query
    /// probes the nearest few lists. Lower edge memory than HNSW, but its centroids
    /// are fit from the data present at build time, so heavy incremental growth drifts
    /// until the next [`crate::Nidus::compact`] rebuild.
    Ivf,
}

/// Configuration for approximate-nearest-neighbour search. When set on
/// [`crate::Config::ann`] the store maintains an in-RAM ANN index and `search`
/// consults it — walking the index for an over-fetched candidate set, then applying
/// the scope/filter/`min_score` and an exact f32 rerank. Approximate: recall is
/// traded for speed past brute-force's comfort zone (≫ a few million vectors).
///
/// Construct with [`AnnConfig::hnsw`] or [`AnnConfig::ivf`] and adjust via the
/// setters. ANN may be combined with [`Quantization`]: the index walk then scores
/// quantized codes for cheaper candidate selection, and the exact f32 rerank over the
/// resulting candidates restores accuracy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnnConfig {
    /// Which index drives the candidate walk.
    pub kind: AnnKind,
    /// HNSW: max neighbours kept per node above layer 0 (layer 0 keeps `2 * m`).
    /// Higher = better recall, more memory. Ignored for IVF.
    pub m: usize,
    /// HNSW: beam width used while *building* the graph. Higher = better-connected
    /// graph (better recall), slower inserts. Ignored for IVF.
    pub ef_construction: usize,
    /// HNSW: beam width used while *searching*. The effective beam is
    /// `max(ef_search, top_k * overscan)`. Higher = better recall, slower queries.
    /// Ignored for IVF.
    pub ef_search: usize,
    /// IVF: number of k-means centroids (inverted lists). `0` = pick `~sqrt(n)` at
    /// build time. Ignored for HNSW.
    pub n_lists: usize,
    /// IVF: how many of the nearest lists a query scans. Higher = better recall,
    /// slower queries. Ignored for HNSW.
    pub n_probe: usize,
    /// Over-fetch multiple: the walk collects `top_k * overscan` candidates before the
    /// scope/filter/`min_score` post-filter and f32 rerank, so a metadata filter or a
    /// collection-subset scope still has survivors to rank. Higher = better recall
    /// under selective filters, slower queries.
    pub overscan: usize,
    /// Seed for the index's PRNG (HNSW level assignment, IVF centroid init), so a
    /// build is deterministic and tests are reproducible.
    pub seed: u64,
}

impl AnnConfig {
    /// HNSW with sensible defaults (`m = 16`, `ef_construction = 200`,
    /// `ef_search = 64`, `overscan = 4`). The default ANN index.
    pub fn hnsw() -> Self {
        Self {
            kind: AnnKind::Hnsw,
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            n_lists: 0,
            n_probe: 8,
            overscan: 4,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// IVF with sensible defaults (`n_lists = 0` → `~sqrt(n)`, `n_probe = 8`,
    /// `overscan = 4`).
    pub fn ivf() -> Self {
        Self {
            kind: AnnKind::Ivf,
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            n_lists: 0,
            n_probe: 8,
            overscan: 4,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Set the HNSW max-neighbours-per-node (clamped to at least 1).
    pub fn m(mut self, m: usize) -> Self {
        self.m = m.max(1);
        self
    }

    /// Set the HNSW build beam width (clamped to at least 1).
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef.max(1);
        self
    }

    /// Set the HNSW search beam width (clamped to at least 1).
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef.max(1);
        self
    }

    /// Set the IVF centroid count (`0` = auto `~sqrt(n)`).
    pub fn n_lists(mut self, n: usize) -> Self {
        self.n_lists = n;
        self
    }

    /// Set the IVF probe count (clamped to at least 1).
    pub fn n_probe(mut self, n: usize) -> Self {
        self.n_probe = n.max(1);
        self
    }

    /// Set the candidate over-fetch multiple (clamped to at least 1).
    pub fn overscan(mut self, n: usize) -> Self {
        self.overscan = n.max(1);
        self
    }

    /// Set the build PRNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
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

/// A document: a caller-supplied id, an **optional** embedding, and typed metadata.
///
/// `vector` is `None` for a **text-only** document — one with no embedding, indexed and
/// retrieved purely by full-text search and metadata. Such a doc occupies no row in the
/// vector matrix and never appears in a vector `search`; it coexists in the same
/// collection as vector-bearing docs. When `Some`, the vector's length must equal the
/// store dimension. Use [`Record::new`] for a vector doc and [`Record::text_only`] for a
/// text-only one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Caller-supplied identity; the upsert key (idempotent within a collection).
    pub id: String,
    /// The embedding, or `None` for a text-only doc. When `Some`, length must equal the
    /// store dimension. Over the wire / in backups the field may be omitted (→ `None`)
    /// and is elided when absent, so a text-only doc is just `{ id, attrs }`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    /// Arbitrary typed metadata.
    pub attrs: BTreeMap<String, Value>,
}

impl Record {
    /// A vector-bearing document. `vector`'s length must equal the store dimension.
    pub fn new(id: impl Into<String>, vector: Vec<f32>, attrs: BTreeMap<String, Value>) -> Self {
        Self {
            id: id.into(),
            vector: Some(vector),
            attrs,
        }
    }

    /// A text-only document — no embedding. Indexed and retrieved by full-text search
    /// and metadata only; never appears in a vector `search`.
    pub fn text_only(id: impl Into<String>, attrs: BTreeMap<String, Value>) -> Self {
        Self {
            id: id.into(),
            vector: None,
            attrs,
        }
    }
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

/// A full-text query: the indexed `field` to search and the raw query `text`. The text
/// is analyzed (lowercase → tokenize → stopword → stem) with the field's configured
/// language at query time, exactly as documents were at index time, so a query term
/// matches a stored term when they share a stem.
#[derive(Clone, Debug)]
pub struct FtsQuery {
    /// The full-text-indexed attribute field to search (declared in the FTS schema).
    pub field: String,
    /// Raw query text.
    pub text: String,
}

impl FtsQuery {
    /// A query over `field` for `text`.
    pub fn new(field: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            text: text.into(),
        }
    }
}

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
    /// Upsert a **text-only** document — no embedding, so no `row` into the data
    /// segment. Appended after the original variants so existing logs (which never
    /// contain it) still decode: bincode tags enum variants by declaration index, so
    /// new variants must only ever be added at the end.
    UpsertText {
        collection: String,
        id: String,
        attrs: BTreeMap<String, Value>,
    },
    /// Declare a collection's full-text-indexed fields (the FTS schema). Replayed on
    /// open to rebuild the inverted index; re-emitted by `compact`. Appended at the end
    /// for the same forward-compatibility reason as `UpsertText`.
    SetFtsSchema {
        collection: String,
        fields: Vec<(String, Language)>,
    },
}
