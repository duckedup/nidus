//! Shared data vocabulary used across nidus modules.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::annotate::{Annotations, HighlightOpts};
use crate::findex::FilterIndexField;
use crate::fts::{FtsField, Language};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnKind {
    /// Hierarchical Navigable Small World graph. Native incremental insert (matches
    /// nidus's append-only upsert), high recall, no training pass. The default.
    Hnsw,
    /// Inverted-file index: k-means centroids partition the space into lists and a query probes the
    /// nearest few. Lower edge memory than HNSW, but centroids are fit at build time, so heavy
    /// incremental growth drifts until the next [`crate::Nidus::compact`].
    Ivf,
}

/// Configuration for approximate-nearest-neighbour search. Set on [`crate::Config::ann`], the store
/// maintains an in-RAM index and `search` walks it for an over-fetched candidate set, then applies
/// scope/filter/`min_score` and an exact f32 rerank — recall traded for speed past brute force.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Over-fetch multiple: the walk collects `top_k * overscan` candidates before the post-filter
    /// and rerank, so a metadata filter or subset scope still has survivors to rank. Higher means
    /// better recall under selective filters and slower queries.
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Str(String),
    Int(i64),
    Bool(bool),
    List(Vec<String>),
    /// A double. Comparison is IEEE: `NaN` matches nothing (not even itself), and
    /// `0.0 == -0.0`. Distinct from [`Value::Int`] — comparisons are same-type only.
    Float(f64),
    /// A UTC instant as **epoch milliseconds**. There is no timezone and no local
    /// time: an instant is absolute, and rendering it is the caller's business.
    DateTime(i64),
    // Append-only: bincode encodes the variant index, so inserting above this line
    // would silently reinterpret every value in every existing store.
}

/// A document: a caller-supplied id, an **optional** embedding, and typed metadata.
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Predicate {
    /// `attrs[key] == value`.
    Eq(String, Value),
    /// `attrs[key]` is present and `!= value`.
    Ne(String, Value),
    /// `attrs[key]` is a [`Value::Str`] matching the glob pattern.
    Glob(String, String),
    /// [`Predicate::Glob`], ignoring ASCII case on both sides. Non-ASCII is not folded.
    IGlob(String, String),
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
    /// `attrs[key]` is a [`Value::List`] containing `value`. Lists hold strings, so a
    /// non-[`Value::Str`] needle never matches. Substring matching on a plain `Str` is
    /// [`Predicate::Glob`], not this.
    Contains(String, Value),
    /// `attrs[key]` is a [`Value::List`] *not* containing `value`. Like [`Predicate::Ne`],
    /// it requires the key present and list-typed.
    NotContains(String, Value),
    /// `attrs[key]` is a [`Value::List`] sharing at least one element with the set.
    /// "Contains all of" is [`Predicate::All`] over several [`Predicate::Contains`].
    ContainsAny(String, Vec<Value>),
    /// Every sub-predicate holds. Empty is `true`, matching [`Filter`]'s empty case.
    All(Vec<Predicate>),
    /// At least one sub-predicate holds. Empty is `false` — the identity for OR.
    Any(Vec<Predicate>),
    /// The sub-predicate does *not* hold. Note this differs from the negative leaf
    /// predicates on an absent key: `Not(Eq(k, v))` is true when `k` is missing, whereas
    /// `Ne(k, v)` is false. Use `Ne`/`NotIn`/`NotContains` to require presence.
    Not(Box<Predicate>),
    /// `attrs[key]` is within N Levenshtein edits of the string, ASCII-case-folded on both
    /// sides. A `List` matches if any element does. An N above `MAX_FUZZY_EDITS` (8) is an
    /// error, not a clamp.
    Fuzzy(String, String, usize),
    /// Every token of the query text appears among `attrs[key]`'s tokens, in any order. A
    /// `List` matches if any single element does. Tokens are ASCII-case-folded runs of
    /// alphanumerics — see `SPEC.md` §7.4.
    ContainsAllTokens(String, String),
    /// At least one token of the query text appears among `attrs[key]`'s tokens. An empty
    /// query never matches, the identity [`Predicate::Any`] and `In` already take.
    ContainsAnyToken(String, String),
    /// The query's tokens appear consecutively and in order in `attrs[key]` — a phrase
    /// match. A `List` matches if any single element carries the whole phrase.
    ContainsTokenSequence(String, String),
    /// `attrs[key]` matches the regular expression, **anchored at both ends** like `Glob`
    /// (`.*` opts back into a substring search). Case folding is the pattern's own `(?i)`.
    /// An unparseable pattern is a caller-facing error — see `SPEC.md` §7.5.
    Regex(String, String),
}

/// A conjunction (AND) of predicates. An empty filter matches everything. Arbitrary
/// boolean shapes nest through the [`Predicate::All`]/[`Predicate::Any`]/[`Predicate::Not`]
/// group variants rather than through this outer list.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Filter(pub Vec<Predicate>);

/// How several [`FtsClause`]s fold into one text score (nidus-m50.10).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FtsCombine {
    /// Add every matched clause's BM25 score, so a doc hit on title *and* body outranks
    /// one hit on either alone. The default.
    #[default]
    #[serde(alias = "sum")]
    Sum,
    /// Take the strongest matched clause, so a long body cannot out-accumulate a precise
    /// title match.
    #[serde(alias = "max")]
    Max,
}

/// One clause of an [`FtsQuery`]: an indexed `field` and the raw query `text` for it. Each
/// clause carries its own text, so `title:"rust"` + `body:"async runtime"` is one query.
#[derive(Clone, Debug, PartialEq)]
pub struct FtsClause {
    /// The full-text-indexed attribute field to search (declared in the FTS schema).
    pub field: String,
    /// Raw query text for this field.
    pub text: String,
}

impl FtsClause {
    /// A clause searching `field` for `text`.
    pub fn new(field: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            text: text.into(),
        }
    }
}

/// A full-text query: one or more [`FtsClause`]s folded by [`FtsCombine`], plus optional
/// highlighting. Each clause's text is analyzed with *that field's* analyzer at query time,
/// exactly as documents were at index time, so a query term matches a term sharing its stem.
#[derive(Clone, Debug)]
pub struct FtsQuery {
    /// The clauses to score. At least one — an empty list is an error, not a match-all.
    pub clauses: Vec<FtsClause>,
    /// How the clauses' scores fold together.
    pub combine: FtsCombine,
    /// Return highlighted fragments of each clause field's stored text. `None` (the
    /// default) returns no fragments and does no extra work.
    pub highlight: Option<HighlightOpts>,
}

impl FtsQuery {
    /// A single-clause query over `field` for `text` — the original shape, unchanged.
    pub fn new(field: impl Into<String>, text: impl Into<String>) -> Self {
        Self::multi([FtsClause::new(field, text)])
    }

    /// A query over several clauses, combined by [`FtsCombine::Sum`] unless
    /// [`combine`](Self::combine) says otherwise.
    pub fn multi(clauses: impl IntoIterator<Item = FtsClause>) -> Self {
        Self {
            clauses: clauses.into_iter().collect(),
            combine: FtsCombine::default(),
            highlight: None,
        }
    }

    /// Set how the clauses fold together.
    pub fn combine(mut self, combine: FtsCombine) -> Self {
        self.combine = combine;
        self
    }

    /// Return highlighted fragments of the matched text.
    pub fn highlight(mut self, opts: HighlightOpts) -> Self {
        self.highlight = Some(opts);
        self
    }

    /// Reject a query with nothing to score. An empty clause list is a caller error rather
    /// than an empty result, so a client bug does not read as "the corpus has no matches".
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if self.clauses.is_empty() {
            anyhow::bail!("a full-text query must carry at least one clause");
        }
        Ok(())
    }
}

/// Which attrs a [`Hit`] carries. An enum, not a pair of lists, so "include *and* exclude"
/// is unrepresentable rather than a precedence rule nobody remembers (nidus-m50.15); the
/// HTTP layer answers `400` for the wire form that sends both.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    /// Every attr the record has — the default, and byte-identical to pre-projection nidus.
    #[default]
    All,
    /// Only the named attrs. A named attr the record lacks is simply absent from the hit.
    Include(Vec<String>),
    /// Every attr except the named ones.
    Exclude(Vec<String>),
}

impl Projection {
    /// Carry only these attrs.
    pub fn include<S: Into<String>>(keys: impl IntoIterator<Item = S>) -> Self {
        Self::Include(keys.into_iter().map(Into::into).collect())
    }

    /// Carry every attr but these.
    pub fn exclude<S: Into<String>>(keys: impl IntoIterator<Item = S>) -> Self {
        Self::Exclude(keys.into_iter().map(Into::into).collect())
    }

    /// Materialize a hit's attrs from the live record's map. Clones only the values that
    /// survive the projection, so an excluded 10 KB body is never copied (nidus-m50.7).
    pub(crate) fn apply(&self, attrs: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
        match self {
            Self::All => attrs.clone(),
            Self::Include(keys) => keys
                .iter()
                .filter_map(|k| attrs.get_key_value(k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            Self::Exclude(keys) => attrs
                .iter()
                .filter(|(k, _)| !keys.iter().any(|drop| drop == *k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }
}

fn default_scale() -> i64 {
    7 * 24 * 60 * 60 * 1000
}

fn default_decay() -> f32 {
    0.5
}

fn default_lambda() -> f32 {
    1.0
}

/// A record with no usable timestamp is **not** penalized, so switching decay on never
/// silently buries data that predates the field (nidus-m50.15 #8).
fn default_missing() -> f32 {
    1.0
}

/// Recency decay over a timestamp attribute. The penalty is **subtracted** from the base
/// score, never multiplied, so it is valid for every [`Distance`] metric and for negative
/// scores alike (nidus-m50.15 #7). See `SPEC.md` §7.6.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decay {
    /// The timestamp attribute: a [`Value::DateTime`] or a [`Value::Int`], epoch milliseconds.
    pub field: String,
    /// "Now", in epoch milliseconds. Ages are measured back from here rather than from the
    /// wall clock, so the same query against an unchanged store ranks the same way twice.
    pub origin: i64,
    /// The age in milliseconds at which the factor equals `decay`. Must be positive.
    #[serde(default = "default_scale")]
    pub scale: i64,
    /// The factor reached at exactly `scale` old — the default `0.5` makes `scale` a
    /// half-life. Must be in `(0, 1)`.
    #[serde(default = "default_decay")]
    pub decay: f32,
    /// Score a fully-decayed hit gives up: the penalty is `lambda * (1 - factor)`.
    #[serde(default = "default_lambda")]
    pub lambda: f32,
    /// The factor for a record whose `field` is missing or not a timestamp. Defaults to
    /// `1.0` — no penalty. Must be in `[0, 1]`.
    #[serde(default = "default_missing")]
    pub missing: f32,
}

impl Decay {
    /// Decay over `field`, aged from `origin` (epoch ms), halving every `scale` ms.
    pub fn new(field: impl Into<String>, origin: i64, scale: i64) -> Self {
        Self {
            field: field.into(),
            origin,
            scale,
            decay: default_decay(),
            lambda: default_lambda(),
            missing: default_missing(),
        }
    }

    /// Set the factor reached at exactly `scale` old (must be in `(0, 1)`).
    pub fn decay(mut self, decay: f32) -> Self {
        self.decay = decay;
        self
    }

    /// Set how much score a fully-decayed hit gives up.
    pub fn lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set the factor used when the timestamp attribute is missing or unusable.
    pub fn missing(mut self, missing: f32) -> Self {
        self.missing = missing;
        self
    }
}

/// A ranking expression layered over the store's distance metric. `None` on
/// [`SearchOpts::rank_by`] is the bare metric — the ranking nidus has always returned.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RankBy {
    /// Subtract a recency penalty from every base score. See [`Decay`].
    Decay(Decay),
}

/// Sort a [`crate::Nidus::list`] by an attribute instead of storage order. Values of a
/// different type than the first orderable one, unorderable values (`Null`/`List`/`NaN`), and
/// records missing the attribute sort into one trailing bucket (nidus-m50.15 #10).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderBy {
    /// The attribute to sort on.
    pub field: String,
    /// Sort descending. The trailing bucket stays trailing either way.
    #[serde(default)]
    pub descending: bool,
}

impl OrderBy {
    /// Ascending order over `field`.
    pub fn asc(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            descending: false,
        }
    }

    /// Descending order over `field`.
    pub fn desc(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            descending: true,
        }
    }
}

/// Cap how many hits may carry any one value of an attribute — "at most 2 hits per file".
/// Records **missing** the attribute form one shared group, so an absent value cannot
/// bypass the cap (nidus-m50.15 #14). Approximate; see `SPEC.md` §7.7.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LimitPer {
    /// The attribute whose distinct values define the groups.
    pub field: String,
    /// Maximum hits kept per distinct value. Must be at least 1.
    pub max: usize,
}

impl LimitPer {
    /// At most `max` hits per distinct value of `field`.
    pub fn new(field: impl Into<String>, max: usize) -> Self {
        Self {
            field: field.into(),
            max,
        }
    }
}

/// Attr key holding the raw remembered text. Canonical home is here (unconditional) so
/// [`RerankOpts::default`] can reference it without the `memory` feature; `memory::META_TEXT`
/// re-exports this constant so its public path is unchanged.
pub const META_TEXT: &str = "nidus.text";

/// Attr keys stamped by `src/chunk`. Live here (not `memory.rs`) so the ungated `chunk`
/// module and the `store/` rollup reach them without `memory`/`embed`. `chunk_index` is
/// `Value::Int` (so `Predicate::Ge` can address it); `parent_id` is `Value::Str`.
pub const META_PARENT_ID: &str = "nidus.parent_id";
pub const META_CHUNK_INDEX: &str = "nidus.chunk_index";

/// Default overscan (see [`RerankOpts::overscan`]): a `top_k=10` query reranks 100 candidates.
pub const DEFAULT_RERANK_OVERSCAN: usize = 10;

/// The post-ranking cross-encoder stage (SPEC §7). Plain config: the reranker itself is
/// async and lives at the edge in `crate::rerank`, never in a `SearchOpts`.
#[derive(Clone, Debug, PartialEq)]
pub struct RerankOpts {
    /// Multiplier on the page depth for the candidate window. Clamped to >= 1.
    pub overscan: usize,
    /// Attr carrying each candidate's text. Defaults to `nidus.text`.
    pub text_attr: String,
}

impl Default for RerankOpts {
    fn default() -> Self {
        Self {
            overscan: DEFAULT_RERANK_OVERSCAN,
            text_attr: META_TEXT.to_string(),
        }
    }
}

/// What [`crate::Nidus::aggregate`] computes over the filter-matching records. Answered from
/// the in-RAM index alone: no [`Record`] is built and no vector is read.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AggregateOpts {
    /// Metadata filter; the default matches every record.
    pub filter: Filter,
    /// Attributes to sum. A missing or non-numeric value is skipped, not counted as zero.
    pub sum: Vec<String>,
    /// Split the answer into one [`Group`] per distinct value of this attribute, alongside the
    /// whole-scope totals. `None` reports the totals alone.
    pub group_by: Option<String>,
}

/// The answer to an [`AggregateOpts`]: how many records matched, plus one tagged [`Value`]
/// per requested sum — `Int` while every addend was an `Int`, else `Float` (nidus-m50.15 #15).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aggregation {
    /// Records matching the filter across the scope.
    pub count: u64,
    /// One entry per [`AggregateOpts::sum`] field, in the same set.
    pub sums: BTreeMap<String, Value>,
    /// One row per distinct [`AggregateOpts::group_by`] value; empty when none was asked for.
    /// Ordered by `count` descending, ties broken by the value for a deterministic answer.
    pub groups: Vec<Group>,
    /// Set when distinct values outran the group cap and later ones were dropped — so a
    /// truncated answer is never mistaken for a complete one.
    pub groups_truncated: bool,
}

/// One distinct value of [`AggregateOpts::group_by`] and the aggregates over just its records.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Group {
    /// The distinct value, or `None` for the records missing the attribute entirely — which
    /// is a different group from those holding [`Value::Null`].
    pub value: Option<Value>,
    /// Records in this group.
    pub count: u64,
    /// The same sum fields as [`Aggregation::sums`], over this group's records alone.
    pub sums: BTreeMap<String, Value>,
}

/// Query parameters for a search.
#[derive(Clone, Debug, Default)]
pub struct SearchOpts {
    /// Maximum number of results.
    pub top_k: usize,
    /// How many top-ranked results to skip before collecting, for pagination. The ranking is
    /// still computed `offset + top_k` deep, and a page is stable only against an unchanging
    /// store (SPEC §7).
    pub offset: usize,
    /// Pre-scoring metadata filter (applied before the dot product).
    pub filter: Filter,
    /// Drop results scoring below this value, on the metric's own scale (cosine similarity,
    /// unless a different [`Distance`] or leg applies). Evaluated **pre-rerank**: a
    /// [`RerankOpts`]-scored hit replaces `Hit::score` afterward, on an unrelated scale.
    pub min_score: Option<f32>,
    /// Force the exact brute-force scan for *this* query, bypassing the ANN walk, the segment
    /// indexes, and the quantized first pass. A guaranteed-exact answer over a narrow subset
    /// without giving up the index for everything else (nidus-m50.12).
    pub exact: bool,
    /// Which attrs the returned hits carry. Default: all of them.
    pub projection: Projection,
    /// Annotate each hit with why it matched (nidus-m50.5). On a `text_search` that is each
    /// clause's own BM25 score; vector `search` ignores it, having a single score to report.
    pub explain: bool,
    /// A ranking expression layered over the distance metric. Deliberately does **not** force
    /// the exact path (that is [`SearchOpts::exact`]): over an ANN or quantized result set it
    /// inherits that path's approximation (nidus-m50.15 #9).
    pub rank_by: Option<RankBy>,
    /// Cap the hits carrying any one value of an attribute. `None` (the default) is uncapped.
    pub limit_per: Option<LimitPer>,
    /// Rerank the candidate window with a hosted cross-encoder (`crate::rerank`, feature-gated).
    /// `None` (the default) leaves the metric ranking untouched.
    pub rerank: Option<RerankOpts>,
}

/// Query parameters for a metadata-only listing (no vector scoring).
#[derive(Clone, Debug)]
pub struct ListOpts {
    /// How many matches to skip before collecting, for pagination.
    pub offset: usize,
    /// Maximum number of records returned. Defaults to 100.
    pub limit: usize,
    /// Metadata filter; the default matches every record.
    pub filter: Filter,
    /// Which attrs the returned hits carry. Default: all of them.
    pub projection: Projection,
    /// Sort by an attribute rather than storage order — ORDER BY with no vector query.
    /// `None` (the default) keeps the stable row-then-id order.
    pub order_by: Option<OrderBy>,
}

impl Default for ListOpts {
    fn default() -> Self {
        Self {
            offset: 0,
            limit: 100,
            filter: Filter::default(),
            projection: Projection::default(),
            order_by: None,
        }
    }
}

/// Options for a hybrid (vector + BM25) search, fused with Reciprocal Rank Fusion.
#[derive(Clone, Debug)]
pub struct HybridOpts {
    /// Final result count after fusion.
    pub top_k: usize,
    /// How many fused results to skip before collecting, for pagination. Applied *after*
    /// fusion, so a page boundary falls on the fused ranking (SPEC §7).
    pub offset: usize,
    /// Metadata filter applied to *both* legs before fusion.
    pub filter: Filter,
    /// RRF rank-bias constant `k`: larger flattens the weight of top ranks. Default 60.
    pub rrf_k: f32,
    /// How deep to pull each leg before fusing (clamped up to at least `top_k`). Larger
    /// improves fusion recall at linear cost. Default 100.
    pub candidates: usize,
    /// Annotate each hit with each leg's own rank and score plus every matched BM25
    /// clause's contribution (nidus-m50.5). Default `false`.
    pub explain: bool,
    /// Weight on the vector leg's RRF contribution. Default `1.0`; both legs at `1.0`
    /// reproduces unweighted RRF bit for bit.
    pub vector_weight: f32,
    /// Weight on the BM25 leg's RRF contribution. Default `1.0`.
    pub text_weight: f32,
    /// Rerank the fused candidate window with a hosted cross-encoder (`crate::rerank`,
    /// feature-gated). `None` (the default) leaves the RRF ranking untouched.
    pub rerank: Option<RerankOpts>,
}

impl Default for HybridOpts {
    fn default() -> Self {
        Self {
            top_k: 10,
            offset: 0,
            filter: Filter::default(),
            rrf_k: 60.0,
            candidates: 100,
            explain: false,
            rerank: None,
            vector_weight: 1.0,
            text_weight: 1.0,
        }
    }
}

/// One search result. Carries its source `collection` (ids are unique only within a
/// collection) and the matched record's `attrs`, but deliberately not its vector.
/// `#[non_exhaustive]`: build one with [`Hit::new`] so added fields stay additive.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Hit {
    pub collection: String,
    pub id: String,
    pub score: f32,
    pub attrs: BTreeMap<String, Value>,
    /// Why this hit matched — per-leg sub-scores and highlighted fragments. `None` unless
    /// the query opted in (`explain` / `FtsQuery::highlight`), so the default is unchanged.
    pub annotations: Option<Annotations>,
}

impl Hit {
    /// One result from `collection`, scored `score`, carrying the record's `attrs` and no
    /// annotations (a search path attaches those afterwards, when asked).
    pub fn new(
        collection: impl Into<String>,
        id: impl Into<String>,
        score: f32,
        attrs: BTreeMap<String, Value>,
    ) -> Self {
        Self {
            collection: collection.into(),
            id: id.into(),
            score,
            attrs,
            annotations: None,
        }
    }
}

/// A cheap, allocation-free snapshot of a store's RAM/disk footprint — the hook a host uses to
/// decide whether it can afford more data before a memory ceiling. `vector_bytes` is the dominant,
/// predictable cost; the in-RAM index of ids and attrs is extra and not counted.
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
    /// Approximate heap held by the opt-in filter index (SPEC §7.4). Zero when no
    /// collection declares one — the other half of that feature's cost trade.
    pub filter_index_bytes: u64,
}

/// What an instance is within a store, for [`ClusterStatus`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Sole writer of a single-node store — holds the plain writer lock.
    Writer,
    /// Read-only opener of a single-node store — holds no lock.
    Reader,
    /// Cluster writer — holds the renewable, fenced writer lease.
    ClusterWriter,
    /// Cluster reader — lock-free, advances via `refresh()`.
    ClusterReader,
    /// In-memory store: no durability, no lock, no peers.
    InMemory,
}

/// Who this instance is and how current it is — the introspection an operator needs
/// during an incident, and what a readiness probe consults (SPEC §14.6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterStatus {
    /// What this instance is.
    pub role: Role,
    /// Whether cluster mode is on (`Config::cluster`).
    pub cluster: bool,
    /// This instance believes it holds the writer handle.
    pub holds_writer_handle: bool,
    /// **This writer has been superseded.** Every subsequent write will fail; the instance
    /// must be replaced. Latched once observed, because the condition is permanent — a
    /// fenced writer never regains the lease, it has to reopen.
    pub fenced: bool,
    /// Our fencing token (owner id) while we hold a cluster lease.
    pub lease_owner: Option<String>,
    /// The manifest commit counter this instance is serving. A reader behind the writer
    /// reports a lower number; comparing across instances shows replication lag.
    pub commit_version: u64,
    /// Seconds since this instance last took up newer state — `0` for a writer (its own
    /// state is by definition current), and for a reader the age of its last successful
    /// `refresh()` (or of its open, if it has not refreshed).
    pub staleness_secs: u64,
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
    /// Upsert a **text-only** document — no embedding, so no `row` into the data segment. Appended
    /// after the original variants so existing logs (which never contain it) still decode: bincode
    /// tags enum variants by declaration index, so new variants must only ever be added at the end.
    UpsertText {
        collection: String,
        id: String,
        attrs: BTreeMap<String, Value>,
    },
    /// **Legacy**, superseded by `SetFtsFields`: an FTS schema carrying only a language per
    /// field. Never written any more, but still decoded on replay — a log written before the
    /// BM25/analyzer params were tunable must still open, with the defaults applied.
    SetFtsSchema {
        collection: String,
        fields: Vec<(String, Language)>,
    },
    /// Declare a collection's full-text-indexed fields with their BM25 and analyzer params.
    /// Replayed on open to rebuild the inverted index; re-emitted by `compact`. Appended at
    /// the end for the same forward-compatibility reason as `UpsertText`.
    SetFtsFields {
        collection: String,
        fields: Vec<FtsField>,
    },
    /// Declare a collection's filter-indexed fields (SPEC §7.4/§7.5). Replayed on open to
    /// rebuild the index; re-emitted by `compact`. Appended at the end for the same
    /// forward-compatibility reason as `UpsertText`.
    SetFilterIndex {
        collection: String,
        fields: Vec<FilterIndexField>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appending_variants_did_not_renumber_the_existing_ones() {
        // bincode tags a variant by its **declaration index**, so inserting one anywhere
        // but the end silently reinterprets every op in every store's existing log.
        let cases: [(Op, u32); 9] = [
            (
                Op::CreateCollection {
                    collection: "c".into(),
                },
                0,
            ),
            (
                Op::DropCollection {
                    collection: "c".into(),
                },
                1,
            ),
            (
                Op::SetMeta {
                    collection: "c".into(),
                    meta: BTreeMap::new(),
                },
                2,
            ),
            (
                Op::Upsert {
                    collection: "c".into(),
                    id: "i".into(),
                    row: 0,
                    attrs: BTreeMap::new(),
                },
                3,
            ),
            (
                Op::Delete {
                    collection: "c".into(),
                    id: "i".into(),
                },
                4,
            ),
            (
                Op::UpsertText {
                    collection: "c".into(),
                    id: "i".into(),
                    attrs: BTreeMap::new(),
                },
                5,
            ),
            (
                Op::SetFtsSchema {
                    collection: "c".into(),
                    fields: vec![("body".into(), Language::English)],
                },
                6,
            ),
            (
                Op::SetFtsFields {
                    collection: "c".into(),
                    fields: vec![FtsField::new("body")],
                },
                7,
            ),
            (
                Op::SetFilterIndex {
                    collection: "c".into(),
                    fields: vec![FilterIndexField::new("body")],
                },
                8,
            ),
        ];
        for (op, want) in cases {
            let bytes = bincode::serialize(&op).unwrap();
            let tag = u32::from_le_bytes(bytes[..4].try_into().unwrap());
            assert_eq!(tag, want, "{op:?} must stay variant {want}");
        }
    }

    #[test]
    fn a_filter_index_op_round_trips() {
        let op = Op::SetFilterIndex {
            collection: "docs".into(),
            fields: vec![
                FilterIndexField::new("body"),
                FilterIndexField::new("title").trigrams(false),
            ],
        };
        let bytes = bincode::serialize(&op).unwrap();
        assert_eq!(bincode::deserialize::<Op>(&bytes).unwrap(), op);
    }

    #[test]
    fn a_log_written_before_the_filter_index_still_decodes() {
        // The forward-compatibility direction that matters: a newer nidus reading an older
        // log. Bytes produced without the new variant must decode unchanged.
        let old = Op::SetFtsFields {
            collection: "docs".into(),
            fields: vec![FtsField::new("body")],
        };
        let bytes = bincode::serialize(&old).unwrap();
        assert_eq!(bincode::deserialize::<Op>(&bytes).unwrap(), old);
    }

    #[test]
    fn a_legacy_fts_schema_op_still_decodes() {
        // The exact bytes an old nidus wrote for `SetFtsSchema { "docs", [("body", English)] }`.
        let op = Op::SetFtsSchema {
            collection: "docs".into(),
            fields: vec![("body".into(), Language::English)],
        };
        let bytes = bincode::serialize(&op).unwrap();
        assert_eq!(bincode::deserialize::<Op>(&bytes).unwrap(), op);
    }
}
