// `deny`, not `forbid`: the single memory-map call in `src/data/mmap.rs` (SPEC §9/§14.6)
// carries a scoped `#[allow(unsafe_code)]`. Every other `unsafe` in the crate stays a hard
// compile error.
#![deny(unsafe_code)]
//! # nidus
//!
//! A small, pure-Rust embeddable vector store: brute-force cosine search over one
//! append-only directory, with typed metadata filters and many collections sharing one
//! embedding space. See `SPEC.md` for the full design.
//!
//! ```no_run
//! use nidus::{Nidus, Config, Record, SearchOpts, Scope};
//! use std::collections::BTreeMap;
//!
//! let mut db = Nidus::open(Config::new("/tmp/store", 3))?;
//! db.create_collection("docs")?;
//! db.upsert("docs", &[Record::new("a", vec![1.0, 0.0, 0.0], BTreeMap::new())])?;
//! let hits = db.search("docs", &[1.0, 0.0, 0.0], &SearchOpts { top_k: 5, ..Default::default() })?;
//! # anyhow::Ok(())
//! ```

mod ann;
// Opt-in result annotations: per-leg sub-scores and highlighted fragments (nidus-m50.5).
mod annotate;
pub mod backend;
// Cooperative cancellation for long scans: a request deadline frees the client, only the
// scan loop can free the CPU.
mod cancel;
// Pure text -> ordered spans chunking for the memory layer (nidus-lvo.1). Ungated: it
// must run under `just miri` and ship to every `cargo add nidus`, not just `memory`.
pub mod chunk;
mod config;
mod data;
// Levelled, logfmt diagnostics (`NIDUS_LOG`). Internal: what an embedding application
// wants from us is a `Result`, not our log stream, so the macro stays `pub(crate)`.
mod diag;
mod filter;
mod findex;
mod fts;
// Reciprocal Rank Fusion over several ranked legs, keeping each leg's own rank/score.
mod fuse;
mod glob;
mod index_cache;
mod lock;
mod log;
mod manifest;
// Metadata keys and clock helpers needed ungated (outside the `memory` feature),
// e.g. by `Nidus::sweep_expired` (nidus-140).
mod meta;
// Process-wide counters, rendered as Prometheus text by `GET /metrics` and readable
// in-process by an embedding application (nidus-abx.4).
pub mod metrics;
mod model;
mod profile;
mod search;
mod store;
// Recall/latency sweep over a caller's own store (nidus-sk9). Ungated: it must run
// under `just miri` and ship to every `cargo add nidus`, not just the `cli` feature.
mod tune;

// The `nidus` binary's guts (CLI + `nidus serve`). Compiled only under the
// non-default `cli` feature, so library consumers never see them and the core
// build stays pure. The thin `main` lives in `src/bin/nidus.rs`.
#[cfg(feature = "cli")]
pub mod cli;
#[cfg(feature = "cli")]
pub mod server;

// ── AI ingest layer (epic nidus-54l) — all behind off-by-default features ────
// The async edge: text-native `remember`/`recall` on top of the sync store
// core, which depends on NONE of this. See Cargo.toml `[features]`.
#[cfg(any(feature = "embed", feature = "summarize", feature = "rerank"))]
mod http;
// Provider capability registry (Embed | Summarize | Rerank): the single source of truth
// the embed/summarize/rerank factories consult before dispatching.
#[cfg(any(feature = "embed", feature = "summarize", feature = "rerank"))]
pub mod providers;
// Embedding abstraction + provider adapters + runtime `AnyEmbedder` selection.
#[cfg(feature = "embed")]
pub mod embed;
// Single-shot summarization abstraction + provider adapters.
#[cfg(feature = "summarize")]
pub mod summarize;
// Hosted cross-encoder reranking: `Reranker` + `AnyReranker` + adapters, mirroring `embed`.
// The pure ranking logic (window, passthrough, re-sort) lives unconditionally in
// `store::rerank`; only the async provider call sits behind this feature.
#[cfg(feature = "rerank")]
pub mod rerank;
// Text-native memory API: `remember(text)` / `recall(query_text)`. Gated on the
// `memory` feature (= `embed`) so building a bare provider (e.g. `embed-voyage`)
// does not require this module to exist.
#[cfg(feature = "memory")]
mod memory;
#[cfg(feature = "memory")]
pub use memory::{
    ChunkedRemembered, META_CREATED_AT, META_DIM, META_EMBEDDER, META_TEXT, META_UPDATED_AT,
    Memory, RecallOpts, RememberMode, RememberOpts, Remembered,
};
// The summarize-mode attr keys are only defined when summaries can be produced.
#[cfg(all(feature = "memory", feature = "summarize"))]
pub use memory::{META_SOURCE, META_SUMMARY};

pub use annotate::{Annotations, ClauseScore, Fragment, Highlight, HighlightOpts, LegScore};
pub use anyhow::Result;
pub use backend::{
    Appender, BackendLock, CasOutcome, ClusterLease, LeaseLost, LeaseRenewer, LocalFs, LocalRam,
    MemoryTier, Persistence, is_lease_lost, open_memory_tier, open_persistence,
};
pub use cancel::Cancel;
pub use config::{Config, Fsync, LeaseWait, OpenMode};
pub use data::SegmentIntegrity;
pub use findex::FilterIndexField;
pub use fts::{Analyzer, FtsField, Language};
pub use meta::META_EXPIRES_AT;
pub use model::{
    AggregateOpts, Aggregation, AnnConfig, AnnKind, ClusterStatus, DEFAULT_RERANK_OVERSCAN, Decay,
    Distance, Filter, Footprint, FtsClause, FtsCombine, FtsQuery, Group, Hit, HybridOpts, LimitPer,
    ListOpts, META_CHUNK_INDEX, META_PARENT_ID, OrderBy, Predicate, Projection, QuantKind,
    Quantization, RankBy, Record, RerankOpts, Role, SearchOpts, Value,
};
pub use profile::OpenProfile;
pub use store::Readiness;
pub use store::SegmentReport;
pub use tune::{TuneCell, TuneOpts, TuneReport, recall_at_k, tune};

use std::collections::BTreeMap;
use std::path::Path;

/// Which collections a [`Nidus::search`] ranks over (SPEC.md §7). Scores are
/// comparable across collections because the whole store shares one embedding
/// space. Accepts `impl Into<Scope>`, so `&str` and `&[&str]` work directly.
pub enum Scope<'a> {
    /// One collection — the common, fast path.
    Collection(&'a str),
    /// A chosen subset.
    Collections(&'a [&'a str]),
    /// Every collection in the store.
    All,
}

impl<'a> From<&'a str> for Scope<'a> {
    fn from(s: &'a str) -> Self {
        Scope::Collection(s)
    }
}

impl<'a> From<&'a [&'a str]> for Scope<'a> {
    fn from(s: &'a [&'a str]) -> Self {
        Scope::Collections(s)
    }
}

/// An open vector store. Synchronous; wrap in `Arc<RwLock<Nidus>>` for concurrent
/// searchers + one writer (SPEC.md §6.5).
pub struct Nidus {
    store: store::Store,
}

impl Nidus {
    /// Open (creating if absent) a store described by `config`.
    pub fn open(config: Config) -> Result<Self> {
        Ok(Self {
            store: store::Store::open(config)?,
        })
    }

    /// Convenience: `open(Config::new(dir, dimension))`.
    pub fn open_dir(dir: impl AsRef<Path>, dimension: usize) -> Result<Self> {
        Self::open(Config::new(dir.as_ref().to_path_buf(), dimension))
    }

    /// An in-memory store (no files, no lock). For tests and ephemeral use.
    pub fn open_in_memory(dimension: usize) -> Result<Self> {
        Ok(Self {
            store: store::Store::in_memory(dimension)?,
        })
    }

    /// The pinned embedding dimension.
    pub fn dimension(&self) -> usize {
        self.store.dimension()
    }

    /// Escape hatch for edge-of-crate wrappers (`crate::rerank`'s async free functions) that
    /// need the store's promoted ranking tails (`Store::finish`/`finish_hybrid`) without
    /// duplicating the page-cut contract (SPEC §7). Not part of the public API.
    #[cfg_attr(not(feature = "rerank"), allow(dead_code))]
    pub(crate) fn store(&self) -> &store::Store {
        &self.store
    }

    /// The configuration this store was opened with.
    pub fn config(&self) -> &Config {
        self.store.config()
    }

    /// The store's currently recorded open-time profile (nidus-141) — see [`OpenProfile`].
    pub fn open_profile(&self) -> &OpenProfile {
        self.store.open_profile()
    }

    /// Record `p` as the store's open-time profile so every later [`open`](Self::open)
    /// resolves ann/quantization/query_threads/mmap without re-passing them. An explicit
    /// act, never implied by a single call's flags.
    pub fn set_open_profile(&mut self, p: &OpenProfile) -> Result<()> {
        self.store.set_open_profile(p)
    }

    /// Clear the recorded profile — later opens fall back to built-in defaults unless
    /// overridden by an explicit [`Config`] flag.
    pub fn clear_open_profile(&mut self) -> Result<()> {
        self.store.clear_open_profile()
    }

    /// A cheap snapshot of the store's vector footprint — rows, dead rows,
    /// `vector_bytes`, and live `doc_count`. Use it to decide whether more data
    /// fits before a memory ceiling (pairs with [`Config::max_vector_bytes`]).
    pub fn footprint(&self) -> Footprint {
        self.store.footprint()
    }

    /// Who this instance is within the store, and how current it is — role, whether it
    /// holds the writer handle, whether it has been **fenced**, and how stale a reader is
    /// (SPEC §14.6). See [`ClusterStatus`].
    pub fn cluster_status(&self) -> ClusterStatus {
        self.store.cluster_status()
    }

    /// A lock-free handle to the facts a readiness probe needs — role, whether this writer
    /// has been fenced, and how stale a reader is. See [`Readiness`].
    pub fn readiness(&self) -> Readiness {
        self.store.readiness()
    }

    /// A handle to this instance's cluster writer lease, for keeping it warm out of band.
    pub fn lease_renewer(&self) -> Option<LeaseRenewer> {
        self.store.lease_renewer()
    }

    // ── Collections ──────────────────────────────────────────────────────

    pub fn create_collection(&mut self, name: &str) -> Result<()> {
        self.store.create_collection(name)
    }

    /// Create `collection` and declare its full-text-indexed fields up front. The recommended
    /// way to enable [BM25 full-text search](Self::text_search): indexing is fully incremental
    /// from the first upsert.
    pub fn create_collection_with_fts(&mut self, name: &str, fields: &[FtsField]) -> Result<()> {
        self.store.create_collection_with_fts(name, fields)
    }

    /// Declare which attribute fields of `collection` are full-text indexed, each with its own
    /// [`FtsField`] tuning (`k1`, `b`, [`Analyzer`]). Redeclaring rebuilds the affected fields.
    ///
    /// ```
    /// # use nidus::{FtsField, Nidus};
    /// # fn main() -> nidus::Result<()> {
    /// # let mut db = Nidus::open_in_memory(3)?;
    /// db.set_fts_schema("docs", &[
    ///     FtsField::new("title").k1(1.5),
    ///     FtsField::new("body").ascii_folding(true).max_token_len(40),
    /// ])?;
    /// # Ok(()) }
    /// ```
    pub fn set_fts_schema(&mut self, collection: &str, fields: &[FtsField]) -> Result<()> {
        self.store.set_fts_schema(collection, fields)
    }

    /// Opt-in index for the text predicates of SPEC §7.4/§7.5, per field. Changes their
    /// **speed, never their results**; costs write time and RAM. Empty `fields` turns it off.
    ///
    /// ```
    /// # use nidus::{FilterIndexField, Nidus};
    /// # fn main() -> nidus::Result<()> {
    /// # let mut db = Nidus::open_in_memory(3)?;
    /// db.set_filter_index("docs", &[
    ///     FilterIndexField::new("body"),
    ///     FilterIndexField::new("tag").trigrams(false),
    /// ])?;
    /// # Ok(()) }
    /// ```
    pub fn set_filter_index(
        &mut self,
        collection: &str,
        fields: &[FilterIndexField],
    ) -> Result<()> {
        self.store.set_filter_index(collection, fields)
    }

    pub fn drop_collection(&mut self, name: &str) -> Result<()> {
        self.store.drop_collection(name)
    }

    pub fn has_collection(&self, name: &str) -> bool {
        self.store.has_collection(name)
    }

    /// Whether `collection` already has a declared FTS schema, without rebuilding it.
    /// The truthful gate for callers that must call [`set_fts_schema`](Self::set_fts_schema)
    /// at most once per collection.
    pub fn has_fts_schema(&self, collection: &str) -> bool {
        self.store.has_fts_schema(collection)
    }

    pub fn collections(&self) -> Vec<String> {
        self.store.collections()
    }

    // ── Per-collection metadata ──────────────────────────────────────────

    pub fn get_meta(&self, collection: &str) -> BTreeMap<String, String> {
        self.store.get_meta(collection)
    }

    pub fn set_meta(&mut self, collection: &str, meta: BTreeMap<String, String>) -> Result<()> {
        self.store.set_meta(collection, meta)
    }

    // ── Documents ────────────────────────────────────────────────────────

    pub fn upsert(&mut self, collection: &str, records: &[Record]) -> Result<usize> {
        self.store.upsert(collection, records)
    }

    pub fn delete(&mut self, collection: &str, ids: &[&str]) -> Result<usize> {
        self.store.delete(collection, ids)
    }

    pub fn delete_where(&mut self, collection: &str, filter: &Filter) -> Result<usize> {
        filter::validate(filter)?;
        self.store.delete_where(collection, filter)
    }

    pub fn get_all(&self, collection: &str) -> Vec<Record> {
        self.store.get_all(collection)
    }

    pub fn get(&self, collection: &str, id: &str) -> Option<Record> {
        self.store.get(collection, id)
    }

    /// Resolve a [`Scope`] to the concrete collection names it covers — shared by
    /// `list`/`search`/`text_search`/`hybrid_search` so the resolution lives in one
    /// place.
    fn scope_names<'a>(&self, scope: impl Into<Scope<'a>>) -> Vec<String> {
        match scope.into() {
            Scope::Collection(c) => vec![c.to_string()],
            Scope::Collections(cs) => cs.iter().map(|s| s.to_string()).collect(),
            Scope::All => self.store.collections(),
        }
    }

    /// List records matching `opts.filter` across a [`Scope`], without vector scoring. With
    /// [`ListOpts::order_by`] set this is a plain ORDER BY over an attribute.
    pub fn list<'a>(&self, scope: impl Into<Scope<'a>>, opts: &ListOpts) -> Result<Vec<Hit>> {
        filter::validate(&opts.filter)?;
        let names = self.scope_names(scope);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.store.list(&refs, opts)
    }

    /// Count the records matching `opts.filter` across a [`Scope`] and sum the attributes named
    /// in [`AggregateOpts::sum`], without materializing a single [`Record`].
    ///
    /// ```
    /// # use nidus::{AggregateOpts, Nidus, Scope};
    /// # fn main() -> nidus::Result<()> {
    /// # let db = Nidus::open_in_memory(3)?;
    /// let stats = db.aggregate(Scope::All, &AggregateOpts {
    ///     sum: vec!["bytes".into()],
    ///     ..Default::default()
    /// })?;
    /// assert_eq!(stats.count, 0);
    /// # Ok(()) }
    /// ```
    pub fn aggregate<'a>(
        &self,
        scope: impl Into<Scope<'a>>,
        opts: &AggregateOpts,
    ) -> Result<Aggregation> {
        filter::validate(&opts.filter)?;
        store::aggregate::validate_group_by(opts.group_by.as_ref())?;
        let names = self.scope_names(scope);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        Ok(self.store.aggregate(&refs, opts))
    }

    /// Search a [`Scope`] — one collection, a subset, or the whole store — for the
    /// nearest neighbours to `query`, merged into one ranking.
    pub fn search<'a>(
        &self,
        scope: impl Into<Scope<'a>>,
        query: &[f32],
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        filter::validate(&opts.filter)?;
        let names = self.scope_names(scope);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.store.search(&refs, query, opts)
    }

    /// "More like this": search a [`Scope`] with the vector already stored at
    /// `collection`/`id`, dropping that source record from the results. `scope` need not be
    /// `collection` itself — one embedding space means any scope can be searched.
    pub fn search_similar<'a>(
        &self,
        scope: impl Into<Scope<'a>>,
        collection: &str,
        id: &str,
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        filter::validate(&opts.filter)?;
        let names = self.scope_names(scope);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.store.search_similar(&refs, collection, id, opts)
    }

    /// Full-text (BM25) search over a [`Scope`], merged into one ranking. Requires the field to be
    /// declared in the collection's FTS schema. Reuses [`SearchOpts`], but `min_score` here is a raw
    /// BM25 floor rather than a cosine one; text-only and vector-bearing docs are both eligible.
    pub fn text_search<'a>(
        &self,
        scope: impl Into<Scope<'a>>,
        query: &FtsQuery,
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        filter::validate(&opts.filter)?;
        let names = self.scope_names(scope);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.store.text_search(&refs, query, opts)
    }

    /// Hybrid search over a [`Scope`]: fuse a vector query and a BM25 text query into one ranking
    /// with Reciprocal Rank Fusion (see [`HybridOpts`]). A doc surfacing in only one leg is still
    /// ranked by that leg.
    pub fn hybrid_search<'a>(
        &self,
        scope: impl Into<Scope<'a>>,
        vector: &[f32],
        text: &FtsQuery,
        opts: &HybridOpts,
    ) -> Result<Vec<Hit>> {
        filter::validate(&opts.filter)?;
        let names = self.scope_names(scope);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.store.hybrid_search(&refs, vector, text, opts)
    }

    // ── Group commit (nidus-xb9.1) ───────────────────────────────────────

    /// Run `f` with the per-batch durable barrier **deferred**, so several mutations can
    /// share one fsync instead of taking one each — the group-commit primitive.
    pub fn deferred<T>(&mut self, f: impl FnOnce(&mut Nidus) -> Result<T>) -> Result<T> {
        let prev = self.store.begin_deferred();
        let out = f(self);
        self.store.end_deferred(prev);
        out
    }

    /// Take the barrier that [`deferred`](Self::deferred) mutations skipped: fsync `data` then
    /// `log`, and in cluster mode publish the commit counter once for the whole group.
    pub fn commit(&mut self) -> Result<()> {
        self.store.commit()
    }

    // ── Maintenance ──────────────────────────────────────────────────────

    /// fsync both files.
    pub fn flush(&mut self) -> Result<()> {
        self.store.flush()
    }

    /// Reclaim dead rows and superseded log records.
    pub fn compact(&mut self) -> Result<()> {
        self.store.compact()
    }

    /// Delete every entry whose `nidus.expires_at` has passed, then reclaim the rows.
    /// One all-or-nothing batch across every collection (nidus-166); `compact` follows as its
    /// own op, so a failed reclaim still leaves the deletes durable and the store consistent.
    pub fn sweep_expired(&mut self) -> Result<usize> {
        let now = meta::now_ms();
        // Bare `Le`, not `not_expired_predicate`'s `Not(Le)`: an absent `expires_at`
        // must not match, or entries that never had a TTL would be deleted.
        let filter = Filter(vec![Predicate::Le(
            META_EXPIRES_AT.to_string(),
            Value::DateTime(now),
        )]);
        let swept = self.store.delete_where_all(&filter)?;
        if swept > 0 {
            self.compact()?;
        }
        Ok(swept)
    }

    /// Adopt a separate writer's newer committed state into a lock-free `ReadOnly` handle without
    /// reopening. Such a handle is a snapshot taken at open; `refresh` advances it to the current
    /// committed state at a single consistent point, never a torn mix.
    pub fn refresh(&mut self) -> Result<bool> {
        self.store.refresh()
    }

    /// Write the ANN index ([`Config::ann`]) to its on-disk cache so the next [`open`](Self::open)
    /// loads it instead of rebuilding the graph. Explicit and out-of-band, never triggered by
    /// `upsert`/`flush`, so call it before shutting down a long-lived handle; `compact()` also does.
    pub fn persist_index(&mut self) -> Result<()> {
        self.store.persist_index()
    }

    /// Check every live segment's vector bytes against its checksum sidecar (#160). Rows written
    /// since the last [`flush`](Self::flush) are reported as *not covered* rather than verified,
    /// and a mismatch is reported, never repaired: re-checksumming corrupt bytes would hide them.
    pub fn verify_integrity(&mut self) -> Result<Vec<SegmentReport>> {
        self.store.verify_integrity()
    }

    /// Retune `ef_search`/`n_probe`/`overscan` in place for the `tune` sweep — no rebuild.
    pub(crate) fn retune_ann(&mut self, cfg: AnnConfig) {
        self.store.retune_ann(cfg)
    }
}
