// `#![deny(unsafe_code)]`, not `forbid`: nidus is unsafe-free everywhere except the single
// memory-map call in `src/data/mmap.rs` (the one conscious FFI opt-in, SPEC §9/§14.6), which
// carries a scoped `#[allow(unsafe_code)]`. `deny` lets that one site opt in; every other use
// of `unsafe` anywhere in the crate is still a hard compile error.
#![deny(unsafe_code)]
//! # nidus
//!
//! A small, pure-Rust embeddable vector store: brute-force cosine search over a
//! single append-only directory, with typed metadata filters and many logical
//! collections sharing one embedding space. No SQL, no query engine; safe Rust
//! throughout but for the one opt-in memory-map call.
//!
//! See `SPEC.md` for the full design.
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
pub mod backend;
mod config;
mod data;
mod filter;
mod fts;
mod glob;
mod index_cache;
mod lock;
mod log;
mod manifest;
mod model;
mod search;
mod store;

// The `nidus` binary's guts (CLI + `nidus serve`). Compiled only under the
// non-default `cli` feature, so library consumers never see them and the core
// build stays pure. The thin `main` lives in `src/bin/nidus.rs`.
#[cfg(feature = "cli")]
pub mod cli;
#[cfg(feature = "cli")]
pub mod server;

pub use anyhow::Result;
pub use backend::{
    Appender, BackendLock, LocalFs, LocalRam, MemoryTier, Persistence, open_memory_tier,
    open_persistence,
};
pub use config::{Config, Fsync, OpenMode};
pub use fts::Language;
pub use model::{
    AnnConfig, AnnKind, Distance, Filter, Footprint, FtsQuery, Hit, HybridOpts, Predicate,
    QuantKind, Quantization, Record, SearchOpts, Value,
};

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

    /// The configuration this store was opened with.
    pub fn config(&self) -> &Config {
        self.store.config()
    }

    /// A cheap snapshot of the store's vector footprint — rows, dead rows,
    /// `vector_bytes`, and live `doc_count`. Use it to decide whether more data
    /// fits before a memory ceiling (pairs with [`Config::max_vector_bytes`]).
    pub fn footprint(&self) -> Footprint {
        self.store.footprint()
    }

    // ── Collections ──────────────────────────────────────────────────────

    pub fn create_collection(&mut self, name: &str) -> Result<()> {
        self.store.create_collection(name)
    }

    /// Create `collection` and declare its full-text-indexed fields up front (each a
    /// `(field, language)` pair). The recommended way to enable [BM25 full-text
    /// search](Self::text_search): indexing is fully incremental from the first upsert.
    pub fn create_collection_with_fts(
        &mut self,
        name: &str,
        fields: &[(String, Language)],
    ) -> Result<()> {
        self.store.create_collection_with_fts(name, fields)
    }

    /// Declare (or redeclare) which attribute fields of `collection` are full-text
    /// indexed for BM25 search, each with its analyzer [`Language`]. May be called
    /// before or after upserting — declaring it on a collection that already holds docs
    /// builds the index from them once. Redeclaring rebuilds the affected fields.
    pub fn set_fts_schema(
        &mut self,
        collection: &str,
        fields: &[(String, Language)],
    ) -> Result<()> {
        self.store.set_fts_schema(collection, fields)
    }

    pub fn drop_collection(&mut self, name: &str) -> Result<()> {
        self.store.drop_collection(name)
    }

    pub fn has_collection(&self, name: &str) -> bool {
        self.store.has_collection(name)
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
        self.store.delete_where(collection, filter)
    }

    pub fn get_all(&self, collection: &str) -> Vec<Record> {
        self.store.get_all(collection)
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

    /// List records matching `filter` across a [`Scope`], without vector scoring.
    /// Skips `offset` matches and returns up to `limit` more, in insertion order,
    /// all with `score: 0.0`. Pass `offset = 0` for the first page; advance by
    /// `limit` to paginate.
    pub fn list<'a>(
        &self,
        scope: impl Into<Scope<'a>>,
        filter: &Filter,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<Hit>> {
        let names = self.scope_names(scope);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.store.list(&refs, filter, offset, limit)
    }

    /// Search a [`Scope`] — one collection, a subset, or the whole store — for the
    /// nearest neighbours to `query`, merged into one ranking.
    pub fn search<'a>(
        &self,
        scope: impl Into<Scope<'a>>,
        query: &[f32],
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        let names = self.scope_names(scope);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.store.search(&refs, query, opts)
    }

    /// Full-text (BM25) search over a [`Scope`] for `query` — the indexed field plus
    /// query text — merged into one ranking. Requires the field to be declared in the
    /// collection's FTS schema (see [`set_fts_schema`](Self::set_fts_schema)). Reuses
    /// [`SearchOpts`] (`top_k`, `filter`); here `min_score` is a raw BM25 floor rather
    /// than a cosine one. Text-only and vector-bearing docs are both eligible.
    pub fn text_search<'a>(
        &self,
        scope: impl Into<Scope<'a>>,
        query: &FtsQuery,
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        let names = self.scope_names(scope);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.store.text_search(&refs, query, opts)
    }

    /// Hybrid search over a [`Scope`]: fuse a vector query and a BM25 text query into a
    /// single ranking with Reciprocal Rank Fusion (see [`HybridOpts`]). A doc that
    /// surfaces in only one leg (e.g. a text-only doc, or one whose vector matches but
    /// whose text does not) is still ranked by that leg.
    pub fn hybrid_search<'a>(
        &self,
        scope: impl Into<Scope<'a>>,
        vector: &[f32],
        text: &FtsQuery,
        opts: &HybridOpts,
    ) -> Result<Vec<Hit>> {
        let names = self.scope_names(scope);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.store.hybrid_search(&refs, vector, text, opts)
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

    /// Adopt a separate writer's newer committed state into a lock-free
    /// [`ReadOnly`](OpenMode::ReadOnly) handle without reopening the store. A `ReadOnly`
    /// handle is a consistent snapshot taken when it opened; `refresh` advances it to the
    /// store's current committed state — picking up the writer's appends, seals, deletes,
    /// and compactions — at a single consistent point (never a torn mix).
    ///
    /// Returns `Ok(true)` when newer state was adopted and `Ok(false)` when the handle was
    /// already current — the cheap common case, a single small manifest read plus a `log`
    /// stat, so it is safe to call before a batch of queries. A `ReadWrite` handle (already
    /// the source of truth) and an in-memory store always return `Ok(false)`. This is the
    /// basis for a search-only process tracking a store another process is writing.
    pub fn refresh(&mut self) -> Result<bool> {
        self.store.refresh()
    }

    /// Write the approximate-nearest-neighbour index ([`Config::ann`]) to its on-disk
    /// cache so the next [`open`](Self::open) loads it instead of rebuilding the graph
    /// (the expensive part of opening an ANN store). This is an explicit, out-of-band
    /// operation — it is never triggered by `upsert`/`flush`, so the write path stays
    /// fast — so call it before shutting down a long-lived handle (e.g. a search
    /// server). A no-op when ANN is disabled, the store is in-memory or read-only, or
    /// nothing changed since the last persist. `compact()` refreshes the cache too.
    pub fn persist_index(&mut self) -> Result<()> {
        self.store.persist_index()
    }
}
