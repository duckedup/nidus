#![forbid(unsafe_code)]
//! # nidus
//!
//! A small, pure-Rust embeddable vector store: brute-force cosine search over a
//! single append-only directory, with typed metadata filters and many logical
//! collections sharing one embedding space. No FFI, no C, no SQL.
//!
//! See `SPEC.md` for the full design.
//!
//! ```no_run
//! use nidus::{Nidus, Config, Record, SearchOpts, Scope};
//! use std::collections::BTreeMap;
//!
//! let mut db = Nidus::open(Config::new("/tmp/store", 3))?;
//! db.create_collection("docs")?;
//! db.upsert("docs", &[Record { id: "a".into(), vector: vec![1.0, 0.0, 0.0], attrs: BTreeMap::new() }])?;
//! let hits = db.search("docs", &[1.0, 0.0, 0.0], &SearchOpts { top_k: 5, ..Default::default() })?;
//! # anyhow::Ok(())
//! ```

mod config;
mod data;
mod filter;
mod glob;
mod lock;
mod log;
mod model;
mod search;
mod store;

pub use anyhow::Result;
pub use config::{Config, Fsync, OpenMode};
pub use model::{Filter, Footprint, Hit, Predicate, Record, SearchOpts, Value};

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

    /// Search a [`Scope`] — one collection, a subset, or the whole store — for the
    /// nearest neighbours to `query`, merged into one ranking.
    pub fn search<'a>(
        &self,
        scope: impl Into<Scope<'a>>,
        query: &[f32],
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        let names: Vec<String> = match scope.into() {
            Scope::Collection(c) => vec![c.to_string()],
            Scope::Collections(cs) => cs.iter().map(|s| s.to_string()).collect(),
            Scope::All => self.store.collections(),
        };
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.store.search(&refs, query, opts)
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
}
