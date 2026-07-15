//! Text-native memory API (epic nidus-54l, tickets .4 + .10).
//!
//! This is the headline "all-in-one memory" surface: text in ([`remember`]),
//! relevant text out ([`recall`]). It sits at the async edge, on top of the
//! synchronous [`Nidus`] core, and turns natural-language text into vectors with
//! an [`AnyEmbedder`] (optionally summarizing first with an [`AnySummarizer`])
//! before handing rows to the store.
//!
//! [`remember`]: Memory::remember
//! [`recall`]: Memory::recall
//!
//! ## Embedder-wiring decision (ticket .4's open question)
//!
//! **[`Memory`] OWNS a [`Nidus`] plus an [`AnyEmbedder`] (and an optional
//! [`AnySummarizer`]).** It does NOT bake embedding into [`Nidus`] itself.
//!
//! The reason is the escape hatch. [`Nidus`]'s raw `upsert`/`search` API takes a
//! caller-supplied `Vec<f32>` and stays completely untouched by this layer — a
//! host that already has its own embeddings (or wants a model nidus doesn't
//! ship an adapter for) keeps using [`Nidus`] directly, with zero async and zero
//! provider deps. `Memory` is a strictly additive convenience wrapper layered
//! over that same store; you can always [`into_inner`](Memory::into_inner) back
//! to the bare [`Nidus`], or reach it via [`db`](Memory::db) /
//! [`db_mut`](Memory::db_mut). Embedding is a property of *this handle*, not of
//! the on-disk store, so one process can wrap a store with an OpenAI embedder
//! while another opens the same directory raw.
//!
//! ## Embedding-space safety
//!
//! Vectors produced by different models live in incomparable spaces, so mixing
//! them in one collection makes cosine ranking meaningless. On the first
//! [`remember`](Memory::remember) into a collection, `Memory` pins the
//! embedder's identity (`"provider/model"`) and dimension into the collection's
//! metadata ([`META_EMBEDDER`] / [`META_DIM`]). Every later write re-checks that
//! identity and **refuses** (an `Err`) if a different embedder is now in play —
//! catching an accidental cross-model write before it corrupts a collection's
//! ranking.

use std::collections::BTreeMap;

use anyhow::{Context, bail};

use crate::embed::{AnyEmbedder, Embedder, embedder_identity};
use crate::{Filter, Hit, Nidus, Record, SearchOpts, Value};

#[cfg(feature = "summarize")]
use crate::summarize::{AnySummarizer, SummarizeOpts, Summarizer};

/// Collection-meta key holding the `"provider/model"` identity of the embedder
/// that produced this collection's vectors.
pub const META_EMBEDDER: &str = "nidus.embedder";
/// Collection-meta key holding the embedding dimension (decimal string).
pub const META_DIM: &str = "nidus.dim";
/// Attr key under which [`RememberMode::Summarize`] stores the generated summary
/// (the text that was actually embedded).
#[cfg(feature = "summarize")]
pub const META_SUMMARY: &str = "nidus.summary";
/// Attr key under which [`RememberMode::Summarize`] stores the original source
/// text, so a recall hit is explainable back to what was ingested.
#[cfg(feature = "summarize")]
pub const META_SOURCE: &str = "nidus.source";

/// Default `top_k` used by [`recall`](Memory::recall) when [`RecallOpts::top_k`]
/// is left at its `0` default.
const DEFAULT_TOP_K: usize = 10;

/// How [`Memory::remember`] prepares the text it stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RememberMode {
    /// Embed the text as given and store it.
    Raw,
    /// Summarize the text first, embed the **summary**, and store both the
    /// summary and a pointer to the source (see [`META_SUMMARY`]/[`META_SOURCE`]).
    /// Requires a summarizer — attach one with
    /// [`with_summarizer`](Memory::with_summarizer).
    #[cfg(feature = "summarize")]
    Summarize,
}

/// Options for [`Memory::recall`], mapped onto the store's [`SearchOpts`].
#[derive(Debug, Clone, Default)]
pub struct RecallOpts {
    /// Maximum number of hits. `0` means "use the default" ([`DEFAULT_TOP_K`]).
    pub top_k: usize,
    /// Drop hits scoring at or below this cosine similarity. `0.0` (the default)
    /// applies no floor.
    pub min_score: f32,
    /// Optional pre-scoring metadata filter.
    pub filter: Option<Filter>,
}

/// A text-native memory handle over a [`Nidus`] store and an embedder.
///
/// See the [module docs](self) for the ownership rationale and embedding-space
/// safety model.
pub struct Memory {
    db: Nidus,
    embedder: AnyEmbedder,
    #[cfg(feature = "summarize")]
    summarizer: Option<AnySummarizer>,
}

impl Memory {
    /// Wrap `db` with `embedder`. All [`remember`](Self::remember) writes will
    /// use `embedder`; the store's dimension should match
    /// [`Embedder::dimension`] (a mismatch is reported on first write).
    pub fn new(db: Nidus, embedder: AnyEmbedder) -> Self {
        Self {
            db,
            embedder,
            #[cfg(feature = "summarize")]
            summarizer: None,
        }
    }

    /// Attach a summarizer, enabling [`RememberMode::Summarize`].
    #[cfg(feature = "summarize")]
    pub fn with_summarizer(mut self, summarizer: AnySummarizer) -> Self {
        self.summarizer = Some(summarizer);
        self
    }

    /// Remember `text` under `id` in `collection`, embedding it (per `mode`) and
    /// upserting a record with `attrs`.
    ///
    /// Creates `collection` if absent. On the first write it pins the embedder's
    /// identity and dimension into the collection metadata; on later writes it
    /// refuses if a different embedder is in play (see the [module docs](self)).
    pub async fn remember(
        &mut self,
        collection: &str,
        id: &str,
        text: &str,
        attrs: BTreeMap<String, Value>,
        mode: RememberMode,
    ) -> anyhow::Result<()> {
        match mode {
            RememberMode::Raw => {
                // Split-borrow: `&mut self.db` and `&self.embedder` are disjoint fields.
                embed_and_store(&mut self.db, &self.embedder, collection, id, text, attrs).await
            }
            #[cfg(feature = "summarize")]
            RememberMode::Summarize => {
                let summarizer = self.summarizer.as_ref().context(
                    "RememberMode::Summarize requires a summarizer; attach one with Memory::with_summarizer(...)",
                )?;
                let summary = summarizer
                    .summarize(text, &SummarizeOpts::default())
                    .await
                    .with_context(|| format!("summarizing text for '{collection}/{id}'"))?;
                // Store the summary (what we embed) and the source text so a hit
                // is explainable back to what was ingested.
                let mut attrs = attrs;
                attrs.insert(META_SUMMARY.to_string(), Value::Str(summary.clone()));
                attrs.insert(META_SOURCE.to_string(), Value::Str(text.to_string()));
                embed_and_store(
                    &mut self.db,
                    &self.embedder,
                    collection,
                    id,
                    &summary,
                    attrs,
                )
                .await
            }
        }
    }

    /// Recall the nearest remembered records to `query_text` from `collection`.
    ///
    /// Embeds the query (via [`Embedder::embed_query`]) and runs a vector search
    /// with `opts` mapped onto [`SearchOpts`].
    pub async fn recall(
        &self,
        collection: &str,
        query_text: &str,
        opts: &RecallOpts,
    ) -> anyhow::Result<Vec<Hit>> {
        recall_with(&self.db, &self.embedder, collection, query_text, opts).await
    }

    /// Borrow the underlying store (raw `Vec<f32>` API escape hatch).
    pub fn db(&self) -> &Nidus {
        &self.db
    }

    /// Mutably borrow the underlying store.
    pub fn db_mut(&mut self) -> &mut Nidus {
        &mut self.db
    }

    /// Unwrap back to the bare [`Nidus`], dropping the embedder/summarizer.
    pub fn into_inner(self) -> Nidus {
        self.db
    }
}

// ── Internals (generic over `impl Embedder` so unit tests can drive them with a
// fake embedder, and so the borrow of `self.db` / `self.embedder` splits cleanly) ──

/// Ensure `collection` exists and its embedding space matches `embedder`,
/// pinning the identity + dimension on first use. Errors on a dimension mismatch
/// with the store, or on an embedder identity that differs from what the
/// collection was first written with.
///
/// `pub(crate)` so the HTTP server (`crate::server`) can reuse the exact same
/// pin/identity logic when it embeds text on behalf of a network client — rather
/// than reimplementing it and risking drift from this write path.
pub(crate) fn ensure_collection_and_pin<E: Embedder>(
    db: &mut Nidus,
    embedder: &E,
    collection: &str,
) -> anyhow::Result<()> {
    let identity = embedder_identity(embedder);
    let store_dim = db.dimension();
    if embedder.dimension() != store_dim {
        bail!(
            "embedder '{identity}' produces {}-dimensional vectors but the store dimension is {store_dim}",
            embedder.dimension()
        );
    }

    if !db.has_collection(collection) {
        db.create_collection(collection)?;
    }

    let mut meta = db.get_meta(collection);
    match meta.get(META_EMBEDDER) {
        Some(existing) => bail_if_identity_differs(collection, existing, &identity)?,
        None => {
            meta.insert(META_EMBEDDER.to_string(), identity);
            meta.insert(META_DIM.to_string(), store_dim.to_string());
            db.set_meta(collection, meta)?;
        }
    }
    Ok(())
}

/// Bail if `collection` was pinned to a different embedder than `identity`.
/// Shared by the write-side pin and the recall-side guard so a cross-model mix
/// is refused symmetrically (comparing vectors from different embedding models
/// is meaningless — see the [module docs](self)).
fn bail_if_identity_differs(
    collection: &str,
    existing: &str,
    identity: &str,
) -> anyhow::Result<()> {
    if existing != identity {
        bail!(
            "collection '{collection}' was written with embedder '{existing}', but this Memory \
             uses '{identity}'; vectors from different embedding models are not comparable — \
             use a separate collection or the matching embedder"
        );
    }
    Ok(())
}

/// Pin the collection, embed `embed_text`, and upsert the resulting record.
async fn embed_and_store<E: Embedder>(
    db: &mut Nidus,
    embedder: &E,
    collection: &str,
    id: &str,
    embed_text: &str,
    attrs: BTreeMap<String, Value>,
) -> anyhow::Result<()> {
    ensure_collection_and_pin(db, embedder, collection)?;
    let vector = embedder
        .embed(embed_text)
        .await
        .with_context(|| format!("embedding text for '{collection}/{id}'"))?;
    db.upsert(collection, &[Record::new(id, vector, attrs)])?;
    Ok(())
}

/// Recall-side identity guard: refuse a recall whose embedder differs from the
/// one `collection` was written with. Symmetric with the write-side pin —
/// recalling with a different (even same-dimension) embedder than the collection
/// was written with would silently return meaningless cross-space rankings, so
/// refuse it up front. A collection with no pinned embedder (never written
/// through `Memory`) imposes no constraint.
///
/// `pub(crate)` so the HTTP server reuses this exact guard on its recall path
/// (the write side reuses [`ensure_collection_and_pin`]).
pub(crate) fn guard_recall_identity<E: Embedder>(
    db: &Nidus,
    embedder: &E,
    collection: &str,
) -> anyhow::Result<()> {
    if let Some(existing) = db.get_meta(collection).get(META_EMBEDDER) {
        bail_if_identity_differs(collection, existing, &embedder_identity(embedder))?;
    }
    Ok(())
}

/// Embed `query_text` as a query and run a vector search mapped from `opts`.
async fn recall_with<E: Embedder>(
    db: &Nidus,
    embedder: &E,
    collection: &str,
    query_text: &str,
    opts: &RecallOpts,
) -> anyhow::Result<Vec<Hit>> {
    guard_recall_identity(db, embedder, collection)?;
    let query = embedder
        .embed_query(query_text)
        .await
        .with_context(|| format!("embedding recall query for '{collection}'"))?;
    let search_opts = SearchOpts {
        top_k: if opts.top_k == 0 {
            DEFAULT_TOP_K
        } else {
            opts.top_k
        },
        filter: opts.filter.clone().unwrap_or_default(),
        min_score: (opts.min_score > 0.0).then_some(opts.min_score),
    };
    db.search(collection, &query, &search_opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::EmbedError;
    use std::future::Future;

    /// A deterministic, network-free [`Embedder`] for tests: it hashes the input
    /// text into a fixed-dimension vector, so the same text always yields the
    /// same vector (a stored doc and a query over its text score ~1.0).
    struct FakeEmbedder {
        dim: usize,
        provider: String,
        model: String,
    }

    impl FakeEmbedder {
        fn new(dim: usize, provider: &str, model: &str) -> Self {
            Self {
                dim,
                provider: provider.to_string(),
                model: model.to_string(),
            }
        }

        fn vector_for(&self, text: &str) -> Vec<f32> {
            // Spread byte contributions across buckets; +0.1 keeps it non-zero
            // (an all-zero vector cannot be unit-normalized by the store).
            let mut v = vec![0.1f32; self.dim];
            for (i, b) in text.bytes().enumerate() {
                v[i % self.dim] += (b as f32) + 1.0;
            }
            v
        }
    }

    impl Embedder for FakeEmbedder {
        fn embed(&self, text: &str) -> impl Future<Output = Result<Vec<f32>, EmbedError>> + Send {
            let v = self.vector_for(text);
            async move { Ok(v) }
        }

        fn embed_batch(
            &self,
            texts: &[&str],
        ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbedError>> + Send {
            let vs: Vec<Vec<f32>> = texts.iter().map(|t| self.vector_for(t)).collect();
            async move { Ok(vs) }
        }

        fn dimension(&self) -> usize {
            self.dim
        }
        fn max_input_tokens(&self) -> usize {
            8192
        }
        fn provider_name(&self) -> &str {
            &self.provider
        }
        fn model_name(&self) -> &str {
            &self.model
        }
    }

    fn open_tmp(dim: usize) -> (tempfile::TempDir, Nidus) {
        let dir = tempfile::tempdir().unwrap();
        let db = Nidus::open_dir(dir.path(), dim).unwrap();
        (dir, db)
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn remember_recall_round_trip() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        embed_and_store(
            &mut db,
            &emb,
            "notes",
            "a",
            "the quick brown fox",
            BTreeMap::new(),
        )
        .await
        .unwrap();
        embed_and_store(
            &mut db,
            &emb,
            "notes",
            "b",
            "lorem ipsum dolor sit",
            BTreeMap::new(),
        )
        .await
        .unwrap();

        let hits = recall_with(
            &db,
            &emb,
            "notes",
            "the quick brown fox",
            &RecallOpts {
                top_k: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, "a", "the exact-text match should rank first");
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn first_write_pins_embedder_identity_and_dim() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        embed_and_store(&mut db, &emb, "notes", "a", "hello world", BTreeMap::new())
            .await
            .unwrap();

        let meta = db.get_meta("notes");
        assert_eq!(meta.get(META_EMBEDDER).map(String::as_str), Some("fake/v1"));
        assert_eq!(meta.get(META_DIM).map(String::as_str), Some("8"));
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn mismatched_embedder_is_refused() {
        let (_dir, mut db) = open_tmp(8);
        let emb_v1 = FakeEmbedder::new(8, "fake", "v1");
        embed_and_store(&mut db, &emb_v1, "notes", "a", "hello", BTreeMap::new())
            .await
            .unwrap();

        // A different model over the same collection must be rejected.
        let emb_v2 = FakeEmbedder::new(8, "fake", "v2");
        let err = ensure_collection_and_pin(&mut db, &emb_v2, "notes").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("fake/v1"),
            "message names the pinned embedder: {msg}"
        );
        assert!(
            msg.contains("fake/v2"),
            "message names the current embedder: {msg}"
        );

        // The same embedder is fine (idempotent re-check).
        ensure_collection_and_pin(&mut db, &emb_v1, "notes").unwrap();
    }

    #[test]
    fn embedder_dimension_must_match_store() {
        let (_dir, mut db) = open_tmp(8);
        let emb_bad = FakeEmbedder::new(4, "fake", "v1");
        let err = ensure_collection_and_pin(&mut db, &emb_bad, "notes").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains('4') && msg.contains('8'),
            "message names both dims: {msg}"
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn recall_with_mismatched_embedder_is_refused() {
        let (_dir, mut db) = open_tmp(8);
        let emb_v1 = FakeEmbedder::new(8, "fake", "v1");
        embed_and_store(&mut db, &emb_v1, "notes", "a", "hello", BTreeMap::new())
            .await
            .unwrap();

        // Same dimension, different model → the recall guard must refuse rather
        // than return meaningless cross-space rankings.
        let emb_v2 = FakeEmbedder::new(8, "fake", "v2");
        let err = recall_with(&db, &emb_v2, "notes", "hello", &RecallOpts::default())
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("fake/v1") && msg.contains("fake/v2"), "{msg}");

        // The matching embedder still recalls fine.
        recall_with(&db, &emb_v1, "notes", "hello", &RecallOpts::default())
            .await
            .unwrap();
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn recall_defaults_top_k_when_zero() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");
        for i in 0..3 {
            embed_and_store(
                &mut db,
                &emb,
                "notes",
                &format!("doc{i}"),
                &format!("content number {i}"),
                BTreeMap::new(),
            )
            .await
            .unwrap();
        }
        // top_k = 0 (RecallOpts default) must fall back to a sensible default,
        // not return zero hits.
        let hits = recall_with(
            &db,
            &emb,
            "notes",
            "content number 1",
            &RecallOpts::default(),
        )
        .await
        .unwrap();
        assert_eq!(hits.len(), 3);
    }
}
