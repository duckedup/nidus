//! Text-native memory API (epic nidus-54l, tickets .4 + .10).

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail};

use crate::embed::{AnyEmbedder, Embedder, embedder_identity};
use crate::{Filter, FtsField, Hit, Nidus, Predicate, Record, SearchOpts, Value};

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
/// Attr key under which [`RememberMode::Summarize`] stored the original source
/// text before `nidus.text` existed. No longer stamped, kept so legacy records
/// (written before nidus-k28) remain readable.
#[cfg(feature = "summarize")]
pub const META_SOURCE: &str = "nidus.source";
/// Attr key holding the raw remembered text, stamped on every `remember` write
/// regardless of mode (nidus-k28.7).
pub const META_TEXT: &str = "nidus.text";
/// Attr key holding the `Value::DateTime` (UTC epoch ms) an entry was first written.
/// Carries forward unchanged on a dedup update-in-place.
pub const META_CREATED_AT: &str = "nidus.created_at";
/// Attr key holding the `Value::DateTime` (UTC epoch ms) an entry was last written.
pub const META_UPDATED_AT: &str = "nidus.updated_at";
/// Attr key holding the `Value::DateTime` (UTC epoch ms) after which an entry is
/// expired. Absent means the entry never expires.
pub const META_EXPIRES_AT: &str = "nidus.expires_at";

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

/// The default full-text schema every `remember`-provisioned collection declares: the raw
/// remembered text, all-default tuning (English analyzer, `k1=1.2`, `b=0.75`).
pub(crate) fn default_fts_fields() -> Vec<FtsField> {
    vec![FtsField::new(META_TEXT)]
}

/// The current time as `Value::DateTime`'s representation: UTC epoch milliseconds.
/// Callers land in a later unit of nidus-k28 (the `remember`/HTTP write paths).
#[allow(dead_code)]
pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Drop caller-supplied recency keys before stamping. `created_at`/`updated_at` would be
/// overwritten anyway, but `expires_at` survives when no TTL is passed — letting a caller
/// set an expiry that never went through `ttl_seconds`.
#[allow(dead_code)]
pub(crate) fn strip_reserved_recency(attrs: &mut BTreeMap<String, Value>) {
    for key in [META_CREATED_AT, META_UPDATED_AT, META_EXPIRES_AT] {
        attrs.remove(key);
    }
}

/// Stamp recency attrs in place: `updated_at` always moves to `now_ms`; `created_at`
/// carries forward from `prior_created` (an update-in-place) or is set to `now_ms` (a
/// fresh entry); `expires_at` is set only when `ttl_seconds` is given.
#[allow(dead_code)]
pub(crate) fn stamp_recency(
    attrs: &mut BTreeMap<String, Value>,
    now_ms: i64,
    prior_created: Option<i64>,
    ttl_seconds: Option<i64>,
) {
    let created = prior_created.unwrap_or(now_ms);
    attrs.insert(META_CREATED_AT.to_string(), Value::DateTime(created));
    attrs.insert(META_UPDATED_AT.to_string(), Value::DateTime(now_ms));
    if let Some(ttl) = ttl_seconds {
        attrs.insert(
            META_EXPIRES_AT.to_string(),
            // Saturating: `ttl_seconds` is unvalidated caller input over the wire, and a
            // plain multiply-add would panic in debug and wrap in release.
            Value::DateTime(now_ms.saturating_add(ttl.saturating_mul(1000))),
        );
    }
}

/// The true-complement "not expired" predicate: true when `nidus.expires_at` is in the
/// future *and* true when it is absent (never TTL'd). A bare `Gt`/`Ge` would be false on
/// an absent key (`range_matches`), silently hiding every memory that never got a TTL.
#[allow(dead_code)]
pub(crate) fn not_expired_predicate(now_ms: i64) -> Predicate {
    Predicate::Not(Box::new(Predicate::Le(
        META_EXPIRES_AT.to_string(),
        Value::DateTime(now_ms),
    )))
}

/// Ensure `collection` exists and its embedding space matches `embedder`, pinning the identity +
/// dimension on first use. Errors on a dimension mismatch with the store, or on an embedder
/// identity that differs from what the collection was first written with.
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

    // Gated on the real schema, not a meta flag: `set_fts_schema` rebuilds the field
    // index from every live doc, so calling it per-write would be O(collection size).
    if !db.has_fts_schema(collection) {
        db.set_fts_schema(collection, &default_fts_fields())?;
    }
    Ok(())
}

/// Bail if `collection` was pinned to a different embedder than `identity`.
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

/// Recall-side identity guard: refuse a recall whose embedder differs from the one `collection` was
/// written with, since even a same-dimension mismatch returns meaningless cross-space rankings. A
/// collection with no pinned embedder — never written through `Memory` — imposes no constraint.
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
        // No `offset` on `RecallOpts` by design (nidus-m50.15): the memory API stays lean.
        ..Default::default()
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
    #[cfg_attr(miri, ignore)] // file-backed via open_tmp; also `memory` is off in the Miri lane.
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

    #[test]
    fn stamp_recency_fresh_entry_sets_created_and_updated() {
        let mut attrs = BTreeMap::new();
        stamp_recency(&mut attrs, 1_000, None, None);
        assert_eq!(attrs.get(META_CREATED_AT), Some(&Value::DateTime(1_000)));
        assert_eq!(attrs.get(META_UPDATED_AT), Some(&Value::DateTime(1_000)));
        assert!(!attrs.contains_key(META_EXPIRES_AT));
    }

    #[test]
    fn stamp_recency_preserves_prior_created_and_sets_ttl() {
        let mut attrs = BTreeMap::new();
        stamp_recency(&mut attrs, 2_000, Some(500), Some(60));
        assert_eq!(attrs.get(META_CREATED_AT), Some(&Value::DateTime(500)));
        assert_eq!(attrs.get(META_UPDATED_AT), Some(&Value::DateTime(2_000)));
        assert_eq!(
            attrs.get(META_EXPIRES_AT),
            Some(&Value::DateTime(2_000 + 60_000))
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    fn not_expired_predicate_matches_absent_and_unexpired_not_expired() {
        let (_dir, mut db) = open_tmp(3);
        db.create_collection("col").unwrap();
        let now = 1_700_000_000_000i64;

        let unexpired =
            BTreeMap::from([(META_EXPIRES_AT.to_string(), Value::DateTime(now + 60_000))]);
        let expired = BTreeMap::from([(META_EXPIRES_AT.to_string(), Value::DateTime(now - 1))]);

        db.upsert(
            "col",
            &[
                Record::new("never_ttld", vec![1.0, 0.0, 0.0], BTreeMap::new()),
                Record::new("unexpired", vec![1.0, 0.0, 0.0], unexpired),
                Record::new("expired", vec![1.0, 0.0, 0.0], expired),
            ],
        )
        .unwrap();

        let opts = SearchOpts {
            top_k: 10,
            filter: Filter(vec![not_expired_predicate(now)]),
            ..Default::default()
        };
        let hits = db.search("col", &[1.0, 0.0, 0.0], &opts).unwrap();
        let mut ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        ids.sort();
        assert_eq!(ids, vec!["never_ttld", "unexpired"]);
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // file-backed via open_tmp; also `memory` is off in the Miri lane.
    async fn first_write_declares_default_fts_schema_and_is_gated() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        assert!(!db.has_fts_schema("notes"));
        embed_and_store(&mut db, &emb, "notes", "a", "hello world", BTreeMap::new())
            .await
            .unwrap();
        assert!(db.has_fts_schema("notes"));

        // Simulate a schema declared directly (not through `remember`). If the second
        // write below is not gated on `has_fts_schema`, it clobbers this back to the
        // default schema — the O(collection size) rebuild the gate exists to avoid.
        db.set_fts_schema("notes", &[crate::FtsField::new("custom_field")])
            .unwrap();

        let mut attrs = BTreeMap::new();
        attrs.insert(
            "custom_field".to_string(),
            Value::Str("bananas".to_string()),
        );
        embed_and_store(&mut db, &emb, "notes", "b", "more text", attrs)
            .await
            .unwrap();

        let hits = db
            .text_search(
                "notes",
                &crate::FtsQuery::new("custom_field", "bananas"),
                &SearchOpts {
                    top_k: 5,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            hits.len(),
            1,
            "a gated second write must not redeclare the default schema over the custom one"
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // file-backed via open_tmp; also `memory` is off in the Miri lane.
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
    #[cfg_attr(miri, ignore)] // file-backed via open_tmp; also `memory` is off in the Miri lane.
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
    #[cfg_attr(miri, ignore)] // file-backed via open_tmp; also `memory` is off in the Miri lane.
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
