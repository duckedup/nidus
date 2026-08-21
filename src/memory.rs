//! Text-native memory API (epic nidus-54l, tickets .4 + .10).

use std::collections::BTreeMap;

use anyhow::{Context, bail};

use crate::diag::diag;
use crate::embed::{AnyEmbedder, Embedder, embedder_identity};
use crate::meta::now_ms;
use crate::{
    Filter, FtsField, Hit, META_CHUNK_INDEX, META_EXPIRES_AT, META_PARENT_ID, Nidus, Predicate,
    Record, SearchOpts, Value,
};

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
/// Attr key under which [`RememberMode::Summarize`] stored the original source text
/// before `nidus.text` existed. No longer stamped by any surface — [`META_TEXT`] carries
/// the raw text — kept so records written before #133 remain readable.
#[cfg(feature = "summarize")]
pub const META_SOURCE: &str = "nidus.source";
/// Attr key holding the raw remembered text, stamped on every `remember` write regardless of
/// mode (nidus-k28.7). Canonical definition moved to `model::META_TEXT` (nidus-4ss, so
/// `RerankOpts::default` can reach it outside this feature); re-exported so the path is unchanged.
pub use crate::model::META_TEXT;
/// Attr key holding the `Value::DateTime` (UTC epoch ms) an entry was first written.
/// Carries forward unchanged on a dedup update-in-place.
pub const META_CREATED_AT: &str = "nidus.created_at";
/// Attr key holding the `Value::DateTime` (UTC epoch ms) an entry was last written.
pub const META_UPDATED_AT: &str = "nidus.updated_at";

/// Default `top_k` used by [`recall`](Memory::recall) when [`RecallOpts::top_k`]
/// is left at its `0` default.
pub(crate) const DEFAULT_TOP_K: usize = 10;

/// How [`Memory::remember`] prepares the text it stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RememberMode {
    /// Embed the text as given and store it.
    #[default]
    Raw,
    /// Summarize the text first, embed the **summary**, and store it under
    /// [`META_SUMMARY`]. The raw text is stamped as [`META_TEXT`] in both modes.
    #[cfg(feature = "summarize")]
    Summarize,
}

/// Options for [`Memory::remember`]. Taken by value because `attrs` moves into the
/// stored record.
#[derive(Debug, Clone, Default)]
pub struct RememberOpts {
    /// How to prepare the text before embedding it.
    pub mode: RememberMode,
    /// Metadata stamped on the record. Reserved `nidus.*` recency keys are dropped:
    /// they are stamped from the store, never accepted from a caller.
    pub attrs: BTreeMap<String, Value>,
    /// Seconds until this memory expires, counted from the write. `None` never expires.
    pub ttl_seconds: Option<i64>,
    /// Cosine floor above which this write updates the nearest existing entry instead of
    /// inserting a competing near-duplicate. `None` disables dedup (a plain upsert by id).
    pub dedupe_threshold: Option<f32>,
}

/// What a [`Memory::remember`] write actually did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Remembered {
    /// The record written. Not the requested id when `deduped`: a near-duplicate match
    /// redirects the write onto the entry it matched.
    pub id: String,
    /// Whether [`RememberOpts::dedupe_threshold`] matched and redirected the write.
    pub deduped: bool,
    /// How many records the upsert touched.
    pub upserted: usize,
}

/// What a [`Memory::remember_chunked`] write actually did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkedRemembered {
    /// The `parent_id` the caller passed in.
    pub parent_id: String,
    /// One entry per emitted chunk, in index order.
    pub chunks: Vec<Remembered>,
    /// Stale tail records (from a longer prior chunking of the same `parent_id`) removed.
    pub pruned: usize,
}

/// Options for [`Memory::recall`], mapped onto the store's [`SearchOpts`].
#[derive(Debug, Clone, Default)]
pub struct RecallOpts {
    /// Maximum number of hits. `0` means "use the default" ([`DEFAULT_TOP_K`]).
    pub top_k: usize,
    /// Drop hits scoring below this cosine similarity. `0.0` (the default)
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

    /// Remember `text` under `id` in `collection`: prepare it per [`RememberOpts::mode`],
    /// embed it, and upsert the row — stamping [`META_TEXT`] and the recency attrs, and
    /// honouring the TTL and dedup knobs. The same write the HTTP and MCP surfaces run.
    pub async fn remember(
        &mut self,
        collection: &str,
        id: &str,
        text: &str,
        opts: RememberOpts,
    ) -> anyhow::Result<Remembered> {
        let RememberOpts {
            mode,
            attrs,
            ttl_seconds,
            dedupe_threshold,
        } = opts;
        // `mut` only on the path where a summary is stamped into it.
        #[cfg_attr(not(feature = "summarize"), allow(unused_mut))]
        let mut attrs = attrs;

        let embed_text = match mode {
            RememberMode::Raw => text.to_string(),
            #[cfg(feature = "summarize")]
            RememberMode::Summarize => {
                let summarizer = self.summarizer.as_ref().context(
                    "RememberMode::Summarize requires a summarizer; attach one with Memory::with_summarizer(...)",
                )?;
                let summary = summarizer
                    .summarize(text, &SummarizeOpts::default())
                    .await
                    .with_context(|| format!("summarizing text for '{collection}/{id}'"))?;
                attrs.insert(META_SUMMARY.to_string(), Value::Str(summary.clone()));
                summary
            }
        };

        // Split-borrow: `&mut self.db` and `&self.embedder` are disjoint fields.
        embed_and_commit(
            &mut self.db,
            &self.embedder,
            collection,
            &embed_text,
            RememberWrite {
                id: id.to_string(),
                text: text.to_string(),
                attrs,
                ttl_seconds,
                dedupe_threshold,
            },
        )
        .await
    }

    /// Chunk `text`, embed in one batch, upsert each as `"{parent_id}#{index}"`, then delete
    /// any survivor with that `parent_id` and index `>= n`. Empty text writes and prunes
    /// nothing, so an accidental empty read cannot wipe the document.
    pub async fn remember_chunked(
        &mut self,
        collection: &str,
        parent_id: &str,
        text: &str,
        chunk_opts: &crate::chunk::ChunkOpts,
        opts: RememberOpts,
    ) -> anyhow::Result<ChunkedRemembered> {
        remember_chunked_with(
            &mut self.db,
            &self.embedder,
            collection,
            parent_id,
            text,
            chunk_opts,
            opts,
        )
        .await
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

/// Drop caller-supplied recency keys before stamping. `created_at`/`updated_at` would be
/// overwritten anyway, but `expires_at` survives when no TTL is passed — letting a caller
/// set an expiry that never went through `ttl_seconds`.
pub(crate) fn strip_reserved_recency(attrs: &mut BTreeMap<String, Value>) {
    for key in [META_CREATED_AT, META_UPDATED_AT, META_EXPIRES_AT] {
        attrs.remove(key);
    }
}

/// Drop caller-supplied chunk provenance. Accepting these would let any write forge a
/// `parent_id`/`chunk_index`, and `remember_chunked`'s stale-tail prune matches on exactly
/// that pair, so a forged row is deletable by an unrelated document's re-ingest.
pub(crate) fn strip_reserved_chunk(attrs: &mut BTreeMap<String, Value>) {
    for key in [META_PARENT_ID, META_CHUNK_INDEX] {
        attrs.remove(key);
    }
}

/// Stamp recency attrs in place: `updated_at` always moves to `now_ms`; `created_at`
/// carries forward from `prior_created` (an update-in-place) or is set to `now_ms` (a
/// fresh entry); `expires_at` is set only when `ttl_seconds` is given.
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
            // Pinning a collection that already holds rows claims vectors we did not embed,
            // which would silence the recall guard for good — so flag it first (nidus-8ki).
            if collection_has_rows(db, collection)? {
                unpinned_collection(db, collection, &identity, "write")?;
            }
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

/// Whether `collection` already holds rows. Counted from the in-RAM index, so this is a
/// map walk rather than a scan of the vectors.
fn collection_has_rows(db: &Nidus, collection: &str) -> anyhow::Result<bool> {
    Ok(db
        .aggregate(collection, &crate::AggregateOpts::default())?
        .count
        > 0)
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

/// One `remember` write, minus the vector: what every surface has in hand before it
/// embeds, and what [`commit_remember`] needs after.
pub(crate) struct RememberWrite {
    pub id: String,
    /// The raw remembered text, stamped as [`META_TEXT`] in every mode (#131). Not
    /// necessarily what was embedded — a summarized write embeds the summary.
    pub text: String,
    pub attrs: BTreeMap<String, Value>,
    pub ttl_seconds: Option<i64>,
    pub dedupe_threshold: Option<f32>,
}

/// Embed `embed_text`, then commit. The seam the unit tests drive with a fake embedder;
/// the server surfaces embed off-lock themselves and call [`commit_remember`] directly.
async fn embed_and_commit<E: Embedder>(
    db: &mut Nidus,
    embedder: &E,
    collection: &str,
    embed_text: &str,
    write: RememberWrite,
) -> anyhow::Result<Remembered> {
    let vector = embedder
        .embed(embed_text)
        .await
        .with_context(|| format!("embedding text for '{collection}/{}'", write.id))?;
    commit_remember(db, embedder, collection, write, vector)
}

/// The seam [`Memory::remember_chunked`] delegates to, and the unit tests drive directly
/// with a fake embedder (`Memory` cannot be poured into, same as [`embed_and_commit`]).
/// See [`Memory::remember_chunked`] for the write-then-prune contract.
async fn remember_chunked_with<E: Embedder>(
    db: &mut Nidus,
    embedder: &E,
    collection: &str,
    parent_id: &str,
    text: &str,
    chunk_opts: &crate::chunk::ChunkOpts,
    opts: RememberOpts,
) -> anyhow::Result<ChunkedRemembered> {
    let RememberOpts {
        mode,
        attrs,
        ttl_seconds,
        dedupe_threshold,
    } = opts;
    // Silently honouring only `Raw` would embed a summarize-mode caller's raw chunks and
    // stamp no summary, reporting success for a write they did not ask for.
    match mode {
        RememberMode::Raw => {}
        #[cfg(feature = "summarize")]
        RememberMode::Summarize => bail!(
            "remember_chunked: RememberMode::Summarize is incompatible with chunking. \
             Summarize the document first and chunk the summary, or chunk with \
             RememberMode::Raw"
        ),
    }
    if dedupe_threshold.is_some() {
        bail!(
            "remember_chunked: dedupe_threshold is incompatible with chunking, since a dedup \
             match could redirect one document's chunk onto another document's, breaking the \
             parent_id/chunk_index invariant rollups group on"
        );
    }

    let chunks = crate::chunk::chunk_text(text, chunk_opts)?;
    if chunks.is_empty() {
        return Ok(ChunkedRemembered {
            parent_id: parent_id.to_string(),
            chunks: Vec::new(),
            pruned: 0,
        });
    }

    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let vectors = embedder.embed_batch(&texts).await.with_context(|| {
        format!(
            "embedding {} chunks for '{collection}/{parent_id}'",
            texts.len()
        )
    })?;

    let n = chunks.len();
    // `zip` would silently drop the tail while `n` still gates the prune below, writing
    // fewer chunks than the prune assumes survive.
    if vectors.len() != n {
        bail!(
            "remember_chunked: embedder returned {} vectors for {n} chunks of \
             '{collection}/{parent_id}'",
            vectors.len()
        );
    }
    let mut remembered = Vec::with_capacity(n);
    for (chunk, vector) in chunks.into_iter().zip(vectors) {
        let index = chunk.index;
        let write = RememberWrite {
            id: format!("{parent_id}#{index}"),
            text: chunk.text,
            attrs: attrs.clone(),
            ttl_seconds,
            dedupe_threshold: None,
        };
        remembered.push(commit_remember_chunk(
            db,
            embedder,
            collection,
            write,
            vector,
            (parent_id, index as i64),
        )?);
    }

    let pruned = db.delete_where(
        collection,
        &Filter(vec![Predicate::All(vec![
            Predicate::Eq(
                META_PARENT_ID.to_string(),
                Value::Str(parent_id.to_string()),
            ),
            Predicate::Ge(META_CHUNK_INDEX.to_string(), Value::Int(n as i64)),
        ])]),
    )?;

    Ok(ChunkedRemembered {
        parent_id: parent_id.to_string(),
        chunks: remembered,
        pruned,
    })
}

/// The store half of a `remember`, shared by the Rust, HTTP, and MCP surfaces so the
/// stamping, dedup, and recency rules cannot drift between them.
///
/// Sync and taking the vector already computed, because the callers that matter embed
/// off-lock and then run this whole body inside one write closure — the read-modify-write
/// is only atomic against other writers if the search, the read-back, and the upsert
/// share a single `&mut Nidus`.
pub(crate) fn commit_remember<E: Embedder>(
    db: &mut Nidus,
    embedder: &E,
    collection: &str,
    write: RememberWrite,
    vector: Vec<f32>,
) -> anyhow::Result<Remembered> {
    commit_remember_inner(db, embedder, collection, write, vector, None)
}

/// [`commit_remember`] for one chunk of a chunked document: same stamping, plus the
/// `(parent_id, chunk_index)` provenance, stamped here so it cannot be forged by a caller.
pub(crate) fn commit_remember_chunk<E: Embedder>(
    db: &mut Nidus,
    embedder: &E,
    collection: &str,
    write: RememberWrite,
    vector: Vec<f32>,
    chunk: (&str, i64),
) -> anyhow::Result<Remembered> {
    commit_remember_inner(db, embedder, collection, write, vector, Some(chunk))
}

fn commit_remember_inner<E: Embedder>(
    db: &mut Nidus,
    embedder: &E,
    collection: &str,
    write: RememberWrite,
    vector: Vec<f32>,
    chunk: Option<(&str, i64)>,
) -> anyhow::Result<Remembered> {
    let RememberWrite {
        id,
        text,
        mut attrs,
        ttl_seconds,
        dedupe_threshold,
    } = write;
    ensure_collection_and_pin(db, embedder, collection)?;

    // Stamped from here, never accepted from a caller — and before the dedup merge below,
    // so this write's text wins over the matched entry's.
    strip_reserved_recency(&mut attrs);
    strip_reserved_chunk(&mut attrs);
    if let Some((parent_id, index)) = chunk {
        attrs.insert(
            META_PARENT_ID.to_string(),
            Value::Str(parent_id.to_string()),
        );
        attrs.insert(META_CHUNK_INDEX.to_string(), Value::Int(index));
    }
    attrs.insert(META_TEXT.to_string(), Value::Str(text));

    let mut target_id = id;
    let mut deduped = false;
    if let Some(threshold) = dedupe_threshold {
        // An expired entry is dead to every read path, so it must not be a dedup candidate
        // either: merging onto one inherits its already-past `expires_at`, landing a write
        // that reports success and is never visible.
        let opts = SearchOpts {
            top_k: 1,
            min_score: Some(threshold),
            filter: Filter(vec![not_expired_predicate(now_ms())]),
            ..Default::default()
        };
        if let Some(hit) = db.search(collection, &vector, &opts)?.into_iter().next() {
            target_id = hit.id;
            deduped = true;
        }
    }

    // Read-before-write under the same borrow: recovers `created_at` from the store rather
    // than from caller attrs, which would let a write forge its own birth date, and — on a
    // dedup match only (D6) — the matched entry's other attrs, so an omitted field survives.
    let existing = db.get(collection, &target_id);
    let prior_created =
        existing
            .as_ref()
            .and_then(|record| match record.attrs.get(META_CREATED_AT) {
                Some(Value::DateTime(ms)) => Some(*ms),
                _ => None,
            });
    if let Some(record) = existing.filter(|_| deduped) {
        for (key, value) in record.attrs {
            attrs.entry(key).or_insert(value);
        }
    }
    stamp_recency(&mut attrs, now_ms(), prior_created, ttl_seconds);

    let upserted = db.upsert(collection, &[Record::new(target_id.clone(), vector, attrs)])?;
    Ok(Remembered {
        id: target_id,
        deduped,
        upserted,
    })
}

/// Recall-side identity guard: refuse a recall whose embedder differs from the one `collection` was
/// written with, since even a same-dimension mismatch returns meaningless cross-space rankings. An
/// unpinned collection cannot be checked, so it warns — or refuses under strict mode (nidus-8ki).
pub(crate) fn guard_recall_identity<E: Embedder>(
    db: &Nidus,
    embedder: &E,
    collection: &str,
) -> anyhow::Result<()> {
    let identity = embedder_identity(embedder);
    match db.get_meta(collection).get(META_EMBEDDER) {
        Some(existing) => bail_if_identity_differs(collection, existing, &identity)?,
        None if db.has_collection(collection) => {
            unpinned_collection(db, collection, &identity, "recall")?;
        }
        None => {}
    }
    Ok(())
}

/// A collection nidus never wrote through `Memory` carries no embedder identity, so a recall
/// against it cannot be checked: scores come back plausible whether or not the spaces agree.
/// Strict mode refuses; otherwise warn once per collection+embedder, since recall is hot.
fn unpinned_collection(
    db: &Nidus,
    collection: &str,
    identity: &str,
    op: &str,
) -> anyhow::Result<()> {
    if db.config().strict_embedder_identity {
        bail!(
            "collection '{collection}' has no pinned embedder ('{META_EMBEDDER}' collection meta), \
             so this {op} with '{identity}' cannot be checked for embedder agreement; it was \
             written outside nidus's memory API — pin it, use a separate collection, or turn \
             strict-embedder-identity off to allow the {op} anyway"
        );
    }
    if warn_once(collection, identity) {
        diag!(
            crate::diag::Level::Warn,
            "memory",
            "collection has no pinned embedder; cross-embedder results would look plausible",
            "collection" => collection,
            "embedder" => identity,
            "op" => op,
        );
    }
    Ok(())
}

/// Whether this `(collection, embedder)` pair still owes a warning. Bounded so a caller
/// naming arbitrary collections cannot grow the set without limit; past the cap every
/// occurrence warns, which is noisy rather than silent.
fn warn_once(collection: &str, identity: &str) -> bool {
    const CAP: usize = 256;
    static WARNED: std::sync::Mutex<Option<std::collections::BTreeSet<String>>> =
        std::sync::Mutex::new(None);
    let key = format!("{collection}\0{identity}");
    let Ok(mut guard) = WARNED.lock() else {
        return true;
    };
    let seen = guard.get_or_insert_with(std::collections::BTreeSet::new);
    if seen.contains(&key) {
        return false;
    }
    if seen.len() < CAP {
        seen.insert(key);
    }
    true
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
    // Same TTL guard as every MCP read tool (D4/D5): AND-ed in, never replacing the
    // caller's own predicates, so an expired entry cannot leak back into recall.
    let mut filter = opts.filter.clone().unwrap_or_default();
    filter.0.push(not_expired_predicate(now_ms()));
    let search_opts = SearchOpts {
        top_k: if opts.top_k == 0 {
            DEFAULT_TOP_K
        } else {
            opts.top_k
        },
        filter,
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

    /// A store opened with `strict_embedder_identity`, the opt-in that turns the unpinned
    /// collection from a warning into a refusal.
    fn open_tmp_strict(dim: usize) -> (tempfile::TempDir, Nidus) {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::Config::new(dir.path(), dim).strict_embedder_identity(true);
        let db = Nidus::open(cfg).unwrap();
        (dir, db)
    }

    /// A collection written straight through `upsert`, the way an external tool does it:
    /// rows, no `nidus.embedder` meta.
    fn upsert_unpinned(db: &mut Nidus, collection: &str, dim: usize) {
        db.create_collection(collection).unwrap();
        let vector = vec![0.5_f32; dim];
        db.upsert(
            collection,
            &[Record::new("foreign", vector, BTreeMap::new())],
        )
        .unwrap();
    }

    /// A `RememberMode::Raw` write through the real path — exactly what `Memory::remember`
    /// runs, minus the `Memory` wrapper the fake embedder cannot be poured into.
    async fn remember_with<E: Embedder>(
        db: &mut Nidus,
        embedder: &E,
        collection: &str,
        id: &str,
        text: &str,
        opts: RememberOpts,
    ) -> anyhow::Result<Remembered> {
        embed_and_commit(
            db,
            embedder,
            collection,
            text,
            RememberWrite {
                id: id.to_string(),
                text: text.to_string(),
                attrs: opts.attrs,
                ttl_seconds: opts.ttl_seconds,
                dedupe_threshold: opts.dedupe_threshold,
            },
        )
        .await
    }

    /// `remember_with` with no TTL and no dedup — the shape most of these tests want.
    async fn remember_raw<E: Embedder>(
        db: &mut Nidus,
        embedder: &E,
        collection: &str,
        id: &str,
        text: &str,
        attrs: BTreeMap<String, Value>,
    ) -> anyhow::Result<Remembered> {
        remember_with(
            db,
            embedder,
            collection,
            id,
            text,
            RememberOpts {
                attrs,
                ..Default::default()
            },
        )
        .await
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn remember_recall_round_trip() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        remember_raw(
            &mut db,
            &emb,
            "notes",
            "a",
            "the quick brown fox",
            BTreeMap::new(),
        )
        .await
        .unwrap();
        remember_raw(
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

        remember_raw(&mut db, &emb, "notes", "a", "hello world", BTreeMap::new())
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
        remember_raw(&mut db, &emb, "notes", "a", "hello world", BTreeMap::new())
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
        remember_raw(&mut db, &emb, "notes", "b", "more text", attrs)
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
        remember_raw(&mut db, &emb_v1, "notes", "a", "hello", BTreeMap::new())
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
        remember_raw(&mut db, &emb_v1, "notes", "a", "hello", BTreeMap::new())
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

    /// The warning is once per collection+embedder: recall is a hot path, and a line per
    /// query would be dropped by whoever set `NIDUS_LOG=error` to escape it.
    #[test]
    fn warn_once_repeats_only_for_a_new_collection_or_embedder() {
        assert!(warn_once("warn-once-notes", "fake/v1"));
        assert!(!warn_once("warn-once-notes", "fake/v1"));
        assert!(warn_once("warn-once-notes", "fake/v2"));
        assert!(warn_once("warn-once-other", "fake/v1"));
    }

    /// nidus-8ki: a collection written by an external tool carries no `nidus.embedder`, so
    /// the identity guard has nothing to compare. It must not pass silently — under the strict
    /// setting the recall is refused outright.
    #[tokio::test]
    #[cfg_attr(miri, ignore)] // file-backed via open_tmp_strict; `memory` is off in the Miri lane.
    async fn strict_mode_refuses_recall_on_an_unpinned_collection() {
        let (_dir, mut db) = open_tmp_strict(8);
        upsert_unpinned(&mut db, "notes", 8);

        let emb = FakeEmbedder::new(8, "fake", "v1");
        let err = recall_with(&db, &emb, "notes", "hello", &RecallOpts::default())
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("no pinned embedder") && msg.contains(META_EMBEDDER),
            "the message should name the missing pin: {msg}"
        );
    }

    /// The default stays permissive — refusing every raw-upsert store would break existing
    /// callers — so the same recall succeeds, having only warned via `diag`.
    #[tokio::test]
    #[cfg_attr(miri, ignore)] // file-backed via open_tmp; also `memory` is off in the Miri lane.
    async fn an_unpinned_collection_still_recalls_by_default() {
        let (_dir, mut db) = open_tmp(8);
        upsert_unpinned(&mut db, "notes", 8);

        let emb = FakeEmbedder::new(8, "fake", "v1");
        recall_with(&db, &emb, "notes", "hello", &RecallOpts::default())
            .await
            .expect("an unpinned collection warns, it does not refuse");
    }

    /// Strict mode also refuses to *pin* a collection that already holds foreign rows: doing so
    /// would claim vectors nidus never embedded and silence the recall guard from then on.
    #[tokio::test]
    #[cfg_attr(miri, ignore)] // file-backed via open_tmp_strict; `memory` is off in the Miri lane.
    async fn strict_mode_refuses_to_pin_a_populated_unpinned_collection() {
        let (_dir, mut db) = open_tmp_strict(8);
        upsert_unpinned(&mut db, "notes", 8);

        let emb = FakeEmbedder::new(8, "fake", "v1");
        let err = remember_raw(&mut db, &emb, "notes", "a", "hello", BTreeMap::new())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("no pinned embedder"),
            "the write should name the missing pin: {err}"
        );
        assert!(
            !db.get_meta("notes").contains_key(META_EMBEDDER),
            "a refused write must not have stamped the identity anyway"
        );
    }

    /// An *empty* collection is nobody's foreign data, so strict mode still lets the first
    /// `remember` create and pin it — the flag guards adoption, not creation.
    #[tokio::test]
    #[cfg_attr(miri, ignore)] // file-backed via open_tmp_strict; `memory` is off in the Miri lane.
    async fn strict_mode_still_pins_a_fresh_collection() {
        let (_dir, mut db) = open_tmp_strict(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        remember_raw(&mut db, &emb, "notes", "a", "hello", BTreeMap::new())
            .await
            .expect("a fresh collection is nobody else's data");
        assert_eq!(
            db.get_meta("notes").get(META_EMBEDDER).map(String::as_str),
            Some("fake/v1")
        );
        recall_with(&db, &emb, "notes", "hello", &RecallOpts::default())
            .await
            .expect("the collection is pinned now, so recall is checkable");
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // file-backed via open_tmp; also `memory` is off in the Miri lane.
    async fn recall_defaults_top_k_when_zero() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");
        for i in 0..3 {
            remember_raw(
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

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // file-backed via open_tmp; also `memory` is off in the Miri lane.
    async fn recall_hides_expired_entries_and_keeps_untimed_ones() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");
        for (id, text) in [("kept", "durable note"), ("gone", "ephemeral note")] {
            remember_raw(&mut db, &emb, "notes", id, text, BTreeMap::new())
                .await
                .unwrap();
        }
        // Backdate one entry's expiry — recall must hide it from that moment on (#106).
        let mut expired = db.get("notes", "gone").unwrap();
        expired.attrs.insert(
            META_EXPIRES_AT.to_string(),
            Value::DateTime(now_ms() - 1_000),
        );
        db.upsert("notes", &[expired]).unwrap();

        let hits = recall_with(&db, &emb, "notes", "durable note", &RecallOpts::default())
            .await
            .unwrap();
        let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        assert!(
            ids.contains(&"kept"),
            "an entry with no expires_at must still surface (D5): {ids:?}"
        );
        assert!(
            !ids.contains(&"gone"),
            "an expired entry must be hidden from recall: {ids:?}"
        );
    }

    // ── Parity with the HTTP and MCP write paths (#133, #131) ───────────────

    /// #131: the first write provisions an FTS schema over `nidus.text`, so a write that
    /// never stamps it maintains a BM25 index over an always-empty field — rebuilt on every
    /// open, and unmatchable by `text_search`. Fails outright without the stamping.
    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn remember_stamps_the_raw_text_and_it_is_full_text_searchable() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        remember_raw(
            &mut db,
            &emb,
            "notes",
            "a",
            "the quick brown fox",
            BTreeMap::new(),
        )
        .await
        .unwrap();

        let stored = db.get("notes", "a").unwrap();
        assert_eq!(
            stored.attrs.get(META_TEXT),
            Some(&Value::Str("the quick brown fox".to_string()))
        );

        let hits = db
            .text_search(
                "notes",
                &crate::FtsQuery::new(META_TEXT, "quick"),
                &SearchOpts {
                    top_k: 5,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            hits.len(),
            1,
            "the provisioned nidus.text index must actually have something in it"
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn remember_stamps_recency_and_carries_created_at_forward() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        remember_raw(&mut db, &emb, "notes", "a", "first", BTreeMap::new())
            .await
            .unwrap();
        let first = db.get("notes", "a").unwrap().attrs;
        let created = match first.get(META_CREATED_AT) {
            Some(Value::DateTime(ms)) => *ms,
            other => panic!("created_at must be stamped, got {other:?}"),
        };
        assert!(matches!(
            first.get(META_UPDATED_AT),
            Some(Value::DateTime(_))
        ));
        assert!(
            !first.contains_key(META_EXPIRES_AT),
            "no ttl means no expiry, not an expiry of zero"
        );

        remember_raw(&mut db, &emb, "notes", "a", "second", BTreeMap::new())
            .await
            .unwrap();
        let second = db.get("notes", "a").unwrap().attrs;
        assert_eq!(
            second.get(META_CREATED_AT),
            Some(&Value::DateTime(created)),
            "a re-remember must not reset the birth date"
        );
    }

    /// A caller cannot hand-write the reserved keys: `expires_at` in particular survives a
    /// write with no `ttl_seconds` unless it is stripped, which is an expiry that never went
    /// through the TTL knob.
    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn caller_supplied_recency_attrs_are_stripped() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        let forged = BTreeMap::from([
            (META_CREATED_AT.to_string(), Value::DateTime(1)),
            (META_EXPIRES_AT.to_string(), Value::DateTime(2)),
            ("keep".to_string(), Value::Str("mine".to_string())),
        ]);
        remember_raw(&mut db, &emb, "notes", "a", "text", forged)
            .await
            .unwrap();

        let stored = db.get("notes", "a").unwrap().attrs;
        assert!(
            !stored.contains_key(META_EXPIRES_AT),
            "a forged expires_at must not survive a write that passed no ttl"
        );
        assert_ne!(
            stored.get(META_CREATED_AT),
            Some(&Value::DateTime(1)),
            "created_at comes from the store, not the caller"
        );
        assert_eq!(
            stored.get("keep"),
            Some(&Value::Str("mine".to_string())),
            "stripping the reserved keys must not touch the caller's own"
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn ttl_sets_expires_at_and_hides_the_entry_from_recall() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        let ttl = RememberOpts {
            ttl_seconds: Some(-1), // already elapsed, so no sleep is needed
            ..Default::default()
        };
        remember_with(&mut db, &emb, "notes", "gone", "ephemeral note", ttl)
            .await
            .unwrap();
        remember_raw(
            &mut db,
            &emb,
            "notes",
            "kept",
            "durable note",
            BTreeMap::new(),
        )
        .await
        .unwrap();

        assert!(
            db.get("notes", "gone")
                .unwrap()
                .attrs
                .contains_key(META_EXPIRES_AT)
        );
        let hits = recall_with(&db, &emb, "notes", "ephemeral note", &RecallOpts::default())
            .await
            .unwrap();
        let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["kept"],
            "the expired entry must be gone and the untimed one must stay: {ids:?}"
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn dedupe_redirects_the_write_and_merges_attrs() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        let first = remember_with(
            &mut db,
            &emb,
            "notes",
            "original",
            "the ranking bug is in the upsert path",
            RememberOpts {
                attrs: BTreeMap::from([
                    ("kind".to_string(), Value::Str("bug".to_string())),
                    ("owner".to_string(), Value::Str("ana".to_string())),
                ]),
                dedupe_threshold: Some(0.99),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(first.id, "original");
        assert!(!first.deduped, "nothing to match on the first write");

        // The same text under a different id: the threshold must fold it onto the original.
        let second = remember_with(
            &mut db,
            &emb,
            "notes",
            "duplicate",
            "the ranking bug is in the upsert path",
            RememberOpts {
                attrs: BTreeMap::from([("kind".to_string(), Value::Str("defect".to_string()))]),
                dedupe_threshold: Some(0.99),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(second.deduped, "a near-duplicate must be detected");
        assert_eq!(second.id, "original", "the write redirects onto the match");
        assert!(
            db.get("notes", "duplicate").is_none(),
            "no competing entry may be created"
        );

        let merged = db.get("notes", "original").unwrap().attrs;
        assert_eq!(
            merged.get("kind"),
            Some(&Value::Str("defect".to_string())),
            "a supplied key wins the collision"
        );
        assert_eq!(
            merged.get("owner"),
            Some(&Value::Str("ana".to_string())),
            "a key this write omitted must survive on the matched entry"
        );
    }

    /// An expired entry is invisible to every read path, so folding onto one would inherit
    /// its already-past `expires_at` — a write that reports success and is never visible.
    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn an_expired_entry_is_not_a_dedupe_candidate() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");
        let text = "the ranking bug is in the upsert path";

        remember_with(
            &mut db,
            &emb,
            "notes",
            "stale",
            text,
            RememberOpts {
                ttl_seconds: Some(-1),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let fresh = remember_with(
            &mut db,
            &emb,
            "notes",
            "fresh",
            text,
            RememberOpts {
                dedupe_threshold: Some(0.99),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert!(
            !fresh.deduped,
            "an expired entry must not be matched as a near-duplicate"
        );
        assert_eq!(fresh.id, "fresh");
        assert!(
            !db.get("notes", "fresh")
                .unwrap()
                .attrs
                .contains_key(META_EXPIRES_AT),
            "the new entry must not inherit the expired one's expiry"
        );
    }

    /// Dedup carries the matched entry's birth date, not the write's own moment — the
    /// merge would otherwise reset the age of every entry it folds onto.
    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn dedupe_preserves_the_matched_entrys_created_at() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");
        let text = "the warm index is shared via redis";

        remember_raw(&mut db, &emb, "notes", "original", text, BTreeMap::new())
            .await
            .unwrap();
        let created = db.get("notes", "original").unwrap().attrs[META_CREATED_AT].clone();

        remember_with(
            &mut db,
            &emb,
            "notes",
            "duplicate",
            text,
            RememberOpts {
                dedupe_threshold: Some(0.99),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(
            db.get("notes", "original").unwrap().attrs[META_CREATED_AT],
            created
        );
    }

    /// 10-char chunks, no overlap, so a run of identical filler chars splits by hard char
    /// count and the chunk count is exactly predictable (`ceil(len / 10)`).
    fn small_chunk_opts() -> crate::chunk::ChunkOpts {
        crate::chunk::ChunkOpts {
            strategy: crate::chunk::ChunkStrategy::Recursive,
            max_chars: 10,
            overlap_chars: 0,
        }
    }

    #[tokio::test]
    async fn remember_chunked_rejects_dedupe_threshold() {
        let mut db = Nidus::open_in_memory(8).unwrap();
        let emb = FakeEmbedder::new(8, "fake", "v1");

        let err = remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-1",
            "some text to chunk",
            &small_chunk_opts(),
            RememberOpts {
                dedupe_threshold: Some(0.9),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("dedup"), "{err}");
    }

    #[tokio::test]
    async fn remember_chunked_empty_text_writes_nothing() {
        let mut db = Nidus::open_in_memory(8).unwrap();
        let emb = FakeEmbedder::new(8, "fake", "v1");

        let result = remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-1",
            "   \n\t  ",
            &small_chunk_opts(),
            RememberOpts::default(),
        )
        .await
        .unwrap();

        assert!(result.chunks.is_empty());
        assert_eq!(result.pruned, 0);
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn remember_chunked_stamps_parent_and_index_on_every_chunk() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        let result = remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-1",
            &"a".repeat(95),
            &small_chunk_opts(),
            RememberOpts::default(),
        )
        .await
        .unwrap();

        assert_eq!(result.chunks.len(), 10);
        assert_eq!(result.pruned, 0);
        for (i, remembered) in result.chunks.iter().enumerate() {
            assert_eq!(remembered.id, format!("doc-1#{i}"));
            let record = db.get("docs", &remembered.id).unwrap();
            assert_eq!(
                record.attrs[META_PARENT_ID],
                Value::Str("doc-1".to_string())
            );
            assert_eq!(record.attrs[META_CHUNK_INDEX], Value::Int(i as i64));
            assert!(record.attrs.contains_key(META_TEXT));
            assert!(record.attrs.contains_key(META_CREATED_AT));
            assert!(record.attrs.contains_key(META_UPDATED_AT));
        }
    }

    /// The whole point of upsert-then-prune: re-chunking a shortened document must not
    /// leave the longer version's tail behind.
    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn remember_chunked_prunes_stale_tail_on_reingest() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");
        let opts = small_chunk_opts();

        let first = remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-1",
            &"a".repeat(95),
            &opts,
            RememberOpts::default(),
        )
        .await
        .unwrap();
        assert_eq!(first.chunks.len(), 10);

        let second = remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-1",
            &"a".repeat(25),
            &opts,
            RememberOpts::default(),
        )
        .await
        .unwrap();
        assert_eq!(second.chunks.len(), 3);
        assert_eq!(second.pruned, 7);

        let ids: Vec<String> = db
            .get_all("docs")
            .into_iter()
            .map(|record| record.id)
            .collect();
        assert_eq!(ids.len(), 3, "the stale indices 3..9 must be gone: {ids:?}");
        for i in 0..3 {
            assert!(ids.contains(&format!("doc-1#{i}")));
        }
        for i in 3..10 {
            assert!(!ids.contains(&format!("doc-1#{i}")));
        }
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn remember_chunked_reingesting_identical_text_is_idempotent() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");
        let opts = small_chunk_opts();
        let text = "a".repeat(25);

        let first = remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-1",
            &text,
            &opts,
            RememberOpts::default(),
        )
        .await
        .unwrap();
        let created: Vec<Value> = first
            .chunks
            .iter()
            .map(|r| db.get("docs", &r.id).unwrap().attrs[META_CREATED_AT].clone())
            .collect();

        std::thread::sleep(std::time::Duration::from_millis(5));
        let second = remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-1",
            &text,
            &opts,
            RememberOpts::default(),
        )
        .await
        .unwrap();

        assert_eq!(second.pruned, 0);
        let first_ids: Vec<&str> = first.chunks.iter().map(|r| r.id.as_str()).collect();
        let second_ids: Vec<&str> = second.chunks.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(first_ids, second_ids);

        for (i, remembered) in second.chunks.iter().enumerate() {
            let record = db.get("docs", &remembered.id).unwrap();
            assert_eq!(record.attrs[META_CREATED_AT], created[i]);
            assert_ne!(record.attrs[META_UPDATED_AT], created[i]);
        }
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn a_forged_parent_id_is_stripped_and_survives_an_unrelated_prune() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");

        // A plain remember forging another document's chunk provenance.
        let mut attrs = BTreeMap::new();
        attrs.insert(META_PARENT_ID.to_string(), Value::Str("doc-1".to_string()));
        attrs.insert(META_CHUNK_INDEX.to_string(), Value::Int(999));
        embed_and_commit(
            &mut db,
            &emb,
            "docs",
            "victim text",
            RememberWrite {
                id: "victim".to_string(),
                text: "victim text".to_string(),
                attrs,
                ttl_seconds: None,
                dedupe_threshold: None,
            },
        )
        .await
        .unwrap();

        let victim = db.get("docs", "victim").unwrap();
        assert!(
            !victim.attrs.contains_key(META_PARENT_ID),
            "caller-supplied parent_id must be stripped: {:?}",
            victim.attrs
        );
        assert!(!victim.attrs.contains_key(META_CHUNK_INDEX));

        // doc-1's own re-ingest prunes chunk_index >= n for parent doc-1. The forged row
        // must not be caught by it.
        remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-1",
            &"a".repeat(20),
            &small_chunk_opts(),
            RememberOpts::default(),
        )
        .await
        .unwrap();

        assert!(
            db.get("docs", "victim").is_some(),
            "an unrelated record was deleted by doc-1's prune"
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn remember_chunked_rejects_a_short_embed_batch() {
        struct ShortEmbedder(FakeEmbedder);
        impl Embedder for ShortEmbedder {
            fn embed(
                &self,
                text: &str,
            ) -> impl Future<Output = Result<Vec<f32>, EmbedError>> + Send {
                self.0.embed(text)
            }
            fn embed_batch(
                &self,
                texts: &[&str],
            ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbedError>> + Send {
                // One vector short: the tail would be silently dropped by `zip`.
                let mut vs: Vec<Vec<f32>> = texts.iter().map(|t| self.0.vector_for(t)).collect();
                vs.pop();
                async move { Ok(vs) }
            }
            fn dimension(&self) -> usize {
                self.0.dimension()
            }
            fn provider_name(&self) -> &str {
                self.0.provider_name()
            }
            fn model_name(&self) -> &str {
                self.0.model_name()
            }
            fn max_input_tokens(&self) -> usize {
                self.0.max_input_tokens()
            }
        }

        let (_dir, mut db) = open_tmp(8);
        let emb = ShortEmbedder(FakeEmbedder::new(8, "fake", "v1"));
        let err = remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-1",
            &"a".repeat(95),
            &small_chunk_opts(),
            RememberOpts::default(),
        )
        .await
        .expect_err("a short embed_batch must be an error, not a silent truncation");
        let msg = err.to_string();
        assert!(msg.contains("vectors"), "unhelpful message: {msg}");
        assert!(
            db.get_all("docs").is_empty(),
            "nothing should have been written"
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // upsert fsyncs
    async fn remember_chunked_does_not_prune_a_different_parent() {
        let (_dir, mut db) = open_tmp(8);
        let emb = FakeEmbedder::new(8, "fake", "v1");
        let opts = small_chunk_opts();

        remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-a",
            &"a".repeat(95),
            &opts,
            RememberOpts::default(),
        )
        .await
        .unwrap();
        remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-b",
            &"b".repeat(25),
            &opts,
            RememberOpts::default(),
        )
        .await
        .unwrap();

        let second = remember_chunked_with(
            &mut db,
            &emb,
            "docs",
            "doc-b",
            &"b".repeat(15),
            &opts,
            RememberOpts::default(),
        )
        .await
        .unwrap();
        assert_eq!(second.pruned, 1);

        let ids: Vec<String> = db
            .get_all("docs")
            .into_iter()
            .map(|record| record.id)
            .collect();
        assert_eq!(
            ids.len(),
            12,
            "doc-a's 10 chunks must survive doc-b's prune: {ids:?}"
        );
        for i in 0..10 {
            assert!(ids.contains(&format!("doc-a#{i}")));
        }
    }
}
