//! Disk-backed Voyage embedder for the retrieval bench (nidus-yq9p.5).
//!
//! Wraps a `voyage-4` [`AnyEmbedder`] in nidus's own [`CachedEmbedder`] rather than writing a
//! second cache format: `CachedEmbedder::open` is already a content-hash disk cache keyed by
//! `(identity, dimension)`, so a model or dimension change invalidates cleanly instead of
//! silently serving vectors from a different embedding space.
//!
//! ## Queries are embedded as documents
//!
//! `CachedEmbedder` deliberately never caches `embed_query` (see its own module docs and
//! tests): a provider that tags document and query text differently could otherwise be made
//! to serve a document-tagged vector back for a query, so the cache refuses to guess and
//! always calls straight through, uncached. BEIR queries are few (300-648 per dataset), but
//! re-embedding them on every rerun is avoidable API spend and, worse, a source of
//! run-to-run variation if the provider is nondeterministic.
//!
//! This lane's caller therefore embeds queries through [`CachedEmbedder::embed_batch`] (the
//! **document** path), not `embed_query`, so query vectors land in the same cache as
//! documents. Voyage's adapter gives `embed_batch` no way to ask for query-style tagging: it
//! always sends `input_type: "document"` on that path, and `"query"` only from `embed_query`
//! (`src/embed/voyage.rs`). So the vectors this lane scores queries with are document-tagged,
//! not the query-tagged tensor Voyage's own published nDCG numbers were measured with. That
//! is a deliberate methodology choice, not an oversight, and the retrieval-quality writeup
//! must carry it forward: expect it to cost a little nDCG relative to Voyage's own numbers.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use nidus::LocalFs;
use nidus::backend::Persistence;
use nidus::embed::cache::CachedEmbedder;
use nidus::embed::{AnyEmbedder, EmbedConfig, EmbedProvider, Embedder};

/// Cache entries to keep per dataset identity: must exceed the largest BEIR corpus this
/// lane runs (FiQA: 57,638 docs) plus its queries, with headroom, or eviction thrashes and
/// a rerun re-embeds instead of hitting cache.
const MAX_CACHE_ENTRIES: usize = 100_000;

/// Environment variable holding the Voyage API key, named so the "absent" error can point
/// at it exactly.
const VOYAGE_API_KEY_VAR: &str = "VOYAGE_API_KEY";

/// Cache identity for a `(model, dataset)` pair, namespacing the cache so a model change
/// (e.g. moving off `voyage-4`) or a different dataset can never reuse another's vectors.
pub fn cache_identity(model: &str, dataset: &str) -> String {
    format!("voyage/{model}/{dataset}")
}

/// Hard-errors when absent: this lane needs a live key and network, and a keyless degraded
/// mode was rejected at scope. Takes the looked-up value, not the environment itself, so a
/// test needs no env mutation.
fn require_api_key(found: Option<String>) -> Result<String> {
    found.ok_or_else(|| {
        anyhow!(
            "{VOYAGE_API_KEY_VAR} is not set: the retrieval bench needs a live Voyage API key \
             and network access"
        )
    })
}

/// Build a `voyage-4` embedder whose vectors persist under `cache_dir`, keyed by `identity`
/// (build one with [`cache_identity`]). The caller **must** call `.save()` when the run
/// ends, or any API spend behind newly-cached vectors is thrown away.
///
/// `cache_dir` is given its OWN subdirectory per identity. `CachedEmbedder` writes a fixed
/// object name and carries the identity only in its validity key, so datasets sharing one
/// directory each load the previous one's blob as stale, re-embed in full, and then
/// overwrite it. One directory each is what actually makes a rerun free.
pub async fn cached_voyage(
    cache_dir: &Path,
    identity: &str,
) -> Result<CachedEmbedder<AnyEmbedder>> {
    let key = require_api_key(std::env::var(VOYAGE_API_KEY_VAR).ok())?;
    // Model is explicit, not `EmbedProvider::Voyage::default_model()`: the published run
    // must record which model it used, and an implicit default would silently follow a
    // future nidus release onto a different one.
    let cfg = EmbedConfig::new("voyage-4").api_key(key);
    let inner = AnyEmbedder::build(EmbedProvider::Voyage, cfg)
        .await
        .map_err(|e| anyhow!("failed to build voyage-4 embedder: {e}"))?;
    // Dimension comes from the built embedder, not a hardcoded 1024, so this can never
    // drift from what `AnyEmbedder`/Voyage itself considers `voyage-4`'s width to be.
    let dimension = inner.dimension();
    let dir = cache_dir.join("embeddings").join(slug(identity));
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let persistence: Arc<dyn Persistence> = Arc::new(LocalFs::new(&dir)?);
    Ok(CachedEmbedder::open(
        inner,
        Some(persistence),
        identity,
        dimension,
        MAX_CACHE_ENTRIES,
    ))
}

/// Filesystem-safe form of a cache identity, for use as a directory name.
fn slug(identity: &str) -> String {
    identity
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_api_key_is_an_error() {
        let err = require_api_key(None).unwrap_err().to_string();
        assert!(
            err.contains(VOYAGE_API_KEY_VAR),
            "error must name {VOYAGE_API_KEY_VAR}, got: {err}"
        );
    }

    #[test]
    fn slug_is_filesystem_safe_and_still_distinguishes() {
        // The slug is a directory name, so it must keep identities apart after mangling:
        // collapsing separators is fine, collapsing two identities into one is not.
        assert_eq!(slug("voyage-4/fiqa"), "voyage-4-fiqa");
        assert_ne!(
            slug(&cache_identity("voyage-4", "fiqa")),
            slug(&cache_identity("voyage-4", "nfcorpus")),
            "two datasets must not share a cache directory"
        );
        assert_ne!(
            slug(&cache_identity("voyage-4", "fiqa")),
            slug(&cache_identity("voyage-4-lite", "fiqa")),
            "two models must not share a cache directory"
        );
        assert!(
            slug(&cache_identity("voyage-4", "fiqa"))
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-'),
            "slug must be safe as a directory name"
        );
    }

    #[test]
    fn identity_distinguishes_model_and_dataset() {
        let a = cache_identity("voyage-4", "fiqa");
        let b = cache_identity("voyage-4-lite", "fiqa");
        let c = cache_identity("voyage-4", "nfcorpus");
        assert_ne!(a, b, "different model must not share a cache identity");
        assert_ne!(a, c, "different dataset must not share a cache identity");
        assert_ne!(b, c);
    }
}
