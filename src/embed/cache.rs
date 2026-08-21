//! Content-hash embedding cache (nidus-lvo.3): an unchanged chunk is never re-embedded.
//! Embedding is the billed, rate-limited part of ingestion, so re-indexing a corpus where
//! three files changed should cost three files' worth of API calls, not the whole corpus.
//!
//! A **sidecar object**, not a reserved collection. The ticket left that open and warned
//! against defaulting to the collection form without resolving the `Scope::All` leak: a
//! cached vector living in a collection is visible to whole-store scans, which the criterion
//! "cached data never leaks into user search results" forbids outright. `crate::index_cache`
//! already gives the sidecar everything needed - CRC'd framing, an opaque validity key, an
//! atomic whole-object `put`, and a load that degrades to `None` rather than erroring.
//!
//! The cache is a **decorator over [`Embedder`]**, so nothing in the memory path knows it
//! exists: wrap an embedder and pass it wherever an `impl Embedder` is taken.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use super::{EmbedError, Embedder};
use crate::backend::Persistence;

/// The sidecar object name under the store's persistence backend.
const OBJECT: &str = "embed-cache";
/// Entries kept before the oldest are evicted. A first ingest embeds the whole corpus, so
/// an uncapped cache would hold it all: 100k chunks at 1024 dims is ~400MB of `f32`.
pub const DEFAULT_MAX_ENTRIES: usize = 50_000;

/// What a cached vector is only valid under. Anything outside this tuple changing means the
/// vectors are unusable rather than stale, so it invalidates the whole cache instead of
/// silently serving a vector from a different model.
fn validity_key(identity: &str, dimension: usize) -> Vec<u8> {
    format!("embed-cache-v1|{identity}|{dimension}").into_bytes()
}

/// Fixed-key hasher, so an entry written by one process is found by the next. `DefaultHasher`
/// is not a cryptographic digest; a collision would serve the wrong vector, which at 64 bits
/// over a corpus of this scale is far below the noise floor of the embedding itself.
fn content_hash(text: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

/// How much work the cache saved, reported so a caller can assert on counts rather than on
/// a duration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evicted: usize,
}

/// The cache body: insertion-ordered so eviction has a defined victim, plus the map for
/// lookup. `order` holds hashes, never a second copy of the vectors.
#[derive(Default)]
struct Entries {
    map: HashMap<u64, Vec<f32>>,
    order: Vec<u64>,
    dirty: bool,
    stats: CacheStats,
}

impl Entries {
    fn insert(&mut self, hash: u64, vector: Vec<f32>, max_entries: usize) {
        if self.map.insert(hash, vector).is_none() {
            self.order.push(hash);
        }
        while self.order.len() > max_entries {
            let victim = self.order.remove(0);
            self.map.remove(&victim);
            self.stats.evicted += 1;
        }
        self.dirty = true;
    }
}

/// An [`Embedder`] that answers from a content-hash cache and only calls `inner` for the
/// texts it has never seen. Wrap the real embedder; every surface above stays unchanged.
pub struct CachedEmbedder<E: Embedder> {
    inner: E,
    persistence: Option<Arc<dyn Persistence>>,
    key: Vec<u8>,
    max_entries: usize,
    entries: Mutex<Entries>,
}

impl<E: Embedder> CachedEmbedder<E> {
    /// Wrap `inner`, loading any cache already stored under `persistence`. A `None` backend
    /// (an in-memory store) still caches for the life of the run; `max_entries == 0` is the
    /// full bypass. A stale or corrupt object loads as empty: a re-embed, never a failure.
    pub fn open(
        inner: E,
        persistence: Option<Arc<dyn Persistence>>,
        identity: &str,
        dimension: usize,
        max_entries: usize,
    ) -> Self {
        let key = validity_key(identity, dimension);
        let mut entries = Entries::default();
        if let Some(p) = persistence.as_deref()
            && let Ok(Some((stored, _))) =
                crate::index_cache::load::<Vec<(u64, Vec<f32>)>>(p, OBJECT, &key)
        {
            for (hash, vector) in stored {
                if vector.len() == dimension && entries.map.insert(hash, vector).is_none() {
                    entries.order.push(hash);
                }
            }
        }
        entries.dirty = false;
        Self {
            inner,
            persistence,
            key,
            max_entries,
            entries: Mutex::new(entries),
        }
    }

    /// Hits, misses and evictions so far.
    pub fn stats(&self) -> CacheStats {
        self.entries.lock().expect("cache mutex").stats
    }

    /// Number of cached vectors.
    pub fn len(&self) -> usize {
        self.entries.lock().expect("cache mutex").order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Persist the cache if anything changed. Called once at the end of a run, not per
    /// batch: the payload is the whole map, so per-batch saves would be quadratic.
    pub fn save(&self) -> anyhow::Result<()> {
        let mut entries = self.entries.lock().expect("cache mutex");
        let Some(p) = self.persistence.as_deref() else {
            return Ok(());
        };
        if !entries.dirty {
            return Ok(());
        }
        let payload: Vec<(u64, Vec<f32>)> = entries
            .order
            .iter()
            .filter_map(|h| entries.map.get(h).map(|v| (*h, v.clone())))
            .collect();
        let watermark = payload.len() as u64;
        crate::index_cache::save(p, OBJECT, &self.key, watermark, &payload)?;
        entries.dirty = false;
        Ok(())
    }

    /// Unwrap back to the inner embedder.
    pub fn into_inner(self) -> E {
        self.inner
    }
}

impl<E: Embedder> Embedder for CachedEmbedder<E> {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self.embed_batch(&[text]).await?.remove(0))
    }

    /// Cache lookups first, then **one** call for the misses only, then splice back into
    /// input order. An all-hit batch makes no call at all, not a zero-length one.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        // `max_entries == 0` is the caller's full bypass (`--no-cache`), so it must not
        // dedupe within the run either — that would be a cache the flag says is off.
        if self.max_entries == 0 {
            return self.inner.embed_batch(texts).await;
        }
        let hashes: Vec<u64> = texts.iter().map(|t| content_hash(t)).collect();
        let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        let mut missing: Vec<usize> = Vec::new();
        {
            let mut entries = self.entries.lock().expect("cache mutex");
            for (i, hash) in hashes.iter().enumerate() {
                let hit = entries.map.get(hash).cloned();
                match hit {
                    Some(v) => {
                        entries.stats.hits += 1;
                        out.push(Some(v));
                    }
                    None => {
                        entries.stats.misses += 1;
                        out.push(None);
                        missing.push(i);
                    }
                }
            }
        }
        if missing.is_empty() {
            return Ok(out.into_iter().map(|v| v.expect("all hits")).collect());
        }

        let wanted: Vec<&str> = missing.iter().map(|&i| texts[i]).collect();
        let fetched = self.inner.embed_batch(&wanted).await?;
        if fetched.len() != wanted.len() {
            return Err(EmbedError::Decode(format!(
                "embedder returned {} vectors for {} texts",
                fetched.len(),
                wanted.len()
            )));
        }
        {
            let mut entries = self.entries.lock().expect("cache mutex");
            for (&i, vector) in missing.iter().zip(fetched) {
                entries.insert(hashes[i], vector.clone(), self.max_entries);
                out[i] = Some(vector);
            }
        }
        Ok(out
            .into_iter()
            .map(|v| v.expect("every slot filled"))
            .collect())
    }

    /// Deliberately uncached: providers that distinguish document from query (Voyage,
    /// Cohere, Gemini, Jina) return a different tensor, so a shared cache would serve a
    /// document vector for a query.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.inner.embed_query(text).await
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn max_input_tokens(&self) -> usize {
        self.inner.max_input_tokens()
    }
    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const DIM: usize = 3;

    /// An in-memory `Persistence` so these tests never touch a file and stay Miri-clean.
    #[derive(Default)]
    struct MemStore(Mutex<HashMap<String, Vec<u8>>>);

    impl Persistence for MemStore {
        fn get(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
            Ok(self.0.lock().unwrap().get(key).cloned())
        }
        fn put(&self, key: &str, bytes: &[u8]) -> anyhow::Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert(key.to_string(), bytes.to_vec());
            Ok(())
        }
        fn delete(&self, key: &str) -> anyhow::Result<()> {
            self.0.lock().unwrap().remove(key);
            Ok(())
        }
        fn list(&self) -> anyhow::Result<Vec<String>> {
            Ok(self.0.lock().unwrap().keys().cloned().collect())
        }
        // The cache never takes a lease; only get/put are exercised.
        fn try_lock(
            &self,
            _key: &str,
            _ttl: std::time::Duration,
        ) -> anyhow::Result<Option<Box<dyn crate::backend::BackendLock>>> {
            Ok(None)
        }
    }

    /// Records every text it was asked for, so a test can assert on *which* chunks were
    /// sent rather than only how many calls happened.
    struct Counting {
        calls: AtomicUsize,
        asked: Mutex<Vec<String>>,
    }

    impl Counting {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                asked: Mutex::new(Vec::new()),
            }
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
        fn asked(&self) -> Vec<String> {
            self.asked.lock().unwrap().clone()
        }
    }

    // A per-text deterministic vector, so a wrong-vector-for-a-text bug is visible.
    fn vector_for(text: &str) -> Vec<f32> {
        let h = content_hash(text);
        vec![
            (h & 0xff) as f32,
            ((h >> 8) & 0xff) as f32,
            ((h >> 16) & 0xff) as f32,
        ]
    }

    impl Embedder for Counting {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            Ok(self.embed_batch(&[text]).await?.remove(0))
        }
        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.asked
                .lock()
                .unwrap()
                .extend(texts.iter().map(|t| t.to_string()));
            Ok(texts.iter().map(|t| vector_for(t)).collect())
        }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.asked.lock().unwrap().push(format!("query:{text}"));
            Ok(vector_for(text))
        }
        fn dimension(&self) -> usize {
            DIM
        }
        fn max_input_tokens(&self) -> usize {
            8192
        }
        fn provider_name(&self) -> &str {
            "test"
        }
        fn model_name(&self) -> &str {
            "fake-v1"
        }
    }

    fn block<F: std::future::Future>(f: F) -> F::Output {
        // A hand-rolled block_on: these futures never yield to a reactor, so a runtime
        // would be a dependency for nothing and would not run under Miri.
        use std::task::{Context, Poll, Waker};
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut f = Box::pin(f);
        loop {
            if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }

    fn store() -> Arc<MemStore> {
        Arc::new(MemStore::default())
    }

    fn cache(inner: Counting, p: &Arc<MemStore>) -> CachedEmbedder<Counting> {
        let p: Arc<dyn Persistence> = p.clone();
        CachedEmbedder::open(inner, Some(p), "test/fake-v1", DIM, DEFAULT_MAX_ENTRIES)
    }

    fn cache_with(
        inner: Counting,
        p: &Arc<MemStore>,
        identity: &str,
        dim: usize,
        max: usize,
    ) -> CachedEmbedder<Counting> {
        let p: Arc<dyn Persistence> = p.clone();
        CachedEmbedder::open(inner, Some(p), identity, dim, max)
    }

    /// lvo.3's central criterion: an unchanged chunk is never re-embedded. Asserts the
    /// inner call count, not that the vectors happened to match.
    #[test]
    fn a_repeat_batch_calls_the_inner_embedder_zero_times() {
        let p = store();
        let texts = ["alpha", "beta"];
        let first = {
            let c = cache(Counting::new(), &p);
            let got = block(c.embed_batch(&texts)).unwrap();
            assert_eq!(c.inner.calls(), 1, "first run must actually embed");
            c.save().unwrap();
            got
        };

        let c = cache(Counting::new(), &p);
        let again = block(c.embed_batch(&texts)).unwrap();
        assert_eq!(
            c.inner.calls(),
            0,
            "a fully cached batch must make no call at all"
        );
        assert_eq!(again, first, "and must return the same vectors");
        assert_eq!(c.stats().hits, 2);
        assert_eq!(c.stats().misses, 0);
    }

    /// The partial case: only the misses go to the provider. Asserts the *texts* sent, so
    /// an implementation that re-sends everything and discards the extras fails.
    #[test]
    fn a_partly_cached_batch_asks_only_for_the_misses() {
        let p = store();
        {
            let c = cache(Counting::new(), &p);
            block(c.embed_batch(&["alpha", "beta"])).unwrap();
            c.save().unwrap();
        }
        let c = cache(Counting::new(), &p);
        block(c.embed_batch(&["alpha", "gamma", "beta"])).unwrap();
        assert_eq!(c.inner.asked(), vec!["gamma"], "only the miss is requested");
        assert_eq!((c.stats().hits, c.stats().misses), (2, 1));
    }

    /// The silent-corruption bug this guards: `remember_chunked` zips vectors onto chunks
    /// positionally, so a hits-then-misses concatenation attaches every chunk the wrong
    /// vector. The miss here is in the middle, where that reordering is visible.
    #[test]
    fn order_survives_a_mixed_hit_miss_batch() {
        let p = store();
        {
            let c = cache(Counting::new(), &p);
            block(c.embed_batch(&["first", "third"])).unwrap();
            c.save().unwrap();
        }
        let c = cache(Counting::new(), &p);
        let texts = ["first", "second", "third"];
        let got = block(c.embed_batch(&texts)).unwrap();
        for (text, vector) in texts.iter().zip(&got) {
            assert_eq!(
                vector,
                &vector_for(text),
                "{text} got another text's vector"
            );
        }
    }

    /// A model swap or a dimension change must miss rather than serve a vector from a
    /// different regime. The model+dim live in the validity key, so this invalidates whole.
    #[test]
    fn a_changed_model_or_dimension_invalidates() {
        let p = store();
        {
            let c = cache(Counting::new(), &p);
            block(c.embed_batch(&["alpha"])).unwrap();
            c.save().unwrap();
        }
        let other_model = cache_with(Counting::new(), &p, "test/other-v2", DIM, 100);
        block(other_model.embed_batch(&["alpha"])).unwrap();
        assert_eq!(other_model.inner.calls(), 1, "a new model must re-embed");

        let other_dim = cache_with(Counting::new(), &p, "test/fake-v1", 4, 100);
        block(other_dim.embed_batch(&["alpha"])).unwrap();
        assert_eq!(other_dim.inner.calls(), 1, "a new dimension must re-embed");
    }

    /// "Cache corruption degrades to a re-embed, never an error" — lvo.3's criterion.
    #[test]
    fn corruption_degrades_to_a_re_embed() {
        let p = store();
        {
            let c = cache(Counting::new(), &p);
            block(c.embed_batch(&["alpha"])).unwrap();
            c.save().unwrap();
        }
        let mut bytes = p.get(OBJECT).unwrap().expect("cache was written");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        p.put(OBJECT, &bytes).unwrap();

        let c = cache(Counting::new(), &p);
        assert!(c.is_empty(), "a corrupt object must load as empty");
        block(c.embed_batch(&["alpha"])).unwrap();
        assert_eq!(c.inner.calls(), 1, "and cost a re-embed, not an error");
    }

    /// An absent object is a cold cache, not a failure.
    #[test]
    fn a_missing_object_is_a_cold_cache() {
        let p = store();
        let c = cache(Counting::new(), &p);
        assert!(c.is_empty());
        block(c.embed_batch(&["alpha"])).unwrap();
        assert_eq!(c.inner.calls(), 1);
    }

    /// The cap is load-bearing: a first ingest embeds the whole corpus, so an uncapped
    /// cache holds it all.
    #[test]
    fn eviction_respects_the_cap_and_keeps_the_newest() {
        let p = store();
        let c = cache_with(Counting::new(), &p, "test/fake-v1", DIM, 2);
        for t in ["a", "b", "c"] {
            block(c.embed_batch(&[t])).unwrap();
        }
        assert_eq!(c.len(), 2, "capped");
        assert_eq!(c.stats().evicted, 1);
        c.save().unwrap();

        let reopened = cache_with(Counting::new(), &p, "test/fake-v1", DIM, 2);
        block(reopened.embed_batch(&["c"])).unwrap();
        assert_eq!(reopened.inner.calls(), 0, "the newest entry survived");
        block(reopened.embed_batch(&["a"])).unwrap();
        assert_eq!(reopened.inner.calls(), 1, "the oldest was evicted");
    }

    /// `max_entries == 0` is `--no-cache`. A cache that still deduped within the run would
    /// contradict the flag, so the bypass has to be total, not just unpersisted.
    #[test]
    fn a_zero_cap_is_a_total_bypass_not_just_an_unpersisted_one() {
        let p = store();
        let c = cache_with(Counting::new(), &p, "test/fake-v1", DIM, 0);
        block(c.embed_batch(&["alpha", "alpha", "alpha"])).unwrap();
        assert_eq!(
            c.inner.asked(),
            vec!["alpha", "alpha", "alpha"],
            "a repeat within one batch must still reach the provider"
        );
        block(c.embed_batch(&["alpha"])).unwrap();
        assert_eq!(c.inner.asked().len(), 4, "and across batches too");
        assert!(c.is_empty(), "nothing retained");
        assert_eq!(c.stats(), CacheStats::default(), "and nothing counted");
    }

    /// A query embedding is a different tensor for Voyage/Cohere/Gemini/Jina, so serving a
    /// cached document vector for a query would be silently wrong.
    #[test]
    fn embed_query_never_reads_or_writes_the_cache() {
        let p = store();
        let c = cache(Counting::new(), &p);
        block(c.embed_batch(&["alpha"])).unwrap();
        let before = c.len();
        block(c.embed_query("alpha")).unwrap();
        assert_eq!(c.inner.asked().last().unwrap(), "query:alpha");
        assert_eq!(c.len(), before, "a query must not populate the cache");
        assert_eq!(c.stats().hits, 0, "nor consume a hit");
    }

    /// An in-memory store has no backend to persist to; that is a pass-through, not an error.
    #[test]
    fn no_persistence_degrades_to_a_pass_through() {
        let c = CachedEmbedder::open(Counting::new(), None, "test/fake-v1", DIM, 100);
        block(c.embed_batch(&["alpha"])).unwrap();
        c.save().unwrap();
        assert_eq!(c.inner.calls(), 1);
        // Still caches in RAM within the one run.
        block(c.embed_batch(&["alpha"])).unwrap();
        assert_eq!(c.inner.calls(), 1);
    }
}
