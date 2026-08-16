//! Persisting the filter index through the shared [`crate::index_cache`] codec. Derived
//! state, never the source of truth: a missing, stale, or corrupt cache rebuilds and is
//! never fatal.

use anyhow::Result;

use crate::backend::Persistence;
use crate::index_cache;

use super::Findex;

/// The cache object's name, alongside `ann` and `fts`.
const OBJECT: &str = "findex";

/// Save the index, valid only for its own schema and only at `watermark` (the committed
/// log offset it reflects).
pub(crate) fn save(p: &dyn Persistence, findex: &Findex, watermark: u64) -> Result<()> {
    index_cache::save(p, OBJECT, &findex.cache_key(), watermark, findex)
}

/// Load the index if the cache is valid for `key`. `Ok(None)` when absent, stale or
/// corrupt; the caller compares the returned watermark against the current log offset and
/// rebuilds unless they match.
pub(crate) fn load(p: &dyn Persistence, key: &[u8]) -> Result<Option<(Findex, u64)>> {
    index_cache::load::<Findex>(p, OBJECT, key)
}
