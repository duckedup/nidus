//! Memory-tier glue (SPEC §13.3): publish the in-RAM working set to a shared
//! [`MemoryTier`](crate::backend::MemoryTier), and adopt it on `open` instead of
//! replaying the log + rebuilding the index.
//!
//! The "working set" serialized here is the **replay-derived index** — the collections
//! (`id → (row, attrs)`), the dead-row count, and the declared FTS schemas. That is the
//! one piece of in-RAM state with no other cache: every worker would otherwise rebuild
//! it by replaying the whole op log. The vectors (`data`) are bulk-read regardless, and
//! the `ann`/`fts` postings keep their own derived caches and fast rebuilds, so they are
//! deliberately *not* duplicated into this blob.
//!
//! It reuses the [`crate::index_cache`] frame/decode codec (magic + version + watermark +
//! validity key + CRC) so there is one on-the-wire format for every derived cache. The
//! **watermark is the log byte offset** and the payload carries the data row count; a
//! snapshot is adopted only when *both* match the just-opened store exactly. The tier is
//! a rebuildable cache, so every error here is swallowed — a missing, stale, or
//! unreachable tier just falls back to the log replay, never failing `open`/`flush`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::Language;
use crate::backend::MemoryTier;
use crate::config::Config;
use crate::fts::Fts;

use super::{Collection, Store};

/// The object name the working-set snapshot is stored under in the memory tier. The
/// `RedisTier` additionally namespaces it with any configured `?prefix=`, so distinct
/// stores sharing one server don't collide.
const WORKING_SET_OBJECT: &str = "workingset";

/// One FTS collection's declared schema in the snapshot (`(field, language)` pairs).
type SchemaEntry<'a> = (&'a str, &'a [(String, Language)]);

/// The borrowing view serialized on publish — no clone of the index.
#[derive(Serialize)]
struct WorkingSetRef<'a> {
    data_rows: u64,
    dead_rows: u64,
    collections: &'a HashMap<String, Collection>,
    fts_schemas: Vec<SchemaEntry<'a>>,
}

/// The owned form decoded on adopt. Its field order/types must mirror [`WorkingSetRef`]
/// exactly (bincode is positional): `&str`↔`String` and `&[T]`↔`Vec<T>` share a layout.
#[derive(Deserialize)]
struct WorkingSet {
    data_rows: u64,
    dead_rows: u64,
    collections: HashMap<String, Collection>,
    fts_schemas: Vec<(String, Vec<(String, Language)>)>,
}

/// A working set adopted from the tier, ready to become the store's in-RAM index.
pub(super) struct AdoptedIndex(WorkingSet);

impl AdoptedIndex {
    /// Decompose into the pieces [`Store::open`] assembles: collections, dead-row count,
    /// and the FTS index with its schemas restored (postings are rebuilt afterwards by
    /// `load_or_build_fts`, exactly as on the replay path).
    pub(super) fn into_parts(self) -> (HashMap<String, Collection>, usize, Fts) {
        let ws = self.0;
        let mut fts = Fts::default();
        for (collection, fields) in &ws.fts_schemas {
            fts.set_schema(collection, fields);
        }
        (ws.collections, ws.dead_rows as usize, fts)
    }
}

/// The validity key for the working-set snapshot: any change to the embedding shape
/// invalidates a cached blob (a differently-shaped store must not adopt it).
pub(super) fn working_set_key(config: &Config) -> Vec<u8> {
    format!("ws-v1:{}:{:?}", config.dimension, config.distance).into_bytes()
}

/// Try to adopt the shared working set: `Ok(Some(index))` only when the tier holds a
/// snapshot whose validity key, log watermark, and data row count all match the
/// just-opened store. Any miss — no tier, absent/stale/evicted blob, or an unreachable
/// tier — is `Ok(None)`, and the caller replays the log. Never fatal.
pub(super) fn try_adopt(
    memory: Option<&dyn MemoryTier>,
    key: &[u8],
    data_rows: u64,
    watermark: u64,
) -> anyhow::Result<Option<AdoptedIndex>> {
    let Some(mem) = memory else {
        return Ok(None);
    };
    // Swallow tier errors: the working set is rebuildable from the log, so an unreachable
    // memory tier degrades to a local rebuild rather than failing the open.
    let Ok(Some(bytes)) = mem.load(WORKING_SET_OBJECT) else {
        return Ok(None);
    };
    match crate::index_cache::decode::<WorkingSet>(&bytes, key) {
        Some((ws, wm)) if wm == watermark && ws.data_rows == data_rows => {
            Ok(Some(AdoptedIndex(ws)))
        }
        _ => Ok(None),
    }
}

impl Store {
    /// Publish the current in-RAM working set to the shared memory tier so peers can
    /// adopt it on open (skipping their own log replay). Best-effort and a no-op when no
    /// external tier is configured: the tier is a rebuildable cache, so a serialization
    /// or transport failure is swallowed, never surfaced as a write/flush error.
    pub(super) fn publish_working_set(&self) {
        let Some(mem) = self.memory.as_deref() else {
            return;
        };
        let _ = self.publish_working_set_inner(mem);
    }

    /// The fallible body of [`publish_working_set`](Self::publish_working_set), kept
    /// separate so the public hook can `let _ =` it.
    fn publish_working_set_inner(&self, mem: &dyn MemoryTier) -> anyhow::Result<()> {
        let fts_schemas: Vec<SchemaEntry<'_>> = self
            .collections
            .keys()
            .filter_map(|name| {
                self.fts
                    .schema_for(name)
                    .map(|fields| (name.as_str(), fields))
            })
            .collect();
        let snapshot = WorkingSetRef {
            data_rows: self.data.row_count(),
            dead_rows: self.dead_rows as u64,
            collections: &self.collections,
            fts_schemas,
        };
        let key = working_set_key(&self.config);
        let watermark = self.log.offset()?;
        let buf = crate::index_cache::frame(&key, watermark, &snapshot)?;
        mem.store(WORKING_SET_OBJECT, &buf, None)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::backend::LocalRam;
    use crate::model::Value;

    /// One collection with one rowed doc, for a minimal working set.
    fn sample_collections() -> HashMap<String, Collection> {
        let mut docs = HashMap::new();
        let mut attrs = BTreeMap::new();
        attrs.insert("lang".to_string(), Value::Str("rust".to_string()));
        docs.insert(
            "doc1".to_string(),
            super::super::DocEntry {
                row: Some(0),
                attrs,
            },
        );
        let mut cols = HashMap::new();
        cols.insert(
            "col".to_string(),
            Collection {
                meta: BTreeMap::new(),
                docs,
            },
        );
        cols
    }

    /// Publish a working set to a tier under `key`/`watermark`/`data_rows`.
    fn publish(tier: &LocalRam, key: &[u8], data_rows: u64, watermark: u64) {
        let cols = sample_collections();
        let snapshot = WorkingSetRef {
            data_rows,
            dead_rows: 0,
            collections: &cols,
            fts_schemas: Vec::new(),
        };
        let buf = crate::index_cache::frame(key, watermark, &snapshot).unwrap();
        tier.store(WORKING_SET_OBJECT, &buf, None).unwrap();
    }

    #[test]
    fn adopts_only_a_matching_snapshot() {
        let tier = LocalRam::new();
        let key = b"ws-v1:3:Cosine";
        publish(&tier, key, 1, 100);

        // Exact match on key + watermark + data_rows → adopted.
        let adopted = try_adopt(Some(&tier), key, 1, 100).unwrap();
        let (cols, dead, _fts) = adopted.expect("matching snapshot adopts").into_parts();
        assert_eq!(dead, 0);
        assert!(cols["col"].docs.contains_key("doc1"));

        // Watermark mismatch (a write happened since) → rebuild.
        assert!(try_adopt(Some(&tier), key, 1, 101).unwrap().is_none());
        // Row-count mismatch → rebuild.
        assert!(try_adopt(Some(&tier), key, 2, 100).unwrap().is_none());
        // Validity-key mismatch (different dim/metric) → rebuild.
        assert!(
            try_adopt(Some(&tier), b"ws-v1:4:Cosine", 1, 100)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn no_tier_and_empty_tier_never_adopt() {
        assert!(try_adopt(None, b"k", 0, 0).unwrap().is_none());
        let empty = LocalRam::new();
        assert!(try_adopt(Some(&empty), b"k", 0, 0).unwrap().is_none());
    }
}
