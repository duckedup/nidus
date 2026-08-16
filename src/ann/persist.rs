//! On-disk codec for the ANN index — a **derived cache**, not source of truth.

use anyhow::Result;

use crate::ann::{Ann, AnnSnapshot, IvfIndex};
use crate::backend::Persistence;
use crate::index_cache;
use crate::model::{AnnConfig, AnnKind, Distance, QuantKind};

/// The object name of the ANN cache on the persistence backend.
const ANN_OBJECT: &str = "ann";

fn kind_to_byte(k: AnnKind) -> u8 {
    match k {
        AnnKind::Hnsw => 0,
        AnnKind::Ivf => 1,
    }
}

fn distance_to_byte(d: Distance) -> u8 {
    match d {
        Distance::Cosine => 0,
        Distance::Euclidean => 1,
        Distance::DotProduct => 2,
    }
}

/// The quantization space the graph/lists were built in. Part of the validity key: a
/// graph built with int8 codes is navigated in int8 space, so a cache built under a
/// different quantization config must be discarded and rebuilt (nidus-ndu).
fn quant_to_byte(quant: Option<QuantKind>) -> u8 {
    match quant {
        None => 0,
        Some(QuantKind::Int8) => 1,
        Some(QuantKind::Binary) => 2,
    }
}

/// The validity key for the shared cache codec: valid only for this exact `(kind, distance, quant,
/// dim, m, ef_construction, n_lists, seed)`, and any mismatch means rebuild. `ef_search`/`n_probe`/
/// `overscan` are query-time tunables that do not change the built structure, so they are excluded.
fn validity_key(
    dim: usize,
    distance: Distance,
    cfg: &AnnConfig,
    quant: Option<QuantKind>,
) -> Vec<u8> {
    let mut k = Vec::with_capacity(3 + 4 * 4 + 8);
    k.push(kind_to_byte(cfg.kind));
    k.push(distance_to_byte(distance));
    k.push(quant_to_byte(quant));
    k.extend_from_slice(&(dim as u32).to_le_bytes());
    k.extend_from_slice(&(cfg.m as u32).to_le_bytes());
    k.extend_from_slice(&(cfg.ef_construction as u32).to_le_bytes());
    k.extend_from_slice(&(cfg.n_lists as u32).to_le_bytes());
    k.extend_from_slice(&cfg.seed.to_le_bytes());
    k
}

/// Save the index to the backend `p` atomically. `covered_rows` is the live row count
/// the index reflects (so a later `open` knows how many rows to incrementally catch up).
#[allow(clippy::too_many_arguments)]
pub(crate) fn save(
    p: &dyn Persistence,
    ann: &Ann,
    covered_rows: u64,
    dim: usize,
    distance: Distance,
    cfg: &AnnConfig,
    quant: Option<QuantKind>,
) -> Result<()> {
    let key = validity_key(dim, distance, cfg, quant);
    index_cache::save(p, ANN_OBJECT, &key, covered_rows, &ann.snapshot_ref())
}

/// Load the index from `p` if present and valid for the current `(dim, distance, cfg, quant)`.
/// `Ok(None)` — never an error — when absent, stale, or corrupt, and the caller rebuilds. On success
/// returns the index and the row count it covers, so the caller can catch up any rows added since.
pub(crate) fn load(
    p: &dyn Persistence,
    dim: usize,
    distance: Distance,
    cfg: &AnnConfig,
    quant: Option<QuantKind>,
) -> Result<Option<(Ann, u64)>> {
    let key = validity_key(dim, distance, cfg, quant);
    Ok(index_cache::load::<AnnSnapshot>(p, ANN_OBJECT, &key)?
        .map(|(snap, covered)| (Ann::from_snapshot(*cfg, dim, distance, snap), covered)))
}

// ── Per-segment IVF sidecars (SPEC §14.3) ────────────────────────────────────────

/// The sidecar object name for one segment's IVF index: `seg-00000001` -> `seg-00000001.ivf`.
/// Sits beside `checksum`'s `<segment>.crc` on the same backend.
pub(crate) fn segment_object_name(segment: &str) -> String {
    format!("{segment}.ivf")
}

/// Which segment a sidecar belongs to: its object name and its **global row range**. Bundled
/// rather than passed positionally because `base` and `rows` are both `u64` — a swap would
/// compile and silently key the cache to the wrong rows.
#[derive(Clone, Copy)]
pub(crate) struct SegmentSlot<'a> {
    pub(crate) name: &'a str,
    pub(crate) base: u64,
    pub(crate) rows: u64,
}

/// Binds a sidecar to one segment's identity **and its global row range**: IVF lists hold
/// global physical rows, so a cache adopted at a different `(base, rows)` would point at the
/// wrong vectors. `n_probe` is excluded for the same reason as [`validity_key`].
pub(crate) fn segment_validity_key(
    slot: SegmentSlot<'_>,
    dim: usize,
    distance: Distance,
    cfg: &AnnConfig,
) -> Vec<u8> {
    let mut k = Vec::with_capacity(1 + 4 + 8 + 8 + 8 + slot.name.len());
    k.push(distance_to_byte(distance));
    k.extend_from_slice(&(dim as u32).to_le_bytes());
    k.extend_from_slice(&(cfg.n_lists as u32).to_le_bytes());
    k.extend_from_slice(&cfg.seed.to_le_bytes());
    k.extend_from_slice(&slot.base.to_le_bytes());
    k.extend_from_slice(&slot.rows.to_le_bytes());
    k.extend_from_slice(slot.name.as_bytes());
    k
}

/// Save one sealed segment's IVF index as its `<segment>.ivf` sidecar. Whole-object `put`,
/// so it is atomic; the watermark is the segment's own row count.
pub(crate) fn save_segment(
    p: &dyn Persistence,
    slot: SegmentSlot<'_>,
    dim: usize,
    distance: Distance,
    cfg: &AnnConfig,
    ix: &IvfIndex,
) -> Result<()> {
    let key = segment_validity_key(slot, dim, distance, cfg);
    index_cache::save(
        p,
        &segment_object_name(slot.name),
        &key,
        slot.rows,
        &ix.snapshot_ref(),
    )
}

/// Load one segment's IVF sidecar when valid for exactly this `(slot, dim, distance, cfg)`.
/// `Ok(None)` — never an error — when absent, stale, or corrupt, and the caller rebuilds.
/// A non-IVF payload is treated as no cache.
pub(crate) fn load_segment(
    p: &dyn Persistence,
    slot: SegmentSlot<'_>,
    dim: usize,
    distance: Distance,
    cfg: &AnnConfig,
) -> Result<Option<IvfIndex>> {
    let key = segment_validity_key(slot, dim, distance, cfg);
    let loaded =
        index_cache::load::<AnnSnapshot>(p, &segment_object_name(slot.name), &key)?.map(|(s, _)| s);
    Ok(match loaded {
        Some(AnnSnapshot::Ivf { centroids, lists }) => {
            Some(IvfIndex::from_parts(*cfg, dim, distance, centroids, lists))
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::Walk;
    use crate::data::Segments;

    fn seg(dim: usize, rows: &[Vec<f32>]) -> Segments {
        let mut d = Segments::in_memory_with(dim, Distance::Cosine);
        for r in rows {
            d.append(r).unwrap();
        }
        d
    }

    /// In-memory round-trip through the shared codec, no filesystem — Miri-clean.
    /// Exercises the same validity key + framing `save`/`load` use.
    fn roundtrip_bytes(
        ann: &Ann,
        dim: usize,
        distance: Distance,
        cfg: &AnnConfig,
        covered: u64,
    ) -> Vec<u8> {
        let key = validity_key(dim, distance, cfg, None);
        index_cache::frame(&key, covered, &ann.snapshot_ref()).unwrap()
    }

    fn decode_bytes(
        bytes: &[u8],
        dim: usize,
        distance: Distance,
        cfg: &AnnConfig,
    ) -> Option<(Ann, u64)> {
        let key = validity_key(dim, distance, cfg, None);
        index_cache::decode::<AnnSnapshot>(bytes, &key)
            .map(|(snap, covered)| (Ann::from_snapshot(*cfg, dim, distance, snap), covered))
    }

    #[test]
    fn hnsw_snapshot_roundtrips_and_searches_the_same() {
        let data = seg(
            3,
            &[
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        );
        let cfg = AnnConfig::hnsw();
        let mut ann = Ann::empty(cfg, 3, Distance::Cosine);
        ann.build(&Walk::exact(&data, Distance::Cosine), &[0, 1, 2], 1);
        let before = ann.search(&Walk::exact(&data, Distance::Cosine), &[0.0, 1.0, 0.0], 3);

        let bytes = roundtrip_bytes(&ann, 3, Distance::Cosine, &cfg, 3);
        let (restored, covered) = decode_bytes(&bytes, 3, Distance::Cosine, &cfg).unwrap();
        assert_eq!(covered, 3);
        let after = restored.search(&Walk::exact(&data, Distance::Cosine), &[0.0, 1.0, 0.0], 3);
        assert_eq!(before, after, "restored graph must search identically");
    }

    #[test]
    fn ivf_snapshot_roundtrips() {
        let rows: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let t = i as f32 / 20.0;
                vec![t.cos(), t.sin()]
            })
            .collect();
        let data = seg(2, &rows);
        let cfg = AnnConfig::ivf().n_lists(4);
        let mut ann = Ann::empty(cfg, 2, Distance::Cosine);
        ann.build(
            &Walk::exact(&data, Distance::Cosine),
            &(0..20).collect::<Vec<_>>(),
            1,
        );
        let before = ann.search(&Walk::exact(&data, Distance::Cosine), &rows[5], 5);

        let bytes = roundtrip_bytes(&ann, 2, Distance::Cosine, &cfg, 20);
        let (restored, _) = decode_bytes(&bytes, 2, Distance::Cosine, &cfg).unwrap();
        let after = restored.search(&Walk::exact(&data, Distance::Cosine), &rows[5], 5);
        assert_eq!(before, after);
    }

    #[test]
    fn config_mismatch_is_rejected() {
        let data = seg(2, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let cfg = AnnConfig::hnsw().m(16);
        let mut ann = Ann::empty(cfg, 2, Distance::Cosine);
        ann.build(&Walk::exact(&data, Distance::Cosine), &[0, 1], 1);
        let bytes = roundtrip_bytes(&ann, 2, Distance::Cosine, &cfg, 2);

        // Different m → cache invalid → None.
        let other = AnnConfig::hnsw().m(32);
        assert!(decode_bytes(&bytes, 2, Distance::Cosine, &other).is_none());
        // Different metric → None.
        assert!(decode_bytes(&bytes, 2, Distance::Euclidean, &cfg).is_none());
        // Different dim → None.
        assert!(decode_bytes(&bytes, 4, Distance::Cosine, &cfg).is_none());
    }

    // ── Per-segment sidecars (nidus-143) ─────────────────────────────────────

    /// A segment sidecar round-trip through `frame`/`decode` only — no filesystem, so it
    /// runs under Miri, exactly like the whole-index tests above.
    fn seg_roundtrip(
        ix: &IvfIndex,
        segment: &str,
        base: u64,
        rows: u64,
        dim: usize,
        cfg: &AnnConfig,
    ) -> Vec<u8> {
        let slot = SegmentSlot {
            name: segment,
            base,
            rows,
        };
        let key = segment_validity_key(slot, dim, Distance::Cosine, cfg);
        index_cache::frame(&key, rows, &ix.snapshot_ref()).unwrap()
    }

    fn seg_decode(
        bytes: &[u8],
        segment: &str,
        base: u64,
        rows: u64,
        dim: usize,
        cfg: &AnnConfig,
    ) -> Option<IvfIndex> {
        let slot = SegmentSlot {
            name: segment,
            base,
            rows,
        };
        let key = segment_validity_key(slot, dim, Distance::Cosine, cfg);
        match index_cache::decode::<AnnSnapshot>(bytes, &key) {
            Some((AnnSnapshot::Ivf { centroids, lists }, _)) => Some(IvfIndex::from_parts(
                *cfg,
                dim,
                Distance::Cosine,
                centroids,
                lists,
            )),
            _ => None,
        }
    }

    fn built_segment_index(data: &Segments, rows: u64, cfg: AnnConfig) -> IvfIndex {
        let mut ix = IvfIndex::new(cfg, data.dimension(), Distance::Cosine);
        ix.build(
            &Walk::exact(data, Distance::Cosine),
            &(0..rows).collect::<Vec<_>>(),
            1,
        );
        ix
    }

    #[test]
    fn segment_sidecar_roundtrips_and_searches_the_same() {
        let rows: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let t = i as f32 / 20.0;
                vec![t.cos(), t.sin()]
            })
            .collect();
        let data = seg(2, &rows);
        let cfg = AnnConfig::ivf().n_lists(4);
        let ix = built_segment_index(&data, 20, cfg);
        let walk = Walk::exact(&data, Distance::Cosine);
        let before = ix.search(&walk, &rows[5], 5);

        let bytes = seg_roundtrip(&ix, "seg-00000001", 0, 20, 2, &cfg);
        let restored = seg_decode(&bytes, "seg-00000001", 0, 20, 2, &cfg).unwrap();
        assert_eq!(before, restored.search(&walk, &rows[5], 5));
    }

    #[test]
    fn segment_sidecar_is_bound_to_its_segment_and_row_range() {
        let data = seg(2, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let cfg = AnnConfig::ivf().n_lists(2);
        let ix = built_segment_index(&data, 2, cfg);
        let bytes = seg_roundtrip(&ix, "seg-00000001", 8, 2, 2, &cfg);

        assert!(seg_decode(&bytes, "seg-00000001", 8, 2, 2, &cfg).is_some());
        // A different segment name, base, or row count must all reject: IVF lists hold
        // *global* rows, so adopting at the wrong range would point at other vectors.
        assert!(seg_decode(&bytes, "seg-00000002", 8, 2, 2, &cfg).is_none());
        assert!(seg_decode(&bytes, "seg-00000001", 0, 2, 2, &cfg).is_none());
        assert!(seg_decode(&bytes, "seg-00000001", 8, 3, 2, &cfg).is_none());
        assert!(seg_decode(&bytes, "seg-00000001", 8, 2, 4, &cfg).is_none());
        // Different IVF tuning → different lists → reject.
        let other = AnnConfig::ivf().n_lists(3);
        assert!(seg_decode(&bytes, "seg-00000001", 8, 2, 2, &other).is_none());
    }

    #[test]
    fn segment_sidecar_object_name_sits_beside_the_checksum() {
        assert_eq!(segment_object_name("data"), "data.ivf");
        assert_eq!(segment_object_name("seg-00000007"), "seg-00000007.ivf");
    }

    #[test]
    fn corrupt_segment_sidecar_is_rejected() {
        let data = seg(2, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let cfg = AnnConfig::ivf().n_lists(2);
        let ix = built_segment_index(&data, 2, cfg);
        let mut bytes = seg_roundtrip(&ix, "data", 0, 2, 2, &cfg);
        let last = bytes.len() - 5; // inside the payload, before the trailing crc32
        bytes[last] ^= 0xFF;
        assert!(seg_decode(&bytes, "data", 0, 2, 2, &cfg).is_none());
    }

    #[test]
    fn corrupt_crc_is_rejected() {
        let data = seg(2, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let cfg = AnnConfig::hnsw();
        let mut ann = Ann::empty(cfg, 2, Distance::Cosine);
        ann.build(&Walk::exact(&data, Distance::Cosine), &[0, 1], 1);
        let mut bytes = roundtrip_bytes(&ann, 2, Distance::Cosine, &cfg, 2);
        // Flip a payload byte; CRC must catch it.
        let mid = bytes.len() - 1;
        bytes[mid] ^= 0xFF;
        assert!(decode_bytes(&bytes, 2, Distance::Cosine, &cfg).is_none());
    }
}
