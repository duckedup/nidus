//! On-disk codec for the ANN index — a **derived cache**, not source of truth.
//!
//! The graph/lists are fully reconstructable from the `data` vectors, so this file is
//! an optimization: it lets `open()` load the index instead of rebuilding it. The
//! framing (header + bincode + CRC + atomic write) lives in [`crate::index_cache`],
//! shared with the FTS cache; this module only composes ANN's **validity key** and
//! converts between the live [`Ann`] and its serializable snapshot. Every load is
//! best-effort — a missing, stale (config/dim/metric/params changed), or CRC-failed
//! file returns `None` and the caller rebuilds; it is never fatal.

use anyhow::Result;

use crate::ann::{Ann, AnnSnapshot};
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

/// The validity key for the shared cache codec: a cache is valid only for this exact
/// `(kind, distance, quant, dim, m, ef_construction, n_lists, seed)`. Any mismatch on
/// load means "rebuild". (`ef_search`, `n_probe`, `overscan` are query-time tunables
/// that don't change the built structure, so they are deliberately excluded.)
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

/// Load the index from `p` if present and valid for the current `(dim, distance, cfg,
/// quant)`. Returns `Ok(None)` — never an error — when the cache is absent, stale, or
/// corrupt; the caller rebuilds. On success returns the index and the row count it
/// covers (the caller incrementally catches up any rows added since).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::Walk;
    use crate::data::DataSegment;

    fn seg(dim: usize, rows: &[Vec<f32>]) -> DataSegment {
        let mut d = DataSegment::in_memory(dim);
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
