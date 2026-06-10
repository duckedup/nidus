//! On-disk codec for the ANN index — a **derived cache**, not source of truth.
//!
//! The graph/lists are fully reconstructable from the `data` vectors, so this file is
//! an optimization: it lets `open()` load the index instead of rebuilding it. Every
//! load is therefore best-effort — a missing, stale (config/dim/metric/params
//! changed), or CRC-failed file simply returns `None` and the caller rebuilds. It is
//! never fatal.
//!
//! Layout mirrors the `data`/`log` files: a fixed 64-byte header (magic `NIDUS\0` +
//! version + the params the cache is only valid for), then a `bincode` payload
//! ([`AnnSnapshot`]), then a `crc32fast` checksum over the payload. Writes are atomic
//! (temp file + fsync + rename) so a crash can't leave a torn cache.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::ann::{Ann, AnnSnapshot};
use crate::model::{AnnConfig, AnnKind, Distance};

/// Magic bytes: "NIDUS\0" (shared convention with `data`/`log`).
const MAGIC: &[u8; 6] = b"NIDUS\0";
/// Format version of the `ann` cache file.
const VERSION: u16 = 1;
/// Fixed header size in bytes.
const HEADER_LEN: usize = 64;

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

/// Encode the header for a cache valid only for this exact `(dim, distance, cfg,
/// covered_rows)` tuple — any mismatch on load means "rebuild".
fn encode_header(
    dim: usize,
    distance: Distance,
    cfg: &AnnConfig,
    covered_rows: u64,
) -> [u8; HEADER_LEN] {
    let mut b = [0u8; HEADER_LEN];
    b[..6].copy_from_slice(MAGIC);
    b[6..8].copy_from_slice(&VERSION.to_le_bytes());
    b[8] = kind_to_byte(cfg.kind);
    b[9] = distance_to_byte(distance);
    // 10..12 pad
    b[12..16].copy_from_slice(&(dim as u32).to_le_bytes());
    b[16..24].copy_from_slice(&covered_rows.to_le_bytes());
    b[24..28].copy_from_slice(&(cfg.m as u32).to_le_bytes());
    b[28..32].copy_from_slice(&(cfg.ef_construction as u32).to_le_bytes());
    b[32..36].copy_from_slice(&(cfg.n_lists as u32).to_le_bytes());
    b[36..44].copy_from_slice(&cfg.seed.to_le_bytes());
    b
}

/// Save the index to `path` atomically. `covered_rows` is the live row count the
/// index reflects (so a later `open` knows how many rows to incrementally catch up).
pub(crate) fn save(
    path: &Path,
    ann: &Ann,
    covered_rows: u64,
    dim: usize,
    distance: Distance,
    cfg: &AnnConfig,
) -> Result<()> {
    let payload =
        bincode::serialize(&ann.snapshot_ref()).context("serialize ANN index snapshot")?;
    let crc = crc32fast::hash(&payload);
    let header = encode_header(dim, distance, cfg, covered_rows);

    // Atomic: write to a temp sibling, fsync, then rename over the target.
    let tmp = path.with_extension("ann.tmp");
    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)
            .with_context(|| format!("create ANN cache temp {tmp:?}"))?;
        f.write_all(&header)?;
        f.write_all(&payload)?;
        f.write_all(&crc.to_le_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("rename ANN cache into {path:?}"))?;
    Ok(())
}

/// Load the index from `path` if present and valid for the current `(dim, distance,
/// cfg)`. Returns `Ok(None)` — never an error — when the cache is absent, stale, or
/// corrupt; the caller rebuilds. On success returns the index and the row count it
/// covers (the caller incrementally catches up any rows added since).
pub(crate) fn load(
    path: &Path,
    dim: usize,
    distance: Distance,
    cfg: &AnnConfig,
) -> Result<Option<(Ann, u64)>> {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("open ANN cache {path:?}")),
    };
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)
        .with_context(|| format!("read ANN cache {path:?}"))?;

    // Too short to hold header + crc, or header doesn't match this store's config →
    // discard and rebuild. None of these are errors.
    if bytes.len() < HEADER_LEN + 4 {
        return Ok(None);
    }
    let expected = encode_header(dim, distance, cfg, 0);
    // Compare every header field except the covered_rows slot (16..24), which is data,
    // not a validity key.
    let header = &bytes[..HEADER_LEN];
    let matches = header[..16] == expected[..16] && header[24..44] == expected[24..44];
    if !matches {
        return Ok(None);
    }
    let covered_rows = u64::from_le_bytes(header[16..24].try_into().unwrap());

    let payload = &bytes[HEADER_LEN..bytes.len() - 4];
    let stored_crc = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
    if crc32fast::hash(payload) != stored_crc {
        return Ok(None);
    }
    let snap: AnnSnapshot = match bincode::deserialize(payload) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    Ok(Some((
        Ann::from_snapshot(*cfg, dim, distance, snap),
        covered_rows,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataSegment;

    fn seg(dim: usize, rows: &[Vec<f32>]) -> DataSegment {
        let mut d = DataSegment::in_memory(dim);
        for r in rows {
            d.append(r).unwrap();
        }
        d
    }

    /// In-memory round-trip of the header + payload + CRC framing, no filesystem —
    /// Miri-clean. Exercises the same encode/validate logic `save`/`load` use.
    fn roundtrip_bytes(
        ann: &Ann,
        dim: usize,
        distance: Distance,
        cfg: &AnnConfig,
        covered: u64,
    ) -> Vec<u8> {
        let payload = bincode::serialize(&ann.snapshot_ref()).unwrap();
        let crc = crc32fast::hash(&payload);
        let mut out = encode_header(dim, distance, cfg, covered).to_vec();
        out.extend_from_slice(&payload);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }

    fn decode_bytes(
        bytes: &[u8],
        dim: usize,
        distance: Distance,
        cfg: &AnnConfig,
    ) -> Option<(Ann, u64)> {
        if bytes.len() < HEADER_LEN + 4 {
            return None;
        }
        let expected = encode_header(dim, distance, cfg, 0);
        let header = &bytes[..HEADER_LEN];
        if header[..16] != expected[..16] || header[24..44] != expected[24..44] {
            return None;
        }
        let covered = u64::from_le_bytes(header[16..24].try_into().unwrap());
        let payload = &bytes[HEADER_LEN..bytes.len() - 4];
        let stored = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
        if crc32fast::hash(payload) != stored {
            return None;
        }
        let snap: AnnSnapshot = bincode::deserialize(payload).ok()?;
        Some((Ann::from_snapshot(*cfg, dim, distance, snap), covered))
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
        ann.build(&data, &[0, 1, 2]);
        let before = ann.search(&data, &[0.0, 1.0, 0.0], 3);

        let bytes = roundtrip_bytes(&ann, 3, Distance::Cosine, &cfg, 3);
        let (restored, covered) = decode_bytes(&bytes, 3, Distance::Cosine, &cfg).unwrap();
        assert_eq!(covered, 3);
        let after = restored.search(&data, &[0.0, 1.0, 0.0], 3);
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
        ann.build(&data, &(0..20).collect::<Vec<_>>());
        let before = ann.search(&data, &rows[5], 5);

        let bytes = roundtrip_bytes(&ann, 2, Distance::Cosine, &cfg, 20);
        let (restored, _) = decode_bytes(&bytes, 2, Distance::Cosine, &cfg).unwrap();
        let after = restored.search(&data, &rows[5], 5);
        assert_eq!(before, after);
    }

    #[test]
    fn config_mismatch_is_rejected() {
        let data = seg(2, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let cfg = AnnConfig::hnsw().m(16);
        let mut ann = Ann::empty(cfg, 2, Distance::Cosine);
        ann.build(&data, &[0, 1]);
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
        ann.build(&data, &[0, 1]);
        let mut bytes = roundtrip_bytes(&ann, 2, Distance::Cosine, &cfg, 2);
        // Flip a payload byte; CRC must catch it.
        let mid = HEADER_LEN + 1;
        bytes[mid] ^= 0xFF;
        assert!(decode_bytes(&bytes, 2, Distance::Cosine, &cfg).is_none());
    }
}
