//! Per-segment checksum sidecar (`<segment>.crc`): a hard integrity claim over a segment's row
//! bytes, framed via the shared `index_cache` codec but never treated as rebuildable (#160).
//! Unlike the `ann`/`fts` caches, a mismatch here is a real finding: recomputing the crc over
//! corrupted bytes would launder the corruption into a fresh, valid-looking checksum.

use anyhow::{Context, Result, anyhow, bail};

use super::{Backing, DataSegment, HEADER_LEN, distance_to_byte};
use crate::backend::Persistence;
use crate::index_cache;
use crate::model::Distance;

/// The sidecar object name for a segment: `data` -> `data.crc`.
pub(crate) fn object_name(segment: &str) -> String {
    format!("{segment}.crc")
}

/// Binds a sidecar to exactly this segment name + dimension + distance (mirrors
/// `ann::persist::validity_key`), so it can't be adopted by the wrong segment or a
/// store reopened at a different pinned dimension.
pub(crate) fn key(segment: &str, dim: usize, distance: Distance) -> Vec<u8> {
    let mut k = Vec::with_capacity(5 + segment.len());
    k.extend_from_slice(&(dim as u32).to_le_bytes());
    k.push(distance_to_byte(distance));
    k.extend_from_slice(segment.as_bytes());
    k
}

/// crc32 over `rows` row-bytes starting at [`HEADER_LEN`]. Streams through `read_exact_at` in
/// bounded chunks for an appender-backed segment; a memory-mapped one is already RAM-resident
/// so its bytes are hashed directly — neither path materializes a fresh whole-segment buffer.
pub(crate) fn compute(seg: &mut DataSegment, rows: u64) -> Result<u32> {
    let stride = seg.dimension * 4;
    let total = rows as usize * stride;
    let mut hasher = crc32fast::Hasher::new();

    if let Backing::Mmap {
        map,
        rows: mapped_rows,
    } = &seg.backing
    {
        if rows > *mapped_rows {
            bail!(
                "checksum requested for {rows} rows but the mapped segment only holds {mapped_rows}"
            );
        }
        hasher.update(&map.bytes()[HEADER_LEN..HEADER_LEN + total]);
        return Ok(hasher.finalize());
    }

    let ap = seg
        .appender
        .as_mut()
        .ok_or_else(|| anyhow!("segment has no backing store to checksum"))?;
    let mut offset = HEADER_LEN as u64;
    let mut remaining = total;
    let mut buf = [0u8; 8192];
    while remaining > 0 {
        let take = remaining.min(buf.len());
        ap.read_exact_at(offset, &mut buf[..take])
            .context("failed to read segment rows for checksum")?;
        hasher.update(&buf[..take]);
        offset += take as u64;
        remaining -= take;
    }
    Ok(hasher.finalize())
}

/// Write the sidecar for `rows` rows. A whole-object [`Persistence::put`], so it is atomic.
pub(crate) fn save(
    p: &dyn Persistence,
    segment: &str,
    key: &[u8],
    rows: u64,
    crc: u32,
) -> Result<()> {
    index_cache::save(p, &object_name(segment), key, rows, &crc)
}

/// The stored `(crc, rows_covered)`, or `None` when absent, keyed for a different
/// segment/dimension/distance, or corrupt framing. `None` means "no usable claim" — it
/// must never be treated as license to silently recompute and re-save (see module docs).
pub(crate) fn load(p: &dyn Persistence, segment: &str, key: &[u8]) -> Result<Option<(u32, u64)>> {
    index_cache::load::<u32>(p, &object_name(segment), key)
}

/// What verifying one segment's sidecar found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentIntegrity {
    /// The sidecar's claim checked out. `rows_covered` can be less than `rows_total` when
    /// rows were appended after the sidecar was stamped — that tail is UNVERIFIED, not clean.
    Ok { rows_covered: u64, rows_total: u64 },
    /// The stored crc disagrees with the row bytes it claims to cover. Real corruption or a
    /// torn write — this is never rebuilt or re-saved automatically.
    Mismatch {
        rows_covered: u64,
        expected: u32,
        actual: u32,
    },
    /// No usable sidecar (absent, wrong key, or corrupt framing) — unverified, not
    /// vouched-for-clean.
    NoChecksum { rows_total: u64 },
}

/// Verify `segment`'s sidecar against its current bytes. A stored claim that outruns the
/// segment's own row count is a hard error, never silently clamped.
pub(crate) fn verify(
    seg: &mut DataSegment,
    p: &dyn Persistence,
    segment: &str,
    key: &[u8],
) -> Result<SegmentIntegrity> {
    let rows_total = seg.row_count();
    let Some((expected, rows_covered)) = load(p, segment, key)? else {
        return Ok(SegmentIntegrity::NoChecksum { rows_total });
    };
    if rows_covered > rows_total {
        bail!(
            "segment {segment} checksum sidecar claims {rows_covered} rows but only \
             {rows_total} are present"
        );
    }
    let actual = compute(seg, rows_covered)?;
    if actual == expected {
        Ok(SegmentIntegrity::Ok {
            rows_covered,
            rows_total,
        })
    } else {
        Ok(SegmentIntegrity::Mismatch {
            rows_covered,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;
    use crate::backend::MemAppender;

    /// A minimal whole-object [`Persistence`] over a `Mutex<HashMap>` — enough to exercise
    /// `save`/`load`/`verify` without touching the filesystem, so these tests run under Miri.
    #[derive(Default)]
    struct MemStore(Mutex<HashMap<String, Vec<u8>>>);

    impl Persistence for MemStore {
        fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
            Ok(self.0.lock().unwrap().get(key).cloned())
        }
        fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert(key.to_string(), bytes.to_vec());
            Ok(())
        }
        fn delete(&self, key: &str) -> Result<()> {
            self.0.lock().unwrap().remove(key);
            Ok(())
        }
        fn list(&self) -> Result<Vec<String>> {
            Ok(self.0.lock().unwrap().keys().cloned().collect())
        }
        // Writer exclusion is irrelevant to a checksum: these tests drive save/load/verify
        // directly, never through the lock-taking open path.
        fn try_lock(
            &self,
            _key: &str,
            _ttl: std::time::Duration,
        ) -> Result<Option<Box<dyn crate::backend::BackendLock>>> {
            Ok(None)
        }
    }

    /// A RAM-backed, appender-backed segment (via [`MemAppender`]) — real `read_exact_at`
    /// plumbing, no filesystem, so `compute`'s streaming path is genuinely exercised.
    fn seg_with_rows(dim: usize, distance: Distance, rows: &[Vec<f32>]) -> DataSegment {
        let mut d = DataSegment::open_with(Box::new(MemAppender::new()), dim, distance).unwrap();
        for r in rows {
            d.append(r).unwrap();
        }
        d
    }

    #[test]
    fn object_name_appends_crc_suffix() {
        assert_eq!(object_name("data"), "data.crc");
        assert_eq!(object_name("seg-00000001"), "seg-00000001.crc");
    }

    #[test]
    fn round_trip_save_then_verify_clean() {
        let mut seg = seg_with_rows(
            2,
            Distance::Cosine,
            &[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
        );
        let p = MemStore::default();
        let k = key("data", 2, Distance::Cosine);
        let rows = seg.row_count();
        let crc = compute(&mut seg, rows).unwrap();
        save(&p, "data", &k, rows, crc).unwrap();

        match verify(&mut seg, &p, "data", &k).unwrap() {
            SegmentIntegrity::Ok {
                rows_covered,
                rows_total,
            } => {
                assert_eq!(rows_covered, rows);
                assert_eq!(rows_total, rows);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    // THE LOAD-BEARING TEST: a byte flipped inside the row region must be a hard
    // Mismatch, never silently treated as an absent/rebuildable checksum.
    #[test]
    fn corrupted_row_byte_is_a_hard_mismatch() {
        let mut seg = seg_with_rows(
            3,
            Distance::Cosine,
            &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        );
        let p = MemStore::default();
        let k = key("data", 3, Distance::Cosine);
        let rows = seg.row_count();
        let crc = compute(&mut seg, rows).unwrap();
        save(&p, "data", &k, rows, crc).unwrap();

        // Flip a byte well inside the row region (past the 64-byte header).
        let mut buf = Vec::new();
        seg.appender
            .as_mut()
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        buf[HEADER_LEN + 1] ^= 0xFF;
        seg.appender.as_mut().unwrap().rewrite(&buf).unwrap();

        match verify(&mut seg, &p, "data", &k).unwrap() {
            SegmentIntegrity::Mismatch { rows_covered, .. } => assert_eq!(rows_covered, rows),
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    #[test]
    fn tail_appended_after_stamping_is_reported_uncovered() {
        let mut seg = seg_with_rows(2, Distance::Cosine, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let p = MemStore::default();
        let k = key("data", 2, Distance::Cosine);
        let n = seg.row_count();
        let crc = compute(&mut seg, n).unwrap();
        save(&p, "data", &k, n, crc).unwrap();

        seg.append(&[2.0, 2.0]).unwrap();
        let m = seg.row_count();
        assert!(m > n);

        match verify(&mut seg, &p, "data", &k).unwrap() {
            SegmentIntegrity::Ok {
                rows_covered,
                rows_total,
            } => {
                assert_eq!(rows_covered, n);
                assert_eq!(rows_total, m);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn sidecar_keyed_for_a_different_segment_or_dimension_is_not_adopted() {
        let mut seg = seg_with_rows(2, Distance::Cosine, &[vec![1.0, 0.0]]);
        let p = MemStore::default();
        let own_key = key("data", 2, Distance::Cosine);
        let rows = seg.row_count();
        let crc = compute(&mut seg, rows).unwrap();
        save(&p, "data", &own_key, rows, crc).unwrap();

        // Wrong segment name.
        let wrong_name = key("seg-00000001", 2, Distance::Cosine);
        assert!(load(&p, "data", &wrong_name).unwrap().is_none());

        // Wrong dimension.
        let wrong_dim = key("data", 4, Distance::Cosine);
        assert!(load(&p, "data", &wrong_dim).unwrap().is_none());

        // A mismatched key must read as "unverified", never a false Mismatch.
        match verify(&mut seg, &p, "data", &wrong_dim).unwrap() {
            SegmentIntegrity::NoChecksum { rows_total } => assert_eq!(rows_total, rows),
            other => panic!("expected NoChecksum, got {other:?}"),
        }
    }

    #[test]
    fn stored_coverage_beyond_current_rows_is_an_error() {
        // A sidecar claiming more rows than the segment holds is a structurally impossible
        // claim (you cannot honestly report having verified rows that don't exist).
        let mut seg = seg_with_rows(2, Distance::Cosine, &[vec![1.0, 0.0]]);
        let p = MemStore::default();
        let k = key("data", 2, Distance::Cosine);
        save(&p, "data", &k, 99, 0).unwrap();
        assert!(verify(&mut seg, &p, "data", &k).is_err());
    }
}
