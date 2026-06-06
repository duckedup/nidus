//! The `data` file: an append-only, fixed-stride, row-major `f32` matrix loaded
//! into RAM. Contract: see the root `SPEC.md` §5.1, §5.3, §6 (incl. §6.6).

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

use crate::model::Distance;

// ── Header constants ──────────────────────────────────────────────────────────

/// Magic bytes: "NIDUS\0"
const MAGIC: &[u8; 6] = b"NIDUS\0";
/// Format version stored as little-endian u16 after the magic.
const VERSION: u16 = 1;
/// Total header size in bytes (cache-line aligned).
pub(crate) const HEADER_LEN: usize = 64;

/// The vector segment. Holds every row in memory; appends go to the tail of the
/// backing file and are mirrored in `vectors`. Implementers may add fields (a file
/// handle, etc.) but must keep the method signatures below.
pub struct DataSegment {
    dimension: usize,
    distance: Distance,
    vectors: Vec<f32>,
    /// `None` when the segment is in-memory only (no backing file).
    file: Option<FileState>,
    /// Test-only fault seam: when `Some(n)`, the next `n` appends succeed and the
    /// one after fails — lets tests exercise mid-batch rollback deterministically
    /// (real ENOSPC is not reproducible in a unit test).
    #[cfg(test)]
    fail_after: Option<usize>,
}

struct FileState {
    /// Path to the data file (needed for `rewrite`).
    path: PathBuf,
    /// Open file handle (append position maintained via seek on rewrite).
    handle: File,
}

// ── Header encode / decode ────────────────────────────────────────────────────

/// Encode the 64-byte header into a fixed-size array.
fn encode_header(dimension: usize, distance: Distance) -> [u8; HEADER_LEN] {
    let mut buf = [0u8; HEADER_LEN];
    // bytes 0..6: magic
    buf[..6].copy_from_slice(MAGIC);
    // bytes 6..8: version (little-endian u16)
    buf[6..8].copy_from_slice(&VERSION.to_le_bytes());
    // bytes 8..12: dimension (little-endian u32)
    let dim_u32 = dimension as u32;
    buf[8..12].copy_from_slice(&dim_u32.to_le_bytes());
    // byte 12: distance metric (0=Cosine, 1=Euclidean, 2=DotProduct)
    buf[12] = distance_to_byte(distance);
    // bytes 13..64: zero-padding (already zeroed)
    buf
}

fn distance_to_byte(d: Distance) -> u8 {
    match d {
        Distance::Cosine => 0,
        Distance::Euclidean => 1,
        Distance::DotProduct => 2,
    }
}

fn byte_to_distance(b: u8) -> Result<Distance> {
    match b {
        0 => Ok(Distance::Cosine),
        1 => Ok(Distance::Euclidean),
        2 => Ok(Distance::DotProduct),
        _ => bail!("unknown distance metric byte {b} in data file header"),
    }
}

/// Decode and verify the 64-byte header. Returns `(dimension, distance)`.
fn decode_header(buf: &[u8; HEADER_LEN]) -> Result<(usize, Distance)> {
    if &buf[..6] != MAGIC {
        bail!("data file has wrong magic bytes — not a nidus data file");
    }
    let version = u16::from_le_bytes([buf[6], buf[7]]);
    if version != VERSION {
        bail!(
            "data file version {} is not supported (expected {})",
            version,
            VERSION
        );
    }
    let dim = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let distance = byte_to_distance(buf[12])?;
    Ok((dim, distance))
}

// ── f32 vector I/O ────────────────────────────────────────────────────────────

/// Encode a slice of `f32` values into a `Vec<u8>` (little-endian).
fn floats_to_bytes(floats: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(floats.len() * 4);
    for &f in floats {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Decode `n` little-endian f32 values from `bytes`. Returns `Err` if the byte
/// length is not exactly `n * 4`. (Test-only: `open` streams + converts in place;
/// this whole-buffer form is retained for the codec round-trip tests.)
#[cfg(test)]
fn bytes_to_floats(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() != n * 4 {
        bail!(
            "expected {} bytes for {} floats, got {}",
            n * 4,
            n,
            bytes.len()
        );
    }
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

impl DataSegment {
    /// Open or create `path` (the `data` file). Verifies/writes the 64-byte header
    /// (magic + version + dimension + distance), then reads every fully-written row
    /// into RAM. Errors on magic mismatch, truncated header, or a dimension/distance
    /// that differs from the requested values.
    pub fn open(path: &Path, dimension: usize, distance: Distance) -> Result<DataSegment> {
        // Open or create the file with read+write access.
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("failed to open data file at {}", path.display()))?;

        let file_len = file
            .seek(SeekFrom::End(0))
            .context("failed to seek data file")?;
        file.seek(SeekFrom::Start(0))
            .context("failed to rewind data file")?;

        let vectors: Vec<f32>;

        if file_len == 0 {
            // New file — write the header.
            let header = encode_header(dimension, distance);
            file.write_all(&header)
                .context("failed to write data file header")?;
            vectors = Vec::new();
            // File position is now at byte 64 (end of header == append point).
        } else {
            // Existing file — read and verify the header.
            if file_len < HEADER_LEN as u64 {
                bail!(
                    "data file at {} is truncated: {} bytes (need at least {} for header)",
                    path.display(),
                    file_len,
                    HEADER_LEN
                );
            }
            let mut header_buf = [0u8; HEADER_LEN];
            file.read_exact(&mut header_buf)
                .context("failed to read data file header")?;
            let (stored_dim, stored_distance) = decode_header(&header_buf)
                .with_context(|| format!("invalid header in {}", path.display()))?;
            if stored_dim != dimension {
                bail!(
                    "data file dimension mismatch: file has dimension {}, requested {}",
                    stored_dim,
                    dimension
                );
            }
            if stored_distance != distance {
                bail!(
                    "data file distance metric mismatch: file has {stored_distance:?}, requested {distance:?}"
                );
            }

            // Calculate how many whole rows are present (ignore partial tail).
            let row_stride = dimension * 4; // bytes per row
            let data_bytes = file_len - HEADER_LEN as u64;
            let row_count = if row_stride == 0 {
                0u64
            } else {
                data_bytes / row_stride as u64
            };
            let whole_data_bytes = row_count * row_stride as u64;

            // Read exactly the whole rows. Stream into a single pre-reserved
            // `Vec<f32>`, converting LE bytes in bounded chunks — this both makes
            // the load fallible (try_reserve → Err instead of an allocator abort)
            // and avoids the old raw-bytes + decoded-floats double allocation (~2×
            // transient peak at open time).
            let total_floats = (row_count as usize) * dimension;
            vectors = if total_floats == 0 {
                Vec::new()
            } else {
                let mut v: Vec<f32> = Vec::new();
                v.try_reserve_exact(total_floats).map_err(|_| {
                    anyhow!(
                        "out of memory loading {} rows ({} bytes) from data file",
                        row_count,
                        whole_data_bytes
                    )
                })?;
                let mut remaining = whole_data_bytes as usize;
                let mut buf = [0u8; 8192]; // multiple of 4 (f32 width)
                while remaining > 0 {
                    let take = remaining.min(buf.len());
                    file.read_exact(&mut buf[..take])
                        .context("failed to read data file rows")?;
                    for chunk in buf[..take].chunks_exact(4) {
                        v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                    remaining -= take;
                }
                v
            };

            // Seek (and effectively truncate) to the end of the last whole row,
            // discarding any partial tail so future appends are aligned.
            let good_end = HEADER_LEN as u64 + whole_data_bytes;
            if file_len > good_end {
                // Truncate partial tail.
                file.set_len(good_end)
                    .context("failed to truncate partial tail from data file")?;
            }
            // Position the file cursor at the write end.
            file.seek(SeekFrom::End(0))
                .context("failed to seek to end of data file")?;
        }

        Ok(DataSegment {
            dimension,
            distance,
            vectors,
            file: Some(FileState {
                path: path.to_path_buf(),
                handle: file,
            }),
            #[cfg(test)]
            fail_after: None,
        })
    }

    /// An in-memory-only segment (no backing file, Cosine distance).
    #[cfg(test)]
    pub fn in_memory(dimension: usize) -> DataSegment {
        Self::in_memory_with(dimension, Distance::default())
    }

    /// An in-memory-only segment with a specific distance metric.
    pub fn in_memory_with(dimension: usize, distance: Distance) -> DataSegment {
        DataSegment {
            dimension,
            distance,
            vectors: Vec::new(),
            file: None,
            #[cfg(test)]
            fail_after: None,
        }
    }

    /// The pinned dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of rows currently stored.
    pub fn row_count(&self) -> u64 {
        (self.vectors.len() / self.dimension.max(1)) as u64
    }

    /// Borrow the entire flat f32 buffer (all rows, contiguous).
    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }

    /// Borrow row `i` as a `dimension`-length slice.
    pub fn row(&self, i: u64) -> &[f32] {
        let dim = self.dimension;
        let start = i as usize * dim;
        &self.vectors[start..start + dim]
    }

    /// Append one vector (length must equal `dimension`), returning its row index.
    /// Updates RAM + the file tail. Does NOT fsync — the caller batches then calls
    /// [`sync`](Self::sync).
    ///
    /// **Atomic per row.** If the file write fails partway (e.g. ENOSPC), the file
    /// is rolled back to the row boundary it started at, so a torn partial row
    /// never persists for the next append to write past — and RAM is left
    /// untouched. On success, RAM and the file advance together.
    pub fn append(&mut self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dimension {
            bail!(
                "vector length {} does not match segment dimension {}",
                vector.len(),
                self.dimension
            );
        }
        let row_index = self.row_count();

        // Test-only fault injection (see `fail_after`). Fires before any mutation,
        // so a failed append leaves RAM + file exactly as they were.
        #[cfg(test)]
        if let Some(n) = self.fail_after {
            if n == 0 {
                bail!("injected append failure (test fault seam) at row {row_index}");
            }
            self.fail_after = Some(n - 1);
        }

        // Reserve RAM first: an OOM here returns an error before the file is
        // touched, so the append stays atomic (and never aborts the process).
        self.vectors.try_reserve(self.dimension).map_err(|_| {
            anyhow!(
                "out of memory growing vector matrix by {} bytes",
                self.dimension * 4
            )
        })?;

        // Write to file first (if backed), then mirror into RAM.
        if let Some(ref mut fs) = self.file {
            let bytes = floats_to_bytes(vector);
            let start = fs
                .handle
                .stream_position()
                .context("failed to read data file position before append")?;
            if let Err(e) = fs.handle.write_all(&bytes) {
                // Roll back any partial bytes to the row boundary (best-effort —
                // the row never landed, so leave the file aligned for the next try).
                let _ = fs.handle.set_len(start);
                let _ = fs.handle.seek(SeekFrom::Start(start));
                return Err(anyhow::Error::new(e))
                    .with_context(|| format!("failed to append row {row_index} to data file"));
            }
        }

        // Infallible — capacity reserved above.
        self.vectors.extend_from_slice(vector);
        Ok(row_index)
    }

    /// Roll the segment back to exactly `rows` rows, discarding everything after
    /// (RAM + the file tail). The batch-rollback primitive: a writer captures
    /// `row_count()` before a batch and calls this to undo a failed one. `rows`
    /// must not exceed the current row count.
    pub fn truncate_to(&mut self, rows: u64) -> Result<()> {
        let keep_floats = rows as usize * self.dimension;
        if keep_floats > self.vectors.len() {
            bail!(
                "truncate_to({}) exceeds current row count {}",
                rows,
                self.row_count()
            );
        }
        if let Some(ref mut fs) = self.file {
            let good_end = HEADER_LEN as u64 + rows * (self.dimension as u64) * 4;
            fs.handle
                .set_len(good_end)
                .context("failed to truncate data file")?;
            fs.handle
                .seek(SeekFrom::Start(good_end))
                .context("failed to seek after data truncate")?;
        }
        self.vectors.truncate(keep_floats);
        Ok(())
    }

    /// Test-only: arm the fault seam so the `(n+1)`-th subsequent append fails.
    #[cfg(test)]
    pub fn fail_after(&mut self, n: usize) {
        self.fail_after = Some(n);
    }

    /// fsync the backing file (no-op for in-memory).
    pub fn sync(&mut self) -> Result<()> {
        if let Some(ref mut fs) = self.file {
            fs.handle.sync_all().context("failed to fsync data file")?;
        }
        Ok(())
    }

    /// Atomically rewrite the backing file to contain exactly `rows` (compaction),
    /// then swap in-RAM state. `rows.len()` must be a multiple of `dimension`.
    pub fn rewrite(&mut self, rows: &[f32]) -> Result<()> {
        let dim = self.dimension;
        if dim > 0 && !rows.len().is_multiple_of(dim) {
            bail!(
                "rows.len() ({}) is not a multiple of dimension ({})",
                rows.len(),
                dim
            );
        }

        match self.file {
            None => {
                // In-memory only: just swap the RAM buffer.
                self.vectors = try_clone_floats(rows)?;
                return Ok(());
            }
            Some(ref fs) => {
                let data_path = fs.path.clone();

                // Determine the sibling temp file path (same directory for atomic rename).
                let dir = data_path
                    .parent()
                    .context("data file path has no parent directory")?;
                let tmp_path = dir.join("data.tmp");

                // Write header + rows to the temp file.
                {
                    let mut tmp = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&tmp_path)
                        .with_context(|| {
                            format!("failed to create temp file at {}", tmp_path.display())
                        })?;

                    let header = encode_header(dim, self.distance);
                    tmp.write_all(&header)
                        .context("failed to write header to temp data file")?;

                    if !rows.is_empty() {
                        let bytes = floats_to_bytes(rows);
                        tmp.write_all(&bytes)
                            .context("failed to write rows to temp data file")?;
                    }

                    tmp.sync_all().context("failed to fsync temp data file")?;
                    // `tmp` is dropped (and closed) here.
                }

                // Atomic rename over the original data file.
                std::fs::rename(&tmp_path, &data_path).with_context(|| {
                    format!(
                        "failed to rename {} to {}",
                        tmp_path.display(),
                        data_path.display()
                    )
                })?;

                // Reopen the file for appending.
                let mut new_handle = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&data_path)
                    .with_context(|| {
                        format!(
                            "failed to reopen data file after rewrite at {}",
                            data_path.display()
                        )
                    })?;

                new_handle
                    .seek(SeekFrom::End(0))
                    .context("failed to seek to end of data file after rewrite")?;

                // Update the FileState handle.
                self.file = Some(FileState {
                    path: data_path,
                    handle: new_handle,
                });
            }
        }

        // Swap in-RAM buffer.
        self.vectors = try_clone_floats(rows)?;
        Ok(())
    }
}

/// Copy `rows` into a fresh `Vec<f32>`, reserving fallibly so an OOM surfaces as an
/// error instead of aborting the process (the `Vec::to_vec` replacement on the
/// compaction path).
fn try_clone_floats(rows: &[f32]) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    out.try_reserve_exact(rows.len())
        .map_err(|_| anyhow!("out of memory copying {} floats during rewrite", rows.len()))?;
    out.extend_from_slice(rows);
    Ok(out)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pure helpers (Miri-clean) ─────────────────────────────────────────

    #[test]
    fn header_encode_magic() {
        let h = encode_header(128, Distance::default());
        assert_eq!(&h[..6], b"NIDUS\0");
    }

    #[test]
    fn header_encode_version() {
        let h = encode_header(128, Distance::default());
        let v = u16::from_le_bytes([h[6], h[7]]);
        assert_eq!(v, VERSION);
    }

    #[test]
    fn header_encode_dimension() {
        let h = encode_header(384, Distance::default());
        let d = u32::from_le_bytes([h[8], h[9], h[10], h[11]]);
        assert_eq!(d, 384);
    }

    #[test]
    fn header_encode_zero_padding() {
        let h = encode_header(3, Distance::default());
        // Bytes 12..64 must all be zero.
        assert!(h[12..64].iter().all(|&b| b == 0));
    }

    #[test]
    fn header_length_is_64() {
        let h = encode_header(1, Distance::default());
        assert_eq!(h.len(), 64);
    }

    #[test]
    fn header_round_trip() {
        let h = encode_header(512, Distance::default());
        let (dim, dist) = decode_header(&h).unwrap();
        assert_eq!(dim, 512);
        assert_eq!(dist, Distance::Cosine);
    }

    #[test]
    fn header_bad_magic_errors() {
        let mut h = encode_header(3, Distance::default());
        h[0] = b'X';
        assert!(decode_header(&h).is_err());
    }

    #[test]
    fn header_bad_version_errors() {
        let mut h = encode_header(3, Distance::default());
        // Force version to 0.
        h[6] = 0;
        h[7] = 0;
        assert!(decode_header(&h).is_err());
    }

    #[test]
    fn floats_to_bytes_round_trip() {
        let floats = vec![1.0_f32, -0.5, 0.0, f32::INFINITY];
        let bytes = floats_to_bytes(&floats);
        assert_eq!(bytes.len(), floats.len() * 4);
        let back = bytes_to_floats(&bytes, floats.len()).unwrap();
        assert_eq!(back, floats);
    }

    #[test]
    fn floats_to_bytes_little_endian() {
        // 1.0_f32 in little-endian IEEE 754 is [0x00, 0x00, 0x80, 0x3F].
        let bytes = floats_to_bytes(&[1.0_f32]);
        assert_eq!(bytes, &[0x00, 0x00, 0x80, 0x3F]);
    }

    #[test]
    fn bytes_to_floats_wrong_length_errors() {
        let bytes = vec![0u8; 7]; // not a multiple of 4
        assert!(bytes_to_floats(&bytes, 2).is_err());
    }

    #[test]
    fn in_memory_row_count_starts_zero() {
        let seg = DataSegment::in_memory(4);
        assert_eq!(seg.row_count(), 0);
    }

    #[test]
    fn in_memory_dimension() {
        let seg = DataSegment::in_memory(128);
        assert_eq!(seg.dimension(), 128);
    }

    #[test]
    fn in_memory_append_and_row() {
        let mut seg = DataSegment::in_memory(3);
        let v = [1.0_f32, 2.0, 3.0];
        let idx = seg.append(&v).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(seg.row_count(), 1);
        assert_eq!(seg.row(0), &v);
    }

    #[test]
    fn in_memory_multiple_appends() {
        let mut seg = DataSegment::in_memory(2);
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert_eq!(seg.append(&a).unwrap(), 0);
        assert_eq!(seg.append(&b).unwrap(), 1);
        assert_eq!(seg.row_count(), 2);
        assert_eq!(seg.row(0), &a);
        assert_eq!(seg.row(1), &b);
    }

    #[test]
    fn in_memory_append_wrong_dimension_errors() {
        let mut seg = DataSegment::in_memory(3);
        assert!(seg.append(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn in_memory_sync_is_noop() {
        let mut seg = DataSegment::in_memory(4);
        seg.sync().unwrap(); // must not panic
    }

    #[test]
    fn in_memory_rewrite_swaps_vectors() {
        let mut seg = DataSegment::in_memory(2);
        seg.append(&[1.0, 2.0]).unwrap();
        seg.append(&[3.0, 4.0]).unwrap();
        let new_rows = vec![5.0_f32, 6.0];
        seg.rewrite(&new_rows).unwrap();
        assert_eq!(seg.row_count(), 1);
        assert_eq!(seg.row(0), &[5.0_f32, 6.0]);
    }

    #[test]
    fn in_memory_rewrite_non_multiple_errors() {
        let mut seg = DataSegment::in_memory(3);
        assert!(seg.rewrite(&[1.0_f32, 2.0]).is_err()); // 2 % 3 != 0
    }

    #[test]
    fn row_count_dimension_zero() {
        // dimension=0 is a degenerate edge case; row_count should not panic.
        let seg = DataSegment::in_memory(0);
        assert_eq!(seg.row_count(), 0);
    }

    #[test]
    fn in_memory_truncate_to_shrinks() {
        let mut seg = DataSegment::in_memory(2);
        seg.append(&[1.0, 2.0]).unwrap();
        seg.append(&[3.0, 4.0]).unwrap();
        seg.append(&[5.0, 6.0]).unwrap();
        seg.truncate_to(1).unwrap();
        assert_eq!(seg.row_count(), 1);
        assert_eq!(seg.row(0), &[1.0_f32, 2.0]);
    }

    #[test]
    fn in_memory_truncate_to_zero() {
        let mut seg = DataSegment::in_memory(2);
        seg.append(&[1.0, 2.0]).unwrap();
        seg.truncate_to(0).unwrap();
        assert_eq!(seg.row_count(), 0);
    }

    #[test]
    fn truncate_to_beyond_row_count_errors() {
        let mut seg = DataSegment::in_memory(2);
        seg.append(&[1.0, 2.0]).unwrap();
        assert!(seg.truncate_to(5).is_err());
    }

    #[test]
    fn append_is_atomic_under_fault() {
        // A failed append must leave RAM exactly as it was.
        let mut seg = DataSegment::in_memory(2);
        seg.append(&[1.0, 2.0]).unwrap();
        seg.fail_after(0); // the very next append fails
        assert!(seg.append(&[3.0, 4.0]).is_err());
        assert_eq!(seg.row_count(), 1, "failed append must not mutate RAM");
        assert_eq!(seg.row(0), &[1.0_f32, 2.0]);
    }

    // ── File-backed tests (ignored under Miri) ────────────────────────────

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_open_create_new() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let seg = DataSegment::open(&path, 4, Distance::default()).unwrap();
        assert_eq!(seg.dimension(), 4);
        assert_eq!(seg.row_count(), 0);
        assert!(path.exists());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_append_and_row() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let mut seg = DataSegment::open(&path, 3, Distance::default()).unwrap();
        let v = [1.0_f32, 2.0, 3.0];
        let idx = seg.append(&v).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(seg.row(0), &v);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_append_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");

        let rows = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        {
            let mut seg = DataSegment::open(&path, 3, Distance::default()).unwrap();
            for r in &rows {
                seg.append(r).unwrap();
            }
            seg.sync().unwrap();
        }

        // Reopen and verify all rows are present.
        let seg2 = DataSegment::open(&path, 3, Distance::default()).unwrap();
        assert_eq!(seg2.row_count(), 2);
        assert_eq!(seg2.row(0), &rows[0]);
        assert_eq!(seg2.row(1), &rows[1]);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_partial_tail_truncated_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");

        // Write one complete row then a partial one.
        {
            let mut seg = DataSegment::open(&path, 4, Distance::default()).unwrap();
            seg.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
            seg.sync().unwrap();
        }

        // Manually append a partial row (3 bytes, not 16) to simulate crash.
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&[0xFF, 0xFF, 0xFF]).unwrap();
        }

        // Reopening should silently ignore the partial tail.
        let seg2 = DataSegment::open(&path, 4, Distance::default()).unwrap();
        assert_eq!(seg2.row_count(), 1, "partial tail must be discarded");
        assert_eq!(seg2.row(0), &[1.0_f32, 2.0, 3.0, 4.0]);

        // Verify the file was physically truncated.
        let expected_len = HEADER_LEN as u64 + 4 * 4;
        let actual_len = std::fs::metadata(&path).unwrap().len();
        assert_eq!(actual_len, expected_len);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_dimension_mismatch_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        DataSegment::open(&path, 4, Distance::default()).unwrap();
        // Reopen with a different dimension must fail.
        let result = DataSegment::open(&path, 8, Distance::default());
        assert!(result.is_err(), "expected dimension-mismatch error");
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("dimension"),
            "error message should mention dimension: {msg}"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_rewrite_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");

        // Initial: write two rows.
        {
            let mut seg = DataSegment::open(&path, 2, Distance::default()).unwrap();
            seg.append(&[1.0, 2.0]).unwrap();
            seg.append(&[3.0, 4.0]).unwrap();
            seg.sync().unwrap();
            // Rewrite with only one row.
            seg.rewrite(&[9.0_f32, 8.0]).unwrap();
            assert_eq!(seg.row_count(), 1);
            assert_eq!(seg.row(0), &[9.0_f32, 8.0]);
        }

        // Reopen and verify the compacted state.
        let seg2 = DataSegment::open(&path, 2, Distance::default()).unwrap();
        assert_eq!(seg2.row_count(), 1);
        assert_eq!(seg2.row(0), &[9.0_f32, 8.0]);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_rewrite_then_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let mut seg = DataSegment::open(&path, 2, Distance::default()).unwrap();
        seg.append(&[1.0, 2.0]).unwrap();
        seg.rewrite(&[5.0_f32, 6.0]).unwrap();
        // Should be able to append after rewrite.
        let idx = seg.append(&[7.0, 8.0]).unwrap();
        assert_eq!(idx, 1);
        seg.sync().unwrap();

        let seg2 = DataSegment::open(&path, 2, Distance::default()).unwrap();
        assert_eq!(seg2.row_count(), 2);
        assert_eq!(seg2.row(0), &[5.0_f32, 6.0]);
        assert_eq!(seg2.row(1), &[7.0_f32, 8.0]);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_append_wrong_dimension_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let mut seg = DataSegment::open(&path, 3, Distance::default()).unwrap();
        assert!(seg.append(&[1.0, 2.0]).is_err());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_truncated_header_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        // Write only a partial header.
        std::fs::write(&path, b"NIDUS").unwrap();
        let result = DataSegment::open(&path, 3, Distance::default());
        assert!(result.is_err(), "expected truncated-header error");
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("truncated") || msg.contains("header"),
            "error should mention truncated/header: {msg}"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_bad_magic_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        // Write a full-length buffer with wrong magic.
        let mut buf = [0u8; HEADER_LEN];
        buf[..6].copy_from_slice(b"WRONG\0");
        buf[6..8].copy_from_slice(&VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&3u32.to_le_bytes());
        std::fs::write(&path, buf).unwrap();
        assert!(DataSegment::open(&path, 3, Distance::default()).is_err());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_exact_bytes_on_disk() {
        // Verify that the on-disk layout matches the spec exactly.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let mut seg = DataSegment::open(&path, 2, Distance::default()).unwrap();
        seg.append(&[1.0_f32, -1.0]).unwrap();
        seg.sync().unwrap();

        let raw = std::fs::read(&path).unwrap();
        // Header: 64 bytes
        assert_eq!(raw.len(), HEADER_LEN + 2 * 4);
        assert_eq!(&raw[..6], b"NIDUS\0");
        // Row 0 starts at byte 64
        let r0 = &raw[HEADER_LEN..HEADER_LEN + 8];
        assert_eq!(&r0[..4], &1.0_f32.to_le_bytes());
        assert_eq!(&r0[4..8], &(-1.0_f32).to_le_bytes());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_truncate_to_realigns_and_reopens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        {
            let mut seg = DataSegment::open(&path, 2, Distance::default()).unwrap();
            seg.append(&[1.0, 2.0]).unwrap();
            seg.append(&[3.0, 4.0]).unwrap();
            seg.append(&[5.0, 6.0]).unwrap();
            seg.truncate_to(1).unwrap();
            seg.sync().unwrap();
            assert_eq!(seg.row_count(), 1);
            // File physically shrank to header + one row (1 row × dim 2 × 4 bytes).
            let row_stride = 2 * 4;
            let expected = HEADER_LEN as u64 + row_stride;
            assert_eq!(std::fs::metadata(&path).unwrap().len(), expected);
            // Appending after truncate stays row-aligned.
            assert_eq!(seg.append(&[7.0, 8.0]).unwrap(), 1);
            seg.sync().unwrap();
        }
        let seg2 = DataSegment::open(&path, 2, Distance::default()).unwrap();
        assert_eq!(seg2.row_count(), 2);
        assert_eq!(seg2.row(0), &[1.0_f32, 2.0]);
        assert_eq!(seg2.row(1), &[7.0_f32, 8.0]);
    }
}
