//! The `log` file: an append-only, framed, checksummed op stream — the commit
//! record and crash-recovery mechanism. Contract: see `SPEC.md` in this directory.
//!
//! Each record is framed `[len: u32][payload: bincode(Op)][crc32: u32]`, all
//! little-endian; `crc32` (via `crc32fast`) covers the payload.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::model::Op;

// ---------------------------------------------------------------------------
// Pure helpers — frame encode/decode over byte buffers (Miri-safe, no file IO)
// ---------------------------------------------------------------------------

/// Encode `op` as a framed record and append to `buf`.
/// Frame: `[len: u32 LE][payload: bincode(op)][crc32: u32 LE]`
fn frame(op: &Op, buf: &mut Vec<u8>) -> Result<()> {
    let payload = bincode::serialize(op).context("bincode serialize")?;
    let len = u32::try_from(payload.len()).context("payload too large for u32 len")?;
    let crc = crc32fast::hash(&payload);

    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&payload);
    buf.extend_from_slice(&crc.to_le_bytes());
    Ok(())
}

/// Result of attempting to parse a single frame from `data` at byte offset `pos`.
enum ParseResult {
    /// A complete, CRC-valid frame was decoded. Contains the Op and the new offset.
    Good(Op, usize),
    /// The frame is at the tail but incomplete or CRC-bad (torn write).
    TornTail,
    /// End of data — no bytes remain.
    Eof,
    /// A bad frame with good data following it — genuine corruption.
    Corruption(String),
}

/// Try to parse a single frame from `data` starting at `pos`. Uses `total_len` to
/// determine if we are at the tail (nothing follows) vs. in the middle.
fn parse_frame(data: &[u8], pos: usize) -> ParseResult {
    // Need at least 4 bytes for the length header.
    if pos >= data.len() {
        return ParseResult::Eof;
    }

    let remaining = &data[pos..];

    // Not enough bytes for the length field?
    if remaining.len() < 4 {
        return ParseResult::TornTail;
    }

    let len = u32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]) as usize;

    // 4 (len) + len (payload) + 4 (crc)
    let frame_size = 4 + len + 4;

    if remaining.len() < frame_size {
        // Not enough bytes for the full frame.
        return ParseResult::TornTail;
    }

    let payload = &remaining[4..4 + len];
    let stored_crc = u32::from_le_bytes([
        remaining[4 + len],
        remaining[4 + len + 1],
        remaining[4 + len + 2],
        remaining[4 + len + 3],
    ]);

    let computed_crc = crc32fast::hash(payload);
    if computed_crc != stored_crc {
        // CRC mismatch — is there more data after this frame?
        if remaining.len() > frame_size {
            return ParseResult::Corruption(format!(
                "CRC mismatch at offset {pos}: expected {computed_crc:#010x}, got {stored_crc:#010x}"
            ));
        } else {
            return ParseResult::TornTail;
        }
    }

    // Decode the payload.
    match bincode::deserialize::<Op>(payload) {
        Ok(op) => {
            // Is there more data after this frame?
            let next = pos + frame_size;
            if next < data.len() || next == data.len() {
                ParseResult::Good(op, next)
            } else {
                // Shouldn't happen, but guard anyway.
                ParseResult::Good(op, next)
            }
        }
        Err(e) => {
            // Decode failure — is this the tail?
            if remaining.len() > frame_size {
                ParseResult::Corruption(format!("bincode decode failed at offset {pos}: {e}"))
            } else {
                ParseResult::TornTail
            }
        }
    }
}

/// Parse all frames from a byte buffer. Returns `(ops, good_byte_count)`.
/// On a torn tail the function stops and returns the last known-good position.
/// On corruption (bad record followed by good records) returns an error.
fn parse_all_frames(data: &[u8]) -> Result<(Vec<Op>, usize)> {
    let mut ops = Vec::new();
    let mut pos = 0usize;
    let mut last_good = 0usize;

    loop {
        match parse_frame(data, pos) {
            ParseResult::Good(op, next) => {
                ops.push(op);
                last_good = next;
                pos = next;
            }
            ParseResult::Eof => {
                break;
            }
            ParseResult::TornTail => {
                // Stop here; truncate to last_good later.
                break;
            }
            ParseResult::Corruption(msg) => {
                bail!("log corruption: {msg}");
            }
        }
    }

    Ok((ops, last_good))
}

// ---------------------------------------------------------------------------
// OpLog
// ---------------------------------------------------------------------------

/// Backing storage — either a real file or an in-memory buffer.
enum Backend {
    File { file: File, path: PathBuf },
    Memory { buf: Vec<u8> },
}

/// Append-only handle to the op log.
pub struct OpLog {
    backend: Backend,
}

impl OpLog {
    /// Open or create the log at `path`, **replay** all committed records into a
    /// `Vec<Op>` (in order), and return the write handle alongside them. A torn or
    /// CRC-failing *tail* record (crash mid-append) is recovered by truncating the
    /// file to the last good record — not an error. A bad record in the *middle*
    /// is corruption (error).
    pub fn open(path: &Path) -> Result<(OpLog, Vec<Op>)> {
        // Open (or create) the file for reading + writing.
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("open log at {}", path.display()))?;

        // Read the entire file.
        let mut data = Vec::new();
        file.read_to_end(&mut data).context("read log file")?;

        // Parse all frames, recovering torn tails.
        let (ops, good_len) = parse_all_frames(&data)?;

        // Truncate the file to the last good position if it has trailing garbage.
        if good_len < data.len() {
            file.set_len(good_len as u64)
                .context("truncate torn log tail")?;
        }

        // Seek to the end (good_len) for appending.
        file.seek(SeekFrom::Start(good_len as u64))
            .context("seek to end of log")?;

        let path = path.to_path_buf();
        Ok((
            OpLog {
                backend: Backend::File { file, path },
            },
            ops,
        ))
    }

    /// An in-memory-only log (no backing file). For tests.
    pub fn in_memory() -> OpLog {
        OpLog {
            backend: Backend::Memory { buf: Vec::new() },
        }
    }

    /// Append one framed record. Does NOT fsync — the caller batches then calls
    /// [`sync`](Self::sync).
    pub fn append(&mut self, op: &Op) -> Result<()> {
        let mut frame_buf = Vec::new();
        frame(op, &mut frame_buf)?;

        match &mut self.backend {
            Backend::File { file, .. } => {
                file.write_all(&frame_buf).context("write log frame")?;
            }
            Backend::Memory { buf } => {
                buf.extend_from_slice(&frame_buf);
            }
        }
        Ok(())
    }

    /// fsync the backing file (no-op for in-memory).
    pub fn sync(&mut self) -> Result<()> {
        match &mut self.backend {
            Backend::File { file, .. } => {
                file.sync_all().context("sync log")?;
            }
            Backend::Memory { .. } => {}
        }
        Ok(())
    }

    /// Atomically rewrite the log to contain exactly `ops` (compaction).
    /// Writes all ops as frames to a temp file in the same directory, fsyncs,
    /// atomically renames over the log, then reopens the append handle.
    pub fn rewrite(&mut self, ops: &[Op]) -> Result<()> {
        match &mut self.backend {
            Backend::File { path, .. } => {
                let path = path.clone();
                let dir = path.parent().unwrap_or(Path::new("."));

                // Build the full frame content.
                let mut frame_buf = Vec::new();
                for op in ops {
                    frame(op, &mut frame_buf)?;
                }

                // Write to a temp file in the same directory.
                let tmp_path = dir.join("log.tmp");
                {
                    let mut tmp = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&tmp_path)
                        .context("open log.tmp")?;
                    tmp.write_all(&frame_buf).context("write log.tmp")?;
                    tmp.sync_all().context("sync log.tmp")?;
                }

                // Atomically rename over the log.
                std::fs::rename(&tmp_path, &path).context("rename log.tmp over log")?;

                // Reopen the file for appending.
                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(false)
                    .truncate(false)
                    .open(&path)
                    .context("reopen log after rewrite")?;

                let new_end = frame_buf.len() as u64;
                file.seek(SeekFrom::Start(new_end))
                    .context("seek to end after rewrite")?;

                self.backend = Backend::File { file, path };
                Ok(())
            }
            Backend::Memory { buf } => {
                // In-memory: just rebuild the buffer.
                buf.clear();
                for op in ops {
                    frame(op, buf)?;
                }
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::Write;

    use super::*;
    use crate::model::Op;

    // --- Pure helpers (Miri-safe) ---

    fn make_ops() -> Vec<Op> {
        vec![
            Op::CreateCollection {
                collection: "col1".into(),
            },
            Op::SetMeta {
                collection: "col1".into(),
                meta: {
                    let mut m = BTreeMap::new();
                    m.insert("k".into(), "v".into());
                    m
                },
            },
            Op::Upsert {
                collection: "col1".into(),
                id: "doc1".into(),
                row: 0,
                attrs: BTreeMap::new(),
            },
            Op::Delete {
                collection: "col1".into(),
                id: "doc1".into(),
            },
            Op::DropCollection {
                collection: "col1".into(),
            },
        ]
    }

    #[test]
    fn frame_round_trip_single() {
        let op = Op::CreateCollection {
            collection: "test".into(),
        };
        let mut buf = Vec::new();
        frame(&op, &mut buf).unwrap();

        let (ops, good_len) = parse_all_frames(&buf).unwrap();
        assert_eq!(ops, vec![op]);
        assert_eq!(good_len, buf.len());
    }

    #[test]
    fn frame_round_trip_many() {
        let orig = make_ops();
        let mut buf = Vec::new();
        for op in &orig {
            frame(op, &mut buf).unwrap();
        }

        let (ops, good_len) = parse_all_frames(&buf).unwrap();
        assert_eq!(ops, orig);
        assert_eq!(good_len, buf.len());
    }

    #[test]
    fn parse_empty_buffer() {
        let (ops, good_len) = parse_all_frames(&[]).unwrap();
        assert!(ops.is_empty());
        assert_eq!(good_len, 0);
    }

    #[test]
    fn parse_truncated_length_field() {
        // Only 2 bytes — too short for even the length header.
        let (ops, good_len) = parse_all_frames(&[0x01, 0x00]).unwrap();
        assert!(ops.is_empty());
        assert_eq!(good_len, 0);
    }

    #[test]
    fn parse_torn_tail_truncated_payload() {
        let orig = make_ops();
        let mut buf = Vec::new();
        for op in &orig {
            frame(op, &mut buf).unwrap();
        }
        let good_len = buf.len();

        // Append a partial frame: length says 50 bytes but we write only 10.
        buf.extend_from_slice(&50u32.to_le_bytes());
        buf.extend_from_slice(&[0xABu8; 10]);

        let (ops, recovered_len) = parse_all_frames(&buf).unwrap();
        assert_eq!(ops, orig);
        assert_eq!(recovered_len, good_len);
    }

    #[test]
    fn parse_torn_tail_bad_crc() {
        let orig = make_ops();
        let mut buf = Vec::new();
        for op in &orig {
            frame(op, &mut buf).unwrap();
        }
        let good_len = buf.len();

        // Append a frame with a bad CRC at the very end.
        let last_op = Op::CreateCollection {
            collection: "last".into(),
        };
        let payload = bincode::serialize(&last_op).unwrap();
        let len = payload.len() as u32;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&payload);
        buf.extend_from_slice(&0xDEADBEEFu32.to_le_bytes()); // wrong CRC

        let (ops, recovered_len) = parse_all_frames(&buf).unwrap();
        assert_eq!(ops, orig);
        assert_eq!(recovered_len, good_len);
    }

    #[test]
    fn parse_corruption_in_middle() {
        let op1 = Op::CreateCollection {
            collection: "a".into(),
        };
        let op2 = Op::CreateCollection {
            collection: "b".into(),
        };

        let mut buf = Vec::new();
        frame(&op1, &mut buf).unwrap();

        // Insert a corrupt frame (bad CRC, but more data follows).
        let bad_op = Op::CreateCollection {
            collection: "bad".into(),
        };
        let payload = bincode::serialize(&bad_op).unwrap();
        let len = payload.len() as u32;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&payload);
        buf.extend_from_slice(&0xDEADBEEFu32.to_le_bytes()); // wrong CRC

        // Add a valid frame after the corrupt one.
        frame(&op2, &mut buf).unwrap();

        let result = parse_all_frames(&buf);
        assert!(result.is_err(), "expected corruption error");
        assert!(result.unwrap_err().to_string().contains("corruption"));
    }

    // --- File-backed tests (require real IO; Miri-ignored) ---

    #[cfg_attr(miri, ignore)]
    #[test]
    fn file_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");

        let ops = make_ops();

        // Write ops.
        {
            let (mut log, replayed) = OpLog::open(&path).unwrap();
            assert!(replayed.is_empty());
            for op in &ops {
                log.append(op).unwrap();
            }
            log.sync().unwrap();
        }

        // Reopen and replay.
        {
            let (_, replayed) = OpLog::open(&path).unwrap();
            assert_eq!(replayed, ops);
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn crash_recovery_garbage_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");

        let ops = make_ops();

        // Write good ops.
        {
            let (mut log, _) = OpLog::open(&path).unwrap();
            for op in &ops {
                log.append(op).unwrap();
            }
            log.sync().unwrap();
        }

        let good_len = std::fs::metadata(&path).unwrap().len();

        // Append random garbage bytes (simulating a crash mid-append).
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&[0xFF, 0xAB, 0x12, 0x99, 0x00, 0xDE, 0xAD, 0xBE, 0xEF])
                .unwrap();
        }

        assert!(std::fs::metadata(&path).unwrap().len() > good_len);

        // Open should recover and truncate.
        let (_, replayed) = OpLog::open(&path).unwrap();
        assert_eq!(replayed, ops);

        // File should be physically truncated to good_len.
        assert_eq!(std::fs::metadata(&path).unwrap().len(), good_len);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn crash_recovery_truncated_mid_frame() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");

        let ops = make_ops();

        // Write good ops.
        {
            let (mut log, _) = OpLog::open(&path).unwrap();
            for op in &ops {
                log.append(op).unwrap();
            }
            log.sync().unwrap();
        }

        let good_len = std::fs::metadata(&path).unwrap().len();

        // Append a partial frame: length header claiming 100 bytes, but only 20 written.
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&100u32.to_le_bytes()).unwrap();
            f.write_all(&[0x55u8; 20]).unwrap();
        }

        // Open should recover and truncate.
        let (_, replayed) = OpLog::open(&path).unwrap();
        assert_eq!(replayed, ops);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), good_len);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn crash_recovery_bad_crc_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");

        let ops = make_ops();

        // Write good ops.
        {
            let (mut log, _) = OpLog::open(&path).unwrap();
            for op in &ops {
                log.append(op).unwrap();
            }
            log.sync().unwrap();
        }

        let good_len = std::fs::metadata(&path).unwrap().len();

        // Append a fully-sized frame but with a wrong CRC at the tail.
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            let bad_op = Op::CreateCollection {
                collection: "bad".into(),
            };
            let payload = bincode::serialize(&bad_op).unwrap();
            let len = payload.len() as u32;
            f.write_all(&len.to_le_bytes()).unwrap();
            f.write_all(&payload).unwrap();
            f.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap();
        }

        let (_, replayed) = OpLog::open(&path).unwrap();
        assert_eq!(replayed, ops);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), good_len);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn rewrite_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");

        // Write initial ops.
        let (mut log, _) = OpLog::open(&path).unwrap();
        for op in &make_ops() {
            log.append(op).unwrap();
        }
        log.sync().unwrap();

        // Rewrite with a smaller set.
        let compacted = vec![
            Op::CreateCollection {
                collection: "col1".into(),
            },
            Op::Upsert {
                collection: "col1".into(),
                id: "docX".into(),
                row: 0,
                attrs: BTreeMap::new(),
            },
        ];
        log.rewrite(&compacted).unwrap();

        // Verify compacted content is replayed correctly.
        let (_, replayed) = OpLog::open(&path).unwrap();
        assert_eq!(replayed, compacted);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn append_after_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");

        let (mut log, _) = OpLog::open(&path).unwrap();
        log.append(&Op::CreateCollection {
            collection: "c".into(),
        })
        .unwrap();
        log.sync().unwrap();

        // Rewrite to empty.
        log.rewrite(&[]).unwrap();

        // Append new op after rewrite.
        let new_op = Op::Upsert {
            collection: "c".into(),
            id: "x".into(),
            row: 42,
            attrs: BTreeMap::new(),
        };
        log.append(&new_op).unwrap();
        log.sync().unwrap();

        let (_, replayed) = OpLog::open(&path).unwrap();
        assert_eq!(replayed, vec![new_op]);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn in_memory_append_and_rewrite() {
        let mut log = OpLog::in_memory();
        let op = Op::CreateCollection {
            collection: "m".into(),
        };
        log.append(&op).unwrap();
        log.sync().unwrap();

        let ops2 = vec![Op::DropCollection {
            collection: "m".into(),
        }];
        log.rewrite(&ops2).unwrap();
        // No panic, no error — that's the contract for in-memory.
    }
}
