//! The `log` file: an append-only, framed, checksummed op stream — the commit
//! record and crash-recovery mechanism. Contract: see the root `SPEC.md` §5.2, §6.1, §6.6.
//!
//! Each record is framed `[len: u32][payload: bincode(Op)][crc32: u32]`, all
//! little-endian; `crc32` (via `crc32fast`) covers the payload.

use anyhow::{Context, Result, bail};

use crate::backend::{Appender, MemAppender};
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

/// Append-only handle to the op log over a persistence [`Appender`] (a local file, or
/// RAM for an in-memory store).
pub struct OpLog {
    appender: Box<dyn Appender>,
}

impl OpLog {
    /// Open or create the `log` file at `path` (convenience over
    /// [`open_with`](Self::open_with): wraps a local `FileAppender`). The store path
    /// goes through the persistence backend's appender via `open_with`; this direct
    /// path-based form backs the log's own file tests.
    #[cfg(test)]
    pub fn open(path: &std::path::Path) -> Result<(OpLog, Vec<Op>)> {
        let appender = crate::backend::FileAppender::open(path)
            .with_context(|| format!("open log at {}", path.display()))?;
        Self::open_with(Box::new(appender))
    }

    /// Open the log over an already-opened persistence [`Appender`]: **replay** all
    /// committed records into a `Vec<Op>` (in order) and return the write handle
    /// alongside them. A torn or CRC-failing *tail* record (crash mid-append) is
    /// recovered by truncating to the last good record — not an error. A bad record in
    /// the *middle* is corruption (error).
    pub fn open_with(mut appender: Box<dyn Appender>) -> Result<(OpLog, Vec<Op>)> {
        // Read the entire log (the `read_to_end` default reserves fallibly, so a huge
        // log fails with an error instead of aborting the process on allocation).
        let mut data = Vec::new();
        appender.read_to_end(&mut data).context("read log file")?;

        // Parse all frames, recovering torn tails.
        let (ops, good_len) = parse_all_frames(&data)?;

        // Truncate to the last good position if there is trailing garbage.
        if good_len < data.len() {
            appender
                .truncate_to(good_len as u64)
                .context("truncate torn log tail")?;
        }

        Ok((OpLog { appender }, ops))
    }

    /// An in-memory-only log (no backing file). For tests and in-memory stores.
    pub fn in_memory() -> OpLog {
        OpLog {
            appender: Box::new(MemAppender::new()),
        }
    }

    /// Append one framed record. Does NOT fsync — the caller batches then calls
    /// [`sync`](Self::sync).
    ///
    /// **Atomic per frame.** If the write fails partway (e.g. ENOSPC), the appender
    /// rolls back to the offset it started at, so a torn frame never persists mid-file
    /// — without this, the next append would write past the partial bytes and
    /// `parse_all_frames` would reject the result as hard `Corruption` on reopen.
    pub fn append(&mut self, op: &Op) -> Result<()> {
        let mut frame_buf = Vec::new();
        frame(op, &mut frame_buf)?;
        self.appender.append(&frame_buf).context("write log frame")
    }

    /// The committed byte length — the append point. A writer captures this before
    /// a batch and passes it to [`truncate_to`](Self::truncate_to) to undo a failed
    /// one.
    pub fn offset(&self) -> Result<u64> {
        self.appender.len().context("log length")
    }

    /// Roll the log back to `offset`, discarding any frames appended after it.
    /// The batch-rollback primitive (counterpart to [`offset`](Self::offset)).
    pub fn truncate_to(&mut self, offset: u64) -> Result<()> {
        self.appender.truncate_to(offset).context("truncate log")
    }

    /// fsync the backing appender (no-op for in-memory).
    pub fn sync(&mut self) -> Result<()> {
        self.appender.sync().context("sync log")
    }

    /// Atomically rewrite the log to contain exactly `ops` (compaction). The appender
    /// handles the atomic temp + fsync + rename (or, in-memory, the buffer swap).
    pub fn rewrite(&mut self, ops: &[Op]) -> Result<()> {
        let mut frame_buf = Vec::new();
        for op in ops {
            frame(op, &mut frame_buf)?;
        }
        self.appender.rewrite(&frame_buf).context("rewrite log")
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

    #[test]
    fn in_memory_offset_tracks_appends_and_truncate() {
        let mut log = OpLog::in_memory();
        assert_eq!(log.offset().unwrap(), 0);
        log.append(&Op::CreateCollection {
            collection: "a".into(),
        })
        .unwrap();
        let mark = log.offset().unwrap();
        assert!(mark > 0);
        log.append(&Op::CreateCollection {
            collection: "b".into(),
        })
        .unwrap();
        assert!(log.offset().unwrap() > mark);
        log.truncate_to(mark).unwrap();
        assert_eq!(log.offset().unwrap(), mark);
    }

    #[test]
    fn in_memory_truncate_beyond_errors() {
        let mut log = OpLog::in_memory();
        log.append(&Op::CreateCollection {
            collection: "a".into(),
        })
        .unwrap();
        assert!(log.truncate_to(9999).is_err());
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
    fn file_truncate_to_drops_tail_frames_and_replays() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");

        let a = Op::CreateCollection {
            collection: "a".into(),
        };
        let b = Op::CreateCollection {
            collection: "b".into(),
        };

        let (mut log, _) = OpLog::open(&path).unwrap();
        log.append(&a).unwrap();
        log.append(&b).unwrap();
        let mark = log.offset().unwrap();
        // Append a third frame, then roll back to the mark.
        log.append(&Op::CreateCollection {
            collection: "c".into(),
        })
        .unwrap();
        log.truncate_to(mark).unwrap();
        log.sync().unwrap();
        drop(log);

        // Reopen: only the first two frames survive, replayed cleanly (no corruption).
        let (_, replayed) = OpLog::open(&path).unwrap();
        assert_eq!(replayed, vec![a, b]);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), mark);
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
