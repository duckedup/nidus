# `log` module — spec

Implement `OpLog` in `mod.rs`: the append-only, framed, checksummed op stream that
is the store's commit record and crash-recovery mechanism. **Do not change the
public signatures.** Root design: `../../SPEC.md` §5.2, §6.1. The `Op` enum lives in
`crate::model` (already defined) — do not redefine it.

## Frame format
Each record: `[ len: u32 ][ payload: len bytes ][ crc32: u32 ]`, all little-endian.
- `payload = bincode::serialize(op)`.
- `crc32 = crc32fast::hash(payload)` (covers the payload bytes).

## Methods (keep signatures)
- `open(path) -> Result<(OpLog, Vec<Op>)>`: create if absent. **Replay**: read
  records sequentially; for each, read `len`, then `len` payload bytes, then the
  `crc32`; verify crc and bincode-decode. Collect decoded `Op`s in order. **Torn
  tail recovery**: if at the tail there are fewer bytes than a full frame, or the
  final record's crc/decode fails, treat it as a crash mid-append → truncate the
  file to the end of the last good record (`set_len`) and stop, returning the good
  ops. A length that overflows the remaining file at the tail is also a torn tail.
  (A corrupt record that is *not* at the tail — i.e. good records follow it — is
  genuine corruption: `bail!`.) Keep a writable append handle positioned at the
  truncation point.
- `in_memory()`: already stubbed — keep it (append/sync may be no-ops that still
  succeed, or buffer in a `Vec<u8>`; replay isn't needed for in-memory).
- `append(&mut self, op) -> Result<()>`: serialize, frame, write. **No fsync.**
- `sync(&mut self) -> Result<()>`: `File::sync_all`; no-op in-memory.
- `rewrite(&mut self, ops) -> Result<()>`: write all `ops` as frames to a temp file
  in the same dir, fsync, atomically rename over `log`, reopen append handle.

## Constraints
Pure safe Rust (`#![forbid(unsafe_code)]`). Deps: `bincode`, `crc32fast`, `anyhow`
(all already in `Cargo.toml`). Be robust to truncated/garbage tails — never panic on
bad bytes; convert to a torn-tail truncation or an `anyhow` error.

## Tests
Frame round-trip (encode one/many ops → replay → identical `Vec<Op>`) can run on a
temp file. MUST include crash-recovery tests: build a valid log, then (a) append
random/garbage trailing bytes and (b) truncate the file mid-final-record; assert
`open` recovers the prior good ops and physically truncates the file to the
last-good length. File-backed tests use `tempfile` and `#[cfg_attr(miri, ignore)]`.
If you factor out a pure `frame()/parse_frame()` helper over `Vec<u8>`, test that
Miri-clean (no file).
