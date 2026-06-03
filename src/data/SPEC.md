# `data` module — spec

Implement `DataSegment` in `mod.rs`: the append-only, fixed-stride, row-major `f32`
matrix backing the `data` file, held entirely in RAM. **Do not change the public
signatures.** Root design: `../../SPEC.md` §5.1, §5.3, §6.

## On-disk layout (`data` file)
```
[ 64-byte header ][ row 0: dim×f32 ][ row 1: dim×f32 ] ...
header: b"NIDUS\0" (6) | version: u16 | dimension: u32 | zero-pad to 64 bytes
```
All little-endian. Rows begin at byte offset 64. Row `i` occupies bytes
`64 + i*dim*4 .. 64 + (i+1)*dim*4`.

## Methods (keep signatures)
- `open(path, dimension) -> Result<DataSegment>`: create the file with a fresh
  header if absent; else read the header and verify magic + that the stored
  dimension equals `dimension` (else `anyhow::bail!` a clear dimension-mismatch /
  corrupt error). Then read **only fully-written rows**: `rows = (file_len - 64) /
  (dim*4)`; ignore any trailing partial row (a crash mid-append). Load them into an
  in-RAM `Vec<f32>`. Keep a writable append handle (seek to end of the last whole
  row, i.e. truncate off any partial tail).
- `in_memory(dimension)`: already implemented in the stub — keep it.
- `dimension(&self) -> usize`, `row_count(&self) -> u64`: provided in stub; keep
  consistent if you change fields.
- `row(&self, i) -> &[f32]`: slice `vectors[i*dim .. (i+1)*dim]`.
- `append(&mut self, vector) -> Result<u64>`: `bail!` if `vector.len() != dim`;
  push into RAM, write the row's `dim*4` bytes to the file tail, return the new row
  index. **Do not fsync here.**
- `sync(&mut self) -> Result<()>`: fsync the file (`File::sync_all`); no-op when
  in-memory.
- `rewrite(&mut self, rows) -> Result<()>`: `bail!` unless `rows.len() % dim == 0`.
  Write header + `rows` to a sibling temp file in the same dir, fsync, atomically
  rename over the `data` file, reopen the append handle, swap the in-RAM `Vec`.

## f32 ↔ bytes
Little-endian. `f32::to_le_bytes` / `from_le_bytes` is fine; you may bulk-read into a
pre-sized `Vec<f32>` if you keep it pure safe Rust (no `unsafe`, no transmute). No
new dependency required.

## Constraints
Pure safe Rust (`#![forbid(unsafe_code)]`). Use `anyhow` for errors
(`bail!`/`.context()`). No mmap.

## Tests (`tests` submodule)
Pure helpers (header encode/parse, row math) MUST be Miri-clean. File-backed tests
(open/append/reopen/rewrite, partial-tail truncation, dimension-mismatch error) use
`tempfile::tempdir()` and carry `#[cfg_attr(miri, ignore)]` (they fsync). Verify:
append→row round-trips bytes exactly; reopen sees prior rows; a file with a partial
trailing row opens and reports the whole-row count; reopening with a different
dimension errors; `rewrite` compacts and survives reopen.
