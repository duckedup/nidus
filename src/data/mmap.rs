//! The one memory-map seam (SPEC §9 / §14.6 phase 3). This module is the **sole** place in
//! the crate that uses `unsafe`: it wraps the platform `mmap` syscall (via `memmap2`) behind a
//! safe [`MappedSegment`] that hands out a read-only `&[u8]` view of an immutable segment file.
//!
//! Everything outside this module — including the `&[u8]` → `&[f32]` reinterpret in
//! [`DataSegment`](super::DataSegment) — stays in safe Rust (the cast goes through
//! `bytemuck::cast_slice`, sound by the on-disk layout invariant: `mmap` returns a
//! page-aligned base and the fixed 64-byte header leaves the row region 4-aligned with a
//! length that is a multiple of `size_of::<f32>()` — see `mmap_rows` in the parent module).

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use memmap2::Mmap;

/// A read-only memory-map of a sealed (immutable) segment file. Holding it keeps the mapping
/// alive; dropping it unmaps. nidus only ever maps segments the manifest marks immutable, which
/// are never written again — the invariant that makes the map sound.
pub struct MappedSegment {
    map: Mmap,
}

impl MappedSegment {
    /// Map `path` read-only. The file must be non-empty (a sealed segment always has a header
    /// plus at least one row, so this holds for every segment nidus maps).
    pub fn open(path: &Path) -> Result<MappedSegment> {
        let file = File::open(path)
            .with_context(|| format!("failed to open segment for mmap at {}", path.display()))?;
        // SAFETY: this is the crate's only `unsafe`. `Mmap::map` is `unsafe` because the mapped
        // bytes must not be mutated underneath the mapping. nidus maps **only immutable
        // segments** — a sealed segment is never appended to, truncated, or rewritten in place
        // (the manifest's commit discipline, SPEC §14.2; compaction renames a fresh object over
        // the name and drops this map first). So the bytes are stable for the map's lifetime.
        #[allow(unsafe_code)]
        let map = unsafe {
            Mmap::map(&file)
                .with_context(|| format!("failed to mmap segment at {}", path.display()))?
        };
        Ok(MappedSegment { map })
    }

    /// The mapped bytes (the whole segment object: 64-byte header followed by f32 rows).
    pub fn bytes(&self) -> &[u8] {
        &self.map
    }
}
