//! The one memory-map seam (SPEC §9 / §14.6 phase 3). This module is the **sole** place in
//! the crate that uses `unsafe`: it wraps the platform `mmap` syscall (via `memmap2`) behind a
//! safe [`MappedSegment`] that hands out a read-only `&[u8]` view of an immutable segment file.

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
        // SAFETY: the crate's only `unsafe`. `Mmap::map` requires the mapped bytes not be mutated
        // underneath it, and nidus maps only immutable segments — never appended to, truncated, or
        // rewritten in place (SPEC §14.2), with compaction dropping this map first.
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
