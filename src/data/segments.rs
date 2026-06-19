//! [`Segments`]: the live segment set as one logical vector matrix (SPEC §14.2).
//!
//! A store's vectors live in an ordered list of immutable **segment** objects plus one
//! **active** (appendable) segment — the last in the list. [`Segments`] presents them to the
//! rest of the store as a single dense **global row-id space**: global row `R` lives in the
//! segment whose cumulative `[base, base+rows)` range contains it, served at local row
//! `R - base`. This is the key that keeps segmentation transparent to the search/quant/ANN
//! paths — they address vectors by dense global row exactly as they did over the one
//! monolithic matrix; only *where* a row physically sits changed.
//!
//! Each segment is a [`DataSegment`] over its own backend object (via
//! [`crate::backend::appender_for`]); the [`Manifest`](crate::manifest::Manifest) names them
//! and is the atomic commit point. Sealing rotates the active segment to immutable and starts
//! a fresh one — no data is moved (a sealed segment is simply never appended to again).

use std::borrow::Cow;
use std::sync::Arc;

use anyhow::{Result, bail};

use super::{DataSegment, HEADER_LEN};
use crate::backend::{Persistence, appender_for};
use crate::manifest::{BASE_SEGMENT, Manifest};
use crate::model::Distance;

/// One open segment: its object name and the loaded [`DataSegment`].
struct Seg {
    name: String,
    data: DataSegment,
}

/// A freshly re-read active segment plus the global row count it implies — staged by
/// [`Segments::reopen_active`] so the store can finish all fallible IO before the one
/// infallible swap ([`Segments::install_active`]), keeping an incremental refresh atomic.
pub struct PendingActive {
    data: DataSegment,
    row_count: u64,
}

impl PendingActive {
    /// The global row count once this staged active segment is installed.
    pub fn row_count(&self) -> u64 {
        self.row_count
    }
}

/// The live segment set, addressed as one dense global row space. The last segment is the
/// active (appendable) one; the rest are immutable.
pub struct Segments {
    distance: Distance,
    /// The backend the segment objects live on; `None` for an in-memory store.
    persistence: Option<Arc<dyn Persistence>>,
    /// Open segments in global-row order (last = active). Never empty.
    segs: Vec<Seg>,
    /// `base[i]` = global row index of `segs[i]`'s first row (cumulative rows before it).
    /// Only the active (last) segment grows, so these are stable except on seal/rewrite.
    base: Vec<u64>,
    /// Next id for minting a fresh `seg-NNNNNNNN` name (monotonic, never reused).
    next_id: u64,
    /// Monotonic manifest version (Phase-4 reader refresh).
    version: u64,
    /// Compare-and-swap fencing for the object-store appenders (cluster mode, SPEC §14.6).
    /// Threaded into every [`appender_for`] this set opens — the active segment, segments
    /// minted on [`seal`](Self::seal), and the base on [`rewrite`](Self::rewrite) — so a
    /// superseded cluster writer's whole-object rewrite is fenced. `false` for the
    /// single-writer default and in-memory stores.
    cas: bool,
}

impl Segments {
    /// Open every segment named by `manifest` over `persistence`, assembled into one global
    /// row space. `cap` (the store's `max_vector_bytes`) is enforced **before** loading any
    /// segment into RAM — summing each segment object's vector bytes and refusing past the
    /// cap (§6.6 "refuse before allocating", generalized across segments).
    pub fn open(
        persistence: Arc<dyn Persistence>,
        manifest: &Manifest,
        cap: Option<u64>,
        mmap: bool,
        cas: bool,
    ) -> Result<Segments> {
        let dimension = manifest.dimension as usize;
        let distance = manifest.distance;
        if manifest.segments.is_empty() {
            bail!("manifest names no segments — corrupt store");
        }

        // The cap (`max_vector_bytes`) bounds total vector bytes across every segment,
        // residency-independent — it counts mmap'd segments too (§6.6, generalized).
        let mut acc_bytes: u64 = 0;
        let mut tally = |seg_bytes: u64| -> Result<()> {
            if let Some(cap) = cap {
                acc_bytes = acc_bytes.saturating_add(seg_bytes);
                if acc_bytes > cap {
                    bail!(
                        "store segments hold {acc_bytes} bytes of vectors, exceeding \
                         max_vector_bytes ({cap} bytes)"
                    );
                }
            }
            Ok(())
        };

        let mut segs: Vec<Seg> = Vec::new();
        let mut base: Vec<u64> = Vec::new();
        let mut acc_rows: u64 = 0;
        let n = manifest.segments.len();
        for (i, name) in manifest.segments.iter().enumerate() {
            // Memory-map a segment only when it is **immutable** (not the active last segment),
            // mmap is requested, the host is little-endian (the on-disk f32 layout, §5.1), and
            // the backend stores it as a mappable local file. Otherwise load it into RAM.
            let is_active = i == n - 1;
            let local_path = (mmap && !is_active && cfg!(target_endian = "little"))
                .then(|| persistence.local_path(name))
                .flatten();
            let data = match local_path {
                Some(path) => {
                    let data = DataSegment::open_mmap(&path, dimension, distance)?;
                    tally(data.row_count() * dimension as u64 * 4)?;
                    data
                }
                None => {
                    // Open the append handle once, size-check it (pre-allocation guard), then load.
                    let ap = appender_for(&persistence, name, cas)?;
                    tally(ap.len()?.saturating_sub(HEADER_LEN as u64))?;
                    DataSegment::open_with(ap, dimension, distance)?
                }
            };
            base.push(acc_rows);
            acc_rows += data.row_count();
            segs.push(Seg {
                name: name.clone(),
                data,
            });
        }

        Ok(Segments {
            distance,
            persistence: Some(persistence),
            segs,
            base,
            next_id: manifest.next_id,
            version: manifest.version,
            cas,
        })
    }

    /// Re-read **only** the active (last) segment object — picking up rows a separate writer
    /// appended — leaving every immutable segment untouched (they never change). The result is
    /// *staged*, not installed: the caller finishes its remaining fallible work and then calls
    /// [`install_active`](Self::install_active), so a [`refresh`](crate::Nidus::refresh) that
    /// races a concurrent writer stays atomic (all IO into locals, one infallible swap).
    ///
    /// This is the incremental refresh fast path (SPEC §14.6 / nidus-bdg): valid only when the
    /// manifest version is unchanged (no seal/compaction restructured the set). On a version
    /// change the caller re-opens the whole set via [`open`](Self::open). The `cap`
    /// (`max_vector_bytes`) is re-checked against the grown total before the new bytes resolve.
    pub fn reopen_active(&self, cap: Option<u64>) -> Result<PendingActive> {
        let p = self
            .persistence
            .as_ref()
            .expect("incremental refresh requires a durable backend");
        let last = self.segs.len() - 1;
        let dim = self.dimension();
        // The active segment is never memory-mapped (mmap is for immutable segments only), so a
        // plain RAM-loading appender is always correct here.
        let ap = appender_for(p, &self.segs[last].name, false)?;
        let data = DataSegment::open_with(ap, dim, self.distance)?;
        let row_count = self.base[last] + data.row_count();
        if let Some(cap) = cap {
            let bytes = row_count.saturating_mul(dim as u64).saturating_mul(4);
            if bytes > cap {
                bail!(
                    "store segments would hold {bytes} bytes of vectors after refresh, exceeding \
                     max_vector_bytes ({cap} bytes)"
                );
            }
        }
        Ok(PendingActive { data, row_count })
    }

    /// Install a [`reopen_active`](Self::reopen_active)-staged active segment and adopt the new
    /// manifest `version` — the infallible swap that completes an incremental refresh. Immutable
    /// segments and bases are unchanged (only the active segment grows); adopting `version` keeps
    /// the next [`refresh`](crate::Nidus::refresh) currency check accurate.
    pub fn install_active(&mut self, pending: PendingActive, version: u64) {
        let last = self.segs.len() - 1;
        self.segs[last].data = pending.data;
        self.version = version;
    }

    /// Whether `names` is exactly this set's current segment list (same names, same order) — the
    /// **structural** change signal for [`refresh`](crate::Nidus::refresh): an unchanged list
    /// means only the active segment grew (plain appends → incremental path); a changed list
    /// means a seal/compaction restructured the set (→ full re-open). Used instead of the manifest
    /// `version`, which in cluster mode advances on *every* commit (it is the commit counter).
    pub fn segment_names_match(&self, names: &[String]) -> bool {
        self.segs.len() == names.len() && self.segs.iter().zip(names).all(|(s, n)| &s.name == n)
    }

    /// An in-memory-only single-segment store (no backing objects, no manifest on disk).
    pub fn in_memory_with(dimension: usize, distance: Distance) -> Segments {
        Segments {
            distance,
            persistence: None,
            segs: vec![Seg {
                name: BASE_SEGMENT.to_string(),
                data: DataSegment::in_memory_with(dimension, distance),
            }],
            base: vec![0],
            next_id: 1,
            version: 1,
            cas: false,
        }
    }

    /// The pinned dimension (every segment shares it; read from the base segment).
    pub fn dimension(&self) -> usize {
        self.segs[0].data.dimension()
    }

    /// Total rows across all segments (the global row count).
    pub fn row_count(&self) -> u64 {
        let last = self.segs.len() - 1;
        self.base[last] + self.segs[last].data.row_count()
    }

    /// Rows in the active (appendable) segment — the seal-threshold yardstick.
    pub fn active_rows(&self) -> u64 {
        self.segs.last().unwrap().data.row_count()
    }

    /// The manifest version this segment set was loaded/sealed at — the Phase-4
    /// reader-refresh signal (SPEC §14.6): a [`ReadOnly`](crate::OpenMode::ReadOnly) reader
    /// adopts a newer manifest when the on-disk version exceeds this. Bumped on every
    /// seal/rewrite (the structural commits); plain appends leave it unchanged.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Advance the manifest version by one and return it — the cluster **commit counter**
    /// (SPEC §14.6 phase 5). In cluster mode the writer calls this on every durable batch so
    /// the published manifest version strictly increases on *any* change (not just seals), and
    /// a reader's [`refresh`](crate::Nidus::refresh) detects every commit with one manifest read.
    pub fn bump_version(&mut self) -> u64 {
        self.version += 1;
        self.version
    }

    /// `(base, rows)` for each segment in global-row order — the last entry is the active
    /// (appendable) segment. Lets the store align a per-segment IVF index to each segment's
    /// global row range `[base, base + rows)` (SPEC §14.3) without exposing segment internals.
    pub fn segment_ranges(&self) -> Vec<(u64, u64)> {
        self.segs
            .iter()
            .enumerate()
            .map(|(i, s)| (self.base[i], s.data.row_count()))
            .collect()
    }

    /// Number of live segments (last is the active one).
    #[cfg(test)]
    pub fn segment_count(&self) -> usize {
        self.segs.len()
    }

    /// Borrow global row `R` as a `dimension`-length slice, dispatching to its owning
    /// segment. Single-segment stores (the default) take the direct fast path.
    pub fn row(&self, global: u64) -> &[f32] {
        if self.segs.len() == 1 {
            return self.segs[0].data.row(global);
        }
        let i = match self.base.binary_search(&global) {
            Ok(i) => i,      // global is the first row of segment i
            Err(i) => i - 1, // base[i-1] <= global < base[i]
        };
        self.segs[i].data.row(global - self.base[i])
    }

    /// Append one vector to the **active** segment, returning its **global** row index.
    pub fn append(&mut self, vector: &[f32]) -> Result<u64> {
        let last = self.segs.len() - 1;
        let base = self.base[last];
        let local = self.segs[last].data.append(vector)?;
        Ok(base + local)
    }

    /// Roll the **active** segment back so the global row count is exactly `rows` (batch
    /// rollback — a batch only ever appends to the active segment, so `rows` is always ≥ the
    /// active segment's base).
    pub fn truncate_to(&mut self, rows: u64) -> Result<()> {
        let last = self.segs.len() - 1;
        let base = self.base[last];
        if rows < base {
            bail!(
                "truncate_to({rows}) is below the active segment base ({base}) — \
                 cannot roll back across a sealed segment boundary"
            );
        }
        self.segs[last].data.truncate_to(rows - base)
    }

    /// fsync the active segment (sealed segments are already durable and never re-written).
    pub fn sync(&mut self) -> Result<()> {
        let last = self.segs.len() - 1;
        self.segs[last].data.sync()
    }

    /// All current vectors as one contiguous slice — borrowed for the common single-segment
    /// store, otherwise concatenated into a fresh buffer. Only the O(N) quant *rebuild* uses
    /// this (open/compact/refit); the per-batch paths read row-by-row via [`row`](Self::row).
    pub fn vectors(&self) -> Cow<'_, [f32]> {
        if self.segs.len() == 1 {
            return Cow::Borrowed(self.segs[0].data.vectors());
        }
        let mut all: Vec<f32> = Vec::with_capacity(self.row_count() as usize * self.dimension());
        for seg in &self.segs {
            all.extend_from_slice(seg.data.vectors());
        }
        Cow::Owned(all)
    }

    /// Seal the active segment into an immutable one and start a fresh active segment,
    /// returning `true` when a seal happened. A no-op (returns `false`) when the active
    /// segment is empty — sealing nothing would leave an empty segment that breaks the
    /// strictly-increasing `base` invariant. No data is moved; the previously-active segment
    /// is simply never appended to again. The caller publishes the new manifest (the commit
    /// point) afterward.
    pub fn seal(&mut self) -> Result<bool> {
        if self.active_rows() == 0 {
            return Ok(false);
        }
        let new_base = self.row_count();
        let name = format!("seg-{:08}", self.next_id);
        let data = match &self.persistence {
            Some(p) => {
                let ap = appender_for(p, &name, self.cas)?;
                let mut d = DataSegment::open_with(ap, self.dimension(), self.distance)?;
                // Make the new (empty) segment's header durable before it is named by the
                // manifest, so a crash can never leave the manifest pointing at a
                // header-less object.
                d.sync()?;
                d
            }
            None => DataSegment::in_memory_with(self.dimension(), self.distance),
        };
        self.next_id += 1;
        self.version += 1;
        self.base.push(new_base);
        self.segs.push(Seg { name, data });
        Ok(true)
    }

    /// Collapse every segment into a single fresh active [`BASE_SEGMENT`] holding exactly
    /// `rows` (the compaction path). Returns the names of the segments that are no longer
    /// referenced (the caller deletes those objects). `rows.len()` must be a multiple of
    /// `dimension`.
    pub fn rewrite(&mut self, rows: &[f32]) -> Result<Vec<String>> {
        // The base segment (always `BASE_SEGMENT`, by construction) is rewritten in place;
        // the rest become unreferenced.
        let dropped: Vec<String> = self.segs[1..].iter().map(|s| s.name.clone()).collect();
        if self.segs[0].data.is_mmap() {
            // The base is memory-mapped (immutable) and has no writable appender. Open a fresh
            // write handle over `BASE_SEGMENT` *without* paging the old rows into RAM, then let
            // `rewrite` atomically replace the object. Replacing `segs[0]` drops the old map.
            let (dim, distance) = (self.dimension(), self.distance);
            let p = self
                .persistence
                .as_ref()
                .expect("a memory-mapped segment implies a local-FS backend");
            let ap = appender_for(p, BASE_SEGMENT, self.cas)?;
            let mut data = DataSegment::rewrite_target(ap, dim, distance);
            data.rewrite(rows)?;
            self.segs[0] = Seg {
                name: BASE_SEGMENT.to_string(),
                data,
            };
        } else {
            self.segs[0].data.rewrite(rows)?;
            self.segs[0].name = BASE_SEGMENT.to_string();
        }
        self.segs.truncate(1);
        self.base = vec![0];
        self.version += 1;
        Ok(dropped)
    }

    /// A manifest snapshot of the current segment set — what seal/compaction persist.
    pub fn manifest(&self) -> Manifest {
        let names = self.segs.iter().map(|s| s.name.clone()).collect();
        Manifest::new(
            self.dimension(),
            self.distance,
            names,
            self.next_id,
            self.version,
        )
    }

    /// Test-only fault seam: arm the active segment so its `(n+1)`-th subsequent append
    /// fails — lets the store tests exercise mid-batch rollback deterministically.
    #[cfg(test)]
    pub fn fail_after(&mut self, n: usize) {
        let last = self.segs.len() - 1;
        self.segs[last].data.fail_after(n);
    }
}

// `Segments` is shared inside `Arc<RwLock<Nidus>>`; searchers take `&self` and only `row()`
// (a shared read), writers hold `&mut self`. Both `DataSegment` and the `Arc<dyn Persistence>`
// are `Send + Sync`, so this is sound — asserted here so a future field can't silently break it.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Segments>();
};

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-segment in-memory `Segments`: seal after the first `split` rows, so global rows
    /// span a sealed segment and the active one. Pure RAM — Miri-clean.
    fn two_segment(dim: usize, rows: &[Vec<f32>], split: usize) -> Segments {
        let mut s = Segments::in_memory_with(dim, Distance::Cosine);
        for (i, r) in rows.iter().enumerate() {
            if i == split {
                assert!(s.seal().unwrap(), "expected a seal at the split point");
            }
            s.append(r).unwrap();
        }
        s
    }

    #[test]
    fn global_row_dispatch_across_segments() {
        let rows = vec![
            vec![1.0_f32, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 3.0],
        ];
        let s = two_segment(2, &rows, 2);
        assert_eq!(s.segment_count(), 2);
        assert_eq!(s.row_count(), 4);
        assert_eq!(s.active_rows(), 2);
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(s.row(i as u64), r.as_slice(), "global row {i}");
        }
    }

    #[test]
    fn vectors_concatenates_in_global_order() {
        let rows = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let s = two_segment(2, &rows, 1);
        assert_eq!(s.vectors().as_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn append_returns_global_indices() {
        let mut s = Segments::in_memory_with(2, Distance::Cosine);
        assert_eq!(s.append(&[1.0, 0.0]).unwrap(), 0);
        assert!(s.seal().unwrap());
        assert_eq!(
            s.append(&[0.0, 1.0]).unwrap(),
            1,
            "global index continues past seal"
        );
        assert_eq!(s.append(&[1.0, 1.0]).unwrap(), 2);
        assert_eq!(s.row_count(), 3);
    }

    #[test]
    fn seal_is_noop_when_active_empty() {
        let mut s = Segments::in_memory_with(2, Distance::Cosine);
        s.append(&[1.0, 0.0]).unwrap();
        assert!(s.seal().unwrap());
        // Active segment is now empty — a second seal must not create a dangling segment.
        assert!(!s.seal().unwrap());
        assert_eq!(s.segment_count(), 2);
    }

    #[test]
    fn truncate_rolls_back_active_segment() {
        let mut s = Segments::in_memory_with(2, Distance::Cosine);
        s.append(&[1.0, 0.0]).unwrap();
        assert!(s.seal().unwrap());
        s.append(&[0.0, 1.0]).unwrap();
        s.append(&[1.0, 1.0]).unwrap();
        s.truncate_to(2).unwrap(); // drop the last active row, keep across the seal
        assert_eq!(s.row_count(), 2);
        assert_eq!(s.row(0), &[1.0, 0.0]);
        assert_eq!(s.row(1), &[0.0, 1.0]);
    }

    #[test]
    fn truncate_below_active_base_errors() {
        let mut s = Segments::in_memory_with(2, Distance::Cosine);
        s.append(&[1.0, 0.0]).unwrap();
        s.seal().unwrap();
        s.append(&[0.0, 1.0]).unwrap();
        // Row 0 is in the sealed segment — rolling back into it is rejected.
        assert!(s.truncate_to(0).is_err());
    }

    #[test]
    fn rewrite_collapses_to_single_segment() {
        let rows = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut s = two_segment(2, &rows, 1);
        let dropped = s.rewrite(&[9.0, 8.0, 7.0, 6.0]).unwrap();
        assert_eq!(dropped.len(), 1, "the sealed seg-* became unreferenced");
        assert_eq!(s.segment_count(), 1);
        assert_eq!(s.row_count(), 2);
        assert_eq!(s.row(0), &[9.0, 8.0]);
        assert_eq!(s.row(1), &[7.0, 6.0]);
        // After collapse, appends continue on the single base segment.
        assert_eq!(s.append(&[1.0, 1.0]).unwrap(), 2);
    }

    #[test]
    fn segment_ranges_track_bases_and_counts() {
        let rows = vec![
            vec![1.0_f32, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 3.0],
        ];
        let s = two_segment(2, &rows, 2);
        // Two segments: [0,2) sealed, [2,4) active. Last entry is the active segment.
        assert_eq!(s.segment_ranges(), vec![(0, 2), (2, 2)]);
    }

    #[test]
    fn manifest_reflects_segment_set() {
        let rows = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        let s = two_segment(2, &rows, 1);
        let m = s.manifest();
        assert_eq!(
            m.segments,
            vec!["data".to_string(), "seg-00000001".to_string()]
        );
        assert_eq!(m.next_id, 2);
        assert!(m.version >= 2, "version advanced on seal");
    }
}
