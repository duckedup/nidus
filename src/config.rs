//! Store configuration (SPEC.md §4.1). The store location is always the caller's
//! choice — nidus contributes no path defaults of its own.

use std::path::PathBuf;
use std::time::Duration;

use crate::model::{Distance, Quantization};

/// How aggressively writes are flushed to disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fsync {
    /// fsync after every `upsert`/`delete` call (durable per batch). Default.
    PerBatch,
    /// fsync only on explicit `flush()`/close (faster, weaker durability).
    OnFlush,
}

/// Whether the store may be written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenMode {
    /// Takes the writer lock; mutations allowed.
    ReadWrite,
    /// No writer lock taken; mutations rejected. For search-only processes and
    /// the future search server (SPEC.md §9).
    ReadOnly,
}

/// Everything needed to open a store. Construct with [`Config::new`] and adjust
/// via the builder setters.
#[derive(Clone, Debug)]
pub struct Config {
    /// The store directory (REQUIRED). Created if absent.
    pub path: PathBuf,
    /// The pinned embedding dimension (REQUIRED). Must match the on-disk header.
    pub dimension: usize,
    /// The similarity / distance metric. Pinned at store creation; must match the
    /// on-disk header on reopen. Default [`Distance::Cosine`].
    pub distance: Distance,
    /// Durability granularity. Default [`Fsync::PerBatch`].
    pub fsync: Fsync,
    /// Read/write vs read-only. Default [`OpenMode::ReadWrite`].
    pub open_mode: OpenMode,
    /// Dead-row ratio that triggers compaction on open; `None` disables.
    /// Default `Some(0.5)`.
    pub auto_compact: Option<f32>,
    /// Stale writer-lock reclamation window. Default 60s.
    pub lock_ttl: Duration,
    /// Hard ceiling on the vector matrix (`rows * dimension * 4` bytes); `None`
    /// disables (the default — no behavior change). Enforced *before* allocating:
    /// `upsert` refuses a batch that would exceed it, and `open` refuses a data
    /// file already over it. This is the only exhaustion guard that holds under
    /// memory overcommit, where the kernel SIGKILLs before an allocation fails and
    /// `try_reserve` never fires. Counts physical rows incl. not-yet-compacted dead
    /// rows, so `compact` can reclaim headroom.
    pub max_vector_bytes: Option<u64>,
    /// int8 scalar quantization for faster search. `None` disables (the default).
    /// When enabled, the store maintains an in-memory int8 vector matrix and uses
    /// a two-pass search: int8 first-pass → f32 rerank.
    pub quantization: Option<Quantization>,
    /// Worker threads for a single search. Default `1` (single-threaded, no behavior
    /// change). When `> 1`, a large scan is split across this many `std::thread::scope`
    /// workers, each with its own bounded heap, merged at the end — both the exact f32
    /// scan and (when quantization is on) the int8 first pass. The f32 scan is
    /// bandwidth-bound (sublinear gain); the int8 first pass is compute-bound and scales
    /// better. Parallelizes *one* query across cores — leave it at `1` when you already
    /// have query-level concurrency (many readers under `Arc<RwLock<Nidus>>`).
    pub query_threads: usize,
}

impl Config {
    /// A config with required fields set and everything else defaulted.
    pub fn new(path: impl Into<PathBuf>, dimension: usize) -> Self {
        Self {
            path: path.into(),
            dimension,
            distance: Distance::default(),
            fsync: Fsync::PerBatch,
            open_mode: OpenMode::ReadWrite,
            auto_compact: Some(0.5),
            lock_ttl: Duration::from_secs(60),
            max_vector_bytes: None,
            quantization: None,
            query_threads: 1,
        }
    }

    /// Set the distance metric.
    pub fn distance(mut self, d: Distance) -> Self {
        self.distance = d;
        self
    }

    /// Set the fsync policy.
    pub fn fsync(mut self, f: Fsync) -> Self {
        self.fsync = f;
        self
    }

    /// Set read/write vs read-only.
    pub fn open_mode(mut self, m: OpenMode) -> Self {
        self.open_mode = m;
        self
    }

    /// Set the auto-compaction dead-row ratio (`None` to disable).
    pub fn auto_compact(mut self, ratio: Option<f32>) -> Self {
        self.auto_compact = ratio;
        self
    }

    /// Set the stale-lock reclamation window.
    pub fn lock_ttl(mut self, ttl: Duration) -> Self {
        self.lock_ttl = ttl;
        self
    }

    /// Set the vector-matrix size ceiling (`None` to disable).
    pub fn max_vector_bytes(mut self, bytes: Option<u64>) -> Self {
        self.max_vector_bytes = bytes;
        self
    }

    /// Enable int8 scalar quantization for faster search.
    pub fn quantization(mut self, q: Option<Quantization>) -> Self {
        self.quantization = q;
        self
    }

    /// Set the number of worker threads for a single exact search (`1` = serial).
    pub fn query_threads(mut self, n: usize) -> Self {
        self.query_threads = n;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let c = Config::new("/tmp/store", 1024);
        assert_eq!(c.dimension, 1024);
        assert_eq!(c.fsync, Fsync::PerBatch);
        assert_eq!(c.open_mode, OpenMode::ReadWrite);
        assert_eq!(c.auto_compact, Some(0.5));
        assert_eq!(c.lock_ttl, Duration::from_secs(60));
    }

    #[test]
    fn builder_overrides() {
        let c = Config::new("/tmp/store", 8)
            .fsync(Fsync::OnFlush)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .lock_ttl(Duration::from_secs(5));
        assert_eq!(c.fsync, Fsync::OnFlush);
        assert_eq!(c.open_mode, OpenMode::ReadOnly);
        assert_eq!(c.auto_compact, None);
        assert_eq!(c.lock_ttl, Duration::from_secs(5));
    }
}
