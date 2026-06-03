//! Store configuration (SPEC.md §4.1). The store location is always the caller's
//! choice — nidus contributes no path defaults of its own.

use std::path::PathBuf;
use std::time::Duration;

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
    /// Durability granularity. Default [`Fsync::PerBatch`].
    pub fsync: Fsync,
    /// Read/write vs read-only. Default [`OpenMode::ReadWrite`].
    pub open_mode: OpenMode,
    /// Dead-row ratio that triggers compaction on open; `None` disables.
    /// Default `Some(0.5)`.
    pub auto_compact: Option<f32>,
    /// Stale writer-lock reclamation window. Default 60s.
    pub lock_ttl: Duration,
}

impl Config {
    /// A config with required fields set and everything else defaulted.
    pub fn new(path: impl Into<PathBuf>, dimension: usize) -> Self {
        Self {
            path: path.into(),
            dimension,
            fsync: Fsync::PerBatch,
            open_mode: OpenMode::ReadWrite,
            auto_compact: Some(0.5),
            lock_ttl: Duration::from_secs(60),
        }
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
