//! Store configuration (SPEC.md §4.1). The store location is always the caller's
//! choice — nidus contributes no path defaults of its own.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail};

use crate::model::{AnnConfig, Distance, QuantKind, Quantization};
use crate::profile::OpenProfile;

/// What a [`OpenMode::ReadWrite`] open does when another instance already holds the
/// writer handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LeaseWait {
    /// Fail immediately on contention. The default.
    #[default]
    Fail,
    /// Retry until acquired, or until this long has elapsed and then fail. For a
    /// script or one-shot command that should not hang forever.
    Timeout(Duration),
    /// Retry indefinitely — a standby whose whole job is to wait for promotion.
    Forever,
}

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
    /// What to do when another instance holds the writer handle. Default
    /// [`LeaseWait::Fail`] — see [`LeaseWait`] for why waiting is what a standby needs.
    pub lease_wait: LeaseWait,
    /// How stale a read-only instance may be before it should stop serving. `None` (the
    /// default) means no bound — the historical behaviour, where a reader serves its
    /// snapshot for as long as it likes.
    pub max_staleness: Option<Duration>,
    /// Hard ceiling on the vector matrix (`rows * dimension * 4` bytes); `None` disables.
    /// Enforced *before* allocating, so it is the only exhaustion guard that holds under memory
    /// overcommit, where the kernel SIGKILLs before `try_reserve` ever fires.
    pub max_vector_bytes: Option<u64>,
    /// Vector quantization for faster search; `None` disables. Keeps an in-memory quantized matrix
    /// and a two-pass search: a cheap first pass selects, an f32 rerank restores exact scores.
    /// [`int8`](Quantization::int8) is 4× and any metric, [`binary`](Quantization::binary) 32× and cosine only.
    pub quantization: Option<Quantization>,
    /// Approximate-nearest-neighbour index; `None` leaves exact brute force. When set, `search` walks
    /// an in-RAM HNSW or IVF index for an over-fetched candidate set, then applies scope/filter and an
    /// exact f32 rerank. Composes with [`Config::quantization`], which makes the walk cheaper.
    pub ann: Option<AnnConfig>,
    /// Worker threads for a single search; default `1`. Above that, a large scan splits across scoped
    /// workers each with its own bounded heap. The f32 scan is bandwidth-bound and the int8 first pass
    /// compute-bound; leave at `1` when you already have query-level concurrency.
    pub query_threads: usize,
    /// Where the durable bytes live (SPEC §13.2): empty or `file://`/bare path → local files under
    /// [`path`](Self::path); `s3://…`/`gs://…` → a live object-backed store, rewriting whole
    /// objects on flush. Empty defaults to `path`.
    pub persistence: String,
    /// Roll the active segment into an immutable one once it reaches this many rows
    /// (SPEC §14.2/§14.4 — "WAL→segment"). `None` (the default) never auto-seals, so a
    /// store stays a single segment and behaves exactly as the pre-segment monolith did.
    pub segment_max_rows: Option<u64>,
    /// IVF-index each immutable segment holding at least this many rows (SPEC §14.3); `None`
    /// brute-forces everything, so exact-vs-approximate follows segment size. Needs
    /// [`segment_max_rows`], ignored under [`ann`](Self::ann), rejected with [`quantization`](Self::quantization).
    pub segment_index_min_rows: Option<u64>,
    /// Memory-map sealed segments rather than loading them into RAM (SPEC §9 / §14.6 phase 3) — what
    /// lets a store outgrow one node's RAM. The active segment stays in RAM. Local-FS only, needs
    /// [`segment_max_rows`](Self::segment_max_rows), and results are identical either way.
    pub mmap: bool,
    /// Where the in-RAM working set is shared (SPEC §13.3). Empty/`local`/`ram` → the process heap
    /// only; a `redis://…` URL publishes the working set on `flush` and loads it on `open`, so other
    /// workers skip replay and rebuild. A rebuildable cache, never fatal.
    pub memory: String,
    /// Cooperating instances over one shared backend (SPEC §14.6 phase 5): one `ReadWrite` writer
    /// holding a heartbeated lease and advancing the manifest per commit, plus any number of
    /// `ReadOnly` readers picking its writes up via `refresh`. Default `false`.
    pub cluster: bool,
    /// Refuse a memory recall against a collection carrying no pinned embedder identity
    /// (`nidus.embedder`) rather than warning about it — the un-pinned collection an
    /// external writer leaves behind cannot be checked for embedder agreement (nidus-8ki).
    pub strict_embedder_identity: bool,
    /// Open a point-in-time snapshot at this commit version instead of the current one
    /// (nidus-bnf); requires [`history_versions`](Self::history_versions) to have recorded
    /// it and [`OpenMode::ReadOnly`]. `None` (the default) opens current, as always.
    pub at_version: Option<u64>,
    /// Keep the last N commit points addressable by [`at_version`](Self::at_version);
    /// `None` (the default) records no history. Enabling it makes every durable batch a
    /// commit point — seals alone are far too rare to pin to — so the write path pays for it.
    pub history_versions: Option<usize>,
    /// Which of the profile-eligible knobs were set explicitly via a builder call, so
    /// [`Config::apply_profile`] knows an explicit setter must beat a recorded default.
    explicit: ExplicitFlags,
}

/// Tracks which profile-eligible [`Config`] knobs were set explicitly, independent of the
/// resolved value each field holds (a caller can explicitly set `ann(None)`, which must still
/// beat a recorded profile).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ExplicitFlags {
    pub(crate) quantization: bool,
    pub(crate) ann: bool,
    pub(crate) query_threads: bool,
    pub(crate) mmap: bool,
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
            lease_wait: LeaseWait::default(),
            max_staleness: None,
            max_vector_bytes: None,
            quantization: None,
            ann: None,
            query_threads: 1,
            segment_max_rows: None,
            segment_index_min_rows: None,
            mmap: false,
            persistence: String::new(),
            memory: String::new(),
            cluster: false,
            strict_embedder_identity: false,
            at_version: None,
            history_versions: None,
            explicit: ExplicitFlags::default(),
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

    /// Set how stale a read-only instance may be before its readiness probe should fail
    /// (`None` = no bound, the default). See [`max_staleness`](Self::max_staleness).
    pub fn max_staleness(mut self, max: Option<Duration>) -> Self {
        self.max_staleness = max;
        self
    }

    /// Set what a `ReadWrite` open does when another instance holds the writer handle
    /// (default [`LeaseWait::Fail`]). [`LeaseWait::Forever`] makes this instance a
    /// standby that waits for promotion instead of exiting.
    pub fn lease_wait(mut self, wait: LeaseWait) -> Self {
        self.lease_wait = wait;
        self
    }

    /// Set the vector-matrix size ceiling (`None` to disable).
    pub fn max_vector_bytes(mut self, bytes: Option<u64>) -> Self {
        self.max_vector_bytes = bytes;
        self
    }

    /// Enable vector quantization for faster search (int8 or binary; `None` disables).
    pub fn quantization(mut self, q: Option<Quantization>) -> Self {
        self.quantization = q;
        self.explicit.quantization = true;
        self
    }

    /// Enable approximate-nearest-neighbour search (HNSW or IVF; `None` disables —
    /// the default exact brute-force). May be combined with quantization for a quantized
    /// index walk plus an exact f32 rerank.
    pub fn ann(mut self, ann: Option<AnnConfig>) -> Self {
        self.ann = ann;
        self.explicit.ann = true;
        self
    }

    /// Set the number of worker threads for a single exact search (`1` = serial).
    pub fn query_threads(mut self, n: usize) -> Self {
        self.query_threads = n;
        self.explicit.query_threads = true;
        self
    }

    /// Set the active-segment seal threshold in rows (`None` = never seal, single segment).
    pub fn segment_max_rows(mut self, rows: Option<u64>) -> Self {
        self.segment_max_rows = rows;
        self
    }

    /// Set the minimum row count for a sealed segment to be IVF-indexed (`None` = never
    /// index, the exact brute-force default). See
    /// [`segment_index_min_rows`](Self::segment_index_min_rows).
    pub fn segment_index_min_rows(mut self, rows: Option<u64>) -> Self {
        self.segment_index_min_rows = rows;
        self
    }

    /// Enable memory-mapping of immutable segments (`false` = all-RAM, the default). See
    /// [`mmap`](Self::mmap).
    pub fn mmap(mut self, on: bool) -> Self {
        self.mmap = on;
        self.explicit.mmap = true;
        self
    }

    /// Set the persistence location (`s3://…` / `gs://…` for a live object-store-backed
    /// store; empty or a path → local files). See [`persistence`](Self::persistence).
    pub fn persistence(mut self, location: impl Into<String>) -> Self {
        self.persistence = location.into();
        self
    }

    /// Set the shared memory-tier location (`redis://…` / `valkey://…`; empty/`local` →
    /// the process heap only). See [`memory`](Self::memory).
    pub fn memory(mut self, location: impl Into<String>) -> Self {
        self.memory = location.into();
        self
    }

    /// Enable cooperating-instances cluster mode (requires shared object-store
    /// [`persistence`](Self::persistence) **and** a shared [`memory`](Self::memory) tier).
    /// See [`cluster`](Self::cluster).
    pub fn cluster(mut self, on: bool) -> Self {
        self.cluster = on;
        self
    }

    /// Refuse, rather than warn, when a memory recall names a collection with no pinned
    /// embedder identity. See [`strict_embedder_identity`](Self::strict_embedder_identity).
    pub fn strict_embedder_identity(mut self, on: bool) -> Self {
        self.strict_embedder_identity = on;
        self
    }

    /// Pin the open to a past commit version (`None` = current). See
    /// [`at_version`](Self::at_version); requires [`OpenMode::ReadOnly`].
    pub fn at_version(mut self, version: Option<u64>) -> Self {
        self.at_version = version;
        self
    }

    /// Record the last N commit points as addressable history (`None` disables). See
    /// [`history_versions`](Self::history_versions).
    pub fn history_versions(mut self, n: Option<usize>) -> Self {
        self.history_versions = n;
        self
    }

    /// Resolve recorded [`OpenProfile`] defaults against this config: an explicit builder call
    /// always wins, otherwise a recorded value fills in, otherwise the built-in default (already
    /// in place) stands. Applied per knob independently.
    pub(crate) fn apply_profile(&mut self, p: &OpenProfile) {
        if !self.explicit.quantization {
            self.quantization = p.quantization.or(self.quantization);
        }
        if !self.explicit.ann {
            self.ann = p.ann.or(self.ann);
        }
        if !self.explicit.query_threads {
            self.query_threads = p.query_threads.unwrap_or(self.query_threads);
        }
        if !self.explicit.mmap {
            self.mmap = p.mmap.unwrap_or(self.mmap);
        }
    }

    /// Build an [`OpenProfile`] capturing only the knobs this config set explicitly, ready to
    /// hand to `Nidus::set_open_profile`. Knobs left at their defaults stay unrecorded, so
    /// recording never freezes a value the caller never chose.
    pub fn to_profile(&self) -> OpenProfile {
        OpenProfile {
            ann: if self.explicit.ann { self.ann } else { None },
            quantization: if self.explicit.quantization {
                self.quantization
            } else {
                None
            },
            query_threads: if self.explicit.query_threads {
                Some(self.query_threads)
            } else {
                None
            },
            mmap: if self.explicit.mmap {
                Some(self.mmap)
            } else {
                None
            },
        }
    }

    /// Validate cross-field invariants that do not depend on the backend — called by every
    /// store constructor before any IO. (Backend-specific checks, e.g. cluster mode needing a
    /// shared store, live in `Store::open_with`, which has the resolved backend in hand.)
    pub(crate) fn validate(&self) -> Result<()> {
        // Quantization and per-segment indexing do not compose: a per-segment fan-out never
        // consults the quantized matrix, so quantization would cost full memory/CPU for no effect.
        // Rejected so the trade-off is explicit rather than a silent no-op (nidus-tku).
        if self.quantization.is_some()
            && self.segment_max_rows.is_some()
            && self.segment_index_min_rows.is_some()
        {
            bail!(
                "Config::quantization cannot be combined with per-segment indexing \
                 (segment_max_rows + segment_index_min_rows): once a segment is IVF-indexed, \
                 search fans out per-segment and never uses the quantized matrix, so \
                 quantization would cost memory and CPU with no effect — enable one or the other"
            );
        }
        if self.at_version.is_some() && self.open_mode == OpenMode::ReadWrite {
            bail!("a pinned read (at_version) is read-only: open with OpenMode::ReadOnly");
        }
        if self.history_versions == Some(0) {
            bail!(
                "Config::history_versions(Some(0)) has no spelling distinct from disabled — \
                 use None to turn history off"
            );
        }
        // Also enforced when the quantized matrix is built, but a recorded profile resolves at
        // *open*, so an unbuildable combination must be rejected before it can be persisted —
        // otherwise every later open fails, including the one needed to clear it (nidus-141).
        if matches!(self.quantization, Some(q) if q.kind == QuantKind::Binary)
            && self.distance != Distance::Cosine
        {
            bail!(
                "binary quantization requires Distance::Cosine (sign codes are an angular \
                 proxy and ignore magnitude); use int8 quantization for a dot-product or \
                 Euclidean store"
            );
        }
        Ok(())
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

    #[test]
    fn explicit_setters_are_tracked_independently() {
        let c = Config::new("/tmp/s", 8).mmap(true);
        assert!(c.explicit.mmap);
        assert!(!c.explicit.ann);
        assert!(!c.explicit.quantization);
        assert!(!c.explicit.query_threads);
    }

    #[test]
    fn apply_profile_fills_unset_knobs() {
        let mut c = Config::new("/tmp/s", 8);
        let p = OpenProfile {
            ann: Some(AnnConfig::hnsw()),
            quantization: Some(Quantization::int8()),
            query_threads: Some(4),
            mmap: Some(true),
        };
        c.apply_profile(&p);
        assert_eq!(c.ann, Some(AnnConfig::hnsw()));
        assert_eq!(c.quantization, Some(Quantization::int8()));
        assert_eq!(c.query_threads, 4);
        assert!(c.mmap);
    }

    #[test]
    fn apply_profile_never_overrides_an_explicit_setter() {
        let mut c = Config::new("/tmp/s", 8)
            .ann(None)
            .quantization(Some(Quantization::binary()))
            .query_threads(2)
            .mmap(false);
        let p = OpenProfile {
            ann: Some(AnnConfig::hnsw()),
            quantization: Some(Quantization::int8()),
            query_threads: Some(4),
            mmap: Some(true),
        };
        c.apply_profile(&p);
        assert_eq!(
            c.ann, None,
            "explicit ann(None) must beat a recorded profile"
        );
        assert_eq!(c.quantization, Some(Quantization::binary()));
        assert_eq!(c.query_threads, 2);
        assert!(!c.mmap);
    }

    #[test]
    fn to_profile_captures_only_explicit_knobs() {
        let c = Config::new("/tmp/s", 8).mmap(true).query_threads(3);
        let p = c.to_profile();
        assert_eq!(p.mmap, Some(true));
        assert_eq!(p.query_threads, Some(3));
        assert_eq!(p.ann, None);
        assert_eq!(p.quantization, None);
    }

    #[test]
    fn to_profile_round_trips_through_apply_profile() {
        let source = Config::new("/tmp/s", 8)
            .ann(Some(AnnConfig::ivf()))
            .quantization(Some(Quantization::binary()));
        let profile = source.to_profile();

        let mut target = Config::new("/tmp/other", 8);
        target.apply_profile(&profile);
        assert_eq!(target.ann, Some(AnnConfig::ivf()));
        assert_eq!(target.quantization, Some(Quantization::binary()));
        // Knobs the source never set explicitly stay untouched on the target.
        assert_eq!(target.query_threads, 1);
        assert!(!target.mmap);
    }

    #[test]
    fn quantization_and_per_segment_indexing_are_mutually_exclusive() {
        // All three set → per-segment indexing activates and shadows quantization → rejected.
        let bad = Config::new("/tmp/s", 8)
            .quantization(Some(Quantization::default()))
            .segment_max_rows(Some(1000))
            .segment_index_min_rows(Some(500));
        let err = bad
            .validate()
            .expect_err("the combination must be rejected")
            .to_string();
        assert!(err.contains("quantization"), "{err}");
        assert!(err.contains("per-segment indexing"), "{err}");
    }

    #[test]
    fn at_version_requires_read_only() {
        let bad = Config::new("/tmp/s", 8).at_version(Some(3));
        assert!(bad.validate().is_err());
        let ok = Config::new("/tmp/s", 8)
            .at_version(Some(3))
            .open_mode(OpenMode::ReadOnly);
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn history_versions_zero_is_rejected() {
        let bad = Config::new("/tmp/s", 8).history_versions(Some(0));
        assert!(bad.validate().is_err());
        let ok = Config::new("/tmp/s", 8).history_versions(Some(2));
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn quantization_with_sealing_but_no_indexing_is_allowed() {
        // Sealing alone keeps quantization fully in effect (the matrix spans every segment).
        let ok = Config::new("/tmp/s", 8)
            .quantization(Some(Quantization::default()))
            .segment_max_rows(Some(1000));
        assert!(ok.validate().is_ok());

        // segment_index_min_rows without segment_max_rows is itself a no-op (nothing seals to
        // index), so it does not shadow quantization either.
        let ok2 = Config::new("/tmp/s", 8)
            .quantization(Some(Quantization::default()))
            .segment_index_min_rows(Some(500));
        assert!(ok2.validate().is_ok());
    }
}
