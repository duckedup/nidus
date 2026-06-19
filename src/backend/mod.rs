//! Pluggable storage & memory backends (SPEC.md §13).
//!
//! nidus generalizes "a local directory, vectors in local RAM" along **two
//! independent, composable axes**, each behind a small **sync, `dyn`-safe** trait:
//!
//! - [`Persistence`] — where the durable *source-of-truth* bytes (`data`/`log`) and
//!   the derived caches (`ann`/`fts`) live. [`LocalFs`] (default), plus [`S3`] and
//!   [`Gcs`] object stores. Object-granular: whole-object get/put/delete/list, plus an
//!   **optional** native [`Appender`] capability (local files have it; object stores
//!   return `None` and rewrite whole objects) and a best-effort [`try_lock`].
//! - [`MemoryTier`] — where the in-RAM working set is held so it can be *shared* and
//!   *reloaded without a rebuild*. [`LocalRam`] (default — the process heap) and a
//!   [`RedisTier`] over the RESP family (Redis/Valkey/KeyDB/DragonflyDB).
//!
//! Both are **sync deliberately** (SPEC §13.4): search is CPU-over-RAM and never
//! touches a backend, every backend *can* be sync, and a sync trait is `dyn`-safe out
//! of the box — genuine runtime plug-and-play. Selection is by **URL scheme**
//! ([`open_persistence`] / [`open_memory_tier`]).
//!
//! The live store runs over these traits: its `data`/`log` segments are
//! [`Appender`]s the [`Persistence`] backend hands out, and its `ann`/`fts` caches and
//! writer lock go through `get`/`put`/`try_lock`. Routing snapshot/backup through
//! [`Persistence`] lands alongside the remote backends, where it first becomes meaningful.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};

mod cloud;
mod gcs;
mod local;
mod object;
mod ram;
mod redis;
mod s3;

#[cfg(test)]
mod tests;

pub use gcs::Gcs;
pub use local::{FileAppender, LocalFs};
pub use object::ObjectAppender;
pub(crate) use object::{ClusterLease, locked_error, object_try_lock};
pub use ram::LocalRam;
pub(crate) use ram::MemAppender;
pub use redis::RedisTier;
pub use s3::S3;

/// Where the durable bytes live: whole **named byte objects** in two classes —
/// source-of-truth (`data`/`log`, never reconstructable) and derived caches
/// (`ann`/`fts`, droppable). The common denominator of local files / S3 / GCS.
///
/// `key` is a single flat object name (e.g. `"data"`, `"ann"`); it must not contain
/// path separators or `..` (implementations reject those).
pub trait Persistence: Send + Sync {
    /// Fetch a whole object. `Ok(None)` when it does not exist (not an error).
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Write a whole object atomically (a reader sees either the old bytes or the
    /// new ones, never a torn mix).
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()>;

    /// Remove an object. Removing an absent object is a no-op, not an error.
    fn delete(&self, key: &str) -> Result<()>;

    /// List the object keys present.
    fn list(&self) -> Result<Vec<String>>;

    /// A native append handle for `key`, if this backend supports in-place appends
    /// (local files do — the `data`/`log` discipline of §6). Object stores return
    /// `Ok(None)`; their callers rewrite whole objects via [`put`](Self::put)
    /// instead. `Err` is a real IO failure opening the handle.
    fn appender(&self, key: &str) -> Result<Option<Box<dyn Appender>>> {
        let _ = key;
        Ok(None)
    }

    /// Atomically create object `key` with `bytes` **only if it does not already
    /// exist** — the create-if-absent primitive a race-free object writer lock needs
    /// (S3 `If-None-Match: *`, GCS `ifGenerationMatch=0`). Returns `Ok(Some(true))`
    /// when this call created it, `Ok(Some(false))` when it already existed (lost the
    /// race — **not** an error), and `Ok(None)` when the backend offers no atomic
    /// create-if-absent (the object-lock caller then falls back to the best-effort
    /// advisory put). `Err` is a real IO failure.
    ///
    /// This is the `expected: None` case of [`put_cas`](Self::put_cas) — the default impl
    /// delegates there, so a backend gets create-if-absent for free once it implements
    /// `put_cas`. (Kept as its own method because the writer-lock paths read more clearly
    /// in create-if-absent terms.)
    fn try_create_exclusive(&self, key: &str, bytes: &[u8]) -> Result<Option<bool>> {
        match self.put_cas(key, bytes, None)? {
            CasOutcome::Written(_) => Ok(Some(true)),
            CasOutcome::Stale => Ok(Some(false)),
            CasOutcome::Unsupported => Ok(None),
        }
    }

    /// Read an object together with an opaque **CAS token** (an S3 `ETag` / GCS generation)
    /// identifying its current version, for a later conditional [`put_cas`](Self::put_cas).
    /// `Ok(None)` when the object is absent. The token is `None` when the backend supports no
    /// compare-and-swap — the default impl returns [`get`](Self::get)'s bytes with no token,
    /// so a caller that finds `None` knows to fall back to a plain [`put`](Self::put).
    fn get_cas(&self, key: &str) -> Result<Option<(Vec<u8>, Option<String>)>> {
        Ok(self.get(key)?.map(|bytes| (bytes, None)))
    }

    /// Atomically write `bytes` to `key` **only if** its current CAS token equals
    /// `expected` (`Some(token)`), or **only if it does not exist** (`None`) — the
    /// compare-and-swap that fences a superseded cluster writer (SPEC §14.6): a stale
    /// writer's token no longer matches, so its write is refused rather than clobbering a
    /// peer's committed state (S3 `If-Match`/`If-None-Match`, GCS `ifGenerationMatch`). See
    /// [`CasOutcome`]. Default: [`Unsupported`](CasOutcome::Unsupported) — the caller falls
    /// back to a plain [`put`](Self::put).
    fn put_cas(&self, key: &str, bytes: &[u8], expected: Option<&str>) -> Result<CasOutcome> {
        let _ = (key, bytes, expected);
        Ok(CasOutcome::Unsupported)
    }

    /// Best-effort exclusive lock on `key` (the writer-exclusion primitive, §6.3).
    /// `Ok(Some(guard))` on success — the lock releases when the guard drops;
    /// `Ok(None)` when another holder has it (contention is **not** an error);
    /// `Err` only on a real IO failure. `ttl` reclaims a lock older than it (a
    /// crashed holder).
    fn try_lock(&self, key: &str, ttl: Duration) -> Result<Option<Box<dyn BackendLock>>>;

    /// The filesystem path of object `key`, when this backend stores it as a plain local
    /// file that can be memory-mapped (SPEC §9 / §14.6 phase 3). `None` for object-store and
    /// in-RAM backends — those cannot be mapped, so a caller with `Config::mmap` falls back to
    /// loading the segment into RAM. The default is `None`; [`LocalFs`] overrides it.
    fn local_path(&self, key: &str) -> Option<PathBuf> {
        let _ = key;
        None
    }

    /// Whether [`try_lock`](Self::try_lock) provides a real exclusive lock (local
    /// `O_EXCL`). Whole-object stores return `false`: a live object-backed store goes
    /// through the [`object_try_lock`] path instead — race-free where the backend
    /// implements [`try_create_exclusive`](Self::try_create_exclusive) (S3/GCS),
    /// advisory otherwise. Default `true`.
    fn has_native_lock(&self) -> bool {
        true
    }
}

/// A durable, append-shaped byte stream — the native local-FS capability that the
/// `data`/`log` segments need (§5–§6): random-access read (to load/replay on open),
/// append with per-write rollback, truncate to a byte boundary, fsync, and atomic
/// whole-file rewrite (compaction). Object-store backends do not provide this (see
/// [`Persistence::appender`]).
///
/// `Send + Sync` because a [`Store`](crate::Nidus) holding an appender is shared as
/// `Arc<RwLock<Nidus>>`: searchers take `&self` and never touch the appender (it is a
/// `&mut self`-only, write-path resource), so sharing `&dyn Appender` across threads is
/// sound — both concrete impls (a `File`, an in-RAM `Vec<u8>`) are themselves `Sync`.
pub trait Appender: Send + Sync {
    /// The current committed length in bytes — the append point.
    fn len(&self) -> Result<u64>;

    /// Whether the stream is currently empty (no bytes appended).
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Read exactly `buf.len()` bytes starting at byte `offset` into `buf`. Errors if
    /// fewer than that many bytes remain. The load/replay primitive — lets a caller
    /// stream a large segment in bounded chunks (no whole-object materialization, so
    /// `data` keeps its low transient open-time footprint).
    fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()>;

    /// Append `bytes`. **Atomic:** on a partial write (e.g. ENOSPC) the stream is
    /// rolled back to the length it had before the call, so no torn suffix persists
    /// for the next append to write past.
    fn append(&mut self, bytes: &[u8]) -> Result<()>;

    /// Truncate to exactly `offset` bytes, discarding any suffix. The batch-rollback
    /// counterpart to capturing [`len`](Self::len) before a batch.
    fn truncate_to(&mut self, offset: u64) -> Result<()>;

    /// Make all appended bytes durable (fsync).
    fn sync(&mut self) -> Result<()>;

    /// Atomically replace the entire contents with `bytes` (temp + fsync + rename),
    /// then leave the handle positioned to append after them. The compaction path.
    fn rewrite(&mut self, bytes: &[u8]) -> Result<()>;

    /// Append the entire current contents to `out`. Provided over
    /// [`read_exact_at`](Self::read_exact_at) with a fallible reserve, so an oversized
    /// stream surfaces an `Err` instead of aborting the process.
    fn read_to_end(&mut self, out: &mut Vec<u8>) -> Result<()> {
        let len = self.len()? as usize;
        let start = out.len();
        out.try_reserve_exact(len)
            .map_err(|_| anyhow::anyhow!("out of memory reading {len} bytes from appender"))?;
        out.resize(start + len, 0);
        self.read_exact_at(0, &mut out[start..])
    }
}

/// The outcome of a compare-and-swap object write ([`Persistence::put_cas`]).
pub enum CasOutcome {
    /// The write committed. Carries the object's **new** CAS token when the backend reports
    /// one (S3 returns it as the response `ETag`); `None` when the backend has no cheap way
    /// to report it, in which case the caller re-reads via [`Persistence::get_cas`] to learn
    /// the new token before its next conditional write.
    Written(Option<String>),
    /// The precondition failed: the object's current token differs from `expected` (a
    /// concurrent writer changed it since), or — for `expected: None` — the object already
    /// exists. **Not an error**: the caller treats it as "lost the race / I am fenced".
    Stale,
    /// This backend offers no compare-and-swap; the caller falls back to a plain
    /// [`put`](Persistence::put). (The default [`put_cas`](Persistence::put_cas) returns this.)
    Unsupported,
}

/// A held backend lock, released on drop (RAII). Returned by
/// [`Persistence::try_lock`]; the concrete guard owns whatever the backend needs to
/// release (a lock file, a conditional-PUT marker, …). `Send + Sync` for the same
/// reason as [`Appender`] — a held lock lives inside the shared [`Store`](crate::Nidus).
pub trait BackendLock: Send + Sync {}

/// Where the in-RAM working set is held so it can be **shared across processes** and
/// **reloaded without a rebuild** (SPEC §13.3). A rebuildable cache of the serialized
/// working state — model (a): an empty or evicted tier is never fatal (the
/// persistence tier is the truth). [`LocalRam`] is the trivial impl.
pub trait MemoryTier: Send + Sync {
    /// Pull the shared working-set blob for `key`. `Ok(None)` when absent/evicted.
    fn load(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Publish the shared working-set blob for `key`. `ttl`, when set, is an
    /// expiry hint a sharing tier may honour (local RAM ignores it — it never evicts).
    fn store(&self, key: &str, bytes: &[u8], ttl: Option<Duration>) -> Result<()>;
}

/// Share one tier behind several handles: `Arc<dyn MemoryTier>` (or `Arc<LocalRam>`) is
/// itself a [`MemoryTier`], so multiple stores can publish to / adopt from one instance.
impl<T: MemoryTier + ?Sized> MemoryTier for Arc<T> {
    fn load(&self, key: &str) -> Result<Option<Vec<u8>>> {
        (**self).load(key)
    }
    fn store(&self, key: &str, bytes: &[u8], ttl: Option<Duration>) -> Result<()> {
        (**self).store(key, bytes, ttl)
    }
}

/// The append handle for segment/`log` object `key`: the backend's native [`Appender`]
/// when it has one (local files — the `data`/`log` discipline of §6), else an
/// [`ObjectAppender`] (an in-RAM buffer rewritten as a whole object on sync) over the
/// shared backend handle (object stores). Shared by [`Store`](crate::Nidus)'s `data`/`log`
/// wiring and the [`Segments`](crate::data::Segments) aggregator so every durable byte
/// stream is opened identically.
///
/// Note: the object-store path loads the whole object into RAM here, *before* a caller's
/// `max_vector_bytes` check (§6.6) can run. So for object stores that "refuse before
/// allocating" guard relaxes to "refuse after one full copy is resident" — inherent to a
/// whole-object backend, and acceptable at nidus's dev/small-scale positioning.
/// `cas` selects the [`ObjectAppender`] commit discipline for an object-store segment:
/// `false` (the default) rewrites the whole object on every sync; `true` (cluster mode,
/// SPEC §14.6) makes each sync a **compare-and-swap** so a superseded writer's whole-object
/// rewrite is fenced rather than clobbering a peer's committed bytes. Native local-FS
/// appenders ignore it (cluster mode requires an object store, never a local file).
pub(crate) fn appender_for(
    persistence: &Arc<dyn Persistence>,
    key: &str,
    cas: bool,
) -> Result<Box<dyn Appender>> {
    match persistence.appender(key)? {
        Some(native) => Ok(native),
        None => Ok(Box::new(ObjectAppender::open(
            persistence.clone(),
            key,
            cas,
        )?)),
    }
}

/// Reject a key that is not a single flat object name — no path separators, no `..`,
/// not empty. Shared by every backend so keys behave identically across local and
/// (future) object stores.
pub(crate) fn validate_key(key: &str) -> Result<()> {
    if key.is_empty() {
        bail!("backend object key must not be empty");
    }
    if key.contains('/') || key.contains('\\') || key.split(['/', '\\']).any(|c| c == "..") {
        bail!("backend object key {key:?} must be a flat name (no path separators or `..`)");
    }
    Ok(())
}

/// Open a **persistence** backend from a URL/location string (SPEC §13.4):
///
/// - `file://<path>` or a bare `<path>` → [`LocalFs`] rooted at that directory.
/// - `s3://…` / `gs://…` (`gcs://…`) → a clear "not yet" error (planned, Phase 3).
///
/// The directory is created if absent.
pub fn open_persistence(location: &str) -> Result<Box<dyn Persistence>> {
    if let Some(rest) = strip_scheme(location, "s3") {
        return Ok(Box::new(S3::from_url(rest)?));
    }
    if let Some(rest) = strip_scheme(location, "gs").or_else(|| strip_scheme(location, "gcs")) {
        return Ok(Box::new(Gcs::from_url(rest)?));
    }
    // `file://<path>` or a bare path → local files.
    let path = strip_scheme(location, "file").unwrap_or(location);
    Ok(Box::new(LocalFs::new(path)?))
}

/// Open a **memory tier** backend from a URL/location string (SPEC §13.3):
///
/// - empty, `local`, or `ram` → [`LocalRam`] (the process heap; nothing shared).
/// - `redis://…` / `rediss://…` (TLS), and the RESP-compatible aliases `valkey://…`,
///   `valkeys://…`, `keydb://…`, `dragonfly://…` → a shared [`RedisTier`].
///
/// An unrecognized scheme is rejected with a clear error rather than a silent fallback.
pub fn open_memory_tier(location: &str) -> Result<Box<dyn MemoryTier>> {
    match location {
        "" | "local" | "ram" => return Ok(Box::new(LocalRam::new())),
        _ => {}
    }
    for scheme in REDIS_SCHEMES {
        if strip_scheme(location, scheme).is_some() {
            return Ok(Box::new(RedisTier::from_url(location)?));
        }
    }
    bail!(
        "unknown memory-tier location {location:?} \
         (expected `local`, or a Redis-family URL like `redis://…` / `valkey://…`)"
    )
}

/// The RESP-protocol URL schemes [`open_memory_tier`] routes to [`RedisTier`] — Redis
/// and its wire-compatible kin (Valkey, KeyDB, DragonflyDB), plain and TLS.
pub(crate) const REDIS_SCHEMES: [&str; 6] =
    ["redis", "rediss", "valkey", "valkeys", "keydb", "dragonfly"];

/// Open the persistence backend holding a single named **object** addressed by
/// `location` — splitting it into a backend root and an object key at the last `/`
/// (a bare name, no `/`, roots at the current directory). Used for snapshots, whose
/// destination/source is one archive object on any backend: `./snap.tar.gz`,
/// `file:///backups/snap.tar.gz`, or (once it lands) `s3://bucket/snap.tar.gz`.
pub fn open_object_location(location: &str) -> Result<(Box<dyn Persistence>, String)> {
    let (root, key) = split_object_location(location)?;
    Ok((open_persistence(root)?, key.to_string()))
}

/// Split a location into `(root_location, object_key)` at the last `/`. Pure string
/// logic (no IO), so it is unit-tested directly.
fn split_object_location(location: &str) -> Result<(&str, &str)> {
    match location.rsplit_once('/') {
        Some((_, "")) => bail!("location {location:?} ends in '/' — it has no object name"),
        // Last '/' is the root's trailing slash (e.g. `file:///x` → root `/`).
        Some((root, key)) => Ok((if root.is_empty() { "/" } else { root }, key)),
        // No '/' at all → a bare object name in the current directory.
        None => Ok((".", location)),
    }
}

/// If `s` begins with `<scheme>://`, return the remainder; else `None`.
fn strip_scheme<'a>(s: &'a str, scheme: &str) -> Option<&'a str> {
    let prefix_len = scheme.len() + 3; // "://"
    if s.len() >= prefix_len
        && s.is_char_boundary(prefix_len)
        && s[..prefix_len].eq_ignore_ascii_case(&format!("{scheme}://"))
    {
        Some(&s[prefix_len..])
    } else {
        None
    }
}
