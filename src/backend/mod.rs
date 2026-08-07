//! Pluggable storage & memory backends (SPEC.md §13).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};

mod aws_creds;
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
pub(crate) use object::{latch_fenced, locked_error, object_try_lock};
// Public: a cluster writer's lease handle is part of the API surface, so an async host
// (`nidus serve`, or an embedding application) can keep the lease warm on a timer while a
// long write holds the store lock — see `Nidus::lease_handle`.
pub use object::{ClusterLease, LeaseLost, LeaseRenewer, is_lease_lost};
pub use ram::LocalRam;
pub(crate) use ram::MemAppender;
pub use redis::RedisTier;
pub use s3::S3;

/// Where the durable bytes live: whole **named byte objects** in two classes —
/// source-of-truth (`data`/`log`, never reconstructable) and derived caches
/// (`ann`/`fts`, droppable). The common denominator of local files / S3 / GCS.
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

    /// A native append handle for `key`, if this backend supports in-place appends (local files do
    /// — the `data`/`log` discipline of §6). Object stores return `Ok(None)`; their callers rewrite
    /// whole objects via [`put`](Self::put) instead. `Err` is a real IO failure opening the handle.
    fn appender(&self, key: &str) -> Result<Option<Box<dyn Appender>>> {
        let _ = key;
        Ok(None)
    }

    /// Atomically create `key` only if absent — the primitive a race-free object writer lock
    /// needs. `Some(true)` = this call created it, `Some(false)` = it existed (lost the race, not
    /// an error), `None` = the backend has no atomic create-if-absent. `Err` is real IO failure.
    fn try_create_exclusive(&self, key: &str, bytes: &[u8]) -> Result<Option<bool>> {
        match self.put_cas(key, bytes, None)? {
            CasOutcome::Written(_) => Ok(Some(true)),
            CasOutcome::Stale => Ok(Some(false)),
            CasOutcome::Unsupported => Ok(None),
        }
    }

    /// Read an object together with an opaque **CAS token** (an S3 `ETag` / GCS generation)
    /// identifying its current version, for a later conditional [`put_cas`](Self::put_cas).
    fn get_cas(&self, key: &str) -> Result<Option<(Vec<u8>, Option<String>)>> {
        Ok(self.get(key)?.map(|bytes| (bytes, None)))
    }

    /// Write `bytes` to `key` only if its CAS token equals `expected`, or only if absent when
    /// `None` — the compare-and-swap that fences a superseded cluster writer (SPEC §14.6). See
    /// [`CasOutcome`]; the default is `Unsupported`, and the caller falls back to a plain `put`.
    fn put_cas(&self, key: &str, bytes: &[u8], expected: Option<&str>) -> Result<CasOutcome> {
        let _ = (key, bytes, expected);
        Ok(CasOutcome::Unsupported)
    }

    /// Best-effort exclusive lock on `key` (the writer-exclusion primitive, §6.3).
    fn try_lock(&self, key: &str, ttl: Duration) -> Result<Option<Box<dyn BackendLock>>>;

    /// The filesystem path of `key` when this backend stores it as a mappable local file (SPEC §9
    /// / §14.6 phase 3). `None` for object-store and in-RAM backends, so a caller with
    /// `Config::mmap` falls back to loading into RAM. [`LocalFs`] overrides the `None` default.
    fn local_path(&self, key: &str) -> Option<PathBuf> {
        let _ = key;
        None
    }

    /// Whether [`try_lock`](Self::try_lock) is a real exclusive lock (local `O_EXCL`). Whole-object
    /// stores return `false` and go through the object-lock path instead — race-free where the
    /// backend implements `try_create_exclusive`, advisory otherwise. Default `true`.
    fn has_native_lock(&self) -> bool {
        true
    }

    /// Whether this backend implements a real compare-and-swap
    /// ([`put_cas`](Self::put_cas) / [`get_cas`](Self::get_cas)).
    fn supports_cas(&self) -> bool {
        false
    }
}

/// A durable, append-shaped byte stream — the native local-FS capability `data`/`log` need
/// (§5–§6): random-access read, append with per-write rollback, truncate to a boundary, fsync, and
/// atomic whole-file rewrite. Object-store backends do not provide this.
pub trait Appender: Send + Sync {
    /// The current committed length in bytes — the append point.
    fn len(&self) -> Result<u64>;

    /// Whether the stream is currently empty (no bytes appended).
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Read exactly `buf.len()` bytes from `offset`, erroring if fewer remain. The load/replay
    /// primitive: a caller streams a large segment in bounded chunks, so `data` keeps its low
    /// transient open-time footprint.
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
    /// The write committed, carrying the object's new CAS token when the backend reports one.
    /// `None` when it has no cheap way to, in which case the caller re-reads via
    /// [`Persistence::get_cas`] before its next conditional write.
    Written(Option<String>),
    /// The precondition failed: the object's current token differs from `expected` (a
    /// concurrent writer changed it since), or — for `expected: None` — the object already
    /// exists. **Not an error**: the caller treats it as "lost the race / I am fenced".
    Stale,
    /// This backend offers no compare-and-swap; the caller falls back to a plain
    /// [`put`](Persistence::put). (The default [`put_cas`](Persistence::put_cas) returns this.)
    Unsupported,
}

/// A held backend lock, released on drop. The concrete guard owns whatever the backend needs to
/// release — a lock file, a conditional-PUT marker. `Send + Sync` for the same reason as
/// [`Appender`]: a held lock lives inside the shared [`Store`](crate::Nidus).
pub trait BackendLock: Send + Sync {}

/// Where the in-RAM working set is held so it can be shared across processes and reloaded without
/// a rebuild (SPEC §13.3). A rebuildable cache: an empty or evicted tier is never fatal, since the
/// persistence tier is the truth. [`LocalRam`] is the trivial impl.
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

/// The append handle for segment/`log` object `key`: the backend's native [`Appender`] for local
/// files, else an `ObjectAppender` (an in-RAM buffer rewritten whole on sync). Shared by the
/// `data`/`log` wiring and [`Segments`](crate::data::Segments) so every stream opens identically.
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

/// Open the persistence backend holding the single object at `location`, split into a backend root
/// and an object key at the last `/` (a bare name roots at the current directory). For snapshots,
/// whose source/destination is one archive object on any backend.
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
