//! Pluggable storage & memory backends (SPEC.md §13).
//!
//! nidus generalizes "a local directory, vectors in local RAM" along **two
//! independent, composable axes**, each behind a small **sync, `dyn`-safe** trait:
//!
//! - [`Persistence`] — where the durable *source-of-truth* bytes (`data`/`log`) and
//!   the derived caches (`ann`/`fts`) live. [`LocalFs`] (default) today; S3/GCS are
//!   the planned members (Phase 3). Object-granular: whole-object get/put/delete/list,
//!   plus an **optional** native [`Appender`] capability (local files have it; object
//!   stores return `None` and rewrite whole objects) and a best-effort [`try_lock`].
//! - [`MemoryTier`] — where the in-RAM working set is held so it can be *shared* and
//!   *reloaded without a rebuild*. [`LocalRam`] (default — the process heap) today;
//!   Redis/Valkey/Memcached are planned (Phase 2).
//!
//! Both are **sync deliberately** (SPEC §13.4): search is CPU-over-RAM and never
//! touches a backend, every backend *can* be sync, and a sync trait is `dyn`-safe out
//! of the box — genuine runtime plug-and-play. Selection is by **URL scheme**
//! ([`open_persistence`] / [`open_memory_tier`]).
//!
//! This module is the **trait surface + the two local impls**. Wiring the live
//! `data`/`log` path onto [`Appender`] and routing snapshots through [`Persistence`]
//! land alongside the remote backends, where they first become meaningful.

use std::time::Duration;

use anyhow::{Result, bail};

mod local;
mod ram;

#[cfg(test)]
mod tests;

pub use local::{FileAppender, LocalFs};
pub use ram::LocalRam;

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

    /// Best-effort exclusive lock on `key` (the writer-exclusion primitive, §6.3).
    /// `Ok(Some(guard))` on success — the lock releases when the guard drops;
    /// `Ok(None)` when another holder has it (contention is **not** an error);
    /// `Err` only on a real IO failure. `ttl` reclaims a lock older than it (a
    /// crashed holder).
    fn try_lock(&self, key: &str, ttl: Duration) -> Result<Option<Box<dyn BackendLock>>>;
}

/// A durable, append-shaped byte stream — the native local-FS capability that the
/// `data`/`log` segments need (§5–§6): append with per-write rollback, truncate to a
/// byte boundary, fsync, and atomic whole-file rewrite (compaction). Object-store
/// backends do not provide this (see [`Persistence::appender`]).
pub trait Appender: Send {
    /// The current committed length in bytes — the append point.
    fn len(&self) -> Result<u64>;

    /// Whether the stream is currently empty (no bytes appended).
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

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

    /// Append the entire current contents to `out`. Used once on open, before any
    /// append, to load/replay the stream.
    fn read_to_end(&mut self, out: &mut Vec<u8>) -> Result<()>;
}

/// A held backend lock, released on drop (RAII). Returned by
/// [`Persistence::try_lock`]; the concrete guard owns whatever the backend needs to
/// release (a lock file, a conditional-PUT marker, …).
pub trait BackendLock: Send {}

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
        let _ = rest;
        bail!(
            "the S3 persistence backend ({location:?}) is not yet implemented \
             (planned: SPEC §13.2, nidus-870 Phase 3)"
        );
    }
    if strip_scheme(location, "gs").is_some() || strip_scheme(location, "gcs").is_some() {
        bail!(
            "the Google Cloud Storage persistence backend ({location:?}) is not yet \
             implemented (planned: SPEC §13.2, nidus-870 Phase 3)"
        );
    }
    // `file://<path>` or a bare path → local files.
    let path = strip_scheme(location, "file").unwrap_or(location);
    Ok(Box::new(LocalFs::new(path)?))
}

/// Open a **memory tier** backend from a URL/location string (SPEC §13.3):
///
/// - empty, `local`, or `ram` → [`LocalRam`] (the process heap; nothing shared).
/// - `redis://…` / `rediss://…` / `valkey://…` → "not yet" (planned, Phase 2).
/// - `memcache://…` / `memcached://…` → "not yet" (planned, Phase 2).
pub fn open_memory_tier(location: &str) -> Result<Box<dyn MemoryTier>> {
    match location {
        "" | "local" | "ram" => return Ok(Box::new(LocalRam::new())),
        _ => {}
    }
    for scheme in ["redis", "rediss", "valkey"] {
        if strip_scheme(location, scheme).is_some() {
            bail!(
                "the Redis/Valkey memory tier ({location:?}) is not yet implemented \
                 (planned: SPEC §13.3, nidus-870 Phase 2)"
            );
        }
    }
    for scheme in ["memcache", "memcached"] {
        if strip_scheme(location, scheme).is_some() {
            bail!(
                "the Memcached memory tier ({location:?}) is not yet implemented \
                 (planned: SPEC §13.3, nidus-870 Phase 2)"
            );
        }
    }
    bail!(
        "unknown memory-tier location {location:?} (expected `local`, `redis://…`, or `memcache://…`)"
    )
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
