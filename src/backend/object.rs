//! Object-store live-backing adapters (SPEC §13.7): the seam that lets a **live** store
//! run on a whole-object [`Persistence`] backend (S3/GCS) that has no native append or
//! `O_EXCL` lock.
//!
//! - [`ObjectAppender`] backs a `data`/`log` segment by an in-RAM buffer (reusing
//!   [`MemAppender`]'s append/truncate/read mechanics) and rewrites the whole object with
//!   one atomic [`Persistence::put`] on `sync`/`rewrite`. So the segments keep their exact
//!   append-then-fsync discipline; the object store just turns each "fsync" into a
//!   whole-object rewrite (O(object), the cost §13.5 names).
//! - [`object_try_lock`] is the writer lock for object stores: a TTL'd lock object,
//!   released by deleting it on drop. A fresh acquire goes through the backend's atomic
//!   create-if-absent ([`Persistence::try_create_exclusive`] — S3 `If-None-Match: *`,
//!   GCS `ifGenerationMatch=0`), so exactly one of N racing writers wins — **race-free**.
//!   A backend without that primitive falls back to a best-effort get-then-put
//!   (**advisory**: two writers racing the gap could both acquire), which still suits
//!   nidus's single-writer / low-write-rate positioning.
//!
//! Both hold an `Arc` of the same backend the store uses, so segments, caches, and the
//! lock all go through one client.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use super::{Appender, BackendLock, MemAppender, Persistence, validate_key};

/// An append handle backed by a single whole object on a [`Persistence`] backend: edits
/// buffer in RAM and become durable as one atomic `put` on [`sync`](Appender::sync).
pub struct ObjectAppender {
    persistence: Arc<dyn Persistence>,
    key: String,
    /// In-RAM mirror of the object's bytes — the append point and read source.
    buf: MemAppender,
}

impl ObjectAppender {
    /// Open the object `key` on `persistence`, loading its current bytes into the RAM
    /// buffer (absent object → empty, matching a fresh local segment).
    pub fn open(persistence: Arc<dyn Persistence>, key: &str) -> Result<ObjectAppender> {
        validate_key(key)?;
        let bytes = persistence.get(key)?.unwrap_or_default();
        Ok(ObjectAppender {
            persistence,
            key: key.to_string(),
            buf: MemAppender::from_bytes(bytes),
        })
    }

    /// Persist the whole buffer as one atomic object write.
    fn flush_object(&self) -> Result<()> {
        self.persistence
            .put(&self.key, self.buf.bytes())
            .with_context(|| format!("rewrite object {:?} on sync", self.key))
    }
}

impl Appender for ObjectAppender {
    fn len(&self) -> Result<u64> {
        self.buf.len()
    }

    fn read_exact_at(&mut self, offset: u64, out: &mut [u8]) -> Result<()> {
        self.buf.read_exact_at(offset, out)
    }

    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        // Buffered in RAM; durability is deferred to `sync`, exactly as a file append is
        // deferred to fsync. The store's commit protocol calls `sync` at the batch end.
        self.buf.append(bytes)
    }

    fn truncate_to(&mut self, offset: u64) -> Result<()> {
        self.buf.truncate_to(offset)
    }

    fn sync(&mut self) -> Result<()> {
        self.flush_object()
    }

    fn rewrite(&mut self, bytes: &[u8]) -> Result<()> {
        self.buf.rewrite(bytes)?;
        self.flush_object()
    }
}

/// A held advisory lock over a whole-object backend: a lock object exists for the
/// lifetime of this guard and is deleted on drop. `Send + Sync` like every
/// [`BackendLock`] (it lives inside the shared store).
pub struct ObjectLock {
    persistence: Arc<dyn Persistence>,
    key: String,
}

impl BackendLock for ObjectLock {}

impl Drop for ObjectLock {
    fn drop(&mut self) {
        // Best-effort release; if it fails the TTL still reclaims the lock eventually.
        let _ = self.persistence.delete(&self.key);
    }
}

/// Writer lock over a whole-object backend (S3/GCS). `Ok(Some)` when the lock object was
/// absent (claimed) or older than `ttl` (a crashed holder — reclaimed); `Ok(None)` when a
/// fresh holder has it (contention, never an error).
///
/// A fresh acquire uses the backend's atomic create-if-absent
/// ([`Persistence::try_create_exclusive`]), so among N writers racing an unlocked store
/// exactly one wins — **race-free**. Only a backend that returns `None` from that method
/// (no atomic primitive) falls back to a best-effort get-then-put (**advisory** — the read
/// and write are not atomic).
pub fn object_try_lock(
    persistence: &Arc<dyn Persistence>,
    key: &str,
    ttl: Duration,
) -> Result<Option<Box<dyn BackendLock>>> {
    validate_key(key)?;
    let now = now_secs();
    let stamp = now.to_string();

    // Fast path: atomic create-if-absent. A fresh acquire (no prior holder) is fully
    // race-free — exactly one of N racing writers gets `Some(true)`.
    match persistence.try_create_exclusive(key, stamp.as_bytes())? {
        Some(true) => return Ok(Some(guard(persistence, key))),
        Some(false) => {} // a holder exists — fall through to the staleness check
        None => return advisory_claim(persistence, key, ttl, now, &stamp), // no atomic primitive
    }

    // A lock object exists. Reclaim only if its holder is stale (older than `ttl`).
    let held_at = persistence.get(key)?.map(|b| parse_stamp(&b)).unwrap_or(0);
    if now.saturating_sub(held_at) < ttl.as_secs() {
        return Ok(None); // a live holder owns it
    }
    // Stale: the prior holder crashed. Delete it, then re-attempt the atomic create so
    // that among several writers reclaiming at once exactly one wins (still race-free).
    persistence.delete(key).context("clear stale lock object")?;
    match persistence.try_create_exclusive(key, stamp.as_bytes())? {
        Some(true) => Ok(Some(guard(persistence, key))),
        _ => Ok(None), // another writer reclaimed first
    }
}

/// The best-effort get-then-put claim for a backend with no atomic create-if-absent.
/// **Advisory** — the staleness read and the claiming write are not atomic, so two writers
/// racing the gap could both acquire. Kept as the fallback for the single-writer positioning.
fn advisory_claim(
    persistence: &Arc<dyn Persistence>,
    key: &str,
    ttl: Duration,
    now: u64,
    stamp: &str,
) -> Result<Option<Box<dyn BackendLock>>> {
    if let Some(existing) = persistence.get(key)?
        && now.saturating_sub(parse_stamp(&existing)) < ttl.as_secs()
    {
        return Ok(None); // a live holder owns it (else: stale — reclaim by overwriting below)
    }
    persistence
        .put(key, stamp.as_bytes())
        .context("write advisory lock object")?;
    Ok(Some(guard(persistence, key)))
}

/// Build the held-lock guard (deletes the lock object on drop).
fn guard(persistence: &Arc<dyn Persistence>, key: &str) -> Box<dyn BackendLock> {
    Box::new(ObjectLock {
        persistence: persistence.clone(),
        key: key.to_string(),
    })
}

/// Current unix time in seconds (a clock before the epoch reads as 0 — makes any lock
/// look stale, the safe-to-reclaim direction).
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Parse the unix-seconds stamp a lock object stores; an unreadable body reads as `0`
/// (epoch), which makes it look stale and so reclaimable — the safe direction.
fn parse_stamp(bytes: &[u8]) -> u64 {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Surface a clear "store is locked" error for the advisory path (shared with the native
/// lock's message at the call site).
pub fn locked_error(location: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "store at {location:?} is locked by another writer (an advisory `lock` object \
         exists and is not yet stale) — stop that writer, or wait for the lock TTL to elapse"
    )
}
