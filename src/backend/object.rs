//! Object-store live-backing adapters (SPEC §13.7): the seam that lets a **live** store
//! run on a whole-object [`Persistence`] backend (S3/GCS) that has no native append or
//! `O_EXCL` lock.
//!
//! - [`ObjectAppender`] backs a `data`/`log` segment by an in-RAM buffer (reusing
//!   [`MemAppender`]'s append/truncate/read mechanics) and rewrites the whole object with
//!   one atomic [`Persistence::put`] on `sync`/`rewrite`. So the segments keep their exact
//!   append-then-fsync discipline; the object store just turns each "fsync" into a
//!   whole-object rewrite (O(object), the cost §13.5 names).
//! - [`advisory_try_lock`] is the writer lock for object stores: a best-effort
//!   get-then-put lock object with a TTL, released by deleting it on drop. It is
//!   **advisory** (not race-free — two writers racing the gap could both acquire), which
//!   suits nidus's single-writer / low-write-rate positioning; a race-free conditional-PUT
//!   lock is the follow-up (`If-None-Match` / `ifGenerationMatch=0`).
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

/// Best-effort advisory writer lock over a whole-object backend (S3/GCS). `Ok(Some)` when
/// the lock object is absent or older than `ttl` (a crashed holder — reclaimed); `Ok(None)`
/// when a fresh holder has it (contention, never an error). **Advisory:** the read and the
/// claiming write are not atomic, so two writers racing the gap could both acquire — fine
/// for the single-writer/dev positioning, not for true concurrent writers.
pub fn advisory_try_lock(
    persistence: &Arc<dyn Persistence>,
    key: &str,
    ttl: Duration,
) -> Result<Option<Box<dyn BackendLock>>> {
    validate_key(key)?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Some(existing) = persistence.get(key)? {
        let held_at = parse_stamp(&existing);
        if now.saturating_sub(held_at) < ttl.as_secs() {
            return Ok(None); // a live holder owns it
        }
        // else: stale (older than ttl) — reclaim by overwriting below.
    }
    persistence
        .put(key, now.to_string().as_bytes())
        .context("write advisory lock object")?;
    Ok(Some(Box::new(ObjectLock {
        persistence: persistence.clone(),
        key: key.to_string(),
    })))
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
