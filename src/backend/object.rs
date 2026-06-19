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

use anyhow::{Context, Result, bail};

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
    // A plain writer lock carries only a timestamp body (no owner — it is never renewed,
    // just held until drop). `Some(())` means we hold it now.
    let body = lock_body(now_secs(), None);
    Ok(try_claim(persistence, key, ttl, &body)?.map(|()| guard(persistence, key)))
}

/// The shared acquire core for both [`object_try_lock`] and [`ClusterLease`]: write `body`
/// to lock object `key`, returning `Some(())` if we now hold it (the object was absent and
/// we created it, or its prior holder was stale and we reclaimed it) and `None` if a live
/// holder owns it (contention — not an error). Race-free via
/// [`Persistence::try_create_exclusive`] where the backend has it; best-effort get-then-put
/// (**advisory**) otherwise.
fn try_claim(
    persistence: &Arc<dyn Persistence>,
    key: &str,
    ttl: Duration,
    body: &[u8],
) -> Result<Option<()>> {
    let now = now_secs();

    // Fast path: atomic create-if-absent. A fresh acquire (no prior holder) is fully
    // race-free — exactly one of N racing writers gets `Some(true)`.
    match persistence.try_create_exclusive(key, body)? {
        Some(true) => return Ok(Some(())),
        Some(false) => {} // a holder exists — fall through to the staleness check
        None => return advisory_claim(persistence, key, ttl, now, body), // no atomic primitive
    }

    // A lock object exists. Reclaim only if its holder is stale (older than `ttl`).
    let held_at = persistence.get(key)?.map(|b| parse_stamp(&b)).unwrap_or(0);
    if now.saturating_sub(held_at) < ttl.as_secs() {
        return Ok(None); // a live holder owns it
    }
    // Stale: the prior holder crashed. Delete it, then re-attempt the atomic create so
    // that among several writers reclaiming at once exactly one wins (still race-free).
    persistence.delete(key).context("clear stale lock object")?;
    match persistence.try_create_exclusive(key, body)? {
        Some(true) => Ok(Some(())),
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
    body: &[u8],
) -> Result<Option<()>> {
    if let Some(existing) = persistence.get(key)?
        && now.saturating_sub(parse_stamp(&existing)) < ttl.as_secs()
    {
        return Ok(None); // a live holder owns it (else: stale — reclaim by overwriting below)
    }
    persistence
        .put(key, body)
        .context("write advisory lock object")?;
    Ok(Some(()))
}

/// Build the held-lock guard (deletes the lock object on drop).
fn guard(persistence: &Arc<dyn Persistence>, key: &str) -> Box<dyn BackendLock> {
    Box::new(ObjectLock {
        persistence: persistence.clone(),
        key: key.to_string(),
    })
}

// ── Cluster writer lease (SPEC §14.6 phase 5) ────────────────────────────────────

/// A **heartbeated writer lease** over a shared object store: like [`ObjectLock`] but it
/// carries an **owner** identity and is **renewed** on every write batch
/// ([`renew`](ClusterLease::renew)), so a long-lived writer keeps it indefinitely while an
/// idle one (silent past the TTL) can be taken over. Renew also **fences**: it verifies the
/// lease still names this owner before re-stamping, so a writer that was superseded while
/// paused fails its next renew rather than clobbering the store. Released on drop.
pub struct ClusterLease {
    persistence: Arc<dyn Persistence>,
    key: String,
    /// This writer instance's unique owner id (PID + acquire time); the fencing token.
    owner: String,
}

impl ClusterLease {
    /// Acquire the lease for `key`, minting a fresh owner id. `Ok(Some)` when held,
    /// `Ok(None)` when a live writer already holds it (contention — not an error). Reclaims a
    /// stale lease (a crashed holder past `ttl`) race-free, exactly as [`object_try_lock`].
    pub fn acquire(
        persistence: &Arc<dyn Persistence>,
        key: &str,
        ttl: Duration,
    ) -> Result<Option<ClusterLease>> {
        validate_key(key)?;
        let owner = mint_owner();
        let body = lock_body(now_secs(), Some(&owner));
        Ok(
            try_claim(persistence, key, ttl, &body)?.map(|()| ClusterLease {
                persistence: persistence.clone(),
                key: key.to_string(),
                owner,
            }),
        )
    }

    /// Renew the lease before a write batch: **fence** (verify the lease still names this
    /// owner) then re-stamp it with a fresh timestamp. Errors when another writer has taken
    /// over — the caller must stop writing, as it no longer holds the store. A no-op-shaped
    /// success otherwise. The renewal is what keeps an active writer's lease from ever going
    /// stale; the fence is what stops a paused-then-superseded writer from clobbering. (No TTL
    /// argument: while we still own the lease no peer can have reclaimed it, so we just
    /// re-stamp; staleness only gates a *peer*'s takeover via [`try_claim`].)
    pub fn renew(&self) -> Result<()> {
        match self.persistence.get(&self.key)? {
            Some(bytes) => {
                let owner = parse_owner(&bytes);
                if owner.as_deref() != Some(self.owner.as_str()) {
                    bail!(
                        "writer lease lost: the store's lease is now held by another writer \
                         (this instance was superseded while paused past the lease TTL) — \
                         stop writing and reopen"
                    );
                }
                // We still own it — re-stamp to extend the TTL.
                self.persistence
                    .put(&self.key, &lock_body(now_secs(), Some(&self.owner)))
                    .context("renew writer lease")
            }
            None => {
                // The lease object vanished (a peer found ours stale and deleted it, or it was
                // never persisted). Re-claim atomically: if a peer beat us to it we are fenced.
                let body = lock_body(now_secs(), Some(&self.owner));
                match self.persistence.try_create_exclusive(&self.key, &body)? {
                    Some(true) => Ok(()), // reclaimed cleanly
                    Some(false) => bail!(
                        "writer lease lost: another writer re-created the lease — stop writing \
                         and reopen"
                    ),
                    None => self
                        .persistence
                        .put(&self.key, &body)
                        .context("re-create writer lease (advisory backend)"),
                }
            }
        }
    }
}

impl BackendLock for ClusterLease {}

impl Drop for ClusterLease {
    fn drop(&mut self) {
        // Release only if we still own it — never delete a lease a peer has taken over.
        if let Ok(Some(bytes)) = self.persistence.get(&self.key)
            && parse_owner(&bytes).as_deref() == Some(self.owner.as_str())
        {
            let _ = self.persistence.delete(&self.key);
        }
    }
}

/// A unique owner id for a writer instance: process id + acquire time (nanos). Distinct
/// across processes and across restarts of the same process, so a stale lease can never be
/// mistaken for a live one of a reborn writer.
fn mint_owner() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{}", std::process::id(), nanos)
}

/// Encode a lock object body: the unix-seconds stamp first, then an optional owner token,
/// space-separated (`"<ts>"` for a plain lock, `"<ts> <owner>"` for a lease). The stamp is
/// first so [`parse_stamp`] reads it the same way for both shapes.
fn lock_body(ts: u64, owner: Option<&str>) -> Vec<u8> {
    match owner {
        Some(o) => format!("{ts} {o}").into_bytes(),
        None => ts.to_string().into_bytes(),
    }
}

/// The owner token from a lease body (`"<ts> <owner>"`), or `None` for an owner-less plain
/// lock body / an unparseable one.
fn parse_owner(bytes: &[u8]) -> Option<String> {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.split_whitespace().nth(1))
        .map(|o| o.to_string())
}

/// Current unix time in seconds (a clock before the epoch reads as 0 — makes any lock
/// look stale, the safe-to-reclaim direction).
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Parse the unix-seconds stamp a lock object stores — the **first** whitespace token, so it
/// reads identically from a plain `"<ts>"` body and a lease `"<ts> <owner>"` body. An
/// unreadable body reads as `0` (epoch), which makes it look stale and so reclaimable — the
/// safe direction.
fn parse_stamp(bytes: &[u8]) -> u64 {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.split_whitespace().next())
        .and_then(|t| t.parse().ok())
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
