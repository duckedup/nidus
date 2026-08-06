//! Object-store live-backing adapters (SPEC §13.7): the seam that lets a **live** store
//! run on a whole-object [`Persistence`] backend (S3/GCS) that has no native append or
//! `O_EXCL` lock.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

use super::{Appender, BackendLock, CasOutcome, MemAppender, Persistence, validate_key};

/// An append handle backed by a single whole object on a [`Persistence`] backend: edits
/// buffer in RAM and become durable as one atomic `put` on [`sync`](Appender::sync).
pub struct ObjectAppender {
    persistence: Arc<dyn Persistence>,
    key: String,
    /// In-RAM mirror of the object's bytes — the append point and read source.
    buf: MemAppender,
    /// Compare-and-swap fencing (cluster mode, SPEC §14.6). `None` = plain mode, every sync
    /// rewrites unconditionally. `Some(token)` = each sync is conditional on the version this
    /// writer last saw, so a superseded writer's sync is refused rather than clobbering a peer.
    cas_token: Option<Option<String>>,
}

impl ObjectAppender {
    /// Open object `key`, loading its bytes into the RAM buffer (absent → empty, matching a fresh
    /// local segment). `cas` selects the commit discipline; in CAS mode the current version token
    /// is captured here for the first conditional sync.
    pub fn open(persistence: Arc<dyn Persistence>, key: &str, cas: bool) -> Result<ObjectAppender> {
        validate_key(key)?;
        let (bytes, cas_token) = if cas {
            match persistence.get_cas(key)? {
                Some((bytes, token)) => (bytes, Some(token)),
                None => (Vec::new(), Some(None)), // absent → expect-absent on first write
            }
        } else {
            (persistence.get(key)?.unwrap_or_default(), None)
        };
        Ok(ObjectAppender {
            persistence,
            key: key.to_string(),
            buf: MemAppender::from_bytes(bytes),
            cas_token,
        })
    }

    /// Persist the whole buffer as one atomic object write. In CAS mode the write is
    /// conditional on the captured token and a mismatch **fences** this writer (a hard error)
    /// rather than overwriting a peer's bytes; the token is advanced on success.
    fn flush_object(&mut self) -> Result<()> {
        let Some(token) = self.cas_token.clone() else {
            return self
                .persistence
                .put(&self.key, self.buf.bytes())
                .with_context(|| format!("rewrite object {:?} on sync", self.key));
        };
        match self
            .persistence
            .put_cas(&self.key, self.buf.bytes(), token.as_deref())?
        {
            CasOutcome::Written(new) => {
                self.cas_token = Some(match new {
                    Some(t) => Some(t),
                    // Backend reported no new token — re-read it for the next conditional write.
                    None => self.persistence.get_cas(&self.key)?.and_then(|(_, t)| t),
                });
                Ok(())
            }
            CasOutcome::Stale => bail!(
                "writer fenced: object {:?} was modified by another writer — this instance was \
                 superseded (its lease was taken over while it stalled); stop writing and reopen",
                self.key
            ),
            // No CAS on this backend: fall back to a plain rewrite (advisory, as the non-cluster
            // path). Cluster correctness then rests on the per-batch lease fence alone.
            CasOutcome::Unsupported => self
                .persistence
                .put(&self.key, self.buf.bytes())
                .with_context(|| format!("rewrite object {:?} on sync", self.key)),
        }
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
        lease_debug(format_args!(
            "ObjectLock::drop key={:?} — DELETING the lock object",
            self.key
        ));
        let _ = self.persistence.delete(&self.key);
    }
}

/// Writer lock over a whole-object backend (S3/GCS). `Ok(Some)` when the lock object was
/// absent (claimed) or older than `ttl` (a crashed holder — reclaimed); `Ok(None)` when a
/// fresh holder has it (contention, never an error).
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

/// The shared acquire core for [`object_try_lock`] and [`ClusterLease`]: `Some(())` = we hold it
/// (created, or reclaimed from a stale holder), `None` = a live holder owns it. Race-free on a
/// CAS-capable backend; without CAS it degrades to a best-effort, advisory get-then-put.
fn try_claim(
    persistence: &Arc<dyn Persistence>,
    key: &str,
    ttl: Duration,
    body: &[u8],
) -> Result<Option<()>> {
    let now = now_secs();
    lease_debug(format_args!(
        "try_claim key={key:?} ttl={}s now={now} body={:?}",
        ttl.as_secs(),
        String::from_utf8_lossy(body)
    ));

    // Fast path: atomic create-if-absent. A fresh acquire (no prior holder) is fully
    // race-free — exactly one of N racing writers gets `Some(true)`.
    match persistence.try_create_exclusive(key, body)? {
        Some(true) => {
            lease_debug(format_args!(
                "try_claim key={key:?} ACQUIRED via create-if-absent (object was ABSENT)"
            ));
            return Ok(Some(()));
        }
        Some(false) => {} // a holder exists — fall through to the staleness check
        None => return advisory_claim(persistence, key, ttl, now, body), // no atomic primitive
    }

    // A lock object exists; reclaim only if its holder is stale. The holder's CAS token is captured
    // alongside its stamp so the reclaim is conditional (nidus-5kj): the holder might renew between
    // our read and write, and an unconditional delete-then-create would steal a live lease.
    let Some((held, token)) = persistence.get_cas(key)? else {
        // Vanished since the create attempt above — race a fresh atomic create for it.
        lease_debug(format_args!(
            "try_claim key={key:?} VANISHED between create-probe and read — racing a create"
        ));
        return reclaim_create(persistence, key, body);
    };
    lease_debug(format_args!(
        "try_claim key={key:?} held={:?} stamp={} age={}s ttl={}s token={}",
        String::from_utf8_lossy(&held),
        parse_stamp(&held),
        now.saturating_sub(parse_stamp(&held)),
        ttl.as_secs(),
        token.as_deref().unwrap_or("<none>")
    ));
    if is_live(now, parse_stamp(&held), ttl) {
        return Ok(None); // a live holder owns it
    }
    lease_debug(format_args!(
        "try_claim key={key:?} holder judged STALE — attempting reclaim"
    ));
    match token {
        // CAS-capable backend (S3/GCS): reclaim with a compare-and-swap gated on the stale
        // object's token. A holder that renewed in the gap (its token moved) or a peer that
        // reclaimed first (likewise) defeats us cleanly — fully race-free, no live lease stolen.
        Some(tok) => match persistence.put_cas(key, body, Some(&tok))? {
            CasOutcome::Written(_) => Ok(Some(())),
            CasOutcome::Stale => Ok(None), // holder renewed, or a peer reclaimed first
            CasOutcome::Unsupported => reclaim_create(persistence, key, body),
        },
        // No conditional-overwrite CAS (create-if-absent only): fall back to the best-effort
        // delete-then-create reclaim (one winner among reclaimers, but not fenced against a
        // holder renewing in the gap — the limit of a backend without compare-and-swap).
        None => reclaim_create(persistence, key, body),
    }
}

/// Best-effort stale reclaim for a backend without conditional-overwrite CAS: clear the stale
/// object then race a fresh atomic create, so among several reclaimers exactly one wins.
fn reclaim_create(
    persistence: &Arc<dyn Persistence>,
    key: &str,
    body: &[u8],
) -> Result<Option<()>> {
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
        && is_live(now, parse_stamp(&existing), ttl)
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

/// A heartbeated writer lease over a shared object store: carries an owner identity and is renewed
/// per write batch, so an active writer keeps it indefinitely while one silent past the TTL can be
/// taken over. Renew also fences, verifying the owner before re-stamping. Released on drop.
pub struct ClusterLease {
    persistence: Arc<dyn Persistence>,
    key: String,
    /// This writer instance's unique owner id (PID + acquire time); the fencing token.
    owner: String,
}

/// A **non-owning** handle that can renew a lease but never releases it.
#[derive(Clone)]
pub struct LeaseRenewer {
    persistence: Arc<dyn Persistence>,
    key: String,
    owner: String,
    /// Latched when a renewal proves the lease **definitively** lost. Shared with the owning
    /// store's `fenced` flag, so an out-of-band renewal failure is *observable* rather than
    /// only logged (nidus-lp4.7).
    fenced: Arc<std::sync::atomic::AtomicBool>,
}

impl LeaseRenewer {
    /// Re-stamp the lease, extending its TTL. Errors if this instance has been superseded
    /// — the same fencing signal a write would hit.
    pub fn renew(&self) -> Result<()> {
        match renew_lease_object(&self.persistence, &self.key, &self.owner) {
            Err(e) if is_lease_lost(&e) => {
                latch_fenced(&self.fenced);
                Err(e)
            }
            other => other,
        }
    }
}

/// Latch a store's `fenced` flag, counting the transition exactly once.
pub(crate) fn latch_fenced(fenced: &std::sync::atomic::AtomicBool) {
    if !fenced.swap(true, std::sync::atomic::Ordering::AcqRel) {
        crate::metrics::metrics().lease_fenced.inc();
    }
}

impl ClusterLease {
    /// A `Drop`-free handle for renewing this lease out of band — see [`LeaseRenewer`].
    pub fn renewer(&self, fenced: Arc<std::sync::atomic::AtomicBool>) -> LeaseRenewer {
        LeaseRenewer {
            persistence: self.persistence.clone(),
            key: self.key.clone(),
            owner: self.owner.clone(),
            fenced,
        }
    }

    /// This instance's fencing token — the owner id written into the lease object.
    pub fn owner(&self) -> &str {
        &self.owner
    }

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

    /// Renew before a write batch: fence (verify the lease still names this owner), then re-stamp.
    /// Errors when another writer has taken over, and the caller must stop writing. No TTL argument
    /// — while we own the lease no peer can have reclaimed it, so this just re-stamps.
    pub fn renew(&self) -> Result<()> {
        renew_lease_object(&self.persistence, &self.key, &self.owner)
    }
}

impl BackendLock for ClusterLease {}

/// Re-stamp lease object `key` for `owner`, the shared body of [`ClusterLease::renew`] and
/// [`LeaseRenewer::renew`] — one implementation, so the owning guard and the out-of-band
/// renewer can never drift on what "renew" means.
fn renew_lease_object(persistence: &Arc<dyn Persistence>, key: &str, owner: &str) -> Result<()> {
    // Counted here rather than at the two call sites: both the owning guard and the background
    // renewer route through this one function, so the attempt/outcome split — which would show an
    // object store misbehaving early (nidus-abx.4) — cannot drift between them.
    let m = crate::metrics::metrics();
    m.lease_renew_attempts.inc();
    let outcome = renew_lease_object_inner(persistence, key, owner);
    match &outcome {
        Ok(()) => m.lease_renew_ok.inc(),
        Err(e) if is_lease_lost(e) => m.lease_renew_lost.inc(),
        Err(_) => m.lease_renew_transient_failures.inc(),
    }
    outcome
}

fn renew_lease_object_inner(
    persistence: &Arc<dyn Persistence>,
    key: &str,
    owner: &str,
) -> Result<()> {
    match persistence.get_cas(key)? {
        Some((bytes, token)) => {
            lease_debug(format_args!(
                "renew key={key:?} owner={owner} saw={:?} token={}",
                String::from_utf8_lossy(&bytes),
                token.as_deref().unwrap_or("<none>")
            ));
            if parse_owner(&bytes).as_deref() != Some(owner) {
                return Err(lease_lost(
                    "writer lease lost: the store's lease is now held by another writer \
                     (this instance was superseded while paused past the lease TTL) — \
                     stop writing and reopen",
                ));
            }
            // We still own it — re-stamp conditionally on the token just read (nidus-lp4.7). An
            // unconditional put is a read-modify-write race: a peer reclaiming in the gap would be
            // silently overwritten, leaving two live writers each believing it holds the lease.
            let body = lock_body(now_secs(), Some(owner));
            match token {
                Some(tok) => match persistence.put_cas(key, &body, Some(&tok))? {
                    CasOutcome::Written(_) => Ok(()),
                    CasOutcome::Stale => {
                        // Someone wrote the lease in the gap: either a peer taking over (we are
                        // fenced) or this instance's other renewer, whose re-stamp is idempotent and
                        // harmless. Re-read to tell them apart rather than fencing a healthy writer.
                        match persistence.get(key)? {
                            Some(now_held) if parse_owner(&now_held).as_deref() == Some(owner) => {
                                lease_debug(format_args!(
                                    "renew key={key:?} owner={owner} CAS lost to our own \
                                     concurrent renewal — still ours, treating as renewed"
                                ));
                                Ok(())
                            }
                            _ => Err(lease_lost(
                                "writer lease lost: another writer reclaimed the lease while \
                                 this instance was renewing it — stop writing and reopen",
                            )),
                        }
                    }
                    // No conditional overwrite on this backend: fall back to the plain put.
                    CasOutcome::Unsupported => {
                        persistence.put(key, &body).context("renew writer lease")
                    }
                },
                None => persistence.put(key, &body).context("renew writer lease"),
            }
        }
        None => {
            // The lease object vanished (a peer found ours stale and deleted it, or it was
            // never persisted). Re-claim atomically: if a peer beat us to it we are fenced.
            lease_debug(format_args!(
                "renew key={key:?} owner={owner} — lease object ABSENT, re-creating"
            ));
            let body = lock_body(now_secs(), Some(owner));
            match persistence.try_create_exclusive(key, &body)? {
                Some(true) => Ok(()), // reclaimed cleanly
                Some(false) => Err(lease_lost(
                    "writer lease lost: another writer re-created the lease — stop writing \
                     and reopen",
                )),
                None => persistence
                    .put(key, &body)
                    .context("re-create writer lease (advisory backend)"),
            }
        }
    }
}

impl Drop for ClusterLease {
    fn drop(&mut self) {
        // Release only if we still own it — never delete a lease a peer has taken over.
        if let Ok(Some(bytes)) = self.persistence.get(&self.key)
            && parse_owner(&bytes).as_deref() == Some(self.owner.as_str())
        {
            lease_debug(format_args!(
                "ClusterLease::drop key={:?} owner={} — DELETING the lease object",
                self.key, self.owner
            ));
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

/// Lease tracing at `debug` level (`NIDUS_LOG=debug`, or the legacy `NIDUS_LEASE_DEBUG=1`).
fn lease_debug(args: std::fmt::Arguments<'_>) {
    crate::diag::diag!(
        crate::diag::Level::Debug,
        "lease",
        args,
        "pid" => std::process::id(),
    );
}

/// The error a renewal returns when this instance has **definitively** lost the lease to
/// another — as distinct from a transient backend failure (a dropped connection, a 5xx from
/// the object store), which says nothing about who holds the lease.
#[derive(Debug)]
pub struct LeaseLost(String);

impl std::fmt::Display for LeaseLost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LeaseLost {}

/// Whether `err` is the definitive [`LeaseLost`] signal rather than a transient backend
/// failure. Searches the whole `anyhow` chain, so added context cannot hide it.
pub fn is_lease_lost(err: &anyhow::Error) -> bool {
    err.downcast_ref::<LeaseLost>().is_some()
}

/// Build a [`LeaseLost`] error. Deliberately not `bail!`/`anyhow!`: the *type* is the signal
/// callers classify on, and a plain string error would be indistinguishable from IO trouble.
fn lease_lost(msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(LeaseLost(msg.into()))
}

/// Whether a lock/lease stamped at `stamp` is still **live** at `now` under `ttl`.
pub(super) fn is_live(now: u64, stamp: u64, ttl: Duration) -> bool {
    now.saturating_sub(stamp) <= ttl.as_secs()
}

/// Current unix time in seconds (a clock before the epoch reads as 0 — makes any lock
/// look stale, the safe-to-reclaim direction).
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Parse the unix-seconds stamp a lock object stores — the **first** whitespace token, so it reads
/// identically from a plain `"<ts>"` body and a lease `"<ts> <owner>"` body. An unreadable body
/// reads as `0` (epoch), which makes it look stale and so reclaimable — the safe direction.
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
