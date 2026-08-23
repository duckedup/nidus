//! Backend tests. Pure/in-RAM cases (LocalRam, scheme parsing, key validation) run
//! under Miri; file-backed LocalFs cases fsync and so are `#[cfg_attr(miri, ignore)]`.

use std::time::Duration;

use super::*;

// ── Key validation (pure, Miri-clean) ──────────────────────────────────────────

#[test]
fn validate_key_accepts_flat_names() {
    for k in ["data", "log", "ann", "fts", "snap.tar.gz", "a_b-c.1"] {
        assert!(validate_key(k).is_ok(), "{k} should be valid");
    }
}

#[test]
fn validate_key_rejects_bad_names() {
    for k in ["", "a/b", "..", "../escape", "dir/../x", "c:\\win"] {
        assert!(validate_key(k).is_err(), "{k:?} should be rejected");
    }
}

// ── Scheme parsing (pure, Miri-clean) ───────────────────────────────────────────

#[test]
fn memory_tier_redis_family_schemes_open() {
    // Construction is lazy (no connection yet), so every RESP-family scheme resolves
    // to a RedisTier without touching the network.
    for loc in [
        "redis://h:6379",
        "rediss://h",
        "valkey://h",
        "valkeys://h",
        "keydb://h",
        "dragonfly://h/0",
        "redis://h:6379/0?prefix=nidus",
    ] {
        assert!(open_memory_tier(loc).is_ok(), "{loc} should open");
    }
}

#[test]
fn memory_tier_unknown_scheme_errors() {
    // Memcached is intentionally unsupported; so is any non-RESP scheme.
    for loc in ["kafka://h", "memcache://h", "memcached://h"] {
        let err = open_memory_tier(loc).err().unwrap().to_string();
        assert!(err.contains("unknown memory-tier location"), "{loc}: {err}");
    }
}

#[test]
fn split_object_location_cases() {
    assert_eq!(
        split_object_location("snap.tar.gz").unwrap(),
        (".", "snap.tar.gz")
    );
    assert_eq!(
        split_object_location("./snap.tar.gz").unwrap(),
        (".", "snap.tar.gz")
    );
    assert_eq!(
        split_object_location("/a/b/snap.tgz").unwrap(),
        ("/a/b", "snap.tgz")
    );
    assert_eq!(split_object_location("/snap").unwrap(), ("/", "snap"));
    assert_eq!(
        split_object_location("file:///backups/snap.tar.gz").unwrap(),
        ("file:///backups", "snap.tar.gz")
    );
    assert_eq!(
        split_object_location("s3://bucket/snap.tar.gz").unwrap(),
        ("s3://bucket", "snap.tar.gz")
    );
    assert!(split_object_location("dir/").is_err());
}

#[test]
fn strip_scheme_is_case_insensitive_and_bounded() {
    assert_eq!(strip_scheme("S3://x", "s3"), Some("x"));
    assert_eq!(strip_scheme("file:///abs", "file"), Some("/abs"));
    assert_eq!(strip_scheme("gs://bucket/p", "gs"), Some("bucket/p"));
    assert_eq!(strip_scheme("gcs://bucket/p", "gcs"), Some("bucket/p"));
    assert_eq!(strip_scheme("s3:/x", "s3"), None); // missing one slash
    assert_eq!(strip_scheme("s3", "s3"), None);
}

// ── LocalRam (pure, Miri-clean) ─────────────────────────────────────────────────

#[test]
fn local_ram_round_trips() {
    let tier = LocalRam::new();
    assert!(tier.load("k").unwrap().is_none());
    tier.store("k", b"hello", None).unwrap();
    assert_eq!(
        tier.load("k").unwrap().as_deref(),
        Some(b"hello".as_slice())
    );
    // Overwrite.
    tier.store("k", b"world", Some(Duration::from_secs(5)))
        .unwrap();
    assert_eq!(
        tier.load("k").unwrap().as_deref(),
        Some(b"world".as_slice())
    );
}

#[test]
fn open_memory_tier_local_aliases() {
    for loc in ["", "local", "ram"] {
        let tier = open_memory_tier(loc).unwrap();
        tier.store("x", b"1", None).unwrap();
        assert_eq!(tier.load("x").unwrap().as_deref(), Some(b"1".as_slice()));
    }
}

// ── object_try_lock over a whole-object backend (pure/in-RAM, Miri-clean) ───────

use std::collections::HashMap;
use std::sync::Mutex;

/// The map + a monotonic generation counter, behind one lock (so a CAS read-modify-write is
/// atomic and there is no lock-ordering between map and counter).
struct MapState {
    objects: HashMap<String, (Vec<u8>, u64)>, // bytes + the version it was written at
    next_gen: u64,
}

/// A whole-object [`Persistence`] over an in-RAM map. `cas == true` models a real CAS object store,
/// `false` one with no atomic primitive (the advisory fallback). `inject_renew` makes the next
/// `get_cas` do a concurrent peer write, staling a reclaimer's token in the read→write gap.
struct MapBackend {
    state: Mutex<MapState>,
    cas: bool,
    inject_renew: Mutex<Option<Vec<u8>>>,
    /// When set, reads fail with an IO-shaped error (see [`MapBackend::fail_reads`]).
    fail_reads: Mutex<bool>,
}

impl MapBackend {
    fn arc(cas: bool) -> Arc<dyn Persistence> {
        Self::arc_injecting(cas, None)
    }
    fn arc_injecting(cas: bool, renew_with: Option<Vec<u8>>) -> Arc<dyn Persistence> {
        Self::concrete(cas, renew_with)
    }
    /// The same double as a **concrete** handle, so a test can keep calling inherent methods
    /// ([`inject_next_read`](Self::inject_next_read)) after handing the trait object to the
    /// code under test. Coerce with `let dyn_: Arc<dyn Persistence> = concrete.clone();`.
    fn concrete(cas: bool, renew_with: Option<Vec<u8>>) -> Arc<MapBackend> {
        Arc::new(MapBackend {
            state: Mutex::new(MapState {
                objects: HashMap::new(),
                next_gen: 0,
            }),
            cas,
            inject_renew: Mutex::new(renew_with),
            fail_reads: Mutex::new(false),
        })
    }

    /// Arrange for the **next** `get_cas` to write `body` after taking its snapshot —
    /// simulating a peer that wrote the object in the read→write gap. Set after acquiring, so
    /// the injected body can name an owner the test only learns at acquire time.
    fn inject_next_read(&self, body: Vec<u8>) {
        *self.inject_renew.lock().unwrap() = Some(body);
    }

    /// Make every read fail with an IO-shaped error — a dropped connection, not a verdict on
    /// who holds the lease.
    fn fail_reads(&self, fail: bool) {
        *self.fail_reads.lock().unwrap() = fail;
    }
}

impl Persistence for MapBackend {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if *self.fail_reads.lock().unwrap() {
            anyhow::bail!("injected transient backend failure: Peer disconnected");
        }
        Ok(self
            .state
            .lock()
            .unwrap()
            .objects
            .get(key)
            .map(|(b, _)| b.clone()))
    }
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        s.next_gen += 1;
        let g = s.next_gen;
        s.objects.insert(key.to_string(), (bytes.to_vec(), g));
        Ok(())
    }
    fn delete(&self, key: &str) -> Result<()> {
        self.state.lock().unwrap().objects.remove(key);
        Ok(())
    }
    fn list(&self) -> Result<Vec<String>> {
        Ok(self.state.lock().unwrap().objects.keys().cloned().collect())
    }
    fn get_cas(&self, key: &str) -> Result<Option<(Vec<u8>, Option<String>)>> {
        if *self.fail_reads.lock().unwrap() {
            anyhow::bail!("injected transient backend failure: Peer disconnected");
        }
        if !self.cas {
            return Ok(self.get(key)?.map(|b| (b, None)));
        }
        let snap = self.state.lock().unwrap().objects.get(key).cloned();
        // Test hook: a peer renews in the read→write gap, bumping the generation so the token
        // we return below is already stale by the time the caller's `put_cas` runs.
        if let Some(fresh) = self.inject_renew.lock().unwrap().take() {
            self.put(key, &fresh)?;
        }
        Ok(snap.map(|(b, g)| (b, Some(g.to_string()))))
    }
    fn supports_cas(&self) -> bool {
        true
    }

    fn put_cas(&self, key: &str, bytes: &[u8], expected: Option<&str>) -> Result<CasOutcome> {
        if !self.cas {
            return Ok(CasOutcome::Unsupported); // forces the advisory fallback
        }
        let mut s = self.state.lock().unwrap();
        let current = s.objects.get(key).map(|(_, g)| g.to_string());
        let matches = match (expected, current.as_deref()) {
            (None, None) => true,         // create-if-absent: absent → write
            (None, Some(_)) => false,     // create-if-absent: present → fail
            (Some(t), Some(c)) => t == c, // conditional overwrite: tokens must match
            (Some(_), None) => false,     // expected a version but it is gone → fail
        };
        if !matches {
            return Ok(CasOutcome::Stale);
        }
        s.next_gen += 1;
        let g = s.next_gen;
        s.objects.insert(key.to_string(), (bytes.to_vec(), g));
        Ok(CasOutcome::Written(Some(g.to_string())))
    }
    fn try_lock(&self, _key: &str, _ttl: Duration) -> Result<Option<Box<dyn BackendLock>>> {
        anyhow::bail!("no native lock");
    }
    fn has_native_lock(&self) -> bool {
        false
    }
}

#[test]
fn object_lock_is_exclusive_and_releases_on_drop() {
    // Both backends behave the same on the happy path — CAS (race-free) and advisory.
    for cas in [true, false] {
        let backend = MapBackend::arc(cas);
        let ttl = Duration::from_secs(60);

        let guard = object_try_lock(&backend, "lock", ttl).unwrap();
        assert!(guard.is_some(), "first acquire wins (cas={cas})");
        // A live holder → contention returns Ok(None), never an error.
        assert!(object_try_lock(&backend, "lock", ttl).unwrap().is_none());
        // Dropping the guard deletes the lock object, freeing it.
        drop(guard);
        assert!(backend.get("lock").unwrap().is_none(), "released on drop");
        assert!(object_try_lock(&backend, "lock", ttl).unwrap().is_some());
    }
}

#[test]
fn object_lock_reclaims_a_stale_holder() {
    for cas in [true, false] {
        let backend = MapBackend::arc(cas);
        // Plant a lock stamped far in the past (a crashed holder).
        backend.put("lock", b"1").unwrap();
        // With a zero TTL every existing lock is already stale → reclaimable. On a CAS backend
        // this goes through the conditional `put_cas` reclaim; advisory backends overwrite.
        let guard = object_try_lock(&backend, "lock", Duration::from_secs(0)).unwrap();
        assert!(guard.is_some(), "stale lock reclaimed (cas={cas})");
    }
}

#[test]
fn object_lock_does_not_steal_a_lease_renewed_in_the_reclaim_gap() {
    // The TOCTOU fix (nidus-5kj): a reclaimer reads a stale lock's token, but the holder renews
    // (a fresh stamp + token) in the gap before the reclaiming write. The CAS-gated reclaim must
    // refuse rather than delete-and-overwrite a lease that came back to life.
    let renewed = b"9999999999 live-holder".to_vec();
    let backend = MapBackend::arc_injecting(true, Some(renewed.clone()));
    // Plant a stale lock (epoch stamp → already past any TTL).
    backend.put("lock", b"0 old-holder").unwrap();
    // ttl 60s: the planted stamp is ancient, so the reclaim is attempted — but the injected
    // renew moves the token first, so the conditional write loses the race.
    let got = object_try_lock(&backend, "lock", Duration::from_secs(60)).unwrap();
    assert!(
        got.is_none(),
        "must not steal a lease renewed in the reclaim gap"
    );
    assert_eq!(
        backend.get("lock").unwrap().as_deref(),
        Some(renewed.as_slice()),
        "the holder's renewed lease must survive the refused reclaim",
    );
}

// ── Cluster writer lease (pure/in-RAM, Miri-clean) ──────────────────────────────

#[test]
fn cluster_lease_excludes_renews_and_releases() {
    for cas in [true, false] {
        let backend = MapBackend::arc(cas);
        let ttl = Duration::from_secs(60);

        let lease = ClusterLease::acquire(&backend, "lock", ttl)
            .unwrap()
            .expect("first acquire wins");
        // A second acquire while it is live → contention (Ok(None), not an error).
        assert!(
            ClusterLease::acquire(&backend, "lock", ttl)
                .unwrap()
                .is_none()
        );
        // Renewing keeps it ours (the op-driven heartbeat) — never errors while we own it.
        lease.renew().unwrap();
        lease.renew().unwrap();
        // Drop releases the lease object so a fresh writer can take it.
        drop(lease);
        assert!(backend.get("lock").unwrap().is_none(), "released on drop");
        assert!(
            ClusterLease::acquire(&backend, "lock", ttl)
                .unwrap()
                .is_some()
        );
    }
}

#[test]
fn cluster_lease_renew_fences_a_superseded_writer() {
    let backend = MapBackend::arc(true);
    let lease = ClusterLease::acquire(&backend, "lock", Duration::from_secs(60))
        .unwrap()
        .unwrap();
    // A peer takes over (a fresh stamp under a different owner).
    backend.put("lock", b"9999999999 other-owner").unwrap();
    // The superseded writer's next renew detects it and refuses — the fence.
    let err = lease
        .renew()
        .expect_err("a superseded lease must fail to renew");
    assert!(err.to_string().contains("lease lost"), "{err}");
    // And dropping the fenced lease must NOT delete the peer's lease object.
    drop(lease);
    assert_eq!(
        backend.get("lock").unwrap().as_deref(),
        Some(&b"9999999999 other-owner"[..])
    );
}

#[test]
fn cluster_lease_renew_reclaims_a_vanished_lease() {
    for cas in [true, false] {
        let backend = MapBackend::arc(cas);
        let lease = ClusterLease::acquire(&backend, "lock", Duration::from_secs(60))
            .unwrap()
            .unwrap();
        // The lease object disappears (e.g. a peer found it stale and deleted it, but no one
        // re-created it). Renew should re-establish our ownership rather than error.
        backend.delete("lock").unwrap();
        lease.renew().unwrap();
        assert!(
            backend.get("lock").unwrap().is_some(),
            "lease re-created on renew"
        );
    }
}

/// **A live lease must never read as stale** just because stamps are truncated to whole
/// seconds (nidus-lp4.7).
#[test]
fn a_lease_stays_live_for_its_whole_ttl_despite_truncated_stamps() {
    use super::object::is_live;

    let ttl = Duration::from_secs(2);
    // Stamped at 1, read at 3 → age 2. As little as 1.02s may actually have elapsed
    // (renewed at t=1.99, read at t=3.01), so this lease must still count as live.
    assert!(
        is_live(3, 1, ttl),
        "age == ttl must stay live: truncation can inflate the age by nearly a second"
    );
    assert!(is_live(1, 1, ttl), "a just-stamped lease is live");
    // Once the age is *certainly* past the TTL, it is reclaimable — failover still happens.
    assert!(!is_live(4, 1, ttl), "age > ttl is reclaimable");
    assert!(!is_live(9_999, 1, ttl), "an ancient stamp is reclaimable");
}

/// **A renewal must not steal back a lease a peer already reclaimed** (nidus-lp4.7).
#[test]
fn cluster_lease_renewal_cannot_steal_back_a_lease_a_peer_reclaimed() {
    let backend = MapBackend::concrete(true, None);
    let dyn_backend: Arc<dyn Persistence> = backend.clone();
    let lease = ClusterLease::acquire(&dyn_backend, "lock", Duration::from_secs(60))
        .unwrap()
        .expect("first acquire wins");

    // A peer reclaims the lease in the read→write gap of our renewal.
    let peer = b"9999999999 peer-owner".to_vec();
    backend.inject_next_read(peer.clone());

    let err = lease
        .renew()
        .expect_err("a renewal that lost the CAS to a peer must fail, not overwrite it");
    assert!(
        is_lease_lost(&err),
        "losing the lease is definitive, not transient: {err:#}"
    );
    assert_eq!(
        backend.get("lock").unwrap().as_deref(),
        Some(peer.as_slice()),
        "the peer's lease must survive — the renewal must not have stamped ours back over it",
    );
}

/// The flip side: an instance's **own** concurrent renewal must not fence it (nidus-lp4.7).
#[test]
fn cluster_lease_renewal_tolerates_this_instances_own_concurrent_renewal() {
    let backend = MapBackend::concrete(true, None);
    let dyn_backend: Arc<dyn Persistence> = backend.clone();
    let lease = ClusterLease::acquire(&dyn_backend, "lock", Duration::from_secs(60))
        .unwrap()
        .unwrap();

    // Our other renewer re-stamps in the gap: same owner, fresh stamp.
    backend.inject_next_read(format!("9999999999 {}", lease.owner()).into_bytes());

    lease
        .renew()
        .expect("losing the CAS to our own renewal is not a lost lease");
}

/// **A transient backend error is not a lost lease** (nidus-lp4.7).
#[test]
fn a_transient_backend_error_is_not_a_lost_lease() {
    let backend = MapBackend::concrete(true, None);
    let dyn_backend: Arc<dyn Persistence> = backend.clone();
    let lease = ClusterLease::acquire(&dyn_backend, "lock", Duration::from_secs(60))
        .unwrap()
        .unwrap();

    backend.fail_reads(true);
    let err = lease
        .renew()
        .expect_err("the read failed, so renewal fails");
    assert!(
        !is_lease_lost(&err),
        "a dropped connection says nothing about who holds the lease: {err:#}"
    );

    // And a renewer must not latch the store's fence on it.
    let fenced = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let renewer = lease.renewer(fenced.clone());
    assert!(renewer.renew().is_err());
    assert!(
        !fenced.load(std::sync::atomic::Ordering::Acquire),
        "a transient error must not fence a healthy writer"
    );

    // Recovering the backend recovers the writer — no restart needed.
    backend.fail_reads(false);
    renewer.renew().expect("still ours once the blip passes");
    assert!(!fenced.load(std::sync::atomic::Ordering::Acquire));
}

/// **A lease lost on a *background* renewal must latch the shared fence** (nidus-lp4.7).
#[test]
fn a_background_renewal_that_loses_the_lease_latches_the_shared_fence() {
    let backend = MapBackend::arc(true);
    let lease = ClusterLease::acquire(&backend, "lock", Duration::from_secs(60))
        .unwrap()
        .unwrap();
    let fenced = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let renewer = lease.renewer(fenced.clone());
    renewer.renew().expect("ours to begin with");
    assert!(!fenced.load(std::sync::atomic::Ordering::Acquire));

    // A peer takes over.
    backend.put("lock", b"9999999999 peer-owner").unwrap();

    let err = renewer.renew().expect_err("superseded");
    assert!(is_lease_lost(&err), "{err:#}");
    assert!(
        fenced.load(std::sync::atomic::Ordering::Acquire),
        "the store's fence must be latched by the background renewer, not only by a write"
    );
}

// ── LocalFs object ops (file-backed, Miri-ignored) ──────────────────────────────

#[cfg_attr(miri, ignore)]
#[test]
fn local_fs_object_round_trip_and_list() {
    let dir = tempfile::tempdir().unwrap();
    let fs = LocalFs::new(dir.path()).unwrap();

    assert!(fs.get("data").unwrap().is_none());
    assert!(fs.list().unwrap().is_empty());

    fs.put("data", b"\x01\x02\x03").unwrap();
    fs.put("ann", b"cache").unwrap();
    assert_eq!(fs.get("data").unwrap().as_deref(), Some(&[1u8, 2, 3][..]));
    assert_eq!(
        fs.list().unwrap(),
        vec!["ann".to_string(), "data".to_string()]
    );

    // Overwrite is atomic and replaces.
    fs.put("data", b"new").unwrap();
    assert_eq!(fs.get("data").unwrap().as_deref(), Some(b"new".as_slice()));

    fs.delete("data").unwrap();
    assert!(fs.get("data").unwrap().is_none());
    fs.delete("data").unwrap(); // deleting absent is a no-op
}

#[cfg_attr(miri, ignore)]
#[test]
fn local_fs_rejects_bad_keys() {
    let dir = tempfile::tempdir().unwrap();
    let fs = LocalFs::new(dir.path()).unwrap();
    assert!(fs.get("../escape").is_err());
    assert!(fs.put("a/b", b"x").is_err());
}

#[cfg_attr(miri, ignore)]
#[test]
fn open_persistence_file_scheme_and_bare_path() {
    let dir = tempfile::tempdir().unwrap();
    let bare = dir.path().join("bare");
    let url = format!("file://{}", dir.path().join("urled").display());

    for loc in [bare.display().to_string(), url] {
        let p = open_persistence(&loc).unwrap();
        p.put("k", b"v").unwrap();
        assert_eq!(p.get("k").unwrap().as_deref(), Some(b"v".as_slice()));
    }
}

// ── FileAppender parity with the data/log discipline (Miri-ignored) ─────────────

#[cfg_attr(miri, ignore)]
#[test]
fn appender_append_len_sync_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let fs = LocalFs::new(dir.path()).unwrap();

    {
        let mut ap = fs.appender("log").unwrap().unwrap();
        assert_eq!(ap.len().unwrap(), 0);
        ap.append(b"abc").unwrap();
        ap.append(b"de").unwrap();
        assert_eq!(ap.len().unwrap(), 5);
        ap.sync().unwrap();
    }
    // Reopen positions at the end; read_to_end yields the whole stream.
    let mut ap = fs.appender("log").unwrap().unwrap();
    assert_eq!(ap.len().unwrap(), 5);
    let mut buf = Vec::new();
    ap.read_to_end(&mut buf).unwrap();
    assert_eq!(buf, b"abcde");
    // Appends after a read_to_end still land at the end.
    ap.append(b"fg").unwrap();
    assert_eq!(ap.len().unwrap(), 7);
}

#[cfg_attr(miri, ignore)]
#[test]
fn appender_truncate_to_rolls_back() {
    let dir = tempfile::tempdir().unwrap();
    let fs = LocalFs::new(dir.path()).unwrap();
    let mut ap = fs.appender("data").unwrap().unwrap();
    ap.append(b"0123456789").unwrap();
    let mark = ap.len().unwrap();
    ap.append(b"XXXX").unwrap();
    ap.truncate_to(mark).unwrap();
    assert_eq!(ap.len().unwrap(), mark);
    let mut buf = Vec::new();
    ap.read_to_end(&mut buf).unwrap();
    assert_eq!(buf, b"0123456789");
}

#[cfg_attr(miri, ignore)]
#[test]
fn appender_rewrite_then_append() {
    let dir = tempfile::tempdir().unwrap();
    let fs = LocalFs::new(dir.path()).unwrap();
    let mut ap = fs.appender("data").unwrap().unwrap();
    ap.append(b"original-contents").unwrap();
    ap.sync().unwrap();
    ap.rewrite(b"new").unwrap();
    assert_eq!(ap.len().unwrap(), 3);
    ap.append(b"-tail").unwrap();
    let mut buf = Vec::new();
    ap.read_to_end(&mut buf).unwrap();
    assert_eq!(buf, b"new-tail");

    // The rewrite is durable across reopen, with no leftover temp object.
    drop(ap);
    let mut ap2 = fs.appender("data").unwrap().unwrap();
    let mut buf2 = Vec::new();
    ap2.read_to_end(&mut buf2).unwrap();
    assert_eq!(buf2, b"new-tail");
    assert!(!fs.list().unwrap().iter().any(|k| k.ends_with(".tmp")));
}

// ── LocalFs::try_lock (Miri-ignored) ────────────────────────────────────────────

#[cfg_attr(miri, ignore)]
#[test]
fn local_fs_lock_excludes_then_releases() {
    let dir = tempfile::tempdir().unwrap();
    let fs = LocalFs::new(dir.path()).unwrap();
    let ttl = Duration::from_secs(60);

    let guard = fs.try_lock("lock", ttl).unwrap();
    assert!(guard.is_some(), "first lock should succeed");
    // Contention returns Ok(None), not an error.
    assert!(fs.try_lock("lock", ttl).unwrap().is_none());
    // Releasing the guard frees the lock.
    drop(guard);
    assert!(fs.try_lock("lock", ttl).unwrap().is_some());
}

// ── OpfsFs pool (pure/in-RAM via a fake SyncHandle, Miri-clean) ─────────────────

use super::opfs::test_support::FakeHandle;

fn fake_pool(body_slots: usize) -> Vec<Box<dyn SyncHandle>> {
    (0..=body_slots)
        .map(|_| Box::new(FakeHandle::new()) as Box<dyn SyncHandle>)
        .collect()
}

#[test]
fn opfs_pool_allocates_reuses_and_exhausts_slots() {
    let fs = OpfsFs::adopt(fake_pool(2)).unwrap(); // 1 directory slot + 2 body slots

    fs.put("a", b"1").unwrap();
    fs.put("b", b"2").unwrap();
    let err = fs.put("c", b"3").unwrap_err().to_string();
    assert!(err.contains("exhausted"), "{err}");

    // Freeing a slot lets a new key reuse it.
    fs.delete("a").unwrap();
    fs.put("c", b"3").unwrap();
    assert_eq!(fs.get("c").unwrap().as_deref(), Some(b"3".as_slice()));
    assert!(fs.get("a").unwrap().is_none());
    assert_eq!(fs.list().unwrap(), vec!["b".to_string(), "c".to_string()]);
}

#[test]
fn opfs_directory_map_round_trips_across_reopen() {
    let fakes: Vec<FakeHandle> = (0..=2).map(|_| FakeHandle::new()).collect();
    let as_handles = |fakes: &[FakeHandle]| -> Vec<Box<dyn SyncHandle>> {
        fakes
            .iter()
            .cloned()
            .map(|h| Box::new(h) as Box<dyn SyncHandle>)
            .collect()
    };

    let fs = OpfsFs::adopt(as_handles(&fakes)).unwrap();
    fs.put("k1", b"hello").unwrap();
    fs.put("k2", b"world").unwrap();
    fs.delete("k1").unwrap();
    drop(fs);

    // "Reopen": fresh handles cloned from the same underlying fakes, exactly as a real
    // worker would acquire fresh sync access handles over the same OPFS files.
    let reopened = OpfsFs::adopt(as_handles(&fakes)).unwrap();
    assert_eq!(reopened.list().unwrap(), vec!["k2".to_string()]);
    assert_eq!(
        reopened.get("k2").unwrap().as_deref(),
        Some(b"world".as_slice())
    );
    assert!(reopened.get("k1").unwrap().is_none());
}

#[test]
fn opfs_a_body_write_without_a_map_update_stays_invisible() {
    let fakes: Vec<FakeHandle> = (0..=1).map(|_| FakeHandle::new()).collect();
    let handles: Vec<Box<dyn SyncHandle>> = fakes
        .iter()
        .cloned()
        .map(|h| Box::new(h) as Box<dyn SyncHandle>)
        .collect();
    let fs = OpfsFs::adopt(handles).unwrap();

    // Simulate a crash between step 1 (body write) and step 2 (map write): write
    // straight into the body slot, bypassing `put`, so the directory map never learns
    // about it.
    fakes[1].write_at(0, b"orphaned").unwrap();

    assert!(
        fs.list().unwrap().is_empty(),
        "an orphaned body must not appear in list()"
    );
    assert!(
        fs.get("ghost").unwrap().is_none(),
        "no key names the orphaned slot, so get() finds nothing"
    );
    // The slot is still unreferenced by the map, so a real put safely reuses it.
    fs.put("real", b"data").unwrap();
    assert_eq!(fs.get("real").unwrap().as_deref(), Some(b"data".as_slice()));
}

#[test]
fn opfs_rejects_bad_keys() {
    let fs = OpfsFs::adopt(fake_pool(1)).unwrap();
    assert!(fs.get("../escape").is_err());
    assert!(fs.put("a/b", b"x").is_err());
    assert!(fs.delete("").is_err());
}

#[test]
fn opfs_adopt_rejects_an_empty_pool() {
    assert!(OpfsFs::adopt(Vec::new()).is_err());
}

#[test]
fn opfs_adopt_rejects_a_corrupt_directory_map() {
    let dir = FakeHandle::new();
    dir.write_at(0, b"not a valid directory map frame").unwrap();
    let handles: Vec<Box<dyn SyncHandle>> = vec![Box::new(dir)];
    assert!(OpfsFs::adopt(handles).is_err());
}

#[test]
fn opfs_overwrite_does_not_reuse_the_key_s_own_slot() {
    // `Persistence::put` promises old-or-new, never torn. An in-place overwrite of the
    // key's own slot breaks that on a crash mid-write, so assert the slot MOVES.
    let fakes: Vec<FakeHandle> = (0..3).map(|_| FakeHandle::new()).collect();
    let handles: Vec<Box<dyn SyncHandle>> = fakes
        .iter()
        .cloned()
        .map(|h| Box::new(h) as Box<dyn SyncHandle>)
        .collect();
    let fs = OpfsFs::adopt(handles).unwrap();
    fs.put("k", b"old").unwrap();
    let first = fakes.iter().position(|h| h.bytes() == b"old").unwrap();
    fs.put("k", b"new").unwrap();
    let second = fakes.iter().position(|h| h.bytes() == b"new").unwrap();
    assert_ne!(
        first, second,
        "overwrite reused the slot it was reading from"
    );
    assert_eq!(fs.get("k").unwrap().as_deref(), Some(b"new".as_slice()));
    // The old bytes are still on disk, unreferenced — that is what makes the put atomic.
    assert_eq!(fakes[first].bytes(), b"old");
}

#[test]
fn opfs_try_lock_is_trivially_held_and_reports_reality() {
    let fs = OpfsFs::adopt(fake_pool(0)).unwrap();
    assert!(
        fs.try_lock("lock", Duration::from_secs(1))
            .unwrap()
            .is_some()
    );
    // The two capability flags are trait defaults, so assert them on the `Persistence`
    // a caller actually gets (the fieldless proxy), not on the pool behind it.
    std::thread::spawn(|| {
        super::opfs::register_pool(OpfsFs::adopt(fake_pool(0)).unwrap());
        let p = super::opfs::open_registered("s").unwrap();
        assert!(p.has_native_lock(), "an OPFS sync handle is exclusive");
        assert!(!p.supports_cas(), "no second writer to fence in one worker");
    })
    .join()
    .unwrap();
}

#[test]
fn opfs_grow_increases_capacity_for_new_writes() {
    let fs = OpfsFs::adopt(fake_pool(0)).unwrap(); // directory slot only
    assert_eq!(fs.capacity(), 1);
    assert!(fs.put("a", b"1").is_err(), "no body slots yet");

    fs.grow(vec![Box::new(FakeHandle::new()) as Box<dyn SyncHandle>])
        .unwrap();
    assert_eq!(fs.capacity(), 2);
    fs.put("a", b"1").unwrap();
    assert_eq!(fs.get("a").unwrap().as_deref(), Some(b"1".as_slice()));
}

#[test]
fn open_persistence_opfs_scheme_is_wasm_only_on_native() {
    // `Box<dyn Persistence>` is not Debug, so `unwrap_err` will not do here.
    let Err(err) = open_persistence("opfs://anything") else {
        panic!("opfs:// resolved on a native target");
    };
    let err = err.to_string();
    assert!(err.contains("wasm32"), "{err}");
}

// ── OPFS registry handoff (pure/in-RAM, Miri-clean) ─────────────────────────────
// Each case spawns a fresh OS thread, so its `thread_local` registry always starts
// empty regardless of how the test harness reuses threads across other tests.

#[test]
fn opfs_open_registered_without_a_pool_names_the_init_call() {
    let err = std::thread::spawn(|| match super::opfs::open_registered("mystore") {
        Err(e) => e.to_string(),
        Ok(_) => panic!("resolved an opfs:// store with no registered pool"),
    })
    .join()
    .unwrap();
    assert!(err.contains("register_pool"), "{err}");
}

#[test]
fn opfs_open_registered_resolves_after_register_pool() {
    std::thread::spawn(|| {
        assert!(super::opfs::open_registered("mystore").is_err());
        super::opfs::register_pool(OpfsFs::adopt(fake_pool(1)).unwrap());
        let p = super::opfs::open_registered("mystore").unwrap();
        p.put("k", b"v").unwrap();
        assert_eq!(p.get("k").unwrap().as_deref(), Some(b"v".as_slice()));
    })
    .join()
    .unwrap();
}

#[test]
fn opfs_grow_pool_adds_capacity_for_new_keys() {
    std::thread::spawn(|| {
        super::opfs::register_pool(OpfsFs::adopt(fake_pool(0)).unwrap()); // no body slots yet
        let p = super::opfs::open_registered("s").unwrap();
        assert!(p.put("a", b"1").is_err(), "no body slots yet");
        grow_pool(vec![Box::new(FakeHandle::new())]).unwrap();
        p.put("a", b"1").unwrap();
        assert_eq!(p.get("a").unwrap().as_deref(), Some(b"1".as_slice()));
    })
    .join()
    .unwrap();
}

#[test]
fn opfs_grow_pool_errors_without_a_registered_pool() {
    std::thread::spawn(|| {
        let err = grow_pool(vec![Box::new(FakeHandle::new())])
            .unwrap_err()
            .to_string();
        assert!(err.contains("register_pool"), "{err}");
    })
    .join()
    .unwrap();
}

// ── wasm-only: cloud backends must not silently fall back (nidus-y67) ──────────

#[cfg(target_family = "wasm")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn wasm_rejects_cloud_persistence_and_memory_tier_backends() {
    // Neither `Box<dyn Persistence>` nor `Box<dyn MemoryTier>` is Debug, so `unwrap_err`
    // will not compile here.
    let Err(err) = open_persistence("s3://bucket/key") else {
        panic!("s3:// resolved on wasm32");
    };
    assert!(err.to_string().contains("wasm32"), "{err}");
    let Err(err) = open_memory_tier("redis://host:6379") else {
        panic!("redis:// resolved on wasm32");
    };
    assert!(err.to_string().contains("wasm32"), "{err}");
}
