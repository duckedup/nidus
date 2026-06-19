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

/// A whole-object [`Persistence`] backed by an in-RAM map, parameterised by whether it
/// offers the atomic create-if-absent primitive — so one type exercises both the
/// race-free conditional-create lock path and the advisory get-then-put fallback.
struct MapBackend {
    objects: Mutex<HashMap<String, Vec<u8>>>,
    atomic_create: bool,
}

impl MapBackend {
    fn arc(atomic_create: bool) -> Arc<dyn Persistence> {
        Arc::new(MapBackend {
            objects: Mutex::new(HashMap::new()),
            atomic_create,
        })
    }
}

impl Persistence for MapBackend {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.objects.lock().unwrap().get(key).cloned())
    }
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        self.objects
            .lock()
            .unwrap()
            .insert(key.to_string(), bytes.to_vec());
        Ok(())
    }
    fn delete(&self, key: &str) -> Result<()> {
        self.objects.lock().unwrap().remove(key);
        Ok(())
    }
    fn list(&self) -> Result<Vec<String>> {
        Ok(self.objects.lock().unwrap().keys().cloned().collect())
    }
    fn try_create_exclusive(&self, key: &str, bytes: &[u8]) -> Result<Option<bool>> {
        if !self.atomic_create {
            return Ok(None); // forces the advisory fallback
        }
        let mut map = self.objects.lock().unwrap();
        if map.contains_key(key) {
            Ok(Some(false)) // already exists — lost the race
        } else {
            map.insert(key.to_string(), bytes.to_vec());
            Ok(Some(true)) // created
        }
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
    // Both backends behave the same on the happy path — race-free and advisory.
    for atomic in [true, false] {
        let backend = MapBackend::arc(atomic);
        let ttl = Duration::from_secs(60);

        let guard = object_try_lock(&backend, "lock", ttl).unwrap();
        assert!(guard.is_some(), "first acquire wins (atomic={atomic})");
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
    for atomic in [true, false] {
        let backend = MapBackend::arc(atomic);
        // Plant a lock stamped far in the past (a crashed holder).
        backend.put("lock", b"1").unwrap();
        // With a zero TTL every existing lock is already stale → reclaimable.
        let guard = object_try_lock(&backend, "lock", Duration::from_secs(0)).unwrap();
        assert!(guard.is_some(), "stale lock reclaimed (atomic={atomic})");
    }
}

// ── Cluster writer lease (pure/in-RAM, Miri-clean) ──────────────────────────────

#[test]
fn cluster_lease_excludes_renews_and_releases() {
    for atomic in [true, false] {
        let backend = MapBackend::arc(atomic);
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
    for atomic in [true, false] {
        let backend = MapBackend::arc(atomic);
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
