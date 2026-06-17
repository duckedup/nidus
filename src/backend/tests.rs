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
fn gcs_scheme_is_clear_not_yet() {
    for loc in ["gs://bucket/p", "gcs://bucket/p"] {
        let err = open_persistence(loc).err().unwrap().to_string();
        assert!(err.contains("not yet implemented"), "{loc}: {err}");
        assert!(err.contains("Phase 3"), "{loc}: {err}");
    }
}

#[test]
fn memory_tier_remote_schemes_are_clear_not_yet() {
    for loc in [
        "redis://h:6379",
        "rediss://h",
        "valkey://h",
        "memcache://h",
        "memcached://h",
    ] {
        let err = open_memory_tier(loc).err().unwrap().to_string();
        assert!(err.contains("not yet implemented"), "{loc}: {err}");
        assert!(err.contains("Phase 2"), "{loc}: {err}");
    }
}

#[test]
fn memory_tier_unknown_scheme_errors() {
    let err = open_memory_tier("kafka://h").err().unwrap().to_string();
    assert!(err.contains("unknown memory-tier location"), "{err}");
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
