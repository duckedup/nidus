use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::Record;
use crate::backend::{MemoryTier, Persistence, open_persistence};
use crate::store::Store;

#[test]
fn name_validation_rejects_path_like_and_reserved_names() {
    assert!(validate_name("tenant-1").is_ok());
    assert!(validate_name("").is_err());
    assert!(validate_name(".").is_err());
    assert!(validate_name("..").is_err());
    assert!(validate_name("a/b").is_err());
    assert!(validate_name("a\\b").is_err());
    assert!(validate_name("a b").is_err());
}

#[test]
fn get_rejects_an_invalid_name_rather_than_sanitising() {
    let mut ns = Namespaces::new(Config::new("unused", 3), "unused-base");
    assert!(ns.get("../escape").is_err());
}

#[test]
fn memory_url_rewrite_gives_each_namespace_its_own_prefix() {
    assert_eq!(
        namespaced_memory_url("redis://host/?prefix=app", "ns1"),
        Some("redis://host/?prefix=app:ns1".to_string())
    );
    assert_eq!(
        namespaced_memory_url("redis://host", "ns2"),
        Some("redis://host?prefix=ns2".to_string())
    );
    assert_eq!(namespaced_memory_url("local", "ns"), None);
    assert_eq!(namespaced_memory_url("", "ns"), None);
}

#[test]
#[cfg_attr(miri, ignore)] // LocalFs fsyncs on open
fn get_never_touches_a_sibling_namespace() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let a_dir = base.join("a");
    std::fs::create_dir_all(&a_dir).unwrap();
    // A manifest that fails to decode: if opening "b" ever touched "a" too — the shape a
    // "list the base and open everything" bug would take — this surfaces as an error.
    std::fs::write(a_dir.join(crate::manifest::MANIFEST_KEY), b"not a manifest").unwrap();

    let mut ns = Namespaces::new(
        Config::new("unused", 3),
        base.to_string_lossy().into_owned(),
    );
    let got = ns.get("b");
    assert!(got.is_ok(), "opening \"b\" touched \"a\": {:?}", got.err());
}

#[test]
#[cfg_attr(miri, ignore)] // upsert fsyncs
fn evicting_a_namespace_releases_its_writer_lock() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    {
        let mut db = Nidus::open_dir(base.join("a"), 3).unwrap();
        db.create_collection("c").unwrap();
        db.upsert(
            "c",
            &[Record::new("1", vec![1.0, 0.0, 0.0], BTreeMap::new())],
        )
        .unwrap();
        db.flush().unwrap();
    }

    let mut ns = Namespaces::new(
        Config::new("unused", 3),
        base.to_string_lossy().into_owned(),
    )
    .budget_bytes(1);
    ns.get("a").unwrap();
    assert_eq!(ns.warm(), vec!["a".to_string()]);

    ns.get("b").unwrap();
    assert!(!ns.warm().contains(&"a".to_string()));

    let reopened = Nidus::open(Config::new(base.join("a"), 3));
    assert!(
        reopened.is_ok(),
        "a's writer lock was not released on eviction: {:?}",
        reopened.err()
    );
}

#[test]
#[cfg_attr(miri, ignore)] // upsert fsyncs
fn budget_evicts_by_bytes_not_by_count() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    for (name, rows) in [("small1", 1usize), ("small2", 1), ("big", 100)] {
        let mut db = Nidus::open_dir(base.join(name), 3).unwrap();
        db.create_collection("c").unwrap();
        let records: Vec<Record> = (0..rows)
            .map(|i| Record::new(i.to_string(), vec![i as f32, 0.0, 0.0], BTreeMap::new()))
            .collect();
        db.upsert("c", &records).unwrap();
        db.flush().unwrap();
    }

    let mut ns = Namespaces::new(
        Config::new("unused", 3),
        base.to_string_lossy().into_owned(),
    )
    .budget_bytes(100);
    ns.get("small1").unwrap();
    ns.get("small2").unwrap();
    assert_eq!(ns.warm_bytes(), 24);

    ns.get("big").unwrap();
    let warm = ns.warm();
    assert!(!warm.contains(&"small1".to_string()));
    assert!(!warm.contains(&"small2".to_string()));
    assert!(warm.contains(&"big".to_string()));
}

/// A shared in-RAM tier with no namespacing of its own (the same shape as `LocalRam`) —
/// reachable from two "stores" at once, so `derive_config`'s own prefix rewrite is what
/// has to keep them apart, not this double.
#[derive(Default)]
struct SharedRam {
    objects: Mutex<HashMap<String, Vec<u8>>>,
}

/// One namespace's view of a [`SharedRam`], mirroring `RedisTier`'s own `<prefix>:<key>`
/// scheme (`src/backend/redis.rs:60-68`).
struct SharedRamView {
    shared: Arc<SharedRam>,
    prefix: String,
}

impl MemoryTier for SharedRamView {
    fn load(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let k = format!("{}:{}", self.prefix, key);
        Ok(self.shared.objects.lock().unwrap().get(&k).cloned())
    }

    fn store(&self, key: &str, bytes: &[u8], _ttl: Option<Duration>) -> Result<()> {
        let k = format!("{}:{}", self.prefix, key);
        self.shared
            .objects
            .lock()
            .unwrap()
            .insert(k, bytes.to_vec());
        Ok(())
    }
}

/// Pulls `?prefix=<value>` back out of a memory-tier URL `derive_config` produced.
fn extract_prefix(memory: &str) -> Option<String> {
    let (_, query) = memory.split_once('?')?;
    query
        .split('&')
        .find_map(|pair| pair.strip_prefix("prefix=").map(str::to_string))
}

/// Opens `cfg` over its own local directory but a shared, namespaced memory tier —
/// `Store::open_with`'s injection seam, reached directly so the test controls the tier.
fn open_store(cfg: &Config, shared: &Arc<SharedRam>, prefix: &str) -> Nidus {
    let location = cfg.path.to_string_lossy().into_owned();
    let persistence: Arc<dyn Persistence> = open_persistence(&location).unwrap().into();
    let memory: Box<dyn MemoryTier> = Box::new(SharedRamView {
        shared: shared.clone(),
        prefix: prefix.to_string(),
    });
    let store = Store::open_with(cfg.clone(), &location, persistence, Some(memory)).unwrap();
    Nidus { store }
}

/// Equal op count, row count and `tag` byte length across namespaces, so every `try_adopt`
/// guard matches (memtier.rs:86) — the only state contamination is detectable in. The `tag`
/// carries the difference because an adopted working set supplies attrs, never vectors.
fn open_and_write(cfg: &Config, shared: &Arc<SharedRam>, prefix: &str, tag: &str) {
    let mut db = open_store(cfg, shared, prefix);
    db.create_collection("c").unwrap();
    let mut attrs = BTreeMap::new();
    attrs.insert("tag".to_string(), crate::model::Value::Str(tag.to_string()));
    db.upsert("c", &[Record::new("only", vec![1.0, 0.0, 0.0], attrs)])
        .unwrap();
    db.flush().unwrap();
}

#[test]
#[cfg_attr(miri, ignore)] // upsert fsyncs
fn memory_tier_reopen_returns_its_own_rows_not_a_siblings() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base").to_string_lossy().into_owned();
    let template = Config::new("unused", 3).memory("redis://fake-host/?prefix=app");

    let cfg_a = derive_config(&template, &base, "a");
    let cfg_b = derive_config(&template, &base, "b");
    let prefix_a = extract_prefix(&cfg_a.memory).expect("redis-family memory keeps a prefix");
    let prefix_b = extract_prefix(&cfg_b.memory).expect("redis-family memory keeps a prefix");

    let shared = Arc::new(SharedRam::default());
    open_and_write(&cfg_a, &shared, &prefix_a, "aaa");
    open_and_write(&cfg_b, &shared, &prefix_b, "bbb");

    // b published last, so without a per-namespace prefix its working set is what sits under
    // the shared `workingset` key and a's reopen adopts it.
    let reopened_a = open_store(&cfg_a, &shared, &prefix_a);
    let tag_a = reopened_a.get("c", "only").expect("a's own row").attrs["tag"].clone();
    assert_eq!(
        tag_a,
        crate::model::Value::Str("aaa".to_string()),
        "a adopted a sibling's working set"
    );

    let reopened_b = open_store(&cfg_b, &shared, &prefix_b);
    let tag_b = reopened_b.get("c", "only").expect("b's own row").attrs["tag"].clone();
    assert_eq!(
        tag_b,
        crate::model::Value::Str("bbb".to_string()),
        "b adopted a sibling's working set"
    );

    assert_ne!(
        prefix_a, prefix_b,
        "two namespaces derived the same memory-tier prefix"
    );
}

/// Eviction must flush first or an `Fsync::OnFlush` caller loses writes made through the
/// handle (SPEC §13.9). Asserted on a per-store artifact, not the process-global barrier
/// counter, which concurrent tests also bump.
#[test]
#[cfg_attr(miri, ignore)] // upsert fsyncs
fn eviction_flushes_before_dropping_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    // `history_versions` is what makes a flush republish the manifest under OnFlush
    // (`note_commit_point`), giving the flush an on-disk trace this test can read.
    let template = Config::new("unused", 3)
        .fsync(crate::config::Fsync::OnFlush)
        .history_versions(Some(4));
    let mut ns = Namespaces::new(template, base.to_string_lossy().into_owned());

    let db = ns.get("a").unwrap();
    db.create_collection("c").unwrap();
    db.upsert(
        "c",
        &[Record::new("1", vec![1.0, 0.0, 0.0], BTreeMap::new())],
    )
    .unwrap();

    let manifest = base.join("a").join(crate::manifest::MANIFEST_KEY);
    let before = std::fs::read(&manifest).unwrap();

    assert!(ns.evict("a").unwrap(), "a was warm");

    let after = std::fs::read(&manifest).unwrap();
    assert_ne!(
        before, after,
        "evicting a dropped the store without flushing it: the manifest was never republished"
    );
}
