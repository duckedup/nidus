//! Black-box tests for `Namespaces` (nidus-pcpc.1) against a real filesystem, through the
//! shipped public API only. Unit 1's inline tests reach `Store::open_with`'s `pub(crate)`
//! injection seam to prove laziness, lock release, byte budgets, and memory isolation; this
//! file proves only what a `tests/` caller can see: per-namespace directories on disk, that
//! a fresh handle reads back what an earlier one wrote, that eviction is not data loss, and
//! that a bad name is rejected rather than sanitised.

use std::collections::BTreeMap;

use nidus::{Config, Namespaces, Record};

/// Opens (or reopens) `name` through `ns`, writes one record into collection `"c"`, and
/// flushes it durably.
fn write_one(ns: &mut Namespaces, name: &str, vector: Vec<f32>) {
    let db = ns.get(name).expect("namespace should open");
    db.create_collection("c").unwrap();
    db.upsert("c", &[Record::new("only", vector, BTreeMap::new())])
        .unwrap();
    db.flush().unwrap();
}

#[test]
#[cfg_attr(miri, ignore)] // LocalFs fsyncs on upsert/flush
fn each_namespace_lands_in_its_own_subdirectory() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");

    let mut ns = Namespaces::new(
        Config::new("unused", 3),
        base.to_string_lossy().into_owned(),
    );
    write_one(&mut ns, "tenant-a", vec![1.0, 0.0, 0.0]);
    write_one(&mut ns, "tenant-b", vec![0.0, 1.0, 0.0]);

    for name in ["tenant-a", "tenant-b"] {
        let sub = base.join(name);
        assert!(
            sub.is_dir(),
            "{name} did not get its own subdirectory under base"
        );
        for object in ["manifest", "log", "data"] {
            assert!(
                sub.join(object).is_file(),
                "{name}/{object} missing: namespace was not persisted as its own store"
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // LocalFs fsyncs on upsert/flush
fn a_fresh_handle_over_the_same_base_reads_back_both_namespaces() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    {
        let mut ns = Namespaces::new(
            Config::new("unused", 3),
            base.to_string_lossy().into_owned(),
        );
        write_one(&mut ns, "tenant-a", vec![1.0, 0.0, 0.0]);
        write_one(&mut ns, "tenant-b", vec![0.0, 1.0, 0.0]);
        // ns dropped here: both writer locks released before the fresh handle opens.
    }

    let mut ns = Namespaces::new(
        Config::new("unused", 3),
        base.to_string_lossy().into_owned(),
    );
    let a = ns
        .get("tenant-a")
        .unwrap()
        .get("c", "only")
        .expect("tenant-a's row");
    assert_eq!(a.vector, Some(vec![1.0, 0.0, 0.0]));
    let b = ns
        .get("tenant-b")
        .unwrap()
        .get("c", "only")
        .expect("tenant-b's row");
    assert_eq!(b.vector, Some(vec![0.0, 1.0, 0.0]));
}

#[test]
#[cfg_attr(miri, ignore)] // LocalFs fsyncs on upsert/flush
fn a_namespace_evicted_by_the_byte_budget_still_returns_its_records() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");

    let mut ns = Namespaces::new(
        Config::new("unused", 3),
        base.to_string_lossy().into_owned(),
    )
    .budget_bytes(1);
    write_one(&mut ns, "tenant-a", vec![1.0, 0.0, 0.0]);
    assert_eq!(ns.warm(), vec!["tenant-a".to_string()]);

    write_one(&mut ns, "tenant-b", vec![0.0, 1.0, 0.0]);
    assert!(
        !ns.warm().contains(&"tenant-a".to_string()),
        "tenant-a should have been pushed out of the warm set by the 1-byte budget"
    );

    let rec = ns
        .get("tenant-a")
        .unwrap()
        .get("c", "only")
        .expect("tenant-a's row survives eviction and reload");
    assert_eq!(rec.vector, Some(vec![1.0, 0.0, 0.0]));
}

#[test]
fn an_invalid_namespace_name_is_an_error_not_a_sanitised_directory() {
    let mut ns = Namespaces::new(Config::new("unused", 3), "unused-base");
    assert!(ns.get("..").is_err());
    assert!(ns.get("a/b").is_err());
    assert!(ns.get("").is_err());
}
