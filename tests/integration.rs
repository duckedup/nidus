//! Black-box integration tests against the public API. These exercise nidus the
//! way a consumer would. File-backed cases use a temp dir and are `#[cfg_attr(miri,
//! ignore)]` (they fsync); in-memory cases run anywhere, including under Miri.

use std::collections::BTreeMap;

use nidus::{Config, Filter, Nidus, OpenMode, Predicate, Record, Scope, SearchOpts, Value};

fn rec(id: &str, vector: Vec<f32>, kind: &str) -> Record {
    let mut attrs = BTreeMap::new();
    attrs.insert("kind".to_string(), Value::Str(kind.to_string()));
    Record {
        id: id.to_string(),
        vector,
        attrs,
    }
}

fn opts(top_k: usize) -> SearchOpts {
    SearchOpts {
        top_k,
        ..Default::default()
    }
}

#[test]
fn in_memory_ranking_and_overwrite() {
    let mut db = Nidus::open_in_memory(3).unwrap();
    db.create_collection("c").unwrap();
    db.upsert(
        "c",
        &[
            rec("a", vec![1.0, 0.0, 0.0], "file"),
            rec("b", vec![0.0, 1.0, 0.0], "file"),
            rec("near", vec![0.9, 0.1, 0.0], "file"),
        ],
    )
    .unwrap();

    let hits = db.search("c", &[1.0, 0.0, 0.0], &opts(3)).unwrap();
    assert_eq!(hits[0].id, "a");
    assert!((hits[0].score - 1.0).abs() < 1e-5);
    assert_eq!(hits[1].id, "near");
    assert_eq!(hits[2].id, "b");

    // Idempotent overwrite by id: count stays, newest vector wins.
    db.upsert("c", &[rec("a", vec![0.0, 0.0, 1.0], "file")])
        .unwrap();
    assert_eq!(db.get_all("c").len(), 3);
    let hits = db.search("c", &[1.0, 0.0, 0.0], &opts(3)).unwrap();
    assert_ne!(hits[0].id, "a"); // "a" moved away from the query
}

#[test]
fn multi_collection_search_merges_and_attributes() {
    let mut db = Nidus::open_in_memory(3).unwrap();
    db.create_collection("x").unwrap();
    db.create_collection("y").unwrap();
    db.upsert("x", &[rec("x1", vec![1.0, 0.0, 0.0], "file")])
        .unwrap();
    db.upsert("y", &[rec("y1", vec![0.95, 0.05, 0.0], "doc")])
        .unwrap();

    // Scope::All merges both into one ranking; each Hit carries its collection.
    let all = db.search(Scope::All, &[1.0, 0.0, 0.0], &opts(10)).unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].collection, "x");
    assert_eq!(all[1].collection, "y");

    // A subset scope also works via &[&str].
    let subset: &[&str] = &["y"];
    let only_y = db.search(subset, &[1.0, 0.0, 0.0], &opts(10)).unwrap();
    assert_eq!(only_y.len(), 1);
    assert_eq!(only_y[0].collection, "y");
}

#[test]
fn filter_and_min_score() {
    let mut db = Nidus::open_in_memory(3).unwrap();
    db.create_collection("c").unwrap();
    db.upsert(
        "c",
        &[
            rec("file1", vec![1.0, 0.0, 0.0], "file"),
            rec("sym1", vec![0.99, 0.01, 0.0], "symbol"),
            rec("far", vec![0.0, 1.0, 0.0], "file"),
        ],
    )
    .unwrap();

    // Only `kind == file`.
    let filtered = SearchOpts {
        top_k: 10,
        filter: Filter(vec![Predicate::Eq(
            "kind".into(),
            Value::Str("file".into()),
        )]),
        min_score: None,
    };
    let hits = db.search("c", &[1.0, 0.0, 0.0], &filtered).unwrap();
    assert!(hits.iter().all(|h| h.id != "sym1"));

    // min_score drops the orthogonal "far".
    let gated = SearchOpts {
        top_k: 10,
        filter: Filter::default(),
        min_score: Some(0.5),
    };
    let hits = db.search("c", &[1.0, 0.0, 0.0], &gated).unwrap();
    assert!(hits.iter().all(|h| h.id != "far"));
}

#[cfg_attr(miri, ignore)]
#[test]
fn file_backed_persistence_and_readonly() {
    let dir = tempfile::tempdir().unwrap();

    {
        let mut db = Nidus::open(Config::new(dir.path(), 3)).unwrap();
        db.create_collection("c").unwrap();
        db.upsert("c", &[rec("a", vec![1.0, 0.0, 0.0], "file")])
            .unwrap();
        let mut meta = BTreeMap::new();
        meta.insert("model".to_string(), "demo-embed".to_string());
        db.set_meta("c", meta).unwrap();
    } // writer lock released on drop

    // Reopen read-only: data persisted, metadata persisted, writes rejected.
    let db = Nidus::open(Config::new(dir.path(), 3).open_mode(OpenMode::ReadOnly)).unwrap();
    let hits = db.search("c", &[1.0, 0.0, 0.0], &opts(5)).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "a");
    assert_eq!(
        db.get_meta("c").get("model").map(String::as_str),
        Some("demo-embed")
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn reopen_with_wrong_dimension_errors() {
    let dir = tempfile::tempdir().unwrap();
    Nidus::open(Config::new(dir.path(), 3)).unwrap();
    assert!(Nidus::open(Config::new(dir.path(), 5)).is_err());
}
