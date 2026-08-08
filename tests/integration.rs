//! Black-box integration tests against the public API. These exercise nidus the
//! way a consumer would. File-backed cases use a temp dir and are `#[cfg_attr(miri,
//! ignore)]` (they fsync); in-memory cases run anywhere, including under Miri.

use std::collections::BTreeMap;

use nidus::{
    AggregateOpts, AnnConfig, Config, Decay, Distance, Filter, LimitPer, ListOpts, Nidus, OpenMode,
    OrderBy, Predicate, Quantization, RankBy, Record, Scope, SearchOpts, Value,
};

fn rec(id: &str, vector: Vec<f32>, kind: &str) -> Record {
    let mut attrs = BTreeMap::new();
    attrs.insert("kind".to_string(), Value::Str(kind.to_string()));
    Record::new(id, vector, attrs)
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
        ..Default::default()
    };
    let hits = db.search("c", &[1.0, 0.0, 0.0], &filtered).unwrap();
    assert!(hits.iter().all(|h| h.id != "sym1"));

    // min_score drops the orthogonal "far".
    let gated = SearchOpts {
        top_k: 10,
        min_score: Some(0.5),
        ..Default::default()
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
fn binary_quantization_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = || {
        Config::new(dir.path(), 3)
            .distance(Distance::Cosine)
            .quantization(Some(Quantization::binary()))
    };

    {
        let mut db = Nidus::open(cfg()).unwrap();
        db.create_collection("c").unwrap();
        db.upsert(
            "c",
            &[
                rec("close", vec![0.9, 0.1, 0.0], "file"),
                rec("far", vec![-1.0, -0.2, 0.3], "file"),
            ],
        )
        .unwrap();
    } // writer lock released; in-RAM binary matrix dropped

    // Reopen: the sign-bit matrix is repacked from `data` by rebuild_quant, so the
    // two-pass binary search still ranks correctly against the persisted vectors.
    let db = Nidus::open(cfg()).unwrap();
    let hits = db.search("c", &[1.0, 0.0, 0.0], &opts(2)).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, "close");
}

#[cfg_attr(miri, ignore)]
#[test]
fn reopen_with_wrong_dimension_errors() {
    let dir = tempfile::tempdir().unwrap();
    Nidus::open(Config::new(dir.path(), 3)).unwrap();
    assert!(Nidus::open(Config::new(dir.path(), 5)).is_err());
}

// ── ANN index persistence ────────────────────────────────────────────────────

fn ann_rec(id: &str, vector: Vec<f32>) -> Record {
    rec(id, vector, "doc")
}

/// persist_index() writes a cache; the next open loads it and searches identically.
#[cfg_attr(miri, ignore)]
#[test]
fn ann_index_persists_and_reloads() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = || Config::new(dir.path(), 3).ann(Some(AnnConfig::hnsw()));
    let query = [0.0, 1.0, 0.0];

    let before = {
        let mut db = Nidus::open(cfg()).unwrap();
        db.upsert(
            "c",
            &[
                ann_rec("a", vec![1.0, 0.0, 0.0]),
                ann_rec("b", vec![0.0, 1.0, 0.0]),
                ann_rec("c", vec![0.0, 0.0, 1.0]),
            ],
        )
        .unwrap();
        let hits = db.search("c", &query, &opts(3)).unwrap();
        db.persist_index().unwrap(); // writes the `ann` cache
        hits
    };

    // The cache file exists, and a reopen returns the same ranking.
    assert!(
        dir.path().join("ann").exists(),
        "persist_index wrote the cache"
    );
    let db = Nidus::open(cfg()).unwrap();
    let after = db.search("c", &query, &opts(3)).unwrap();
    let ids_before: Vec<_> = before.iter().map(|h| &h.id).collect();
    let ids_after: Vec<_> = after.iter().map(|h| &h.id).collect();
    assert_eq!(ids_before, ids_after, "reloaded index ranks identically");
}

/// Rows added after the cache was written are incrementally caught up on open.
#[cfg_attr(miri, ignore)]
#[test]
fn ann_index_incremental_catchup_after_persist() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = || Config::new(dir.path(), 3).ann(Some(AnnConfig::hnsw()));

    {
        let mut db = Nidus::open(cfg()).unwrap();
        db.upsert("c", &[ann_rec("a", vec![1.0, 0.0, 0.0])])
            .unwrap();
        db.persist_index().unwrap(); // cache covers 1 row
        // Add a second row *after* persisting — only in `data`/`log`, not the cache.
        db.upsert("c", &[ann_rec("b", vec![0.0, 1.0, 0.0])])
            .unwrap();
    }

    // Reopen: cache (1 row) loads, then row `b` is incrementally inserted.
    let db = Nidus::open(cfg()).unwrap();
    let hits = db.search("c", &[0.0, 1.0, 0.0], &opts(2)).unwrap();
    assert_eq!(hits[0].id, "b", "caught-up row is searchable and nearest");
    assert_eq!(hits.len(), 2, "both rows present");
}

/// ANN combined with quantization (nidus-ndu): the quantized-walk index persists and
/// reloads, ranking identically, and the f32 rerank keeps the self-query exact.
#[cfg_attr(miri, ignore)]
#[test]
fn ann_with_int8_quantization_persists_and_reloads() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = || {
        Config::new(dir.path(), 3)
            .ann(Some(AnnConfig::hnsw()))
            .quantization(Some(Quantization::default()))
    };
    let query = [0.0, 1.0, 0.0];

    let before = {
        let mut db = Nidus::open(cfg()).unwrap();
        db.upsert(
            "c",
            &[
                ann_rec("a", vec![1.0, 0.0, 0.0]),
                ann_rec("b", vec![0.0, 1.0, 0.0]),
                ann_rec("c", vec![0.0, 0.0, 1.0]),
            ],
        )
        .unwrap();
        let hits = db.search("c", &query, &opts(3)).unwrap();
        db.persist_index().unwrap();
        hits
    };
    assert_eq!(before[0].id, "b", "exact rerank surfaces the true nearest");

    let db = Nidus::open(cfg()).unwrap();
    let after = db.search("c", &query, &opts(3)).unwrap();
    let ids_before: Vec<_> = before.iter().map(|h| &h.id).collect();
    let ids_after: Vec<_> = after.iter().map(|h| &h.id).collect();
    assert_eq!(
        ids_before, ids_after,
        "quantized-walk cache reloads identically"
    );
}

/// Changing the quantization config invalidates the ANN cache (it encodes the quant
/// kind in its validity key), so a reopen rebuilds rather than walking a graph built in
/// a different space.
#[cfg_attr(miri, ignore)]
#[test]
fn ann_cache_invalidated_by_quant_change() {
    let dir = tempfile::tempdir().unwrap();
    let recs = [
        ann_rec("a", vec![1.0, 0.0, 0.0]),
        ann_rec("b", vec![0.0, 1.0, 0.0]),
    ];
    {
        let mut db = Nidus::open(
            Config::new(dir.path(), 3)
                .ann(Some(AnnConfig::hnsw()))
                .quantization(Some(Quantization::default())),
        )
        .unwrap();
        db.upsert("c", &recs).unwrap();
        db.persist_index().unwrap();
    }
    // Reopen with binary quantization instead of int8: the cached graph is in int8
    // space, so it must be discarded and rebuilt. The query still returns correctly.
    let db = Nidus::open(
        Config::new(dir.path(), 3)
            .ann(Some(AnnConfig::hnsw()))
            .quantization(Some(Quantization::binary())),
    )
    .unwrap();
    let hits = db.search("c", &[0.0, 1.0, 0.0], &opts(2)).unwrap();
    assert_eq!(hits[0].id, "b", "rebuilt index searches correctly");
}

/// A corrupt cache file is silently discarded and the index rebuilt — no error.
#[cfg_attr(miri, ignore)]
#[test]
fn ann_corrupt_cache_falls_back_to_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = || Config::new(dir.path(), 3).ann(Some(AnnConfig::hnsw()));
    {
        let mut db = Nidus::open(cfg()).unwrap();
        db.upsert("c", &[ann_rec("a", vec![1.0, 0.0, 0.0])])
            .unwrap();
        db.persist_index().unwrap();
    }
    // Clobber the cache with garbage.
    std::fs::write(dir.path().join("ann"), b"not a valid nidus ann cache").unwrap();

    let db = Nidus::open(cfg()).unwrap(); // must not error
    let hits = db.search("c", &[1.0, 0.0, 0.0], &opts(1)).unwrap();
    assert_eq!(
        hits[0].id, "a",
        "rebuilt from vectors after discarding bad cache"
    );
}

/// A read-only handle loads the persisted cache (and persist_index is a no-op).
#[cfg_attr(miri, ignore)]
#[test]
fn ann_readonly_reopen_loads_cache() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = Nidus::open(Config::new(dir.path(), 3).ann(Some(AnnConfig::hnsw()))).unwrap();
        db.upsert("c", &[ann_rec("a", vec![1.0, 0.0, 0.0])])
            .unwrap();
        db.persist_index().unwrap();
    }
    let mut db = Nidus::open(
        Config::new(dir.path(), 3)
            .ann(Some(AnnConfig::hnsw()))
            .open_mode(OpenMode::ReadOnly),
    )
    .unwrap();
    assert_eq!(
        db.search("c", &[1.0, 0.0, 0.0], &opts(1)).unwrap()[0].id,
        "a"
    );
    db.persist_index().unwrap(); // no-op under ReadOnly, must not error
}

/// Boolean composition and containment through the real scan path (nidus-m50.1/.2).
/// `filter::matches` is unit-tested directly; this pins that the same semantics survive
/// the store's scan, including its empty-filter fast path and `list`/`delete_where`.
#[test]
fn nested_filters_and_containment_through_the_store() {
    fn doc(id: &str, project: &str, tags: &[&str]) -> Record {
        let mut attrs = BTreeMap::new();
        attrs.insert("project".to_string(), Value::Str(project.to_string()));
        attrs.insert(
            "tags".to_string(),
            Value::List(tags.iter().map(|s| s.to_string()).collect()),
        );
        Record::new(id, vec![1.0, 0.0, 0.0], attrs)
    }

    let mut db = Nidus::open_in_memory(3).unwrap();
    db.create_collection("c").unwrap();
    db.upsert(
        "c",
        &[
            doc("a", "nidus", &["rust", "wip"]),
            doc("b", "nidus", &["rust"]),
            doc("c", "beads", &["go"]),
            doc("d", "other", &["rust"]),
        ],
    )
    .unwrap();

    let ids = |hits: Vec<nidus::Hit>| {
        let mut v: Vec<String> = hits.into_iter().map(|h| h.id).collect();
        v.sort();
        v
    };

    // (project = nidus OR project = beads) AND NOT tags contains "wip".
    let f = Filter(vec![
        Predicate::Any(vec![
            Predicate::Eq("project".into(), Value::Str("nidus".into())),
            Predicate::Eq("project".into(), Value::Str("beads".into())),
        ]),
        Predicate::Not(Box::new(Predicate::Contains(
            "tags".into(),
            Value::Str("wip".into()),
        ))),
    ]);
    let hits = db
        .search(
            "c",
            &[1.0, 0.0, 0.0],
            &SearchOpts {
                top_k: 10,
                filter: f.clone(),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(ids(hits), vec!["b", "c"]);

    // The same filter through `list`, which takes the non-scoring path.
    assert_eq!(
        ids(db
            .list(
                "c",
                &ListOpts {
                    filter: f,
                    limit: 10,
                    ..Default::default()
                }
            )
            .unwrap()),
        vec!["b", "c"]
    );

    // ContainsAny overlaps on either candidate.
    let any_tag = Filter(vec![Predicate::ContainsAny(
        "tags".into(),
        vec![Value::Str("go".into()), Value::Str("wip".into())],
    )]);
    assert_eq!(
        ids(db
            .list(
                "c",
                &ListOpts {
                    filter: any_tag,
                    limit: 10,
                    ..Default::default()
                }
            )
            .unwrap()),
        vec!["a", "c"]
    );

    // delete_where resolves a nested filter to ids before logging, so this is also
    // the check that a group survives the write path.
    let removed = db
        .delete_where(
            "c",
            &Filter(vec![Predicate::Not(Box::new(Predicate::ContainsAny(
                "tags".into(),
                vec![Value::Str("rust".into())],
            )))]),
        )
        .unwrap();
    assert_eq!(removed, 1); // only "c" lacks the rust tag
    assert_eq!(
        ids(db
            .list(
                "c",
                &ListOpts {
                    limit: 10,
                    ..Default::default()
                }
            )
            .unwrap()),
        vec!["a", "b", "d"]
    );
}

/// Paging through a public `search` from a consumer's seat: the pages tile the ranking, an
/// offset past the end stops the walk, and the default page is what it always was.
#[test]
fn search_pagination_walks_the_ranking_once() {
    let mut db = Nidus::open_in_memory(3).unwrap();
    db.create_collection("c").unwrap();
    let recs: Vec<Record> = (0..7)
        .map(|i| rec(&format!("d{i}"), vec![1.0, i as f32 * 0.1, 0.0], "file"))
        .collect();
    db.upsert("c", &recs).unwrap();
    let q = [1.0, 0.0, 0.0];

    let mut walked: Vec<String> = Vec::new();
    let mut offset = 0;
    loop {
        let page = db
            .search(
                "c",
                &q,
                &SearchOpts {
                    top_k: 3,
                    offset,
                    ..Default::default()
                },
            )
            .unwrap();
        if page.is_empty() {
            break;
        }
        walked.extend(page.iter().map(|h| h.id.clone()));
        offset += 3;
    }
    assert_eq!(walked, ["d0", "d1", "d2", "d3", "d4", "d5", "d6"]);

    // The default page is untouched by the new knob.
    let default_page: Vec<String> = db
        .search("c", &q, &opts(3))
        .unwrap()
        .iter()
        .map(|h| h.id.clone())
        .collect();
    assert_eq!(default_page, ["d0", "d1", "d2"]);
}

#[test]
fn text_predicates_through_the_store() {
    fn doc(id: &str, title: &str, tags: &[&str]) -> Record {
        let mut attrs = BTreeMap::new();
        attrs.insert("title".to_string(), Value::Str(title.to_string()));
        attrs.insert(
            "tags".to_string(),
            Value::List(tags.iter().map(|s| s.to_string()).collect()),
        );
        Record::new(id, vec![1.0, 0.0, 0.0], attrs)
    }

    let mut db = Nidus::open_in_memory(3).unwrap();
    db.create_collection("c").unwrap();
    db.upsert(
        "c",
        &[
            doc("a", "The quick brown fox", &["rust", "search"]),
            doc("b", "A brown quick hound", &["golang"]),
            doc("c", "Postgres vector notes", &["postgres"]),
        ],
    )
    .unwrap();

    let listed = |f: Filter| {
        db.list(
            "c",
            &ListOpts {
                filter: f,
                ..Default::default()
            },
        )
        .map(|hits| {
            let mut v: Vec<String> = hits.into_iter().map(|h| h.id).collect();
            v.sort();
            v
        })
    };
    let ids = |f: Filter| listed(f).unwrap();

    // Fuzzy reaches a half-remembered word, and looks inside the tag list.
    assert_eq!(
        ids(Filter(vec![Predicate::Fuzzy(
            "tags".into(),
            "postgre".into(),
            1
        )])),
        ["c"]
    );
    assert!(
        ids(Filter(vec![Predicate::Fuzzy(
            "tags".into(),
            "postgre".into(),
            0
        )]))
        .is_empty()
    );

    // Token order is free for ContainsAllTokens, mandatory for the sequence.
    assert_eq!(
        ids(Filter(vec![Predicate::ContainsAllTokens(
            "title".into(),
            "brown quick".into()
        )])),
        ["a", "b"]
    );
    assert_eq!(
        ids(Filter(vec![Predicate::ContainsTokenSequence(
            "title".into(),
            "quick brown".into()
        )])),
        ["a"]
    );
    assert_eq!(
        ids(Filter(vec![Predicate::ContainsAnyToken(
            "title".into(),
            "hound postgres".into()
        )])),
        ["b", "c"]
    );

    // Regex is anchored, so the whole attribute must match; `(?i)` is the case switch.
    assert_eq!(
        ids(Filter(vec![Predicate::Regex(
            "title".into(),
            "(?i)the quick .* fox".into()
        )])),
        ["a"]
    );
    assert!(
        ids(Filter(vec![Predicate::Regex(
            "title".into(),
            "quick brown".into()
        )]))
        .is_empty()
    );

    // An edit budget above the ceiling, and an unparseable pattern, are both errors.
    let over = Filter(vec![Predicate::Fuzzy("title".into(), "fox".into(), 9)]);
    assert!(listed(over.clone()).is_err());
    assert!(listed(Filter(vec![Predicate::Regex("title".into(), "(".into())])).is_err());
    assert!(
        db.search(
            "c",
            &[1.0, 0.0, 0.0],
            &SearchOpts {
                top_k: 10,
                filter: over.clone(),
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(db.delete_where("c", &over).is_err());
}

/// The ranking-expression and aggregation surface through the public API, on a file-backed
/// store reopened between writing and querying (nidus-m50.3, nidus-m50.6).
#[test]
#[cfg_attr(miri, ignore)]
fn ranking_and_aggregation_survive_a_reopen() {
    const DAY: i64 = 86_400_000;
    let origin = 2_000 * DAY;
    let dir = tempfile::tempdir().unwrap();
    let stamped = |id: &str, days: i64, bytes: i64| {
        let mut attrs = BTreeMap::new();
        attrs.insert("ts".to_string(), Value::DateTime(origin - days * DAY));
        attrs.insert("bytes".to_string(), Value::Int(bytes));
        attrs.insert("file".to_string(), Value::Str("a.rs".to_string()));
        Record::new(id, vec![1.0, 0.0, 0.0], attrs)
    };
    {
        let mut db = Nidus::open_dir(dir.path(), 3).unwrap();
        db.upsert("c", &[stamped("fresh", 0, 10), stamped("week", 7, 32)])
            .unwrap();
        db.flush().unwrap();
    }

    let db = Nidus::open_dir(dir.path(), 3).unwrap();
    let decayed = SearchOpts {
        top_k: 10,
        rank_by: Some(RankBy::Decay(Decay::new("ts", origin, 7 * DAY).lambda(0.4))),
        ..Default::default()
    };
    let hits = db.search("c", &[1.0, 0.0, 0.0], &decayed).unwrap();
    assert_eq!(hits[0].id, "fresh");
    assert!((hits[1].score - 0.8).abs() < 1e-5, "{}", hits[1].score);

    // Both docs share one file, so a cap of one leaves exactly one hit.
    let capped = SearchOpts {
        top_k: 10,
        limit_per: Some(LimitPer::new("file", 1)),
        ..Default::default()
    };
    assert_eq!(db.search("c", &[1.0, 0.0, 0.0], &capped).unwrap().len(), 1);

    let ordered = db
        .list(
            "c",
            &ListOpts {
                order_by: Some(OrderBy::desc("bytes")),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(ordered[0].id, "week");

    let stats = db
        .aggregate(
            Scope::All,
            &AggregateOpts {
                sum: vec!["bytes".into()],
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(stats.count, 2);
    assert_eq!(stats.sums["bytes"], Value::Int(42));
}
