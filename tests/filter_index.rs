//! The filter index (nidus-89) as a *narrower*: it may propose extra candidates, never
//! withhold a real match. So the acceptance criterion is differential — an indexed store
//! and an unindexed one must return **identical** results — and everything else here
//! supports that claim.
//!
//! Generators are hand-rolled splitmix64 rather than a property-test crate: this repo has
//! one dev-dependency and a build-speed thesis, and `TermRng` in the benchmarks already
//! sets the precedent.

use std::collections::BTreeMap;

use nidus::{Filter, FilterIndexField, Hit, ListOpts, Nidus, Predicate, Record, SearchOpts, Value};

// ── Deterministic generation ──────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

const DIM: usize = 8;

/// A vocabulary with deliberate traps: a shared prefix (`run`/`running`) so stemming
/// differences would show, a repeated-token phrase case, non-ASCII, and a term short
/// enough to have no trigrams at all.
const VOCAB: [&str; 10] = [
    "alpha", "beta", "gamma", "delta", "run", "running", "café", "ab", "term0", "a",
];

fn docs(seed: u64, n: usize) -> Vec<Record> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|i| {
            let words: Vec<String> = (0..1 + rng.below(6))
                .map(|_| VOCAB[rng.below(VOCAB.len())].to_string())
                .collect();
            let mut attrs = BTreeMap::new();
            // Mix the attr shapes the leaf rule distinguishes: Str, List, wrong-typed,
            // and absent. The last two must never match, indexed or not.
            match rng.below(6) {
                0 => {
                    attrs.insert("text".into(), Value::List(words.clone()));
                }
                1 => {
                    attrs.insert("text".into(), Value::Int(7));
                }
                2 => {}
                _ => {
                    attrs.insert("text".into(), Value::Str(words.join(" ")));
                }
            }
            attrs.insert("n".into(), Value::Int(i as i64));
            let vector: Vec<f32> = (0..DIM).map(|_| rng.unit()).collect();
            Record {
                id: format!("d{i}"),
                vector: Some(vector),
                attrs,
            }
        })
        .collect()
}

fn words(rng: &mut Rng, n: usize) -> String {
    (0..n)
        .map(|_| VOCAB[rng.below(VOCAB.len())])
        .collect::<Vec<_>>()
        .join(" ")
}

fn leaf(rng: &mut Rng) -> Predicate {
    let k = || "text".to_string();
    let which = rng.below(10);
    let n = 1 + rng.below(3);
    let term = VOCAB[rng.below(VOCAB.len())];
    match which {
        0 => Predicate::ContainsAllTokens(k(), words(rng, n)),
        1 => Predicate::ContainsAnyToken(k(), words(rng, n)),
        2 => Predicate::ContainsTokenSequence(k(), words(rng, n)),
        3 => Predicate::ContainsAllTokens(k(), String::new()),
        4 => Predicate::ContainsAnyToken(k(), String::new()),
        5 => {
            let budget = rng.below(3);
            Predicate::Fuzzy(k(), words(rng, n), budget)
        }
        6 => Predicate::Regex(k(), format!(".*{term}.*")),
        7 => Predicate::Regex(k(), ".*".into()),
        // Un-indexed leaves, so mixed boolean shapes are exercised too.
        8 => Predicate::Eq("n".into(), Value::Int(rng.below(40) as i64)),
        _ => Predicate::Glob("text".into(), format!("*{term}*")),
    }
}

/// A filter nested to at most `depth`, mixing indexed and un-indexed leaves under
/// `All`/`Any`/`Not`. Hand-written cases do not find a sign error inside a nested `Any`.
fn predicate(rng: &mut Rng, depth: usize) -> Predicate {
    if depth == 0 {
        return leaf(rng);
    }
    match rng.below(6) {
        0 => {
            let n = 1 + rng.below(2);
            Predicate::All((0..n).map(|_| predicate(rng, depth - 1)).collect())
        }
        1 => {
            let n = 1 + rng.below(2);
            Predicate::Any((0..n).map(|_| predicate(rng, depth - 1)).collect())
        }
        2 => Predicate::Not(Box::new(predicate(rng, depth - 1))),
        _ => leaf(rng),
    }
}

fn filters(seed: u64, n: usize) -> Vec<Filter> {
    let mut rng = Rng(seed ^ 0xF117);
    (0..n)
        .map(|_| {
            let n = 1 + rng.below(2);
            Filter((0..n).map(|_| predicate(&mut rng, 3)).collect())
        })
        .collect()
}

/// Volume for the randomized differential tests: the native number, or a smaller one under
/// Miri, which interprets every step (this file was 53 of the lane's 65 minutes, against
/// 0.24s natively). The `test` job still draws the full volume on every PR (nidus-2r7).
const fn scale(native: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { native }
}

// ── Fixtures ──────────────────────────────────────────────────────────────────────

fn build(records: &[Record], indexed: bool) -> Nidus {
    let mut db = Nidus::open_in_memory(DIM).expect("open");
    db.create_collection("c").expect("create");
    if indexed {
        db.set_filter_index("c", &[FilterIndexField::new("text")])
            .expect("declare");
    }
    db.upsert("c", records).expect("upsert");
    db
}

fn ids(hits: &[Hit]) -> Vec<&str> {
    hits.iter().map(|h| h.id.as_str()).collect()
}

fn query_vec() -> Vec<f32> {
    (0..DIM).map(|i| (i as f32 + 1.0) / DIM as f32).collect()
}

// ── The contract ──────────────────────────────────────────────────────────────────

/// **The acceptance criterion for the whole feature.** Same corpus, same filters, one
/// store with the index and one without: identical hits, order and scores.
#[test]
fn indexed_and_unindexed_results_are_identical() {
    for &seed in [1u64, 2, 3, 5, 8, 13, 21, 34].iter().take(scale(8, 2)) {
        let records = docs(seed, scale(60, 12));
        let plain = build(&records, false);
        let indexed = build(&records, true);

        for filter in filters(seed, scale(40, 6)) {
            let opts = SearchOpts {
                top_k: 10,
                filter: filter.clone(),
                ..Default::default()
            };
            let a = plain
                .search("c", &query_vec(), &opts)
                .expect("plain search");
            let b = indexed
                .search("c", &query_vec(), &opts)
                .expect("indexed search");
            assert_eq!(
                ids(&a),
                ids(&b),
                "seed {seed} search mismatch for {filter:?}"
            );
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.score, y.score, "seed {seed} score mismatch {filter:?}");
            }

            let lopts = ListOpts {
                limit: 100,
                filter: filter.clone(),
                ..Default::default()
            };
            let la = plain.list("c", &lopts).expect("plain list");
            let lb = indexed.list("c", &lopts).expect("indexed list");
            assert_eq!(ids(&la), ids(&lb), "seed {seed} list mismatch {filter:?}");
        }
    }
}

/// `delete_where` shares the filter path, so it shares the risk.
#[test]
fn delete_where_removes_the_same_documents_either_way() {
    for &seed in [4u64, 6, 9].iter().take(scale(3, 1)) {
        let records = docs(seed, scale(40, 10));
        for filter in filters(seed, scale(20, 4)) {
            let mut plain = build(&records, false);
            let mut indexed = build(&records, true);
            let a = plain.delete_where("c", &filter).expect("plain delete");
            let b = indexed.delete_where("c", &filter).expect("indexed delete");
            assert_eq!(a, b, "seed {seed} delete count mismatch for {filter:?}");

            let lopts = ListOpts {
                limit: 200,
                ..Default::default()
            };
            let la = plain.list("c", &lopts).expect("plain list");
            let lb = indexed.list("c", &lopts).expect("indexed list");
            assert_eq!(
                ids(&la),
                ids(&lb),
                "seed {seed} survivors differ {filter:?}"
            );
        }
    }
}

// ── Write-path coverage: a missed upsert is a silently wrong result ───────────────

#[test]
fn a_document_written_after_the_declaration_is_found() {
    let mut db = Nidus::open_in_memory(DIM).unwrap();
    db.create_collection("c").unwrap();
    db.set_filter_index("c", &[FilterIndexField::new("text")])
        .unwrap();
    db.upsert("c", &docs(1, 5)).unwrap();
    let mut attrs = BTreeMap::new();
    attrs.insert("text".into(), Value::Str("zebra quagga".into()));
    db.upsert(
        "c",
        &[Record {
            id: "late".into(),
            vector: Some(vec![0.5; DIM]),
            attrs,
        }],
    )
    .unwrap();

    let hits = db
        .list(
            "c",
            &ListOpts {
                limit: 10,
                filter: Filter(vec![Predicate::ContainsAllTokens(
                    "text".into(),
                    "zebra".into(),
                )]),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(ids(&hits), ["late"]);
}

#[test]
fn documents_written_before_the_declaration_are_indexed_by_it() {
    let mut db = Nidus::open_in_memory(DIM).unwrap();
    db.create_collection("c").unwrap();
    let mut attrs = BTreeMap::new();
    attrs.insert("text".into(), Value::Str("zebra quagga".into()));
    db.upsert(
        "c",
        &[Record {
            id: "early".into(),
            vector: Some(vec![0.5; DIM]),
            attrs,
        }],
    )
    .unwrap();
    // Declaring after the write must backfill, or the doc is invisible to the predicate.
    db.set_filter_index("c", &[FilterIndexField::new("text")])
        .unwrap();

    let hits = db
        .list(
            "c",
            &ListOpts {
                limit: 10,
                filter: Filter(vec![Predicate::ContainsAllTokens(
                    "text".into(),
                    "zebra".into(),
                )]),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(ids(&hits), ["early"]);
}

#[test]
fn an_overwritten_document_is_not_found_under_its_old_text() {
    let mut db = Nidus::open_in_memory(DIM).unwrap();
    db.create_collection("c").unwrap();
    db.set_filter_index("c", &[FilterIndexField::new("text")])
        .unwrap();
    for text in ["zebra quagga", "okapi bongo"] {
        let mut attrs = BTreeMap::new();
        attrs.insert("text".into(), Value::Str(text.into()));
        db.upsert(
            "c",
            &[Record {
                id: "d".into(),
                vector: Some(vec![0.5; DIM]),
                attrs,
            }],
        )
        .unwrap();
    }
    let find = |q: &str| {
        db.list(
            "c",
            &ListOpts {
                limit: 10,
                filter: Filter(vec![Predicate::ContainsAllTokens("text".into(), q.into())]),
                ..Default::default()
            },
        )
        .unwrap()
        .len()
    };
    assert_eq!(find("zebra"), 0);
    assert_eq!(find("okapi"), 1);
}

// ── Adversarial inputs ────────────────────────────────────────────────────────────

/// Each of these is a place the index could disagree with the evaluator: a phrase with a
/// repeated prefix, non-ASCII case (folded for ASCII only), text too short to have any
/// trigram, and a fuzzy budget so wide the bound goes vacuous.
#[test]
fn edge_case_inputs_agree_with_the_unindexed_path() {
    let texts = [
        ("repeat", "a a a b"),
        ("accent", "café"),
        ("accent_upper", "CAFÉ"),
        ("short", "ab"),
        ("empty", ""),
        ("punct", "-- ,, --"),
        ("long", "alpha beta gamma delta run running term0"),
    ];
    let records: Vec<Record> = texts
        .iter()
        .enumerate()
        .map(|(i, (id, text))| {
            let mut attrs = BTreeMap::new();
            attrs.insert("text".into(), Value::Str((*text).into()));
            Record {
                id: (*id).into(),
                vector: Some(vec![i as f32 / 10.0; DIM]),
                attrs,
            }
        })
        .collect();

    let plain = build(&records, false);
    let indexed = build(&records, true);

    let cases = [
        Predicate::ContainsTokenSequence("text".into(), "a a b".into()),
        Predicate::ContainsAllTokens("text".into(), "café".into()),
        Predicate::ContainsAllTokens("text".into(), "CAFÉ".into()),
        Predicate::ContainsAllTokens("text".into(), "ab".into()),
        Predicate::Fuzzy("text".into(), "cafe".into(), 1),
        Predicate::Fuzzy("text".into(), "ab".into(), 2),
        Predicate::Fuzzy(
            "text".into(),
            "alpha beta gamma delta run running term0".into(),
            3,
        ),
        Predicate::Regex("text".into(), ".*".into()),
        Predicate::Regex("text".into(), "caf.".into()),
        Predicate::Regex("text".into(), "alpha|nothing".into()),
    ];
    for pred in cases {
        let opts = ListOpts {
            limit: 50,
            filter: Filter(vec![pred.clone()]),
            ..Default::default()
        };
        assert_eq!(
            ids(&plain.list("c", &opts).unwrap()),
            ids(&indexed.list("c", &opts).unwrap()),
            "mismatch for {pred:?}"
        );
    }
}

#[test]
fn turning_the_index_off_restores_the_unindexed_path() {
    let records = docs(7, 20);
    let mut db = build(&records, true);
    db.set_filter_index("c", &[]).unwrap();
    let plain = build(&records, false);
    let opts = ListOpts {
        limit: 50,
        filter: Filter(vec![Predicate::ContainsAllTokens(
            "text".into(),
            "alpha".into(),
        )]),
        ..Default::default()
    };
    assert_eq!(
        ids(&plain.list("c", &opts).unwrap()),
        ids(&db.list("c", &opts).unwrap())
    );
}

#[test]
fn a_declaration_on_a_field_no_document_carries_is_harmless() {
    let records = docs(11, 20);
    let mut db = Nidus::open_in_memory(DIM).unwrap();
    db.create_collection("c").unwrap();
    db.set_filter_index("c", &[FilterIndexField::new("absent")])
        .unwrap();
    db.upsert("c", &records).unwrap();
    let plain = build(&records, false);
    let opts = ListOpts {
        limit: 50,
        filter: Filter(vec![Predicate::ContainsAllTokens(
            "text".into(),
            "alpha".into(),
        )]),
        ..Default::default()
    };
    assert_eq!(
        ids(&plain.list("c", &opts).unwrap()),
        ids(&db.list("c", &opts).unwrap())
    );
}

#[test]
fn a_field_indexing_nothing_is_rejected() {
    let mut db = Nidus::open_in_memory(DIM).unwrap();
    db.create_collection("c").unwrap();
    let err = db
        .set_filter_index(
            "c",
            &[FilterIndexField::new("text").tokens(false).trigrams(false)],
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("would index nothing"), "{err}");
}

// ── File-backed: persistence, corruption, staleness ───────────────────────────────
// Each case opens a real store, so each carries a Miri ignore for the fsync syscalls
// Miri lacks. The in-memory cases above stay Miri-clean.

use nidus::Config;

fn text_rec(id: &str, text: &str) -> Record {
    let mut attrs = BTreeMap::new();
    attrs.insert("text".into(), Value::Str(text.into()));
    Record {
        id: id.into(),
        vector: Some(vec![0.5; DIM]),
        attrs,
    }
}

fn tokens_filter(q: &str) -> Filter {
    Filter(vec![Predicate::ContainsAllTokens("text".into(), q.into())])
}

fn found(db: &Nidus, q: &str) -> Vec<String> {
    db.list(
        "c",
        &ListOpts {
            limit: 50,
            filter: tokens_filter(q),
            ..Default::default()
        },
    )
    .unwrap()
    .iter()
    .map(|h| h.id.clone())
    .collect()
}

fn seed_store(dir: &std::path::Path) {
    let mut db = Nidus::open(Config::new(dir, DIM)).unwrap();
    db.create_collection("c").unwrap();
    db.set_filter_index("c", &[FilterIndexField::new("text")])
        .unwrap();
    db.upsert(
        "c",
        &[text_rec("a", "zebra quagga"), text_rec("b", "okapi")],
    )
    .unwrap();
    db.flush().unwrap();
    // Derived caches are written out-of-band, never on the upsert/flush path.
    db.persist_index().unwrap();
}

#[cfg_attr(miri, ignore)] // fsync: Miri does not implement it
#[test]
fn the_index_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    seed_store(dir.path());
    let db = Nidus::open(Config::new(dir.path(), DIM)).unwrap();
    assert_eq!(found(&db, "zebra"), ["a"]);
    assert_eq!(found(&db, "okapi"), ["b"]);
}

#[cfg_attr(miri, ignore)] // fsync: Miri does not implement it
#[test]
fn a_corrupt_cache_rebuilds_instead_of_failing() {
    let dir = tempfile::tempdir().unwrap();
    seed_store(dir.path());
    std::fs::write(dir.path().join("findex"), b"not a valid nidus findex cache").unwrap();

    let db = Nidus::open(Config::new(dir.path(), DIM)).unwrap(); // must not error
    assert_eq!(found(&db, "zebra"), ["a"], "rebuilt after discarding cache");
}

#[cfg_attr(miri, ignore)] // fsync: Miri does not implement it
#[test]
fn every_truncation_of_the_cache_rebuilds() {
    // Mirrors `index_cache`'s own truncated-buffer sweep: no prefix may panic or be
    // adopted, at any length.
    let dir = tempfile::tempdir().unwrap();
    seed_store(dir.path());
    let full = std::fs::read(dir.path().join("findex")).unwrap();
    for n in (0..full.len()).step_by(7) {
        std::fs::write(dir.path().join("findex"), &full[..n]).unwrap();
        let db = Nidus::open(Config::new(dir.path(), DIM)).unwrap();
        assert_eq!(found(&db, "zebra"), ["a"], "truncated to {n} bytes");
    }
}

#[cfg_attr(miri, ignore)] // fsync: Miri does not implement it
#[test]
fn a_changed_declaration_invalidates_the_cache() {
    let dir = tempfile::tempdir().unwrap();
    seed_store(dir.path());
    {
        let mut db = Nidus::open(Config::new(dir.path(), DIM)).unwrap();
        db.set_filter_index("c", &[FilterIndexField::new("text").trigrams(false)])
            .unwrap();
        db.flush().unwrap();
        db.persist_index().unwrap();
    }
    let db = Nidus::open(Config::new(dir.path(), DIM)).unwrap();
    assert_eq!(found(&db, "zebra"), ["a"]);
}

/// **The wrong-answer case, not the slow one.** Writes made after the last cache save
/// must not be lost when the cache is adopted on open.
#[cfg_attr(miri, ignore)] // fsync: Miri does not implement it
#[test]
fn writes_after_the_last_cache_save_are_not_lost_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    seed_store(dir.path());
    {
        // `flush` in `seed_store` persisted the cache; this write advances the log past
        // that watermark without persisting again.
        let mut db = Nidus::open(Config::new(dir.path(), DIM)).unwrap();
        db.upsert("c", &[text_rec("late", "pangolin")]).unwrap();
    }
    let db = Nidus::open(Config::new(dir.path(), DIM)).unwrap();
    assert_eq!(
        found(&db, "pangolin"),
        ["late"],
        "a stale-watermark cache must be rebuilt, not adopted"
    );
    assert_eq!(found(&db, "zebra"), ["a"]);
}

#[cfg_attr(miri, ignore)] // fsync: Miri does not implement it
#[test]
fn compaction_leaves_indexed_queries_correct() {
    let dir = tempfile::tempdir().unwrap();
    seed_store(dir.path());
    let mut db = Nidus::open(Config::new(dir.path(), DIM)).unwrap();
    db.delete("c", &["b"]).unwrap();
    db.compact().unwrap();
    assert_eq!(found(&db, "zebra"), ["a"]);
    assert!(found(&db, "okapi").is_empty());

    // The declaration must survive compaction's log rewrite, or a later reopen silently
    // loses the index.
    drop(db);
    let db = Nidus::open(Config::new(dir.path(), DIM)).unwrap();
    assert_eq!(found(&db, "zebra"), ["a"]);
}

#[cfg_attr(miri, ignore)] // fsync: Miri does not implement it
#[test]
fn a_read_only_reopen_answers_from_the_index() {
    let dir = tempfile::tempdir().unwrap();
    seed_store(dir.path());
    let db =
        Nidus::open(Config::new(dir.path(), DIM).open_mode(nidus::OpenMode::ReadOnly)).unwrap();
    assert_eq!(found(&db, "zebra"), ["a"]);
}

#[cfg_attr(miri, ignore)] // fsync: Miri does not implement it
#[test]
fn the_footprint_reports_the_index_and_zero_without_one() {
    let dir = tempfile::tempdir().unwrap();
    seed_store(dir.path());
    let db = Nidus::open(Config::new(dir.path(), DIM)).unwrap();
    assert!(db.footprint().filter_index_bytes > 0);

    let plain = Nidus::open_in_memory(DIM).unwrap();
    assert_eq!(plain.footprint().filter_index_bytes, 0);
}
