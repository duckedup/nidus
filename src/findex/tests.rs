//! Module tests. The superset property is the contract; everything else supports it.

use std::collections::BTreeMap;

use super::{FilterIndexField, Findex};
use crate::filter;
use crate::model::{Predicate, Value};

/// splitmix64, matching the generator the benchmarks use, so a failing case is
/// reproducible from its seed alone.
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
}

const VOCAB: [&str; 8] = [
    "alpha", "beta", "gamma", "delta", "run", "running", "x", "café",
];

/// A doc set mixing `Str`, `List`, wrong-typed and absent attrs — the leaf rule says the
/// last two never match, and the index must not resurrect them.
fn corpus(seed: u64, n: usize) -> Vec<(String, BTreeMap<String, Value>)> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|i| {
            let words: Vec<String> = (0..1 + rng.below(5))
                .map(|_| VOCAB[rng.below(VOCAB.len())].to_string())
                .collect();
            let mut attrs = BTreeMap::new();
            match rng.below(5) {
                0 => {
                    attrs.insert("text".into(), Value::List(words));
                }
                1 => {
                    attrs.insert("text".into(), Value::Int(7));
                }
                2 => {} // absent
                _ => {
                    attrs.insert("text".into(), Value::Str(words.join(" ")));
                }
            }
            (format!("d{i}"), attrs)
        })
        .collect()
}

fn build(docs: &[(String, BTreeMap<String, Value>)]) -> Findex {
    let mut f = Findex::default();
    f.set_schema("c", &[FilterIndexField::new("text")]);
    for (id, attrs) in docs {
        f.index_doc("c", id, attrs);
    }
    f
}

fn predicates() -> Vec<Predicate> {
    let k = || "text".to_string();
    vec![
        Predicate::ContainsAllTokens(k(), "alpha".into()),
        Predicate::ContainsAllTokens(k(), "alpha beta".into()),
        Predicate::ContainsAllTokens(k(), "".into()),
        Predicate::ContainsAnyToken(k(), "alpha delta".into()),
        Predicate::ContainsAnyToken(k(), "".into()),
        Predicate::ContainsTokenSequence(k(), "alpha beta".into()),
        Predicate::ContainsTokenSequence(k(), "run".into()),
        Predicate::Fuzzy(k(), "running".into(), 0),
        Predicate::Fuzzy(k(), "runnimg".into(), 1),
        Predicate::Fuzzy(k(), "alpha beta gamma".into(), 2),
        Predicate::Fuzzy(k(), "x".into(), 1),
        Predicate::Regex(k(), ".*alpha.*".into()),
        Predicate::Regex(k(), "alpha|beta".into()),
        Predicate::Regex(k(), ".*".into()),
    ]
}

/// **The module's contract.** Whatever the index proposes must contain every document the
/// predicate actually matches. A subset here is a silently wrong query result.
#[test]
fn candidates_are_always_a_superset_of_the_true_matches() {
    for seed in [1u64, 2, 3, 7, 11, 42] {
        let docs = corpus(seed, 40);
        let idx = build(&docs);
        for pred in predicates() {
            let truth: Vec<&str> = docs
                .iter()
                .filter(|(_, a)| filter::matches(&crate::model::Filter(vec![pred.clone()]), a))
                .map(|(id, _)| id.as_str())
                .collect();
            let Some(cands) = idx.candidate_ids("c", &pred, usize::MAX) else {
                continue; // "scan everything" is trivially a superset
            };
            for id in truth {
                assert!(
                    cands.iter().any(|c| c == id),
                    "seed {seed} pred {pred:?} dropped {id}"
                );
            }
        }
    }
}

#[test]
fn a_list_attr_matching_across_elements_is_a_candidate_but_fails_verification() {
    // `field_text` joins list elements, so the postings see one run of tokens while
    // `any_text` requires a single element to satisfy the predicate. The index therefore
    // over-approximates here, and the verify step is what makes that safe.
    let mut attrs = BTreeMap::new();
    attrs.insert(
        "text".into(),
        Value::List(vec!["alpha".into(), "beta".into()]),
    );
    let docs = vec![("d0".to_string(), attrs.clone())];
    let idx = build(&docs);

    let pred = Predicate::ContainsAllTokens("text".into(), "alpha beta".into());
    assert_eq!(
        idx.candidate_ids("c", &pred, usize::MAX),
        Some(vec!["d0".to_string()])
    );
    assert!(!filter::matches(&crate::model::Filter(vec![pred]), &attrs));
}

#[test]
fn an_unindexed_predicate_declines_to_narrow() {
    let idx = build(&corpus(1, 5));
    let pred = Predicate::Eq("text".into(), Value::Int(1));
    assert_eq!(idx.candidate_ids("c", &pred, usize::MAX), None);
}

#[test]
fn an_unindexed_field_declines_to_narrow() {
    let idx = build(&corpus(1, 5));
    let pred = Predicate::ContainsAllTokens("other".into(), "alpha".into());
    assert_eq!(idx.candidate_ids("c", &pred, usize::MAX), None);
}

#[test]
fn a_structure_turned_off_declines_the_predicates_it_would_serve() {
    let mut f = Findex::default();
    f.set_schema("c", &[FilterIndexField::new("text").trigrams(false)]);
    let mut attrs = BTreeMap::new();
    attrs.insert("text".into(), Value::Str("alpha beta".into()));
    f.index_doc("c", "d0", &attrs);

    assert!(
        f.candidate_ids(
            "c",
            &Predicate::ContainsAllTokens("text".into(), "alpha".into()),
            usize::MAX,
        )
        .is_some()
    );
    assert_eq!(
        f.candidate_ids(
            "c",
            &Predicate::Fuzzy("text".into(), "alphaa".into(), 1),
            usize::MAX,
        ),
        None
    );
}

#[test]
fn an_empty_any_token_query_narrows_to_nothing_matching_the_predicate() {
    // `ContainsAnyToken("")` is the `Any([])` identity: it matches no document, so an
    // empty candidate set is the correct narrowing rather than a refusal.
    let idx = build(&corpus(1, 5));
    let pred = Predicate::ContainsAnyToken("text".into(), "".into());
    assert_eq!(idx.candidate_ids("c", &pred, usize::MAX), Some(vec![]));
}

#[test]
fn an_empty_all_tokens_query_declines_because_it_matches_everything() {
    let idx = build(&corpus(1, 5));
    let pred = Predicate::ContainsAllTokens("text".into(), "".into());
    assert_eq!(idx.candidate_ids("c", &pred, usize::MAX), None);
}

#[test]
fn a_short_fuzzy_needle_declines_rather_than_narrowing_to_nothing() {
    // 1 trigram against a 1-edit budget leaves a vacuous threshold; returning an empty
    // set here would drop every real match.
    let idx = build(&corpus(1, 5));
    assert_eq!(
        idx.candidate_ids(
            "c",
            &Predicate::Fuzzy("text".into(), "abc".into(), 1),
            usize::MAX,
        ),
        None
    );
}

#[test]
fn a_regex_with_no_required_literal_declines() {
    let idx = build(&corpus(1, 5));
    assert_eq!(
        idx.candidate_ids(
            "c",
            &Predicate::Regex("text".into(), ".*".into()),
            usize::MAX,
        ),
        None
    );
    assert_eq!(
        idx.candidate_ids(
            "c",
            &Predicate::Regex("text".into(), "alpha|beta".into()),
            usize::MAX,
        ),
        None
    );
}

#[test]
fn an_overwritten_doc_is_not_returned_under_its_old_text() {
    let mut f = Findex::default();
    f.set_schema("c", &[FilterIndexField::new("text")]);
    let mut old = BTreeMap::new();
    old.insert("text".into(), Value::Str("alpha".into()));
    f.index_doc("c", "d0", &old);
    let mut new = BTreeMap::new();
    new.insert("text".into(), Value::Str("delta".into()));
    f.index_doc("c", "d0", &new);

    let pred = Predicate::ContainsAllTokens("text".into(), "alpha".into());
    assert_eq!(f.candidate_ids("c", &pred, usize::MAX), Some(vec![]));
    let pred = Predicate::ContainsAllTokens("text".into(), "delta".into());
    assert_eq!(
        f.candidate_ids("c", &pred, usize::MAX),
        Some(vec!["d0".to_string()])
    );
}

#[test]
fn a_removed_doc_is_not_a_candidate() {
    let mut f = build(&[(
        "d0".to_string(),
        BTreeMap::from([("text".to_string(), Value::Str("alpha".into()))]),
    )]);
    f.remove_doc("c", "d0");
    let pred = Predicate::ContainsAllTokens("text".into(), "alpha".into());
    assert_eq!(f.candidate_ids("c", &pred, usize::MAX), Some(vec![]));
}

#[test]
fn dropping_a_collection_forgets_its_schema_and_postings() {
    let mut f = build(&corpus(1, 5));
    f.drop_collection("c");
    assert!(!f.is_active());
    assert_eq!(
        f.candidate_ids(
            "c",
            &Predicate::ContainsAllTokens("text".into(), "alpha".into()),
            usize::MAX,
        ),
        None
    );
}

#[test]
fn cache_key_changes_on_every_schema_parameter() {
    let base = {
        let mut f = Findex::default();
        f.set_schema("c", &[FilterIndexField::new("text")]);
        f.cache_key()
    };
    for variant in [
        FilterIndexField::new("other"),
        FilterIndexField::new("text").tokens(false),
        FilterIndexField::new("text").trigrams(false),
    ] {
        let mut f = Findex::default();
        f.set_schema("c", &[variant]);
        assert_ne!(base, f.cache_key());
    }
}

#[test]
fn cache_key_is_stable_across_declaration_order_of_collections() {
    let mut a = Findex::default();
    a.set_schema("x", &[FilterIndexField::new("t")]);
    a.set_schema("y", &[FilterIndexField::new("t")]);
    let mut b = Findex::default();
    b.set_schema("y", &[FilterIndexField::new("t")]);
    b.set_schema("x", &[FilterIndexField::new("t")]);
    assert_eq!(a.cache_key(), b.cache_key());
}

#[test]
fn clear_indexes_keeps_the_schema_and_forgets_the_postings() {
    let mut f = build(&corpus(1, 20));
    assert!(f.heap_bytes() > 0);
    f.clear_indexes();
    assert!(f.is_active());
    let pred = Predicate::ContainsAllTokens("text".into(), "alpha".into());
    assert_eq!(f.candidate_ids("c", &pred, usize::MAX), Some(vec![]));
}

#[test]
fn the_cache_codec_round_trips_the_whole_index() {
    let f = build(&corpus(3, 20));
    let key = f.cache_key();
    let bytes = crate::index_cache::frame(&key, 99, &f).unwrap();
    let (back, watermark) = crate::index_cache::decode::<Findex>(&bytes, &key).unwrap();
    assert_eq!(watermark, 99);
    let pred = Predicate::ContainsAllTokens("text".into(), "alpha".into());
    assert_eq!(
        back.candidate_ids("c", &pred, usize::MAX),
        f.candidate_ids("c", &pred, usize::MAX)
    );
}
