//! Tests for the store: pure-logic (Miri-clean) unit tests plus file-backed and
//! quantization/ANN behaviour. Lives beside the implementation it exercises; the
//! `pub(super)` quant-state fields let it assert on maintained index state.

use std::collections::BTreeMap;

use super::quant::{BinState, Int8State, Quant};
use super::scoring::PARALLEL_SCAN_WORK_FLOOR;
use super::write::SegmentReport;
use super::*;
use crate::Fsync;
use crate::data::SegmentIntegrity;
use crate::model::{
    Filter, Hit, ListOpts, Predicate, Projection, Quantization, Record, SearchOpts, Suggestions,
    Value,
};
use crate::search::normalize;

/// Extract the int8 state from a store's quant slot, panicking if it is off or binary.
fn int8_state(store: &Store) -> &Int8State {
    match store
        .quant
        .as_ref()
        .expect("quantization should be enabled")
    {
        Quant::Int8(s) => s,
        Quant::Binary(_) => panic!("expected int8 quant state, found binary"),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn rec(id: &str, vector: Vec<f32>) -> Record {
    Record::new(id, vector, BTreeMap::new())
}

fn rec_with(id: &str, vector: Vec<f32>, attrs: BTreeMap<String, Value>) -> Record {
    Record::new(id, vector, attrs)
}

fn default_opts(top_k: usize) -> SearchOpts {
    SearchOpts {
        top_k,
        ..Default::default()
    }
}

// ── Pure-logic tests (Miri-clean) ─────────────────────────────────────

#[test]
fn in_memory_dimension() {
    let store = Store::in_memory(4).unwrap();
    assert_eq!(store.dimension(), 4);
}

#[test]
fn effective_query_threads_matches_config_on_native() {
    // The wasm branch clamps to 1; it cannot be exercised on this (native) host, so U5's
    // wasm test suite asserts that half — see BLUEPRINT-nidus-y67.md.
    let store = Store::in_memory_cfg(Config::new("/dev/null", 3).query_threads(4)).unwrap();
    assert_eq!(store.effective_query_threads(), 4);
}

#[test]
fn create_and_has_collection() {
    let mut store = Store::in_memory(3).unwrap();
    assert!(!store.has_collection("docs"));
    store.create_collection("docs").unwrap();
    assert!(store.has_collection("docs"));
}

#[test]
fn create_collection_idempotent() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("docs").unwrap();
    store.create_collection("docs").unwrap(); // should not error
    assert!(store.has_collection("docs"));
    assert_eq!(store.collections().len(), 1);
}

#[test]
fn drop_collection() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("docs").unwrap();
    store.drop_collection("docs").unwrap();
    assert!(!store.has_collection("docs"));
}

#[test]
fn drop_nonexistent_collection_is_noop() {
    let mut store = Store::in_memory(3).unwrap();
    store.drop_collection("ghost").unwrap(); // no error
}

#[test]
fn collections_sorted() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("zebra").unwrap();
    store.create_collection("apple").unwrap();
    store.create_collection("mango").unwrap();
    let names = store.collections();
    assert_eq!(names, vec!["apple", "mango", "zebra"]);
}

#[test]
fn metadata_round_trip() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("col").unwrap();
    let mut meta = BTreeMap::new();
    meta.insert("model".to_string(), "text-embed-v1".to_string());
    meta.insert("hwm".to_string(), "42".to_string());
    store.set_meta("col", meta.clone()).unwrap();
    assert_eq!(store.get_meta("col"), meta);
}

#[test]
fn get_meta_absent_collection_returns_empty() {
    let store = Store::in_memory(2).unwrap();
    assert!(store.get_meta("nope").is_empty());
}

#[test]
fn upsert_and_search_exact_match() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    // A vector pointing along x.
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "doc1");
    assert!(
        (hits[0].score - 1.0).abs() < 1e-6,
        "exact match should score ~1.0"
    );
}

#[test]
fn upsert_orthogonal_scores_zero() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    // Query along y — orthogonal to doc1's vector.
    let hits = store
        .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].score.abs() < 1e-6,
        "orthogonal vectors should score ~0.0"
    );
}

#[test]
fn search_ranking_order() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    // doc_a is closest to query [1,0,0], doc_b is farther.
    store
        .upsert(
            "col",
            &[
                rec("doc_a", vec![1.0, 0.0, 0.0]),
                rec("doc_b", vec![0.0, 1.0, 0.0]),
            ],
        )
        .unwrap();
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, "doc_a", "highest scorer should be first");
    assert!(hits[0].score > hits[1].score);
}

// ── search_similar: "more like this" by record id ──────────────────────────

#[test]
fn search_similar_excludes_the_source_record() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert(
            "col",
            &[
                rec("src", vec![1.0, 0.0, 0.0]),
                rec("near", vec![0.9, 0.1, 0.0]),
                rec("far", vec![0.0, 1.0, 0.0]),
            ],
        )
        .unwrap();
    let hits = store
        .search_similar(&["col"], "col", "src", &default_opts(10))
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert!(!ids.contains(&"src"), "source record must not self-match");
    assert!(ids.contains(&"near"), "a real neighbour must be returned");
    assert_eq!(hits[0].id, "near", "nearest neighbour ranks first");
}

#[test]
fn search_similar_keeps_a_true_duplicate() {
    // Written as a score test (score < 1.0 - eps) this would wrongly drop the duplicate too;
    // exclusion must be by (collection, id) identity alone.
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert(
            "col",
            &[
                rec("src", vec![1.0, 0.0, 0.0]),
                rec("dup", vec![1.0, 0.0, 0.0]),
            ],
        )
        .unwrap();
    let hits = store
        .search_similar(&["col"], "col", "src", &default_opts(10))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "dup");
    assert!(
        (hits[0].score - 1.0).abs() < 1e-6,
        "byte-identical duplicate must still score ~1.0"
    );
}

#[test]
fn search_similar_on_a_text_only_record_errors_with_the_reason() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert("col", &[text_rec("t1", attrs_one("kind", "note"))])
        .unwrap();
    let err = store
        .search_similar(&["col"], "col", "t1", &default_opts(10))
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("t1"), "message should name the record: {msg}");
    assert!(msg.contains("text-only"), "message should say why: {msg}");
}

#[test]
fn search_similar_on_an_unknown_id_errors_distinctly() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert("col", &[rec("src", vec![1.0, 0.0, 0.0])])
        .unwrap();
    let err = store
        .search_similar(&["col"], "col", "ghost", &default_opts(10))
        .unwrap_err();
    let text_only_err = {
        store
            .upsert("col", &[text_rec("t1", attrs_one("kind", "note"))])
            .unwrap();
        store
            .search_similar(&["col"], "col", "t1", &default_opts(10))
            .unwrap_err()
    };
    assert_ne!(
        err.to_string(),
        text_only_err.to_string(),
        "a missing id and a text-only record must report distinct reasons"
    );
}

#[test]
fn search_similar_returns_a_full_page() {
    // Without the over-fetch slot, ranking only `top_k` deep and then dropping the (always
    // top-ranked) source would return one hit short of a full page.
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert(
            "col",
            &[
                rec("src", vec![1.0, 0.0, 0.0]),
                rec("n1", vec![0.9, 0.1, 0.0]),
                rec("n2", vec![0.8, 0.2, 0.0]),
                rec("n3", vec![0.7, 0.3, 0.0]),
                rec("n4", vec![0.6, 0.4, 0.0]),
            ],
        )
        .unwrap();
    let hits = store
        .search_similar(&["col"], "col", "src", &default_opts(3))
        .unwrap();
    assert_eq!(hits.len(), 3, "a full page of 3, not 2");
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, vec!["n1", "n2", "n3"]);
}

#[test]
fn search_similar_pages_without_the_source() {
    // offset=1 must land on the second-best *neighbour*, proving the source was dropped
    // before the offset was applied, not after.
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert(
            "col",
            &[
                rec("src", vec![1.0, 0.0, 0.0]),
                rec("n1", vec![0.9, 0.1, 0.0]),
                rec("n2", vec![0.8, 0.2, 0.0]),
                rec("n3", vec![0.7, 0.3, 0.0]),
            ],
        )
        .unwrap();
    let opts = SearchOpts {
        top_k: 1,
        offset: 1,
        ..Default::default()
    };
    let hits = store.search_similar(&["col"], "col", "src", &opts).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "n2");
}

#[test]
fn search_similar_can_cross_collections() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert("a", &[rec("src", vec![1.0, 0.0, 0.0])])
        .unwrap();
    store
        .upsert(
            "b",
            &[
                rec("near", vec![0.9, 0.1, 0.0]),
                rec("far", vec![0.0, 1.0, 0.0]),
            ],
        )
        .unwrap();
    let hits = store
        .search_similar(&["a", "b"], "a", "src", &default_opts(10))
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert!(!ids.contains(&"src"));
    assert_eq!(hits[0].collection, "b");
    assert_eq!(hits[0].id, "near");
}

/// The source always ranks first at 1.0, so capping before it is dropped spends its own
/// value's quota and starves the genuine neighbour sharing that value.
#[test]
fn search_similar_does_not_let_the_source_spend_its_own_limit_per_slot() {
    let mut store = Store::in_memory(3).unwrap();
    let mut attrs = BTreeMap::new();
    attrs.insert("cat".to_string(), Value::Str("x".to_string()));
    store
        .upsert(
            "col",
            &[
                rec_with("src", vec![1.0, 0.0, 0.0], attrs.clone()),
                rec_with("near", vec![0.99, 0.1, 0.0], attrs),
                rec_with("far", vec![0.0, 1.0, 0.0], {
                    let mut m = BTreeMap::new();
                    m.insert("cat".to_string(), Value::Str("y".to_string()));
                    m
                }),
            ],
        )
        .unwrap();

    let opts = SearchOpts {
        top_k: 10,
        limit_per: Some(crate::model::LimitPer::new("cat", 1)),
        ..Default::default()
    };
    let hits = store.search_similar(&["col"], "col", "src", &opts).unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert!(!ids.contains(&"src"), "source must be excluded: {ids:?}");
    assert!(
        ids.contains(&"near"),
        "the cap's one `cat=x` slot belongs to the nearest real neighbour, not to the \
         already-excluded source: {ids:?}"
    );
    assert!(
        ids.contains(&"far"),
        "the other group is unaffected: {ids:?}"
    );
}

#[test]
fn upsert_is_idempotent_by_id() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    // Insert doc1 twice with different vectors.
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    store
        .upsert("col", &[rec("doc1", vec![0.0, 1.0, 0.0])])
        .unwrap();
    // Count stays at 1.
    assert_eq!(store.get_all("col").len(), 1);
    // The newest vector wins — query along y should give score ~1.0.
    let hits = store
        .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!((hits[0].score - 1.0).abs() < 1e-6);
}

#[test]
fn delete_removes_doc() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    let removed = store.delete("col", &["doc1"]).unwrap();
    assert_eq!(removed, 1);
    assert!(store.get_all("col").is_empty());
}

#[test]
fn delete_nonexistent_returns_zero() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    let removed = store.delete("col", &["ghost"]).unwrap();
    assert_eq!(removed, 0);
}

#[test]
fn delete_where_by_attr() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    let mut attrs_a = BTreeMap::new();
    attrs_a.insert("kind".to_string(), Value::Str("file".to_string()));
    let mut attrs_b = BTreeMap::new();
    attrs_b.insert("kind".to_string(), Value::Str("section".to_string()));
    store
        .upsert(
            "col",
            &[
                rec_with("doc_a", vec![1.0, 0.0, 0.0], attrs_a),
                rec_with("doc_b", vec![0.0, 1.0, 0.0], attrs_b),
            ],
        )
        .unwrap();
    // Delete only files.
    let filter = Filter(vec![Predicate::Eq(
        "kind".to_string(),
        Value::Str("file".to_string()),
    )]);
    let removed = store.delete_where("col", &filter).unwrap();
    assert_eq!(removed, 1);
    let remaining = store.get_all("col");
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].id, "doc_b");
}

#[test]
fn not_expired_predicate_is_a_true_complement() {
    // Mirrors `memory::not_expired_predicate`, which is behind the `memory` feature and
    // so unimportable here: true when unexpired *and* when the key is absent, where a
    // bare `Gt`/`Ge` would be false and hide every never-TTL'd memory.
    const EXPIRES_AT: &str = "nidus.expires_at";
    let not_expired = |now: i64| -> Predicate {
        Predicate::Not(Box::new(Predicate::Le(
            EXPIRES_AT.to_string(),
            Value::DateTime(now),
        )))
    };

    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    let now = 1_700_000_000_000i64;

    let unexpired = BTreeMap::from([(EXPIRES_AT.to_string(), Value::DateTime(now + 60_000))]);
    let expired = BTreeMap::from([(EXPIRES_AT.to_string(), Value::DateTime(now - 1))]);

    store
        .upsert(
            "col",
            &[
                rec_with("never_ttld", vec![1.0, 0.0, 0.0], BTreeMap::new()),
                rec_with("unexpired", vec![1.0, 0.0, 0.0], unexpired),
                rec_with("expired", vec![1.0, 0.0, 0.0], expired),
            ],
        )
        .unwrap();

    let opts = SearchOpts {
        top_k: 10,
        filter: Filter(vec![not_expired(now)]),
        ..Default::default()
    };
    let hits = store.search(&["col"], &[1.0, 0.0, 0.0], &opts).unwrap();
    let mut ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    ids.sort();
    assert_eq!(
        ids,
        vec!["never_ttld", "unexpired"],
        "absent-key and unexpired must match; expired must not"
    );
}

#[test]
fn min_score_filters_low_results() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    // Query along y — score will be ~0.0, below min_score of 0.5.
    let opts = SearchOpts {
        top_k: 5,
        min_score: Some(0.5),
        ..Default::default()
    };
    let hits = store.search(&["col"], &[0.0, 1.0, 0.0], &opts).unwrap();
    assert!(hits.is_empty(), "doc should be filtered by min_score");
}

#[test]
fn filter_scoping_in_search() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    let mut attrs_rust = BTreeMap::new();
    attrs_rust.insert("lang".to_string(), Value::Str("rust".to_string()));
    let mut attrs_go = BTreeMap::new();
    attrs_go.insert("lang".to_string(), Value::Str("go".to_string()));
    store
        .upsert(
            "col",
            &[
                rec_with("rust_doc", vec![1.0, 0.0, 0.0], attrs_rust),
                rec_with("go_doc", vec![1.0, 0.0, 0.0], attrs_go),
            ],
        )
        .unwrap();
    // Search with a filter restricting to Rust only.
    let opts = SearchOpts {
        top_k: 5,
        filter: Filter(vec![Predicate::Eq(
            "lang".to_string(),
            Value::Str("rust".to_string()),
        )]),
        ..Default::default()
    };
    let hits = store.search(&["col"], &[1.0, 0.0, 0.0], &opts).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "rust_doc");
}

#[test]
fn multi_collection_merged_search() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col_a").unwrap();
    store.create_collection("col_b").unwrap();
    // col_a has the nearest doc to query [1,0,0].
    store
        .upsert("col_a", &[rec("best", vec![1.0, 0.0, 0.0])])
        .unwrap();
    // col_b has a less-close doc.
    let h = std::f32::consts::FRAC_1_SQRT_2;
    store
        .upsert("col_b", &[rec("ok", vec![h, h, 0.0])])
        .unwrap();
    let hits = store
        .search(&["col_a", "col_b"], &[1.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 2);
    // The first hit should be "best" from col_a.
    assert_eq!(hits[0].id, "best");
    assert_eq!(hits[0].collection, "col_a");
    assert_eq!(hits[1].id, "ok");
    assert_eq!(hits[1].collection, "col_b");
}

#[test]
fn multi_collection_hit_collection_field() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("alpha").unwrap();
    store.create_collection("beta").unwrap();
    store.upsert("alpha", &[rec("a1", vec![1.0, 0.0])]).unwrap();
    store.upsert("beta", &[rec("b1", vec![0.0, 1.0])]).unwrap();
    let hits = store
        .search(&["alpha", "beta"], &[1.0, 0.0], &default_opts(5))
        .unwrap();
    // Each hit should carry the correct collection field.
    for hit in &hits {
        if hit.id == "a1" {
            assert_eq!(hit.collection, "alpha");
        } else if hit.id == "b1" {
            assert_eq!(hit.collection, "beta");
        } else {
            panic!("unexpected id: {}", hit.id);
        }
    }
}

#[test]
fn search_missing_collection_skipped() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("real").unwrap();
    store
        .upsert("real", &[rec("doc1", vec![1.0, 0.0])])
        .unwrap();
    // Include a non-existent collection — should not error.
    let hits = store
        .search(&["real", "phantom"], &[1.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "doc1");
}

#[test]
fn upsert_wrong_dimension_errors() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    let result = store.upsert("col", &[rec("doc1", vec![1.0, 0.0])]);
    assert!(result.is_err());
}

#[test]
fn upsert_implicitly_creates_collection() {
    let mut store = Store::in_memory(3).unwrap();
    // No explicit create_collection — upsert should auto-create.
    store
        .upsert("newcol", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    assert!(store.has_collection("newcol"));
}

#[test]
fn get_all_includes_vector() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    let records = store.get_all("col");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id, "doc1");
    // Vector should be unit-normalized (already unit here).
    assert_eq!(records[0].vector.as_deref().unwrap().len(), 3);
}

#[test]
fn get_hits_returns_record_with_attrs() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert(
            "col",
            &[rec_with(
                "doc1",
                vec![1.0, 0.0, 0.0],
                attrs_one("kind", "note"),
            )],
        )
        .unwrap();
    let record = store.get("col", "doc1").expect("doc1 should exist");
    assert_eq!(record.id, "doc1");
    assert_eq!(record.attrs.get("kind"), Some(&Value::Str("note".into())));
}

#[test]
fn get_unknown_id_in_real_collection_is_none() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    assert!(store.get("col", "nope").is_none());
}

#[test]
fn get_unknown_collection_is_none() {
    let store = Store::in_memory(3).unwrap();
    assert!(store.get("nope", "doc1").is_none());
}

#[test]
fn get_text_only_doc_has_no_vector() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert("col", &[text_rec("t1", attrs_one("kind", "note"))])
        .unwrap();
    let record = store.get("col", "t1").expect("t1 should exist");
    assert_eq!(record.vector, None);
}

#[test]
fn get_agrees_with_get_all() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    let via_get = store.get("col", "doc1").unwrap();
    let via_get_all = store
        .get_all("col")
        .into_iter()
        .find(|r| r.id == "doc1")
        .unwrap();
    assert_eq!(via_get.id, via_get_all.id);
    assert_eq!(via_get.vector, via_get_all.vector);
    assert_eq!(via_get.attrs, via_get_all.attrs);
}

#[test]
fn compact_in_memory_preserves_live_docs() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    store
        .upsert("col", &[rec("doc2", vec![0.0, 1.0, 0.0])])
        .unwrap();
    // Overwrite doc1 — creates a dead row.
    store
        .upsert("col", &[rec("doc1", vec![0.0, 0.0, 1.0])])
        .unwrap();
    store.compact().unwrap();
    assert_eq!(store.dead_rows, 0);
    // Both docs should still be searchable.
    let hits = store
        .search(&["col"], &[0.0, 0.0, 1.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, "doc1");
}

#[test]
fn drop_increments_dead_rows() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    store
        .upsert("col", &[rec("doc2", vec![0.0, 1.0, 0.0])])
        .unwrap();
    assert_eq!(store.dead_rows, 0);
    store.drop_collection("col").unwrap();
    assert_eq!(store.dead_rows, 2);
}

#[test]
fn top_k_limits_results() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("col").unwrap();
    for i in 0..10u32 {
        let v = vec![i as f32, 0.0];
        store.upsert("col", &[rec(&format!("doc{i}"), v)]).unwrap();
    }
    let hits = store
        .search(&["col"], &[1.0, 0.0], &default_opts(3))
        .unwrap();
    assert_eq!(hits.len(), 3);
}

#[test]
fn upsert_rolls_back_on_mid_batch_failure() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("col").unwrap();
    store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();

    let rows_before = store.data.row_count();
    let docs_before = store.get_all("col").len();
    let dead_before = store.dead_rows;

    // A 2-record batch where the first append succeeds and the second fails.
    store.data.fail_after(1);
    let res = store.upsert("col", &[rec("b", vec![0.0, 1.0]), rec("c", vec![1.0, 1.0])]);
    assert!(res.is_err());

    // Everything restored: no orphan row, index untouched, dead-count untouched.
    assert_eq!(
        store.data.row_count(),
        rows_before,
        "orphan row must be rolled back"
    );
    assert_eq!(store.get_all("col").len(), docs_before, "index unchanged");
    assert_eq!(store.dead_rows, dead_before);

    // Store remains usable for subsequent writes (disarm the seam first).
    store.data.fail_after(usize::MAX);
    store.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    assert_eq!(store.get_all("col").len(), 2);
}

#[test]
fn footprint_tracks_rows_dead_and_docs() {
    let mut store = Store::in_memory(4).unwrap();
    store.create_collection("col").unwrap();

    let fp0 = store.footprint();
    assert_eq!(fp0.rows, 0);
    assert_eq!(fp0.dead_rows, 0);
    assert_eq!(fp0.dimension, 4);
    assert_eq!(fp0.vector_bytes, 0);
    assert_eq!(fp0.doc_count, 0);

    store
        .upsert("col", &[rec("a", vec![1.0, 0.0, 0.0, 0.0])])
        .unwrap();
    store
        .upsert("col", &[rec("b", vec![0.0, 1.0, 0.0, 0.0])])
        .unwrap();
    let fp1 = store.footprint();
    assert_eq!(fp1.rows, 2);
    assert_eq!(fp1.dead_rows, 0);
    assert_eq!(fp1.vector_bytes, 2 * 4 * 4); // 2 rows × dim 4 × 4 bytes
    assert_eq!(fp1.doc_count, 2);

    // Overwrite "a": a dead row appears, doc_count stays at 2.
    store
        .upsert("col", &[rec("a", vec![0.0, 0.0, 1.0, 0.0])])
        .unwrap();
    let fp2 = store.footprint();
    assert_eq!(fp2.rows, 3);
    assert_eq!(fp2.dead_rows, 1);
    assert_eq!(fp2.doc_count, 2);

    // Compaction reclaims the dead row.
    store.compact().unwrap();
    let fp3 = store.footprint();
    assert_eq!(fp3.rows, 2);
    assert_eq!(fp3.dead_rows, 0);
    assert_eq!(fp3.doc_count, 2);
}

#[test]
fn max_vector_bytes_refuses_over_budget_upsert() {
    // Cap at exactly 2 rows (dim 2 × 4 bytes × 2 rows = 16 bytes).
    let config = Config::new("/dev/null/in-memory", 2)
        .open_mode(OpenMode::ReadWrite)
        .auto_compact(None)
        .max_vector_bytes(Some(16));
    let mut store = Store {
        fenced: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        last_verified: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        baseline_config: config.clone(),
        config,
        data: Segments::in_memory_with(2, Distance::Cosine),
        log: OpLog::in_memory(),
        persistence: None,
        memory: None,
        lock: None,
        lease: None,
        collections: HashMap::new(),
        dead_rows: 0,
        open_profile: OpenProfile::default(),
        quant: None,
        ann: None,
        seg_indexes: Vec::new(),
        seg_index_dirty: Vec::new(),
        ann_dirty: false,
        fts: crate::fts::Fts::default(),
        fts_dirty: false,
        findex: Default::default(),
        findex_dirty: false,
        in_memory: true,
        row_to_doc: Vec::new(),
        scan_order: std::sync::RwLock::new(None),
        loaded_log_offset: 0,
        manifest_cas: None,
        defer_barrier: false,
        pending_barrier: false,
        pinned: None,
        pruned_through: 0,
    };
    store.create_collection("col").unwrap();
    store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    store.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    assert_eq!(store.footprint().vector_bytes, 16);

    // The third row would exceed the cap — refuse, leaving the store intact.
    let res = store.upsert("col", &[rec("c", vec![1.0, 1.0])]);
    assert!(res.is_err());
    assert_eq!(store.footprint().rows, 2, "refused batch must not append");

    // Store stays usable for reads.
    let hits = store
        .search(&["col"], &[1.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 2);
}

// ── File-backed tests (ignored under Miri) ────────────────────────────

#[cfg_attr(miri, ignore)]
#[test]
fn open_refuses_data_file_over_cap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    // Write 3 rows (dim 2) with no cap.
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store.create_collection("col").unwrap();
        store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
        store.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
        store.upsert("col", &[rec("c", vec![1.0, 1.0])]).unwrap();
    }

    // Reopen with a cap below the on-disk size → clean Err, not a panic.
    let res = Store::open(Config::new(&path, 2).max_vector_bytes(Some(8)));
    assert!(res.is_err());
    let msg = res.err().unwrap().to_string();
    assert!(
        msg.contains("max_vector_bytes"),
        "error should mention the cap: {msg}"
    );

    // A cap at/above the size still opens fine.
    let ok = Store::open(Config::new(&path, 2).max_vector_bytes(Some(24)));
    assert!(ok.is_ok());
}

#[cfg_attr(miri, ignore)]
#[test]
fn upsert_rollback_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store.create_collection("col").unwrap();
        store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();

        // Next append fails immediately; the batch must fully roll back.
        store.data.fail_after(0);
        assert!(store.upsert("col", &[rec("b", vec![0.0, 1.0])]).is_err());
        assert_eq!(store.data.row_count(), 1);
        assert_eq!(store.get_all("col").len(), 1);
    }

    // Reopen: only "a" is present, replayed cleanly with no corruption.
    let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
    let recs = store.get_all("col");
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].id, "a");
}

// ── Fsync::OnFlush durability (nidus-4h2) ─────────────────────────────────

/// Everything written under `OnFlush` is present after `flush()` + reopen.
#[cfg_attr(miri, ignore)]
#[test]
fn on_flush_persists_every_batch_once_flushed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 2).fsync(Fsync::OnFlush)).unwrap();
        // Several separate calls: under `OnFlush` none of them syncs on its own, so this
        // is the case where a lost ordering guarantee would show up.
        for i in 0..8 {
            store
                .upsert("col", &[rec(&format!("d{i}"), vec![i as f32, 1.0])])
                .unwrap();
        }
        store.flush().unwrap();
    }

    let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
    assert_eq!(store.get_all("col").len(), 8);
}

/// A crash that leaves the log durable while the rows it names are not must drop the
/// dangling tail and reopen cleanly — never fail, never surface a phantom record.
#[cfg_attr(miri, ignore)]
#[test]
fn a_log_ahead_of_data_tail_is_dropped_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 2).fsync(Fsync::OnFlush)).unwrap();
        for i in 0..6 {
            store
                .upsert("col", &[rec(&format!("d{i}"), vec![i as f32, 1.0])])
                .unwrap();
        }
        store.flush().unwrap();
    }

    // Lose the last two rows of `data` while the log still names all six — rows are a
    // fixed `dim * 4` stride after the header, so dropping bytes off the end drops whole
    // rows and leaves the header and the prefix intact.
    let data_path = path.join("data");
    let len = std::fs::metadata(&data_path).unwrap().len();
    let stride = 2 * std::mem::size_of::<f32>() as u64;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&data_path)
        .unwrap()
        .set_len(len - 2 * stride)
        .unwrap();

    let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
    let recs = store.get_all("col");
    assert_eq!(recs.len(), 4, "the two rowless records must be ignored");
    let mut ids: Vec<&str> = recs.iter().map(|r| r.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, ["d0", "d1", "d2", "d3"]);

    // The recovered prefix is a working store, not just a readable one.
    let hits = store
        .search(
            &["col"],
            &[0.0, 1.0],
            &SearchOpts {
                top_k: 10,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(!hits.is_empty(), "recovered store must still serve search");
    assert!(hits.iter().all(|h| h.id != "d4" && h.id != "d5"));
}

#[cfg_attr(miri, ignore)]
#[test]
fn reopen_sees_prior_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    // Write some data.
    {
        let mut store = Store::open(Config::new(&path, 3)).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        store
            .upsert("col", &[rec("doc2", vec![0.0, 1.0, 0.0])])
            .unwrap();
    }

    // Reopen and verify.
    {
        let store = Store::open(Config::new(&path, 3).open_mode(OpenMode::ReadOnly)).unwrap();
        assert!(store.has_collection("col"));
        let records = store.get_all("col");
        assert_eq!(records.len(), 2);
        let ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
        assert!(ids.contains(&"doc1".to_string()));
        assert!(ids.contains(&"doc2".to_string()));
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn readonly_rejects_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    // Create a store and write something.
    {
        Store::open(Config::new(&path, 2)).unwrap();
    }

    // Open read-only.
    let mut store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();

    assert!(store.create_collection("col").is_err());
    assert!(store.drop_collection("col").is_err());
    assert!(store.set_meta("col", BTreeMap::new()).is_err());
    assert!(store.upsert("col", &[rec("doc1", vec![1.0, 0.0])]).is_err());
    assert!(store.delete("col", &["doc1"]).is_err());
    assert!(store.delete_where("col", &Filter::default()).is_err());
    assert!(store.flush().is_err());
    assert!(store.compact().is_err());
}

#[cfg_attr(miri, ignore)]
#[test]
fn compaction_preserves_live_docs_and_results() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    {
        let mut store = Store::open(Config::new(&path, 3).auto_compact(None)).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        store
            .upsert("col", &[rec("doc2", vec![0.0, 1.0, 0.0])])
            .unwrap();
        // Overwrite doc1 — creates a dead row.
        store
            .upsert("col", &[rec("doc1", vec![0.0, 0.0, 1.0])])
            .unwrap();
        assert_eq!(store.dead_rows, 1);
        store.compact().unwrap();
        assert_eq!(store.dead_rows, 0);

        // Verify search still works after compact.
        let hits = store
            .search(&["col"], &[0.0, 0.0, 1.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "doc1");
    }

    // Reopen and verify compacted state persists.
    {
        let store = Store::open(
            Config::new(&path, 3)
                .open_mode(OpenMode::ReadOnly)
                .auto_compact(None),
        )
        .unwrap();
        let records = store.get_all("col");
        assert_eq!(records.len(), 2);
        let hits = store
            .search(&["col"], &[0.0, 0.0, 1.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "doc1");
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn metadata_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut meta = BTreeMap::new();
    meta.insert("model".to_string(), "text-v3".to_string());

    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store.create_collection("col").unwrap();
        store.set_meta("col", meta.clone()).unwrap();
    }

    {
        let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
        assert_eq!(store.get_meta("col"), meta);
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn auto_compact_triggers_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    // Write with enough dead rows to trigger auto-compact (ratio > 0.5).
    {
        let mut store = Store::open(
            Config::new(&path, 3).auto_compact(None), // disable for setup
        )
        .unwrap();
        store.create_collection("col").unwrap();
        // Insert 3 docs then overwrite 2 of them → 2 dead of 5 total rows = 40%.
        // Then delete 1 more → 3 dead of 5 total > 50%.
        store
            .upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])])
            .unwrap();
        store
            .upsert("col", &[rec("b", vec![0.0, 1.0, 0.0])])
            .unwrap();
        store
            .upsert("col", &[rec("c", vec![0.0, 0.0, 1.0])])
            .unwrap();
        store
            .upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])])
            .unwrap(); // overwrite a
        store
            .upsert("col", &[rec("b", vec![0.0, 1.0, 0.0])])
            .unwrap(); // overwrite b
        // Now we have 5 rows, 2 dead (ratio = 0.4), 3 live docs.
        assert_eq!(store.dead_rows, 2);
    }

    // Reopen with auto_compact = Some(0.3) — should trigger compaction.
    {
        let store = Store::open(Config::new(&path, 3).auto_compact(Some(0.3))).unwrap();
        assert_eq!(store.dead_rows, 0, "auto-compact should have run");
        assert_eq!(store.get_all("col").len(), 3);
    }
}

/// #116: a read-only open past the dead-row threshold must open and skip the
/// compaction — it holds no writer lock to rewrite `data`/`log` with — rather than
/// die in `check_writable`. A writer opening afterwards still compacts.
#[cfg_attr(miri, ignore)]
#[test]
fn auto_compact_is_skipped_on_a_read_only_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    {
        let mut store = Store::open(Config::new(&path, 3).auto_compact(None)).unwrap();
        store.create_collection("col").unwrap();
        for (id, v) in [
            ("a", vec![1.0, 0.0, 0.0]),
            ("b", vec![0.0, 1.0, 0.0]),
            ("c", vec![0.0, 0.0, 1.0]),
        ] {
            store.upsert("col", &[rec(id, v)]).unwrap();
        }
        store
            .upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])])
            .unwrap();
        store
            .upsert("col", &[rec("b", vec![0.0, 1.0, 0.0])])
            .unwrap();
        assert_eq!(store.dead_rows, 2, "2 dead of 5 rows: past a 0.3 threshold");
    }

    {
        let store = Store::open(
            Config::new(&path, 3)
                .auto_compact(Some(0.3))
                .open_mode(OpenMode::ReadOnly),
        )
        .expect("a read-only open past the threshold must not fail (#116)");
        assert_eq!(store.dead_rows, 2, "read-only must not compact");
        assert_eq!(store.get_all("col").len(), 3);
    }

    {
        let store = Store::open(Config::new(&path, 3).auto_compact(Some(0.3))).unwrap();
        assert_eq!(store.dead_rows, 0, "the next writer open still compacts");
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn upsert_idempotent_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store.create_collection("col").unwrap();
        store.upsert("col", &[rec("doc1", vec![1.0, 0.0])]).unwrap();
        // Overwrite with a different vector.
        store.upsert("col", &[rec("doc1", vec![0.0, 1.0])]).unwrap();
    }

    {
        let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
        let records = store.get_all("col");
        assert_eq!(records.len(), 1);
        // The newest vector should win — search along y should score ~1.0.
        let hits = store
            .search(&["col"], &[0.0, 1.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 1.0).abs() < 1e-5);
    }
}

// ── Euclidean distance tests ─────────────────────────────────────────

#[test]
fn euclidean_exact_match_scores_zero() {
    let mut store = Store::in_memory_with(3, Distance::Euclidean).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 2.0, 3.0])])
        .unwrap();
    let hits = store
        .search(&["col"], &[1.0, 2.0, 3.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].score.abs() < 1e-6,
        "identical vectors should score 0.0, got {}",
        hits[0].score
    );
}

#[test]
fn euclidean_ranking_closer_first() {
    let mut store = Store::in_memory_with(3, Distance::Euclidean).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert(
            "col",
            &[
                rec("close", vec![0.9, 0.1, 0.0]),
                rec("far", vec![0.0, 1.0, 0.0]),
            ],
        )
        .unwrap();
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits[0].id, "close", "closer vector should rank first");
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn euclidean_does_not_normalize() {
    let mut store = Store::in_memory_with(2, Distance::Euclidean).unwrap();
    store.create_collection("col").unwrap();
    store.upsert("col", &[rec("doc1", vec![3.0, 4.0])]).unwrap();
    let records = store.get_all("col");
    assert_eq!(
        records[0].vector,
        Some(vec![3.0, 4.0]),
        "raw vectors preserved"
    );
}

#[test]
fn euclidean_min_score_filters() {
    let mut store = Store::in_memory_with(2, Distance::Euclidean).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![10.0, 0.0])])
        .unwrap();
    let opts = SearchOpts {
        top_k: 5,
        min_score: Some(-1.0),
        ..Default::default()
    };
    let hits = store.search(&["col"], &[0.0, 0.0], &opts).unwrap();
    assert!(
        hits.is_empty(),
        "score should be -100, below min_score of -1"
    );
}

// ── DotProduct distance tests ────────────────────────────────────────

#[test]
fn dotproduct_raw_dot() {
    let mut store = Store::in_memory_with(3, Distance::DotProduct).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert(
            "col",
            &[rec("a", vec![2.0, 0.0, 0.0]), rec("b", vec![1.0, 0.0, 0.0])],
        )
        .unwrap();
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits[0].id, "a", "higher magnitude should score higher");
    assert!(
        (hits[0].score - 2.0).abs() < 1e-6,
        "score = raw dot product"
    );
    assert!((hits[1].score - 1.0).abs() < 1e-6);
}

#[test]
fn dotproduct_does_not_normalize() {
    let mut store = Store::in_memory_with(2, Distance::DotProduct).unwrap();
    store.create_collection("col").unwrap();
    store.upsert("col", &[rec("doc1", vec![3.0, 4.0])]).unwrap();
    let records = store.get_all("col");
    assert_eq!(
        records[0].vector,
        Some(vec![3.0, 4.0]),
        "raw vectors preserved"
    );
}

#[test]
fn dotproduct_ranking_by_magnitude() {
    let mut store = Store::in_memory_with(2, Distance::DotProduct).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert(
            "col",
            &[rec("big", vec![10.0, 0.0]), rec("small", vec![1.0, 0.0])],
        )
        .unwrap();
    let hits = store
        .search(&["col"], &[1.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits[0].id, "big");
    assert!(hits[0].score > hits[1].score);
}

// ── Distance metric persistence tests ────────────────────────────────

#[cfg_attr(miri, ignore)]
#[test]
fn euclidean_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 3).distance(Distance::Euclidean)).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 2.0, 3.0])])
            .unwrap();
    }
    {
        let store = Store::open(
            Config::new(&path, 3)
                .distance(Distance::Euclidean)
                .open_mode(OpenMode::ReadOnly),
        )
        .unwrap();
        let records = store.get_all("col");
        assert_eq!(records[0].vector, Some(vec![1.0, 2.0, 3.0]));
        let hits = store
            .search(&["col"], &[1.0, 2.0, 3.0], &default_opts(5))
            .unwrap();
        assert!(hits[0].score.abs() < 1e-6);
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn distance_mismatch_on_reopen_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        Store::open(Config::new(&path, 3).distance(Distance::Euclidean)).unwrap();
    }
    let res = Store::open(Config::new(&path, 3).distance(Distance::Cosine));
    assert!(res.is_err());
    let msg = res.err().unwrap().to_string();
    assert!(
        msg.contains("distance"),
        "error should mention distance: {msg}"
    );
}

// ── Open-profile persistence tests (nidus-141) ────────────────────────

#[cfg_attr(miri, ignore)] // fsync
#[test]
fn recorded_profile_applies_on_reopen_with_bare_config() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 3)).unwrap();
        store
            .set_open_profile(&OpenProfile {
                ann: Some(AnnConfig::hnsw()),
                quantization: Some(Quantization::int8()),
                query_threads: Some(4),
                mmap: None,
            })
            .unwrap();
    }
    // A bare Config sets nothing explicitly, so every knob must resolve from the manifest.
    let store = Store::open(Config::new(&path, 3)).unwrap();
    assert_eq!(store.config().ann, Some(AnnConfig::hnsw()));
    assert_eq!(store.config().quantization, Some(Quantization::int8()));
    assert_eq!(store.config().query_threads, 4);
}

#[cfg_attr(miri, ignore)] // fsync
#[test]
fn explicit_flag_beats_recorded_profile_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 3)).unwrap();
        store
            .set_open_profile(&OpenProfile {
                ann: None,
                quantization: Some(Quantization::int8()),
                query_threads: Some(4),
                mmap: None,
            })
            .unwrap();
    }
    let store = Store::open(Config::new(&path, 3).query_threads(2)).unwrap();
    assert_eq!(
        store.config().query_threads,
        2,
        "an explicit builder call must beat the recorded profile"
    );
    assert_eq!(
        store.config().quantization,
        Some(Quantization::int8()),
        "a knob the caller never set explicitly still adopts the recorded profile"
    );
}

#[cfg_attr(miri, ignore)] // fsync
#[test]
fn no_recorded_profile_opens_on_built_in_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        Store::open(Config::new(&path, 3)).unwrap();
    }
    let store = Store::open(Config::new(&path, 3)).unwrap();
    assert_eq!(store.config().ann, None);
    assert_eq!(store.config().quantization, None);
    assert_eq!(store.config().query_threads, 1);
    assert!(!store.config().mmap);
}

#[cfg_attr(miri, ignore)] // fsync
#[test]
fn cleared_profile_reopens_on_built_in_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 3)).unwrap();
        store
            .set_open_profile(&OpenProfile {
                ann: Some(AnnConfig::hnsw()),
                quantization: Some(Quantization::int8()),
                query_threads: Some(4),
                mmap: None,
            })
            .unwrap();
    }
    {
        let mut store = Store::open(Config::new(&path, 3)).unwrap();
        assert_eq!(
            store.config().ann,
            Some(AnnConfig::hnsw()),
            "sanity: the profile applied before it is cleared"
        );
        store.clear_open_profile().unwrap();
    }
    let store = Store::open(Config::new(&path, 3)).unwrap();
    assert_eq!(store.config().ann, None);
    assert_eq!(store.config().quantization, None);
    assert_eq!(store.config().query_threads, 1);
}

#[cfg_attr(miri, ignore)] // fsync
#[test]
fn profile_survives_a_seal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 2).segment_max_rows(Some(1))).unwrap();
        store
            .set_open_profile(&OpenProfile {
                ann: None,
                quantization: Some(Quantization::int8()),
                query_threads: Some(3),
                mmap: None,
            })
            .unwrap();
        store.create_collection("col").unwrap();
        // The second upsert crosses the 1-row segment cap, forcing a seal (SPEC §14.4) — the
        // hazard this test guards: a seal must carry the profile forward, not drop it.
        store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
        store.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    }
    let store = Store::open(Config::new(&path, 2).segment_max_rows(Some(1))).unwrap();
    assert_eq!(
        store.config().quantization,
        Some(Quantization::int8()),
        "a seal must not revert the recorded profile to defaults"
    );
    assert_eq!(store.config().query_threads, 3);
}

#[cfg_attr(miri, ignore)] // fsync
#[test]
fn pre_profile_v1_manifest_opens_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 3)).unwrap();
        store.create_collection("col").unwrap();
        store
            .upsert("col", &[rec("doc1", vec![1.0, 2.0, 3.0])])
            .unwrap();
    }
    // Overwrite the v2 manifest `open` just wrote with a hand-built v1 blob (bincode's
    // positional format, no `profile` field) — a pre-nidus-141 manifest must still open.
    #[derive(serde::Serialize)]
    struct V1Shape {
        format_version: u16,
        dimension: u64,
        distance: Distance,
        segments: Vec<String>,
        next_id: u64,
        version: u64,
    }
    let v1 = V1Shape {
        format_version: 1,
        dimension: 3,
        distance: Distance::Cosine,
        segments: vec!["data".to_string()],
        next_id: 1,
        version: 1,
    };
    let payload = bincode::serialize(&v1).unwrap();
    let crc = crc32fast::hash(&payload);
    let mut bytes = Vec::with_capacity(4 + payload.len());
    bytes.extend_from_slice(&crc.to_le_bytes());
    bytes.extend_from_slice(&payload);
    std::fs::write(path.join("manifest"), &bytes).unwrap();

    let store = Store::open(Config::new(&path, 3)).unwrap();
    let records = store.get_all("col");
    assert_eq!(records.len(), 1);
    assert_eq!(store.open_profile, OpenProfile::default());
}

// ── list (metadata-only query) tests ─────────────────────────────────

#[test]
fn list_returns_all_matching() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    let mut a_rust = BTreeMap::new();
    a_rust.insert("lang".to_string(), Value::Str("rust".to_string()));
    let mut a_go = BTreeMap::new();
    a_go.insert("lang".to_string(), Value::Str("go".to_string()));
    store
        .upsert(
            "col",
            &[
                rec_with("r1", vec![1.0, 0.0, 0.0], a_rust.clone()),
                rec_with("r2", vec![0.0, 1.0, 0.0], a_rust),
                rec_with("g1", vec![0.0, 0.0, 1.0], a_go),
            ],
        )
        .unwrap();
    let filter = Filter(vec![Predicate::Eq(
        "lang".to_string(),
        Value::Str("rust".to_string()),
    )]);
    let hits = store
        .list(
            &["col"],
            &ListOpts {
                filter,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 2);
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert!(ids.contains(&"r1"));
    assert!(ids.contains(&"r2"));
}

#[test]
fn list_respects_limit() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("col").unwrap();
    for i in 0..10u32 {
        store
            .upsert("col", &[rec(&format!("d{i}"), vec![i as f32, 0.0])])
            .unwrap();
    }
    let hits = store
        .list(
            &["col"],
            &ListOpts {
                limit: 3,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 3);
}

#[test]
fn list_scores_are_zero() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("col").unwrap();
    store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    let hits = store
        .list(
            &["col"],
            &ListOpts {
                limit: 10,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].score, 0.0);
}

#[test]
fn list_empty_filter_returns_all() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("col").unwrap();
    store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    store.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    let hits = store.list(&["col"], &ListOpts::default()).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn list_multi_collection() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("a").unwrap();
    store.create_collection("b").unwrap();
    store.upsert("a", &[rec("x", vec![1.0, 0.0])]).unwrap();
    store.upsert("b", &[rec("y", vec![0.0, 1.0])]).unwrap();
    let hits = store.list(&["a", "b"], &ListOpts::default()).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn list_insertion_order() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("first", vec![1.0, 0.0])])
        .unwrap();
    store
        .upsert("col", &[rec("second", vec![0.0, 1.0])])
        .unwrap();
    let hits = store.list(&["col"], &ListOpts::default()).unwrap();
    assert_eq!(hits[0].id, "first");
    assert_eq!(hits[1].id, "second");
}

#[test]
fn list_offset_paginates() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("col").unwrap();
    for i in 0..10u32 {
        store
            .upsert("col", &[rec(&format!("d{i}"), vec![i as f32, 0.0])])
            .unwrap();
    }
    // Page through in windows of 3; concatenating the pages reproduces the
    // full insertion-ordered list with no gaps or repeats.
    let mut paged: Vec<String> = Vec::new();
    for page in 0..4 {
        let hits = store
            .list(
                &["col"],
                &ListOpts {
                    offset: page * 3,
                    limit: 3,
                    ..Default::default()
                },
            )
            .unwrap();
        paged.extend(hits.into_iter().map(|h| h.id));
    }
    let full: Vec<String> = store
        .list(&["col"], &ListOpts::default())
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(
        paged, full,
        "paginated windows must reconstruct the full list"
    );
    assert_eq!(paged.len(), 10);
}

#[test]
fn list_offset_past_end_is_empty() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("col").unwrap();
    store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    let hits = store
        .list(
            &["col"],
            &ListOpts {
                offset: 5,
                limit: 10,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(hits.is_empty());
}

// ── scan-order cache (nidus-dxt) ─────────────────────────────────────
// The whole-store fast path caches a row-sorted scan across queries; these pin that every write
// changing the doc set invalidates it, so a search after a write never reads a stale order.

#[test]
fn scan_cache_reflects_upsert_between_searches() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    // First search builds the cache.
    let hits = store
        .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 1);
    // A new doc lands on a fresh row — the cache must pick it up next query.
    store
        .upsert("col", &[rec("doc2", vec![0.0, 1.0, 0.0])])
        .unwrap();
    let hits = store
        .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 2, "second search must see the upserted doc");
    assert_eq!(hits[0].id, "doc2", "new doc is the nearest to the query");
}

#[test]
fn scan_cache_reflects_delete_between_searches() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert(
            "col",
            &[
                rec("doc1", vec![1.0, 0.0, 0.0]),
                rec("doc2", vec![0.0, 1.0, 0.0]),
            ],
        )
        .unwrap();
    // Build the cache.
    assert_eq!(
        store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap()
            .len(),
        2
    );
    // Delete and re-search: a stale cache would still rank the dead row.
    store.delete("col", &["doc1"]).unwrap();
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "doc2");
}

#[test]
fn scan_cache_overwrite_uses_new_vector() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    // Build the cache against the original row.
    store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    // Overwrite doc1 — old row goes dead, new row is appended.
    store
        .upsert("col", &[rec("doc1", vec![0.0, 1.0, 0.0])])
        .unwrap();
    let hits = store
        .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        (hits[0].score - 1.0).abs() < 1e-6,
        "search must score the overwritten vector, not the dead row"
    );
}

#[test]
fn scan_cache_survives_compact() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert(
            "col",
            &[
                rec("a", vec![1.0, 0.0, 0.0]),
                rec("b", vec![0.0, 1.0, 0.0]),
                rec("c", vec![0.0, 0.0, 1.0]),
            ],
        )
        .unwrap();
    store.delete("col", &["b"]).unwrap();
    // Build the cache while a dead row exists.
    store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    // Compaction renumbers every live row — the cache must be rebuilt against them.
    store.compact().unwrap();
    let hits = store
        .search(&["col"], &[0.0, 0.0, 1.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, "c");
}

#[test]
fn scan_cache_whole_store_filter_matches_subset_path() {
    // The whole-store cache path filters via a per-entry attr lookup; the subset
    // path filters inline. Both must agree. Build one collection with attrs and
    // compare a filtered whole-store search against the same filter via subset.
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    let tag = |t: &str| {
        let mut m = BTreeMap::new();
        m.insert("tag".to_string(), Value::Str(t.to_string()));
        m
    };
    store
        .upsert(
            "col",
            &[
                rec_with("a", vec![1.0, 0.0, 0.0], tag("keep")),
                rec_with("b", vec![0.9, 0.1, 0.0], tag("drop")),
                rec_with("c", vec![0.8, 0.2, 0.0], tag("keep")),
            ],
        )
        .unwrap();
    let opts = SearchOpts {
        top_k: 5,
        filter: Filter(vec![Predicate::Eq(
            "tag".to_string(),
            Value::Str("keep".to_string()),
        )]),
        ..Default::default()
    };
    let hits = store.search(&["col"], &[1.0, 0.0, 0.0], &opts).unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, vec!["a", "c"], "filter must keep only tagged docs");
}

#[test]
fn scan_cache_subset_scope_excludes_other_collections() {
    // A strict subset scope takes the direct (non-cache) path; it must not leak
    // docs from out-of-scope collections, and the cache (built by a prior whole-
    // store search) must not interfere.
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("a").unwrap();
    store.create_collection("b").unwrap();
    store
        .upsert("a", &[rec("a1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    store
        .upsert("b", &[rec("b1", vec![0.0, 1.0, 0.0])])
        .unwrap();
    // Whole-store search builds the global cache.
    assert_eq!(
        store
            .search(&["a", "b"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap()
            .len(),
        2
    );
    // Subset search must see only collection "a".
    let hits = store
        .search(&["a"], &[1.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "a1");
}

// ── int8 scalar quantization tests ───────────────────────────────────

fn quantized_store(dim: usize) -> Store {
    Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .open_mode(OpenMode::ReadWrite)
            .auto_compact(None)
            .quantization(Some(Quantization::default())),
    )
    .unwrap()
}

#[test]
fn quantized_search_ranking_matches_exact() {
    let mut store = quantized_store(3);
    store.create_collection("col").unwrap();
    store
        .upsert(
            "col",
            &[
                rec("close", vec![0.9, 0.1, 0.0]),
                rec("mid", vec![0.5, 0.5, 0.0]),
                rec("far", vec![0.0, 0.0, 1.0]),
            ],
        )
        .unwrap();
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(3))
        .unwrap();
    assert_eq!(
        hits[0].id, "close",
        "quantized search should rank correctly"
    );
}

#[test]
fn quantized_search_respects_top_k() {
    let mut store = quantized_store(2);
    store.create_collection("col").unwrap();
    for i in 0..20u32 {
        store
            .upsert("col", &[rec(&format!("d{i}"), vec![i as f32, 0.0])])
            .unwrap();
    }
    let hits = store
        .search(&["col"], &[19.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 5);
}

#[test]
fn quantized_search_with_filter() {
    let mut store = quantized_store(3);
    store.create_collection("col").unwrap();
    let mut a_rust = BTreeMap::new();
    a_rust.insert("lang".to_string(), Value::Str("rust".to_string()));
    let mut a_go = BTreeMap::new();
    a_go.insert("lang".to_string(), Value::Str("go".to_string()));
    store
        .upsert(
            "col",
            &[
                rec_with("r1", vec![1.0, 0.0, 0.0], a_rust),
                rec_with("g1", vec![1.0, 0.0, 0.0], a_go),
            ],
        )
        .unwrap();
    let opts = SearchOpts {
        top_k: 5,
        filter: Filter(vec![Predicate::Eq(
            "lang".to_string(),
            Value::Str("rust".to_string()),
        )]),
        ..Default::default()
    };
    let hits = store.search(&["col"], &[1.0, 0.0, 0.0], &opts).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "r1");
}

#[test]
fn quantized_search_euclidean() {
    let mut store = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", 3)
            .distance(Distance::Euclidean)
            .open_mode(OpenMode::ReadWrite)
            .auto_compact(None)
            .quantization(Some(Quantization::default())),
    )
    .unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert(
            "col",
            &[
                rec("exact", vec![1.0, 2.0, 3.0]),
                rec("far", vec![10.0, 20.0, 30.0]),
            ],
        )
        .unwrap();
    let hits = store
        .search(&["col"], &[1.0, 2.0, 3.0], &default_opts(2))
        .unwrap();
    assert_eq!(hits[0].id, "exact");
}

#[test]
fn quantized_survives_compact() {
    let mut store = quantized_store(3);
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])])
        .unwrap();
    store
        .upsert("col", &[rec("a", vec![0.0, 1.0, 0.0])])
        .unwrap();
    store.compact().unwrap();
    let hits = store
        .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!((hits[0].score - 1.0).abs() < 1e-5);
}

#[test]
fn quantized_empty_store_searches_ok() {
    let store = quantized_store(3);
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn quantized_incremental_matches_bulk() {
    // The int8 matrix must stay correct whether built in one batch or many.
    // Build the same data two ways and assert identical search rankings.
    let make = |incremental: bool| {
        let mut store = quantized_store(4);
        store.create_collection("col").unwrap();
        let recs: Vec<Record> = (0..50u32)
            .map(|i| {
                let a = i as f32 * 0.01;
                rec(&format!("d{i}"), vec![a, 1.0 - a, 0.5, -a])
            })
            .collect();
        if incremental {
            for r in &recs {
                store.upsert("col", std::slice::from_ref(r)).unwrap();
            }
        } else {
            store.upsert("col", &recs).unwrap();
        }
        store
    };
    let bulk = make(false);
    let incr = make(true);
    let q = vec![0.2, 0.8, 0.5, -0.2];
    let hb = bulk.search(&["col"], &q, &default_opts(10)).unwrap();
    let hi = incr.search(&["col"], &q, &default_opts(10)).unwrap();
    let ids_b: Vec<&str> = hb.iter().map(|h| h.id.as_str()).collect();
    let ids_i: Vec<&str> = hi.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids_b, ids_i, "incremental and bulk must rank identically");
}

#[test]
fn quantized_incremental_keeps_full_recall() {
    // Drip-feed rows one at a time, then confirm an exact-match query still
    // finds its target (incremental quantization must not lose the vector).
    let mut store = quantized_store(3);
    store.create_collection("col").unwrap();
    for i in 0..30u32 {
        let v = vec![i as f32, (30 - i) as f32, 1.0];
        store.upsert("col", &[rec(&format!("d{i}"), v)]).unwrap();
    }
    // Query exactly matches d7.
    let hits = store
        .search(&["col"], &[7.0, 23.0, 1.0], &default_opts(1))
        .unwrap();
    assert_eq!(hits[0].id, "d7");
}

#[test]
fn quantized_refit_tracks_row_growth() {
    // params_rows must follow the geometric-refit rule: it only jumps when the
    // row count crosses REFIT_GROWTH× the last fit set, not on every batch.
    let mut store = quantized_store(2);
    store.create_collection("col").unwrap();
    // First batch (2 rows): refit from 0 → params_rows = 2.
    store
        .upsert("col", &[rec("a", vec![1.0, 0.0]), rec("b", vec![0.0, 1.0])])
        .unwrap();
    assert_eq!(int8_state(&store).params_rows, 2);
    // One more row (total 3): 3 <= 2*2, so NO refit — params_rows stays 2.
    store.upsert("col", &[rec("c", vec![1.0, 1.0])]).unwrap();
    assert_eq!(int8_state(&store).params_rows, 2);
    // Push past 2*2=4 (total 5): refit fires → params_rows = 5.
    store
        .upsert("col", &[rec("d", vec![2.0, 0.0]), rec("e", vec![0.0, 2.0])])
        .unwrap();
    assert_eq!(int8_state(&store).params_rows, 5);
    // The int8 matrix always covers every physical row.
    let dim = store.data.dimension();
    assert_eq!(
        int8_state(&store).vectors.len(),
        store.data.row_count() as usize * dim
    );
}

// ── binary (sign-bit) quantization tests ─────────────────────────────

/// A deterministic xorshift pseudo-random vector in roughly [-0.5, 0.5)^dim, for
/// recall/parallel tests where structured modulo data would produce Hamming ties.
fn pseudo_vec(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn binary_store(dim: usize) -> Store {
    Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .distance(Distance::Cosine)
            .open_mode(OpenMode::ReadWrite)
            .auto_compact(None)
            .quantization(Some(Quantization::binary())),
    )
    .unwrap()
}

/// Extract the binary state, panicking if quant is off or int8.
fn bin_state(store: &Store) -> &BinState {
    match store
        .quant
        .as_ref()
        .expect("quantization should be enabled")
    {
        Quant::Binary(s) => s,
        Quant::Int8(_) => panic!("expected binary quant state, found int8"),
    }
}

#[test]
fn binary_rejects_non_cosine() {
    // Sign codes are an angular proxy; binary must refuse dot-product / Euclidean.
    for distance in [Distance::DotProduct, Distance::Euclidean] {
        let result = Store::in_memory_cfg(
            Config::new("/dev/null/in-memory", 4)
                .distance(distance)
                .open_mode(OpenMode::ReadWrite)
                .quantization(Some(Quantization::binary())),
        );
        let err = match result {
            Ok(_) => panic!("binary quantization must be rejected for {distance:?}"),
            Err(e) => e,
        };
        assert!(
            err.to_string()
                .contains("binary quantization requires Distance::Cosine"),
            "expected cosine-only rejection, got: {err}"
        );
    }
    // Cosine is accepted.
    assert!(
        Store::in_memory_cfg(
            Config::new("/dev/null/in-memory", 4)
                .distance(Distance::Cosine)
                .open_mode(OpenMode::ReadWrite)
                .quantization(Some(Quantization::binary())),
        )
        .is_ok()
    );
}

#[test]
fn binary_search_ranks_correctly() {
    let mut store = binary_store(3);
    store.create_collection("col").unwrap();
    store
        .upsert(
            "col",
            &[
                rec("close", vec![0.9, 0.1, 0.0]),
                rec("mid", vec![0.6, 0.5, 0.1]),
                rec("far", vec![-1.0, -0.2, 0.3]),
            ],
        )
        .unwrap();
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(3))
        .unwrap();
    assert_eq!(
        hits[0].id, "close",
        "binary first-pass + f32 rerank should rank correctly"
    );
    // The reranked score is the exact f32 cosine, not a Hamming proxy.
    assert!(hits[0].score <= 1.0 + 1e-6 && hits[0].score >= -1.0 - 1e-6);
}

#[test]
fn binary_state_covers_all_rows_multiword() {
    // dim 130 → 3 u64 words per row; words must cover every physical row.
    let mut store = binary_store(130);
    store.create_collection("col").unwrap();
    for i in 0..7u32 {
        store
            .upsert(
                "col",
                &[rec(&format!("d{i}"), pseudo_vec(i as u64 + 1, 130))],
            )
            .unwrap();
    }
    assert_eq!(bin_state(&store).words_per_row, 130usize.div_ceil(64)); // == 3
    assert_eq!(
        bin_state(&store).words.len(),
        store.data.row_count() as usize * 3
    );
}

// Ignored under Miri: builds thousands of rows to make recall meaningful — far too
// slow at Miri's ~100x. Pure in-RAM logic, covered amply by the f32/serial path.
#[cfg_attr(miri, ignore)] // N=2000 x 128-dim binary recall sweep; far too slow under Miri.
#[test]
fn binary_search_recall_high_vs_exact() {
    let dim = 128;
    let n = 2000usize;
    let k = 10usize;
    let mut exact = Store::in_memory_with(dim, Distance::Cosine).unwrap();
    let mut bin = binary_store(dim);
    exact.create_collection("c").unwrap();
    bin.create_collection("c").unwrap();
    for i in 0..n {
        let r = rec(&format!("d{i}"), pseudo_vec(i as u64 + 1, dim));
        exact.upsert("c", std::slice::from_ref(&r)).unwrap();
        bin.upsert("c", &[r]).unwrap();
    }
    let (mut hit, mut total) = (0usize, 0usize);
    for qi in 0..20u64 {
        let q = pseudo_vec(1_000_000 + qi, dim);
        let truth: Vec<String> = exact
            .search(&["c"], &q, &default_opts(k))
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let got: std::collections::HashSet<String> = bin
            .search(&["c"], &q, &default_opts(k))
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        for id in &truth {
            if got.contains(id) {
                hit += 1;
            }
            total += 1;
        }
    }
    let recall = hit as f64 / total as f64;
    assert!(recall >= 0.6, "binary recall@{k} too low: {recall:.3}");
}

/// Build a binary-quantized store with `n` pseudo-random rows and the given threads.
fn binary_pseudo_store(dim: usize, n: usize, threads: usize) -> Store {
    let mut store = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .distance(Distance::Cosine)
            .open_mode(OpenMode::ReadWrite)
            .auto_compact(None)
            .query_threads(threads)
            .quantization(Some(Quantization::binary())),
    )
    .unwrap();
    store.create_collection("col").unwrap();
    let recs: Vec<Record> = (0..n)
        .map(|i| rec(&format!("d{i}"), pseudo_vec(i as u64 + 1, dim)))
        .collect();
    store.upsert("col", &recs).unwrap();
    store
}

// Ignored under Miri — needs to clear PARALLEL_SCAN_WORK_FLOOR to engage threads.
#[cfg_attr(miri, ignore)] // spawns query threads over ~1.5k x 768-dim rows; threads + size not for Miri.
#[test]
fn binary_parallel_matches_serial() {
    // Pseudo-random sign codes make Hamming ties near the overscan boundary
    // vanishingly unlikely, so serial and parallel select the same candidates and
    // rerank to byte-identical ordered results.
    let dim = 768;
    let n = rows_to_parallelize(dim) + 100;
    let serial = binary_pseudo_store(dim, n, 1);
    let parallel = binary_pseudo_store(dim, n, 4);
    let q = pseudo_vec(7_000_001, dim);
    let hs: Vec<String> = serial
        .search(&["col"], &q, &default_opts(20))
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    let hp: Vec<String> = parallel
        .search(&["col"], &q, &default_opts(20))
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(hs, hp, "binary parallel scan must match serial");
}

// ── parallel scan tests ──────────────────────────────────────────────

/// Rows needed at `dim` to clear [`PARALLEL_SCAN_WORK_FLOOR`], so the threaded path
/// actually engages. Keeps the parallel tests robust to the constant's value (and
/// fast: a wide dim hits the work floor at far fewer rows than a narrow one).
fn rows_to_parallelize(dim: usize) -> usize {
    PARALLEL_SCAN_WORK_FLOOR.div_ceil(dim) + 1
}

/// Build an in-memory store with `n` deterministic pseudo-random rows, the given
/// `query_threads`, and optional int8 quantization.
fn threaded_store_cfg(dim: usize, n: usize, threads: usize, quant: bool) -> Store {
    let mut cfg = Config::new("/dev/null/in-memory", dim)
        .open_mode(OpenMode::ReadWrite)
        .auto_compact(None)
        .query_threads(threads);
    if quant {
        cfg = cfg.quantization(Some(Quantization::default()));
    }
    let mut store = Store::in_memory_cfg(cfg).unwrap();
    store.create_collection("col").unwrap();
    let recs: Vec<Record> = (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i * 31 + d * 7) % 97) as f32 - 48.0)
                .collect();
            rec(&format!("d{i}"), v)
        })
        .collect();
    store.upsert("col", &recs).unwrap();
    store
}

fn threaded_store(dim: usize, n: usize, threads: usize) -> Store {
    threaded_store_cfg(dim, n, threads, false)
}

// Ignored under Miri: clearing PARALLEL_SCAN_WORK_FLOOR takes minutes at Miri's ~100x slowdown,
// and the scan is safe Rust over shared `&` reads that the borrow checker already proves
// data-race-free, so Miri adds no coverage.
#[cfg_attr(miri, ignore)] // spawns query threads over ~1.5k x 768-dim rows; threads + size not for Miri.
#[test]
fn parallel_search_matches_serial() {
    // A wide dim clears the work floor at ~1.4k rows — far cheaper than narrow dims.
    let dim = 768;
    let n = rows_to_parallelize(dim) + 100; // exceed the floor so threading engages
    let serial = threaded_store(dim, n, 1);
    let parallel = threaded_store(dim, n, 4);
    let q: Vec<f32> = (0..dim).map(|d| (d * 5 % 13) as f32 - 6.0).collect();
    let hs = serial.search(&["col"], &q, &default_opts(20)).unwrap();
    let hp = parallel.search(&["col"], &q, &default_opts(20)).unwrap();
    assert_eq!(hs.len(), hp.len());
    // The sorted score sequence must be byte-identical (exact f32 over the same
    // data); only tie-breaking among equal scores may differ.
    for (a, b) in hs.iter().zip(&hp) {
        assert!(
            (a.score - b.score).abs() < 1e-6,
            "score mismatch: serial {} vs parallel {}",
            a.score,
            b.score
        );
    }
}

// Ignored under Miri — same reason as `parallel_search_matches_serial`.
#[cfg_attr(miri, ignore)] // spawns query threads over ~1.5k x 768-dim rows; ran >10min under Miri.
#[test]
fn parallel_search_respects_filter_and_min_score() {
    let dim = 768;
    let n = rows_to_parallelize(dim) + 100;
    let parallel = threaded_store(dim, n, 4);
    let q: Vec<f32> = (0..dim).map(|d| (d * 5 % 13) as f32 - 6.0).collect();
    // A min_score floor must be honored across all worker chunks.
    let opts = SearchOpts {
        top_k: 30,
        min_score: Some(0.99),
        ..Default::default()
    };
    let hits = parallel.search(&["col"], &q, &opts).unwrap();
    assert!(hits.iter().all(|h| h.score >= 0.99));
}

// The quantized first pass scales across threads; its parallel and serial candidate
// sets must produce the same final ranking. Ignored under Miri (same cost reason).
#[cfg_attr(miri, ignore)] // spawns query threads over ~1.5k x 768-dim rows; threads + size not for Miri.
#[test]
fn parallel_quantized_matches_serial() {
    let dim = 768;
    let n = rows_to_parallelize(dim) + 100;
    let serial = threaded_store_cfg(dim, n, 1, true);
    let parallel = threaded_store_cfg(dim, n, 4, true);
    let q: Vec<f32> = (0..dim).map(|d| (d * 5 % 13) as f32 - 6.0).collect();
    let hs = serial.search(&["col"], &q, &default_opts(20)).unwrap();
    let hp = parallel.search(&["col"], &q, &default_opts(20)).unwrap();
    assert_eq!(hs.len(), hp.len());
    // Same int8 candidate set (just scored in chunks) → same f32 rerank scores.
    for (a, b) in hs.iter().zip(&hp) {
        assert!(
            (a.score - b.score).abs() < 1e-6,
            "score mismatch: serial {} vs parallel {}",
            a.score,
            b.score
        );
    }
}

#[test]
fn parallel_below_floor_falls_back_to_serial() {
    // Fewer rows than the floor: the parallel branch is skipped, but results
    // must still be correct.
    let store = threaded_store(4, 10, 8);
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 5);
    // Scores are non-increasing.
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score);
    }
}

#[test]
fn parallel_search_with_quantization() {
    // query_threads is set and quantization is on, but the scan is below the work
    // floor: the quantized path runs single-threaded and must still be correct.
    let store = threaded_store_cfg(8, 200, 4, true);
    let q: Vec<f32> = (0..8).map(|d| (d * 2 % 7) as f32).collect();
    let hits = store.search(&["col"], &q, &default_opts(10)).unwrap();
    assert_eq!(hits.len(), 10);
}

// ── ANN ─────────────────────────────────────────────────────────────────────

use crate::ann::SplitMix64;
use crate::model::AnnConfig;

/// `n` deterministic random unit vectors of dimension `dim`.
fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix64::new(seed);
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim)
                .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
                .collect();
            normalize(&mut v);
            v
        })
        .collect()
}

fn ann_store(dim: usize, cfg: AnnConfig, vectors: &[Vec<f32>]) -> Store {
    let mut s = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .auto_compact(None)
            .ann(Some(cfg)),
    )
    .unwrap();
    let recs: Vec<Record> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
        .collect();
    s.upsert("col", &recs).unwrap();
    s
}

fn exact_store(dim: usize, vectors: &[Vec<f32>]) -> Store {
    let mut s = Store::in_memory(dim).unwrap();
    let recs: Vec<Record> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
        .collect();
    s.upsert("col", &recs).unwrap();
    s
}

/// Mean recall@k of `ann` against the exact brute-force `truth` over `queries`.
fn mean_recall(ann: &Store, truth: &Store, queries: &[Vec<f32>], k: usize) -> f32 {
    let mut total = 0.0f32;
    for q in queries {
        let exact: std::collections::HashSet<String> = truth
            .search(&["col"], q, &default_opts(k))
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let got = ann.search(&["col"], q, &default_opts(k)).unwrap();
        let hit = got.iter().filter(|h| exact.contains(&h.id)).count();
        total += hit as f32 / k as f32;
    }
    total / queries.len() as f32
}

#[test]
#[cfg_attr(miri, ignore)] // N=2000 build is too slow under Miri; logic is covered in ann/.
fn hnsw_recall_matches_exact() {
    let (n, dim, k) = (2000, 32, 10);
    let data = random_unit_vectors(n, dim, 1);
    let queries = random_unit_vectors(50, dim, 2);
    let ann = ann_store(dim, AnnConfig::hnsw(), &data);
    let truth = exact_store(dim, &data);
    let recall = mean_recall(&ann, &truth, &queries, k);
    assert!(
        recall >= 0.90,
        "HNSW recall@{k} = {recall:.3}, expected >= 0.90"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // builds a parallel HNSW graph; threads + size not for Miri.
fn hnsw_parallel_build_recall_matches_serial() {
    // A parallel build produces a different-but-equivalent graph; recall should
    // stay in the same ballpark as the serial build on the same data.
    let (n, dim, k) = (1500, 32, 10); // > PARALLEL_BUILD_MIN so the parallel path runs
    let data = random_unit_vectors(n, dim, 7);
    let queries = random_unit_vectors(30, dim, 8);
    let truth = exact_store(dim, &data);

    let serial = ann_store(dim, AnnConfig::hnsw(), &data); // query_threads defaults to 1
    let parallel = {
        let mut s = Store::in_memory_cfg(
            Config::new("/dev/null/in-memory", dim)
                .auto_compact(None)
                .query_threads(4)
                .ann(Some(AnnConfig::hnsw())),
        )
        .unwrap();
        let recs: Vec<Record> = data
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
            .collect();
        // upsert builds incrementally (serial); force the parallel from-scratch
        // build path via compact (rebuild_ann under query_threads=4).
        s.upsert("col", &recs).unwrap();
        s.compact().unwrap();
        s
    };

    let serial_recall = mean_recall(&serial, &truth, &queries, k);
    let parallel_recall = mean_recall(&parallel, &truth, &queries, k);
    assert!(
        parallel_recall >= serial_recall - 0.05,
        "parallel recall {parallel_recall:.3} should track serial {serial_recall:.3}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn ivf_recall_matches_exact() {
    let (n, dim, k) = (2000, 32, 10);
    let data = random_unit_vectors(n, dim, 3);
    let queries = random_unit_vectors(50, dim, 4);
    // Probe a generous fraction of lists so recall is solid.
    let ann = ann_store(dim, AnnConfig::ivf().n_probe(12), &data);
    let truth = exact_store(dim, &data);
    let recall = mean_recall(&ann, &truth, &queries, k);
    assert!(
        recall >= 0.70,
        "IVF recall@{k} = {recall:.3}, expected >= 0.70"
    );
}

/// Small-N correctness that stays Miri-clean (no fsync, tiny build).
#[test]
fn ann_finds_exact_match_small() {
    let data = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    for cfg in [AnnConfig::hnsw(), AnnConfig::ivf().n_probe(8)] {
        let s = ann_store(3, cfg, &data);
        let hits = s
            .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(1))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "d1", "{cfg:?} should find the exact match");
    }
}

#[test]
#[cfg_attr(miri, ignore)] // N=200 HNSW build is slow under Miri; tiny cases cover the path.
fn ann_post_filter_returns_only_matching() {
    // Half the docs carry kind=a, half kind=b; an ANN query filtered to kind=b must
    // never return a kind=a doc.
    let dim = 16;
    let data = random_unit_vectors(200, dim, 5);
    let mut s = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .auto_compact(None)
            .ann(Some(AnnConfig::hnsw().overscan(8))),
    )
    .unwrap();
    let recs: Vec<Record> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut attrs = BTreeMap::new();
            let kind = if i % 2 == 0 { "a" } else { "b" };
            attrs.insert("kind".to_string(), Value::Str(kind.to_string()));
            rec_with(&format!("d{i}"), v.clone(), attrs)
        })
        .collect();
    s.upsert("col", &recs).unwrap();

    let opts = SearchOpts {
        top_k: 10,
        filter: Filter(vec![Predicate::Eq(
            "kind".to_string(),
            Value::Str("b".to_string()),
        )]),
        ..Default::default()
    };
    let hits = s.search(&["col"], &data[1], &opts).unwrap();
    assert!(!hits.is_empty(), "filtered ANN should still return results");
    for h in &hits {
        // d1, d3, … are odd indices = kind b.
        let idx: usize = h.id.trim_start_matches('d').parse().unwrap();
        assert_eq!(idx % 2, 1, "{} leaked into a kind=b query", h.id);
    }
}

#[test]
fn ann_skips_deleted_rows() {
    let data = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    let mut s = ann_store(3, AnnConfig::hnsw(), &data);
    // Delete the nearest doc to a +x query; its graph node is now stale.
    s.delete("col", &["d0"]).unwrap();
    let hits = s
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(3))
        .unwrap();
    assert!(
        hits.iter().all(|h| h.id != "d0"),
        "deleted doc must not appear: {hits:?}"
    );
    // The next-nearest live doc should now lead.
    assert_eq!(hits[0].id, "d1");
}

/// An ANN store that also quantizes — the walk scores quantized codes and the store
/// reranks candidates with the exact f32 score (nidus-ndu).
fn ann_quant_store(dim: usize, cfg: AnnConfig, quant: Quantization, vectors: &[Vec<f32>]) -> Store {
    let mut s = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .auto_compact(None)
            .ann(Some(cfg))
            .quantization(Some(quant)),
    )
    .unwrap();
    let recs: Vec<Record> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
        .collect();
    s.upsert("col", &recs).unwrap();
    s
}

// ANN + quantization: the walk runs in the quantized space and the f32 rerank restores accuracy.
// Coarse codes steer the walk less precisely, so recall sits a touch below exact-walk ANN and the
// thresholds are looser than above — still well clear of chance.

#[test]
#[cfg_attr(miri, ignore)] // N=2000 HNSW build is too slow under Miri.
fn hnsw_int8_walk_recall() {
    let (n, dim, k) = (2000, 32, 10);
    let data = random_unit_vectors(n, dim, 11);
    let queries = random_unit_vectors(50, dim, 12);
    let truth = exact_store(dim, &data);
    let ann = ann_quant_store(dim, AnnConfig::hnsw(), Quantization::default(), &data);
    let recall = mean_recall(&ann, &truth, &queries, k);
    assert!(
        recall >= 0.85,
        "HNSW+int8 recall@{k} = {recall:.3}, expected >= 0.85"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // N=2000 HNSW build is too slow under Miri.
fn hnsw_binary_walk_recall() {
    let (n, dim, k) = (2000, 64, 10);
    let data = random_unit_vectors(n, dim, 13);
    let queries = random_unit_vectors(50, dim, 14);
    let truth = exact_store(dim, &data);
    // Binary codes are the coarsest proxy; a wider beam/over-fetch keeps recall solid.
    let ann = ann_quant_store(
        dim,
        AnnConfig::hnsw().ef_search(128).overscan(16),
        Quantization::binary(),
        &data,
    );
    let recall = mean_recall(&ann, &truth, &queries, k);
    assert!(
        recall >= 0.70,
        "HNSW+binary recall@{k} = {recall:.3}, expected >= 0.70"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // N=2000 IVF build is too slow under Miri.
fn ivf_int8_walk_recall() {
    let (n, dim, k) = (2000, 32, 10);
    let data = random_unit_vectors(n, dim, 15);
    let queries = random_unit_vectors(50, dim, 16);
    let truth = exact_store(dim, &data);
    let ann = ann_quant_store(
        dim,
        AnnConfig::ivf().n_probe(12),
        Quantization::default(),
        &data,
    );
    let recall = mean_recall(&ann, &truth, &queries, k);
    assert!(
        recall >= 0.65,
        "IVF+int8 recall@{k} = {recall:.3}, expected >= 0.65"
    );
}

/// The combination is accepted at `open` (the v1 mutual-exclusion is lifted) and the
/// quantized-walk path returns exactly `top_k` ranked hits on a tiny store — Miri-clean.
#[test]
fn ann_with_quantization_is_accepted() {
    let data: Vec<Vec<f32>> = (0..8)
        .map(|i| {
            let t = i as f32 / 8.0;
            let mut v = vec![t.cos(), t.sin(), 0.25, -0.5];
            normalize(&mut v);
            v
        })
        .collect();
    let s = ann_quant_store(4, AnnConfig::hnsw(), Quantization::default(), &data);
    let hits = s.search(&["col"], &data[2], &default_opts(3)).unwrap();
    assert_eq!(hits.len(), 3);
    // Exact rerank means the self-query's nearest hit is the doc itself.
    assert_eq!(hits[0].id, "d2");
    // Scores are the exact f32 cosine (rerank), not the quantized walk score.
    assert!(hits[0].score > 0.99, "self-match score {}", hits[0].score);
}

/// Build an ANN store and a matching exact (brute-force) store over the same
/// vectors, tagging every `stride`-th doc `kind=rare` (the rest `kind=common`).
fn kinded_stores(
    dim: usize,
    cfg: AnnConfig,
    vectors: &[Vec<f32>],
    stride: usize,
) -> (Store, Store) {
    let recs: Vec<Record> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut attrs = BTreeMap::new();
            let kind = if i % stride == 0 { "rare" } else { "common" };
            attrs.insert("kind".to_string(), Value::Str(kind.to_string()));
            rec_with(&format!("d{i}"), v.clone(), attrs)
        })
        .collect();
    let mut ann = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .auto_compact(None)
            .ann(Some(cfg)),
    )
    .unwrap();
    ann.upsert("col", &recs).unwrap();
    let mut exact = Store::in_memory(dim).unwrap();
    exact.upsert("col", &recs).unwrap();
    (ann, exact)
}

fn rare_filter() -> Filter {
    Filter(vec![Predicate::Eq(
        "kind".to_string(),
        Value::Str("rare".to_string()),
    )])
}

#[test]
#[cfg_attr(miri, ignore)] // N=400 HNSW build is slow under Miri; logic is plain code.
fn ann_selective_filter_keeps_exact_recall() {
    // Only every 40th doc is `rare` (~2.5% selectivity, far below 1/overscan = 25%).
    // The post-filter walk would starve here; the exact-prefilter fallback must
    // instead return *exactly* what brute force over the rare set returns.
    let (dim, k) = (16, 5);
    let data = random_unit_vectors(400, dim, 11);
    let queries = random_unit_vectors(20, dim, 12);
    for cfg in [AnnConfig::hnsw(), AnnConfig::ivf().n_probe(8)] {
        let (ann, exact) = kinded_stores(dim, cfg, &data, 40);
        for q in &queries {
            let opts = SearchOpts {
                top_k: k,
                filter: rare_filter(),
                ..Default::default()
            };
            let got = ann.search(&["col"], q, &opts).unwrap();
            let want = exact.search(&["col"], q, &opts).unwrap();
            // Exact prefilter ⇒ identical ids *and* scores, not just high recall.
            let got_ids: Vec<&str> = got.iter().map(|h| h.id.as_str()).collect();
            let want_ids: Vec<&str> = want.iter().map(|h| h.id.as_str()).collect();
            assert_eq!(
                got_ids, want_ids,
                "{cfg:?}: selective-filter ranking diverged"
            );
            for (g, w) in got.iter().zip(&want) {
                assert!((g.score - w.score).abs() < 1e-6);
            }
            // Every result is genuinely `rare` (the filter is honoured).
            assert!(got.iter().all(|h| {
                let idx: usize = h.id.trim_start_matches('d').parse().unwrap();
                idx.is_multiple_of(40)
            }));
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // N=400 HNSW build is slow under Miri.
fn ann_selective_scope_keeps_exact_recall() {
    // A tiny collection inside a much larger store: the whole-index walk surfaces
    // mostly out-of-scope candidates, starving the post-filter. The exact prefilter
    // (scope alone narrows the population) must match brute force over `small`.
    let (dim, k) = (16, 5);
    let big = random_unit_vectors(400, dim, 21);
    let small = random_unit_vectors(8, dim, 22);
    let queries = random_unit_vectors(20, dim, 23);

    let build = |cfg: Option<AnnConfig>| {
        let mut c = Config::new("/dev/null/in-memory", dim).auto_compact(None);
        if let Some(a) = cfg {
            c = c.ann(Some(a));
        }
        let mut s = Store::in_memory_cfg(c).unwrap();
        let big_recs: Vec<Record> = big
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("b{i}"), v.clone()))
            .collect();
        let small_recs: Vec<Record> = small
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("s{i}"), v.clone()))
            .collect();
        s.upsert("big", &big_recs).unwrap();
        s.upsert("small", &small_recs).unwrap();
        s
    };

    let ann = build(Some(AnnConfig::hnsw()));
    let exact = build(None);
    for q in &queries {
        let got = ann.search(&["small"], q, &default_opts(k)).unwrap();
        let want = exact.search(&["small"], q, &default_opts(k)).unwrap();
        let got_ids: Vec<&str> = got.iter().map(|h| h.id.as_str()).collect();
        let want_ids: Vec<&str> = want.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(got_ids, want_ids, "selective-scope ranking diverged");
        assert!(got.iter().all(|h| h.id.starts_with('s')));
    }
}

#[test]
fn ann_selective_filter_respects_min_score() {
    // The exact-prefilter path must still honour `min_score` (it runs the real f32
    // scorer, so the floor applies exactly as on the brute-force path). Tiny build
    // so this stays Miri-clean.
    let dim = 8;
    let data = random_unit_vectors(16, dim, 31);
    let (ann, _exact) = kinded_stores(dim, AnnConfig::hnsw(), &data, 4);
    let opts = SearchOpts {
        top_k: 10,
        filter: rare_filter(),
        min_score: Some(0.99), // essentially only a near-identical vector clears this
        ..Default::default()
    };
    let hits = ann.search(&["col"], &data[0], &opts).unwrap();
    assert!(hits.iter().all(|h| h.score >= 0.99));
    // d0 is `rare` (index 0) and identical to the query → it must be present.
    assert_eq!(hits[0].id, "d0");
}

// ── Optional vectors: text-only documents ──────────────────────────────────

/// A text-only record (no embedding) — coexists with vector docs in a collection.
fn text_rec(id: &str, attrs: BTreeMap<String, Value>) -> Record {
    Record::text_only(id, attrs)
}

fn attrs_one(key: &str, val: &str) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert(key.to_string(), Value::Str(val.to_string()));
    m
}

#[test]
fn text_only_upsert_adds_no_row() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert("col", &[text_rec("t1", attrs_one("kind", "note"))])
        .unwrap();
    // No vector ⇒ no data row, no vector_bytes, but it is a live doc.
    let fp = store.footprint();
    assert_eq!(fp.rows, 0);
    assert_eq!(fp.vector_bytes, 0);
    assert_eq!(fp.doc_count, 1);
    // get_all returns it with vector None.
    let recs = store.get_all("col");
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].id, "t1");
    assert_eq!(recs[0].vector, None);
}

#[test]
fn vector_search_excludes_text_only_docs() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert(
            "col",
            &[
                rec("v1", vec![1.0, 0.0, 0.0]),
                text_rec("t1", attrs_one("kind", "note")),
            ],
        )
        .unwrap();
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(10))
        .unwrap();
    // Only the vector doc is ranked; the text-only doc never appears.
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "v1");
}

#[test]
fn list_includes_text_only_docs() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert(
            "col",
            &[
                rec("v1", vec![1.0, 0.0, 0.0]),
                text_rec("t1", attrs_one("kind", "note")),
            ],
        )
        .unwrap();
    let hits = store
        .list(
            &["col"],
            &ListOpts {
                limit: 10,
                ..Default::default()
            },
        )
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["v1", "t1"],
        "rowed doc first, then text-only by id"
    );
}

#[test]
fn doc_can_switch_between_vector_and_text_only() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert("col", &[rec("d", vec![1.0, 0.0, 0.0])])
        .unwrap();
    assert_eq!(store.footprint().rows, 1);
    // Re-upsert the same id as text-only: the old row becomes dead.
    store
        .upsert("col", &[text_rec("d", attrs_one("kind", "note"))])
        .unwrap();
    assert_eq!(store.footprint().doc_count, 1);
    assert_eq!(store.footprint().dead_rows, 1);
    // It no longer appears in vector search.
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(10))
        .unwrap();
    assert!(hits.is_empty());
    // Re-upsert with a vector again: searchable once more.
    store
        .upsert("col", &[rec("d", vec![0.0, 1.0, 0.0])])
        .unwrap();
    let hits = store
        .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(10))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "d");
}

#[test]
fn delete_text_only_doc_leaves_no_dead_row() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert("col", &[text_rec("t1", attrs_one("kind", "note"))])
        .unwrap();
    assert_eq!(store.delete("col", &["t1"]).unwrap(), 1);
    assert_eq!(store.footprint().dead_rows, 0);
    assert_eq!(store.footprint().doc_count, 0);
}

#[test]
#[cfg_attr(miri, ignore)]
fn text_only_docs_survive_reopen_and_compact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store
            .upsert(
                "col",
                &[
                    rec("v1", vec![3.0, 4.0]),
                    text_rec("t1", attrs_one("kind", "note")),
                    text_rec("t2", attrs_one("kind", "memo")),
                ],
            )
            .unwrap();
        store.compact().unwrap();
    }
    // Reopen: the UpsertText log records must replay back into live docs.
    let store = Store::open(Config::new(&path, 2)).unwrap();
    assert_eq!(store.footprint().doc_count, 3);
    assert_eq!(store.footprint().rows, 1, "only the vector doc has a row");
    let all = store.get_all("col");
    let mut text_only: Vec<&str> = all
        .iter()
        .filter(|r| r.vector.is_none())
        .map(|r| r.id.as_str())
        .collect();
    text_only.sort();
    assert_eq!(text_only, vec!["t1", "t2"]);
}

// ── Per-segment IVF indexing (SPEC §14.3) ───────────────────────────────────

/// An in-memory store that seals every `seal` rows and IVF-indexes any sealed segment
/// with `≥ index_min` rows — the per-segment "exact tail / indexed cold" split.
fn segmented_store(dim: usize, seal: u64, index_min: u64) -> Store {
    Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .distance(Distance::Cosine)
            .open_mode(OpenMode::ReadWrite)
            .auto_compact(None)
            .segment_max_rows(Some(seal))
            .segment_index_min_rows(Some(index_min)),
    )
    .unwrap()
}

#[test]
fn segmented_indexes_build_as_segments_seal() {
    // Seal at 4 rows; index any sealed segment with >= 2 rows. Insert in batches so the
    // seal fires at the start of a later upsert.
    let mut s = segmented_store(2, 4, 2);
    s.create_collection("col").unwrap();
    // 8 rows in batches of 2 → after 4 rows, the next upsert seals the first segment;
    // after 8, the next seals the second.
    for i in 0..5 {
        s.upsert(
            "col",
            &[
                rec(&format!("a{i}"), vec![(i as f32).cos(), (i as f32).sin()]),
                rec(
                    &format!("b{i}"),
                    vec![(-(i as f32)).sin(), (i as f32).cos()],
                ),
            ],
        )
        .unwrap();
    }
    // seg_indexes is aligned with the segment set; the last (active) slot is never indexed.
    assert_eq!(s.seg_indexes.len(), s.data.segment_count());
    assert!(
        s.seg_indexes.last().unwrap().is_none(),
        "the active segment must stay exhaustive (never indexed)"
    );
    assert!(
        s.seg_indexes.iter().filter(|x| x.is_some()).count() >= 1,
        "at least one sealed segment should be IVF-indexed"
    );
}

#[test]
fn segmented_off_by_default_is_exact() {
    // segment_max_rows set (a multi-segment store) but no index threshold → every segment
    // is brute-forced, so results are byte-for-byte the exact single-segment store.
    let dim = 8;
    let data = random_unit_vectors(40, dim, 11);
    let exact = exact_store(dim, &data);

    let mut seg = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .auto_compact(None)
            .segment_max_rows(Some(8)), // seals, but segment_index_min_rows is None
    )
    .unwrap();
    let recs: Vec<Record> = data
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
        .collect();
    seg.upsert("col", &recs).unwrap();
    assert!(
        seg.seg_indexes.iter().all(Option::is_none),
        "no segment may be indexed when segment_index_min_rows is unset"
    );

    let q = random_unit_vectors(1, dim, 99).pop().unwrap();
    let got: Vec<String> = seg
        .search(&["col"], &q, &default_opts(10))
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    let truth: Vec<String> = exact
        .search(&["col"], &q, &default_opts(10))
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(got, truth, "unindexed multi-segment store must be exact");
}

#[test]
fn segmented_small_segments_are_fully_probed_exact() {
    // A small sealed segment has ~sqrt(rows) lists and n_probe (8) >= that, so it is fully
    // probed — every row scored — and recall is exact. Query each stored vector and expect
    // itself as the top hit, whether it lives in a sealed (indexed) segment or the tail.
    let mut s = segmented_store(2, 4, 2);
    s.create_collection("col").unwrap();
    let vectors: Vec<Vec<f32>> = (0..12)
        .map(|i| {
            let t = i as f32 / 12.0 * std::f32::consts::TAU;
            vec![t.cos(), t.sin()]
        })
        .collect();
    for (i, v) in vectors.iter().enumerate() {
        s.upsert("col", &[rec(&format!("d{i}"), v.clone())])
            .unwrap();
    }
    // Force a final seal so the last full segment is indexed too.
    s.flush().unwrap();
    for (i, v) in vectors.iter().enumerate() {
        let hits = s.search(&["col"], v, &default_opts(1)).unwrap();
        assert_eq!(hits[0].id, format!("d{i}"), "row {i} should be its own NN");
    }
}

#[test]
fn segmented_skips_deleted_rows_in_cold_segments() {
    let mut s = segmented_store(2, 4, 2);
    s.create_collection("col").unwrap();
    let vectors: Vec<Vec<f32>> = (0..12)
        .map(|i| {
            let t = i as f32 / 12.0 * std::f32::consts::TAU;
            vec![t.cos(), t.sin()]
        })
        .collect();
    for (i, v) in vectors.iter().enumerate() {
        s.upsert("col", &[rec(&format!("d{i}"), v.clone())])
            .unwrap();
    }
    s.flush().unwrap(); // seal the tail so its rows are indexed
    // Delete a doc that lives in a sealed (indexed) segment, then query its exact vector.
    s.delete("col", &["d1"]).unwrap();
    let hits = s.search(&["col"], &vectors[1], &default_opts(3)).unwrap();
    assert!(
        hits.iter().all(|h| h.id != "d1"),
        "a deleted row in a cold segment must not surface: {hits:?}"
    );
}

#[test]
fn segmented_compact_collapses_to_exact() {
    let mut s = segmented_store(2, 4, 2);
    s.create_collection("col").unwrap();
    let vectors: Vec<Vec<f32>> = (0..12)
        .map(|i| {
            let t = i as f32 / 12.0 * std::f32::consts::TAU;
            vec![t.cos(), t.sin()]
        })
        .collect();
    for (i, v) in vectors.iter().enumerate() {
        s.upsert("col", &[rec(&format!("d{i}"), v.clone())])
            .unwrap();
    }
    s.flush().unwrap();
    assert!(
        s.data.segment_count() > 1,
        "store should have sealed segments"
    );
    s.compact().unwrap();
    // Compaction collapses every segment into one fresh active segment → fully exact.
    assert_eq!(s.data.segment_count(), 1);
    assert!(s.seg_indexes.iter().all(Option::is_none));
    for (i, v) in vectors.iter().enumerate() {
        let hits = s.search(&["col"], v, &default_opts(1)).unwrap();
        assert_eq!(hits[0].id, format!("d{i}"));
    }
}

// Ignored under Miri: a few thousand rows make recall meaningful but are far too slow at
// Miri's ~100×. The dispatch/merge logic is exercised by the small tests above.
#[cfg_attr(miri, ignore)]
#[test]
fn segmented_recall_matches_exact() {
    let (n, dim, k) = (2000usize, 32, 10);
    let data = random_unit_vectors(n, dim, 3);
    let queries = random_unit_vectors(40, dim, 4);
    let truth = exact_store(dim, &data);

    // Seal every 256 rows; index any sealed segment (>= 64 rows). Insert in 256-row
    // batches so the store fans out into several indexed cold segments plus a tail.
    let mut seg = segmented_store(dim, 256, 64);
    let recs: Vec<Record> = data
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
        .collect();
    for batch in recs.chunks(256) {
        seg.upsert("col", batch).unwrap();
    }
    assert!(
        seg.seg_indexes.iter().filter(|x| x.is_some()).count() >= 2,
        "expected several indexed cold segments"
    );

    let recall = mean_recall(&seg, &truth, &queries, k);
    assert!(
        recall >= 0.80,
        "per-segment IVF recall@{k} = {recall:.3}, expected >= 0.80"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn segmented_survives_reopen() {
    // The per-segment indexes are rebuilt on open from the (immutable) segments; a
    // reopened segmented store must answer queries the same as before.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let vectors: Vec<Vec<f32>> = (0..12)
        .map(|i| {
            let t = i as f32 / 12.0 * std::f32::consts::TAU;
            vec![t.cos(), t.sin()]
        })
        .collect();
    {
        let mut s = Store::open(
            Config::new(&path, 2)
                .auto_compact(None)
                .segment_max_rows(Some(4))
                .segment_index_min_rows(Some(2)),
        )
        .unwrap();
        for (i, v) in vectors.iter().enumerate() {
            s.upsert("col", &[rec(&format!("d{i}"), v.clone())])
                .unwrap();
        }
        s.flush().unwrap();
    }
    let s = Store::open(
        Config::new(&path, 2)
            .auto_compact(None)
            .segment_max_rows(Some(4))
            .segment_index_min_rows(Some(2)),
    )
    .unwrap();
    assert!(
        s.seg_indexes.iter().filter(|x| x.is_some()).count() >= 1,
        "cold-segment indexes should be rebuilt on open"
    );
    for (i, v) in vectors.iter().enumerate() {
        let hits = s.search(&["col"], v, &default_opts(1)).unwrap();
        assert_eq!(hits[0].id, format!("d{i}"), "row {i} after reopen");
    }
}

// ── Memory-mapped immutable segments (SPEC §9 / §14.6 phase 3) ──────────────

/// Write a sealed, multi-segment store to disk at `path` and return the vectors written.
/// Sealing past `seal` rows produces immutable segments that a later `mmap` open can map.
fn write_sealed_store(path: &std::path::Path, n: usize, dim: usize, seal: u64) -> Vec<Vec<f32>> {
    let data = random_unit_vectors(n, dim, 7);
    let mut s = Store::open(
        Config::new(path, dim)
            .auto_compact(None)
            .segment_max_rows(Some(seal)),
    )
    .unwrap();
    let recs: Vec<Record> = data
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
        .collect();
    for batch in recs.chunks(64) {
        s.upsert("col", batch).unwrap();
    }
    s.flush().unwrap();
    data
}

#[cfg_attr(miri, ignore)]
#[test]
fn mmap_search_matches_ram_load() {
    // The whole contract: a memory-mapped open answers byte-for-byte identically to the
    // RAM-loaded open of the same on-disk store — same ids AND same scores, every query.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let (n, dim, k) = (500usize, 16, 10);
    write_sealed_store(&path, n, dim, 64);
    let queries = random_unit_vectors(20, dim, 8);

    // Collect the RAM-load results first, then drop that handle (releasing the writer lock)
    // before opening the mmap handle over the same directory.
    let ram_results: Vec<Vec<(String, f32)>> = {
        let ram = Store::open(Config::new(&path, dim).auto_compact(None).mmap(false)).unwrap();
        queries
            .iter()
            .map(|q| {
                ram.search(&["col"], q, &default_opts(k))
                    .unwrap()
                    .into_iter()
                    .map(|h| (h.id, h.score))
                    .collect()
            })
            .collect()
    };

    let mapped = Store::open(Config::new(&path, dim).auto_compact(None).mmap(true)).unwrap();
    for (q, expected) in queries.iter().zip(&ram_results) {
        let got: Vec<(String, f32)> = mapped
            .search(&["col"], q, &default_opts(k))
            .unwrap()
            .into_iter()
            .map(|h| (h.id, h.score))
            .collect();
        assert_eq!(
            &got, expected,
            "mmap search must match the RAM-loaded search exactly"
        );
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn mmap_excludes_deleted_rows_in_mapped_segments() {
    // Deleting a row that lives in a sealed (mapped) segment removes it from results: the
    // live index is independent of the data backing, so a tombstone hides a mapped row.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let dim = 4;
    let data = write_sealed_store(&path, 100, dim, 32);

    let mut s = Store::open(Config::new(&path, dim).auto_compact(None).mmap(true)).unwrap();
    s.delete("col", &["d10"]).unwrap(); // d10 lives in the first sealed (mapped) segment
    let hits = s.search(&["col"], &data[10], &default_opts(5)).unwrap();
    assert!(
        hits.iter().all(|h| h.id != "d10"),
        "a deleted row in a mapped segment must not appear"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn mmap_compact_collapses_mapped_base() {
    // Compaction must work even when the base `data` segment is memory-mapped: it reopens
    // the base writable, atomically rewrites it, and collapses to a single exact segment.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let dim = 8;
    let data = write_sealed_store(&path, 200, dim, 64); // base `data` + several sealed segments

    let mut s = Store::open(Config::new(&path, dim).auto_compact(None).mmap(true)).unwrap();
    for i in 0..50 {
        s.delete("col", &[format!("d{i}").as_str()]).unwrap();
    }
    s.compact().unwrap();
    assert_eq!(
        s.data.segment_count(),
        1,
        "compaction collapses every segment into one"
    );
    // A surviving doc is still its own nearest neighbour after the collapse.
    let hits = s.search(&["col"], &data[120], &default_opts(1)).unwrap();
    assert_eq!(hits[0].id, "d120");
}

#[cfg_attr(miri, ignore)]
#[test]
fn mmap_single_segment_store_stays_ram() {
    // A store with no sealing is one segment — the active one, which is never mapped. Opening
    // it with mmap on is harmless: it loads to RAM and behaves exactly as without the flag.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let dim = 4;
    let data = random_unit_vectors(20, dim, 5);
    {
        let mut s = Store::open(Config::new(&path, dim)).unwrap();
        let recs: Vec<Record> = data
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
            .collect();
        s.upsert("col", &recs).unwrap();
        s.flush().unwrap();
    }
    let s = Store::open(Config::new(&path, dim).mmap(true)).unwrap();
    let hits = s.search(&["col"], &data[3], &default_opts(1)).unwrap();
    assert_eq!(hits[0].id, "d3");
}

#[cfg_attr(miri, ignore)]
#[test]
fn mmap_with_per_segment_index_keeps_recall() {
    // mmap composes with the Phase-2 per-segment IVF: cold segments are both mapped AND
    // indexed, and search over them still tracks exact recall.
    let (n, dim, k) = (2000usize, 32, 10);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let queries = random_unit_vectors(40, dim, 4);

    let data = {
        let all = random_unit_vectors(n, dim, 3);
        let mut s = Store::open(
            Config::new(&path, dim)
                .auto_compact(None)
                .segment_max_rows(Some(256))
                .segment_index_min_rows(Some(64)),
        )
        .unwrap();
        let recs: Vec<Record> = all
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
            .collect();
        for batch in recs.chunks(256) {
            s.upsert("col", batch).unwrap();
        }
        s.flush().unwrap();
        all
    };
    let truth = exact_store(dim, &data);

    let mapped = Store::open(
        Config::new(&path, dim)
            .auto_compact(None)
            .segment_max_rows(Some(256))
            .segment_index_min_rows(Some(64))
            .mmap(true),
    )
    .unwrap();
    assert!(
        mapped.seg_indexes.iter().filter(|x| x.is_some()).count() >= 2,
        "cold segments should be indexed over the mapped data"
    );
    let recall = mean_recall(&mapped, &truth, &queries, k);
    assert!(
        recall >= 0.80,
        "mmap + per-segment IVF recall@{k} = {recall:.3}, expected >= 0.80"
    );
}

// ── Per-segment IVF sidecars (nidus-143) ────────────────────────────────────

/// A file-backed segmented store at `path`: seals every 8 rows, indexes any sealed
/// segment with >= 4 rows. Mirrors `segmented_store` but on a real backend, so the
/// `<segment>.ivf` sidecars actually exist.
fn segmented_on_disk(path: &std::path::Path, dim: usize) -> Store {
    Store::open(
        Config::new(path, dim)
            .distance(Distance::Cosine)
            .auto_compact(None)
            .segment_max_rows(Some(8))
            .segment_index_min_rows(Some(4)),
    )
    .unwrap()
}

/// Fill `s` with `n` deterministic unit vectors, in batches small enough to seal repeatedly.
fn fill_segmented(s: &mut Store, dim: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
    let data = random_unit_vectors(n, dim, seed);
    let recs: Vec<Record> = data
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
        .collect();
    for batch in recs.chunks(4) {
        s.upsert("col", batch).unwrap();
    }
    s.flush().unwrap();
    data
}

#[test]
#[cfg_attr(miri, ignore)] // fsync: a real file-backed store
fn segment_ivf_sidecars_are_written_and_reloaded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let dim = 8;
    let (data, sidecars) = {
        let mut s = segmented_on_disk(&path, dim);
        let data = fill_segmented(&mut s, dim, 40, 7);
        let indexed = s.seg_indexes.iter().filter(|x| x.is_some()).count();
        assert!(indexed >= 2, "expected several indexed cold segments");
        s.persist_index().unwrap();
        let names = s.data.segment_names();
        let sidecars: Vec<std::path::PathBuf> = names
            .iter()
            .zip(&s.seg_indexes)
            .filter(|(_, ix)| ix.is_some())
            .map(|(n, _)| path.join(format!("{n}.ivf")))
            .collect();
        assert_eq!(sidecars.len(), indexed);
        for f in &sidecars {
            assert!(f.exists(), "sidecar {f:?} written");
        }
        (data, sidecars)
    };

    let q = data[3].clone();
    let ids = |s: &Store| -> Vec<String> {
        s.search(&["col"], &q, &default_opts(10))
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect()
    };

    // Reopen: every cold segment's index comes off its sidecar, so none is dirty.
    let from_cache = {
        let reopened = segmented_on_disk(&path, dim);
        assert_eq!(
            reopened.seg_indexes.iter().filter(|x| x.is_some()).count(),
            sidecars.len()
        );
        assert!(
            reopened.seg_index_dirty.iter().all(|d| !d),
            "an adopted sidecar must not be marked dirty (it would be rewritten every persist)"
        );
        ids(&reopened)
    };

    // Results match the rebuilt-from-scratch store: adopting the cache changes nothing.
    for f in &sidecars {
        std::fs::remove_file(f).unwrap();
    }
    let rebuilt = segmented_on_disk(&path, dim);
    assert!(
        rebuilt.seg_index_dirty.iter().any(|&d| d),
        "with no sidecar every index is freshly built, hence dirty"
    );
    assert_eq!(from_cache, ids(&rebuilt));
}

#[test]
#[cfg_attr(miri, ignore)] // fsync: a real file-backed store
fn a_planted_sidecar_is_actually_adopted() {
    // Proves adoption, not "the store opened": an adopted *empty* index offers no candidates
    // while still excluding its rows from the exhaustive tail, so that segment's docs vanish
    // — an observable a rebuild cannot produce.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let dim = 8;
    let mut s = segmented_on_disk(&path, dim);
    let data = fill_segmented(&mut s, dim, 40, 11);
    let ranges = s.data.segment_ranges();
    let names = s.data.segment_names();
    let victim = s
        .seg_indexes
        .iter()
        .position(Option::is_some)
        .expect("a cold segment is indexed");
    let (base, rows) = ranges[victim];
    let p = s.persistence.clone().unwrap();
    let slot = crate::ann::SegmentSlot {
        name: &names[victim],
        base,
        rows,
    };
    crate::ann::save_segment_index(
        p.as_ref(),
        slot,
        dim,
        Distance::Cosine,
        &AnnConfig::ivf(),
        &IvfIndex::new(AnnConfig::ivf(), dim, Distance::Cosine), // never built: no lists
    )
    .unwrap();
    drop(s);

    // Every doc in the victim segment's row range must now be unreachable.
    let reopened = segmented_on_disk(&path, dim);
    let all: Vec<String> = reopened
        .search(&["col"], &data[0], &default_opts(40))
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(
        all.len() as u64,
        40 - rows,
        "the planted empty index must swallow exactly its own segment's rows"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // fsync: a real file-backed store
fn compaction_drops_stale_segment_sidecars() {
    // Regression (nidus-143): `rewrite` replaces the base's bytes in place at base 0, so a
    // surviving `data.ivf` key-matches a later seal at the same row count. The plant makes that
    // adoption observable — a real stale index holds the same rows and only degrades recall.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let dim = 8;
    let base_sidecar = path.join("data.ivf");

    let mut s = segmented_on_disk(&path, dim);
    let data = fill_segmented(&mut s, dim, 40, 13);
    let (base, rows) = s.data.segment_ranges()[0];
    assert_eq!(base, 0, "the base segment always starts at row 0");
    // Plant an empty-but-valid index over the base segment, keyed exactly as a later reseal
    // at the same row count will be.
    let p = s.persistence.clone().unwrap();
    let slot = crate::ann::SegmentSlot {
        name: "data",
        base,
        rows,
    };
    crate::ann::save_segment_index(
        p.as_ref(),
        slot,
        dim,
        Distance::Cosine,
        &AnnConfig::ivf(),
        &IvfIndex::new(AnnConfig::ivf(), dim, Distance::Cosine),
    )
    .unwrap();
    assert!(base_sidecar.exists());

    // Delete everything past the base segment so compaction collapses to one segment, then compact.
    let doomed: Vec<String> = (rows as usize..40).map(|i| format!("d{i}")).collect();
    let refs: Vec<&str> = doomed.iter().map(String::as_str).collect();
    s.delete("col", &refs).unwrap();
    s.compact().unwrap();
    assert!(
        !base_sidecar.exists(),
        "compaction must delete the base segment's stale IVF sidecar"
    );
    drop(s);

    // Re-fill so `data` seals again at the same `(base, rows)` the plant was keyed for.
    let mut s = segmented_on_disk(&path, dim);
    assert_eq!(
        s.data.segment_ranges()[0].1,
        rows,
        "reseals at the same size"
    );
    let refill: Vec<Record> = (0..32)
        .map(|i| rec(&format!("n{i}"), data[i % data.len()].clone()))
        .collect();
    for batch in refill.chunks(4) {
        s.upsert("col", batch).unwrap();
    }
    s.flush().unwrap();
    assert_eq!(s.data.segment_ranges()[0], (base, rows));
    drop(s);

    // A surviving plant would be adopted here and swallow the base segment's rows.
    let reopened = segmented_on_disk(&path, dim);
    let hits = reopened
        .search(&["col"], &data[0], &default_opts(64))
        .unwrap();
    assert_eq!(
        hits.len(),
        rows as usize + 32,
        "every live doc must still be reachable"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // 512-row k-means over a real file-backed store: too slow under Miri.
fn compaction_stale_sidecar_would_wreck_recall() {
    // The measured counterpart to `compaction_drops_stale_segment_sidecars`: a genuinely built
    // index at a scale where `n_probe` (8) covers only part of the ~23 lists, so adopting one
    // fitted to the pre-compaction vectors probes the wrong lists and recall is what pays.
    let (rows, dim, k) = (512usize, 32usize, 10usize);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let cfg = || {
        Config::new(&path, dim)
            .distance(Distance::Cosine)
            .auto_compact(None)
            .segment_max_rows(Some(rows as u64))
            .segment_index_min_rows(Some(256))
    };
    let fill = |s: &mut Store, vs: &[Vec<f32>]| {
        let recs: Vec<Record> = vs
            .iter()
            .enumerate()
            .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
            .collect();
        for batch in recs.chunks(128) {
            s.upsert("col", batch).unwrap();
        }
        s.flush().unwrap();
    };

    // One extra vector past `rows`: it lands in the fresh active segment and is what pushes
    // the base over the seal threshold at exactly `rows`.
    let old = random_unit_vectors(rows + 1, dim, 21);
    let new = random_unit_vectors(rows + 1, dim, 22);
    let queries = random_unit_vectors(30, dim, 23);

    // 1. Fill with the OLD vectors, seal + index the base, and write its sidecar.
    let mut s = Store::open(cfg()).unwrap();
    fill(&mut s, &old);
    assert_eq!(s.data.segment_ranges()[0], (0, rows as u64));
    s.persist_index().unwrap();
    assert!(path.join("data.ivf").exists());

    // 2. Empty the store and compact: `data` is rewritten in place, so its sidecar now
    //    describes vectors the store no longer holds.
    let ids: Vec<String> = (0..=rows).map(|i| format!("d{i}")).collect();
    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    s.delete("col", &refs).unwrap();
    s.compact().unwrap();

    // 3. Refill with the NEW vectors so the base reseals at exactly the `(base, rows)` the
    //    old sidecar was keyed under. No `persist_index` — nothing may rewrite it.
    fill(&mut s, &new);
    assert_eq!(s.data.segment_ranges()[0], (0, rows as u64));
    drop(s);

    // 4. Reopen: the only point where a surviving sidecar is adopted.
    let reopened = Store::open(cfg()).unwrap();
    let truth = exact_store(dim, &new);
    let recall = mean_recall(&reopened, &truth, &queries, k);

    // Calibrate against a store that reached the same contents without ever compacting, so the
    // bound tracks IVF's own recall at this probe ratio rather than a hardcoded number.
    let ref_dir = tempfile::tempdir().unwrap();
    let ref_path = ref_dir.path().join("store");
    let ref_cfg = || {
        Config::new(&ref_path, dim)
            .distance(Distance::Cosine)
            .auto_compact(None)
            .segment_max_rows(Some(rows as u64))
            .segment_index_min_rows(Some(256))
    };
    let mut r = Store::open(ref_cfg()).unwrap();
    fill(&mut r, &new);
    drop(r);
    let baseline = mean_recall(&Store::open(ref_cfg()).unwrap(), &truth, &queries, k);

    // Measured: 0.770 both sides with the fix, 0.337 through the compaction path without it.
    assert!(
        recall >= baseline - 0.02,
        "recall@{k} = {recall:.3} through the compaction path vs {baseline:.3} without \
         compacting — a stale sidecar was adopted over the rewritten vectors"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // fsync: a real file-backed store
fn a_corrupt_segment_sidecar_rebuilds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let dim = 8;
    let mut s = segmented_on_disk(&path, dim);
    let data = fill_segmented(&mut s, dim, 40, 17);
    s.persist_index().unwrap();
    let clean: Vec<String> = s
        .search(&["col"], &data[0], &default_opts(40))
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    drop(s);

    // Flip a payload byte in every sidecar: the CRC must reject them and open must rebuild.
    let mut corrupted = 0;
    for entry in std::fs::read_dir(&path).unwrap() {
        let f = entry.unwrap().path();
        if f.extension().and_then(|e| e.to_str()) != Some("ivf") {
            continue;
        }
        let mut bytes = std::fs::read(&f).unwrap();
        let last = bytes.len() - 5; // inside the payload, before the trailing crc32
        bytes[last] ^= 0xFF;
        std::fs::write(&f, &bytes).unwrap();
        corrupted += 1;
    }
    assert!(corrupted > 0, "there were sidecars to corrupt");

    let reopened = segmented_on_disk(&path, dim);
    assert!(
        reopened.seg_index_dirty.iter().any(|&d| d),
        "a rejected sidecar means the index was rebuilt, hence dirty"
    );
    let got: Vec<String> = reopened
        .search(&["col"], &data[0], &default_opts(40))
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(got, clean);
}

// ── Full-text search (BM25) ─────────────────────────────────────────────────

use crate::fts::{Analyzer, FtsField, Language};
use crate::model::FtsQuery;

fn doc(id: &str, body: &str) -> Record {
    let mut attrs = BTreeMap::new();
    attrs.insert("body".to_string(), Value::Str(body.to_string()));
    Record::text_only(id, attrs)
}

#[test]
fn text_search_ranks_and_stems() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    store
        .upsert(
            "docs",
            &[
                doc("a", "the cat sat on the mat"),
                doc("b", "cats are running and cats keep running"),
                doc("c", "a dog barked loudly"),
            ],
        )
        .unwrap();
    let hits = store
        .text_search(
            &["docs"],
            &FtsQuery::new("body", "running cats"),
            &default_opts(10),
        )
        .unwrap();
    // b mentions the query terms most; c matches nothing.
    assert_eq!(hits[0].id, "b");
    assert!(!hits.iter().any(|h| h.id == "c"));
}

#[test]
fn text_search_indexes_docs_upserted_before_schema() {
    // Declaring the schema after upserts must index the existing docs.
    let mut store = Store::in_memory(3).unwrap();
    store.upsert("docs", &[doc("a", "alpha beta")]).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    let hits = store
        .text_search(
            &["docs"],
            &FtsQuery::new("body", "alpha"),
            &default_opts(10),
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "a");
}

#[test]
fn text_search_respects_filter_and_delete() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    let mut a = doc("a", "shared term");
    a.attrs
        .insert("lang".to_string(), Value::Str("rust".to_string()));
    let mut b = doc("b", "shared term");
    b.attrs
        .insert("lang".to_string(), Value::Str("go".to_string()));
    store.upsert("docs", &[a, b]).unwrap();

    // Filter to lang=rust → only a.
    let opts = SearchOpts {
        top_k: 10,
        filter: Filter(vec![Predicate::Eq(
            "lang".to_string(),
            Value::Str("rust".to_string()),
        )]),
        ..Default::default()
    };
    let hits = store
        .text_search(&["docs"], &FtsQuery::new("body", "shared"), &opts)
        .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["a"]
    );

    // Delete a → no longer found.
    store.delete("docs", &["a"]).unwrap();
    let hits = store
        .text_search(
            &["docs"],
            &FtsQuery::new("body", "shared"),
            &default_opts(10),
        )
        .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["b"]
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn text_search_survives_reopen_and_compact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store
            .set_fts_schema("docs", &[FtsField::new("body")])
            .unwrap();
        store
            .upsert(
                "docs",
                &[
                    doc("a", "searching for needles"),
                    doc("b", "haystack of hay"),
                ],
            )
            .unwrap();
        store.delete("docs", &["b"]).unwrap();
        store.compact().unwrap();
    }
    let store = Store::open(Config::new(&path, 2)).unwrap();
    let hits = store
        .text_search(
            &["docs"],
            &FtsQuery::new("body", "needle"),
            &default_opts(10),
        )
        .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["a"]
    );
}

#[test]
fn hybrid_collection_text_and_vector_coexist() {
    // A collection can hold vector docs and full-text fields on the same records.
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    let mut r = Record::new("a", vec![1.0, 0.0, 0.0], BTreeMap::new());
    r.attrs.insert(
        "body".to_string(),
        Value::Str("vector and text together".to_string()),
    );
    store.upsert("docs", &[r]).unwrap();

    // Vector search finds it.
    let vhits = store
        .search(&["docs"], &[1.0, 0.0, 0.0], &default_opts(10))
        .unwrap();
    assert_eq!(vhits.len(), 1);
    // Text search finds it too.
    let thits = store
        .text_search(&["docs"], &FtsQuery::new("body", "text"), &default_opts(10))
        .unwrap();
    assert_eq!(thits.len(), 1);
    assert_eq!(thits[0].id, "a");
}

use crate::model::HybridOpts;

#[test]
fn hybrid_search_fuses_vector_and_text() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    // a: strong vector match, weak text. b: weak vector, strong text. c: text-only.
    let mut a = Record::new("a", vec![1.0, 0.0, 0.0], BTreeMap::new());
    a.attrs.insert(
        "body".to_string(),
        Value::Str("unrelated words".to_string()),
    );
    let mut b = Record::new("b", vec![0.0, 1.0, 0.0], BTreeMap::new());
    b.attrs.insert(
        "body".to_string(),
        Value::Str("quantum physics lecture".to_string()),
    );
    let c = doc("c", "quantum physics quantum physics");
    store.upsert("docs", &[a, b, c]).unwrap();

    let opts = HybridOpts {
        top_k: 10,
        ..Default::default()
    };
    let hits = store
        .hybrid_search(
            &["docs"],
            &[1.0, 0.0, 0.0],
            &FtsQuery::new("body", "quantum physics"),
            &opts,
        )
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    // All three surface: a via the vector leg, b and c via the text leg.
    assert!(ids.contains(&"a"));
    assert!(ids.contains(&"b"));
    assert!(
        ids.contains(&"c"),
        "text-only doc ranked by its BM25 leg alone"
    );
    // Fused scores are descending.
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score);
    }
}

#[test]
// Miri evaluates `ln` (in BM25's idf) non-deterministically by design, so the fused
// RRF scores vary by an ULP run-to-run under Miri — the very stability this asserts.
// The tie-break determinism it checks holds under real float semantics.
#[cfg_attr(miri, ignore)]
fn hybrid_search_is_deterministic() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    store
        .upsert(
            "docs",
            &[
                doc("x", "alpha beta"),
                doc("y", "alpha gamma"),
                doc("z", "beta gamma"),
            ],
        )
        .unwrap();
    let opts = HybridOpts::default();
    let q = FtsQuery::new("body", "alpha beta");
    let a = store
        .hybrid_search(&["docs"], &[0.0, 0.0, 0.0], &q, &opts)
        .unwrap();
    let b = store
        .hybrid_search(&["docs"], &[0.0, 0.0, 0.0], &q, &opts)
        .unwrap();
    let ids_a: Vec<&str> = a.iter().map(|h| h.id.as_str()).collect();
    let ids_b: Vec<&str> = b.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids_a, ids_b);
}

#[test]
#[cfg_attr(miri, ignore)]
fn fts_cache_persists_and_reloads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store
            .set_fts_schema("docs", &[FtsField::new("body")])
            .unwrap();
        store
            .upsert("docs", &[doc("a", "alpha beta"), doc("b", "beta gamma")])
            .unwrap();
        // Write the fts cache out of band.
        store.persist_index().unwrap();
        assert!(path.join("fts").exists(), "fts cache file written");
    }
    // Reopen: cache watermark == log offset → adopted, results intact.
    {
        let store = Store::open(Config::new(&path, 2)).unwrap();
        let hits = store
            .text_search(&["docs"], &FtsQuery::new("body", "beta"), &default_opts(10))
            .unwrap();
        let mut ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }
    // A write after the cache was persisted must still be reflected on the next open
    // (watermark mismatch → rebuild from the live docs, including the new doc).
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store.upsert("docs", &[doc("c", "gamma delta")]).unwrap();
        // (no persist_index here — the cache is now stale)
    }
    {
        let store = Store::open(Config::new(&path, 2)).unwrap();
        let hits = store
            .text_search(
                &["docs"],
                &FtsQuery::new("body", "delta"),
                &default_opts(10),
            )
            .unwrap();
        assert_eq!(
            hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
            vec!["c"]
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn text_only_churn_auto_compacts_fts_on_reopen() {
    // Text-only docs create no data rows, so churning them never raises `dead_rows`
    // and the dead-row auto-compact can't see it. The FTS tombstone-ratio trigger gives
    // these workloads automatic relief (nidus-b6i PR feedback).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store
            .set_fts_schema("docs", &[FtsField::new("body")])
            .unwrap();
        // Insert 4 ids, then overwrite each several times → many tombstones, zero rows.
        for round in 0..5 {
            for i in 0..4 {
                let body = format!("term{round} shared");
                store
                    .upsert("docs", &[doc(&format!("d{i}"), &body)])
                    .unwrap();
            }
        }
        assert_eq!(
            store.footprint().dead_rows,
            0,
            "no data rows for text-only docs"
        );
        assert!(
            store.fts.tombstone_ratio() > 0.5,
            "churn should accumulate FTS tombstones"
        );
        store.persist_index().unwrap();
    }
    // Reopen → the FTS tombstone ratio (> auto_compact 0.5) triggers a compaction that
    // rebuilds the index and drops the tombstones.
    let store = Store::open(Config::new(&path, 2)).unwrap();
    assert_eq!(
        store.fts.tombstone_ratio(),
        0.0,
        "reopen auto-compacted the FTS index"
    );
    let hits = store
        .text_search(
            &["docs"],
            &FtsQuery::new("body", "shared"),
            &default_opts(10),
        )
        .unwrap();
    assert_eq!(hits.len(), 4, "all four live docs still searchable");
}

#[test]
fn text_search_across_collections_analyzes_once() {
    // Multi-collection text search returns a correct merged ranking (and analyzes the
    // query once per language internally).
    let mut store = Store::in_memory(3).unwrap();
    for c in ["a", "b"] {
        store.set_fts_schema(c, &[FtsField::new("body")]).unwrap();
    }
    store.upsert("a", &[doc("a1", "running fast")]).unwrap();
    store.upsert("b", &[doc("b1", "runners run")]).unwrap();
    let hits = store
        .text_search(
            &["a", "b"],
            &FtsQuery::new("body", "run"),
            &default_opts(10),
        )
        .unwrap();
    let mut ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    ids.sort();
    assert_eq!(
        ids,
        vec!["a1", "b1"],
        "stemmed match across both collections"
    );
}

// ── suggest: ranked term completions (nidus-ux0) ─────────────────────────────────

#[test]
fn suggest_ranks_by_document_frequency_not_idf() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    store
        .upsert(
            "docs",
            &[
                doc("a", "nidus"),
                doc("b", "nidus"),
                doc("c", "nidus"),
                doc("d", "nidification"),
            ],
        )
        .unwrap();

    // The ticket's own criterion, asserted as words: typing "nid" offers "nidus" above
    // "nidification". Surface forms, not the stems "nidu"/"nidif" a dropdown cannot show.
    let got = store.suggest("docs", "body", "nid", 10);
    assert_eq!(
        got.suggestions
            .iter()
            .map(|s| (s.term.as_str(), s.df))
            .collect::<Vec<_>>(),
        vec![("nidus", 3), ("nidification", 1)],
        "{got:?}"
    );
    assert_eq!(got.matched, 2);

    // text_search's document ranking is unaffected: the rare term still lifts its doc.
    let hits = store
        .text_search(
            &["docs"],
            &FtsQuery::multi([FtsClause::new("body", "nid").prefix()]),
            &default_opts(10),
        )
        .unwrap();
    assert_eq!(
        hits[0].id, "d",
        "the rare term's sole doc still ranks first"
    );
}

/// nidus-dnm: the fragment is folded but never stemmed, so before a prefix clause expanded
/// over surface forms too, typing the whole word "running" matched no document at all.
#[test]
fn a_prefix_clause_matches_a_stem_shortened_word() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    store
        .upsert("docs", &[doc("a", "running late"), doc("b", "quiet")])
        .unwrap();

    for typed in ["runn", "runni", "running"] {
        let hits = store
            .text_search(
                &["docs"],
                &FtsQuery::multi([FtsClause::new("body", typed).prefix()]),
                &default_opts(10),
            )
            .unwrap();
        assert_eq!(
            hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
            vec!["a"],
            "prefix {typed:?} must match the document that spells it"
        );
    }
}

#[test]
fn suggest_folds_the_prefix_like_a_prefix_clause() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body").ascii_folding(true)])
        .unwrap();
    store
        .upsert("docs", &[doc("a", "nidus"), doc("b", "café")])
        .unwrap();

    let upper = store.suggest("docs", "body", "NID", 10);
    let lower = store.suggest("docs", "body", "nid", 10);
    assert_eq!(upper, lower);
    assert_eq!(upper.suggestions.len(), 1, "{upper:?}");

    let folded = store.suggest("docs", "body", "caf", 10);
    assert_eq!(folded.suggestions[0].term, "cafe");
}

#[test]
fn suggest_takes_only_the_trailing_token() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    store.upsert("docs", &[doc("a", "nidus store")]).unwrap();

    assert_eq!(
        store.suggest("docs", "body", "the nid", 10),
        store.suggest("docs", "body", "nid", 10)
    );
}

#[test]
fn suggest_is_empty_for_an_unindexed_field_and_an_unknown_collection() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    store.upsert("docs", &[doc("a", "nidus")]).unwrap();

    assert_eq!(
        store.suggest("docs", "title", "nid", 10),
        Suggestions::default()
    );
    assert_eq!(
        store.suggest("nope", "body", "nid", 10),
        Suggestions::default()
    );
}

#[test]
fn suggest_of_an_empty_prefix_is_empty() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    store.upsert("docs", &[doc("a", "nidus")]).unwrap();

    for prefix in ["", "  ", "!!"] {
        assert_eq!(
            store.suggest("docs", "body", prefix, 10),
            Suggestions::default(),
            "prefix {prefix:?} must yield no suggestions"
        );
    }
}

#[test]
fn suggest_limit_truncates_and_matched_still_reports_the_full_count() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    store
        .upsert(
            "docs",
            &[
                doc("a", "nid1"),
                doc("b", "nid2"),
                doc("c", "nid3"),
                doc("d", "nid4"),
                doc("e", "nid5"),
            ],
        )
        .unwrap();

    let got = store.suggest("docs", "body", "nid", 2);
    assert_eq!(got.suggestions.len(), 2);
    assert_eq!(got.matched, 5);
}

#[test]
fn suggest_limit_zero_is_empty() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    store.upsert("docs", &[doc("a", "nidus")]).unwrap();

    let got = store.suggest("docs", "body", "nid", 0);
    assert!(got.suggestions.is_empty());
    assert_eq!(got.matched, 1);
}

// ── FTS schema: per-field BM25 / analyzer configuration (nidus-m50.13) ───────────

#[test]
fn a_legacy_fts_schema_op_replays_with_the_default_params() {
    // A store written before the params were tunable holds `SetFtsSchema`, not
    // `SetFtsFields`. It must still open, scored exactly as it was.
    let ops = vec![
        Op::CreateCollection {
            collection: "docs".to_string(),
        },
        Op::SetFtsSchema {
            collection: "docs".to_string(),
            fields: vec![("body".to_string(), Language::English)],
        },
    ];
    let (collections, _dead, fts, _findex) = Store::replay_ops(ops, 0);
    assert!(collections.contains_key("docs"));
    let decl = fts.schema_for("docs").expect("legacy schema restored");
    assert_eq!(decl, &[FtsField::new("body")]);
    assert_eq!(decl[0].k1, 1.2);
    assert_eq!(decl[0].b, 0.75);
    assert_eq!(
        fts.field_analyzer("docs", "body"),
        Some(Analyzer::default())
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_old_format_log_still_opens_and_searches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store.create_collection("docs").unwrap();
        // Write the *legacy* op straight to the log, as a pre-m50.13 nidus would have.
        store
            .log
            .append(&Op::SetFtsSchema {
                collection: "docs".to_string(),
                fields: vec![("body".to_string(), Language::English)],
            })
            .unwrap();
        store.flush().unwrap();
        store
            .upsert(
                "docs",
                &[doc("a", "the cat sat"), doc("b", "cats and cats")],
            )
            .unwrap();
        store.flush().unwrap();
    }
    let store = Store::open(Config::new(&path, 2)).unwrap();
    let hits = store
        .text_search(&["docs"], &FtsQuery::new("body", "cat"), &default_opts(10))
        .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["b", "a"]
    );
    // The frozen default score (idf = ln(1.2), norm exactly 1 at k1 = 1.2 / b = 0.75),
    // so the legacy path is not silently re-tuned.
    assert!((hits[1].score - 0.182_321_6).abs() < 1e-5, "{hits:?}");
}

#[test]
fn per_field_params_change_the_ranking_they_are_declared_on() {
    // `b = 0` on `body` removes length normalization, which is enough to flip the
    // short-doc-wins ordering the default produces.
    let ranked = |field: FtsField| {
        let mut store = Store::in_memory(3).unwrap();
        store.set_fts_schema("docs", &[field]).unwrap();
        store
            .upsert(
                "docs",
                &[
                    doc("short", "needle"),
                    doc(
                        "long",
                        "needle needle plus assorted unrelated padding words here",
                    ),
                ],
            )
            .unwrap();
        let hits = store
            .text_search(
                &["docs"],
                &FtsQuery::new("body", "needle"),
                &default_opts(10),
            )
            .unwrap();
        hits[0].id.clone()
    };
    assert_eq!(ranked(FtsField::new("body")), "short");
    assert_eq!(ranked(FtsField::new("body").b(0.0)), "long");
}

#[test]
fn analyzer_options_apply_at_index_and_query_time() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body").ascii_folding(true)])
        .unwrap();
    store.upsert("docs", &[doc("a", "un café noir")]).unwrap();
    // Folding is symmetric: either spelling of the query finds the accented document.
    for q in ["cafe", "café"] {
        let hits = store
            .text_search(&["docs"], &FtsQuery::new("body", q), &default_opts(10))
            .unwrap();
        assert_eq!(hits.len(), 1, "query {q:?} must match the folded term");
    }
}

#[test]
fn set_fts_schema_rejects_params_bm25_cannot_use() {
    let mut store = Store::in_memory(3).unwrap();
    assert!(
        store
            .set_fts_schema("docs", &[FtsField::new("body").b(2.0)])
            .is_err()
    );
    assert!(
        store
            .set_fts_schema("docs", &[FtsField::new("body").k1(-1.0)])
            .is_err()
    );
    // The rejected schema was never applied, so nothing is indexed.
    assert!(store.fts.schema_for("docs").is_none());
}

#[test]
#[cfg_attr(miri, ignore)]
fn changing_a_bm25_param_rebuilds_instead_of_serving_the_stale_cache() {
    // The consequential case: the postings on disk are still valid, so only the cache
    // *key* stands between a reopen and results scored under the old k1/b.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let corpus = [
        doc("short", "needle"),
        doc(
            "long",
            "needle needle plus assorted unrelated padding words here",
        ),
    ];
    let key_default;
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store
            .set_fts_schema("docs", &[FtsField::new("body")])
            .unwrap();
        store.upsert("docs", &corpus).unwrap();
        store.persist_index().unwrap();
        key_default = store.fts.cache_key();
        assert!(path.join("fts").exists());
    }
    {
        // Redeclare with b = 0 and nothing else changed.
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store
            .set_fts_schema("docs", &[FtsField::new("body").b(0.0)])
            .unwrap();
        assert_ne!(store.fts.cache_key(), key_default, "the key must move");
        store.persist_index().unwrap();
    }
    {
        // Reopen: the cache on disk is keyed for b = 0, so the new scores are served.
        let store = Store::open(Config::new(&path, 2)).unwrap();
        let hits = store
            .text_search(
                &["docs"],
                &FtsQuery::new("body", "needle"),
                &default_opts(10),
            )
            .unwrap();
        assert_eq!(hits[0].id, "long", "reopened under the new b");
    }
    {
        // And back: reverting the schema must not adopt the b = 0 cache either.
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store
            .set_fts_schema("docs", &[FtsField::new("body")])
            .unwrap();
        assert_eq!(store.fts.cache_key(), key_default);
        let hits = store
            .text_search(
                &["docs"],
                &FtsQuery::new("body", "needle"),
                &default_opts(10),
            )
            .unwrap();
        assert_eq!(hits[0].id, "short", "back to the default ranking");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn compact_re_emits_the_full_field_configuration() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let field = FtsField::new("body").k1(1.7).b(0.3).ascii_folding(true);
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store
            .set_fts_schema("docs", std::slice::from_ref(&field))
            .unwrap();
        store.upsert("docs", &[doc("a", "un café noir")]).unwrap();
        store.compact().unwrap();
    }
    // The post-compact log must carry the params, not a default-shaped schema.
    let store = Store::open(Config::new(&path, 2)).unwrap();
    assert_eq!(store.fts.schema_for("docs"), Some(&[field][..]));
    let hits = store
        .text_search(&["docs"], &FtsQuery::new("body", "cafe"), &default_opts(10))
        .unwrap();
    assert_eq!(hits.len(), 1, "folding survived the compaction");
}

// ── Segments: seal / manifest / migration (SPEC §14, Phase 1) ────────────────────

/// Eight distinct 2-D vectors at increasing angles — deterministic cosine ranking, so a
/// multi-segment store and a single-segment one must agree exactly.
fn angled(n: usize) -> Vec<Record> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            rec(&format!("doc{i}"), vec![1.0 - t, t])
        })
        .collect()
}

#[cfg_attr(miri, ignore)]
#[test]
fn seal_on_threshold_creates_multiple_segments() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        // Seal every 3 active rows: 7 single upserts → segments [data(3), seg-1(3), seg-2(1)].
        let mut store = Store::open(Config::new(&path, 2).segment_max_rows(Some(3))).unwrap();
        for r in angled(7) {
            store.upsert("col", &[r]).unwrap();
        }
        store.flush().unwrap();
        assert_eq!(store.data.segment_count(), 3, "active sealed twice");
        assert_eq!(store.footprint().rows, 7);
    }
    // The sealed segment objects are physically present, named by the manifest.
    assert!(path.join("data").exists());
    assert!(path.join("seg-00000001").exists());
    assert!(path.join("seg-00000002").exists());
    assert!(path.join("manifest").exists());

    // Reopen: the manifest is read, every segment loaded into one global row space.
    let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
    assert_eq!(store.data.segment_count(), 3);
    assert_eq!(store.footprint().rows, 7);
}

#[cfg_attr(miri, ignore)]
#[test]
fn multi_segment_search_matches_single_segment() {
    let q = [0.6_f32, 0.4];
    let recs = angled(8);

    // Multi-segment store (seal every 2 rows).
    let multi_dir = tempfile::tempdir().unwrap();
    let multi_hits = {
        let mut store =
            Store::open(Config::new(multi_dir.path(), 2).segment_max_rows(Some(2))).unwrap();
        for r in &recs {
            store.upsert("col", std::slice::from_ref(r)).unwrap();
        }
        assert!(store.data.segment_count() > 1, "store should have sealed");
        store.search(&["col"], &q, &default_opts(8)).unwrap()
    };

    // Single-segment store over the identical data.
    let single_dir = tempfile::tempdir().unwrap();
    let single_hits = {
        let mut store = Store::open(Config::new(single_dir.path(), 2)).unwrap();
        store.upsert("col", &recs).unwrap();
        assert_eq!(store.data.segment_count(), 1);
        store.search(&["col"], &q, &default_opts(8)).unwrap()
    };

    let ids = |hits: &[crate::model::Hit]| hits.iter().map(|h| h.id.clone()).collect::<Vec<_>>();
    assert_eq!(
        ids(&multi_hits),
        ids(&single_hits),
        "ranking must match exactly"
    );
    for (m, s) in multi_hits.iter().zip(&single_hits) {
        assert!(
            (m.score - s.score).abs() < 1e-6,
            "scores must match: {m:?} vs {s:?}"
        );
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn compact_collapses_segments_and_reclaims_objects() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut store = Store::open(Config::new(&path, 2).segment_max_rows(Some(2))).unwrap();
    for r in angled(6) {
        store.upsert("col", &[r]).unwrap();
    }
    assert!(store.data.segment_count() > 1);
    assert!(path.join("seg-00000001").exists());

    store.compact().unwrap();
    assert_eq!(
        store.data.segment_count(),
        1,
        "compaction collapses to one segment"
    );
    // The previously-sealed segment objects are reclaimed (no longer named by the manifest).
    assert!(
        !path.join("seg-00000001").exists(),
        "orphaned segment object deleted"
    );
    assert_eq!(store.footprint().rows, 6);
    drop(store);

    let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
    assert_eq!(store.get_all("col").len(), 6);
}

#[cfg_attr(miri, ignore)]
#[test]
fn legacy_store_without_manifest_migrates_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store.upsert("col", &angled(3)).unwrap();
    }
    // Simulate a pre-manifest (legacy) store: delete the manifest, keep `data` + `log`.
    std::fs::remove_file(path.join("manifest")).unwrap();
    assert!(!path.join("manifest").exists());

    // Reopen ReadWrite: the `data` object becomes the implicit base segment and a fresh
    // manifest is written (transparent migration) — the data is intact.
    {
        let store = Store::open(Config::new(&path, 2)).unwrap();
        assert_eq!(store.get_all("col").len(), 3);
    }
    assert!(path.join("manifest").exists(), "migration wrote a manifest");
}

#[cfg_attr(miri, ignore)]
#[test]
fn readonly_open_without_manifest_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store.upsert("col", &angled(2)).unwrap();
    }
    std::fs::remove_file(path.join("manifest")).unwrap();

    // A read-only open reads through a synthesized in-RAM manifest but must not persist one.
    let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
    assert_eq!(store.get_all("col").len(), 2);
    assert!(
        !path.join("manifest").exists(),
        "read-only open must not write a manifest"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn orphan_segment_not_in_manifest_is_ignored() {
    // Models a crash *before* a seal's manifest swap: the fresh segment object exists but
    // the manifest does not yet name it. On reopen it must be invisible — the store reads
    // exactly the manifest's segment set, never a stray object (the §6.2 guarantee, §14).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 2).segment_max_rows(Some(3))).unwrap();
        for r in angled(4) {
            store.upsert("col", &[r]).unwrap();
        }
        store.flush().unwrap();
        assert_eq!(store.data.segment_count(), 2); // [data, seg-1]
    }
    // Drop a stray segment object the manifest does not reference (an interrupted seal).
    std::fs::write(path.join("seg-00000099"), b"garbage-not-a-segment").unwrap();

    let store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
    assert_eq!(store.data.segment_count(), 2, "stray object ignored");
    assert_eq!(
        store.get_all("col").len(),
        4,
        "data intact at pre-crash state"
    );
}

// ── Manifest-versioned reader refresh (SPEC §14.6 phase 4) ──────────────────

/// Open a lock-free `ReadOnly` reader over `path` — the search-only handle that tracks a
/// separate writer via [`Store::refresh`]. Auto-compaction is off (a reader never compacts).
fn open_reader(path: &std::path::Path, dim: usize) -> Store {
    Store::open(
        Config::new(path, dim)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None),
    )
    .unwrap()
}

/// The `(id, score)` ranking a search returns — the comparable shape for parity assertions.
fn ranking(store: &Store, query: &[f32], k: usize) -> Vec<(String, f32)> {
    store
        .search(&["col"], query, &default_opts(k))
        .unwrap()
        .into_iter()
        .map(|h| (h.id, h.score))
        .collect()
}

#[cfg_attr(miri, ignore)]
#[test]
fn refresh_adopts_a_writers_appends() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(Config::new(&path, 2).auto_compact(None)).unwrap();
    w.create_collection("col").unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();

    // The reader opens at a snapshot that holds only "a".
    let mut r = open_reader(&path, 2);
    assert_eq!(r.get_all("col").len(), 1);

    // The writer commits a second doc. The reader does not see it until it refreshes.
    w.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    let before = ranking(&r, &[0.0, 1.0], 5);
    assert_eq!(before.len(), 1, "still the open-time snapshot");
    assert_eq!(before[0].0, "a");

    assert!(r.refresh().unwrap(), "refresh adopts the new write");
    let after = ranking(&r, &[0.0, 1.0], 5);
    assert_eq!(after.len(), 2);
    assert_eq!(after[0].0, "b");
    assert!((after[0].1 - 1.0).abs() < 1e-5);

    // Idempotent: with nothing newer committed, a second refresh is a cheap no-op.
    assert!(!r.refresh().unwrap(), "already current");
}

#[cfg_attr(miri, ignore)]
#[test]
fn refresh_is_a_noop_when_nothing_changed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(Config::new(&path, 2).auto_compact(None)).unwrap();
    w.create_collection("col").unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();

    let mut r = open_reader(&path, 2);
    assert!(!r.refresh().unwrap(), "nothing committed since open");
    assert!(!r.refresh().unwrap(), "still current");
}

#[cfg_attr(miri, ignore)]
#[test]
fn refresh_adopts_deletes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(Config::new(&path, 2).auto_compact(None)).unwrap();
    w.create_collection("col").unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    w.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();

    let mut r = open_reader(&path, 2);
    assert_eq!(r.get_all("col").len(), 2);

    w.delete("col", &["a"]).unwrap();
    assert!(r.refresh().unwrap(), "the delete is adopted");
    let ids: Vec<String> = r.get_all("col").into_iter().map(|rec| rec.id).collect();
    assert_eq!(ids, vec!["b".to_string()]);
}

#[cfg_attr(miri, ignore)]
#[test]
fn refresh_adopts_a_seal_and_matches_a_fresh_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    // Seal every 3 active rows so the writer grows multiple segments.
    let mut w = Store::open(
        Config::new(&path, 2)
            .auto_compact(None)
            .segment_max_rows(Some(3)),
    )
    .unwrap();
    for record in angled(2) {
        w.upsert("col", &[record]).unwrap();
    }
    let mut r = open_reader(&path, 2);
    assert_eq!(
        r.data.segment_count(),
        1,
        "reader opened on a single segment"
    );

    // Drive the writer past two seal points.
    for record in angled(8).into_iter().skip(2) {
        w.upsert("col", &[record]).unwrap();
    }
    w.flush().unwrap();
    assert!(w.data.segment_count() >= 2, "writer sealed at least once");

    assert!(
        r.refresh().unwrap(),
        "the seal advances the manifest version"
    );
    assert_eq!(
        r.data.segment_count(),
        w.data.segment_count(),
        "reader adopts the new segment set"
    );
    assert_eq!(r.data.version(), w.data.version());

    // Search parity against a reader freshly opened over the same on-disk store.
    let fresh = open_reader(&path, 2);
    for q in random_unit_vectors(12, 2, 5) {
        assert_eq!(ranking(&r, &q, 6), ranking(&fresh, &q, 6), "query {q:?}");
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn refresh_adopts_a_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(Config::new(&path, 3).auto_compact(None)).unwrap();
    w.create_collection("col").unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])]).unwrap();
    w.upsert("col", &[rec("b", vec![0.0, 1.0, 0.0])]).unwrap();
    // Overwrite "a" → one dead row.
    w.upsert("col", &[rec("a", vec![0.0, 0.0, 1.0])]).unwrap();

    let mut r = open_reader(&path, 3);
    assert_eq!(r.dead_rows, 1, "reader sees the dead row pre-compaction");

    w.compact().unwrap();
    assert!(r.refresh().unwrap(), "the compaction is adopted");
    assert_eq!(r.dead_rows, 0, "compaction reclaimed the dead row");

    // The compacted store still answers correctly through the reader.
    let hits = r
        .search(&["col"], &[0.0, 0.0, 1.0], &default_opts(5))
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, "a");
    let fresh = open_reader(&path, 3);
    for q in random_unit_vectors(8, 3, 9) {
        assert_eq!(ranking(&r, &q, 5), ranking(&fresh, &q, 5));
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn refresh_is_a_noop_on_a_writer_and_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut w = Store::open(Config::new(&path, 2).auto_compact(None)).unwrap();
    w.create_collection("col").unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    assert!(
        !w.refresh().unwrap(),
        "a writer is already the source of truth"
    );

    let mut mem = Store::in_memory(2).unwrap();
    mem.create_collection("col").unwrap();
    assert!(!mem.refresh().unwrap(), "an in-memory store has no backend");
}

#[cfg_attr(miri, ignore)]
#[test]
fn refresh_keeps_recall_with_per_segment_indexing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let dim = 8;
    let data = random_unit_vectors(240, dim, 3);

    // Writer seals every 64 rows and IVF-indexes any sealed segment ≥ 64 rows (cold), while
    // the active tail stays exact (SPEC §14.3).
    let mut w = Store::open(
        Config::new(&path, dim)
            .auto_compact(None)
            .segment_max_rows(Some(64))
            .segment_index_min_rows(Some(64)),
    )
    .unwrap();
    for (i, v) in data.iter().enumerate().take(80) {
        w.upsert("col", &[rec(&format!("d{i}"), v.clone())])
            .unwrap();
    }

    let mut r = Store::open(
        Config::new(&path, dim)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .segment_max_rows(Some(64))
            .segment_index_min_rows(Some(64)),
    )
    .unwrap();

    // Writer commits the rest, sealing and indexing further cold segments.
    for (i, v) in data.iter().enumerate().skip(80) {
        w.upsert("col", &[rec(&format!("d{i}"), v.clone())])
            .unwrap();
    }
    w.flush().unwrap();
    assert!(r.refresh().unwrap());

    // The refreshed reader matches a fresh reader over the same store, indexes and all.
    let fresh = Store::open(
        Config::new(&path, dim)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .segment_max_rows(Some(64))
            .segment_index_min_rows(Some(64)),
    )
    .unwrap();
    for q in random_unit_vectors(15, dim, 11) {
        assert_eq!(ranking(&r, &q, 10), ranking(&fresh, &q, 10), "query {q:?}");
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn refresh_composes_with_a_memory_mapped_reader() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let dim = 16;
    let data = random_unit_vectors(300, dim, 4);

    let mut w = Store::open(
        Config::new(&path, dim)
            .auto_compact(None)
            .segment_max_rows(Some(64)),
    )
    .unwrap();
    for (i, v) in data.iter().enumerate().take(70) {
        w.upsert("col", &[rec(&format!("d{i}"), v.clone())])
            .unwrap();
    }
    w.flush().unwrap();

    // A memory-mapped reader: immutable segments are mapped from disk, the active stays RAM.
    let mut r = Store::open(
        Config::new(&path, dim)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .segment_max_rows(Some(64))
            .mmap(true),
    )
    .unwrap();

    for (i, v) in data.iter().enumerate().skip(70) {
        w.upsert("col", &[rec(&format!("d{i}"), v.clone())])
            .unwrap();
    }
    w.flush().unwrap();
    assert!(
        r.refresh().unwrap(),
        "the mapped reader adopts the new segments"
    );

    let fresh = Store::open(
        Config::new(&path, dim)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .segment_max_rows(Some(64))
            .mmap(true),
    )
    .unwrap();
    for q in random_unit_vectors(15, dim, 13) {
        assert_eq!(ranking(&r, &q, 10), ranking(&fresh, &q, 10), "query {q:?}");
    }
}

// ── Pinned point-in-time opens (nidus-bnf, SPEC §14.2 history) ──────────────

#[cfg_attr(miri, ignore)]
#[test]
fn pinned_open_survives_a_later_commit_but_not_its_delete() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(
        Config::new(&path, 2)
            .auto_compact(None)
            .segment_max_rows(Some(2))
            .history_versions(Some(10)),
    )
    .unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    w.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    // Every durable batch is a commit point once history is on, so this is a real pin.
    let v = w.data.version();

    w.upsert("col", &[rec("c", vec![1.0, 1.0])]).unwrap();
    w.delete("col", &["a"]).unwrap();
    w.flush().unwrap();

    let pinned = Store::open(
        Config::new(&path, 2)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .at_version(Some(v)),
    )
    .unwrap();
    let ids: std::collections::BTreeSet<String> =
        pinned.get_all("col").into_iter().map(|r| r.id).collect();
    // A row-count-only bound would still replay the later `Delete`, dropping "a" — the log
    // offset is what keeps it. A row-count-only bound would also let "c" leak in.
    assert_eq!(
        ids,
        ["a", "b"].into_iter().map(String::from).collect(),
        "sees a,b as of the pin; not c; still a despite the later delete"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn pinned_open_across_a_compaction_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(
        Config::new(&path, 2)
            .auto_compact(None)
            .segment_max_rows(Some(2))
            .history_versions(Some(10)),
    )
    .unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    w.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    w.upsert("col", &[rec("c", vec![1.0, 1.0])]).unwrap();
    let v = w.data.version();

    w.delete("col", &["a"]).unwrap();
    w.compact().unwrap();

    let err = Store::open(
        Config::new(&path, 2)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .at_version(Some(v)),
    )
    .map(|_| ())
    .unwrap_err()
    .to_string();
    let oldest = w.versions().unwrap().oldest_readable.unwrap();
    assert!(err.contains(&v.to_string()), "{err}");
    assert!(err.contains(&oldest.to_string()), "{err}");
}

/// The floor is the fence, not the delete. `compact` reclaims stale history entries
/// best-effort (it swallows delete failures), so a survivor must still be refused — it
/// describes segments the in-place base rewrite renumbered, and would serve wrong bytes.
#[cfg_attr(miri, ignore)] // fsyncs: Miri has no sync_all
#[test]
fn a_history_entry_that_outlives_a_compaction_is_still_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(
        Config::new(&path, 2)
            .auto_compact(None)
            .segment_max_rows(Some(2))
            .history_versions(Some(10)),
    )
    .unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    w.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    let v = w.data.version();

    let p = crate::open_persistence(path.to_str().unwrap()).unwrap();
    let stale = crate::manifest::history::load_entry(p.as_ref(), v)
        .unwrap()
        .expect("the pinned version was recorded");

    w.delete("col", &["a"]).unwrap();
    w.compact().unwrap();
    // Put the entry back: exactly the state a failed delete leaves behind.
    crate::manifest::history::store_entry(p.as_ref(), &stale).unwrap();

    let err = Store::open(
        Config::new(&path, 2)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .at_version(Some(v)),
    )
    .map(|_| ())
    .unwrap_err()
    .to_string();
    let oldest = w.versions().unwrap().oldest_readable.unwrap();
    assert!(err.contains(&oldest.to_string()), "{err}");
}

/// A pin held *across* a live writer's compaction. The reader keeps serving the snapshot it
/// already has (it never re-reads the rewritten base), `refresh` still refuses to move it,
/// and only a fresh open at that version is refused — the floor cannot un-open a handle.
#[cfg_attr(miri, ignore)] // fsyncs: Miri has no sync_all
#[test]
fn a_pin_held_across_a_compaction_keeps_serving_and_never_moves() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(
        Config::new(&path, 2)
            .auto_compact(None)
            .segment_max_rows(Some(2))
            .history_versions(Some(10)),
    )
    .unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    w.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    let v = w.data.version();

    let mut r = Store::open(
        Config::new(&path, 2)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .at_version(Some(v)),
    )
    .unwrap();
    let before: std::collections::BTreeSet<String> =
        r.get_all("col").into_iter().map(|rec| rec.id).collect();
    assert_eq!(before, ["a", "b"].into_iter().map(String::from).collect());

    w.delete("col", &["a"]).unwrap();
    w.compact().unwrap();

    assert!(!r.refresh().unwrap(), "refresh must never cross a pin");
    let after: std::collections::BTreeSet<String> =
        r.get_all("col").into_iter().map(|rec| rec.id).collect();
    assert_eq!(after, before, "the held pin still serves its own snapshot");

    // But the version is gone for anyone opening now, and `refresh_to` back to it says so.
    let err = r.refresh_to(v).map(|_| ()).unwrap_err().to_string();
    let oldest = w.versions().unwrap().oldest_readable.unwrap();
    assert!(err.contains(&oldest.to_string()), "{err}");
    let survived: std::collections::BTreeSet<String> =
        r.get_all("col").into_iter().map(|rec| rec.id).collect();
    assert_eq!(
        survived, before,
        "a refused refresh_to left the snapshot serving"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn refresh_never_crosses_a_pin_but_refresh_to_does() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(
        Config::new(&path, 2)
            .auto_compact(None)
            .segment_max_rows(Some(2))
            .history_versions(Some(10)),
    )
    .unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    w.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    w.upsert("col", &[rec("c", vec![1.0, 1.0])]).unwrap(); // seals, records v1
    let v1 = w.data.version();

    let mut r = Store::open(
        Config::new(&path, 2)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .at_version(Some(v1)),
    )
    .unwrap();
    let before: std::collections::BTreeSet<String> =
        r.get_all("col").into_iter().map(|rec| rec.id).collect();

    w.upsert("col", &[rec("d", vec![0.0, 0.0])]).unwrap();
    w.upsert("col", &[rec("e", vec![1.0, 0.0])]).unwrap(); // seals again, records v2
    w.flush().unwrap();
    let v2 = w.data.version();
    assert!(v2 > v1, "the second seal must have advanced the version");

    assert!(!r.refresh().unwrap(), "refresh must never cross a pin");
    let still_pinned: std::collections::BTreeSet<String> =
        r.get_all("col").into_iter().map(|rec| rec.id).collect();
    assert_eq!(
        still_pinned, before,
        "refresh left the pinned snapshot alone"
    );

    // The pin is behind the live head, and `versions()` must say so rather than reporting
    // the pin twice (`self.data.version()` is the pinned version on a pinned handle).
    let vs = r.versions().unwrap();
    assert_eq!(vs.pinned, Some(v1));
    assert_eq!(
        vs.commit_version, v2,
        "commit_version is the live head, not the pin"
    );

    r.refresh_to(v2).unwrap();
    assert_eq!(r.pinned(), Some(v2));
    let moved: std::collections::BTreeSet<String> =
        r.get_all("col").into_iter().map(|rec| rec.id).collect();
    assert_ne!(moved, before, "refresh_to actually moved the snapshot");

    let fresh_pin = Store::open(
        Config::new(&path, 2)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .at_version(Some(v2)),
    )
    .unwrap();
    let expected: std::collections::BTreeSet<String> = fresh_pin
        .get_all("col")
        .into_iter()
        .map(|rec| rec.id)
        .collect();
    assert_eq!(moved, expected, "matches a fresh open at the same version");
}

#[cfg_attr(miri, ignore)]
#[test]
fn pinned_handle_rejects_every_mutation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(
        Config::new(&path, 2)
            .auto_compact(None)
            .segment_max_rows(Some(2))
            .history_versions(Some(10)),
    )
    .unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    w.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    w.upsert("col", &[rec("c", vec![1.0, 1.0])]).unwrap();
    let v = w.data.version();
    w.flush().unwrap();

    let mut pinned = Store::open(
        Config::new(&path, 2)
            .open_mode(OpenMode::ReadOnly)
            .auto_compact(None)
            .at_version(Some(v)),
    )
    .unwrap();
    assert!(pinned.upsert("col", &[rec("z", vec![0.0, 0.0])]).is_err());
    assert!(pinned.delete("col", &["a"]).is_err());
    assert!(pinned.compact().is_err());
    assert!(pinned.flush().is_err());
}

#[cfg_attr(miri, ignore)]
#[test]
fn open_at_with_history_off_names_the_reason() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut w = Store::open(Config::new(&path, 2).auto_compact(None)).unwrap();
        w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    }

    let err = Store::open(
        Config::new(&path, 2)
            .open_mode(OpenMode::ReadOnly)
            .at_version(Some(1)),
    )
    .map(|_| ())
    .unwrap_err()
    .to_string();
    assert!(err.contains("no history"), "{err}");
    assert!(err.contains("history_versions"), "{err}");
}

#[cfg_attr(miri, ignore)]
#[test]
fn pruning_moves_the_floor_and_versions_reports_the_survivors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(
        Config::new(&path, 2)
            .auto_compact(None)
            .segment_max_rows(Some(1))
            .history_versions(Some(2)),
    )
    .unwrap();
    for (i, id) in ["a", "b", "c", "d", "e"].into_iter().enumerate() {
        w.upsert("col", &[rec(id, vec![i as f32, 0.0])]).unwrap();
    }
    w.flush().unwrap();

    let vs = w.versions().unwrap();
    assert_eq!(vs.commit_version, w.data.version());
    assert_eq!(vs.pinned, None);
    let oldest = vs.oldest_readable.expect("some history was recorded");
    assert!(oldest > 1, "the earliest seals must have been pruned");
    assert_eq!(
        vs.readable,
        (oldest..=vs.commit_version).collect::<Vec<_>>(),
        "no gaps in the surviving window"
    );

    let err = Store::open(
        Config::new(&path, 2)
            .open_mode(OpenMode::ReadOnly)
            .at_version(Some(oldest - 1)),
    )
    .map(|_| ())
    .unwrap_err()
    .to_string();
    assert!(err.contains(&oldest.to_string()), "{err}");
}

#[cfg_attr(miri, ignore)]
#[test]
fn versions_reports_the_landscape_without_pruning() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();

    let mut w = Store::open(
        Config::new(&path, 2)
            .auto_compact(None)
            .segment_max_rows(Some(1))
            .history_versions(Some(10)),
    )
    .unwrap();
    w.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    w.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap(); // seals, records the only entry
    w.flush().unwrap();
    let v = w.data.version();

    // Version 1 has no entry: it is the manifest a fresh store opens at, published before
    // any batch. History begins at the first commit point, which is the first durable batch.
    let vs = w.versions().unwrap();
    assert_eq!(vs.commit_version, v);
    assert_eq!(vs.pinned, None);
    assert_eq!(vs.oldest_readable, Some(2));
    assert_eq!(vs.readable, vec![2, 3, 4, v]);

    let pinned = Store::open(
        Config::new(&path, 2)
            .open_mode(OpenMode::ReadOnly)
            .at_version(Some(v)),
    )
    .unwrap();
    assert_eq!(pinned.versions().unwrap().pinned, Some(v));
}

#[cfg_attr(miri, ignore)]
#[test]
fn cluster_mode_rejects_a_local_filesystem_store() {
    // Cluster mode needs a shared backend; a local-FS store is single-node and is refused
    // with a clear error (the shared-memory check is never reached).
    let dir = tempfile::tempdir().unwrap();
    let Err(err) = Store::open(Config::new(dir.path(), 2).cluster(true)) else {
        panic!("a local-filesystem cluster store must be rejected");
    };
    assert!(err.to_string().contains("object-store"), "{err}");
}

// ── Live object-store backing + shared memory tier (Miri-clean: all in-RAM) ──────
// Exercises the whole-object live-backing path and the shared memory tier through a fake in-RAM
// whole-object Persistence — no files, no network, no fsync — so it runs under Miri.

mod object_backed {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use anyhow::{Result, bail};

    use super::*;
    use crate::backend::{BackendLock, CasOutcome, LocalRam, MemoryTier, Persistence};

    /// A whole-object Persistence over an in-RAM map: no native appender or lock, forcing the
    /// `ObjectAppender` and advisory-lock paths — the shape S3/GCS present, but synchronous and
    /// Miri-clean. Each object carries a monotonic generation, modelling an `ETag` so CAS is exercised.
    #[derive(Default)]
    struct InMemObjectStore {
        objects: Mutex<HashMap<String, (Vec<u8>, u64)>>,
        next_gen: AtomicU64,
        /// Per-key read count (`get` + `get_cas`), so a test can assert which objects a
        /// refresh fetched — e.g. that an incremental refresh skips immutable segments.
        gets: Mutex<HashMap<String, u64>>,
        /// Per-key durable write count. On a whole-object backend a segment's `sync()` *is* a `put`,
        /// so this counts barriers the way an fsync counter would locally — which makes group-commit
        /// coalescing directly measurable rather than inferred (nidus-xb9.1).
        puts: Mutex<HashMap<String, u64>>,
    }

    impl InMemObjectStore {
        /// Mint a fresh, never-reused generation token for a write.
        fn bump(&self) -> u64 {
            self.next_gen.fetch_add(1, Ordering::Relaxed) + 1
        }
        /// Record a read of `key`.
        fn note_get(&self, key: &str) {
            *self
                .gets
                .lock()
                .unwrap()
                .entry(key.to_string())
                .or_insert(0) += 1;
        }
        /// How many times `key` has been read.
        fn get_count(&self, key: &str) -> u64 {
            self.gets.lock().unwrap().get(key).copied().unwrap_or(0)
        }
        /// Forget all read counts (call before the action under test).
        fn reset_gets(&self) {
            self.gets.lock().unwrap().clear();
        }
        /// Record a durable write of `key`.
        fn note_put(&self, key: &str) {
            *self
                .puts
                .lock()
                .unwrap()
                .entry(key.to_string())
                .or_insert(0) += 1;
        }
        /// How many times `key` has been durably written.
        fn put_count(&self, key: &str) -> u64 {
            self.puts.lock().unwrap().get(key).copied().unwrap_or(0)
        }
        /// Forget all write counts (call before the action under test).
        fn reset_puts(&self) {
            self.puts.lock().unwrap().clear();
        }
    }

    impl Persistence for InMemObjectStore {
        fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
            self.note_get(key);
            Ok(self
                .objects
                .lock()
                .unwrap()
                .get(key)
                .map(|(b, _)| b.clone()))
        }
        fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
            self.note_put(key);
            let g = self.bump();
            self.objects
                .lock()
                .unwrap()
                .insert(key.to_string(), (bytes.to_vec(), g));
            Ok(())
        }
        fn delete(&self, key: &str) -> Result<()> {
            self.objects.lock().unwrap().remove(key);
            Ok(())
        }
        fn list(&self) -> Result<Vec<String>> {
            let mut keys: Vec<String> = self.objects.lock().unwrap().keys().cloned().collect();
            keys.sort();
            Ok(keys)
        }
        fn get_cas(&self, key: &str) -> Result<Option<(Vec<u8>, Option<String>)>> {
            self.note_get(key);
            Ok(self
                .objects
                .lock()
                .unwrap()
                .get(key)
                .map(|(b, g)| (b.clone(), Some(g.to_string()))))
        }
        fn supports_cas(&self) -> bool {
            true
        }

        fn put_cas(&self, key: &str, bytes: &[u8], expected: Option<&str>) -> Result<CasOutcome> {
            // Atomic compare-and-swap under the map lock — the conditional-write primitive
            // S3/GCS give (If-Match / ifGenerationMatch, and If-None-Match:* / =0 when
            // `expected` is None). `try_create_exclusive` rides the `None` arm via its default.
            let mut objs = self.objects.lock().unwrap();
            let current = objs.get(key).map(|(_, g)| g.to_string());
            let matches = match (expected, &current) {
                (None, None) => true,                     // create-if-absent: must be absent
                (Some(want), Some(have)) => want == have, // compare current token
                _ => false,                               // absent-vs-present or token mismatch
            };
            if !matches {
                return Ok(CasOutcome::Stale);
            }
            self.note_put(key);
            let g = self.next_gen.fetch_add(1, Ordering::Relaxed) + 1;
            objs.insert(key.to_string(), (bytes.to_vec(), g));
            Ok(CasOutcome::Written(Some(g.to_string())))
        }
        fn try_lock(&self, _key: &str, _ttl: Duration) -> Result<Option<Box<dyn BackendLock>>> {
            bail!("InMemObjectStore has no native lock — advisory lock is used instead")
        }
        fn has_native_lock(&self) -> bool {
            false
        }
    }

    fn cfg() -> Config {
        Config::new("/unused/object-store", 3)
            .open_mode(OpenMode::ReadWrite)
            .auto_compact(None)
    }

    fn has_key(backend: &Arc<dyn Persistence>, key: &str) -> bool {
        backend.get(key).unwrap().is_some()
    }

    #[test]
    fn live_object_backing_round_trips_through_a_shared_tier() {
        let backend: Arc<dyn Persistence> = Arc::new(InMemObjectStore::default());
        let tier = Arc::new(LocalRam::new());

        // 1. Open over the object backend + shared tier; write, flush, close.
        {
            let mem: Box<dyn MemoryTier> = Box::new(tier.clone());
            let mut store =
                Store::open_with(cfg(), "s3://bucket/store", backend.clone(), Some(mem)).unwrap();
            store.create_collection("col").unwrap();
            store
                .upsert(
                    "col",
                    &[rec("a", vec![1.0, 0.0, 0.0]), rec("b", vec![0.0, 1.0, 0.0])],
                )
                .unwrap();
            store.flush().unwrap();
        }

        // The data/log live as whole objects on the backend, and flush published the
        // working set to the shared tier.
        assert!(has_key(&backend, "data"), "data object written");
        assert!(has_key(&backend, "log"), "log object written");
        assert!(
            tier.load("workingset").unwrap().is_some(),
            "working set published to the tier"
        );
        // The advisory lock object was released on close.
        assert!(!has_key(&backend, "lock"), "lock released on drop");

        // 2. Reopen over the same backend + tier: data is intact and searchable (the
        //    index came from the adopted working set, which by construction matches a
        //    fresh replay of the same log).
        let mem: Box<dyn MemoryTier> = Box::new(tier.clone());
        let store =
            Store::open_with(cfg(), "s3://bucket/store", backend.clone(), Some(mem)).unwrap();
        let hits = store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits[0].id, "a", "nearest neighbour survives the round-trip");
        assert_eq!(hits.len(), 2, "both rows present after reopen");
    }

    #[test]
    fn advisory_lock_excludes_a_second_live_writer() {
        let backend: Arc<dyn Persistence> = Arc::new(InMemObjectStore::default());

        let first = Store::open_with(cfg(), "s3://bucket/store", backend.clone(), None).unwrap();
        // A second writer over the same backend is refused while the first holds the lock.
        let Err(err) = Store::open_with(cfg(), "s3://bucket/store", backend.clone(), None) else {
            panic!("second open must be locked out");
        };
        assert!(err.to_string().contains("locked"), "{err}");

        // Releasing the first lets a new writer in.
        drop(first);
        assert!(
            Store::open_with(cfg(), "s3://bucket/store", backend.clone(), None).is_ok(),
            "lock is reclaimable after the holder drops"
        );
    }

    // ── Cluster mode: shared backend + writer lease + commit-counter refresh (§14.6 ph5) ──

    fn cluster_cfg() -> Config {
        cfg().cluster(true)
    }

    /// A cluster writer over a shared object backend + shared tier.
    fn cluster_writer(backend: &Arc<dyn Persistence>, tier: &Arc<LocalRam>) -> Store {
        let mem: Box<dyn MemoryTier> = Box::new(tier.clone());
        Store::open_with(
            cluster_cfg(),
            "s3://bucket/store",
            backend.clone(),
            Some(mem),
        )
        .unwrap()
    }

    /// A lock-free cluster reader over the same shared backend + tier.
    fn cluster_reader(backend: &Arc<dyn Persistence>, tier: &Arc<LocalRam>) -> Store {
        let mem: Box<dyn MemoryTier> = Box::new(tier.clone());
        Store::open_with(
            cluster_cfg().open_mode(OpenMode::ReadOnly),
            "s3://bucket/store",
            backend.clone(),
            Some(mem),
        )
        .unwrap()
    }

    #[test]
    fn cluster_mode_requires_a_shared_memory_tier() {
        let backend: Arc<dyn Persistence> = Arc::new(InMemObjectStore::default());
        // Object-store persistence is fine, but with no shared memory tier cluster is rejected.
        let Err(err) = Store::open_with(cluster_cfg(), "s3://bucket/store", backend.clone(), None)
        else {
            panic!("cluster without a memory tier must be rejected");
        };
        assert!(err.to_string().contains("shared memory tier"), "{err}");
    }

    #[test]
    fn cluster_writer_lease_excludes_a_second_writer() {
        let backend: Arc<dyn Persistence> = Arc::new(InMemObjectStore::default());
        let tier = Arc::new(LocalRam::new());

        let first = cluster_writer(&backend, &tier);
        // A second cluster writer over the same backend is fenced out by the live lease.
        let mem: Box<dyn MemoryTier> = Box::new(tier.clone());
        let Err(err) = Store::open_with(
            cluster_cfg(),
            "s3://bucket/store",
            backend.clone(),
            Some(mem),
        ) else {
            panic!("second cluster writer must be locked out");
        };
        assert!(err.to_string().contains("locked"), "{err}");

        // Releasing the first lets a new writer take the lease.
        drop(first);
        let mem: Box<dyn MemoryTier> = Box::new(tier.clone());
        assert!(
            Store::open_with(
                cluster_cfg(),
                "s3://bucket/store",
                backend.clone(),
                Some(mem)
            )
            .is_ok(),
            "lease is reclaimable after the holder drops"
        );
    }

    #[test]
    fn cluster_lease_renews_across_many_batches() {
        let backend: Arc<dyn Persistence> = Arc::new(InMemObjectStore::default());
        let tier = Arc::new(LocalRam::new());
        let mut w = cluster_writer(&backend, &tier);
        w.create_collection("col").unwrap();
        // Many sequential batches: op-driven renewal must keep the lease alive throughout
        // (no background thread), so none of these errors.
        for i in 0..25 {
            w.upsert("col", &[rec(&format!("d{i}"), vec![1.0, 0.0, 0.0])])
                .unwrap();
        }
        w.flush().unwrap();
        assert_eq!(w.get_all("col").len(), 25);
    }

    #[test]
    fn cluster_lease_fences_a_superseded_writer() {
        let backend: Arc<dyn Persistence> = Arc::new(InMemObjectStore::default());
        let tier = Arc::new(LocalRam::new());
        let mut w = cluster_writer(&backend, &tier);
        w.create_collection("col").unwrap();
        w.upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])]).unwrap();

        // Simulate a peer taking over the lease while this writer was paused (a fresh stamp
        // under a *different* owner — what a stale-reclaim by another instance writes).
        backend.put("lock", b"9999999999 other-writer-77").unwrap();

        // The next mutation renews-and-fences first, so it errors before clobbering the store.
        let err = w
            .upsert("col", &[rec("b", vec![0.0, 1.0, 0.0])])
            .expect_err("a superseded writer must be fenced");
        assert!(err.to_string().contains("lease lost"), "{err}");
        // The fenced write left no trace: "b" never landed.
        assert_eq!(w.get_all("col").len(), 1);
    }

    #[test]
    fn cluster_reader_refreshes_on_every_commit() {
        let backend: Arc<dyn Persistence> = Arc::new(InMemObjectStore::default());
        let tier = Arc::new(LocalRam::new());

        let mut w = cluster_writer(&backend, &tier);
        w.create_collection("col").unwrap();
        w.upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])]).unwrap();

        let mut r = cluster_reader(&backend, &tier);
        assert_eq!(r.get_all("col").len(), 1);
        let v0 = r.data.version();

        // A *plain* append (no seal) — on an object store this bumps neither the log's
        // visible length nor (pre-phase-5) the manifest. The commit-counter makes it visible.
        w.upsert("col", &[rec("b", vec![0.0, 1.0, 0.0])]).unwrap();
        assert!(
            r.refresh().unwrap(),
            "the commit advanced the manifest version"
        );
        assert!(r.data.version() > v0, "reader adopted the newer version");
        assert_eq!(r.get_all("col").len(), 2);
        let hits = r
            .search(&["col"], &[0.0, 1.0, 0.0], &default_opts(5))
            .unwrap();
        assert_eq!(hits[0].id, "b");

        // Nothing newer committed → a refresh is a no-op.
        assert!(!r.refresh().unwrap(), "already current");
    }

    #[test]
    fn cluster_reader_matches_a_fresh_open_after_refresh() {
        let backend: Arc<dyn Persistence> = Arc::new(InMemObjectStore::default());
        let tier = Arc::new(LocalRam::new());

        let mut w = cluster_writer(&backend, &tier);
        w.create_collection("col").unwrap();
        w.upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])]).unwrap();

        let mut r = cluster_reader(&backend, &tier);

        // Several more commits: appends, an overwrite, a delete.
        w.upsert("col", &[rec("b", vec![0.0, 1.0, 0.0])]).unwrap();
        w.upsert("col", &[rec("c", vec![0.0, 0.0, 1.0])]).unwrap();
        w.upsert("col", &[rec("a", vec![0.5, 0.5, 0.0])]).unwrap(); // overwrite
        w.delete("col", &["b"]).unwrap();
        w.flush().unwrap();

        assert!(r.refresh().unwrap());
        // The refreshed reader matches a reader freshly opened over the same shared store.
        let fresh = cluster_reader(&backend, &tier);
        for q in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ] {
            let got: Vec<(String, f32)> = r
                .search(&["col"], &q, &default_opts(5))
                .unwrap()
                .into_iter()
                .map(|h| (h.id, h.score))
                .collect();
            let exp: Vec<(String, f32)> = fresh
                .search(&["col"], &q, &default_opts(5))
                .unwrap()
                .into_iter()
                .map(|h| (h.id, h.score))
                .collect();
            assert_eq!(got, exp, "query {q:?}");
        }
    }

    #[test]
    fn cluster_cas_fences_a_superseded_writer_and_preserves_committed_data() {
        let backend: Arc<dyn Persistence> = Arc::new(InMemObjectStore::default());
        let tier = Arc::new(LocalRam::new());

        let mut w = cluster_writer(&backend, &tier);
        w.create_collection("col").unwrap();
        w.upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])]).unwrap();

        // Simulate a peer taking over and committing: it rewrites the shared objects, advancing
        // their CAS tokens and stranding this writer's anchors. Its `lock` is left untouched so the
        // lease renew still passes, isolating the CAS fence and the window the lease cannot close.
        for key in ["data", "log", "manifest"] {
            let bytes = backend.get(key).unwrap().unwrap();
            backend.put(key, &bytes).unwrap(); // identical bytes, fresh generation token
        }

        // The superseded writer's next durable write is refused by the compare-and-swap — it
        // fails cleanly instead of clobbering the peer's committed bytes.
        let err = w
            .upsert("col", &[rec("b", vec![0.0, 1.0, 0.0])])
            .expect_err("a superseded cluster writer must be fenced before it clobbers the store");
        let msg = format!("{err:#}"); // include the error source chain (the fence is the cause)
        assert!(
            msg.contains("fenced") || msg.contains("superseded"),
            "expected a fencing error, got: {msg}"
        );

        // The committed state survived intact: a fresh reader sees exactly the peer's "a",
        // never the fenced writer's "b".
        let r = cluster_reader(&backend, &tier);
        let ids: Vec<String> = r.get_all("col").into_iter().map(|h| h.id).collect();
        assert_eq!(ids, vec!["a".to_string()]);
    }

    #[test]
    fn cluster_manifest_commit_is_compare_and_swapped() {
        // Isolate the manifest commit's CAS (the structural commit point) using OnFlush, where
        // `flush` is the sole committer: after a peer bumps only the `manifest` token, the next
        // flush's manifest compare-and-swap must fail rather than republish a stale segment set.
        let backend: Arc<dyn Persistence> = Arc::new(InMemObjectStore::default());
        let tier = Arc::new(LocalRam::new());
        let mem: Box<dyn MemoryTier> = Box::new(tier.clone());
        let mut w = Store::open_with(
            cluster_cfg().fsync(crate::config::Fsync::OnFlush),
            "s3://bucket/store",
            backend.clone(),
            Some(mem),
        )
        .unwrap();
        w.create_collection("col").unwrap();
        w.upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])]).unwrap();
        w.flush().unwrap();

        // A peer republishes the manifest out-of-band, advancing its CAS token.
        let m = backend.get("manifest").unwrap().unwrap();
        backend.put("manifest", &m).unwrap();

        // The next flush commits the manifest conditionally on the token this writer last wrote
        // — now stale — so it is fenced at the commit point.
        let err = w
            .flush()
            .expect_err("a stale manifest commit must be compare-and-swapped out");
        assert!(err.to_string().contains("lease lost"), "{err}");
    }

    #[test]
    fn cluster_incremental_refresh_skips_immutable_segments() {
        // Keep a concrete handle for the GET-count assertions while passing a trait object to
        // the store helpers.
        let raw = Arc::new(InMemObjectStore::default());
        let backend: Arc<dyn Persistence> = raw.clone();
        let tier = Arc::new(LocalRam::new());

        // `segment_max_rows = 2` seals the base after two rows, so the store ends up with an
        // immutable base segment ("data") plus a fresh active segment ("seg-00000001").
        let mem: Box<dyn MemoryTier> = Box::new(tier.clone());
        let mut w = Store::open_with(
            cluster_cfg().segment_max_rows(Some(2)),
            "s3://bucket/store",
            backend.clone(),
            Some(mem),
        )
        .unwrap();
        w.create_collection("col").unwrap();
        w.upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])]).unwrap();
        w.upsert("col", &[rec("b", vec![0.0, 1.0, 0.0])]).unwrap();
        w.upsert("col", &[rec("c", vec![0.0, 0.0, 1.0])]).unwrap(); // seals [a,b]→data, c→seg-1

        let mut r = cluster_reader(&backend, &tier);
        assert_eq!(r.get_all("col").len(), 3);

        // A plain append to the active segment — no new seal, so the segment *list* is unchanged
        // and the reader takes the incremental fast path.
        w.upsert("col", &[rec("d", vec![1.0, 1.0, 0.0])]).unwrap();

        raw.reset_gets();
        assert!(r.refresh().unwrap(), "the commit is detected");
        assert_eq!(
            raw.get_count("data"),
            0,
            "the immutable base segment must not be re-fetched on an incremental refresh"
        );
        assert!(
            raw.get_count("seg-00000001") >= 1,
            "the active segment is re-read to pick up the appended row"
        );
        assert_eq!(
            r.get_all("col").len(),
            4,
            "the new row is visible after refresh"
        );

        // A seal *does* restructure the list → the next refresh re-opens the whole set.
        w.upsert("col", &[rec("e", vec![0.5, 0.5, 0.0])]).unwrap(); // seg-1 now [c,d] → seals, e→seg-2
        raw.reset_gets();
        assert!(r.refresh().unwrap());
        assert!(
            raw.get_count("data") >= 1,
            "a restructure (seal) re-opens every segment, including the immutable base"
        );
        assert_eq!(r.get_all("col").len(), 5);
    }

    /// **Group commit issues fewer barriers than there are batches — counted, not inferred.**
    #[test]
    fn group_commit_coalesces_the_barrier_across_batches() {
        let raw = Arc::new(InMemObjectStore::default());
        let backend: Arc<dyn Persistence> = raw.clone();
        let mut store = Store::open_with(cfg(), "s3://bucket/store", backend, None).unwrap();
        store.create_collection("col").unwrap();

        // Baseline: four separate batches, each taking its own barrier.
        raw.reset_puts();
        for i in 0..4u32 {
            store
                .upsert("col", &[rec(&format!("solo-{i}"), vec![1.0, 0.0, 0.0])])
                .unwrap();
        }
        let (solo_data, solo_log) = (raw.put_count("data"), raw.put_count("log"));
        assert_eq!(
            (solo_data, solo_log),
            (4, 4),
            "without group commit every batch pays its own data+log barrier"
        );

        // The same four batches inside one deferred scope, closed by one `commit`.
        raw.reset_puts();
        let prev = store.begin_deferred();
        for i in 0..4u32 {
            store
                .upsert("col", &[rec(&format!("group-{i}"), vec![0.0, 1.0, 0.0])])
                .unwrap();
        }
        store.end_deferred(prev);
        assert_eq!(
            (raw.put_count("data"), raw.put_count("log")),
            (0, 0),
            "a deferred batch appends without taking a barrier"
        );
        store.commit().unwrap();
        assert_eq!(
            (raw.put_count("data"), raw.put_count("log")),
            (1, 1),
            "one barrier covers the whole group"
        );

        // And the data is all there — coalescing barriers must not lose batches.
        assert_eq!(store.get_all("col").len(), 8);

        // A second `commit` with nothing owed is free: no barrier, so the uncontended
        // single-writer path is not taxed by a redundant fsync.
        raw.reset_puts();
        store.commit().unwrap();
        assert_eq!((raw.put_count("data"), raw.put_count("log")), (0, 0));
    }

    /// **In cluster mode the coalesced group publishes the commit counter once, not per batch.**
    #[test]
    fn group_commit_publishes_one_cluster_commit_for_the_group() {
        let raw = Arc::new(InMemObjectStore::default());
        let backend: Arc<dyn Persistence> = raw.clone();
        let tier = Arc::new(LocalRam::new());
        let mem: Box<dyn MemoryTier> = Box::new(tier.clone());
        let mut w = Store::open_with(
            cluster_cfg(),
            "s3://bucket/store",
            backend.clone(),
            Some(mem),
        )
        .unwrap();
        w.create_collection("col").unwrap();

        let before = Manifest::load(backend.as_ref()).unwrap().unwrap().version;
        raw.reset_puts();
        let prev = w.begin_deferred();
        for i in 0..3u32 {
            w.upsert("col", &[rec(&format!("g-{i}"), vec![1.0, 0.0, 0.0])])
                .unwrap();
        }
        w.end_deferred(prev);
        w.commit().unwrap();
        assert_eq!(
            raw.put_count("manifest"),
            1,
            "three batches, one manifest publish"
        );

        let after = Manifest::load(backend.as_ref()).unwrap().unwrap().version;
        assert!(
            after > before,
            "the commit counter must still advance, or a reader never sees the group \
             ({before} → {after})"
        );
        let mut r = cluster_reader(&backend, &tier);
        assert_eq!(
            r.get_all("col").len(),
            3,
            "and a peer picks the whole group up on one refresh"
        );
        assert!(!r.refresh().unwrap(), "which leaves it current");
    }

    /// **A failed barrier fails every write in its group.**
    #[test]
    fn a_failed_group_barrier_is_reported_not_swallowed() {
        let raw = Arc::new(InMemObjectStore::default());
        let backend: Arc<dyn Persistence> = raw.clone();
        let tier = Arc::new(LocalRam::new());
        let mem: Box<dyn MemoryTier> = Box::new(tier.clone());
        let mut w = Store::open_with(
            cluster_cfg(),
            "s3://bucket/store",
            backend.clone(),
            Some(mem),
        )
        .unwrap();
        w.create_collection("col").unwrap();

        let prev = w.begin_deferred();
        for i in 0..3u32 {
            w.upsert("col", &[rec(&format!("g-{i}"), vec![1.0, 0.0, 0.0])])
                .unwrap();
        }
        w.end_deferred(prev);

        // Someone else advances the manifest: this writer's CAS token is now stale.
        let m = backend.get("manifest").unwrap().unwrap();
        backend.put("manifest", &m).unwrap();

        let err = w
            .commit()
            .expect_err("a superseded writer's group barrier must fail");
        assert!(err.to_string().contains("lease lost"), "{err}");

        // Still owed: a failed barrier must not clear the debt, or the next `commit` would
        // report durability that never happened.
        let err = w
            .commit()
            .expect_err("the barrier is still owed and retried");
        assert!(err.to_string().contains("lease lost"), "{err}");
    }

    /// **A scope-wide delete that fails mid-flight deletes nothing (nidus-166).**
    #[test]
    fn a_failed_scope_wide_delete_leaves_every_collection_intact() {
        let raw = Arc::new(InMemObjectStore::default());
        let backend: Arc<dyn Persistence> = raw.clone();
        let tier = Arc::new(LocalRam::new());
        let mut w = cluster_writer(&backend, &tier);

        let expired =
            BTreeMap::from([(crate::meta::META_EXPIRES_AT.to_string(), Value::DateTime(1))]);
        for c in ["a", "b"] {
            w.create_collection(c).unwrap();
            w.upsert(
                c,
                &[rec_with(
                    &format!("{c}-1"),
                    vec![1.0, 0.0, 0.0],
                    expired.clone(),
                )],
            )
            .unwrap();
        }

        // Someone else advances the manifest, so this writer's CAS token is stale and the
        // sweep's durable barrier fails *after* its log records are appended.
        let m = backend.get("manifest").unwrap().unwrap();
        backend.put("manifest", &m).unwrap();

        let filter = Filter(vec![Predicate::Le(
            crate::meta::META_EXPIRES_AT.to_string(),
            Value::DateTime(i64::MAX),
        )]);
        w.delete_where_all(&filter)
            .expect_err("a failed barrier must fail the whole sweep");

        // Looping `delete_where` instead renewed the lease per collection and committed each
        // to RAM before its barrier, so "a" was already gone while the caller saw an error.
        for c in ["a", "b"] {
            assert_eq!(
                w.collections.get(c).unwrap().docs.len(),
                1,
                "collection {c} must be untouched by a failed sweep"
            );
        }
    }
}

// ── Cooperative cancellation ────────────────────────────────────────────────

/// **A cancelled scan stops instead of running to completion.**
#[test]
fn a_cancelled_search_stops_and_errors() {
    let mut store = Store::in_memory(8).unwrap();
    store.create_collection("col").unwrap();
    let recs: Vec<Record> = (0..64)
        .map(|i| {
            let v: Vec<f32> = (0..8).map(|d| ((i * 7 + d) % 13) as f32).collect();
            Record::new(format!("d{i}"), v, Default::default())
        })
        .collect();
    store.upsert("col", &recs).unwrap();

    let q = vec![1.0f32; 8];
    let opts = SearchOpts {
        top_k: 10,
        ..Default::default()
    };

    // Uncancelled: the ambient token is absent, so nothing changes for ordinary callers —
    // the overwhelmingly common case, and the one a regression here would break.
    assert_eq!(store.search(&["col"], &q, &opts).unwrap().len(), 10);

    // Cancelled: it must refuse rather than return partial results, which would look like
    // a legitimate (but wrong) ranking.
    let token = crate::Cancel::new();
    token.cancel();
    let err = token
        .scope(|| store.search(&["col"], &q, &opts))
        .expect_err("a cancelled search must not return results");
    assert!(
        format!("{err:#}").contains("cancelled"),
        "the error should say why: {err:#}"
    );

    // An installed but un-cancelled token is not cancellation.
    let live = crate::Cancel::new();
    assert_eq!(
        live.scope(|| store.search(&["col"], &q, &opts))
            .unwrap()
            .len(),
        10
    );
}

// ── Query-dimension validation (nidus-c5v) ───────────────────────────────────
// Dimension is pinned in the `data` header, so a wrong-length query is unanswerable. It used to be
// answered anyway: `dot` zips and stops at the shorter, returning a plausible-looking prefix score.

/// The wording matters beyond readability: the server's `classify` maps this substring to
/// `400`, so a reworded message would silently downgrade the HTTP status to `500`.
fn assert_dim_error(err: anyhow::Error, want_len: usize, want_dim: usize) {
    let msg = format!("{err:#}");
    assert!(
        msg.contains("does not match store dimension"),
        "message must keep the substring `classify` matches for 400, got: {msg}"
    );
    assert!(
        msg.contains(&want_len.to_string()) && msg.contains(&want_dim.to_string()),
        "message should name both lengths ({want_len} vs {want_dim}), got: {msg}"
    );
}

#[test]
fn search_rejects_a_query_of_the_wrong_dimension() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert("col", &[rec("a", vec![1.0, 0.0, 0.0])])
        .unwrap();

    // Short, long, and empty. The short case is the dangerous one: it used to score
    // against a prefix and return `Ok`, so "too few dimensions" was indistinguishable
    // from a real answer.
    for bad in [vec![1.0, 0.0], vec![1.0, 0.0, 0.0, 9.0], vec![]] {
        let n = bad.len();
        let err = match store.search(&["col"], &bad, &default_opts(5)) {
            Ok(hits) => panic!(
                "a {n}-dim query against a dim-3 store must be refused, got {} hit(s)",
                hits.len()
            ),
            Err(e) => e,
        };
        assert_dim_error(err, n, 3);
    }

    // The correct length still works — the guard must not have broken the happy path.
    assert_eq!(
        store
            .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn search_rejects_a_bad_dimension_even_on_an_empty_store() {
    // No rows, so a scan would find nothing and return `Ok(vec![])` regardless. That is
    // the whole point: "wrong shape" must not be reported as "nothing matched", because a
    // caller cannot tell those apart and the wrong-shape case is a bug in their code.
    let store = Store::in_memory(8).unwrap();
    let err = store
        .search(&["col"], &[1.0, 2.0], &default_opts(5))
        .expect_err("dimension is pinned at creation, not inferred from the rows present");
    assert_dim_error(err, 2, 8);
}

#[test]
fn search_rejects_a_bad_dimension_on_the_ann_path_too() {
    // `search` validates before dispatching, so every path inherits the guard. Asserted
    // rather than assumed: the ANN walk is a separate code path that never touches
    // `rank_scan`, and it is the path a production store is most likely to be running.
    let vectors: Vec<Vec<f32>> = (0..32)
        .map(|i| {
            let x = i as f32;
            vec![x, 1.0 - x, x * 0.5, 2.0 - x]
        })
        .collect();
    let store = ann_store(4, AnnConfig::hnsw(), &vectors);
    let err = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
        .expect_err("the ANN path must refuse a mismatched query too");
    assert_dim_error(err, 3, 4);
}

#[test]
fn hybrid_search_rejects_a_bad_dimension_regardless_of_top_k() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    store
        .upsert("docs", &[doc("c", "quantum physics")])
        .unwrap();

    let query = FtsQuery::new("body", "quantum");
    for top_k in [10, 0] {
        let opts = HybridOpts {
            top_k,
            ..Default::default()
        };
        // `top_k: 0` short-circuits before the vector leg runs, so without an explicit guard in
        // `hybrid_search` the same bad query would be accepted at `top_k: 0` and refused at `top_k:
        // 10`. A verdict that depends on `top_k` is a miserable thing to debug.
        let err = store
            .hybrid_search(&["docs"], &[1.0, 0.0], &query, &opts)
            .expect_err("hybrid must refuse a mismatched vector at any top_k");
        assert_dim_error(err, 2, 3);
    }
}

#[test]
fn a_refused_query_is_not_counted_as_a_served_one() {
    // `search_queries` underpins every ratio built on it (ANN-vs-exact share, mean rows
    // scanned). Counting a request the store refused would overstate the denominator.
    let store = Store::in_memory(3).unwrap();
    // The counter is process-global and sibling tests search concurrently, so a moved
    // counter is ambiguous while a still one is proof. Pass on the first clean window; a
    // real regression increments in every window and still fails.
    let unmoved = (0..64).any(|_| {
        let before = crate::metrics::metrics().search_queries.get();
        assert!(store.search(&["col"], &[1.0], &default_opts(5)).is_err());
        crate::metrics::metrics().search_queries.get() == before
    });
    assert!(
        unmoved,
        "a rejected query must not move the served-query counter"
    );
}

/// A caller-supplied `top_k` reached the bounded heap's reservation unclamped and aborted
/// on "capacity overflow" (nidus-m50.17). It must simply return everything there is.
#[test]
fn an_absurd_top_k_returns_every_doc_instead_of_panicking() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert(
            "col",
            &[rec("a", vec![1.0, 0.0, 0.0]), rec("b", vec![0.0, 1.0, 0.0])],
        )
        .unwrap();
    let huge = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(usize::MAX / 2))
        .unwrap();
    // Identical to a sane `top_k` over the same store: the clamp bounds only the allocation.
    let sane = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(10))
        .unwrap();
    assert_eq!(huge, sane);
    assert_eq!(huge.len(), 2);
    assert_eq!(huge[0].id, "a");
}

/// A frozen record of `hybrid_search`'s exact output, asserted bit-for-bit so the RRF
/// extraction into `crate::fuse` cannot quietly change a fused score or a tie-break.
#[test]
// BM25's `idf` calls `ln`, which Miri evaluates non-deterministically; the fused bits
// then vary by an ULP run-to-run. Real float semantics make these exact.
#[cfg_attr(miri, ignore)]
fn hybrid_search_is_unchanged_by_the_fusion_extraction() {
    let (store, vector, text) = golden_fixture();
    // Ids paired with the exact bits of their fused score, most relevant first.
    let ranked = |top_k, rrf_k, candidates| -> Vec<(String, u32)> {
        let opts = HybridOpts {
            top_k,
            rrf_k,
            candidates,
            ..Default::default()
        };
        store
            .hybrid_search(&["docs"], &vector, &text, &opts)
            .unwrap()
            .iter()
            .map(|h| (h.id.clone(), h.score.to_bits()))
            .collect()
    };
    let frozen = |pairs: &[(&str, u32)]| -> Vec<(String, u32)> {
        pairs
            .iter()
            .map(|(id, b)| ((*id).to_string(), *b))
            .collect()
    };

    assert_eq!(
        ranked(5, 60.0, 100),
        frozen(&[
            ("a", 1023751753),
            ("d", 1023476752),
            ("c", 1015434122),
            ("b", 1015292168),
            ("e", 1015154721),
        ])
    );
    // A non-default `rrf_k`, and `candidates` below `top_k` so the leg-depth clamp runs.
    assert_eq!(
        ranked(3, 5.0, 2),
        frozen(&[("a", 1050573288), ("c", 1042983595), ("b", 1041385765)])
    );
}

/// Five docs spanning every fusion case: both legs, vector-only, text-only, and a doc
/// each leg ranks differently — so a change in either leg or in the fusion shows up.
fn golden_fixture() -> (Store, Vec<f32>, FtsQuery) {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    let mut a = Record::new("a", vec![1.0, 0.0, 0.0], BTreeMap::new());
    a.attrs.insert(
        "body".to_string(),
        Value::Str("quantum physics lecture notes".to_string()),
    );
    let mut b = Record::new("b", vec![0.9, 0.1, 0.0], BTreeMap::new());
    b.attrs.insert(
        "body".to_string(),
        Value::Str("unrelated cooking recipe".to_string()),
    );
    let c = doc("c", "quantum physics quantum entanglement");
    let mut d = Record::new("d", vec![0.0, 1.0, 0.0], BTreeMap::new());
    d.attrs.insert(
        "body".to_string(),
        Value::Str("the physics of cooking".to_string()),
    );
    let e = Record::new("e", vec![0.5, 0.5, 0.0], BTreeMap::new());
    store.upsert("docs", &[a, b, c, d, e]).unwrap();
    (
        store,
        vec![1.0, 0.0, 0.0],
        FtsQuery::new("body", "quantum physics"),
    )
}

// ── Pagination (nidus-m50.8) ───────────────────────────────────────────────

/// Ten docs whose cosine scores against `[1, 0, 0]` are strictly decreasing in `d0..d9`,
/// so the expected ranking is known without recomputing it.
fn ranked_store() -> Store {
    let mut store = Store::in_memory(3).unwrap();
    let recs: Vec<Record> = (0..10)
        .map(|i| rec(&format!("d{i}"), vec![1.0, i as f32 * 0.1, 0.0]))
        .collect();
    store.upsert("docs", &recs).unwrap();
    store
}

fn ids(hits: &[crate::model::Hit]) -> Vec<String> {
    hits.iter().map(|h| h.id.clone()).collect()
}

fn page(top_k: usize, offset: usize) -> SearchOpts {
    SearchOpts {
        top_k,
        offset,
        ..Default::default()
    }
}

#[test]
fn search_pages_partition_the_ranking() {
    let store = ranked_store();
    let q = [1.0, 0.0, 0.0];
    let first = store.search(&["docs"], &q, &page(3, 0)).unwrap();
    let second = store.search(&["docs"], &q, &page(3, 3)).unwrap();
    let both = store.search(&["docs"], &q, &page(6, 0)).unwrap();

    assert_eq!(ids(&first).len(), 3);
    assert_eq!(ids(&second).len(), 3);
    // Page 1 ++ page 2 is exactly the top 6 — no gap, no overlap, no reordering.
    let joined: Vec<String> = ids(&first).into_iter().chain(ids(&second)).collect();
    assert_eq!(joined, ids(&both));
    assert_eq!(joined, vec!["d0", "d1", "d2", "d3", "d4", "d5"]);
}

#[test]
fn search_offset_past_the_result_set_is_an_empty_page() {
    let store = ranked_store();
    let q = [1.0, 0.0, 0.0];
    // Past the ranking entirely, and past it by exactly one — both are empty, not an error.
    assert!(
        store
            .search(&["docs"], &q, &page(5, 10))
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .search(&["docs"], &q, &page(5, 1_000))
            .unwrap()
            .is_empty()
    );
    assert_eq!(store.search(&["docs"], &q, &page(5, 9)).unwrap().len(), 1);
}

#[test]
fn search_with_the_default_offset_is_unchanged() {
    let store = ranked_store();
    let q = [1.0, 0.0, 0.0];
    let implicit = store.search(&["docs"], &q, &default_opts(4)).unwrap();
    let explicit = store.search(&["docs"], &q, &page(4, 0)).unwrap();
    assert_eq!(ids(&implicit), ids(&explicit));
    assert_eq!(ids(&implicit), vec!["d0", "d1", "d2", "d3"]);
}

#[test]
fn a_zero_top_k_page_is_empty_at_any_offset() {
    let store = ranked_store();
    let q = [1.0, 0.0, 0.0];
    assert!(store.search(&["docs"], &q, &page(0, 0)).unwrap().is_empty());
    assert!(store.search(&["docs"], &q, &page(0, 3)).unwrap().is_empty());
}

/// Every doc scores identically here, so only the tie-break decides the ranking — which is
/// exactly the case that made pagination incoherent before the total order.
#[test]
fn tied_scores_paginate_without_repeating_or_dropping_a_doc() {
    let mut store = Store::in_memory(3).unwrap();
    let recs: Vec<Record> = ["e", "c", "a", "f", "b", "d"]
        .iter()
        .map(|id| rec(id, vec![1.0, 0.0, 0.0]))
        .collect();
    store.upsert("docs", &recs).unwrap();
    let q = [1.0, 0.0, 0.0];

    let mut walked: Vec<String> = Vec::new();
    for offset in [0, 2, 4] {
        walked.extend(ids(&store.search(&["docs"], &q, &page(2, offset)).unwrap()));
    }
    assert_eq!(walked, vec!["a", "b", "c", "d", "e", "f"]);
}

/// The same query must give the same page every time, or a caller paging through a static
/// store sees documents move between pages.
#[test]
fn tied_scores_are_stable_across_repeated_searches() {
    let mut store = Store::in_memory(3).unwrap();
    let recs: Vec<Record> = (0..12)
        .map(|i| rec(&format!("t{i:02}"), vec![1.0, 0.0, 0.0]))
        .collect();
    store.upsert("docs", &recs).unwrap();
    let q = [1.0, 0.0, 0.0];

    let first = ids(&store.search(&["docs"], &q, &page(4, 4)).unwrap());
    assert_eq!(first, vec!["t04", "t05", "t06", "t07"]);
    for _ in 0..20 {
        assert_eq!(
            ids(&store.search(&["docs"], &q, &page(4, 4)).unwrap()),
            first
        );
    }
}

/// Ties across collections break on the collection name first, then the id.
#[test]
fn tied_scores_break_on_collection_before_id() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert("zeta", &[rec("a", vec![1.0, 0.0, 0.0])])
        .unwrap();
    store
        .upsert("alpha", &[rec("z", vec![1.0, 0.0, 0.0])])
        .unwrap();
    let hits = store
        .search(&["zeta", "alpha"], &[1.0, 0.0, 0.0], &page(2, 0))
        .unwrap();
    assert_eq!(hits[0].collection, "alpha");
    assert_eq!(hits[1].collection, "zeta");
}

#[test]
fn text_search_paginates() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    // Term frequency decreasing in i, so BM25 ranks t0 .. t5 in order.
    let recs: Vec<Record> = (0..6)
        .map(|i| doc(&format!("t{i}"), &"quantum ".repeat(6 - i)))
        .collect();
    store.upsert("docs", &recs).unwrap();

    let q = FtsQuery::new("body", "quantum");
    let first = store.text_search(&["docs"], &q, &page(2, 0)).unwrap();
    let second = store.text_search(&["docs"], &q, &page(2, 2)).unwrap();
    let both = store.text_search(&["docs"], &q, &page(4, 0)).unwrap();
    let joined: Vec<String> = ids(&first).into_iter().chain(ids(&second)).collect();
    assert_eq!(joined, ids(&both));
    assert!(
        store
            .text_search(&["docs"], &q, &page(2, 99))
            .unwrap()
            .is_empty()
    );
}

#[test]
// BM25's `ln` is non-deterministic under Miri by design, so a fused ranking can reorder by
// an ULP there — the same reason `hybrid_search_is_deterministic` is ignored.
#[cfg_attr(miri, ignore)]
fn hybrid_search_paginates_the_fused_ranking() {
    let (store, vector, text) = golden_fixture();
    let full = store
        .hybrid_search(
            &["docs"],
            &vector,
            &text,
            &HybridOpts {
                top_k: 4,
                ..Default::default()
            },
        )
        .unwrap();
    let second = store
        .hybrid_search(
            &["docs"],
            &vector,
            &text,
            &HybridOpts {
                top_k: 2,
                offset: 2,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(ids(&second), ids(&full)[2..4].to_vec());

    let past_the_end = store
        .hybrid_search(
            &["docs"],
            &vector,
            &text,
            &HybridOpts {
                top_k: 3,
                offset: 50,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(past_the_end.is_empty());
}

/// The page must be cut *after* the top-k cap. Deepening by `offset` is what makes page 2
/// exist at all: rank only `top_k` deep first and it would be empty.
#[test]
fn a_page_past_the_first_is_not_starved_by_the_top_k_cap() {
    let store = ranked_store();
    let hits = store
        .search(&["docs"], &[1.0, 0.0, 0.0], &page(2, 6))
        .unwrap();
    assert_eq!(ids(&hits), vec!["d6", "d7"]);
}

// ── Per-query exact search (nidus-m50.12) ───────────────────────────────────

/// `exact: true` on an ANN store must answer exactly what a store with no index answers —
/// same ids, same scores, same order — while leaving the index in place for other queries.
#[test]
#[cfg_attr(miri, ignore)] // N=2000 HNSW build is too slow under Miri.
fn exact_bypasses_the_ann_walk_and_matches_brute_force() {
    let (n, dim, k) = (2000, 32, 10);
    let data = random_unit_vectors(n, dim, 21);
    let queries = random_unit_vectors(20, dim, 22);
    let ann = ann_store(dim, AnnConfig::hnsw(), &data);
    let truth = exact_store(dim, &data);

    let forced = SearchOpts {
        top_k: k,
        exact: true,
        ..Default::default()
    };
    let mut approximate_ever_differed = false;
    for q in &queries {
        let want = truth.search(&["col"], q, &default_opts(k)).unwrap();
        assert_eq!(
            ann.search(&["col"], q, &forced).unwrap(),
            want,
            "exact: true must reproduce brute force"
        );
        approximate_ever_differed |= ann.search(&["col"], q, &default_opts(k)).unwrap() != want;
    }
    assert!(
        approximate_ever_differed,
        "the ANN path should differ somewhere, else this proves nothing"
    );
}

/// The quantized first pass is an approximation too, so `exact` must bypass it as well —
/// with a coarse binary code and no over-fetch, the default path visibly loses hits.
#[test]
fn exact_bypasses_the_quantized_first_pass() {
    let (dim, k) = (32, 10);
    let data = random_unit_vectors(120, dim, 23);
    let mut quantized = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .auto_compact(None)
            .quantization(Some(Quantization::binary().rescore(1))),
    )
    .unwrap();
    let recs: Vec<Record> = data
        .iter()
        .enumerate()
        .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
        .collect();
    quantized.upsert("col", &recs).unwrap();
    let truth = exact_store(dim, &data);

    let q = &random_unit_vectors(1, dim, 24)[0];
    let want = truth.search(&["col"], q, &default_opts(k)).unwrap();
    let forced = SearchOpts {
        top_k: k,
        exact: true,
        ..Default::default()
    };
    assert_eq!(quantized.search(&["col"], q, &forced).unwrap(), want);
    assert_ne!(
        quantized.search(&["col"], q, &default_opts(k)).unwrap(),
        want,
        "rescore(1) binary should be lossy, else the bypass proves nothing"
    );
}

/// `exact: false` is the default and must leave an indexed store on the index — asserted by
/// the ANN path answering identically whether the flag is omitted or spelled out.
#[test]
#[cfg_attr(miri, ignore)] // N=600 ANN build; ran >10min under Miri.
fn exact_false_is_the_untouched_approximate_path() {
    let (n, dim, k) = (600, 16, 5);
    let data = random_unit_vectors(n, dim, 25);
    let ann = ann_store(dim, AnnConfig::hnsw(), &data);
    let q = &random_unit_vectors(1, dim, 26)[0];
    let spelled_out = SearchOpts {
        top_k: k,
        exact: false,
        ..Default::default()
    };
    assert_eq!(
        ann.search(&["col"], q, &default_opts(k)).unwrap(),
        ann.search(&["col"], q, &spelled_out).unwrap()
    );
}

// ── Projection (nidus-m50.7) ────────────────────────────────────────────────

/// Three attrs, one of them a long body — the payload projection exists to leave behind.
fn projected_store() -> Store {
    let mut store = Store::in_memory(2).unwrap();
    let attrs = |id: &str| {
        BTreeMap::from([
            ("title".to_string(), Value::Str(format!("title of {id}"))),
            ("body".to_string(), Value::Str("x".repeat(4096))),
            ("lang".to_string(), Value::Str("rust".to_string())),
        ])
    };
    let recs: Vec<Record> = ["a", "b"]
        .iter()
        .enumerate()
        .map(|(i, id)| rec_with(id, vec![1.0, i as f32 * 0.1], attrs(id)))
        .collect();
    store.upsert("col", &recs).unwrap();
    store
}

fn attr_keys(hit: &Hit) -> Vec<&str> {
    hit.attrs.keys().map(String::as_str).collect()
}

#[test]
fn the_default_projection_returns_every_attr() {
    let store = projected_store();
    let hits = store
        .search(&["col"], &[1.0, 0.0], &default_opts(2))
        .unwrap();
    assert_eq!(attr_keys(&hits[0]), vec!["body", "lang", "title"]);
}

#[test]
fn include_returns_only_the_named_attrs() {
    let store = projected_store();
    let opts = SearchOpts {
        top_k: 2,
        projection: Projection::include(["title", "missing"]),
        ..Default::default()
    };
    let hits = store.search(&["col"], &[1.0, 0.0], &opts).unwrap();
    assert_eq!(hits.len(), 2);
    for hit in &hits {
        // A named attr the record lacks is simply absent — not an error, not a Null.
        assert_eq!(attr_keys(hit), vec!["title"]);
    }
}

#[test]
fn exclude_removes_only_the_named_attrs() {
    let store = projected_store();
    let opts = SearchOpts {
        top_k: 2,
        projection: Projection::exclude(["body"]),
        ..Default::default()
    };
    let hits = store.search(&["col"], &[1.0, 0.0], &opts).unwrap();
    assert_eq!(attr_keys(&hits[0]), vec!["lang", "title"]);
}

#[test]
fn projection_leaves_the_ranking_alone() {
    let store = projected_store();
    let ranked = |projection: Projection| {
        let opts = SearchOpts {
            top_k: 2,
            projection,
            ..Default::default()
        };
        let hits = store.search(&["col"], &[1.0, 0.0], &opts).unwrap();
        hits.into_iter()
            .map(|h| (h.id, h.score))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        ranked(Projection::All),
        ranked(Projection::include(["lang"]))
    );
    assert_eq!(
        ranked(Projection::All),
        ranked(Projection::exclude(["lang"]))
    );
}

#[test]
fn list_projects_attrs() {
    let store = projected_store();
    let listed = |projection: Projection| {
        store
            .list(
                &["col"],
                &ListOpts {
                    projection,
                    ..Default::default()
                },
            )
            .unwrap()
    };
    assert_eq!(
        attr_keys(&listed(Projection::All)[0]),
        vec!["body", "lang", "title"]
    );
    assert_eq!(
        attr_keys(&listed(Projection::include(["lang"]))[0]),
        vec!["lang"]
    );
    assert_eq!(
        attr_keys(&listed(Projection::exclude(["body", "lang"]))[0]),
        vec!["title"]
    );
}

#[test]
fn text_search_projects_attrs() {
    let mut store = projected_store();
    store
        .set_fts_schema("col", &[crate::FtsField::new("title")])
        .unwrap();
    let opts = SearchOpts {
        top_k: 2,
        projection: Projection::include(["lang"]),
        ..Default::default()
    };
    let hits = store
        .text_search(&["col"], &FtsQuery::new("title", "title"), &opts)
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(attr_keys(&hits[0]), vec!["lang"]);
}

/// Hits materialize on the index paths too, so projection has to reach the ANN walk's rerank
/// and the quantized two-pass tail — not only the brute-force scan.
#[test]
#[cfg_attr(miri, ignore)] // ANN + quantized build over 1024-dim rows; ran >10min under Miri.
fn projection_applies_on_the_ann_and_quantized_paths() {
    let dim = 16;
    let data = random_unit_vectors(300, dim, 27);
    let attrs = BTreeMap::from([
        ("keep".to_string(), Value::Int(1)),
        ("drop".to_string(), Value::Str("x".repeat(1024))),
    ]);
    let build = |store: &mut Store| {
        let recs: Vec<Record> = data
            .iter()
            .enumerate()
            .map(|(i, v)| rec_with(&format!("d{i}"), v.clone(), attrs.clone()))
            .collect();
        store.upsert("col", &recs).unwrap();
    };
    let mut ann = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .auto_compact(None)
            .ann(Some(AnnConfig::hnsw())),
    )
    .unwrap();
    build(&mut ann);
    let mut quantized = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", dim)
            .auto_compact(None)
            .quantization(Some(Quantization::default())),
    )
    .unwrap();
    build(&mut quantized);

    let q = &random_unit_vectors(1, dim, 28)[0];
    let opts = SearchOpts {
        top_k: 5,
        projection: Projection::exclude(["drop"]),
        ..Default::default()
    };
    for store in [&ann, &quantized] {
        let hits = store.search(&["col"], q, &opts).unwrap();
        assert_eq!(hits.len(), 5);
        for hit in &hits {
            assert_eq!(attr_keys(hit), vec!["keep"]);
        }
    }
}

// ── Multi-clause BM25 + result annotations (nidus-m50.10, nidus-m50.5) ───────

use crate::annotate::HighlightOpts;
use crate::model::{FtsClause, FtsCombine};

/// A text-only doc with both indexed fields populated.
fn titled(id: &str, title: &str, body: &str) -> Record {
    let mut attrs = BTreeMap::new();
    attrs.insert("title".to_string(), Value::Str(title.to_string()));
    attrs.insert("body".to_string(), Value::Str(body.to_string()));
    Record::text_only(id, attrs)
}

/// A store indexing `title` and `body` with length normalization off, so the fixtures'
/// scores depend on term frequency alone.
fn two_field_store(docs: &[Record]) -> Store {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema(
            "docs",
            &[FtsField::new("title").b(0.0), FtsField::new("body").b(0.0)],
        )
        .unwrap();
    store.upsert("docs", docs).unwrap();
    store
}

/// The Sum/Max fixture: `spread` matches both fields weakly, `focused` matches only `body`
/// but strongly, `filler` matches only `title`.
fn combine_fixture() -> Store {
    two_field_store(&[
        titled("spread", "needle", "needle"),
        titled("focused", "alpha", "needle needle needle needle"),
        titled("filler", "needle", "gamma"),
    ])
}

fn hit_ids(hits: &[Hit]) -> Vec<&str> {
    hits.iter().map(|h| h.id.as_str()).collect()
}

#[test]
fn a_clause_per_field_searches_both_at_once() {
    let store = two_field_store(&[
        titled("a", "rust vector store", "unrelated prose"),
        titled("b", "unrelated heading", "an embedded vector store"),
        titled("c", "nothing here", "nothing there"),
    ]);
    let q = FtsQuery::multi([
        FtsClause::new("title", "rust"),
        FtsClause::new("body", "embedded"),
    ]);
    let hits = store.text_search(&["docs"], &q, &default_opts(10)).unwrap();
    // Each doc is found through a different clause; neither could find both alone.
    let mut found = hit_ids(&hits);
    found.sort_unstable();
    assert_eq!(found, vec!["a", "b"]);
}

#[test]
fn sum_and_max_rank_the_same_corpus_differently() {
    let store = combine_fixture();
    let q = || {
        FtsQuery::multi([
            FtsClause::new("title", "needle"),
            FtsClause::new("body", "needle"),
        ])
    };
    let sum = store
        .text_search(&["docs"], &q(), &default_opts(10))
        .unwrap();
    // Sum adds both clauses, so matching two fields beats matching one field harder.
    assert_eq!(hit_ids(&sum), vec!["spread", "focused", "filler"]);

    let max = store
        .text_search(&["docs"], &q().combine(FtsCombine::Max), &default_opts(10))
        .unwrap();
    // Max takes the strongest single clause, which flips the top of the ranking.
    assert_eq!(max[0].id, "focused");
    assert!(max[0].score < sum[0].score, "{max:?} vs {sum:?}");
}

#[test]
// The bit-exactness this asserts is the point, so it is not loosened to a tolerance.
// BM25's `idf` calls `ln`, which Miri evaluates non-deterministically; real float
// semantics make these exact, so the guarantee is enforced on the native run.
#[cfg_attr(miri, ignore)] // asserts exact BM25 score bits; Miri's `ln` differs from host libm by 1 ULP.
fn a_single_clause_scores_exactly_as_the_one_field_query_always_did() {
    let store = two_field_store(&[
        titled("d1", "t", "the cat sat on the mat"),
        titled("d2", "t", "cats and more cats running with cats"),
        titled("d3", "t", "a dog barked"),
    ]);
    let run = |q: FtsQuery| store.text_search(&["docs"], &q, &default_opts(10)).unwrap();
    let baseline = run(FtsQuery::new("body", "cat"));
    assert_eq!(hit_ids(&baseline), vec!["d2", "d1"]);
    // The multi-clause machinery must not perturb a one-clause query under either fold.
    for combine in [FtsCombine::Sum, FtsCombine::Max] {
        let hits = run(FtsQuery::multi([FtsClause::new("body", "cat")]).combine(combine));
        assert_eq!(hits, baseline, "{combine:?}");
    }
}

#[test]
fn a_clause_list_must_not_be_empty() {
    let store = combine_fixture();
    let empty = FtsQuery::multi([]);
    assert!(
        store
            .text_search(&["docs"], &empty, &default_opts(10))
            .is_err()
    );
    assert!(
        store
            .hybrid_search(&["docs"], &[0.0, 0.0, 0.0], &empty, &HybridOpts::default())
            .is_err()
    );
}

#[test]
fn a_clause_naming_an_unindexed_field_contributes_nothing() {
    let store = combine_fixture();
    let q = FtsQuery::multi([
        FtsClause::new("title", "needle"),
        FtsClause::new("author", "needle"),
    ]);
    let hits = store.text_search(&["docs"], &q, &default_opts(10)).unwrap();
    let mut found = hit_ids(&hits);
    found.sort_unstable();
    assert_eq!(found, vec!["filler", "spread"]);
}

#[test]
fn hybrid_fuses_a_vector_leg_with_several_text_clauses() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("title"), FtsField::new("body")])
        .unwrap();
    let rec = |id: &str, v: Vec<f32>, title: &str, body: &str| {
        let mut r = titled(id, title, body);
        Record::new(id, v, std::mem::take(&mut r.attrs))
    };
    store
        .upsert(
            "docs",
            &[
                rec("vec", vec![1.0, 0.0, 0.0], "nothing", "nothing"),
                rec("title-hit", vec![0.0, 1.0, 0.0], "quantum", "nothing"),
                rec("body-hit", vec![0.0, 0.0, 1.0], "nothing", "photon physics"),
            ],
        )
        .unwrap();
    let q = FtsQuery::multi([
        FtsClause::new("title", "quantum"),
        FtsClause::new("body", "photon"),
    ]);
    let hits = store
        .hybrid_search(&["docs"], &[1.0, 0.0, 0.0], &q, &HybridOpts::default())
        .unwrap();
    let mut found = hit_ids(&hits);
    found.sort_unstable();
    // The vector leg carries `vec`; the two clauses each carry one more document.
    assert_eq!(found, vec!["body-hit", "title-hit", "vec"]);
}

#[test]
fn explain_reports_each_matched_clauses_own_score() {
    let store = combine_fixture();
    let q = FtsQuery::multi([
        FtsClause::new("title", "needle"),
        FtsClause::new("body", "needle"),
    ]);
    let plain = store.text_search(&["docs"], &q, &default_opts(10)).unwrap();
    assert!(
        plain.iter().all(|h| h.annotations.is_none()),
        "annotations must stay opt-in"
    );

    let opts = SearchOpts {
        top_k: 10,
        explain: true,
        ..Default::default()
    };
    let hits = store.text_search(&["docs"], &q, &opts).unwrap();
    let by_id = |id: &str| {
        hits.iter()
            .find(|h| h.id == id)
            .unwrap()
            .annotations
            .clone()
            .unwrap()
    };
    let spread = by_id("spread");
    assert_eq!(
        spread
            .clauses
            .iter()
            .map(|c| c.field.as_str())
            .collect::<Vec<_>>(),
        vec!["title", "body"],
        "matched clauses are reported in query order"
    );
    // The parts must add up to the fused score the hit carries.
    let total: f32 = spread.clauses.iter().map(|c| c.score).sum();
    let hit_score = hits.iter().find(|h| h.id == "spread").unwrap().score;
    assert!((total - hit_score).abs() < 1e-6);
    // A doc that matched one clause reports one clause, not a zero row for the other.
    assert_eq!(by_id("focused").clauses.len(), 1);
    assert_eq!(by_id("focused").clauses[0].field, "body");
}

#[test]
fn explain_reports_each_legs_rank_and_score_on_a_hybrid_hit() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    let mut attrs = BTreeMap::new();
    attrs.insert("body".to_string(), Value::Str("quantum physics".into()));
    store
        .upsert(
            "docs",
            &[
                Record::new("both", vec![1.0, 0.0, 0.0], attrs.clone()),
                Record::new("vec-only", vec![0.9, 0.1, 0.0], BTreeMap::new()),
            ],
        )
        .unwrap();
    let q = FtsQuery::new("body", "quantum");
    let opts = HybridOpts {
        explain: true,
        ..Default::default()
    };
    let hits = store
        .hybrid_search(&["docs"], &[1.0, 0.0, 0.0], &q, &opts)
        .unwrap();
    let both = hits.iter().find(|h| h.id == "both").unwrap();
    let a = both.annotations.clone().unwrap();
    let vector = a.vector.expect("vector leg reported");
    let text = a.text.expect("text leg reported");
    assert_eq!(vector.rank, 0);
    assert!((vector.score - 1.0).abs() < 1e-6, "{vector:?}");
    assert_eq!(text.rank, 0);
    assert_eq!(a.clauses.len(), 1);
    assert!((text.score - a.clauses[0].score).abs() < 1e-6);

    // A doc only the vector leg returned reports that leg alone.
    let vec_only = hits.iter().find(|h| h.id == "vec-only").unwrap();
    let a = vec_only.annotations.clone().unwrap();
    assert!(a.vector.is_some() && a.text.is_none() && a.clauses.is_empty());

    // And without `explain` the hits carry nothing extra.
    let plain = store
        .hybrid_search(&["docs"], &[1.0, 0.0, 0.0], &q, &HybridOpts::default())
        .unwrap();
    assert!(plain.iter().all(|h| h.annotations.is_none()));
}

#[test]
fn highlight_spans_index_the_original_text_across_the_stemmer() {
    let store = two_field_store(&[
        titled("a", "t", "The engineers were running experiments overnight"),
        titled("b", "t", "We run the suite nightly"),
    ]);
    // "running" (query) and "run" (document) share a stem, and vice versa: the span must
    // cover the *document's* spelling either way, which no substring search would find.
    for (query, want) in [("running", "run"), ("run", "running")] {
        let q = FtsQuery::new("body", query).highlight(HighlightOpts::default());
        let hits = store.text_search(&["docs"], &q, &default_opts(10)).unwrap();
        let hit = hits
            .iter()
            .find(|h| h.id == if want == "run" { "b" } else { "a" })
            .unwrap();
        let hl = &hit.annotations.as_ref().unwrap().highlights;
        assert_eq!(hl.len(), 1);
        assert_eq!(hl[0].field, "body");
        let frag = &hl[0].fragments[0];
        let marked: Vec<&str> = frag.spans.iter().map(|&(s, e)| &frag.text[s..e]).collect();
        assert_eq!(marked, vec![want], "query {query:?}");
    }
}

#[test]
fn highlighting_reads_the_stored_text_even_when_projection_drops_the_field() {
    // The combination the feature exists for: drop the 10 KB body from the payload and keep
    // only the snippet that explains the match (nidus-m50.5).
    let store = two_field_store(&[titled("a", "heading", "the engineers were running late")]);
    let q = FtsQuery::new("body", "run").highlight(HighlightOpts::default());
    let opts = SearchOpts {
        top_k: 10,
        projection: Projection::include(["title"]),
        ..Default::default()
    };
    let hits = store.text_search(&["docs"], &q, &opts).unwrap();
    assert_eq!(
        attr_keys(&hits[0]),
        vec!["title"],
        "body was projected away"
    );
    let hl = &hits[0].annotations.as_ref().unwrap().highlights;
    assert!(hl[0].fragments[0].text.contains("running"));

    // Excluding it is the same story.
    let opts = SearchOpts {
        top_k: 10,
        projection: Projection::exclude(["body"]),
        ..Default::default()
    };
    let hits = store.text_search(&["docs"], &q, &opts).unwrap();
    assert_eq!(attr_keys(&hits[0]), vec!["title"]);
    assert!(!hits[0].annotations.as_ref().unwrap().highlights.is_empty());
}

/// nidus-lvo.4's projection/highlighting criterion, both at once: a text search over a
/// chunked corpus that projects the body away must still highlight the match AND still widen
/// the hit, since both read the stored record rather than the payload.
#[test]
fn expansion_and_highlighting_coexist_over_a_projected_away_body() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body").b(0.0)])
        .unwrap();
    let chunk = |i: i64, start: i64, body: &str| {
        let mut attrs = BTreeMap::new();
        attrs.insert("body".to_string(), Value::Str(body.to_string()));
        attrs.insert(
            crate::model::META_TEXT.to_string(),
            Value::Str(body.to_string()),
        );
        attrs.insert(
            crate::model::META_PARENT_ID.to_string(),
            Value::Str("doc".to_string()),
        );
        attrs.insert(crate::model::META_CHUNK_INDEX.to_string(), Value::Int(i));
        attrs.insert(crate::model::META_CHAR_START.to_string(), Value::Int(start));
        Record::text_only(format!("doc#{i}"), attrs)
    };
    store
        .upsert(
            "docs",
            &[
                chunk(0, 0, "the engineers were"),
                chunk(1, 4, "engineers were running late"),
            ],
        )
        .unwrap();

    let q = FtsQuery::new("body", "run").highlight(HighlightOpts::default());
    let opts = SearchOpts {
        top_k: 10,
        projection: Projection::exclude(["body", crate::model::META_TEXT]),
        expand: Some(crate::Expand::new(1)),
        ..Default::default()
    };
    let hits = store.text_search(&["docs"], &q, &opts).unwrap();
    let hit = hits.iter().find(|h| h.id == "doc#1").expect("chunk 1 hit");
    assert!(!hit.attrs.contains_key("body"), "body was projected away");
    let hl = &hit.annotations.as_ref().unwrap().highlights;
    assert!(
        hl[0].fragments[0].text.contains("running"),
        "the fragment still comes from the stored text"
    );
    assert_eq!(
        hit.context.as_deref(),
        Some("the engineers were running late"),
        "and the window is still stitched from the stored chunks"
    );
}

#[test]
fn highlighting_covers_every_matched_clause_and_survives_fusion() {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("title"), FtsField::new("body")])
        .unwrap();
    let mut attrs = BTreeMap::new();
    attrs.insert("title".to_string(), Value::Str("running order".into()));
    attrs.insert(
        "body".to_string(),
        Value::Str("the tests all passed".into()),
    );
    store
        .upsert("docs", &[Record::new("a", vec![1.0, 0.0, 0.0], attrs)])
        .unwrap();
    let q = FtsQuery::multi([
        FtsClause::new("title", "run"),
        FtsClause::new("body", "test"),
    ])
    .highlight(HighlightOpts::default());

    let hits = store.text_search(&["docs"], &q, &default_opts(10)).unwrap();
    let fields: Vec<&str> = hits[0]
        .annotations
        .as_ref()
        .unwrap()
        .highlights
        .iter()
        .map(|h| h.field.as_str())
        .collect();
    assert_eq!(fields, vec!["title", "body"]);

    // Fusion rebuilds the hit, so highlighting is applied to the fused page, not the leg.
    let hits = store
        .hybrid_search(&["docs"], &[1.0, 0.0, 0.0], &q, &HybridOpts::default())
        .unwrap();
    assert_eq!(hits[0].annotations.as_ref().unwrap().highlights.len(), 2);
}

// ── Ranking expressions (nidus-m50.3) ─────────────────────────────────────

use crate::model::{AggregateOpts, Aggregation, Decay, LimitPer, OrderBy, RankBy};

const DAY: i64 = 86_400_000;

/// `n` days before `origin`, as an epoch-millis `DateTime`.
fn days_ago(origin: i64, n: i64) -> Value {
    Value::DateTime(origin - n * DAY)
}

fn stamped(id: &str, vector: Vec<f32>, ts: Value) -> Record {
    rec_with(id, vector, BTreeMap::from([("ts".to_string(), ts)]))
}

/// A store where every doc shares one vector, so the base score is identical and any
/// reordering can only have come from the ranking expression.
fn decay_store(origin: i64) -> Store {
    let mut store = Store::in_memory(3).unwrap();
    store
        .upsert(
            "docs",
            &[
                stamped("fresh", vec![1.0, 0.0, 0.0], days_ago(origin, 0)),
                stamped("week", vec![1.0, 0.0, 0.0], days_ago(origin, 7)),
                stamped("year", vec![1.0, 0.0, 0.0], days_ago(origin, 365)),
            ],
        )
        .unwrap();
    store
}

fn decayed(origin: i64, lambda: f32) -> SearchOpts {
    SearchOpts {
        top_k: 10,
        rank_by: Some(RankBy::Decay(
            Decay::new("ts", origin, 7 * DAY).lambda(lambda),
        )),
        ..Default::default()
    }
}

#[test]
fn decay_reorders_by_age_and_subtracts_lambda_times_one_minus_factor() {
    let origin = 2_000 * DAY;
    let store = decay_store(origin);
    let hits = store
        .search(&["docs"], &[1.0, 0.0, 0.0], &decayed(origin, 0.4))
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, vec!["fresh", "week", "year"], "newest first");

    // base = 1.0 (identical unit vectors), and the formula is
    // base − lambda × (1 − decay^(age/scale)): age 0 → factor 1 → no penalty,
    // age == scale → factor 0.5 → penalty lambda/2.
    assert!((hits[0].score - 1.0).abs() < 1e-5, "{}", hits[0].score);
    assert!(
        (hits[1].score - (1.0 - 0.4 * 0.5)).abs() < 1e-5,
        "{}",
        hits[1].score
    );
    let year_factor = 0.5f32.powf(365.0 / 7.0);
    assert!(
        (hits[2].score - (1.0 - 0.4 * (1.0 - year_factor))).abs() < 1e-5,
        "{}",
        hits[2].score
    );
}

#[test]
fn decay_off_by_default_leaves_the_ranking_and_the_scores_alone() {
    let origin = 2_000 * DAY;
    let store = decay_store(origin);
    let plain = store
        .search(&["docs"], &[1.0, 0.0, 0.0], &default_opts(10))
        .unwrap();
    assert_eq!(plain.len(), 3);
    for hit in &plain {
        assert!((hit.score - 1.0).abs() < 1e-6, "base score untouched");
    }
    let ids: Vec<&str> = plain.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, vec!["fresh", "week", "year"], "tie-break unchanged");
}

#[test]
fn a_record_with_no_timestamp_is_not_penalized() {
    let origin = 2_000 * DAY;
    let mut store = decay_store(origin);
    store
        .upsert("docs", &[rec("undated", vec![1.0, 0.0, 0.0])])
        .unwrap();
    let hits = store
        .search(&["docs"], &[1.0, 0.0, 0.0], &decayed(origin, 0.9))
        .unwrap();
    let undated = hits.iter().find(|h| h.id == "undated").unwrap();
    assert!((undated.score - 1.0).abs() < 1e-5, "{}", undated.score);
    // It therefore outranks the aged docs instead of being buried beneath them.
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids[..2], ["fresh", "undated"], "{ids:?}");
}

#[test]
fn a_missing_timestamp_can_be_opted_into_a_penalty() {
    let origin = 2_000 * DAY;
    let mut store = decay_store(origin);
    store
        .upsert("docs", &[rec("undated", vec![1.0, 0.0, 0.0])])
        .unwrap();
    let opts = SearchOpts {
        top_k: 10,
        rank_by: Some(RankBy::Decay(
            Decay::new("ts", origin, 7 * DAY).lambda(0.5).missing(0.0),
        )),
        ..Default::default()
    };
    let hits = store.search(&["docs"], &[1.0, 0.0, 0.0], &opts).unwrap();
    let undated = hits.iter().find(|h| h.id == "undated").unwrap();
    assert!((undated.score - 0.5).abs() < 1e-5, "{}", undated.score);
}

/// The whole point of subtracting: Euclidean scores in (−∞, 0] and dot product anywhere at
/// all, and one penalty is valid for both. A multiplied factor would need a Cosine-only clamp.
#[test]
fn decay_works_under_euclidean_and_dot_product() {
    let origin = 2_000 * DAY;
    for distance in [Distance::Euclidean, Distance::DotProduct] {
        let mut store = Store::in_memory_with(3, distance).unwrap();
        store
            .upsert(
                "docs",
                &[
                    stamped("fresh", vec![2.0, 0.0, 0.0], days_ago(origin, 0)),
                    stamped("week", vec![2.0, 0.0, 0.0], days_ago(origin, 7)),
                ],
            )
            .unwrap();
        let base = store
            .search(&["docs"], &[2.0, 0.0, 0.0], &default_opts(10))
            .unwrap();
        assert!(
            (base[0].score - base[1].score).abs() < 1e-6,
            "{distance:?}: identical vectors must tie before decay"
        );
        let hits = store
            .search(&["docs"], &[2.0, 0.0, 0.0], &decayed(origin, 1.0))
            .unwrap();
        assert_eq!(hits[0].id, "fresh", "{distance:?}");
        assert!(
            (hits[1].score - (base[1].score - 0.5)).abs() < 1e-4,
            "{distance:?}: one half-life costs lambda/2, got {}",
            hits[1].score
        );
    }
}

/// nidus-m50.15 #9: `rank_by` does not force the exact path — it applies over an ANN result
/// set, inheriting that path's approximation rather than silently disabling the index.
#[test]
fn decay_applies_over_an_ann_result_set_without_forcing_exact() {
    let origin = 2_000 * DAY;
    let mut store = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", 3)
            .auto_compact(None)
            .ann(Some(AnnConfig::hnsw())),
    )
    .unwrap();
    store
        .upsert(
            "docs",
            &[
                stamped("fresh", vec![1.0, 0.0, 0.0], days_ago(origin, 0)),
                stamped("week", vec![1.0, 0.0, 0.0], days_ago(origin, 7)),
            ],
        )
        .unwrap();
    let hits = store
        .search(&["docs"], &[1.0, 0.0, 0.0], &decayed(origin, 0.4))
        .unwrap();
    assert_eq!(hits[0].id, "fresh");
    assert!((hits[1].score - (1.0 - 0.4 * 0.5)).abs() < 1e-5, "{hits:?}");
}

#[test]
fn decay_applies_over_the_quantized_two_pass_search() {
    let origin = 2_000 * DAY;
    let mut store = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", 3)
            .auto_compact(None)
            .quantization(Some(Quantization::default())),
    )
    .unwrap();
    store
        .upsert(
            "docs",
            &[
                stamped("fresh", vec![1.0, 0.0, 0.0], days_ago(origin, 0)),
                stamped("week", vec![1.0, 0.0, 0.0], days_ago(origin, 7)),
            ],
        )
        .unwrap();
    let hits = store
        .search(&["docs"], &[1.0, 0.0, 0.0], &decayed(origin, 0.4))
        .unwrap();
    assert_eq!(hits[0].id, "fresh");
    assert!((hits[1].score - (1.0 - 0.4 * 0.5)).abs() < 1e-5, "{hits:?}");
}

#[test]
fn min_score_gates_the_decayed_score_not_the_base_one() {
    let origin = 2_000 * DAY;
    let store = decay_store(origin);
    let opts = SearchOpts {
        min_score: Some(0.9),
        ..decayed(origin, 0.4)
    };
    let ids: Vec<String> = store
        .search(&["docs"], &[1.0, 0.0, 0.0], &opts)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    // Every base score is 1.0; only `fresh` still clears 0.9 once the penalty lands.
    assert_eq!(ids, vec!["fresh"]);
}

#[test]
fn a_degenerate_ranking_expression_is_refused() {
    let store = decay_store(0);
    for bad in [
        Decay::new("ts", 0, 0),
        Decay::new("ts", 0, DAY).decay(1.0),
        Decay::new("ts", 0, DAY).lambda(f32::NAN),
        Decay::new("", 0, DAY),
    ] {
        let opts = SearchOpts {
            top_k: 3,
            rank_by: Some(RankBy::Decay(bad.clone())),
            ..Default::default()
        };
        assert!(
            store.search(&["docs"], &[1.0, 0.0, 0.0], &opts).is_err(),
            "{bad:?} must be refused"
        );
    }
}

// ── ORDER BY with no vector query (nidus-m50.3) ───────────────────────────

fn ordered_store() -> Store {
    let mut store = Store::in_memory(2).unwrap();
    let with_n = |id: &str, n: i64| {
        rec_with(
            id,
            vec![1.0, 0.0],
            BTreeMap::from([("n".to_string(), Value::Int(n))]),
        )
    };
    store
        .upsert("docs", &[with_n("b", 2), with_n("c", 3), with_n("a", 1)])
        .unwrap();
    store
}

fn listed(store: &Store, order: Option<OrderBy>) -> Vec<String> {
    store
        .list(
            &["docs"],
            &ListOpts {
                order_by: order,
                ..Default::default()
            },
        )
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect()
}

#[test]
fn order_by_sorts_a_plain_attribute_both_ways() {
    let store = ordered_store();
    assert_eq!(listed(&store, Some(OrderBy::asc("n"))), ["a", "b", "c"]);
    assert_eq!(listed(&store, Some(OrderBy::desc("n"))), ["c", "b", "a"]);
    // No `order_by` keeps the storage order the upsert produced.
    assert_eq!(listed(&store, None), ["b", "c", "a"]);
}

#[test]
fn order_by_sorts_strings_and_datetimes_too() {
    let mut store = Store::in_memory(2).unwrap();
    let with =
        |id: &str, v: Value| rec_with(id, vec![1.0, 0.0], BTreeMap::from([("k".to_string(), v)]));
    store
        .upsert(
            "docs",
            &[
                with("mid", Value::Str("m".into())),
                with("low", Value::Str("a".into())),
                with("high", Value::Str("z".into())),
                with("older", Value::DateTime(1)),
            ],
        )
        .unwrap();
    let asc = listed(&store, Some(OrderBy::asc("k")));
    assert_eq!(&asc[..3], ["low", "mid", "high"], "{asc:?}");
    assert_eq!(asc[3], "older", "a DateTime does not order against a Str");
}

/// nidus-m50.15 #10: values that do not order against the witness type — a different
/// variant, an unorderable one, or an absent attribute — land in ONE trailing bucket, and
/// stay trailing when the sort is reversed.
#[test]
fn order_by_puts_cross_type_and_missing_values_in_a_trailing_bucket() {
    let mut store = ordered_store();
    store
        .upsert(
            "docs",
            &[
                rec_with(
                    "str",
                    vec![1.0, 0.0],
                    BTreeMap::from([("n".to_string(), Value::Str("zzz".into()))]),
                ),
                rec_with(
                    "null",
                    vec![1.0, 0.0],
                    BTreeMap::from([("n".to_string(), Value::Null)]),
                ),
                rec("absent", vec![1.0, 0.0]),
            ],
        )
        .unwrap();
    let trailing = ["str", "null", "absent"];
    let asc = listed(&store, Some(OrderBy::asc("n")));
    assert_eq!(&asc[..3], ["a", "b", "c"], "{asc:?}");
    for id in trailing {
        assert!(asc[3..].contains(&id.to_string()), "{asc:?}");
    }
    let desc = listed(&store, Some(OrderBy::desc("n")));
    assert_eq!(&desc[..3], ["c", "b", "a"], "{desc:?}");
    for id in trailing {
        assert!(
            desc[3..].contains(&id.to_string()),
            "reversing must not promote the trailing bucket: {desc:?}"
        );
    }
}

#[test]
fn order_by_an_attribute_nobody_has_keeps_the_storage_order() {
    let store = ordered_store();
    assert_eq!(listed(&store, Some(OrderBy::asc("nope"))), ["b", "c", "a"]);
}

#[test]
fn order_by_runs_before_the_page_is_cut() {
    let store = ordered_store();
    let page = store
        .list(
            &["docs"],
            &ListOpts {
                offset: 1,
                limit: 1,
                order_by: Some(OrderBy::asc("n")),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(page.len(), 1);
    assert_eq!(page[0].id, "b", "the second row of the SORTED set");
}

// ── Per-leg hybrid weights (nidus-m50.3) ──────────────────────────────────

fn weighted_hybrid_store() -> Store {
    let mut store = Store::in_memory(3).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    // One doc per leg and nothing in both, so a weight decides the winner outright.
    let mut vecdoc = Record::new("vecdoc", vec![1.0, 0.0, 0.0], BTreeMap::new());
    vecdoc
        .attrs
        .insert("body".to_string(), Value::Str("nothing here".to_string()));
    store
        .upsert("docs", &[vecdoc, doc("textdoc", "quantum physics")])
        .unwrap();
    store
}

fn fused(store: &Store, opts: HybridOpts) -> Vec<Hit> {
    store
        .hybrid_search(
            &["docs"],
            &[1.0, 0.0, 0.0],
            &FtsQuery::new("body", "quantum physics"),
            &opts,
        )
        .unwrap()
}

fn weighted(vector_weight: f32, text_weight: f32) -> HybridOpts {
    HybridOpts {
        top_k: 10,
        vector_weight,
        text_weight,
        ..Default::default()
    }
}

#[test]
fn both_leg_weights_at_one_are_identical_to_the_unweighted_fusion() {
    let store = weighted_hybrid_store();
    let default = fused(
        &store,
        HybridOpts {
            top_k: 10,
            ..Default::default()
        },
    );
    // Bit-identical, not approximately equal: multiplying by 1.0 is exact.
    assert_eq!(fused(&store, weighted(1.0, 1.0)), default);
}

#[test]
fn a_leg_weight_shifts_which_leg_wins_the_fusion() {
    let store = weighted_hybrid_store();
    assert_eq!(fused(&store, weighted(4.0, 1.0))[0].id, "vecdoc");
    assert_eq!(fused(&store, weighted(1.0, 4.0))[0].id, "textdoc");
    // The winner's score is exactly the weighted reciprocal rank of its leading leg.
    let heavy = fused(&store, weighted(4.0, 1.0));
    assert!(
        (heavy[0].score - 4.0 / 61.0).abs() < 1e-6,
        "{}",
        heavy[0].score
    );
}

#[test]
fn a_poisonous_leg_weight_is_refused() {
    let store = weighted_hybrid_store();
    for (v, t) in [(f32::NAN, 1.0), (1.0, -1.0), (f32::INFINITY, 1.0)] {
        assert!(
            store
                .hybrid_search(
                    &["docs"],
                    &[1.0, 0.0, 0.0],
                    &FtsQuery::new("body", "quantum"),
                    &weighted(v, t)
                )
                .is_err(),
            "({v}, {t}) must be refused"
        );
    }
}

// ── Aggregation (nidus-m50.6) ─────────────────────────────────────────────

fn agg_store() -> Store {
    let mut store = Store::in_memory(2).unwrap();
    let doc = |id: &str, kind: &str, bytes: Value| {
        rec_with(
            id,
            vec![1.0, 0.0],
            BTreeMap::from([
                ("kind".to_string(), Value::Str(kind.to_string())),
                ("bytes".to_string(), bytes),
            ]),
        )
    };
    store
        .upsert(
            "docs",
            &[
                doc("a", "note", Value::Int(10)),
                doc("b", "note", Value::Int(32)),
                doc("c", "code", Value::Int(5)),
                rec("d", vec![0.0, 1.0]),
            ],
        )
        .unwrap();
    store
}

fn agg(store: &Store, filter: Filter, sum: &[&str]) -> Aggregation {
    store.aggregate(
        &["docs"],
        &AggregateOpts {
            filter,
            sum: sum.iter().map(|s| s.to_string()).collect(),
            group_by: None,
        },
    )
}

fn agg_by(store: &Store, field: &str, sum: &[&str]) -> Aggregation {
    store.aggregate(
        &["docs"],
        &AggregateOpts {
            filter: Filter::default(),
            sum: sum.iter().map(|s| s.to_string()).collect(),
            group_by: Some(field.to_string()),
        },
    )
}

/// `group_by` splits the same pass into per-value rows while the whole-scope totals stay put,
/// so a caller gets both without a second query (nidus-bmh).
#[test]
fn group_by_reports_a_row_per_distinct_value_beside_the_totals() {
    let store = agg_store();
    let out = agg_by(&store, "kind", &["bytes"]);

    assert_eq!(out.count, 4, "the totals still cover every record");
    assert_eq!(out.sums["bytes"], Value::Int(47));

    // Ordered by count descending; `note` holds two of the four records.
    let kinds: Vec<Option<&Value>> = out.groups.iter().map(|g| g.value.as_ref()).collect();
    assert_eq!(kinds[0], Some(&Value::Str("note".into())));
    assert_eq!(out.groups[0].count, 2);
    assert_eq!(out.groups[0].sums["bytes"], Value::Int(42));

    // Every group's sum covers only its own records, and they add back up to the total.
    let regrouped: i64 = out
        .groups
        .iter()
        .map(|g| match g.sums["bytes"] {
            Value::Int(n) => n,
            ref other => panic!("expected Int, got {other:?}"),
        })
        .sum();
    assert_eq!(regrouped, 47);
    assert!(!out.groups_truncated);
}

/// A record with no `kind` at all forms its own group with a `None` value — it is not folded
/// into `Null` and not silently dropped, so the group counts still sum to the total.
#[test]
fn a_record_missing_the_group_attribute_forms_its_own_group() {
    let store = agg_store();
    let out = agg_by(&store, "kind", &["bytes"]);

    let missing = out
        .groups
        .iter()
        .find(|g| g.value.is_none())
        .expect("the attribute-less record must be represented");
    assert_eq!(missing.count, 1);
    // It carries no `bytes` either, and a skipped addend is not a zero addend.
    assert_eq!(missing.sums["bytes"], Value::Int(0));

    let total: u64 = out.groups.iter().map(|g| g.count).sum();
    assert_eq!(total, out.count, "every record lands in exactly one group");
}

/// Grouping on an attribute nothing carries is answerable, not an error: one group holding
/// everything. The empty *name* is the caller mistake, and that is rejected in `Nidus`.
#[test]
fn grouping_on_an_unknown_attribute_yields_one_missing_group() {
    let store = agg_store();
    let out = agg_by(&store, "nope", &[]);
    assert_eq!(out.groups.len(), 1);
    assert_eq!(out.groups[0].value, None);
    assert_eq!(out.groups[0].count, 4);
}

/// Past MAX_GROUPS the answer says so. A short list of groups is indistinguishable from a
/// complete one, which is the whole reason the flag exists rather than a silent truncation.
/// Ignored under Miri only for its size — 10k records is minutes there, milliseconds natively.
#[cfg_attr(miri, ignore)] // upserts MAX_GROUPS+1 = 10_001 records; ran >10min under Miri.
#[test]
fn outrunning_the_group_cap_is_reported_not_hidden() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("docs").unwrap();
    let recs: Vec<Record> = (0..super::aggregate::MAX_GROUPS + 1)
        .map(|i| {
            Record::new(
                format!("r{i}"),
                vec![1.0, 0.0],
                BTreeMap::from([("k".to_string(), Value::Int(i as i64))]),
            )
        })
        .collect();
    store.upsert("docs", &recs).unwrap();

    let out = store.aggregate(
        &["docs"],
        &AggregateOpts {
            filter: Filter::default(),
            sum: Vec::new(),
            group_by: Some("k".to_string()),
        },
    );
    assert!(out.groups_truncated, "the cap was hit and must be reported");
    assert_eq!(out.groups.len(), super::aggregate::MAX_GROUPS);
    // The totals are still exact: truncation drops GROUPS, never records from the count.
    assert_eq!(out.count as usize, super::aggregate::MAX_GROUPS + 1);
}

/// Two calls over one store must agree exactly. `HashMap` iteration order is deliberately
/// unspecified, so without the sort the tied rows would swap between otherwise identical calls.
#[test]
fn group_order_is_stable_across_calls() {
    let store = agg_store();
    let a = agg_by(&store, "kind", &["bytes"]);
    let b = agg_by(&store, "kind", &["bytes"]);
    let values = |x: &Aggregation| -> Vec<Option<Value>> {
        x.groups.iter().map(|g| g.value.clone()).collect()
    };
    assert_eq!(values(&a), values(&b));
}

#[test]
fn count_and_sum_over_a_filter() {
    let store = agg_store();
    let all = agg(&store, Filter::default(), &["bytes"]);
    assert_eq!(all.count, 4);
    assert_eq!(all.sums["bytes"], Value::Int(47));

    let notes = agg(
        &store,
        Filter(vec![Predicate::Eq(
            "kind".into(),
            Value::Str("note".into()),
        )]),
        &["bytes"],
    );
    assert_eq!(notes.count, 2);
    assert_eq!(notes.sums["bytes"], Value::Int(42));
}

#[test]
fn an_empty_match_counts_zero_and_sums_zero() {
    let store = agg_store();
    let none = agg(
        &store,
        Filter(vec![Predicate::Eq(
            "kind".into(),
            Value::Str("nope".into()),
        )]),
        &["bytes"],
    );
    assert_eq!(none.count, 0);
    assert_eq!(none.sums["bytes"], Value::Int(0));
    // An unknown collection answers the same way rather than erroring.
    let empty = Store::in_memory(2).unwrap();
    let out = empty.aggregate(&["missing"], &AggregateOpts::default());
    assert_eq!(out.count, 0);
    assert!(out.sums.is_empty());
}

#[test]
fn a_sum_reports_a_tagged_value_and_promotes_to_float() {
    let mut store = Store::in_memory(2).unwrap();
    let with =
        |id: &str, v: Value| rec_with(id, vec![1.0, 0.0], BTreeMap::from([("x".to_string(), v)]));
    store
        .upsert(
            "docs",
            &[
                with("i", Value::Int(2)),
                with("f", Value::Float(0.5)),
                with("s", Value::Str("nope".into())),
                rec("none", vec![1.0, 0.0]),
            ],
        )
        .unwrap();
    // Non-numeric and missing values are skipped, not zeroed; one Float promotes the total.
    assert_eq!(
        agg(&store, Filter::default(), &["x"]).sums["x"],
        Value::Float(2.5)
    );
}

#[test]
fn aggregate_sums_several_fields_in_one_pass() {
    let store = agg_store();
    let out = agg(&store, Filter::default(), &["bytes", "kind"]);
    assert_eq!(out.sums["bytes"], Value::Int(47));
    // A field with no numeric value anywhere sums to Int(0), not an error.
    assert_eq!(out.sums["kind"], Value::Int(0));
}

// ── Result diversity: limit_per (nidus-m50.6) ─────────────────────────────

/// Six docs over two files, each at a distinct score so the ranking is unambiguous.
fn diverse_store() -> Store {
    let mut store = Store::in_memory(2).unwrap();
    let recs: Vec<Record> = (0..6)
        .map(|i| {
            let file = if i % 2 == 0 { "a.rs" } else { "b.rs" };
            let angle = 0.02 * i as f32;
            rec_with(
                &format!("d{i}"),
                vec![1.0 - angle, angle],
                BTreeMap::from([("file".to_string(), Value::Str(file.to_string()))]),
            )
        })
        .collect();
    store.upsert("docs", &recs).unwrap();
    store
}

fn capped(store: &Store, cap: Option<LimitPer>, top_k: usize) -> Vec<String> {
    store
        .search(
            &["docs"],
            &[1.0, 0.0],
            &SearchOpts {
                top_k,
                limit_per: cap,
                ..Default::default()
            },
        )
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect()
}

#[test]
fn limit_per_caps_the_hits_carrying_one_value() {
    let store = diverse_store();
    assert_eq!(capped(&store, None, 6).len(), 6);
    let ids = capped(&store, Some(LimitPer::new("file", 2)), 6);
    // The best two of each group survive, in rank order.
    assert_eq!(ids, ["d0", "d1", "d2", "d3"], "{ids:?}");
    assert_eq!(capped(&store, Some(LimitPer::new("file", 1)), 6).len(), 2);
}

/// nidus-m50.15 #14: without a shared group for the absent value, dropping the attribute
/// from a doc would be a way to opt out of the cap entirely.
#[test]
fn records_missing_the_group_attribute_share_one_group() {
    let mut store = Store::in_memory(2).unwrap();
    let recs: Vec<Record> = (0..4)
        .map(|i| {
            rec(
                &format!("d{i}"),
                vec![1.0 - 0.02 * i as f32, 0.02 * i as f32],
            )
        })
        .collect();
    store.upsert("docs", &recs).unwrap();
    let ids = capped(&store, Some(LimitPer::new("file", 2)), 4);
    assert_eq!(ids, ["d0", "d1"], "all four share one group: {ids:?}");
}

#[test]
fn limit_per_reads_the_live_record_not_the_projected_hit() {
    let store = diverse_store();
    let ids: Vec<String> = store
        .search(
            &["docs"],
            &[1.0, 0.0],
            &SearchOpts {
                top_k: 6,
                limit_per: Some(LimitPer::new("file", 1)),
                projection: Projection::exclude(["file"]),
                ..Default::default()
            },
        )
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(
        ids,
        ["d0", "d1"],
        "excluding the attr must not lift the cap"
    );
}

#[test]
fn limit_per_composes_with_pagination() {
    let store = diverse_store();
    let page = store
        .search(
            &["docs"],
            &[1.0, 0.0],
            &SearchOpts {
                top_k: 2,
                offset: 2,
                limit_per: Some(LimitPer::new("file", 2)),
                ..Default::default()
            },
        )
        .unwrap();
    let ids: Vec<&str> = page.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, ["d2", "d3"], "the page is cut AFTER the cap");
}

#[test]
fn limit_per_caps_a_text_search_too() {
    let mut store = Store::in_memory(2).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    let docs: Vec<Record> = (0..4)
        .map(|i| {
            let mut r = doc(&format!("d{i}"), "quantum physics");
            r.attrs
                .insert("file".to_string(), Value::Str("a.rs".to_string()));
            r
        })
        .collect();
    store.upsert("docs", &docs).unwrap();
    let hits = store
        .text_search(
            &["docs"],
            &FtsQuery::new("body", "quantum"),
            &SearchOpts {
                top_k: 4,
                limit_per: Some(LimitPer::new("file", 2)),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn a_degenerate_cap_is_refused() {
    let store = diverse_store();
    for bad in [LimitPer::new("file", 0), LimitPer::new("", 2)] {
        let opts = SearchOpts {
            top_k: 3,
            limit_per: Some(bad.clone()),
            ..Default::default()
        };
        assert!(
            store.search(&["docs"], &[1.0, 0.0], &opts).is_err(),
            "{bad:?} must be refused"
        );
    }
}

// ── Rerank (`crate::store::rerank`) — pure logic, exercised through `Store` ─────────────
// The async provider call lives behind the `rerank` feature; these run on the default lane,
// standing in for it with a length-keyed fake so the test picks which candidate wins.

use crate::model::RerankOpts;

fn texted(id: &str, vector: Vec<f32>, text: &str) -> Record {
    rec_with(
        id,
        vector,
        BTreeMap::from([("nidus.text".to_string(), Value::Str(text.to_string()))]),
    )
}

fn rerank_opts(top_k: usize) -> SearchOpts {
    SearchOpts {
        top_k,
        rerank: Some(RerankOpts::default()),
        ..Default::default()
    }
}

/// Runs the pure rerank stage over an already metric-ranked `hits` set, standing in for
/// `crate::rerank::rerank_hits` plus the promoted tail: a fake score keyed on text length, so
/// the metric-worst candidate wins by being given the longest text.
fn reranked(store: &Store, hits: Vec<Hit>, opts: &SearchOpts) -> Vec<Hit> {
    let text_attr = opts
        .rerank
        .as_ref()
        .expect("opts.rerank must be set")
        .text_attr
        .clone();
    let (texts, passthrough) = super::rerank::candidate_texts(&hits, &text_attr);
    let candidate_idx: Vec<usize> = (0..hits.len())
        .filter(|i| !passthrough.contains(i))
        .collect();
    let scored: Vec<(usize, f32)> = candidate_idx
        .into_iter()
        .zip(texts.iter().map(|t| t.len() as f32))
        .collect();
    store.finish(
        super::rerank::apply_scores(hits, &scored, passthrough),
        opts,
    )
}

/// Same, but for a fused hybrid ranking (`HybridOpts` has no `limit_per`, so the tail is
/// `finish_hybrid` rather than `finish`).
fn reranked_hybrid(store: &Store, hits: Vec<Hit>, opts: &HybridOpts) -> Vec<Hit> {
    let text_attr = opts
        .rerank
        .as_ref()
        .expect("opts.rerank must be set")
        .text_attr
        .clone();
    let (texts, passthrough) = super::rerank::candidate_texts(&hits, &text_attr);
    let candidate_idx: Vec<usize> = (0..hits.len())
        .filter(|i| !passthrough.contains(i))
        .collect();
    let scored: Vec<(usize, f32)> = candidate_idx
        .into_iter()
        .zip(texts.iter().map(|t| t.len() as f32))
        .collect();
    store.finish_hybrid(
        super::rerank::apply_scores(hits, &scored, passthrough),
        opts,
    )
}

fn cosine_docs() -> Store {
    let mut store = Store::in_memory(2).unwrap();
    store
        .upsert(
            "docs",
            &[
                texted("d0", vec![1.0, 0.0], "a"),
                texted("d1", vec![0.99, 0.02], "bb"),
                texted("d2", vec![0.9, 0.2], "ccc"),
            ],
        )
        .unwrap();
    store
}

#[test]
fn rerank_reorders_by_provider_score() {
    let store = cosine_docs();
    let opts = rerank_opts(3);
    let hits = store.search(&["docs"], &[1.0, 0.0], &opts).unwrap();
    assert_eq!(
        hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["d0", "d1", "d2"],
        "metric order, before reranking"
    );
    let out = reranked(&store, hits, &opts);
    assert_eq!(
        out.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["d2", "d1", "d0"],
        "the metric-worst candidate (longest text) now ranks first"
    );
}

#[test]
fn rerank_widens_the_candidate_window() {
    let mut store = Store::in_memory(2).unwrap();
    let texts = ["a", "aa", "aaa", "aaaa", "aaaaa"];
    let recs: Vec<Record> = (0..5)
        .map(|i| {
            let angle = 0.05 * i as f32;
            texted(&format!("d{i}"), vec![1.0 - angle, angle], texts[i])
        })
        .collect();
    store.upsert("docs", &recs).unwrap();

    let plain_hits = store
        .search(&["docs"], &[1.0, 0.0], &default_opts(2))
        .unwrap();
    assert_eq!(
        plain_hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["d0", "d1"],
        "without widening only the metric-best two are seen"
    );

    let opts = SearchOpts {
        top_k: 2,
        rerank: Some(RerankOpts {
            overscan: 5,
            text_attr: "nidus.text".to_string(),
        }),
        ..Default::default()
    };
    let widened = store
        .search(
            &["docs"],
            &[1.0, 0.0],
            &SearchOpts {
                top_k: super::rerank::rerank_depth(&opts),
                ..opts.clone()
            },
        )
        .unwrap();
    assert_eq!(widened.len(), 5, "the window now covers the whole corpus");
    let out = reranked(&store, widened, &opts);
    assert_eq!(out.len(), 2, "top_k still caps the final page");
    assert_eq!(
        out[0].id, "d4",
        "metric-worst (longest text) now finishes first, having been outside the plain page"
    );
}

#[test]
fn a_record_with_no_text_attr_is_passed_through_unranked() {
    let mut null_attrs = BTreeMap::new();
    null_attrs.insert("nidus.text".to_string(), Value::Null);
    let mut int_attrs = BTreeMap::new();
    int_attrs.insert("nidus.text".to_string(), Value::Int(3));
    let mut store = Store::in_memory(2).unwrap();
    store
        .upsert(
            "docs",
            &[
                texted("has_text", vec![0.9, 0.1], "hello"),
                rec("absent", vec![0.8, 0.2]),
                rec_with("nullish", vec![0.7, 0.3], null_attrs),
                rec_with("nonstr", vec![0.6, 0.4], int_attrs),
                texted("empty", vec![1.0, 0.0], ""),
            ],
        )
        .unwrap();
    let opts = rerank_opts(5);
    let hits = store.search(&["docs"], &[1.0, 0.0], &opts).unwrap();
    let metric_order: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();

    let out = reranked(&store, hits.clone(), &opts);
    assert_eq!(
        out[0].id, "has_text",
        "the only candidate with usable text ranks first"
    );
    let passthrough_ids: Vec<&str> = out[1..].iter().map(|h| h.id.as_str()).collect();
    let expected: Vec<&str> = metric_order
        .iter()
        .filter(|id| id.as_str() != "has_text")
        .map(String::as_str)
        .collect();
    assert_eq!(
        passthrough_ids, expected,
        "absent/Null/non-Str/empty-string all pass through in original metric order"
    );
    for h in &out[1..] {
        let original = hits.iter().find(|o| o.id == h.id).unwrap();
        assert_eq!(
            h.score, original.score,
            "passthrough score untouched: {}",
            h.id
        );
    }
}

#[test]
fn rerank_applies_over_an_ann_result_set() {
    let mut store = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", 2)
            .auto_compact(None)
            .ann(Some(AnnConfig::hnsw())),
    )
    .unwrap();
    store
        .upsert(
            "docs",
            &[
                texted("d0", vec![1.0, 0.0], "a"),
                texted("d1", vec![0.9, 0.1], "bb"),
            ],
        )
        .unwrap();
    let opts = rerank_opts(2);
    let hits = store.search(&["docs"], &[1.0, 0.0], &opts).unwrap();
    assert_eq!(hits[0].id, "d0", "ANN still returns the metric-best first");
    let out = reranked(&store, hits, &opts);
    assert_eq!(
        out[0].id, "d1",
        "reranking still applies over the ANN result set, without forcing exact"
    );
}

#[test]
fn rerank_preserves_the_collection_id_tie_break() {
    let mut store = Store::in_memory(2).unwrap();
    store
        .upsert(
            "docs",
            &[
                texted("z", vec![1.0, 0.0], "aa"),
                texted("a", vec![0.99, 0.02], "bb"),
            ],
        )
        .unwrap();
    let opts = rerank_opts(2);
    let hits = store.search(&["docs"], &[1.0, 0.0], &opts).unwrap();
    let out = reranked(&store, hits, &opts);
    assert_eq!(out[0].score, out[1].score, "equal-length texts tie exactly");
    assert_eq!(
        out.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["a", "z"],
        "ties break on (collection, id) ascending, same as the metric path"
    );
}

#[test]
fn limit_per_survives_a_rerank() {
    let mut store = Store::in_memory(2).unwrap();
    let mk = |id: &str, angle: f32, file: &str, text: &str| {
        rec_with(
            id,
            vec![1.0 - angle, angle],
            BTreeMap::from([
                ("file".to_string(), Value::Str(file.to_string())),
                ("nidus.text".to_string(), Value::Str(text.to_string())),
            ]),
        )
    };
    store
        .upsert(
            "docs",
            &[
                mk("a1", 0.01, "a.rs", "a"),
                mk("a2", 0.05, "a.rs", "aaaa"),
                mk("b1", 0.02, "b.rs", "bb"),
                mk("b2", 0.06, "b.rs", "bbb"),
            ],
        )
        .unwrap();
    let opts = SearchOpts {
        top_k: 4,
        limit_per: Some(LimitPer::new("file", 1)),
        rerank: Some(RerankOpts::default()),
        ..Default::default()
    };

    let plain = store.search(&["docs"], &[1.0, 0.0], &opts).unwrap();
    assert_eq!(
        plain.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["a1", "b1"],
        "the plain metric cap, for contrast"
    );

    // The candidate gather for rerank drops `limit_per` (re-applied post-rerank, decision 3):
    // capping it here would freeze each group's winner at metric order.
    let widened = store
        .search(
            &["docs"],
            &[1.0, 0.0],
            &SearchOpts {
                top_k: super::rerank::rerank_depth(&opts),
                limit_per: None,
                ..opts.clone()
            },
        )
        .unwrap();
    let out = reranked(&store, widened, &opts);
    assert_eq!(
        out.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["a2", "b2"],
        "limit_per re-applied AFTER rerank keeps the reranked winner per group"
    );
}

#[test]
fn min_score_applies_before_rerank() {
    let store = cosine_docs();
    let opts = SearchOpts {
        top_k: 3,
        min_score: Some(0.98),
        rerank: Some(RerankOpts::default()),
        ..Default::default()
    };
    let hits = store.search(&["docs"], &[1.0, 0.0], &opts).unwrap();
    assert!(
        hits.iter().all(|h| h.id != "d2"),
        "d2's cosine score is below min_score, so it never enters the rerank window: {hits:?}"
    );
    let out = reranked(&store, hits, &opts);
    assert!(
        out.iter().all(|h| h.id != "d2"),
        "reranking cannot resurrect a hit min_score already excluded"
    );
}

#[test]
fn rerank_of_a_hybrid_ranking() {
    let mut store = Store::in_memory(2).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    let mk = |id: &str, vector: Vec<f32>, body: &str, text: &str| {
        rec_with(
            id,
            vector,
            BTreeMap::from([
                ("body".to_string(), Value::Str(body.to_string())),
                ("nidus.text".to_string(), Value::Str(text.to_string())),
            ]),
        )
    };
    store
        .upsert(
            "docs",
            &[
                mk("d0", vec![1.0, 0.0], "quantum physics", "a"),
                mk("d1", vec![0.9, 0.1], "quantum physics", "aaaa"),
            ],
        )
        .unwrap();
    let opts = HybridOpts {
        top_k: 2,
        rerank: Some(RerankOpts::default()),
        ..Default::default()
    };
    let fused = store
        .hybrid_search(
            &["docs"],
            &[1.0, 0.0],
            &FtsQuery::new("body", "quantum"),
            &opts,
        )
        .unwrap();
    assert_eq!(fused[0].id, "d0", "d0 wins the plain fused ranking");
    let out = reranked_hybrid(&store, fused, &opts);
    assert_eq!(
        out[0].id, "d1",
        "reranking still applies over a fused RRF result set"
    );
}

#[test]
fn a_nan_rerank_score_does_not_poison_the_order() {
    let store = cosine_docs();
    let opts = rerank_opts(3);
    let hits = store.search(&["docs"], &[1.0, 0.0], &opts).unwrap();
    let scored = vec![(0, f32::NAN), (1, 2.0), (2, 1.0)];
    let out = super::rerank::apply_scores(hits, &scored, vec![]);
    let out = store.finish(out, &opts);
    assert_eq!(
        out.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        vec!["d1", "d2", "d0"],
        "a NaN score sorts last, never displacing a real result"
    );
}

/// Regression, review finding: `refresh` re-merged onto the already-merged config, so
/// `p.ann.or(self.ann)` reproduced the stale value and a cleared knob never retracted.
#[cfg_attr(miri, ignore)] // fsync
#[test]
fn refresh_retracts_a_cleared_profile_knob() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut writer = Store::open(Config::new(&path, 3)).unwrap();
    writer
        .set_open_profile(&OpenProfile {
            ann: Some(AnnConfig::hnsw()),
            ..Default::default()
        })
        .unwrap();

    let mut reader = Store::open(Config::new(&path, 3).open_mode(OpenMode::ReadOnly)).unwrap();
    assert_eq!(
        reader.config().ann,
        Some(AnnConfig::hnsw()),
        "reader should adopt the recorded profile at open"
    );

    writer.clear_open_profile().unwrap();
    reader.refresh().unwrap();
    assert_eq!(
        reader.config().ann,
        None,
        "a live reader must drop an ann default the writer cleared"
    );
}

/// Regression, review finding: an unbuildable profile could be persisted, and since a profile
/// resolves at open, every later open failed — including the `clear` needed to undo it.
#[cfg_attr(miri, ignore)] // fsync
#[test]
fn set_open_profile_rejects_a_combination_that_could_never_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut store = Store::open(Config::new(&path, 3).distance(Distance::Euclidean)).unwrap();

    let err = store
        .set_open_profile(&OpenProfile {
            quantization: Some(Quantization::binary()),
            ..Default::default()
        })
        .expect_err("binary quantization on a Euclidean store must be refused");
    assert!(
        err.to_string().contains("binary quantization requires"),
        "{err}"
    );

    drop(store);
    Store::open(Config::new(&path, 3).distance(Distance::Euclidean))
        .expect("the store must still open: nothing unbuildable was persisted");
}

// ── Checksum sidecar integrity (#160) ──────────────────────────────────

#[cfg_attr(miri, ignore)] // fsync
#[test]
fn flush_writes_a_checksum_sidecar_and_verify_comes_back_clean() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut store = Store::open(Config::new(&path, 2)).unwrap();
    store
        .upsert("col", &[rec("a", vec![1.0, 0.0]), rec("b", vec![0.0, 1.0])])
        .unwrap();
    store.flush().unwrap();

    let reports = store.verify_integrity().unwrap();
    assert!(!reports.is_empty());
    for r in &reports {
        match r.integrity {
            SegmentIntegrity::Ok {
                rows_covered,
                rows_total,
            } => assert_eq!(
                rows_covered, rows_total,
                "segment {} should be fully covered right after flush",
                r.segment
            ),
            other => panic!("segment {}: expected Ok, got {other:?}", r.segment),
        }
    }
}

/// Rows appended after the last `flush` (with no further flush) must be reported
/// uncovered, not silently treated as verified.
#[cfg_attr(miri, ignore)] // fsync
#[test]
fn verify_reports_rows_written_after_flush_as_uncovered() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut store = Store::open(Config::new(&path, 2)).unwrap();
    store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    store.flush().unwrap();

    // Appended, but no flush after this — the sidecar must not claim to cover it.
    store.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();

    let reports = store.verify_integrity().unwrap();
    let active = reports.last().unwrap();
    match active.integrity {
        SegmentIntegrity::Ok {
            rows_covered,
            rows_total,
        } => {
            assert!(
                rows_covered < rows_total,
                "the row written after the last flush must be uncovered: \
                 covered {rows_covered}, total {rows_total}"
            );
        }
        other => panic!("expected a partial Ok, got {other:?}"),
    }
}

/// LOAD-BEARING: flip a byte inside a flushed row, then reopen and verify. This must fail
/// without group 1's checksum plumbing — a test that only asserted "verification ran"
/// would pass even with a corrupted store, which is the exact shape SKILL.md warns about.
#[cfg_attr(miri, ignore)] // fsync
#[test]
fn verify_reports_a_mismatch_naming_the_segment_after_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store
            .upsert("col", &[rec("a", vec![1.0, 0.0]), rec("b", vec![0.0, 1.0])])
            .unwrap();
        store.flush().unwrap();
    }

    // Flip the file's last byte — inside the last row's bytes, past the header — after the
    // store (and its sidecar) is closed, so a fresh open reads the corrupted bytes cleanly.
    let data_path = path.join("data");
    let mut bytes = std::fs::read(&data_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&data_path, &bytes).unwrap();

    let mut store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
    let reports = store.verify_integrity().unwrap();
    let mismatch = reports
        .iter()
        .find(|r| matches!(r.integrity, SegmentIntegrity::Mismatch { .. }))
        .unwrap_or_else(|| panic!("expected a mismatch, got {reports:?}"));
    assert_eq!(mismatch.segment, "data");
}

#[test]
fn segment_report_field_shape() {
    // Pure-logic sanity check on the report type itself (Miri-clean): a report just
    // pairs a segment name with the checksum finding, nothing more.
    let report = SegmentReport {
        segment: "data".to_string(),
        integrity: SegmentIntegrity::NoChecksum { rows_total: 0 },
    };
    assert_eq!(report.segment, "data");
    assert!(matches!(
        report.integrity,
        SegmentIntegrity::NoChecksum { rows_total: 0 }
    ));
}

// ── Result diversity: MMR in vector space (nidus-tx2) ─────────────────────

/// Three near-duplicates crowding the query plus one genuinely different doc that scores
/// lower. Without MMR the duplicates own every page; with it the outlier surfaces.
fn crowded_store() -> Store {
    let mut store = Store::in_memory(3).unwrap();
    let recs = vec![
        rec_with(
            "dup0",
            vec![1.0, 0.02, 0.0],
            BTreeMap::from([("file".to_string(), Value::Str("a.rs".into()))]),
        ),
        rec_with(
            "dup1",
            vec![1.0, 0.03, 0.0],
            BTreeMap::from([("file".to_string(), Value::Str("a.rs".into()))]),
        ),
        rec_with(
            "dup2",
            vec![1.0, 0.04, 0.0],
            BTreeMap::from([("file".to_string(), Value::Str("a.rs".into()))]),
        ),
        rec_with(
            "novel",
            vec![0.6, 0.8, 0.0],
            BTreeMap::from([("file".to_string(), Value::Str("b.rs".into()))]),
        ),
    ];
    store.upsert("docs", &recs).unwrap();
    store
}

fn spread(store: &Store, lambda: Option<f32>, top_k: usize) -> Vec<String> {
    store
        .search(
            &["docs"],
            &[1.0, 0.0, 0.0],
            &SearchOpts {
                top_k,
                diversity: lambda,
                ..Default::default()
            },
        )
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect()
}

#[test]
fn diversity_changes_the_returned_set_on_a_near_duplicate_corpus() {
    let store = crowded_store();
    // The plain ranking is three interchangeable duplicates; the different doc never shows.
    assert_eq!(spread(&store, None, 2), ["dup0", "dup1"]);
    // MMR keeps rank 1 and spends slot 2 on the outlier instead of a second near-copy.
    // A duplicate scoring 0.9996 against 0.9998 only loses to spread once lambda tips past
    // relevance, which is MMR working, not a threshold to tune away.
    assert_eq!(spread(&store, Some(0.3), 2), ["dup0", "novel"]);
}

#[test]
fn diversity_unset_is_a_no_op() {
    let store = crowded_store();
    let plain = store
        .search(&["docs"], &[1.0, 0.0, 0.0], &default_opts(4))
        .unwrap();
    let explicit = store
        .search(
            &["docs"],
            &[1.0, 0.0, 0.0],
            &SearchOpts {
                top_k: 4,
                diversity: None,
                ..Default::default()
            },
        )
        .unwrap();
    let ids: Vec<&str> = plain.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids, ["dup0", "dup1", "dup2", "novel"]);
    assert_eq!(plain, explicit);
}

/// An unset knob must not deepen the scan either, or "no-op" would still cost a wider ranking.
#[test]
fn diversity_only_over_fetches_when_it_is_set() {
    let plain = SearchOpts {
        top_k: 10,
        ..Default::default()
    };
    assert_eq!(read::depth(&plain), 10);
    let spread = SearchOpts {
        diversity: Some(0.5),
        ..plain.clone()
    };
    assert_eq!(read::depth(&spread), 10 * super::diversity::MMR_OVERFETCH);
    // The larger factor wins rather than the two multiplying into an unbounded scan.
    let both = SearchOpts {
        limit_per: Some(LimitPer::new("file", 1)),
        ..spread
    };
    assert_eq!(
        read::depth(&both),
        10 * super::aggregate::LIMIT_PER_OVERFETCH
    );
}

#[test]
fn lambda_one_leaves_the_metric_ranking_alone() {
    let store = crowded_store();
    assert_eq!(spread(&store, Some(1.0), 4), spread(&store, None, 4));
}

#[test]
fn diversity_never_displaces_the_top_hit() {
    let store = crowded_store();
    for lambda in [0.0, 0.25, 0.5, 0.75, 1.0] {
        assert_eq!(
            spread(&store, Some(lambda), 4)[0],
            "dup0",
            "lambda {lambda}"
        );
    }
}

/// Greedy selection is order-dependent, so identical candidates must resolve the way the
/// ranking already did — else SPEC §7's total order (and pagination with it) stops holding.
#[test]
fn identical_vectors_keep_the_deterministic_total_order() {
    let mut store = Store::in_memory(2).unwrap();
    let recs: Vec<Record> = ["c", "a", "b", "d"]
        .iter()
        .map(|id| rec(id, vec![1.0, 0.0]))
        .collect();
    store.upsert("docs", &recs).unwrap();
    let ids = spread_q(&store, &[1.0, 0.0], Some(0.3), 4);
    assert_eq!(ids, ["a", "b", "c", "d"], "{ids:?}");
    // And stable across repeated calls, not merely sorted once.
    assert_eq!(spread_q(&store, &[1.0, 0.0], Some(0.3), 4), ids);
}

#[test]
fn diversity_composes_with_pagination() {
    let store = crowded_store();
    let page0 = store
        .search(
            &["docs"],
            &[1.0, 0.0, 0.0],
            &SearchOpts {
                top_k: 1,
                diversity: Some(0.3),
                ..Default::default()
            },
        )
        .unwrap();
    let page1 = store
        .search(
            &["docs"],
            &[1.0, 0.0, 0.0],
            &SearchOpts {
                top_k: 1,
                offset: 1,
                diversity: Some(0.3),
                ..Default::default()
            },
        )
        .unwrap();
    // Diversify then paginate: page 2 is the second MMR pick, not the second raw score.
    assert_eq!(page0[0].id, "dup0");
    assert_eq!(page1[0].id, "novel");
}

/// The cap is a hard constraint and runs first, so MMR only ever reorders legal survivors.
#[test]
fn diversity_composes_with_limit_per() {
    let store = crowded_store();
    let ids: Vec<String> = store
        .search(
            &["docs"],
            &[1.0, 0.0, 0.0],
            &SearchOpts {
                top_k: 4,
                limit_per: Some(LimitPer::new("file", 2)),
                diversity: Some(0.3),
                ..Default::default()
            },
        )
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    // The cap keeps two of `a.rs` and MMR then spends slot 2 on the other file, so both
    // mechanisms are visibly in force: uncapped this would be dup0/dup1/dup2.
    assert_eq!(ids, ["dup0", "novel", "dup1"], "{ids:?}");
}

/// Both knobs must be in force at once: `rank_by` reshapes the scores MMR then spreads, so
/// a diversified ranking still reflects the decay penalty rather than the raw metric.
#[test]
fn diversity_composes_with_rank_by() {
    const DAY: i64 = 86_400_000;
    let origin = 2_000 * DAY;
    let mut store = crowded_store();
    // Only `dup1` carries a timestamp, so `missing(0.0)` penalizes every other doc and `dup1`
    // takes rank 1 away from `dup0`.
    store
        .upsert(
            "docs",
            &[rec_with(
                "dup1",
                vec![1.0, 0.03, 0.0],
                BTreeMap::from([
                    ("file".to_string(), Value::Str("a.rs".into())),
                    ("ts".to_string(), Value::Int(origin)),
                ]),
            )],
        )
        .unwrap();
    let opts = SearchOpts {
        top_k: 2,
        rank_by: Some(RankBy::Decay(
            Decay::new("ts", origin, 7 * DAY).lambda(0.5).missing(0.0),
        )),
        diversity: Some(0.1),
        ..Default::default()
    };
    let ids: Vec<String> = store
        .search(&["docs"], &[1.0, 0.0, 0.0], &opts)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(ids, ["dup1", "novel"], "{ids:?}");
    // Without the decay the same diversity lambda keeps `dup0` at rank 1, so neither knob is
    // quietly overriding the other.
    assert_eq!(spread(&store, Some(0.1), 2), ["dup0", "novel"]);
}

/// Vectors are normalized on insert only for `Cosine`, so MMR must divide by real norms
/// rather than assume the dot product already is a cosine.
#[test]
fn diversity_measures_cosine_on_an_unnormalized_store() {
    let mut store = Store::in_memory_with(2, Distance::DotProduct).unwrap();
    store
        .upsert(
            "docs",
            &[
                rec("big", vec![9.0, 0.0]),
                rec("small", vec![1.0, 0.0]),
                rec("side", vec![0.0, 2.0]),
            ],
        )
        .unwrap();
    let ids = spread_q(&store, &[1.0, 0.0], Some(0.0), 2);
    // `big` and `small` are collinear (cosine 1.0) despite very different dot products, so
    // pure-diversity selection must pick the orthogonal doc second.
    assert_eq!(ids, ["big", "side"], "{ids:?}");
}

fn spread_q(store: &Store, q: &[f32], lambda: Option<f32>, top_k: usize) -> Vec<String> {
    store
        .search(
            &["docs"],
            q,
            &SearchOpts {
                top_k,
                diversity: lambda,
                ..Default::default()
            },
        )
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect()
}

#[test]
fn a_bad_lambda_is_a_caller_fault() {
    let store = crowded_store();
    for bad in [f32::NAN, -0.1, 1.1, f32::INFINITY] {
        let err = store
            .search(
                &["docs"],
                &[1.0, 0.0, 0.0],
                &SearchOpts {
                    top_k: 2,
                    diversity: Some(bad),
                    ..Default::default()
                },
            )
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains(read::BAD_QUERY), "{bad}: {msg}");
        assert!(msg.contains("diversity"), "{bad}: {msg}");
    }
}

/// The window is bounded on purpose (MMR is O(W² · dim)). Past it the ranking keeps its
/// score order rather than the cost growing with whatever depth was over-fetched.
#[test]
#[cfg_attr(miri, ignore)] // runtime cost: a 512-wide MMR window is 512² pairs under the interpreter
fn beyond_the_window_bound_the_tail_keeps_score_order() {
    let n = super::diversity::MAX_DIVERSITY_WINDOW + 8;
    let mut store = Store::in_memory(2).unwrap();
    let recs: Vec<Record> = (0..n)
        .map(|i| {
            let angle = i as f32 * 1e-4;
            rec(&format!("d{i:05}"), vec![1.0, angle])
        })
        .collect();
    store.upsert("docs", &recs).unwrap();
    let hits = store
        .search(
            &["docs"],
            &[1.0, 0.0],
            &SearchOpts {
                top_k: n,
                diversity: Some(0.5),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hits.len(), n);
    let tail: Vec<&str> = hits[super::diversity::MAX_DIVERSITY_WINDOW..]
        .iter()
        .map(|h| h.id.as_str())
        .collect();
    let mut sorted = tail.clone();
    sorted.sort_unstable();
    assert_eq!(
        tail, sorted,
        "the tail past the window must stay score-ordered"
    );
}

/// The approximate paths hand `finish` the same `Vec<Hit>` the exact one does, so MMR must
/// spread an ANN or quantized ranking too rather than silently only working on brute force.
#[test]
fn diversity_spreads_the_ann_and_quantized_rankings() {
    let crowd = [
        vec![1.0, 0.02, 0.0],
        vec![1.0, 0.03, 0.0],
        vec![1.0, 0.04, 0.0],
        vec![0.6, 0.8, 0.0],
    ];
    let configs: Vec<(&str, Store)> = vec![
        ("hnsw", ann_store(3, AnnConfig::hnsw(), &crowd)),
        ("ivf", ann_store(3, AnnConfig::ivf(), &crowd)),
        ("int8", crowded_quant_store(Quantization::default())),
        ("binary", crowded_quant_store(Quantization::binary())),
    ];
    for (name, store) in configs {
        let plain = spread_col(&store, None, 2);
        let mmr = spread_col(&store, Some(0.3), 2);
        assert_eq!(plain[0], mmr[0], "{name}: rank 1 must survive");
        assert_ne!(plain, mmr, "{name}: MMR changed nothing");
        assert!(
            mmr[1].contains('3'),
            "{name}: expected the outlier, got {mmr:?}"
        );
    }
}

/// `crowded_store`'s corpus in collection `col`, under a quantization kind.
fn crowded_quant_store(q: Quantization) -> Store {
    let mut store = Store::in_memory_cfg(
        Config::new("/dev/null/in-memory", 3)
            .open_mode(OpenMode::ReadWrite)
            .auto_compact(None)
            .quantization(Some(q)),
    )
    .unwrap();
    let recs: Vec<Record> = [
        vec![1.0, 0.02, 0.0],
        vec![1.0, 0.03, 0.0],
        vec![1.0, 0.04, 0.0],
        vec![0.6, 0.8, 0.0],
    ]
    .iter()
    .enumerate()
    .map(|(i, v)| rec(&format!("d{i}"), v.clone()))
    .collect();
    store.upsert("col", &recs).unwrap();
    store
}

fn spread_col(store: &Store, lambda: Option<f32>, top_k: usize) -> Vec<String> {
    store
        .search(
            &["col"],
            &[1.0, 0.0, 0.0],
            &SearchOpts {
                top_k,
                diversity: lambda,
                ..Default::default()
            },
        )
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect()
}

/// A text-only record has no vector, so it cannot be measurably redundant with anything —
/// MMR must carry it on its BM25 score instead of dropping or mis-penalizing it.
#[test]
fn diversity_handles_a_text_only_hit() {
    let mut store = Store::in_memory(2).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    let body = |t: &str| BTreeMap::from([("body".to_string(), Value::Str(t.to_string()))]);
    store
        .upsert(
            "docs",
            &[
                rec_with("vec0", vec![1.0, 0.0], body("alpha alpha alpha")),
                rec_with("vec1", vec![1.0, 0.0], body("alpha alpha")),
                Record::text_only("textual", body("alpha beta gamma delta")),
            ],
        )
        .unwrap();
    let opts = SearchOpts {
        top_k: 3,
        diversity: Some(0.3),
        ..Default::default()
    };
    let ids: Vec<String> = store
        .text_search(&["docs"], &FtsQuery::new("body", "alpha"), &opts)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    // BM25 alone ranks the vectorless doc last (one term, longest body). The two identical
    // vectors penalize each other and it carries no penalty, so MMR promotes it past the
    // second copy rather than burying or dropping it.
    assert_eq!(ids, ["vec0", "textual", "vec1"], "{ids:?}");
}

// ── Reinforcement stamp (nidus-gk6) ──────────────────────────────────────────

use crate::findex::FilterIndexField;
use crate::meta::{META_ACCESS_COUNT, META_EXPIRES_AT, META_LAST_ACCESSED};

#[test]
fn reinforce_increments_from_absent_and_then_from_a_prior_count() {
    let mut store = Store::in_memory(2).unwrap();
    store.upsert("docs", &[rec("a", vec![1.0, 0.0])]).unwrap();

    assert_eq!(store.reinforce("docs", &["a"], 1_000, None).unwrap(), 1);
    let first = store.get("docs", "a").unwrap();
    assert_eq!(first.attrs.get(META_ACCESS_COUNT), Some(&Value::Int(1)));
    assert_eq!(
        first.attrs.get(META_LAST_ACCESSED),
        Some(&Value::DateTime(1_000))
    );

    store.reinforce("docs", &["a"], 2_000, None).unwrap();
    let second = store.get("docs", "a").unwrap();
    assert_eq!(second.attrs.get(META_ACCESS_COUNT), Some(&Value::Int(2)));
    assert_eq!(
        second.attrs.get(META_LAST_ACCESSED),
        Some(&Value::DateTime(2_000))
    );
}

/// The regression test for the `UpsertText` vector-stripping trap: reinforcing a vectored
/// doc must leave its row and score untouched, not just leave `row` as `Some`.
#[test]
fn reinforce_leaves_the_row_and_every_other_attr_alone() {
    let mut store = Store::in_memory(2).unwrap();
    store
        .upsert(
            "docs",
            &[rec_with(
                "a",
                vec![1.0, 0.0],
                BTreeMap::from([("keep".to_string(), Value::Str("mine".to_string()))]),
            )],
        )
        .unwrap();
    let before = store
        .search(&["docs"], &[1.0, 0.0], &default_opts(1))
        .unwrap();
    assert_eq!(before.len(), 1, "must be searchable before reinforcing");

    store.reinforce("docs", &["a"], 1_000, None).unwrap();

    let after = store
        .search(&["docs"], &[1.0, 0.0], &default_opts(1))
        .unwrap();
    assert_eq!(
        after.len(),
        1,
        "the doc must still be searchable — its row must survive"
    );
    assert_eq!(
        after[0].score, before[0].score,
        "reinforce must not change the score — the vector/row is untouched"
    );
    assert_eq!(
        after[0].attrs.get("keep"),
        Some(&Value::Str("mine".to_string())),
        "every other attr must survive untouched"
    );
}

#[test]
fn reinforce_does_not_count_a_dead_row() {
    let mut store = Store::in_memory(2).unwrap();
    store.upsert("docs", &[rec("a", vec![1.0, 0.0])]).unwrap();
    let before = store.dead_rows;
    for i in 0..5 {
        store.reinforce("docs", &["a"], 1_000 + i, None).unwrap();
    }
    assert_eq!(
        store.dead_rows, before,
        "pure reads must never fire compaction bookkeeping"
    );
}

#[cfg_attr(miri, ignore)] // fsync — syscalls Miri does not implement
#[test]
fn reinforce_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store.upsert("docs", &[rec("a", vec![1.0, 0.0])]).unwrap();
        store.reinforce("docs", &["a"], 5_000, None).unwrap();
    }
    let store = Store::open(Config::new(&path, 2)).unwrap();
    let record = store.get("docs", "a").unwrap();
    assert_eq!(record.attrs.get(META_ACCESS_COUNT), Some(&Value::Int(1)));
    assert_eq!(
        record.attrs.get(META_LAST_ACCESSED),
        Some(&Value::DateTime(5_000))
    );
}

#[test]
fn reinforce_skips_an_absent_id_without_erroring() {
    let mut store = Store::in_memory(2).unwrap();
    store.upsert("docs", &[rec("a", vec![1.0, 0.0])]).unwrap();
    let stamped = store
        .reinforce("docs", &["a", "ghost"], 1_000, None)
        .unwrap();
    assert_eq!(stamped, 1, "only the present id is stamped");
    assert_eq!(
        store
            .reinforce("no-such-collection", &["a"], 1_000, None)
            .unwrap(),
        0,
        "an absent collection is skipped too, not an error"
    );
}

#[cfg_attr(miri, ignore)] // fsync — syscalls Miri does not implement
#[test]
fn reinforce_is_refused_on_a_read_only_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    {
        let mut store = Store::open(Config::new(&path, 2)).unwrap();
        store.upsert("docs", &[rec("a", vec![1.0, 0.0])]).unwrap();
    }
    let mut store = Store::open(Config::new(&path, 2).open_mode(OpenMode::ReadOnly)).unwrap();
    assert!(store.reinforce("docs", &["a"], 1_000, None).is_err());
}

#[test]
fn extend_ttl_moves_an_existing_expiry_forward_only() {
    let mut store = Store::in_memory(2).unwrap();
    store
        .upsert(
            "docs",
            &[
                rec("no-expiry", vec![1.0, 0.0]),
                rec_with(
                    "near-expiry",
                    vec![0.0, 1.0],
                    BTreeMap::from([(META_EXPIRES_AT.to_string(), Value::DateTime(1_500))]),
                ),
                rec_with(
                    "far-expiry",
                    vec![1.0, 1.0],
                    BTreeMap::from([(META_EXPIRES_AT.to_string(), Value::DateTime(10_000))]),
                ),
            ],
        )
        .unwrap();

    // now = 1_000, extend = 5s → a floor of 6_000.
    store
        .reinforce(
            "docs",
            &["no-expiry", "near-expiry", "far-expiry"],
            1_000,
            Some(5),
        )
        .unwrap();

    let no_expiry = store.get("docs", "no-expiry").unwrap();
    assert!(
        !no_expiry.attrs.contains_key(META_EXPIRES_AT),
        "never had an expiry, must not gain one"
    );
    let near = store.get("docs", "near-expiry").unwrap();
    assert_eq!(
        near.attrs.get(META_EXPIRES_AT),
        Some(&Value::DateTime(6_000)),
        "a nearer expiry moves out to now + extend"
    );
    let far = store.get("docs", "far-expiry").unwrap();
    assert_eq!(
        far.attrs.get(META_EXPIRES_AT),
        Some(&Value::DateTime(10_000)),
        "an already-further expiry must not move back"
    );
}

#[test]
fn reinforce_makes_the_count_filterable() {
    let mut store = Store::in_memory(2).unwrap();
    store.upsert("docs", &[rec("a", vec![1.0, 0.0])]).unwrap();
    store
        .set_filter_index("docs", &[FilterIndexField::new(META_ACCESS_COUNT)])
        .unwrap();

    store.reinforce("docs", &["a"], 1_000, None).unwrap();

    let hits = store
        .list(
            &["docs"],
            &ListOpts {
                filter: Filter(vec![Predicate::Ge(
                    META_ACCESS_COUNT.to_string(),
                    Value::Int(1),
                )]),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        hits.len(),
        1,
        "the reinforced doc must be found via the filter"
    );
}

/// `extend_ttl_seconds` and a stored count both arrive unvalidated (the wire for the first,
/// a raw `upsert` for the second), so the arithmetic must saturate rather than panic in debug
/// and wrap in release, exactly as `stamp_recency` already does.
#[test]
fn reinforce_saturates_instead_of_overflowing() {
    let mut store = Store::in_memory(2).unwrap();
    let mut attrs = BTreeMap::new();
    attrs.insert(META_ACCESS_COUNT.to_string(), Value::Int(i64::MAX));
    attrs.insert(META_EXPIRES_AT.to_string(), Value::DateTime(i64::MAX - 1));
    store
        .upsert(
            "docs",
            &[Record {
                id: "a".into(),
                vector: Some(vec![1.0, 0.0]),
                attrs,
            }],
        )
        .unwrap();

    store
        .reinforce("docs", &["a"], 1_000, Some(i64::MAX))
        .unwrap();

    let got = store.attrs_of("docs", "a").unwrap();
    assert_eq!(got.get(META_ACCESS_COUNT), Some(&Value::Int(i64::MAX)));
    // Saturated to i64::MAX, which is still *forward* of the old MAX - 1. The bug this
    // guards is the wrap: an unchecked multiply lands on a past instant and expires the
    // entry the recall was extending.
    assert_eq!(got.get(META_EXPIRES_AT), Some(&Value::DateTime(i64::MAX)));
}

/// A stamp writes only the two reinforcement attrs, so a collection whose FTS schema covers
/// something else must not be reindexed: `FieldIndex::index` tombstones and re-appends, so a
/// reindex per recall would leak a dead posting set per stamp on a read path.
#[test]
fn reinforce_does_not_reindex_a_collection_it_does_not_touch() {
    let mut store = Store::in_memory(2).unwrap();
    store
        .set_fts_schema("docs", &[FtsField::new("body")])
        .unwrap();
    let mut attrs = BTreeMap::new();
    attrs.insert("body".to_string(), Value::Str("deploys run at noon".into()));
    store
        .upsert(
            "docs",
            &[Record {
                id: "a".into(),
                vector: Some(vec![1.0, 0.0]),
                attrs,
            }],
        )
        .unwrap();
    let before = store.fts_posting_count();

    for _ in 0..5 {
        store.reinforce("docs", &["a"], 1_000, None).unwrap();
    }

    assert_eq!(
        store.fts_posting_count(),
        before,
        "a stamp touching no indexed field must not re-append postings"
    );
    // And the text is still findable, so skipping the reindex did not drop the doc.
    let hits = store
        .text_search(
            &["docs"],
            &FtsQuery::new("body", "deploys"),
            &default_opts(10),
        )
        .unwrap();
    assert_eq!(hits.len(), 1, "the doc must still be findable by text");
}

// ── Query plan (nidus-cvz) ───────────────────────────────────────────────────

use crate::plan::{Candidates, Narrowing, QueryPath};

#[test]
fn plan_reports_exact_path_with_rows_scanned() {
    let data = random_unit_vectors(20, 3, 1);
    let store = exact_store(3, &data);
    let q = random_unit_vectors(1, 3, 2).pop().unwrap();
    let (hits, plan) = store
        .search_with_plan(&["col"], &q, &default_opts(5))
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(plan.path, QueryPath::Exact);
    assert_eq!(plan.rows_scanned, Some(20));
    assert_eq!(plan.narrowing, Narrowing::Inactive);
}

#[test]
fn plan_reports_quantized_path_with_rows_scanned() {
    let mut store = quantized_store(3);
    store.create_collection("col").unwrap();
    store
        .upsert(
            "col",
            &[
                rec("a", vec![0.9, 0.1, 0.0]),
                rec("b", vec![0.5, 0.5, 0.0]),
                rec("c", vec![0.0, 0.0, 1.0]),
            ],
        )
        .unwrap();
    let (hits, plan) = store
        .search_with_plan(&["col"], &[1.0, 0.0, 0.0], &default_opts(3))
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(plan.path, QueryPath::Quantized);
    assert_eq!(plan.rows_scanned, Some(3));
}

#[test]
fn plan_reports_ann_path_with_no_rows_scanned() {
    let data = random_unit_vectors(50, 4, 3);
    let store = ann_store(4, AnnConfig::hnsw(), &data);
    let q = random_unit_vectors(1, 4, 4).pop().unwrap();
    let (hits, plan) = store
        .search_with_plan(&["col"], &q, &default_opts(5))
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(plan.path, QueryPath::Ann);
    assert_eq!(plan.rows_scanned, None);
    assert!(plan.candidates.is_some());
}

#[test]
fn plan_reports_ann_prefilter_fallback_on_a_selective_filter() {
    let data = random_unit_vectors(50, 4, 5);
    let mut store = ann_store(4, AnnConfig::hnsw(), &data);
    let mut attrs = BTreeMap::new();
    attrs.insert("tag".to_string(), Value::Int(1));
    // The only tagged doc: a maximally selective filter, narrow enough to starve the
    // ANN walk and force the exact-prefilter fallback (nidus-0ou).
    store
        .upsert(
            "col",
            &[rec_with("tagged", vec![1.0, 0.0, 0.0, 0.0], attrs)],
        )
        .unwrap();
    let opts = SearchOpts {
        filter: Filter(vec![Predicate::Eq("tag".to_string(), Value::Int(1))]),
        ..default_opts(5)
    };
    let (hits, plan) = store
        .search_with_plan(&["col"], &[1.0, 0.0, 0.0, 0.0], &opts)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(plan.path, QueryPath::AnnPrefilterFallback);
    assert_eq!(
        plan.rows_scanned, None,
        "the fallback is not the counted brute-force scan"
    );
}

#[test]
fn plan_reports_segmented_path() {
    let mut s = segmented_store(2, 4, 2);
    s.create_collection("col").unwrap();
    for i in 0..5 {
        s.upsert(
            "col",
            &[
                rec(&format!("a{i}"), vec![(i as f32).cos(), (i as f32).sin()]),
                rec(
                    &format!("b{i}"),
                    vec![(-(i as f32)).sin(), (i as f32).cos()],
                ),
            ],
        )
        .unwrap();
    }
    let (hits, plan) = s
        .search_with_plan(&["col"], &[1.0, 0.0], &default_opts(5))
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(plan.path, QueryPath::Segmented);
    assert_eq!(plan.rows_scanned, None);
}

#[test]
fn plan_candidates_survived_and_dropped_sum_to_surfaced() {
    let data = random_unit_vectors(80, 4, 7);
    let store = ann_store(4, AnnConfig::hnsw(), &data);
    let q = random_unit_vectors(1, 4, 8).pop().unwrap();
    let (_, plan) = store
        .search_with_plan(&["col"], &q, &default_opts(5))
        .unwrap();
    let c: Candidates = plan.candidates.unwrap();
    assert!(c.surfaced >= c.survived);
    let dropped =
        c.dropped_out_of_scope + c.dropped_stale + c.dropped_filtered + c.dropped_min_score;
    assert_eq!(c.surfaced, c.survived + dropped);
}

#[test]
fn plan_narrowing_reports_inactive_declined_and_narrowed() {
    let mut store = Store::in_memory(2).unwrap();
    store
        .upsert(
            "col",
            &[
                rec_with(
                    "a",
                    vec![1.0, 0.0],
                    BTreeMap::from([("tag".to_string(), Value::Str("hello world".into()))]),
                ),
                rec("b", vec![0.0, 1.0]),
            ],
        )
        .unwrap();

    // No filter index declared anywhere in the store: Inactive.
    let (_, plan) = store
        .search_with_plan(&["col"], &[1.0, 0.0], &default_opts(5))
        .unwrap();
    assert_eq!(plan.narrowing, Narrowing::Inactive);

    store
        .set_filter_index("col", &[FilterIndexField::new("tag")])
        .unwrap();

    // `Eq` is not a text predicate the filter index can answer: Declined.
    let opts = SearchOpts {
        filter: Filter(vec![Predicate::Eq(
            "tag".to_string(),
            Value::Str("hello world".into()),
        )]),
        ..default_opts(5)
    };
    let (_, plan) = store
        .search_with_plan(&["col"], &[1.0, 0.0], &opts)
        .unwrap();
    assert_eq!(plan.narrowing, Narrowing::Declined);

    // `ContainsAllTokens` on the indexed field: Narrowed.
    let opts = SearchOpts {
        filter: Filter(vec![Predicate::ContainsAllTokens(
            "tag".to_string(),
            "hello".to_string(),
        )]),
        ..default_opts(5)
    };
    let (_, plan) = store
        .search_with_plan(&["col"], &[1.0, 0.0], &opts)
        .unwrap();
    assert!(matches!(plan.narrowing, Narrowing::Narrowed { .. }));
}

#[test]
fn plan_opt_out_search_ignores_the_plan_but_still_answers() {
    let data = random_unit_vectors(10, 3, 9);
    let store = exact_store(3, &data);
    let hits = store
        .search(&["col"], &[1.0, 0.0, 0.0], &default_opts(5))
        .unwrap();
    assert!(!hits.is_empty());
}
