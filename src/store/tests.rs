//! Tests for the store: pure-logic (Miri-clean) unit tests plus file-backed and
//! quantization/ANN behaviour. Lives beside the implementation it exercises; the
//! `pub(super)` quant-state fields let it assert on maintained index state.

use std::collections::BTreeMap;

use super::quant::{BinState, Int8State, Quant};
use super::scoring::PARALLEL_SCAN_WORK_FLOOR;
use super::*;
use crate::model::{Filter, Predicate, Quantization, Record, SearchOpts, Value};
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
        filter: Filter::default(),
        min_score: None,
    }
}

// ── Pure-logic tests (Miri-clean) ─────────────────────────────────────

#[test]
fn in_memory_dimension() {
    let store = Store::in_memory(4).unwrap();
    assert_eq!(store.dimension(), 4);
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
fn min_score_filters_low_results() {
    let mut store = Store::in_memory(3).unwrap();
    store.create_collection("col").unwrap();
    store
        .upsert("col", &[rec("doc1", vec![1.0, 0.0, 0.0])])
        .unwrap();
    // Query along y — score will be ~0.0, below min_score of 0.5.
    let opts = SearchOpts {
        top_k: 5,
        filter: Filter::default(),
        min_score: Some(0.5),
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
        min_score: None,
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
        config,
        data: Segments::in_memory_with(2, Distance::Cosine),
        log: OpLog::in_memory(),
        persistence: None,
        memory: None,
        lock: None,
        collections: HashMap::new(),
        dead_rows: 0,
        quant: None,
        ann: None,
        seg_indexes: Vec::new(),
        ann_dirty: false,
        fts: crate::fts::Fts::default(),
        fts_dirty: false,
        in_memory: true,
        row_to_doc: Vec::new(),
        scan_order: std::sync::RwLock::new(None),
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
        filter: Filter::default(),
        min_score: Some(-1.0),
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
    let hits = store.list(&["col"], &filter, 0, 100).unwrap();
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
    let hits = store.list(&["col"], &Filter::default(), 0, 3).unwrap();
    assert_eq!(hits.len(), 3);
}

#[test]
fn list_scores_are_zero() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("col").unwrap();
    store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    let hits = store.list(&["col"], &Filter::default(), 0, 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].score, 0.0);
}

#[test]
fn list_empty_filter_returns_all() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("col").unwrap();
    store.upsert("col", &[rec("a", vec![1.0, 0.0])]).unwrap();
    store.upsert("col", &[rec("b", vec![0.0, 1.0])]).unwrap();
    let hits = store.list(&["col"], &Filter::default(), 0, 100).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn list_multi_collection() {
    let mut store = Store::in_memory(2).unwrap();
    store.create_collection("a").unwrap();
    store.create_collection("b").unwrap();
    store.upsert("a", &[rec("x", vec![1.0, 0.0])]).unwrap();
    store.upsert("b", &[rec("y", vec![0.0, 1.0])]).unwrap();
    let hits = store.list(&["a", "b"], &Filter::default(), 0, 100).unwrap();
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
    let hits = store.list(&["col"], &Filter::default(), 0, 100).unwrap();
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
            .list(&["col"], &Filter::default(), page * 3, 3)
            .unwrap();
        paged.extend(hits.into_iter().map(|h| h.id));
    }
    let full: Vec<String> = store
        .list(&["col"], &Filter::default(), 0, 100)
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
    let hits = store.list(&["col"], &Filter::default(), 5, 10).unwrap();
    assert!(hits.is_empty());
}

// ── scan-order cache (nidus-dxt) ─────────────────────────────────────
//
// The whole-store fast path caches a row-sorted scan across queries; these pin
// that it stays consistent with the doc set — i.e. every write that changes the
// docs invalidates it, so a search after a write never reads a stale order.

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
        min_score: None,
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
        min_score: None,
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
#[cfg_attr(miri, ignore)]
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
#[cfg_attr(miri, ignore)]
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

// Ignored under Miri: needs enough work to clear PARALLEL_SCAN_WORK_FLOOR to engage
// the threaded path, which Miri runs at ~100x slowdown (minutes). The thread::scope
// scan is `#![forbid(unsafe_code)]` safe Rust over shared `&` reads — the borrow
// checker already proves it data-race-free, so Miri adds no coverage here.
#[cfg_attr(miri, ignore)]
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
#[cfg_attr(miri, ignore)]
#[test]
fn parallel_search_respects_filter_and_min_score() {
    let dim = 768;
    let n = rows_to_parallelize(dim) + 100;
    let parallel = threaded_store(dim, n, 4);
    let q: Vec<f32> = (0..dim).map(|d| (d * 5 % 13) as f32 - 6.0).collect();
    // A min_score floor must be honored across all worker chunks.
    let opts = SearchOpts {
        top_k: 30,
        filter: Filter::default(),
        min_score: Some(0.99),
    };
    let hits = parallel.search(&["col"], &q, &opts).unwrap();
    assert!(hits.iter().all(|h| h.score >= 0.99));
}

// The quantized first pass scales across threads; its parallel and serial candidate
// sets must produce the same final ranking. Ignored under Miri (same cost reason).
#[cfg_attr(miri, ignore)]
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
        min_score: None,
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

// ANN + quantization combined: the index walk runs in the quantized space and the f32
// rerank restores accuracy. Recall is necessarily a touch below the exact-walk ANN
// (the coarse codes steer the walk less precisely), so the thresholds are looser than
// the pure-ANN tests above — but well clear of chance.

#[test]
#[cfg_attr(miri, ignore)]
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
#[cfg_attr(miri, ignore)]
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
#[cfg_attr(miri, ignore)]
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
                min_score: None,
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
    let hits = store.list(&["col"], &Filter::default(), 0, 10).unwrap();
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

// ── Full-text search (BM25) ─────────────────────────────────────────────────

use crate::Language;
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
        .set_fts_schema("docs", &[("body".to_string(), Language::English)])
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
        .set_fts_schema("docs", &[("body".to_string(), Language::English)])
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
        .set_fts_schema("docs", &[("body".to_string(), Language::English)])
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
        min_score: None,
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
            .set_fts_schema("docs", &[("body".to_string(), Language::English)])
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
        .set_fts_schema("docs", &[("body".to_string(), Language::English)])
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
        .set_fts_schema("docs", &[("body".to_string(), Language::English)])
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
        .set_fts_schema("docs", &[("body".to_string(), Language::English)])
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
            .set_fts_schema("docs", &[("body".to_string(), Language::English)])
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
            .set_fts_schema("docs", &[("body".to_string(), Language::English)])
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
        store
            .set_fts_schema(c, &[("body".to_string(), Language::English)])
            .unwrap();
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

// ── Live object-store backing + shared memory tier (Miri-clean: all in-RAM) ──────
//
// Exercises the whole-object live-backing path (ObjectAppender rewrites the whole
// `data`/`log` object on each sync) and the shared memory tier (publish on flush, adopt
// on reopen) through a fake in-RAM whole-object Persistence — no files, no network, no
// fsync — so it runs under Miri alongside the rest of the store logic.

mod object_backed {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use anyhow::{Result, bail};

    use super::*;
    use crate::backend::{BackendLock, LocalRam, MemoryTier, Persistence};

    /// A whole-object Persistence backed by an in-RAM map: no native appender (forcing an
    /// `ObjectAppender`) and no native lock (forcing the advisory object lock) — exactly
    /// the shape S3/GCS present, but synchronous and Miri-clean.
    #[derive(Default)]
    struct InMemObjectStore {
        objects: Mutex<HashMap<String, Vec<u8>>>,
    }

    impl Persistence for InMemObjectStore {
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
            let mut keys: Vec<String> = self.objects.lock().unwrap().keys().cloned().collect();
            keys.sort();
            Ok(keys)
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
        let err = Store::open_with(cfg(), "s3://bucket/store", backend.clone(), None)
            .err()
            .expect("second open must be locked out");
        assert!(err.to_string().contains("locked"), "{err}");

        // Releasing the first lets a new writer in.
        drop(first);
        assert!(
            Store::open_with(cfg(), "s3://bucket/store", backend.clone(), None).is_ok(),
            "lock is reclaimable after the holder drops"
        );
    }
}
