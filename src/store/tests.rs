//! Tests for the store: pure-logic (Miri-clean) unit tests plus file-backed and
//! quantization/ANN behaviour. Lives beside the implementation it exercises; the
//! `pub(super)` quant-state fields let it assert on maintained index state.

use std::collections::BTreeMap;

use super::quant::{BinState, Int8State, Quant};
use super::scoring::PARALLEL_SCAN_WORK_FLOOR;
use super::*;
use crate::Fsync;
use crate::model::{Filter, ListOpts, Predicate, Quantization, Record, SearchOpts, Value};
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
        config,
        data: Segments::in_memory_with(2, Distance::Cosine),
        log: OpLog::in_memory(),
        persistence: None,
        memory: None,
        lock: None,
        lease: None,
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
        loaded_log_offset: 0,
        manifest_cas: None,
        defer_barrier: false,
        pending_barrier: false,
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

// Ignored under Miri: clearing PARALLEL_SCAN_WORK_FLOOR takes minutes at Miri's ~100x slowdown,
// and the scan is safe Rust over shared `&` reads that the borrow checker already proves
// data-race-free, so Miri adds no coverage.
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
        min_score: Some(0.99),
        ..Default::default()
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
    let (collections, _dead, fts) = Store::replay_ops(ops, 0);
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
