#![cfg(target_family = "wasm")]
//! Node-run wasm suite (nidus-y67 U5): the clock hazard, the thread-degradation
//! clamp, and ranking correctness, all executed on `wasm32-unknown-unknown` via
//! `wasm-pack test --node`. No browser needed: the one store here that needs a
//! `Persistence` backend uses an in-process `SyncHandle`, never real OPFS — see
//! `tests/wasm_opfs/main.rs` for the suite that runs in a real browser.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use nidus::backend::{OpfsFs, SyncHandle, register_pool};
use nidus::{Config, Nidus, Record, SearchOpts, Value};
use wasm_bindgen_test::wasm_bindgen_test;

/// An in-RAM stand-in for a real OPFS sync access handle (pure Rust, no JS
/// calls). Reimplemented here rather than shared: `src/backend/opfs.rs`'s
/// own `test_support::FakeHandle` is `pub(crate)`, unreachable from here.
#[derive(Clone, Default)]
struct RamHandle(Arc<Mutex<Vec<u8>>>);

impl SyncHandle for RamHandle {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> nidus::Result<usize> {
        let data = self.0.lock().unwrap();
        let start = offset as usize;
        if start >= data.len() {
            return Ok(0);
        }
        let n = buf.len().min(data.len() - start);
        buf[..n].copy_from_slice(&data[start..start + n]);
        Ok(n)
    }

    fn write_at(&self, offset: u64, buf: &[u8]) -> nidus::Result<usize> {
        let mut data = self.0.lock().unwrap();
        let start = offset as usize;
        let end = start + buf.len();
        if end > data.len() {
            data.resize(end, 0);
        }
        data[start..end].copy_from_slice(buf);
        Ok(buf.len())
    }

    fn truncate(&self, size: u64) -> nidus::Result<()> {
        self.0.lock().unwrap().truncate(size as usize);
        Ok(())
    }

    fn size(&self) -> nidus::Result<u64> {
        Ok(self.0.lock().unwrap().len() as u64)
    }

    fn flush(&self) -> nidus::Result<()> {
        Ok(())
    }
}

/// A fresh pool of `body_slots` body slots (plus the directory slot at index 0),
/// registered on this thread so the next `opfs://…` open resolves it.
fn register_fresh_pool(body_slots: usize) {
    let handles: Vec<Box<dyn SyncHandle>> = (0..=body_slots)
        .map(|_| Box::new(RamHandle::default()) as Box<dyn SyncHandle>)
        .collect();
    register_pool(OpfsFs::adopt(handles).expect("adopt a fresh OPFS pool"));
}

/// A cheap deterministic xorshift64 generator, so two stores upserting "the
/// same data" get byte-identical vectors without pulling in an RNG crate.
fn synth_vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// The clock hazard (nidus-y67 U2): pre-fix, a wall-clock read here panics.
/// A bare `upsert` does not reach `clock::now_unix_millis()` by default
/// (`history_versions` is off); `reinforce` does, unconditionally.
#[wasm_bindgen_test]
fn upsert_then_reinforce_stamps_a_plausible_wall_clock_timestamp() {
    let mut db = Nidus::open_in_memory(3).expect("open an in-memory store");
    db.create_collection("docs").expect("create collection");

    let n = db
        .upsert(
            "docs",
            &[Record::new("doc1", vec![1.0, 0.0, 0.0], BTreeMap::new())],
        )
        .expect("upsert must return Ok, not panic on the wasm clock");
    assert_eq!(n, 1, "exactly one row committed");

    db.reinforce("docs", &["doc1"], None)
        .expect("reinforce must return Ok, not panic on the wasm clock");

    let rec = db.get("docs", "doc1").expect("doc1 must still be present");
    let stamped = match rec.attrs.get(nidus::META_LAST_ACCESSED) {
        Some(Value::DateTime(ms)) => *ms,
        other => panic!("expected a DateTime nidus.last_accessed attr, got {other:?}"),
    };
    // Floor, not an exact bound (mirrors src/clock.rs's own test): guards against
    // a clock stuck at 0 (a "did not panic" check alone would miss that), not skew.
    assert!(
        stamped > 1_577_836_800_000,
        "implausible epoch-ms timestamp: {stamped}"
    );
}

/// The thread hazard (nidus-y67 U2): `effective_query_threads` clamps to 1
/// on wasm, so `thread::scope` (store/scoring.rs) is never reached. Proven
/// by a real `query_threads(4)` store ranking identically to a serial run.
#[wasm_bindgen_test]
fn query_threads_above_one_degrades_to_identical_serial_ranking() {
    const DIM: usize = 4096;
    const ROWS: usize = 300; // 300 * 4096 ~= 1.23M, clears the 1<<20 floor with margin.

    let records: Vec<Record> = (0..ROWS)
        .map(|i| {
            Record::new(
                format!("r{i}"),
                synth_vector(i as u64, DIM),
                BTreeMap::new(),
            )
        })
        .collect();
    let query = synth_vector(0xDEAD_BEEF, DIM);
    let opts = SearchOpts {
        top_k: 20,
        ..Default::default()
    };

    // Known-serial baseline: `open_in_memory` always pins `query_threads` at 1
    // (Store::in_memory_with), so this cannot reach the parallel path at all.
    let mut serial = Nidus::open_in_memory(DIM).expect("open in-memory baseline");
    serial.create_collection("docs").expect("create collection");
    serial
        .upsert("docs", &records)
        .expect("baseline upsert must succeed");
    let baseline = serial
        .search("docs", &query, &opts)
        .expect("baseline search must succeed");
    assert!(
        !baseline.is_empty(),
        "baseline must actually rank something"
    );

    // Same data, `query_threads(4)` configured for real: the only public path
    // to a non-default `query_threads` is `Nidus::open`, which needs a real
    // backend even on wasm (no in-memory + custom-Config constructor exists).
    register_fresh_pool(8);
    let cfg = Config::new("opfs://wasm-main-thread-clamp", DIM).query_threads(4);
    let mut parallel = Nidus::open(cfg).expect("open opfs-backed store with query_threads=4");
    parallel
        .create_collection("docs")
        .expect("create collection");
    parallel
        .upsert("docs", &records)
        .expect("query_threads=4 upsert must succeed, not fail on thread::scope");
    let clamped = parallel
        .search("docs", &query, &opts)
        .expect("query_threads=4 search must succeed, not fail on thread::scope");

    assert_eq!(
        baseline.len(),
        clamped.len(),
        "the clamp must not silently narrow the result set"
    );
    for (b, c) in baseline.iter().zip(clamped.iter()) {
        assert_eq!(b.id, c.id, "ranking order must be identical");
        assert!(
            (b.score - c.score).abs() < 1e-4,
            "score mismatch for {}: serial {} vs query_threads=4 {}",
            b.id,
            b.score,
            c.score
        );
    }
}

/// Ranking correctness on wasm, not just "non-empty" (mirrors
/// `tests/e2e/scale.rs`): cosine computed independently over hand-written
/// vectors, compared against `Nidus::search`'s own order and scores.
#[wasm_bindgen_test]
fn search_ranking_matches_hand_computed_cosine_ground_truth() {
    let docs: Vec<(&str, Vec<f32>)> = vec![
        ("a", vec![1.0, 0.0, 0.0, 0.0]),
        ("b", vec![0.0, 1.0, 0.0, 0.0]),
        ("c", vec![0.9, 0.1, 0.0, 0.0]),
        ("d", vec![-1.0, 0.0, 0.0, 0.0]),
        ("e", vec![0.5, 0.5, 0.5, 0.5]),
        ("f", vec![0.2, 0.2, 0.9, 0.1]),
    ];
    let query = vec![0.8, 0.2, 0.1, 0.0];

    let mut expected: Vec<(&str, f32)> = docs
        .iter()
        .map(|(id, v)| (*id, cosine(&query, v)))
        .collect();
    expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut db = Nidus::open_in_memory(4).expect("open in-memory store");
    db.create_collection("docs").expect("create collection");
    let records: Vec<Record> = docs
        .iter()
        .map(|(id, v)| Record::new(*id, v.clone(), BTreeMap::new()))
        .collect();
    db.upsert("docs", &records)
        .expect("upsert ground-truth docs");

    let opts = SearchOpts {
        top_k: expected.len(),
        ..Default::default()
    };
    let hits = db
        .search("docs", &query, &opts)
        .expect("search must succeed");

    assert_eq!(hits.len(), expected.len());
    for (hit, (eid, escore)) in hits.iter().zip(expected.iter()) {
        assert_eq!(&hit.id, eid, "ranking order mismatch");
        assert!(
            (hit.score - escore).abs() < 1e-4,
            "score mismatch for {eid}: got {} want {escore}",
            hit.score
        );
    }
}
