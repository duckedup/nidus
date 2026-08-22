#![cfg(target_family = "wasm")]
//! Real-headless-browser wasm suite (nidus-y67 U5): the OPFS handle pool
//! through a real `Store::open`, run via `wasm_bindgen_test_configure!(run_in_browser)`.
//!
//! U4's JS binding (the async `navigator.storage.getDirectory()` handshake that
//! acquires real `FileSystemSyncAccessHandle`s from a worker) is being written in
//! parallel and is not in this tree, so these tests drive
//! `nidus::backend::{SyncHandle, OpfsFs, register_pool, grow_pool}` directly with
//! an in-process handle instead of a real one — see the session report for what
//! swapping in U4's real acquisition still needs to prove (real OPFS durability
//! across a real page reload, which an in-process handle cannot exercise).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use nidus::backend::{OpfsFs, SyncHandle, open_persistence, register_pool};
use nidus::{Config, Nidus, Record, SearchOpts};
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// An in-RAM stand-in for a real OPFS sync access handle — see `tests/wasm/main.rs`
/// for why this is reimplemented here rather than shared: `src/backend/opfs.rs`'s
/// own `test_support::FakeHandle` is `pub(crate)`, unreachable from an integration test.
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

fn as_handles(fakes: &[RamHandle]) -> Vec<Box<dyn SyncHandle>> {
    fakes
        .iter()
        .cloned()
        .map(|h| Box::new(h) as Box<dyn SyncHandle>)
        .collect()
}

fn register_fresh_pool(body_slots: usize) {
    let fakes: Vec<RamHandle> = (0..=body_slots).map(|_| RamHandle::default()).collect();
    register_pool(OpfsFs::adopt(as_handles(&fakes)).expect("adopt a fresh OPFS pool"));
}

/// (b)(1): a real pool (in-process handle; see the module doc), a real
/// `Store::open("opfs://…")`, upsert, search, ranking check — the store's whole
/// life over the OPFS-shaped persistence path in one real `Nidus`.
#[wasm_bindgen_test]
fn upsert_and_search_over_a_fresh_opfs_pool() {
    register_fresh_pool(8);
    let mut db = Nidus::open(Config::new("opfs://wasm-e2e-basic", 4)).expect("open opfs:// store");
    db.create_collection("docs").expect("create collection");
    db.upsert(
        "docs",
        &[
            Record::new("a", vec![1.0, 0.0, 0.0, 0.0], BTreeMap::new()),
            Record::new("b", vec![0.0, 1.0, 0.0, 0.0], BTreeMap::new()),
            Record::new("c", vec![0.0, 0.0, 1.0, 0.0], BTreeMap::new()),
        ],
    )
    .expect("upsert over opfs:// must succeed");

    let hits = db
        .search(
            "docs",
            &[0.9, 0.1, 0.0, 0.0],
            &SearchOpts {
                top_k: 3,
                ..Default::default()
            },
        )
        .expect("search over opfs:// must succeed");
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].id, "a", "closest vector must rank first");
}

/// (b)(2): durability across a reopen — the same underlying slot bytes
/// adopted twice, proving slot 0's directory map and the body-then-map
/// write order (`src/backend/opfs.rs`) survive a real close and reopen.
#[wasm_bindgen_test]
fn durability_survives_a_full_close_and_reopen() {
    let fakes: Vec<RamHandle> = (0..=8).map(|_| RamHandle::default()).collect();

    register_pool(OpfsFs::adopt(as_handles(&fakes)).expect("adopt pool for the writer"));
    {
        let mut db = Nidus::open(Config::new("opfs://wasm-e2e-reopen", 4))
            .expect("open opfs:// store (writer)");
        db.create_collection("docs").expect("create collection");
        db.upsert(
            "docs",
            &[
                Record::new("a", vec![1.0, 0.0, 0.0, 0.0], BTreeMap::new()),
                Record::new("b", vec![0.0, 1.0, 0.0, 0.0], BTreeMap::new()),
            ],
        )
        .expect("upsert must succeed");
        db.flush().expect("flush must succeed");
        // `db` drops here: the writer lock (an OPFS-object CAS guard, not a real
        // file lock) releases, exactly like a worker tearing down after a commit.
    }

    // A fresh "worker" adopts the SAME underlying slots — a real page reload
    // would reacquire the same OPFS handles the same way.
    register_pool(OpfsFs::adopt(as_handles(&fakes)).expect("re-adopt pool after reopen"));
    let reopened =
        Nidus::open(Config::new("opfs://wasm-e2e-reopen", 4)).expect("reopen opfs:// store");
    let hits = reopened
        .search(
            "docs",
            &[0.9, 0.1, 0.0, 0.0],
            &SearchOpts {
                top_k: 2,
                ..Default::default()
            },
        )
        .expect("search after reopen must succeed");
    assert_eq!(hits.len(), 2, "both rows must survive the reopen");
    assert_eq!(hits[0].id, "a");
    assert_eq!(hits[1].id, "b");
}

/// (b)(3): pool exhaustion is a clean, named `Err`, and prior writes survive.
/// `segment_max_rows(1)` mints a fresh key every upsert against a tiny pool;
/// this drives upserts until one fails rather than hard-coding which row.
#[wasm_bindgen_test]
fn pool_exhaustion_is_a_named_error_and_prior_writes_survive() {
    // 1 directory slot + 4 body slots: comfortably fits manifest/data/log (no
    // "lock" object — see the module-level `has_native_lock` note in the
    // session report) with one spare, but not a second seal's worth of keys.
    register_fresh_pool(4);
    let cfg = Config::new("opfs://wasm-e2e-exhaustion", 4).segment_max_rows(Some(1));
    let mut db = Nidus::open(cfg).expect("open opfs:// store");
    db.create_collection("docs").expect("create collection");

    let basis = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut committed: Vec<String> = Vec::new();
    // Format the error to a `String` immediately: `anyhow::Error`'s own type is
    // not nameable here (`anyhow` is nidus's dependency, not this test crate's).
    let mut failure: Option<(String, String)> = None;
    for i in 0..8 {
        let id = format!("r{i}");
        let v = basis[i % basis.len()].to_vec();
        match db.upsert("docs", &[Record::new(id.clone(), v, BTreeMap::new())]) {
            Ok(_) => committed.push(id),
            Err(e) => {
                failure = Some((id, format!("{e:#}")));
                break;
            }
        }
    }

    let (failed_id, msg) = failure.expect(
        "a 4-body-slot pool must exhaust within 8 sealing rows; if this fires, the pool \
         needs sizing down rather than the test loosening",
    );
    assert!(
        msg.contains("pool"),
        "error must name the pool as the cause, got: {msg}"
    );
    assert!(
        !committed.is_empty(),
        "the first upsert needs no seal at all, so at least one row must commit \
         before exhaustion"
    );

    // Nothing written before exhaustion was corrupted or lost.
    for id in &committed {
        assert!(db.get("docs", id).is_some(), "{id} must survive exhaustion");
    }
    assert!(
        db.get("docs", &failed_id).is_none(),
        "{failed_id} must not appear half-written"
    );
    let hits = db
        .search(
            "docs",
            &[1.0, 0.0, 0.0, 0.0],
            &SearchOpts {
                top_k: 20,
                ..Default::default()
            },
        )
        .expect("search must still work after a failed write");
    assert_eq!(
        hits.len(),
        committed.len(),
        "only the pre-exhaustion rows are live"
    );
}

/// (b)(4): the write-order invariant, via the public `Persistence` object
/// `open_persistence` hands back (`OpfsFs`'s own get/put/list are
/// `pub(crate)`). Mirrors the host-side unit test in `src/backend/tests.rs`.
#[wasm_bindgen_test]
fn a_body_write_without_a_map_update_stays_invisible() {
    let fakes: Vec<RamHandle> = (0..=1).map(|_| RamHandle::default()).collect();
    register_pool(OpfsFs::adopt(as_handles(&fakes)).expect("adopt pool"));
    let backend =
        open_persistence("opfs://wasm-e2e-write-order").expect("resolve the registered pool");

    // Simulate a crash between step 1 (body write) and step 2 (map write): write
    // straight into the body slot, bypassing `put`, so the directory map never
    // learns about it.
    fakes[1]
        .write_at(0, b"orphaned")
        .expect("direct body write");

    assert!(
        backend.list().expect("list must succeed").is_empty(),
        "an orphaned body must not appear in list()"
    );
    assert!(
        backend.get("ghost").expect("get must succeed").is_none(),
        "no key names the orphaned slot, so get() must find nothing"
    );
    // The slot is still unreferenced by the map, so a real put safely reuses it
    // rather than treating it as already occupied.
    backend
        .put("real", b"data")
        .expect("put must reuse the free slot");
    assert_eq!(
        backend.get("real").expect("get must succeed").as_deref(),
        Some(b"data".as_slice())
    );
}
