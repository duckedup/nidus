#![cfg(target_family = "wasm")]
//! Real-browser wasm suite (nidus-21z, following nidus-y67 U5): the OPFS handle pool
//! through a real `Store::open`. Two kinds of coverage live here side by side: four
//! in-RAM `RamHandle` tests exercising pool logic (fresh-pool search, reopen, exhaustion,
//! orphaned writes) cheaply and context-agnostically, and `opfs_handle`-backed tests below
//! them that drive genuine `FileSystemSyncAccessHandle`s acquired via
//! `navigator.storage.getDirectory()` — the real OPFS path U5 could not yet reach because
//! the binding's handshake was still being written in parallel.
//!
//! Configured `run_in_dedicated_worker`, not `run_in_browser`: a sync access handle exists
//! only inside the one thread that opened it (SPEC §13.8), so acquiring one at all — not
//! just using it — requires this whole binary to run in a worker.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use nidus::backend::{OpfsFs, SyncHandle, open_persistence, register_pool};
use nidus::{Config, Nidus, Record, SearchOpts};
use wasm_bindgen_test::wasm_bindgen_test;

mod opfs_handle;
use opfs_handle::{FaultyHandle, acquire_fresh, reacquire};

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_dedicated_worker);

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

fn boxed(handles: Vec<opfs_handle::JsSyncHandle>) -> Vec<Box<dyn SyncHandle>> {
    handles
        .into_iter()
        .map(|h| Box::new(h) as Box<dyn SyncHandle>)
        .collect()
}

/// T1 (nidus-21z): real acquisition through `navigator.storage.getDirectory()`, driving a
/// real `Nidus::open("opfs://…")` end to end — upsert, search, rank.
#[wasm_bindgen_test]
async fn upsert_and_search_over_a_real_opfs_pool() {
    let handles = acquire_fresh("t1-basic", 9)
        .await
        .expect("acquire 1 real directory slot + 8 real body slots");
    register_pool(OpfsFs::adopt(boxed(handles)).expect("adopt a fresh real OPFS pool"));

    let mut db =
        Nidus::open(Config::new("opfs://wasm-e2e-t1-basic", 4)).expect("open a real opfs:// store");
    db.create_collection("docs").expect("create collection");
    db.upsert(
        "docs",
        &[
            Record::new("a", vec![1.0, 0.0, 0.0, 0.0], BTreeMap::new()),
            Record::new("b", vec![0.0, 1.0, 0.0, 0.0], BTreeMap::new()),
            Record::new("c", vec![0.0, 0.0, 1.0, 0.0], BTreeMap::new()),
        ],
    )
    .expect("upsert over a real opfs:// store must succeed");

    let hits = db
        .search(
            "docs",
            &[0.9, 0.1, 0.0, 0.0],
            &SearchOpts {
                top_k: 3,
                ..Default::default()
            },
        )
        .expect("search over a real opfs:// store must succeed");
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].id, "a", "closest vector must rank first");
}

/// T2 (nidus-21z), the acceptance criterion: durability AND ranking survive a real
/// close-and-reacquire cycle on the SAME underlying OPFS files, not just the same
/// in-process objects — `close()` is called on every handle before reacquiring.
#[wasm_bindgen_test]
async fn durability_survives_a_real_close_and_reopen() {
    let dir = "t2-reopen";
    let writer = acquire_fresh(dir, 9)
        .await
        .expect("acquire real handles for the writer");
    register_pool(OpfsFs::adopt(boxed(writer.clone())).expect("adopt pool for the writer"));
    {
        let mut db = Nidus::open(Config::new("opfs://wasm-e2e-t2-reopen", 4))
            .expect("open a real opfs:// store (writer)");
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
        // `db` drops here: the writer lock releases, exactly like a worker tearing
        // down after a commit.
    }
    for h in &writer {
        h.close().expect("close the real sync access handle");
    }

    // A fresh acquisition on the SAME slot files — the real reacquisition a page
    // reload performs, not the same live JS objects reused.
    let reopened = reacquire(dir, 9)
        .await
        .expect("re-acquire real handles after close");
    register_pool(OpfsFs::adopt(boxed(reopened)).expect("re-adopt pool after reopen"));
    let db = Nidus::open(Config::new("opfs://wasm-e2e-t2-reopen", 4))
        .expect("reopen a real opfs:// store");
    let hits = db
        .search(
            "docs",
            &[0.9, 0.1, 0.0, 0.0],
            &SearchOpts {
                top_k: 2,
                ..Default::default()
            },
        )
        .expect("search after reopen must succeed");
    assert_eq!(hits.len(), 2, "both rows must survive the real reopen");
    assert_eq!(hits[0].id, "a");
    assert_eq!(hits[1].id, "b");
}

/// T3 (nidus-21z): the write-order invariant with a REAL failing mode, over real OPFS
/// handles wrapped in `FaultyHandle`. Unlike `a_body_write_without_a_map_update_stays_invisible`
/// (passes under either order), this fails if `put`'s body-then-map order is ever inverted.
#[wasm_bindgen_test]
async fn a_faulty_body_write_never_corrupts_the_directory_map() {
    let raw = acquire_fresh("t3-write-order", 3)
        .await
        .expect("acquire 1 real directory slot + 2 real body slots");
    let faulty: Vec<FaultyHandle<opfs_handle::JsSyncHandle>> =
        raw.into_iter().map(FaultyHandle::new).collect();
    let pool: Vec<Box<dyn SyncHandle>> = faulty
        .iter()
        .cloned()
        .map(|f| Box::new(f) as Box<dyn SyncHandle>)
        .collect();
    register_pool(OpfsFs::adopt(pool).expect("adopt real pool for the write-order test"));
    let backend =
        open_persistence("opfs://wasm-e2e-t3-write-order").expect("resolve the registered pool");

    backend.put("k", b"v1").expect("first put must succeed");
    // `allocate_slot` skips slot 1 (referenced by "k"), so the next body write
    // deterministically lands on slot 2 — arm the fault there.
    faulty[2].arm();

    let err = backend
        .put("k", b"v2")
        .expect_err("a faulty body write must fail the whole put");
    assert!(
        format!("{err:#}").contains("injected OPFS write failure"),
        "error must name the injected fault, got: {err:#}"
    );

    // THE assertion: body-then-map order means the map never learned about the
    // failed slot 2 write, so "k" still resolves to its old, intact body.
    assert_eq!(
        backend.get("k").expect("get must succeed").as_deref(),
        Some(b"v1".as_slice()),
        "a failed write must never corrupt the previously-committed value"
    );
    assert_eq!(backend.list().expect("list must succeed"), vec!["k".to_string()]);
}
