//! [`OpfsFs`]: a [`Persistence`] backend over a pre-opened pool of OPFS sync access-handle
//! slots (nidus-y67). OPFS gives synchronous file IO but only asynchronous handle
//! acquisition, and nidus mints new object keys at runtime (sealing a segment writes
//! `seg-NNNNNNNN`) — a synchronous `put()` for a brand-new key cannot itself acquire a
//! handle. So the design pre-opens every handle it will ever need, asynchronously, before
//! any sync call happens: the sqlite-wasm `opfs-sahpool` shape.
//!
//! Slot 0 holds a serialized directory map (`key -> slot`) and is the **commit point**;
//! slots `1..N` hold object bodies. `put` writes the body, `flush`es it, and only then
//! rewrites and flushes slot 0 — the same discipline as nidus's own `data`-then-`log`
//! fsync order (SPEC §6): a crash between the two steps leaves a body nothing references,
//! which is recoverable, never a torn or half-visible object.
//!
//! All async work (opening handles) happens in JS, outside this crate; this module only
//! ever performs synchronous reads/writes over already-open handles, abstracted behind
//! [`SyncHandle`] so the pool logic is host-testable (and Miri-clean) without a browser.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};

use super::{BackendLock, Persistence, validate_key};

/// The synchronous subset of `FileSystemSyncAccessHandle` (read/write/truncate/getSize/
/// flush) that one OPFS pool slot needs. U4's binding implements this over the real JS
/// handle; [`test_support::FakeHandle`] implements it over an in-RAM buffer for tests.
pub trait SyncHandle {
    /// Read up to `buf.len()` bytes starting at `offset`; returns the count actually read
    /// (short if `offset + buf.len()` runs past the slot's current length).
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize>;

    /// Write `buf` at `offset`, growing the slot if the write runs past its current
    /// length; returns the count actually written.
    fn write_at(&self, offset: u64, buf: &[u8]) -> Result<usize>;

    /// Discard any bytes beyond `size` (growing is undefined; callers only shrink).
    fn truncate(&self, size: u64) -> Result<()>;

    /// The slot's current committed length.
    fn size(&self) -> Result<u64>;

    /// Make prior writes durable (OPFS's own `flush`).
    fn flush(&self) -> Result<()>;
}

/// The pool state guarded by one lock: the slot handles and the directory map naming
/// which slot (`1..slots.len()`) holds each key's body. `slots[0]` is the directory slot.
struct Inner {
    slots: Vec<Box<dyn SyncHandle>>,
    map: BTreeMap<String, u32>,
}

/// A pre-opened OPFS handle pool ([`adopt`](OpfsFs::adopt)ed, never itself async). See the
/// module docs for the write-order invariant that makes it crash-safe.
pub struct OpfsFs {
    inner: Mutex<Inner>,
}

impl OpfsFs {
    /// Adopt a pool of already-opened handles: `handles[0]` is the directory slot,
    /// `handles[1..]` are body slots. Reads and validates the current directory map from
    /// slot 0 — all async acquisition already happened in JS before this call.
    pub fn adopt(handles: Vec<Box<dyn SyncHandle>>) -> Result<OpfsFs> {
        if handles.is_empty() {
            bail!("OPFS pool must have at least one slot (slot 0 is the directory map)");
        }
        let map = read_map(handles[0].as_ref())?;
        for &slot in map.values() {
            if slot == 0 || slot as usize >= handles.len() {
                bail!(
                    "OPFS directory map references out-of-range slot {slot} (pool has {} slots)",
                    handles.len()
                );
            }
        }
        Ok(OpfsFs {
            inner: Mutex::new(Inner {
                slots: handles,
                map,
            }),
        })
    }

    /// Total slots in the pool, including slot 0.
    pub fn capacity(&self) -> usize {
        self.lock().map(|g| g.slots.len()).unwrap_or(0)
    }

    /// Add freshly-opened handles to the pool. This is the binding's async-growth step,
    /// run **between** operations (never from inside a sync call — acquiring a handle is
    /// impossible from there), which is why `put`'s exhaustion error exists at all.
    pub fn grow(&self, new_handles: impl IntoIterator<Item = Box<dyn SyncHandle>>) -> Result<()> {
        self.lock()?.slots.extend(new_handles);
        Ok(())
    }

    fn lock(&self) -> Result<MutexGuard<'_, Inner>> {
        self.inner
            .lock()
            .map_err(|_| anyhow!("OPFS pool lock poisoned"))
    }
}

impl OpfsFs {
    pub(crate) fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        validate_key(key)?;
        let inner = self.lock()?;
        match inner.map.get(key) {
            Some(&slot) => Ok(Some(read_whole(inner.slots[slot as usize].as_ref())?)),
            None => Ok(None),
        }
    }

    pub(crate) fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        validate_key(key)?;
        let mut inner = self.lock()?;
        // Copy-on-write, never in place: `put` promises old-or-new, never torn, and
        // `allocate_slot` skips every slot the map references — so the key's current bytes
        // stay readable until the map flush below commits the new slot.
        let slot = allocate_slot(&inner)?;
        // Load-bearing order (module docs): body written and flushed FIRST, directory
        // map rewritten and flushed SECOND. Never reverse this.
        write_whole(inner.slots[slot as usize].as_ref(), bytes)?;
        inner.slots[slot as usize].flush()?;
        inner.map.insert(key.to_string(), slot);
        write_map(&inner)?;
        Ok(())
    }

    pub(crate) fn delete(&self, key: &str) -> Result<()> {
        validate_key(key)?;
        let mut inner = self.lock()?;
        if inner.map.remove(key).is_some() {
            write_map(&inner)?;
        }
        Ok(())
    }

    pub(crate) fn list(&self) -> Result<Vec<String>> {
        let inner = self.lock()?;
        let mut keys: Vec<String> = inner.map.keys().cloned().collect();
        keys.sort();
        Ok(keys)
    }

    pub(crate) fn try_lock(
        &self,
        _key: &str,
        _ttl: Duration,
    ) -> Result<Option<Box<dyn BackendLock>>> {
        // A browser store is single-writer by construction: one worker owns this pool,
        // and OPFS itself grants each sync access handle exclusively. No real second
        // holder can exist, so the guard is trivially always held.
        Ok(Some(Box::new(OpfsLock)))
    }
}

/// The trivial always-held guard `try_lock` returns — see its comment for why a real
/// exclusion protocol is unnecessary here.
struct OpfsLock;
impl BackendLock for OpfsLock {}

/// The first body slot (`1..slots.len()`) not currently referenced by the map, or a clear
/// exhaustion error naming the pool size and the async growth call that fixes it.
fn allocate_slot(inner: &Inner) -> Result<u32> {
    let used: HashSet<u32> = inner.map.values().copied().collect();
    (1..inner.slots.len() as u32)
        .find(|s| !used.contains(s))
        .ok_or_else(|| {
            anyhow!(
                "OPFS pool exhausted: all {} body slots are occupied ({} slots total \
             including the directory slot). Every write takes a FREE slot, overwrites \
             included, because a put must not tear an existing object; growing the pool \
             needs an async step (acquire more handles, then call \
             `nidus::backend::grow_pool`) that cannot happen inside this synchronous write",
                inner.slots.len().saturating_sub(1),
                inner.slots.len(),
            )
        })
}

/// Slot-0 codec: `[len: u32 LE][payload: bincode(map)][crc32: u32 LE]` — the same framing
/// discipline as the op-log (`src/log/mod.rs`), so a torn or corrupt directory map is
/// detectable rather than silently misread.
fn encode_map(map: &BTreeMap<String, u32>) -> Result<Vec<u8>> {
    let payload = bincode::serialize(map).context("failed to encode OPFS directory map")?;
    let len = u32::try_from(payload.len()).context("OPFS directory map too large for u32 len")?;
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    Ok(out)
}

fn decode_map(bytes: &[u8]) -> Result<BTreeMap<String, u32>> {
    if bytes.is_empty() {
        return Ok(BTreeMap::new());
    }
    if bytes.len() < 8 {
        bail!("OPFS directory map is truncated: {} bytes", bytes.len());
    }
    let len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    if bytes.len() != 8 + len {
        bail!(
            "OPFS directory map length mismatch: header says {len}, have {} bytes of payload",
            bytes.len().saturating_sub(8)
        );
    }
    let payload = &bytes[4..4 + len];
    let stored_crc = u32::from_le_bytes(bytes[4 + len..8 + len].try_into().unwrap());
    let computed_crc = crc32fast::hash(payload);
    if computed_crc != stored_crc {
        bail!(
            "OPFS directory map CRC mismatch: expected {computed_crc:#010x}, got {stored_crc:#010x}"
        );
    }
    bincode::deserialize(payload).context("failed to decode OPFS directory map")
}

fn read_whole(handle: &dyn SyncHandle) -> Result<Vec<u8>> {
    let len = usize::try_from(handle.size()?).context("OPFS slot larger than usize")?;
    // Fallible reserve (SPEC §6.6): an oversized slot surfaces an `Err`, never an abort.
    let mut buf = Vec::new();
    buf.try_reserve_exact(len)
        .map_err(|_| anyhow!("out of memory reading {len} bytes from an OPFS slot"))?;
    buf.resize(len, 0);
    let read = handle.read_at(0, &mut buf)?;
    buf.truncate(read);
    Ok(buf)
}

fn write_whole(handle: &dyn SyncHandle, bytes: &[u8]) -> Result<()> {
    let written = handle.write_at(0, bytes)?;
    if written != bytes.len() {
        bail!(
            "OPFS slot write incomplete: wrote {written} of {} bytes",
            bytes.len()
        );
    }
    handle.truncate(bytes.len() as u64)
}

fn read_map(handle: &dyn SyncHandle) -> Result<BTreeMap<String, u32>> {
    decode_map(&read_whole(handle)?)
}

fn write_map(inner: &Inner) -> Result<()> {
    write_whole(inner.slots[0].as_ref(), &encode_map(&inner.map)?)?;
    inner.slots[0].flush()
}

/// The `Persistence` the `opfs://` arm hands out: **fieldless on purpose**. The pool's
/// handles wrap a `!Send` `JsValue`, so keeping them in the thread_local rather than in a
/// trait object is what lets `Persistence` keep `Send + Sync` with no `unsafe`.
struct Shared;

/// Run `f` against this thread's registered pool, or fail naming the init call.
fn with_pool<T>(f: impl FnOnce(&OpfsFs) -> Result<T>) -> Result<T> {
    REGISTRY.with(|r| match r.borrow().as_ref() {
        Some(fs) => f(fs),
        None => Err(anyhow!(
            "no OPFS pool registered on this thread; call `nidus::backend::register_pool` \
             (the wasm binding's `init_opfs_pool`) before using an opfs:// store"
        )),
    })
}

impl Persistence for Shared {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        with_pool(|fs| fs.get(key))
    }
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        with_pool(|fs| fs.put(key, bytes))
    }
    fn delete(&self, key: &str) -> Result<()> {
        with_pool(|fs| fs.delete(key))
    }
    fn list(&self) -> Result<Vec<String>> {
        with_pool(|fs| fs.list())
    }
    fn try_lock(&self, key: &str, ttl: Duration) -> Result<Option<Box<dyn BackendLock>>> {
        with_pool(|fs| fs.try_lock(key, ttl))
    }
}

thread_local! {
    // Per-thread by design: OPFS sync access handles exist only inside the one worker
    // that opened them, so the pool they back is naturally that worker's own.
    static REGISTRY: RefCell<Option<OpfsFs>> = const { RefCell::new(None) };
}

/// Register a pool of already-opened handles (the wasm binding's `init_opfs_pool`) so a
/// later `Nidus::open("opfs://…")` on this thread can resolve it. Synchronous and does no
/// IO itself — all acquisition already happened in JS before this call.
pub fn register_pool(fs: OpfsFs) {
    REGISTRY.with(|r| *r.borrow_mut() = Some(fs));
}

/// Add freshly-opened handles to the pool already registered on this thread — the async
/// growth step `put`'s exhaustion error points callers at. Errors if none is registered.
pub fn grow_pool(new_handles: Vec<Box<dyn SyncHandle>>) -> Result<()> {
    REGISTRY.with(|r| {
        let guard = r.borrow();
        let fs = guard
            .as_ref()
            .context("cannot grow an OPFS pool before register_pool (init_opfs_pool) runs")?;
        fs.grow(new_handles)
    })
}

/// Resolve an `opfs://` location against this thread's registered pool, or a clear error
/// naming the initialisation call when none has been registered yet.
pub(crate) fn open_registered(location: &str) -> Result<Box<dyn Persistence>> {
    REGISTRY.with(|r| match r.borrow().as_ref() {
        Some(_) => Ok(Box::new(Shared) as Box<dyn Persistence>),
        None => Err(anyhow!(
            "opfs:// location {location:?} has no registered pool on this thread; call \
             `nidus::backend::register_pool` (the wasm binding's `init_opfs_pool`) \
             before opening a store here"
        )),
    })
}

/// A fake [`SyncHandle`] for pure-logic, Miri-clean pool tests — no browser required.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{Result, SyncHandle};
    use std::sync::{Arc, Mutex};

    /// An in-RAM OPFS slot. `Clone` shares the same underlying bytes, modelling a
    /// "reopen" of the same file with a fresh sync access handle.
    #[derive(Clone, Default)]
    pub(crate) struct FakeHandle(Arc<Mutex<Vec<u8>>>);

    impl FakeHandle {
        pub(crate) fn new() -> FakeHandle {
            FakeHandle::default()
        }

        /// The slot's raw bytes — lets a test see WHICH slot a write landed in, which is
        /// how the copy-on-write invariant is asserted rather than assumed.
        pub(crate) fn bytes(&self) -> Vec<u8> {
            self.0.lock().unwrap().clone()
        }
    }

    impl SyncHandle for FakeHandle {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
            let data = self.0.lock().unwrap();
            let start = offset as usize;
            if start >= data.len() {
                return Ok(0);
            }
            let n = buf.len().min(data.len() - start);
            buf[..n].copy_from_slice(&data[start..start + n]);
            Ok(n)
        }

        fn write_at(&self, offset: u64, buf: &[u8]) -> Result<usize> {
            let mut data = self.0.lock().unwrap();
            let start = offset as usize;
            let end = start + buf.len();
            if end > data.len() {
                data.resize(end, 0);
            }
            data[start..end].copy_from_slice(buf);
            Ok(buf.len())
        }

        fn truncate(&self, size: u64) -> Result<()> {
            self.0.lock().unwrap().truncate(size as usize);
            Ok(())
        }

        fn size(&self) -> Result<u64> {
            Ok(self.0.lock().unwrap().len() as u64)
        }

        fn flush(&self) -> Result<()> {
            Ok(())
        }
    }
}
