//! Real OPFS acquisition (nidus-21z): the async handshake
//! (`getDirectory` → `getFileHandle` → `createSyncAccessHandle`), a [`SyncHandle`] over one
//! live handle, and [`FaultyHandle`], a decorator that injects a real write failure for T3's
//! write-order counterfactual. No `web-sys`: dispatch is via `js_sys::Reflect`/`Function`,
//! mirroring `bindings/wasm/src/lib.rs`'s private `JsSyncHandle` (duplicated here — this
//! crate cannot depend on `bindings/wasm`, a dev-dependency cycle back through `nidus`
//! itself). Keep the two call shapes in sync by hand if the OPFS surface ever changes.

use std::cell::Cell;
use std::rc::Rc;

use js_sys::{Function, Object, Promise, Reflect, Uint8Array};
use nidus::backend::SyncHandle;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

/// Remove any prior run's subdirectory (OPFS is persistent per origin, so a stale slot file
/// would otherwise make the second run of these tests fail) and open `count` fresh handles
/// in it: index 0 is the directory slot, `1..count` are body slots.
pub async fn acquire_fresh(test_dir: &str, count: usize) -> anyhow::Result<Vec<JsSyncHandle>> {
    let root = get_directory().await?;
    remove_entry_tolerant(&root, test_dir).await?;
    let dir = get_directory_handle(&root, test_dir).await?;
    open_slots(&dir, count).await
}

/// Open `count` fresh handles on the SAME already-existing slot files under `test_dir`,
/// without removing anything first — a genuine reacquisition, for T2's reopen cycle.
pub async fn reacquire(test_dir: &str, count: usize) -> anyhow::Result<Vec<JsSyncHandle>> {
    let root = get_directory().await?;
    let dir = get_directory_handle(&root, test_dir).await?;
    open_slots(&dir, count).await
}

async fn open_slots(dir: &JsValue, count: usize) -> anyhow::Result<Vec<JsSyncHandle>> {
    let mut handles = Vec::with_capacity(count);
    for i in 0..count {
        let file = get_file_handle(dir, &format!("slot-{i}")).await?;
        handles.push(JsSyncHandle(create_sync_access_handle(&file).await?));
    }
    Ok(handles)
}

async fn get_directory() -> anyhow::Result<JsValue> {
    let global = js_sys::global();
    let navigator = Reflect::get(&global, &JsValue::from_str("navigator"))
        .map_err(|e| anyhow::anyhow!("no `navigator` on the worker global scope: {e:?}"))?;
    let storage = Reflect::get(&navigator, &JsValue::from_str("storage"))
        .map_err(|e| anyhow::anyhow!("no `navigator.storage`: {e:?}"))?;
    call_async(&storage, "getDirectory", &[]).await
}

async fn get_directory_handle(root: &JsValue, name: &str) -> anyhow::Result<JsValue> {
    call_async(
        root,
        "getDirectoryHandle",
        &[JsValue::from_str(name), bool_options("create", true)?],
    )
    .await
}

async fn get_file_handle(dir: &JsValue, name: &str) -> anyhow::Result<JsValue> {
    call_async(
        dir,
        "getFileHandle",
        &[JsValue::from_str(name), bool_options("create", true)?],
    )
    .await
}

async fn create_sync_access_handle(file: &JsValue) -> anyhow::Result<JsValue> {
    call_async(file, "createSyncAccessHandle", &[]).await
}

/// `root.removeEntry(name, {recursive: true})`, tolerating a `NotFoundError` rejection (a
/// clean profile has nothing to remove yet) — everything else propagates as a real failure.
async fn remove_entry_tolerant(root: &JsValue, name: &str) -> anyhow::Result<()> {
    let promise = call_promise(
        root,
        "removeEntry",
        &[JsValue::from_str(name), bool_options("recursive", true)?],
    )?;
    match JsFuture::from(promise).await {
        Ok(_) => Ok(()),
        Err(e) => {
            let kind = Reflect::get(&e, &JsValue::from_str("name"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default();
            if kind == "NotFoundError" {
                Ok(())
            } else {
                Err(anyhow::anyhow!("removeEntry({name}) rejected: {e:?}"))
            }
        }
    }
}

/// `{key: value}` as a plain JS object, e.g. `{create: true}` for `getFileHandle`.
fn bool_options(key: &str, value: bool) -> anyhow::Result<JsValue> {
    let opts = Object::new();
    Reflect::set(&opts, &JsValue::from_str(key), &JsValue::from_bool(value))
        .map_err(|e| anyhow::anyhow!("failed to build `{{{key}: {value}}}`: {e:?}"))?;
    Ok(opts.into())
}

/// `obj[name](args...)`, asserted to return a promise, awaited to completion.
async fn call_async(obj: &JsValue, name: &str, args: &[JsValue]) -> anyhow::Result<JsValue> {
    let promise = call_promise(obj, name, args)?;
    JsFuture::from(promise)
        .await
        .map_err(|e| anyhow::anyhow!("`{name}` rejected: {e:?}"))
}

fn call_promise(obj: &JsValue, name: &str, args: &[JsValue]) -> anyhow::Result<Promise> {
    let prop = Reflect::get(obj, &JsValue::from_str(name))
        .map_err(|e| anyhow::anyhow!("no `{name}`: {e:?}"))?;
    let f: Function = prop
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("`{name}` is not callable"))?;
    let result = match args.len() {
        0 => f.call0(obj),
        1 => f.call1(obj, &args[0]),
        2 => f.call2(obj, &args[0], &args[1]),
        n => return Err(anyhow::anyhow!("`{name}` called with {n} arguments")),
    }
    .map_err(|e| anyhow::anyhow!("`{name}` failed synchronously: {e:?}"))?;
    result
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("`{name}` did not return a promise"))
}

/// A live `FileSystemSyncAccessHandle`, dispatched by property name via `Reflect`/
/// `Function` rather than a typed `web-sys` binding — see the module doc for why. Cheap to
/// `Clone` (a JS-object-reference clone), so a copy can be retained for `close()` post-move.
#[derive(Clone)]
pub struct JsSyncHandle(JsValue);

impl JsSyncHandle {
    /// Close the handle (synchronous per spec — no promise). Real reacquisition (via
    /// [`reacquire`]) requires this: without it, T2 would just reuse the same live JS
    /// object rather than proving a fresh `createSyncAccessHandle` sees the same bytes.
    pub fn close(&self) -> anyhow::Result<()> {
        call_method(&self.0, "close", &[])?;
        Ok(())
    }
}

impl SyncHandle for JsSyncHandle {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> anyhow::Result<usize> {
        let view = Uint8Array::new_with_length(buf.len() as u32);
        let n = call_method(&self.0, "read", &[view.clone().into(), at_options(offset)?])?
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("OPFS read() did not return a number"))?;
        view.copy_to(buf);
        Ok((n as usize).min(buf.len()))
    }

    fn write_at(&self, offset: u64, buf: &[u8]) -> anyhow::Result<usize> {
        let view = Uint8Array::new_with_length(buf.len() as u32);
        view.copy_from(buf);
        let n = call_method(&self.0, "write", &[view.into(), at_options(offset)?])?
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("OPFS write() did not return a number"))?;
        Ok(n as usize)
    }

    fn truncate(&self, size: u64) -> anyhow::Result<()> {
        call_method(&self.0, "truncate", &[JsValue::from_f64(size as f64)])?;
        Ok(())
    }

    fn size(&self) -> anyhow::Result<u64> {
        let n = call_method(&self.0, "getSize", &[])?
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("OPFS getSize() did not return a number"))?;
        Ok(n as u64)
    }

    fn flush(&self) -> anyhow::Result<()> {
        call_method(&self.0, "flush", &[])?;
        Ok(())
    }
}

/// `{at: offset}`, the `FileSystemReadWriteOptions` shape `read`/`write` take.
fn at_options(offset: u64) -> anyhow::Result<JsValue> {
    let opts = Object::new();
    Reflect::set(
        &opts,
        &JsValue::from_str("at"),
        &JsValue::from_f64(offset as f64),
    )
    .map_err(|e| anyhow::anyhow!("failed to build OPFS read/write options: {e:?}"))?;
    Ok(opts.into())
}

/// Call `obj[name](args...)` synchronously (the sync-handle methods themselves, unlike
/// acquisition, never return promises).
fn call_method(obj: &JsValue, name: &str, args: &[JsValue]) -> anyhow::Result<JsValue> {
    let prop = Reflect::get(obj, &JsValue::from_str(name))
        .map_err(|e| anyhow::anyhow!("OPFS handle has no `{name}`: {e:?}"))?;
    let f: Function = prop
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("OPFS handle's `{name}` is not callable"))?;
    let result = match args.len() {
        0 => f.call0(obj),
        1 => f.call1(obj, &args[0]),
        2 => f.call2(obj, &args[0], &args[1]),
        n => return Err(anyhow::anyhow!("OPFS `{name}` called with {n} arguments")),
    };
    result.map_err(|e| anyhow::anyhow!("OPFS `{name}` failed: {e:?}"))
}

/// Decorates any [`SyncHandle`] with an arm-able failure on `write_at` — T3's counterfactual
/// for the write-order invariant (the real handle stays underneath). `fail_writes` is
/// `Rc<Cell<bool>>` so `.clone()` shares one flag between the pool's copy and the test's.
#[derive(Clone)]
pub struct FaultyHandle<H> {
    inner: H,
    fail_writes: Rc<Cell<bool>>,
}

impl<H> FaultyHandle<H> {
    pub fn new(inner: H) -> Self {
        Self {
            inner,
            fail_writes: Rc::new(Cell::new(false)),
        }
    }

    /// Arm the injected failure: the next `write_at` (and every one after) returns `Err`.
    pub fn arm(&self) {
        self.fail_writes.set(true);
    }
}

impl<H: SyncHandle> SyncHandle for FaultyHandle<H> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> anyhow::Result<usize> {
        self.inner.read_at(offset, buf)
    }

    fn write_at(&self, offset: u64, buf: &[u8]) -> anyhow::Result<usize> {
        if self.fail_writes.get() {
            anyhow::bail!("injected OPFS write failure");
        }
        self.inner.write_at(offset, buf)
    }

    fn truncate(&self, size: u64) -> anyhow::Result<()> {
        self.inner.truncate(size)
    }

    fn size(&self) -> anyhow::Result<u64> {
        self.inner.size()
    }

    fn flush(&self) -> anyhow::Result<()> {
        self.inner.flush()
    }
}
