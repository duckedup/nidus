//! `wasm-bindgen` binding over nidus for the browser (nidus-y67, U4).
//!
//! Two exported surfaces: the OPFS pool handshake (`init_opfs_pool`, `grow_opfs_pool`) that
//! adopts already-open `FileSystemSyncAccessHandle`s from a worker, and [`NidusHandle`], a
//! thin wrapper over `Nidus` (open/upsert/search/flush/close). Both MUST run on the one
//! worker thread that opened the handles — the pool nidus registers is a `thread_local`.

#![deny(unsafe_code)]

use std::collections::BTreeMap;

use js_sys::{Function, Object, Reflect, Uint8Array};
use nidus::backend::{OpfsFs, SyncHandle, grow_pool, register_pool};
use nidus::{Config, Nidus, Record, SearchOpts, Value};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Adopt a fresh pool of already-opened OPFS handles (`handles[0]` is the directory slot,
/// `handles[1..]` are body slots) and register it on this worker thread. Must run before any
/// `NidusHandle::open("opfs://…")` call on the same thread.
#[wasm_bindgen]
pub fn init_opfs_pool(handles: js_sys::Array) -> Result<(), JsValue> {
    let fs = OpfsFs::adopt(wrap_handles(handles)).map_err(js_err)?;
    register_pool(fs);
    Ok(())
}

/// Add freshly-opened handles to the pool already registered on this thread — the async
/// growth step a `put` exhaustion error asks for. Same thread as `init_opfs_pool` only.
#[wasm_bindgen]
pub fn grow_opfs_pool(handles: js_sys::Array) -> Result<(), JsValue> {
    grow_pool(wrap_handles(handles)).map_err(js_err)
}

fn wrap_handles(handles: js_sys::Array) -> Vec<Box<dyn SyncHandle>> {
    handles
        .iter()
        .map(|h| Box::new(JsSyncHandle(h)) as Box<dyn SyncHandle>)
        .collect()
}

/// An open store. Wraps `nidus::Nidus`; every method must run on the worker thread that
/// registered its OPFS pool (for an `opfs://` location).
#[wasm_bindgen]
pub struct NidusHandle(Nidus);

#[wasm_bindgen]
impl NidusHandle {
    /// Open (creating if absent) a store at `location` — `opfs://name` (needs
    /// `init_opfs_pool` first, same thread) or `file://…`/a bare path.
    pub fn open(location: &str, dimension: u32) -> Result<NidusHandle, JsValue> {
        let cfg = Config::new(location, dimension as usize).persistence(location);
        Nidus::open(cfg).map(NidusHandle).map_err(js_err)
    }

    /// Upsert records given as a JS array of `{id, vector?, attrs}` (mirrors
    /// `server::dto::UpsertRequest`'s `Record` shape); returns the count written.
    pub fn upsert(&mut self, collection: &str, records: JsValue) -> Result<u32, JsValue> {
        let records: Vec<Record> = serde_wasm_bindgen::from_value(records).map_err(js_value_err)?;
        let n = self.0.upsert(collection, &records).map_err(js_err)?;
        Ok(n as u32)
    }

    /// Nearest-neighbour search in `collection`, returning a JS array of hits shaped like
    /// `server::dto::HitDto`: `{collection, id, score, attrs}`.
    pub fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        top_k: u32,
    ) -> Result<JsValue, JsValue> {
        let opts = SearchOpts {
            top_k: top_k as usize,
            ..Default::default()
        };
        let hits = self.0.search(collection, &query, &opts).map_err(js_err)?;
        let dtos: Vec<HitDto> = hits.into_iter().map(HitDto::from).collect();
        serde_wasm_bindgen::to_value(&dtos).map_err(js_value_err)
    }

    /// fsync both files.
    pub fn flush(&mut self) -> Result<(), JsValue> {
        self.0.flush().map_err(js_err)
    }

    /// Close the store. Consumes the handle: nidus's `Drop` releases the writer lock (a
    /// trivial always-held guard on OPFS), and the JS wrapper is invalidated with it.
    pub fn close(self) {}
}

/// Serializable mirror of `crate::Hit` (which carries no serde derive) — same shape as
/// `server::dto::HitDto`, minus the annotation/context fields this minimal surface never sets.
#[derive(Serialize)]
struct HitDto {
    collection: String,
    id: String,
    score: f32,
    attrs: BTreeMap<String, Value>,
}

impl From<nidus::Hit> for HitDto {
    fn from(h: nidus::Hit) -> Self {
        Self {
            collection: h.collection,
            id: h.id,
            score: h.score,
            attrs: h.attrs,
        }
    }
}

/// Wraps one live `FileSystemSyncAccessHandle` as an opaque `JsValue`. Calls are dispatched
/// by property name via `Reflect`/`Function` (like `nidus::clock`'s `performance_now_ms`)
/// rather than typed `web-sys` bindings, so this has no dependency on a generated method name.
struct JsSyncHandle(JsValue);

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

/// A `FileSystemReadWriteOptions`-shaped plain object: `{at: offset}`.
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

/// Call `obj[name](args...)`, dispatched by property lookup rather than a typed binding —
/// see [`JsSyncHandle`]'s doc comment for why.
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

/// Flatten an `anyhow::Error`'s whole cause chain into the `JsValue` thrown to JS, so a
/// wrapped error (e.g. pool exhaustion surfacing through a commit) still names its root cause.
fn js_err(err: anyhow::Error) -> JsValue {
    let mut msg = err.to_string();
    for cause in err.chain().skip(1) {
        msg.push_str(": ");
        msg.push_str(&cause.to_string());
    }
    JsValue::from_str(&msg)
}

fn js_value_err(err: serde_wasm_bindgen::Error) -> JsValue {
    JsValue::from_str(&err.to_string())
}
