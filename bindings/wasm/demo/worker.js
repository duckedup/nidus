// The dedicated worker that owns the wasm module and the OPFS handle pool for one nidus
// store (nidus-y67, U4). OPFS sync access handles exist only inside the worker that opened
// them, so `init_opfs_pool`/`NidusHandle.open`/every later call all happen right here.
import init, { NidusHandle, grow_opfs_pool, init_opfs_pool } from "../pkg/nidus_wasm.js";

const DIMENSION = 3;
const STORE_LOCATION = "opfs://nidus-demo";
const COLLECTION = "docs";
const SLOTS_PER_GROW = 8; // 1 directory slot + 7 body slots per batch of opened handles

let handle = null;
let root = null;
let nextSlotName = 0;

// The one piece of async JS work OPFS needs: getDirectory() plus one getFileHandle() +
// createSyncAccessHandle() per slot. Everything after this is synchronous Rust/wasm.
async function openSlots(count) {
  const opened = [];
  for (let i = 0; i < count; i++) {
    const name = `slot-${nextSlotName++}`;
    const fileHandle = await root.getFileHandle(name, { create: true });
    opened.push(await fileHandle.createSyncAccessHandle());
  }
  return opened;
}

async function setup() {
  await init();
  root = await navigator.storage.getDirectory();
  init_opfs_pool(await openSlots(SLOTS_PER_GROW));
  handle = NidusHandle.open(STORE_LOCATION, DIMENSION);
}

// `upsert` fails synchronously when the pool runs out of body slots (by design: growing
// needs an async step a sync write cannot perform). The fix is this retry loop: open more
// handles, hand them to `grow_opfs_pool`, then repeat the write — not a bug, the design.
async function withPoolGrowth(writeOnce) {
  try {
    return writeOnce();
  } catch (e) {
    if (!String(e).includes("OPFS pool exhausted")) throw e;
    grow_opfs_pool(await openSlots(SLOTS_PER_GROW));
    return writeOnce();
  }
}

// Records arrive from the page as plain `{tag: value}` JS objects; nidus's wire `Value`
// is externally tagged (`{"Str": "x"}`), so wrap each attr before crossing into wasm.
function toAttrs(plain) {
  const attrs = {};
  for (const [k, v] of Object.entries(plain || {})) {
    if (typeof v === "string") attrs[k] = { Str: v };
    else if (typeof v === "boolean") attrs[k] = { Bool: v };
    else if (typeof v === "number") attrs[k] = { Float: v };
  }
  return attrs;
}

self.onmessage = async (ev) => {
  const { id, cmd, payload } = ev.data;
  try {
    let result;
    switch (cmd) {
      case "init":
        await setup();
        break;
      case "upsert": {
        const records = payload.records.map((r) => ({
          id: r.id,
          vector: r.vector,
          attrs: toAttrs(r.attrs),
        }));
        result = await withPoolGrowth(() => handle.upsert(COLLECTION, records));
        break;
      }
      case "search":
        result = handle.search(COLLECTION, payload.query, payload.topK ?? 5);
        break;
      case "close":
        handle.close();
        handle = null;
        break;
      case "reopen":
        handle = NidusHandle.open(STORE_LOCATION, DIMENSION);
        break;
      default:
        throw new Error(`unknown worker command: ${cmd}`);
    }
    self.postMessage({ id, ok: true, result });
  } catch (e) {
    self.postMessage({ id, ok: false, error: e && e.message ? e.message : String(e) });
  }
};
