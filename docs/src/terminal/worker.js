// The docs terminal's dedicated worker (nidus-7pj): owns the wasm module, the OPFS
// pool, and the one live store. Mirrors bindings/wasm/demo/worker.js: see its
// comments for why this must run on one dedicated thread.
import { DIM, hashVector, SEEDS } from "./corpus.js";

const DIMENSION = DIM;
const STORE_LOCATION = "opfs://nidus-docs-demo";
// The pool's slot files, and so the key->slot map in slot 0, are shared by every store
// opened against them. The store name alone therefore isolates nothing: keying this
// DIRECTORY on DIM is what gives a reader a fresh store when the vectors change, rather
// than "manifest has 64, requested 256" on a store they cannot clear.
const STORE_DIR = `nidus-docs-demo-d${DIM}`;
const COLLECTION = "docs";
const SLOTS_PER_GROW = 8;

let wasmMod = null;
let handle = null;
let root = null;
let nextSlotName = 0;

async function openSlots(count) {
  const opened = [];
  for (let i = 0; i < count; i++) {
    const name = `slot-${nextSlotName++}`;
    const fileHandle = await root.getFileHandle(name, { create: true });
    opened.push(await fileHandle.createSyncAccessHandle());
  }
  return opened;
}

// Growing the pool needs an async step a sync write cannot perform, so pool
// exhaustion throws "OPFS pool exhausted"; catch it once and retry after growing.
async function withPoolGrowth(writeOnce) {
  try {
    return writeOnce();
  } catch (e) {
    if (!String(e).includes("OPFS pool exhausted")) throw e;
    wasmMod.grow_opfs_pool(await openSlots(SLOTS_PER_GROW));
    return writeOnce();
  }
}

// nidus's wire `Value` is externally tagged (`{"Str": "x"}`); a bare JS value
// crossing the boundary fails deserialization, so wrap every attr first.
function toAttrs(plain) {
  const attrs = {};
  for (const [k, v] of Object.entries(plain || {})) {
    if (typeof v === "string") attrs[k] = { Str: v };
    else if (typeof v === "boolean") attrs[k] = { Bool: v };
    else if (typeof v === "number") attrs[k] = { Float: v };
  }
  return attrs;
}

// Idempotent: upsert is keyed by id, so re-seeding an already-seeded store just
// overwrites the same rows.
async function seed() {
  const records = SEEDS.map((s) => ({
    id: s.id,
    vector: hashVector(s.text),
    attrs: toAttrs(s.attrs),
  }));
  await withPoolGrowth(() => handle.upsert(COLLECTION, records));
}

// Try the OPFS path; any failure (unsupported API, a non-secure context, a denied
// permission) falls back to an in-memory store. A reliable in-memory terminal beats
// an intermittent persistent one, and the caller is told which path won.
async function setup() {
  if (!wasmMod) {
    // A literal specifier on purpose: Vite must see it to emit the .wasm the glue
    // loads via `new URL(..., import.meta.url)`. copy-wasm.mjs guarantees the path
    // exists, writing a throwing stub when no artifact was built.
    wasmMod = await import("../generated/nidus-wasm/nidus_wasm.js");
    await wasmMod.default();
  }
  try {
    const opfsRoot = await navigator.storage.getDirectory();
    root = await opfsRoot.getDirectoryHandle(STORE_DIR, { create: true });
    wasmMod.init_opfs_pool(await openSlots(SLOTS_PER_GROW));
    handle = wasmMod.NidusHandle.open(STORE_LOCATION, DIMENSION);
    // Count BEFORE seeding: this is what survived the reader's last visit, which is
    // the whole point of persisting, and seeding would otherwise mask it.
    const restored = handle.footprint().doc_count;
    await seed();
    return { mode: "opfs", restored };
  } catch (e) {
    handle = wasmMod.NidusHandle.open_in_memory(DIMENSION);
    await seed();
    return { mode: "memory", reason: e && e.message ? e.message : String(e) };
  }
}

self.onmessage = async (ev) => {
  const { id, cmd, payload } = ev.data;
  try {
    let result;
    switch (cmd) {
      case "init":
        result = await setup();
        break;
      case "upsert": {
        const records = [{ id: payload.id, vector: hashVector(payload.text), attrs: toAttrs({}) }];
        result = await withPoolGrowth(() => handle.upsert(COLLECTION, records));
        break;
      }
      case "search":
        result = handle.search(COLLECTION, hashVector(payload.text), payload.topK ?? 5);
        break;
      case "stats":
        result = handle.footprint();
        break;
      case "similar": {
        // The record's own text re-vectorized: nidus has the stored vector, but the
        // binding exposes no get(), and this is the same vector by construction.
        const seed = SEEDS.find((s) => s.id === payload.id);
        if (!seed) throw new Error(`no seeded record with id "${payload.id}"`);
        const hits = handle.search(COLLECTION, hashVector(seed.text), (payload.topK ?? 4) + 1);
        result = hits.filter((h) => h.id !== payload.id).slice(0, payload.topK ?? 4);
        break;
      }
      case "clear":
        await withPoolGrowth(() => handle.drop_collection(COLLECTION));
        await seed();
        result = { cleared: true };
        break;
      default:
        throw new Error(`unknown worker command: ${cmd}`);
    }
    self.postMessage({ id, ok: true, result });
  } catch (e) {
    self.postMessage({ id, ok: false, error: e && e.message ? e.message : String(e) });
  }
};
