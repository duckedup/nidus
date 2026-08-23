//! `@duckedup/nidus/wasm` — browser helper over the generated `nidus_wasm` module.
//
// Must typecheck with `bindings/wasm/pkg` absent (nidus-3hc): the module's surface is
// declared locally below and reached via a dynamic `import()` of a non-literal
// specifier, never a static import of the `nidus_wasm.js` this package copies into place.

// OPFS sync access is worker-only and still missing from TS's lib.dom.d.ts.
declare global {
  interface FileSystemFileHandle {
    createSyncAccessHandle(): Promise<FileSystemSyncAccessHandle>;
  }
  interface FileSystemSyncAccessHandle {
    close(): void;
  }
}

/** Mirrors `bindings/wasm/src/lib.rs`'s `#[wasm_bindgen]` exports. Can drift if that changes. */
export interface NidusHandleInstance {
  upsert(collection: string, records: unknown): number;
  search(collection: string, query: number[], topK: number): unknown;
  flush(): void;
  close(): void;
}

export interface NidusHandleClass {
  open(location: string, dimension: number): NidusHandleInstance;
}

interface WasmModule {
  default: (input?: unknown) => Promise<unknown>;
  NidusHandle: NidusHandleClass;
  init_opfs_pool: (handles: FileSystemSyncAccessHandle[]) => void;
  grow_opfs_pool: (handles: FileSystemSyncAccessHandle[]) => void;
}

let modPromise: Promise<WasmModule> | undefined;

async function loadWasm(): Promise<WasmModule> {
  if (!modPromise) {
    const specifier = "./nidus_wasm.js";
    modPromise = import(specifier) as Promise<WasmModule>;
  }
  const mod = await modPromise;
  await mod.default();
  return mod;
}

export interface OpfsPoolOptions {
  /** Handles opened per grow (directory slot + body slots). Defaults to 8. */
  slots?: number;
  /** Directory name under the OPFS root to open slot files in. Defaults to "nidus". */
  dir?: string;
}

export interface OpfsPool {
  /** The store constructor, ready to use: the pool it needs is already registered. */
  readonly NidusHandle: NidusHandleClass;
  /** Open `slots` more handles and register them with the wasm pool (async growth step). */
  grow(): Promise<void>;
  /** Run `writeOnce`; on a pool-exhausted error, grow the pool once and retry. */
  withPoolGrowth<T>(writeOnce: () => T): Promise<T>;
}

// Mirrors bindings/wasm/demo/worker.js:17-25 (openSlots): the one async step OPFS needs
// per handle (getFileHandle + createSyncAccessHandle); everything after is sync wasm.
async function openSlots(dir: FileSystemDirectoryHandle, count: number, next: { n: number }) {
  const opened: FileSystemSyncAccessHandle[] = [];
  for (let i = 0; i < count; i++) {
    const fileHandle = await dir.getFileHandle(`slot-${next.n++}`, { create: true });
    opened.push(await fileHandle.createSyncAccessHandle());
  }
  return opened;
}

/**
 * Load the wasm module and open+register an initial pool of OPFS sync access handles.
 * Mirrors `bindings/wasm/demo/worker.js`'s `openSlots`/`withPoolGrowth` (:17-25, :36-44).
 */
export async function acquireOpfsPool(opts?: OpfsPoolOptions): Promise<OpfsPool> {
  const slots = opts?.slots ?? 8;
  const mod = await loadWasm();
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle(opts?.dir ?? "nidus", { create: true });
  const next = { n: 0 };

  mod.init_opfs_pool(await openSlots(dir, slots, next));

  return {
    NidusHandle: mod.NidusHandle,
    async grow() {
      mod.grow_opfs_pool(await openSlots(dir, slots, next));
    },
    // Retries once on "OPFS pool exhausted" after growing (nidus-y67's documented
    // design, not a bug): a sync write can't perform the async grow step itself.
    async withPoolGrowth<T>(writeOnce: () => T): Promise<T> {
      try {
        return writeOnce();
      } catch (e) {
        if (!String(e).includes("OPFS pool exhausted")) throw e;
        mod.grow_opfs_pool(await openSlots(dir, slots, next));
        return writeOnce();
      }
    },
  };
}
