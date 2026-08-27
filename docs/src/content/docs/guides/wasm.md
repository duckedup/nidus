---
title: Running in the browser
description: Build nidus for wasm32 and store its data in the browser's Origin Private File System, from a dedicated worker.
---

nidus compiles for `wasm32-unknown-unknown`, so a browser page (or an edge runtime that
speaks wasm) can hold a nidus store and search it entirely client-side, no server round
trip. Persistence is a new location scheme, `opfs://`, backed by the browser's [Origin
Private File System](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)
(OPFS): every byte stays in storage scoped to your page's own origin.

No `Cargo.toml` feature flag is involved. The `wasm32` target itself selects which parts
of the tree compile: the core store, search, and the local/RAM/OPFS backends build and run
there, while the S3, GCS, and Redis-family backends are simply absent (their client crates
depend on native TLS and certificate stores that do not target `wasm32-unknown-unknown`).
If you depend on `nidus` directly from your own crate targeting `wasm32`, set
`default-features = false` on that dependency: the default build pulls in `cli`,
`serve`, and the provider crates, none of which target `wasm32-unknown-unknown`.

## Installing

```sh
npm install @duckedup/nidus
```

```ts
import { acquireOpfsPool } from "@duckedup/nidus/wasm";

const pool = await acquireOpfsPool({ slots: 8 });
const store = pool.NidusHandle.open("opfs://my-store", /* dimension */ 3);
```

`@duckedup/nidus/wasm` is a separate, **ESM-only** subpath of the JS SDK: it resolves its
wasm payload relative to `import.meta.url`, so it works from `import` but not `require`.
It is a subpath rather than part of the package's default entry precisely so the wasm
payload (roughly 477 KB) never lands in a bundle that only wanted the plain HTTP client.
Import it lazily, from the dedicated worker that needs it (see below), not from your
app's top level.

`acquireOpfsPool` does the handle acquisition and pool registration described in the next
two sections for you, so a consumer never has to hand-write that loop.

## Building from the repo

Most consumers only need `npm install @duckedup/nidus` above. Build straight from the
repo instead when you are working on the binding itself:

```bash
# Build the library for the target. `--no-default-features` is required: the default
# build pulls the server stack (tokio, axum, reqwest), none of which targets wasm32.
cargo build --no-default-features --target wasm32-unknown-unknown --lib

# Build the JS binding (bindings/wasm)
just build-wasm-binding
```

That writes an ES module plus its `.d.ts` to `bindings/wasm/pkg`, which you import
directly:

```ts
import init, { NidusHandle, init_opfs_pool, grow_opfs_pool } from "./pkg/nidus_wasm.js";
```

A full working example, including the worker and a page that drives it, lives in
[`bindings/wasm/demo`](https://github.com/duckedup/nidus/tree/main/bindings/wasm/demo).
Serve it over `http://localhost` or any secure context; OPFS is unavailable on a plain
`file://` page.

## Why a dedicated worker

An OPFS `FileSystemSyncAccessHandle` exists only inside the thread that opened it: it
cannot be shared with, or handed to, another worker or the main thread. So the whole
lifecycle of an `opfs://` store (opening the handle pool, then `open`/`upsert`/`search`/
`flush` on that store) has to run on **one dedicated worker thread**, start to finish.
The demo's `worker.js` shows the shape: the main page only ever posts messages to the
worker and awaits replies; every actual store call happens inside the worker's own message
handler.

Two async browser calls do the handle acquisition, once, up front: this is exactly what
`acquireOpfsPool` runs for you, in a loop, one slot at a time:

```js
const root = await navigator.storage.getDirectory();
const fileHandle = await root.getFileHandle(name, { create: true });
const syncHandle = await fileHandle.createSyncAccessHandle();
```

Everything after that, every nidus call, is synchronous.

nidus's own browser test suite (`tests/wasm_opfs`) follows the same rule and drives its
handles from a dedicated worker too, for the same reason.

## The pool handshake

Hand a batch of already-open sync access handles to `init_opfs_pool` before opening any
store:

```js
import init, { NidusHandle, init_opfs_pool } from "../pkg/nidus_wasm.js";

await init();
const handles = await openSlots(8); // your own async loop over createSyncAccessHandle()
init_opfs_pool(handles);
const store = NidusHandle.open("opfs://my-store", /* dimension */ 3);
```

**`handles[0]` is reserved as the directory slot**: it holds a small, checksummed map of
object key to slot number, not a data object, and `handles[1..]` are the body slots, one
per stored object (`data`, `log`, and so on). So a batch of `N` handles gives you `N - 1`
usable object slots, not `N`. Passing an empty array is rejected outright.

`init_opfs_pool` (and everything that follows) registers the pool on the calling thread
only. Calling it on the worker and then calling `NidusHandle.open`/`upsert`/`search` from
anywhere else fails with an error naming the missing registration: the same thread
affinity rule as above, enforced at the Rust layer too.

## Pool exhaustion and the retry loop

A store mints new object keys as it runs (sealing a segment, for example), and opening a
handle is async while `put` is not. So a synchronous write that needs a slot the pool does
not have fails outright, with an error that says so:

```
OPFS pool exhausted: all N body slots are occupied (...); growing the pool needs an
async step (acquire more handles, then call `nidus::backend::grow_pool`) that cannot
happen inside this synchronous write
```

The fix is a retry loop on the JS side: catch that specific error, open another batch of
handles asynchronously, hand them to `grow_opfs_pool`, and redo the write:

```js
async function withPoolGrowth(writeOnce) {
  try {
    return writeOnce();
  } catch (e) {
    if (!String(e).includes("OPFS pool exhausted")) throw e;
    grow_opfs_pool(await openSlots(8));
    return writeOnce(); // retry, now that the pool has room
  }
}
```

This is not a bug to work around once and forget: any write can hit it again as the store
grows, so keep the retry wrapper around every write path, the way the demo's `worker.js`
wraps `upsert`.

## What is not there yet

Be aware of these gaps rather than discovering them after adopting wasm support:

- **No threads.** `wasm32-unknown-unknown` has no `std::thread::scope`, so
  `Config::query_threads` above `1` and the parallel HNSW build path have nothing to run
  on. Setting them is not an error; they are silently ignored and the serial path runs.
- **No `s3://`, `gs://`, or Redis-family (`redis://`, `valkey://`, …) backends.** These
  depend on native TLS/cert stores that are simply not available on this target. Naming one
  of these locations in a wasm build returns a clear error rather than a silent local
  fallback.
- **No mmap.** `Config::mmap` is for a mappable local file; OPFS objects are not one, so
  the setting is accepted and quietly has no effect (the same fallback a `redis`-backed or
  in-memory store already takes on native).

None of this is a smaller version of nidus by design choice: it's what one browser thread
with OPFS and no native TLS stack can offer today, and it may grow over time.
