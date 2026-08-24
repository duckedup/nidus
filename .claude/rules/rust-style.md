---
paths:
  - "src/**/*.rs"
  - "tests/**/*.rs"
  - "benchmarks/**/*.rs"
  - "bindings/**/*.rs"
---

# Rust conventions

- **Safe Rust.** `#![deny(unsafe_code)]` — `deny`, not `forbid`, because the single
  memory-map call in `src/data/mmap.rs` carries a scoped `#[allow(unsafe_code)]`. That module
  is the **only** place the allow appears; a second one is a design change (D0006).
- **Sync API.** nidus is synchronous (CPU + blocking file IO). Async callers wrap it in
  `Arc<Mutex<Nidus>>` + `spawn_blocking`, the same way one wraps a blocking embedded DB.
- **One embedding space per store.** Dimension is pinned in the `data` header at creation;
  reopening with a different dimension is a hard error. Many collections, one dimension.
- **Errors are `anyhow::Result` everywhere** (`anyhow!`/`bail!`/`.context()`). No hand-rolled
  error enum, no `thiserror`.
- **Codec discipline.** All on-disk encoding is little-endian and explicit; every record is
  length-prefixed and CRC32-checked so a torn tail is detectable and recoverable. Test codecs
  against in-memory buffers so they stay Miri-clean.
- **Comments cap at 3 lines**, counting the whole block including `///` blank separators. Doc
  examples (a ``` fence) and `//!` module docs are exempt. Rationale longer than that goes in
  the commit message, the PR, `SPEC.md`, or a `bd` issue (D0009).

## Keep files focused — split by concern, not by size cap

There is no hard line limit, but a module covering several distinct concerns should become a
directory of sibling files, each owning one, with `mod.rs` holding the core type plus the glue.
`src/store/` is the worked example: `scoring`, `quant`, `read`, `write`, `tests` each stand
alone.

In Rust this costs almost nothing: child modules see the parent's private items and private
struct fields, so an inherent `impl Store` can span several files with **no** field made `pub`.
Only something called across sibling submodules needs `pub(super)` (e.g. `hits_from_topk`,
`rebuild_quant`). Keep state types beside the code that reads their internals (`Int8State`
lives with the quantized search) so their fields stay private.

Adding a big new concern to an already-large module? Prefer a new sibling file over appending,
and move the matching tests into the module's own `tests.rs` rather than growing one giant
test block.
