# D0006 — `deny(unsafe_code)`, not `forbid`, for exactly one scoped allow

**Status:** accepted
**Rule:** `src/data/mmap.rs` is the only place `#[allow(unsafe_code)]` may appear. A second scoped allow is a design change, not an implementation detail.

## Why

Our own code carries `#![deny(unsafe_code)]`. It is `deny` rather than `forbid` for one
reason: the single memory-map call in `src/data/mmap.rs` (SPEC §9/§14.6) carries a scoped
`#[allow(unsafe_code)]`, and `forbid` cannot be locally overridden.

That one module is the whole exception. Every other `unsafe` in the crate stays a hard
compile error, which is what keeps the weakening visible: a diff that adds a second allow is
proposing a new trust boundary, not writing code.

`nidus-check laws` has a detector for both halves — an `unsafe` use outside the sanctioned
module, and a weakened crate attribute — so this is checked rather than remembered.

## Evidence

- `lib/laws.mjs` — `unsafeUse` and `crateAttrWeakened`.
