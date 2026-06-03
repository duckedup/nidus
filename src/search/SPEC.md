# `search` module — spec

Implement the cosine kernels and bounded top-k in `mod.rs`. **Do not change the
public signatures.** Root design: `../../SPEC.md` §7.

## Functions
- `pub fn normalize(v: &mut [f32])` — scale `v` to unit L2 length in place. If the
  norm is 0 or non-finite / below a tiny epsilon (e.g. `1e-12`), leave `v` unchanged
  (it must then score 0 against everything via `dot`).
- `pub fn dot(a: &[f32], b: &[f32]) -> f32` — sum of `a[i]*b[i]`. Assume equal
  length (callers guarantee it). Write it as a simple loop / iterator that the
  compiler can autovectorize; `f32` accumulation is fine. May `debug_assert_eq!`
  the lengths.
- `pub struct TopK<T>` with:
  - `pub fn new(k: usize) -> Self`
  - `pub fn offer(&mut self, score: f32, item: T)` — retain the item only if it is
    among the top `k` scores offered so far. O(log k) per offer via a min-heap
    keyed on score (smallest score on top, evicted when full and a better score
    arrives). `k == 0` keeps nothing.
  - `pub fn into_sorted_desc(self) -> Vec<(f32, T)>` — kept items, highest score
    first. Ties broken arbitrarily.

`f32` is not `Ord`. Wrap scores for the heap with a total order (e.g. a newtype using
`f32::total_cmp`). Treat `NaN` as the lowest possible score so it never displaces a
real result.

## Constraints
- Pure safe Rust (`#![forbid(unsafe_code)]`). Use `std::collections::BinaryHeap` or
  a hand-rolled heap. No new dependencies.
- `TopK<T>` must work for any `T` (the store uses `T = usize` row index or a small id
  struct). You may add a bound like `T` only where strictly needed; prefer none.

## Tests (inline, Miri-clean)
`normalize` of `[3,0,0]` → `[1,0,0]`; zero vector unchanged; `dot` of orthonormal
basis vectors = 0 and of equal unit vectors ≈ 1. `TopK`: offering more than `k`
keeps the k highest; `into_sorted_desc` is descending; `k==0` empty; NaN scores are
discarded in favor of real ones; stable enough that the documented ranking holds.
