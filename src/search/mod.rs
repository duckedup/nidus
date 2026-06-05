//! Cosine kernels + bounded top-k selection. Contract: see the root `SPEC.md` §7.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Normalize `v` to unit length in place. A zero (or near-zero) vector is left
/// unchanged so it scores 0 against everything.
pub fn normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    // Leave zero or non-finite vectors unchanged; they will score 0 against everything.
    if norm < 1e-12 || !norm.is_finite() {
        return;
    }
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Number of independent accumulator lanes in [`dot`]. Eight `f32` lanes is a
/// 256-bit-wide reduction, which maps onto common SIMD register widths (AVX `ymm`,
/// or two NEON `q` registers) and keeps enough chains in flight to hide FMA latency.
const DOT_LANES: usize = 8;

/// Dot product of two equal-length slices. For unit vectors this is cosine
/// similarity. Panics or is undefined if lengths differ — callers guarantee equal
/// length (the store pins the dimension).
///
/// The hot path of brute-force search. A naive `zip().map().sum()` forces a single
/// sequential accumulator: because f32 addition is not associative, LLVM may not
/// reorder it, so without fast-math it stays scalar and leaves the vector units idle.
/// Here we keep [`DOT_LANES`] *independent* running sums — lane `i` only ever touches
/// elements `i, i+LANES, i+2·LANES, …` — so each lane is its own associative reduction
/// chain the optimizer is free to vectorize, and the lanes share no dependency. The
/// per-lane partials are folded at the end (pairwise, again independent of input
/// length). Summation order differs from the naive left fold, so the f32 result can
/// round a hair differently; that only matters at exact-tie boundaries in ranking,
/// which are arbitrary anyway.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot: slice lengths must be equal");

    let mut acc = [0.0f32; DOT_LANES];
    let mut a_chunks = a.chunks_exact(DOT_LANES);
    let mut b_chunks = b.chunks_exact(DOT_LANES);

    // Bulk of the work: full LANES-wide chunks, each lane an independent FMA chain.
    for (ca, cb) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for lane in 0..DOT_LANES {
            acc[lane] += ca[lane] * cb[lane];
        }
    }

    // Fold the lanes pairwise (8 → 4 → 2 → 1), then add the < LANES tail remainder.
    let mut half = DOT_LANES / 2;
    while half >= 1 {
        for lane in 0..half {
            acc[lane] += acc[lane + half];
        }
        half /= 2;
    }
    let mut sum = acc[0];
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        sum += x * y;
    }
    sum
}

/// Negative squared Euclidean distance: −Σ(aᵢ − bᵢ)². Result is in (−∞, 0];
/// 0 = identical vectors. Higher (closer to zero) = more similar, so the same
/// top-k heap that works for dot/cosine works here without modification.
pub fn euclidean_neg_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "euclidean_neg_sq: slice lengths must be equal"
    );

    let mut acc = [0.0f32; DOT_LANES];
    let mut a_chunks = a.chunks_exact(DOT_LANES);
    let mut b_chunks = b.chunks_exact(DOT_LANES);

    for (ca, cb) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for lane in 0..DOT_LANES {
            let d = ca[lane] - cb[lane];
            acc[lane] += d * d;
        }
    }

    let mut half = DOT_LANES / 2;
    while half >= 1 {
        for lane in 0..half {
            acc[lane] += acc[lane + half];
        }
        half /= 2;
    }
    let mut sum = acc[0];
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        let d = x - y;
        sum += d * d;
    }
    -sum
}

// ── Internal total-order wrapper for f32 scores ──────────────────────────────
//
// `BinaryHeap` requires `Ord`. Since `f32` is not `Ord` (NaN), we wrap the
// score with a newtype that uses `f32::total_cmp`, which places NaN below all
// finite values (at the bottom of the total order). This means NaN scores are
// the lowest possible and will be evicted before any real result.

#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        // Custom total order that places NaN as the *lowest* possible score,
        // so NaN never displaces a real result in the heap.
        // Note: f32::total_cmp places NaN as the *highest* (largest bit pattern),
        // which is the opposite of what we want here.
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,    // NaN < any real
            (false, true) => Ordering::Greater, // any real > NaN
            (false, false) => {
                // Both are non-NaN; partial_cmp is total here.
                self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
            }
        }
    }
}

// ── Entry stored in the min-heap ──────────────────────────────────────────────
//
// The heap is a max-heap by default; to get a min-heap we reverse the ordering
// so the smallest score sits at the top and is the first candidate for eviction.

struct Entry<T> {
    score: OrdF32,
    item: T,
}

impl<T> PartialEq for Entry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl<T> Eq for Entry<T> {}

impl<T> PartialOrd for Entry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Entry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse the score ordering: smallest score → largest in this Ord →
        // floats to the top of BinaryHeap (which is a max-heap).
        other.score.cmp(&self.score)
    }
}

/// A bounded collector that retains the `k` highest-scoring items seen, in O(N log k)
/// using a min-heap keyed by score. Ties may be broken arbitrarily.
pub struct TopK<T> {
    k: usize,
    heap: BinaryHeap<Entry<T>>,
}

impl<T> TopK<T> {
    /// A collector that keeps the top `k` items.
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
        }
    }

    /// Offer one scored item; kept only if it ranks within the top `k` so far.
    pub fn offer(&mut self, score: f32, item: T) {
        if self.k == 0 {
            return;
        }
        let ord_score = OrdF32(score);
        if self.heap.len() < self.k {
            // Heap not full yet — always accept.
            self.heap.push(Entry {
                score: ord_score,
                item,
            });
        } else if let Some(worst) = self.heap.peek() {
            // Heap is full. Replace the worst (smallest-score) entry if the new
            // score is strictly better. Using total_cmp: NaN < everything finite,
            // so NaN scores are never preferred over real results.
            if ord_score > worst.score {
                self.heap.pop();
                self.heap.push(Entry {
                    score: ord_score,
                    item,
                });
            }
        }
    }

    /// Consume the collector, returning the kept items sorted by score descending.
    pub fn into_sorted_desc(self) -> Vec<(f32, T)> {
        // Drain the heap into a Vec, then sort descending.
        // BinaryHeap::into_iter_sorted would give descending order too (it's a
        // max-heap with reversed Ord), but we use sort for clarity and stability
        // vis-à-vis ties.
        let mut v: Vec<(f32, T)> = self.heap.into_iter().map(|e| (e.score.0, e.item)).collect();
        // Sort highest first; NaN treated as lowest (sorted to end).
        v.sort_by_key(|&(s, _)| std::cmp::Reverse(OrdF32(s)));
        v
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── normalize ──────────────────────────────────────────────────────────────

    #[test]
    fn normalize_simple() {
        let mut v = [3.0f32, 0.0, 0.0];
        normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-7, "x component should be 1");
        assert!((v[1]).abs() < 1e-7);
        assert!((v[2]).abs() < 1e-7);
    }

    #[test]
    fn normalize_zero_vector_unchanged() {
        let mut v = [0.0f32, 0.0, 0.0];
        normalize(&mut v);
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_near_zero_vector_unchanged() {
        // Norm below 1e-12 threshold should leave the vector alone.
        let mut v = [1e-14f32, 0.0, 0.0];
        normalize(&mut v);
        // The values should be essentially unchanged (not divided by the tiny norm).
        assert!((v[0] - 1e-14f32).abs() < 1e-20);
    }

    #[test]
    fn normalize_produces_unit_vector() {
        let mut v = [1.0f32, 2.0, 3.0];
        normalize(&mut v);
        let len: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (len - 1.0).abs() < 1e-6,
            "normalized vector should have unit length, got {len}"
        );
    }

    #[test]
    fn normalize_negative_components() {
        let mut v = [-4.0f32, 3.0];
        normalize(&mut v);
        let len: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-6);
        assert!(v[0] < 0.0, "sign should be preserved");
    }

    #[test]
    fn normalize_already_unit() {
        let mut v = [1.0f32, 0.0, 0.0];
        normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-7);
        assert!((v[1]).abs() < 1e-7);
        assert!((v[2]).abs() < 1e-7);
    }

    #[test]
    fn normalize_single_element() {
        let mut v = [5.0f32];
        normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn normalize_empty_slice() {
        let mut v: [f32; 0] = [];
        // Should not panic on an empty slice.
        normalize(&mut v);
    }

    // ── dot ───────────────────────────────────────────────────────────────────

    #[test]
    fn dot_orthogonal_basis_vectors() {
        let x = [1.0f32, 0.0, 0.0];
        let y = [0.0f32, 1.0, 0.0];
        let z = [0.0f32, 0.0, 1.0];
        assert!(
            (dot(&x, &y)).abs() < 1e-7,
            "orthogonal vectors should have dot=0"
        );
        assert!((dot(&x, &z)).abs() < 1e-7);
        assert!((dot(&y, &z)).abs() < 1e-7);
    }

    #[test]
    fn dot_unit_vectors_equal_approx_one() {
        let v = [1.0f32, 0.0, 0.0];
        assert!((dot(&v, &v) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn dot_equal_unit_vectors() {
        // After normalizing [1,1,0], dot with itself should be 1.
        let mut a = [1.0f32, 1.0, 0.0];
        normalize(&mut a);
        let b = a;
        assert!((dot(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dot_antiparallel() {
        let a = [1.0f32, 0.0];
        let b = [-1.0f32, 0.0];
        assert!(
            (dot(&a, &b) + 1.0).abs() < 1e-7,
            "antiparallel unit vectors → -1"
        );
    }

    #[test]
    fn dot_known_value() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((dot(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn dot_with_zero_vector() {
        let a = [1.0f32, 2.0, 3.0];
        let z = [0.0f32, 0.0, 0.0];
        assert!((dot(&a, &z)).abs() < 1e-7);
    }

    #[test]
    fn dot_empty_slices() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        assert_eq!(dot(&a, &b), 0.0);
    }

    /// Naive sequential reference the chunked `dot` must agree with (modulo f32
    /// rounding from a different summation order).
    fn dot_naive(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn dot_matches_naive_across_lengths() {
        // Cover every residue class around the LANES=8 chunk boundary, including
        // lengths shorter than one chunk (pure remainder) and exact multiples.
        for len in [
            0usize, 1, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 384, 768, 1000,
        ] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.013 - 0.5).collect();
            let b: Vec<f32> = (0..len).map(|i| 0.25 - (i as f32) * 0.007).collect();
            let got = dot(&a, &b);
            let want = dot_naive(&a, &b);
            // Different summation order → tiny rounding drift; scale tolerance by len.
            let tol = 1e-4 * (len as f32).max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "len={len}: chunked dot {got} vs naive {want}"
            );
        }
    }

    #[test]
    fn dot_tail_only_no_full_chunk() {
        // Length < LANES exercises the remainder path with zero full chunks.
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn dot_exact_chunk_boundary() {
        // Exactly LANES elements: one full chunk, empty remainder.
        let a = [1.0f32; 8];
        let b = [2.0f32; 8];
        assert!((dot(&a, &b) - 16.0).abs() < 1e-5);
    }

    // ── TopK ──────────────────────────────────────────────────────────────────

    #[test]
    fn topk_k_zero_keeps_nothing() {
        let mut tk: TopK<i32> = TopK::new(0);
        tk.offer(1.0, 42);
        tk.offer(2.0, 99);
        assert!(tk.into_sorted_desc().is_empty());
    }

    #[test]
    fn topk_fewer_than_k_offers() {
        let mut tk: TopK<&str> = TopK::new(5);
        tk.offer(0.9, "a");
        tk.offer(0.5, "b");
        let result = tk.into_sorted_desc();
        assert_eq!(result.len(), 2);
        // Highest score first.
        assert!((result[0].0 - 0.9).abs() < 1e-7);
        assert_eq!(result[0].1, "a");
        assert!((result[1].0 - 0.5).abs() < 1e-7);
        assert_eq!(result[1].1, "b");
    }

    #[test]
    fn topk_keeps_top_k_of_many() {
        let mut tk: TopK<usize> = TopK::new(3);
        // Offer 6 items; top-3 by score are 0.9, 0.8, 0.7.
        tk.offer(0.5, 1);
        tk.offer(0.9, 2);
        tk.offer(0.3, 3);
        tk.offer(0.7, 4);
        tk.offer(0.8, 5);
        tk.offer(0.1, 6);
        let result = tk.into_sorted_desc();
        assert_eq!(result.len(), 3);
        let scores: Vec<f32> = result.iter().map(|(s, _)| *s).collect();
        assert!(
            (scores[0] - 0.9).abs() < 1e-7,
            "first should be 0.9, got {}",
            scores[0]
        );
        assert!(
            (scores[1] - 0.8).abs() < 1e-7,
            "second should be 0.8, got {}",
            scores[1]
        );
        assert!(
            (scores[2] - 0.7).abs() < 1e-7,
            "third should be 0.7, got {}",
            scores[2]
        );
    }

    #[test]
    fn topk_into_sorted_desc_order() {
        let mut tk: TopK<u8> = TopK::new(4);
        tk.offer(0.2, 10);
        tk.offer(0.8, 20);
        tk.offer(0.5, 30);
        tk.offer(0.1, 40);
        let result = tk.into_sorted_desc();
        // Scores should be non-increasing.
        for w in result.windows(2) {
            assert!(
                w[0].0 >= w[1].0,
                "scores should be non-increasing: {} >= {}",
                w[0].0,
                w[1].0
            );
        }
    }

    #[test]
    fn topk_exact_k_offers_all_kept() {
        let mut tk: TopK<i32> = TopK::new(3);
        tk.offer(0.1, 1);
        tk.offer(0.2, 2);
        tk.offer(0.3, 3);
        let result = tk.into_sorted_desc();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn topk_nan_score_discarded_in_favor_of_real() {
        // NaN should be treated as the lowest possible score.
        // When the heap is full with real scores, a NaN offer should be ignored.
        let mut tk: TopK<i32> = TopK::new(2);
        tk.offer(0.8, 1);
        tk.offer(0.6, 2);
        // Heap is full. NaN should NOT displace either real entry.
        tk.offer(f32::NAN, 99);
        let result = tk.into_sorted_desc();
        assert_eq!(result.len(), 2, "NaN should not have been added");
        let items: Vec<i32> = result.iter().map(|(_, i)| *i).collect();
        assert!(items.contains(&1));
        assert!(items.contains(&2));
        assert!(!items.contains(&99), "NaN item should not be in results");
    }

    #[test]
    fn topk_nan_when_heap_not_full_then_evicted_by_real() {
        // If NaN sneaks in when the heap is not yet full, a real score should
        // later evict it because NaN < any real number under total_cmp.
        let mut tk: TopK<i32> = TopK::new(2);
        tk.offer(f32::NAN, 99); // heap not full, NaN gets in
        tk.offer(0.5, 1); // still not full, both in
        tk.offer(0.9, 2); // heap is full; NaN (worst) gets evicted
        let result = tk.into_sorted_desc();
        assert_eq!(result.len(), 2);
        let items: Vec<i32> = result.iter().map(|(_, i)| *i).collect();
        assert!(items.contains(&1));
        assert!(items.contains(&2));
        assert!(!items.contains(&99), "NaN item should have been evicted");
    }

    #[test]
    fn topk_negative_scores_ranked_correctly() {
        let mut tk: TopK<&str> = TopK::new(2);
        tk.offer(-0.5, "bad");
        tk.offer(-0.1, "ok");
        tk.offer(-0.9, "worse");
        let result = tk.into_sorted_desc();
        assert_eq!(result.len(), 2);
        assert!((result[0].0 - (-0.1)).abs() < 1e-7);
        assert_eq!(result[0].1, "ok");
        assert!((result[1].0 - (-0.5)).abs() < 1e-7);
        assert_eq!(result[1].1, "bad");
    }

    #[test]
    fn topk_k_one_keeps_best() {
        let mut tk: TopK<&str> = TopK::new(1);
        tk.offer(0.3, "a");
        tk.offer(0.7, "b");
        tk.offer(0.5, "c");
        let result = tk.into_sorted_desc();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, "b");
        assert!((result[0].0 - 0.7).abs() < 1e-7);
    }

    #[test]
    fn topk_cosine_workflow() {
        // Simulate typical store usage: normalize a query, compute dot products,
        // collect top-2.
        let mut query = [1.0f32, 1.0, 0.0];
        normalize(&mut query);

        let stored = [
            [1.0f32, 0.0, 0.0], // row 0 — not normalized, but let's pre-normalize
            [0.0f32, 1.0, 0.0], // row 1
            [0.0f32, 0.0, 1.0], // row 2 — orthogonal to query
        ];

        let mut tk: TopK<usize> = TopK::new(2);
        for (i, row) in stored.iter().enumerate() {
            let score = dot(row, &query);
            tk.offer(score, i);
        }

        let result = tk.into_sorted_desc();
        assert_eq!(result.len(), 2);
        // Rows 0 and 1 should tie (both 1/sqrt(2) ≈ 0.707), row 2 should be excluded.
        let rows: Vec<usize> = result.iter().map(|(_, r)| *r).collect();
        assert!(rows.contains(&0), "row 0 should be in top-2");
        assert!(rows.contains(&1), "row 1 should be in top-2");
        assert!(
            !rows.contains(&2),
            "row 2 (orthogonal) should not be in top-2"
        );
    }

    #[test]
    fn topk_ties_all_kept_within_k() {
        // When all scores are equal, k items are kept.
        let mut tk: TopK<i32> = TopK::new(3);
        for i in 0..5 {
            tk.offer(0.5, i);
        }
        let result = tk.into_sorted_desc();
        assert_eq!(result.len(), 3);
        // All retained scores should equal 0.5.
        for (s, _) in &result {
            assert!((*s - 0.5).abs() < 1e-7);
        }
    }

    // ── euclidean_neg_sq ─────────────────────────────────────────────────

    #[test]
    fn euclidean_identical_vectors() {
        let a = [1.0f32, 2.0, 3.0];
        assert!((euclidean_neg_sq(&a, &a)).abs() < 1e-7, "identical → 0");
    }

    #[test]
    fn euclidean_known_value() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        // ||a-b||² = 1+1 = 2, negated → -2
        assert!((euclidean_neg_sq(&a, &b) + 2.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_empty_slices() {
        let a: [f32; 0] = [];
        assert_eq!(euclidean_neg_sq(&a, &a), 0.0);
    }

    fn euclidean_naive(a: &[f32], b: &[f32]) -> f32 {
        -a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>()
    }

    #[test]
    fn euclidean_matches_naive_across_lengths() {
        for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 384, 768] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.013 - 0.5).collect();
            let b: Vec<f32> = (0..len).map(|i| 0.25 - (i as f32) * 0.007).collect();
            let got = euclidean_neg_sq(&a, &b);
            let want = euclidean_naive(&a, &b);
            let tol = 1e-4 * (len as f32).max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "len={len}: chunked {got} vs naive {want}"
            );
        }
    }

    #[test]
    fn euclidean_closer_vector_scores_higher() {
        let q = [1.0f32, 0.0, 0.0];
        let close = [0.9f32, 0.1, 0.0];
        let far = [0.0f32, 1.0, 0.0];
        assert!(euclidean_neg_sq(&q, &close) > euclidean_neg_sq(&q, &far));
    }
}
