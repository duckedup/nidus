//! Quantization: the in-RAM int8 / binary state, its build-and-extend maintenance, and
//! the two-pass quantized search path (a quantized first pass selects candidates, the
//! exact f32 [`rerank_candidates`](Store::rerank_candidates) orders them). Keeping the
//! state types and the search that reads their internals in one module means their
//! fields stay private here. Pure-f32 brute force lives in [`super::read`].

use anyhow::{Result, bail};

use super::Store;
use super::scoring::{parallel_topk, score_chunk_bin, score_chunk_i8};
use crate::model::{Distance, Hit, QuantKind, SearchOpts};
use crate::search::{QuantParams, TopK, pack_signs, pack_signs_into};

/// int8 scalar quantization state. `vectors` mirrors the f32 `data` rows one-for-one
/// (same physical row indices). Fields are `pub(super)` so the store's test module
/// (a sibling of this one) can assert on the maintained state.
pub(super) struct Int8State {
    params: QuantParams,
    /// Quantized vectors, flat and row-major, `data.row_count() * dim` int8 values.
    pub(super) vectors: Vec<i8>,
    /// How many rows `params` was fit from. Upserts quantize new rows against the
    /// current `params` and only refit (rescan for a fresh scale) once the row count
    /// outgrows this by [`REFIT_GROWTH`], keeping incremental upsert amortized O(1)/row.
    pub(super) params_rows: u64,
}

/// Binary (sign-bit) quantization state. **Scale-free:** each row's code is just its
/// sign bits, so there is no scale to fit and no refit — incremental upsert is a plain
/// append. `words` mirrors the f32 `data` rows one-for-one, `words_per_row` u64 each.
pub(super) struct BinState {
    /// Packed sign-bit codes, flat and row-major, `row_count * words_per_row` u64 values.
    pub(super) words: Vec<u64>,
    /// `dim.div_ceil(64)` — words per row's code.
    pub(super) words_per_row: usize,
}

/// The active quantization scheme's in-RAM state, maintained when `Config::quantization`
/// is set (`None` when quantization is off — the f32 brute-force default).
pub(super) enum Quant {
    Int8(Int8State),
    Binary(BinState),
}

impl Quant {
    /// An empty quant state for `kind`, validating metric compatibility up front.
    /// Binary codes are an angular proxy (they ignore magnitude), so they are rejected
    /// for any metric but cosine — a clear error beats a silently wrong ranking.
    pub(super) fn empty(kind: QuantKind, dim: usize, distance: Distance) -> Result<Self> {
        match kind {
            QuantKind::Int8 => Ok(Quant::Int8(Int8State {
                params: QuantParams::from_vectors(&[]),
                vectors: Vec::new(),
                params_rows: 0,
            })),
            QuantKind::Binary => {
                if distance != Distance::Cosine {
                    bail!(
                        "binary quantization requires Distance::Cosine (sign codes are an \
                         angular proxy and ignore magnitude); use int8 quantization for a \
                         dot-product or Euclidean store"
                    );
                }
                Ok(Quant::Binary(BinState {
                    words: Vec::new(),
                    words_per_row: dim.div_ceil(64),
                }))
            }
        }
    }
}

/// Pack a flat row-major f32 matrix (`dim` floats per row) into row-major sign-bit
/// codes, `dim.div_ceil(64)` u64 per row. The whole-matrix build used by `rebuild_quant`.
fn pack_matrix(vectors: &[f32], dim: usize) -> Vec<u64> {
    if dim == 0 {
        return Vec::new();
    }
    let words_per_row = dim.div_ceil(64);
    let rows = vectors.len() / dim;
    let mut out = vec![0u64; rows * words_per_row];
    for r in 0..rows {
        let src = &vectors[r * dim..(r + 1) * dim];
        let dst = &mut out[r * words_per_row..(r + 1) * words_per_row];
        pack_signs_into(src, dst);
    }
    out
}

/// Refit the quantization scale once the live row count grows past this multiple of
/// the count it was last fit from. Geometric (doubling) → amortized O(1) per row over
/// a full incremental build, while bounding how stale the shared scale can get.
const REFIT_GROWTH: u64 = 2;

impl Store {
    /// Rebuild the quantized matrix from *all* current vectors. O(N) — used on `open`,
    /// `compact`, and the occasional int8 geometric refit, not per upsert batch. int8
    /// re-fits the scale and re-quantizes; binary repacks sign bits (scale-free).
    pub(super) fn rebuild_quant(&mut self) {
        let dim = self.data.dimension();
        let all = self.data.vectors();
        match self.quant {
            None => {}
            Some(Quant::Int8(ref mut s)) => {
                s.params = QuantParams::from_vectors(all);
                s.vectors = s.params.quantize_all(all);
                s.params_rows = self.data.row_count();
            }
            Some(Quant::Binary(ref mut s)) => {
                s.words_per_row = dim.div_ceil(64);
                s.words = pack_matrix(all, dim);
            }
        }
    }

    /// Incrementally extend the quantized matrix after `upsert` appended rows
    /// `[prev_rows, row_count())` — O(batch), not O(N). int8 quantizes the new rows
    /// against the existing scale, falling back to a full [`rebuild_quant`] when there
    /// is no scale yet or the row count has grown past [`REFIT_GROWTH`]× the fit set (so
    /// a drifting distribution can't keep saturating a stale scale). Binary is scale-free
    /// — it just packs the new rows' sign bits, never refits.
    pub(super) fn extend_quant(&mut self, prev_rows: u64) {
        let total = self.data.row_count();
        let dim = self.data.dimension();
        // Decide whether int8 needs a full refit before taking the mutable state borrow.
        let refit = match self.quant {
            None => return,
            Some(Quant::Int8(ref s)) => s.params_rows == 0 || total > s.params_rows * REFIT_GROWTH,
            Some(Quant::Binary(_)) => false, // scale-free: never refits
        };
        if refit {
            self.rebuild_quant();
            return;
        }
        let all = self.data.vectors();
        match self.quant {
            None => {}
            Some(Quant::Int8(ref mut s)) => {
                s.vectors.resize(total as usize * dim, 0);
                for row in prev_rows as usize..total as usize {
                    let base = row * dim;
                    let (src, dst) = (&all[base..base + dim], &mut s.vectors[base..base + dim]);
                    s.params.quantize(src, dst);
                }
            }
            Some(Quant::Binary(ref mut s)) => {
                let wpr = s.words_per_row;
                s.words.resize(total as usize * wpr, 0);
                for row in prev_rows as usize..total as usize {
                    let src = &all[row * dim..(row + 1) * dim];
                    pack_signs_into(src, &mut s.words[row * wpr..(row + 1) * wpr]);
                }
            }
        }
    }

    /// The configured overscan factor (the first pass keeps `top_k × rescore`
    /// candidates for the f32 rerank). 1 when quantization is off — unused on that path.
    fn rescore(&self) -> usize {
        self.config.quantization.map_or(1, |q| q.rescore)
    }

    /// If a populated quantized matrix is present, run the matching two-pass search and
    /// return its hits; `None` means quantization is off (or not yet built), so the
    /// caller falls back to the exact f32 scan in [`super::read`]. This is the seam that
    /// keeps every read of the [`Quant`] state's private fields inside this module.
    pub(super) fn search_quantized<'a>(
        &self,
        q: &[f32],
        scan: &mut [(u64, &'a str, &'a str)],
        opts: &SearchOpts,
        score_fn: fn(&[f32], &[f32]) -> f32,
        workers: usize,
    ) -> Option<Result<Vec<Hit>>> {
        if self.data.dimension() == 0 {
            return None;
        }
        match self.quant {
            Some(Quant::Int8(ref s)) if !s.vectors.is_empty() => {
                Some(self.search_int8(q, scan, opts, score_fn, workers))
            }
            Some(Quant::Binary(ref s)) if !s.words.is_empty() => {
                Some(self.search_binary(q, scan, opts, score_fn, workers))
            }
            _ => None,
        }
    }

    /// Two-pass int8 search: int8 first-pass selects candidates, f32 reranks. The int8
    /// first pass is the lever that scales with threads — int8 moves 4× fewer bytes than
    /// f32, so it is compute- not bandwidth-bound — so it splits across `workers` (when
    /// engaged), while the f32 rerank stays serial (only `top_k × rescore` rows, too few
    /// to amortize a second fan-out).
    pub(super) fn search_int8<'a>(
        &self,
        q: &[f32],
        scan: &mut [(u64, &'a str, &'a str)],
        opts: &SearchOpts,
        score_fn: fn(&[f32], &[f32]) -> f32,
        workers: usize,
    ) -> Result<Vec<Hit>> {
        let Some(Quant::Int8(s)) = self.quant.as_ref() else {
            return Ok(Vec::new());
        };
        let dim = self.data.dimension();
        let overscan = opts.top_k.saturating_mul(self.rescore()).max(opts.top_k);

        // Quantize the query vector with the same shared scale as the stored rows.
        let mut q_i8 = vec![0i8; dim];
        s.params.quantize(q, &mut q_i8);

        // First pass: int8 scoring to select overscan candidates. The int8 score is
        // monotonic with the f32 score (shared symmetric scale), so it picks the right
        // candidate set; exact scores come from the f32 rerank below. Parallel when
        // engaged (the int8 sweep is the part that scales with threads), else serial.
        let is_euclidean = self.config.distance == Distance::Euclidean;
        let topk_q = if workers > 1 {
            parallel_topk(scan, workers, overscan, |chunk| {
                score_chunk_i8(&s.vectors, dim, chunk, &q_i8, is_euclidean, overscan)
            })?
        } else {
            // `scan` arrives row-sorted from `with_sorted_scan` — score it in place.
            score_chunk_i8(&s.vectors, dim, scan, &q_i8, is_euclidean, overscan)
        };

        let candidates = topk_q.into_sorted_desc();
        Ok(self.rerank_candidates(q, &candidates, score_fn, opts))
    }

    /// Two-pass binary search: a Hamming first-pass over the 32×-smaller sign-bit matrix
    /// selects candidates, f32 reranks. Cosine only (enforced at `open`), so the query is
    /// already unit-normalized when it reaches here; its sign code is invariant to that.
    /// The binary first pass moves 32× fewer bytes than f32, so it scales with `workers`
    /// even harder than int8; the f32 rerank stays serial.
    pub(super) fn search_binary<'a>(
        &self,
        q: &[f32],
        scan: &mut [(u64, &'a str, &'a str)],
        opts: &SearchOpts,
        score_fn: fn(&[f32], &[f32]) -> f32,
        workers: usize,
    ) -> Result<Vec<Hit>> {
        let Some(Quant::Binary(s)) = self.quant.as_ref() else {
            return Ok(Vec::new());
        };
        let overscan = opts.top_k.saturating_mul(self.rescore()).max(opts.top_k);

        // Pack the query's sign bits with the same rule as the stored rows.
        let q_words = pack_signs(q);
        let wpr = s.words_per_row;

        // First pass: Hamming scoring selects overscan candidates. Score = -(hamming),
        // monotone with cosine rank for unit vectors; exact scores come from the rerank.
        let topk_q = if workers > 1 {
            parallel_topk(scan, workers, overscan, |chunk| {
                score_chunk_bin(&s.words, wpr, chunk, &q_words, overscan)
            })?
        } else {
            // `scan` arrives row-sorted from `with_sorted_scan` — score it in place.
            score_chunk_bin(&s.words, wpr, scan, &q_words, overscan)
        };

        let candidates = topk_q.into_sorted_desc();
        Ok(self.rerank_candidates(q, &candidates, score_fn, opts))
    }

    /// Exact f32 rerank of first-pass candidates → final ranked `Hit`s. Shared by both
    /// two-pass paths: the first pass is only an ordering proxy, so the true score (and
    /// `min_score`) is computed here from the original f32 vectors.
    fn rerank_candidates(
        &self,
        q: &[f32],
        candidates: &[(f32, (u64, &str, &str))],
        score_fn: fn(&[f32], &[f32]) -> f32,
        opts: &SearchOpts,
    ) -> Vec<Hit> {
        let mut topk: TopK<(&str, &str)> = TopK::new(opts.top_k);
        for (_, (row, col_name, id)) in candidates {
            let score = score_fn(q, self.data.row(*row));
            if let Some(min) = opts.min_score
                && score < min
            {
                continue;
            }
            topk.offer(score, (*col_name, *id));
        }
        self.hits_from_topk(topk)
    }
}
