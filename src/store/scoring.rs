//! Scan kernels: the per-chunk scoring functions and the parallel-scan engine that
//! the brute-force ([`super::read`]) and quantized ([`super::quant`]) search paths
//! share. Pure functions over borrowed data — no [`Store`](super::Store) state.

use anyhow::{Result, anyhow};

use crate::data::Segments;
use crate::search::{TopK, dot_i8, euclidean_neg_sq_i8, hamming};

/// Minimum total scan *work* — candidate rows × dimension — before a parallel search
/// splits across worker threads. Below this, thread spawn/join overhead outweighs the
/// scan, so we stay serial even when `Config::query_threads > 1`. The floor is on work
/// rather than a flat row count because per-row scan cost scales with dimension: a fixed
/// row floor over-parallelizes narrow vectors and under-parallelizes wide ones. ~1.05M
/// units ≈ 4096 rows at dim 256, or ~1365 rows at dim 768.
pub(super) const PARALLEL_SCAN_WORK_FLOOR: usize = 1 << 20;

/// Score a slice of candidate rows into a fresh bounded top-k heap. The unit of
/// parallel work: each worker scores one chunk independently, then the caller
/// merges the per-chunk heaps. Pure read of `data` (shared `&` across threads).
pub(super) fn score_chunk<'a>(
    data: &Segments,
    chunk: &[(u64, &'a str, &'a str)],
    q: &[f32],
    score_fn: fn(&[f32], &[f32]) -> f32,
    top_k: usize,
    min_score: Option<f32>,
) -> TopK<(&'a str, &'a str)> {
    let mut topk: TopK<(&'a str, &'a str)> = TopK::new(top_k);
    for &(row, col_name, id) in chunk {
        let score = score_fn(q, data.row(row));
        if let Some(min) = min_score
            && score < min
        {
            continue;
        }
        topk.offer(score, (col_name, id));
    }
    topk
}

/// Score a chunk against the **int8** matrix into a bounded top-k of `overscan`
/// candidates — the quantized first-pass unit of parallel work, mirroring
/// [`score_chunk`] for the f32 path. The int8 score is monotonic with the f32 score
/// (shared symmetric scale), so it picks the right candidate set; exact scores come
/// from the caller's f32 rerank. Carries `row` in the item so the rerank can re-read
/// the f32 vector. `min_score` is *not* applied here — the int8 score is only an
/// ordering proxy, so the floor is enforced on the exact f32 score during rerank.
pub(super) fn score_chunk_i8<'a>(
    quant_vectors: &[i8],
    dim: usize,
    chunk: &[(u64, &'a str, &'a str)],
    q_i8: &[i8],
    is_euclidean: bool,
    overscan: usize,
) -> TopK<(u64, &'a str, &'a str)> {
    let mut topk: TopK<(u64, &'a str, &'a str)> = TopK::new(overscan);
    for &(row, col_name, id) in chunk {
        let base = row as usize * dim;
        let end = base + dim;
        if end > quant_vectors.len() {
            continue;
        }
        let stored_i8 = &quant_vectors[base..end];
        let approx_score = if is_euclidean {
            euclidean_neg_sq_i8(q_i8, stored_i8) as f32
        } else {
            dot_i8(q_i8, stored_i8) as f32
        };
        topk.offer(approx_score, (row, col_name, id));
    }
    topk
}

/// Score a chunk against the **binary** (sign-bit) matrix into a bounded top-k of
/// `overscan` candidates — the binary first-pass unit of parallel work, mirroring
/// [`score_chunk_i8`]. Score is `-(hamming)` (higher = better), monotone with cosine
/// rank for unit vectors, so it picks the right candidate set; exact scores come from
/// the caller's f32 rerank. Carries `row` so the rerank can re-read the f32 vector.
/// `min_score` is *not* applied here — Hamming is only an ordering proxy.
pub(super) fn score_chunk_bin<'a>(
    words: &[u64],
    words_per_row: usize,
    chunk: &[(u64, &'a str, &'a str)],
    q_words: &[u64],
    overscan: usize,
) -> TopK<(u64, &'a str, &'a str)> {
    let mut topk: TopK<(u64, &'a str, &'a str)> = TopK::new(overscan);
    for &(row, col_name, id) in chunk {
        let base = row as usize * words_per_row;
        let end = base + words_per_row;
        if end > words.len() {
            continue;
        }
        let approx_score = -(hamming(q_words, &words[base..end]) as f32);
        topk.offer(approx_score, (row, col_name, id));
    }
    topk
}

/// Split `scan` across `workers` threads, score each chunk with `score_one` into its
/// own bounded top-k of capacity `cap`, then merge the per-worker results into one.
/// The shared parallel-scan engine behind both the f32 and int8 first passes.
///
/// Each worker sorts its own chunk by physical row before scoring, so the per-chunk
/// sweep stays storage-ordered for the prefetcher — the global row-sort is skipped on
/// the parallel path (the prefetch win is per-chunk, and per-chunk sorts run in
/// parallel instead of as serial pre-work, cutting the Amdahl tax). Reads of `data` /
/// the quant matrix are shared `&` across threads; the only mutation is each worker
/// reordering its disjoint `&mut` chunk.
pub(super) fn parallel_topk<'a, T, F>(
    scan: &mut [(u64, &'a str, &'a str)],
    workers: usize,
    cap: usize,
    score_one: F,
) -> Result<TopK<T>>
where
    T: Send,
    F: Fn(&[(u64, &'a str, &'a str)]) -> TopK<T> + Sync,
{
    let chunk_len = scan.len().div_ceil(workers);
    let score_one = &score_one;
    let locals = std::thread::scope(|s| -> Result<Vec<Vec<(f32, T)>>> {
        let handles: Vec<_> = scan
            .chunks_mut(chunk_len)
            .map(|chunk| {
                s.spawn(move || {
                    chunk.sort_unstable_by_key(|&(row, _, _)| row);
                    score_one(chunk).into_sorted_desc()
                })
            })
            .collect();
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(
                h.join()
                    .map_err(|_| anyhow!("search worker thread panicked"))?,
            );
        }
        Ok(out)
    })?;

    let mut merged: TopK<T> = TopK::new(cap);
    for local in locals {
        for (score, item) in local {
            merged.offer(score, item);
        }
    }
    Ok(merged)
}
