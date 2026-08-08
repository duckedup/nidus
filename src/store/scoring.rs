//! Scan kernels: the per-chunk scoring functions and the parallel-scan engine that
//! the brute-force ([`super::read`]) and quantized ([`super::quant`]) search paths
//! share. Pure functions over borrowed data — no [`Store`](super::Store) state.

use anyhow::{Result, anyhow};

use crate::cancel::{CHECK_EVERY, check};
use crate::data::Segments;
use crate::search::{TopK, dot_i8, euclidean_neg_sq_i8, hamming};

/// Minimum scan work — rows × dimension — before a parallel search splits across threads; below it
/// spawn/join outweighs the scan. On work rather than a row count because per-row cost scales with
/// dimension. ~1.05M units ≈ 4096 rows at dim 256, or ~1365 at dim 768.
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
) -> Result<TopK<(&'a str, &'a str)>> {
    let mut topk: TopK<(&'a str, &'a str)> = TopK::new(top_k);
    // Cooperative cancellation: the only thing that can stop this loop is this loop. Hoisted out of
    // the per-row body by walking blocks — one atomic load per CHECK_EVERY rows and nothing per row,
    // since at small dimensions even a mask-and-branch is a measurable share of the work.
    for block in chunk.chunks(CHECK_EVERY) {
        check()?;
        for &(row, col_name, id) in block {
            let score = score_fn(q, data.row(row));
            if let Some(min) = min_score
                && score < min
            {
                continue;
            }
            topk.offer(score, (col_name, id));
        }
    }
    Ok(topk)
}

/// Score a chunk against the int8 matrix into a bounded top-k of `overscan` candidates — the
/// quantized first-pass unit of parallel work, mirroring [`score_chunk`]. Monotonic with f32, so it
/// picks the right set; carries `row` for the rerank, which is where `min_score` is enforced.
pub(super) fn score_chunk_i8<'a>(
    quant_vectors: &[i8],
    dim: usize,
    chunk: &[(u64, &'a str, &'a str)],
    q_i8: &[i8],
    is_euclidean: bool,
    overscan: usize,
) -> Result<TopK<(u64, &'a str, &'a str)>> {
    let mut topk: TopK<(u64, &'a str, &'a str)> = TopK::new(overscan);
    for block in chunk.chunks(CHECK_EVERY) {
        check()?;
        for &(row, col_name, id) in block {
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
    }
    Ok(topk)
}

/// Score a chunk against the binary (sign-bit) matrix into a bounded top-k of `overscan`
/// candidates, mirroring [`score_chunk_i8`]. Score is `-(hamming)`, monotone with cosine rank for
/// unit vectors, and carries `row` for the rerank. `min_score` is applied there, not here.
pub(super) fn score_chunk_bin<'a>(
    words: &[u64],
    words_per_row: usize,
    chunk: &[(u64, &'a str, &'a str)],
    q_words: &[u64],
    overscan: usize,
) -> Result<TopK<(u64, &'a str, &'a str)>> {
    let mut topk: TopK<(u64, &'a str, &'a str)> = TopK::new(overscan);
    for block in chunk.chunks(CHECK_EVERY) {
        check()?;
        for &(row, col_name, id) in block {
            let base = row as usize * words_per_row;
            let end = base + words_per_row;
            if end > words.len() {
                continue;
            }
            let approx_score = -(hamming(q_words, &words[base..end]) as f32);
            topk.offer(approx_score, (row, col_name, id));
        }
    }
    Ok(topk)
}

/// Split `scan` across `workers` threads, score each chunk with `score_one` into its
/// own bounded top-k of capacity `cap`, then merge the per-worker results into one.
/// The shared parallel-scan engine behind both the f32 and int8 first passes.
pub(super) fn parallel_topk<'a, T, F>(
    scan: &mut [(u64, &'a str, &'a str)],
    workers: usize,
    cap: usize,
    score_one: F,
) -> Result<TopK<T>>
where
    T: Ord + Send,
    F: Fn(&[(u64, &'a str, &'a str)]) -> Result<TopK<T>> + Sync,
{
    let chunk_len = scan.len().div_ceil(workers);
    let score_one = &score_one;
    // Cancellation is ambient per-thread (`crate::cancel`) and a fresh worker inherits nothing, so
    // capture the caller's token here and re-install it in each worker. The one handoff the ambient
    // model needs, which is why it lives in the shared fan-out rather than each call site.
    let token = crate::cancel::current();
    let locals = std::thread::scope(|s| -> Result<Vec<Vec<(f32, T)>>> {
        let handles: Vec<_> = scan
            .chunks_mut(chunk_len)
            .map(|chunk| {
                let token = token.clone();
                s.spawn(move || {
                    chunk.sort_unstable_by_key(|&(row, _, _)| row);
                    let score = || score_one(chunk).map(TopK::into_sorted_desc);
                    match token {
                        Some(token) => token.scope(score),
                        None => score(),
                    }
                })
            })
            .collect();
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(
                h.join()
                    .map_err(|_| anyhow!("search worker thread panicked"))??,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cancel;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// **Cancellation must reach the parallel scan's worker threads.**
    #[test]
    fn cancellation_reaches_every_parallel_worker() {
        const WORKERS: usize = 4;
        let mut scan: Vec<(u64, &str, &str)> = (0..64).map(|i| (i, "col", "id")).collect();

        // Each worker records whether it observed the token. All four must.
        let saw_token = AtomicUsize::new(0);
        let ran = AtomicUsize::new(0);
        let token = Cancel::new();
        token.cancel();
        let result: Result<TopK<(&str, &str)>> = token.scope(|| {
            parallel_topk(&mut scan, WORKERS, 4, |_chunk| {
                ran.fetch_add(1, Ordering::Relaxed);
                if crate::cancel::cancelled() {
                    saw_token.fetch_add(1, Ordering::Relaxed);
                }
                crate::cancel::check()?;
                Ok(TopK::new(4))
            })
        });

        assert_eq!(ran.load(Ordering::Relaxed), WORKERS, "every chunk ran");
        assert_eq!(
            saw_token.load(Ordering::Relaxed),
            WORKERS,
            "every worker must observe the caller's cancellation, not just the first"
        );
        assert!(
            result.is_err(),
            "a cancelled parallel scan must surface the error, not partial results"
        );
    }

    /// Without a token installed, workers see no cancellation — the ordinary path, and the
    /// one a bug in the handoff would most plausibly break.
    #[test]
    fn workers_see_no_cancellation_when_none_is_installed() {
        let mut scan: Vec<(u64, &str, &str)> = (0..64).map(|i| (i, "col", "id")).collect();
        let saw_token = AtomicUsize::new(0);
        let result: Result<TopK<(&str, &str)>> = parallel_topk(&mut scan, 4, 4, |_chunk| {
            if crate::cancel::cancelled() {
                saw_token.fetch_add(1, Ordering::Relaxed);
            }
            Ok(TopK::new(4))
        });
        assert_eq!(saw_token.load(Ordering::Relaxed), 0);
        assert!(result.is_ok());
    }
}
