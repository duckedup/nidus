//! Scan kernels: the per-chunk scoring functions and the parallel-scan engine that
//! the brute-force ([`super::read`]) and quantized ([`super::quant`]) search paths
//! share. Pure functions over borrowed data — no [`Store`](super::Store) state.

use anyhow::{Result, anyhow};

use crate::cancel::{CHECK_EVERY, check};
use crate::data::Segments;
use crate::model::Pool;
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

// ── Named-vector reduction (nidus-85t) ────────────────────────────────────────

/// Score a named-vector scan into a bounded top-k, pooling each record's contiguous run of
/// named rows into one score **before** the push — `TopK` has no dedup by key, so pushing every
/// row would yield duplicate hits. `chunk` must be row-sorted, so one record's rows are adjacent.
pub(super) struct NamedReduce<'w> {
    /// Per-name multiplier, indexed by the scan entry's name index. Resolved once per query
    /// rather than looked up per scanned row: a `BTreeMap<String, f32>` probe inside the hot
    /// loop cost more than the dot products it guarded (nidus-85t review).
    pub(super) weights: &'w [f32],
    /// How a record's weighted per-name scores fold into its one record score.
    pub(super) pool: Pool,
}

pub(super) fn score_chunk_named<'a>(
    data: &Segments,
    chunk: &[(u64, &'a str, &'a str, u32)],
    q: &[f32],
    score_fn: fn(&[f32], &[f32]) -> f32,
    top_k: usize,
    min_score: Option<f32>,
    reduce: &NamedReduce<'_>,
) -> Result<TopK<(&'a str, &'a str)>> {
    let NamedReduce { weights, pool } = *reduce;
    let mut topk: TopK<(&'a str, &'a str)> = TopK::new(top_k);
    let mut i = 0;
    let mut groups_seen: usize = 0;
    while i < chunk.len() {
        let (_, col, id, _) = chunk[i];
        let mut j = i + 1;
        while j < chunk.len() && chunk[j].1 == col && chunk[j].2 == id {
            j += 1;
        }
        groups_seen += 1;
        if groups_seen.is_multiple_of(CHECK_EVERY) {
            check()?;
        }
        let mut acc: Option<f32> = None;
        for &(row, _, _, name_ix) in &chunk[i..j] {
            let w = weights.get(name_ix as usize).copied().unwrap_or(1.0);
            let weighted = score_fn(q, data.row(row)) * w;
            acc = Some(match (acc, pool) {
                (None, _) => weighted,
                (Some(a), Pool::Max) => a.max(weighted),
                (Some(a), Pool::Sum) => a + weighted,
            });
        }
        if let Some(score) = acc
            && min_score.is_none_or(|m| score >= m)
        {
            topk.offer(score, (col, id));
        }
        i = j;
    }
    Ok(topk)
}

/// Split `scan` into up to `workers` pieces, nudging each boundary forward past any run sharing
/// `(collection, id)` with the entry just before it — so a record's contiguous rows always land
/// in one shard, and pooling inside [`score_chunk_named`] stays exact.
fn record_aligned_split<'a, 'b>(
    scan: &'a mut [(u64, &'b str, &'b str, u32)],
    workers: usize,
) -> Vec<&'a mut [(u64, &'b str, &'b str, u32)]> {
    let n = scan.len();
    if workers <= 1 || n == 0 {
        return vec![scan];
    }
    let target = n.div_ceil(workers);
    let mut cuts: Vec<usize> = Vec::new();
    let mut pos = target.min(n);
    while pos < n {
        while pos < n && scan[pos].1 == scan[pos - 1].1 && scan[pos].2 == scan[pos - 1].2 {
            pos += 1;
        }
        cuts.push(pos);
        pos += target;
    }
    let mut out = Vec::with_capacity(cuts.len() + 1);
    let mut rest: &mut [_] = scan;
    let mut prev = 0;
    for cut in cuts {
        let (a, b) = rest.split_at_mut(cut - prev);
        out.push(a);
        rest = b;
        prev = cut;
    }
    out.push(rest);
    out
}

/// [`parallel_topk`] for a named-vector scan: shards by [`record_aligned_split`] rather than an
/// even `chunks_mut`, so a record's rows never straddle two workers — pooling then stays exact
/// *and* parallel, with no serial fallback and no post-hoc collapse.
pub(super) fn parallel_topk_grouped<'a, T, F>(
    scan: &mut [(u64, &'a str, &'a str, u32)],
    workers: usize,
    cap: usize,
    score_one: F,
) -> Result<TopK<T>>
where
    T: Ord + Send,
    F: Fn(&[(u64, &'a str, &'a str, u32)]) -> Result<TopK<T>> + Sync,
{
    let score_one = &score_one;
    let token = crate::cancel::current();
    let shards = record_aligned_split(scan, workers);
    let locals = std::thread::scope(|s| -> Result<Vec<Vec<(f32, T)>>> {
        let handles: Vec<_> = shards
            .into_iter()
            .map(|chunk| {
                let token = token.clone();
                s.spawn(move || {
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

    // ── Named-vector reduction (nidus-85t) ────────────────────────────────────

    /// **A record's rows must never straddle two shards.** Sized so a plain `chunks_mut`
    /// split (target = 2) would cut `r1`'s 3-row run in half; `record_aligned_split` must
    /// snap the boundary forward instead, or pooling silently double-counts a partial group.
    #[test]
    fn parallel_topk_grouped_never_splits_a_records_rows_across_shards() {
        let owned: Vec<(u64, String, String, u32)> = vec![
            (0, "c".into(), "r1".into(), 0),
            (1, "c".into(), "r1".into(), 1),
            (2, "c".into(), "r1".into(), 2),
            (3, "c".into(), "r2".into(), 0),
            (4, "c".into(), "r3".into(), 0),
            (5, "c".into(), "r3".into(), 1),
        ];
        let mut scan: Vec<(u64, &str, &str, u32)> = owned
            .iter()
            .map(|(r, c, i, n)| (*r, c.as_str(), i.as_str(), *n))
            .collect();

        // The per-shard scorer reports each group's row count as its "score", revealing a
        // split group as two smaller entries for the same id instead of one whole one.
        let topk = parallel_topk_grouped(&mut scan, 4, 10, |chunk| {
            let mut out: TopK<(&str, &str)> = TopK::new(10);
            let mut i = 0;
            while i < chunk.len() {
                let (_, col, id, _) = chunk[i];
                let mut j = i + 1;
                while j < chunk.len() && chunk[j].1 == col && chunk[j].2 == id {
                    j += 1;
                }
                out.offer((j - i) as f32, (col, id));
                i = j;
            }
            Ok(out)
        })
        .unwrap();

        let mut seen: std::collections::HashMap<&str, Vec<f32>> = std::collections::HashMap::new();
        for (score, (_, id)) in topk.into_sorted_desc() {
            seen.entry(id).or_default().push(score);
        }
        assert_eq!(seen.len(), 3, "no id split into extra entries: {seen:?}");
        assert_eq!(seen["r1"], vec![3.0], "r1's 3 rows must land in one shard");
        assert_eq!(seen["r2"], vec![1.0]);
        assert_eq!(seen["r3"], vec![2.0], "r3's 2 rows must land in one shard");
    }

    #[test]
    fn score_chunk_named_pools_max_and_sum_and_skips_an_absent_name() {
        // Two rows for "a" (names x, y), one row for "b" (name x only).
        let mut store_data =
            crate::data::Segments::in_memory_with(2, crate::model::Distance::Cosine);
        let ra = store_data.append(&[1.0, 0.0]).unwrap();
        let rb = store_data.append(&[0.0, 1.0]).unwrap();
        let rc = store_data.append(&[1.0, 0.0]).unwrap();
        let chunk = [
            (ra, "col", "a", 0u32),
            (rb, "col", "a", 1u32),
            (rc, "col", "b", 0u32),
        ];
        let q = [1.0, 0.0];
        let weights: Vec<f32> = vec![1.0, 1.0];

        let max_topk = score_chunk_named(
            &store_data,
            &chunk,
            &q,
            crate::search::dot,
            10,
            None,
            &NamedReduce {
                weights: &weights,
                pool: Pool::Max,
            },
        )
        .unwrap();
        let max: std::collections::HashMap<&str, f32> = max_topk
            .into_sorted_desc()
            .into_iter()
            .map(|(s, (_, id))| (id, s))
            .collect();
        assert!(
            (max["a"] - 1.0).abs() < 1e-6,
            "max(1.0, 0.0) == 1.0, {max:?}"
        );
        assert!(
            (max["b"] - 1.0).abs() < 1e-6,
            "b has only `x`, unpenalized for missing `y`"
        );

        let sum_topk = score_chunk_named(
            &store_data,
            &chunk,
            &q,
            crate::search::dot,
            10,
            None,
            &NamedReduce {
                weights: &weights,
                pool: Pool::Sum,
            },
        )
        .unwrap();
        let sum: std::collections::HashMap<&str, f32> = sum_topk
            .into_sorted_desc()
            .into_iter()
            .map(|(s, (_, id))| (id, s))
            .collect();
        assert!((sum["a"] - 1.0).abs() < 1e-6, "1.0 + 0.0 == 1.0, {sum:?}");
        assert!(
            (sum["b"] - 1.0).abs() < 1e-6,
            "single term: sum == that term"
        );
    }
}
