//! Recall/latency sweep against the caller's own store (nidus-sk9). Exact brute-force and
//! the ANN walk share one binary, so ground truth needs no external dataset: sample the
//! store's own vectors as queries, score each configured cell against an exact-search
//! ground truth, and recommend the cheapest cell that clears a target recall.

use std::collections::HashSet;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use serde::Serialize;

use crate::ann::SplitMix64;
use crate::{AnnKind, Config, Distance, Nidus, OpenMode, QuantKind, Quantization, SearchOpts};

/// recall@k = mean over queries of |returned ∩ truth| / |truth|. Generic over the id
/// type: [`tune`] keys on `(collection, id)`, since `Hit` ids are unique only within a
/// collection; a single-collection caller can key on a bare id instead.
pub fn recall_at_k<K: Eq + std::hash::Hash>(returned: &[Vec<K>], truth: &[Vec<K>]) -> f64 {
    let q = returned.len().min(truth.len());
    if q == 0 {
        return 1.0;
    }
    let mut acc = 0.0;
    for i in 0..q {
        let t: HashSet<&K> = truth[i].iter().collect();
        let hit = returned[i].iter().filter(|id| t.contains(id)).count();
        acc += hit as f64 / truth[i].len().max(1) as f64;
    }
    acc / q as f64
}

/// Sweep knobs for [`tune`]. Quantization is the outer loop (each value needs its own
/// store open); `params` sweeps `ef_search` (HNSW) or `n_probe` (IVF) per `Config::ann`'s
/// kind, crossed with `overscan`.
#[derive(Clone, Debug)]
pub struct TuneOpts {
    /// `k` in recall@k, and `top_k` for every search the sweep runs.
    pub top_k: usize,
    /// How many stored vectors to sample as queries (clamped to the scope's own size).
    pub sample_size: usize,
    /// SplitMix64 seed for the deterministic sample.
    pub seed: u64,
    /// Minimum recall a cell must clear to be recommended by lowest p50 latency.
    pub target_recall: f64,
    /// `ef_search` (HNSW) or `n_probe` (IVF) values to sweep.
    pub params: Vec<usize>,
    /// Overscan values to sweep, crossed with `params`.
    pub overscan: Vec<usize>,
    /// Quantization cells to sweep, outermost; `None` is the unquantized f32 index.
    pub quantizations: Vec<Option<Quantization>>,
    /// Collections to sample and search over; `None` sweeps every collection.
    pub collections: Option<Vec<String>>,
}

impl Default for TuneOpts {
    fn default() -> Self {
        Self {
            top_k: 10,
            sample_size: 200,
            seed: 0x5EED,
            target_recall: 0.95,
            params: vec![16, 32, 64, 128],
            overscan: vec![1, 2, 4],
            quantizations: vec![
                None,
                Some(Quantization::int8()),
                Some(Quantization::binary()),
            ],
            collections: None,
        }
    }
}

/// One swept cell's measurement.
#[derive(Clone, Debug, Serialize)]
pub struct TuneCell {
    pub quantization: Option<Quantization>,
    /// The `ef_search` (HNSW) or `n_probe` (IVF) value this cell used.
    pub param: usize,
    pub overscan: usize,
    pub recall_at_k: f64,
    pub p50_micros: u64,
    pub p95_micros: u64,
}

/// The full sweep: every measured cell, plus a recommendation.
#[derive(Clone, Debug, Serialize)]
pub struct TuneReport {
    pub cells: Vec<TuneCell>,
    /// Human-readable self-hit policy, so a CLI can print it verbatim.
    pub self_hit_policy: String,
    /// The lowest-p50 cell clearing `target_recall`, or the highest-recall cell if none did.
    pub recommended: TuneCell,
    /// Whether `recommended` actually cleared `TuneOpts::target_recall`.
    pub target_met: bool,
}

/// One sampled query: a stored vector plus the `(collection, id)` it must not count as
/// its own hit when scoring recall.
#[derive(Clone, Debug, PartialEq)]
struct SampledQuery {
    collection: String,
    id: String,
    vector: Vec<f32>,
}

/// Deterministic sample of up to `sample_size` vector-bearing records from `collections`.
/// Sorted by `(collection, id)` first so the SplitMix64 draw depends only on `seed` and
/// the row count, never on the store's own (hash-map) iteration order.
fn sample_queries(
    nidus: &Nidus,
    collections: &[String],
    sample_size: usize,
    seed: u64,
) -> Vec<SampledQuery> {
    let mut all: Vec<SampledQuery> = Vec::new();
    for c in collections {
        for rec in nidus.get_all(c) {
            if let Some(vector) = rec.vector {
                all.push(SampledQuery {
                    collection: c.clone(),
                    id: rec.id,
                    vector,
                });
            }
        }
    }
    all.sort_by(|a, b| (&a.collection, &a.id).cmp(&(&b.collection, &b.id)));
    let n = all.len();
    let take = sample_size.min(n);
    if take == 0 {
        return Vec::new();
    }
    let mut rng = SplitMix64::new(seed);
    for i in 0..take {
        let j = i + rng.below(n - i);
        all.swap(i, j);
    }
    all.truncate(take);
    all
}

/// The self-hit policy: drop the hit matching the query's own `(collection, id)` — a
/// query sampled from the store's own vectors is a guaranteed exact match against
/// itself, which would otherwise flatter every cell's recall — then truncate to `k`.
fn strip_self_hit(
    hits: Vec<crate::Hit>,
    self_collection: &str,
    self_id: &str,
    k: usize,
) -> Vec<(String, String)> {
    hits.into_iter()
        .filter(|h| !(h.collection == self_collection && h.id == self_id))
        .take(k)
        .map(|h| (h.collection, h.id))
        .collect()
}

fn percentile_micros(sorted: &[u64], pct: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(sorted.len() * pct / 100).min(sorted.len() - 1)]
}

/// Whether a quantization cell can even be opened against `config`. Binary codes are an
/// angular proxy valid only for cosine (model.rs), and no quantization composes with
/// per-segment indexing (config.rs) — skip those cells rather than failing the whole sweep.
fn quant_supported(q: Option<Quantization>, config: &Config) -> bool {
    let Some(q) = q else {
        return true;
    };
    if config.segment_max_rows.is_some() && config.segment_index_min_rows.is_some() {
        return false;
    }
    q.kind != QuantKind::Binary || config.distance == Distance::Cosine
}

/// Sweep `Config::ann`'s query-time knobs and `opts.quantizations` against the store's own
/// vectors, and recommend a cell. Quantization is the outer loop (its own read-only open
/// per value); the ann grid retunes in place inside that open — one index build per cell.
pub fn tune(config: &Config, opts: &TuneOpts) -> Result<TuneReport> {
    let base_ann = config
        .ann
        .ok_or_else(|| anyhow!("Config::ann must be set to tune ef_search/n_probe/overscan"))?;
    if opts.params.is_empty() || opts.overscan.is_empty() {
        bail!("TuneOpts::params and TuneOpts::overscan must both be non-empty");
    }

    let mut cells = Vec::new();
    for quant in opts
        .quantizations
        .iter()
        .copied()
        .filter(|q| quant_supported(*q, config))
    {
        let cell_config = config
            .clone()
            .open_mode(OpenMode::ReadOnly)
            .quantization(quant)
            .ann(Some(base_ann));
        let mut nidus = Nidus::open(cell_config)?;

        let names = opts
            .collections
            .clone()
            .unwrap_or_else(|| nidus.collections());
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();

        let sample = sample_queries(&nidus, &names, opts.sample_size, opts.seed);
        if sample.is_empty() {
            bail!("no vector-bearing records found to sample in the given scope");
        }

        let mut truth: Vec<Vec<(String, String)>> = Vec::with_capacity(sample.len());
        for s in &sample {
            let hits = nidus.search(
                refs.as_slice(),
                &s.vector,
                &SearchOpts {
                    top_k: opts.top_k + 1,
                    exact: true,
                    ..Default::default()
                },
            )?;
            truth.push(strip_self_hit(hits, &s.collection, &s.id, opts.top_k));
        }

        for &raw_param in &opts.params {
            for &raw_overscan in &opts.overscan {
                // Clamp once and report the clamped values: a cell that measured with 1 must not
                // claim it swept 0, or the recommended block hands back a config never benchmarked.
                let (param, overscan) = (raw_param.max(1), raw_overscan.max(1));
                let mut ann_cfg = base_ann;
                match ann_cfg.kind {
                    AnnKind::Hnsw => ann_cfg.ef_search = param,
                    AnnKind::Ivf => ann_cfg.n_probe = param,
                }
                ann_cfg.overscan = overscan;
                nidus.retune_ann(ann_cfg);

                let mut returned = Vec::with_capacity(sample.len());
                let mut micros = Vec::with_capacity(sample.len());
                for s in &sample {
                    let started = Instant::now();
                    let hits = nidus.search(
                        refs.as_slice(),
                        &s.vector,
                        &SearchOpts {
                            top_k: opts.top_k + 1,
                            ..Default::default()
                        },
                    )?;
                    micros.push(started.elapsed().as_micros() as u64);
                    returned.push(strip_self_hit(hits, &s.collection, &s.id, opts.top_k));
                }
                micros.sort_unstable();

                cells.push(TuneCell {
                    quantization: quant,
                    param,
                    overscan,
                    recall_at_k: recall_at_k(&returned, &truth),
                    p50_micros: percentile_micros(&micros, 50),
                    p95_micros: percentile_micros(&micros, 95),
                });
            }
        }
    }

    if cells.is_empty() {
        bail!(
            "no quantization cell in TuneOpts::quantizations was compatible with this store's \
             config (binary needs Distance::Cosine; no quantization composes with per-segment \
             indexing)"
        );
    }
    let qualifying = cells
        .iter()
        .filter(|c| c.recall_at_k >= opts.target_recall)
        .min_by(|a, b| a.p50_micros.cmp(&b.p50_micros));
    let (recommended, target_met) = match qualifying {
        Some(c) => (c.clone(), true),
        None => (
            cells
                .iter()
                .max_by(|a, b| a.recall_at_k.total_cmp(&b.recall_at_k))
                .cloned()
                .expect("cells is non-empty, checked above"),
            false,
        ),
    };

    Ok(TuneReport {
        cells,
        self_hit_policy: "each leg fetches k+1 and drops the hit matching the sampled query's \
            own (collection, id) before truncating to k"
            .to_string(),
        recommended,
        target_met,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnnConfig;
    use std::collections::BTreeMap;

    // ── recall_at_k (pure, Miri-clean) ───────────────────────────────────

    #[test]
    fn recall_at_k_perfect_overlap() {
        let returned = vec![vec![1u64, 2, 3]];
        let truth = vec![vec![1u64, 2, 3]];
        assert_eq!(recall_at_k(&returned, &truth), 1.0);
    }

    #[test]
    fn recall_at_k_partial_overlap() {
        let returned = vec![vec![1u64, 2, 99]];
        let truth = vec![vec![1u64, 2, 3]];
        assert!((recall_at_k(&returned, &truth) - (2.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn recall_at_k_zero_overlap() {
        let returned = vec![vec![97u64, 98, 99]];
        let truth = vec![vec![1u64, 2, 3]];
        assert_eq!(recall_at_k(&returned, &truth), 0.0);
    }

    #[test]
    fn recall_at_k_no_queries_is_perfect() {
        let returned: Vec<Vec<u64>> = vec![];
        let truth: Vec<Vec<u64>> = vec![];
        assert_eq!(recall_at_k(&returned, &truth), 1.0);
    }

    #[test]
    fn recall_at_k_empty_truth_list_scores_zero_not_nan() {
        let returned = vec![vec![1u64]];
        let truth = vec![vec![]];
        assert_eq!(recall_at_k(&returned, &truth), 0.0);
    }

    #[test]
    fn recall_at_k_same_id_different_collection_is_not_a_match() {
        // Two collections sharing id "42": only the (col, id) pair identifies a document.
        let returned = vec![vec![("a".to_string(), "42".to_string())]];
        let truth = vec![vec![("b".to_string(), "42".to_string())]];
        assert_eq!(recall_at_k(&returned, &truth), 0.0);
    }

    // ── sampler (pure, Miri-clean) ────────────────────────────────────────

    fn seed_store(n: usize, dim: usize) -> Nidus {
        let mut db = Nidus::open_in_memory(dim).unwrap();
        let recs: Vec<crate::Record> = (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0;
                crate::Record::new(format!("d{i}"), v, BTreeMap::new())
            })
            .collect();
        db.upsert("col", &recs).unwrap();
        db
    }

    #[test]
    fn sample_queries_is_deterministic_for_same_seed_and_size() {
        let db = seed_store(50, 8);
        let names = vec!["col".to_string()];
        let a = sample_queries(&db, &names, 10, 42);
        let b = sample_queries(&db, &names, 10, 42);
        assert_eq!(a, b);
        assert_eq!(a.len(), 10);
    }

    #[test]
    fn sample_queries_clamps_when_sample_size_exceeds_store() {
        let db = seed_store(3, 8);
        let names = vec!["col".to_string()];
        let sample = sample_queries(&db, &names, 1000, 1);
        assert_eq!(sample.len(), 3);
    }

    // ── self-hit strip (pure, Miri-clean) ────────────────────────────────

    #[test]
    fn strip_self_hit_drops_only_the_query_and_keeps_k_others() {
        let hits = vec![
            crate::Hit::new("col", "self", 1.0, BTreeMap::new()),
            crate::Hit::new("col", "a", 0.9, BTreeMap::new()),
            crate::Hit::new("col", "b", 0.8, BTreeMap::new()),
            crate::Hit::new("col", "c", 0.7, BTreeMap::new()),
        ];
        let kept = strip_self_hit(hits, "col", "self", 2);
        assert_eq!(
            kept,
            vec![
                ("col".to_string(), "a".to_string()),
                ("col".to_string(), "b".to_string()),
            ]
        );
    }

    // ── file-backed sweep (needs a real store; not Miri-clean) ───────────

    /// A swept `0` is clamped to `1` before it is measured, so the cell must REPORT `1` —
    /// reporting the raw `0` would advertise a config that was never benchmarked, and the
    /// output tells the user to persist it with `nidus configure`.
    #[test]
    #[cfg_attr(miri, ignore)] // runtime cost: a few-hundred-vector IVF build is too slow under Miri.
    fn swept_zero_is_reported_as_the_clamped_value_actually_measured() {
        let dir = tempfile::tempdir().unwrap();
        let (n, dim) = (60, 8);
        let base_ann = AnnConfig::ivf().n_lists(4);
        {
            let mut db = Nidus::open(
                Config::new(dir.path(), dim)
                    .ann(Some(base_ann))
                    .auto_compact(None),
            )
            .unwrap();
            let recs: Vec<crate::Record> = (0..n)
                .map(|i| {
                    let mut v = vec![0.0f32; dim];
                    v[i % dim] = 1.0;
                    crate::Record::new(format!("d{i}"), v, BTreeMap::new())
                })
                .collect();
            db.upsert("col", &recs).unwrap();
            db.flush().unwrap();
        }

        let report = tune(
            &Config::new(dir.path(), dim).ann(Some(base_ann)),
            &TuneOpts {
                top_k: 3,
                sample_size: 10,
                seed: 1,
                target_recall: 0.9,
                params: vec![0],
                overscan: vec![0],
                quantizations: vec![None],
                collections: None,
            },
        )
        .unwrap();

        let cell = &report.cells[0];
        assert_eq!(
            cell.param, 1,
            "a swept 0 must report the clamped 1 it measured"
        );
        assert_eq!(
            cell.overscan, 1,
            "a swept overscan 0 must report the clamped 1"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // runtime cost: a few-hundred-vector IVF build is too slow under Miri.
    fn sweep_scores_generous_cell_exact_and_starved_cell_lower() {
        // IVF with `n_probe == n_lists` scans every list, i.e. every row: with no
        // quantization that is scored *exactly*, unlike HNSW's approximate graph walk —
        // the only way to make the "generous cell hits 1.0" assertion non-flaky.
        let dir = tempfile::tempdir().unwrap();
        let (n, dim, n_lists) = (300, 16, 10);
        let mut rng = SplitMix64::new(11);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
                    .collect();
                crate::search::normalize(&mut v);
                v
            })
            .collect();

        let base_ann = AnnConfig::ivf().n_lists(n_lists);
        {
            let mut db = Nidus::open(
                Config::new(dir.path(), dim)
                    .ann(Some(base_ann))
                    .auto_compact(None),
            )
            .unwrap();
            let recs: Vec<crate::Record> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| crate::Record::new(format!("d{i}"), v.clone(), BTreeMap::new()))
                .collect();
            db.upsert("col", &recs).unwrap();
            db.flush().unwrap();
        }

        let read_config = Config::new(dir.path(), dim).ann(Some(base_ann));
        let opts = TuneOpts {
            top_k: 5,
            sample_size: 40,
            seed: 5,
            target_recall: 0.99,
            params: vec![1, n_lists],
            overscan: vec![1, 8],
            quantizations: vec![None],
            collections: None,
        };
        let report = tune(&read_config, &opts).unwrap();

        let generous = report
            .cells
            .iter()
            .find(|c| c.param == n_lists && c.overscan == 8)
            .unwrap();
        let starved = report
            .cells
            .iter()
            .find(|c| c.param == 1 && c.overscan == 1)
            .unwrap();
        assert_eq!(
            generous.recall_at_k, 1.0,
            "probing every IVF list with no quantization must match the exact ground truth"
        );
        assert!(
            starved.recall_at_k < generous.recall_at_k,
            "starved cell ({}) should score lower than generous ({})",
            starved.recall_at_k,
            generous.recall_at_k
        );
    }
}
