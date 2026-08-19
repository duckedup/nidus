//! `nidus tune`: parse the sweep-list flags into a [`TuneOpts`], call `nidus::tune`
//! against the caller's own store, and render the report. All measurement lives in
//! `crate::tune`; this module is the thin CLI adapter, print-only, never mutating.

use anyhow::{Result, anyhow};

use super::{StoreArgs, print_json};
use crate::server::dto::AnnDto;
use crate::{AnnConfig, AnnKind, OpenMode, TuneCell, TuneOpts, TuneReport};

/// `nidus tune`: opens read-only, sweeps `ef_search`/`n_probe`/`overscan` (and
/// optionally quantization) against the store's own vectors, and prints the report.
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    store: StoreArgs,
    collection: Option<String>,
    top_k: usize,
    sample: usize,
    ef_search: Vec<usize>,
    n_probe: Vec<usize>,
    overscan: Vec<usize>,
    sweep_quantization: bool,
    target_recall: f64,
    seed: Option<u64>,
) -> Result<()> {
    let cfg = store.config(OpenMode::ReadOnly)?;
    let base_ann = cfg.ann.ok_or_else(|| {
        anyhow!("--ann hnsw|ivf must be set on `tune` to pick a baseline index to sweep")
    })?;

    let defaults = TuneOpts::default();
    let params = match base_ann.kind {
        AnnKind::Hnsw if !ef_search.is_empty() => ef_search,
        AnnKind::Ivf if !n_probe.is_empty() => n_probe,
        _ => defaults.params.clone(),
    };
    let overscan = if overscan.is_empty() {
        defaults.overscan.clone()
    } else {
        overscan
    };
    let quantizations = if sweep_quantization {
        defaults.quantizations.clone()
    } else {
        vec![cfg.quantization]
    };

    let opts = TuneOpts {
        top_k,
        sample_size: sample,
        seed: seed.unwrap_or(defaults.seed),
        target_recall,
        params,
        overscan,
        quantizations,
        collections: collection.map(|c| vec![c]),
    };

    let report = crate::tune(&cfg, &opts)?;
    render(base_ann, target_recall, opts.seed, sample, &report)
}

/// Render a [`TuneReport`] as the JSON document `nidus tune` prints: every swept
/// cell, the recommended `Config` knobs, the sample the sweep drew from (including
/// the self-hit policy in words, per the ticket), and how to persist the pick.
fn render(
    base_ann: AnnConfig,
    target_recall: f64,
    seed: u64,
    sample_size: usize,
    report: &TuneReport,
) -> Result<()> {
    let cells: Vec<_> = report
        .cells
        .iter()
        .map(|c| cell_json(base_ann, c))
        .collect();
    print_json(&serde_json::json!({
        "cells": cells,
        "recommended": {
            "config": cell_json(base_ann, &report.recommended),
            "target_recall": target_recall,
            "target_met": report.target_met,
        },
        "sample": {
            "size": sample_size,
            "seed": seed,
            "self_hit_policy": report.self_hit_policy,
        },
        "next_step": "persist the recommended ann/quantization knobs with `nidus configure`",
    }))
}

/// One cell's `AnnConfig` (the swept param + overscan folded back onto the
/// baseline), quantization, recall@k, and latency, as a JSON object.
fn cell_json(base_ann: AnnConfig, cell: &TuneCell) -> serde_json::Value {
    let mut ann = base_ann;
    match ann.kind {
        AnnKind::Hnsw => ann.ef_search = cell.param,
        AnnKind::Ivf => ann.n_probe = cell.param,
    }
    ann.overscan = cell.overscan;
    serde_json::json!({
        "ann": AnnDto::from(ann),
        "quantization": cell.quantization,
        "recall_at_k": cell.recall_at_k,
        "p50_micros": cell.p50_micros,
        "p95_micros": cell.p95_micros,
    })
}
