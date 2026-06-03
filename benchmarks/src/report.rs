//! Human-readable table + machine-readable JSON, plus the parity threshold check.

use std::time::Duration;

use serde_json::{Value, json};

use crate::{Cell, EngineResult, RunCfg, topk_agreement};

const REFERENCE: &str = "nidus";

// ── formatting helpers ───────────────────────────────────────────────────────

pub fn fmt_dur(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us < 1.0 {
        format!("{:.0}ns", us * 1e3)
    } else if us < 1000.0 {
        format!("{us:.1}us")
    } else {
        format!("{:.2}ms", us / 1000.0)
    }
}

pub fn fmt_count(x: f64) -> String {
    if x >= 1e6 {
        format!("{:.2}M", x / 1e6)
    } else if x >= 1e3 {
        format!("{:.0}k", x / 1e3)
    } else {
        format!("{x:.0}")
    }
}

pub fn fmt_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.2}GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1}MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.1}KB", b / 1e3)
    } else {
        format!("{b:.0}B")
    }
}

// ── table ────────────────────────────────────────────────────────────────────

/// Print one cell's results as an aligned table, plus the parity cross-check.
pub fn print_cell(cell: Cell, results: &[EngineResult]) {
    println!("\n── exact KNN  {cell}  ──────────────────────────────────────");
    println!(
        "{:<10} {:>10} {:>12} {:>10} {:>10} {:>10} {:>10}",
        "engine", "build", "ingest/s", "q_p50", "q_p95", "q_p99", "disk"
    );
    for r in results {
        println!(
            "{:<10} {:>10} {:>12} {:>10} {:>10} {:>10} {:>10}",
            r.engine,
            fmt_dur(r.build),
            fmt_count(r.ingest_per_s),
            fmt_dur(r.query.p50),
            fmt_dur(r.query.p95),
            fmt_dur(r.query.p99),
            fmt_bytes(r.disk_bytes),
        );
    }

    // Parity: top-k agreement of every other engine vs the reference (nidus).
    if let Some(reference) = results.iter().find(|r| r.engine == REFERENCE) {
        let others: Vec<_> = results.iter().filter(|r| r.engine != REFERENCE).collect();
        if !others.is_empty() {
            print!("  parity vs {REFERENCE} (top-k id agreement):");
            for o in others {
                print!(
                    "  {}={:.1}%",
                    o.engine,
                    topk_agreement(reference, o) * 100.0
                );
            }
            println!();
        }
    }
}

// ── threshold ──────────────────────────────────────────────────────────────

/// For one cell: is nidus's p50 within `factor`× the best (lowest) p50 across engines?
/// `None` when nidus isn't present or it's the only engine (nothing to compare).
pub fn cell_passes(results: &[EngineResult], factor: f64) -> Option<bool> {
    let nidus = results.iter().find(|r| r.engine == REFERENCE)?;
    let best = results
        .iter()
        .map(|r| r.query.p50)
        .min()
        .filter(|_| results.len() > 1)?;
    Some(nidus.query.p50.as_secs_f64() <= best.as_secs_f64() * factor)
}

// ── JSON artifact ────────────────────────────────────────────────────────────

fn timings_json(r: &EngineResult) -> Value {
    json!({
        "build_ms": r.build.as_secs_f64() * 1e3,
        "ingest_per_s": r.ingest_per_s,
        "q_p50_ms": r.query.p50.as_secs_f64() * 1e3,
        "q_p95_ms": r.query.p95.as_secs_f64() * 1e3,
        "q_p99_ms": r.query.p99.as_secs_f64() * 1e3,
        "q_mean_ms": r.query.mean.as_secs_f64() * 1e3,
        "q_samples": r.query.count,
        "disk_bytes": r.disk_bytes,
    })
}

/// Assemble the full run as JSON for diffing across runs. `stamp` is supplied by the
/// caller (e.g. a unix timestamp) so this module stays free of wall-clock calls.
pub fn to_json(seed: u64, cfg: &RunCfg, cells: &[(Cell, Vec<EngineResult>)], stamp: u64) -> Value {
    let cell_json: Vec<Value> = cells
        .iter()
        .map(|(cell, results)| {
            let engines: serde_json::Map<String, Value> = results
                .iter()
                .map(|r| (r.engine.to_string(), timings_json(r)))
                .collect();
            json!({
                "n": cell.n,
                "dim": cell.dim,
                "top_k": cell.top_k,
                "engines": Value::Object(engines),
            })
        })
        .collect();

    json!({
        "stamp": stamp,
        "seed": seed,
        "warmup": cfg.warmup,
        "iters": cfg.iters,
        "search": "exact-knn-cosine",
        "cells": cell_json,
    })
}
