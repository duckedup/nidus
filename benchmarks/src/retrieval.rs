//! nidus-bench-retrieval — BEIR retrieval quality (nidus-yq9p.5): nDCG@10 and Recall@100 for
//! four legs (FTS only, vector only, RRF fusion at the shipped defaults, fusion + rerank) over
//! one store per dataset carrying both an FTS field and voyage-4 vectors, against the BEIR
//! paper's own published BM25 numbers. Needs a live `VOYAGE_API_KEY` and network; never run in
//! CI (`bench-compiles` only compiles it). Mirrors `named.rs`: hand-rolled `key=value` args,
//! `fn main() -> ExitCode`, `write_json` reading `NIDUS_VERSION` from the environment.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result, anyhow, bail};
use nidus::embed::Embedder;
use nidus::rerank::{AnyReranker, RerankConfig, RerankProvider, hybrid_reranked};
use nidus::{Config, FtsField, FtsQuery, HybridOpts, Nidus, Record, RerankOpts, SearchOpts, Value};
use nidus_bench::beir::{self, DatasetSpec};
use nidus_bench::{embed_cache, qmetrics, recall_at_k};
use serde_json::{Value as Json, json};

const COLLECTION: &str = "beir";

/// Records per `upsert` call while indexing a corpus, so FiQA's 57,638 docs never build one
/// giant batch in RAM.
const UPSERT_BATCH: usize = 500;

const EMBED_PROVIDER: &str = "voyage";
const EMBED_MODEL: &str = "voyage-4";
const RERANK_PROVIDER: &str = "voyage";
const RERANK_MODEL: &str = "rerank-2.5";

/// Named so the "absent" error can point at it exactly (`embed_cache`'s own copy of this
/// check fires only after a dataset is already downloaded; this one fires before any of it).
const VOYAGE_API_KEY_VAR: &str = "VOYAGE_API_KEY";

// ── args ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Args {
    datasets: Vec<&'static str>,
    top_k: usize,
    recall_k: usize,
    threshold: f32,
    cache: PathBuf,
    json: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            datasets: beir::DATASETS.iter().map(|s| s.name).collect(),
            top_k: 10,
            recall_k: 100,
            threshold: 0.0,
            cache: PathBuf::from("benchmarks/.cache"),
            json: None,
        }
    }
}

fn usage() {
    println!("nidus-bench-retrieval — BEIR retrieval quality: nDCG@10 and Recall@100 for FTS,");
    println!("vector, RRF fusion and fusion + rerank (nidus-yq9p.5). Needs VOYAGE_API_KEY.");
    println!(
        "args: dataset=scifact,nfcorpus,fiqa  top_k=10  recall_k=100  threshold=0  \
         cache=benchmarks/.cache  json=<path>  help"
    );
}

fn parse_args() -> Result<Args> {
    parse_args_from(std::env::args().skip(1))
}

/// The testable core of [`parse_args`], over an arbitrary token iterator rather than the real
/// process argv, so the inline tests need no process exit and no network.
fn parse_args_from(tokens: impl Iterator<Item = String>) -> Result<Args> {
    let mut a = Args::default();
    for tok in tokens {
        if tok == "help" || tok == "--help" || tok == "-h" {
            usage();
            std::process::exit(0);
        }
        let Some((key, val)) = tok.split_once('=') else {
            bail!("expected key=value, got `{tok}` (try `help`)");
        };
        match key {
            "dataset" => a.datasets = parse_datasets(val)?,
            "top_k" => a.top_k = val.parse()?,
            "recall_k" => a.recall_k = val.parse()?,
            "threshold" => a.threshold = val.parse()?,
            "cache" => a.cache = PathBuf::from(val),
            "json" => a.json = Some(PathBuf::from(val)),
            other => bail!("unknown arg `{other}` (try `help`)"),
        }
    }
    Ok(a)
}

/// Resolve a comma-separated `dataset=` value against [`beir::DATASETS`], failing loudly on
/// an unknown name rather than silently skipping it.
fn parse_datasets(val: &str) -> Result<Vec<&'static str>> {
    val.split(',')
        .map(str::trim)
        .map(|name| {
            beir::DATASETS
                .iter()
                .find(|s| s.name == name)
                .map(|s| s.name)
                .with_context(|| {
                    let known: Vec<&str> = beir::DATASETS.iter().map(|s| s.name).collect();
                    format!("unknown dataset `{name}`; available: {}", known.join(", "))
                })
        })
        .collect()
}

// ── hybrid opts (the fusion leg must not tune anything) ─────────────────────

/// `HybridOpts::default()`, mutated only to request `recall_k` results and, if `recall_k`
/// exceeds the default `candidates` depth, to widen it to match (nidus-yq9p.5's one
/// documented caveat) — every RRF weighting knob stays at its shipped default.
fn fusion_opts(recall_k: usize) -> HybridOpts {
    let mut opts = HybridOpts {
        top_k: recall_k,
        ..HybridOpts::default()
    };
    if recall_k > opts.candidates {
        opts.candidates = recall_k;
    }
    opts
}

// ── per-dataset run ──────────────────────────────────────────────────────────

/// One leg's ranked results, ready for [`qmetrics::ndcg_at_k`] (`ranked`) and [`recall_at_k`]
/// (`returned`) — the same ids, so the two metrics can never see different rankings.
struct LegRun {
    label: &'static str,
    ranked: Vec<(String, Vec<String>)>,
    returned: Vec<Vec<String>>,
}

impl LegRun {
    fn new(label: &'static str, capacity: usize) -> Self {
        LegRun {
            label,
            ranked: Vec::with_capacity(capacity),
            returned: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, query_id: &str, ids: Vec<String>) {
        self.returned.push(ids.clone());
        self.ranked.push((query_id.to_string(), ids));
    }

    fn metrics(&self, qrels: &qmetrics::Qrels, top_k: usize, truth: &[Vec<String>]) -> (f64, f64) {
        (
            qmetrics::ndcg_at_k(&self.ranked, qrels, top_k),
            recall_at_k(&self.returned, truth),
        )
    }
}

/// Embedded corpus vectors and embedded query vectors, in their datasets' own order.
type Embeddings = (Vec<Vec<f32>>, Vec<Vec<f32>>);

/// Embed a dataset's corpus and query texts through `embedder`'s document path (see
/// `embed_cache`'s module docs for why queries use `embed_batch`, not `embed_query`), saving
/// as soon as each half succeeds so a later failure never throws away spend already made.
fn embed_all(
    rt: &tokio::runtime::Runtime,
    embedder: &nidus::embed::cache::CachedEmbedder<nidus::embed::AnyEmbedder>,
    dataset: &str,
    doc_texts: &[String],
    query_texts: &[String],
) -> Result<Embeddings> {
    let doc_refs: Vec<&str> = doc_texts.iter().map(String::as_str).collect();
    let doc_vectors = match rt.block_on(embedder.embed_batch(&doc_refs)) {
        Ok(v) => v,
        Err(e) => {
            let _ = embedder.save();
            return Err(anyhow!("embedding {dataset} corpus: {e}"));
        }
    };
    embedder.save()?;

    let query_refs: Vec<&str> = query_texts.iter().map(String::as_str).collect();
    let query_vectors = match rt.block_on(embedder.embed_batch(&query_refs)) {
        Ok(v) => v,
        Err(e) => {
            let _ = embedder.save();
            return Err(anyhow!("embedding {dataset} queries: {e}"));
        }
    };
    embedder.save()?;

    Ok((doc_vectors, query_vectors))
}

/// Build the one store this dataset shares across all four legs: a single collection
/// carrying both the FTS field and every doc's vector, indexed by `text_attr` — the same
/// attr [`RerankOpts::default`] reads, checked once here rather than assumed.
fn build_store(
    text_attr: &str,
    docs: &[(String, String, String)],
    doc_texts: &[String],
    doc_vectors: &[Vec<f32>],
    dim: usize,
) -> Result<(Nidus, tempfile::TempDir)> {
    let dir = tempfile::tempdir()?;
    let mut db = Nidus::open(Config::new(dir.path().join("store"), dim))?;
    db.create_collection_with_fts(COLLECTION, &[FtsField::new(text_attr)])?;

    let mut batch = Vec::with_capacity(UPSERT_BATCH);
    for (i, (doc_id, _, _)) in docs.iter().enumerate() {
        let mut attrs = BTreeMap::new();
        attrs.insert(text_attr.to_string(), Value::Str(doc_texts[i].clone()));
        batch.push(Record::new(doc_id.clone(), doc_vectors[i].clone(), attrs));
        if batch.len() == UPSERT_BATCH {
            db.upsert(COLLECTION, &batch)?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        db.upsert(COLLECTION, &batch)?;
    }
    Ok((db, dir))
}

fn run_dataset(
    spec: &DatasetSpec,
    args: &Args,
    text_attr: &str,
    fusion: &HybridOpts,
    fusion_reranked: &HybridOpts,
    rt: &tokio::runtime::Runtime,
    reranker: &AnyReranker,
) -> Result<Json> {
    println!("\n═══ {} ═══════════════════════════════════════════", spec.name);
    println!("loading corpus (cache: {})...", args.cache.display());
    let corpus = beir::load(spec, &args.cache)?;

    let doc_texts: Vec<String> = corpus
        .docs
        .iter()
        .map(|(_, title, text)| format!("{title}\n\n{text}"))
        .collect();
    let query_texts: Vec<String> = corpus.queries.iter().map(|(_, t)| t.clone()).collect();

    println!(
        "embedding {} docs + {} queries (voyage-4)...",
        corpus.docs.len(),
        corpus.queries.len()
    );
    let identity = embed_cache::cache_identity(EMBED_MODEL, spec.name);
    let embedder = rt.block_on(embed_cache::cached_voyage(&args.cache, &identity))?;
    let (doc_vectors, query_vectors) =
        embed_all(rt, &embedder, spec.name, &doc_texts, &query_texts)?;
    let dim = embedder.dimension();

    let (db, _guard) = build_store(text_attr, &corpus.docs, &doc_texts, &doc_vectors, dim)?;

    println!("running four legs over {} queries...", corpus.queries.len());
    let leg_opts = SearchOpts {
        top_k: args.recall_k,
        ..Default::default()
    };
    let mut fts = LegRun::new("fts only", corpus.queries.len());
    let mut vector = LegRun::new("vector only", corpus.queries.len());
    let mut rrf = LegRun::new("rrf fusion", corpus.queries.len());
    let mut reranked = LegRun::new("fusion + rerank", corpus.queries.len());

    for (i, (query_id, query_text)) in corpus.queries.iter().enumerate() {
        let qvec = &query_vectors[i];
        let text_query = FtsQuery::new(text_attr, query_text.as_str());

        let ids = db
            .text_search(COLLECTION, &text_query, &leg_opts)?
            .into_iter()
            .map(|h| h.id)
            .collect();
        fts.push(query_id, ids);

        let ids = db
            .search(COLLECTION, qvec, &leg_opts)?
            .into_iter()
            .map(|h| h.id)
            .collect();
        vector.push(query_id, ids);

        let ids = db
            .hybrid_search(COLLECTION, qvec, &text_query, fusion)?
            .into_iter()
            .map(|h| h.id)
            .collect();
        rrf.push(query_id, ids);

        let ids = rt
            .block_on(hybrid_reranked(
                &db,
                reranker,
                COLLECTION,
                qvec,
                &text_query,
                query_text.as_str(),
                fusion_reranked,
            ))?
            .into_iter()
            .map(|h| h.id)
            .collect();
        reranked.push(query_id, ids);
    }

    let query_ids: Vec<String> = corpus.queries.iter().map(|(id, _)| id.clone()).collect();
    let truth = qmetrics::binary_truth(&query_ids, &corpus.qrels, args.threshold);
    let legs = [&fts, &vector, &rrf, &reranked].map(|leg| {
        let (ndcg, recall) = leg.metrics(&corpus.qrels, args.top_k, &truth);
        (leg.label, ndcg, recall)
    });

    print_table(spec, corpus.docs.len(), corpus.queries.len(), args, &legs);

    Ok(json!({
        "dataset": spec.name,
        "doc_count": corpus.docs.len(),
        "query_count": corpus.queries.len(),
        "candidates_used": fusion.candidates,
        "legs": {
            "fts_only": {"ndcg_at_10": legs[0].1, "recall_at_100": legs[0].2},
            "vector_only": {"ndcg_at_10": legs[1].1, "recall_at_100": legs[1].2},
            "rrf_fusion": {"ndcg_at_10": legs[2].1, "recall_at_100": legs[2].2},
            "fusion_and_rerank": {"ndcg_at_10": legs[3].1, "recall_at_100": legs[3].2},
        },
        "bm25_reference_ndcg_at_10": spec.bm25_ndcg_at_10,
    }))
}

// ── reporting ────────────────────────────────────────────────────────────────

fn print_table(
    spec: &DatasetSpec,
    doc_count: usize,
    query_count: usize,
    args: &Args,
    legs: &[(&str, f64, f64); 4],
) {
    println!("\n{}  docs={doc_count}  queries={query_count}", spec.name);
    println!(
        "{:<20}{:>12}{:>14}",
        "leg",
        format!("ndcg@{}", args.top_k),
        format!("recall@{}", args.recall_k)
    );
    for (label, ndcg, recall) in legs {
        println!("{label:<20}{ndcg:>12.4}{recall:>14.4}");
    }
    println!(
        "{:<20}{:>12.4}{:>14}",
        "bm25 (BEIR)", spec.bm25_ndcg_at_10, "--"
    );
    println!(
        "note: `bm25 (BEIR)` is the published BEIR paper baseline, not a nidus run; the \
         comparison is directional (BEIR's BM25 is Elasticsearch-default analysis, nidus's \
         FTS leg is k1=1.2/b=0.75 over a Porter English analyzer)."
    );
}

fn write_json(
    path: &std::path::Path,
    args: &Args,
    fusion: &HybridOpts,
    nidus_version: &str,
    cells: &[Json],
) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let doc = json!({
        "bench": "nidus-bench-retrieval",
        "nidus_version": nidus_version,
        "inputs": {
            "datasets": args.datasets, "top_k": args.top_k, "recall_k": args.recall_k,
            "threshold": args.threshold, "cache": args.cache.display().to_string(),
            "embed_provider": EMBED_PROVIDER, "embed_model": EMBED_MODEL,
            "rerank_provider": RERANK_PROVIDER, "rerank_model": RERANK_MODEL,
            "rrf_k": fusion.rrf_k, "candidates": fusion.candidates,
            "vector_weight": fusion.vector_weight, "text_weight": fusion.text_weight,
        },
        "cells": cells,
    });
    std::fs::write(path, format!("{}\n", serde_json::to_string_pretty(&doc)?))
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!("\nwrote {}", path.display());
    Ok(())
}

// ── entry point ──────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode> {
    let args = parse_args()?;

    // Fail before any download or embed call, not partway through one.
    let api_key = std::env::var(VOYAGE_API_KEY_VAR).map_err(|_| {
        anyhow!(
            "{VOYAGE_API_KEY_VAR} is not set: the retrieval bench needs a live Voyage API key \
             and network access"
        )
    })?;

    let nidus_version = std::env::var("NIDUS_VERSION").unwrap_or_else(|_| "unknown".into());
    let text_attr = RerankOpts::default().text_attr;
    let fusion = fusion_opts(args.recall_k);
    // Same base as the RRF-only leg (`fusion_opts`'s untouched-default knobs plus the
    // `recall_k`-sized `top_k`/`candidates` both legs need) — `rerank` is the only field
    // that differs between the two hybrid legs.
    let fusion_reranked = HybridOpts {
        rerank: Some(RerankOpts::default()),
        ..fusion.clone()
    };

    println!("nidus-bench-retrieval — BEIR retrieval quality (nidus-yq9p.5)");
    println!(
        "nidus={nidus_version}  embed={EMBED_PROVIDER}/{EMBED_MODEL}  \
         rerank={RERANK_PROVIDER}/{RERANK_MODEL}"
    );
    println!(
        "rrf_k={}  candidates={}  vector_weight={}  text_weight={}",
        fusion.rrf_k, fusion.candidates, fusion.vector_weight, fusion.text_weight
    );
    println!(
        "top_k={}  recall_k={}  threshold={}  cache={}",
        args.top_k,
        args.recall_k,
        args.threshold,
        args.cache.display()
    );

    let rt = tokio::runtime::Runtime::new()?;
    let reranker = AnyReranker::build(
        RerankProvider::Voyage,
        RerankConfig::new(RERANK_MODEL).api_key(api_key),
    )?;

    let mut cells = Vec::with_capacity(args.datasets.len());
    for &name in &args.datasets {
        let spec = beir::DATASETS
            .iter()
            .find(|s| s.name == name)
            .expect("dataset name validated in parse_args");
        cells.push(run_dataset(
            spec,
            &args,
            &text_attr,
            &fusion,
            &fusion_reranked,
            &rt,
            &reranker,
        )?);
    }

    if let Some(path) = &args.json {
        write_json(path, &args, &fusion, &nidus_version, &cells)?;
    }

    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> impl Iterator<Item = String> {
        s.split_whitespace()
            .map(String::from)
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn parse_args_applies_defaults() {
        let a = parse_args_from(toks("")).unwrap();
        assert_eq!(a.top_k, 10);
        assert_eq!(a.recall_k, 100);
        assert_eq!(a.threshold, 0.0);
        assert_eq!(a.datasets, vec!["scifact", "nfcorpus", "fiqa"]);
        assert_eq!(a.cache, PathBuf::from("benchmarks/.cache"));
        assert!(a.json.is_none());
    }

    #[test]
    fn parse_args_overrides_every_knob() {
        let a = parse_args_from(toks(
            "dataset=scifact,fiqa top_k=5 recall_k=50 threshold=1 cache=/tmp/x json=out.json",
        ))
        .unwrap();
        assert_eq!(a.datasets, vec!["scifact", "fiqa"]);
        assert_eq!(a.top_k, 5);
        assert_eq!(a.recall_k, 50);
        assert_eq!(a.threshold, 1.0);
        assert_eq!(a.cache, PathBuf::from("/tmp/x"));
        assert_eq!(a.json, Some(PathBuf::from("out.json")));
    }

    #[test]
    fn parse_args_rejects_a_bad_token() {
        let err = parse_args_from(toks("not-key-value")).unwrap_err();
        assert!(err.to_string().contains("key=value"));
    }

    #[test]
    fn parse_args_rejects_an_unknown_dataset() {
        let err = parse_args_from(toks("dataset=not-a-real-dataset")).unwrap_err();
        assert!(err.to_string().contains("unknown dataset"));
    }

    #[test]
    fn fusion_opts_leaves_defaults_untouched_at_the_default_recall_k() {
        let opts = fusion_opts(100);
        let default = HybridOpts::default();
        assert_eq!(opts.rrf_k, default.rrf_k);
        assert_eq!(opts.vector_weight, default.vector_weight);
        assert_eq!(opts.text_weight, default.text_weight);
        assert_eq!(opts.candidates, default.candidates);
        assert_eq!(opts.top_k, 100);
    }

    #[test]
    fn fusion_opts_raises_candidates_only_when_recall_k_exceeds_it() {
        let opts = fusion_opts(200);
        assert_eq!(opts.candidates, 200);
        assert_eq!(opts.rrf_k, HybridOpts::default().rrf_k);
    }
}
