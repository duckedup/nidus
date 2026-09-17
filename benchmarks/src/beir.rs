//! BEIR dataset acquisition and parsing (nidus-yq9p.5).
//!
//! Each dataset ships as a zip at a stable public URL containing `corpus.jsonl`
//! (`{"_id","title","text"}` per line), `queries.jsonl` (every query, not just the test
//! split) and `qrels/test.tsv` (tab-separated, header row, graded relevance). The test
//! split is exactly the query ids that appear in the qrels file. Doc and query ids are
//! opaque strings and are never parsed as integers: the qrels join is by exact string
//! match, and an integer round-trip would silently drop leading zeros or non-numeric ids.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::qmetrics::Qrels;

/// One BEIR dataset, loaded and ready to index.
pub struct Corpus {
    pub name: &'static str,
    /// (doc id, title, body) in file order.
    pub docs: Vec<(String, String, String)>,
    /// (query id, query text), test split only.
    pub queries: Vec<(String, String)>,
    pub qrels: Qrels,
}

/// The three datasets this lane publishes, with their expected shape.
pub struct DatasetSpec {
    pub name: &'static str,
    pub url: &'static str,
    /// Asserted after parse, so a truncated or silently rehosted corpus fails loudly
    /// instead of quietly moving the published numbers.
    pub expect_docs: usize,
    pub expect_queries: usize,
    /// The BEIR paper's published BM25 nDCG@10 for this dataset, the FTS leg's reference line.
    pub bm25_ndcg_at_10: f64,
}

/// The three datasets this lane publishes. Counts are the full corpus and the test-split
/// query count, verified against a real download; `bm25_ndcg_at_10` is Table 2 of Thakur
/// et al., "BEIR: A Heterogeneous Benchmark..." (NeurIPS 2021 Datasets and Benchmarks).
pub const DATASETS: &[DatasetSpec] = &[
    DatasetSpec {
        name: "scifact",
        url: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        expect_docs: 5_183,
        expect_queries: 300,
        bm25_ndcg_at_10: 0.665,
    },
    DatasetSpec {
        name: "nfcorpus",
        url: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        expect_docs: 3_633,
        expect_queries: 323,
        bm25_ndcg_at_10: 0.325,
    },
    DatasetSpec {
        name: "fiqa",
        url: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        expect_docs: 57_638,
        expect_queries: 648,
        bm25_ndcg_at_10: 0.236,
    },
];

/// Load a dataset, downloading and extracting into `cache_dir` only on a cache miss. A
/// cache hit (the three expected files already on disk) parses straight from disk and
/// makes no network call at all.
pub fn load(spec: &DatasetSpec, cache_dir: &Path) -> Result<Corpus> {
    let dataset_dir = cache_dir.join(spec.name);
    if !has_expected_files(&dataset_dir) {
        fetch(spec, cache_dir, &dataset_dir)?;
    }
    parse_dataset(spec, &dataset_dir)
}

fn has_expected_files(dir: &Path) -> bool {
    dir.join("corpus.jsonl").is_file()
        && dir.join("queries.jsonl").is_file()
        && dir.join("qrels").join("test.tsv").is_file()
}

/// Download `spec.url`, extract into a scratch directory beside `dataset_dir`, then
/// rename the extracted tree into place. Extract-then-rename matters: a half-extracted
/// directory from an interrupted run must never read as a cache hit on the next one.
fn fetch(spec: &DatasetSpec, cache_dir: &Path, dataset_dir: &Path) -> Result<()> {
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating cache dir {}", cache_dir.display()))?;

    let zip_path = cache_dir.join(format!("{}.zip.part", spec.name));
    download(spec.url, &zip_path)?;

    let scratch = cache_dir.join(format!(".{}.extracting", spec.name));
    if scratch.exists() {
        fs::remove_dir_all(&scratch)
            .with_context(|| format!("clearing stale extraction dir {}", scratch.display()))?;
    }
    extract_zip(&zip_path, &scratch)?;
    let _ = fs::remove_file(&zip_path);

    let extracted_root = find_dataset_root(&scratch, spec.name)?;
    fs::rename(&extracted_root, dataset_dir).with_context(|| {
        format!(
            "moving extracted {} into place at {}",
            extracted_root.display(),
            dataset_dir.display()
        )
    })?;
    let _ = fs::remove_dir_all(&scratch);
    Ok(())
}

/// `GET url`, streaming the response straight to `dest`. BEIR's host is a plain static
/// file server: no auth, and a non-2xx status comes back as an error by default.
fn download(url: &str, dest: &Path) -> Result<()> {
    let res = ureq::get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let mut reader = res.into_body().into_reader();
    let mut file = File::create(dest).with_context(|| format!("creating {}", dest.display()))?;
    io::copy(&mut reader, &mut file)
        .with_context(|| format!("writing {} from {url}", dest.display()))?;
    Ok(())
}

/// Extract every entry of `zip_path` under `dest`. `enclosed_name()` rejects zip-slip
/// entries (absolute paths, `..` components) rather than trusting the archive.
fn extract_zip(zip_path: &Path, dest: &Path) -> Result<()> {
    let file = File::open(zip_path)
        .with_context(|| format!("opening downloaded zip {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading zip archive {}", zip_path.display()))?;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .with_context(|| format!("reading entry {i} of {}", zip_path.display()))?;
        let Some(rel_path) = entry.enclosed_name() else {
            continue;
        };
        let out_path = dest.join(rel_path);
        if entry.is_dir() {
            fs::create_dir_all(&out_path)
                .with_context(|| format!("creating {}", out_path.display()))?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let mut out_file =
            File::create(&out_path).with_context(|| format!("creating {}", out_path.display()))?;
        io::copy(&mut entry, &mut out_file)
            .with_context(|| format!("extracting {}", out_path.display()))?;
    }
    Ok(())
}

/// BEIR zips extract to one top-level `<name>/` directory; fall back to the scratch root
/// itself in case a host ships the three files flat, with no wrapper folder.
fn find_dataset_root(scratch: &Path, name: &str) -> Result<PathBuf> {
    let nested = scratch.join(name);
    if nested.join("corpus.jsonl").is_file() {
        return Ok(nested);
    }
    if scratch.join("corpus.jsonl").is_file() {
        return Ok(scratch.to_path_buf());
    }
    bail!(
        "extracted {name} but found no corpus.jsonl under {} or {}",
        scratch.display(),
        nested.display()
    );
}

#[derive(Deserialize)]
struct CorpusLine {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
}

#[derive(Deserialize)]
struct QueryLine {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

fn parse_dataset(spec: &DatasetSpec, dir: &Path) -> Result<Corpus> {
    let corpus_path = dir.join("corpus.jsonl");
    let corpus_text = fs::read_to_string(&corpus_path)
        .with_context(|| format!("reading {}", corpus_path.display()))?;
    let docs = parse_corpus(&corpus_text, &corpus_path)?;

    let queries_path = dir.join("queries.jsonl");
    let queries_text = fs::read_to_string(&queries_path)
        .with_context(|| format!("reading {}", queries_path.display()))?;
    let all_queries = parse_queries(&queries_text, &queries_path)?;

    let qrels_path = dir.join("qrels").join("test.tsv");
    let qrels_text = fs::read_to_string(&qrels_path)
        .with_context(|| format!("reading {}", qrels_path.display()))?;
    let qrels = parse_qrels(&qrels_text, &qrels_path)?;

    let queries = restrict_queries_to_qrels(all_queries, &qrels, &qrels_path)?;

    if docs.len() != spec.expect_docs || queries.len() != spec.expect_queries {
        bail!(
            "{}: expected {} docs / {} test queries, got {} docs / {} queries; delete {} and retry",
            spec.name,
            spec.expect_docs,
            spec.expect_queries,
            docs.len(),
            queries.len(),
            dir.display()
        );
    }

    let doc_ids: HashSet<&str> = docs.iter().map(|(id, _, _)| id.as_str()).collect();
    let missing_corpus_ids = qrels
        .values()
        .flat_map(|by_doc| by_doc.keys())
        .filter(|id| !doc_ids.contains(id.as_str()))
        .count();
    if missing_corpus_ids > 0 {
        eprintln!(
            "warning: {missing_corpus_ids} qrels rows in {} reference corpus ids absent \
             from {}; the join is broken and metrics over them are meaningless",
            qrels_path.display(),
            corpus_path.display()
        );
    }

    Ok(Corpus {
        name: spec.name,
        docs,
        queries,
        qrels,
    })
}

/// Parse `corpus.jsonl`, keeping file order (deterministic indexing order across runs).
fn parse_corpus(text: &str, path: &Path) -> Result<Vec<(String, String, String)>> {
    let mut docs = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let rec: CorpusLine = serde_json::from_str(line)
            .with_context(|| format!("parsing {} line {}", path.display(), i + 1))?;
        docs.push((rec.id, rec.title, rec.text));
    }
    Ok(docs)
}

/// Parse `queries.jsonl`. This is every query in the dataset; the test split is applied
/// afterwards by [`restrict_queries_to_qrels`].
fn parse_queries(text: &str, path: &Path) -> Result<Vec<(String, String)>> {
    let mut queries = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let rec: QueryLine = serde_json::from_str(line)
            .with_context(|| format!("parsing {} line {}", path.display(), i + 1))?;
        queries.push((rec.id, rec.text));
    }
    Ok(queries)
}

/// Parse `qrels/test.tsv`: tab-separated `query-id corpus-id score`, header row skipped
/// unconditionally (it is not data under any parse). Scores stay graded, never binarized.
fn parse_qrels(text: &str, path: &Path) -> Result<Qrels> {
    let mut qrels: Qrels = Qrels::new();
    for (i, line) in text.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = line.split('\t');
        let query_id = fields
            .next()
            .with_context(|| format!("{} line {}: missing query-id", path.display(), i + 1))?;
        let corpus_id = fields
            .next()
            .with_context(|| format!("{} line {}: missing corpus-id", path.display(), i + 1))?;
        let score: f32 = fields
            .next()
            .with_context(|| format!("{} line {}: missing score", path.display(), i + 1))?
            .trim()
            .parse()
            .with_context(|| format!("{} line {}: score is not a number", path.display(), i + 1))?;
        qrels
            .entry(query_id.to_string())
            .or_default()
            .insert(corpus_id.to_string(), score);
    }
    Ok(qrels)
}

/// Drop every query whose id has no qrels entry, and fail loudly if a qrels row names a
/// query id that `queries.jsonl` never defined: that is a broken join, not an empty split.
fn restrict_queries_to_qrels(
    all_queries: Vec<(String, String)>,
    qrels: &Qrels,
    qrels_path: &Path,
) -> Result<Vec<(String, String)>> {
    let known_ids: HashSet<&str> = all_queries.iter().map(|(id, _)| id.as_str()).collect();
    for query_id in qrels.keys() {
        if !known_ids.contains(query_id.as_str()) {
            bail!(
                "{}: qrels references query id {query_id:?} absent from queries.jsonl",
                qrels_path.display()
            );
        }
    }
    Ok(all_queries
        .into_iter()
        .filter(|(id, _)| qrels.contains_key(id))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PATH: &str = "test-fixture";

    #[test]
    fn corpus_jsonl_parses_title_and_text() {
        let fixture = concat!(
            "{\"_id\": \"1\", \"title\": \"A Title\", \"text\": \"body one\"}\n",
            "{\"_id\": \"2\", \"title\": \"\", \"text\": \"body two\"}\n",
        );
        let docs = parse_corpus(fixture, Path::new(TEST_PATH)).unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(
            docs[0],
            (
                "1".to_string(),
                "A Title".to_string(),
                "body one".to_string()
            )
        );
        assert_eq!(
            docs[1],
            ("2".to_string(), String::new(), "body two".to_string())
        );
    }

    #[test]
    fn qrels_tsv_skips_the_header_row() {
        let fixture = concat!("query-id\tcorpus-id\tscore\n", "q1\td1\t1\n",);
        let qrels = parse_qrels(fixture, Path::new(TEST_PATH)).unwrap();
        assert!(!qrels.contains_key("query-id"));
        assert_eq!(qrels.len(), 1);
        assert_eq!(qrels["q1"]["d1"], 1.0);
    }

    #[test]
    fn ids_stay_strings() {
        let corpus = concat!(
            "{\"_id\": \"007\", \"title\": \"\", \"text\": \"leading zero\"}\n",
            "{\"_id\": \"doc-abc\", \"title\": \"\", \"text\": \"non numeric\"}\n",
        );
        let docs = parse_corpus(corpus, Path::new(TEST_PATH)).unwrap();
        assert_eq!(docs[0].0, "007");
        assert_eq!(docs[1].0, "doc-abc");

        let qrels_fixture = concat!(
            "query-id\tcorpus-id\tscore\n",
            "q1\t007\t1\n",
            "q1\tdoc-abc\t2\n",
        );
        let qrels = parse_qrels(qrels_fixture, Path::new(TEST_PATH)).unwrap();
        assert_eq!(qrels["q1"]["007"], 1.0);
        assert_eq!(qrels["q1"]["doc-abc"], 2.0);
    }

    #[test]
    fn queries_are_restricted_to_the_qrels_split() {
        let queries = vec![
            ("q1".to_string(), "in split".to_string()),
            ("q2".to_string(), "not in split".to_string()),
        ];
        let mut qrels: Qrels = Qrels::new();
        qrels
            .entry("q1".to_string())
            .or_default()
            .insert("d1".to_string(), 1.0);

        let restricted = restrict_queries_to_qrels(queries, &qrels, Path::new(TEST_PATH)).unwrap();
        assert_eq!(restricted.len(), 1);
        assert_eq!(restricted[0].0, "q1");
    }

    #[test]
    fn qrels_referencing_an_unknown_query_id_is_an_error() {
        let queries = vec![("q1".to_string(), "known".to_string())];
        let mut qrels: Qrels = Qrels::new();
        qrels
            .entry("ghost".to_string())
            .or_default()
            .insert("d1".to_string(), 1.0);

        let err = restrict_queries_to_qrels(queries, &qrels, Path::new(TEST_PATH)).unwrap_err();
        assert!(err.to_string().contains("ghost"));
    }

    #[test]
    fn graded_scores_survive_parsing() {
        let fixture = concat!(
            "query-id\tcorpus-id\tscore\n",
            "q1\td0\t0\n",
            "q1\td1\t1\n",
            "q1\td2\t2\n",
        );
        let qrels = parse_qrels(fixture, Path::new(TEST_PATH)).unwrap();
        assert_eq!(qrels["q1"]["d0"], 0.0);
        assert_eq!(qrels["q1"]["d1"], 1.0);
        assert_eq!(qrels["q1"]["d2"], 2.0);
        assert_ne!(qrels["q1"]["d2"], 1.0);
    }
}
