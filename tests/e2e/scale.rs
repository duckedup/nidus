//! Scale and correctness end-to-end: a realistic corpus pushed through the HTTP API,
//! with results checked against ground truth computed here in the test.

use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::harness::{RunningServer, Server};

/// Corpus size and dimension: large enough that ranking is non-trivial and payloads are
/// realistic, small enough to ingest and verify in seconds on every PR.
const N: usize = 10_000;
const DIM: usize = 384;
/// Records per upsert request — keeps any single body well clear of the default body cap
/// while still exercising repeated batched writes.
const BATCH: usize = 500;
const TOP_K: usize = 10;

/// Deterministic vectors without a PRNG dependency: SplitMix64, the same
/// seeded-and-dependency-free approach `src/ann/` takes. Values are centred on zero so
/// the vectors point in genuinely different directions.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float in `[-1, 1)`.
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 8_388_608.0 - 1.0
    }

    fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_unit()).collect()
    }
}

/// The corpus, plus the query set — generated once and shared by the tests below.
struct Corpus {
    vectors: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
}

impl Corpus {
    fn generate(seed: u64, n: usize, dim: usize, queries: usize) -> Corpus {
        let mut rng = Rng(seed);
        Corpus {
            vectors: (0..n).map(|_| rng.vector(dim)).collect(),
            queries: (0..queries).map(|_| rng.vector(dim)).collect(),
        }
    }

    fn id(i: usize) -> String {
        format!("doc-{i:06}")
    }

    /// Every record as JSON, with an attr that partitions the corpus into ten buckets so
    /// filtered search has something meaningful to select on.
    fn batches(&self) -> impl Iterator<Item = Value> + '_ {
        self.vectors.chunks(BATCH).enumerate().map(|(b, chunk)| {
            let records: Vec<Value> = chunk
                .iter()
                .enumerate()
                .map(|(j, v)| {
                    let i = b * BATCH + j;
                    json!({
                        "id": Corpus::id(i),
                        "vector": v,
                        "attrs": {"bucket": {"Int": (i % 10) as i64}}
                    })
                })
                .collect();
            json!({"records": records})
        })
    }

    /// Exact top-k by cosine, computed here so the server's answer is checked against an
    /// independent implementation rather than against itself.
    fn ground_truth(
        &self,
        query: &[f32],
        k: usize,
        keep: impl Fn(usize) -> bool,
    ) -> Vec<(String, f32)> {
        let q = normalise(query);
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| keep(*i))
            .map(|(i, v)| (i, dot(&normalise(v), &q)))
            .collect();
        // Descending by score; ties broken by index so the order is deterministic.
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored
            .into_iter()
            .take(k)
            .map(|(i, s)| (Corpus::id(i), s))
            .collect()
    }
}

fn normalise(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Ingest the whole corpus, returning how long it took.
fn ingest(server: &RunningServer, corpus: &Corpus) -> Duration {
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
    let started = Instant::now();
    for (i, batch) in corpus.batches().enumerate() {
        let (status, body) = server.post("/collections/docs/upsert", &batch);
        assert_eq!(status, 200, "batch {i} failed: {body}");
        assert_eq!(body["upserted"], BATCH, "batch {i} short-wrote: {body}");
    }
    started.elapsed()
}

/// Ids from a search response, in rank order.
fn hit_ids(hits: &Value) -> Vec<String> {
    hits.as_array()
        .expect("hits array")
        .iter()
        .filter_map(|h| h["id"].as_str().map(str::to_string))
        .collect()
}

/// Fraction of the true top-k the server actually returned — the standard recall@k, used
/// for the approximate paths where an exact match is not the contract.
fn recall_at_k(returned: &[String], truth: &[(String, f32)]) -> f64 {
    let truth_ids: std::collections::HashSet<&str> =
        truth.iter().map(|(id, _)| id.as_str()).collect();
    let hits = returned
        .iter()
        .filter(|id| truth_ids.contains(id.as_str()))
        .count();
    hits as f64 / truth.len() as f64
}

/// **Exact search over a realistic corpus, through HTTP, must be exactly right.**
#[test]
fn exact_search_at_scale_matches_ground_truth() {
    let corpus = Corpus::generate(0xC0FFEE, N, DIM, 5);
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM).start();

    let elapsed = ingest(&server, &corpus);
    let rate = N as f64 / elapsed.as_secs_f64();
    println!("ingest: {N} x {DIM}-d in {elapsed:.2?} ({rate:.0} vec/s, debug build)");

    let (_, stats) = server.get("/stats");
    assert_eq!(stats["footprint"]["doc_count"], N as u64);

    let mut latencies = Vec::new();
    for (qi, query) in corpus.queries.iter().enumerate() {
        let started = Instant::now();
        let (status, hits) = server.post(
            "/search",
            &json!({"query": query, "top_k": TOP_K, "collections": ["docs"]}),
        );
        latencies.push(started.elapsed());
        assert_eq!(status, 200, "query {qi} failed: {hits}");

        let truth = corpus.ground_truth(query, TOP_K, |_| true);
        let expected: Vec<String> = truth.iter().map(|(id, _)| id.clone()).collect();
        assert_eq!(
            hit_ids(&hits),
            expected,
            "query {qi}: exact search must reproduce the ground-truth ranking"
        );

        // Scores must agree too, not just the ordering — a normalisation bug could
        // preserve rank order while returning wrong similarities.
        for (rank, (id, want)) in truth.iter().enumerate() {
            let got = hits[rank]["score"].as_f64().expect("score") as f32;
            assert!(
                (got - want).abs() < 1e-4,
                "query {qi} rank {rank} ({id}): score {got} != expected {want}"
            );
        }
    }

    let worst = latencies.iter().max().expect("queries ran");
    println!(
        "query latency: worst {worst:.2?} over {} queries",
        latencies.len()
    );

    // Order-of-magnitude ceilings only — see the module docs. A debug-build exhaustive
    // scan of 10k x 384 is milliseconds; seconds means something is structurally wrong.
    assert!(
        *worst < Duration::from_secs(5),
        "worst-case query took {worst:.2?}; exhaustive search over {N} vectors should not \
         be anywhere near this even unoptimised"
    );
    assert!(
        elapsed < Duration::from_secs(120),
        "ingesting {N} vectors took {elapsed:.2?}, far beyond a batched-write path"
    );
}

/// A filter applied over a realistic corpus selects exactly the right subset and ranks it
/// correctly. At three vectors a filter bug is invisible; over 10k with a tenth of rows
/// admitted, an off-by-one or a filter applied after top-k shows up immediately.
#[test]
fn filtered_search_at_scale_matches_ground_truth() {
    let corpus = Corpus::generate(0xF117E4, N, DIM, 3);
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM).start();
    ingest(&server, &corpus);

    for (qi, query) in corpus.queries.iter().enumerate() {
        let (status, hits) = server.post(
            "/search",
            &json!({
                "query": query,
                "top_k": TOP_K,
                "filter": [{"Eq": ["bucket", {"Int": 7}]}]
            }),
        );
        assert_eq!(status, 200, "query {qi} failed: {hits}");

        let ids = hit_ids(&hits);
        let truth = corpus.ground_truth(query, TOP_K, |i| i % 10 == 7);
        let expected: Vec<String> = truth.iter().map(|(id, _)| id.clone()).collect();
        assert_eq!(ids, expected, "query {qi}: filtered ranking mismatch");
        // Every hit must genuinely satisfy the predicate — a filter that leaked would
        // still rank plausibly, so assert membership directly.
        for id in &ids {
            let i: usize = id
                .strip_prefix("doc-")
                .and_then(|n| n.parse().ok())
                .expect("parse id");
            assert_eq!(i % 10, 7, "{id} does not satisfy bucket == 7");
        }
    }
}

/// **The ANN recall canary under a selective filter** (#114, SPEC §9's named limitation):
/// HNSW over the same corpus, filter, and ground truth as the exact filtered test above,
/// held to a recall floor — the acceptance test for the exact-prefilter follow-up.
#[test]
fn ann_filtered_search_recall_stays_above_the_floor() {
    let corpus = Corpus::generate(0xF117E4, N, DIM, 3);
    let dir = tempfile::tempdir().unwrap();
    // Small build params on purpose: the default `ef_construction = 200` makes a debug-build
    // ingest blow the harness's per-request timeout, and the canary is about the filtered
    // search path, not graph quality — the floor below was observed with exactly these.
    let server = Server::new(dir.path(), DIM)
        .args([
            "--ann",
            "hnsw",
            "--ann-m",
            "8",
            "--ann-ef-construction",
            "16",
        ])
        .start();

    // Hand-rolled ingest at a fifth of `BATCH`: graph inserts are slow enough in a debug
    // build that a 500-record batch would flirt with the harness's 30s request timeout.
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
    for (b, chunk) in corpus.vectors.chunks(100).enumerate() {
        let records: Vec<Value> = chunk
            .iter()
            .enumerate()
            .map(|(j, v)| {
                let i = b * 100 + j;
                json!({
                    "id": Corpus::id(i),
                    "vector": v,
                    "attrs": {"bucket": {"Int": (i % 10) as i64}}
                })
            })
            .collect();
        let (status, body) = server.post("/collections/docs/upsert", &json!({"records": records}));
        assert_eq!(status, 200, "batch {b} failed: {body}");
    }

    let mut recalls = Vec::new();
    for (qi, query) in corpus.queries.iter().enumerate() {
        let (status, hits) = server.post(
            "/search",
            &json!({
                "query": query,
                "top_k": TOP_K,
                "filter": [{"Eq": ["bucket", {"Int": 7}]}]
            }),
        );
        assert_eq!(status, 200, "query {qi} failed: {hits}");

        let ids = hit_ids(&hits);
        // Approximate or not, a hit that fails the filter is a correctness bug, not a
        // recall trade — assert membership before measuring recall.
        for id in &ids {
            let i: usize = id
                .strip_prefix("doc-")
                .and_then(|n| n.parse().ok())
                .expect("parse id");
            assert_eq!(i % 10, 7, "{id} does not satisfy bucket == 7");
        }
        let truth = corpus.ground_truth(query, TOP_K, |i| i % 10 == 7);
        recalls.push(recall_at_k(&ids, &truth));
    }

    let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
    println!(
        "hnsw filtered recall@{TOP_K}: mean {mean:.3} over {} queries",
        recalls.len()
    );
    // A collapse detector, not a benchmark: observed 1.000 (the exact-prefilter fallback,
    // nidus-0ou, kicks in at this selectivity), so 0.6 is comfortably clear of noise while
    // still catching the starvation SPEC §9 warns about if the fallback ever regresses.
    assert!(
        mean >= 0.6,
        "hnsw filtered recall@{TOP_K} collapsed to {mean:.3} — the ANN candidate walk is \
         being starved by the filter and nothing is compensating"
    );
}

/// The quantized first pass is a speed/recall trade, so the contract is recall, not an exact match.
/// This pins that the trade is made well end-to-end: int8 codes select, an exact f32 rerank orders,
/// and what comes back over HTTP still contains nearly all of the true top-k.
#[test]
fn quantized_search_at_scale_keeps_high_recall() {
    let corpus = Corpus::generate(0x00A17_u64, N, DIM, 3);
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM)
        .args(["--quantization", "int8"])
        .start();
    ingest(&server, &corpus);

    let mut recalls = Vec::new();
    for (qi, query) in corpus.queries.iter().enumerate() {
        let (status, hits) = server.post("/search", &json!({"query": query, "top_k": TOP_K}));
        assert_eq!(status, 200, "query {qi} failed: {hits}");
        let truth = corpus.ground_truth(query, TOP_K, |_| true);
        recalls.push(recall_at_k(&hit_ids(&hits), &truth));
    }

    let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
    println!(
        "int8 recall@{TOP_K}: mean {mean:.3} over {} queries",
        recalls.len()
    );
    assert!(
        mean >= 0.8,
        "int8 recall@{TOP_K} collapsed to {mean:.3} — the quantized first pass is not \
         selecting sensible candidates"
    );
}
