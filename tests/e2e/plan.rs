//! e2e: the starvation case, end to end (nidus-cvz). A selective filter over an ANN store
//! must fall back to an exact prefilter, and that fallback must be *visible* from outside
//! the process: `plan.path`, the unset-plan response shape, and the slow-query stderr line.
//!
//! Every assertion here names something only true because `QueryPlan` exists — a `path`
//! string, array-vs-object shape, a stderr line — never just "200 with some hits", which
//! would still pass if the whole feature were reverted.

use serde_json::{Value, json};

use crate::harness::Server;

/// Deterministic vectors without a PRNG dependency: SplitMix64, the same seeded approach
/// `tests/e2e/tune.rs` and `tests/e2e/scale.rs` use (copied, not shared — each e2e suite
/// stands alone).
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

const DIM: usize = 8;
/// Large enough that the ANN path actually engages and a 1-in-50 filter still leaves the
/// exact-prefilter fallback well clear of its cap; small enough to ingest in a debug build
/// in well under the harness's per-request timeout.
const N: usize = 1_000;
const TOP_K: usize = 5;
/// Number of buckets the corpus is partitioned into; querying for one bucket selects ~2%
/// of the corpus — a tiny fraction, not merely "less than everything".
const BUCKETS: i64 = 50;

/// Small HNSW build params (mirrors `scale.rs`'s `ann_filtered_search_recall_stays_above_the_floor`):
/// the default `ef_construction = 200` risks a slow debug-build ingest, and this suite is
/// about the fallback-reporting path, not graph quality.
const ANN_ARGS: [&str; 6] = [
    "--ann",
    "hnsw",
    "--ann-m",
    "8",
    "--ann-ef-construction",
    "16",
];

fn id(i: usize) -> String {
    format!("doc-{i:06}")
}

/// Ingest `N` deterministic vectors into `docs`, each tagged with `bucket` in `0..BUCKETS`,
/// batched so no single request body is large.
fn ingest(server: &crate::harness::RunningServer, vectors: &[Vec<f32>]) {
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
    for (b, chunk) in vectors.chunks(200).enumerate() {
        let records: Vec<Value> = chunk
            .iter()
            .enumerate()
            .map(|(j, v)| {
                let i = b * 200 + j;
                json!({
                    "id": id(i),
                    "vector": v,
                    "attrs": {"bucket": {"Int": (i as i64) % BUCKETS}}
                })
            })
            .collect();
        let (status, body) = server.post("/collections/docs/upsert", &json!({"records": records}));
        assert_eq!(status, 200, "batch {b} failed: {body}");
    }
}

/// **The fallback is reported.** A filter selecting ~2% of a 1000-row ANN-indexed corpus
/// starves the walk, so `search_ann` bails to the exact prefilter (`src/store/read.rs`) and
/// the plan must say so.
#[cfg_attr(miri, ignore)] // spawns a real `nidus serve` process
#[test]
fn selective_filter_over_ann_reports_the_prefilter_fallback() {
    let mut rng = Rng(0xC0FFEE);
    let vectors: Vec<Vec<f32>> = (0..N).map(|_| rng.vector(DIM)).collect();
    let query = rng.vector(DIM);

    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM).args(ANN_ARGS).start();
    ingest(&server, &vectors);

    let (status, out) = server.post(
        "/search",
        &json!({
            "query": query,
            "top_k": TOP_K,
            "filter": [{"Eq": ["bucket", {"Int": 7}]}],
            "plan": true
        }),
    );
    assert_eq!(status, 200, "{out}");
    assert_eq!(
        out["plan"]["path"], "ann_prefilter_fallback",
        "a ~2% filter over an ANN store should have starved the walk: {out}"
    );
}

/// **The unfiltered control.** The identical store and query with no filter must report
/// `"ann"`, not the fallback — proving the *filter* caused it, not the store or query.
/// Without this, test 1 would pass even if every ANN query reported the fallback.
#[cfg_attr(miri, ignore)] // spawns a real `nidus serve` process
#[test]
fn unfiltered_ann_search_does_not_report_the_fallback() {
    let mut rng = Rng(0xC0FFEE);
    let vectors: Vec<Vec<f32>> = (0..N).map(|_| rng.vector(DIM)).collect();
    let query = rng.vector(DIM);

    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), DIM).args(ANN_ARGS).start();
    ingest(&server, &vectors);

    let (status, out) = server.post(
        "/search",
        &json!({"query": query, "top_k": TOP_K, "plan": true}),
    );
    assert_eq!(status, 200, "{out}");
    assert_eq!(
        out["plan"]["path"], "ann",
        "an unfiltered query over the whole store should walk the graph, not fall back: {out}"
    );
}

/// **Byte-identical when unset.** The same request without `plan` returns a bare JSON array;
/// with `plan: true` it returns `{hits, plan}`. Assert the array-ness directly, not just 200.
#[cfg_attr(miri, ignore)] // spawns a real `nidus serve` process
#[test]
fn plan_flag_switches_response_shape() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), 3).start();
    assert_eq!(server.post("/collections/docs", &json!({})).0, 200);
    assert_eq!(
        server
            .post(
                "/collections/docs/upsert",
                &json!({"records": [{"id": "a", "vector": [1, 0, 0], "attrs": {}}]}),
            )
            .0,
        200
    );

    let (status, without_plan) = server.post("/search", &json!({"query": [1, 0, 0], "top_k": 5}));
    assert_eq!(status, 200);
    assert!(
        without_plan.is_array(),
        "no `plan` field must stay byte-identical to today's bare array: {without_plan}"
    );

    let (status, with_plan) = server.post(
        "/search",
        &json!({"query": [1, 0, 0], "top_k": 5, "plan": true}),
    );
    assert_eq!(status, 200);
    assert!(
        with_plan.is_object(),
        "`plan: true` must switch to the {{hits, plan}} object shape: {with_plan}"
    );
    assert!(with_plan["hits"].is_array(), "{with_plan}");
    assert!(with_plan["plan"]["path"].is_string(), "{with_plan}");
}

/// A corpus large enough that an exact brute-force scan measurably clears a 1ms threshold
/// even in a debug build — a 3-vector store risks finishing in microseconds and never
/// tripping `NIDUS_SLOW_QUERY_MS=1` at all.
const SLOW_N: usize = 5_000;
const SLOW_DIM: usize = 256;

/// **The slow-query line.** `NIDUS_SLOW_QUERY_MS=1` makes a real scan "slow", so a search
/// must produce a stderr line carrying `msg=` and a `path=` field — asserted on content,
/// not merely that the server survived the request.
#[cfg_attr(miri, ignore)] // spawns a real `nidus serve` process
#[test]
fn slow_query_threshold_logs_a_stderr_line_with_a_path() {
    let mut rng = Rng(0xDECAF);
    let vectors: Vec<Vec<f32>> = (0..SLOW_N).map(|_| rng.vector(SLOW_DIM)).collect();
    let query = rng.vector(SLOW_DIM);

    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path(), SLOW_DIM)
        .env("NIDUS_SLOW_QUERY_MS", "1")
        .start();
    ingest(&server, &vectors);

    let (status, hits) = server.post("/search", &json!({"query": query, "top_k": 5}));
    assert_eq!(status, 200, "{hits}");

    let stderr = server.stderr();
    let slow_line = stderr
        .lines()
        .find(|l| l.contains("msg=") && l.contains("slow query"))
        .unwrap_or_else(|| panic!("no slow-query line in stderr:\n{stderr}"));
    assert!(
        slow_line.contains("path="),
        "slow-query line missing path= field: {slow_line}"
    );
}
