//! The async half of the rerank stage: free functions over `&Nidus` + `&impl Reranker`,
//! mirroring `memory::recall_with` — nidus stays synchronous, gaining no async method. The
//! sync search runs first (widened by `store::rerank::rerank_depth`); only the resulting
//! candidate texts, if any, cross the network.

use anyhow::bail;

use super::Reranker;
use crate::model::RerankOpts;
use crate::store::rerank::{apply_scores, candidate_texts, rerank_depth};
use crate::{FtsQuery, Hit, HybridOpts, Nidus, Result, Scope, SearchOpts};

/// Rerank an already-ranked `hits` set: extract candidate texts, score them with `reranker`,
/// substitute scores, and re-sort — passthroughs keep their metric score and relative order
/// (SPEC §7, decision 1). Skips the network call when nothing in `hits` has usable text.
pub async fn rerank_hits<R: Reranker>(
    reranker: &R,
    query: &str,
    hits: Vec<Hit>,
    opts: &RerankOpts,
) -> Result<Vec<Hit>> {
    let (texts, passthrough) = candidate_texts(&hits, &opts.text_attr);
    if texts.is_empty() {
        return Ok(hits);
    }
    let candidate_indices: Vec<usize> = (0..hits.len())
        .filter(|i| !passthrough.contains(i))
        .collect();
    let scores = reranker.rerank(query, &texts).await?;
    if scores.len() != candidate_indices.len() {
        bail!(
            "reranker returned {} scores for {} documents",
            scores.len(),
            candidate_indices.len()
        );
    }
    let scored: Vec<(usize, f32)> = candidate_indices.into_iter().zip(scores).collect();
    Ok(apply_scores(hits, &scored, passthrough))
}

/// Vector search widened to [`rerank_depth`], reranked, then tailed with the caller's real
/// `opts` (`Store::finish`, applying `limit_per`/pagination/`top_k` once, post-rerank). A
/// pass-through to `db.search` when `opts.rerank` is `None` — safe to call unconditionally.
pub async fn search_reranked<'a, R: Reranker>(
    db: &Nidus,
    reranker: &R,
    scope: impl Into<Scope<'a>>,
    query_vector: &[f32],
    query_text: &str,
    opts: &SearchOpts,
) -> Result<Vec<Hit>> {
    let Some(rerank_opts) = opts.rerank.clone() else {
        return db.search(scope, query_vector, opts);
    };
    // `limit_per` is dropped here and re-applied once, post-rerank, via `Store::finish` below
    // (decision 3): capping the pre-rerank window would freeze the group winners at their
    // metric order, defeating the point of reranking them.
    let widened = SearchOpts {
        top_k: rerank_depth(opts),
        offset: 0,
        limit_per: None,
        ..opts.clone()
    };
    let hits = db.search(scope, query_vector, &widened)?;
    let reranked = rerank_hits(reranker, query_text, hits, &rerank_opts).await?;
    Ok(db.store().finish(reranked, opts))
}

/// Hybrid (vector + BM25) search, reranked the same way as [`search_reranked`]. `HybridOpts`
/// has no `limit_per`, so the tail is just the page cut (`Store::finish_hybrid`).
pub async fn hybrid_reranked<'a, R: Reranker>(
    db: &Nidus,
    reranker: &R,
    scope: impl Into<Scope<'a>>,
    vector: &[f32],
    text: &FtsQuery,
    query_text: &str,
    opts: &HybridOpts,
) -> Result<Vec<Hit>> {
    let Some(rerank_opts) = opts.rerank.clone() else {
        return db.hybrid_search(scope, vector, text, opts);
    };
    let overscan = rerank_opts.overscan.max(1);
    let widened = HybridOpts {
        top_k: opts
            .offset
            .saturating_add(opts.top_k)
            .saturating_mul(overscan),
        offset: 0,
        candidates: opts.candidates.saturating_mul(overscan),
        ..opts.clone()
    };
    let hits = db.hybrid_search(scope, vector, text, &widened)?;
    let reranked = rerank_hits(reranker, query_text, hits, &rerank_opts).await?;
    Ok(db.store().finish_hybrid(reranked, opts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtsField;
    use crate::model::{LimitPer, META_TEXT, Record, Value};
    use crate::rerank::RerankError;
    use std::collections::BTreeMap;

    /// Scores a document by its text length, so a test picks the winner by writing a longer
    /// string. Deterministic and network-free, mirroring `embed`'s `Fake`.
    struct LenReranker;

    impl Reranker for LenReranker {
        async fn rerank(&self, _query: &str, documents: &[&str]) -> Result<Vec<f32>, RerankError> {
            Ok(documents.iter().map(|d| d.len() as f32).collect())
        }
        fn provider_name(&self) -> &str {
            "fake"
        }
        fn model_name(&self) -> &str {
            "len"
        }
        fn max_documents(&self) -> usize {
            1000
        }
    }

    /// Answers a fixed score list in input order, for the alignment checks.
    struct FixedReranker(Vec<f32>);

    impl Reranker for FixedReranker {
        async fn rerank(&self, _query: &str, documents: &[&str]) -> Result<Vec<f32>, RerankError> {
            Ok(self.0.iter().copied().take(documents.len()).collect())
        }
        fn provider_name(&self) -> &str {
            "fake"
        }
        fn model_name(&self) -> &str {
            "fixed"
        }
        fn max_documents(&self) -> usize {
            1000
        }
    }

    fn attrs(text: &str, group: &str) -> BTreeMap<String, Value> {
        BTreeMap::from([
            (META_TEXT.to_string(), Value::Str(text.to_string())),
            ("group".to_string(), Value::Str(group.to_string())),
            ("body".to_string(), Value::Str(text.to_string())),
        ])
    }

    /// Four docs whose cosine order against `[1.0, 0.0]` is a, b, c, d — and whose text
    /// lengths run the other way, so `LenReranker` must invert the ranking.
    fn store() -> Nidus {
        let mut db = Nidus::open_in_memory(2).unwrap();
        db.set_fts_schema("docs", &[FtsField::new("body")]).unwrap();
        let recs = vec![
            Record::new("a", vec![1.0, 0.0], attrs("w", "g1")),
            Record::new("b", vec![0.9, 0.1], attrs("ww", "g1")),
            Record::new("c", vec![0.7, 0.3], attrs("www", "g2")),
            Record::new("d", vec![0.5, 0.5], attrs("wwww", "g2")),
        ];
        db.upsert("docs", &recs).unwrap();
        db
    }

    fn ids(hits: &[Hit]) -> Vec<&str> {
        hits.iter().map(|h| h.id.as_str()).collect()
    }

    #[tokio::test]
    async fn search_reranked_inverts_the_metric_order() {
        let db = store();
        let plain = SearchOpts {
            top_k: 4,
            ..Default::default()
        };
        assert_eq!(
            ids(&db.search("docs", &[1.0, 0.0], &plain).unwrap()),
            vec!["a", "b", "c", "d"],
            "baseline must be the cosine order, or the flip below proves nothing"
        );

        let opts = SearchOpts {
            top_k: 4,
            rerank: Some(RerankOpts::default()),
            ..Default::default()
        };
        let out = search_reranked(&db, &LenReranker, "docs", &[1.0, 0.0], "q", &opts)
            .await
            .unwrap();
        assert_eq!(ids(&out), vec!["d", "c", "b", "a"]);
        assert_eq!(
            out[0].score, 4.0,
            "the rerank score replaces the cosine one"
        );
    }

    /// The whole point of the over-fetch window: a record outside the plain `top_k` finishes
    /// inside it after reranking.
    #[tokio::test]
    async fn search_reranked_pulls_a_winner_in_from_outside_the_page() {
        let db = store();
        let narrow = SearchOpts {
            top_k: 2,
            ..Default::default()
        };
        assert_eq!(
            ids(&db.search("docs", &[1.0, 0.0], &narrow).unwrap()),
            vec!["a", "b"]
        );

        let opts = SearchOpts {
            top_k: 2,
            rerank: Some(RerankOpts::default()),
            ..Default::default()
        };
        let out = search_reranked(&db, &LenReranker, "docs", &[1.0, 0.0], "q", &opts)
            .await
            .unwrap();
        assert_eq!(
            ids(&out),
            vec!["d", "c"],
            "d was never in the un-widened page"
        );
    }

    /// `offset` is cut once, post-rerank, against the reranked order.
    #[tokio::test]
    async fn search_reranked_paginates_the_reranked_order() {
        let db = store();
        let opts = SearchOpts {
            top_k: 2,
            offset: 1,
            rerank: Some(RerankOpts::default()),
            ..Default::default()
        };
        let out = search_reranked(&db, &LenReranker, "docs", &[1.0, 0.0], "q", &opts)
            .await
            .unwrap();
        assert_eq!(ids(&out), vec!["c", "b"]);
    }

    /// Decision 3: the per-value cap is re-applied after reranking, so a reordering cannot
    /// silently break the diversity guarantee the caller asked for.
    #[tokio::test]
    async fn search_reranked_reapplies_limit_per() {
        let db = store();
        let opts = SearchOpts {
            top_k: 4,
            limit_per: Some(LimitPer::new("group", 1)),
            rerank: Some(RerankOpts::default()),
            ..Default::default()
        };
        let out = search_reranked(&db, &LenReranker, "docs", &[1.0, 0.0], "q", &opts)
            .await
            .unwrap();
        assert_eq!(
            ids(&out),
            vec!["d", "b"],
            "one hit per group, in reranked order"
        );
    }

    /// A hit with no text attr is passed through unranked, after the reranked hits, keeping
    /// its metric score (decision 1).
    #[tokio::test]
    async fn a_candidate_without_text_is_passed_through_after_the_reranked_hits() {
        let mut db = Nidus::open_in_memory(2).unwrap();
        let mut bare = BTreeMap::new();
        bare.insert("group".to_string(), Value::Str("g".to_string()));
        db.upsert(
            "docs",
            &[
                Record::new("has-text", vec![0.5, 0.5], attrs("wwww", "g")),
                Record::new("no-text", vec![1.0, 0.0], bare),
            ],
        )
        .unwrap();

        let opts = SearchOpts {
            top_k: 2,
            rerank: Some(RerankOpts::default()),
            ..Default::default()
        };
        let out = search_reranked(&db, &LenReranker, "docs", &[1.0, 0.0], "q", &opts)
            .await
            .unwrap();
        assert_eq!(ids(&out), vec!["has-text", "no-text"]);
        assert_eq!(out[1].score, 1.0, "passthrough keeps its cosine score");
    }

    /// `opts.rerank == None` must be a pass-through, so an edge can call this unconditionally.
    #[tokio::test]
    async fn no_rerank_opts_is_a_plain_search() {
        let db = store();
        let opts = SearchOpts {
            top_k: 3,
            ..Default::default()
        };
        let out = search_reranked(&db, &LenReranker, "docs", &[1.0, 0.0], "q", &opts)
            .await
            .unwrap();
        assert_eq!(
            ids(&out),
            ids(&db.search("docs", &[1.0, 0.0], &opts).unwrap())
        );
    }

    #[tokio::test]
    async fn hybrid_reranked_reorders_the_fused_ranking() {
        let db = store();
        let query = crate::FtsQuery::new("body", "w");
        let plain = HybridOpts {
            top_k: 4,
            ..Default::default()
        };
        let baseline = db
            .hybrid_search("docs", &[1.0, 0.0], &query, &plain)
            .unwrap();
        assert!(baseline.len() >= 3, "need a non-trivial fused baseline");

        let opts = HybridOpts {
            top_k: 4,
            rerank: Some(RerankOpts::default()),
            ..Default::default()
        };
        let out = hybrid_reranked(&db, &LenReranker, "docs", &[1.0, 0.0], &query, "q", &opts)
            .await
            .unwrap();
        assert_eq!(ids(&out), vec!["d", "c", "b", "a"]);
        assert_ne!(
            ids(&out),
            ids(&baseline),
            "rerank must change the fused order"
        );
    }

    /// A provider returning fewer scores than candidates is an error, not a silent misalignment
    /// of every score onto the wrong document.
    #[tokio::test]
    async fn a_short_score_vector_is_an_error() {
        let db = store();
        let opts = SearchOpts {
            top_k: 4,
            rerank: Some(RerankOpts::default()),
            ..Default::default()
        };
        let err = search_reranked(
            &db,
            &FixedReranker(vec![1.0, 2.0]),
            "docs",
            &[1.0, 0.0],
            "q",
            &opts,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("2 scores for 4 documents"),
            "must name the mismatch: {err}"
        );
    }
}
