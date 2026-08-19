//! Jina AI rerank adapter (`jina-reranker-v2-base-multilingual` default).

use serde::Deserialize;

use super::{RerankConfig, RerankError, Reranker, post_json, resolve_base};
use crate::http::RetryPolicy;

const DEFAULT_BASE: &str = "https://api.jina.ai";
/// Jina's documented per-request document cap.
const MAX_DOCUMENTS: usize = 1000;

#[derive(Debug)]
pub struct JinaReranker {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
}

impl JinaReranker {
    pub fn new(config: RerankConfig) -> Result<Self, RerankError> {
        if config.api_key.is_empty() {
            return Err(RerankError::Config("Jina requires an api_key".into()));
        }
        Ok(Self {
            client: reqwest::Client::new(),
            api_key: config.api_key,
            base_url: resolve_base(config.base_url.as_deref(), DEFAULT_BASE),
            model: config.model,
            extra_headers: config.extra_headers,
        })
    }
}

impl Reranker for JinaReranker {
    async fn rerank(
        &self,
        query: &str,
        docs: &[&str],
        model: Option<&str>,
    ) -> Result<Vec<f32>, RerankError> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let mut scores = vec![0.0f32; docs.len()];
        for (offset, chunk) in docs.chunks(MAX_DOCUMENTS).enumerate() {
            let body = build_body(model.unwrap_or(&self.model), query, chunk);
            let url = format!("{}/v1/rerank", self.base_url);
            let policy = RetryPolicy::standard(3, 1000);
            let resp: ApiResponse = post_json(
                &self.client,
                &policy,
                "Jina rerank API",
                &url,
                Some(&self.api_key),
                &self.extra_headers,
                &body,
            )
            .await?;
            scatter(
                &mut scores,
                offset * MAX_DOCUMENTS,
                chunk.len(),
                resp.results,
            )?;
        }
        Ok(scores)
    }

    fn provider_name(&self) -> &str {
        "jina"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
    fn max_documents(&self) -> usize {
        MAX_DOCUMENTS
    }
}

fn build_body(model: &str, query: &str, docs: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "query": query,
        "documents": docs,
        "return_documents": false,
    })
}

/// Scatter `results`' `{index, relevance_score}` pairs back into `scores[base..base+len]`,
/// undoing whatever order/truncation the provider applied. An index outside `0..len` is a
/// decode error, not a silently dropped candidate.
fn scatter(
    scores: &mut [f32],
    base: usize,
    len: usize,
    results: Vec<ScoredDoc>,
) -> Result<(), RerankError> {
    if results.len() != len {
        return Err(RerankError::Decode(format!(
            "Jina returned {} scores for {len} documents",
            results.len()
        )));
    }
    for d in results {
        if d.index >= len {
            return Err(RerankError::Decode(format!(
                "Jina returned out-of-range index {} for {len} documents",
                d.index
            )));
        }
        scores[base + d.index] = d.relevance_score;
    }
    Ok(())
}

#[derive(Deserialize)]
struct ApiResponse {
    results: Vec<ScoredDoc>,
}

#[derive(Deserialize)]
struct ScoredDoc {
    index: usize,
    relevance_score: f32,
}

#[cfg(test)]
mod tests {
    use super::super::testutil::mock_once;
    use super::*;

    #[test]
    fn body_shape() {
        let b = build_body("jina-reranker-v2-base-multilingual", "q", &["a", "b"]);
        assert_eq!(b["model"], "jina-reranker-v2-base-multilingual");
        assert_eq!(b["query"], "q");
        assert_eq!(b["documents"][1], "b");
        assert_eq!(b["return_documents"], false);
    }

    #[test]
    fn response_parsing() {
        let json =
            r#"{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}"#;
        let r: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.results.len(), 2);
    }

    #[test]
    fn scatter_restores_input_order_from_a_resorted_response() {
        let results = vec![
            ScoredDoc {
                index: 1,
                relevance_score: 0.9,
            },
            ScoredDoc {
                index: 0,
                relevance_score: 0.1,
            },
        ];
        let mut scores = vec![0.0; 2];
        scatter(&mut scores, 0, 2, results).unwrap();
        assert_eq!(scores, vec![0.1, 0.9]);
    }

    #[test]
    fn scatter_rejects_an_out_of_range_index() {
        let results = vec![ScoredDoc {
            index: 5,
            relevance_score: 0.9,
        }];
        let mut scores = vec![0.0; 1];
        assert!(scatter(&mut scores, 0, 1, results).is_err());
    }

    #[test]
    fn constructor_requires_key() {
        assert!(
            JinaReranker::new(RerankConfig::new("jina-reranker-v2-base-multilingual")).is_err()
        );
        let r =
            JinaReranker::new(RerankConfig::new("jina-reranker-v2-base-multilingual").api_key("k"))
                .unwrap();
        assert_eq!(r.provider_name(), "jina");
        assert_eq!(r.max_documents(), 1000);
    }

    #[tokio::test]
    async fn rerank_hits_endpoint_and_returns_input_order() {
        let server = mock_once(
            200,
            r#"{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}"#,
        );
        let r = JinaReranker::new(
            RerankConfig::new("jina-reranker-v2-base-multilingual")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let scores = r.rerank("q", &["doc-a", "doc-b"], None).await.unwrap();
        assert_eq!(scores, vec![0.1, 0.9]);
        let cap = server.captured();
        assert_eq!(cap.path, "/v1/rerank");
        assert!(cap.body.contains("\"return_documents\":false"));
    }

    #[tokio::test]
    async fn non_2xx_maps_to_api_error() {
        let server = mock_once(400, r#"{"error":"bad"}"#);
        let r = JinaReranker::new(
            RerankConfig::new("jina-reranker-v2-base-multilingual")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let err = r.rerank("q", &["x"], None).await.unwrap_err();
        match err {
            RerankError::Api { status, .. } => assert_eq!(status, 400),
            other => panic!("expected Api error, got {other:?}"),
        }
    }
}
