//! Cohere rerank adapter (`rerank-v3.5` default).

use serde::Deserialize;

use super::{RerankConfig, RerankError, Reranker, post_json, resolve_base, scatter_by_index};
use crate::http::RetryPolicy;

const DEFAULT_BASE: &str = "https://api.cohere.com";
const MAX_DOCUMENTS: usize = 1000;

#[derive(Debug)]
pub struct CohereReranker {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
}

impl CohereReranker {
    pub fn new(config: RerankConfig) -> Result<Self, RerankError> {
        if config.api_key.is_empty() {
            return Err(RerankError::Config("Cohere requires an api_key".into()));
        }
        Ok(Self {
            client: reqwest::Client::new(),
            api_key: config.api_key,
            base_url: resolve_base(config.base_url.as_deref(), DEFAULT_BASE),
            model: config.model,
            extra_headers: config.extra_headers,
        })
    }

    async fn call(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, RerankError> {
        let body = build_body(&self.model, query, documents);
        let url = format!("{}/v2/rerank", self.base_url);
        let policy = RetryPolicy::standard(3, 1000);
        let resp: ApiResponse = post_json(
            &self.client,
            &policy,
            "Cohere rerank API",
            &url,
            &self.api_key,
            &self.extra_headers,
            &body,
        )
        .await?;
        scatter_by_index(
            documents.len(),
            resp.results
                .into_iter()
                .map(|d| (d.index, d.relevance_score)),
            "Cohere rerank API",
        )
    }
}

impl Reranker for CohereReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, RerankError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let mut all = Vec::with_capacity(documents.len());
        for chunk in documents.chunks(MAX_DOCUMENTS) {
            all.extend(self.call(query, chunk).await?);
        }
        Ok(all)
    }

    fn provider_name(&self) -> &str {
        "cohere"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
    fn max_documents(&self) -> usize {
        MAX_DOCUMENTS
    }
}

/// Every document gets a score: the `Reranker` contract is one score per input document in
/// input order, and the caller's over-fetch window is what narrows the result set. Asking the
/// provider for its own top-n would return fewer scores than candidates and misalign them.
fn build_body(model: &str, query: &str, documents: &[&str]) -> serde_json::Value {
    serde_json::json!({ "model": model, "query": query, "documents": documents })
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
    fn constructor_requires_key() {
        assert!(CohereReranker::new(RerankConfig::new("rerank-v3.5")).is_err());
        let r = CohereReranker::new(RerankConfig::new("rerank-v3.5").api_key("k")).unwrap();
        assert_eq!(r.provider_name(), "cohere");
        assert_eq!(r.model_name(), "rerank-v3.5");
    }

    #[test]
    fn body_sets_expected_fields() {
        let b = build_body("rerank-v3.5", "q", &["a", "b"]);
        assert_eq!(b["model"], "rerank-v3.5");
        assert_eq!(b["query"], "q");
        assert_eq!(b["documents"][0], "a");
        assert_eq!(b["documents"][1], "b");
        // No provider-side top-n: a truncated response would misalign scores with candidates.
        assert!(b.get("top_n").is_none());
    }

    #[test]
    fn response_scatters_out_of_order_indices() {
        let json =
            r#"{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}"#;
        let r: ApiResponse = serde_json::from_str(json).unwrap();
        let scores = scatter_by_index(
            2,
            r.results.into_iter().map(|d| (d.index, d.relevance_score)),
            "test",
        )
        .unwrap();
        assert_eq!(scores, vec![0.1, 0.9]);
    }

    #[test]
    fn response_short_by_one_is_decode_error() {
        let json = r#"{"results":[{"index":0,"relevance_score":0.1}]}"#;
        let r: ApiResponse = serde_json::from_str(json).unwrap();
        let err = scatter_by_index(
            2,
            r.results.into_iter().map(|d| (d.index, d.relevance_score)),
            "test",
        )
        .unwrap_err();
        assert!(matches!(err, RerankError::Decode(_)));
    }

    #[tokio::test]
    async fn rerank_hits_v2_endpoint_and_parses() {
        let server = mock_once(
            200,
            r#"{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.2}]}"#,
        );
        let r = CohereReranker::new(
            RerankConfig::new("rerank-v3.5")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let scores = r.rerank("q", &["a", "b"]).await.unwrap();
        assert_eq!(scores, vec![0.2, 0.9]);
        let cap = server.captured();
        assert_eq!(cap.method, "POST");
        assert_eq!(cap.path, "/v2/rerank");
    }

    #[tokio::test]
    async fn non_2xx_maps_to_api_error() {
        let server = mock_once(400, r#"{"error":"bad"}"#);
        let r = CohereReranker::new(
            RerankConfig::new("rerank-v3.5")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let err = r.rerank("q", &["a"]).await.unwrap_err();
        match err {
            RerankError::Api { status, .. } => assert_eq!(status, 400),
            other => panic!("expected Api error, got {other:?}"),
        }
    }
}
