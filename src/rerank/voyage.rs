//! Voyage AI rerank adapter (`rerank-2.5` default).

use serde::Deserialize;

use super::{RerankConfig, RerankError, Reranker, post_json, resolve_base};
use crate::http::RetryPolicy;

const DEFAULT_BASE: &str = "https://api.voyageai.com";
/// Voyage's documented per-request document cap.
const MAX_DOCUMENTS: usize = 1000;

#[derive(Debug)]
pub struct VoyageReranker {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
}

impl VoyageReranker {
    pub fn new(config: RerankConfig) -> Result<Self, RerankError> {
        if config.api_key.is_empty() {
            return Err(RerankError::Config("Voyage requires an api_key".into()));
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

impl Reranker for VoyageReranker {
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
                "Voyage rerank API",
                &url,
                Some(&self.api_key),
                &self.extra_headers,
                &body,
            )
            .await?;
            scatter(&mut scores, offset * MAX_DOCUMENTS, chunk.len(), resp.data)?;
        }
        Ok(scores)
    }

    fn provider_name(&self) -> &str {
        "voyage"
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

/// Scatter `data`'s `{index, relevance_score}` pairs back into `scores[base..base+len]`,
/// undoing whatever order/truncation the provider applied. An index outside `0..len` is a
/// decode error, not a silently dropped candidate.
fn scatter(
    scores: &mut [f32],
    base: usize,
    len: usize,
    data: Vec<ScoredDoc>,
) -> Result<(), RerankError> {
    if data.len() != len {
        return Err(RerankError::Decode(format!(
            "Voyage returned {} scores for {len} documents",
            data.len()
        )));
    }
    for d in data {
        if d.index >= len {
            return Err(RerankError::Decode(format!(
                "Voyage returned out-of-range index {} for {len} documents",
                d.index
            )));
        }
        scores[base + d.index] = d.relevance_score;
    }
    Ok(())
}

#[derive(Deserialize)]
struct ApiResponse {
    data: Vec<ScoredDoc>,
}

#[derive(Deserialize)]
struct ScoredDoc {
    index: usize,
    relevance_score: f32,
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{mock_once, mock_persistent};
    use super::*;

    #[test]
    fn body_shape() {
        let b = build_body("rerank-2.5", "q", &["a", "b"]);
        assert_eq!(b["model"], "rerank-2.5");
        assert_eq!(b["query"], "q");
        assert_eq!(b["documents"][0], "a");
        assert_eq!(b["return_documents"], false);
    }

    #[test]
    fn response_parsing() {
        let json =
            r#"{"data":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}"#;
        let r: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.data.len(), 2);
    }

    #[test]
    fn scatter_restores_input_order_from_a_resorted_response() {
        let data = vec![
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
        scatter(&mut scores, 0, 2, data).unwrap();
        assert_eq!(scores, vec![0.1, 0.9]);
    }

    #[test]
    fn scatter_rejects_an_out_of_range_index() {
        let data = vec![ScoredDoc {
            index: 5,
            relevance_score: 0.9,
        }];
        let mut scores = vec![0.0; 1];
        assert!(scatter(&mut scores, 0, 1, data).is_err());
    }

    #[test]
    fn constructor_requires_key() {
        assert!(VoyageReranker::new(RerankConfig::new("rerank-2.5")).is_err());
        let r = VoyageReranker::new(RerankConfig::new("rerank-2.5").api_key("k")).unwrap();
        assert_eq!(r.provider_name(), "voyage");
        assert_eq!(r.model_name(), "rerank-2.5");
        assert_eq!(r.max_documents(), 1000);
    }

    #[tokio::test]
    async fn rerank_hits_endpoint_and_returns_input_order() {
        let server = mock_once(
            200,
            r#"{"data":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}"#,
        );
        let r = VoyageReranker::new(
            RerankConfig::new("rerank-2.5")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let scores = r.rerank("q", &["doc-a", "doc-b"], None).await.unwrap();
        assert_eq!(scores, vec![0.1, 0.9]);
        let cap = server.captured();
        assert_eq!(cap.method, "POST");
        assert_eq!(cap.path, "/v1/rerank");
        assert!(cap.body.contains("\"return_documents\":false"));
    }

    #[tokio::test]
    async fn a_per_call_model_override_reaches_the_wire() {
        let server = mock_once(200, r#"{"data":[{"index":0,"relevance_score":0.5}]}"#);
        let r = VoyageReranker::new(
            RerankConfig::new("rerank-2.5")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        r.rerank("q", &["d"], Some("rerank-lite-1")).await.unwrap();
        let body = server.captured().body;
        assert!(body.contains("\"model\":\"rerank-lite-1\""), "{body}");
        assert!(!body.contains("rerank-2.5"), "{body}");
    }

    #[tokio::test]
    async fn rerank_chunks_at_max_documents() {
        // Twice the per-request cap: two equal-sized round trips against the same mock,
        // each chunk's scores scattered into its own absolute slot range.
        let owned: Vec<String> = (0..2 * MAX_DOCUMENTS).map(|i| format!("doc-{i}")).collect();
        let docs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let entries: Vec<String> = (0..MAX_DOCUMENTS)
            .map(|i| {
                format!(
                    "{{\"index\":{i},\"relevance_score\":{}}}",
                    (MAX_DOCUMENTS - i) as f32
                )
            })
            .collect();
        let resp_body = format!("{{\"data\":[{}]}}", entries.join(","));
        let server = mock_persistent(200, &resp_body, 2);
        let r = VoyageReranker::new(
            RerankConfig::new("rerank-2.5")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let scores = r.rerank("q", &docs, None).await.unwrap();
        assert_eq!(scores.len(), 2 * MAX_DOCUMENTS);
        assert_eq!(scores[0], MAX_DOCUMENTS as f32);
        assert_eq!(scores[MAX_DOCUMENTS - 1], 1.0);
        assert_eq!(scores[MAX_DOCUMENTS], MAX_DOCUMENTS as f32);
        assert_eq!(scores[2 * MAX_DOCUMENTS - 1], 1.0);
        assert_eq!(server.captured_n(2).len(), 2);
    }

    #[tokio::test]
    async fn non_2xx_maps_to_api_error() {
        let server = mock_once(400, r#"{"error":"bad"}"#);
        let r = VoyageReranker::new(
            RerankConfig::new("rerank-2.5")
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
