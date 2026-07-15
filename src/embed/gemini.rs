//! Google Gemini embedding adapter (`text-embedding-004` default).
//!
//! Bespoke wire shape (Generative Language API), authenticated with the
//! `x-goog-api-key` header:
//! - single: `POST /v1beta/models/{model}:embedContent` →
//!   `{embedding:{values:[..]}}`
//! - batch:  `POST /v1beta/models/{model}:batchEmbedContents` with
//!   `{requests:[..]}` → `{embeddings:[{values:[..]}]}`
//!
//! `taskType` is `RETRIEVAL_DOCUMENT` for documents, `RETRIEVAL_QUERY` for
//! queries.

use serde::Deserialize;

use super::{EmbedConfig, EmbedError, Embedder, post_json, resolve_base};
use crate::http::RetryPolicy;

const DEFAULT_BASE: &str = "https://generativelanguage.googleapis.com";
const MAX_BATCH: usize = 100;
const MAX_TOKENS: usize = 2_048;
const TASK_DOC: &str = "RETRIEVAL_DOCUMENT";
const TASK_QUERY: &str = "RETRIEVAL_QUERY";

/// Default output dimension per model. The legacy `text-embedding-004` /
/// `embedding-001` emit 768; the GA `gemini-embedding-001` defaults to 3072.
/// (gemini-embedding-001 supports MRL truncation via `output_dimensionality`,
/// which we do not set, so it returns its 3072 default.)
fn dimension_for_model(model: &str) -> usize {
    match model {
        "text-embedding-004" | "embedding-001" => 768,
        "gemini-embedding-001" | "gemini-embedding-exp-03-07" => 3072,
        _ => 768,
    }
}

#[derive(Debug)]
pub struct GeminiEmbedder {
    client: reqwest::Client,
    model: String,
    base_url: String,
    dimension: usize,
    /// Extra headers already merged with `x-goog-api-key`.
    headers: Vec<(String, String)>,
}

impl GeminiEmbedder {
    pub fn new(config: EmbedConfig) -> Result<Self, EmbedError> {
        if config.api_key.is_empty() {
            return Err(EmbedError::Config("Gemini requires an api_key".into()));
        }
        let mut headers = config.extra_headers;
        headers.push(("x-goog-api-key".to_string(), config.api_key));
        Ok(Self {
            client: reqwest::Client::new(),
            base_url: resolve_base(config.base_url.as_deref(), DEFAULT_BASE),
            dimension: dimension_for_model(&config.model),
            model: config.model,
            headers,
        })
    }

    fn model_path(&self) -> String {
        format!("models/{}", self.model)
    }

    fn content(text: &str) -> serde_json::Value {
        serde_json::json!({ "parts": [ { "text": text } ] })
    }

    async fn embed_single(&self, text: &str, task: &str) -> Result<Vec<f32>, EmbedError> {
        let body = serde_json::json!({
            "model": self.model_path(),
            "content": Self::content(text),
            "taskType": task,
        });
        let url = format!(
            "{}/v1beta/models/{}:embedContent",
            self.base_url, self.model
        );
        let policy = RetryPolicy::standard(3, 1000);
        let resp: SingleResponse = post_json(
            &self.client,
            &policy,
            "Gemini API",
            &url,
            None,
            &self.headers,
            &body,
        )
        .await?;
        Ok(resp.embedding.values)
    }

    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let requests: Vec<serde_json::Value> = texts
            .iter()
            .map(|t| {
                serde_json::json!({
                    "model": self.model_path(),
                    "content": Self::content(t),
                    "taskType": TASK_DOC,
                })
            })
            .collect();
        let body = serde_json::json!({ "requests": requests });
        let url = format!(
            "{}/v1beta/models/{}:batchEmbedContents",
            self.base_url, self.model
        );
        let policy = RetryPolicy::standard(3, 1000);
        let resp: BatchResponse = post_json(
            &self.client,
            &policy,
            "Gemini API",
            &url,
            None,
            &self.headers,
            &body,
        )
        .await?;
        Ok(resp.embeddings.into_iter().map(|e| e.values).collect())
    }
}

impl Embedder for GeminiEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_single(text, TASK_DOC).await
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(MAX_BATCH) {
            all.extend(self.embed_many(chunk).await?);
        }
        Ok(all)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_single(text, TASK_QUERY).await
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
    fn max_input_tokens(&self) -> usize {
        MAX_TOKENS
    }
    fn provider_name(&self) -> &str {
        "gemini"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
}

#[derive(Deserialize)]
struct SingleResponse {
    embedding: Values,
}

#[derive(Deserialize)]
struct BatchResponse {
    embeddings: Vec<Values>,
}

#[derive(Deserialize)]
struct Values {
    values: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::super::testutil::mock_once;
    use super::*;

    #[test]
    fn constructor_requires_key() {
        assert!(GeminiEmbedder::new(EmbedConfig::new("text-embedding-004")).is_err());
        let e = GeminiEmbedder::new(EmbedConfig::new("text-embedding-004").api_key("k")).unwrap();
        assert_eq!(e.dimension(), 768);
        assert_eq!(e.provider_name(), "gemini");
    }

    #[test]
    fn dimension_is_model_aware() {
        assert_eq!(dimension_for_model("text-embedding-004"), 768);
        assert_eq!(dimension_for_model("gemini-embedding-001"), 3072);
        // The GA model users actually pick must report 3072, not the legacy 768.
        let e = GeminiEmbedder::new(EmbedConfig::new("gemini-embedding-001").api_key("k")).unwrap();
        assert_eq!(e.dimension(), 3072);
    }

    #[test]
    fn response_parsing() {
        let single = r#"{"embedding":{"values":[0.1,0.2,0.3]}}"#;
        let r: SingleResponse = serde_json::from_str(single).unwrap();
        assert_eq!(r.embedding.values, vec![0.1, 0.2, 0.3]);

        let batch = r#"{"embeddings":[{"values":[0.1]},{"values":[0.2]}]}"#;
        let b: BatchResponse = serde_json::from_str(batch).unwrap();
        assert_eq!(b.embeddings.len(), 2);
    }

    #[tokio::test]
    async fn embed_single_endpoint_and_task() {
        let server = mock_once(200, r#"{"embedding":{"values":[1.0,2.0]}}"#);
        let e = GeminiEmbedder::new(
            EmbedConfig::new("text-embedding-004")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        assert_eq!(e.embed("d").await.unwrap(), vec![1.0, 2.0]);
        let cap = server.captured();
        assert_eq!(cap.path, "/v1beta/models/text-embedding-004:embedContent");
        assert!(cap.body.contains("\"taskType\":\"RETRIEVAL_DOCUMENT\""));
        assert!(cap.body.contains("\"model\":\"models/text-embedding-004\""));
    }

    #[tokio::test]
    async fn batch_uses_batch_endpoint() {
        let server = mock_once(200, r#"{"embeddings":[{"values":[1.0]},{"values":[2.0]}]}"#);
        let e = GeminiEmbedder::new(
            EmbedConfig::new("text-embedding-004")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let out = e.embed_batch(&["a", "b"]).await.unwrap();
        assert_eq!(out, vec![vec![1.0], vec![2.0]]);
        assert_eq!(
            server.captured().path,
            "/v1beta/models/text-embedding-004:batchEmbedContents"
        );
    }

    #[tokio::test]
    async fn query_uses_query_task() {
        let server = mock_once(200, r#"{"embedding":{"values":[1.0]}}"#);
        let e = GeminiEmbedder::new(
            EmbedConfig::new("text-embedding-004")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        e.embed_query("q").await.unwrap();
        assert!(
            server
                .captured()
                .body
                .contains("\"taskType\":\"RETRIEVAL_QUERY\"")
        );
    }
}
