//! Voyage AI embedding adapter (`voyage-3` default).
//!
//! Bespoke wire shape: `POST /v1/embeddings` with an `input_type` tag
//! (`document` vs `query`), response `{data:[{embedding}]}`.

use serde::Deserialize;

use super::{EmbedConfig, EmbedError, Embedder, post_json, resolve_base};
use crate::http::RetryPolicy;

const DEFAULT_BASE: &str = "https://api.voyageai.com";
const MAX_BATCH: usize = 128;
const MAX_TOKENS: usize = 16_000;

#[derive(Debug)]
pub struct VoyageEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
    dimension: usize,
}

impl VoyageEmbedder {
    pub fn new(config: EmbedConfig) -> Result<Self, EmbedError> {
        if config.api_key.is_empty() {
            return Err(EmbedError::Config("Voyage requires an api_key".into()));
        }
        let dimension = dimension_for_model(&config.model);
        Ok(Self {
            client: reqwest::Client::new(),
            api_key: config.api_key,
            base_url: resolve_base(config.base_url.as_deref(), DEFAULT_BASE),
            model: config.model,
            extra_headers: config.extra_headers,
            dimension,
        })
    }

    async fn call(&self, texts: &[&str], input_type: &str) -> Result<Vec<Vec<f32>>, EmbedError> {
        let body = build_body(&self.model, texts, input_type);
        let url = format!("{}/v1/embeddings", self.base_url);
        let policy = RetryPolicy::standard(3, 1000);
        let resp: ApiResponse = post_json(
            &self.client,
            &policy,
            "Voyage API",
            &url,
            Some(&self.api_key),
            &self.extra_headers,
            &body,
        )
        .await?;
        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

impl Embedder for VoyageEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        one(self.call(&[text], "document").await?, "Voyage")
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(MAX_BATCH) {
            all.extend(self.call(chunk, "document").await?);
        }
        Ok(all)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        one(self.call(&[text], "query").await?, "Voyage")
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
    fn max_input_tokens(&self) -> usize {
        MAX_TOKENS
    }
    fn provider_name(&self) -> &str {
        "voyage"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
}

fn build_body(model: &str, texts: &[&str], input_type: &str) -> serde_json::Value {
    serde_json::json!({ "model": model, "input": texts, "input_type": input_type })
}

fn dimension_for_model(model: &str) -> usize {
    match model {
        "voyage-3-lite" => 512,
        // The 1536-dim generation. All current voyage-3.x models are 1024.
        "voyage-code-2" | "voyage-large-2" => 1536,
        _ => 1024,
    }
}

fn one(v: Vec<Vec<f32>>, who: &str) -> Result<Vec<f32>, EmbedError> {
    v.into_iter()
        .next()
        .ok_or_else(|| EmbedError::Decode(format!("{who} returned no embedding")))
}

#[derive(Deserialize)]
struct ApiResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::super::testutil::mock_once;
    use super::*;

    #[test]
    fn body_sets_input_type() {
        let doc = build_body("voyage-3", &["hi"], "document");
        assert_eq!(doc["input_type"], "document");
        assert_eq!(doc["model"], "voyage-3");
        assert_eq!(doc["input"][0], "hi");
        let q = build_body("voyage-3", &["hi"], "query");
        assert_eq!(q["input_type"], "query");
    }

    #[test]
    fn dimension_lookup() {
        assert_eq!(dimension_for_model("voyage-3"), 1024);
        assert_eq!(dimension_for_model("voyage-3-lite"), 512);
        assert_eq!(dimension_for_model("voyage-code-2"), 1536);
        assert_eq!(dimension_for_model("voyage-large-2"), 1536);
        assert_eq!(dimension_for_model("unknown"), 1024);
    }

    #[test]
    fn response_parsing() {
        let json = r#"{"data":[{"embedding":[0.1,0.2,0.3],"index":0}]}"#;
        let r: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.data[0].embedding, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn constructor_requires_key() {
        assert!(VoyageEmbedder::new(EmbedConfig::new("voyage-3")).is_err());
        let e = VoyageEmbedder::new(EmbedConfig::new("voyage-3").api_key("k")).unwrap();
        assert_eq!(e.dimension(), 1024);
        assert_eq!(e.provider_name(), "voyage");
        assert_eq!(e.model_name(), "voyage-3");
    }

    #[tokio::test]
    async fn embed_hits_endpoint_and_parses() {
        let server = mock_once(200, r#"{"data":[{"embedding":[1.0,2.0],"index":0}]}"#);
        let e = VoyageEmbedder::new(
            EmbedConfig::new("voyage-3")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let v = e.embed("hello").await.unwrap();
        assert_eq!(v, vec![1.0, 2.0]);
        let cap = server.captured();
        assert_eq!(cap.method, "POST");
        assert_eq!(cap.path, "/v1/embeddings");
        assert!(cap.body.contains("\"input_type\":\"document\""));
    }

    #[tokio::test]
    async fn query_uses_query_input_type() {
        let server = mock_once(200, r#"{"data":[{"embedding":[9.0],"index":0}]}"#);
        let e = VoyageEmbedder::new(
            EmbedConfig::new("voyage-3")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        e.embed_query("q").await.unwrap();
        assert!(server.captured().body.contains("\"input_type\":\"query\""));
    }

    #[tokio::test]
    async fn non_2xx_maps_to_api_error() {
        let server = mock_once(400, r#"{"error":"bad"}"#);
        let e = VoyageEmbedder::new(
            EmbedConfig::new("voyage-3")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let err = e.embed("x").await.unwrap_err();
        match err {
            EmbedError::Api { status, .. } => assert_eq!(status, 400),
            other => panic!("expected Api error, got {other:?}"),
        }
    }
}
