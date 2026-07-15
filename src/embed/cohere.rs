//! Cohere embedding adapter (`embed-english-v3.0` default).
//!
//! Bespoke wire shape: `POST /v2/embed` with `{model, texts, input_type,
//! embedding_types:["float"]}`, response `{embeddings:{float:[[..]]}}`.
//! `input_type` is `search_document` for documents, `search_query` for queries.

use serde::Deserialize;

use super::{EmbedConfig, EmbedError, Embedder, post_json, resolve_base};
use crate::http::RetryPolicy;

const DEFAULT_BASE: &str = "https://api.cohere.com";
const MAX_BATCH: usize = 96;
const MAX_TOKENS: usize = 512;
const INPUT_DOC: &str = "search_document";
const INPUT_QUERY: &str = "search_query";

/// Output dimension per model. Cohere's v3 "light" models emit 384; the
/// standard v3 models (and the default) emit 1024.
fn dimension_for_model(model: &str) -> usize {
    match model {
        "embed-english-light-v3.0" | "embed-multilingual-light-v3.0" => 384,
        _ => 1024,
    }
}

#[derive(Debug)]
pub struct CohereEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    dimension: usize,
    extra_headers: Vec<(String, String)>,
}

impl CohereEmbedder {
    pub fn new(config: EmbedConfig) -> Result<Self, EmbedError> {
        if config.api_key.is_empty() {
            return Err(EmbedError::Config("Cohere requires an api_key".into()));
        }
        Ok(Self {
            client: reqwest::Client::new(),
            api_key: config.api_key,
            base_url: resolve_base(config.base_url.as_deref(), DEFAULT_BASE),
            dimension: dimension_for_model(&config.model),
            model: config.model,
            extra_headers: config.extra_headers,
        })
    }

    async fn call(&self, texts: &[&str], input_type: &str) -> Result<Vec<Vec<f32>>, EmbedError> {
        let body = serde_json::json!({
            "model": self.model,
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        });
        let url = format!("{}/v2/embed", self.base_url);
        let policy = RetryPolicy::standard(3, 1000);
        let resp: ApiResponse = post_json(
            &self.client,
            &policy,
            "Cohere API",
            &url,
            Some(&self.api_key),
            &self.extra_headers,
            &body,
        )
        .await?;
        Ok(resp.embeddings.float)
    }
}

impl Embedder for CohereEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        one(self.call(&[text], INPUT_DOC).await?)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(MAX_BATCH) {
            all.extend(self.call(chunk, INPUT_DOC).await?);
        }
        Ok(all)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        one(self.call(&[text], INPUT_QUERY).await?)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
    fn max_input_tokens(&self) -> usize {
        MAX_TOKENS
    }
    fn provider_name(&self) -> &str {
        "cohere"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
}

fn one(v: Vec<Vec<f32>>) -> Result<Vec<f32>, EmbedError> {
    v.into_iter()
        .next()
        .ok_or_else(|| EmbedError::Decode("Cohere returned no embedding".into()))
}

#[derive(Deserialize)]
struct ApiResponse {
    embeddings: FloatEmbeddings,
}

#[derive(Deserialize)]
struct FloatEmbeddings {
    float: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::super::testutil::mock_once;
    use super::*;

    #[test]
    fn constructor_requires_key() {
        assert!(CohereEmbedder::new(EmbedConfig::new("embed-english-v3.0")).is_err());
        let e = CohereEmbedder::new(EmbedConfig::new("embed-english-v3.0").api_key("k")).unwrap();
        assert_eq!(e.dimension(), 1024);
        assert_eq!(e.provider_name(), "cohere");
    }

    #[test]
    fn light_models_report_384() {
        assert_eq!(dimension_for_model("embed-english-light-v3.0"), 384);
        assert_eq!(dimension_for_model("embed-multilingual-light-v3.0"), 384);
        assert_eq!(dimension_for_model("embed-english-v3.0"), 1024);
        let e =
            CohereEmbedder::new(EmbedConfig::new("embed-english-light-v3.0").api_key("k")).unwrap();
        assert_eq!(e.dimension(), 384);
    }

    #[test]
    fn response_parsing() {
        let json = r#"{"id":"x","embeddings":{"float":[[0.1,0.2],[0.3,0.4]]}}"#;
        let r: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.embeddings.float.len(), 2);
        assert_eq!(r.embeddings.float[0], vec![0.1, 0.2]);
    }

    #[tokio::test]
    async fn embed_hits_v2_endpoint_with_doc_input() {
        let server = mock_once(200, r#"{"embeddings":{"float":[[7.0,8.0]]}}"#);
        let e = CohereEmbedder::new(
            EmbedConfig::new("embed-english-v3.0")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        assert_eq!(e.embed("d").await.unwrap(), vec![7.0, 8.0]);
        let cap = server.captured();
        assert_eq!(cap.path, "/v2/embed");
        assert!(cap.body.contains("\"input_type\":\"search_document\""));
        assert!(cap.body.contains("\"embedding_types\":[\"float\"]"));
    }

    #[tokio::test]
    async fn query_uses_search_query() {
        let server = mock_once(200, r#"{"embeddings":{"float":[[1.0]]}}"#);
        let e = CohereEmbedder::new(
            EmbedConfig::new("embed-english-v3.0")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        e.embed_query("q").await.unwrap();
        assert!(
            server
                .captured()
                .body
                .contains("\"input_type\":\"search_query\"")
        );
    }
}
