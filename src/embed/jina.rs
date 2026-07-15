//! Jina embedding adapter (`jina-embeddings-v3` default).
//!
//! OpenAI-shaped `/v1/embeddings` plus a `task` field
//! (`retrieval.passage` for documents, `retrieval.query` for queries); reuses
//! [`super::openai_shaped`]. Fixed 1024-dimension output.

use super::{EmbedConfig, EmbedError, Embedder, openai_shaped, resolve_base};

const DEFAULT_BASE: &str = "https://api.jina.ai";
const MAX_BATCH: usize = 128;
const MAX_TOKENS: usize = 8_192;
const DIMENSION: usize = 1024;
const TASK_DOC: &str = "retrieval.passage";
const TASK_QUERY: &str = "retrieval.query";

#[derive(Debug)]
pub struct JinaEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
}

impl JinaEmbedder {
    pub fn new(config: EmbedConfig) -> Result<Self, EmbedError> {
        if config.api_key.is_empty() {
            return Err(EmbedError::Config("Jina requires an api_key".into()));
        }
        Ok(Self {
            client: reqwest::Client::new(),
            api_key: config.api_key,
            base_url: resolve_base(config.base_url.as_deref(), DEFAULT_BASE),
            model: config.model,
            extra_headers: config.extra_headers,
        })
    }

    async fn call(&self, texts: &[&str], task: &str) -> Result<Vec<Vec<f32>>, EmbedError> {
        let url = format!("{}/v1/embeddings", self.base_url);
        openai_shaped::embed_call(
            &self.client,
            &url,
            Some(&self.api_key),
            &self.extra_headers,
            &self.model,
            texts,
            Some(task),
            "Jina API",
        )
        .await
    }
}

impl Embedder for JinaEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.call(&[text], TASK_DOC)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Decode("Jina returned no embedding".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(MAX_BATCH) {
            all.extend(self.call(chunk, TASK_DOC).await?);
        }
        Ok(all)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.call(&[text], TASK_QUERY)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Decode("Jina returned no embedding".into()))
    }

    fn dimension(&self) -> usize {
        DIMENSION
    }
    fn max_input_tokens(&self) -> usize {
        MAX_TOKENS
    }
    fn provider_name(&self) -> &str {
        "jina"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::mock_once;
    use super::*;

    #[test]
    fn constructor_requires_key() {
        assert!(JinaEmbedder::new(EmbedConfig::new("jina-embeddings-v3")).is_err());
        let e = JinaEmbedder::new(EmbedConfig::new("jina-embeddings-v3").api_key("k")).unwrap();
        assert_eq!(e.dimension(), 1024);
        assert_eq!(e.provider_name(), "jina");
    }

    #[tokio::test]
    async fn embed_sets_passage_task() {
        let server = mock_once(200, r#"{"data":[{"embedding":[0.1],"index":0}]}"#);
        let e = JinaEmbedder::new(
            EmbedConfig::new("jina-embeddings-v3")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        e.embed("d").await.unwrap();
        assert!(
            server
                .captured()
                .body
                .contains("\"task\":\"retrieval.passage\"")
        );
    }

    #[tokio::test]
    async fn query_sets_query_task() {
        let server = mock_once(200, r#"{"data":[{"embedding":[0.1],"index":0}]}"#);
        let e = JinaEmbedder::new(
            EmbedConfig::new("jina-embeddings-v3")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        e.embed_query("q").await.unwrap();
        assert!(
            server
                .captured()
                .body
                .contains("\"task\":\"retrieval.query\"")
        );
    }
}
