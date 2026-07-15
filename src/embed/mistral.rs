//! Mistral embedding adapter (`mistral-embed` default).
//!
//! OpenAI-shaped `/v1/embeddings`; reuses [`super::openai_shaped`]. Fixed
//! 1024-dimension output.

use super::{EmbedConfig, EmbedError, Embedder, openai_shaped, resolve_base};

const DEFAULT_BASE: &str = "https://api.mistral.ai";
const MAX_BATCH: usize = 128;
const MAX_TOKENS: usize = 8_192;
const DIMENSION: usize = 1024;

#[derive(Debug)]
pub struct MistralEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
}

impl MistralEmbedder {
    pub fn new(config: EmbedConfig) -> Result<Self, EmbedError> {
        if config.api_key.is_empty() {
            return Err(EmbedError::Config("Mistral requires an api_key".into()));
        }
        Ok(Self {
            client: reqwest::Client::new(),
            api_key: config.api_key,
            base_url: resolve_base(config.base_url.as_deref(), DEFAULT_BASE),
            model: config.model,
            extra_headers: config.extra_headers,
        })
    }

    async fn call(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let url = format!("{}/v1/embeddings", self.base_url);
        openai_shaped::embed_call(
            &self.client,
            &url,
            Some(&self.api_key),
            &self.extra_headers,
            &self.model,
            texts,
            None,
            "Mistral API",
        )
        .await
    }
}

impl Embedder for MistralEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.call(&[text])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Decode("Mistral returned no embedding".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(MAX_BATCH) {
            all.extend(self.call(chunk).await?);
        }
        Ok(all)
    }

    fn dimension(&self) -> usize {
        DIMENSION
    }
    fn max_input_tokens(&self) -> usize {
        MAX_TOKENS
    }
    fn provider_name(&self) -> &str {
        "mistral"
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
        assert!(MistralEmbedder::new(EmbedConfig::new("mistral-embed")).is_err());
        let e = MistralEmbedder::new(EmbedConfig::new("mistral-embed").api_key("k")).unwrap();
        assert_eq!(e.dimension(), 1024);
        assert_eq!(e.provider_name(), "mistral");
    }

    #[tokio::test]
    async fn embed_parses_and_hits_endpoint() {
        let server = mock_once(200, r#"{"data":[{"embedding":[0.5,0.6],"index":0}]}"#);
        let e = MistralEmbedder::new(
            EmbedConfig::new("mistral-embed")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        assert_eq!(e.embed("hi").await.unwrap(), vec![0.5, 0.6]);
        assert_eq!(server.captured().path, "/v1/embeddings");
    }
}
