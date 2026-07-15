//! Generic OpenAI-compatible embedding adapter.
//!
//! Reaches any gateway that speaks the standard `/v1/embeddings` shape (Azure
//! OpenAI, Together, Fireworks, vLLM, LiteLLM, DeepInfra, …). It **requires** an
//! explicit `base_url`; the API key is optional (some gateways are keyless).
//! Because the model dimension is not knowable for an arbitrary gateway, the
//! constructor is async and probes it with one embed call (like Ollama).

use super::{EmbedConfig, EmbedError, Embedder, openai_shaped, resolve_base};

const MAX_BATCH: usize = 128;
const MAX_TOKENS: usize = 8_192;

#[derive(Debug)]
pub struct OpenAiCompatEmbedder {
    client: reqwest::Client,
    api_key: Option<String>,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
    dimension: usize,
}

impl OpenAiCompatEmbedder {
    pub async fn new(config: EmbedConfig) -> Result<Self, EmbedError> {
        let base = config.base_url.as_deref().ok_or_else(|| {
            EmbedError::Config("openai-compat requires an explicit base_url".into())
        })?;
        if config.model.is_empty() {
            return Err(EmbedError::Config(
                "openai-compat requires an explicit model".into(),
            ));
        }
        // Accept both the host root ("https://api.together.xyz") and the
        // /v1-suffixed form the OpenAI SDK / gateway docs publish
        // ("https://api.together.xyz/v1") — we append "/v1/embeddings"
        // ourselves, so strip a trailing "/v1" to avoid "/v1/v1/embeddings".
        let base_url = resolve_base(Some(base), "");
        let base_url = base_url
            .strip_suffix("/v1")
            .map(str::to_string)
            .unwrap_or(base_url);
        let api_key = if config.api_key.is_empty() {
            None
        } else {
            Some(config.api_key)
        };

        let mut this = Self {
            client: reqwest::Client::new(),
            api_key,
            model: config.model,
            base_url,
            extra_headers: config.extra_headers,
            dimension: 0,
        };
        let probe = this
            .call(&["dimension probe"])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Decode("openai-compat returned no embedding".into()))?;
        this.dimension = probe.len();
        if this.dimension == 0 {
            return Err(EmbedError::Decode(
                "openai-compat returned a zero-dimension embedding".into(),
            ));
        }
        Ok(this)
    }

    async fn call(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let url = format!("{}/v1/embeddings", self.base_url);
        openai_shaped::embed_call(
            &self.client,
            &url,
            self.api_key.as_deref(),
            &self.extra_headers,
            &self.model,
            texts,
            None,
            "OpenAI-compatible API",
        )
        .await
    }
}

impl Embedder for OpenAiCompatEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.call(&[text])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Decode("openai-compat returned no embedding".into()))
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
        self.dimension
    }
    fn max_input_tokens(&self) -> usize {
        MAX_TOKENS
    }
    fn provider_name(&self) -> &str {
        "openai-compat"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::mock_once;
    use super::*;

    #[tokio::test]
    async fn requires_base_url() {
        let err = OpenAiCompatEmbedder::new(EmbedConfig::new("some-model").api_key("k"))
            .await
            .unwrap_err();
        assert!(matches!(err, EmbedError::Config(_)));
    }

    #[tokio::test]
    async fn requires_model() {
        let err = OpenAiCompatEmbedder::new(EmbedConfig::new("").base_url("http://x"))
            .await
            .unwrap_err();
        assert!(matches!(err, EmbedError::Config(_)));
    }

    #[tokio::test]
    async fn probes_dimension_and_keyless_ok() {
        let server = mock_once(200, r#"{"data":[{"embedding":[1.0,2.0,3.0],"index":0}]}"#);
        let e =
            OpenAiCompatEmbedder::new(EmbedConfig::new("gateway-model").base_url(&server.base_url))
                .await
                .unwrap();
        assert_eq!(e.dimension(), 3);
        assert_eq!(e.provider_name(), "openai-compat");
        assert_eq!(server.captured().path, "/v1/embeddings");
    }

    #[tokio::test]
    async fn v1_suffixed_base_url_does_not_double_up() {
        let server = mock_once(200, r#"{"data":[{"embedding":[1.0,2.0],"index":0}]}"#);
        // Gateways (Together/Fireworks/LiteLLM/…) publish a "/v1"-suffixed base
        // URL; it must still resolve to a single "/v1/embeddings".
        let base = format!("{}/v1", server.base_url);
        let e = OpenAiCompatEmbedder::new(EmbedConfig::new("gateway-model").base_url(base))
            .await
            .unwrap();
        assert_eq!(e.dimension(), 2);
        assert_eq!(server.captured().path, "/v1/embeddings");
    }
}
