//! OpenAI embedding adapter (`text-embedding-3-small` default).

use super::{EmbedConfig, EmbedError, Embedder, openai_shaped, resolve_base};

const DEFAULT_BASE: &str = "https://api.openai.com";
const MAX_BATCH: usize = 2048;
const MAX_TOKENS: usize = 8_191;

#[derive(Debug)]
pub struct OpenAiEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
    dimension: usize,
    output_dimension: Option<usize>,
}

impl OpenAiEmbedder {
    pub fn new(config: EmbedConfig) -> Result<Self, EmbedError> {
        if config.api_key.is_empty() {
            return Err(EmbedError::Config("OpenAI requires an api_key".into()));
        }
        let output_dimension = match config.output_dimension {
            Some(d) => Some(validate_output_dimension(&config.model, d)?),
            None => None,
        };
        let dimension = output_dimension.unwrap_or_else(|| dimension_for_model(&config.model));
        Ok(Self {
            client: reqwest::Client::new(),
            api_key: config.api_key,
            base_url: resolve_base(config.base_url.as_deref(), DEFAULT_BASE),
            model: config.model,
            extra_headers: config.extra_headers,
            dimension,
            output_dimension,
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
            self.output_dimension,
            "OpenAI API",
        )
        .await
    }
}

impl Embedder for OpenAiEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.call(&[text])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Decode("OpenAI returned no embedding".into()))
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
        "openai"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
}

fn dimension_for_model(model: &str) -> usize {
    match model {
        "text-embedding-3-large" => 3072,
        "text-embedding-3-small" | "text-embedding-ada-002" => 1536,
        _ => 1536,
    }
}

/// Whether `model` honours the `dimensions` request field (Matryoshka
/// truncation). Only the `text-embedding-3` pair does; ada-002 ignores it.
fn supports_output_dimension(model: &str) -> bool {
    matches!(model, "text-embedding-3-small" | "text-embedding-3-large")
}

/// OpenAI truncates to any width up to the model's native one, so the bound is a
/// range rather than Voyage's fixed set.
fn validate_output_dimension(model: &str, d: usize) -> Result<usize, EmbedError> {
    if !supports_output_dimension(model) {
        return Err(EmbedError::Config(format!(
            "OpenAI model '{model}' has a fixed output dimension; drop output_dimension"
        )));
    }
    let native = dimension_for_model(model);
    if d == 0 || d > native {
        return Err(EmbedError::Config(format!(
            "OpenAI output_dimension must be between 1 and {native} for '{model}', got {d}"
        )));
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::super::testutil::mock_once;
    use super::*;

    #[test]
    fn dimension_lookup() {
        assert_eq!(dimension_for_model("text-embedding-3-large"), 3072);
        assert_eq!(dimension_for_model("text-embedding-3-small"), 1536);
        assert_eq!(dimension_for_model("text-embedding-ada-002"), 1536);
        assert_eq!(dimension_for_model("unknown"), 1536);
    }

    #[test]
    fn constructor_requires_key() {
        assert!(OpenAiEmbedder::new(EmbedConfig::new("text-embedding-3-small")).is_err());
    }

    #[test]
    fn output_dimension_is_validated() {
        assert_eq!(
            validate_output_dimension("text-embedding-3-large", 3072).unwrap(),
            3072
        );
        assert_eq!(
            validate_output_dimension("text-embedding-3-small", 256).unwrap(),
            256
        );
        // Past the model's native width the API cannot fill the vector.
        assert!(validate_output_dimension("text-embedding-3-small", 3072).is_err());
        assert!(validate_output_dimension("text-embedding-3-small", 0).is_err());
        // ada-002 would silently ignore the field, so refuse it.
        assert!(validate_output_dimension("text-embedding-ada-002", 256).is_err());
    }

    #[test]
    fn output_dimension_drives_reported_dimension() {
        let e = OpenAiEmbedder::new(
            EmbedConfig::new("text-embedding-3-large")
                .api_key("k")
                .output_dimension(256),
        )
        .unwrap();
        assert_eq!(e.dimension(), 256);

        let native =
            OpenAiEmbedder::new(EmbedConfig::new("text-embedding-3-large").api_key("k")).unwrap();
        assert_eq!(native.dimension(), 3072);
    }

    #[tokio::test]
    async fn output_dimension_reaches_the_wire() {
        let server = mock_once(200, r#"{"data":[{"embedding":[1.0,2.0],"index":0}]}"#);
        let e = OpenAiEmbedder::new(
            EmbedConfig::new("text-embedding-3-small")
                .api_key("k")
                .base_url(&server.base_url)
                .output_dimension(512),
        )
        .unwrap();
        e.embed("hello").await.unwrap();
        assert!(server.captured().body.contains("\"dimensions\":512"));
    }

    #[tokio::test]
    async fn body_omits_dimensions_unless_asked() {
        let server = mock_once(200, r#"{"data":[{"embedding":[1.0],"index":0}]}"#);
        let e = OpenAiEmbedder::new(
            EmbedConfig::new("text-embedding-3-small")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        e.embed("hello").await.unwrap();
        assert!(!server.captured().body.contains("dimensions"));
    }

    #[tokio::test]
    async fn batch_reorders_by_index() {
        // Returned out of order (index 1 first) — must come back sorted.
        let server = mock_once(
            200,
            r#"{"data":[{"embedding":[3.0],"index":1},{"embedding":[1.0],"index":0}]}"#,
        );
        let e = OpenAiEmbedder::new(
            EmbedConfig::new("text-embedding-3-small")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        let out = e.embed_batch(&["a", "b"]).await.unwrap();
        assert_eq!(out, vec![vec![1.0], vec![3.0]]);
        let cap = server.captured();
        assert_eq!(cap.path, "/v1/embeddings");
        assert!(cap.body.contains("\"model\":\"text-embedding-3-small\""));
    }

    #[tokio::test]
    async fn non_2xx_maps_to_api_error() {
        let server = mock_once(401, "nope");
        let e = OpenAiEmbedder::new(
            EmbedConfig::new("text-embedding-3-small")
                .api_key("k")
                .base_url(&server.base_url),
        )
        .unwrap();
        match e.embed("x").await.unwrap_err() {
            EmbedError::Api { status, .. } => assert_eq!(status, 401),
            other => panic!("expected Api, got {other:?}"),
        }
    }
}
