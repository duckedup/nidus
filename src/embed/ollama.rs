//! Ollama local embedding adapter (`nomic-embed-text` default), keyless.
//!
//! Ollama serves arbitrary user-installed models, so the dimension is not known
//! statically: the constructor is async and probes it with one embed call.
//! Bespoke wire shape: `POST /api/embed` with `{model, input:text}`, response
//! `{embeddings:[[..]]}`. Batches are sent sequentially.

use serde::Deserialize;

use super::{EmbedConfig, EmbedError, Embedder, post_json, resolve_base};
use crate::http::RetryPolicy;

const DEFAULT_BASE: &str = "http://localhost:11434";
const MAX_TOKENS: usize = 8_192;

#[derive(Debug)]
pub struct OllamaEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    extra_headers: Vec<(String, String)>,
    dimension: usize,
}

impl OllamaEmbedder {
    /// Construct and probe the embedding dimension with one call.
    pub async fn new(config: EmbedConfig) -> Result<Self, EmbedError> {
        let client = reqwest::Client::new();
        let base_url = resolve_base(config.base_url.as_deref(), DEFAULT_BASE);
        let model = config.model;
        let extra_headers = config.extra_headers;

        let probe = embed_one(
            &client,
            &base_url,
            &model,
            &extra_headers,
            "dimension probe",
        )
        .await?;
        let dimension = probe.len();
        if dimension == 0 {
            return Err(EmbedError::Decode(format!(
                "Ollama returned a zero-dimension embedding for model '{model}'"
            )));
        }
        Ok(Self {
            client,
            base_url,
            model,
            extra_headers,
            dimension,
        })
    }
}

async fn embed_one(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    headers: &[(String, String)],
    text: &str,
) -> Result<Vec<f32>, EmbedError> {
    let body = serde_json::json!({ "model": model, "input": text });
    let url = format!("{base_url}/api/embed");
    let policy = RetryPolicy::server_errors(3, 500);
    let resp: EmbedResponse =
        post_json(client, &policy, "Ollama API", &url, None, headers, &body).await?;
    resp.embeddings
        .into_iter()
        .next()
        .ok_or_else(|| EmbedError::Decode("Ollama returned no embedding".into()))
}

impl Embedder for OllamaEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        embed_one(
            &self.client,
            &self.base_url,
            &self.model,
            &self.extra_headers,
            text,
        )
        .await
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
    fn max_input_tokens(&self) -> usize {
        MAX_TOKENS
    }
    fn provider_name(&self) -> &str {
        "ollama"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::super::testutil::mock_once;
    use super::*;

    #[test]
    fn response_parsing() {
        let json = r#"{"model":"nomic-embed-text","embeddings":[[0.1,0.2,0.3]]}"#;
        let r: EmbedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.embeddings[0], vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn constructor_probes_dimension() {
        let server = mock_once(200, r#"{"embeddings":[[0.1,0.2,0.3,0.4]]}"#);
        let e =
            OllamaEmbedder::new(EmbedConfig::new("nomic-embed-text").base_url(&server.base_url))
                .await
                .unwrap();
        assert_eq!(e.dimension(), 4);
        assert_eq!(e.provider_name(), "ollama");
        let cap = server.captured();
        assert_eq!(cap.path, "/api/embed");
        assert!(cap.body.contains("\"input\":\"dimension probe\""));
    }

    #[tokio::test]
    async fn server_error_surfaces() {
        // 400 is a client error — server_errors policy does not retry it, so it
        // surfaces immediately as an Api error rather than looping.
        let server = mock_once(400, "bad model");
        let err = OllamaEmbedder::new(EmbedConfig::new("missing").base_url(&server.base_url))
            .await
            .unwrap_err();
        match err {
            EmbedError::Api { status, .. } => assert_eq!(status, 400),
            other => panic!("expected Api, got {other:?}"),
        }
    }
}
