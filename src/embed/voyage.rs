//! Voyage AI embedding adapter (`voyage-4` default), including the Voyage 4
//! family and its Matryoshka `output_dimension`.

use serde::Deserialize;

use super::{EmbedConfig, EmbedError, Embedder, post_json, resolve_base};
use crate::http::RetryPolicy;

const DEFAULT_BASE: &str = "https://api.voyageai.com";
const MAX_BATCH: usize = 128;
/// Context of the oldest models Voyage still serves; newer ones report more.
const FALLBACK_TOKENS: usize = 16_000;
/// The widths Matryoshka models accept, smallest first.
const MRL_DIMENSIONS: [usize; 4] = [256, 512, 1024, 2048];

#[derive(Debug)]
pub struct VoyageEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
    dimension: usize,
    output_dimension: Option<usize>,
    max_tokens: usize,
}

impl VoyageEmbedder {
    pub fn new(config: EmbedConfig) -> Result<Self, EmbedError> {
        if config.api_key.is_empty() {
            return Err(EmbedError::Config("Voyage requires an api_key".into()));
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
            max_tokens: max_tokens_for_model(&config.model),
            model: config.model,
            extra_headers: config.extra_headers,
            dimension,
            output_dimension,
        })
    }

    async fn call(&self, texts: &[&str], input_type: &str) -> Result<Vec<Vec<f32>>, EmbedError> {
        let body = build_body(&self.model, texts, input_type, self.output_dimension);
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
        self.max_tokens
    }
    fn provider_name(&self) -> &str {
        "voyage"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
}

fn build_body(
    model: &str,
    texts: &[&str],
    input_type: &str,
    output_dimension: Option<usize>,
) -> serde_json::Value {
    let mut body = serde_json::json!({ "model": model, "input": texts, "input_type": input_type });
    if let Some(d) = output_dimension {
        body["output_dimension"] = serde_json::json!(d);
    }
    body
}

/// Whether `model` accepts `output_dimension` (Matryoshka truncation): the
/// Voyage 4 family plus the 3.x models that shipped with it.
fn supports_output_dimension(model: &str) -> bool {
    matches!(
        model,
        "voyage-4-large"
            | "voyage-4"
            | "voyage-4-lite"
            | "voyage-4-nano"
            | "voyage-code-4"
            | "voyage-3-large"
            | "voyage-3.5"
            | "voyage-3.5-lite"
            | "voyage-code-3"
    )
}

fn validate_output_dimension(model: &str, d: usize) -> Result<usize, EmbedError> {
    if !supports_output_dimension(model) {
        return Err(EmbedError::Config(format!(
            "Voyage model '{model}' has a fixed output dimension; drop output_dimension"
        )));
    }
    if !MRL_DIMENSIONS.contains(&d) {
        return Err(EmbedError::Config(format!(
            "Voyage output_dimension must be one of {MRL_DIMENSIONS:?}, got {d}"
        )));
    }
    Ok(d)
}

/// Native output width, or `None` for a model this table has never heard of.
fn known_dimension(model: &str) -> Option<usize> {
    Some(match model {
        "voyage-4-large"
        | "voyage-4"
        | "voyage-4-lite"
        | "voyage-4-nano"
        | "voyage-code-4"
        | "voyage-3-large"
        | "voyage-3.5"
        | "voyage-3"
        | "voyage-code-3"
        | "voyage-multilingual-2"
        | "voyage-finance-2"
        | "voyage-law-2"
        | "voyage-large-2-instruct"
        | "voyage-2"
        | "voyage-3.5-lite" => 1024,
        "voyage-3-lite" => 512,
        "voyage-code-2" | "voyage-large-2" => 1536,
        _ => return None,
    })
}

/// Guesses 1024 for an unknown model — the width of everything Voyage has
/// shipped since voyage-3 — but says so, because a wrong guess pins the store
/// to a dimension the API will then refuse to fill.
fn dimension_for_model(model: &str) -> usize {
    known_dimension(model).unwrap_or_else(|| {
        crate::diag::diag!(
            crate::diag::Level::Warn,
            "embed",
            "unknown Voyage model, assuming 1024 dimensions",
            "model" => model,
        );
        1024
    })
}

/// Context window. Unknown models get the smaller legacy figure: under-reporting
/// only makes a caller chunk more finely, over-reporting gets a 400.
fn max_tokens_for_model(model: &str) -> usize {
    match model {
        "voyage-4-large"
        | "voyage-4"
        | "voyage-4-lite"
        | "voyage-4-nano"
        | "voyage-code-4"
        | "voyage-3-large"
        | "voyage-3.5"
        | "voyage-3.5-lite"
        | "voyage-3"
        | "voyage-3-lite"
        | "voyage-code-3"
        | "voyage-multilingual-2"
        | "voyage-finance-2" => 32_000,
        "voyage-2" => 4_000,
        _ => FALLBACK_TOKENS,
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
        let doc = build_body("voyage-3", &["hi"], "document", None);
        assert_eq!(doc["input_type"], "document");
        assert_eq!(doc["model"], "voyage-3");
        assert_eq!(doc["input"][0], "hi");
        let q = build_body("voyage-3", &["hi"], "query", None);
        assert_eq!(q["input_type"], "query");
    }

    #[test]
    fn body_omits_output_dimension_unless_asked() {
        let native = build_body("voyage-4", &["hi"], "document", None);
        assert!(native.get("output_dimension").is_none());
        let mrl = build_body("voyage-4", &["hi"], "document", Some(256));
        assert_eq!(mrl["output_dimension"], 256);
    }

    #[test]
    fn dimension_lookup() {
        assert_eq!(dimension_for_model("voyage-3"), 1024);
        // The fallback is 1024 too, so assert the arm exists, not the width.
        assert_eq!(known_dimension("voyage-code-3"), Some(1024));
        assert_eq!(dimension_for_model("voyage-3-lite"), 512);
        assert_eq!(dimension_for_model("voyage-code-2"), 1536);
        assert_eq!(dimension_for_model("voyage-large-2"), 1536);
        assert_eq!(dimension_for_model("unknown"), 1024);
    }

    #[test]
    fn voyage_4_family_is_explicit_not_fallback() {
        for m in [
            "voyage-4-large",
            "voyage-4",
            "voyage-4-lite",
            "voyage-4-nano",
            "voyage-code-4",
            "voyage-code-3",
        ] {
            assert_eq!(known_dimension(m), Some(1024), "{m}");
            assert_eq!(max_tokens_for_model(m), 32_000, "{m}");
            assert!(supports_output_dimension(m), "{m}");
        }
        assert_eq!(known_dimension("voyage-4-turbo-does-not-exist"), None);
    }

    #[test]
    fn context_window_is_per_model() {
        assert_eq!(max_tokens_for_model("voyage-3"), 32_000);
        assert_eq!(max_tokens_for_model("voyage-2"), 4_000);
        assert_eq!(max_tokens_for_model("voyage-large-2"), FALLBACK_TOKENS);
        assert_eq!(max_tokens_for_model("unknown"), FALLBACK_TOKENS);
    }

    #[test]
    fn output_dimension_is_validated() {
        assert_eq!(validate_output_dimension("voyage-4", 2048).unwrap(), 2048);
        // A fixed-width model would silently ignore the field, so refuse it.
        assert!(validate_output_dimension("voyage-3", 512).is_err());
        assert!(validate_output_dimension("voyage-4", 768).is_err());
    }

    #[test]
    fn output_dimension_drives_reported_dimension() {
        let e = VoyageEmbedder::new(
            EmbedConfig::new("voyage-4-large")
                .api_key("k")
                .output_dimension(256),
        )
        .unwrap();
        assert_eq!(e.dimension(), 256);
        assert_eq!(e.max_input_tokens(), 32_000);

        let native = VoyageEmbedder::new(EmbedConfig::new("voyage-4-large").api_key("k")).unwrap();
        assert_eq!(native.dimension(), 1024);
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
    async fn output_dimension_reaches_the_wire() {
        let server = mock_once(200, r#"{"data":[{"embedding":[1.0,2.0],"index":0}]}"#);
        let e = VoyageEmbedder::new(
            EmbedConfig::new("voyage-4")
                .api_key("k")
                .base_url(&server.base_url)
                .output_dimension(512),
        )
        .unwrap();
        e.embed("hello").await.unwrap();
        assert!(server.captured().body.contains("\"output_dimension\":512"));
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
