//! Anthropic Messages API summarizer adapter.
//!
//! Sends the assembled prompt to the Claude Messages API
//! (`POST /v1/messages`) and parses the first text block out of the response.
//! Transient failures (429/5xx, Anthropic's 529 "overloaded") are retried with
//! bounded exponential backoff via [`crate::http`].

use serde::Deserialize;

use crate::http::RetryPolicy;
use crate::summarize::{SummarizeConfig, SummarizeError, SummarizeOpts, Summarizer, prompts};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const PROVIDER_NAME: &str = "anthropic";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// A summarizer backed by the Anthropic Messages API.
#[derive(Debug)]
pub struct AnthropicSummarizer {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
    max_tokens: usize,
    system_prompt: String,
}

impl AnthropicSummarizer {
    /// Build from `config`. Requires a non-empty API key.
    pub fn new(config: SummarizeConfig) -> Result<Self, SummarizeError> {
        if config.api_key.trim().is_empty() {
            return Err(SummarizeError::Config(
                "an API key is required for the anthropic summarizer".into(),
            ));
        }
        let base_url = config
            .base_url
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
            .trim_end_matches('/')
            .to_string();
        let system_prompt = config
            .system_prompt
            .unwrap_or_else(|| prompts::DEFAULT_SYSTEM_PROMPT.to_string());
        Ok(Self {
            client: reqwest::Client::new(),
            api_key: config.api_key,
            model: config.model,
            base_url,
            extra_headers: config.extra_headers,
            max_tokens: config.max_tokens,
            system_prompt,
        })
    }
}

impl Summarizer for AnthropicSummarizer {
    async fn summarize(&self, text: &str, opts: &SummarizeOpts) -> Result<String, SummarizeError> {
        let system = opts.system.as_deref().unwrap_or(&self.system_prompt);
        let user = prompts::user_message(text, opts.instructions.as_deref());
        let max_tokens = opts.max_tokens.unwrap_or(self.max_tokens);
        let url = format!("{}/v1/messages", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{ "role": "user", "content": user }],
        });

        let policy = RetryPolicy::standard(3, 1000);
        let resp = super::send_checked(&policy, "Anthropic API", || {
            let mut req = self
                .client
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .json(&body);
            for (name, value) in &self.extra_headers {
                req = req.header(name, value);
            }
            req
        })
        .await?;

        let parsed: MessagesResponse = resp
            .json()
            .await
            .map_err(|e| SummarizeError::Decode(e.to_string()))?;
        extract_text(&parsed)
    }

    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[derive(Debug, Deserialize)]
struct MessagesResponse {
    #[serde(default)]
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    /// Present only on `text` blocks; `None` for `tool_use` etc., which we skip.
    text: Option<String>,
}

/// The first block that carries text. Non-text blocks (which have no `text`
/// field / a null `text`) are skipped.
fn extract_text(resp: &MessagesResponse) -> Result<String, SummarizeError> {
    resp.content
        .iter()
        .find_map(|block| block.text.clone())
        .ok_or_else(|| {
            SummarizeError::Decode("Anthropic response contained no text content".into())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::summarize::test_server::serve_once;

    // ── Response parsing ──────────────────────────────────────────────

    #[test]
    fn parse_successful_response() {
        let json = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                { "type": "text", "text": "Dense retrieval-friendly summary." }
            ],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 100, "output_tokens": 20 }
        }"#;
        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            extract_text(&resp).unwrap(),
            "Dense retrieval-friendly summary."
        );
    }

    #[test]
    fn parse_response_with_no_text_errors() {
        let resp: MessagesResponse = serde_json::from_str(r#"{ "content": [] }"#).unwrap();
        assert!(matches!(
            extract_text(&resp),
            Err(SummarizeError::Decode(_))
        ));
    }

    #[test]
    fn parse_response_skips_non_text_blocks() {
        let json = r#"{
            "content": [
                { "type": "tool_use", "text": null },
                { "type": "text", "text": "The actual summary." }
            ]
        }"#;
        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(extract_text(&resp).unwrap(), "The actual summary.");
    }

    // ── Construction ──────────────────────────────────────────────────

    #[test]
    fn new_without_key_errors() {
        let err = AnthropicSummarizer::new(SummarizeConfig::new("claude")).unwrap_err();
        assert!(matches!(err, SummarizeError::Config(_)));
    }

    #[test]
    fn new_trims_trailing_slash_from_base_url() {
        let s = AnthropicSummarizer::new(
            SummarizeConfig::new("claude")
                .api_key("k")
                .base_url("http://localhost:9/"),
        )
        .unwrap();
        assert_eq!(s.base_url, "http://localhost:9");
        assert_eq!(s.provider_name(), "anthropic");
        assert_eq!(s.model_name(), "claude");
    }

    // ── End-to-end wire ───────────────────────────────────────────────

    #[tokio::test]
    async fn summarize_sends_correct_request_and_parses_response() {
        let (base, rx) = serve_once(
            200,
            r#"{"content":[{"type":"text","text":"a dense summary"}]}"#,
        );
        let s = AnthropicSummarizer::new(
            SummarizeConfig::new("claude-haiku-4-5-20251001")
                .api_key("secret-key")
                .base_url(base),
        )
        .unwrap();

        let out = s
            .summarize(
                "some source text to condense",
                &SummarizeOpts {
                    instructions: Some("Summarize tersely.".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(out, "a dense summary");

        let req = rx.recv().unwrap();
        assert!(req.head.starts_with("POST /v1/messages"));
        assert!(
            req.head
                .to_ascii_lowercase()
                .contains("x-api-key: secret-key")
        );
        assert!(
            req.head
                .to_ascii_lowercase()
                .contains("anthropic-version: 2023-06-01")
        );
        // Body shape: model, system, and the assembled user message.
        assert!(req.body.contains("claude-haiku-4-5-20251001"));
        assert!(req.body.contains("some source text to condense"));
        assert!(req.body.contains("Summarize tersely."));
        assert!(req.body.contains("\"role\":\"user\""));
    }

    #[tokio::test]
    async fn non_success_status_maps_to_api_error() {
        // 400 is non-retryable, so this returns immediately (no backoff wait).
        let (base, _rx) = serve_once(400, r#"{"error":{"message":"bad request"}}"#);
        let s =
            AnthropicSummarizer::new(SummarizeConfig::new("claude").api_key("k").base_url(base))
                .unwrap();
        match s.summarize("x", &SummarizeOpts::default()).await {
            Err(SummarizeError::Api { status, body }) => {
                assert_eq!(status, 400);
                assert!(body.contains("bad request"));
            }
            other => panic!("expected Api error, got {other:?}"),
        }
    }
}
