//! OpenAI-compatible chat-completions summarizer adapter.
//!
//! Sends the assembled prompt to `POST /v1/chat/completions` with a bearer
//! token and parses `choices[0].message.content`. The `base_url` override makes
//! the same adapter serve Azure OpenAI, LiteLLM, vLLM, and Ollama's `/v1`
//! surface. Transient failures (429/5xx) are retried via [`crate::http`].

use serde::Deserialize;

use crate::http::RetryPolicy;
use crate::summarize::{SummarizeConfig, SummarizeError, SummarizeOpts, Summarizer, prompts};

const DEFAULT_BASE_URL: &str = "https://api.openai.com";
const PROVIDER_NAME: &str = "openai";

/// A summarizer backed by the OpenAI-compatible chat-completions API.
#[derive(Debug)]
pub struct OpenAiSummarizer {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
    max_tokens: usize,
    system_prompt: String,
}

impl OpenAiSummarizer {
    /// Build from `config`. Requires a non-empty API key.
    pub fn new(config: SummarizeConfig) -> Result<Self, SummarizeError> {
        if config.api_key.trim().is_empty() {
            return Err(SummarizeError::Config(
                "an API key is required for the openai summarizer".into(),
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

impl Summarizer for OpenAiSummarizer {
    async fn summarize(&self, text: &str, opts: &SummarizeOpts) -> Result<String, SummarizeError> {
        let system = opts.system.as_deref().unwrap_or(&self.system_prompt);
        let user = prompts::user_message(text, opts.instructions.as_deref());
        let max_tokens = opts.max_tokens.unwrap_or(self.max_tokens);
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user },
            ],
        });

        let policy = RetryPolicy::standard(3, 1000);
        let resp = super::send_checked(&policy, "OpenAI API", || {
            let mut req = self
                .client
                .post(&url)
                .bearer_auth(&self.api_key)
                .json(&body);
            for (name, value) in &self.extra_headers {
                req = req.header(name, value);
            }
            req
        })
        .await?;

        let parsed: ChatResponse = resp
            .json()
            .await
            .map_err(|e| SummarizeError::Decode(e.to_string()))?;
        extract_content(&parsed)
    }

    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

/// The content of the first choice's message.
fn extract_content(resp: &ChatResponse) -> Result<String, SummarizeError> {
    resp.choices
        .first()
        .and_then(|choice| choice.message.content.clone())
        .ok_or_else(|| {
            SummarizeError::Decode("OpenAI response contained no message content".into())
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
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": { "role": "assistant", "content": "A dense summary." },
                    "finish_reason": "stop"
                }
            ]
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(extract_content(&resp).unwrap(), "A dense summary.");
    }

    #[test]
    fn parse_response_with_no_choices_errors() {
        let resp: ChatResponse = serde_json::from_str(r#"{ "choices": [] }"#).unwrap();
        assert!(matches!(
            extract_content(&resp),
            Err(SummarizeError::Decode(_))
        ));
    }

    #[test]
    fn parse_response_with_null_content_errors() {
        let json = r#"{ "choices": [ { "message": { "role": "assistant", "content": null } } ] }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(
            extract_content(&resp),
            Err(SummarizeError::Decode(_))
        ));
    }

    // ── Construction ──────────────────────────────────────────────────

    #[test]
    fn new_without_key_errors() {
        let err = OpenAiSummarizer::new(SummarizeConfig::new("gpt-4o-mini")).unwrap_err();
        assert!(matches!(err, SummarizeError::Config(_)));
    }

    #[test]
    fn new_trims_trailing_slash_from_base_url() {
        let s = OpenAiSummarizer::new(
            SummarizeConfig::new("gpt-4o-mini")
                .api_key("k")
                .base_url("http://localhost:9/"),
        )
        .unwrap();
        assert_eq!(s.base_url, "http://localhost:9");
        assert_eq!(s.provider_name(), "openai");
        assert_eq!(s.model_name(), "gpt-4o-mini");
    }

    // ── End-to-end wire ───────────────────────────────────────────────

    #[tokio::test]
    async fn summarize_sends_correct_request_and_parses_response() {
        let (base, rx) = serve_once(
            200,
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"condensed"}}]}"#,
        );
        let s = OpenAiSummarizer::new(
            SummarizeConfig::new("gpt-4o-mini")
                .api_key("sk-test")
                .base_url(base),
        )
        .unwrap();

        let out = s
            .summarize("raw input text", &SummarizeOpts::default())
            .await
            .unwrap();
        assert_eq!(out, "condensed");

        let req = rx.recv().unwrap();
        assert!(req.head.starts_with("POST /v1/chat/completions"));
        assert!(
            req.head
                .to_ascii_lowercase()
                .contains("authorization: bearer sk-test")
        );
        assert!(req.body.contains("gpt-4o-mini"));
        assert!(req.body.contains("raw input text"));
        assert!(req.body.contains("\"role\":\"system\""));
        assert!(req.body.contains("\"role\":\"user\""));
    }

    #[tokio::test]
    async fn non_success_status_maps_to_api_error() {
        let (base, _rx) = serve_once(401, r#"{"error":{"message":"invalid api key"}}"#);
        let s = OpenAiSummarizer::new(
            SummarizeConfig::new("gpt-4o-mini")
                .api_key("k")
                .base_url(base),
        )
        .unwrap();
        match s.summarize("x", &SummarizeOpts::default()).await {
            Err(SummarizeError::Api { status, body }) => {
                assert_eq!(status, 401);
                assert!(body.contains("invalid api key"));
            }
            other => panic!("expected Api error, got {other:?}"),
        }
    }
}
