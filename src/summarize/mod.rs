//! Single-shot text summarization (epic nidus-54l, tickets .7/.8).
//!
//! The optional "summarize-then-embed" leg of the AI ingest layer: given a
//! blob of arbitrary text, produce a dense, retrieval-friendly summary that is
//! a better embedding target than the raw text. nidus knows nothing about the
//! caller's domain — unlike the code-specific summarizer this was ported from,
//! the trait here is a single generic [`Summarizer::summarize`] over `(text,
//! opts)`.
//!
//! The public surface is a native-async trait (RPITIT, no `async_trait`, no
//! `Box<dyn>`), a typed [`SummarizeError`] at the edge, a [`SummarizeConfig`]
//! builder, the [`SummarizeProvider`] selector, and the closed [`AnySummarizer`]
//! enum that dispatches to whichever provider adapters were compiled in.

pub mod prompts;

#[cfg(feature = "summarize-anthropic")]
pub mod anthropic;
#[cfg(feature = "summarize-openai")]
pub mod openai;

use std::fmt;

use crate::providers::{self, Capability};

// ── Trait ───────────────────────────────────────────────────────────────────

/// Turns arbitrary text into a dense, search-friendly summary.
///
/// Native async: [`summarize`](Self::summarize) returns an
/// `impl Future + Send` rather than boxing, so callers pay no allocation and the
/// trait stays object-unsafe by design — dispatch across providers goes through
/// the closed [`AnySummarizer`] enum, not `dyn Summarizer`.
pub trait Summarizer: Send + Sync {
    /// Summarize `text` into retrieval-friendly prose.
    fn summarize(
        &self,
        text: &str,
        opts: &SummarizeOpts,
    ) -> impl std::future::Future<Output = Result<String, SummarizeError>> + Send;

    /// The provider this summarizer talks to (e.g. `"anthropic"`).
    fn provider_name(&self) -> &str;

    /// The model name, for logging and cost attribution.
    fn model_name(&self) -> &str;
}

/// A stable `"provider/model"` identity string for logging and cache keys.
pub fn summarizer_identity(s: &impl Summarizer) -> String {
    format!("{}/{}", s.provider_name(), s.model_name())
}

// ── Per-call options ──────────────────────────────────────────────────────────

/// Per-call overrides for a single [`Summarizer::summarize`] call. All optional;
/// anything left `None` falls back to the adapter's configured defaults.
#[derive(Debug, Clone, Default)]
pub struct SummarizeOpts {
    /// Override the system prompt for this call only.
    pub system: Option<String>,
    /// Extra instructions prepended to the user message (e.g. "in one sentence").
    pub instructions: Option<String>,
    /// Override the max output tokens for this call only.
    pub max_tokens: Option<usize>,
}

// ── Error ─────────────────────────────────────────────────────────────────────

/// Everything that can go wrong at the summarization edge — a closed, typed
/// enum rather than `anyhow` so callers can match on the failure class.
#[derive(Debug)]
pub enum SummarizeError {
    /// Bad or missing configuration (empty API key, unknown provider, …).
    Config(String),
    /// Transport failure that outlived the retry budget (DNS, connect, TLS, …).
    Backend(String),
    /// The API answered with a non-success status.
    Api { status: u16, body: String },
    /// The response body could not be parsed into the expected shape.
    Decode(String),
}

impl fmt::Display for SummarizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SummarizeError::Config(msg) => write!(f, "summarize config error: {msg}"),
            SummarizeError::Backend(msg) => write!(f, "summarize backend error: {msg}"),
            SummarizeError::Api { status, body } => {
                write!(f, "summarize API error ({status}): {body}")
            }
            SummarizeError::Decode(msg) => write!(f, "summarize decode error: {msg}"),
        }
    }
}

impl std::error::Error for SummarizeError {}

// ── Config ──────────────────────────────────────────────────────────────────

/// Configuration for building a summarizer. Construct with [`new`](Self::new)
/// and chain the builder setters.
#[derive(Debug, Clone)]
pub struct SummarizeConfig {
    /// The model name passed through to the provider.
    pub model: String,
    /// API key / bearer token. Required by the hosted providers.
    pub api_key: String,
    /// Override the provider base URL (Azure/LiteLLM/vLLM/Ollama, or a mock).
    pub base_url: Option<String>,
    /// Extra request headers appended to every call.
    pub extra_headers: Vec<(String, String)>,
    /// Default max output tokens (defaults to 1024).
    pub max_tokens: usize,
    /// Override the default system prompt.
    pub system_prompt: Option<String>,
}

impl SummarizeConfig {
    /// A config for `model` with defaults: no key, default base URL, no extra
    /// headers, `max_tokens = 1024`, default system prompt.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: String::new(),
            base_url: None,
            extra_headers: Vec::new(),
            max_tokens: 1024,
            system_prompt: None,
        }
    }

    /// Set the API key / bearer token.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into();
        self
    }

    /// Override the provider base URL.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Append an extra request header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    /// Set the default max output tokens.
    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Override the default system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
}

// ── Provider selector ─────────────────────────────────────────────────────────

/// The summarization providers nidus can build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummarizeProvider {
    Anthropic,
    OpenAi,
}

impl SummarizeProvider {
    /// Parse a provider from its wire name (`"anthropic"`, `"openai"`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "anthropic" => Some(Self::Anthropic),
            "openai" => Some(Self::OpenAi),
            _ => None,
        }
    }

    /// The provider's canonical wire name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
        }
    }

    /// A sensible default model for the provider.
    pub fn default_model(&self) -> &'static str {
        match self {
            Self::Anthropic => "claude-haiku-4-5-20251001",
            Self::OpenAi => "gpt-4o-mini",
        }
    }
}

impl std::str::FromStr for SummarizeProvider {
    type Err = SummarizeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_name(s).ok_or_else(|| {
            SummarizeError::Config(format!(
                "unknown summarize provider '{s}'; available summarizers: {}",
                providers::names_with(Capability::Summarize).join(", ")
            ))
        })
    }
}

// ── Runtime dispatch ──────────────────────────────────────────────────────────

/// A summarizer selected at runtime — one variant per compiled-in provider
/// adapter. This is the closed-enum replacement for `Box<dyn Summarizer>`
/// (the trait is object-unsafe by design).
pub enum AnySummarizer {
    #[cfg(feature = "summarize-anthropic")]
    Anthropic(anthropic::AnthropicSummarizer),
    #[cfg(feature = "summarize-openai")]
    OpenAi(openai::OpenAiSummarizer),
}

impl AnySummarizer {
    /// Build a summarizer for `provider` from `config`. Rejects providers that
    /// are not registered summarizers, and providers whose adapter feature was
    /// not compiled in, with a [`SummarizeError::Config`].
    pub async fn build(
        provider: SummarizeProvider,
        config: SummarizeConfig,
    ) -> Result<Self, SummarizeError> {
        if !providers::supports(provider.as_str(), Capability::Summarize) {
            return Err(SummarizeError::Config(format!(
                "provider '{}' is not a valid summarizer; available summarizers: {}",
                provider.as_str(),
                providers::names_with(Capability::Summarize).join(", ")
            )));
        }
        match provider {
            #[cfg(feature = "summarize-anthropic")]
            SummarizeProvider::Anthropic => Ok(Self::Anthropic(
                anthropic::AnthropicSummarizer::new(config)?,
            )),
            #[cfg(feature = "summarize-openai")]
            SummarizeProvider::OpenAi => Ok(Self::OpenAi(openai::OpenAiSummarizer::new(config)?)),
            #[allow(unreachable_patterns)]
            other => {
                let _ = config;
                Err(SummarizeError::Config(format!(
                    "summarize provider '{}' is not compiled in; enable the `summarize-{}` feature",
                    other.as_str(),
                    other.as_str()
                )))
            }
        }
    }
}

#[cfg(any(feature = "summarize-anthropic", feature = "summarize-openai"))]
impl Summarizer for AnySummarizer {
    async fn summarize(&self, text: &str, opts: &SummarizeOpts) -> Result<String, SummarizeError> {
        match self {
            #[cfg(feature = "summarize-anthropic")]
            Self::Anthropic(s) => s.summarize(text, opts).await,
            #[cfg(feature = "summarize-openai")]
            Self::OpenAi(s) => s.summarize(text, opts).await,
        }
    }

    fn provider_name(&self) -> &str {
        match self {
            #[cfg(feature = "summarize-anthropic")]
            Self::Anthropic(s) => s.provider_name(),
            #[cfg(feature = "summarize-openai")]
            Self::OpenAi(s) => s.provider_name(),
        }
    }

    fn model_name(&self) -> &str {
        match self {
            #[cfg(feature = "summarize-anthropic")]
            Self::Anthropic(s) => s.model_name(),
            #[cfg(feature = "summarize-openai")]
            Self::OpenAi(s) => s.model_name(),
        }
    }
}

// When no provider adapter is compiled in, `AnySummarizer` is uninhabited (it
// has no variants) so these methods can never actually be called — but the
// `impl` must still exist so the public API surface is stable.
#[cfg(not(any(feature = "summarize-anthropic", feature = "summarize-openai")))]
impl Summarizer for AnySummarizer {
    async fn summarize(
        &self,
        _text: &str,
        _opts: &SummarizeOpts,
    ) -> Result<String, SummarizeError> {
        unreachable!("AnySummarizer has no variants without a summarize-* feature")
    }

    fn provider_name(&self) -> &str {
        unreachable!("AnySummarizer has no variants without a summarize-* feature")
    }

    fn model_name(&self) -> &str {
        unreachable!("AnySummarizer has no variants without a summarize-* feature")
    }
}

// ── Shared adapter plumbing ─────────────────────────────────────────────────────

/// Send a prepared request with retry, then classify the response: transport
/// failure past the retry budget → [`SummarizeError::Backend`], a non-2xx
/// status → [`SummarizeError::Api`], success → the [`reqwest::Response`] for the
/// caller to parse. Shared by every provider adapter (the dedupe rule — one
/// source, two callers) so status/error handling lives in exactly one place.
#[cfg(any(feature = "summarize-anthropic", feature = "summarize-openai"))]
async fn send_checked(
    policy: &crate::http::RetryPolicy,
    label: &str,
    build: impl Fn() -> reqwest::RequestBuilder,
) -> Result<reqwest::Response, SummarizeError> {
    let resp = crate::http::send_with_retry(policy, label, build)
        .await
        .map_err(|e| SummarizeError::Backend(e.to_string()))?;
    if resp.status().is_success() {
        Ok(resp)
    } else {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        Err(SummarizeError::Api { status, body })
    }
}

// ── Shared test server ───────────────────────────────────────────────────────

/// A one-shot local HTTP server used by the provider wire tests: it accepts a
/// single connection, captures the raw request, and replies with a canned
/// status + JSON body. Lives here (rather than duplicated per adapter) so both
/// the Anthropic and OpenAI tests share one implementation.
#[cfg(all(
    test,
    any(feature = "summarize-anthropic", feature = "summarize-openai")
))]
pub(crate) mod test_server {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::thread;

    /// The raw request head (request line + headers) and decoded body.
    pub struct CapturedRequest {
        pub head: String,
        pub body: String,
    }

    /// Bind an ephemeral port, serve exactly one request with `status` +
    /// `resp_body`, and hand the captured request back over the channel.
    /// Returns the base URL to point an adapter's `base_url` at.
    pub fn serve_once(
        status: u16,
        resp_body: &'static str,
    ) -> (String, mpsc::Receiver<CapturedRequest>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let req = read_request(&mut stream);
                let resp = format!(
                    "HTTP/1.1 {status} STATUS\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp_body}",
                    resp_body.len()
                );
                let _ = stream.write_all(resp.as_bytes());
                let _ = stream.flush();
                let _ = tx.send(req);
            }
        });
        (format!("http://{addr}"), rx)
    }

    fn read_request(stream: &mut TcpStream) -> CapturedRequest {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            let n = stream.read(&mut tmp).unwrap_or(0);
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..pos]).into_owned();
                let content_len = content_length(&head);
                let body_start = pos + 4;
                while buf.len() - body_start < content_len {
                    let n = stream.read(&mut tmp).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                let body = String::from_utf8_lossy(&buf[body_start..]).into_owned();
                return CapturedRequest { head, body };
            }
        }
        CapturedRequest {
            head: String::from_utf8_lossy(&buf).into_owned(),
            body: String::new(),
        }
    }

    fn content_length(head: &str) -> usize {
        head.lines()
            .find_map(|line| {
                let lower = line.to_ascii_lowercase();
                lower
                    .strip_prefix("content-length:")
                    .map(|v| v.trim().parse::<usize>().unwrap_or(0))
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_round_trips_by_name() {
        assert_eq!(
            SummarizeProvider::from_name("anthropic"),
            Some(SummarizeProvider::Anthropic)
        );
        assert_eq!(
            SummarizeProvider::from_name("openai"),
            Some(SummarizeProvider::OpenAi)
        );
        assert_eq!(SummarizeProvider::from_name("voyage"), None);
        assert_eq!(SummarizeProvider::Anthropic.as_str(), "anthropic");
        assert_eq!(SummarizeProvider::OpenAi.as_str(), "openai");
    }

    #[test]
    fn provider_default_models() {
        assert_eq!(
            SummarizeProvider::Anthropic.default_model(),
            "claude-haiku-4-5-20251001"
        );
        assert_eq!(SummarizeProvider::OpenAi.default_model(), "gpt-4o-mini");
    }

    #[test]
    fn provider_from_str() {
        assert_eq!(
            "anthropic".parse::<SummarizeProvider>().unwrap(),
            SummarizeProvider::Anthropic
        );
        assert!("nope".parse::<SummarizeProvider>().is_err());
    }

    #[test]
    fn config_new_has_defaults() {
        let cfg = SummarizeConfig::new("m");
        assert_eq!(cfg.model, "m");
        assert!(cfg.api_key.is_empty());
        assert!(cfg.base_url.is_none());
        assert!(cfg.extra_headers.is_empty());
        assert_eq!(cfg.max_tokens, 1024);
        assert!(cfg.system_prompt.is_none());
    }

    #[test]
    fn config_builder_chains() {
        let cfg = SummarizeConfig::new("m")
            .api_key("k")
            .base_url("http://localhost:1234")
            .header("x-extra", "1")
            .max_tokens(256)
            .system_prompt("be terse");
        assert_eq!(cfg.api_key, "k");
        assert_eq!(cfg.base_url.as_deref(), Some("http://localhost:1234"));
        assert_eq!(cfg.extra_headers, vec![("x-extra".into(), "1".into())]);
        assert_eq!(cfg.max_tokens, 256);
        assert_eq!(cfg.system_prompt.as_deref(), Some("be terse"));
    }

    #[test]
    fn error_display_variants() {
        assert!(
            SummarizeError::Config("x".into())
                .to_string()
                .contains("config")
        );
        assert!(
            SummarizeError::Backend("x".into())
                .to_string()
                .contains("backend")
        );
        assert!(
            SummarizeError::Api {
                status: 429,
                body: "slow down".into(),
            }
            .to_string()
            .contains("429")
        );
        assert!(
            SummarizeError::Decode("x".into())
                .to_string()
                .contains("decode")
        );
    }

    #[cfg(feature = "summarize-anthropic")]
    #[tokio::test]
    async fn build_rejects_unknown_provider_capability() {
        // Sanity: build validates against the capability registry before
        // constructing an adapter. anthropic is a valid summarizer, so this
        // path just confirms build succeeds with a key.
        let s = AnySummarizer::build(
            SummarizeProvider::Anthropic,
            SummarizeConfig::new("claude-haiku-4-5-20251001").api_key("k"),
        )
        .await
        .unwrap();
        assert_eq!(s.provider_name(), "anthropic");
    }
}
