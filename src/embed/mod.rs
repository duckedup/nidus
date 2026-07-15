//! Embedding abstraction + provider adapters (epic nidus-54l, tickets .1/.2/.3).
//!
//! The [`Embedder`] trait turns natural-language text into a dense `Vec<f32>`
//! ready to hand to the sync store. Concrete adapters live in the sibling
//! provider files, each gated behind its own `embed-<name>` feature; the
//! runtime-selectable [`AnyEmbedder`] enum wraps whichever ones were compiled
//! in and is what callers hold.
//!
//! The trait uses **native `async` methods** (return-position `impl Future`),
//! so it is intentionally **not** object-safe — there is no `Box<dyn Embedder>`.
//! Dispatch across providers goes through the closed [`AnyEmbedder`] enum.
//!
//! ## Providers
//!
//! - Voyage, OpenAI, Ollama, Cohere, Gemini, Mistral, Jina embedders, plus the
//!   generic `openai-compat` catch-all (Azure/Together/Fireworks/vLLM/LiteLLM/…).
//! - OpenAI, Mistral, Jina, and openai-compat all speak the same
//!   `/v1/embeddings` wire shape, so they share one request/parse helper
//!   ([`openai_shaped`]). Voyage, Cohere, and Gemini are bespoke.

use std::fmt;

// ── Provider adapter modules (one per `embed-<name>` feature) ────────────────
// Public so the concrete types can appear in the public `AnyEmbedder` variants
// without tripping the private-interface lint.
#[cfg(feature = "embed-cohere")]
pub mod cohere;
#[cfg(feature = "embed-gemini")]
pub mod gemini;
#[cfg(feature = "embed-jina")]
pub mod jina;
#[cfg(feature = "embed-mistral")]
pub mod mistral;
#[cfg(feature = "embed-ollama")]
pub mod ollama;
#[cfg(feature = "embed-openai")]
pub mod openai;
#[cfg(feature = "embed-openai-compat")]
pub mod openai_compat;
#[cfg(feature = "embed-voyage")]
pub mod voyage;

// ── Errors ───────────────────────────────────────────────────────────────────

/// A typed error at the embedding edge (the public surface uses this enum, not
/// `anyhow`).
#[derive(Debug)]
pub enum EmbedError {
    /// Bad/missing configuration (no API key, missing base URL, unknown or
    /// not-compiled-in provider).
    Config(String),
    /// A transport-level failure that survived the retry budget.
    Backend(String),
    /// The API returned a non-2xx status. `body` is the raw response text.
    Api { status: u16, body: String },
    /// A 2xx response whose body could not be parsed into the expected shape.
    Decode(String),
}

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbedError::Config(m) => write!(f, "embed config error: {m}"),
            EmbedError::Backend(m) => write!(f, "embed backend error: {m}"),
            EmbedError::Api { status, body } => {
                write!(f, "embed API error ({status}): {body}")
            }
            EmbedError::Decode(m) => write!(f, "embed decode error: {m}"),
        }
    }
}

impl std::error::Error for EmbedError {}

// ── Config ─────────────────────────────────────────────────────────────────

/// Everything an [`AnyEmbedder`] needs to reach a provider. Built fluently:
///
/// ```ignore
/// let cfg = EmbedConfig::new("text-embedding-3-small")
///     .api_key(std::env::var("OPENAI_API_KEY").unwrap())
///     .base_url("https://api.openai.com");
/// ```
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    /// Model name. Empty means "use the provider default" (see
    /// [`EmbedProvider::default_model`]); openai-compat has no default and
    /// requires an explicit model.
    pub model: String,
    /// Bearer/API key. Keyless providers (Ollama, some openai-compat gateways)
    /// leave this empty.
    pub api_key: String,
    /// Override the provider's default base URL. Required for openai-compat.
    pub base_url: Option<String>,
    /// Extra request headers applied to every call (e.g. gateway auth).
    pub extra_headers: Vec<(String, String)>,
}

impl EmbedConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: String::new(),
            base_url: None,
            extra_headers: Vec::new(),
        }
    }

    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = k.into();
        self
    }

    pub fn base_url(mut self, u: impl Into<String>) -> Self {
        self.base_url = Some(u.into());
        self
    }

    pub fn header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.extra_headers.push((k.into(), v.into()));
        self
    }
}

// ── Provider enum ────────────────────────────────────────────────────────────

/// Which embedding backend to build. The `as_str` values match the
/// [`crate::providers`] registry names exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedProvider {
    Voyage,
    OpenAi,
    Ollama,
    Cohere,
    Gemini,
    Mistral,
    Jina,
    OpenAiCompat,
}

impl EmbedProvider {
    /// Parse a registry name (`voyage`, `openai`, …, `openai-compat`).
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "voyage" => EmbedProvider::Voyage,
            "openai" => EmbedProvider::OpenAi,
            "ollama" => EmbedProvider::Ollama,
            "cohere" => EmbedProvider::Cohere,
            "gemini" => EmbedProvider::Gemini,
            "mistral" => EmbedProvider::Mistral,
            "jina" => EmbedProvider::Jina,
            "openai-compat" => EmbedProvider::OpenAiCompat,
            _ => return None,
        })
    }

    /// The registry name — matches [`crate::providers`].
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbedProvider::Voyage => "voyage",
            EmbedProvider::OpenAi => "openai",
            EmbedProvider::Ollama => "ollama",
            EmbedProvider::Cohere => "cohere",
            EmbedProvider::Gemini => "gemini",
            EmbedProvider::Mistral => "mistral",
            EmbedProvider::Jina => "jina",
            EmbedProvider::OpenAiCompat => "openai-compat",
        }
    }

    /// The model used when [`EmbedConfig::model`] is empty. openai-compat has
    /// none (`""`) — the caller must supply a model.
    pub fn default_model(&self) -> &'static str {
        match self {
            EmbedProvider::Voyage => "voyage-3",
            EmbedProvider::OpenAi => "text-embedding-3-small",
            EmbedProvider::Ollama => "nomic-embed-text",
            EmbedProvider::Cohere => "embed-english-v3.0",
            EmbedProvider::Gemini => "text-embedding-004",
            EmbedProvider::Mistral => "mistral-embed",
            EmbedProvider::Jina => "jina-embeddings-v3",
            EmbedProvider::OpenAiCompat => "",
        }
    }
}

impl std::str::FromStr for EmbedProvider {
    type Err = EmbedError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        EmbedProvider::from_name(s).ok_or_else(|| {
            EmbedError::Config(format!(
                "unknown embed provider '{s}'; available: {}",
                crate::providers::names_with(crate::providers::Capability::Embed).join(", ")
            ))
        })
    }
}

// ── The trait ────────────────────────────────────────────────────────────────

/// Turns text into a dense embedding. Native async (RPITIT) — **not**
/// object-safe by design; dispatch via [`AnyEmbedder`], never `Box<dyn>`.
pub trait Embedder: Send + Sync {
    /// Embed a document (the indexed side).
    fn embed(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Vec<f32>, EmbedError>> + Send;

    /// Embed a batch of documents. Providers chunk to their per-request maximum
    /// internally.
    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> impl std::future::Future<Output = Result<Vec<Vec<f32>>, EmbedError>> + Send;

    /// Embed a search query. Providers that distinguish document vs. query
    /// (Voyage, Cohere, Gemini, Jina) tag the request accordingly; the default
    /// delegates to [`embed`](Self::embed).
    fn embed_query(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Vec<f32>, EmbedError>> + Send {
        async move { self.embed(text).await }
    }

    fn dimension(&self) -> usize;
    fn max_input_tokens(&self) -> usize;
    fn provider_name(&self) -> &str;
    fn model_name(&self) -> &str;
}

/// `"provider/model"` — a stable identity string for an embedder (used to tag
/// stored vectors with what produced them).
pub fn embedder_identity(e: &impl Embedder) -> String {
    format!("{}/{}", e.provider_name(), e.model_name())
}

// ── AnyEmbedder: the closed, runtime-selectable enum ─────────────────────────

/// One variant per compiled-in provider. This is the concrete type callers
/// hold; it implements [`Embedder`] by delegating to the wrapped adapter.
///
/// With **zero** provider features enabled the enum has no variants (it is
/// uninhabited) and [`build`](AnyEmbedder::build) always returns a
/// [`EmbedError::Config`] telling you which feature to turn on.
pub enum AnyEmbedder {
    #[cfg(feature = "embed-voyage")]
    Voyage(voyage::VoyageEmbedder),
    #[cfg(feature = "embed-openai")]
    OpenAi(openai::OpenAiEmbedder),
    #[cfg(feature = "embed-ollama")]
    Ollama(ollama::OllamaEmbedder),
    #[cfg(feature = "embed-cohere")]
    Cohere(cohere::CohereEmbedder),
    #[cfg(feature = "embed-gemini")]
    Gemini(gemini::GeminiEmbedder),
    #[cfg(feature = "embed-mistral")]
    Mistral(mistral::MistralEmbedder),
    #[cfg(feature = "embed-jina")]
    Jina(jina::JinaEmbedder),
    #[cfg(feature = "embed-openai-compat")]
    OpenAiCompat(openai_compat::OpenAiCompatEmbedder),
}

/// The error returned when a provider was requested but its feature is off.
/// (Dead only under `embed-all`, where every not-compiled branch is stripped.)
#[allow(dead_code)]
fn feature_missing(provider: &str) -> EmbedError {
    EmbedError::Config(format!(
        "provider '{provider}' requires the embed-{provider} feature; enable it"
    ))
}

impl AnyEmbedder {
    /// Build an embedder for `provider` from `config`. Async because some
    /// adapters (Ollama, openai-compat) probe their embedding dimension with a
    /// live call during construction.
    ///
    /// Returns [`EmbedError::Config`] when the requested provider's feature was
    /// not compiled in.
    pub async fn build(provider: EmbedProvider, config: EmbedConfig) -> Result<Self, EmbedError> {
        // Fill in the provider default when the caller left the model empty.
        // This references both `provider` and `config` unconditionally, so the
        // zero-provider build has no unused-variable warnings.
        let mut config = config;
        if config.model.is_empty() {
            config.model = provider.default_model().to_string();
        }

        // Each arm compiles exactly one `#[cfg]` tail-block: the real
        // constructor when the feature is on, else the feature-missing error.
        match provider {
            EmbedProvider::Voyage => {
                #[cfg(feature = "embed-voyage")]
                {
                    voyage::VoyageEmbedder::new(config).map(AnyEmbedder::Voyage)
                }
                #[cfg(not(feature = "embed-voyage"))]
                {
                    Err(feature_missing("voyage"))
                }
            }
            EmbedProvider::OpenAi => {
                #[cfg(feature = "embed-openai")]
                {
                    openai::OpenAiEmbedder::new(config).map(AnyEmbedder::OpenAi)
                }
                #[cfg(not(feature = "embed-openai"))]
                {
                    Err(feature_missing("openai"))
                }
            }
            EmbedProvider::Ollama => {
                #[cfg(feature = "embed-ollama")]
                {
                    ollama::OllamaEmbedder::new(config)
                        .await
                        .map(AnyEmbedder::Ollama)
                }
                #[cfg(not(feature = "embed-ollama"))]
                {
                    Err(feature_missing("ollama"))
                }
            }
            EmbedProvider::Cohere => {
                #[cfg(feature = "embed-cohere")]
                {
                    cohere::CohereEmbedder::new(config).map(AnyEmbedder::Cohere)
                }
                #[cfg(not(feature = "embed-cohere"))]
                {
                    Err(feature_missing("cohere"))
                }
            }
            EmbedProvider::Gemini => {
                #[cfg(feature = "embed-gemini")]
                {
                    gemini::GeminiEmbedder::new(config).map(AnyEmbedder::Gemini)
                }
                #[cfg(not(feature = "embed-gemini"))]
                {
                    Err(feature_missing("gemini"))
                }
            }
            EmbedProvider::Mistral => {
                #[cfg(feature = "embed-mistral")]
                {
                    mistral::MistralEmbedder::new(config).map(AnyEmbedder::Mistral)
                }
                #[cfg(not(feature = "embed-mistral"))]
                {
                    Err(feature_missing("mistral"))
                }
            }
            EmbedProvider::Jina => {
                #[cfg(feature = "embed-jina")]
                {
                    jina::JinaEmbedder::new(config).map(AnyEmbedder::Jina)
                }
                #[cfg(not(feature = "embed-jina"))]
                {
                    Err(feature_missing("jina"))
                }
            }
            EmbedProvider::OpenAiCompat => {
                #[cfg(feature = "embed-openai-compat")]
                {
                    openai_compat::OpenAiCompatEmbedder::new(config)
                        .await
                        .map(AnyEmbedder::OpenAiCompat)
                }
                #[cfg(not(feature = "embed-openai-compat"))]
                {
                    Err(feature_missing("openai-compat"))
                }
            }
        }
    }
}

/// Delegate a method call to the wrapped adapter. The `not(any(...))` wildcard
/// arm keeps the match exhaustive (and this method compilable) when the enum
/// has no variants; it is `unreachable` because `build` can never construct an
/// uninhabited value.
macro_rules! delegate {
    ($self:ident, $e:ident => $call:expr) => {
        match $self {
            #[cfg(feature = "embed-voyage")]
            AnyEmbedder::Voyage($e) => $call,
            #[cfg(feature = "embed-openai")]
            AnyEmbedder::OpenAi($e) => $call,
            #[cfg(feature = "embed-ollama")]
            AnyEmbedder::Ollama($e) => $call,
            #[cfg(feature = "embed-cohere")]
            AnyEmbedder::Cohere($e) => $call,
            #[cfg(feature = "embed-gemini")]
            AnyEmbedder::Gemini($e) => $call,
            #[cfg(feature = "embed-mistral")]
            AnyEmbedder::Mistral($e) => $call,
            #[cfg(feature = "embed-jina")]
            AnyEmbedder::Jina($e) => $call,
            #[cfg(feature = "embed-openai-compat")]
            AnyEmbedder::OpenAiCompat($e) => $call,
            #[cfg(not(any(
                feature = "embed-voyage",
                feature = "embed-openai",
                feature = "embed-ollama",
                feature = "embed-cohere",
                feature = "embed-gemini",
                feature = "embed-mistral",
                feature = "embed-jina",
                feature = "embed-openai-compat",
            )))]
            _ => unreachable!("AnyEmbedder has no compiled-in providers"),
        }
    };
}

// With zero providers compiled in, `delegate!` collapses to only the
// `unreachable!` arm, leaving the `text`/`texts` params unused — expected.
#[cfg_attr(
    not(any(
        feature = "embed-voyage",
        feature = "embed-openai",
        feature = "embed-ollama",
        feature = "embed-cohere",
        feature = "embed-gemini",
        feature = "embed-mistral",
        feature = "embed-jina",
        feature = "embed-openai-compat",
    )),
    allow(unused_variables)
)]
impl Embedder for AnyEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        delegate!(self, e => e.embed(text).await)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        delegate!(self, e => e.embed_batch(texts).await)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        delegate!(self, e => e.embed_query(text).await)
    }

    fn dimension(&self) -> usize {
        delegate!(self, e => e.dimension())
    }
    fn max_input_tokens(&self) -> usize {
        delegate!(self, e => e.max_input_tokens())
    }
    fn provider_name(&self) -> &str {
        delegate!(self, e => e.provider_name())
    }
    fn model_name(&self) -> &str {
        delegate!(self, e => e.model_name())
    }
}

// ── Shared wire helpers (used only by the compiled-in adapters) ──────────────

/// Resolve the effective base URL: caller override else the provider default,
/// with any trailing slash trimmed.
#[cfg(any(
    feature = "embed-voyage",
    feature = "embed-openai",
    feature = "embed-ollama",
    feature = "embed-cohere",
    feature = "embed-gemini",
    feature = "embed-mistral",
    feature = "embed-jina",
    feature = "embed-openai-compat",
))]
pub(crate) fn resolve_base(base_url: Option<&str>, default: &str) -> String {
    base_url
        .unwrap_or(default)
        .trim_end_matches('/')
        .to_string()
}

/// POST `body` as JSON to `url`, with bounded retry, mapping the outcome onto
/// [`EmbedError`]: transport failure → [`EmbedError::Backend`], non-2xx →
/// [`EmbedError::Api`], and a 2xx body that will not parse → [`EmbedError::Decode`].
///
/// `api_key` becomes a bearer header when `Some`; `headers` are applied verbatim
/// (that is how Gemini passes `x-goog-api-key` and how callers pass their extra
/// headers). This is the ONE shared request/response helper for every hosted
/// adapter (the dedupe rule).
#[cfg(any(
    feature = "embed-voyage",
    feature = "embed-openai",
    feature = "embed-ollama",
    feature = "embed-cohere",
    feature = "embed-gemini",
    feature = "embed-mistral",
    feature = "embed-jina",
    feature = "embed-openai-compat",
))]
pub(crate) async fn post_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    policy: &crate::http::RetryPolicy,
    label: &str,
    url: &str,
    api_key: Option<&str>,
    headers: &[(String, String)],
    body: &serde_json::Value,
) -> Result<T, EmbedError> {
    let resp = crate::http::send_with_retry(policy, label, || {
        let mut rb = client.post(url).json(body);
        if let Some(k) = api_key {
            rb = rb.bearer_auth(k);
        }
        for (h, v) in headers {
            rb = rb.header(h.as_str(), v.as_str());
        }
        rb
    })
    .await
    .map_err(|e| EmbedError::Backend(format!("{label}: {e:#}")))?;

    let status = resp.status().as_u16();
    if (200..300).contains(&status) {
        resp.json::<T>()
            .await
            .map_err(|e| EmbedError::Decode(format!("{label}: {e}")))
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(EmbedError::Api { status, body })
    }
}

/// The shared OpenAI `/v1/embeddings` request+parse path, reused by the OpenAI,
/// Mistral, Jina, and openai-compat adapters (the dedupe mandate). Handles the
/// optional `task` field (Jina) and the index-reorder OpenAI does not guarantee.
#[cfg(any(
    feature = "embed-openai",
    feature = "embed-mistral",
    feature = "embed-jina",
    feature = "embed-openai-compat",
))]
pub(crate) mod openai_shaped {
    use super::{EmbedError, post_json};
    use crate::http::RetryPolicy;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Data {
        embedding: Vec<f32>,
        #[serde(default)]
        index: usize,
    }

    #[derive(Deserialize)]
    struct Resp {
        data: Vec<Data>,
    }

    /// One `/v1/embeddings` call for a single (already-chunked) batch of texts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn embed_call(
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
        headers: &[(String, String)],
        model: &str,
        texts: &[&str],
        task: Option<&str>,
        label: &str,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut body = serde_json::json!({ "model": model, "input": texts });
        if let Some(t) = task {
            body["task"] = serde_json::Value::String(t.to_string());
        }
        let policy = RetryPolicy::standard(3, 1000);
        let resp: Resp = post_json(client, &policy, label, url, api_key, headers, &body).await?;
        let mut data = resp.data;
        data.sort_by_key(|d| d.index);
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn response_sorts_by_index() {
            let json =
                r#"{"data":[{"embedding":[0.3,0.4],"index":1},{"embedding":[0.1,0.2],"index":0}]}"#;
            let resp: Resp = serde_json::from_str(json).unwrap();
            let mut data = resp.data;
            data.sort_by_key(|d| d.index);
            assert_eq!(data[0].embedding, vec![0.1, 0.2]);
            assert_eq!(data[1].embedding, vec![0.3, 0.4]);
        }

        #[test]
        fn response_without_index_defaults_to_zero() {
            let json = r#"{"data":[{"embedding":[0.5,0.6]}]}"#;
            let resp: Resp = serde_json::from_str(json).unwrap();
            assert_eq!(resp.data[0].index, 0);
        }
    }
}

// ── Test-only in-process mock HTTP server, shared by adapter wire tests ──────

#[cfg(all(
    test,
    any(
        feature = "embed-voyage",
        feature = "embed-openai",
        feature = "embed-ollama",
        feature = "embed-cohere",
        feature = "embed-gemini",
        feature = "embed-mistral",
        feature = "embed-jina",
        feature = "embed-openai-compat",
    )
))]
pub(crate) mod testutil {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    /// What the mock captured about the single request it served.
    pub struct Captured {
        pub method: String,
        pub path: String,
        pub body: String,
    }

    /// A one-shot HTTP/1.1 server: accepts exactly one connection, replies with
    /// `status`/`resp_body`, and lets the test read back the request it saw.
    pub struct MockServer {
        pub base_url: String,
        rx: mpsc::Receiver<Captured>,
    }

    impl MockServer {
        /// Block until the request has been served and return what it captured.
        pub fn captured(self) -> Captured {
            self.rx.recv().expect("mock server captured a request")
        }
    }

    /// Spin a one-shot mock returning `status` with JSON `resp_body`.
    pub fn mock_once(status: u16, resp_body: &str) -> MockServer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let addr = listener.local_addr().expect("mock server addr");
        let base_url = format!("http://{addr}");
        let (tx, rx) = mpsc::channel();
        let resp_body = resp_body.to_string();

        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept mock connection");

            // Read headers, then the Content-Length body.
            let mut buf = Vec::new();
            let mut tmp = [0u8; 1024];
            let header_end = loop {
                let n = stream.read(&mut tmp).expect("read request");
                if n == 0 {
                    break buf.len();
                }
                buf.extend_from_slice(&tmp[..n]);
                if let Some(pos) = find_subsequence(&buf, b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let head = String::from_utf8_lossy(&buf[..header_end.min(buf.len())]).to_string();
            let content_length = head
                .lines()
                .find_map(|l| {
                    let l = l.to_ascii_lowercase();
                    l.strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            while buf.len() < header_end + content_length {
                let n = stream.read(&mut tmp).expect("read body");
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }

            let mut lines = head.lines();
            let request_line = lines.next().unwrap_or("");
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or("").to_string();
            let path = parts.next().unwrap_or("").to_string();
            let body = String::from_utf8_lossy(&buf[header_end.min(buf.len())..]).to_string();

            let response = format!(
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                resp_body.len(),
                resp_body
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            let _ = tx.send(Captured { method, path, body });
        });

        MockServer { base_url, rx }
    }

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }
}

// ── Pure unit tests (provider-agnostic) ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_roundtrip_names() {
        for p in [
            EmbedProvider::Voyage,
            EmbedProvider::OpenAi,
            EmbedProvider::Ollama,
            EmbedProvider::Cohere,
            EmbedProvider::Gemini,
            EmbedProvider::Mistral,
            EmbedProvider::Jina,
            EmbedProvider::OpenAiCompat,
        ] {
            assert_eq!(EmbedProvider::from_name(p.as_str()), Some(p));
        }
    }

    #[test]
    fn provider_names_match_registry() {
        for name in crate::providers::names_with(crate::providers::Capability::Embed) {
            assert!(
                EmbedProvider::from_name(name).is_some(),
                "registry name {name} has no EmbedProvider"
            );
        }
    }

    #[test]
    fn from_str_unknown_is_config_error() {
        let e = "nope".parse::<EmbedProvider>().unwrap_err();
        assert!(matches!(e, EmbedError::Config(_)));
    }

    #[test]
    fn default_models() {
        assert_eq!(
            EmbedProvider::OpenAi.default_model(),
            "text-embedding-3-small"
        );
        assert_eq!(EmbedProvider::Voyage.default_model(), "voyage-3");
        assert_eq!(EmbedProvider::OpenAiCompat.default_model(), "");
    }

    #[test]
    fn config_builder() {
        let c = EmbedConfig::new("m")
            .api_key("k")
            .base_url("http://x")
            .header("a", "b");
        assert_eq!(c.model, "m");
        assert_eq!(c.api_key, "k");
        assert_eq!(c.base_url.as_deref(), Some("http://x"));
        assert_eq!(c.extra_headers, vec![("a".to_string(), "b".to_string())]);
    }

    #[test]
    fn error_display() {
        assert!(
            EmbedError::Config("x".into())
                .to_string()
                .contains("config")
        );
        assert!(
            EmbedError::Api {
                status: 429,
                body: "slow down".into(),
            }
            .to_string()
            .contains("429")
        );
    }

    struct Fake;
    impl Embedder for Fake {
        async fn embed(&self, _: &str) -> Result<Vec<f32>, EmbedError> {
            Ok(vec![])
        }
        async fn embed_batch(&self, t: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
            Ok(vec![vec![]; t.len()])
        }
        fn dimension(&self) -> usize {
            3
        }
        fn max_input_tokens(&self) -> usize {
            8192
        }
        fn provider_name(&self) -> &str {
            "test"
        }
        fn model_name(&self) -> &str {
            "fake-v1"
        }
    }

    #[test]
    fn identity_format() {
        assert_eq!(embedder_identity(&Fake), "test/fake-v1");
    }
}
