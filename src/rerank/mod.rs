//! Retrieve-then-rerank over a hosted cross-encoder (nidus-4ss): [`Reranker`] +
//! [`AnyReranker`] mirror `src/embed/`'s shape exactly. `stage.rs` is declared in
//! `src/lib.rs` ungated (pure ranking logic, Miri-covered); every item below is gated
//! `#[cfg(feature = "rerank")]` individually, since this module itself is not.

pub mod stage;

#[cfg(feature = "rerank")]
use std::fmt;

// ── Provider adapter modules (one per `rerank-<name>` feature) ──────────────
// Public so the concrete types can appear in the public `AnyReranker` variants
// without tripping the private-interface lint (mirrors `src/embed/mod.rs`).
#[cfg(feature = "rerank-cohere")]
pub mod cohere;
#[cfg(feature = "rerank-jina")]
pub mod jina;
#[cfg(feature = "rerank-voyage")]
pub mod voyage;

#[cfg(feature = "rerank")]
pub mod apply;

// ── Errors ───────────────────────────────────────────────────────────────────

/// A typed error at the rerank edge (the public surface uses this enum, not `anyhow`).
#[cfg(feature = "rerank")]
#[derive(Debug)]
pub enum RerankError {
    /// Bad/missing configuration (no API key, unknown or not-compiled-in provider).
    Config(String),
    /// A transport-level failure that survived the retry budget.
    Backend(String),
    /// The API returned a non-2xx status. `body` is the raw response text.
    Api { status: u16, body: String },
    /// A 2xx response whose body could not be parsed into the expected shape.
    Decode(String),
}

#[cfg(feature = "rerank")]
impl fmt::Display for RerankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RerankError::Config(m) => write!(f, "rerank config error: {m}"),
            RerankError::Backend(m) => write!(f, "rerank backend error: {m}"),
            RerankError::Api { status, body } => {
                write!(f, "rerank API error ({status}): {body}")
            }
            RerankError::Decode(m) => write!(f, "rerank decode error: {m}"),
        }
    }
}

#[cfg(feature = "rerank")]
impl std::error::Error for RerankError {}

// ── Config ─────────────────────────────────────────────────────────────────

/// Everything an [`AnyReranker`] needs to reach a provider. Built fluently:
///
/// ```ignore
/// let cfg = RerankConfig::new("rerank-2.5").api_key(std::env::var("VOYAGE_API_KEY")?);
/// ```
#[cfg(feature = "rerank")]
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// Model name. Empty means "use the provider default" ([`RerankProvider::default_model`]).
    pub model: String,
    /// Bearer/API key. All three providers require one.
    pub api_key: String,
    /// Override the provider's default base URL (used to point at a local mock).
    pub base_url: Option<String>,
    /// Extra request headers applied to every call (e.g. gateway auth).
    pub extra_headers: Vec<(String, String)>,
}

#[cfg(feature = "rerank")]
impl RerankConfig {
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

/// Which rerank backend to build. The `as_str` values match the [`crate::providers`]
/// registry names exactly.
#[cfg(feature = "rerank")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankProvider {
    Voyage,
    Cohere,
    Jina,
}

#[cfg(feature = "rerank")]
impl RerankProvider {
    /// Parse a registry name (`voyage`, `cohere`, `jina`).
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "voyage" => RerankProvider::Voyage,
            "cohere" => RerankProvider::Cohere,
            "jina" => RerankProvider::Jina,
            _ => return None,
        })
    }

    /// The registry name — matches [`crate::providers`].
    pub fn as_str(&self) -> &'static str {
        match self {
            RerankProvider::Voyage => "voyage",
            RerankProvider::Cohere => "cohere",
            RerankProvider::Jina => "jina",
        }
    }

    /// The model used when [`RerankConfig::model`] is empty.
    pub fn default_model(&self) -> &'static str {
        match self {
            RerankProvider::Voyage => "rerank-2.5",
            RerankProvider::Cohere => "rerank-v3.5",
            RerankProvider::Jina => "jina-reranker-v2-base-multilingual",
        }
    }
}

#[cfg(feature = "rerank")]
impl std::str::FromStr for RerankProvider {
    type Err = RerankError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        RerankProvider::from_name(s).ok_or_else(|| {
            RerankError::Config(format!(
                "unknown rerank provider '{s}'; available: {}",
                crate::providers::names_with(crate::providers::Capability::Rerank).join(", ")
            ))
        })
    }
}

// ── The trait ────────────────────────────────────────────────────────────────

/// Scores `(query, doc)` pairs at a hosted cross-encoder. Native async (RPITIT) — **not**
/// object-safe by design; dispatch via [`AnyReranker`], never `Box<dyn>`.
#[cfg(feature = "rerank")]
pub trait Reranker: Send + Sync {
    /// One score per doc, in `docs` order — NOT the provider's returned order, which an
    /// adapter must undo or the merge step's contract breaks. `model` overrides the adapter's
    /// configured model for this call only.
    fn rerank(
        &self,
        query: &str,
        docs: &[&str],
        model: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Vec<f32>, RerankError>> + Send;

    fn provider_name(&self) -> &str;
    fn model_name(&self) -> &str;
    /// Largest doc count the provider accepts per request; adapters chunk internally.
    fn max_documents(&self) -> usize;
}

/// `"provider/model"` — a stable identity string for a reranker.
#[cfg(feature = "rerank")]
pub fn reranker_identity(r: &impl Reranker) -> String {
    format!("{}/{}", r.provider_name(), r.model_name())
}

// ── AnyReranker: the closed, runtime-selectable enum ─────────────────────────

/// One variant per compiled-in provider. This is the concrete type callers hold; it
/// implements [`Reranker`] by delegating to the wrapped adapter.
#[cfg(feature = "rerank")]
#[derive(Debug)]
pub enum AnyReranker {
    #[cfg(feature = "rerank-voyage")]
    Voyage(voyage::VoyageReranker),
    #[cfg(feature = "rerank-cohere")]
    Cohere(cohere::CohereReranker),
    #[cfg(feature = "rerank-jina")]
    Jina(jina::JinaReranker),
}

/// The error returned when a provider was requested but its feature is off. (Dead only
/// under `rerank-all`, where every not-compiled branch is stripped.)
#[cfg(feature = "rerank")]
#[allow(dead_code)]
fn feature_missing(provider: &str) -> RerankError {
    RerankError::Config(format!(
        "provider '{provider}' requires the rerank-{provider} feature; enable it"
    ))
}

#[cfg(feature = "rerank")]
impl AnyReranker {
    /// Build a reranker for `provider` from `config`. Sync — unlike
    /// `AnyEmbedder`, no adapter probes anything with a live call during
    /// construction.
    pub fn build(provider: RerankProvider, config: RerankConfig) -> Result<Self, RerankError> {
        let mut config = config;
        if config.model.is_empty() {
            config.model = provider.default_model().to_string();
        }

        // Each arm compiles exactly one `#[cfg]` tail-block: the real constructor when
        // the feature is on, else the feature-missing error.
        match provider {
            RerankProvider::Voyage => {
                #[cfg(feature = "rerank-voyage")]
                {
                    voyage::VoyageReranker::new(config).map(AnyReranker::Voyage)
                }
                #[cfg(not(feature = "rerank-voyage"))]
                {
                    Err(feature_missing("voyage"))
                }
            }
            RerankProvider::Cohere => {
                #[cfg(feature = "rerank-cohere")]
                {
                    cohere::CohereReranker::new(config).map(AnyReranker::Cohere)
                }
                #[cfg(not(feature = "rerank-cohere"))]
                {
                    Err(feature_missing("cohere"))
                }
            }
            RerankProvider::Jina => {
                #[cfg(feature = "rerank-jina")]
                {
                    jina::JinaReranker::new(config).map(AnyReranker::Jina)
                }
                #[cfg(not(feature = "rerank-jina"))]
                {
                    Err(feature_missing("jina"))
                }
            }
        }
    }
}

/// Delegate a method call to the wrapped adapter. The `not(any(...))` wildcard arm keeps the match
/// exhaustive (and this method compilable) when the enum has no variants; it is `unreachable`
/// because `build` can never construct an uninhabited value.
#[cfg(feature = "rerank")]
macro_rules! delegate {
    ($self:ident, $e:ident => $call:expr) => {
        match $self {
            #[cfg(feature = "rerank-voyage")]
            AnyReranker::Voyage($e) => $call,
            #[cfg(feature = "rerank-cohere")]
            AnyReranker::Cohere($e) => $call,
            #[cfg(feature = "rerank-jina")]
            AnyReranker::Jina($e) => $call,
            #[cfg(not(any(
                feature = "rerank-voyage",
                feature = "rerank-cohere",
                feature = "rerank-jina",
            )))]
            _ => unreachable!("AnyReranker has no compiled-in providers"),
        }
    };
}

// With zero providers compiled in, `delegate!` collapses to only the `unreachable!` arm,
// leaving the `query`/`docs` params unused — expected.
#[cfg(feature = "rerank")]
#[cfg_attr(
    not(any(
        feature = "rerank-voyage",
        feature = "rerank-cohere",
        feature = "rerank-jina",
    )),
    allow(unused_variables)
)]
impl Reranker for AnyReranker {
    async fn rerank(
        &self,
        query: &str,
        docs: &[&str],
        model: Option<&str>,
    ) -> Result<Vec<f32>, RerankError> {
        delegate!(self, e => e.rerank(query, docs, model).await)
    }

    fn provider_name(&self) -> &str {
        delegate!(self, e => e.provider_name())
    }
    fn model_name(&self) -> &str {
        delegate!(self, e => e.model_name())
    }
    fn max_documents(&self) -> usize {
        delegate!(self, e => e.max_documents())
    }
}

// ── Shared wire helpers (used only by the compiled-in adapters) ──────────────
// Not embed's `resolve_base`/`post_json`: those are `pub(crate)` behind embed's own
// feature gates, so a `rerank`-only build would not have them.

/// Resolve the effective base URL: caller override else the provider default, with any
/// trailing slash trimmed.
#[cfg(any(
    feature = "rerank-voyage",
    feature = "rerank-cohere",
    feature = "rerank-jina",
))]
pub(crate) fn resolve_base(base_url: Option<&str>, default: &str) -> String {
    base_url
        .unwrap_or(default)
        .trim_end_matches('/')
        .to_string()
}

/// POST `body` as JSON to `url`, with bounded retry, mapping the outcome onto
/// [`RerankError`]: transport failure → [`RerankError::Backend`], non-2xx →
/// [`RerankError::Api`], and a 2xx body that will not parse → [`RerankError::Decode`].
#[cfg(any(
    feature = "rerank-voyage",
    feature = "rerank-cohere",
    feature = "rerank-jina",
))]
pub(crate) async fn post_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    policy: &crate::http::RetryPolicy,
    label: &str,
    url: &str,
    api_key: Option<&str>,
    headers: &[(String, String)],
    body: &serde_json::Value,
) -> Result<T, RerankError> {
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
    .map_err(|e| RerankError::Backend(format!("{label}: {e:#}")))?;

    let status = resp.status().as_u16();
    if (200..300).contains(&status) {
        resp.json::<T>()
            .await
            .map_err(|e| RerankError::Decode(format!("{label}: {e}")))
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(RerankError::Api { status, body })
    }
}

// ── Test-only in-process mock HTTP server, shared by adapter wire tests ──────

#[cfg(all(
    test,
    any(
        feature = "rerank-voyage",
        feature = "rerank-cohere",
        feature = "rerank-jina",
    )
))]
pub(crate) mod testutil {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    /// What the mock captured about one request it served.
    #[allow(dead_code)]
    pub struct Captured {
        pub method: String,
        pub path: String,
        pub body: String,
    }

    /// A mock HTTP/1.1 server: accepts up to `times` connections, replies with
    /// `status`/`resp_body` on each, and lets the test read back what it saw.
    pub struct MockServer {
        pub base_url: String,
        rx: mpsc::Receiver<Captured>,
    }

    impl MockServer {
        /// Block until the single request has been served (the one-shot case).
        pub fn captured(self) -> Captured {
            self.rx.recv().expect("mock server captured a request")
        }

        /// Block until `n` requests have been served, in the order received (the chunking
        /// case, where one rerank call spans more than one provider round trip).
        pub fn captured_n(self, n: usize) -> Vec<Captured> {
            (0..n)
                .map(|_| self.rx.recv().expect("mock server captured a request"))
                .collect()
        }
    }

    /// Spin a one-shot mock returning `status` with JSON `resp_body`.
    pub fn mock_once(status: u16, resp_body: &str) -> MockServer {
        mock_n(status, resp_body, 1)
    }

    /// Spin a mock that answers `times` requests with the same `status`/`resp_body`.
    pub fn mock_persistent(status: u16, resp_body: &str, times: usize) -> MockServer {
        mock_n(status, resp_body, times)
    }

    fn mock_n(status: u16, resp_body: &str, times: usize) -> MockServer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let addr = listener.local_addr().expect("mock server addr");
        let base_url = format!("http://{addr}");
        let (tx, rx) = mpsc::channel();
        let resp_body = resp_body.to_string();

        thread::spawn(move || {
            for _ in 0..times {
                let (mut stream, _) = listener.accept().expect("accept mock connection");

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
            }
        });

        MockServer { base_url, rx }
    }

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }
}

// ── Pure unit tests (provider-agnostic) ───────────────────────────────────────

#[cfg(all(test, feature = "rerank"))]
mod tests {
    use super::*;

    #[test]
    fn provider_roundtrip_names() {
        for p in [
            RerankProvider::Voyage,
            RerankProvider::Cohere,
            RerankProvider::Jina,
        ] {
            assert_eq!(RerankProvider::from_name(p.as_str()), Some(p));
        }
    }

    #[test]
    fn provider_names_match_registry() {
        for name in crate::providers::names_with(crate::providers::Capability::Rerank) {
            assert!(
                RerankProvider::from_name(name).is_some(),
                "registry name {name} has no RerankProvider"
            );
        }
    }

    #[test]
    fn from_str_unknown_is_config_error() {
        let e = "nope".parse::<RerankProvider>().unwrap_err();
        assert!(matches!(e, RerankError::Config(_)));
    }

    #[test]
    fn default_models() {
        assert_eq!(RerankProvider::Voyage.default_model(), "rerank-2.5");
        assert_eq!(RerankProvider::Cohere.default_model(), "rerank-v3.5");
        assert_eq!(
            RerankProvider::Jina.default_model(),
            "jina-reranker-v2-base-multilingual"
        );
    }

    #[test]
    fn config_builder() {
        let c = RerankConfig::new("m")
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
            RerankError::Config("x".into())
                .to_string()
                .contains("config")
        );
        assert!(
            RerankError::Api {
                status: 429,
                body: "slow down".into(),
            }
            .to_string()
            .contains("429")
        );
    }

    struct Fake;
    impl Reranker for Fake {
        async fn rerank(
            &self,
            _query: &str,
            docs: &[&str],
            _model: Option<&str>,
        ) -> Result<Vec<f32>, RerankError> {
            Ok(vec![0.0; docs.len()])
        }
        fn provider_name(&self) -> &str {
            "test"
        }
        fn model_name(&self) -> &str {
            "fake-v1"
        }
        fn max_documents(&self) -> usize {
            1000
        }
    }

    #[test]
    fn identity_format() {
        assert_eq!(reranker_identity(&Fake), "test/fake-v1");
    }
}
