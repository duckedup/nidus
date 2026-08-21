//! Hosted cross-encoder reranking (epic nidus-4ss): the async post-ranking provider seam,
//! **not** the quantized int8→f32 rescore `src/store/quant.rs` also calls a rerank. The pure
//! window/passthrough/re-sort logic that stage shares with this one lives, unconditionally,
//! in [`crate::store::rerank`]; only the network call sits behind this feature.

use std::fmt;

#[cfg(feature = "rerank-cohere")]
pub mod cohere;
#[cfg(feature = "rerank-voyage")]
pub mod voyage;

mod apply;
pub use apply::{hybrid_reranked, rerank_hits, search_reranked, text_search_reranked};

// ── Errors ───────────────────────────────────────────────────────────────────

/// A typed error at the rerank edge (the public surface uses this enum, not `anyhow`).
#[derive(Debug)]
pub enum RerankError {
    /// Bad/missing configuration (no API key, unknown or not-compiled-in provider).
    Config(String),
    /// A transport-level failure that survived the retry budget.
    Backend(String),
    /// The API returned a non-2xx status. `body` is the raw response text.
    Api { status: u16, body: String },
    /// A 2xx response whose body could not be parsed into the expected shape, or whose
    /// scores could not be safely scattered back to input order (missing/out-of-range
    /// index, or fewer scores than documents).
    Decode(String),
}

impl fmt::Display for RerankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RerankError::Config(m) => write!(f, "rerank config error: {m}"),
            RerankError::Backend(m) => write!(f, "rerank backend error: {m}"),
            RerankError::Api { status, body } => write!(f, "rerank API error ({status}): {body}"),
            RerankError::Decode(m) => write!(f, "rerank decode error: {m}"),
        }
    }
}

impl std::error::Error for RerankError {}

// ── Config ─────────────────────────────────────────────────────────────────

/// Everything an [`AnyReranker`] needs to reach a provider. Built fluently.
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// Model name. Empty means "use the provider default" (see
    /// [`RerankProvider::default_model`]).
    pub model: String,
    /// Bearer/API key.
    pub api_key: String,
    /// Override the provider's default base URL.
    pub base_url: Option<String>,
    /// Extra request headers applied to every call (e.g. gateway auth).
    pub extra_headers: Vec<(String, String)>,
}

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankProvider {
    Voyage,
    Cohere,
}

impl RerankProvider {
    /// Parse a registry name (`voyage`, `cohere`).
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "voyage" => RerankProvider::Voyage,
            "cohere" => RerankProvider::Cohere,
            _ => return None,
        })
    }

    /// The registry name — matches [`crate::providers`].
    pub fn as_str(&self) -> &'static str {
        match self {
            RerankProvider::Voyage => "voyage",
            RerankProvider::Cohere => "cohere",
        }
    }

    /// The model used when [`RerankConfig::model`] is empty.
    pub fn default_model(&self) -> &'static str {
        match self {
            RerankProvider::Voyage => "rerank-2.5",
            RerankProvider::Cohere => "rerank-v3.5",
        }
    }
}

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

/// Scores documents against a query. Native async (RPITIT) — **not** object-safe by design;
/// dispatch via [`AnyReranker`], never `Box<dyn>`.
pub trait Reranker: Send + Sync {
    /// Score each document against `query`, one score per input document **in input order**.
    /// Providers return `{index, score}` pairs, possibly reordered or short; normalising that
    /// in the adapter is what keeps `crate::store::rerank` provider-agnostic.
    fn rerank(
        &self,
        query: &str,
        documents: &[&str],
    ) -> impl std::future::Future<Output = Result<Vec<f32>, RerankError>> + Send;

    fn provider_name(&self) -> &str;
    fn model_name(&self) -> &str;
    fn max_documents(&self) -> usize;
}

/// `"provider/model"` — a stable identity string for a reranker.
pub fn reranker_identity(r: &impl Reranker) -> String {
    format!("{}/{}", r.provider_name(), r.model_name())
}

// ── AnyReranker: the closed, runtime-selectable enum ─────────────────────────

/// One variant per compiled-in provider. This is the concrete type callers hold; it
/// implements [`Reranker`] by delegating to the wrapped adapter.
pub enum AnyReranker {
    #[cfg(feature = "rerank-voyage")]
    Voyage(voyage::VoyageReranker),
    #[cfg(feature = "rerank-cohere")]
    Cohere(cohere::CohereReranker),
}

/// The error returned when a provider was requested but its feature is off.
/// (Dead only under `rerank-all`, where every not-compiled branch is stripped.)
#[allow(dead_code)]
fn feature_missing(provider: &str) -> RerankError {
    RerankError::Config(format!(
        "provider '{provider}' requires the rerank-{provider} feature; enable it"
    ))
}

impl AnyReranker {
    /// Build a reranker for `provider` from `config`.
    pub fn build(provider: RerankProvider, config: RerankConfig) -> Result<Self, RerankError> {
        let mut config = config;
        if config.model.is_empty() {
            config.model = provider.default_model().to_string();
        }

        // Each arm compiles exactly one `#[cfg]` tail-block: the real constructor when the
        // feature is on, else the feature-missing error.
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
        }
    }
}

/// Delegate a method call to the wrapped adapter. The `not(any(...))` wildcard arm keeps the
/// match exhaustive when the enum has no variants; it is `unreachable` because `build` can
/// never construct an uninhabited value.
macro_rules! delegate {
    ($self:ident, $e:ident => $call:expr) => {
        match $self {
            #[cfg(feature = "rerank-voyage")]
            AnyReranker::Voyage($e) => $call,
            #[cfg(feature = "rerank-cohere")]
            AnyReranker::Cohere($e) => $call,
            #[cfg(not(any(feature = "rerank-voyage", feature = "rerank-cohere")))]
            _ => unreachable!("AnyReranker has no compiled-in providers"),
        }
    };
}

// With zero providers compiled in, `delegate!` collapses to only the `unreachable!` arm,
// leaving the params unused — expected.
#[cfg_attr(
    not(any(feature = "rerank-voyage", feature = "rerank-cohere")),
    allow(unused_variables)
)]
impl Reranker for AnyReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, RerankError> {
        delegate!(self, e => e.rerank(query, documents).await)
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

/// Resolve the effective base URL: caller override else the provider default, with any
/// trailing slash trimmed.
#[cfg(any(feature = "rerank-voyage", feature = "rerank-cohere"))]
pub(crate) fn resolve_base(base_url: Option<&str>, default: &str) -> String {
    base_url
        .unwrap_or(default)
        .trim_end_matches('/')
        .to_string()
}

/// POST `body` as JSON to `url`, with bounded retry, mapping the outcome onto
/// [`RerankError`]: transport failure → [`RerankError::Backend`], non-2xx →
/// [`RerankError::Api`], and a 2xx body that will not parse → [`RerankError::Decode`].
#[cfg(any(feature = "rerank-voyage", feature = "rerank-cohere"))]
pub(crate) async fn post_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    policy: &crate::http::RetryPolicy,
    label: &str,
    url: &str,
    api_key: &str,
    headers: &[(String, String)],
    body: &serde_json::Value,
) -> Result<T, RerankError> {
    let resp = crate::http::send_with_retry(policy, label, || {
        let mut rb = client.post(url).bearer_auth(api_key).json(body);
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

/// Scatter `(index, relevance_score)` pairs back into input order, over `n` documents.
/// `Decode`s if an index is missing, out of range, or duplicated, or if fewer scores than
/// documents came back — a silently short/misaligned result is the worst failure here.
#[cfg(any(feature = "rerank-voyage", feature = "rerank-cohere"))]
pub(crate) fn scatter_by_index(
    n: usize,
    pairs: impl IntoIterator<Item = (usize, f32)>,
    label: &str,
) -> Result<Vec<f32>, RerankError> {
    let mut slots: Vec<Option<f32>> = vec![None; n];
    for (idx, score) in pairs {
        match slots.get_mut(idx) {
            Some(slot @ None) => *slot = Some(score),
            Some(Some(_)) => {
                return Err(RerankError::Decode(format!(
                    "{label}: duplicate index {idx} in rerank response"
                )));
            }
            None => {
                return Err(RerankError::Decode(format!(
                    "{label}: out-of-range index {idx} in rerank response (n={n})"
                )));
            }
        }
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            s.ok_or_else(|| RerankError::Decode(format!("{label}: missing score for index {i}")))
        })
        .collect()
}

// ── Test-only in-process mock HTTP server, shared by adapter wire tests ──────

#[cfg(all(test, any(feature = "rerank-voyage", feature = "rerank-cohere")))]
pub(crate) mod testutil {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    /// What the mock captured about the single request it served.
    #[allow(dead_code)]
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
        for p in [RerankProvider::Voyage, RerankProvider::Cohere] {
            assert_eq!(RerankProvider::from_name(p.as_str()), Some(p));
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

    #[cfg(any(feature = "rerank-voyage", feature = "rerank-cohere"))]
    #[test]
    fn scatter_out_of_order_indices() {
        let scores = scatter_by_index(3, [(2, 0.3), (0, 0.1), (1, 0.2)], "test").unwrap();
        assert_eq!(scores, vec![0.1, 0.2, 0.3]);
    }

    #[cfg(any(feature = "rerank-voyage", feature = "rerank-cohere"))]
    #[test]
    fn scatter_rejects_short_response() {
        let err = scatter_by_index(3, [(0, 0.1), (1, 0.2)], "test").unwrap_err();
        assert!(matches!(err, RerankError::Decode(_)));
    }

    #[cfg(any(feature = "rerank-voyage", feature = "rerank-cohere"))]
    #[test]
    fn scatter_rejects_out_of_range_index() {
        let err = scatter_by_index(2, [(0, 0.1), (5, 0.2)], "test").unwrap_err();
        assert!(matches!(err, RerankError::Decode(_)));
    }

    struct Fake;
    impl Reranker for Fake {
        async fn rerank(&self, _: &str, docs: &[&str]) -> Result<Vec<f32>, RerankError> {
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
