//! Generic HTTP retry infrastructure shared by every reqwest-based adapter.

// Shared retry infra whose only consumers are the per-provider adapters. A feature set enabling a
// base but no adapter — or an adapter not using a given helper — leaves items unreferenced here,
// which is expected rather than dead, so this is allowed module-wide.
#![allow(dead_code)]

use std::time::Duration;

use anyhow::{Context, Result};

/// Retry classification + timing for a family of requests.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub base_delay_ms: u64,
    pub retryable: fn(u16) -> bool,
}

impl RetryPolicy {
    /// Retry on rate limits and the common transient server statuses
    /// (429, 500, 502, 503, 529). Used by the hosted AI APIs.
    pub fn standard(max_retries: usize, base_delay_ms: u64) -> Self {
        Self {
            max_retries,
            base_delay_ms,
            retryable: is_retryable_standard,
        }
    }

    /// Retry on any 5xx server error. Used by Ollama (local), which treats
    /// client errors as terminal.
    pub fn server_errors(max_retries: usize, base_delay_ms: u64) -> Self {
        Self {
            max_retries,
            base_delay_ms,
            retryable: is_server_error,
        }
    }
}

/// The standard retryable set for hosted AI APIs: rate limiting plus the
/// transient server statuses (including Anthropic's 529 "overloaded").
pub fn is_retryable_standard(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 529)
}

/// Any 5xx status.
pub fn is_server_error(status: u16) -> bool {
    status >= 500
}

/// Exponential backoff: `base_delay_ms * 2^attempt`.
pub fn backoff(policy: &RetryPolicy, attempt: usize) -> Duration {
    Duration::from_millis(policy.base_delay_ms * 2u64.pow(attempt as u32))
}

/// Send a request with bounded exponential-backoff retry.
pub async fn send_with_retry<F>(
    policy: &RetryPolicy,
    label: &str,
    build: F,
) -> Result<reqwest::Response>
where
    F: Fn() -> reqwest::RequestBuilder,
{
    let mut attempt = 0;
    loop {
        match build().send().await {
            Ok(resp) => {
                if (policy.retryable)(resp.status().as_u16()) && attempt < policy.max_retries {
                    let delay = backoff(policy, attempt);
                    // Counted as well as logged: a backend that has been retrying all
                    // morning is invisible in a log nobody is tailing, and looks like
                    // nothing at all until it fails outright (nidus-abx.4).
                    crate::metrics::metrics().backend_retries.inc();
                    crate::diag::diag!(
                        crate::diag::Level::Warn,
                        "backend",
                        "request returned a retryable status, retrying",
                        "label" => label,
                        "status" => resp.status(),
                        "attempt" => attempt + 1,
                        "delay_ms" => delay.as_millis(),
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
                return Ok(resp);
            }
            Err(e) => {
                if attempt < policy.max_retries {
                    let delay = backoff(policy, attempt);
                    crate::metrics::metrics().backend_retries.inc();
                    crate::diag::diag!(
                        crate::diag::Level::Warn,
                        "backend",
                        "request failed at the transport, retrying",
                        "label" => label,
                        "err" => e,
                        "attempt" => attempt + 1,
                        "delay_ms" => delay.as_millis(),
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
                return Err(e).with_context(|| format!("{label} request failed"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_retryable_set() {
        assert!(is_retryable_standard(429));
        assert!(is_retryable_standard(500));
        assert!(is_retryable_standard(502));
        assert!(is_retryable_standard(503));
        assert!(is_retryable_standard(529));
        assert!(!is_retryable_standard(200));
        assert!(!is_retryable_standard(400));
        assert!(!is_retryable_standard(401));
        assert!(!is_retryable_standard(404));
    }

    #[test]
    fn server_error_set() {
        assert!(is_server_error(500));
        assert!(is_server_error(502));
        assert!(is_server_error(599));
        assert!(!is_server_error(200));
        assert!(!is_server_error(429));
        assert!(!is_server_error(404));
    }

    #[test]
    fn backoff_is_exponential() {
        let p = RetryPolicy::standard(3, 1000);
        assert_eq!(backoff(&p, 0), Duration::from_secs(1));
        assert_eq!(backoff(&p, 1), Duration::from_secs(2));
        assert_eq!(backoff(&p, 2), Duration::from_secs(4));
        assert_eq!(backoff(&p, 3), Duration::from_secs(8));
    }

    #[test]
    fn backoff_respects_base_delay() {
        let p = RetryPolicy::server_errors(3, 500);
        assert_eq!(backoff(&p, 0), Duration::from_millis(500));
        assert_eq!(backoff(&p, 1), Duration::from_secs(1));
        assert_eq!(backoff(&p, 2), Duration::from_secs(2));
    }

    #[test]
    fn constructors_wire_the_right_predicate() {
        let std = RetryPolicy::standard(3, 1000);
        assert!((std.retryable)(429));
        let srv = RetryPolicy::server_errors(3, 1000);
        assert!(!(srv.retryable)(429));
        assert!((srv.retryable)(500));
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
        feature = "rerank-voyage",
        feature = "rerank-cohere",
        feature = "summarize-anthropic",
        feature = "summarize-openai",
    )
))]
pub(crate) mod mock {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    /// What the mock captured about the single request it served.
    #[allow(dead_code)]
    pub struct Captured {
        pub method: String,
        pub path: String,
        pub head: String,
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
            let _ = tx.send(Captured {
                method,
                path,
                head,
                body,
            });
        });

        MockServer { base_url, rx }
    }

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }
}
