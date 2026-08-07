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
