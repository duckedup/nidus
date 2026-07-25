//! Shared blocking HTTP for the object-store backends (S3/GCS).
//!
//! A thin [`ureq`] wrapper that returns `(status, body)` and **never** treats a non-2xx
//! response as an error — the callers map status themselves (404 → `None`, etc.). TLS
//! is ureq's default rustls + `ring`, self-contained via `webpki-roots` (no system cert
//! store, no OpenSSL). The sans-IO clients (`rusty-s3`/`tame-gcs`) build and sign the
//! requests; this only executes them.

use anyhow::{Context, Result, anyhow};
use http::{HeaderMap, Response};
use ureq::{Agent, Body};

/// A reusable blocking HTTP client (one pooled `ureq::Agent`).
pub(crate) struct Http {
    agent: Agent,
}

/// Whether a transport failure is worth one immediate retry.
///
/// The case this exists for is a **stale pooled connection**: the agent keeps connections
/// alive, the server (or an intervening proxy) closes one after its own idle timeout, and
/// the next request reuses the dead socket and fails with "Peer disconnected" / a reset
/// before any bytes are served. There is nothing wrong with the request — a fresh
/// connection succeeds — so failing the caller's operation on it is a self-inflicted error.
///
/// Observed for real: a `nidus serve` instance exited during a cluster e2e run because a
/// lease claim hit exactly this, which is why it is handled here rather than left to every
/// call site.
///
/// Only *connection-level* failures qualify. An HTTP status is never retried here (the
/// client is built with `http_status_as_error(false)`, so statuses are not errors at all),
/// and a timeout is not retried either — that would double the caller's worst case.
fn worth_retrying(e: &ureq::Error) -> bool {
    match e {
        ureq::Error::Io(io) => matches!(
            io.kind(),
            std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::UnexpectedEof
                | std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::NotConnected
        ),
        _ => false,
    }
}

/// Run `attempt`, retrying once if the first failure looks like a dropped pooled
/// connection. Requests reaching here are idempotent or compare-and-swap-guarded, so a
/// single replay cannot double-apply anything: a CAS whose first attempt actually landed
/// finds its token consumed on the replay and reports `Stale`, which the callers already
/// handle.
fn with_one_retry<T>(
    mut attempt: impl FnMut() -> std::result::Result<T, ureq::Error>,
) -> Result<T> {
    match attempt() {
        Ok(v) => Ok(v),
        Err(first) if worth_retrying(&first) => attempt().map_err(|second| {
            net_err(second).context(format!("after retrying a dropped connection ({first})"))
        }),
        Err(e) => Err(net_err(e)),
    }
}

impl Http {
    pub(crate) fn new() -> Http {
        Http::new_with_timeout(None)
    }

    /// Like [`new`](Self::new) but with an overall per-request timeout. Used for the
    /// credential-metadata calls (STS / ECS / EC2 IMDS): off-cloud the link-local IMDS
    /// address is unroutable, so a short timeout turns a multi-second hang into a quick,
    /// clear "no credentials" error.
    pub(crate) fn new_with_timeout(timeout: Option<std::time::Duration>) -> Http {
        // `http_status_as_error(false)`: a 4xx/5xx comes back as a normal response so a
        // backend can distinguish 404 (absent → `None`) from a transport failure.
        let config = ureq::config::Config::builder()
            .http_status_as_error(false)
            .timeout_global(timeout)
            .build();
        Http {
            agent: Agent::new_with_config(config),
        }
    }

    /// `GET url`, returning the response status and body.
    pub(crate) fn get(&self, url: &str) -> Result<(u16, Vec<u8>)> {
        let (status, body, _headers) = self.get_h(url)?;
        Ok((status, body))
    }

    /// `GET url`, also returning the response headers (the compare-and-swap paths read the
    /// object's version token from them — S3 `ETag`, GCS `x-goog-generation`).
    pub(crate) fn get_h(&self, url: &str) -> Result<(u16, Vec<u8>, HeaderMap)> {
        finish(with_one_retry(|| self.agent.get(url).call())?)
    }

    /// `PUT url` with `body` and any extra request `headers` (e.g. a signed
    /// `If-None-Match: *` for a conditional create — those headers must be sent
    /// verbatim because they are part of the SigV4 signature).
    pub(crate) fn put(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<(u16, Vec<u8>)> {
        let (status, body, _headers) = self.put_h(url, headers, body)?;
        Ok((status, body))
    }

    /// `PUT url`, also returning the response headers — the conditional-write path reads the
    /// object's **new** version token (S3 returns it as the response `ETag`).
    pub(crate) fn put_h(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<(u16, Vec<u8>, HeaderMap)> {
        finish(with_one_retry(|| {
            let mut req = self.agent.put(url);
            for (name, value) in headers {
                req = req.header(*name, *value);
            }
            req.send(body)
        })?)
    }

    /// `DELETE url`.
    pub(crate) fn delete(&self, url: &str) -> Result<(u16, Vec<u8>)> {
        let (status, body, _headers) = finish(with_one_retry(|| self.agent.delete(url).call())?)?;
        Ok((status, body))
    }

    /// Run a fully-built request (method + uri + headers + body) — used by GCS, whose
    /// sans-IO client (and `tame-oauth`) emit [`http::Request`]s.
    pub(crate) fn run(&self, req: http::Request<Vec<u8>>) -> Result<(u16, Vec<u8>)> {
        let (status, body, _headers) = self.run_h(req)?;
        Ok((status, body))
    }

    /// Like [`run`](Self::run) but also returns the response headers — GCS reads the object's
    /// generation (`x-goog-generation`, its CAS token) from a download response.
    pub(crate) fn run_h(&self, req: http::Request<Vec<u8>>) -> Result<(u16, Vec<u8>, HeaderMap)> {
        // Same dropped-pooled-connection retry as the S3 paths. `http::Request` is not
        // `Clone`, so rebuild the one field the replay needs rather than cloning the request.
        finish(with_one_retry(|| {
            let mut replay = http::Request::builder().method(req.method()).uri(req.uri());
            for (name, value) in req.headers() {
                replay = replay.header(name, value);
            }
            let replay = replay
                .body(req.body().clone())
                .expect("rebuilding a request that already parsed");
            self.agent.run(replay)
        })?)
    }
}

/// Read a response into `(status, body, headers)`. Headers are captured before the body is
/// consumed (the version-token readers need them; most callers drop them).
fn finish(res: Response<Body>) -> Result<(u16, Vec<u8>, HeaderMap)> {
    let status = res.status().as_u16();
    let headers = res.headers().clone();
    let body = res
        .into_body()
        .read_to_vec()
        .context("read HTTP response body")?;
    Ok((status, body, headers))
}

fn net_err(e: ureq::Error) -> anyhow::Error {
    anyhow!("HTTP request failed: {e}")
}
