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

impl Http {
    pub(crate) fn new() -> Http {
        // `http_status_as_error(false)`: a 4xx/5xx comes back as a normal response so a
        // backend can distinguish 404 (absent → `None`) from a transport failure.
        let config = ureq::config::Config::builder()
            .http_status_as_error(false)
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
        finish(self.agent.get(url).call().map_err(net_err)?)
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
        let mut req = self.agent.put(url);
        for (name, value) in headers {
            req = req.header(*name, *value);
        }
        finish(req.send(body).map_err(net_err)?)
    }

    /// `DELETE url`.
    pub(crate) fn delete(&self, url: &str) -> Result<(u16, Vec<u8>)> {
        let (status, body, _headers) = finish(self.agent.delete(url).call().map_err(net_err)?)?;
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
        finish(self.agent.run(req).map_err(net_err)?)
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
