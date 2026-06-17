//! Shared blocking HTTP for the object-store backends (S3/GCS).
//!
//! A thin [`ureq`] wrapper that returns `(status, body)` and **never** treats a non-2xx
//! response as an error — the callers map status themselves (404 → `None`, etc.). TLS
//! is ureq's default rustls + `ring`, self-contained via `webpki-roots` (no system cert
//! store, no OpenSSL). The sans-IO clients (`rusty-s3`/`tame-gcs`) build and sign the
//! requests; this only executes them.

use anyhow::{Context, Result, anyhow};
use http::Response;
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
        finish(self.agent.get(url).call().map_err(net_err)?)
    }

    /// `PUT url` with `body`.
    pub(crate) fn put(&self, url: &str, body: &[u8]) -> Result<(u16, Vec<u8>)> {
        finish(self.agent.put(url).send(body).map_err(net_err)?)
    }

    /// `DELETE url`.
    pub(crate) fn delete(&self, url: &str) -> Result<(u16, Vec<u8>)> {
        finish(self.agent.delete(url).call().map_err(net_err)?)
    }

    /// Run a fully-built request (method + uri + headers + body) — used by GCS, whose
    /// sans-IO client (and `tame-oauth`) emit [`http::Request`]s.
    pub(crate) fn run(&self, req: http::Request<Vec<u8>>) -> Result<(u16, Vec<u8>)> {
        finish(self.agent.run(req).map_err(net_err)?)
    }
}

/// Read a response into `(status, body)`.
fn finish(res: Response<Body>) -> Result<(u16, Vec<u8>)> {
    let status = res.status().as_u16();
    let body = res
        .into_body()
        .read_to_vec()
        .context("read HTTP response body")?;
    Ok((status, body))
}

fn net_err(e: ureq::Error) -> anyhow::Error {
    anyhow!("HTTP request failed: {e}")
}
