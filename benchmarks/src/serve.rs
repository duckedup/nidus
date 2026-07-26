//! A live `nidus serve` child process, for benchmarks that measure the HTTP path.
//!
//! Extracted from [`crate::engines::server`] so the write-path decomposition
//! (the `nidus-bench-write` binary) drives the server through the *same* spawn,
//! readiness, and POST code rather than a second copy of it. The engine adapter is a
//! [`VectorStore`](crate::VectorStore) impl on top of this; the write bench needs a
//! different shape entirely (its own batch sizes, its own flags, timing split around the
//! socket call), so what they share is the process, not the interface.
//!
//! Not shared with `tests/e2e/harness.rs` despite the overlap: that lives in another
//! crate's test target (unreachable from here), and its contract is deliberately
//! different — it panics with the child's stderr attached, which is right for an
//! assertion and wrong for a benchmark that must return `Result` and stay quiet.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

/// A running `nidus serve`, killed when dropped.
pub struct ServeProcess {
    child: Child,
    /// `http://127.0.0.1:<port>` — the port the child actually bound.
    pub base: String,
    /// The store directory the child was pointed at.
    pub dir: PathBuf,
    /// Shared client. `ureq::Agent` is `Clone + Send + Sync`, so a concurrency sweep can
    /// hand a clone to each thread and reuse the connection pool.
    pub agent: ureq::Agent,
}

impl ServeProcess {
    /// Where the `nidus` binary is. `NIDUS_BIN` wins; otherwise prefer a release build,
    /// since benchmarking a debug binary measures nothing useful.
    pub fn binary() -> Result<PathBuf> {
        if let Ok(p) = std::env::var("NIDUS_BIN") {
            let p = PathBuf::from(p);
            if p.is_file() {
                return Ok(p);
            }
            bail!("NIDUS_BIN={} is not a file", p.display());
        }
        for candidate in ["target/release/nidus", "target/debug/nidus"] {
            let p = PathBuf::from(candidate);
            if p.is_file() {
                if candidate.contains("debug") {
                    eprintln!(
                        "warning: benchmarking the DEBUG nidus binary — build with \
                         `cargo build --release --features cli` for meaningful numbers"
                    );
                }
                return Ok(p);
            }
        }
        bail!(
            "no nidus binary found — build one with `cargo build --release --features cli` \
             or set NIDUS_BIN (see `just bench-server`)"
        )
    }

    /// Spawn `nidus serve --dir <dir> --dim <dim>` on an ephemeral port and wait until it
    /// reports ready. `extra` is appended verbatim, for per-benchmark flags.
    pub fn spawn(dir: &Path, dim: usize, extra: &[&str]) -> Result<Self> {
        let mut child = Command::new(Self::binary()?)
            .arg("serve")
            .arg("--dir")
            .arg(dir)
            .args(["--dim", &dim.to_string()])
            // Port 0 so parallel or repeated runs never collide.
            .args(["--addr", "127.0.0.1:0"])
            // A million-vector cell in large batches still makes sizeable bodies.
            .args(["--max-body-bytes", &(512 * 1024 * 1024).to_string()])
            .args(extra)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawn nidus serve")?;

        // The bound address is on the first stderr line.
        let pipe = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("nidus serve stderr not piped"))?;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                if let Some((_, rest)) = line.split_once("http://")
                    && let Some(addr) = rest.split_whitespace().next()
                {
                    let _ = tx.send(addr.to_string());
                }
            }
        });
        let addr = match rx.recv_timeout(STARTUP_TIMEOUT) {
            Ok(addr) => addr,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                bail!("nidus serve never reported an address ({e})");
            }
        };

        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .http_status_as_error(false)
                // Ingesting a large cell is one long request; do not time it out.
                .timeout_global(Some(Duration::from_secs(600)))
                .build(),
        );
        let proc = ServeProcess {
            child,
            base: format!("http://{addr}"),
            dir: dir.to_path_buf(),
            agent,
        };

        // Wait for the store to be OPEN, not merely for the process to be alive.
        //
        // `/ready` rather than `/health`: since nidus-abx.1 the liveness probe answers as
        // soon as the router is up and deliberately does *not* gate on the store, so a
        // first request racing the open gets a `503 store is not open yet`. Readiness is
        // the probe that means "will serve traffic", which is exactly the precondition a
        // benchmark needs before it starts a clock.
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        loop {
            if let Ok(res) = proc.agent.get(format!("{}/ready", proc.base)).call()
                && res.status().as_u16() == 200
            {
                return Ok(proc);
            }
            if Instant::now() >= deadline {
                bail!("nidus serve bound {} but /ready never answered", proc.base);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// POST a JSON value, encoding it here.
    pub fn post(&self, path: &str, body: &Value) -> Result<Value> {
        self.post_bytes(path, &serde_json::to_vec(body)?)
    }

    /// POST an already-encoded body.
    ///
    /// Separate from [`post`](Self::post) so a caller can keep client-side JSON encoding
    /// *outside* the timed section — which is the whole point of the write-path
    /// decomposition, where encode and transport are the two costs being told apart.
    pub fn post_bytes(&self, path: &str, bytes: &[u8]) -> Result<Value> {
        post_on(&self.agent, &self.base, path, bytes)
    }

    /// One unlabelled sample from `GET /metrics`.
    ///
    /// A benchmark that reports only wall clock can say a change made things faster; it
    /// cannot say *why*. Reading the server's own counters lets a run attribute the number
    /// to the mechanism — e.g. how many writes actually shared a barrier (nidus-xb9.1)
    /// rather than inferring it from the shape of the curve.
    pub fn metric(&self, name: &str) -> Result<f64> {
        let text = self
            .agent
            .get(format!("{}/metrics", self.base))
            .call()
            .context("scrape /metrics")?
            .into_body()
            .read_to_string()?;
        text.lines()
            .find_map(|l| {
                l.strip_prefix(name)
                    .filter(|rest| rest.starts_with(' '))
                    .and_then(|rest| rest.trim().parse().ok())
            })
            .with_context(|| format!("no sample named {name} in the scrape"))
    }
}

/// [`ServeProcess::post_bytes`] against a borrowed agent, for worker threads that hold a
/// clone of the agent rather than the process itself.
pub fn post_on(agent: &ureq::Agent, base: &str, path: &str, bytes: &[u8]) -> Result<Value> {
    let res = agent
        .post(format!("{base}{path}"))
        .header("content-type", "application/json")
        .send(bytes)
        .with_context(|| format!("POST {path}"))?;
    let status = res.status().as_u16();
    let body = res.into_body().read_to_vec()?;
    if status != 200 {
        bail!(
            "POST {path} -> {status}: {}",
            String::from_utf8_lossy(&body)
        );
    }
    Ok(serde_json::from_slice(&body).unwrap_or(Value::Null))
}

impl Drop for ServeProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
