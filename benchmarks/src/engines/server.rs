//! `nidus serve` adapter — the SAME nidus, reached over HTTP instead of in-process.
//!
//! Every other engine here (including [`super::nidus`]) drives a library in-process, so
//! the parity table measures storage and search but says nothing about what the *server*
//! costs. Running the identical dataset through this adapter puts a "nidus (server)" row
//! beside "nidus", and the gap between them is precisely the HTTP overhead: JSON-encoding
//! `dim` floats per request, the `Arc<RwLock<Nidus>>` + `spawn_blocking` hop, and socket
//! framing.
//!
//! Needs the `nidus` binary. `env!("CARGO_BIN_EXE_nidus")` is only defined for the
//! defining package's own tests, so the path comes from `NIDUS_BIN`, falling back to the
//! usual target directories — `just bench-server` builds it first and passes the path.
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
use serde_json::{Value, json};

use crate::VectorStore;
use crate::metrics::disk_bytes;

const COLLECTION: &str = "bench";
/// Records per upsert request. Large enough that per-request overhead does not dominate
/// ingest, small enough that a million-vector cell never builds one enormous body.
const BATCH: usize = 1_000;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

pub struct NidusServerEngine {
    child: Child,
    base: String,
    dir: PathBuf,
    dim: usize,
    agent: ureq::Agent,
}

impl NidusServerEngine {
    /// Where the `nidus` binary is. `NIDUS_BIN` wins; otherwise prefer a release build,
    /// since benchmarking a debug binary measures nothing useful.
    fn binary() -> Result<PathBuf> {
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

    fn post(&self, path: &str, body: &Value) -> Result<Value> {
        let bytes = serde_json::to_vec(body)?;
        let res = self
            .agent
            .post(format!("{}{path}", self.base))
            .header("content-type", "application/json")
            .send(&bytes[..])
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
}

impl VectorStore for NidusServerEngine {
    const NAME: &'static str = "nidus (server)";

    fn create(dim: usize, dir: &Path) -> Result<Self> {
        let store_dir = dir.join("nidus-server-store");
        let mut child = Command::new(Self::binary()?)
            .arg("serve")
            .arg("--dir")
            .arg(&store_dir)
            .args(["--dim", &dim.to_string()])
            // Port 0 so parallel or repeated runs never collide.
            .args(["--addr", "127.0.0.1:0"])
            // A million-vector cell in batches of 1000 still makes sizeable bodies.
            .args(["--max-body-bytes", &(512 * 1024 * 1024).to_string()])
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
        let engine = Self {
            child,
            base: format!("http://{addr}"),
            dir: store_dir,
            dim,
            agent,
        };

        // Wait for the store to be open and the router live, not merely bound.
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        loop {
            if let Ok(res) = engine.agent.get(format!("{}/health", engine.base)).call()
                && res.status().as_u16() == 200
            {
                break;
            }
            if Instant::now() >= deadline {
                bail!(
                    "nidus serve bound {} but /health never answered",
                    engine.base
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        }

        engine.post(&format!("/collections/{COLLECTION}"), &json!({}))?;
        Ok(engine)
    }

    fn ingest(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        let dim = self.dim;
        for (b, chunk) in ids.chunks(BATCH).enumerate() {
            let records: Vec<Value> = chunk
                .iter()
                .enumerate()
                .map(|(j, id)| {
                    let row = b * BATCH + j;
                    json!({
                        "id": id.to_string(),
                        "vector": &vectors[row * dim..(row + 1) * dim],
                        "attrs": {}
                    })
                })
                .collect();
            let n = records.len();
            let body = self.post(
                &format!("/collections/{COLLECTION}/upsert"),
                &json!({"records": records}),
            )?;
            let upserted = body["upserted"].as_u64().unwrap_or(0) as usize;
            if upserted != n {
                bail!("upsert reported {upserted} of {n} records");
            }
        }
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>> {
        let hits = self.post(
            "/search",
            &json!({"query": query, "top_k": top_k, "collections": [COLLECTION]}),
        )?;
        hits.as_array()
            .ok_or_else(|| anyhow!("search did not return an array: {hits}"))?
            .iter()
            .map(|h| {
                let id = h["id"]
                    .as_str()
                    .ok_or_else(|| anyhow!("hit without an id: {h}"))?
                    .parse::<u64>()?;
                let score = h["score"]
                    .as_f64()
                    .ok_or_else(|| anyhow!("hit without a score: {h}"))?
                    as f32;
                Ok((id, score))
            })
            .collect()
    }

    fn disk_bytes(&self) -> u64 {
        // Ask the server to flush first: unlike the in-process engine, buffered writes
        // here are on the other side of a socket, and an unflushed store would measure
        // small for the wrong reason.
        let _ = self.post("/flush", &json!({}));
        disk_bytes(&self.dir)
    }
}

impl Drop for NidusServerEngine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
