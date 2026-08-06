//! `nidus serve` adapter — the SAME nidus, reached over HTTP instead of in-process.

use std::path::Path;

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use crate::VectorStore;
use crate::metrics::disk_bytes;
use crate::serve::ServeProcess;

const COLLECTION: &str = "bench";
/// Records per upsert request. Large enough that per-request overhead does not dominate
/// ingest, small enough that a million-vector cell never builds one enormous body.
const BATCH: usize = 1_000;

pub struct NidusServerEngine {
    proc: ServeProcess,
    dim: usize,
}

impl VectorStore for NidusServerEngine {
    const NAME: &'static str = "nidus (server)";

    fn create(dim: usize, dir: &Path) -> Result<Self> {
        let proc = ServeProcess::spawn(&dir.join("nidus-server-store"), dim, &[])?;
        proc.post(&format!("/collections/{COLLECTION}"), &json!({}))?;
        Ok(NidusServerEngine { proc, dim })
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
            let body = self.proc.post(
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
        let hits = self.proc.post(
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
        let _ = self.proc.post("/flush", &json!({}));
        disk_bytes(&self.proc.dir)
    }
}
