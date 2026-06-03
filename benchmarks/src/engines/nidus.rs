//! nidus adapter — the reference engine. Uses the public API only.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use nidus::{Config, Nidus, Record, SearchOpts};

use crate::VectorStore;
use crate::metrics::disk_bytes;

const COLLECTION: &str = "bench";

pub struct NidusEngine {
    db: Nidus,
    dir: PathBuf,
    dim: usize,
}

impl VectorStore for NidusEngine {
    const NAME: &'static str = "nidus";

    fn create(dim: usize, dir: &Path) -> Result<Self> {
        // A subdirectory so disk measurement covers exactly the nidus store.
        let store_dir = dir.join("nidus-store");
        let mut db = Nidus::open(Config::new(&store_dir, dim))?;
        db.create_collection(COLLECTION)?;
        Ok(Self {
            db,
            dir: store_dir,
            dim,
        })
    }

    fn ingest(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        let dim = self.dim;
        let records: Vec<Record> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| Record {
                id: id.to_string(),
                vector: vectors[i * dim..(i + 1) * dim].to_vec(),
                attrs: BTreeMap::new(),
            })
            .collect();
        self.db.upsert(COLLECTION, &records)?;
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>> {
        let opts = SearchOpts {
            top_k,
            ..Default::default()
        };
        let hits = self.db.search(COLLECTION, query, &opts)?;
        hits.into_iter()
            .map(|h| Ok((h.id.parse::<u64>()?, h.score)))
            .collect()
    }

    fn disk_bytes(&self) -> u64 {
        disk_bytes(&self.dir)
    }
}
