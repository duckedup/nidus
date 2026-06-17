//! [`LocalRam`]: the trivial [`MemoryTier`] — the working set *is* the process heap,
//! nothing shared. It is the default and the behavioural baseline for the sharing
//! tiers (Redis/Valkey/Memcached, Phase 2): same load/store contract, just no other
//! process can see it. `ttl` is ignored — local RAM never evicts.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Result, bail};

use super::{Appender, MemoryTier};

/// An in-process memory tier backed by a `Mutex<HashMap>`. Shared between threads of
/// one process (so it composes with `Arc<RwLock<Nidus>>`), but never across processes.
#[derive(Default)]
pub struct LocalRam {
    objects: Mutex<HashMap<String, Vec<u8>>>,
}

impl LocalRam {
    /// A fresh, empty tier.
    pub fn new() -> LocalRam {
        LocalRam::default()
    }
}

impl MemoryTier for LocalRam {
    fn load(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let map = self
            .objects
            .lock()
            .map_err(|_| anyhow::anyhow!("local memory tier lock poisoned"))?;
        Ok(map.get(key).cloned())
    }

    fn store(&self, key: &str, bytes: &[u8], _ttl: Option<Duration>) -> Result<()> {
        let mut map = self
            .objects
            .lock()
            .map_err(|_| anyhow::anyhow!("local memory tier lock poisoned"))?;
        map.insert(key.to_string(), bytes.to_vec());
        Ok(())
    }
}

/// An in-RAM [`Appender`] over a `Vec<u8>` — the backing for an in-memory store's
/// `log` (no files, no fsync). Same append/truncate/rewrite contract as the file
/// appender, so the segment code is identical whether backed by a file or RAM.
#[derive(Default)]
pub(crate) struct MemAppender {
    buf: Vec<u8>,
}

impl MemAppender {
    pub(crate) fn new() -> MemAppender {
        MemAppender::default()
    }
}

impl Appender for MemAppender {
    fn len(&self) -> Result<u64> {
        Ok(self.buf.len() as u64)
    }

    fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let start = offset as usize;
        let end = start
            .checked_add(buf.len())
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| anyhow::anyhow!("read past end of in-memory appender"))?;
        buf.copy_from_slice(&self.buf[start..end]);
        Ok(())
    }

    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    fn truncate_to(&mut self, offset: u64) -> Result<()> {
        let o = offset as usize;
        if o > self.buf.len() {
            bail!(
                "truncate_to({offset}) exceeds in-memory length {}",
                self.buf.len()
            );
        }
        self.buf.truncate(o);
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }

    fn rewrite(&mut self, bytes: &[u8]) -> Result<()> {
        self.buf.clear();
        self.buf.extend_from_slice(bytes);
        Ok(())
    }
}
