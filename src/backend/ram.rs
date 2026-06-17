//! [`LocalRam`]: the trivial [`MemoryTier`] — the working set *is* the process heap,
//! nothing shared. It is the default and the behavioural baseline for the sharing
//! tiers (Redis/Valkey/Memcached, Phase 2): same load/store contract, just no other
//! process can see it. `ttl` is ignored — local RAM never evicts.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Result;

use super::MemoryTier;

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
