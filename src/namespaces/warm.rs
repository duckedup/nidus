//! Warm-set accounting and eviction (nidus-pcpc.1): each namespace's open store plus its
//! byte footprint and access order, evicted least-recently-used until a byte budget is
//! met. An evicted store is flushed before it drops, so an `Fsync::OnFlush` caller does
//! not lose the writes it made through the handle; the drop then releases its writer lock
//! via `WriteLock`/`ObjectLock`/`ClusterLease`, which need no unlock call of their own.

use std::collections::HashMap;

use anyhow::Result;

use crate::Nidus;

/// One warm namespace: its open store, its byte size as of the last refresh, and a
/// monotonic access stamp (never a wall clock — cheap, immune to clock skew) for LRU.
struct Entry {
    store: Nidus,
    size: u64,
    stamp: u64,
}

/// `vector_bytes + filter_index_bytes` from `Nidus::footprint` — the one byte-counting
/// scheme the warm set uses, not a second one. Cheap: live counters, no IO.
fn footprint_bytes(n: &Nidus) -> u64 {
    let fp = n.footprint();
    fp.vector_bytes + fp.filter_index_bytes
}

/// The set of currently-open namespace stores. Pure accounting; opening a store and
/// deriving its `Config` are `Namespaces`'s job, not this one's.
#[derive(Default)]
pub(super) struct WarmSet {
    entries: HashMap<String, Entry>,
    clock: u64,
}

impl WarmSet {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Touches `name`'s LRU stamp and returns its store, if warm.
    pub(super) fn get_mut(&mut self, name: &str) -> Option<&mut Nidus> {
        self.clock += 1;
        let stamp = self.clock;
        let entry = self.entries.get_mut(name)?;
        entry.stamp = stamp;
        Some(&mut entry.store)
    }

    pub(super) fn names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Flushes `name` (when `writable`) and drops it, releasing any writer lock it held.
    pub(super) fn remove(&mut self, name: &str, writable: bool) -> Result<bool> {
        match self.entries.remove(name) {
            // Removed first, so a failed flush still frees the lock rather than stranding
            // a store nobody can reach; the error still reaches the caller.
            Some(mut entry) => {
                if writable {
                    entry.store.flush()?;
                }
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub(super) fn total_bytes(&self) -> u64 {
        self.entries.values().map(|e| e.size).sum()
    }

    /// Inserts a freshly opened `name`. Sizing and budget enforcement are `enforce`'s job:
    /// a store is measured at 0 rows here and only grows once the caller writes to it.
    pub(super) fn admit(&mut self, name: String, store: Nidus) {
        self.clock += 1;
        let stamp = self.clock;
        let size = footprint_bytes(&store);
        self.entries.insert(name, Entry { store, size, stamp });
    }

    /// Re-measures every warm store, then evicts LRU entries (never `keep`) until the
    /// total fits `budget`. Re-measuring is load-bearing: a namespace opens at 0 and grows
    /// only once written through, so sizing once at admission would bound nothing.
    pub(super) fn enforce(&mut self, keep: &str, budget: u64, writable: bool) -> Result<()> {
        for entry in self.entries.values_mut() {
            entry.size = footprint_bytes(&entry.store);
        }
        while self.total_bytes() > budget {
            let victim = self
                .entries
                .iter()
                .filter(|(k, _)| k.as_str() != keep)
                .min_by_key(|(_, e)| e.stamp)
                .map(|(k, _)| k.clone());
            let Some(victim) = victim else { break };
            self.remove(&victim, writable)?;
        }
        Ok(())
    }
}
