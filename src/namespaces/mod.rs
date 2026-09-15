//! Maps a namespace name to its own [`Nidus`] store under one base location (a local
//! directory or an object-store prefix), opened lazily on first access and held in a warm
//! set bounded by a byte budget (nidus-pcpc.1). `Store::open`/`Store::open_with`
//! (`src/store/mod.rs:287`/`:309`) are unchanged: this is a handle in front of them, not a
//! new backend.

use std::path::PathBuf;

use anyhow::{Result, bail};

use crate::{Config, Nidus};

mod warm;
pub use warm::Handle;
use warm::WarmSet;

#[cfg(test)]
mod tests;

/// Default warm-set byte budget: 1 GiB.
const DEFAULT_BUDGET_BYTES: u64 = 1 << 30;

/// Redis-family memory-tier schemes whose `?prefix=` needs a per-namespace rewrite.
const REDIS_SCHEMES: [&str; 6] = ["redis", "rediss", "valkey", "valkeys", "keydb", "dragonfly"];

/// Maps a namespace name to its own [`Nidus`] store under `base`, opened lazily and held
/// in a warm set bounded by a byte budget.
pub struct Namespaces {
    template: Config,
    base: String,
    warm: WarmSet,
    budget: u64,
}

impl Namespaces {
    /// Handle over `base` (a local directory or an object-store prefix), deriving each
    /// namespace's `Config` from `template`.
    pub fn new(template: Config, base: impl Into<String>) -> Self {
        Self {
            template,
            base: base.into(),
            warm: WarmSet::new(),
            budget: DEFAULT_BUDGET_BYTES,
        }
    }

    /// Byte budget for the warm set (default 1 GiB). A store's size is unknowable before
    /// it opens, so one larger than `bytes` alone is a transient overshoot, not a refusal.
    pub fn budget_bytes(mut self, bytes: u64) -> Self {
        self.budget = bytes;
        self
    }

    /// A shared handle to `name`'s store, opening it on first access. Admitting it may
    /// evict other warm namespaces (never `name` itself) to fit the byte budget. Owned, not
    /// borrowed: the caller can drop its own guard over this set before using the store.
    pub fn get(&mut self, name: &str) -> Result<Handle> {
        validate_name(name)?;
        if !self.warm.contains(name) {
            let cfg = derive_config(&self.template, &self.base, name);
            self.warm.admit(name.to_string(), Nidus::open(cfg)?);
        }
        // Re-measures every warm store before evicting: sizes taken once at admission
        // would all be 0, since a namespace is opened before it is written to.
        self.warm.enforce(name, self.budget, self.writable())?;
        Ok(self
            .warm
            .handle(name)
            .expect("just admitted or already warm"))
    }

    /// Namespace names currently warm. Opens nothing.
    pub fn warm(&self) -> Vec<String> {
        self.warm.names()
    }

    /// Warm namespaces with their last-measured byte size. Opens nothing — this is what
    /// makes the byte budget observable (the `nidus serve` namespace-listing route) without
    /// perturbing it.
    pub fn warm_entries(&self) -> Vec<(String, u64)> {
        self.warm.entries()
    }

    /// Flush `name` and drop it from the warm set, releasing its writer lock via `Drop`.
    /// `Ok(true)` if it was warm. The flush is what keeps an `Fsync::OnFlush` caller from
    /// losing writes it made through the handle.
    pub fn evict(&mut self, name: &str) -> Result<bool> {
        self.warm.remove(name, self.writable())
    }

    /// Bytes currently held by the warm set, as of the last `get`.
    pub fn warm_bytes(&self) -> u64 {
        self.warm.total_bytes()
    }

    /// Whether these namespaces may be written. A `ReadOnly` store refuses `flush`
    /// (`check_writable`), so eviction must not attempt one.
    fn writable(&self) -> bool {
        self.template.open_mode == crate::config::OpenMode::ReadWrite
    }
}

/// Reject empty, `.`, `..`, and anything outside `[A-Za-z0-9._-]` — one place so a local
/// directory and an object-store prefix can never diverge on what a name means.
fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("namespace name must not be empty");
    }
    if name == "." || name == ".." {
        bail!("namespace name {name:?} is reserved");
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        bail!("namespace name {name:?} must match [A-Za-z0-9._-]");
    }
    Ok(())
}

/// Derive `name`'s own `Config` from `template`: its location under `base`, `mmap` on
/// unless pinned explicitly, and (for a Redis-family memory tier) its own `?prefix=` so
/// same-shaped namespaces never adopt each other's rows (`src/store/memtier.rs:77-86`).
fn derive_config(template: &Config, base: &str, name: &str) -> Config {
    let mut cfg = template.clone();
    if template.persistence.is_empty() {
        cfg.path = PathBuf::from(base).join(name);
    } else {
        cfg.persistence = format!("{base}/{name}");
    }
    if template.to_profile().mmap.is_none() {
        cfg.mmap = true;
    }
    if let Some(rewritten) = namespaced_memory_url(&template.memory, name) {
        cfg.memory = rewritten;
    }
    cfg
}

/// Rewrite `memory`'s `?prefix=` so `name` gets its own key namespace; `None` for a
/// non-Redis-family scheme (`local`/`ram`/empty need no shared tier and are untouched).
fn namespaced_memory_url(memory: &str, name: &str) -> Option<String> {
    let scheme = memory.split("://").next()?;
    if !REDIS_SCHEMES.iter().any(|s| s.eq_ignore_ascii_case(scheme)) {
        return None;
    }
    let (base, query) = match memory.split_once('?') {
        Some((b, q)) => (b, Some(q)),
        None => (memory, None),
    };
    let mut found_prefix = false;
    let mut params: Vec<String> = Vec::new();
    if let Some(q) = query {
        for pair in q.split('&').filter(|p| !p.is_empty()) {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            if k == "prefix" {
                found_prefix = true;
                params.push(format!("prefix={v}:{name}"));
            } else {
                params.push(pair.to_string());
            }
        }
    }
    if !found_prefix {
        params.push(format!("prefix={name}"));
    }
    Some(format!("{base}?{}", params.join("&")))
}
