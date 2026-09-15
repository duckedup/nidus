//! The namespaced-mode registry (nidus-pcpc.2): `Namespaces` (nidus-pcpc.1) plus the
//! per-namespace side-tables — group commit, readiness, lease renewal — a single
//! process-wide store never needed. `Namespaces::get` takes `&mut self`, so both it and the
//! side-tables that must track it share one guard: reconciliation runs inside the very call
//! that can evict, which is what makes it exact rather than best-effort.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use tokio::task::JoinHandle;

use crate::namespaces::Handle;
use crate::{Config, Namespaces, Nidus};

use super::commit::Committer;

/// One namespace's side-tables. Its lease-renewal task is aborted when the entry is dropped
/// (eviction, or the whole registry going away) — never left renewing a lease nobody holds.
struct Side {
    committer: Arc<Committer>,
    readiness: crate::Readiness,
    lease_task: JoinHandle<()>,
}

impl Drop for Side {
    fn drop(&mut self) {
        self.lease_task.abort();
    }
}

struct Inner {
    namespaces: Namespaces,
    side: HashMap<String, Side>,
}

/// A poisoned lock means a panic left in-RAM state untrustworthy. Single-store mode fails
/// every data route in that case (`/ready` and the handlers both check), so namespaced mode
/// must refuse too rather than serving possibly-torn rows.
fn poisoned() -> anyhow::Error {
    anyhow::anyhow!(
        "store lock poisoned: a panic left this instance's in-RAM state untrustworthy — \
         it must be restarted"
    )
}

/// Many `Nidus` stores behind one byte-bounded warm set, each with its own group-commit
/// queue — see `commit::Target` for why sharing one across namespaces would corrupt data.
pub(super) struct Registry {
    inner: Mutex<Inner>,
    renew_every: Duration,
}

impl Registry {
    pub(super) fn new(
        template: Config,
        base: String,
        budget_bytes: Option<u64>,
        renew_every: Duration,
    ) -> Arc<Registry> {
        let mut namespaces = Namespaces::new(template, base);
        if let Some(b) = budget_bytes {
            namespaces = namespaces.budget_bytes(b);
        }
        Arc::new(Registry {
            inner: Mutex::new(Inner {
                namespaces,
                side: HashMap::new(),
            }),
            renew_every,
        })
    }

    fn lock(&self) -> anyhow::Result<MutexGuard<'_, Inner>> {
        self.inner.lock().map_err(|_| poisoned())
    }

    pub(super) fn is_poisoned(&self) -> bool {
        self.inner.is_poisoned()
    }

    /// Ensure `name`'s side-table entry exists, spawning its lease-renewal task on first
    /// admission. Runs under the same guard as the `get`/reconcile that precedes it.
    fn ensure_side(self: &Arc<Self>, guard: &mut Inner, name: &str, readiness: &crate::Readiness) {
        if guard.side.contains_key(name) {
            return;
        }
        let lease_task =
            spawn_namespace_lease_renewal(Arc::clone(self), name.to_string(), self.renew_every);
        guard.side.insert(
            name.to_string(),
            Side {
                committer: Committer::new(),
                readiness: readiness.clone(),
                lease_task,
            },
        );
    }

    /// Admit `name` (opening it if cold), reconcile the side-tables, and hand back its
    /// handle, committer and readiness. The guard drops before returning, so the caller holds
    /// only that one namespace's lock and tenants never serialize (nidus-pcpc.2).
    fn admit_handle(
        self: &Arc<Self>,
        name: &str,
    ) -> anyhow::Result<(Handle, Arc<Committer>, crate::Readiness)> {
        let mut guard = self.lock()?;
        let handle = guard.namespaces.get(name)?;
        let readiness = handle.read().map_err(|_| poisoned())?.readiness();
        // Reconcile here, inside the guard that just ran the only call that can evict.
        let warm = guard.namespaces.warm();
        guard.side.retain(|k, _| warm.iter().any(|w| w == k));
        self.ensure_side(&mut guard, name, &readiness);
        let side = guard.side.get(name).expect("just ensured");
        Ok((handle, side.committer.clone(), side.readiness.clone()))
    }

    /// The write seam: admit `name` and hand back its committer and readiness handle.
    pub(super) fn admit(
        self: &Arc<Self>,
        name: &str,
    ) -> anyhow::Result<(Arc<Committer>, crate::Readiness)> {
        let (_, committer, readiness) = self.admit_handle(name)?;
        Ok((committer, readiness))
    }

    /// The read seam: admit `name`, then run `f` under its own read lock. Concurrent readers
    /// of one namespace, and readers of different namespaces, all proceed in parallel.
    pub(super) fn read<T>(
        self: &Arc<Self>,
        name: &str,
        f: impl FnOnce(&Nidus) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let (handle, _, _) = self.admit_handle(name)?;
        let db = handle.read().map_err(|_| poisoned())?;
        f(&db)
    }

    /// Apply the group-commit leader's closure against `name`'s store — reached only
    /// through [`super::commit::Target::Namespace`]. Goes through `admit_handle` like every
    /// other seam, so a namespace reopened here gets its side-tables too.
    pub(super) fn write<T>(
        self: &Arc<Self>,
        name: &str,
        f: impl FnOnce(&mut Nidus) -> T,
    ) -> anyhow::Result<T> {
        let (handle, _, _) = self.admit_handle(name)?;
        let mut db = handle.write().map_err(|_| poisoned())?;
        Ok(f(&mut db))
    }

    /// Every warm namespace's name, byte size, and readiness, for the listing route. Opens
    /// nothing: `warm_entries` only reads what `enforce` already measured.
    pub(super) fn warm_snapshot(&self) -> Vec<(String, u64, Option<crate::Readiness>)> {
        let Ok(guard) = self.lock() else {
            return Vec::new();
        };
        guard
            .namespaces
            .warm_entries()
            .into_iter()
            .map(|(name, bytes)| {
                let readiness = guard.side.get(&name).map(|s| s.readiness.clone());
                (name, bytes, readiness)
            })
            .collect()
    }
}

/// Renew `name`'s writer lease on a timer, mirroring `super::spawn_lease_renewal` for the
/// single-store case. Checks the warm set before touching the store, under the same lock,
/// so an evicted namespace is never reopened just to renew a lease nobody holds.
fn spawn_namespace_lease_renewal(
    reg: Arc<Registry>,
    name: String,
    ttl: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(ttl);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let reg = reg.clone();
            let name = name.clone();
            let _ = tokio::task::spawn_blocking(move || {
                // Look the namespace up under the guard, then release it before touching
                // the store: a renewal must never hold the registry against every tenant.
                let handle = {
                    let Ok(mut guard) = reg.lock() else { return };
                    if !guard.namespaces.warm().iter().any(|w| w == &name) {
                        return;
                    }
                    let Ok(h) = guard.namespaces.get(&name) else {
                        return;
                    };
                    h
                };
                let Ok(db) = handle.read() else { return };
                let Some(lease) = db.lease_renewer() else {
                    return;
                };
                drop(db);
                renew_or_diag(&name, &lease);
            })
            .await;
        }
    })
}

/// Renew one lease and diagnose the outcome — split out only so `spawn_namespace_lease_renewal`
/// stays under the file's own comment budget, not because this is reused.
fn renew_or_diag(name: &str, lease: &crate::LeaseRenewer) {
    if let Err(e) = lease.renew() {
        if crate::backend::is_lease_lost(&e) {
            crate::diag::diag!(
                crate::diag::Level::Error,
                "lease",
                "writer lease LOST on background renewal — this namespace is fenced",
                "namespace" => name,
                "err" => format!("{e:#}"),
            );
        } else {
            crate::diag::diag!(
                crate::diag::Level::Warn,
                "lease",
                "background lease renewal failed transiently, will retry",
                "namespace" => name,
                "err" => format!("{e:#}"),
            );
        }
    }
}
