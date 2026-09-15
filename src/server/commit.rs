//! Group commit for the write path (nidus-xb9.1): concurrent writes queue, the first to
//! reach the store applies the whole queue under one exclusive guard, and one barrier covers
//! them all. No timed window, so a lone write still pays exactly what it did. SPEC §6.4.
//!
//! A [`Committer`] is one queue plus one leadership flag. In namespaced mode (nidus-pcpc.2)
//! every namespace gets its OWN `Committer` — see [`Target`] for why sharing one across
//! namespaces would be cross-tenant data corruption, not merely a missed optimisation.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::anyhow;
use tokio::sync::oneshot;

use crate::Nidus;

use super::registry::Registry;

/// What a group's leader commits against: the single process-wide store, or one namespace
/// inside a [`Registry`]. Each namespace owns its own [`Committer`] so a leader here only
/// ever sees jobs meant for this one store (see the module docs).
pub(super) enum Target {
    Single(Arc<RwLock<Option<Nidus>>>),
    Namespace {
        registry: Arc<Registry>,
        name: String,
    },
}

impl Target {
    /// Exclusive store access for the leader, or `None` if it cannot be reached — the group
    /// is then dropped, and its members recover via `submit`'s oneshot fallback.
    fn with_store<T>(&self, f: impl FnOnce(&mut Nidus) -> T) -> Option<T> {
        match self {
            Target::Single(db) => {
                let mut guard = db.write().ok()?;
                let db = guard.as_mut()?;
                Some(f(db))
            }
            Target::Namespace { registry, name } => registry.write(name, f).ok(),
        }
    }
}

/// Deliver one queued write's answer, now that the group's barrier has been resolved.
type Ack = Box<dyn FnOnce(Option<&str>) + Send>;

/// Apply one queued write under the leader's store guard, yielding its [`Ack`] and whether
/// it now **needs** the barrier (false if it failed and rolled back, so a group of nothing
/// but failures issues no fsync at all).
type Apply = Box<dyn FnOnce(&mut Nidus) -> (Ack, bool) + Send>;

/// The write queue and its leadership flag.
pub(super) struct Committer {
    inner: Mutex<Inner>,
    /// Groups committed, and writes applied across them.
    groups: AtomicU64,
    writes: AtomicU64,
    /// Writes submitted, ever; `submitted - writes` is the current backlog. Monotonic rather than
    /// the queue's length because an atomic reads without taking the queue's mutex, and it does not
    /// evaporate the instant a leader drains — which makes it something a test can synchronise on.
    submitted: AtomicU64,
}

struct Inner {
    queue: VecDeque<Apply>,
    /// A leader is draining right now, and will re-check the queue before it stands down —
    /// so a write that sees this is guaranteed to be picked up without electing a second
    /// leader to compete for the same store guard.
    leader: bool,
}

impl Committer {
    pub(super) fn new() -> Arc<Committer> {
        Arc::new(Committer {
            inner: Mutex::new(Inner {
                queue: VecDeque::new(),
                leader: false,
            }),
            groups: AtomicU64::new(0),
            writes: AtomicU64::new(0),
            submitted: AtomicU64::new(0),
        })
    }

    /// `(groups committed, writes applied in them)` — see the `groups` field.
    pub(super) fn stats(&self) -> (u64, u64) {
        (
            self.groups.load(Ordering::Relaxed),
            self.writes.load(Ordering::Relaxed),
        )
    }

    /// Writes submitted, ever — the monotonic counter behind the queue-depth gauge.
    pub(super) fn submitted(&self) -> u64 {
        self.submitted.load(Ordering::Relaxed)
    }

    /// Writes submitted but not yet applied: the current write backlog.
    pub(super) fn depth(&self) -> u64 {
        self.submitted()
            .saturating_sub(self.writes.load(Ordering::Relaxed))
    }

    /// The queue mutex, recovering from a poisoned lock rather than propagating.
    fn inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Submit `f` and wait for the barrier that makes it durable.
    pub(super) async fn submit<F, T>(
        self: &Arc<Self>,
        target: Target,
        cancel: Option<crate::Cancel>,
        f: F,
    ) -> anyhow::Result<T>
    where
        F: FnOnce(&mut Nidus) -> anyhow::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = oneshot::channel::<anyhow::Result<T>>();

        // The whole job, typed at the call site and erased for the queue. The outcome is
        // captured by the `Ack` closure rather than sent immediately: that gap between
        // "applied" and "acknowledged" is where the shared barrier goes.
        let apply: Apply = Box::new(move |db: &mut Nidus| {
            let outcome = db.deferred(|db| match &cancel {
                Some(cancel) => cancel.scope(|| f(db)),
                None => f(db),
            });
            let needs_barrier = outcome.is_ok();
            let ack: Ack = Box::new(move |barrier: Option<&str>| {
                let answer = match (outcome, barrier) {
                    (Ok(value), None) => Ok(value),
                    (Ok(_), Some(reason)) => Err(anyhow!(
                        "the write was applied but its batch could not be made durable, so it \
                         is NOT acknowledged: {reason}"
                    )),
                    // Its own failure is the honest answer whatever the barrier did — it
                    // rolled back and contributed nothing to the group.
                    (Err(e), _) => Err(e),
                };
                // A dropped receiver means the client already gave up (deadline or
                // disconnect). The write still happened and is still durable; there is
                // simply nobody to tell.
                let _ = tx.send(answer);
            });
            (ack, needs_barrier)
        });

        let lead = {
            let mut inner = self.inner();
            inner.queue.push_back(apply);
            self.submitted.fetch_add(1, Ordering::Relaxed);
            // Enqueue and election in one critical section: either a leader exists that has
            // not yet made its final queue check (so it will take this job), or there is
            // none and we become it. No window in between.
            !std::mem::replace(&mut inner.leader, true)
        };

        if lead {
            let me = Arc::clone(self);
            // Detached, not awaited: the leader may keep working through later groups after
            // our own answer is ready, and this response must not wait on other people's
            // writes. `spawn_blocking` because it takes the store guard and fsyncs.
            tokio::task::spawn_blocking(move || me.drive(&target));
        }

        rx.await.unwrap_or_else(|_| {
            // The leader dropped our job without answering: the target could not be reached
            // (single-store lock poisoned/not open, or a namespace failed to admit), or the
            // leader's blocking task never ran because the runtime is shutting down.
            Err(anyhow!(
                "store is not open yet: the write could not be committed"
            ))
        })
    }

    /// Lead: commit group after group until the queue is empty, then stand down.
    fn drive(&self, target: &Target) {
        // Backstop for an unwind out of `commit_group`: a panicking write must not leave the queue
        // leaderless, which would hang every write behind it. Disarmed on the orderly exit below,
        // where standing down happens under the same lock as the emptiness check behind it.
        let mut standdown = StandDown { c: Some(self) };
        loop {
            let group: Vec<Apply> = {
                let mut inner = self.inner();
                if inner.queue.is_empty() {
                    // Atomic with the check: a write arriving after this sees `leader == false`
                    // and elects itself, so nothing is ever left unattended.
                    inner.leader = false;
                    standdown.c = None;
                    return;
                }
                inner.queue.drain(..).collect()
            };
            self.commit_group(target, group);
        }
    }

    /// Apply one group under a single exclusive store guard, then one barrier for all of it.
    fn commit_group(&self, target: &Target, group: Vec<Apply>) {
        let groups = &self.groups;
        let writes = &self.writes;
        // `None` means the store could not be reached; the group's jobs are dropped and
        // recover through `submit`'s oneshot fallback, same as an unavailable guard before.
        let ran = target.with_store(move |nidus| {
            groups.fetch_add(1, Ordering::Relaxed);
            writes.fetch_add(group.len() as u64, Ordering::Relaxed);

            // Phase 1: apply every write, each with its own barrier deferred.
            let mut acks: Vec<Ack> = Vec::with_capacity(group.len());
            let mut needs_barrier = false;
            for apply in group {
                let (ack, needed) = apply(nidus);
                needs_barrier |= needed;
                acks.push(ack);
            }

            // Phase 2: one barrier for the group. Skipped entirely when every write in it
            // failed and rolled back — there is nothing to make durable.
            let barrier = if needs_barrier {
                nidus.commit().err().map(|e| format!("{e:#}"))
            } else {
                None
            };
            (acks, barrier)
        });

        // Phase 3: answer, outside the store guard (`with_store` already released it), so a
        // slow client cannot hold the store lock.
        if let Some((acks, barrier)) = ran {
            for ack in acks {
                ack(barrier.as_deref());
            }
        }
    }
}

/// Clears the leadership flag if [`Committer::drive`] unwinds. See its comment.
struct StandDown<'a> {
    c: Option<&'a Committer>,
}

impl Drop for StandDown<'_> {
    fn drop(&mut self) {
        if let Some(c) = self.c {
            c.inner().leader = false;
        }
    }
}
