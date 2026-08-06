//! Group commit for the write path (nidus-xb9.1).
//!
//! ## The measurement this exists to move
//!
//! `just bench-write` put a number on the write ceiling: **~7.6ms of fixed cost per upsert
//! call**, paid whether that call carries one record or a thousand — a disk barrier
//! (`fsync` of `data`, then `log`) plus, in cluster mode, a manifest publish. It is why
//! `batch=1` gets 125 vec/s over HTTP while `batch=1000` gets 33k from the same code path.
//!
//! A single `nidus serve` already has 2–8 upserts genuinely in flight at once, and the
//! concurrency sweep plateaus at ~85k vec/s (384-d) because **every one of them pays its own
//! barrier**. The plateau sits exactly on the in-process durable rate, so the exclusive write
//! section is the asymptote, and merging the barriers is how that asymptote moves.
//!
//! ## The shape
//!
//! Classic WAL group commit. Writes are submitted to a queue instead of racing for the store
//! lock. Whichever request finds no leader becomes one: it takes the store's exclusive guard
//! **once**, applies every write queued at that moment with the barrier deferred
//! ([`Nidus::deferred`]), then takes **one** barrier for the whole group
//! ([`Nidus::commit`]) and only then answers all of them.
//!
//! ```text
//!   without group commit          with group commit
//!   ────────────────────          ─────────────────
//!   append ─ fsync ─ 200          append ┐
//!            append ─ fsync ─ 200 append ├─ fsync ─ 200 ×N
//!                     append ─ …  append ┘
//! ```
//!
//! Every existing invariant holds. The durable write order is untouched (each batch appends
//! data before log; the barrier syncs data before log). No request is answered before its own
//! bytes are durable — that is the entire point of splitting apply from acknowledge, and the
//! reason [`Nidus::deferred`] documents the obligation rather than hiding it. A failed barrier
//! fails *every* member of its group, because none of them can honestly be called durable.
//!
//! ## Why no window, no timer, no wait
//!
//! The classic mistake in a group-commit implementation is to *wait* for a group to form —
//! which buys throughput under load by taxing every single-client write with a delay it
//! gains nothing from. There is no timer here. The leader drains whatever is **already**
//! queued and goes; with one client that is always a group of one, and the path is the same
//! append-then-barrier it was before, minus a queue push. Batching only ever happens because
//! work was genuinely waiting.
//!
//! ## Leadership, and why it is a flag rather than a thread
//!
//! A dedicated commit thread was the alternative. This is less machinery for the same
//! behaviour: no lifecycle to own, nothing to shut down, and no thread sitting idle in a
//! read-only instance. The election is one `bool` under the queue's mutex, and it is
//! *elected together with the enqueue* — a write either finds a leader that is guaranteed to
//! come back for it, or becomes the leader itself. There is no third outcome, which is what
//! rules out the lost-wakeup this design would otherwise be prone to.
//!
//! One incidental win: under write load the blocking pool now holds roughly one task per
//! *group* instead of one per request.
//!
//! The queue is unbounded because it is already bounded — the concurrency semaphore in
//! [`super::limits`] caps store-touching requests in flight, and nothing can be queued here
//! without holding one of those permits.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::anyhow;
use tokio::sync::oneshot;

use crate::Nidus;

/// The store slot every write goes through, shared with [`super::AppState`].
type Db = Arc<RwLock<Option<Nidus>>>;

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
    /// Writes submitted, ever. `submitted - writes` is the current backlog — reported as a
    /// gauge, and the reason this is a monotonic counter rather than the queue's length: an
    /// atomic can be read by a scrape without taking the queue's mutex, and it does not
    /// evaporate the instant a leader drains the queue, which makes it something a test can
    /// synchronise on.
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
        db: Db,
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
            tokio::task::spawn_blocking(move || me.drive(&db));
        }

        rx.await.unwrap_or_else(|_| {
            // The leader dropped our job without answering: the store slot was empty or its
            // lock was poisoned (see `commit_group`), or the leader's blocking task never ran
            // because the runtime is shutting down.
            Err(anyhow!(
                "store is not open yet: the write could not be committed"
            ))
        })
    }

    /// Lead: commit group after group until the queue is empty, then stand down.
    fn drive(&self, db: &Db) {
        // Backstop for an unwind out of `commit_group` — a panicking write must not leave the
        // queue leaderless, which would hang every write that came after it. Disarmed on the
        // orderly exit below, where standing down happens under the same lock as the emptiness
        // check it is based on.
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
            self.commit_group(db, group);
        }
    }

    /// Apply one group under a single exclusive store guard, then one barrier for all of it.
    fn commit_group(&self, db: &Db, group: Vec<Apply>) {
        // A poisoned store lock means a previous write panicked mid-mutation; the store's
        // invariants are unknown, so refuse rather than write into it. Dropping the jobs
        // answers their submitters through the `rx` fallback.
        let Ok(mut guard) = db.write() else {
            return;
        };
        // Still opening, or a standby that never got the writer handle. Same fallback.
        let Some(nidus) = guard.as_mut() else {
            return;
        };

        self.groups.fetch_add(1, Ordering::Relaxed);
        self.writes.fetch_add(group.len() as u64, Ordering::Relaxed);

        // Phase 1: apply every write, each with its own barrier deferred.
        let mut acks: Vec<Ack> = Vec::with_capacity(group.len());
        let mut needs_barrier = false;
        for apply in group {
            let (ack, needed) = apply(nidus);
            needs_barrier |= needed;
            acks.push(ack);
        }

        // Phase 2: one barrier for the group. Skipped entirely when every write in it failed
        // and rolled back — there is nothing to make durable.
        let barrier = if needs_barrier {
            nidus.commit().err().map(|e| format!("{e:#}"))
        } else {
            None
        };

        // Phase 3: answer. Outside the guard, so a slow client cannot hold the store lock.
        drop(guard);
        for ack in acks {
            ack(barrier.as_deref());
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
