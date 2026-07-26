//! Cooperative cancellation for long scans (nidus-abx.2 follow-up).
//!
//! A server request deadline frees the *client*: axum drops the response future and the
//! caller gets its `504`. It does not free the *CPU* — the search is running on a
//! `spawn_blocking` task, and blocking tasks are not cancellable, so an abandoned query
//! went on paying for a full brute-force scan with nobody left to receive it. Under load
//! that is the worst possible time to be doing free work.
//!
//! Nothing outside the scan can fix that: the only place that can stop a running loop is
//! the loop. So the scan kernels check a flag, and whoever owns the request sets it.
//!
//! ## Why ambient rather than a parameter
//!
//! The obvious design is a field on [`SearchOpts`](crate::SearchOpts). It was rejected:
//! that struct is public and constructed with struct literals in ~30 places in this repo
//! alone, so a new field is a breaking change for every embedding application, in exchange
//! for a concern almost none of them have. A parallel set of `*_cancellable` methods was
//! rejected for the same reason in reverse — it doubles the search API surface forever.
//!
//! So the token is **ambient**: [`Cancel::scope`] installs it for the current thread, and
//! the kernels consult it. That is honest about the actual lifetime involved — one call, on
//! one thread — and it means every scan path is covered, including ones added later, with
//! no signature to remember to thread through.
//!
//! The cost of ambience is that it does not cross a thread boundary by itself, which
//! matters because a parallel scan fans out to worker threads. [`current`] exists for
//! exactly that: the fan-out captures the token before spawning and re-installs it in each
//! worker. That handoff is the one place this has to be got right, and it is one place.
//!
//! ## Cost
//!
//! One relaxed atomic load per [`CHECK_EVERY`] rows, and nothing at all when no token is
//! installed (a thread-local read of `None`). The check is batched rather than per-row
//! because a per-row atomic load would be a measurable tax on the tightest loop in the
//! crate, paid by every query to help the rare abandoned one.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};

/// How many rows a scan kernel processes between cancellation checks.
///
/// 1024 rows is a few microseconds of scanning even at large dimensions — fine-grained
/// enough that an abandoned query stops promptly, coarse enough that the atomic load
/// disappears into the scoring work beside it.
pub(crate) const CHECK_EVERY: usize = 1024;

/// A shared "stop what you are doing" flag.
///
/// Cloning is cheap and every clone observes the same signal. Cancellation is one-way:
/// once set it stays set, because a token that could be un-cancelled would invite a race
/// where a scan resumes work its caller has already given up on.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Cancel {
        Cancel::default()
    }

    /// Signal every holder to stop. Idempotent.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Run `f` with this token installed as the ambient cancellation signal for the
    /// current thread, restoring whatever was there before — including on unwind, which is
    /// why the restore is in a `Drop` guard rather than written after the call.
    pub fn scope<T>(&self, f: impl FnOnce() -> T) -> T {
        struct Restore(Option<Cancel>);
        impl Drop for Restore {
            fn drop(&mut self) {
                CURRENT.with(|c| *c.borrow_mut() = self.0.take());
            }
        }
        let previous = CURRENT.with(|c| c.borrow_mut().replace(self.clone()));
        let _restore = Restore(previous);
        f()
    }
}

thread_local! {
    /// The token in force on this thread, if any.
    static CURRENT: RefCell<Option<Cancel>> = const { RefCell::new(None) };
}

/// The ambient token, for handing across a thread boundary. See the module docs.
pub(crate) fn current() -> Option<Cancel> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Whether the caller has given up on this work.
pub(crate) fn cancelled() -> bool {
    CURRENT.with(|c| c.borrow().as_ref().is_some_and(Cancel::is_cancelled))
}

/// `Err` once the caller has given up, for the `?` sites in the scan kernels.
///
/// The message is deliberately distinctive: it can surface in a log line, and "the client
/// left" must not read like a store fault.
pub(crate) fn check() -> Result<()> {
    if cancelled() {
        bail!("search cancelled: the caller stopped waiting for this request");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_is_shared_by_its_clones() {
        let a = Cancel::new();
        let b = a.clone();
        assert!(!a.is_cancelled() && !b.is_cancelled());
        b.cancel();
        assert!(a.is_cancelled(), "clones observe one signal");
    }

    #[test]
    fn nothing_is_cancelled_without_a_scope() {
        assert!(!cancelled(), "no ambient token means no cancellation");
        assert!(check().is_ok());
    }

    #[test]
    fn scope_installs_and_restores() {
        let outer = Cancel::new();
        outer.scope(|| {
            assert!(!cancelled());
            let inner = Cancel::new();
            inner.cancel();
            inner.scope(|| assert!(cancelled(), "innermost token wins"));
            assert!(!cancelled(), "and the outer one is restored");
            outer.cancel();
            assert!(cancelled());
        });
        assert!(!cancelled(), "the scope does not leak past its call");
    }

    /// The restore must survive a panic, or one cancelled request would leave every later
    /// request on that worker thread permanently cancelled.
    #[test]
    fn scope_restores_on_unwind() {
        let token = Cancel::new();
        token.cancel();
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(|| token.scope(|| panic!("boom")));
        std::panic::set_hook(hook);
        assert!(caught.is_err());
        assert!(
            !cancelled(),
            "a panic inside a scope must not leave the thread cancelled"
        );
    }

    /// `current()` is what lets a parallel scan hand the token to its workers — the one
    /// place ambience needs help.
    #[test]
    fn current_can_be_carried_to_another_thread() {
        let token = Cancel::new();
        token.scope(|| {
            let carried = current().expect("a token is installed");
            let seen = std::thread::spawn(move || {
                // A fresh thread inherits nothing until the token is re-installed.
                let before = cancelled();
                carried.cancel();
                let after = carried.scope(cancelled);
                (before, after)
            })
            .join()
            .unwrap();
            assert_eq!(seen, (false, true));
            assert!(cancelled(), "and the parent sees the worker's signal");
        });
    }
}
