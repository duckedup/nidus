//! The crate-internal query-plan recorder (nidus-cvz): a zero-cost `Off` variant plus an
//! `On` variant instrumented sites populate. `Store::search`/`search_similar`/`hybrid_search`
//! thread a `&mut PlanRec` through their inner bodies; `finish` hands back the public
//! [`crate::plan::QueryPlan`] the `*_with_plan` methods return.

use std::time::{Duration, Instant};

use anyhow::Result;

use super::Store;
use crate::plan::{Candidates, Narrowing, QueryPath, QueryPlan, Timings};

/// Which [`Timings`] field a timed block feeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Narrow,
    Gather,
    Walk,
    Resolve,
    FirstPass,
    Rescore,
    Score,
}

/// Accumulates one query's plan. Lives only inside [`PlanRec::On`], so its fields are
/// touched only when recording is actually on.
#[derive(Default)]
pub(crate) struct Rec {
    path: Option<QueryPath>,
    rows_scanned: Option<u64>,
    candidates: Option<Candidates>,
    narrowing: Option<Narrowing>,
    timings: Timings,
}

/// The recorder every search path threads through. `Off` is the default and must allocate
/// nothing and take no clock reading — every method below is `#[inline]` and falls straight
/// through on that arm, which is what keeps an unset `SearchOpts::plan` free.
pub(crate) enum PlanRec {
    Off,
    On(Box<Rec>),
}

impl PlanRec {
    #[inline]
    pub(crate) fn new(on: bool) -> Self {
        if on {
            Self::On(Box::default())
        } else {
            Self::Off
        }
    }

    #[inline]
    pub(crate) fn is_on(&self) -> bool {
        matches!(self, Self::On(_))
    }

    #[inline]
    pub(crate) fn path(&mut self, p: QueryPath) {
        if let Self::On(r) = self {
            r.path = Some(p);
        }
    }

    #[inline]
    pub(crate) fn rows_scanned(&mut self, n: u64) {
        if let Self::On(r) = self {
            r.rows_scanned = Some(n);
        }
    }

    #[inline]
    pub(crate) fn narrowing(&mut self, n: Narrowing) {
        if let Self::On(r) = self {
            r.narrowing = Some(n);
        }
    }

    /// A mutable handle to the accumulated candidate counts, created on first touch.
    #[inline]
    pub(crate) fn candidates(&mut self) -> Option<&mut Candidates> {
        match self {
            Self::Off => None,
            Self::On(r) => Some(r.candidates.get_or_insert_with(Candidates::default)),
        }
    }

    /// Time `f`, recording into the phase `slot` picks out. Runs `f` untimed when `Off` —
    /// the one `Instant::now()` that must never fire on the disabled path.
    #[inline]
    pub(crate) fn phase<T>(&mut self, slot: Phase, f: impl FnOnce() -> T) -> T {
        match self {
            Self::Off => f(),
            Self::On(r) => {
                let start = Instant::now();
                let out = f();
                r.timings.set(slot, start.elapsed());
                out
            }
        }
    }

    /// Open-coded timing for a block that cannot be a closure — one holding a borrow that
    /// must outlive it. `None` when `Off`, so no clock is read.
    #[inline]
    pub(crate) fn start(&self) -> Option<Instant> {
        match self {
            Self::Off => None,
            Self::On(_) => Some(Instant::now()),
        }
    }

    /// Close a [`PlanRec::start`] span into `slot`. A `None` start is a no-op.
    #[inline]
    pub(crate) fn stop(&mut self, slot: Phase, start: Option<Instant>) {
        if let (Self::On(r), Some(t)) = (&mut *self, start) {
            r.timings.set(slot, t.elapsed());
        }
    }

    /// Consume the recorder into a [`QueryPlan`], `None` on `Off`. `total` is measured by
    /// the caller around the whole traced call, since a phase closure cannot also borrow
    /// `self` to record itself.
    pub(crate) fn finish(self, total: Duration) -> Option<QueryPlan> {
        match self {
            Self::Off => None,
            Self::On(r) => Some(QueryPlan {
                path: r.path.unwrap_or(QueryPath::Exact),
                rows_scanned: r.rows_scanned,
                candidates: r.candidates,
                narrowing: r.narrowing.unwrap_or(Narrowing::Inactive),
                timings: Timings { total, ..r.timings },
            }),
        }
    }
}

impl Timings {
    fn set(&mut self, slot: Phase, d: Duration) {
        match slot {
            Phase::Narrow => self.narrow = Some(d),
            Phase::Gather => self.gather = Some(d),
            Phase::Walk => self.walk = Some(d),
            Phase::Resolve => self.resolve = Some(d),
            Phase::FirstPass => self.first_pass = Some(d),
            Phase::Rescore => self.rescore = Some(d),
            Phase::Score => self.score = Some(d),
        }
    }
}

impl Store {
    /// The one entry point every `search`/`search_with_plan` sibling pair funnels through:
    /// build the recorder (on when `plan_requested` or `NIDUS_SLOW_QUERY_MS` is set, so the
    /// env var works with no per-query opt-in), run `f`, then log-if-slow and hand back the plan.
    pub(super) fn traced<T>(
        &self,
        plan_requested: bool,
        f: impl FnOnce(&mut PlanRec) -> Result<T>,
    ) -> Result<(T, Option<QueryPlan>)> {
        let mut rec = PlanRec::new(plan_requested || crate::diag::slow_query_threshold().is_some());
        let start = rec.is_on().then(Instant::now);
        let out = f(&mut rec)?;
        let Some(start) = start else {
            return Ok((out, None));
        };
        let total = start.elapsed();
        let plan = rec.finish(total);
        if let Some(p) = &plan {
            Self::log_if_slow(total, p);
        }
        Ok((out, plan))
    }

    /// Short-form slow-query line: path, `total_us`, rows scanned, candidates. Not the whole
    /// plan — logfmt readability and the comment cap both argue for the short form.
    fn log_if_slow(total: Duration, plan: &QueryPlan) {
        if !crate::diag::is_slow(total) {
            return;
        }
        crate::diag::diag!(
            crate::diag::Level::Warn,
            "search",
            "slow query",
            "path" => format!("{:?}", plan.path),
            "total_us" => total.as_micros(),
            "rows_scanned" => plan.rows_scanned.map_or_else(String::new, |n| n.to_string()),
            "candidates" => plan.candidates.map_or_else(String::new, |c| format!("{c:?}")),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_records_nothing_and_finishes_to_none() {
        let mut rec = PlanRec::Off;
        assert!(!rec.is_on());
        // The closure still runs — only the timing is skipped.
        let ran = rec.phase(Phase::Score, || true);
        assert!(ran);
        assert!(rec.candidates().is_none());
        assert!(rec.finish(Duration::from_secs(1)).is_none());
    }

    #[test]
    fn on_accumulates_path_rows_and_phase_timing() {
        let mut rec = PlanRec::new(true);
        assert!(rec.is_on());
        rec.path(QueryPath::Quantized);
        rec.rows_scanned(42);
        rec.narrowing(Narrowing::Narrowed { candidates: 7 });
        let doubled = rec.phase(Phase::FirstPass, || 2 + 2);
        assert_eq!(doubled, 4);
        if let Some(c) = rec.candidates() {
            c.surfaced = 10;
            c.survived = 6;
        }
        let plan = rec.finish(Duration::from_micros(500)).unwrap();
        assert_eq!(plan.path, QueryPath::Quantized);
        assert_eq!(plan.rows_scanned, Some(42));
        assert_eq!(plan.narrowing, Narrowing::Narrowed { candidates: 7 });
        assert_eq!(plan.timings.total, Duration::from_micros(500));
        assert!(plan.timings.first_pass.is_some());
        assert_eq!(plan.candidates.unwrap().surfaced, 10);
    }
}
