//! Process-wide counters, exported as Prometheus text by `GET /metrics` (nidus-abx.4).

use std::sync::atomic::{AtomicU64, Ordering};

/// A monotonically increasing count. Never decreases, never resets — Prometheus expects
/// exactly that and handles restarts itself.
#[derive(Debug, Default)]
pub struct Counter(AtomicU64);

impl Counter {
    /// A fresh zeroed counter. `const` so a registry can be a `static`.
    pub const fn new() -> Counter {
        Counter(AtomicU64::new(0))
    }

    /// Record one event.
    pub fn inc(&self) {
        self.add(1);
    }

    /// Record `n` events (e.g. rows scanned by one query).
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }

    /// Current value.
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// A value that goes up and down — in-flight requests, and nothing else so far.
#[derive(Debug, Default)]
pub struct Gauge(AtomicU64);

impl Gauge {
    /// A fresh zeroed gauge. `const` so a registry can be a `static`.
    pub const fn new() -> Gauge {
        Gauge(AtomicU64::new(0))
    }

    pub fn inc(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    pub fn dec(&self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Every counter the library maintains, as one `static`.
#[derive(Debug)]
pub struct Metrics {
    // ── Writer lease (cluster mode) ──────────────────────────────────────────
    /// Lease renewals attempted, from either the write path or the background renewer.
    pub lease_renew_attempts: Counter,
    /// Lease renewals that succeeded.
    pub lease_renew_ok: Counter,
    /// Renewals that failed **transiently** — a backend blip, the lease is still ours.
    pub lease_renew_transient_failures: Counter,
    /// Renewals that failed **definitively** — this instance was superseded and fenced.
    pub lease_renew_lost: Counter,
    /// Times this instance latched its `fenced` flag. At most one per process lifetime.
    pub lease_fenced: Counter,
    /// Backend errors seen while a standby waited for the writer handle.
    pub lease_wait_errors: Counter,

    // ── Write path / durability (nidus-xb9.1) ────────────────────────────────
    /// Mutating batches that needed a durable barrier (upserts, deletes, …).
    pub write_batches: Counter,
    /// Durable barriers actually issued — one fsync of `data` then `log`.
    pub durability_barriers: Counter,

    // ── Object store / memory tier ───────────────────────────────────────────
    /// HTTP requests to a backend that were retried (per retry, not per request).
    pub backend_retries: Counter,
    /// Scheduled `refresh()` calls that failed on a read-only instance.
    pub refresh_failures: Counter,

    // ── Search path ──────────────────────────────────────────────────────────
    /// Vector searches served, whatever path took them.
    pub search_queries: Counter,
    /// Searches answered by walking the ANN index (`Config::ann`).
    pub search_ann: Counter,
    /// Searches answered by the per-segment IVF fan-out.
    pub search_segmented: Counter,
    /// Searches answered by the two-pass quantized scan (int8 / binary).
    pub search_quantized: Counter,
    /// Searches answered by the exact f32 brute-force scan.
    pub search_exact: Counter,
    /// Rows fed to the brute-force scan, summed over queries. **Not** incremented on the
    /// ANN or segmented paths, where no full scan happens — that is the point of the split.
    pub search_vectors_scanned: Counter,
    /// Candidates reranked in f32 by the quantized path's second pass.
    pub search_reranked: Counter,
    /// Queries whose filter was narrowed by the opt-in filter index before scanning.
    pub search_findex_narrowed: Counter,
}

impl Metrics {
    const fn new() -> Metrics {
        Metrics {
            lease_renew_attempts: Counter::new(),
            lease_renew_ok: Counter::new(),
            lease_renew_transient_failures: Counter::new(),
            lease_renew_lost: Counter::new(),
            lease_fenced: Counter::new(),
            lease_wait_errors: Counter::new(),
            write_batches: Counter::new(),
            durability_barriers: Counter::new(),
            backend_retries: Counter::new(),
            refresh_failures: Counter::new(),
            search_queries: Counter::new(),
            search_ann: Counter::new(),
            search_segmented: Counter::new(),
            search_quantized: Counter::new(),
            search_exact: Counter::new(),
            search_vectors_scanned: Counter::new(),
            search_reranked: Counter::new(),
            search_findex_narrowed: Counter::new(),
        }
    }

    /// `(metric name, HELP text, value)` for every counter, in scrape order.
    pub fn counters(&self) -> Vec<(&'static str, &'static str, u64)> {
        let Metrics {
            lease_renew_attempts,
            lease_renew_ok,
            lease_renew_transient_failures,
            lease_renew_lost,
            lease_fenced,
            lease_wait_errors,
            write_batches,
            durability_barriers,
            backend_retries,
            refresh_failures,
            search_queries,
            search_ann,
            search_segmented,
            search_quantized,
            search_exact,
            search_vectors_scanned,
            search_reranked,
            search_findex_narrowed,
        } = self;
        vec![
            (
                "nidus_lease_renew_attempts_total",
                "Writer-lease renewals attempted",
                lease_renew_attempts.get(),
            ),
            (
                "nidus_lease_renew_ok_total",
                "Writer-lease renewals that succeeded",
                lease_renew_ok.get(),
            ),
            (
                "nidus_lease_renew_transient_failures_total",
                "Writer-lease renewals that failed transiently (lease still held)",
                lease_renew_transient_failures.get(),
            ),
            (
                "nidus_lease_renew_lost_total",
                "Writer-lease renewals that failed definitively (superseded)",
                lease_renew_lost.get(),
            ),
            (
                "nidus_lease_fenced_total",
                "Times this instance latched its fenced flag",
                lease_fenced.get(),
            ),
            (
                "nidus_lease_wait_errors_total",
                "Backend errors while waiting for the writer handle",
                lease_wait_errors.get(),
            ),
            (
                "nidus_write_batches_total",
                "Mutating batches that needed a durable barrier",
                write_batches.get(),
            ),
            (
                "nidus_durability_barriers_total",
                "Durable barriers issued (one fsync of data then log)",
                durability_barriers.get(),
            ),
            (
                "nidus_backend_retries_total",
                "Backend HTTP requests retried",
                backend_retries.get(),
            ),
            (
                "nidus_refresh_failures_total",
                "Scheduled refresh attempts that failed",
                refresh_failures.get(),
            ),
            (
                "nidus_search_queries_total",
                "Vector searches served",
                search_queries.get(),
            ),
            (
                "nidus_search_ann_total",
                "Searches answered from the ANN index",
                search_ann.get(),
            ),
            (
                "nidus_search_segmented_total",
                "Searches answered by the per-segment IVF fan-out",
                search_segmented.get(),
            ),
            (
                "nidus_search_quantized_total",
                "Searches answered by the two-pass quantized scan",
                search_quantized.get(),
            ),
            (
                "nidus_search_exact_total",
                "Searches answered by the exact f32 brute-force scan",
                search_exact.get(),
            ),
            (
                "nidus_search_vectors_scanned_total",
                "Rows fed to the brute-force scan, summed over queries",
                search_vectors_scanned.get(),
            ),
            (
                "nidus_search_reranked_total",
                "Candidates reranked in f32 by the quantized second pass",
                search_reranked.get(),
            ),
            (
                "nidus_search_findex_narrowed_total",
                "Queries whose filter was narrowed by the filter index before scanning",
                search_findex_narrowed.get(),
            ),
        ]
    }
}

/// The process-wide registry.
static METRICS: Metrics = Metrics::new();

/// Access the process-wide counters.
pub fn metrics() -> &'static Metrics {
    &METRICS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_accumulate() {
        let c = Counter::new();
        assert_eq!(c.get(), 0);
        c.inc();
        c.add(4);
        assert_eq!(c.get(), 5);
    }

    #[test]
    fn gauge_goes_both_ways() {
        let g = Gauge::new();
        g.inc();
        g.inc();
        g.dec();
        assert_eq!(g.get(), 1);
    }

    /// Coverage is enforced by the compiler (`counters()` destructures `Metrics`), so what
    /// is left to check is that the exported *names* are well-formed and unique — a typo
    /// or a copy-paste duplicate would otherwise ship a scrape Prometheus rejects.
    #[test]
    fn exported_names_are_well_formed_and_unique() {
        let names: Vec<&str> = metrics().counters().iter().map(|(n, _, _)| *n).collect();
        assert!(!names.is_empty());
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate metric name");
        for n in names {
            assert!(n.starts_with("nidus_"), "{n} is missing the nidus_ prefix");
            assert!(n.ends_with("_total"), "{n} is a counter but lacks _total");
        }
    }
}
