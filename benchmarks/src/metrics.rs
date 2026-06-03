//! Timing summaries and on-disk size measurement.

use std::fs;
use std::path::Path;
use std::time::Duration;

/// Latency percentiles over a set of per-query samples.
#[derive(Clone, Copy, Debug)]
pub struct Timings {
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub mean: Duration,
    pub count: usize,
}

impl Timings {
    /// Summarize raw latency samples. Consumes/sorts the input.
    pub fn summarize(mut samples: Vec<Duration>) -> Self {
        assert!(!samples.is_empty(), "no latency samples to summarize");
        samples.sort_unstable();
        let count = samples.len();
        let sum: Duration = samples.iter().sum();
        Timings {
            p50: percentile(&samples, 0.50),
            p95: percentile(&samples, 0.95),
            p99: percentile(&samples, 0.99),
            mean: sum / count as u32,
            count,
        }
    }
}

/// Nearest-rank percentile of a pre-sorted slice. `p` in `[0.0, 1.0]`.
pub fn percentile(sorted: &[Duration], p: f64) -> Duration {
    debug_assert!(!sorted.is_empty());
    let p = p.clamp(0.0, 1.0);
    // nearest-rank: ceil(p * n) clamped to [1, n], then 0-indexed.
    let rank = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

/// Total bytes of a store: the file itself, or every file under a directory tree.
pub fn disk_bytes(path: &Path) -> u64 {
    let meta = match fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return 0,
    };
    if meta.is_file() {
        return meta.len();
    }
    let mut total = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            total += disk_bytes(&entry.path());
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ds(ms: &[u64]) -> Vec<Duration> {
        ms.iter().map(|&m| Duration::from_millis(m)).collect()
    }

    #[test]
    fn percentiles_pick_expected_ranks() {
        let mut s = ds(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        s.sort_unstable();
        assert_eq!(percentile(&s, 0.50), Duration::from_millis(50));
        assert_eq!(percentile(&s, 0.95), Duration::from_millis(100));
        assert_eq!(percentile(&s, 0.99), Duration::from_millis(100));
        assert_eq!(percentile(&s, 0.0), Duration::from_millis(10));
    }

    #[test]
    fn summarize_mean() {
        let t = Timings::summarize(ds(&[10, 20, 30]));
        assert_eq!(t.mean, Duration::from_millis(20));
        assert_eq!(t.count, 3);
    }
}
