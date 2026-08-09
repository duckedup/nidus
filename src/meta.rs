//! Metadata keys and clock helpers used outside the `memory` feature (nidus-140).
//! Lives ungated so [`Nidus::sweep_expired`] compiles in every build.

use std::time::{SystemTime, UNIX_EPOCH};

/// Attr key holding the `Value::DateTime` (UTC epoch ms) after which an entry is
/// expired. Absent means the entry never expires.
pub const META_EXPIRES_AT: &str = "nidus.expires_at";

/// The current time as `Value::DateTime`'s representation: UTC epoch milliseconds.
pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
