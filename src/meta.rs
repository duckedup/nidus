//! Metadata keys and clock helpers used outside the `memory` feature (nidus-140).
//! Lives ungated so [`Nidus::sweep_expired`] compiles in every build.

/// Attr key holding the `Value::DateTime` (UTC epoch ms) after which an entry is
/// expired. Absent means the entry never expires.
pub const META_EXPIRES_AT: &str = "nidus.expires_at";

/// Attr key counting how many times an entry has been returned by a reinforced recall.
/// `Value::Int`. Absent means never recalled (which the count ranking term reads as 0).
pub const META_ACCESS_COUNT: &str = "nidus.access_count";
/// Attr key holding the `Value::DateTime` (UTC epoch ms) of the last reinforced recall.
pub const META_LAST_ACCESSED: &str = "nidus.last_accessed";

/// The current time as `Value::DateTime`'s representation: UTC epoch milliseconds.
pub(crate) fn now_ms() -> i64 {
    crate::clock::now_unix_millis()
}
