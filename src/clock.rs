//! The crate's single time seam (nidus-y67). `wasm32` has no `SystemTime`/`Instant` epoch
//! and no threads, so every wall-clock or monotonic read that a wasm build compiles routes
//! through this file. A raw `Instant::now`/`SystemTime::now` elsewhere is allowed only where
//! wasm never compiles it: `backend/object.rs` and `backend/aws_creds.rs` (native-only), and
//! `server/metrics.rs` / `cli/backup.rs` (`cli`-gated).

#[cfg(not(target_family = "wasm"))]
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Wall-clock UTC epoch milliseconds. Native: `SystemTime::now()`, `.unwrap_or(0)` on a
/// pre-epoch clock (the crate's historical behaviour, `src/meta.rs`). Wasm: `Date.now()`.
pub(crate) fn now_unix_millis() -> i64 {
    #[cfg(not(target_family = "wasm"))]
    {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }
    #[cfg(target_family = "wasm")]
    {
        js_sys::Date::now() as i64
    }
}

/// Monotonic clock base for the lock-free staleness stamp: `Instant` cannot live in an atomic,
/// and staleness must read without the store lock. NOT the wall clock — it can jump
/// backwards, making a reader look *younger* than it is, in a browser too.
#[cfg(not(target_family = "wasm"))]
fn mono_base() -> Instant {
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *BASE.get_or_init(Instant::now)
}

/// Milliseconds on nidus's monotonic clock. Native: elapsed since [`mono_base`] via
/// `Instant`. Wasm: `performance.now()` — monotonic and worker-available, unlike
/// `Date.now()` (see the doc above).
pub(crate) fn mono_millis() -> u64 {
    #[cfg(not(target_family = "wasm"))]
    {
        mono_base().elapsed().as_millis() as u64
    }
    #[cfg(target_family = "wasm")]
    {
        performance_now_ms() as u64
    }
}

/// Microseconds on nidus's monotonic clock — the resolution a latency sweep needs, where
/// `mono_millis` would round most searches to 0. On wasm the browser clamps
/// `performance.now()` (5us-100us), so a sweep's timings are coarser there than on native.
pub(crate) fn mono_micros() -> u64 {
    #[cfg(not(target_family = "wasm"))]
    {
        mono_base().elapsed().as_micros() as u64
    }
    #[cfg(target_family = "wasm")]
    {
        (performance_now_ms() * 1_000.0) as u64
    }
}

/// `performance.now()` reached via `js_sys::Reflect` rather than typed `web_sys::Performance`,
/// so this seam needs no `Window`/`Performance` feature on the `web-sys` dependency (out of
/// this unit's scope) and works from both a window and a worker global scope alike.
#[cfg(target_family = "wasm")]
fn performance_now_ms() -> f64 {
    use js_sys::{Function, Reflect};
    use wasm_bindgen::{JsCast, JsValue};

    let global = js_sys::global();
    let Ok(perf) = Reflect::get(&global, &JsValue::from_str("performance")) else {
        return 0.0;
    };
    if perf.is_undefined() {
        return 0.0;
    }
    let Ok(now_fn) = Reflect::get(&perf, &JsValue::from_str("now")) else {
        return 0.0;
    };
    let Ok(now_fn) = now_fn.dyn_into::<Function>() else {
        return 0.0;
    };
    now_fn
        .call0(&perf)
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_unix_millis_is_plausible() {
        // 2020-01-01T00:00:00Z in epoch ms: a floor, not an exact bound — guards against a
        // clock stuck at 0, not against skew.
        assert!(now_unix_millis() > 1_577_836_800_000);
    }

    #[test]
    fn now_unix_millis_is_monotonic_ish_across_two_calls() {
        let a = now_unix_millis();
        let b = now_unix_millis();
        assert!(b >= a, "wall clock went backwards: {a} then {b}");
    }

    #[test]
    fn mono_micros_does_not_go_backwards() {
        let a = mono_micros();
        let b = mono_micros();
        assert!(b >= a, "monotonic clock went backwards: {a} then {b}");
    }

    #[test]
    fn mono_millis_does_not_go_backwards() {
        let a = mono_millis();
        let b = mono_millis();
        assert!(b >= a, "monotonic clock went backwards: {a} then {b}");
    }
}
