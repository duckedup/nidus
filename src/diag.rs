//! Levelled, machine-parseable diagnostic logging (nidus-abx.4).
//!
//! Before this, every diagnostic in the crate was a bare `eprintln!`: no level, so there
//! was no way to turn detail up in production or down in a test; no structure, so a log
//! aggregator could not query any of it; and no correlation id, so a slow request could
//! not be tied to the lease renewal that overlapped it. `NIDUS_LEASE_DEBUG=1` was a
//! deliberately crude on/off for one subsystem and showed what the general mechanism
//! should be — this is that mechanism, and it subsumes it.
//!
//! **Why not `tracing`.** `tracing` + `tracing-subscriber` is the idiomatic answer and
//! would be a fine choice for a bigger service, but it is a real dependency-tree decision
//! rather than an implementation detail (CLAUDE.md), and it would land in the *library*
//! tree — every `cargo add nidus` consumer would pay for it. What the crate actually needs
//! is levels, key=value structure, and a request id. That is this file: no dependency, and
//! it holds the same shape a `tracing` migration would keep if one is ever wanted.
//!
//! **Format** — logfmt, one event per line on stderr:
//!
//! ```text
//! ts=2026-07-25T18:04:11.482Z level=warn target=lease msg="renewal failed" pid=4711 attempt=3
//! ```
//!
//! `ts`/`level`/`target`/`msg` always lead, in that order, followed by the caller's own
//! key=value pairs. Values containing a space, a quote, or an `=` are quoted and escaped,
//! so a naive `key=value` splitter never mis-parses a message.
//!
//! **Level** comes from `NIDUS_LOG` (`error|warn|info|debug|trace`, default `info`), read
//! once into a `OnceLock` — so a suppressed `trace!` in a hot path costs one relaxed atomic
//! load and no formatting at all.

use std::fmt::Write as _;
use std::sync::OnceLock;

/// Severity, ordered least to most verbose. `Off` silences everything.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Level {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl Level {
    fn as_str(self) -> &'static str {
        match self {
            Level::Off => "off",
            Level::Error => "error",
            Level::Warn => "warn",
            Level::Info => "info",
            Level::Debug => "debug",
            Level::Trace => "trace",
        }
    }

    /// Parse a `NIDUS_LOG` value. Unknown text falls back to `Info` rather than erroring:
    /// a typo in a log-level env var must never stop a store from opening.
    fn parse(s: &str) -> Level {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "silent" => Level::Off,
            "error" => Level::Error,
            "warn" | "warning" => Level::Warn,
            "debug" => Level::Debug,
            "trace" => Level::Trace,
            _ => Level::Info,
        }
    }
}

/// The configured threshold, read from `NIDUS_LOG` exactly once.
pub(crate) fn level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        // `NIDUS_LEASE_DEBUG` predates this and is still honoured: it used to be the only
        // way to see lease tracing, and a runbook or a CI job may still set it. It now
        // simply means "debug", so the crude hook becomes a special case of the general
        // mechanism instead of sitting beside it.
        if std::env::var_os("NIDUS_LEASE_DEBUG").is_some()
            && std::env::var_os("NIDUS_LOG").is_none()
        {
            return Level::Debug;
        }
        std::env::var("NIDUS_LOG").map_or(Level::Info, |v| Level::parse(&v))
    })
}

/// Whether an event at `lvl` would be emitted. Callers use this to skip formatting.
pub(crate) fn enabled(lvl: Level) -> bool {
    lvl <= level()
}

/// Emit one logfmt line. Called only from [`diag!`], which has already checked the level.
///
/// `msg` is a `Display` rather than a `&str` so a call site can hand over a
/// `format_args!` without allocating for a line that a level check might still drop.
pub(crate) fn emit(
    lvl: Level,
    target: &str,
    msg: &dyn std::fmt::Display,
    fields: &[(&str, &dyn std::fmt::Display)],
) {
    let mut line = String::with_capacity(128);
    let _ = write!(
        line,
        "ts={} level={} target={} msg=",
        Timestamp::now(),
        lvl.as_str(),
        target
    );
    write_value(&mut line, msg);
    for (k, v) in fields {
        let _ = write!(line, " {k}=");
        write_value(&mut line, v);
    }
    // One `eprintln!` per event: a single write syscall per line, so concurrent threads
    // interleave whole lines rather than fragments.
    eprintln!("{line}");
}

/// Append a value, quoting and escaping only when a naive `key=value` split would
/// otherwise break. Bare values stay bare, which keeps the common line readable.
fn write_value(out: &mut String, v: &dyn std::fmt::Display) {
    let s = v.to_string();
    let needs_quotes = s.is_empty()
        || s.chars()
            .any(|c| c.is_whitespace() || c == '"' || c == '=' || c == '\\');
    if !needs_quotes {
        out.push_str(&s);
        return;
    }
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Emit a levelled, structured event.
///
/// ```ignore
/// diag!(Level::Warn, "lease", "renewal failed transiently", "attempt" => attempt, "err" => format!("{e:#}"));
/// ```
///
/// Named `diag!` rather than `log!` because `crate::log` is already the op-log codec —
/// two `log`s in one crate, in different namespaces, is exactly the ambiguity a reader
/// should never have to resolve.
///
/// The level check happens *before* the arguments are evaluated, so a suppressed event
/// costs an atomic load — formatting an error chain for a `debug` line nobody asked for
/// would otherwise be a real cost on a hot path.
macro_rules! diag {
    ($lvl:expr, $target:expr, $msg:expr $(, $k:expr => $v:expr)* $(,)?) => {{
        let lvl = $lvl;
        if $crate::diag::enabled(lvl) {
            $crate::diag::emit(lvl, $target, &$msg, &[$(($k, &$v as &dyn ::std::fmt::Display)),*]);
        }
    }};
}
pub(crate) use diag;

/// A UTC RFC 3339 timestamp with millisecond precision, from `std` alone.
///
/// Rendering a human-readable date is the one thing structured logging needs that `std`
/// does not hand over. Pulling `chrono`/`time` into the *library* tree for it would be a
/// dependency decision (CLAUDE.md) for ~20 lines of arithmetic, and unix millis in the log
/// would push the conversion onto whoever is reading stderr at 3am. So: the standard
/// civil-from-days algorithm, era-based and exact for every date `SystemTime` can hold.
struct Timestamp(u64);

impl Timestamp {
    fn now() -> Timestamp {
        Timestamp(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as u64),
        )
    }
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let secs = self.0 / 1000;
        let millis = self.0 % 1000;
        let days = (secs / 86_400) as i64;
        let tod = secs % 86_400;
        let (y, m, d) = civil_from_days(days);
        write!(
            f,
            "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{millis:03}Z",
            tod / 3600,
            (tod % 3600) / 60,
            tod % 60,
        )
    }
}

/// Days since the Unix epoch → `(year, month, day)`. Howard Hinnant's era-based
/// `civil_from_days`, which is exact and branch-light; the shift by 719_468 moves the
/// epoch to 0000-03-01 so leap days land at the end of the internal year.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], March-based
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_order_least_to_most_verbose() {
        assert!(Level::Error < Level::Warn);
        assert!(Level::Warn < Level::Info);
        assert!(Level::Info < Level::Debug);
        assert!(Level::Debug < Level::Trace);
        assert!(Level::Off < Level::Error);
    }

    #[test]
    fn parse_is_forgiving() {
        assert_eq!(Level::parse("WARN"), Level::Warn);
        assert_eq!(Level::parse(" debug "), Level::Debug);
        assert_eq!(Level::parse("off"), Level::Off);
        // A typo must not be fatal, and must not silence the logs either.
        assert_eq!(Level::parse("verbose"), Level::Info);
    }

    /// A value with a space, a quote, or an `=` must be quoted, so a consumer splitting on
    /// whitespace-then-`=` cannot be fooled by an error message that contains either.
    #[test]
    fn values_are_quoted_only_when_ambiguous() {
        let mut s = String::new();
        write_value(&mut s, &"plain");
        assert_eq!(s, "plain");

        let mut s = String::new();
        write_value(&mut s, &"two words");
        assert_eq!(s, "\"two words\"");

        let mut s = String::new();
        write_value(&mut s, &"a=b");
        assert_eq!(s, "\"a=b\"");

        let mut s = String::new();
        write_value(&mut s, &"say \"hi\"");
        assert_eq!(s, "\"say \\\"hi\\\"\"");

        // An empty value would otherwise render as a bare `key=`, which several logfmt
        // parsers read as a missing key rather than an empty string.
        let mut s = String::new();
        write_value(&mut s, &"");
        assert_eq!(s, "\"\"");
    }

    #[test]
    fn timestamp_renders_rfc3339() {
        // 2026-07-25T18:04:11.482Z — a fixed instant, so this asserts the arithmetic
        // rather than the clock.
        assert_eq!(
            Timestamp(1_785_002_651_482).to_string(),
            "2026-07-25T18:04:11.482Z"
        );
        // The epoch itself, and a leap day, are the two cases the era arithmetic is
        // easiest to get wrong on.
        assert_eq!(Timestamp(0).to_string(), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            Timestamp(1_709_164_800_000).to_string(),
            "2024-02-29T00:00:00.000Z"
        );
    }
}
