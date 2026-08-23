//! Levelled diagnostics (nidus-abx.4): logfmt on stderr, `ts`/`level`/`target`/`msg` leading.
//! Level from `NIDUS_LOG`, read once into a `OnceLock`, so a suppressed event costs one relaxed
//! load. Hand-rolled to keep `tracing` out of the library dependency tree.

use std::fmt::Write as _;
use std::sync::OnceLock;
use std::time::Duration;

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
        // `NIDUS_LEASE_DEBUG` predates this and is still honoured, since a runbook or CI job may
        // set it. It now simply means "debug", making the crude hook a special case of the general
        // mechanism rather than something sitting beside it.
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

/// Parse `NIDUS_SLOW_QUERY_MS`'s raw value. Unset, unparseable, and `0` all mean
/// disabled (`None`), never "every query is slow" — a pure fn so it is unit-testable
/// without the `OnceLock`'s once-per-process caching getting in the way.
fn parse_slow_query_ms(v: Option<&str>) -> Option<Duration> {
    let ms: u64 = v?.trim().parse().ok()?;
    (ms > 0).then(|| Duration::from_millis(ms))
}

/// `NIDUS_SLOW_QUERY_MS`, read once. `None` disables the slow-query line entirely.
pub(crate) fn slow_query_threshold() -> Option<Duration> {
    static THRESHOLD: OnceLock<Option<Duration>> = OnceLock::new();
    *THRESHOLD
        .get_or_init(|| parse_slow_query_ms(std::env::var("NIDUS_SLOW_QUERY_MS").ok().as_deref()))
}

/// Whether `elapsed` at `threshold` counts as slow. Pure, so it is directly testable;
/// `is_slow` below is the impure wrapper the search paths actually call.
fn crosses(elapsed: Duration, threshold: Option<Duration>) -> bool {
    threshold.is_some_and(|t| elapsed >= t)
}

/// Whether `elapsed` crosses the configured threshold.
pub(crate) fn is_slow(elapsed: Duration) -> bool {
    crosses(elapsed, slow_query_threshold())
}

/// Emit one logfmt line. Called only from [`diag!`], which has already checked the level.
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

/// Emit a levelled, structured event. Named `diag!`, not `log!`, because `crate::log` is the
/// op-log codec; the level check precedes argument evaluation, so a suppressed event is one load.
///
/// ```ignore
/// diag!(Level::Warn, "lease", "renewal failed transiently", "attempt" => attempt, "err" => format!("{e:#}"));
/// ```
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
struct Timestamp(u64);

impl Timestamp {
    fn now() -> Timestamp {
        Timestamp(crate::clock::now_unix_millis().max(0) as u64)
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
    fn slow_query_ms_parse_disables_on_bad_input() {
        assert_eq!(parse_slow_query_ms(None), None);
        assert_eq!(parse_slow_query_ms(Some("")), None);
        assert_eq!(parse_slow_query_ms(Some("nope")), None);
        assert_eq!(parse_slow_query_ms(Some("-5")), None);
        // `0` disables rather than meaning "every query is slow".
        assert_eq!(parse_slow_query_ms(Some("0")), None);
        assert_eq!(
            parse_slow_query_ms(Some("250")),
            Some(Duration::from_millis(250))
        );
        assert_eq!(
            parse_slow_query_ms(Some(" 250 ")),
            Some(Duration::from_millis(250))
        );
    }

    #[test]
    fn crosses_is_disabled_when_threshold_is_none() {
        assert!(!crosses(Duration::from_secs(1), None));
        assert!(!crosses(
            Duration::from_millis(99),
            Some(Duration::from_millis(100))
        ));
        assert!(crosses(
            Duration::from_millis(100),
            Some(Duration::from_millis(100))
        ));
        assert!(crosses(
            Duration::from_millis(101),
            Some(Duration::from_millis(100))
        ));
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
