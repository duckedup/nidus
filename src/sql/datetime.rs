//! `timestamp '...'` literal parsing: an ISO-8601 UTC instant to epoch milliseconds.
//! Pure integer civil-date math (Howard Hinnant's `days_from_civil`), no date dependency,
//! so it runs on the lean lane and under Miri like the rest of `src/sql/`.

/// Accepted: `YYYY-MM-DDTHH:MM:SS[.fff]Z`, with a space in place of `T` and a missing `Z`
/// both tolerated. Always UTC: there is no offset syntax, because `Value::DateTime` is an
/// absolute instant (`SPEC.md` §3). `None` on anything else.
pub(super) fn iso8601_to_millis(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 19 {
        return None;
    }
    if !matches!(b[10], b'T' | b't' | b' ') {
        return None;
    }
    if b[4] != b'-' || b[7] != b'-' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let n = |r: std::ops::Range<usize>| -> Option<i64> {
        let t = std::str::from_utf8(&b[r]).ok()?;
        if t.bytes().all(|c| c.is_ascii_digit()) {
            t.parse().ok()
        } else {
            None
        }
    };
    let (year, month, day) = (n(0..4)?, n(5..7)?, n(8..10)?);
    let (hour, min, sec) = (n(11..13)?, n(14..16)?, n(17..19)?);
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    if hour > 23 || min > 59 || sec > 60 {
        return None;
    }
    let millis = parse_fraction(&b[19..])?;

    // days_from_civil: days since 1970-01-01, valid for years before it too.
    let y = if month <= 2 { year - 1 } else { year };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    let secs = days.checked_mul(86400)? + hour * 3600 + min * 60 + sec;
    secs.checked_mul(1000)?.checked_add(millis)
}

/// The optional `.fff` and trailing `Z`, as milliseconds. More than three fractional
/// digits truncate; fewer pad, so `.5` is 500ms.
fn parse_fraction(rest: &[u8]) -> Option<i64> {
    let rest = match rest {
        [] => return Some(0),
        [b'Z' | b'z'] => return Some(0),
        r => r,
    };
    if rest[0] != b'.' {
        return None;
    }
    let digits: Vec<u8> = rest[1..]
        .iter()
        .copied()
        .take_while(u8::is_ascii_digit)
        .collect();
    if digits.is_empty() {
        return None;
    }
    match &rest[1 + digits.len()..] {
        [] | [b'Z' | b'z'] => {}
        _ => return None,
    }
    let mut ms: i64 = 0;
    for i in 0..3 {
        ms = ms * 10 + i64::from(digits.get(i).map_or(0, |d| d - b'0'));
    }
    Some(ms)
}

#[cfg(test)]
mod tests {
    use super::iso8601_to_millis as p;

    #[test]
    fn the_epoch_itself() {
        assert_eq!(p("1970-01-01T00:00:00Z"), Some(0));
    }

    #[test]
    fn a_known_instant() {
        assert_eq!(p("2023-11-14T22:13:20Z"), Some(1_700_000_000_000));
    }

    #[test]
    fn milliseconds_are_kept() {
        assert_eq!(p("1970-01-01T00:00:00.250Z"), Some(250));
        assert_eq!(p("1970-01-01T00:00:00.5Z"), Some(500));
        assert_eq!(p("1970-01-01T00:00:00.1234Z"), Some(123));
    }

    #[test]
    fn before_the_epoch_is_negative() {
        assert_eq!(p("1969-12-31T23:59:59Z"), Some(-1000));
    }

    #[test]
    fn a_leap_day_is_a_real_day() {
        assert_eq!(p("2024-02-29T00:00:00Z"), Some(1_709_164_800_000));
    }

    #[test]
    fn the_z_and_the_t_are_optional() {
        assert_eq!(p("2023-11-14T22:13:20"), Some(1_700_000_000_000));
        assert_eq!(p("2023-11-14 22:13:20Z"), Some(1_700_000_000_000));
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        for bad in [
            "",
            "2023-11-14",
            "2023-11-14T22:13",
            "2023/11/14T22:13:20Z",
            "2023-13-01T00:00:00Z",
            "2023-11-32T00:00:00Z",
            "2023-11-14T24:13:20Z",
            "2023-11-14T22:61:20Z",
            "20a3-11-14T22:13:20Z",
            "2023-11-14T22:13:20+01:00",
            "2023-11-14T22:13:20.Z",
            "2023-11-14T22:13:20xZ",
        ] {
            assert_eq!(p(bad), None, "{bad:?} should not parse");
        }
    }
}
