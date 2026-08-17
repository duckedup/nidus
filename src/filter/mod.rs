//! Filter evaluation against a record's attributes. Contract: see the root `SPEC.md` §7, §7.1.

mod narrow;
mod pattern;
mod text;

pub(crate) use narrow::candidate_ids;
#[cfg(test)]
pub(crate) use text::levenshtein_ascii_ci;
pub(crate) use text::tokens;

use std::cmp::Ordering;
use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};

use crate::model::{Filter, Predicate, Value};

/// Same-type ordering of two [`Value`]s for the range predicates: `Int` numeric, `Str` lexical,
/// `Bool` `false < true`, `Float` by IEEE partial order, `DateTime` chronological. `None` for
/// mismatched or non-orderable variants, so a range predicate fails rather than matches spuriously.
pub(crate) fn value_cmp(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
        (Value::Str(x), Value::Str(y)) => Some(x.cmp(y)),
        (Value::Bool(x), Value::Bool(y)) => Some(x.cmp(y)),
        // `partial_cmp` is `None` for NaN, so a NaN bound or attribute simply fails the
        // predicate rather than imposing a total order that would rank it somewhere.
        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y),
        (Value::DateTime(x), Value::DateTime(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// True iff `attrs[key]` is present and orders against `bound` as one of `wanted`.
/// Absent key or an incomparable pair (`value_cmp` → `None`) is never a match.
fn range_matches(
    attrs: &BTreeMap<String, Value>,
    key: &str,
    bound: &Value,
    wanted: &[Ordering],
) -> bool {
    match attrs.get(key).and_then(|v| value_cmp(v, bound)) {
        Some(ord) => wanted.contains(&ord),
        None => false,
    }
}

/// True iff `attrs[key]` is a list and `pred` holds over its elements. An absent or
/// non-list attribute is never a match, so `NotContains` requires the key present and
/// list-typed exactly as `Ne` requires it present.
fn list_matches(
    attrs: &BTreeMap<String, Value>,
    key: &str,
    pred: impl Fn(&[String]) -> bool,
) -> bool {
    match attrs.get(key) {
        Some(Value::List(items)) => pred(items),
        _ => false,
    }
}

/// The string a list element could equal. Lists hold strings, so any other variant is a
/// needle that can never be found — reported as `None` rather than silently coerced.
fn needle(value: &Value) -> Option<&str> {
    match value {
        Value::Str(s) => Some(s.as_str()),
        _ => None,
    }
}

/// True iff every predicate in `filter` matches `attrs`; an empty filter matches everything. Every
/// predicate requires `key` present, so an absent attribute matches nothing — including the negative
/// and range predicates. `SPEC.md` §7.1 has the full per-predicate semantics.
pub fn matches(filter: &Filter, attrs: &BTreeMap<String, Value>) -> bool {
    filter.0.iter().all(|p| matches_one(p, attrs))
}

/// Evaluate one predicate, recursing through the [`Predicate::All`]/[`Predicate::Any`]/
/// [`Predicate::Not`] groups. Untrusted input cannot recurse without bound: serde_json
/// caps nesting at 128 before a filter ever reaches here, and no `Op` carries a filter.
fn matches_one(predicate: &Predicate, attrs: &BTreeMap<String, Value>) -> bool {
    use Ordering::{Equal, Greater, Less};
    match predicate {
        Predicate::Eq(key, expected) => attrs.get(key) == Some(expected),
        Predicate::Ne(key, expected) => matches!(attrs.get(key), Some(v) if v != expected),
        Predicate::Glob(key, pattern) => match attrs.get(key) {
            Some(Value::Str(s)) => crate::glob::glob_match(pattern, s),
            _ => false,
        },
        Predicate::IGlob(key, pattern) => match attrs.get(key) {
            Some(Value::Str(s)) => crate::glob::glob_match_ascii_ci(pattern, s),
            _ => false,
        },
        Predicate::In(key, set) => match attrs.get(key) {
            Some(v) => set.contains(v),
            None => false,
        },
        Predicate::NotIn(key, set) => matches!(attrs.get(key), Some(v) if !set.contains(v)),
        Predicate::Lt(key, bound) => range_matches(attrs, key, bound, &[Less]),
        Predicate::Le(key, bound) => range_matches(attrs, key, bound, &[Less, Equal]),
        Predicate::Gt(key, bound) => range_matches(attrs, key, bound, &[Greater]),
        Predicate::Ge(key, bound) => range_matches(attrs, key, bound, &[Greater, Equal]),
        Predicate::Contains(key, value) => match needle(value) {
            Some(want) => list_matches(attrs, key, |items| items.iter().any(|i| i == want)),
            None => false,
        },
        Predicate::NotContains(key, value) => match needle(value) {
            Some(want) => list_matches(attrs, key, |items| !items.iter().any(|i| i == want)),
            // An unfindable needle is trivially absent, but the key must still be a list.
            None => list_matches(attrs, key, |_| true),
        },
        Predicate::ContainsAny(key, set) => list_matches(attrs, key, |items| {
            set.iter()
                .filter_map(needle)
                .any(|want| items.iter().any(|i| i == want))
        }),
        Predicate::All(preds) => preds.iter().all(|p| matches_one(p, attrs)),
        Predicate::Any(preds) => preds.iter().any(|p| matches_one(p, attrs)),
        Predicate::Not(pred) => !matches_one(pred, attrs),
        Predicate::Fuzzy(key, needle, max_edits) => text::fuzzy(attrs.get(key), needle, *max_edits),
        Predicate::ContainsAllTokens(key, query) => {
            text::contains_all_tokens(attrs.get(key), query)
        }
        Predicate::ContainsAnyToken(key, query) => text::contains_any_token(attrs.get(key), query),
        Predicate::ContainsTokenSequence(key, query) => {
            text::contains_token_sequence(attrs.get(key), query)
        }
        Predicate::Regex(key, p) => pattern::regex_matches(attrs.get(key), p),
    }
}

/// Prepare a filter once per query, before any row is scanned: reject what cannot be honoured
/// as written, and compile every `Regex` into the pattern cache so the per-record path never
/// pays a compile. Every public `Nidus` query method calls this first.
pub(crate) fn validate(filter: &Filter) -> Result<()> {
    // Marker INLINE rather than via `.context`, which would become the outermost message
    // and hide which predicate was wrong. Tags it 400 (a caller mistake, not a 5xx).
    filter
        .0
        .iter()
        .try_for_each(validate_one)
        .map_err(|e| anyhow::anyhow!("{}: {e:#}", crate::store::BAD_QUERY))
}

/// Recurses through the boolean groups exactly as `matches_one` does, so a nested predicate
/// is prepared too. Depth is bounded by the same serde_json nesting cap.
fn validate_one(predicate: &Predicate) -> Result<()> {
    match predicate {
        Predicate::Fuzzy(key, _, max_edits) if *max_edits > text::MAX_FUZZY_EDITS => bail!(
            "Fuzzy on `{key}` allows {max_edits} edits, above the maximum of {}",
            text::MAX_FUZZY_EDITS
        ),
        Predicate::Regex(key, p) => pattern::compile(p)
            .map(|_| ())
            .with_context(|| format!("Regex on `{key}`")),
        Predicate::All(preds) | Predicate::Any(preds) => preds.iter().try_for_each(validate_one),
        Predicate::Not(pred) => validate_one(pred),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::model::{Filter, Predicate, Value};

    use super::matches;

    // ── Helpers ──────────────────────────────────────────────────────────────────

    fn attrs(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn filter(predicates: Vec<Predicate>) -> Filter {
        Filter(predicates)
    }

    // ── Empty filter ─────────────────────────────────────────────────────────────

    #[test]
    fn empty_filter_matches_empty_attrs() {
        assert!(matches(&filter(vec![]), &BTreeMap::new()));
    }

    #[test]
    fn empty_filter_matches_nonempty_attrs() {
        let a = attrs(&[("x", Value::Int(1))]);
        assert!(matches(&filter(vec![]), &a));
    }

    // ── Eq predicate ─────────────────────────────────────────────────────────────

    #[test]
    fn eq_str_match() {
        let a = attrs(&[("lang", Value::Str("rust".into()))]);
        let f = filter(vec![Predicate::Eq(
            "lang".into(),
            Value::Str("rust".into()),
        )]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn eq_str_mismatch() {
        let a = attrs(&[("lang", Value::Str("go".into()))]);
        let f = filter(vec![Predicate::Eq(
            "lang".into(),
            Value::Str("rust".into()),
        )]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn eq_int_match() {
        let a = attrs(&[("count", Value::Int(42))]);
        let f = filter(vec![Predicate::Eq("count".into(), Value::Int(42))]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn eq_int_mismatch() {
        let a = attrs(&[("count", Value::Int(42))]);
        let f = filter(vec![Predicate::Eq("count".into(), Value::Int(0))]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn eq_bool_match() {
        let a = attrs(&[("active", Value::Bool(true))]);
        let f = filter(vec![Predicate::Eq("active".into(), Value::Bool(true))]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn eq_bool_mismatch() {
        let a = attrs(&[("active", Value::Bool(true))]);
        let f = filter(vec![Predicate::Eq("active".into(), Value::Bool(false))]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn eq_list_match() {
        let a = attrs(&[("tags", Value::List(vec!["a".into(), "b".into()]))]);
        let f = filter(vec![Predicate::Eq(
            "tags".into(),
            Value::List(vec!["a".into(), "b".into()]),
        )]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn eq_list_mismatch_order() {
        // List equality is order-sensitive (Vec PartialEq)
        let a = attrs(&[("tags", Value::List(vec!["b".into(), "a".into()]))]);
        let f = filter(vec![Predicate::Eq(
            "tags".into(),
            Value::List(vec!["a".into(), "b".into()]),
        )]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn eq_null_present_matches() {
        // Key present with Value::Null must match Eq(key, Null)
        let a = attrs(&[("edge", Value::Null)]);
        let f = filter(vec![Predicate::Eq("edge".into(), Value::Null)]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn eq_null_absent_does_not_match() {
        // Key absent must NOT equal Null — the critical Null-vs-absent distinction
        let a = BTreeMap::new();
        let f = filter(vec![Predicate::Eq("edge".into(), Value::Null)]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn eq_absent_key_fails() {
        let a = attrs(&[("other", Value::Int(1))]);
        let f = filter(vec![Predicate::Eq("missing".into(), Value::Int(1))]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn eq_type_mismatch_str_vs_int() {
        // Same textual value but different types must not match
        let a = attrs(&[("x", Value::Str("1".into()))]);
        let f = filter(vec![Predicate::Eq("x".into(), Value::Int(1))]);
        assert!(!matches(&f, &a));
    }

    // ── Glob predicate ───────────────────────────────────────────────────────────

    #[test]
    fn glob_str_match_star() {
        let a = attrs(&[("path", Value::Str("src/main.rs".into()))]);
        let f = filter(vec![Predicate::Glob("path".into(), "src/*".into())]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn glob_str_no_match() {
        let a = attrs(&[("path", Value::Str("tests/foo.rs".into()))]);
        let f = filter(vec![Predicate::Glob("path".into(), "src/*".into())]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn glob_str_extension_match() {
        let a = attrs(&[("file", Value::Str("Cargo.toml".into()))]);
        let f = filter(vec![Predicate::Glob("file".into(), "*.toml".into())]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn glob_str_question_mark() {
        let a = attrs(&[("name", Value::Str("file1.rs".into()))]);
        let f = filter(vec![Predicate::Glob("name".into(), "file?.rs".into())]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn glob_non_str_value_fails() {
        // Glob on an Int attr must return false (only Str is matchable)
        let a = attrs(&[("x", Value::Int(42))]);
        let f = filter(vec![Predicate::Glob("x".into(), "*".into())]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn glob_null_value_fails() {
        let a = attrs(&[("x", Value::Null)]);
        let f = filter(vec![Predicate::Glob("x".into(), "*".into())]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn glob_bool_value_fails() {
        let a = attrs(&[("flag", Value::Bool(true))]);
        let f = filter(vec![Predicate::Glob("flag".into(), "*".into())]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn glob_list_value_fails() {
        // Glob is Str-only by design; looking inside a list is `Contains`.
        let a = attrs(&[("tags", Value::List(vec!["rust".into()]))]);
        let f = filter(vec![Predicate::Glob("tags".into(), "*".into())]);
        assert!(!matches(&f, &a));
        let c = filter(vec![Predicate::Contains(
            "tags".into(),
            Value::Str("rust".into()),
        )]);
        assert!(matches(&c, &a));
    }

    #[test]
    fn glob_absent_key_fails() {
        let a = BTreeMap::new();
        let f = filter(vec![Predicate::Glob("path".into(), "src/*".into())]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn glob_char_class_match() {
        let a = attrs(&[("ver", Value::Str("v3".into()))]);
        let f = filter(vec![Predicate::Glob("ver".into(), "v[0-9]".into())]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn glob_exact_str_match() {
        let a = attrs(&[("kind", Value::Str("file".into()))]);
        let f = filter(vec![Predicate::Glob("kind".into(), "file".into())]);
        assert!(matches(&f, &a));
    }

    // ── IGlob predicate ──────────────────────────────────────────────────────────

    #[test]
    fn iglob_folds_ascii_case() {
        let a = attrs(&[("path", Value::Str("src/finance/rates.rs".into()))]);
        let f = filter(vec![Predicate::IGlob(
            "path".into(),
            "Src/Finance/*".into(),
        )]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn iglob_where_glob_would_miss() {
        // The predicate pair on identical inputs — the whole point of IGlob.
        let a = attrs(&[("file", Value::Str("README.md".into()))]);
        assert!(!matches(
            &filter(vec![Predicate::Glob("file".into(), "*.MD".into())]),
            &a
        ));
        assert!(matches(
            &filter(vec![Predicate::IGlob("file".into(), "*.MD".into())]),
            &a
        ));
    }

    #[test]
    fn iglob_still_respects_the_pattern() {
        let a = attrs(&[("path", Value::Str("tests/foo.rs".into()))]);
        let f = filter(vec![Predicate::IGlob("path".into(), "SRC/*".into())]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn iglob_non_str_value_fails() {
        // Same type discipline as Glob: only a Str attr is matchable.
        let a = attrs(&[("x", Value::Int(42))]);
        let f = filter(vec![Predicate::IGlob("x".into(), "*".into())]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn iglob_absent_key_fails() {
        let a = BTreeMap::new();
        let f = filter(vec![Predicate::IGlob("path".into(), "src/*".into())]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn iglob_non_ascii_is_not_folded() {
        let a = attrs(&[("path", Value::Str("docs/café.md".into()))]);
        assert!(matches(
            &filter(vec![Predicate::IGlob("path".into(), "DOCS/café.MD".into())]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::IGlob("path".into(), "DOCS/CAFÉ.MD".into())]),
            &a
        ));
    }

    // ── In predicate ─────────────────────────────────────────────────────────────

    #[test]
    fn in_membership_match() {
        let a = attrs(&[("lang", Value::Str("rust".into()))]);
        let f = filter(vec![Predicate::In(
            "lang".into(),
            vec![Value::Str("go".into()), Value::Str("rust".into())],
        )]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn in_membership_no_match() {
        let a = attrs(&[("lang", Value::Str("python".into()))]);
        let f = filter(vec![Predicate::In(
            "lang".into(),
            vec![Value::Str("go".into()), Value::Str("rust".into())],
        )]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn in_empty_set_always_false() {
        // Even if key is present, empty set = false
        let a = attrs(&[("lang", Value::Str("rust".into()))]);
        let f = filter(vec![Predicate::In("lang".into(), vec![])]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn in_absent_key_fails() {
        let a = BTreeMap::new();
        let f = filter(vec![Predicate::In(
            "lang".into(),
            vec![Value::Str("rust".into())],
        )]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn in_null_present_in_set() {
        let a = attrs(&[("edge", Value::Null)]);
        let f = filter(vec![Predicate::In(
            "edge".into(),
            vec![Value::Null, Value::Str("x".into())],
        )]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn in_null_absent_not_in_set() {
        // Absent key fails even when Null is in the set
        let a = BTreeMap::new();
        let f = filter(vec![Predicate::In("edge".into(), vec![Value::Null])]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn in_int_match() {
        let a = attrs(&[("code", Value::Int(200))]);
        let f = filter(vec![Predicate::In(
            "code".into(),
            vec![Value::Int(200), Value::Int(201), Value::Int(204)],
        )]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn in_int_no_match() {
        let a = attrs(&[("code", Value::Int(404))]);
        let f = filter(vec![Predicate::In(
            "code".into(),
            vec![Value::Int(200), Value::Int(201)],
        )]);
        assert!(!matches(&f, &a));
    }

    // ── Multiple predicates (AND) ─────────────────────────────────────────────────

    #[test]
    fn and_all_pass() {
        let a = attrs(&[
            ("lang", Value::Str("rust".into())),
            ("path", Value::Str("src/lib.rs".into())),
            ("active", Value::Bool(true)),
        ]);
        let f = filter(vec![
            Predicate::Eq("lang".into(), Value::Str("rust".into())),
            Predicate::Glob("path".into(), "src/*.rs".into()),
            Predicate::In("active".into(), vec![Value::Bool(true)]),
        ]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn and_first_fails() {
        let a = attrs(&[
            ("lang", Value::Str("go".into())),
            ("path", Value::Str("src/lib.rs".into())),
        ]);
        let f = filter(vec![
            Predicate::Eq("lang".into(), Value::Str("rust".into())),
            Predicate::Glob("path".into(), "src/*".into()),
        ]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn and_second_fails() {
        let a = attrs(&[
            ("lang", Value::Str("rust".into())),
            ("path", Value::Str("tests/foo.rs".into())),
        ]);
        let f = filter(vec![
            Predicate::Eq("lang".into(), Value::Str("rust".into())),
            Predicate::Glob("path".into(), "src/*".into()),
        ]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn and_last_fails() {
        let a = attrs(&[
            ("lang", Value::Str("rust".into())),
            ("path", Value::Str("src/main.rs".into())),
            ("kind", Value::Str("test".into())),
        ]);
        let f = filter(vec![
            Predicate::Eq("lang".into(), Value::Str("rust".into())),
            Predicate::Glob("path".into(), "src/*".into()),
            Predicate::In(
                "kind".into(),
                vec![Value::Str("file".into()), Value::Str("section".into())],
            ),
        ]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn and_mixed_predicate_types_all_pass() {
        let a = attrs(&[
            ("name", Value::Str("foo.rs".into())),
            ("size", Value::Int(512)),
            ("enabled", Value::Bool(true)),
        ]);
        let f = filter(vec![
            Predicate::Glob("name".into(), "*.rs".into()),
            Predicate::In(
                "size".into(),
                vec![Value::Int(256), Value::Int(512), Value::Int(1024)],
            ),
            Predicate::Eq("enabled".into(), Value::Bool(true)),
        ]);
        assert!(matches(&f, &a));
    }

    // ── Single predicate edge cases ───────────────────────────────────────────────

    #[test]
    fn single_eq_predicate_passes() {
        let a = attrs(&[("x", Value::Int(7))]);
        let f = filter(vec![Predicate::Eq("x".into(), Value::Int(7))]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn single_glob_predicate_passes() {
        let a = attrs(&[("p", Value::Str("hello".into()))]);
        let f = filter(vec![Predicate::Glob("p".into(), "hel*".into())]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn single_in_predicate_passes() {
        let a = attrs(&[("v", Value::Bool(false))]);
        let f = filter(vec![Predicate::In(
            "v".into(),
            vec![Value::Bool(false), Value::Bool(true)],
        )]);
        assert!(matches(&f, &a));
    }

    // ── Extra attrs beyond filter scope are ignored ───────────────────────────────

    #[test]
    fn extra_attrs_do_not_affect_result() {
        let a = attrs(&[
            ("lang", Value::Str("rust".into())),
            ("unrelated", Value::Int(999)),
            ("also_unrelated", Value::Null),
        ]);
        let f = filter(vec![Predicate::Eq(
            "lang".into(),
            Value::Str("rust".into()),
        )]);
        assert!(matches(&f, &a));
    }

    // ── Ne predicate ─────────────────────────────────────────────────────────────

    #[test]
    fn ne_present_and_different_matches() {
        let a = attrs(&[("status", Value::Str("active".into()))]);
        let f = filter(vec![Predicate::Ne(
            "status".into(),
            Value::Str("archived".into()),
        )]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn ne_present_and_equal_fails() {
        let a = attrs(&[("status", Value::Str("archived".into()))]);
        let f = filter(vec![Predicate::Ne(
            "status".into(),
            Value::Str("archived".into()),
        )]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn ne_absent_key_fails() {
        // Absent key matches no predicate, including negative ones (SPEC §7.1).
        let a = attrs(&[("other", Value::Int(1))]);
        let f = filter(vec![Predicate::Ne(
            "status".into(),
            Value::Str("archived".into()),
        )]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn ne_different_type_matches() {
        // Different type is "not equal" — a present Int is != a Str bound.
        let a = attrs(&[("k", Value::Int(1))]);
        let f = filter(vec![Predicate::Ne("k".into(), Value::Str("1".into()))]);
        assert!(matches(&f, &a));
    }

    // ── NotIn predicate ──────────────────────────────────────────────────────────

    #[test]
    fn not_in_present_and_absent_from_set_matches() {
        let a = attrs(&[("tag", Value::Str("rust".into()))]);
        let f = filter(vec![Predicate::NotIn(
            "tag".into(),
            vec![Value::Str("go".into()), Value::Str("python".into())],
        )]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn not_in_present_and_in_set_fails() {
        let a = attrs(&[("tag", Value::Str("go".into()))]);
        let f = filter(vec![Predicate::NotIn(
            "tag".into(),
            vec![Value::Str("go".into()), Value::Str("python".into())],
        )]);
        assert!(!matches(&f, &a));
    }

    #[test]
    fn not_in_empty_set_present_key_matches() {
        // Nothing is in the empty set, so a present value is always "not in" it.
        let a = attrs(&[("tag", Value::Str("rust".into()))]);
        let f = filter(vec![Predicate::NotIn("tag".into(), vec![])]);
        assert!(matches(&f, &a));
    }

    #[test]
    fn not_in_absent_key_fails() {
        let a = attrs(&[("other", Value::Int(1))]);
        let f = filter(vec![Predicate::NotIn(
            "tag".into(),
            vec![Value::Str("go".into())],
        )]);
        assert!(!matches(&f, &a));
    }

    // ── Range predicates (Lt/Le/Gt/Ge) ─────────────────────────────────────────────

    #[test]
    fn range_int_lt_gt() {
        let a = attrs(&[("age", Value::Int(30))]);
        assert!(matches(
            &filter(vec![Predicate::Lt("age".into(), Value::Int(40))]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::Lt("age".into(), Value::Int(30))]),
            &a
        ));
        assert!(matches(
            &filter(vec![Predicate::Gt("age".into(), Value::Int(20))]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::Gt("age".into(), Value::Int(30))]),
            &a
        ));
    }

    #[test]
    fn range_int_le_ge_boundary() {
        let a = attrs(&[("n", Value::Int(10))]);
        // Boundary equal: Le/Ge include it, Lt/Gt exclude it.
        assert!(matches(
            &filter(vec![Predicate::Le("n".into(), Value::Int(10))]),
            &a
        ));
        assert!(matches(
            &filter(vec![Predicate::Ge("n".into(), Value::Int(10))]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::Lt("n".into(), Value::Int(10))]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::Gt("n".into(), Value::Int(10))]),
            &a
        ));
    }

    #[test]
    fn range_negative_ints() {
        let a = attrs(&[("balance", Value::Int(-5))]);
        assert!(matches(
            &filter(vec![Predicate::Lt("balance".into(), Value::Int(0))]),
            &a
        ));
        assert!(matches(
            &filter(vec![Predicate::Gt("balance".into(), Value::Int(-10))]),
            &a
        ));
    }

    #[test]
    fn range_str_lexical() {
        let a = attrs(&[("name", Value::Str("mango".into()))]);
        assert!(matches(
            &filter(vec![Predicate::Gt(
                "name".into(),
                Value::Str("apple".into())
            )]),
            &a
        ));
        assert!(matches(
            &filter(vec![Predicate::Lt(
                "name".into(),
                Value::Str("zebra".into())
            )]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::Lt(
                "name".into(),
                Value::Str("apple".into())
            )]),
            &a
        ));
    }

    #[test]
    fn range_bool_false_lt_true() {
        let a = attrs(&[("flag", Value::Bool(false))]);
        assert!(matches(
            &filter(vec![Predicate::Lt("flag".into(), Value::Bool(true))]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::Gt("flag".into(), Value::Bool(true))]),
            &a
        ));
    }

    #[test]
    fn range_cross_type_never_matches() {
        // Int attr vs Str bound is incomparable → no range match, for every operator.
        let a = attrs(&[("k", Value::Int(5))]);
        for p in [
            Predicate::Lt("k".into(), Value::Str("5".into())),
            Predicate::Le("k".into(), Value::Str("5".into())),
            Predicate::Gt("k".into(), Value::Str("5".into())),
            Predicate::Ge("k".into(), Value::Str("5".into())),
        ] {
            assert!(!matches(&filter(vec![p]), &a));
        }
    }

    #[test]
    fn range_null_and_list_never_match() {
        let a = attrs(&[("nul", Value::Null), ("lst", Value::List(vec!["a".into()]))]);
        assert!(!matches(
            &filter(vec![Predicate::Lt("nul".into(), Value::Int(1))]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::Gt("lst".into(), Value::Int(0))]),
            &a
        ));
    }

    #[test]
    fn range_absent_key_fails() {
        let a = attrs(&[("other", Value::Int(1))]);
        assert!(!matches(
            &filter(vec![Predicate::Lt("age".into(), Value::Int(40))]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::Ge("age".into(), Value::Int(0))]),
            &a
        ));
    }

    #[test]
    fn range_and_negation_compose_as_and() {
        // 18 <= age < 65 AND tier != "free": a half-open range plus an exclusion.
        let a = attrs(&[("age", Value::Int(40)), ("tier", Value::Str("pro".into()))]);
        let f = filter(vec![
            Predicate::Ge("age".into(), Value::Int(18)),
            Predicate::Lt("age".into(), Value::Int(65)),
            Predicate::Ne("tier".into(), Value::Str("free".into())),
        ]);
        assert!(matches(&f, &a));

        // Same filter, but age out of range → fails.
        let b = attrs(&[("age", Value::Int(70)), ("tier", Value::Str("pro".into()))]);
        assert!(!matches(&f, &b));
    }

    // ── serde round-trip (the variants flow through CLI --where / server) ──────────

    #[test]
    fn new_predicates_round_trip_through_serde() {
        // bincode is the core serializer (serde_json is a cli-only dep); a clean
        // round-trip here confirms the derived Serialize/Deserialize covers the new
        // variants, which is exactly what carries them across the CLI/server wire.
        let preds = vec![
            Predicate::Ne("a".into(), Value::Int(1)),
            Predicate::NotIn("b".into(), vec![Value::Str("x".into())]),
            Predicate::Lt("c".into(), Value::Int(10)),
            Predicate::Le("d".into(), Value::Int(10)),
            Predicate::Gt("e".into(), Value::Int(10)),
            Predicate::Ge("f".into(), Value::Int(10)),
            Predicate::IGlob("g".into(), "src/*".into()),
        ];
        let f = filter(preds);
        let bytes = bincode::serialize(&f).unwrap();
        let back: Filter = bincode::deserialize(&bytes).unwrap();
        assert_eq!(f, back);
    }

    // ── Array containment (nidus-m50.2) ──────────────────────────────────────────

    fn tagged(tags: &[&str]) -> BTreeMap<String, Value> {
        attrs(&[(
            "tags",
            Value::List(tags.iter().map(|s| s.to_string()).collect()),
        )])
    }

    fn contains(needle: &str) -> Filter {
        filter(vec![Predicate::Contains(
            "tags".into(),
            Value::Str(needle.into()),
        )])
    }

    #[test]
    fn contains_finds_an_element() {
        assert!(matches(&contains("rust"), &tagged(&["go", "rust"])));
    }

    #[test]
    fn contains_missing_element_fails() {
        assert!(!matches(&contains("zig"), &tagged(&["go", "rust"])));
    }

    #[test]
    fn contains_empty_list_fails() {
        assert!(!matches(&contains("rust"), &tagged(&[])));
    }

    #[test]
    fn contains_absent_key_fails() {
        assert!(!matches(&contains("rust"), &BTreeMap::new()));
    }

    #[test]
    fn contains_is_not_substring_matching() {
        // "rust" must not match the element "rustacean" — Glob covers substrings.
        assert!(!matches(&contains("rust"), &tagged(&["rustacean"])));
    }

    #[test]
    fn contains_on_a_scalar_string_fails() {
        // A plain Str is not a one-element list; Glob is the tool for that.
        let a = attrs(&[("tags", Value::Str("rust".into()))]);
        assert!(!matches(&contains("rust"), &a));
    }

    #[test]
    fn contains_non_string_needle_never_matches() {
        // Lists hold strings, so an Int needle is unfindable rather than coerced.
        let f = filter(vec![Predicate::Contains("tags".into(), Value::Int(1))]);
        assert!(!matches(&f, &tagged(&["1"])));
    }

    #[test]
    fn not_contains_requires_the_key_to_be_a_present_list() {
        let f = filter(vec![Predicate::NotContains(
            "tags".into(),
            Value::Str("zig".into()),
        )]);
        assert!(matches(&f, &tagged(&["rust"])));
        assert!(matches(&f, &tagged(&[])));
        // Absent key and wrong type both fail, exactly as `Ne` does.
        assert!(!matches(&f, &BTreeMap::new()));
        assert!(!matches(&f, &attrs(&[("tags", Value::Str("zig".into()))])));
    }

    #[test]
    fn contains_any_needs_one_overlap() {
        let f = filter(vec![Predicate::ContainsAny(
            "tags".into(),
            vec![Value::Str("zig".into()), Value::Str("rust".into())],
        )]);
        assert!(matches(&f, &tagged(&["rust", "go"])));
        assert!(!matches(&f, &tagged(&["go"])));
    }

    #[test]
    fn contains_any_with_an_empty_set_fails() {
        // No candidate can overlap, so this is false even for a populated list.
        let f = filter(vec![Predicate::ContainsAny("tags".into(), vec![])]);
        assert!(!matches(&f, &tagged(&["rust"])));
    }

    #[test]
    fn contains_all_is_expressed_by_composing_with_all() {
        // No dedicated ContainsAll variant: boolean composition already covers it.
        let f = filter(vec![Predicate::All(vec![
            Predicate::Contains("tags".into(), Value::Str("rust".into())),
            Predicate::Contains("tags".into(), Value::Str("go".into())),
        ])]);
        assert!(matches(&f, &tagged(&["go", "rust", "zig"])));
        assert!(!matches(&f, &tagged(&["rust"])));
    }

    // ── Boolean composition (nidus-m50.1) ────────────────────────────────────────

    fn project(name: &str) -> Predicate {
        Predicate::Eq("project".into(), Value::Str(name.into()))
    }

    #[test]
    fn any_is_a_disjunction() {
        let f = filter(vec![Predicate::Any(vec![
            project("nidus"),
            project("beads"),
        ])]);
        assert!(matches(
            &f,
            &attrs(&[("project", Value::Str("nidus".into()))])
        ));
        assert!(matches(
            &f,
            &attrs(&[("project", Value::Str("beads".into()))])
        ));
        assert!(!matches(
            &f,
            &attrs(&[("project", Value::Str("other".into()))])
        ));
    }

    #[test]
    fn empty_group_identities() {
        // AND of nothing is true, OR of nothing is false — the standard identities,
        // and All's matches Filter's own "empty matches everything".
        assert!(matches(
            &filter(vec![Predicate::All(vec![])]),
            &BTreeMap::new()
        ));
        assert!(!matches(
            &filter(vec![Predicate::Any(vec![])]),
            &BTreeMap::new()
        ));
    }

    #[test]
    fn not_inverts_a_group() {
        let a = attrs(&[
            ("kind", Value::Str("scratch".into())),
            ("stale", Value::Bool(true)),
        ]);
        let f = filter(vec![Predicate::Not(Box::new(Predicate::All(vec![
            Predicate::Eq("kind".into(), Value::Str("scratch".into())),
            Predicate::Eq("stale".into(), Value::Bool(true)),
        ])))]);
        assert!(!matches(&f, &a));
        // Flipping either conjunct makes the inner AND false, so Not becomes true.
        let b = attrs(&[
            ("kind", Value::Str("scratch".into())),
            ("stale", Value::Bool(false)),
        ]);
        assert!(matches(&f, &b));
    }

    #[test]
    fn not_differs_from_ne_on_an_absent_key() {
        // The trap worth pinning: Ne asserts a present-and-different attribute, while
        // Not(Eq) is satisfied by the attribute simply not being there.
        let empty = BTreeMap::new();
        assert!(!matches(
            &filter(vec![Predicate::Ne("k".into(), Value::Int(1))]),
            &empty
        ));
        assert!(matches(
            &filter(vec![Predicate::Not(Box::new(Predicate::Eq(
                "k".into(),
                Value::Int(1)
            )))]),
            &empty
        ));
    }

    #[test]
    fn groups_nest_arbitrarily() {
        // (project = nidus OR project = beads) AND NOT (tags contains wip)
        let f = filter(vec![
            Predicate::Any(vec![project("nidus"), project("beads")]),
            Predicate::Not(Box::new(Predicate::Contains(
                "tags".into(),
                Value::Str("wip".into()),
            ))),
        ]);
        let mut ok = tagged(&["done"]);
        ok.insert("project".into(), Value::Str("nidus".into()));
        assert!(matches(&f, &ok));

        let mut wip = tagged(&["wip"]);
        wip.insert("project".into(), Value::Str("nidus".into()));
        assert!(!matches(&f, &wip));
    }

    #[test]
    fn deeply_nested_groups_evaluate() {
        // 64 levels of Not — well inside serde_json's 128 nesting cap, which is what
        // bounds this from an untrusted body.
        let mut p = Predicate::Eq("k".into(), Value::Int(1));
        for _ in 0..64 {
            p = Predicate::Not(Box::new(p));
        }
        // An even number of negations is the identity.
        assert!(matches(&filter(vec![p]), &attrs(&[("k", Value::Int(1))])));
    }

    #[test]
    fn group_and_containment_predicates_round_trip_through_serde() {
        let f = filter(vec![
            Predicate::Contains("tags".into(), Value::Str("rust".into())),
            Predicate::NotContains("tags".into(), Value::Str("wip".into())),
            Predicate::ContainsAny("tags".into(), vec![Value::Str("a".into())]),
            Predicate::Any(vec![project("nidus")]),
            Predicate::All(vec![project("beads")]),
            Predicate::Not(Box::new(project("other"))),
        ]);
        let bytes = bincode::serialize(&f).unwrap();
        let back: Filter = bincode::deserialize(&bytes).unwrap();
        assert_eq!(f, back);
    }

    // ── Float and DateTime (nidus-m50.4) ─────────────────────────────────────────

    #[test]
    fn float_range_and_equality() {
        let a = attrs(&[("score", Value::Float(0.75))]);
        assert!(matches(
            &filter(vec![Predicate::Gt("score".into(), Value::Float(0.5))]),
            &a
        ));
        assert!(matches(
            &filter(vec![Predicate::Le("score".into(), Value::Float(0.75))]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::Lt("score".into(), Value::Float(0.5))]),
            &a
        ));
        assert!(matches(
            &filter(vec![Predicate::Eq("score".into(), Value::Float(0.75))]),
            &a
        ));
    }

    #[test]
    fn float_does_not_compare_against_int() {
        // Same-type only, as for Str/Int — documented, and consistent with the rest.
        let a = attrs(&[("score", Value::Int(1))]);
        assert!(!matches(
            &filter(vec![Predicate::Gt("score".into(), Value::Float(0.5))]),
            &a
        ));
        let b = attrs(&[("score", Value::Float(1.0))]);
        assert!(!matches(
            &filter(vec![Predicate::Gt("score".into(), Value::Int(0))]),
            &b
        ));
    }

    #[test]
    fn nan_matches_nothing_including_itself() {
        // partial_cmp is None for NaN, so every range predicate fails rather than
        // NaN being silently ordered somewhere.
        let a = attrs(&[("score", Value::Float(f64::NAN))]);
        for p in [
            Predicate::Lt("score".into(), Value::Float(0.0)),
            Predicate::Le("score".into(), Value::Float(0.0)),
            Predicate::Gt("score".into(), Value::Float(0.0)),
            Predicate::Ge("score".into(), Value::Float(0.0)),
            Predicate::Eq("score".into(), Value::Float(f64::NAN)),
        ] {
            assert!(!matches(&filter(vec![p]), &a));
        }
    }

    #[test]
    fn negative_zero_equals_zero() {
        // IEEE equality, inherited from f64's PartialEq. Worth pinning: it is the one
        // place Value equality is not bitwise.
        let a = attrs(&[("z", Value::Float(-0.0))]);
        assert!(matches(
            &filter(vec![Predicate::Eq("z".into(), Value::Float(0.0))]),
            &a
        ));
    }

    #[test]
    fn datetime_orders_chronologically_and_is_not_an_int() {
        let a = attrs(&[("at", Value::DateTime(1_700_000_000_000))]);
        assert!(matches(
            &filter(vec![Predicate::Ge(
                "at".into(),
                Value::DateTime(1_600_000_000_000)
            )]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::Ge(
                "at".into(),
                Value::DateTime(1_800_000_000_000)
            )]),
            &a
        ));
        // A DateTime is not an Int, so the epoch number alone does not match it.
        assert!(!matches(
            &filter(vec![Predicate::Ge("at".into(), Value::Int(0))]),
            &a
        ));
    }

    #[test]
    fn appending_variants_did_not_renumber_the_existing_ones() {
        // bincode encodes the variant *index*. Every value in every existing store is
        // decoded by that index, so the pre-0.44 variants must still sit at 0..=4 —
        // this is the whole back-compat contract for adding a Value type.
        for (want_index, value) in [
            (0u32, Value::Null),
            (1, Value::Str("s".into())),
            (2, Value::Int(1)),
            (3, Value::Bool(true)),
            (4, Value::List(vec!["a".into()])),
            (5, Value::Float(1.0)),
            (6, Value::DateTime(1)),
        ] {
            let bytes = bincode::serialize(&value).unwrap();
            let tag = u32::from_le_bytes(bytes[..4].try_into().unwrap());
            assert_eq!(tag, want_index, "variant index moved for {value:?}");
        }
    }

    #[test]
    fn a_value_encoded_before_the_new_variants_still_decodes() {
        // The literal bytes an older nidus wrote for Int(42): tag 2, then the i64.
        let mut old = 2u32.to_le_bytes().to_vec();
        old.extend_from_slice(&42i64.to_le_bytes());
        let back: Value = bincode::deserialize(&old).unwrap();
        assert_eq!(back, Value::Int(42));
    }

    #[test]
    fn float_and_datetime_round_trip_through_serde() {
        let f = filter(vec![
            Predicate::Eq("a".into(), Value::Float(1.5)),
            Predicate::Ge("b".into(), Value::DateTime(1_700_000_000_000)),
        ]);
        let bytes = bincode::serialize(&f).unwrap();
        assert_eq!(bincode::deserialize::<Filter>(&bytes).unwrap(), f);
    }

    // ── Fuzzy (nidus-m50.9) ──────────────────────────────────────────────────────

    fn fuzzy(key: &str, needle: &str, n: usize) -> Filter {
        filter(vec![Predicate::Fuzzy(key.into(), needle.into(), n)])
    }

    #[test]
    fn fuzzy_zero_edits_is_exact_equality() {
        let a = attrs(&[("id", Value::Str("nidus".into()))]);
        assert!(matches(&fuzzy("id", "nidus", 0), &a));
        assert!(!matches(&fuzzy("id", "nidua", 0), &a));
    }

    #[test]
    fn fuzzy_substitution_at_the_budget_boundary() {
        let a = attrs(&[("id", Value::Str("nidus".into()))]);
        assert!(!matches(&fuzzy("id", "nidux", 0), &a));
        assert!(matches(&fuzzy("id", "nidux", 1), &a));
    }

    #[test]
    fn fuzzy_insertion_at_the_budget_boundary() {
        let a = attrs(&[("id", Value::Str("colour".into()))]);
        assert!(!matches(&fuzzy("id", "color", 0), &a));
        assert!(matches(&fuzzy("id", "color", 1), &a));
    }

    #[test]
    fn fuzzy_deletion_at_the_budget_boundary() {
        let a = attrs(&[("id", Value::Str("color".into()))]);
        assert!(!matches(&fuzzy("id", "colour", 0), &a));
        assert!(matches(&fuzzy("id", "colour", 1), &a));
    }

    #[test]
    fn fuzzy_transposition_needs_two_edits() {
        // Plain Levenshtein, not Damerau — a swap is a delete plus an insert.
        let a = attrs(&[("word", Value::Str("form".into()))]);
        assert!(!matches(&fuzzy("word", "from", 1), &a));
        assert!(matches(&fuzzy("word", "from", 2), &a));
    }

    #[test]
    fn fuzzy_folds_ascii_case_on_both_sides() {
        let a = attrs(&[("id", Value::Str("NidusStore".into()))]);
        assert!(matches(&fuzzy("id", "nidusstore", 0), &a));
        assert!(matches(&fuzzy("id", "NIDUSSTORF", 1), &a));
    }

    #[test]
    fn fuzzy_does_not_fold_non_ascii() {
        let a = attrs(&[("id", Value::Str("café".into()))]);
        assert!(matches(&fuzzy("id", "CAFé", 0), &a));
        assert!(!matches(&fuzzy("id", "CAFÉ", 0), &a));
    }

    #[test]
    fn fuzzy_looks_inside_a_list_and_matches_any_element() {
        let a = attrs(&[("tags", Value::List(vec!["rust".into(), "postgres".into()]))]);
        assert!(matches(&fuzzy("tags", "postgre", 1), &a));
        assert!(!matches(&fuzzy("tags", "postgre", 0), &a));
        assert!(!matches(&fuzzy("tags", "elixir", 1), &a));
    }

    #[test]
    fn fuzzy_absent_key_and_wrong_types_never_match() {
        let a = attrs(&[
            ("n", Value::Int(5)),
            ("b", Value::Bool(true)),
            ("nul", Value::Null),
        ]);
        assert!(!matches(&fuzzy("missing", "", 8), &a));
        assert!(!matches(&fuzzy("n", "5", 8), &a));
        assert!(!matches(&fuzzy("b", "true", 8), &a));
        assert!(!matches(&fuzzy("nul", "", 8), &a));
    }

    #[test]
    fn fuzzy_length_gap_beyond_the_budget_fails() {
        let a = attrs(&[("id", Value::Str("ab".into()))]);
        assert!(!matches(&fuzzy("id", "abcdefgh", 3), &a));
        assert!(matches(&fuzzy("id", "abcdefgh", 6), &a));
    }

    // ── Token predicates (nidus-m50.9) ───────────────────────────────────────────

    #[test]
    fn contains_all_tokens_is_order_free() {
        let a = attrs(&[("body", Value::Str("the quick brown fox".into()))]);
        assert!(matches(
            &filter(vec![Predicate::ContainsAllTokens(
                "body".into(),
                "fox quick".into()
            )]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::ContainsAllTokens(
                "body".into(),
                "fox hound".into()
            )]),
            &a
        ));
    }

    #[test]
    fn contains_all_tokens_folds_case_and_ignores_punctuation() {
        let a = attrs(&[("body", Value::Str("Hello, World!".into()))]);
        assert!(matches(
            &filter(vec![Predicate::ContainsAllTokens(
                "body".into(),
                "world HELLO".into()
            )]),
            &a
        ));
    }

    #[test]
    fn contains_all_tokens_matches_whole_tokens_not_substrings() {
        let a = attrs(&[("body", Value::Str("rustacean".into()))]);
        assert!(!matches(
            &filter(vec![Predicate::ContainsAllTokens(
                "body".into(),
                "rust".into()
            )]),
            &a
        ));
    }

    #[test]
    fn contains_all_tokens_with_an_empty_query_matches_any_present_text() {
        let a = attrs(&[
            ("body", Value::Str("anything".into())),
            ("n", Value::Int(1)),
        ]);
        assert!(matches(
            &filter(vec![Predicate::ContainsAllTokens("body".into(), "".into())]),
            &a
        ));
        // Still a leaf: the key must be present and text-bearing.
        assert!(!matches(
            &filter(vec![Predicate::ContainsAllTokens("n".into(), "".into())]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::ContainsAllTokens(
                "missing".into(),
                "".into()
            )]),
            &a
        ));
    }

    #[test]
    fn contains_any_token_needs_one_hit_and_an_empty_query_never_matches() {
        let a = attrs(&[("body", Value::Str("the quick brown fox".into()))]);
        assert!(matches(
            &filter(vec![Predicate::ContainsAnyToken(
                "body".into(),
                "hound fox".into()
            )]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::ContainsAnyToken(
                "body".into(),
                "hound wolf".into()
            )]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::ContainsAnyToken("body".into(), "".into())]),
            &a
        ));
    }

    #[test]
    fn contains_token_sequence_requires_order_and_adjacency() {
        let a = attrs(&[("body", Value::Str("the quick brown fox".into()))]);
        assert!(matches(
            &filter(vec![Predicate::ContainsTokenSequence(
                "body".into(),
                "quick brown".into()
            )]),
            &a
        ));
        // Same tokens, wrong order.
        assert!(!matches(
            &filter(vec![Predicate::ContainsTokenSequence(
                "body".into(),
                "brown quick".into()
            )]),
            &a
        ));
        // In order but not adjacent.
        assert!(!matches(
            &filter(vec![Predicate::ContainsTokenSequence(
                "body".into(),
                "quick fox".into()
            )]),
            &a
        ));
    }

    #[test]
    fn contains_token_sequence_longer_than_the_text_fails() {
        let a = attrs(&[("body", Value::Str("quick fox".into()))]);
        assert!(!matches(
            &filter(vec![Predicate::ContainsTokenSequence(
                "body".into(),
                "the quick brown fox".into()
            )]),
            &a
        ));
    }

    #[test]
    fn token_predicates_look_inside_a_list_element_wise() {
        let a = attrs(&[(
            "notes",
            Value::List(vec!["quick brown".into(), "lazy dog".into()]),
        )]);
        // One element carries the whole phrase.
        assert!(matches(
            &filter(vec![Predicate::ContainsTokenSequence(
                "notes".into(),
                "quick brown".into()
            )]),
            &a
        ));
        // A phrase never spans two elements, and neither does "all tokens".
        assert!(!matches(
            &filter(vec![Predicate::ContainsTokenSequence(
                "notes".into(),
                "brown lazy".into()
            )]),
            &a
        ));
        assert!(!matches(
            &filter(vec![Predicate::ContainsAllTokens(
                "notes".into(),
                "brown dog".into()
            )]),
            &a
        ));
        assert!(matches(
            &filter(vec![Predicate::ContainsAnyToken(
                "notes".into(),
                "dog".into()
            )]),
            &a
        ));
    }

    #[test]
    fn token_predicates_reject_absent_keys_and_non_text_values() {
        let a = attrs(&[("n", Value::Int(42)), ("nul", Value::Null)]);
        for p in [
            Predicate::ContainsAllTokens("n".into(), "42".into()),
            Predicate::ContainsAnyToken("n".into(), "42".into()),
            Predicate::ContainsTokenSequence("n".into(), "42".into()),
            Predicate::ContainsAllTokens("nul".into(), "x".into()),
            Predicate::ContainsAnyToken("missing".into(), "x".into()),
            Predicate::ContainsTokenSequence("missing".into(), "x".into()),
        ] {
            assert!(!matches(&filter(vec![p.clone()]), &a), "{p:?} matched");
        }
    }

    #[test]
    fn the_text_predicates_compose_with_not() {
        // `Not` negates a truth value, so it flips an absent key to true — the leaf
        // predicates stay false there. Same asymmetry as §7.3, retested for the new leaves.
        let a = attrs(&[("other", Value::Int(1))]);
        assert!(!matches(&fuzzy("id", "nidus", 1), &a));
        assert!(matches(
            &filter(vec![Predicate::Not(Box::new(Predicate::Fuzzy(
                "id".into(),
                "nidus".into(),
                1
            )))]),
            &a
        ));
    }

    // ── Validation: the fuzzy edit ceiling (nidus-m50.9) ─────────────────────────

    #[test]
    fn a_fuzzy_budget_at_the_ceiling_validates() {
        let f = fuzzy("id", "nidus", super::text::MAX_FUZZY_EDITS);
        assert!(super::validate(&f).is_ok());
    }

    #[test]
    fn a_fuzzy_budget_over_the_ceiling_is_an_error_not_a_clamp() {
        let f = fuzzy("id", "nidus", super::text::MAX_FUZZY_EDITS + 1);
        let err = super::validate(&f).unwrap_err().to_string();
        assert!(err.contains("Fuzzy"), "{err}");
        assert!(err.contains("id"), "{err}");
    }

    #[test]
    fn an_over_budget_fuzzy_nested_in_a_group_is_still_caught() {
        let bad = Predicate::Fuzzy("id".into(), "nidus".into(), 99);
        let f = filter(vec![Predicate::Not(Box::new(Predicate::Any(vec![
            Predicate::Eq("k".into(), Value::Int(1)),
            Predicate::All(vec![bad]),
        ])))]);
        assert!(super::validate(&f).is_err());
    }

    #[test]
    fn validation_passes_a_filter_with_no_fuzzy_predicate() {
        let f = filter(vec![
            Predicate::ContainsAllTokens("body".into(), "a b".into()),
            Predicate::Eq("k".into(), Value::Int(1)),
        ]);
        assert!(super::validate(&f).is_ok());
    }

    // ── Predicate variant indices (the bincode back-compat contract) ─────────────

    #[test]
    fn appending_predicates_did_not_renumber_the_existing_ones() {
        // bincode tags an enum by its declaration index, and clients hard-code filters, so
        // a new Predicate must only ever be appended. Inserting one silently reinterprets
        // every filter already in flight.
        for (want_index, predicate) in [
            (0u32, Predicate::Eq("k".into(), Value::Null)),
            (1, Predicate::Ne("k".into(), Value::Null)),
            (2, Predicate::Glob("k".into(), "*".into())),
            (3, Predicate::IGlob("k".into(), "*".into())),
            (4, Predicate::In("k".into(), vec![])),
            (5, Predicate::NotIn("k".into(), vec![])),
            (6, Predicate::Lt("k".into(), Value::Int(0))),
            (7, Predicate::Le("k".into(), Value::Int(0))),
            (8, Predicate::Gt("k".into(), Value::Int(0))),
            (9, Predicate::Ge("k".into(), Value::Int(0))),
            (10, Predicate::Contains("k".into(), Value::Null)),
            (11, Predicate::NotContains("k".into(), Value::Null)),
            (12, Predicate::ContainsAny("k".into(), vec![])),
            (13, Predicate::All(vec![])),
            (14, Predicate::Any(vec![])),
            (15, Predicate::Not(Box::new(Predicate::All(vec![])))),
            (16, Predicate::Fuzzy("k".into(), "s".into(), 1)),
            (17, Predicate::ContainsAllTokens("k".into(), "s".into())),
            (18, Predicate::ContainsAnyToken("k".into(), "s".into())),
            (19, Predicate::ContainsTokenSequence("k".into(), "s".into())),
            (20, Predicate::Regex("k".into(), "s".into())),
        ] {
            let bytes = bincode::serialize(&predicate).unwrap();
            let tag = u32::from_le_bytes(bytes[..4].try_into().unwrap());
            assert_eq!(tag, want_index, "variant index moved for {predicate:?}");
        }
    }

    #[test]
    fn the_text_predicates_round_trip_through_serde() {
        let f = filter(vec![
            Predicate::Fuzzy("a".into(), "nidus".into(), 2),
            Predicate::ContainsAllTokens("b".into(), "quick brown".into()),
            Predicate::ContainsAnyToken("c".into(), "fox hound".into()),
            Predicate::ContainsTokenSequence("d".into(), "lazy dog".into()),
            Predicate::Regex("e".into(), "(?i)^v[0-9]+$".into()),
        ]);
        let bytes = bincode::serialize(&f).unwrap();
        assert_eq!(bincode::deserialize::<Filter>(&bytes).unwrap(), f);
    }

    // ── Regex (nidus-m50.9) ──────────────────────────────────────────────────────

    fn regex(key: &str, p: &str) -> Filter {
        filter(vec![Predicate::Regex(key.into(), p.into())])
    }

    #[test]
    fn regex_is_anchored_at_both_ends_like_glob() {
        let a = attrs(&[("path", Value::Str("src/store/read.rs".into()))]);
        assert!(matches(&regex("path", "src/.*\\.rs"), &a));
        // A bare substring does not match; `.*` opts back in.
        assert!(!matches(&regex("path", "store"), &a));
        assert!(matches(&regex("path", ".*store.*"), &a));
    }

    #[test]
    fn regex_case_sensitivity_is_the_patterns_own_flag() {
        let a = attrs(&[("file", Value::Str("README.md".into()))]);
        assert!(!matches(&regex("file", "readme\\.md"), &a));
        assert!(matches(&regex("file", "(?i)readme\\.md"), &a));
    }

    #[test]
    fn regex_looks_inside_a_list() {
        let a = attrs(&[("tags", Value::List(vec!["v1".into(), "beta".into()]))]);
        assert!(matches(&regex("tags", "v[0-9]+"), &a));
        assert!(!matches(&regex("tags", "v[0-9]{2}"), &a));
    }

    #[test]
    fn regex_absent_key_and_non_text_values_never_match() {
        let a = attrs(&[("n", Value::Int(42)), ("nul", Value::Null)]);
        assert!(!matches(&regex("missing", ".*"), &a));
        assert!(!matches(&regex("n", ".*"), &a));
        assert!(!matches(&regex("nul", ".*"), &a));
    }

    #[test]
    fn an_invalid_regex_is_a_validation_error() {
        let err = super::validate(&regex("path", "(unclosed"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Regex on `path`"), "{err}");
    }

    #[test]
    fn an_invalid_regex_nested_in_a_group_is_still_caught() {
        let f = filter(vec![Predicate::Not(Box::new(Predicate::Any(vec![
            Predicate::Regex("k".into(), "[".into()),
        ])))]);
        assert!(super::validate(&f).is_err());
    }

    #[test]
    fn a_valid_regex_validates() {
        assert!(super::validate(&regex("path", "src/[a-z_]+\\.rs")).is_ok());
    }
}
