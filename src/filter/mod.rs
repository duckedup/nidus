//! Filter evaluation against a record's attributes. Contract: see `SPEC.md` here.

use std::collections::BTreeMap;

use crate::model::{Filter, Predicate, Value};

/// True iff every predicate in `filter` matches `attrs` (an empty filter matches
/// everything). See `SPEC.md` for per-predicate semantics (`Eq`, `Glob`, `In`).
pub fn matches(filter: &Filter, attrs: &BTreeMap<String, Value>) -> bool {
    filter.0.iter().all(|predicate| match predicate {
        Predicate::Eq(key, expected) => attrs.get(key) == Some(expected),
        Predicate::Glob(key, pattern) => match attrs.get(key) {
            Some(Value::Str(s)) => crate::glob::glob_match(pattern, s),
            _ => false,
        },
        Predicate::In(key, set) => match attrs.get(key) {
            Some(v) => set.contains(v),
            None => false,
        },
    })
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
        let a = attrs(&[("tags", Value::List(vec!["rust".into()]))]);
        let f = filter(vec![Predicate::Glob("tags".into(), "*".into())]);
        assert!(!matches(&f, &a));
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
}
