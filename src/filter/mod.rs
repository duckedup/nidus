//! Filter evaluation against a record's attributes. Contract: see the root `SPEC.md` §7, §7.1.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::model::{Filter, Predicate, Value};

/// Same-type ordering of two [`Value`]s for the range predicates. Returns `None` when
/// the two values are not comparable: different variants, or a non-orderable variant
/// (`Null`, `List`). `Int`↔`Int` is numeric, `Str`↔`Str` lexical, `Bool`↔`Bool` orders
/// `false < true`. The `None` case is what makes a range predicate fail on an
/// absent/wrong-type attribute rather than match spuriously.
fn value_cmp(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
        (Value::Str(x), Value::Str(y)) => Some(x.cmp(y)),
        (Value::Bool(x), Value::Bool(y)) => Some(x.cmp(y)),
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

/// True iff every predicate in `filter` matches `attrs` (an empty filter matches
/// everything). Every predicate requires `key` to be present — an absent attribute
/// matches nothing, including the negative (`Ne`/`NotIn`) and range predicates. See the
/// root `SPEC.md` §7.1 for the full per-predicate semantics.
pub fn matches(filter: &Filter, attrs: &BTreeMap<String, Value>) -> bool {
    use Ordering::{Equal, Greater, Less};
    filter.0.iter().all(|predicate| match predicate {
        Predicate::Eq(key, expected) => attrs.get(key) == Some(expected),
        Predicate::Ne(key, expected) => matches!(attrs.get(key), Some(v) if v != expected),
        Predicate::Glob(key, pattern) => match attrs.get(key) {
            Some(Value::Str(s)) => crate::glob::glob_match(pattern, s),
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
        ];
        let f = filter(preds);
        let bytes = bincode::serialize(&f).unwrap();
        let back: Filter = bincode::deserialize(&bytes).unwrap();
        assert_eq!(f, back);
    }
}
