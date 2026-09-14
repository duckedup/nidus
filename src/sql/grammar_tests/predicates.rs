//! The `WHERE` grammar, exhaustively: every comparison operator against every literal type,
//! every predicate spelling, precedence/grouping/negation, field-name shapes, and keyword
//! casing. Each test asserts the parsed [`PredNode`] tree or a real `err_at` offset, never
//! merely that parsing succeeded (root blueprint's bar).

use super::super::error::{SEC_ARRAY, SEC_GLOB, SEC_REGEX, SEC_SYNTAX, SEC_TEXT};
use super::super::parse::{CmpOp, Lit, PredNode};
use super::{err_at, one, same};

// ── every comparison operator, every literal type ───────────────────────────────

#[test]
fn eq_int() {
    let s = one("SELECT * FROM docs WHERE a = 1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn eq_negative_int() {
    let s = one("SELECT * FROM docs WHERE a = -1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Int(-1)
        })
    );
}

#[test]
fn eq_float() {
    let s = one("SELECT * FROM docs WHERE a = 1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Float(1.5)
        })
    );
}

#[test]
fn eq_negative_float() {
    let s = one("SELECT * FROM docs WHERE a = -1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Float(-1.5)
        })
    );
}

#[test]
fn eq_string() {
    let s = one("SELECT * FROM docs WHERE a = 'x'");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Str("x".into())
        })
    );
}

#[test]
fn eq_true() {
    let s = one("SELECT * FROM docs WHERE a = true");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Bool(true)
        })
    );
}

#[test]
fn eq_false() {
    let s = one("SELECT * FROM docs WHERE a = false");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Bool(false)
        })
    );
}

#[test]
fn eq_null() {
    let s = one("SELECT * FROM docs WHERE a = null");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Null
        })
    );
}

#[test]
fn ne_int() {
    let s = one("SELECT * FROM docs WHERE a != 1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ne,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn ne_negative_int() {
    let s = one("SELECT * FROM docs WHERE a != -1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ne,
            value: Lit::Int(-1)
        })
    );
}

#[test]
fn ne_float() {
    let s = one("SELECT * FROM docs WHERE a != 1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ne,
            value: Lit::Float(1.5)
        })
    );
}

#[test]
fn ne_negative_float() {
    let s = one("SELECT * FROM docs WHERE a != -1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ne,
            value: Lit::Float(-1.5)
        })
    );
}

#[test]
fn ne_string() {
    let s = one("SELECT * FROM docs WHERE a != 'x'");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ne,
            value: Lit::Str("x".into())
        })
    );
}

#[test]
fn ne_true() {
    let s = one("SELECT * FROM docs WHERE a != true");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ne,
            value: Lit::Bool(true)
        })
    );
}

#[test]
fn ne_false() {
    let s = one("SELECT * FROM docs WHERE a != false");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ne,
            value: Lit::Bool(false)
        })
    );
}

#[test]
fn ne_null() {
    let s = one("SELECT * FROM docs WHERE a != null");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ne,
            value: Lit::Null
        })
    );
}

#[test]
fn lt_int() {
    let s = one("SELECT * FROM docs WHERE a < 1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Lt,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn lt_negative_int() {
    let s = one("SELECT * FROM docs WHERE a < -1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Lt,
            value: Lit::Int(-1)
        })
    );
}

#[test]
fn lt_float() {
    let s = one("SELECT * FROM docs WHERE a < 1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Lt,
            value: Lit::Float(1.5)
        })
    );
}

#[test]
fn lt_negative_float() {
    let s = one("SELECT * FROM docs WHERE a < -1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Lt,
            value: Lit::Float(-1.5)
        })
    );
}

#[test]
fn lt_string() {
    let s = one("SELECT * FROM docs WHERE a < 'x'");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Lt,
            value: Lit::Str("x".into())
        })
    );
}

#[test]
fn lt_true() {
    let s = one("SELECT * FROM docs WHERE a < true");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Lt,
            value: Lit::Bool(true)
        })
    );
}

#[test]
fn lt_false() {
    let s = one("SELECT * FROM docs WHERE a < false");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Lt,
            value: Lit::Bool(false)
        })
    );
}

#[test]
fn lt_null() {
    let s = one("SELECT * FROM docs WHERE a < null");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Lt,
            value: Lit::Null
        })
    );
}

#[test]
fn le_int() {
    let s = one("SELECT * FROM docs WHERE a <= 1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Le,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn le_negative_int() {
    let s = one("SELECT * FROM docs WHERE a <= -1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Le,
            value: Lit::Int(-1)
        })
    );
}

#[test]
fn le_float() {
    let s = one("SELECT * FROM docs WHERE a <= 1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Le,
            value: Lit::Float(1.5)
        })
    );
}

#[test]
fn le_negative_float() {
    let s = one("SELECT * FROM docs WHERE a <= -1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Le,
            value: Lit::Float(-1.5)
        })
    );
}

#[test]
fn le_string() {
    let s = one("SELECT * FROM docs WHERE a <= 'x'");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Le,
            value: Lit::Str("x".into())
        })
    );
}

#[test]
fn le_true() {
    let s = one("SELECT * FROM docs WHERE a <= true");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Le,
            value: Lit::Bool(true)
        })
    );
}

#[test]
fn le_false() {
    let s = one("SELECT * FROM docs WHERE a <= false");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Le,
            value: Lit::Bool(false)
        })
    );
}

#[test]
fn le_null() {
    let s = one("SELECT * FROM docs WHERE a <= null");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Le,
            value: Lit::Null
        })
    );
}

#[test]
fn gt_int() {
    let s = one("SELECT * FROM docs WHERE a > 1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Gt,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn gt_negative_int() {
    let s = one("SELECT * FROM docs WHERE a > -1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Gt,
            value: Lit::Int(-1)
        })
    );
}

#[test]
fn gt_float() {
    let s = one("SELECT * FROM docs WHERE a > 1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Gt,
            value: Lit::Float(1.5)
        })
    );
}

#[test]
fn gt_negative_float() {
    let s = one("SELECT * FROM docs WHERE a > -1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Gt,
            value: Lit::Float(-1.5)
        })
    );
}

#[test]
fn gt_string() {
    let s = one("SELECT * FROM docs WHERE a > 'x'");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Gt,
            value: Lit::Str("x".into())
        })
    );
}

#[test]
fn gt_true() {
    let s = one("SELECT * FROM docs WHERE a > true");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Gt,
            value: Lit::Bool(true)
        })
    );
}

#[test]
fn gt_false() {
    let s = one("SELECT * FROM docs WHERE a > false");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Gt,
            value: Lit::Bool(false)
        })
    );
}

#[test]
fn gt_null() {
    let s = one("SELECT * FROM docs WHERE a > null");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Gt,
            value: Lit::Null
        })
    );
}

#[test]
fn ge_int() {
    let s = one("SELECT * FROM docs WHERE a >= 1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ge,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn ge_negative_int() {
    let s = one("SELECT * FROM docs WHERE a >= -1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ge,
            value: Lit::Int(-1)
        })
    );
}

#[test]
fn ge_float() {
    let s = one("SELECT * FROM docs WHERE a >= 1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ge,
            value: Lit::Float(1.5)
        })
    );
}

#[test]
fn ge_negative_float() {
    let s = one("SELECT * FROM docs WHERE a >= -1.5");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ge,
            value: Lit::Float(-1.5)
        })
    );
}

#[test]
fn ge_string() {
    let s = one("SELECT * FROM docs WHERE a >= 'x'");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ge,
            value: Lit::Str("x".into())
        })
    );
}

#[test]
fn ge_true() {
    let s = one("SELECT * FROM docs WHERE a >= true");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ge,
            value: Lit::Bool(true)
        })
    );
}

#[test]
fn ge_false() {
    let s = one("SELECT * FROM docs WHERE a >= false");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ge,
            value: Lit::Bool(false)
        })
    );
}

#[test]
fn ge_null() {
    let s = one("SELECT * FROM docs WHERE a >= null");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Ge,
            value: Lit::Null
        })
    );
}

// ── IN / NOT IN ──────────────────────────────────────────────────────────────

#[test]
fn in_one_element() {
    let s = one("SELECT * FROM docs WHERE a IN (1)");
    assert_eq!(
        s.filter,
        Some(PredNode::In {
            field: "a".into(),
            negate: false,
            values: vec![Lit::Int(1)]
        })
    );
}

#[test]
fn in_many_elements() {
    let s = one("SELECT * FROM docs WHERE a IN (1, 2, 3)");
    assert_eq!(
        s.filter,
        Some(PredNode::In {
            field: "a".into(),
            negate: false,
            values: vec![Lit::Int(1), Lit::Int(2), Lit::Int(3)]
        })
    );
}

#[test]
fn in_nested_quotes() {
    let s = one("SELECT * FROM docs WHERE a IN ('it''s here', 'plain')");
    assert_eq!(
        s.filter,
        Some(PredNode::In {
            field: "a".into(),
            negate: false,
            values: vec![Lit::Str("it's here".into()), Lit::Str("plain".into())]
        })
    );
}

#[test]
fn in_with_mixed_literal_types() {
    let s = one("SELECT * FROM docs WHERE a IN (1, 'x', true, null)");
    assert_eq!(
        s.filter,
        Some(PredNode::In {
            field: "a".into(),
            negate: false,
            values: vec![
                Lit::Int(1),
                Lit::Str("x".into()),
                Lit::Bool(true),
                Lit::Null
            ]
        })
    );
}

#[test]
fn not_in_one_element() {
    let s = one("SELECT * FROM docs WHERE a NOT IN (1)");
    assert_eq!(
        s.filter,
        Some(PredNode::In {
            field: "a".into(),
            negate: true,
            values: vec![Lit::Int(1)]
        })
    );
}

#[test]
fn not_in_many_elements() {
    let s = one("SELECT * FROM docs WHERE a NOT IN ('x', 'y', 'z')");
    assert_eq!(
        s.filter,
        Some(PredNode::In {
            field: "a".into(),
            negate: true,
            values: vec![
                Lit::Str("x".into()),
                Lit::Str("y".into()),
                Lit::Str("z".into())
            ]
        })
    );
}

#[test]
fn not_in_with_escaped_quote() {
    let s = one("SELECT * FROM docs WHERE a NOT IN ('it''s ok')");
    assert_eq!(
        s.filter,
        Some(PredNode::In {
            field: "a".into(),
            negate: true,
            values: vec![Lit::Str("it's ok".into())]
        })
    );
}

#[test]
fn in_missing_open_paren_errors() {
    let sql = "SELECT * FROM docs WHERE a IN 1";
    err_at(sql, sql.find('1').unwrap(), "expected '('", SEC_SYNTAX);
}

#[test]
fn in_empty_list_errors() {
    let sql = "SELECT * FROM docs WHERE a IN ()";
    err_at(sql, sql.rfind(')').unwrap(), "expected a value", SEC_SYNTAX);
}

#[test]
fn in_unclosed_paren_errors() {
    let sql = "SELECT * FROM docs WHERE a IN (1";
    err_at(sql, sql.len(), "expected ')'", SEC_SYNTAX);
}

#[test]
fn not_in_after_not_without_in_or_like_errors() {
    let sql = "SELECT * FROM docs WHERE a NOT 5";
    err_at(
        sql,
        sql.find('5').unwrap(),
        "expected 'IN' or 'LIKE'",
        SEC_SYNTAX,
    );
}

// ── LIKE / NOT LIKE / ILIKE / regex (~) ──────────────────────────────────────

#[test]
fn like_happy() {
    let s = one("SELECT * FROM docs WHERE a LIKE 'x*'");
    assert_eq!(
        s.filter,
        Some(PredNode::Like {
            field: "a".into(),
            negate: false,
            pattern: "x*".into()
        })
    );
}

#[test]
fn not_like_happy() {
    let s = one("SELECT * FROM docs WHERE a NOT LIKE 'y*'");
    assert_eq!(
        s.filter,
        Some(PredNode::Like {
            field: "a".into(),
            negate: true,
            pattern: "y*".into()
        })
    );
}

#[test]
fn like_pattern_with_escaped_quote() {
    let s = one("SELECT * FROM docs WHERE a LIKE 'it''s*'");
    assert_eq!(
        s.filter,
        Some(PredNode::Like {
            field: "a".into(),
            negate: false,
            pattern: "it's*".into()
        })
    );
}

#[test]
fn like_missing_pattern_errors() {
    let sql = "SELECT * FROM docs WHERE a LIKE";
    err_at(sql, sql.len(), "expected a string literal", SEC_GLOB);
}

#[test]
fn like_non_string_pattern_errors() {
    let sql = "SELECT * FROM docs WHERE a LIKE 5";
    err_at(
        sql,
        sql.find('5').unwrap(),
        "expected a string literal",
        SEC_GLOB,
    );
}

#[test]
fn not_like_missing_pattern_errors() {
    let sql = "SELECT * FROM docs WHERE a NOT LIKE";
    err_at(sql, sql.len(), "expected a string literal", SEC_GLOB);
}

#[test]
fn ilike_happy() {
    let s = one("SELECT * FROM docs WHERE a ILIKE 'Z*'");
    assert_eq!(
        s.filter,
        Some(PredNode::ILike {
            field: "a".into(),
            pattern: "Z*".into()
        })
    );
}

#[test]
fn ilike_pattern_case_preserved() {
    let s = one("SELECT * FROM docs WHERE a ILIKE 'MiXeD'");
    assert_eq!(
        s.filter,
        Some(PredNode::ILike {
            field: "a".into(),
            pattern: "MiXeD".into()
        })
    );
}

#[test]
fn ilike_missing_pattern_errors() {
    let sql = "SELECT * FROM docs WHERE a ILIKE";
    err_at(sql, sql.len(), "expected a string literal", SEC_GLOB);
}

#[test]
fn ilike_non_string_pattern_errors() {
    let sql = "SELECT * FROM docs WHERE a ILIKE true";
    err_at(
        sql,
        sql.find("true").unwrap(),
        "expected a string literal",
        SEC_GLOB,
    );
}

#[test]
fn regex_happy() {
    let s = one("SELECT * FROM docs WHERE a ~ 'v[0-9]+'");
    assert_eq!(
        s.filter,
        Some(PredNode::Regex {
            field: "a".into(),
            pattern: "v[0-9]+".into()
        })
    );
}

#[test]
fn regex_missing_pattern_errors() {
    let sql = "SELECT * FROM docs WHERE a ~";
    err_at(sql, sql.len(), "expected a string literal", SEC_REGEX);
}

#[test]
fn regex_non_string_pattern_errors() {
    let sql = "SELECT * FROM docs WHERE a ~ 5";
    err_at(
        sql,
        sql.find('5').unwrap(),
        "expected a string literal",
        SEC_REGEX,
    );
}

// ── function predicates ──────────────────────────────────────────────────────

#[test]
fn contains_happy() {
    let s = one("SELECT * FROM docs WHERE contains(tags, 'rust')");
    assert_eq!(
        s.filter,
        Some(PredNode::Contains {
            field: "tags".into(),
            value: Lit::Str("rust".into())
        })
    );
}

#[test]
fn contains_missing_comma_errors() {
    let sql = "SELECT * FROM docs WHERE contains(tags 'rust')";
    err_at(sql, sql.find('\'').unwrap(), "expected ','", SEC_ARRAY);
}

#[test]
fn contains_missing_value_errors() {
    let sql = "SELECT * FROM docs WHERE contains(tags,)";
    err_at(sql, sql.rfind(')').unwrap(), "expected a value", SEC_SYNTAX);
}

#[test]
fn contains_missing_field_errors() {
    let sql = "SELECT * FROM docs WHERE contains(, 'x')";
    err_at(
        sql,
        sql.find(',').unwrap(),
        "expected a field name",
        SEC_ARRAY,
    );
}

#[test]
fn not_contains_happy() {
    let s = one("SELECT * FROM docs WHERE not_contains(tags, 'wip')");
    assert_eq!(
        s.filter,
        Some(PredNode::NotContains {
            field: "tags".into(),
            value: Lit::Str("wip".into())
        })
    );
}

#[test]
fn contains_any_happy_two() {
    let s = one("SELECT * FROM docs WHERE contains_any(tags, 'a', 'b')");
    assert_eq!(
        s.filter,
        Some(PredNode::ContainsAny {
            field: "tags".into(),
            values: vec![Lit::Str("a".into()), Lit::Str("b".into())]
        })
    );
}

#[test]
fn contains_any_happy_one() {
    let s = one("SELECT * FROM docs WHERE contains_any(tags, 'a')");
    assert_eq!(
        s.filter,
        Some(PredNode::ContainsAny {
            field: "tags".into(),
            values: vec![Lit::Str("a".into())]
        })
    );
}

#[test]
fn contains_any_with_three_elements() {
    let s = one("SELECT * FROM docs WHERE contains_any(tags, 'a', 'b', 'c')");
    assert_eq!(
        s.filter,
        Some(PredNode::ContainsAny {
            field: "tags".into(),
            values: vec![
                Lit::Str("a".into()),
                Lit::Str("b".into()),
                Lit::Str("c".into())
            ]
        })
    );
}

#[test]
fn contains_any_unclosed_paren_errors() {
    let sql = "SELECT * FROM docs WHERE contains_any(tags, 'a'";
    err_at(sql, sql.len(), "expected ')'", SEC_ARRAY);
}

#[test]
fn fuzzy_happy() {
    let sql = "SELECT * FROM docs WHERE fuzzy(id, 'nidus', 2)";
    let s = one(sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Fuzzy {
            field: "id".into(),
            needle: "nidus".into(),
            edits: 2,
            at: sql.rfind('2').unwrap(),
        })
    );
}

#[test]
fn fuzzy_edits_as_float_errors() {
    let sql = "SELECT * FROM docs WHERE fuzzy(id, 'nidus', 2.5)";
    err_at(
        sql,
        sql.rfind(')').unwrap(),
        "expected an integer, found a float",
        SEC_SYNTAX,
    );
}

#[test]
fn fuzzy_needle_non_string_errors() {
    let sql = "SELECT * FROM docs WHERE fuzzy(id, 2)";
    err_at(
        sql,
        sql.find('2').unwrap(),
        "expected a string literal",
        SEC_TEXT,
    );
}

#[test]
fn match_all_happy() {
    let s = one("SELECT * FROM docs WHERE match_all(body, 'a b')");
    assert_eq!(
        s.filter,
        Some(PredNode::MatchAll {
            field: "body".into(),
            query: "a b".into()
        })
    );
}

#[test]
fn match_all_non_string_query_errors() {
    let sql = "SELECT * FROM docs WHERE match_all(body, 5)";
    err_at(
        sql,
        sql.find('5').unwrap(),
        "expected a string literal",
        SEC_TEXT,
    );
}

#[test]
fn match_any_happy() {
    let s = one("SELECT * FROM docs WHERE match_any(body, 'c d')");
    assert_eq!(
        s.filter,
        Some(PredNode::MatchAny {
            field: "body".into(),
            query: "c d".into()
        })
    );
}

#[test]
fn match_any_missing_comma_errors() {
    let sql = "SELECT * FROM docs WHERE match_any(body)";
    err_at(sql, sql.rfind(')').unwrap(), "expected ','", SEC_TEXT);
}

#[test]
fn phrase_happy() {
    let s = one("SELECT * FROM docs WHERE phrase(body, 'e f')");
    assert_eq!(
        s.filter,
        Some(PredNode::Phrase {
            field: "body".into(),
            query: "e f".into()
        })
    );
}

#[test]
fn phrase_missing_comma_errors() {
    let sql = "SELECT * FROM docs WHERE phrase(body 'e f')";
    err_at(sql, sql.find('\'').unwrap(), "expected ','", SEC_TEXT);
}

#[test]
fn phrase_unclosed_paren_errors() {
    let sql = "SELECT * FROM docs WHERE phrase(body, 'e f'";
    err_at(sql, sql.len(), "expected ')'", SEC_TEXT);
}

#[test]
fn bare_word_without_parens_is_treated_as_a_field_name() {
    let sql = "SELECT * FROM docs WHERE contains tags";
    err_at(
        sql,
        sql.find("tags").unwrap(),
        "expected a comparison operator",
        SEC_SYNTAX,
    );
}

// ── precedence, grouping, negation ───────────────────────────────────────────

#[test]
fn or_binds_looser_than_and() {
    // "a=1 OR b=2 AND c=3" is Or(a, And(b, c)).
    let s = one("SELECT * FROM docs WHERE a = 1 OR b = 2 AND c = 3");
    let PredNode::Any(items) = s.filter.unwrap() else {
        panic!("expected Any at the top")
    };
    assert_eq!(items.len(), 2);
    assert!(matches!(items[0], PredNode::Cmp { .. }));
    let PredNode::All(inner) = &items[1] else {
        panic!("expected All")
    };
    assert_eq!(inner.len(), 2);
}

#[test]
fn and_binds_tighter_than_or() {
    // "a=1 AND b=2 OR c=3" is Or(And(a, b), c).
    let s = one("SELECT * FROM docs WHERE a = 1 AND b = 2 OR c = 3");
    let PredNode::Any(items) = s.filter.unwrap() else {
        panic!("expected Any at the top")
    };
    assert_eq!(items.len(), 2);
    let PredNode::All(inner) = &items[0] else {
        panic!("expected All")
    };
    assert_eq!(inner.len(), 2);
    assert!(matches!(items[1], PredNode::Cmp { .. }));
}

#[test]
fn parens_override_or_then_and() {
    let s = one("SELECT * FROM docs WHERE (a = 1 OR b = 2) AND c = 3");
    let PredNode::All(items) = s.filter.unwrap() else {
        panic!("expected All at the top")
    };
    assert_eq!(items.len(), 2);
    assert!(matches!(items[0], PredNode::Any(_)));
    assert!(matches!(items[1], PredNode::Cmp { .. }));
}

#[test]
fn parens_override_and_then_or() {
    let s = one("SELECT * FROM docs WHERE a = 1 AND (b = 2 OR c = 3)");
    let PredNode::All(items) = s.filter.unwrap() else {
        panic!("expected All at the top")
    };
    assert_eq!(items.len(), 2);
    assert!(matches!(items[0], PredNode::Cmp { .. }));
    assert!(matches!(items[1], PredNode::Any(_)));
}

#[test]
fn not_binds_tighter_than_and() {
    // "NOT a=1 AND b=2" is And(Not(a), b).
    let s = one("SELECT * FROM docs WHERE NOT a = 1 AND b = 2");
    let PredNode::All(items) = s.filter.unwrap() else {
        panic!("expected All at the top")
    };
    assert_eq!(items.len(), 2);
    assert!(matches!(items[0], PredNode::Not(_)));
    assert!(matches!(items[1], PredNode::Cmp { .. }));
}

#[test]
fn bare_double_not_errors() {
    // A bare repeated NOT is not legal: `parse_not` only strips one NOT before
    // descending to `primary`, which does not itself accept a leading NOT.
    let sql = "SELECT * FROM docs WHERE NOT NOT a = 1";
    err_at(
        sql,
        sql.rfind("NOT").unwrap(),
        "expected a field name",
        SEC_SYNTAX,
    );
}

#[test]
fn parenthesized_double_not() {
    let s = one("SELECT * FROM docs WHERE NOT (NOT a = 1)");
    let PredNode::Not(outer) = s.filter.unwrap() else {
        panic!("expected Not at the top")
    };
    assert!(matches!(*outer, PredNode::Not(_)));
}

#[test]
fn not_over_parenthesized_group() {
    let s = one("SELECT * FROM docs WHERE NOT (a = 1 OR b = 2)");
    let PredNode::Not(inner) = s.filter.unwrap() else {
        panic!("expected Not at the top")
    };
    assert!(matches!(*inner, PredNode::Any(_)));
}

#[test]
fn three_levels_mixed_without_parens() {
    // "a=1 OR b=2 AND NOT c=3 OR d=4" is Any[Cmp(a), All[Cmp(b), Not(Cmp(c))], Cmp(d)] flat.
    let s = one("SELECT * FROM docs WHERE a = 1 OR b = 2 AND NOT c = 3 OR d = 4");
    let PredNode::Any(items) = s.filter.unwrap() else {
        panic!("expected Any at the top")
    };
    assert_eq!(items.len(), 3);
    assert!(matches!(items[0], PredNode::Cmp { .. }));
    let PredNode::All(mid) = &items[1] else {
        panic!("expected All in the middle")
    };
    assert!(matches!(mid[0], PredNode::Cmp { .. }));
    assert!(matches!(mid[1], PredNode::Not(_)));
    assert!(matches!(items[2], PredNode::Cmp { .. }));
}

#[test]
fn four_levels_mixed_with_parens() {
    let s = one("SELECT * FROM docs WHERE (a = 1 OR b = 2) AND (c = 3 OR NOT d = 4)");
    let PredNode::All(items) = s.filter.unwrap() else {
        panic!("expected All at the top")
    };
    assert_eq!(items.len(), 2);
    let PredNode::Any(left) = &items[0] else {
        panic!("expected Any on the left")
    };
    assert_eq!(left.len(), 2);
    let PredNode::Any(right) = &items[1] else {
        panic!("expected Any on the right")
    };
    assert_eq!(right.len(), 2);
    assert!(matches!(right[1], PredNode::Not(_)));
}

#[test]
fn not_over_a_nested_and_or_group() {
    let s = one("SELECT * FROM docs WHERE NOT ((a = 1 OR b = 2) AND c = 3)");
    let PredNode::Not(inner) = s.filter.unwrap() else {
        panic!("expected Not at the top")
    };
    let PredNode::All(items) = *inner else {
        panic!("expected All inside the Not")
    };
    assert!(matches!(items[0], PredNode::Any(_)));
    assert!(matches!(items[1], PredNode::Cmp { .. }));
}

#[test]
fn and_chain_of_three_is_flat() {
    let s = one("SELECT * FROM docs WHERE a = 1 AND b = 2 AND c = 3");
    let PredNode::All(items) = s.filter.unwrap() else {
        panic!("expected All at the top")
    };
    assert_eq!(items.len(), 3);
}

#[test]
fn or_chain_of_three_is_flat() {
    let s = one("SELECT * FROM docs WHERE a = 1 OR b = 2 OR c = 3");
    let PredNode::Any(items) = s.filter.unwrap() else {
        panic!("expected Any at the top")
    };
    assert_eq!(items.len(), 3);
}

#[test]
fn same_and_or_precedence_matches_explicit_parens() {
    same(
        "SELECT * FROM docs WHERE a = 1 AND b = 2 OR c = 3",
        "SELECT * FROM docs WHERE (a = 1 AND b = 2) OR c = 3",
    );
}

#[test]
fn same_or_and_precedence_matches_explicit_parens() {
    same(
        "SELECT * FROM docs WHERE a = 1 OR b = 2 AND c = 3",
        "SELECT * FROM docs WHERE a = 1 OR (b = 2 AND c = 3)",
    );
}

#[test]
fn same_not_and_precedence_matches_explicit_parens() {
    same(
        "SELECT * FROM docs WHERE NOT a = 1 AND b = 2",
        "SELECT * FROM docs WHERE (NOT a = 1) AND b = 2",
    );
}

// ── field names ───────────────────────────────────────────────────────────────

#[test]
fn bare_field_name() {
    let s = one("SELECT * FROM docs WHERE myfield = 1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "myfield".into(),
            op: CmpOp::Eq,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn dotted_field_name() {
    let s = one("SELECT * FROM docs WHERE nidus.text = 1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "nidus.text".into(),
            op: CmpOp::Eq,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn quoted_field_with_a_dot() {
    let s = one(r#"SELECT * FROM docs WHERE "a.b" = 1"#);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a.b".into(),
            op: CmpOp::Eq,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn quoted_field_with_a_space() {
    let s = one(r#"SELECT * FROM docs WHERE "a field" = 1"#);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a field".into(),
            op: CmpOp::Eq,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn quoted_reserved_word_order() {
    let s = one(r#"SELECT * FROM docs WHERE "order" = 1"#);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "order".into(),
            op: CmpOp::Eq,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn quoted_reserved_word_select() {
    let s = one(r#"SELECT * FROM docs WHERE "select" = 1"#);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "select".into(),
            op: CmpOp::Eq,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn quoted_reserved_word_limit() {
    let s = one(r#"SELECT * FROM docs WHERE "limit" = 1"#);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "limit".into(),
            op: CmpOp::Eq,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn leading_underscore_field() {
    let s = one("SELECT * FROM docs WHERE _id = 1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "_id".into(),
            op: CmpOp::Eq,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn field_with_digits_after_first_char() {
    let s = one("SELECT * FROM docs WHERE a1b2 = 1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a1b2".into(),
            op: CmpOp::Eq,
            value: Lit::Int(1)
        })
    );
}

#[test]
fn underscore_then_digit_field() {
    let s = one("SELECT * FROM docs WHERE _1 = 1");
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "_1".into(),
            op: CmpOp::Eq,
            value: Lit::Int(1)
        })
    );
}

// ── keyword casing ────────────────────────────────────────────────────────────

#[test]
fn and_lowercase_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE a = 1 and b = 2",
        "SELECT * FROM docs WHERE a = 1 AND b = 2",
    );
}

#[test]
fn and_mixed_case_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE a = 1 AnD b = 2",
        "SELECT * FROM docs WHERE a = 1 AND b = 2",
    );
}

#[test]
fn or_lowercase_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE a = 1 or b = 2",
        "SELECT * FROM docs WHERE a = 1 OR b = 2",
    );
}

#[test]
fn or_mixed_case_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE a = 1 Or b = 2",
        "SELECT * FROM docs WHERE a = 1 OR b = 2",
    );
}

#[test]
fn not_lowercase_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE not a = 1",
        "SELECT * FROM docs WHERE NOT a = 1",
    );
}

#[test]
fn not_mixed_case_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE NoT a = 1",
        "SELECT * FROM docs WHERE NOT a = 1",
    );
}

#[test]
fn in_lowercase_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE a in (1)",
        "SELECT * FROM docs WHERE a IN (1)",
    );
}

#[test]
fn in_mixed_case_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE a In (1)",
        "SELECT * FROM docs WHERE a IN (1)",
    );
}

#[test]
fn like_lowercase_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE a like 'x'",
        "SELECT * FROM docs WHERE a LIKE 'x'",
    );
}

#[test]
fn like_mixed_case_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE a LiKe 'x'",
        "SELECT * FROM docs WHERE a LIKE 'x'",
    );
}

#[test]
fn ilike_lowercase_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE a ilike 'x'",
        "SELECT * FROM docs WHERE a ILIKE 'x'",
    );
}

#[test]
fn ilike_mixed_case_same_as_uppercase() {
    same(
        "SELECT * FROM docs WHERE a IlIkE 'x'",
        "SELECT * FROM docs WHERE a ILIKE 'x'",
    );
}
