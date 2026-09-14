//! Depth, size, and hostile-input abuse tests for the SQL grammar (nidus-yq9p.6, P5):
//! the parser must fail cleanly at its documented limits and never panic, overflow, or
//! silently truncate on attacker-shaped or merely huge input.

use super::super::MAX_NEST_DEPTH;
use super::super::error::{SEC_BOOL, SEC_REGEX, SEC_SYNTAX};
use super::super::parse::{CmpOp, Lit, PredNode, Ranking, Selection};
use super::{err_at, many, one, same};

/// Every depth/size fixture below builds on a `WHERE` clause off this prefix.
const PREFIX: &str = "SELECT * FROM docs WHERE ";

/// Build `depth` levels of `PredNode::Not` wrapping `leaf` — the expected shape for a
/// `NOT (NOT (... leaf ...))` fixture at that depth.
fn nested_not(depth: usize, leaf: PredNode) -> PredNode {
    let mut node = leaf;
    for _ in 0..depth {
        node = PredNode::Not(Box::new(node));
    }
    node
}

/// Assert `sql` parses the same way twice, and that leading/trailing whitespace
/// (spaces, newlines, tabs) around it never changes the result.
fn assert_idempotent(sql: &str) {
    same(sql, sql);
    same(sql, &format!("  \n\t{sql}\t\n  "));
}

// ── the depth cap (§7.3), at the boundary: 127, 128, and 129 nested parens ──────────

#[test]
fn depth_127_nested_parens_still_parses() {
    let n = MAX_NEST_DEPTH - 1;
    let sql = format!("{PREFIX}{}a = 1{}", "(".repeat(n), ")".repeat(n));
    let s = one(&sql);
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
fn depth_128_at_the_cap_still_parses() {
    let n = MAX_NEST_DEPTH;
    let sql = format!("{PREFIX}{}a = 1{}", "(".repeat(n), ")".repeat(n));
    let s = one(&sql);
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
fn depth_129_exceeds_the_cap_with_a_clean_error() {
    let n = MAX_NEST_DEPTH + 1;
    let sql = format!("{PREFIX}{}a = 1{}", "(".repeat(n), ")".repeat(n));
    let at = PREFIX.len() + MAX_NEST_DEPTH;
    err_at(&sql, at, "nesting", SEC_BOOL);
}

// Nested NOT is only reachable through a paren at each level (`parse_not` does not loop),
// so this walks the same counter through a NOT-shaped leaf instead of a bare comparison.

#[test]
fn depth_127_nested_not_still_parses() {
    let n = MAX_NEST_DEPTH - 1;
    let sql = format!("{PREFIX}{}a = 1{}", "NOT (".repeat(n), ")".repeat(n));
    let s = one(&sql);
    let leaf = PredNode::Cmp {
        field: "a".into(),
        op: CmpOp::Eq,
        value: Lit::Int(1),
    };
    assert_eq!(s.filter, Some(nested_not(n, leaf)));
}

#[test]
fn depth_128_nested_not_at_the_cap_still_parses() {
    let n = MAX_NEST_DEPTH;
    let sql = format!("{PREFIX}{}a = 1{}", "NOT (".repeat(n), ")".repeat(n));
    let s = one(&sql);
    let leaf = PredNode::Cmp {
        field: "a".into(),
        op: CmpOp::Eq,
        value: Lit::Int(1),
    };
    assert_eq!(s.filter, Some(nested_not(n, leaf)));
}

#[test]
fn depth_129_nested_not_exceeds_the_cap_with_a_clean_error() {
    let unit = "NOT (";
    let n = MAX_NEST_DEPTH + 1;
    let sql = format!("{PREFIX}{}a = 1{}", unit.repeat(n), ")".repeat(n));
    let at = PREFIX.len() + MAX_NEST_DEPTH * unit.len() + (unit.len() - 1);
    err_at(&sql, at, "nesting", SEC_BOOL);
}

// Function predicates don't recurse themselves; this proves the same paren counter
// governs a fn_predicate leaf exactly as it does a bare comparison.

#[test]
fn depth_127_wrapping_a_fn_predicate_still_parses() {
    let n = MAX_NEST_DEPTH - 1;
    let sql = format!(
        "{PREFIX}{}contains(tags, 'x'){}",
        "(".repeat(n),
        ")".repeat(n)
    );
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Contains {
            field: "tags".into(),
            value: Lit::Str("x".into())
        })
    );
}

#[test]
fn depth_128_wrapping_a_fn_predicate_at_the_cap_still_parses() {
    let n = MAX_NEST_DEPTH;
    let sql = format!(
        "{PREFIX}{}contains(tags, 'x'){}",
        "(".repeat(n),
        ")".repeat(n)
    );
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Contains {
            field: "tags".into(),
            value: Lit::Str("x".into())
        })
    );
}

#[test]
fn depth_129_wrapping_a_fn_predicate_exceeds_the_cap_with_a_clean_error() {
    let n = MAX_NEST_DEPTH + 1;
    let sql = format!(
        "{PREFIX}{}contains(tags, 'x'){}",
        "(".repeat(n),
        ")".repeat(n)
    );
    let at = PREFIX.len() + MAX_NEST_DEPTH;
    err_at(&sql, at, "nesting", SEC_BOOL);
}

// The counter must reset between sibling branches, not accumulate across an OR split.

#[test]
fn depth_is_independent_across_sibling_or_branches() {
    let n = MAX_NEST_DEPTH;
    let sql = format!(
        "{PREFIX}{}a = 1{} OR {}b = 2{}",
        "(".repeat(n),
        ")".repeat(n),
        "(".repeat(n),
        ")".repeat(n)
    );
    let s = one(&sql);
    let PredNode::Any(items) = s.filter.unwrap() else {
        panic!("expected Any")
    };
    assert_eq!(
        items,
        vec![
            PredNode::Cmp {
                field: "a".into(),
                op: CmpOp::Eq,
                value: Lit::Int(1)
            },
            PredNode::Cmp {
                field: "b".into(),
                op: CmpOp::Eq,
                value: Lit::Int(2)
            },
        ]
    );
}

#[test]
fn depth_exceeded_only_in_the_second_branch_reports_that_branchs_offset() {
    let n = MAX_NEST_DEPTH + 1;
    let branch1 = "a = 1 OR ";
    let sql = format!("{PREFIX}{branch1}{}b = 2{}", "(".repeat(n), ")".repeat(n));
    let at = PREFIX.len() + branch1.len() + MAX_NEST_DEPTH;
    err_at(&sql, at, "nesting", SEC_BOOL);
}

// ── size: chains, lists, vectors, and raw byte volume ───────────────────────────────

#[test]
fn or_chain_of_500_terms_parses_cleanly() {
    let terms: Vec<String> = (0..500).map(|i| format!("a = {i}")).collect();
    let sql = format!("{PREFIX}{}", terms.join(" OR "));
    let s = one(&sql);
    let PredNode::Any(items) = s.filter.unwrap() else {
        panic!("expected Any")
    };
    assert_eq!(items.len(), 500);
    assert_eq!(
        items[0],
        PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Int(0)
        }
    );
    assert_eq!(
        items[499],
        PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Int(499)
        }
    );
}

#[test]
fn and_chain_of_500_terms_parses_cleanly() {
    let terms: Vec<String> = (0..500).map(|i| format!("a = {i}")).collect();
    let sql = format!("{PREFIX}{}", terms.join(" AND "));
    let s = one(&sql);
    let PredNode::All(items) = s.filter.unwrap() else {
        panic!("expected All")
    };
    assert_eq!(items.len(), 500);
    assert_eq!(
        items[0],
        PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Int(0)
        }
    );
    assert_eq!(
        items[499],
        PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Int(499)
        }
    );
}

#[test]
fn in_list_with_1000_elements_parses_cleanly() {
    let vals: Vec<String> = (0..1000).map(|i| i.to_string()).collect();
    let sql = format!("{PREFIX}a IN ({})", vals.join(", "));
    let s = one(&sql);
    let PredNode::In {
        field,
        negate,
        values,
    } = s.filter.unwrap()
    else {
        panic!("expected In")
    };
    assert_eq!(field, "a");
    assert!(!negate);
    assert_eq!(values.len(), 1000);
    assert_eq!(values[0], Lit::Int(0));
    assert_eq!(values[999], Lit::Int(999));
}

#[test]
fn knn_vector_with_4096_components_parses_cleanly() {
    let nums: Vec<String> = (0..4096).map(|i| format!("{i}.0")).collect();
    let sql = format!("SELECT * FROM docs ORDER BY knn([{}])", nums.join(", "));
    let s = one(&sql);
    let Some(Ranking::Knn { vector, .. }) = s.order else {
        panic!("expected Knn")
    };
    assert_eq!(vector.len(), 4096);
    assert_eq!(vector[0], 0.0);
    assert_eq!(vector[4095], 4095.0);
}

#[test]
fn hundred_kb_input_by_repetition_parses_cleanly() {
    let mut sql = PREFIX.to_string();
    let mut n = 0usize;
    while sql.len() < 100_000 {
        if n > 0 {
            sql.push_str(" OR ");
        }
        sql.push_str("a = ");
        sql.push_str(&n.to_string());
        n += 1;
    }
    assert!(sql.len() >= 100_000);
    let s = one(&sql);
    let PredNode::Any(items) = s.filter.unwrap() else {
        panic!("expected Any")
    };
    assert_eq!(items.len(), n);
}

#[test]
fn single_identifier_10000_characters_long_parses_cleanly() {
    let field = "a".repeat(10_000);
    let sql = format!("{PREFIX}{field} = 1");
    let s = one(&sql);
    let PredNode::Cmp { field: f, .. } = s.filter.unwrap() else {
        panic!("expected Cmp")
    };
    assert_eq!(f.len(), 10_000);
    assert_eq!(f, field);
}

#[test]
fn string_literal_50000_characters_long_round_trips() {
    let text = "x".repeat(50_000);
    let sql = format!("{PREFIX}a = '{text}'");
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Str(text)
        })
    );
}

// ── hostile and adversarial strings ──────────────────────────────────────────────────

#[test]
fn string_literal_containing_semicolon_and_parens_round_trips() {
    let value = "a;(b)c";
    let sql = format!("{PREFIX}x = '{value}'");
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "x".into(),
            op: CmpOp::Eq,
            value: Lit::Str(value.into())
        })
    );
}

#[test]
fn string_literal_containing_an_escaped_quote_round_trips() {
    let sql = format!("{PREFIX}x = 'it''s here'");
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "x".into(),
            op: CmpOp::Eq,
            value: Lit::Str("it's here".into())
        })
    );
}

#[test]
fn string_literal_containing_newline_and_tab_round_trips() {
    let value = "line1\nline2\ttabbed";
    let sql = format!("{PREFIX}x = '{value}'");
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "x".into(),
            op: CmpOp::Eq,
            value: Lit::Str(value.into())
        })
    );
}

#[test]
fn string_literal_containing_a_nul_ish_escape_round_trips() {
    let value = "a\u{0}b";
    let sql = format!("{PREFIX}x = '{value}'");
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "x".into(),
            op: CmpOp::Eq,
            value: Lit::Str(value.into())
        })
    );
}

#[test]
fn sql_injection_shaped_string_stays_one_statement() {
    let value = "'; DROP TABLE x; --";
    let escaped = value.replace('\'', "''");
    let sql = format!("{PREFIX}x = '{escaped}'");
    let stmts = many(&sql);
    assert_eq!(stmts.len(), 1);
    assert_eq!(
        stmts[0].filter,
        Some(PredNode::Cmp {
            field: "x".into(),
            op: CmpOp::Eq,
            value: Lit::Str(value.into())
        })
    );
}

#[test]
fn quoted_identifier_containing_knn_paren_is_a_plain_field_name() {
    // A quoted ident lexes as `Kind::QuotedIdent`, which `ident_lc` never matches, so
    // this can never be mistaken for the `knn(` ranking call.
    let sql = "SELECT * FROM docs ORDER BY \"knn(\"";
    let s = one(sql);
    assert_eq!(
        s.order,
        Some(Ranking::Field {
            field: "knn(".into(),
            desc: false
        })
    );
}

#[test]
fn string_value_containing_knn_paren_is_a_plain_string_literal() {
    let sql = format!("{PREFIX}x = 'knn('");
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "x".into(),
            op: CmpOp::Eq,
            value: Lit::Str("knn(".into())
        })
    );
}

#[test]
fn unicode_characters_in_a_string_literal_round_trip() {
    let value = "héllo wörld 日本語 🎉";
    let sql = format!("{PREFIX}x = '{value}'");
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "x".into(),
            op: CmpOp::Eq,
            value: Lit::Str(value.into())
        })
    );
}

#[test]
fn unicode_characters_in_a_quoted_identifier_round_trip() {
    let sql = "SELECT \"日本語 field\" FROM docs";
    let s = one(sql);
    assert_eq!(s.select, Selection::Fields(vec!["日本語 field".into()]));
}

#[test]
fn unicode_strings_inside_an_in_list_round_trip() {
    let sql = "SELECT * FROM docs WHERE lang IN ('café', '日本語')";
    let s = one(sql);
    let PredNode::In { values, .. } = s.filter.unwrap() else {
        panic!("expected In")
    };
    assert_eq!(
        values,
        vec![Lit::Str("café".into()), Lit::Str("日本語".into())]
    );
}

#[test]
fn offset_after_a_multibyte_char_in_a_string_is_bytes_not_chars() {
    // 'é' is 2 bytes / 1 char; if offsets were char-counted this would land short.
    let sql = "SELECT * FROM docs WHERE x = 'café' AND y ~";
    assert_ne!(
        sql.chars().count(),
        sql.len(),
        "fixture needs a real multi-byte char"
    );
    err_at(sql, sql.len(), "string literal", SEC_REGEX);
}

#[test]
fn offset_after_a_multibyte_char_in_a_quoted_ident_is_bytes_not_chars() {
    let sql = "SELECT * FROM docs WHERE \"日本語\" ~";
    assert_ne!(
        sql.chars().count(),
        sql.len(),
        "fixture needs a real multi-byte char"
    );
    err_at(sql, sql.len(), "string literal", SEC_REGEX);
}

#[test]
fn crlf_line_endings_between_every_token_still_parses() {
    let sql = "SELECT\r\n*\r\nFROM\r\ndocs\r\nWHERE\r\na\r\n=\r\n1";
    same(sql, "SELECT * FROM docs WHERE a = 1");
}

#[test]
fn mixed_tabs_and_spaces_between_every_token_still_parses() {
    let sql = "SELECT\t*\t FROM\t docs \tWHERE\ta\t=\t1";
    same(sql, "SELECT * FROM docs WHERE a = 1");
}

#[test]
fn only_whitespace_input_is_an_empty_query_error() {
    err_at("   \t\n  ", 0, "empty query", SEC_SYNTAX);
}

#[test]
fn only_a_semicolon_input_is_a_syntax_error() {
    err_at(";", 0, "SELECT", SEC_SYNTAX);
}

#[test]
fn like_pattern_containing_hostile_shapes_round_trips() {
    let pattern = "*/../../etc/passwd*";
    let sql = format!("{PREFIX}x LIKE '{pattern}'");
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Like {
            field: "x".into(),
            negate: false,
            pattern: pattern.into()
        })
    );
}

#[test]
fn in_list_containing_a_sql_injection_shaped_string_round_trips() {
    let sql = r#"SELECT * FROM docs WHERE x IN ('a''; DROP TABLE t; --', 'clean')"#;
    let s = one(sql);
    let PredNode::In { values, .. } = s.filter.unwrap() else {
        panic!("expected In")
    };
    assert_eq!(
        values,
        vec![
            Lit::Str("a'; DROP TABLE t; --".into()),
            Lit::Str("clean".into())
        ]
    );
}

#[test]
fn quoted_string_spelling_null_is_a_string_not_the_null_literal() {
    let sql = format!("{PREFIX}x = 'NULL'");
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "x".into(),
            op: CmpOp::Eq,
            value: Lit::Str("NULL".into())
        })
    );
}

#[test]
fn bare_null_keyword_is_the_null_literal() {
    let sql = format!("{PREFIX}x = NULL");
    let s = one(&sql);
    assert_eq!(
        s.filter,
        Some(PredNode::Cmp {
            field: "x".into(),
            op: CmpOp::Eq,
            value: Lit::Null
        })
    );
}

// ── idempotence: 20+ varied valid queries, twice-parsed and whitespace-padded ───────

#[test]
fn idempotent_select_star_from_docs() {
    assert_idempotent("SELECT * FROM docs");
}

#[test]
fn idempotent_select_fields_from_docs() {
    assert_idempotent("SELECT a, b FROM docs");
}

#[test]
fn idempotent_select_star_except() {
    assert_idempotent("SELECT * EXCEPT (a, b) FROM docs");
}

#[test]
fn idempotent_from_multiple_collections() {
    assert_idempotent("SELECT * FROM docs, notes");
}

#[test]
fn idempotent_where_simple_eq() {
    assert_idempotent("SELECT * FROM docs WHERE a = 1");
}

#[test]
fn idempotent_where_ne_and_lt() {
    assert_idempotent("SELECT * FROM docs WHERE a != 1 AND b < 2");
}

#[test]
fn idempotent_where_in_list() {
    assert_idempotent("SELECT * FROM docs WHERE a IN (1, 2, 3)");
}

#[test]
fn idempotent_where_not_in() {
    assert_idempotent("SELECT * FROM docs WHERE a NOT IN (1, 2)");
}

#[test]
fn idempotent_where_like() {
    assert_idempotent("SELECT * FROM docs WHERE a LIKE 'x*'");
}

#[test]
fn idempotent_where_not_like() {
    assert_idempotent("SELECT * FROM docs WHERE a NOT LIKE 'y*'");
}

#[test]
fn idempotent_where_ilike() {
    assert_idempotent("SELECT * FROM docs WHERE a ILIKE 'Z*'");
}

#[test]
fn idempotent_where_regex() {
    assert_idempotent("SELECT * FROM docs WHERE a ~ 'v[0-9]+'");
}

#[test]
fn idempotent_where_contains_family() {
    assert_idempotent("SELECT * FROM docs WHERE contains(tags, 'x') AND not_contains(tags, 'y')");
}

#[test]
fn idempotent_where_fuzzy_and_tokens() {
    assert_idempotent("SELECT * FROM docs WHERE fuzzy(id, 'nidus', 2) AND match_all(body, 'a b')");
}

#[test]
fn idempotent_where_grouped_precedence() {
    assert_idempotent("SELECT * FROM docs WHERE (a = 1 OR b = 2) AND NOT c = 3");
}

#[test]
fn idempotent_group_by_with_sums() {
    assert_idempotent("SELECT * FROM docs GROUP BY project, SUM(bytes)");
}

#[test]
fn idempotent_order_by_knn_vector() {
    assert_idempotent("SELECT * FROM docs ORDER BY knn([1.0, 0.0, -0.5])");
}

#[test]
fn idempotent_order_by_knn_with_decay() {
    assert_idempotent(
        "SELECT * FROM docs ORDER BY knn([1.0]) - decay(created_at, 1700000000000, 604800000, 0.5, 2.0)",
    );
}

#[test]
fn idempotent_order_by_match() {
    assert_idempotent("SELECT * FROM docs ORDER BY match(title, 'rust', body, 'async' PREFIX)");
}

#[test]
fn idempotent_order_by_knn_fuse_match() {
    assert_idempotent("SELECT * FROM docs ORDER BY knn([1.0]) FUSE match(body, 'rust')");
}

#[test]
fn idempotent_order_by_field_desc_limit_offset() {
    assert_idempotent("SELECT * FROM docs ORDER BY score DESC LIMIT 10 OFFSET 5");
}

#[test]
fn idempotent_with_flags() {
    assert_idempotent("SELECT * FROM docs ORDER BY knn([1.0]) WITH (annotations, plan, exact)");
}

#[test]
fn idempotent_with_scalars() {
    assert_idempotent(
        "SELECT * FROM docs ORDER BY knn([1.0]) WITH (min_score = 0.5, diversity = 0.2)",
    );
}

#[test]
fn idempotent_with_limit_per() {
    assert_idempotent("SELECT * FROM docs ORDER BY knn([1.0]) WITH (limit_per = (file, 2))");
}

#[test]
fn idempotent_with_context() {
    assert_idempotent(
        r#"SELECT * FROM docs ORDER BY knn([1.0]) WITH (context = (radius 2, parent "pid", index "idx", text "txt"))"#,
    );
}
