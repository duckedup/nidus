//! Every error class the grammar can raise, each pinned to its byte offset and its §7.x
//! section (nidus-yq9p.6, root `BLUEPRINT-nidus-yq9p-2-3-6.md`'s error model). Every test
//! goes through [`err_at`](super::err_at), so every one asserts offset, message, and section.

use super::super::error::{
    SEC_AGG, SEC_ANNOTATE, SEC_ARRAY, SEC_BATCH, SEC_BOOL, SEC_GLOB, SEC_RANK, SEC_REGEX,
    SEC_SYNTAX, SEC_TEXT,
};
use super::{err_at, one};

// ── unexpected token, in every clause position ──────────────────────────────────────

#[test]
fn select_missing_field_name() {
    err_at("SELECT FROM docs", 7, "field name", SEC_SYNTAX);
}

#[test]
fn from_missing_collection_name() {
    err_at("SELECT * FROM WHERE a = 1", 14, "field name", SEC_SYNTAX);
}

#[test]
fn where_missing_predicate_unexpected_token() {
    err_at(
        "SELECT * FROM docs WHERE ORDER BY x",
        25,
        "field name",
        SEC_SYNTAX,
    );
}

#[test]
fn order_by_missing_target_unexpected_token() {
    err_at(
        "SELECT * FROM docs ORDER BY LIMIT 5",
        28,
        "field name",
        SEC_SYNTAX,
    );
}

#[test]
fn group_by_missing_field_unexpected_token() {
    err_at(
        "SELECT * FROM docs GROUP BY LIMIT 5",
        28,
        "field name",
        SEC_AGG,
    );
}

#[test]
fn limit_wrong_type_unexpected_token() {
    err_at("SELECT * FROM docs LIMIT 'x'", 25, "a number", SEC_SYNTAX);
}

#[test]
fn with_unexpected_token_inside() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0]) WITH (42)",
        45,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn function_call_unexpected_token() {
    err_at(
        "SELECT * FROM docs WHERE contains(tags, )",
        40,
        "a value",
        SEC_SYNTAX,
    );
}

// ── unexpected end of input: truncate right after every keyword, offsets march forward ──

#[test]
fn truncate_after_select() {
    err_at("SELECT", 6, "field name", SEC_SYNTAX);
}

#[test]
fn truncate_after_from() {
    err_at("SELECT a FROM", 13, "field name", SEC_SYNTAX);
}

#[test]
fn truncate_after_where() {
    err_at("SELECT a FROM docs WHERE", 24, "field name", SEC_SYNTAX);
}

#[test]
fn truncate_after_group() {
    err_at("SELECT a FROM docs WHERE b = 1 GROUP", 36, "'BY'", SEC_AGG);
}

#[test]
fn truncate_after_group_by() {
    err_at(
        "SELECT a FROM docs WHERE b = 1 GROUP BY",
        39,
        "field name",
        SEC_AGG,
    );
}

#[test]
fn truncate_after_order() {
    err_at(
        "SELECT a FROM docs WHERE b = 1 GROUP BY c ORDER",
        47,
        "'BY'",
        SEC_SYNTAX,
    );
}

#[test]
fn truncate_after_order_by() {
    err_at(
        "SELECT a FROM docs WHERE b = 1 GROUP BY c ORDER BY",
        50,
        "field name",
        SEC_SYNTAX,
    );
}

#[test]
fn truncate_after_limit() {
    err_at(
        "SELECT a FROM docs WHERE b = 1 GROUP BY c ORDER BY d DESC LIMIT",
        63,
        "a number",
        SEC_SYNTAX,
    );
}

#[test]
fn truncate_after_offset() {
    err_at(
        "SELECT a FROM docs WHERE b = 1 GROUP BY c ORDER BY d DESC LIMIT 5 OFFSET",
        72,
        "a number",
        SEC_SYNTAX,
    );
}

#[test]
fn truncate_after_with() {
    err_at(
        "SELECT a FROM docs WHERE b = 1 GROUP BY c ORDER BY d DESC LIMIT 5 OFFSET 2 WITH",
        79,
        "'('",
        SEC_ANNOTATE,
    );
}

// ── unterminated: string, paren, bracket, quoted identifier ─────────────────────────

#[test]
fn unterminated_string() {
    err_at(
        "SELECT * FROM docs WHERE a = 'oops",
        29,
        "unterminated string literal",
        SEC_SYNTAX,
    );
}

#[test]
fn unterminated_paren() {
    err_at("SELECT * FROM docs WHERE (a = 1", 31, "')'", SEC_BOOL);
}

#[test]
fn unterminated_bracket() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0, 2.0",
        41,
        "']'",
        SEC_RANK,
    );
}

#[test]
fn unterminated_quoted_ident() {
    err_at(
        r#"SELECT "oops FROM docs"#,
        7,
        "unterminated quoted identifier",
        SEC_SYNTAX,
    );
}

// ── bad literal: wrong type, overflow, malformed ─────────────────────────────────────

#[test]
fn float_where_int_belongs() {
    // The float token is consumed before the int-vs-float check runs, so the error lands
    // on whatever follows it (here, end of input) rather than on the float itself.
    err_at(
        "SELECT * FROM docs LIMIT 1.5",
        28,
        "found a float",
        SEC_SYNTAX,
    );
}

#[test]
fn string_where_number_belongs() {
    err_at("SELECT * FROM docs OFFSET 'x'", 26, "a number", SEC_SYNTAX);
}

#[test]
fn integer_overflow_i64() {
    // Same as `float_where_int_belongs`: the oversized token is consumed first, so the
    // error position is past it, at end of input.
    err_at(
        "SELECT * FROM docs WHERE a = 99999999999999999999",
        49,
        "invalid integer literal",
        SEC_SYNTAX,
    );
}

#[test]
fn malformed_number_two_dots() {
    // "1.2" lexes as a float; the second '.' is then a bare unexpected character.
    err_at(
        "SELECT * FROM docs WHERE a = 1.2.3",
        32,
        "unexpected character",
        SEC_SYNTAX,
    );
}

#[test]
fn malformed_number_bare_e() {
    err_at(
        "SELECT * FROM docs WHERE a = 1e",
        29,
        "malformed number",
        SEC_SYNTAX,
    );
}

#[test]
fn malformed_number_digits_running_into_letters() {
    err_at(
        "SELECT * FROM docs WHERE a = 12abc",
        29,
        "malformed number",
        SEC_SYNTAX,
    );
}

#[test]
fn a_well_formed_exponent_still_lexes() {
    let s = one("SELECT * FROM docs WHERE a = 1e3");
    assert!(format!("{s:?}").contains("1000.0"), "{s:?}");
}

#[test]
fn lone_minus() {
    err_at("SELECT * FROM docs WHERE a = -", 30, "a number", SEC_SYNTAX);
}

// ── bad arity: every function predicate, too few and too many arguments ─────────────

#[test]
fn contains_too_few_args() {
    err_at(
        "SELECT * FROM docs WHERE contains(tags)",
        38,
        "','",
        SEC_ARRAY,
    );
}

#[test]
fn contains_too_many_args() {
    err_at(
        "SELECT * FROM docs WHERE contains(tags, 'x', 'y')",
        43,
        "')'",
        SEC_ARRAY,
    );
}

#[test]
fn not_contains_too_few_args() {
    err_at(
        "SELECT * FROM docs WHERE not_contains(tags)",
        42,
        "','",
        SEC_ARRAY,
    );
}

#[test]
fn not_contains_too_many_args() {
    err_at(
        "SELECT * FROM docs WHERE not_contains(tags, 'x', 'y')",
        47,
        "')'",
        SEC_ARRAY,
    );
}

#[test]
fn contains_any_too_few_args() {
    err_at(
        "SELECT * FROM docs WHERE contains_any(tags)",
        42,
        "','",
        SEC_ARRAY,
    );
}

#[test]
fn contains_any_trailing_comma_is_a_missing_value() {
    // Variadic, so "too many" has no natural ceiling; a dangling comma is the analogue.
    err_at(
        "SELECT * FROM docs WHERE contains_any(tags, 'a',)",
        48,
        "a value",
        SEC_SYNTAX,
    );
}

#[test]
fn fuzzy_too_few_args() {
    err_at(
        "SELECT * FROM docs WHERE fuzzy(id, 'x')",
        38,
        "','",
        SEC_TEXT,
    );
}

#[test]
fn fuzzy_too_many_args() {
    err_at(
        "SELECT * FROM docs WHERE fuzzy(id, 'x', 2, 3)",
        41,
        "')'",
        SEC_TEXT,
    );
}

#[test]
fn match_all_too_few_args() {
    err_at(
        "SELECT * FROM docs WHERE match_all(body)",
        39,
        "','",
        SEC_TEXT,
    );
}

#[test]
fn match_all_too_many_args() {
    err_at(
        "SELECT * FROM docs WHERE match_all(body, 'x', 'y')",
        44,
        "')'",
        SEC_TEXT,
    );
}

#[test]
fn match_any_too_few_args() {
    err_at(
        "SELECT * FROM docs WHERE match_any(body)",
        39,
        "','",
        SEC_TEXT,
    );
}

#[test]
fn match_any_too_many_args() {
    err_at(
        "SELECT * FROM docs WHERE match_any(body, 'x', 'y')",
        44,
        "')'",
        SEC_TEXT,
    );
}

#[test]
fn phrase_too_few_args() {
    err_at("SELECT * FROM docs WHERE phrase(body)", 36, "','", SEC_TEXT);
}

#[test]
fn phrase_too_many_args() {
    err_at(
        "SELECT * FROM docs WHERE phrase(body, 'x', 'y')",
        41,
        "')'",
        SEC_TEXT,
    );
}

// ── unknown name ─────────────────────────────────────────────────────────────────────

#[test]
fn unknown_function_predicate_is_read_as_field_then_bad_comparison() {
    // The grammar has no notion of "unknown function": a name the parser doesn't
    // recognize as one of the fixed predicate functions is just a field name, and the
    // '(' that follows is an unexpected token where a comparison operator belongs.
    err_at(
        "SELECT * FROM docs WHERE unknownfn(field, 1)",
        34,
        "comparison operator",
        SEC_SYNTAX,
    );
}

// NOTE: an unknown `WITH` key (`WITH (nonsense)`) is not a parse-time error — any identifier
// is a legal option name. Rejecting one is `compile.rs`'s `unknown_with_key`, unreachable
// through the raw `parse()` this file's `err_at` exercises. Out of this file's scope.

// ── section mapping: at least two dedicated errors per §7 section ──────────────────

#[test]
fn glob_like_wrong_type() {
    err_at(
        "SELECT * FROM docs WHERE a LIKE 123",
        32,
        "a string literal",
        SEC_GLOB,
    );
}

#[test]
fn glob_ilike_eof() {
    err_at(
        "SELECT * FROM docs WHERE a ILIKE",
        32,
        "a string literal",
        SEC_GLOB,
    );
}

#[test]
fn array_contains_missing_comma() {
    err_at(
        "SELECT * FROM docs WHERE contains(tags 'x')",
        39,
        "','",
        SEC_ARRAY,
    );
}

#[test]
fn array_not_contains_bad_field() {
    err_at(
        "SELECT * FROM docs WHERE not_contains(999, 'x')",
        38,
        "field name",
        SEC_ARRAY,
    );
}

#[test]
fn bool_unmatched_open_paren_dedicated() {
    err_at(
        "SELECT * FROM docs WHERE (a = 1 OR b = 2",
        40,
        "')'",
        SEC_BOOL,
    );
}

#[test]
fn bool_depth_cap_dedicated() {
    // One past the cap (128): the smallest input that must still error, not a big margin.
    let mut sql = "SELECT * FROM docs WHERE ".to_string();
    sql.push_str(&"(".repeat(129));
    sql.push_str("a = 1");
    sql.push_str(&")".repeat(129));
    err_at(&sql, 153, "nesting", SEC_BOOL);
}

#[test]
fn text_fuzzy_missing_comma() {
    err_at(
        "SELECT * FROM docs WHERE fuzzy(id 'x', 2)",
        34,
        "','",
        SEC_TEXT,
    );
}

#[test]
fn text_phrase_bad_field() {
    err_at(
        "SELECT * FROM docs WHERE phrase(999, 'x')",
        32,
        "field name",
        SEC_TEXT,
    );
}

#[test]
fn regex_eof_dedicated() {
    err_at(
        "SELECT * FROM docs WHERE a ~",
        28,
        "a string literal",
        SEC_REGEX,
    );
}

#[test]
fn regex_wrong_type_dedicated() {
    err_at(
        "SELECT * FROM docs WHERE a ~ 42",
        29,
        "a string literal",
        SEC_REGEX,
    );
}

#[test]
fn rank_knn_missing_bracket() {
    err_at("SELECT * FROM docs ORDER BY knn(1.0)", 32, "'['", SEC_RANK);
}

#[test]
fn rank_fuse_missing_match_keyword() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0]) FUSE",
        43,
        "'match'",
        SEC_RANK,
    );
}

#[test]
fn agg_sum_missing_paren() {
    err_at("SELECT * FROM docs GROUP BY x, SUM", 34, "'('", SEC_AGG);
}

#[test]
fn agg_sum_empty_field() {
    err_at(
        "SELECT * FROM docs GROUP BY x, SUM()",
        35,
        "field name",
        SEC_AGG,
    );
}

#[test]
fn batch_two_statements_after_where() {
    err_at(
        "SELECT * FROM docs WHERE a = 1 SELECT * FROM other",
        31,
        "expected ';'",
        SEC_BATCH,
    );
}

#[test]
fn batch_two_statements_after_limit() {
    err_at(
        "SELECT * FROM docs LIMIT 5 SELECT * FROM other",
        27,
        "expected ';'",
        SEC_BATCH,
    );
}

#[test]
fn syntax_missing_select() {
    err_at("FROM docs", 0, "'SELECT'", SEC_SYNTAX);
}

#[test]
fn syntax_missing_comparison_operator_eof() {
    err_at(
        "SELECT * FROM docs WHERE a",
        26,
        "comparison operator",
        SEC_SYNTAX,
    );
}

// ── the offset is not always 0: errors deep inside a long query, well past byte 40 ──

#[test]
fn deep_glob_missing_pattern() {
    err_at(
        "SELECT * FROM docs WHERE a = 1 AND b = 2 AND c = 3 AND d LIKE",
        61,
        "a string literal",
        SEC_GLOB,
    );
}

#[test]
fn deep_regex_missing_pattern() {
    err_at(
        "SELECT * FROM docs WHERE a = 1 AND b = 2 AND c ~",
        48,
        "a string literal",
        SEC_REGEX,
    );
}

#[test]
fn deep_group_by_missing_field() {
    err_at(
        "SELECT * FROM docs WHERE a = 1 AND b = 2 GROUP BY",
        49,
        "field name",
        SEC_AGG,
    );
}

#[test]
fn deep_sum_missing_paren() {
    err_at(
        "SELECT * FROM docs WHERE a = 1 AND b = 2 AND c = 3 GROUP BY x, SUM",
        66,
        "'('",
        SEC_AGG,
    );
}

#[test]
fn deep_unterminated_bracket() {
    err_at(
        "SELECT * FROM docs WHERE a = 1 AND b = 2 AND c = 3 ORDER BY knn([1.0, 2.0",
        73,
        "']'",
        SEC_RANK,
    );
}

#[test]
fn deep_contains_arity() {
    err_at(
        "SELECT * FROM docs WHERE a=1 AND b=2 AND c=3 AND contains(tags)",
        62,
        "','",
        SEC_ARRAY,
    );
}

// ── comparison / IN / NOT edge cases ─────────────────────────────────────────────────

#[test]
fn not_without_in_or_like() {
    err_at(
        "SELECT * FROM docs WHERE a NOT 5",
        31,
        "IN' or 'LIKE'",
        SEC_SYNTAX,
    );
}

#[test]
fn in_missing_open_paren() {
    err_at("SELECT * FROM docs WHERE a IN 1", 30, "'('", SEC_SYNTAX);
}

#[test]
fn in_missing_close_paren() {
    err_at("SELECT * FROM docs WHERE a IN (1, 2", 35, "')'", SEC_SYNTAX);
}

#[test]
fn in_empty_list_is_a_missing_value() {
    err_at(
        "SELECT * FROM docs WHERE a IN ()",
        31,
        "a value",
        SEC_SYNTAX,
    );
}

#[test]
fn not_in_missing_open_paren() {
    err_at("SELECT * FROM docs WHERE a NOT IN 1", 34, "'('", SEC_SYNTAX);
}

#[test]
fn comparison_missing_value_after_ne() {
    err_at("SELECT * FROM docs WHERE a !=", 29, "a value", SEC_SYNTAX);
}

#[test]
fn bang_alone_is_a_lex_error() {
    err_at(
        "SELECT * FROM docs WHERE a !",
        27,
        "'=' after '!'",
        SEC_SYNTAX,
    );
}

#[test]
fn unexpected_character_is_a_lex_error() {
    err_at(
        "SELECT * FROM docs WHERE a @ 1",
        27,
        "unexpected character",
        SEC_SYNTAX,
    );
}

// ── WITH clause: shapes beyond a single bad option name ──────────────────────────────

#[test]
fn with_missing_comma_between_options() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0]) WITH (annotations plan)",
        57,
        "')'",
        SEC_ANNOTATE,
    );
}

#[test]
fn with_missing_close_paren() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0]) WITH (annotations",
        56,
        "')'",
        SEC_ANNOTATE,
    );
}

#[test]
fn with_list_missing_close_paren() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0]) WITH (limit_per = (file, 2",
        65,
        "')'",
        SEC_ANNOTATE,
    );
}

#[test]
fn with_keyed_arg_extra_token() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0]) WITH (context = (radius 2 3))",
        65,
        "')'",
        SEC_ANNOTATE,
    );
}

#[test]
fn with_value_after_eq_missing() {
    // The literal parser's own error is always the generic SEC_SYNTAX, even inside WITH.
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0]) WITH (min_score =",
        56,
        "a value",
        SEC_SYNTAX,
    );
}

#[test]
fn with_empty_parens() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0]) WITH ()",
        45,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

// ── SELECT / scope / field-list edge cases ───────────────────────────────────────────

#[test]
fn select_except_missing_paren() {
    err_at("SELECT * EXCEPT body FROM docs", 16, "'('", SEC_SYNTAX);
}

#[test]
fn select_except_empty_list() {
    err_at("SELECT * EXCEPT () FROM docs", 17, "field name", SEC_SYNTAX);
}

#[test]
fn select_trailing_comma_field_list() {
    err_at("SELECT a, FROM docs", 10, "field name", SEC_SYNTAX);
}

#[test]
fn scope_trailing_comma() {
    err_at(
        "SELECT * FROM docs, WHERE a=1",
        20,
        "field name",
        SEC_SYNTAX,
    );
}

#[test]
fn quoted_field_unterminated_inside_select() {
    err_at(
        r#"SELECT a, "oops FROM docs WHERE x=1"#,
        10,
        "unterminated quoted identifier",
        SEC_SYNTAX,
    );
}

#[test]
fn group_by_sum_wrong_close() {
    err_at("SELECT * FROM docs GROUP BY x, SUM(y", 36, "')'", SEC_AGG);
}

// ── ranking / decay edge cases ───────────────────────────────────────────────────────

#[test]
fn knn_vector_bad_element() {
    // Numeric-literal errors inside a vector are the generic SEC_SYNTAX, not SEC_RANK.
    err_at(
        "SELECT * FROM docs ORDER BY knn(['x'])",
        33,
        "a number",
        SEC_SYNTAX,
    );
}

#[test]
fn knn_vector_trailing_comma() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0,])",
        37,
        "a number",
        SEC_SYNTAX,
    );
}

#[test]
fn decay_missing_comma() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0]) - decay(created_at 100, 200)",
        58,
        "','",
        SEC_RANK,
    );
}

#[test]
fn decay_missing_close_paren() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0]) - decay(created_at, 100, 200",
        67,
        "')'",
        SEC_RANK,
    );
}

#[test]
fn decay_and_fuse_conflict() {
    err_at(
        "SELECT * FROM docs ORDER BY knn([1.0]) - decay(a, 1, 2) FUSE match(body, 'x')",
        56,
        "decay and FUSE",
        SEC_RANK,
    );
}

#[test]
fn match_clause_missing_comma() {
    err_at(
        "SELECT * FROM docs ORDER BY match(title 'x')",
        40,
        "','",
        SEC_RANK,
    );
}

// ── table-driven: adding a new class later is one line ──────────────────────────────

#[test]
fn table_driven_error_offsets_and_sections() {
    let cases: &[(&str, usize, &str, &str)] = &[
        ("SELECT FROM docs", 7, "field name", SEC_SYNTAX),
        ("SELECT * FROM WHERE a = 1", 14, "field name", SEC_SYNTAX),
        (
            "SELECT * FROM docs WHERE a LIKE 123",
            32,
            "a string literal",
            SEC_GLOB,
        ),
        (
            "SELECT * FROM docs WHERE a ILIKE",
            32,
            "a string literal",
            SEC_GLOB,
        ),
        (
            "SELECT * FROM docs WHERE contains(tags 'x')",
            39,
            "','",
            SEC_ARRAY,
        ),
        (
            "SELECT * FROM docs WHERE not_contains(999, 'x')",
            38,
            "field name",
            SEC_ARRAY,
        ),
        (
            "SELECT * FROM docs WHERE (a = 1 OR b = 2",
            40,
            "')'",
            SEC_BOOL,
        ),
        (
            "SELECT * FROM docs WHERE fuzzy(id 'x', 2)",
            34,
            "','",
            SEC_TEXT,
        ),
        (
            "SELECT * FROM docs WHERE phrase(999, 'x')",
            32,
            "field name",
            SEC_TEXT,
        ),
        (
            "SELECT * FROM docs WHERE a ~",
            28,
            "a string literal",
            SEC_REGEX,
        ),
        (
            "SELECT * FROM docs WHERE a ~ 42",
            29,
            "a string literal",
            SEC_REGEX,
        ),
        ("SELECT * FROM docs ORDER BY knn(1.0)", 32, "'['", SEC_RANK),
        (
            "SELECT * FROM docs ORDER BY knn([1.0]) FUSE",
            43,
            "'match'",
            SEC_RANK,
        ),
        ("SELECT * FROM docs GROUP BY x, SUM", 34, "'('", SEC_AGG),
        (
            "SELECT * FROM docs GROUP BY x, SUM()",
            35,
            "field name",
            SEC_AGG,
        ),
        (
            "SELECT * FROM docs WHERE a = 1 SELECT * FROM other",
            31,
            "expected ';'",
            SEC_BATCH,
        ),
        (
            "SELECT * FROM docs LIMIT 5 SELECT * FROM other",
            27,
            "expected ';'",
            SEC_BATCH,
        ),
        ("FROM docs", 0, "'SELECT'", SEC_SYNTAX),
        (
            "SELECT * FROM docs WHERE a",
            26,
            "comparison operator",
            SEC_SYNTAX,
        ),
        (
            "SELECT * FROM docs WHERE contains(tags)",
            38,
            "','",
            SEC_ARRAY,
        ),
        (
            "SELECT * FROM docs WHERE contains(tags, 'x', 'y')",
            43,
            "')'",
            SEC_ARRAY,
        ),
        (
            "SELECT * FROM docs WHERE fuzzy(id, 'x')",
            38,
            "','",
            SEC_TEXT,
        ),
        (
            "SELECT * FROM docs WHERE match_all(body, 'x', 'y')",
            44,
            "')'",
            SEC_TEXT,
        ),
        (
            "SELECT * FROM docs WHERE unknownfn(field, 1)",
            34,
            "comparison operator",
            SEC_SYNTAX,
        ),
        (
            "SELECT * FROM docs WHERE a NOT 5",
            31,
            "IN' or 'LIKE'",
            SEC_SYNTAX,
        ),
        (
            "SELECT * FROM docs WHERE a IN ()",
            31,
            "a value",
            SEC_SYNTAX,
        ),
        (
            "SELECT * FROM docs ORDER BY knn(['x'])",
            33,
            "a number",
            SEC_SYNTAX,
        ),
        (
            "SELECT * FROM docs ORDER BY knn([1.0]) - decay(created_at 100, 200)",
            58,
            "','",
            SEC_RANK,
        ),
        (
            "SELECT * FROM docs ORDER BY match(title 'x')",
            40,
            "','",
            SEC_RANK,
        ),
        (
            "SELECT * FROM docs ORDER BY knn([1.0]) WITH ()",
            45,
            "WITH option name",
            SEC_ANNOTATE,
        ),
    ];
    assert!(cases.len() >= 25, "table must carry 25+ cases");
    for &(sql, at, needle, section) in cases {
        err_at(sql, at, needle, section);
    }
}
