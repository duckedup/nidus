//! `WITH (...)` option-bag grammar (nidus-yq9p.6, `BLUEPRINT-with-opts.md`): every key's
//! flag/scalar/tuple shapes, the bag's own punctuation rules, and the points where
//! `parse_with_options` is deliberately permissive and defers validation to `compile.rs`.

use super::super::error::{SEC_AGG, SEC_ANNOTATE, SEC_BATCH, SEC_RANK, SEC_SYNTAX};
use super::super::parse::{AnyLit, Lit, OptArg, OptValue, WithOption};
use super::{compile_err_at, compiled_one, err_at, many, one};

/// The `with` field of the sole statement `sql` parses to.
fn withs(sql: &str) -> Vec<WithOption> {
    one(sql).with
}

// ── flags: annotations / plan / exact ───────────────────────────────────────────

#[test]
fn annotations_bare_flag() {
    let w = withs("SELECT * FROM t WITH (annotations)");
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].name, "annotations");
    assert_eq!(w[0].value, OptValue::Flag);
}

#[test]
fn plan_bare_flag() {
    let w = withs("SELECT * FROM t WITH (plan)");
    assert_eq!(w[0].name, "plan");
    assert_eq!(w[0].value, OptValue::Flag);
}

#[test]
fn exact_bare_flag() {
    let w = withs("SELECT * FROM t WITH (exact)");
    assert_eq!(w[0].name, "exact");
    assert_eq!(w[0].value, OptValue::Flag);
}

#[test]
fn annotations_key_eq_true_is_a_scalar_not_a_flag() {
    let w = withs("SELECT * FROM t WITH (annotations = true)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Bool(true))));
}

#[test]
fn plan_key_eq_false_is_a_scalar_not_a_flag() {
    let w = withs("SELECT * FROM t WITH (plan = false)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Bool(false))));
}

#[test]
fn exact_key_eq_int_is_a_scalar_not_a_flag() {
    let w = withs("SELECT * FROM t WITH (exact = 1)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Int(1))));
}

// ── scalar numeric keys: min_score / diversity / candidates / rrf_k ────────────

#[test]
fn min_score_float() {
    let w = withs("SELECT * FROM t WITH (min_score = 0.5)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Float(0.5))));
}

#[test]
fn min_score_int() {
    let w = withs("SELECT * FROM t WITH (min_score = 1)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Int(1))));
}

#[test]
fn min_score_negative_float() {
    let w = withs("SELECT * FROM t WITH (min_score = -0.25)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Float(-0.25))));
}

#[test]
fn diversity_float() {
    let w = withs("SELECT * FROM t WITH (diversity = 0.3)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Float(0.3))));
}

#[test]
fn diversity_negative_int() {
    let w = withs("SELECT * FROM t WITH (diversity = -1)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Int(-1))));
}

#[test]
fn candidates_int() {
    let w = withs("SELECT * FROM t WITH (candidates = 100)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Int(100))));
}

#[test]
fn candidates_negative_int_parses_though_compile_would_reject_it() {
    let w = withs("SELECT * FROM t WITH (candidates = -5)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Int(-5))));
}

#[test]
fn rrf_k_float() {
    let w = withs("SELECT * FROM t WITH (rrf_k = 60.0)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Float(60.0))));
}

#[test]
fn rrf_k_scientific_notation() {
    let w = withs("SELECT * FROM t WITH (rrf_k = 1e2)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Float(100.0))));
}

// ── tuple key: limit_per = (field, n) ────────────────────────────────────────────

#[test]
fn limit_per_ident_field() {
    let w = withs("SELECT * FROM t WITH (limit_per = (file, 2))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Bare(AnyLit::Ident("file".into())),
            OptArg::Bare(AnyLit::Val(Lit::Int(2))),
        ])
    );
}

#[test]
fn limit_per_quoted_ident_field() {
    let w = withs(r#"SELECT * FROM t WITH (limit_per = ("file name", 2))"#);
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Bare(AnyLit::Ident("file name".into())),
            OptArg::Bare(AnyLit::Val(Lit::Int(2))),
        ])
    );
}

#[test]
fn limit_per_string_literal_field() {
    let w = withs("SELECT * FROM t WITH (limit_per = ('file', 2))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Bare(AnyLit::Val(Lit::Str("file".into()))),
            OptArg::Bare(AnyLit::Val(Lit::Int(2))),
        ])
    );
}

#[test]
fn limit_per_empty_tuple_parses_as_an_empty_list() {
    let w = withs("SELECT * FROM t WITH (limit_per = ())");
    assert_eq!(w[0].value, OptValue::List(vec![]));
}

#[test]
fn limit_per_zero_max() {
    let w = withs("SELECT * FROM t WITH (limit_per = (file, 0))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Bare(AnyLit::Ident("file".into())),
            OptArg::Bare(AnyLit::Val(Lit::Int(0))),
        ])
    );
}

// ── tuple key: context = (radius n [, parent f, index f, text f]) ──────────────

#[test]
fn context_radius_only() {
    let w = withs("SELECT * FROM t WITH (context = (radius 2))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![OptArg::Keyed(
            "radius".into(),
            AnyLit::Val(Lit::Int(2))
        )])
    );
}

#[test]
fn context_radius_and_parent() {
    let w = withs("SELECT * FROM t WITH (context = (radius 2, parent pid))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Keyed("radius".into(), AnyLit::Val(Lit::Int(2))),
            OptArg::Keyed("parent".into(), AnyLit::Ident("pid".into())),
        ])
    );
}

#[test]
fn context_all_four_args() {
    let w = withs(
        r#"SELECT * FROM t WITH (context = (radius 2, parent "pid", index "idx", text "txt"))"#,
    );
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Keyed("radius".into(), AnyLit::Val(Lit::Int(2))),
            OptArg::Keyed("parent".into(), AnyLit::Ident("pid".into())),
            OptArg::Keyed("index".into(), AnyLit::Ident("idx".into())),
            OptArg::Keyed("text".into(), AnyLit::Ident("txt".into())),
        ])
    );
}

#[test]
fn context_args_in_reverse_order_are_kept_as_written() {
    let w = withs(r#"SELECT * FROM t WITH (context = (text "txt", radius 2))"#);
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Keyed("text".into(), AnyLit::Ident("txt".into())),
            OptArg::Keyed("radius".into(), AnyLit::Val(Lit::Int(2))),
        ])
    );
}

#[test]
fn context_radius_and_index_only() {
    let w = withs("SELECT * FROM t WITH (context = (radius 2, index myidx))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Keyed("radius".into(), AnyLit::Val(Lit::Int(2))),
            OptArg::Keyed("index".into(), AnyLit::Ident("myidx".into())),
        ])
    );
}

#[test]
fn context_radius_and_text_only() {
    let w = withs("SELECT * FROM t WITH (context = (radius 2, text mytxt))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Keyed("radius".into(), AnyLit::Val(Lit::Int(2))),
            OptArg::Keyed("text".into(), AnyLit::Ident("mytxt".into())),
        ])
    );
}

#[test]
fn context_radius_as_a_string_literal_parses_though_compile_would_reject_it() {
    let w = withs("SELECT * FROM t WITH (context = (radius '2'))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![OptArg::Keyed(
            "radius".into(),
            AnyLit::Val(Lit::Str("2".into()))
        )])
    );
}

// ── tuple key: rerank = ([overscan n] [, text f]) ───────────────────────────────

#[test]
fn rerank_overscan_only() {
    let w = withs("SELECT * FROM t WITH (rerank = (overscan 10))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![OptArg::Keyed(
            "overscan".into(),
            AnyLit::Val(Lit::Int(10))
        )])
    );
}

#[test]
fn rerank_text_only() {
    let w = withs(r#"SELECT * FROM t WITH (rerank = (text "body"))"#);
    assert_eq!(
        w[0].value,
        OptValue::List(vec![OptArg::Keyed(
            "text".into(),
            AnyLit::Ident("body".into())
        )])
    );
}

#[test]
fn rerank_overscan_and_text() {
    let w = withs(r#"SELECT * FROM t WITH (rerank = (overscan 10, text "body"))"#);
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Keyed("overscan".into(), AnyLit::Val(Lit::Int(10))),
            OptArg::Keyed("text".into(), AnyLit::Ident("body".into())),
        ])
    );
}

#[test]
fn rerank_args_in_reverse_order() {
    let w = withs(r#"SELECT * FROM t WITH (rerank = (text "body", overscan 10))"#);
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Keyed("text".into(), AnyLit::Ident("body".into())),
            OptArg::Keyed("overscan".into(), AnyLit::Val(Lit::Int(10))),
        ])
    );
}

#[test]
fn rerank_zero_overscan() {
    let w = withs("SELECT * FROM t WITH (rerank = (overscan 0))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![OptArg::Keyed(
            "overscan".into(),
            AnyLit::Val(Lit::Int(0))
        )])
    );
}

#[test]
fn rerank_accepts_an_unknown_keyed_arg_name_at_parse_time() {
    let w = withs("SELECT * FROM t WITH (rerank = (frobnicate 5))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![OptArg::Keyed(
            "frobnicate".into(),
            AnyLit::Val(Lit::Int(5))
        )])
    );
}

// ── tuple key: weights = (vector_weight, text_weight) ───────────────────────────

#[test]
fn weights_two_floats() {
    let w = withs("SELECT * FROM t WITH (weights = (1.0, 2.0))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Bare(AnyLit::Val(Lit::Float(1.0))),
            OptArg::Bare(AnyLit::Val(Lit::Float(2.0))),
        ])
    );
}

#[test]
fn weights_two_ints() {
    let w = withs("SELECT * FROM t WITH (weights = (1, 2))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Bare(AnyLit::Val(Lit::Int(1))),
            OptArg::Bare(AnyLit::Val(Lit::Int(2))),
        ])
    );
}

#[test]
fn weights_keyed_form_parses_though_only_bare_args_are_documented() {
    let w = withs("SELECT * FROM t WITH (weights = (vector 1.0, text 2.0))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Keyed("vector".into(), AnyLit::Val(Lit::Float(1.0))),
            OptArg::Keyed("text".into(), AnyLit::Val(Lit::Float(2.0))),
        ])
    );
}

#[test]
fn weights_negative_number() {
    let w = withs("SELECT * FROM t WITH (weights = (-1.0, 2.0))");
    assert_eq!(
        w[0].value,
        OptValue::List(vec![
            OptArg::Bare(AnyLit::Val(Lit::Float(-1.0))),
            OptArg::Bare(AnyLit::Val(Lit::Float(2.0))),
        ])
    );
}

// ── permissiveness at other keys: any literal type, any key name ───────────────

#[test]
fn min_score_accepts_a_bool_at_parse_time() {
    let w = withs("SELECT * FROM t WITH (min_score = true)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Bool(true))));
}

#[test]
fn candidates_accepts_null_at_parse_time() {
    let w = withs("SELECT * FROM t WITH (candidates = null)");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Null)));
}

#[test]
fn annotations_accepts_a_string_value_at_parse_time() {
    let w = withs("SELECT * FROM t WITH (annotations = 'yes')");
    assert_eq!(
        w[0].value,
        OptValue::Scalar(AnyLit::Val(Lit::Str("yes".into())))
    );
}

#[test]
fn diversity_accepts_a_bare_ident_value_at_parse_time() {
    let w = withs("SELECT * FROM t WITH (diversity = somefield)");
    assert_eq!(
        w[0].value,
        OptValue::Scalar(AnyLit::Ident("somefield".into()))
    );
}

// ── the bag itself: how many keys, duplicates, unknown keys ────────────────────

#[test]
fn one_key() {
    let w = withs("SELECT * FROM t WITH (annotations)");
    assert_eq!(w.len(), 1);
}

#[test]
fn two_keys() {
    let w = withs("SELECT * FROM t WITH (annotations, plan)");
    assert_eq!(w.len(), 2);
    assert_eq!(w[0].name, "annotations");
    assert_eq!(w[1].name, "plan");
}

#[test]
fn three_keys() {
    let w = withs("SELECT * FROM t WITH (annotations, plan, exact)");
    assert_eq!(w.len(), 3);
    for opt in &w {
        assert_eq!(opt.value, OptValue::Flag);
    }
}

#[test]
fn every_key_at_once() {
    let w = withs(
        "SELECT * FROM t WITH (annotations, plan, exact, min_score = 0.5, \
         diversity = 0.2, limit_per = (file, 2), \
         context = (radius 2, parent pid, index idx, text txt), \
         rerank = (overscan 10, text body), candidates = 100, rrf_k = 60, \
         weights = (1.0, 2.0))",
    );
    assert_eq!(w.len(), 11);
    let names: Vec<&str> = w.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "annotations",
            "plan",
            "exact",
            "min_score",
            "diversity",
            "limit_per",
            "context",
            "rerank",
            "candidates",
            "rrf_k",
            "weights",
        ]
    );
    assert_eq!(w[0].value, OptValue::Flag);
    assert_eq!(w[3].value, OptValue::Scalar(AnyLit::Val(Lit::Float(0.5))));
    assert_eq!(w[8].value, OptValue::Scalar(AnyLit::Val(Lit::Int(100))));
    let OptValue::List(context_args) = &w[6].value else {
        panic!("expected a list")
    };
    assert_eq!(context_args.len(), 4);
}

#[test]
fn duplicate_key_same_value_parses_as_two_entries() {
    let w = withs("SELECT * FROM t WITH (exact, exact)");
    assert_eq!(w.len(), 2);
    assert_eq!(w[0].name, "exact");
    assert_eq!(w[1].name, "exact");
}

#[test]
fn duplicate_key_different_values_parses_as_two_entries() {
    let w = withs("SELECT * FROM t WITH (min_score = 1, min_score = 2)");
    assert_eq!(w.len(), 2);
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Int(1))));
    assert_eq!(w[1].value, OptValue::Scalar(AnyLit::Val(Lit::Int(2))));
}

#[test]
fn unknown_key_parses_fine_at_the_grammar_level() {
    let w = withs("SELECT * FROM t WITH (bogus)");
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].name, "bogus");
    assert_eq!(w[0].value, OptValue::Flag);
}

#[test]
fn unknown_key_amid_known_keys() {
    let w = withs("SELECT * FROM t WITH (annotations, bogus, plan)");
    let names: Vec<&str> = w.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, vec!["annotations", "bogus", "plan"]);
}

#[test]
fn whitespace_and_newlines_between_keys_are_insignificant() {
    let w = withs("SELECT * FROM t WITH (\n    annotations,\n    plan\n)");
    assert_eq!(w.len(), 2);
    assert_eq!(w[0].name, "annotations");
    assert_eq!(w[1].name, "plan");
}

#[test]
fn each_statement_in_a_script_keeps_its_own_with_bag() {
    let stmts = many("SELECT * FROM a WITH (annotations); SELECT * FROM b WITH (plan)");
    assert_eq!(stmts[0].with[0].name, "annotations");
    assert_eq!(stmts[1].with[0].name, "plan");
}

#[test]
fn limit_offset_and_with_all_present_together() {
    let s = one("SELECT * FROM t LIMIT 5 OFFSET 2 WITH (annotations)");
    assert_eq!(s.limit, Some(5));
    assert_eq!(s.offset, Some(2));
    assert_eq!(s.with[0].name, "annotations");
}

#[test]
fn group_by_with_a_with_clause_parses_though_compile_would_reject_the_combination() {
    let s = one("SELECT * FROM t GROUP BY g WITH (annotations)");
    assert_eq!(s.group_by, Some("g".to_string()));
    assert_eq!(s.with[0].name, "annotations");
}

// ── the key name field preserves the caller's case (compile.rs lowercases it) ──

#[test]
fn key_name_case_is_preserved_uppercase() {
    let w = withs("SELECT * FROM t WITH (ANNOTATIONS)");
    assert_eq!(w[0].name, "ANNOTATIONS");
}

#[test]
fn key_name_case_is_preserved_mixed_case() {
    let w = withs("SELECT * FROM t WITH (Min_Score = 1)");
    assert_eq!(w[0].name, "Min_Score");
    assert_eq!(w[0].value, OptValue::Scalar(AnyLit::Val(Lit::Int(1))));
}

// ── the bag's own punctuation: parens, commas, position ─────────────────────────

#[test]
fn empty_parens_is_an_error() {
    err_at(
        "SELECT * FROM t WITH ()",
        22,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn leading_comma_is_an_error() {
    err_at(
        "SELECT * FROM t WITH (, annotations)",
        22,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn with_and_no_parens_at_all_is_an_error() {
    err_at("SELECT * FROM t WITH", 20, "expected '('", SEC_ANNOTATE);
}

#[test]
fn unterminated_with_open_paren_is_an_error() {
    err_at(
        "SELECT * FROM t WITH (",
        22,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn trailing_comma_after_the_last_key_is_an_error() {
    err_at(
        "SELECT * FROM t WITH (annotations,)",
        34,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn doubled_comma_between_keys_is_an_error() {
    err_at(
        "SELECT * FROM t WITH (annotations,, plan)",
        34,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn missing_comma_between_keys_is_an_error() {
    err_at(
        "SELECT * FROM t WITH (annotations plan)",
        34,
        "expected ')'",
        SEC_ANNOTATE,
    );
}

#[test]
fn with_before_order_by_is_the_wrong_position() {
    err_at(
        "SELECT * FROM t WITH (annotations) ORDER BY knn([1.0])",
        35,
        "expected ';'",
        SEC_BATCH,
    );
}

#[test]
fn with_before_limit_is_the_wrong_position() {
    err_at(
        "SELECT * FROM t WITH (annotations) LIMIT 5",
        35,
        "expected ';'",
        SEC_BATCH,
    );
}

#[test]
fn trailing_comma_inside_a_tuple_value_is_an_error() {
    err_at(
        "SELECT * FROM t WITH (limit_per = (file, 2,))",
        43,
        "expected a value",
        SEC_SYNTAX,
    );
}

#[test]
fn doubled_comma_inside_a_tuple_value_is_an_error() {
    err_at(
        "SELECT * FROM t WITH (limit_per = (file,, 2))",
        40,
        "expected a value",
        SEC_SYNTAX,
    );
}

#[test]
fn unterminated_outer_paren_after_a_tuple_value_is_an_error() {
    err_at(
        "SELECT * FROM t WITH (limit_per = (file, 2)",
        43,
        "expected ')'",
        SEC_ANNOTATE,
    );
}

#[test]
fn unterminated_inner_and_outer_parens_is_an_error() {
    err_at(
        "SELECT * FROM t WITH (context = (radius 2)",
        42,
        "expected ')'",
        SEC_ANNOTATE,
    );
}

// ── a WITH key name colliding with a reserved keyword ───────────────────────────

#[test]
fn key_named_select_collides_with_the_keyword() {
    err_at(
        "SELECT * FROM t WITH (select)",
        22,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn key_named_from_collides_with_the_keyword() {
    err_at(
        "SELECT * FROM t WITH (from)",
        22,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn key_named_limit_collides_with_the_keyword() {
    err_at(
        "SELECT * FROM t WITH (limit)",
        22,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn key_named_and_collides_with_the_keyword() {
    err_at(
        "SELECT * FROM t WITH (and)",
        22,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn key_named_true_collides_with_the_keyword() {
    err_at(
        "SELECT * FROM t WITH (true)",
        22,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn key_named_null_collides_with_the_keyword() {
    err_at(
        "SELECT * FROM t WITH (null)",
        22,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn key_named_in_collides_with_the_keyword() {
    err_at(
        "SELECT * FROM t WITH (in)",
        22,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

#[test]
fn key_named_desc_collides_with_the_keyword() {
    err_at(
        "SELECT * FROM t WITH (desc)",
        22,
        "WITH option name",
        SEC_ANNOTATE,
    );
}

// ── WITH-key semantics: rejected at the COMPILE layer, not by `parse` ──────────
// `parse_with_*` never looks at a key's name, so value type and tuple arity are semantics.
// These assert where the rejection actually happens, so each one can fail for a real reason.

#[test]
fn with_key_names_are_case_insensitive_end_to_end() {
    let a = compiled_one("SELECT * FROM t ORDER BY knn([1.0]) WITH (min_score = 0.5)");
    let b = compiled_one("SELECT * FROM t ORDER BY knn([1.0]) WITH (MIN_SCORE = 0.5)");
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

#[test]
fn min_score_wants_a_number_not_a_string() {
    compile_err_at(
        "SELECT * FROM t ORDER BY knn([1.0]) WITH (min_score = 'x')",
        42,
        "a number",
        SEC_RANK,
    );
}

#[test]
fn candidates_is_rejected_without_a_fuse_ranking() {
    compile_err_at(
        "SELECT * FROM t ORDER BY knn([1.0]) WITH (candidates = 'x')",
        42,
        "FUSE",
        SEC_RANK,
    );
}

#[test]
fn weights_wants_two_numbers_not_strings() {
    compile_err_at(
        "SELECT * FROM t ORDER BY knn([1.0]) FUSE match(b, 'q') WITH (weights = ('a', 'b'))",
        61,
        "two numbers",
        SEC_RANK,
    );
}

#[test]
fn weights_with_one_argument_is_rejected() {
    compile_err_at(
        "SELECT * FROM t ORDER BY knn([1.0]) FUSE match(b, 'q') WITH (weights = (1.0))",
        61,
        "two numbers",
        SEC_RANK,
    );
}

#[test]
fn weights_with_four_arguments_is_rejected() {
    compile_err_at(
        "SELECT * FROM t ORDER BY knn([1.0]) FUSE match(b, 'q') WITH (weights = (1.0, 2.0, 3.0, 4.0))",
        61,
        "two numbers",
        SEC_RANK,
    );
}

#[test]
fn limit_per_swapped_argument_types_is_rejected() {
    compile_err_at(
        "SELECT * FROM t ORDER BY knn([1.0]) WITH (limit_per = (5, x))",
        42,
        "(field, n)",
        SEC_AGG,
    );
}

#[test]
fn limit_per_with_one_argument_is_rejected() {
    compile_err_at(
        "SELECT * FROM t ORDER BY knn([1.0]) WITH (limit_per = (file))",
        42,
        "(field, n)",
        SEC_AGG,
    );
}

#[test]
fn limit_per_with_three_arguments_is_rejected() {
    compile_err_at(
        "SELECT * FROM t ORDER BY knn([1.0]) WITH (limit_per = (file, 2, 3))",
        42,
        "(field, n)",
        SEC_AGG,
    );
}

#[test]
fn limit_per_with_a_negative_count_is_rejected() {
    compile_err_at(
        "SELECT * FROM t ORDER BY knn([1.0]) WITH (limit_per = (file, -1))",
        42,
        "(field, n)",
        SEC_AGG,
    );
}
