//! Statement shape and every clause (nidus-yq9p.6, `BLUEPRINT-clauses.md`): `SELECT`,
//! `FROM`, `ORDER BY`'s ranking heads, `GROUP BY`/`LIMIT`/`OFFSET`, clause-order and
//! clause-presence combinations, and `;`-separated statement batching (§7.9).

use super::super::error::{SEC_AGG, SEC_BATCH, SEC_RANK, SEC_SYNTAX};
use super::super::parse::*;
use super::{compiled_one, err_at, many, one, same};

// ── SELECT ───────────────────────────────────────────────────────────────────

#[test]
fn select_star_is_all() {
    let s = one("SELECT * FROM t");
    assert_eq!(s.select, Selection::All);
}

#[test]
fn select_single_field() {
    let s = one("SELECT a FROM t");
    assert_eq!(s.select, Selection::Fields(vec!["a".into()]));
}

#[test]
fn select_many_fields() {
    let s = one("SELECT a, b, c FROM t");
    assert_eq!(
        s.select,
        Selection::Fields(vec!["a".into(), "b".into(), "c".into()])
    );
}

#[test]
fn select_field_charset_allows_underscores_and_digits() {
    let s = one("SELECT field_1, a2 FROM t");
    assert_eq!(
        s.select,
        Selection::Fields(vec!["field_1".into(), "a2".into()])
    );
}

#[test]
fn select_dotted_field() {
    let s = one("SELECT a.b FROM t");
    assert_eq!(s.select, Selection::Fields(vec!["a.b".into()]));
}

#[test]
fn select_quoted_field() {
    let s = one(r#"SELECT "a field" FROM t"#);
    assert_eq!(s.select, Selection::Fields(vec!["a field".into()]));
}

#[test]
fn select_quoted_field_with_dot_and_space() {
    let s = one(r#"SELECT "a.b c" FROM t"#);
    assert_eq!(s.select, Selection::Fields(vec!["a.b c".into()]));
}

#[test]
fn select_star_except_one_name() {
    let s = one("SELECT * EXCEPT (body) FROM t");
    assert_eq!(s.select, Selection::AllExcept(vec!["body".into()]));
}

#[test]
fn select_star_except_many_names() {
    let s = one("SELECT * EXCEPT (body, embedding) FROM t");
    assert_eq!(
        s.select,
        Selection::AllExcept(vec!["body".into(), "embedding".into()])
    );
}

#[test]
fn select_star_except_dotted_name() {
    let s = one("SELECT * EXCEPT (a.b) FROM t");
    assert_eq!(s.select, Selection::AllExcept(vec!["a.b".into()]));
}

#[test]
fn select_field_list_trailing_comma_is_an_error() {
    let sql = "SELECT a, FROM t";
    err_at(
        sql,
        sql.find("FROM").unwrap(),
        "expected a field name",
        SEC_SYNTAX,
    );
}

#[test]
fn select_star_except_empty_parens_is_an_error() {
    let sql = "SELECT * EXCEPT () FROM t";
    err_at(
        sql,
        sql.find(')').unwrap(),
        "expected a field name",
        SEC_SYNTAX,
    );
}

#[test]
fn select_star_except_trailing_comma_is_an_error() {
    let sql = "SELECT * EXCEPT (a,) FROM t";
    err_at(
        sql,
        sql.find(')').unwrap(),
        "expected a field name",
        SEC_SYNTAX,
    );
}

#[test]
fn select_star_except_missing_closing_paren_is_an_error() {
    let sql = "SELECT * EXCEPT (a FROM t";
    err_at(sql, sql.find("FROM").unwrap(), "')'", SEC_SYNTAX);
}

#[test]
fn select_except_without_a_leading_star_is_an_error() {
    // `EXCEPT` only means anything right after `*`; otherwise it is just an unexpected
    // keyword where a field name was expected.
    let sql = "SELECT EXCEPT (a) FROM t";
    err_at(
        sql,
        sql.find("EXCEPT").unwrap(),
        "expected a field name",
        SEC_SYNTAX,
    );
}

// `sum(x)`/`count(*)` are GROUP BY's aggregation syntax (§7.7: `GROUP BY f, SUM(n)`), never
// a SELECT-list function call; `Selection` has no such variant. These document that the
// SELECT list rejects them rather than silently accepting a form nothing downstream models.
#[test]
fn select_sum_projects_a_whole_scope_aggregate() {
    let s = one("SELECT sum(x) FROM docs");
    assert_eq!(
        s.select,
        Selection::Aggregates {
            sums: vec!["x".to_string()],
            count: false
        }
    );
    assert_eq!(s.group_by, None);
}

#[test]
fn select_several_sums_and_count_star() {
    let s = one("SELECT sum(a), count(*), sum(b) FROM docs");
    assert_eq!(
        s.select,
        Selection::Aggregates {
            sums: vec!["a".to_string(), "b".to_string()],
            count: true
        }
    );
}

#[test]
fn select_count_star_alone_is_an_aggregate() {
    let s = one("SELECT count(*) FROM docs");
    assert_eq!(
        s.select,
        Selection::Aggregates {
            sums: vec![],
            count: true
        }
    );
}

#[test]
fn count_is_not_reserved_and_stays_usable_as_a_field() {
    let s = one("SELECT count FROM docs");
    assert_eq!(s.select, Selection::Fields(vec!["count".to_string()]));
}

#[test]
fn sum_over_a_field_named_count_still_parses() {
    let s = one("SELECT sum(count) FROM docs");
    assert_eq!(
        s.select,
        Selection::Aggregates {
            sums: vec!["count".to_string()],
            count: false
        }
    );
}

#[test]
fn count_of_something_other_than_star_is_rejected() {
    err_at("SELECT count(x) FROM docs", 13, "'*'", SEC_AGG);
}

// ── FROM ─────────────────────────────────────────────────────────────────────

#[test]
fn from_one_collection() {
    let s = one("SELECT * FROM docs");
    assert_eq!(s.from, vec!["docs".to_string()]);
}

#[test]
fn from_many_collections() {
    let s = one("SELECT * FROM a, b, c");
    assert_eq!(
        s.from,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}

#[test]
fn from_star_is_the_empty_scope() {
    let s = one("SELECT * FROM *");
    assert_eq!(s.from, Vec::<String>::new());
}

#[test]
fn from_quoted_name() {
    let s = one(r#"SELECT * FROM "my coll.v2""#);
    assert_eq!(s.from, vec!["my coll.v2".to_string()]);
}

#[test]
fn from_dotted_name() {
    let s = one("SELECT * FROM my_ns.sub_coll");
    assert_eq!(s.from, vec!["my_ns.sub_coll".to_string()]);
}

#[test]
fn from_missing_entirely_is_an_error() {
    let sql = "SELECT *";
    err_at(sql, sql.len(), "'FROM'", SEC_SYNTAX);
}

#[test]
fn from_with_no_name_is_an_error() {
    let sql = "SELECT * FROM";
    err_at(sql, sql.len(), "expected a field name", SEC_SYNTAX);
}

#[test]
fn from_trailing_comma_is_an_error() {
    let sql = "SELECT * FROM a,";
    err_at(sql, sql.len(), "expected a field name", SEC_SYNTAX);
}

#[test]
fn from_missing_comma_leaves_a_trailing_token_and_is_a_batch_error() {
    // No comma between two names: only the first is consumed as `FROM`, and the second
    // becomes a stray token the batcher chokes on.
    let sql = "SELECT * FROM alpha beta";
    err_at(
        sql,
        sql.find("beta").unwrap(),
        "between statements",
        SEC_BATCH,
    );
}

// ── ORDER BY: every ranking head ─────────────────────────────────────────────

#[test]
fn order_by_knn_one_component() {
    let s = one("SELECT * FROM t ORDER BY knn([1.0])");
    let Some(Ranking::Knn { vector, decay, .. }) = s.order else {
        panic!("expected Knn")
    };
    assert_eq!(vector, vec![1.0]);
    assert!(decay.is_none());
}

#[test]
fn order_by_knn_two_components() {
    let s = one("SELECT * FROM t ORDER BY knn([1.0, 2.0])");
    let Some(Ranking::Knn { vector, .. }) = s.order else {
        panic!("expected Knn")
    };
    assert_eq!(vector, vec![1.0, 2.0]);
}

#[test]
fn order_by_knn_many_components() {
    let s = one("SELECT * FROM t ORDER BY knn([1.0, 2.0, 3.0, 4.0])");
    let Some(Ranking::Knn { vector, .. }) = s.order else {
        panic!("expected Knn")
    };
    assert_eq!(vector, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn order_by_knn_large_vector() {
    let s = one("SELECT * FROM t ORDER BY knn([1,2,3,4,5,6,7,8,9,10])");
    let Some(Ranking::Knn { vector, .. }) = s.order else {
        panic!("expected Knn")
    };
    assert_eq!(
        vector,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    );
}

#[test]
fn order_by_knn_mixed_int_and_float_components() {
    let s = one("SELECT * FROM t ORDER BY knn([1, 2.5, -3])");
    let Some(Ranking::Knn { vector, .. }) = s.order else {
        panic!("expected Knn")
    };
    assert_eq!(vector, vec![1.0, 2.5, -3.0]);
}

#[test]
fn order_by_knn_negative_components() {
    let s = one("SELECT * FROM t ORDER BY knn([-1.5, -2.0])");
    let Some(Ranking::Knn { vector, .. }) = s.order else {
        panic!("expected Knn")
    };
    assert_eq!(vector, vec![-1.5, -2.0]);
}

#[test]
fn order_by_knn_empty_vector() {
    let s = one("SELECT * FROM t ORDER BY knn([])");
    let Some(Ranking::Knn { vector, .. }) = s.order else {
        panic!("expected Knn")
    };
    assert_eq!(vector, Vec::<f32>::new());
}

#[test]
fn order_by_knn_vector_trailing_comma_is_an_error() {
    let sql = "SELECT * FROM t ORDER BY knn([1.0, 2.0,])";
    err_at(sql, sql.find(']').unwrap(), "expected a number", SEC_SYNTAX);
}

#[test]
fn order_by_knn_missing_closing_bracket_is_an_error() {
    let sql = "SELECT * FROM t ORDER BY knn([1.0)";
    err_at(sql, sql.find(')').unwrap(), "']'", SEC_RANK);
}

#[test]
fn order_by_knn_decay_minimal_args() {
    let s = one("SELECT * FROM t ORDER BY knn([1.0]) - decay(f, 0, 1)");
    let Some(Ranking::Knn { decay: Some(d), .. }) = s.order else {
        panic!("expected Knn with decay")
    };
    assert_eq!(d.field, "f");
    assert_eq!(d.origin, 0);
    assert_eq!(d.scale, 1);
    assert_eq!(d.decay, None);
    assert_eq!(d.lambda, None);
}

#[test]
fn order_by_knn_decay_with_decay_arg() {
    let s = one("SELECT * FROM t ORDER BY knn([1.0]) - decay(f, 0, 1, 0.5)");
    let Some(Ranking::Knn { decay: Some(d), .. }) = s.order else {
        panic!("expected Knn with decay")
    };
    assert_eq!(d.decay, Some(0.5));
    assert_eq!(d.lambda, None);
}

#[test]
fn order_by_knn_decay_with_decay_and_lambda() {
    let s = one("SELECT * FROM t ORDER BY knn([1.0]) - decay(f, 0, 1, 0.5, 2.0)");
    let Some(Ranking::Knn { decay: Some(d), .. }) = s.order else {
        panic!("expected Knn with decay")
    };
    assert_eq!(d.decay, Some(0.5));
    assert_eq!(d.lambda, Some(2.0));
}

#[test]
fn order_by_knn_decay_negative_origin() {
    let s = one("SELECT * FROM t ORDER BY knn([1.0]) - decay(f, -100, 50)");
    let Some(Ranking::Knn { decay: Some(d), .. }) = s.order else {
        panic!("expected Knn with decay")
    };
    assert_eq!(d.origin, -100);
    assert_eq!(d.scale, 50);
}

#[test]
fn order_by_knn_decay_missing_scale_is_an_error() {
    let sql = "SELECT * FROM t ORDER BY knn([1.0]) - decay(f, 0)";
    err_at(sql, sql.rfind(')').unwrap(), "','", SEC_RANK);
}

#[test]
fn order_by_match_single_clause() {
    let s = one("SELECT * FROM t ORDER BY match(body, 'rust async')");
    let Some(Ranking::Match { clauses, .. }) = s.order else {
        panic!("expected Match")
    };
    assert_eq!(clauses.len(), 1);
    assert_eq!(clauses[0].field, "body");
    assert_eq!(clauses[0].text, "rust async");
    assert!(!clauses[0].prefix);
}

#[test]
fn order_by_match_multi_clause() {
    let s = one("SELECT * FROM t ORDER BY match(title, 'rust', body, 'async')");
    let Some(Ranking::Match { clauses, .. }) = s.order else {
        panic!("expected Match")
    };
    assert_eq!(clauses.len(), 2);
    assert_eq!(clauses[0].field, "title");
    assert_eq!(clauses[1].field, "body");
}

#[test]
fn order_by_match_prefix() {
    let s = one("SELECT * FROM t ORDER BY match(title, 'rust', body, 'async' PREFIX)");
    let Some(Ranking::Match { clauses, .. }) = s.order else {
        panic!("expected Match")
    };
    assert!(!clauses[0].prefix);
    assert!(clauses[1].prefix);
}

#[test]
fn order_by_match_missing_comma_between_field_and_text_is_an_error() {
    let sql = "SELECT * FROM t ORDER BY match(f 'q')";
    err_at(sql, sql.find('\'').unwrap(), "','", SEC_RANK);
}

#[test]
fn order_by_knn_fuse_match() {
    let s = one("SELECT * FROM t ORDER BY knn([1.0, 0.0]) FUSE match(body, 'rust')");
    let Some(Ranking::Hybrid {
        vector, clauses, ..
    }) = s.order
    else {
        panic!("expected Hybrid")
    };
    assert_eq!(vector, vec![1.0, 0.0]);
    assert_eq!(clauses.len(), 1);
    assert_eq!(clauses[0].field, "body");
}

#[test]
fn order_by_decay_combined_with_fuse_is_an_error() {
    let sql = "SELECT * FROM t ORDER BY knn([1.0]) - decay(f, 0, 1) FUSE match(g, 'q')";
    err_at(
        sql,
        sql.find("FUSE").unwrap(),
        "decay and FUSE cannot combine",
        SEC_RANK,
    );
}

#[test]
fn order_by_bare_field_defaults_ascending() {
    let s = one("SELECT * FROM t ORDER BY score");
    assert_eq!(
        s.order,
        Some(Ranking::Field {
            field: "score".into(),
            desc: false
        })
    );
}

#[test]
fn order_by_field_asc_explicit() {
    let s = one("SELECT * FROM t ORDER BY score ASC");
    assert_eq!(
        s.order,
        Some(Ranking::Field {
            field: "score".into(),
            desc: false
        })
    );
}

#[test]
fn order_by_field_desc() {
    let s = one("SELECT * FROM t ORDER BY score DESC");
    assert_eq!(
        s.order,
        Some(Ranking::Field {
            field: "score".into(),
            desc: true
        })
    );
}

#[test]
fn order_by_field_with_dotted_name() {
    let s = one("SELECT * FROM t ORDER BY a.b DESC");
    assert_eq!(
        s.order,
        Some(Ranking::Field {
            field: "a.b".into(),
            desc: true
        })
    );
}

#[test]
fn order_by_unknown_ranking_head_is_a_field_reference_then_a_batch_error() {
    // Only `knn`/`match` are special-cased ranking heads; any other `name(...)` reads the
    // name as a bare field and leaves the parens as a stray, unbatched token.
    let sql = "SELECT * FROM t ORDER BY unknownfn(x)";
    err_at(sql, sql.find('(').unwrap(), "between statements", SEC_BATCH);
}

#[test]
fn order_by_knn_with_group_by_parses_at_the_grammar_level() {
    // The GROUP BY + vector-ranking contradiction is `compile.rs`'s job (SEC_AGG at
    // compile time); the grammar itself has no cross-clause validation to reject this.
    let s = one("SELECT * FROM t GROUP BY g ORDER BY knn([1.0])");
    assert_eq!(s.group_by, Some("g".to_string()));
    assert!(matches!(s.order, Some(Ranking::Knn { .. })));
}

#[test]
fn order_by_is_case_insensitive() {
    same(
        "SELECT * FROM t ORDER BY x ASC",
        "select * from t order by x asc",
    );
}

#[test]
fn order_by_knn_fuse_match_is_case_insensitive() {
    same(
        "SELECT * FROM t ORDER BY knn([1.0]) FUSE match(f, 'q')",
        "SELECT * FROM t ORDER BY KNN([1.0]) fuse MATCH(f, 'q')",
    );
}

// ── GROUP BY, LIMIT, OFFSET ──────────────────────────────────────────────────

#[test]
fn group_by_single_field() {
    let s = one("SELECT * FROM t GROUP BY g");
    assert_eq!(s.group_by, Some("g".to_string()));
    assert!(s.sums.is_empty());
}

#[test]
fn group_by_dotted_field() {
    let s = one("SELECT * FROM t GROUP BY a.b");
    assert_eq!(s.group_by, Some("a.b".to_string()));
}

#[test]
fn group_by_with_single_sum() {
    let s = one("SELECT * FROM t GROUP BY g, SUM(x)");
    assert_eq!(s.sums, vec!["x".to_string()]);
}

#[test]
fn group_by_with_sums() {
    let s = one("SELECT * FROM t GROUP BY project, SUM(bytes), SUM(count)");
    assert_eq!(s.group_by, Some("project".to_string()));
    assert_eq!(s.sums, vec!["bytes".to_string(), "count".to_string()]);
}

#[test]
fn group_by_is_case_insensitive() {
    same(
        "SELECT * FROM t GROUP BY g, SUM(x)",
        "select * from t group by g, sum(x)",
    );
}

#[test]
fn group_by_missing_by_keyword_is_an_error() {
    let sql = "SELECT * FROM t GROUP";
    err_at(sql, sql.len(), "'BY'", SEC_AGG);
}

#[test]
fn group_by_missing_field_is_an_error() {
    let sql = "SELECT * FROM t GROUP BY";
    err_at(sql, sql.len(), "expected a field name", SEC_AGG);
}

#[test]
fn group_by_sum_missing_paren_is_an_error() {
    let sql = "SELECT * FROM t GROUP BY g, SUM";
    err_at(sql, sql.len(), "'('", SEC_AGG);
}

#[test]
fn limit_zero() {
    let s = one("SELECT * FROM t LIMIT 0");
    assert_eq!(s.limit, Some(0));
}

#[test]
fn limit_large_value() {
    let s = one("SELECT * FROM t LIMIT 9999999999");
    assert_eq!(s.limit, Some(9_999_999_999));
}

#[test]
fn limit_negative_value_parses() {
    // `parse.rs` has no range check of its own; a negative LIMIT is a valid `i64` here and
    // whatever rejects it (if anything should) lives downstream, not in the grammar.
    let s = one("SELECT * FROM t LIMIT -5");
    assert_eq!(s.limit, Some(-5));
}

#[test]
fn limit_float_value_is_an_error() {
    let sql = "SELECT * FROM t LIMIT 1.5";
    err_at(sql, sql.len(), "integer", SEC_SYNTAX);
}

#[test]
fn limit_missing_value_is_an_error() {
    let sql = "SELECT * FROM t LIMIT";
    err_at(sql, sql.len(), "expected a number", SEC_SYNTAX);
}

#[test]
fn offset_without_limit() {
    let s = one("SELECT * FROM t OFFSET 5");
    assert_eq!(s.limit, None);
    assert_eq!(s.offset, Some(5));
}

#[test]
fn offset_zero() {
    let s = one("SELECT * FROM t OFFSET 0");
    assert_eq!(s.offset, Some(0));
}

#[test]
fn offset_missing_value_is_an_error() {
    let sql = "SELECT * FROM t OFFSET";
    err_at(sql, sql.len(), "expected a number", SEC_SYNTAX);
}

#[test]
fn offset_float_value_is_an_error() {
    let sql = "SELECT * FROM t OFFSET 2.5";
    err_at(sql, sql.len(), "integer", SEC_SYNTAX);
}

#[test]
fn limit_then_offset_both_present() {
    let s = one("SELECT * FROM t LIMIT 10 OFFSET 5");
    assert_eq!(s.limit, Some(10));
    assert_eq!(s.offset, Some(5));
}

#[test]
fn limit_offset_zero_both() {
    let s = one("SELECT * FROM t LIMIT 0 OFFSET 0");
    assert_eq!(s.limit, Some(0));
    assert_eq!(s.offset, Some(0));
}

#[test]
fn limit_offset_large_values_together() {
    let s = one("SELECT * FROM t LIMIT 1000000 OFFSET 500000");
    assert_eq!(s.limit, Some(1_000_000));
    assert_eq!(s.offset, Some(500_000));
}

#[test]
fn offset_before_limit_is_rejected() {
    // The grammar accepts only `LIMIT ... OFFSET ...`; the reverse order leaves `LIMIT`
    // as a stray token.
    let sql = "SELECT * FROM t OFFSET 5 LIMIT 10";
    err_at(
        sql,
        sql.find("LIMIT").unwrap(),
        "between statements",
        SEC_BATCH,
    );
}

// ── Clause combinations ──────────────────────────────────────────────────────

#[test]
fn every_subset_of_optional_clauses_parses_and_preserves_presence() {
    for mask in 0u8..64 {
        let mut sql = String::from("SELECT * FROM t");
        if mask & 0b1 != 0 {
            sql.push_str(" WHERE a = 1");
        }
        if mask & 0b10 != 0 {
            sql.push_str(" GROUP BY g");
        }
        if mask & 0b100 != 0 {
            sql.push_str(" ORDER BY score DESC");
        }
        if mask & 0b1000 != 0 {
            sql.push_str(" LIMIT 5");
        }
        if mask & 0b10000 != 0 {
            sql.push_str(" OFFSET 2");
        }
        if mask & 0b100000 != 0 {
            sql.push_str(" WITH (exact)");
        }
        let s = one(&sql);
        assert_eq!(s.filter.is_some(), mask & 0b1 != 0, "WHERE for {sql:?}");
        assert_eq!(
            s.group_by.is_some(),
            mask & 0b10 != 0,
            "GROUP BY for {sql:?}"
        );
        assert_eq!(s.order.is_some(), mask & 0b100 != 0, "ORDER BY for {sql:?}");
        assert_eq!(s.limit.is_some(), mask & 0b1000 != 0, "LIMIT for {sql:?}");
        assert_eq!(
            s.offset.is_some(),
            mask & 0b10000 != 0,
            "OFFSET for {sql:?}"
        );
        assert_eq!(!s.with.is_empty(), mask & 0b100000 != 0, "WITH for {sql:?}");
    }
}

#[test]
fn bare_select_star_from_has_no_optional_clauses() {
    let s = one("SELECT * FROM t");
    assert!(s.filter.is_none());
    assert!(s.group_by.is_none());
    assert!(s.order.is_none());
    assert!(s.limit.is_none());
    assert!(s.offset.is_none());
    assert!(s.with.is_empty());
}

#[test]
fn where_alone_sets_filter_only() {
    let s = one("SELECT * FROM t WHERE a = 1");
    assert!(s.filter.is_some());
    assert!(s.group_by.is_none());
    assert!(s.order.is_none());
}

#[test]
fn all_six_optional_clauses_together_full_shape_check() {
    let s = one(
        "SELECT * FROM t WHERE a = 1 GROUP BY g ORDER BY score DESC LIMIT 5 OFFSET 2 \
         WITH (exact)",
    );
    assert!(s.filter.is_some());
    assert_eq!(s.group_by, Some("g".to_string()));
    assert_eq!(
        s.order,
        Some(Ranking::Field {
            field: "score".into(),
            desc: true
        })
    );
    assert_eq!(s.limit, Some(5));
    assert_eq!(s.offset, Some(2));
    assert_eq!(s.with.len(), 1);
    assert_eq!(s.with[0].name, "exact");
    assert_eq!(s.with[0].value, OptValue::Flag);
}

#[test]
fn where_after_group_by_is_rejected() {
    let sql = "SELECT * FROM t GROUP BY g WHERE a = 1";
    err_at(
        sql,
        sql.find("WHERE").unwrap(),
        "between statements",
        SEC_BATCH,
    );
}

#[test]
fn order_by_before_where_is_rejected() {
    let sql = "SELECT * FROM t ORDER BY score WHERE a = 1";
    err_at(
        sql,
        sql.find("WHERE").unwrap(),
        "between statements",
        SEC_BATCH,
    );
}

#[test]
fn group_by_after_order_by_is_rejected() {
    let sql = "SELECT * FROM t ORDER BY score GROUP BY g";
    err_at(
        sql,
        sql.find("GROUP").unwrap(),
        "between statements",
        SEC_BATCH,
    );
}

#[test]
fn limit_before_order_by_is_rejected() {
    let sql = "SELECT * FROM t LIMIT 5 ORDER BY score";
    err_at(
        sql,
        sql.find("ORDER").unwrap(),
        "between statements",
        SEC_BATCH,
    );
}

#[test]
fn with_before_limit_is_rejected() {
    let sql = "SELECT * FROM t WITH (exact) LIMIT 5";
    err_at(
        sql,
        sql.find("LIMIT").unwrap(),
        "between statements",
        SEC_BATCH,
    );
}

#[test]
fn offset_after_with_is_rejected() {
    let sql = "SELECT * FROM t WITH (exact) OFFSET 5";
    err_at(
        sql,
        sql.find("OFFSET").unwrap(),
        "between statements",
        SEC_BATCH,
    );
}

#[test]
fn from_star_combined_with_where() {
    let s = one("SELECT * FROM * WHERE a = 1");
    assert_eq!(s.from, Vec::<String>::new());
    assert!(s.filter.is_some());
}

// ── Statements and batching (§7.9) ───────────────────────────────────────────

#[test]
fn one_statement() {
    let stmts = many("SELECT * FROM a");
    assert_eq!(stmts.len(), 1);
    assert_eq!(stmts[0].from, vec!["a".to_string()]);
}

#[test]
fn two_statements() {
    let stmts = many("SELECT * FROM a; SELECT * FROM b");
    assert_eq!(stmts.len(), 2);
    assert_eq!(stmts[0].from, vec!["a".to_string()]);
    assert_eq!(stmts[1].from, vec!["b".to_string()]);
}

#[test]
fn three_statements() {
    let stmts = many("SELECT * FROM a; SELECT * FROM b; SELECT * FROM c");
    assert_eq!(stmts.len(), 3);
    assert_eq!(stmts[2].from, vec!["c".to_string()]);
}

#[test]
fn four_statements() {
    let stmts = many("SELECT * FROM a; SELECT * FROM b; SELECT * FROM c; SELECT * FROM d");
    assert_eq!(stmts.len(), 4);
    assert_eq!(stmts[3].from, vec!["d".to_string()]);
}

#[test]
fn five_statements() {
    let stmts =
        many("SELECT * FROM a; SELECT * FROM b; SELECT * FROM c; SELECT * FROM d; SELECT * FROM e");
    assert_eq!(stmts.len(), 5);
    assert_eq!(stmts[4].from, vec!["e".to_string()]);
}

#[test]
fn script_with_mixed_clause_shapes_per_statement() {
    let stmts = many("SELECT a FROM x WHERE p = 1; SELECT * FROM y ORDER BY z DESC LIMIT 3");
    assert_eq!(stmts.len(), 2);
    assert_eq!(stmts[0].select, Selection::Fields(vec!["a".into()]));
    assert!(stmts[0].filter.is_some());
    assert_eq!(stmts[1].select, Selection::All);
    assert_eq!(
        stmts[1].order,
        Some(Ranking::Field {
            field: "z".into(),
            desc: true
        })
    );
    assert_eq!(stmts[1].limit, Some(3));
}

#[test]
fn trailing_semicolon_is_optional() {
    assert_eq!(many("SELECT * FROM a").len(), 1);
    assert_eq!(many("SELECT * FROM a;").len(), 1);
}

#[test]
fn whitespace_around_semicolons_is_insignificant() {
    let stmts = many("SELECT * FROM a ; SELECT * FROM b");
    assert_eq!(stmts.len(), 2);
    assert_eq!(stmts[1].from, vec!["b".to_string()]);
}

#[test]
fn missing_select_keyword_is_an_error() {
    let sql = "FROM t";
    err_at(sql, 0, "'SELECT'", SEC_SYNTAX);
}

#[test]
fn leading_semicolon_is_an_error() {
    let sql = ";SELECT * FROM t";
    err_at(sql, 0, "'SELECT'", SEC_SYNTAX);
}

#[test]
fn lone_semicolon_is_an_error() {
    let sql = ";";
    err_at(sql, 0, "'SELECT'", SEC_SYNTAX);
}

#[test]
fn double_semicolon_is_an_error() {
    let sql = "SELECT * FROM a;;SELECT * FROM b";
    err_at(sql, sql.find(";;").unwrap() + 1, "'SELECT'", SEC_SYNTAX);
}

#[test]
fn empty_input_is_an_error() {
    let sql = "";
    err_at(sql, 0, "empty query", SEC_SYNTAX);
}

#[test]
fn whitespace_only_input_is_an_error() {
    let sql = "   \n\t  ";
    err_at(sql, 0, "empty query", SEC_SYNTAX);
}

#[test]
fn semicolon_inside_string_literal_does_not_split_the_script() {
    let stmts = many("SELECT * FROM t WHERE a = 'x;y'");
    assert_eq!(stmts.len(), 1);
    assert_eq!(
        stmts[0].filter,
        Some(PredNode::Cmp {
            field: "a".into(),
            op: CmpOp::Eq,
            value: Lit::Str("x;y".into())
        })
    );
}

// ── an omitted LIMIT on a ranked query (nidus-yq9p.6) ───────────────────────────
// `SearchOpts::default().top_k` is 0, so a bare ranked SELECT would otherwise return
// nothing and read as "no matches".

#[test]
fn knn_without_limit_defaults_to_ten() {
    let c = compiled_one("SELECT * FROM docs ORDER BY knn([1.0, 0.0])");
    assert!(format!("{c:?}").contains("top_k: 10"), "{c:?}");
}

#[test]
fn match_without_limit_defaults_to_ten() {
    let c = compiled_one("SELECT * FROM docs ORDER BY match(body, 'rust')");
    assert!(format!("{c:?}").contains("top_k: 10"), "{c:?}");
}

#[test]
fn fuse_without_limit_defaults_to_ten() {
    let c = compiled_one("SELECT * FROM docs ORDER BY knn([1.0]) FUSE match(body, 'rust')");
    assert!(format!("{c:?}").contains("top_k: 10"), "{c:?}");
}

#[test]
fn an_explicit_limit_zero_is_honoured_not_defaulted() {
    let c = compiled_one("SELECT * FROM docs ORDER BY knn([1.0, 0.0]) LIMIT 0");
    assert!(format!("{c:?}").contains("top_k: 0"), "{c:?}");
}

#[test]
fn an_explicit_limit_beats_the_default() {
    let c = compiled_one("SELECT * FROM docs ORDER BY knn([1.0, 0.0]) LIMIT 3");
    assert!(format!("{c:?}").contains("top_k: 3"), "{c:?}");
}

#[test]
fn list_dispatch_keeps_its_own_hundred_default() {
    let c = compiled_one("SELECT * FROM docs WHERE a = 1");
    assert!(format!("{c:?}").contains("limit: 100"), "{c:?}");
}

// ── timestamp literals (§7.12 G1): Value::DateTime is reachable from SQL ───────

#[test]
fn timestamp_string_literal_compiles_to_a_datetime() {
    let c = compiled_one("SELECT * FROM docs WHERE created > timestamp '2023-11-14T22:13:20Z'");
    assert!(
        format!("{c:?}").contains("DateTime(1700000000000)"),
        "{c:?}"
    );
}

#[test]
fn timestamp_epoch_millis_compiles_to_a_datetime() {
    let c = compiled_one("SELECT * FROM docs WHERE created > timestamp 1700000000000");
    assert!(
        format!("{c:?}").contains("DateTime(1700000000000)"),
        "{c:?}"
    );
}

#[test]
fn timestamp_accepts_a_pre_epoch_instant() {
    let c = compiled_one("SELECT * FROM docs WHERE created > timestamp '1969-12-31T23:59:59Z'");
    assert!(format!("{c:?}").contains("DateTime(-1000)"), "{c:?}");
}

#[test]
fn timestamp_keeps_milliseconds() {
    let c = compiled_one("SELECT * FROM docs WHERE created = timestamp '1970-01-01T00:00:00.250Z'");
    assert!(format!("{c:?}").contains("DateTime(250)"), "{c:?}");
}

#[test]
fn a_plain_integer_is_still_an_int_not_a_datetime() {
    let c = compiled_one("SELECT * FROM docs WHERE created > 1700000000000");
    assert!(format!("{c:?}").contains("Int(1700000000000)"), "{c:?}");
}

#[test]
fn timestamp_in_an_in_list_and_a_range_pair() {
    let c = compiled_one(
        "SELECT * FROM docs WHERE created >= timestamp '2020-01-01T00:00:00Z' \
         AND created < timestamp '2021-01-01T00:00:00Z'",
    );
    assert_eq!(format!("{c:?}").matches("DateTime(").count(), 2, "{c:?}");
}

#[test]
fn a_malformed_timestamp_string_is_rejected() {
    err_at(
        "SELECT * FROM docs WHERE created > timestamp 'not-a-date'",
        35,
        "ISO-8601",
        SEC_SYNTAX,
    );
}

#[test]
fn a_timestamp_without_a_value_is_rejected() {
    err_at(
        "SELECT * FROM docs WHERE created > timestamp",
        35,
        "expected a value",
        SEC_SYNTAX,
    );
}

#[test]
fn a_fractional_timestamp_number_is_rejected() {
    err_at(
        "SELECT * FROM docs WHERE created > timestamp 1.5",
        35,
        "whole milliseconds",
        SEC_SYNTAX,
    );
}

#[test]
fn timestamp_is_not_reserved_as_a_field_name() {
    let s = one("SELECT timestamp FROM docs");
    assert_eq!(s.select, Selection::Fields(vec!["timestamp".to_string()]));
}

// ── SUM without GROUP BY (§7.12 G2): whole-scope totals are reachable ──────────

#[test]
fn select_sum_without_group_by_compiles_to_whole_scope_totals() {
    let c = compiled_one("SELECT sum(bytes) FROM files");
    let d = format!("{c:?}");
    assert!(d.contains("Aggregate"), "{d}");
    assert!(d.contains("group_by: None"), "{d}");
    assert!(d.contains(r#"sum: ["bytes"]"#), "{d}");
}

#[test]
fn select_sum_with_group_by_keeps_the_group() {
    let c = compiled_one("SELECT sum(bytes) FROM files GROUP BY lang");
    let d = format!("{c:?}");
    assert!(d.contains(r#"group_by: Some("lang")"#), "{d}");
    assert!(d.contains(r#"sum: ["bytes"]"#), "{d}");
}

#[test]
fn count_star_alone_compiles_to_an_aggregate_with_no_sums() {
    let c = compiled_one("SELECT count(*) FROM files");
    let d = format!("{c:?}");
    assert!(d.contains("Aggregate"), "{d}");
    assert!(d.contains("sum: []"), "{d}");
}

#[test]
fn group_by_alone_still_dispatches_to_aggregate() {
    let c = compiled_one("SELECT * FROM files GROUP BY lang");
    assert!(format!("{c:?}").contains("Aggregate"), "{c:?}");
}

#[test]
fn an_aggregate_projection_rejects_limit() {
    err_at_compile_or_parse("SELECT sum(bytes) FROM files LIMIT 5", "LIMIT/OFFSET");
}

#[test]
fn an_aggregate_projection_rejects_order_by() {
    err_at_compile_or_parse("SELECT sum(bytes) FROM files ORDER BY lang", "ORDER BY");
}

/// Compile-layer rejection whose offset is the statement, not a token: assert the message.
fn err_at_compile_or_parse(sql: &str, needle: &str) {
    let e = match super::super::compile_all(sql) {
        Ok(_) => panic!("expected {sql:?} to be rejected, but it compiled"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        e.contains(needle),
        "message for {sql:?} lacks {needle:?}: {e}"
    );
    assert!(
        e.contains("at byte"),
        "message for {sql:?} lacks an offset: {e}"
    );
}
