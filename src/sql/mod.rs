//! SQL-shaped read syntax (nidus-yq9p.2/.3/.6): a hand-rolled, dependency-free front end
//! that lexes and parses one `SELECT` statement, then compiles it to the same typed
//! `SearchOpts`/`HybridOpts`/`ListOpts`/`AggregateOpts`/`Filter`/`FtsQuery` values the rest
//! of the crate already uses. There is no second execution path: see the root
//! `BLUEPRINT-nidus-yq9p-2-3-6.md` for the full grammar, the `WITH` key table, the dispatch
//! table, and the error format this module implements.
//!
//! Lean-lane, dependency-free (SPEC §1 Speed): no feature gate, no new crate. `serde_json`
//! is not available here (`Cargo.toml` makes it optional), so [`compile`] returns the typed
//! [`Compiled`] value, not a JSON string — surfaces that have `serde_json` render it there.

mod compile;
mod datetime;
mod error;
mod lex;
mod parse;

pub(crate) use error::SQL_PARSE_ERROR;

use crate::model::{AggregateOpts, Aggregation, FtsQuery, Hit, HybridOpts, ListOpts, SearchOpts};
use crate::plan::QueryPlan;

/// Nested-`(...)` recursion guard for the `WHERE` boolean tree (root blueprint): a
/// hand-rolled parser inherits none of `serde_json`'s de facto 128-level cap, and unbounded
/// recursion on attacker-supplied text is a stack overflow, not an error (SPEC §1 Stable).
const MAX_NEST_DEPTH: usize = 128;

/// One statement, compiled: the store entry point its `ORDER BY` head selected, and the
/// typed options that entry point runs with. Never executed by this module — `src/lib.rs`
/// matches over it and calls the existing `Store` methods (root blueprint's dispatch table).
#[derive(Clone, Debug)]
pub enum Compiled {
    Search {
        collections: Vec<String>,
        vector: Vec<f32>,
        opts: SearchOpts,
    },
    TextSearch {
        collections: Vec<String>,
        query: FtsQuery,
        opts: SearchOpts,
    },
    Hybrid {
        collections: Vec<String>,
        vector: Vec<f32>,
        text: FtsQuery,
        opts: HybridOpts,
    },
    List {
        collections: Vec<String>,
        opts: ListOpts,
    },
    Aggregate {
        collections: Vec<String>,
        opts: AggregateOpts,
    },
}

/// What running a [`Compiled`] value produces. `plan` is `Some` only when the statement
/// asked for one (`WITH (plan)`, §7.11) — always `None` on [`Aggregation`], which has no
/// plan-carrying entry point.
#[derive(Clone, Debug, PartialEq)]
pub enum QueryAnswer {
    Hits {
        hits: Vec<Hit>,
        plan: Option<QueryPlan>,
    },
    Aggregation(Aggregation),
}

/// Parse and compile every statement in `sql` (§7.9: `;`-separated). The one place a
/// [`error::SqlError`] becomes an `anyhow::Error` — every downstream surface sees the same
/// `Display` text through the chain that already exists (root blueprint's error model).
pub(crate) fn compile_all(sql: &str) -> anyhow::Result<Vec<Compiled>> {
    parse::parse(sql)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .iter()
        .map(|stmt| compile::compile(stmt).map_err(|e| anyhow::anyhow!("{e}")))
        .collect()
}

#[cfg(test)]
mod grammar_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Filter, Predicate, Value};

    fn compiled_one(sql: &str) -> Compiled {
        let mut c = compile_all(sql).unwrap();
        assert_eq!(c.len(), 1, "expected exactly one compiled statement");
        c.remove(0)
    }

    // ── every dispatch arm reaches the right `Compiled` variant ─────────────────

    #[test]
    fn knn_dispatches_to_search() {
        let c =
            compiled_one("SELECT * FROM docs WHERE lang = 'rust' ORDER BY knn([1.0, 0.0]) LIMIT 5");
        let Compiled::Search {
            collections,
            vector,
            opts,
        } = c
        else {
            panic!("expected Search")
        };
        assert_eq!(collections, vec!["docs".to_string()]);
        assert_eq!(vector, vec![1.0, 0.0]);
        assert_eq!(opts.top_k, 5);
        assert_eq!(
            opts.filter,
            Filter(vec![Predicate::Eq(
                "lang".into(),
                Value::Str("rust".into())
            )])
        );
    }

    #[test]
    fn match_dispatches_to_text_search() {
        let c = compiled_one("SELECT * FROM docs ORDER BY match(body, 'rust async')");
        assert!(matches!(c, Compiled::TextSearch { .. }));
    }

    #[test]
    fn knn_fuse_match_dispatches_to_hybrid() {
        let c = compiled_one("SELECT * FROM docs ORDER BY knn([1.0]) FUSE match(body, 'rust')");
        assert!(matches!(c, Compiled::Hybrid { .. }));
    }

    #[test]
    fn bare_field_order_by_dispatches_to_list() {
        let c = compiled_one("SELECT * FROM docs ORDER BY score DESC LIMIT 20");
        let Compiled::List { opts, .. } = c else {
            panic!("expected List")
        };
        assert_eq!(opts.limit, 20);
        assert!(opts.order_by.unwrap().descending);
    }

    #[test]
    fn no_order_by_dispatches_to_list_too() {
        let c = compiled_one("SELECT * FROM docs");
        assert!(matches!(c, Compiled::List { .. }));
    }

    #[test]
    fn group_by_dispatches_to_aggregate() {
        let c = compiled_one("SELECT * FROM docs GROUP BY project, SUM(bytes)");
        let Compiled::Aggregate { opts, .. } = c else {
            panic!("expected Aggregate")
        };
        assert_eq!(opts.group_by, Some("project".to_string()));
        assert_eq!(opts.sum, vec!["bytes".to_string()]);
    }

    // ── FROM *, and an omitted FROM name, are both "every collection" ───────────

    #[test]
    fn from_star_is_the_empty_scope() {
        let c = compiled_one("SELECT * FROM *");
        let Compiled::List { collections, .. } = c else {
            panic!("expected List")
        };
        assert_eq!(collections, Vec::<String>::new());
    }

    // ── WHERE compiles to the same Filter/Predicate/Value a typed caller writes ──

    #[test]
    fn boolean_composition_and_negation_compile_correctly() {
        let c = compiled_one(
            "SELECT * FROM docs WHERE (a = 1 OR b = 2) AND NOT c LIKE 'x*' ORDER BY score",
        );
        let Compiled::List { opts, .. } = c else {
            panic!("expected List")
        };
        assert_eq!(
            opts.filter,
            Filter(vec![
                Predicate::Any(vec![
                    Predicate::Eq("a".into(), Value::Int(1)),
                    Predicate::Eq("b".into(), Value::Int(2)),
                ]),
                Predicate::Not(Box::new(Predicate::Glob("c".into(), "x*".into()))),
            ])
        );
    }

    #[test]
    fn every_predicate_family_compiles_through_the_pipeline() {
        let c = compiled_one(
            "SELECT * FROM docs WHERE \
             a IN (1, 2) AND b NOT IN (3) AND c ILIKE 'X*' AND d ~ 'v[0-9]+' AND \
             contains(tags, 'x') AND not_contains(tags, 'y') AND contains_any(tags, 'p', 'q') AND \
             fuzzy(id, 'nidus', 2) AND match_all(body, 'a b') AND match_any(body, 'c d') AND \
             phrase(body, 'e f')",
        );
        let Compiled::List { opts, .. } = c else {
            panic!("expected List")
        };
        let Filter(preds) = opts.filter;
        assert_eq!(
            preds,
            vec![
                Predicate::In("a".into(), vec![Value::Int(1), Value::Int(2)]),
                Predicate::NotIn("b".into(), vec![Value::Int(3)]),
                Predicate::IGlob("c".into(), "X*".into()),
                Predicate::Regex("d".into(), "v[0-9]+".into()),
                Predicate::Contains("tags".into(), Value::Str("x".into())),
                Predicate::NotContains("tags".into(), Value::Str("y".into())),
                Predicate::ContainsAny(
                    "tags".into(),
                    vec![Value::Str("p".into()), Value::Str("q".into())]
                ),
                Predicate::Fuzzy("id".into(), "nidus".into(), 2),
                Predicate::ContainsAllTokens("body".into(), "a b".into()),
                Predicate::ContainsAnyToken("body".into(), "c d".into()),
                Predicate::ContainsTokenSequence("body".into(), "e f".into()),
            ]
        );
    }

    // ── every WITH key, on a dispatch that can honour it ────────────────────────

    #[test]
    fn with_annotations_sets_explain() {
        let c = compiled_one("SELECT * FROM docs ORDER BY knn([1.0]) WITH (annotations)");
        let Compiled::Search { opts, .. } = c else {
            panic!()
        };
        assert!(opts.explain);
    }

    #[test]
    fn with_plan_sets_plan() {
        let c = compiled_one("SELECT * FROM docs ORDER BY knn([1.0]) WITH (plan)");
        let Compiled::Search { opts, .. } = c else {
            panic!()
        };
        assert!(opts.plan);
    }

    #[test]
    fn with_exact_sets_exact() {
        let c = compiled_one("SELECT * FROM docs ORDER BY knn([1.0]) WITH (exact)");
        let Compiled::Search { opts, .. } = c else {
            panic!()
        };
        assert!(opts.exact);
    }

    #[test]
    fn with_min_score_and_diversity() {
        let c = compiled_one(
            "SELECT * FROM docs ORDER BY knn([1.0]) WITH (min_score = 0.5, diversity = 0.3)",
        );
        let Compiled::Search { opts, .. } = c else {
            panic!()
        };
        assert_eq!(opts.min_score, Some(0.5));
        assert_eq!(opts.diversity, Some(0.3));
    }

    #[test]
    fn with_limit_per() {
        let c = compiled_one("SELECT * FROM docs ORDER BY knn([1.0]) WITH (limit_per = (file, 2))");
        let Compiled::Search { opts, .. } = c else {
            panic!()
        };
        let cap = opts.limit_per.unwrap();
        assert_eq!(cap.field, "file");
        assert_eq!(cap.max, 2);
    }

    #[test]
    fn with_context_builds_expand() {
        let c = compiled_one(
            r#"SELECT * FROM docs ORDER BY knn([1.0]) WITH (context = (radius 2, parent "pid", index "idx", text "txt"))"#,
        );
        let Compiled::Search { opts, .. } = c else {
            panic!()
        };
        let expand = opts.expand.unwrap();
        assert_eq!(expand.radius, 2);
        assert_eq!(expand.parent_field, "pid");
        assert_eq!(expand.index_field, "idx");
        assert_eq!(expand.text_field, "txt");
    }

    #[test]
    fn with_rerank_builds_rerank_opts() {
        let c = compiled_one(
            r#"SELECT * FROM docs ORDER BY knn([1.0]) WITH (rerank = (overscan 20, text "body"))"#,
        );
        let Compiled::Search { opts, .. } = c else {
            panic!()
        };
        let rr = opts.rerank.unwrap();
        assert_eq!(rr.overscan, 20);
        assert_eq!(rr.text_attr, "body");
    }

    #[test]
    fn with_candidates_rrf_k_weights_on_a_fuse_query() {
        let c = compiled_one(
            "SELECT * FROM docs ORDER BY knn([1.0]) FUSE match(body, 'x') \
             WITH (candidates = 200, rrf_k = 30, weights = (2.0, 1.0))",
        );
        let Compiled::Hybrid { opts, .. } = c else {
            panic!()
        };
        assert_eq!(opts.candidates, 200);
        assert_eq!(opts.rrf_k, 30.0);
        assert_eq!(opts.vector_weight, 2.0);
        assert_eq!(opts.text_weight, 1.0);
    }

    // ── contradictions: a WITH key the dispatch's opts cannot hold ──────────────

    #[test]
    fn weights_on_a_non_fuse_query_is_a_compile_error() {
        let err = compile_all("SELECT * FROM docs ORDER BY knn([1.0]) WITH (weights = (1.0, 2.0))")
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(SQL_PARSE_ERROR), "{msg}");
        assert!(msg.contains("weights"), "{msg}");
        assert!(msg.contains("§7.6"), "{msg}");
    }

    #[test]
    fn annotations_on_a_plain_list_is_a_compile_error() {
        // No ranking at all, so `SearchOpts::explain` is simply not in play.
        let err = compile_all("SELECT * FROM docs WITH (annotations)").unwrap_err();
        assert!(err.to_string().contains("annotations"));
    }

    #[test]
    fn select_columns_on_a_hybrid_query_is_a_compile_error() {
        let err = compile_all("SELECT a FROM docs ORDER BY knn([1.0]) FUSE match(body, 'x')")
            .unwrap_err();
        assert!(err.to_string().contains("SELECT columns"));
    }

    #[test]
    fn group_by_with_a_vector_ranking_is_a_compile_error() {
        let err =
            compile_all("SELECT * FROM docs GROUP BY project ORDER BY knn([1.0])").unwrap_err();
        assert!(err.to_string().contains("GROUP BY"));
    }

    #[test]
    fn unknown_with_key_is_a_compile_error() {
        let err =
            compile_all("SELECT * FROM docs ORDER BY knn([1.0]) WITH (nonsense)").unwrap_err();
        assert!(err.to_string().contains("unknown WITH option"));
    }

    // ── multi-query batching (§7.9) ──────────────────────────────────────────────

    #[test]
    fn compile_all_splits_a_script() {
        let c = compile_all("SELECT * FROM a; SELECT * FROM b").unwrap();
        assert_eq!(c.len(), 2);
    }

    // ── the error format itself, end to end through `compile_all` ───────────────

    #[test]
    fn a_parse_error_carries_the_marker_offset_and_section() {
        let err = compile_all("SELECT * FROM docs WHERE a =")
            .unwrap_err()
            .to_string();
        assert!(err.starts_with(SQL_PARSE_ERROR), "{err}");
        assert!(err.contains("at byte"), "{err}");
        assert!(err.contains('§'), "{err}");
    }
}
