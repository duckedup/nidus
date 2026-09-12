//! The shared code-search BM25 query, so the CLI, HTTP and MCP surfaces cannot drift apart
//! (nidus-hij). Callers own scope, `top_k` and any [`crate::Filter`]; this builds only the
//! [`crate::FtsQuery`] leg.

use crate::model::META_TEXT;

use super::{META_DOC, META_PATH, META_SYMBOL};

/// What code search's BM25 leg is declared over: the chunk body, plus the three fields a
/// caller is most likely to name directly. `code ingest` declares exactly these as the FTS
/// schema, so the CLI, HTTP and MCP surfaces cannot drift apart again (nidus-hij).
pub const CODE_FTS_FIELDS: [&str; 4] = [META_TEXT, META_PATH, META_SYMBOL, META_DOC];

/// One clause per declared field: a query word may live in the body, the path, the symbol
/// name or the doc comment. `Max` rather than `Sum`, so a long body cannot out-accumulate an
/// exact symbol-name match.
pub fn fts_query(query: &str) -> crate::FtsQuery {
    crate::FtsQuery {
        combine: crate::FtsCombine::Max,
        ..crate::FtsQuery::multi(
            CODE_FTS_FIELDS
                .into_iter()
                .map(|f| crate::FtsClause::new(f, query.to_string())),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_query_covers_exactly_the_four_declared_fields() {
        let q = fts_query("parse_manifest");
        let fields: Vec<&str> = q.clauses.iter().map(|c| c.field.as_str()).collect();
        assert_eq!(
            fields,
            vec!["nidus.text", "code.path", "code.symbol", "code.doc"]
        );
    }

    #[test]
    fn fts_query_combines_by_max_not_the_default_sum() {
        let q = fts_query("anything");
        assert_eq!(q.combine, crate::FtsCombine::Max);
    }

    #[test]
    fn fts_query_carries_the_callers_text_on_every_clause() {
        let q = fts_query("parse_manifest");
        for c in &q.clauses {
            assert_eq!(c.text, "parse_manifest");
        }
    }
}
