//! The parse-grammar test corpus (nidus-yq9p.6). Split by concern so each file stays
//! readable; `helpers` holds the shared entry points every file uses.

mod abuse;
mod clauses;
mod errors;
mod predicates;
mod with_opts;

use super::error::SqlError;
use super::parse::{PredNode, Ranking, Statement, parse};
use super::{Compiled, compile_all};

/// Parse exactly one statement, panicking with the SQL and the error on failure.
pub(super) fn one(sql: &str) -> Statement {
    let mut stmts = parse(sql).unwrap_or_else(|e| panic!("parse({sql:?}) failed: {e}"));
    assert_eq!(
        stmts.len(),
        1,
        "expected exactly one statement from {sql:?}"
    );
    stmts.remove(0)
}

/// Parse a whole `;`-separated script.
pub(super) fn many(sql: &str) -> Vec<Statement> {
    parse(sql).unwrap_or_else(|e| panic!("parse({sql:?}) failed: {e}"))
}

/// The error a statement must produce, panicking if it parsed instead.
pub(super) fn err(sql: &str) -> SqlError {
    match parse(sql) {
        Ok(_) => panic!("expected {sql:?} to fail parsing, but it parsed"),
        Err(e) => e,
    }
}

/// Assert an input fails at `at` with `needle` in the message and `section` named. Every
/// error test goes through here so no test can settle for "it errored".
pub(super) fn err_at(sql: &str, at: usize, needle: &str, section: &str) {
    let e = err(sql);
    let shown = e.to_string();
    assert_eq!(e.at, at, "wrong byte offset for {sql:?}: {shown}");
    assert!(
        shown.contains(needle),
        "message for {sql:?} lacks {needle:?}: {shown}"
    );
    assert!(
        shown.contains(section),
        "message for {sql:?} lacks section {section:?}: {shown}"
    );
}

/// Assert two spellings parse to the same tree. Byte offsets are zeroed first: padding or
/// re-casing an input legitimately moves every `at`, and this helper is about shape.
pub(super) fn same(a: &str, b: &str) {
    let (mut x, mut y) = (one(a), one(b));
    zero_offsets(&mut x);
    zero_offsets(&mut y);
    assert_eq!(x, y, "{a:?} and {b:?} should parse alike");
}

fn zero_offsets(s: &mut Statement) {
    s.at = 0;
    if let Some(f) = s.filter.as_mut() {
        zero_pred(f);
    }
    if let Some(r) = s.order.as_mut() {
        match r {
            Ranking::Knn { at, .. } | Ranking::Match { at, .. } | Ranking::Hybrid { at, .. } => {
                *at = 0;
            }
            Ranking::Field { .. } => {}
        }
    }
    for w in &mut s.with {
        w.name_at = 0;
    }
}

fn zero_pred(p: &mut PredNode) {
    match p {
        PredNode::Fuzzy { at, .. } => *at = 0,
        PredNode::All(items) | PredNode::Any(items) => items.iter_mut().for_each(zero_pred),
        PredNode::Not(inner) => zero_pred(inner),
        _ => {}
    }
}

/// Compile a statement, panicking with the SQL on failure. `WITH`-key value types and tuple
/// arity are semantics, not syntax, so they are rejected here rather than by `parse`.
pub(super) fn compiled_one(sql: &str) -> Compiled {
    let mut c = compile_all(sql).unwrap_or_else(|e| panic!("compile({sql:?}) failed: {e}"));
    assert_eq!(c.len(), 1, "expected exactly one compiled statement");
    c.remove(0)
}

/// Assert a statement parses but fails to COMPILE, with the offset, message and §7 section.
pub(super) fn compile_err_at(sql: &str, at: usize, needle: &str, section: &str) {
    parse(sql).unwrap_or_else(|e| panic!("{sql:?} should parse and fail at compile, got: {e}"));
    let shown = match compile_all(sql) {
        Ok(_) => panic!("expected {sql:?} to fail compiling, but it compiled"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        shown.contains(&format!("at byte {at}")),
        "wrong byte offset for {sql:?} (wanted {at}): {shown}"
    );
    assert!(
        shown.contains(needle),
        "message for {sql:?} lacks {needle:?}: {shown}"
    );
    assert!(
        shown.contains(section),
        "message for {sql:?} lacks section {section:?}: {shown}"
    );
}
