//! The one SQL error format (root `BLUEPRINT-nidus-yq9p-2-3-6.md`'s error model): a byte
//! offset into the original text, a message, and the `§7.x` section the violated rule lives
//! in. Every surface sees this same `Display` text through the `anyhow::Error` chain.

use std::fmt;

/// The greppable marker every SQL front-end failure carries, so `classify()`
/// (`src/server/mod.rs`, a later unit) can answer 400 rather than 500 — mirrors
/// `crate::store::BAD_QUERY`.
pub(crate) const SQL_PARSE_ERROR: &str = "sql parse error";

// `SPEC.md` §7 section tails, lowercase-first to match the one Display format. §7.12 does
// not exist yet in this worktree (a later unit adds it); it is the general-syntax catch-all.
pub(super) const SEC_SYNTAX: &str = "§7.12 SQL syntax";
pub(super) const SEC_GLOB: &str = "§7.1 glob subset";
pub(super) const SEC_ARRAY: &str = "§7.2 array containment";
pub(super) const SEC_BOOL: &str = "§7.3 boolean composition";
pub(super) const SEC_TEXT: &str = "§7.4 fuzzy and token text predicates";
pub(super) const SEC_REGEX: &str = "§7.5 regular expressions";
pub(super) const SEC_RANK: &str = "§7.6 ranking expressions";
pub(super) const SEC_AGG: &str = "§7.7 aggregation and result diversity";
pub(super) const SEC_ANNOTATE: &str = "§7.8 result annotations";
pub(super) const SEC_BATCH: &str = "§7.9 multi-query batching";
pub(super) const SEC_EXPAND: &str = "§7.10 parent rollup and neighbour expansion";
pub(super) const SEC_PLAN: &str = "§7.11 query plans";

/// A SQL front-end failure: where in the source it happened, what went wrong, and which
/// `SPEC.md` §7 section owns the rule. The one type every lexer/parser/compiler fn returns.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SqlError {
    pub at: usize,
    pub message: String,
    pub section: &'static str,
}

impl SqlError {
    pub(super) fn new(at: usize, message: impl Into<String>, section: &'static str) -> Self {
        Self {
            at,
            message: message.into(),
            section,
        }
    }
}

impl fmt::Display for SqlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{SQL_PARSE_ERROR} at byte {}: {} ({})",
            self.at, self.message, self.section
        )
    }
}

impl std::error::Error for SqlError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_matches_the_one_format() {
        let e = SqlError::new(17, "expected a value after '='", SEC_BOOL);
        assert_eq!(
            e.to_string(),
            "sql parse error at byte 17: expected a value after '=' (§7.3 boolean composition)"
        );
    }
}
