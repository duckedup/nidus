//! `Predicate::Regex`: compiled patterns and the per-query compile cache. See `SPEC.md` §7.5.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use anyhow::{Context, Result};
use regex::Regex;

use crate::model::Value;

use super::text::any_text;

/// Distinct patterns held compiled. Patterns arrive from untrusted bodies, so an overflowing
/// cache is cleared wholesale rather than grown without bound — the next query recompiles.
const CACHE_CAPACITY: usize = 256;

/// Compiled patterns, keyed by the *caller's* pattern text. A filter is evaluated once per
/// scanned record, so compiling there would dwarf the scan; the cache turns the per-record
/// cost into a read-locked lookup.
static COMPILED: LazyLock<RwLock<HashMap<String, Arc<Regex>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Anchored at both ends, matching `Glob`/`IGlob` — the whole attribute must match, and
/// `.*` on either side opts back into a substring search. Case folding is the pattern's own
/// `(?i)`, so there is no second predicate variant for it.
fn anchored(pattern: &str) -> String {
    format!("^(?:{pattern})$")
}

/// Compile `pattern` into the cache, surfacing a syntax error to the caller. Called once per
/// query from `filter::validate`, never from the per-record path.
pub(super) fn compile(pattern: &str) -> Result<Arc<Regex>> {
    if let Some(hit) = COMPILED.read().ok().and_then(|c| c.get(pattern).cloned()) {
        return Ok(hit);
    }
    let compiled = Arc::new(
        Regex::new(&anchored(pattern))
            .with_context(|| format!("invalid Regex pattern `{pattern}`"))?,
    );
    if let Ok(mut cache) = COMPILED.write() {
        if cache.len() >= CACHE_CAPACITY {
            cache.clear();
        }
        cache.insert(pattern.to_string(), Arc::clone(&compiled));
    }
    Ok(compiled)
}

/// [`crate::Predicate::Regex`]. Reads any text the attribute carries, like the other text
/// predicates. An unparseable pattern matches nothing here — `validate` has already turned
/// it into an error on every path a caller can reach.
pub(super) fn regex_matches(value: Option<&Value>, pattern: &str) -> bool {
    let Ok(re) = compile(pattern) else {
        return false;
    };
    any_text(value, |text| re.is_match(text))
}

#[cfg(test)]
mod tests {
    use super::{compile, regex_matches};
    use crate::model::Value;

    #[test]
    fn an_invalid_pattern_is_an_error_not_a_panic() {
        let err = compile("(").unwrap_err().to_string();
        assert!(err.contains("invalid Regex pattern"), "{err}");
    }

    #[test]
    fn compiling_the_same_pattern_twice_returns_the_cached_arc() {
        let a = compile("cached-[0-9]+").unwrap();
        let b = compile("cached-[0-9]+").unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn matching_is_anchored_at_both_ends() {
        let v = Value::Str("quick brown fox".into());
        assert!(regex_matches(Some(&v), "quick brown fox"));
        assert!(!regex_matches(Some(&v), "brown"));
        assert!(regex_matches(Some(&v), ".*brown.*"));
    }

    #[test]
    fn an_alternation_is_anchored_as_a_whole_not_per_branch() {
        // The wrapping group is why `a|b` cannot leak into an unanchored `^a|b$`.
        let v = Value::Str("b".into());
        assert!(regex_matches(Some(&v), "a|b"));
        assert!(!regex_matches(Some(&Value::Str("xbx".into())), "a|b"));
    }

    #[test]
    fn the_inline_flag_is_the_case_switch() {
        let v = Value::Str("README.md".into());
        assert!(!regex_matches(Some(&v), "readme\\.md"));
        assert!(regex_matches(Some(&v), "(?i)readme\\.md"));
    }

    #[test]
    fn a_caller_supplied_anchor_still_works() {
        let v = Value::Str("abc".into());
        assert!(regex_matches(Some(&v), "^abc$"));
    }
}
