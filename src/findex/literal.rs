//! Required-literal extraction from a regex, for narrowing `Predicate::Regex` through the
//! trigram index. Contract: see the root `SPEC.md` §7.5.
//!
//! A literal is *required* only if **every** string the pattern can match contains it.
//! Anything less and narrowing drops real matches, so every uncertain construct here
//! returns "no requirement" and the caller falls back to the full scan.

use regex_syntax::hir::{Hir, HirKind};

/// Literals every match must contain. Empty means no narrowing is possible — the honest
/// answer for `.*`, for an alternation whose branches share nothing, and for anything this
/// analysis does not understand.
pub(crate) fn required_literals(pattern: &str) -> Vec<String> {
    // The predicate anchors both ends (`filter/pattern.rs`), but anchoring changes only
    // where a match may sit, never which literals it must contain, so parse the raw form.
    let Ok(hir) = regex_syntax::parse(pattern) else {
        return Vec::new();
    };
    let mut out = required(&hir);
    out.retain(|l| !l.is_empty());
    out.sort_unstable();
    out.dedup();
    out
}

/// Literals required by `hir`. Concatenation accumulates; alternation may only keep what
/// **all** branches require; everything optional or unbounded contributes nothing.
fn required(hir: &Hir) -> Vec<String> {
    match hir.kind() {
        HirKind::Literal(lit) => String::from_utf8(lit.0.to_vec())
            .map(|s| vec![s])
            .unwrap_or_default(),

        // A concatenation requires each part's literals. Adjacent literals are joined so
        // `ab` `cd` yields `abcd` — a longer literal narrows far better than two short ones.
        HirKind::Concat(parts) => {
            let mut out: Vec<String> = Vec::new();
            let mut run = String::new();
            for p in parts {
                if let HirKind::Literal(lit) = p.kind()
                    && let Ok(s) = std::str::from_utf8(&lit.0)
                {
                    run.push_str(s);
                    continue;
                }
                if !run.is_empty() {
                    out.push(std::mem::take(&mut run));
                }
                out.extend(required(p));
            }
            if !run.is_empty() {
                out.push(run);
            }
            out
        }

        // Only literals required by EVERY branch survive. `a|b` requires neither, so an
        // empty branch set collapses the whole alternation to no requirement.
        HirKind::Alternation(branches) => {
            let mut common: Option<Vec<String>> = None;
            for b in branches {
                let here = required(b);
                if here.is_empty() {
                    return Vec::new();
                }
                common = Some(match common {
                    None => here,
                    Some(prev) => prev.into_iter().filter(|l| here.contains(l)).collect(),
                });
                if common.as_ref().is_some_and(Vec::is_empty) {
                    return Vec::new();
                }
            }
            common.unwrap_or_default()
        }

        // A repetition guarantees its sub-pattern only when it must occur at least once.
        HirKind::Repetition(r) if r.min >= 1 => required(&r.sub),

        HirKind::Capture(c) => required(&c.sub),

        // Empty, Look (anchors/boundaries), Class, and min-zero repetitions all match
        // without contributing a required literal.
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::required_literals;

    #[test]
    fn a_plain_literal_is_required() {
        assert_eq!(required_literals("hello"), ["hello"]);
    }

    #[test]
    fn adjacent_literals_join_into_one_longer_requirement() {
        assert_eq!(required_literals("(?:ab)(?:cd)"), ["abcd"]);
    }

    #[test]
    fn a_wildcard_requires_nothing() {
        assert!(required_literals(".*").is_empty());
        assert!(required_literals(".+").is_empty());
    }

    #[test]
    fn an_alternation_requires_only_what_every_branch_requires() {
        // The bug this guards: taking "a" from the first branch would drop every "b" match.
        assert!(required_literals("a|b").is_empty());
        // Branches sharing a literal keep it; branches that do not keep nothing.
        assert_eq!(required_literals("ab|ab"), ["ab"]);
        assert!(required_literals("xay|xby").is_empty());
    }

    /// The safety contract, as a property rather than an example: anything reported as
    /// required must genuinely appear in every string the pattern matches. A literal that
    /// is not really required would narrow away real matches.
    #[test]
    fn every_reported_literal_really_appears_in_every_match() {
        let patterns = [
            "hello",
            "a|b",
            "ab|ab",
            ".*term0 .*",
            "abc(xyz)?",
            "(?:ab)+",
            "v[0-9]+x",
            "[a-z]+",
            "^abc$",
            "foo|.*",
            "(?i)readme",
            "a?b",
            "(ab|ac)d",
        ];
        let corpus = [
            "hello", "a", "b", "ab", "abc", "abcxyz", "term0 x", "xterm0 y", "v12x", "vx",
            "readme", "README", "abd", "acd", "zzz", "",
        ];
        for p in patterns {
            let lits = required_literals(p);
            if lits.is_empty() {
                continue;
            }
            let re = regex::Regex::new(p).unwrap();
            for text in corpus {
                if !re.is_match(text) {
                    continue;
                }
                for l in &lits {
                    assert!(
                        text.contains(l.as_str()),
                        "pattern {p:?} matched {text:?} but required literal {l:?} is absent"
                    );
                }
            }
        }
    }

    #[test]
    fn an_alternation_with_one_unconstrained_branch_requires_nothing() {
        assert!(required_literals("foo|.*").is_empty());
    }

    #[test]
    fn an_optional_group_requires_nothing_but_its_neighbours_still_count() {
        assert_eq!(required_literals("abc(xyz)?"), ["abc"]);
        assert!(required_literals("(xyz)?").is_empty());
    }

    #[test]
    fn a_one_or_more_repetition_requires_its_body() {
        assert_eq!(required_literals("(?:ab)+"), ["ab"]);
        assert!(required_literals("(?:ab)*").is_empty());
    }

    #[test]
    fn a_character_class_requires_nothing() {
        assert!(required_literals("[a-z]+").is_empty());
        assert_eq!(required_literals("v[0-9]+x"), ["v", "x"]);
    }

    #[test]
    fn surrounding_wildcards_do_not_erase_the_literal() {
        assert_eq!(required_literals(".*term0 .*"), ["term0 "]);
    }

    #[test]
    fn an_unparseable_pattern_requires_nothing() {
        assert!(required_literals("(").is_empty());
    }

    #[test]
    fn an_anchor_contributes_nothing_and_breaks_no_run() {
        assert_eq!(required_literals("^abc$"), ["abc"]);
    }
}
