//! Fuzzy (Levenshtein) and token text predicates. Contract: see the root `SPEC.md` §7.4.

use crate::model::Value;

/// The largest edit budget [`crate::Predicate::Fuzzy`] accepts. Past this the DP stops being
/// a cheap per-row check and the match stops meaning anything, so an over-budget filter is
/// rejected rather than clamped — a clamp answers a different question than the one asked.
pub(crate) const MAX_FUZZY_EDITS: usize = 8;

/// True iff some text this attribute carries satisfies `pred`: a `Str` offers itself, a
/// `List` offers each element, and every other variant offers nothing — so an absent or
/// wrong-type attribute is never a match, per the leaf rule in §7.1.
pub(super) fn any_text(value: Option<&Value>, pred: impl Fn(&str) -> bool) -> bool {
    match value {
        Some(Value::Str(s)) => pred(s),
        Some(Value::List(items)) => items.iter().any(|i| pred(i)),
        _ => false,
    }
}

/// ASCII-case-folded Levenshtein distance (substitution, insertion, deletion — a
/// transposition therefore costs 2), computed with a two-row DP. Non-ASCII is not folded,
/// following `IGlob`: a locale-dependent fold means different things on different machines.
pub(crate) fn levenshtein_ascii_ci(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().map(|c| c.to_ascii_lowercase()).collect();
    let b: Vec<char> = b.chars().map(|c| c.to_ascii_lowercase()).collect();
    if a.is_empty() || b.is_empty() {
        return a.len().max(b.len());
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let subst = prev[j] + usize::from(ca != cb);
            cur[j + 1] = subst.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Split text into ASCII-case-folded runs of alphanumerics. Deliberately simpler than the
/// FTS analyzer — no stemming, no stopwords — because these are *filter* predicates, where
/// a term either is or is not present, not ranking.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

/// [`crate::Predicate::Fuzzy`]. The length pre-check is not just a speed-up: it skips the
/// whole DP for the common case of a needle nowhere near the attribute's size.
pub(super) fn fuzzy(value: Option<&Value>, needle: &str, max_edits: usize) -> bool {
    let needle_chars = needle.chars().count();
    any_text(value, |text| {
        text.chars().count().abs_diff(needle_chars) <= max_edits
            && levenshtein_ascii_ci(text, needle) <= max_edits
    })
}

/// [`crate::Predicate::ContainsAllTokens`]. An empty query matches any present text
/// attribute, the same vacuous-truth identity `All([])` takes.
pub(super) fn contains_all_tokens(value: Option<&Value>, query: &str) -> bool {
    let want = tokenize(query);
    any_text(value, |text| {
        let have = tokenize(text);
        want.iter().all(|w| have.contains(w))
    })
}

/// [`crate::Predicate::ContainsAnyToken`]. An empty query matches nothing, the identity
/// `Any([])` and an empty `In` set already take.
pub(super) fn contains_any_token(value: Option<&Value>, query: &str) -> bool {
    let want = tokenize(query);
    any_text(value, |text| {
        let have = tokenize(text);
        want.iter().any(|w| have.contains(w))
    })
}

/// [`crate::Predicate::ContainsTokenSequence`]: the query's tokens as a consecutive,
/// in-order run. A phrase never spans two `List` elements — each element is its own text.
pub(super) fn contains_token_sequence(value: Option<&Value>, query: &str) -> bool {
    let want = tokenize(query);
    any_text(value, |text| {
        if want.is_empty() {
            return true;
        }
        tokenize(text).windows(want.len()).any(|w| w == want)
    })
}

#[cfg(test)]
mod tests {
    use super::{levenshtein_ascii_ci, tokenize};

    // ── Levenshtein ──────────────────────────────────────────────────────────────

    #[test]
    fn identical_strings_are_zero_edits() {
        assert_eq!(levenshtein_ascii_ci("kitten", "kitten"), 0);
    }

    #[test]
    fn empty_against_anything_is_its_length() {
        assert_eq!(levenshtein_ascii_ci("", "abc"), 3);
        assert_eq!(levenshtein_ascii_ci("abc", ""), 3);
        assert_eq!(levenshtein_ascii_ci("", ""), 0);
    }

    #[test]
    fn substitution_costs_one_per_char() {
        assert_eq!(levenshtein_ascii_ci("cat", "bat"), 1);
        assert_eq!(levenshtein_ascii_ci("cat", "bar"), 2);
    }

    #[test]
    fn insertion_and_deletion_cost_one() {
        assert_eq!(levenshtein_ascii_ci("color", "colour"), 1);
        assert_eq!(levenshtein_ascii_ci("colour", "color"), 1);
    }

    #[test]
    fn transposition_costs_two_not_one() {
        // Plain Levenshtein, not Damerau: a swap is a delete plus an insert.
        assert_eq!(levenshtein_ascii_ci("form", "from"), 2);
        assert_eq!(levenshtein_ascii_ci("ab", "ba"), 2);
    }

    #[test]
    fn the_textbook_case() {
        assert_eq!(levenshtein_ascii_ci("kitten", "sitting"), 3);
        assert_eq!(levenshtein_ascii_ci("saturday", "sunday"), 3);
    }

    #[test]
    fn distance_is_symmetric() {
        assert_eq!(
            levenshtein_ascii_ci("nidus", "nimbus"),
            levenshtein_ascii_ci("nimbus", "nidus")
        );
    }

    #[test]
    fn ascii_case_is_folded_but_non_ascii_is_not() {
        assert_eq!(levenshtein_ascii_ci("Kitten", "kITTEN"), 0);
        assert_eq!(levenshtein_ascii_ci("café", "CAFÉ"), 1);
    }

    #[test]
    fn distance_counts_chars_not_bytes() {
        // "é" is two bytes; a byte-wise DP would report 2 here.
        assert_eq!(levenshtein_ascii_ci("café", "cafe"), 1);
    }

    // ── Tokenizer ────────────────────────────────────────────────────────────────

    #[test]
    fn tokenizer_splits_on_punctuation_and_whitespace() {
        assert_eq!(
            tokenize("the quick, brown\tfox!"),
            ["the", "quick", "brown", "fox"]
        );
    }

    #[test]
    fn tokenizer_folds_ascii_case() {
        assert_eq!(tokenize("Rust AND Go"), ["rust", "and", "go"]);
    }

    #[test]
    fn tokenizer_drops_empty_runs() {
        assert!(tokenize("   ---   ").is_empty());
        assert_eq!(tokenize("--a--b--"), ["a", "b"]);
    }

    #[test]
    fn tokenizer_keeps_digits_and_splits_underscores() {
        assert_eq!(tokenize("src_main2.rs"), ["src", "main2", "rs"]);
    }

    #[test]
    fn tokenizer_does_not_stem_or_drop_stopwords() {
        // The FTS analyzer would stem "running" and drop "the"; a filter must not.
        assert_eq!(tokenize("the running"), ["the", "running"]);
    }
}
