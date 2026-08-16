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

/// Runs of alphanumerics, borrowed and **unfolded** — folding is the comparison's job
/// (`eq_ignore_ascii_case`), so a scan allocates nothing per row. Simpler than the FTS
/// analyzer on purpose: no stemming, no stopwords, since a filter term is present or not.
pub(crate) fn tokens(text: &str) -> impl Iterator<Item = &str> + Clone {
    text.split(is_separator).filter(is_nonempty)
}

/// Named rather than inline so the `tokens` iterator stays `Clone` — the sequence walk below
/// restarts from cloned cursors.
fn is_separator(c: char) -> bool {
    !c.is_alphanumeric()
}

fn is_nonempty(token: &&str) -> bool {
    !token.is_empty()
}

/// True iff `have` opens with every token of `want`. An empty `want` is vacuously true,
/// which is `ContainsTokenSequence`'s empty-query identity.
fn starts_with_tokens<'h, 'w>(
    mut have: impl Iterator<Item = &'h str>,
    want: impl Iterator<Item = &'w str>,
) -> bool {
    want.into_iter()
        .all(|w| have.next().is_some_and(|h| h.eq_ignore_ascii_case(w)))
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
    any_text(value, |text| {
        tokens(query).all(|w| tokens(text).any(|h| h.eq_ignore_ascii_case(w)))
    })
}

/// [`crate::Predicate::ContainsAnyToken`]. An empty query matches nothing, the identity
/// `Any([])` and an empty `In` set already take.
pub(super) fn contains_any_token(value: Option<&Value>, query: &str) -> bool {
    any_text(value, |text| {
        tokens(query).any(|w| tokens(text).any(|h| h.eq_ignore_ascii_case(w)))
    })
}

/// [`crate::Predicate::ContainsTokenSequence`]: the query's tokens as a consecutive, in-order
/// run, never spanning two `List` elements. Restarts at every token — a single cursor reset on
/// mismatch would miss `"a a b"` inside `"a a a b"`.
pub(super) fn contains_token_sequence(value: Option<&Value>, query: &str) -> bool {
    any_text(value, |text| {
        let mut cursor = tokens(text);
        loop {
            if starts_with_tokens(cursor.clone(), tokens(query)) {
                return true;
            }
            if cursor.next().is_none() {
                return false;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{contains_all_tokens, contains_token_sequence, levenshtein_ascii_ci, tokens};
    use crate::model::Value;

    fn toks(text: &str) -> Vec<&str> {
        tokens(text).collect()
    }

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
            toks("the quick, brown\tfox!"),
            ["the", "quick", "brown", "fox"]
        );
    }

    #[test]
    fn tokenizer_drops_empty_runs() {
        assert!(toks("   ---   ").is_empty());
        assert_eq!(toks("--a--b--"), ["a", "b"]);
    }

    #[test]
    fn tokenizer_keeps_digits_and_splits_underscores() {
        assert_eq!(toks("src_main2.rs"), ["src", "main2", "rs"]);
    }

    #[test]
    fn tokenizer_does_not_stem_or_drop_stopwords() {
        // The FTS analyzer would stem "running" and drop "the"; a filter must not.
        assert_eq!(toks("the running"), ["the", "running"]);
    }

    #[test]
    fn tokenizer_borrows_the_text_verbatim_and_folds_nothing() {
        // Folding moved to the comparison; the tokens themselves are slices of `text`.
        assert_eq!(toks("Rust AND Go"), ["Rust", "AND", "Go"]);
    }

    // ── Case folding, now the comparison's job ───────────────────────────────────

    #[test]
    fn token_matching_folds_ascii_case_on_both_sides() {
        let v = Value::Str("Rust AND Go".into());
        assert!(contains_all_tokens(Some(&v), "rust go"));
        assert!(contains_all_tokens(Some(&v), "RUST Go"));
        assert!(contains_token_sequence(Some(&v), "rust AND go"));
    }

    #[test]
    fn token_matching_does_not_fold_non_ascii_case() {
        // Mirrors `ascii_case_is_folded_but_non_ascii_is_not`: a locale-dependent fold
        // would mean different things on different machines.
        let v = Value::Str("café".into());
        assert!(contains_all_tokens(Some(&v), "café"));
        assert!(!contains_all_tokens(Some(&v), "CAFÉ"));
    }

    // ── Sequence restart ─────────────────────────────────────────────────────────

    #[test]
    fn a_sequence_is_retried_from_every_token_not_just_the_first() {
        // A single cursor reset to the start of `want` on mismatch fails this: it consumes
        // both leading "a"s, mismatches "b" against the third "a", and never retries.
        let v = Value::Str("a a a b".into());
        assert!(contains_token_sequence(Some(&v), "a a b"));
    }
}
