//! Highlighted fragments: excerpts of a field's **original** text with the byte ranges a
//! query term matched. The analyzer stems and folds, so a matched term is generally not a
//! substring of the text — the offsets come from [`analyze_spans`], never from a substring search.

use std::collections::HashSet;

use super::analyzer::{Analyzer, analyze_spans};
use crate::annotate::{Fragment, HighlightOpts};

/// Fragments of `text` around the terms matching `query_terms` (already analyzed with the
/// same `cfg`). Each fragment's `spans` are byte ranges into that fragment's own `text`.
/// Empty when nothing matched.
pub(crate) fn fragments(
    text: &str,
    cfg: Analyzer,
    query_terms: &[String],
    opts: &HighlightOpts,
) -> Vec<Fragment> {
    let wanted: HashSet<&str> = query_terms.iter().map(String::as_str).collect();
    let matches: Vec<(usize, usize)> = analyze_spans(text, cfg)
        .into_iter()
        .filter(|t| wanted.contains(t.term.as_str()))
        .map(|t| (t.start, t.end))
        .collect();

    // Clamped, not rejected: a zero here is a caller typo, and an empty fragment list would
    // read as "nothing matched" rather than "you asked for nothing".
    let max_fragments = opts.max_fragments.max(1);
    let budget = opts.fragment_chars.max(1);

    let mut out = Vec::new();
    let mut i = 0;
    while i < matches.len() && out.len() < max_fragments {
        let (first_start, first_end) = matches[i];
        // Half the budget of leading context, then fill forward; never cut so short that the
        // match that opened the window falls outside it.
        let start = snap_start(text, back_chars(text, first_start, budget / 2), first_start);
        let end = fwd_chars(text, start, budget).max(first_end);
        let mut spans = Vec::new();
        let mut last_end = first_end;
        while i < matches.len() && matches[i].1 <= end {
            spans.push((matches[i].0 - start, matches[i].1 - start));
            last_end = matches[i].1;
            i += 1;
        }
        let end = snap_end(text, end, last_end);
        out.push(Fragment {
            text: text[start..end].to_string(),
            spans,
        });
    }
    out
}

/// The byte index `n` chars before `from`, clamped to the start of `text`.
fn back_chars(text: &str, from: usize, n: usize) -> usize {
    text[..from]
        .char_indices()
        .rev()
        .take(n)
        .last()
        .map_or(from, |(i, _)| i)
}

/// The byte index `n` chars after `from`, clamped to the end of `text`.
fn fwd_chars(text: &str, from: usize, n: usize) -> usize {
    text[from..]
        .char_indices()
        .nth(n)
        .map_or(text.len(), |(i, _)| from + i)
}

/// Move `start` forward past the first token boundary so a fragment does not open mid-word,
/// never past `limit` (the match that opened the window).
fn snap_start(text: &str, start: usize, limit: usize) -> usize {
    if start == 0 {
        return 0;
    }
    match text[start..limit]
        .char_indices()
        .find(|(_, c)| !c.is_alphanumeric())
    {
        Some((i, c)) => start + i + c.len_utf8(),
        None => start,
    }
}

/// Move `end` back to the last token boundary so a fragment does not close mid-word, never
/// before `limit` (the end of the last match inside it).
fn snap_end(text: &str, end: usize, limit: usize) -> usize {
    if end == text.len() {
        return end;
    }
    match text[limit..end]
        .char_indices()
        .rev()
        .find(|(_, c)| !c.is_alphanumeric())
    {
        Some((i, _)) => limit + i,
        None => end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fts::analyze;

    /// Highlight `text` for `query`, both analyzed with `cfg`.
    fn hl(text: &str, query: &str, opts: HighlightOpts) -> Vec<Fragment> {
        let cfg = Analyzer::default();
        fragments(text, cfg, &analyze(query, cfg), &opts)
    }

    /// The literal substrings a fragment's spans point at — the whole point of the module.
    fn marked(f: &Fragment) -> Vec<&str> {
        f.spans.iter().map(|&(s, e)| &f.text[s..e]).collect()
    }

    #[test]
    fn a_stemmed_query_highlights_the_surface_form() {
        // "running" (query) stems to "run"; the document spells it "run". The span must
        // cover the document's spelling, which a substring search for "running" never finds.
        let f = hl("we run tests daily", "running", HighlightOpts::default());
        assert_eq!(marked(&f[0]), vec!["run"]);
        assert_eq!(f[0].text, "we run tests daily");
    }

    #[test]
    fn a_surface_query_highlights_the_stemmed_document() {
        let f = hl(
            "the runner was running fast",
            "run",
            HighlightOpts::default(),
        );
        assert_eq!(marked(&f[0]), vec!["running"]);
    }

    #[test]
    fn spans_land_on_multibyte_text_at_char_boundaries() {
        let text = "el café estaba abierto";
        let cfg = Analyzer::default().ascii_folding(true);
        let f = fragments(text, cfg, &analyze("cafe", cfg), &HighlightOpts::default());
        assert_eq!(marked(&f[0]), vec!["café"]);
    }

    #[test]
    fn several_matches_in_one_window_share_a_fragment() {
        let f = hl("alpha beta alpha gamma", "alpha", HighlightOpts::default());
        assert_eq!(f.len(), 1);
        assert_eq!(marked(&f[0]), vec!["alpha", "alpha"]);
    }

    #[test]
    fn a_short_budget_splits_distant_matches_into_separate_fragments() {
        let text = "needle at the very start, then a long stretch of unrelated padding \
                    words that carry no signal whatsoever, and finally another needle";
        let f = hl(
            text,
            "needle",
            HighlightOpts::default().max_fragments(2).fragment_chars(30),
        );
        assert_eq!(f.len(), 2, "{f:?}");
        assert_eq!(marked(&f[0]), vec!["needle"]);
        assert_eq!(marked(&f[1]), vec!["needle"]);
        // Each fragment is an excerpt, not the whole field.
        assert!(f.iter().all(|frag| frag.text.len() < text.len()));
    }

    #[test]
    fn max_fragments_caps_the_output() {
        let text = "one needle two needle three needle four needle five needle six needle";
        let f = hl(
            text,
            "needle",
            HighlightOpts::default().max_fragments(2).fragment_chars(12),
        );
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn a_fragment_does_not_open_or_close_mid_word() {
        let text = "extraordinarily complicated preamble before the needle appears here";
        let f = hl(text, "needle", HighlightOpts::default().fragment_chars(24));
        let frag = &f[0];
        assert!(
            !text.starts_with(&frag.text),
            "should be an interior window"
        );
        assert!(text.contains(&frag.text));
        // Neither edge splits a word: the char before/after the excerpt is a boundary.
        let start = text.find(&frag.text).unwrap();
        assert!(text[..start].ends_with(|c: char| !c.is_alphanumeric()));
        let after = start + frag.text.len();
        assert!(text[after..].starts_with(|c: char| !c.is_alphanumeric()));
    }

    #[test]
    fn no_match_and_degenerate_options_return_nothing_surprising() {
        assert!(hl("alpha beta", "gamma", HighlightOpts::default()).is_empty());
        assert!(hl("", "alpha", HighlightOpts::default()).is_empty());
        assert!(hl("alpha", "the and of", HighlightOpts::default()).is_empty());
        // Zero budgets clamp rather than swallow the match.
        let f = hl(
            "alpha",
            "alpha",
            HighlightOpts {
                max_fragments: 0,
                fragment_chars: 0,
            },
        );
        assert_eq!(marked(&f[0]), vec!["alpha"]);
    }
}
