//! Character-trigram postings: the index behind `Fuzzy` (via the edit bound below) and
//! `Regex` (via required-literal extraction in `literal.rs`).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// One trigram, ASCII-case-folded. `char`-based rather than byte-based to match
/// `levenshtein_ascii_ci`'s char DP, and folded the same way it folds: ASCII only, since a
/// locale-dependent fold means different things on different machines.
pub(crate) type Trigram = [char; 3];

/// Every trigram of `text`, in order, with duplicates. Text shorter than 3 chars has none,
/// which is why a short needle can never be narrowed and must fall back to a full scan.
pub(crate) fn trigrams(text: &str) -> Vec<Trigram> {
    let chars: Vec<char> = text.chars().map(|c| c.to_ascii_lowercase()).collect();
    chars.windows(3).map(|w| [w[0], w[1], w[2]]).collect()
}

/// The distinct trigrams of `text`, sorted — the form both indexing and querying want.
pub(crate) fn distinct_trigrams(text: &str) -> Vec<Trigram> {
    let mut t = trigrams(text);
    t.sort_unstable();
    t.dedup();
    t
}

/// trigram → docnums, ascending.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct TrigramPostings {
    postings: HashMap<Trigram, Vec<u32>>,
}

impl TrigramPostings {
    pub(crate) fn index(&mut self, docnum: u32, text: &str) {
        for t in distinct_trigrams(text) {
            let list = self.postings.entry(t).or_default();
            if list.last() != Some(&docnum) {
                list.push(docnum);
            }
        }
    }

    pub(crate) fn get(&self, t: &Trigram) -> &[u32] {
        self.postings.get(t).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.postings
            .values()
            .map(|v| size_of::<Trigram>() + v.len() * size_of::<u32>() + size_of::<Vec<u32>>())
            .sum()
    }

    /// Docnums sharing at least `threshold` of `wanted`, ascending. `threshold == 0` is
    /// rejected by the caller, never answered here — it would mean "every doc".
    pub(crate) fn at_least(&self, wanted: &[Trigram], threshold: usize) -> Vec<u32> {
        let mut hits: HashMap<u32, usize> = HashMap::new();
        for t in wanted {
            for d in self.get(t) {
                *hits.entry(*d).or_insert(0) += 1;
            }
        }
        let mut out: Vec<u32> = hits
            .into_iter()
            .filter(|(_, n)| *n >= threshold)
            .map(|(d, _)| d)
            .collect();
        out.sort_unstable();
        out
    }
}

/// The minimum trigrams a string within `max_edits` of `needle` must share with it.
///
/// A single edit (substitution, insertion or deletion) destroys **at most 3** trigrams,
/// since a character participates in at most 3 windows. So `|trigrams(needle)| - 3*d` is a
/// sound lower bound. `None` means the bound is vacuous (`<= 0`) and no narrowing is
/// possible — the caller must scan everything rather than narrow to nothing.
pub(crate) fn fuzzy_threshold(needle: &str, max_edits: usize) -> Option<usize> {
    let n = distinct_trigrams(needle).len();
    n.checked_sub(3usize.saturating_mul(max_edits))
        .filter(|t| *t > 0)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{TrigramPostings, distinct_trigrams, fuzzy_threshold, trigrams};
    use crate::filter::levenshtein_ascii_ci;

    #[test]
    fn trigrams_are_char_windows_and_ascii_folded() {
        assert_eq!(trigrams("AbCd"), [['a', 'b', 'c'], ['b', 'c', 'd']]);
    }

    #[test]
    fn text_shorter_than_three_chars_has_no_trigrams() {
        assert!(trigrams("").is_empty());
        assert!(trigrams("ab").is_empty());
        assert_eq!(trigrams("abc").len(), 1);
    }

    #[test]
    fn non_ascii_case_is_not_folded() {
        // Mirrors `levenshtein_ascii_ci`: folding É would mean different things per locale.
        assert_ne!(trigrams("café"), trigrams("cafÉ"));
    }

    #[test]
    fn trigrams_count_chars_not_bytes() {
        // "é" is two bytes; a byte-window index would produce a different count.
        assert_eq!(trigrams("café").len(), 2);
    }

    #[test]
    fn a_vacuous_threshold_is_none_not_zero() {
        assert_eq!(fuzzy_threshold("abcdef", 1), Some(1)); // 4 trigrams - 3
        assert_eq!(fuzzy_threshold("abcde", 1), None); // 3 trigrams - 3 = 0
        assert_eq!(fuzzy_threshold("abcd", 1), None); // 2 trigrams, saturates
        assert_eq!(fuzzy_threshold("ab", 0), None); // no trigrams at all
        assert_eq!(fuzzy_threshold("abcd", 0), Some(2));
    }

    #[test]
    fn at_least_counts_shared_trigrams() {
        let mut p = TrigramPostings::default();
        p.index(0, "hello");
        p.index(1, "help");
        let want = distinct_trigrams("hello");
        assert_eq!(p.at_least(&want, want.len()), [0]);
        assert_eq!(p.at_least(&want, 1), [0, 1]);
    }

    /// Every string of length `lens` over `alphabet`.
    fn words_of_len(alphabet: &[char], lens: std::ops::RangeInclusive<usize>) -> Vec<String> {
        let mut out = Vec::new();
        let mut level: Vec<String> = vec![String::new()];
        for len in 1..=*lens.end() {
            level = level
                .iter()
                .flat_map(|w| alphabet.iter().map(move |c| format!("{w}{c}")))
                .collect();
            if lens.contains(&len) {
                out.extend(level.iter().cloned());
            }
        }
        out
    }

    /// Every string reachable from `word` in at most `d` single edits. Generating the
    /// neighbourhood directly is what makes this test bite: enumerating a whole small
    /// alphabet instead only reaches needles too short for the bound to be non-vacuous, so
    /// the `3` in `3*d` was never actually exercised (a 2*d bound passed that version).
    fn edit_neighborhood(word: &str, d: usize, alphabet: &[char]) -> BTreeSet<String> {
        let mut frontier = BTreeSet::from([word.to_string()]);
        let mut all = frontier.clone();
        for _ in 0..d {
            let mut next = BTreeSet::new();
            for w in &frontier {
                let chars: Vec<char> = w.chars().collect();
                for i in 0..chars.len() {
                    let mut c = chars.clone();
                    c.remove(i);
                    next.insert(c.into_iter().collect::<String>());
                }
                for i in 0..chars.len() {
                    for &a in alphabet.iter().filter(|a| **a != chars[i]) {
                        let mut c = chars.clone();
                        c[i] = a;
                        next.insert(c.into_iter().collect::<String>());
                    }
                }
                for i in 0..=chars.len() {
                    for &a in alphabet {
                        let mut c = chars.clone();
                        c.insert(i, a);
                        next.insert(c.into_iter().collect::<String>());
                    }
                }
            }
            all.extend(next.iter().cloned());
            frontier = next;
        }
        all
    }

    /// The inequality every `Fuzzy` narrowing rests on. If it is wrong in the tight
    /// direction, real matches are silently dropped, so this walks the true edit
    /// neighbourhood of each needle rather than sampling a corpus.
    #[cfg_attr(miri, ignore)] // runtime cost: ~10^5 neighbourhood strings
    #[test]
    fn the_edit_bound_never_drops_a_true_match() {
        let alphabet = ['a', 'b', 'c'];
        let mut checked = 0usize;

        let mut cases: Vec<(String, usize)> = Vec::new();
        for w in words_of_len(&['a', 'b'], 6..=8) {
            cases.push((w, 1));
        }
        for w in words_of_len(&['a', 'b'], 9..=9) {
            cases.push((w, 2));
        }

        for (needle, d) in cases {
            let Some(threshold) = fuzzy_threshold(&needle, d) else {
                continue;
            };
            let want = distinct_trigrams(&needle);
            for text in edit_neighborhood(&needle, d, &alphabet) {
                debug_assert!(levenshtein_ascii_ci(&text, &needle) <= d);
                let have = distinct_trigrams(&text);
                let shared = want.iter().filter(|t| have.contains(t)).count();
                assert!(
                    shared >= threshold,
                    "needle={needle:?} text={text:?} d={d} shared={shared} threshold={threshold}"
                );
                checked += 1;
            }
        }

        // Guard against the whole test going vacuous again if the bound or the needle set
        // changes: a threshold that is always `None` would skip every case silently.
        assert!(checked > 10_000, "only {checked} pairs checked");
    }
}
