//! Raw-token postings: the index behind `ContainsAllTokens`, `ContainsAnyToken` and
//! `ContainsTokenSequence`. Terms come from the *same* tokenizer the predicates use
//! (`filter::text::tokens`), unstemmed — that identity is the whole point.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::filter::tokens as split_tokens;

/// term → docnums, ascending. The ASCII fold happens here, once per document at write
/// time, rather than once per row per query — which is where the predicates deliberately
/// no longer do it.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct TokenPostings {
    postings: HashMap<String, Vec<u32>>,
}

impl TokenPostings {
    /// Add `text`'s tokens under `docnum`. Docnums arrive ascending, so each posting list
    /// stays sorted without a sort; a repeated token in one doc is recorded once.
    pub(crate) fn index(&mut self, docnum: u32, text: &str) {
        for t in split_tokens(text) {
            let list = self.postings.entry(t.to_ascii_lowercase()).or_default();
            if list.last() != Some(&docnum) {
                list.push(docnum);
            }
        }
    }

    /// Docnums carrying `token`, or an empty slice if the term is unknown.
    pub(crate) fn get(&self, token: &str) -> &[u32] {
        self.postings
            .get(&token.to_ascii_lowercase())
            .map_or(&[], Vec::as_slice)
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.postings
            .iter()
            .map(|(k, v)| k.len() + v.len() * size_of::<u32>() + size_of::<Vec<u32>>())
            .sum()
    }
}

/// Intersect ascending docnum lists. An empty input yields `None` — "nothing to narrow
/// with" — never an empty result, which would claim no document matches.
pub(crate) fn intersect(lists: &[&[u32]]) -> Option<Vec<u32>> {
    let (first, rest) = lists.split_first()?;
    let mut acc: Vec<u32> = first.to_vec();
    for list in rest {
        acc.retain(|d| list.binary_search(d).is_ok());
        if acc.is_empty() {
            break;
        }
    }
    Some(acc)
}

/// Union ascending docnum lists into one sorted, deduplicated list.
pub(crate) fn union(lists: &[&[u32]]) -> Vec<u32> {
    let mut acc: Vec<u32> = lists.iter().flat_map(|l| l.iter().copied()).collect();
    acc.sort_unstable();
    acc.dedup();
    acc
}

#[cfg(test)]
mod tests {
    use super::{TokenPostings, intersect, union};

    fn built() -> TokenPostings {
        let mut p = TokenPostings::default();
        p.index(0, "the quick brown fox");
        p.index(1, "the lazy dog");
        p.index(2, "Quick Brown Bear");
        p
    }

    #[test]
    fn postings_are_ascii_folded_at_index_time() {
        let p = built();
        assert_eq!(p.get("quick"), [0, 2]);
        assert_eq!(p.get("QUICK"), [0, 2]);
    }

    #[test]
    fn a_repeated_token_records_one_docnum() {
        let mut p = TokenPostings::default();
        p.index(7, "a a a b a");
        assert_eq!(p.get("a"), [7]);
    }

    #[test]
    fn an_unknown_term_has_no_postings() {
        let p = built();
        assert!(p.get("absent").is_empty());
    }

    #[test]
    fn intersect_of_nothing_is_none_not_empty() {
        // The distinction is load-bearing: `None` means "cannot narrow", an empty Vec
        // would claim no document matches.
        assert_eq!(intersect(&[]), None);
        assert_eq!(intersect(&[&[1, 2][..]]), Some(vec![1, 2]));
    }

    #[test]
    fn intersect_keeps_only_common_docnums() {
        assert_eq!(
            intersect(&[&[1, 2, 3][..], &[2, 3, 4][..], &[2, 3, 9][..]]),
            Some(vec![2, 3])
        );
        assert_eq!(intersect(&[&[1][..], &[2][..]]), Some(vec![]));
    }

    #[test]
    fn union_sorts_and_dedups() {
        assert_eq!(union(&[&[3, 1][..], &[1, 2][..]]), vec![1, 2, 3]);
        assert!(union(&[]).is_empty());
    }
}
