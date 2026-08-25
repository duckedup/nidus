//! Fallback fuzzy leg for `suggest` (nidus-972): a length-derived edit budget plus a
//! prefix-aware edit-distance DP run down the surface map's shared prefixes.

use std::collections::BTreeMap;

/// The longest fragment the fuzzy leg will consider. A typeahead fragment is a word being
/// typed, and no surface form this long has a useful completion, but `Analyzer::max_token_len`
/// defaults to `None`, so without this an untrusted `prefix` sizes the DP (nidus-972).
pub(crate) const MAX_FUZZY_FRAGMENT: usize = 64;

/// The edit budget a fragment of `chars` characters earns, `0` meaning do not run the leg at
/// all. Short fragments earn nothing, or a budget there ranks noise above the completion the
/// typist meant; over-long ones earn nothing so a caller cannot choose the DP's size.
pub(crate) fn budget(chars: usize) -> usize {
    match chars {
        0..=3 => 0,
        4..=7 => 1,
        c if c <= MAX_FUZZY_FRAGMENT => 2,
        _ => 0,
    }
}

/// `fragment`'s length in chars, counting at most `MAX_FUZZY_FRAGMENT + 1` of them. Counting
/// them all would itself walk an untrusted string, which is the cost this guard refuses; a
/// separate function so that early stop is assertable without timing anything.
fn capped_len(fragment: &str) -> usize {
    fragment.chars().take(MAX_FUZZY_FRAGMENT + 1).count()
}

/// `budget` for `fragment`, over its capped length.
pub(crate) fn budget_for(fragment: &str) -> usize {
    budget(capped_len(fragment))
}

/// Longest common prefix of two char slices.
fn lcp(a: &[char], b: &[char]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// One DP row extended by a single candidate character. `row[i]` is the edit distance from
/// `needle[..i]` to the candidate prefix the row stands for, so `row[0]` is that prefix's length.
fn extend(prev: &[usize], needle: &[char], c: char, prefix_len: usize) -> Vec<usize> {
    let mut cur = Vec::with_capacity(prev.len());
    cur.push(prefix_len);
    for i in 1..prev.len() {
        let sub = prev[i - 1] + usize::from(needle[i - 1] != c);
        cur.push(sub.min(prev[i] + 1).min(cur[i - 1] + 1));
    }
    cur
}

/// Surface forms within `max` **prefix** edits of `needle`, with that distance; returns the DP
/// rows computed, which is the narrowing (SPEC §7). A matched ancestor settles its subtree
/// rather than pruning it, or a found completion is dropped as deeper rows drift.
pub(crate) fn for_each_within<'a>(
    surface: &'a BTreeMap<String, String>,
    needle: &str,
    max: usize,
    mut on_match: impl FnMut(&'a String, &'a String, usize),
) -> usize {
    let needle: Vec<char> = needle.chars().collect();
    let n = needle.len();
    let mut rows: Vec<Vec<usize>> = vec![(0..=n).collect()];
    let mut prev_key: Vec<char> = Vec::new();
    // Nothing under this prefix length can match, so its keys are skipped without any DP.
    let mut pruned_at: Option<usize> = None;
    // An ancestor matched at this distance and no deeper row can improve on it, so its keys are
    // emitted without any DP either. The distance is final: deeper rows all exceed `max`.
    let mut settled_at: Option<(usize, usize)> = None;
    let mut rows_computed = 0usize;

    for (key, stem) in surface {
        let kc: Vec<char> = key.chars().collect();
        let shared = lcp(&prev_key, &kc);
        if let Some(p) = pruned_at {
            if shared >= p {
                continue;
            }
            pruned_at = None;
        }
        if let Some((f, distance)) = settled_at {
            if shared >= f {
                on_match(key, stem, distance);
                continue;
            }
            settled_at = None;
        }
        rows.truncate(shared + 1);
        // Prefix semantics: the running best is over every prefix walked so far, not the last.
        let mut best = rows.iter().map(|r| r[n]).min().unwrap_or(usize::MAX);
        for k in shared..kc.len() {
            let row = extend(&rows[k], &needle, kc[k], k + 1);
            rows_computed += 1;
            let row_min = row.iter().min().copied().unwrap_or(usize::MAX);
            best = best.min(row[n]);
            rows.push(row);
            if row_min > max {
                // No descendant can improve, so either they all match at `best` or none can.
                if best <= max {
                    settled_at = Some((k + 1, best));
                } else {
                    pruned_at = Some(k + 1);
                }
                break;
            }
        }
        prev_key = kc;
        if best <= max {
            on_match(key, stem, best);
        }
    }
    rows_computed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(words: &[&str]) -> BTreeMap<String, String> {
        words
            .iter()
            .map(|w| (w.to_string(), w.to_string()))
            .collect()
    }

    /// `(term, distance)` for every match, sorted by distance then term.
    fn within(words: &[&str], needle: &str, max: usize) -> Vec<(String, usize)> {
        let v = vocab(words);
        let mut got = Vec::new();
        for_each_within(&v, needle, max, |k, _, d| got.push((k.clone(), d)));
        got.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        got
    }

    #[test]
    fn a_correct_prefix_is_distance_zero() {
        assert_eq!(within(&["running"], "runn", 1), [("running".into(), 0)]);
    }

    #[test]
    fn a_single_typo_is_distance_one() {
        assert_eq!(within(&["running"], "runing", 1), [("running".into(), 1)]);
    }

    #[test]
    fn beyond_the_budget_does_not_match() {
        assert!(within(&["running"], "xyzzy", 1).is_empty());
    }

    #[test]
    fn a_needle_longer_than_the_candidate_still_matches_within_budget() {
        assert_eq!(within(&["running"], "runningg", 1), [("running".into(), 1)]);
    }

    #[test]
    fn distance_counts_chars_not_bytes() {
        // A byte DP reaches "caf\xC3" from "cafe" in one substitution, so the needle carries the
        // multi-byte char instead: "café" -> "cafe" is one char edit but two byte edits, and only
        // a char-based DP answers at max 1.
        assert_eq!(within(&["cafe"], "café", 1), [("cafe".into(), 1)]);
    }

    /// "runing" matches "running" by depth 7, then the rows drift past `max` on "shoes". Pruning
    /// there drops a completion already found. Reachable: the leg runs only when nothing has the
    /// needle as a prefix, so completing to a longer word is its ordinary case.
    #[test]
    fn a_typo_completes_to_a_much_longer_word() {
        // Two keys sharing the drifted prefix: the first is emitted on its own walk, but the
        // second is reached only if the subtree is marked settled rather than pruned.
        assert_eq!(
            within(&["runningshoes", "runningsocks"], "runing", 1),
            [("runningshoes".into(), 1), ("runningsocks".into(), 1)]
        );
    }

    #[test]
    fn budget_boundaries() {
        assert_eq!(budget(3), 0);
        assert_eq!(budget(4), 1);
        assert_eq!(budget(7), 1);
        assert_eq!(budget(8), 2);
    }

    /// An over-long fragment earns no budget, so the leg never allocates DP state sized to it.
    #[test]
    fn an_over_long_fragment_earns_no_budget() {
        assert_eq!(budget(MAX_FUZZY_FRAGMENT), 2);
        assert_eq!(budget(MAX_FUZZY_FRAGMENT + 1), 0);
    }

    /// The guard refuses without walking the fragment: counting every char of an untrusted
    /// `prefix` is itself the cost being refused. Asserted structurally rather than by a clock,
    /// which measures the machine (and, under Miri, the interpreter) instead of the code.
    #[test]
    fn an_over_long_fragment_is_refused_without_being_walked() {
        let long = "a".repeat(10_000);
        assert_eq!(capped_len(&long), MAX_FUZZY_FRAGMENT + 1);
        assert_eq!(budget_for(&long), 0);
    }

    /// The narrowing SPEC §7 claims, as a number rather than an argument. Without subtree
    /// pruning every character of every word costs a row (36 here); with it, the four `zz*`
    /// words collapse to the two rows that prove "zz" is out of budget.
    #[test]
    fn a_subtree_out_of_budget_is_skipped_whole() {
        let words = [
            "running", "runs", "zzaaaaa", "zzbbbbb", "zzccccc", "zzddddd",
        ];
        let total_chars: usize = words.iter().map(|w| w.chars().count()).sum();
        assert_eq!(total_chars, 39);
        let v = vocab(&words);
        let rows = for_each_within(&v, "runn", 1, |_, _, _| {});
        assert!(
            rows < total_chars / 2,
            "pruning computed {rows} rows of a possible {total_chars}"
        );
    }
}
