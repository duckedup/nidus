//! Pure ranking logic for the hosted-reranker provider stage (nidus-4ss). No `cfg`, no
//! network, no I/O: this file is covered by `just miri` in every feature combination.

use std::collections::BTreeMap;

use crate::model::{Hit, Value};

/// Default candidate over-fetch multiple: rank `(offset + top_k) * OVERSCAN` deep, then let
/// the reranker pick. Distinct from `AnnConfig::overscan` and `Quantization::rescore` —
/// this one is the hosted-reranker window (nidus-4ss).
pub const DEFAULT_OVERSCAN: usize = 4;

/// Largest accepted [`RerankOpts::overscan`]. A request surface rejects a bigger one rather
/// than clamping, so an absurd value is a caller error and not a silent cap.
pub const MAX_OVERSCAN: usize = 64;

/// Ceiling on the over-fetched window, so no `overscan` can push a search past the depth a
/// plain query is capped at. Mirrors `server::dto::MAX_TOP_K`, which a test pins.
pub const MAX_RERANK_DEPTH: usize = 10_000;

/// The attr a candidate's text is read from when the caller names none. The same key
/// `remember` stamps, so a memory store reranks with no configuration.
pub const DEFAULT_TEXT_FIELD: &str = "nidus.text";

/// Per-request knobs for the reranker provider stage. Deliberately not a field on
/// `SearchOpts`/`HybridOpts`: the sync store would provably ignore such a field (SPEC §9).
#[derive(Clone, Debug, PartialEq)]
pub struct RerankOpts {
    /// Which attr carries the candidate text. `None` means [`DEFAULT_TEXT_FIELD`].
    pub text_field: Option<String>,
    /// Candidate over-fetch multiple. `0` and `1` both mean "no over-fetch".
    pub overscan: usize,
    /// Per-request model override; `None` uses the reranker's configured model.
    pub model: Option<String>,
    /// The text scored against each candidate. Required on the raw-vector `search` path,
    /// which has no query text of its own; elsewhere `None` means "the surface's own query".
    pub query: Option<String>,
}

impl Default for RerankOpts {
    fn default() -> Self {
        Self {
            text_field: None,
            overscan: DEFAULT_OVERSCAN,
            model: None,
            query: None,
        }
    }
}

/// How deep to rank before handing candidates to the reranker: `(offset + top_k) *
/// max(overscan, 1)`, saturating and then capped at [`MAX_RERANK_DEPTH`] — the cap is what
/// stops an over-fetch from outgrowing the ceiling a plain query is held to.
pub fn depth(offset: usize, top_k: usize, overscan: usize) -> usize {
    offset
        .saturating_add(top_k)
        .saturating_mul(overscan.max(1))
        .min(MAX_RERANK_DEPTH)
}

/// The candidate's text at `field`, or `None` if absent or not a [`Value::Str`] — a
/// `List`/`Int`/`Null` there is not text and must never be stringified.
pub fn text_of<'a>(attrs: &'a BTreeMap<String, Value>, field: &str) -> Option<&'a str> {
    match attrs.get(field) {
        Some(Value::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// Replace each reranked hit's score with the provider's and resort: reranked hits rank by
/// that score (desc, `(collection, id)` asc on ties — `search::TopK`'s total order);
/// an unscored hit keeps its metric score but sorts below every reranked one regardless.
pub fn merge(hits: Vec<Hit>, scored: &[(usize, f32)]) -> Vec<Hit> {
    let mut reranked = vec![false; hits.len()];
    let mut hits = hits;
    for &(idx, score) in scored {
        if let Some(h) = hits.get_mut(idx) {
            h.score = score;
            reranked[idx] = true;
        }
    }
    let mut pairs: Vec<(bool, Hit)> = hits
        .into_iter()
        .zip(reranked)
        .map(|(h, r)| (r, h))
        .collect();
    pairs.sort_by(|(ra, a), (rb, b)| {
        rb.cmp(ra)
            .then_with(|| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                (a.collection.as_str(), a.id.as_str()).cmp(&(b.collection.as_str(), b.id.as_str()))
            })
    });
    pairs.into_iter().map(|(_, h)| h).collect()
}

/// The tail cap a sync search path would normally apply, run here instead: the reranked
/// window stays deeper than `top_k` until this point.
pub fn page(hits: Vec<Hit>, offset: usize, top_k: usize) -> Vec<Hit> {
    hits.into_iter().skip(offset).take(top_k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn hit(collection: &str, id: &str, score: f32) -> Hit {
        Hit::new(collection, id, score, BTreeMap::new())
    }

    fn ids(hits: &[Hit]) -> Vec<&str> {
        hits.iter().map(|h| h.id.as_str()).collect()
    }

    // ── depth ──────────────────────────────────────────────────────────────

    #[test]
    fn depth_overscan_zero_and_one_are_equivalent() {
        assert_eq!(depth(5, 10, 0), 15);
        assert_eq!(depth(5, 10, 1), 15);
    }

    #[test]
    fn depth_multiplies_by_overscan() {
        assert_eq!(depth(0, 10, 4), 40);
        assert_eq!(depth(5, 10, 4), 60);
    }

    #[test]
    fn depth_saturates_then_caps_rather_than_panicking() {
        assert_eq!(depth(usize::MAX, usize::MAX, 4), MAX_RERANK_DEPTH);
        assert_eq!(depth(usize::MAX, 1, 1), MAX_RERANK_DEPTH);
        assert_eq!(depth(0, MAX_RERANK_DEPTH, MAX_OVERSCAN), MAX_RERANK_DEPTH);
    }

    // ── text_of ────────────────────────────────────────────────────────────

    #[test]
    fn text_of_reads_only_str_values() {
        let mut attrs = BTreeMap::new();
        attrs.insert("body".to_string(), Value::Str("hello".to_string()));
        attrs.insert("n".to_string(), Value::Int(3));
        attrs.insert("tags".to_string(), Value::List(vec!["a".to_string()]));
        attrs.insert("missing_kind".to_string(), Value::Null);

        assert_eq!(text_of(&attrs, "body"), Some("hello"));
        assert_eq!(text_of(&attrs, "n"), None);
        assert_eq!(text_of(&attrs, "tags"), None);
        assert_eq!(text_of(&attrs, "missing_kind"), None);
        assert_eq!(text_of(&attrs, "absent"), None);
    }

    // ── merge ──────────────────────────────────────────────────────────────

    #[test]
    fn merge_with_no_scored_candidates_is_byte_identical_to_input() {
        let hits = vec![hit("c", "a", 0.9), hit("c", "b", 0.5), hit("c", "c", 0.1)];
        let out = merge(hits, &[]);
        assert_eq!(ids(&out), vec!["a", "b", "c"]);
    }

    #[test]
    fn reranked_hits_sort_above_unranked_regardless_of_metric_score() {
        // "loser" has the highest metric score but no text, so it is never scored by the
        // provider; "winner" has a low metric score but the provider ranks it first.
        let hits = vec![hit("c", "loser", 0.99), hit("c", "winner", 0.01)];
        let out = merge(hits, &[(1, 5.0)]);
        assert_eq!(ids(&out), vec!["winner", "loser"]);
    }

    #[test]
    fn ties_break_on_collection_then_id_ascending() {
        let hits = vec![hit("c", "b", 0.1), hit("c", "a", 0.1)];
        let out = merge(hits, &[(0, 1.0), (1, 1.0)]);
        assert_eq!(ids(&out), vec!["a", "b"]);
    }

    #[test]
    fn scored_hits_sort_by_provider_score_descending() {
        let hits = vec![hit("c", "a", 0.9), hit("c", "b", 0.5)];
        let out = merge(hits, &[(0, 1.0), (1, 9.0)]);
        assert_eq!(ids(&out), vec!["b", "a"]);
        assert_eq!(out[0].score, 9.0);
    }

    // ── page ───────────────────────────────────────────────────────────────

    #[test]
    fn page_applies_offset_and_top_k() {
        let hits = vec![hit("c", "a", 1.0), hit("c", "b", 1.0), hit("c", "c", 1.0)];
        let out = page(hits, 1, 1);
        assert_eq!(ids(&out), vec!["b"]);
    }

    #[test]
    fn page_offset_past_end_is_empty() {
        let hits = vec![hit("c", "a", 1.0)];
        assert!(page(hits, 5, 10).is_empty());
    }

    #[cfg(feature = "memory")]
    #[test]
    fn default_text_field_is_the_key_remember_stamps() {
        assert_eq!(DEFAULT_TEXT_FIELD, crate::memory::META_TEXT);
    }
}
