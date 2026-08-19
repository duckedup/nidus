//! The pure, unconditional half of the rerank stage (SPEC §7): the over-fetch window, text
//! extraction, and score substitution + re-sort. This is the hosted cross-encoder seam
//! (`crate::rerank`), never the quantized int8→f32 rescore in [`super::quant`]. Unconditional
//! (no `#[cfg]`) so `just ci`/`just miri` cover it with no provider feature compiled in; the
//! async provider call lives at the edge in `crate::rerank`.

// Only the feature-gated async edge (`crate::rerank`) calls these; with `rerank` off the
// module's own tests are the sole consumer, which dead_code does not count. Same reason
// `crate::http` carries a module-wide allow.
#![cfg_attr(not(feature = "rerank"), allow(dead_code))]

use std::cmp::Ordering;

use super::read::depth;
use crate::model::{Hit, Projection, SearchOpts, Value};

/// How deep to rank before reranking: one page (`read::depth`) times the overscan, so a
/// `top_k=N` query sees `N * overscan` candidates to rerank over.
pub(crate) fn rerank_depth(opts: &SearchOpts) -> usize {
    let overscan = opts.rerank.as_ref().map(|r| r.overscan.max(1)).unwrap_or(1);
    depth(opts).saturating_mul(overscan)
}

/// The pre-rerank fetch's opts: [`rerank_depth`] deep, unpaginated, `limit_per` deferred, and
/// the text attr force-included — `hits_from_topk` projects before a `Hit` exists, so
/// inheriting a projection that drops it would silently skip the rerank (nidus-d6z).
pub(crate) fn widened_opts(opts: &SearchOpts) -> (SearchOpts, bool) {
    let text_attr = opts
        .rerank
        .as_ref()
        .map(|r| r.text_attr.as_str())
        .unwrap_or(crate::model::META_TEXT);
    let already_kept = projection_carries(&opts.projection, text_attr);
    let projection = if already_kept {
        opts.projection.clone()
    } else {
        force_include(&opts.projection, text_attr)
    };
    let widened = SearchOpts {
        top_k: rerank_depth(opts),
        offset: 0,
        limit_per: None,
        projection,
        ..opts.clone()
    };
    (widened, already_kept)
}

/// Whether `p` would already carry `field`.
fn projection_carries(p: &Projection, field: &str) -> bool {
    match p {
        Projection::All => true,
        Projection::Include(keys) => keys.iter().any(|k| k == field),
        Projection::Exclude(keys) => !keys.iter().any(|k| k == field),
    }
}

/// `p` widened to carry `field`. Only called when it does not already.
fn force_include(p: &Projection, field: &str) -> Projection {
    match p {
        Projection::All => Projection::All,
        Projection::Include(keys) => {
            let mut keys = keys.clone();
            keys.push(field.to_string());
            Projection::Include(keys)
        }
        Projection::Exclude(keys) => {
            Projection::Exclude(keys.iter().filter(|k| *k != field).cloned().collect())
        }
    }
}

/// Undo [`widened_opts`]'s force-include, so the response honours the projection the caller
/// actually asked for. A no-op when their projection already carried the attr.
pub(crate) fn retrim(hits: &mut [Hit], opts: &SearchOpts, already_kept: bool) {
    if already_kept {
        return;
    }
    let Some(r) = opts.rerank.as_ref() else {
        return;
    };
    for h in hits.iter_mut() {
        h.attrs.remove(&r.text_attr);
    }
}

/// Split `hits` into the candidate texts (non-empty `Value::Str` under `text_attr`, in order)
/// and the original indices of the hits that had none — those are passed through unranked.
pub(crate) fn candidate_texts<'a>(hits: &'a [Hit], text_attr: &str) -> (Vec<&'a str>, Vec<usize>) {
    let mut texts = Vec::new();
    let mut passthrough = Vec::new();
    for (i, hit) in hits.iter().enumerate() {
        match hit.attrs.get(text_attr) {
            Some(Value::Str(s)) if !s.is_empty() => texts.push(s.as_str()),
            _ => passthrough.push(i),
        }
    }
    (texts, passthrough)
}

/// Total order for reranked scores: descending, NaN sorting deterministically last — the same
/// rule [`crate::search`]'s `OrdF32`/`TopK::into_sorted_desc` use, just not importable from here
/// (that type is private to `search`), so the semantics are copied, not invented.
fn cmp_desc(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // NaN sorts last under a descending order
        (false, true) => Ordering::Less,
        (false, false) => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
    }
}

/// Substitute `scored` (original-hit-index, rerank score) into the hits they came from, and
/// re-sort them: rerank score descending (NaN last), ties broken on `(collection, id)`
/// ascending — then append `passthrough` in their original relative order, scores untouched.
pub(crate) fn apply_scores(
    hits: Vec<Hit>,
    scored: &[(usize, f32)],
    passthrough: Vec<usize>,
) -> Vec<Hit> {
    let mut slots: Vec<Option<Hit>> = hits.into_iter().map(Some).collect();

    let mut reranked: Vec<Hit> = scored
        .iter()
        .filter_map(|&(idx, score)| {
            slots.get_mut(idx)?.take().map(|mut hit| {
                hit.score = score;
                hit
            })
        })
        .collect();
    reranked.sort_by(|a, b| {
        cmp_desc(a.score, b.score).then_with(|| {
            (a.collection.as_str(), a.id.as_str()).cmp(&(b.collection.as_str(), b.id.as_str()))
        })
    });

    let tail = passthrough
        .into_iter()
        .filter_map(|i| slots.get_mut(i)?.take());
    reranked.extend(tail);
    reranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LimitPer;
    use std::collections::BTreeMap;

    fn hit(collection: &str, id: &str, score: f32, text: Option<&str>) -> Hit {
        let mut attrs = BTreeMap::new();
        if let Some(t) = text {
            attrs.insert("nidus.text".to_string(), Value::Str(t.to_string()));
        }
        Hit::new(collection, id, score, attrs)
    }

    #[test]
    fn rerank_depth_multiplies_the_page_by_overscan() {
        let opts = SearchOpts {
            top_k: 10,
            rerank: Some(crate::model::RerankOpts {
                overscan: 5,
                text_attr: "nidus.text".to_string(),
            }),
            ..Default::default()
        };
        assert_eq!(rerank_depth(&opts), 50);
    }

    #[test]
    fn rerank_depth_clamps_overscan_to_at_least_one() {
        let opts = SearchOpts {
            top_k: 10,
            rerank: Some(crate::model::RerankOpts {
                overscan: 0,
                text_attr: "nidus.text".to_string(),
            }),
            ..Default::default()
        };
        assert_eq!(rerank_depth(&opts), 10);
    }

    #[test]
    fn rerank_depth_respects_limit_per_overfetch() {
        let plain = SearchOpts {
            top_k: 10,
            rerank: Some(crate::model::RerankOpts {
                overscan: 2,
                text_attr: "nidus.text".to_string(),
            }),
            ..Default::default()
        };
        let with_cap = SearchOpts {
            limit_per: Some(LimitPer::new("f", 1)),
            ..plain.clone()
        };
        assert!(rerank_depth(&with_cap) > rerank_depth(&plain));
    }

    #[test]
    fn candidate_texts_separates_non_empty_str_from_everything_else() {
        let hits = vec![
            hit("c", "a", 1.0, Some("hello")),
            hit("c", "b", 1.0, None),
            hit("c", "d", 1.0, Some("")),
        ];
        let (texts, passthrough) = candidate_texts(&hits, "nidus.text");
        assert_eq!(texts, vec!["hello"]);
        assert_eq!(passthrough, vec![1, 2]);
    }

    #[test]
    fn candidate_texts_treats_null_and_non_str_as_passthrough() {
        let mut null_attrs = BTreeMap::new();
        null_attrs.insert("nidus.text".to_string(), Value::Null);
        let mut int_attrs = BTreeMap::new();
        int_attrs.insert("nidus.text".to_string(), Value::Int(3));
        let hits = vec![
            Hit::new("c", "a", 1.0, null_attrs),
            Hit::new("c", "b", 1.0, int_attrs),
        ];
        let (texts, passthrough) = candidate_texts(&hits, "nidus.text");
        assert!(texts.is_empty());
        assert_eq!(passthrough, vec![0, 1]);
    }

    #[test]
    fn apply_scores_reorders_by_score_then_appends_passthroughs_in_order() {
        let hits = vec![
            hit("c", "a", 0.9, Some("a")),
            hit("c", "b", 0.5, Some("b")),
            hit("c", "z-passthrough", 0.99, None),
        ];
        // metric-worst (b) scores highest from the reranker.
        let scored = vec![(0, 0.1), (1, 0.8)];
        let out = apply_scores(hits, &scored, vec![2]);
        let ids: Vec<&str> = out.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids, vec!["b", "a", "z-passthrough"]);
        assert_eq!(out[0].score, 0.8);
        assert_eq!(out[1].score, 0.1);
        assert_eq!(out[2].score, 0.99, "passthrough score untouched");
    }

    #[test]
    fn apply_scores_breaks_ties_on_collection_then_id() {
        let hits = vec![hit("z", "a", 1.0, Some("x")), hit("a", "b", 1.0, Some("y"))];
        let scored = vec![(0, 5.0), (1, 5.0)];
        let out = apply_scores(hits, &scored, vec![]);
        assert_eq!(
            out.iter()
                .map(|h| h.collection.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
    }

    #[test]
    fn a_nan_rerank_score_sorts_deterministically_last() {
        let hits = vec![
            hit("c", "a", 1.0, Some("x")),
            hit("c", "b", 1.0, Some("y")),
            hit("c", "c", 1.0, Some("z")),
        ];
        let scored = vec![(0, f32::NAN), (1, 2.0), (2, 1.0)];
        let out = apply_scores(hits, &scored, vec![]);
        let ids: Vec<&str> = out.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids, vec!["b", "c", "a"], "NaN never displaces a real score");
    }
}
