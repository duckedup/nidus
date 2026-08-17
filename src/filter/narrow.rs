//! Filter-level candidate narrowing: walking a whole [`Filter`] through the index in
//! `src/findex/`. Contract: see the root `SPEC.md` §7.4.
//!
//! `None` throughout means **"consider every document"**, not "no document matches". That
//! is the safe direction, and every construct this walk does not understand returns it. The
//! caller still runs [`super::matches`] on whatever survives, so a superset costs time
//! while a subset would drop real results.

use std::collections::BTreeSet;

use crate::findex::Findex;
use crate::model::{Filter, Predicate};

/// Past this share of a collection's live docs, walking a candidate list and then scanning
/// most of the store anyway costs more than the straight scan. A heuristic, not a contract.
const MAX_CANDIDATE_SHARE: f64 = 0.5;

/// Candidate doc ids for `filter`, or `None` to scan everything.
pub(crate) fn candidate_ids(
    findex: &Findex,
    collection: &str,
    filter: &Filter,
) -> Option<Vec<String>> {
    if !findex.is_active() {
        return None;
    }
    // The budget is passed down so a hot term is rejected from its posting-list lengths
    // alone, before any candidate list is built.
    let live = findex.live_docs(collection)?;
    let limit = (live as f64 * MAX_CANDIDATE_SHARE) as usize;
    let set = conjunction(findex, collection, &filter.0, limit)?;
    if set.len() > limit {
        return None;
    }
    Some(set.into_iter().collect())
}

/// AND: intersect what the indexable children give, ignoring the ones that cannot narrow.
/// All-unnarrowable is `None`; a narrowable child alone still constrains the whole.
fn conjunction(
    findex: &Findex,
    collection: &str,
    preds: &[Predicate],
    limit: usize,
) -> Option<BTreeSet<String>> {
    let mut acc: Option<BTreeSet<String>> = None;
    for p in preds {
        let Some(here) = one(findex, collection, p, limit) else {
            continue;
        };
        acc = Some(match acc {
            None => here,
            Some(prev) => prev.intersection(&here).cloned().collect(),
        });
        if acc.as_ref().is_some_and(BTreeSet::is_empty) {
            break;
        }
    }
    acc
}

/// OR: every child must be narrowable, or the whole node is not. A union that silently
/// omits an unnarrowable branch drops every document that matched only through it.
fn disjunction(
    findex: &Findex,
    collection: &str,
    preds: &[Predicate],
    limit: usize,
) -> Option<BTreeSet<String>> {
    let mut acc = BTreeSet::new();
    for p in preds {
        acc.extend(one(findex, collection, p, limit)?);
    }
    Some(acc)
}

fn one(
    findex: &Findex,
    collection: &str,
    pred: &Predicate,
    limit: usize,
) -> Option<BTreeSet<String>> {
    match pred {
        Predicate::All(preds) => conjunction(findex, collection, preds, limit),
        Predicate::Any(preds) => disjunction(findex, collection, preds, limit),
        // The complement of a superset is a SUBSET of the complement, so narrowing through
        // a negation drops real matches. There is no cheap sound answer; scan everything.
        Predicate::Not(_) => None,
        _ => findex
            .candidate_ids(collection, pred, limit)
            .map(|ids| ids.into_iter().collect()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::candidate_ids;
    use crate::findex::{FilterIndexField, Findex};
    use crate::model::{Filter, Predicate, Value};

    fn store() -> Findex {
        let mut f = Findex::default();
        f.set_schema("c", &[FilterIndexField::new("text")]);
        for (id, text) in [
            ("a", "alpha beta gamma"),
            ("b", "beta delta"),
            ("c", "gamma epsilon"),
            ("d", "zeta eta theta iota kappa"),
        ] {
            let mut attrs = BTreeMap::new();
            attrs.insert("text".to_string(), Value::Str(text.to_string()));
            f.index_doc("c", id, &attrs);
        }
        f
    }

    fn all_tokens(q: &str) -> Predicate {
        Predicate::ContainsAllTokens("text".into(), q.into())
    }

    fn ids(f: &Findex, filter: Filter) -> Option<Vec<String>> {
        candidate_ids(f, "c", &filter)
    }

    #[test]
    fn an_inactive_index_never_narrows() {
        let f = Findex::default();
        assert_eq!(ids(&f, Filter(vec![all_tokens("alpha")])), None);
    }

    #[test]
    fn a_single_indexed_leaf_narrows() {
        let f = store();
        assert_eq!(
            ids(&f, Filter(vec![all_tokens("alpha")])),
            Some(vec!["a".to_string()])
        );
    }

    #[test]
    fn a_conjunction_intersects() {
        let f = store();
        let filter = Filter(vec![all_tokens("beta"), all_tokens("gamma")]);
        assert_eq!(ids(&f, filter), Some(vec!["a".to_string()]));
    }

    #[test]
    fn an_unindexed_leaf_does_not_constrain_but_does_not_block() {
        let f = store();
        let filter = Filter(vec![
            all_tokens("alpha"),
            Predicate::Eq("other".into(), Value::Int(1)),
        ]);
        // The Eq cannot narrow, but the indexed leaf still can — an AND only tightens.
        assert_eq!(ids(&f, filter), Some(vec!["a".to_string()]));
    }

    #[test]
    fn a_negation_never_narrows() {
        // Complementing a superset yields a subset, which would drop real matches.
        let f = store();
        let filter = Filter(vec![Predicate::Not(Box::new(all_tokens("alpha")))]);
        assert_eq!(ids(&f, filter), None);
    }

    #[test]
    fn a_disjunction_of_indexed_leaves_unions() {
        let f = store();
        let filter = Filter(vec![Predicate::Any(vec![
            all_tokens("alpha"),
            all_tokens("delta"),
        ])]);
        assert_eq!(
            ids(&f, filter),
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn a_disjunction_with_one_unindexed_branch_never_narrows() {
        // Union-ing only the indexed branch would drop every doc matching via the Eq.
        let f = store();
        let filter = Filter(vec![Predicate::Any(vec![
            all_tokens("alpha"),
            Predicate::Eq("other".into(), Value::Int(1)),
        ])]);
        assert_eq!(ids(&f, filter), None);
    }

    #[test]
    fn a_disjunction_containing_a_negation_never_narrows() {
        let f = store();
        let filter = Filter(vec![Predicate::Any(vec![
            all_tokens("alpha"),
            Predicate::Not(Box::new(all_tokens("delta"))),
        ])]);
        assert_eq!(ids(&f, filter), None);
    }

    #[test]
    fn a_genuinely_empty_result_is_narrowed_not_abandoned() {
        // Distinct from None: no document carries this token, so the scan can skip them all.
        let f = store();
        assert_eq!(ids(&f, Filter(vec![all_tokens("absent")])), Some(vec![]));
    }

    #[test]
    fn the_cost_guard_declines_when_narrowing_barely_narrows() {
        // "beta" hits 2 of 4 docs; at half the collection the candidate walk stops paying.
        let f = store();
        let filter = Filter(vec![Predicate::Any(vec![
            all_tokens("beta"),
            all_tokens("gamma"),
        ])]);
        assert_eq!(ids(&f, filter), None);
    }

    #[test]
    fn nesting_composes_all_three_rules() {
        let f = store();
        let filter = Filter(vec![Predicate::All(vec![
            all_tokens("beta"),
            Predicate::Any(vec![all_tokens("alpha"), all_tokens("delta")]),
        ])]);
        assert_eq!(
            ids(&f, filter),
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }
}
