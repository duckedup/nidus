//! Retrieval-quality metrics against graded relevance judgements (nidus-yq9p.5).

use std::collections::HashMap;

/// Graded relevance for one query: doc id -> qrel score. BEIR qrels are graded 0-2
/// (NFCorpus goes higher), so the gain is the score itself, not a binary flag.
pub type Qrels = HashMap<String, HashMap<String, f32>>;

/// The rank discount for a 1-based position: `log2(i + 1)`.
fn discount(rank_1based: usize) -> f64 {
    ((rank_1based + 1) as f64).log2()
}

/// DCG@k for one query's judged gains, taken in the order given and truncated to `k`.
fn dcg(gains: impl Iterator<Item = f64>, k: usize) -> f64 {
    gains
        .take(k)
        .enumerate()
        .map(|(i, rel)| rel / discount(i + 1))
        .sum()
}

/// Mean nDCG@k over queries, gains taken from `qrels` and the ideal DCG computed over that
/// query's own judgements. Queries with no judgements are skipped, not scored zero.
pub fn ndcg_at_k(ranked: &[(String, Vec<String>)], qrels: &Qrels, k: usize) -> f64 {
    let mut total = 0.0f64;
    let mut scored = 0usize;
    for (query_id, docs) in ranked {
        let Some(judged) = qrels.get(query_id) else {
            continue;
        };
        if judged.is_empty() {
            continue;
        }
        let run_dcg = dcg(
            docs.iter().map(|d| *judged.get(d).unwrap_or(&0.0) as f64),
            k,
        );
        let mut ideal: Vec<f64> = judged.values().map(|&r| r as f64).collect();
        ideal.sort_unstable_by(|a, b| b.total_cmp(a));
        let ideal_dcg = dcg(ideal.into_iter(), k);
        if ideal_dcg > 0.0 {
            total += run_dcg / ideal_dcg;
            scored += 1;
        }
    }
    if scored == 0 {
        0.0
    } else {
        total / scored as f64
    }
}

/// Flatten graded qrels into the binary truth sets `nidus::recall_at_k` expects: a doc counts
/// as relevant when its score is strictly greater than `threshold` (BEIR convention: 0.0).
pub fn binary_truth(query_ids: &[String], qrels: &Qrels, threshold: f32) -> Vec<Vec<String>> {
    query_ids
        .iter()
        .map(|q| {
            qrels
                .get(q)
                .map(|judged| {
                    judged
                        .iter()
                        .filter(|&(_, &score)| score > threshold)
                        .map(|(doc, _)| doc.clone())
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recall_at_k;
    use std::collections::HashSet;

    fn qrels(entries: &[(&str, &[(&str, f32)])]) -> Qrels {
        entries
            .iter()
            .map(|(q, docs)| {
                (
                    q.to_string(),
                    docs.iter().map(|(d, r)| (d.to_string(), *r)).collect(),
                )
            })
            .collect()
    }

    fn ranked(query: &str, docs: &[&str]) -> Vec<(String, Vec<String>)> {
        vec![(
            query.to_string(),
            docs.iter().map(|d| d.to_string()).collect(),
        )]
    }

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        // Ranking matches the ideal order exactly, so DCG == IDCG term for term.
        let q = qrels(&[("q1", &[("a", 3.0), ("b", 2.0), ("c", 1.0)])]);
        let r = ranked("q1", &["a", "b", "c"]);
        assert_eq!(ndcg_at_k(&r, &q, 3), 1.0);
    }

    #[test]
    fn ndcg_uses_graded_gains_not_binary() {
        // a=2.0, b=1.0, ranked worst-first [b, a], k=2. Graded: DCG=1/log2(2)+2/log2(3)=
        // 2.261859..., IDCG=2/log2(2)+1/log2(3)=2.630929..., nDCG=0.859718699852...
        // Binary (gain=1.0 for any judged doc) gives DCG==IDCG, nDCG=1.0: must differ.
        let q = qrels(&[("q1", &[("a", 2.0), ("b", 1.0)])]);
        let r = ranked("q1", &["b", "a"]);
        let got = ndcg_at_k(&r, &q, 2);
        assert!((got - 0.8597186998521972).abs() < 1e-9);
        assert!((got - 1.0).abs() > 1e-3);
    }

    #[test]
    fn ndcg_discount_is_log2_rank_plus_one() {
        // One judged doc (rel=1.0) at rank 2 of 3; ideal places it at rank 1.
        // DCG=1/log2(3), IDCG=1/log2(2)=1.0, nDCG=1/log2(3)=0.6309297535714575.
        // An off-by-one discount gives a different plausible number (e.g. 0.5).
        let q = qrels(&[("q1", &[("x", 1.0)])]);
        let r = ranked("q1", &["unjudged", "x", "unjudged2"]);
        let got = ndcg_at_k(&r, &q, 3);
        assert!((got - 0.6309297535714575).abs() < 1e-6);
    }

    #[test]
    fn ndcg_normalises_against_the_querys_own_ideal() {
        // 3 judgements (3,2,1) but k=2: ideal truncates to top two. Run ranks a,c:
        // DCG=3/log2(2)+1/log2(3)=3.630929..., IDCG=3/log2(2)+2/log2(3)=4.261859...,
        // nDCG=0.851959044517... An unnormalised score is the raw DCG, which is > 1.0.
        let q = qrels(&[("q1", &[("a", 3.0), ("b", 2.0), ("c", 1.0)])]);
        let r = ranked("q1", &["a", "c"]);
        let got = ndcg_at_k(&r, &q, 2);
        assert!(got <= 1.0);
        assert!((got - 0.8519590445170673).abs() < 1e-9);
    }

    #[test]
    fn ndcg_truncates_at_k() {
        // The only judged doc sits at rank k+1, so it never enters the k=2 window: DCG=0.
        // IDCG (that same doc, ideally at rank 1) = 1/log2(2) = 1.0, so nDCG = 0.0.
        let q = qrels(&[("q1", &[("x", 1.0)])]);
        let r = ranked("q1", &["unjudged1", "unjudged2", "x"]);
        assert_eq!(ndcg_at_k(&r, &q, 2), 0.0);
    }

    #[test]
    fn ndcg_skips_unjudged_queries() {
        // q1 is judged and ranked perfectly (nDCG=1.0); q2 has no qrels entry at all and
        // must be skipped, not scored 0.0 -- otherwise the mean would be 0.5, not 1.0.
        let q = qrels(&[("q1", &[("a", 1.0)])]);
        let r = vec![
            ("q1".to_string(), vec!["a".to_string()]),
            ("q2".to_string(), vec!["z".to_string()]),
        ];
        assert_eq!(ndcg_at_k(&r, &q, 1), 1.0);
    }

    #[test]
    fn ndcg_unjudged_documents_score_zero() {
        // Padding the ranking with ids absent from qrels contributes gain 0 at every rank,
        // so DCG and IDCG are unchanged from the unpadded case: nDCG stays 1.0.
        let q = qrels(&[("q1", &[("a", 1.0)])]);
        let r = ranked("q1", &["a", "pad1", "pad2"]);
        assert_eq!(ndcg_at_k(&r, &q, 3), 1.0);
    }

    #[test]
    fn binary_truth_thresholds_on_score() {
        // q1: "a" scores 1.0 (kept), "b" scores 0.0 (excluded at threshold 0.0).
        // q2 has no qrels entry: an empty set, not a missing element.
        let q = qrels(&[("q1", &[("a", 1.0), ("b", 0.0)])]);
        let ids = vec!["q1".to_string(), "q2".to_string()];
        let truth = binary_truth(&ids, &q, 0.0);
        assert_eq!(truth.len(), 2);
        assert_eq!(
            truth[0].iter().cloned().collect::<HashSet<_>>(),
            HashSet::from(["a".to_string()])
        );
        assert_eq!(truth[1], Vec::<String>::new());
    }

    #[test]
    fn binary_truth_feeds_recall_at_k() {
        // qrels for q1: "a"=1.0 and "x"=2.0 are relevant (> 0.0), "b"=0.0 is not.
        // returned = ["a", "b", "c"]: 1 of the 2 relevant docs ("a") comes back, so
        // recall = 1/2 = 0.5.
        let q = qrels(&[("q1", &[("a", 1.0), ("b", 0.0), ("x", 2.0)])]);
        let ids = vec!["q1".to_string()];
        let truth = binary_truth(&ids, &q, 0.0);
        let returned = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        assert_eq!(recall_at_k(&returned, &truth), 0.5);
    }
}
