//! Ranking expressions (nidus-m50.3): the recency-decay penalty every scoring path layers over
//! its base score, and attribute ordering for the vector-free `list`. Validation lives here so a
//! malformed expression is refused once per query, never once per record.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use anyhow::{Result, bail};

use super::Store;
use crate::cancel::{CHECK_EVERY, check};
use crate::filter::value_cmp;
use crate::model::{Decay, Hit, OrderBy, RankBy, SearchOpts, Value};
use crate::search::TopK;

/// Reject a ranking expression that cannot be honoured as written. Called once per query from
/// the `Nidus` entry points, beside `filter::validate`.
pub(crate) fn validate(rank_by: Option<&RankBy>) -> Result<()> {
    let Some(RankBy::Decay(d)) = rank_by else {
        return Ok(());
    };
    if d.field.is_empty() {
        bail!("Decay needs a timestamp field name");
    }
    if d.scale <= 0 {
        bail!("Decay scale must be positive, got {}", d.scale);
    }
    if !(d.decay > 0.0 && d.decay < 1.0) {
        bail!("Decay decay must be in (0, 1), got {}", d.decay);
    }
    if !d.lambda.is_finite() || d.lambda < 0.0 {
        bail!(
            "Decay lambda must be finite and non-negative, got {}",
            d.lambda
        );
    }
    if !(d.missing >= 0.0 && d.missing <= 1.0) {
        bail!("Decay missing must be in [0, 1], got {}", d.missing);
    }
    Ok(())
}

/// The decay factor for one record: `decay^(age / scale)`, so a record exactly `scale` old
/// sits at `decay` and a future timestamp is simply un-aged. `missing` stands in for an
/// absent or non-timestamp attribute.
fn factor(d: &Decay, attrs: &BTreeMap<String, Value>) -> f32 {
    let ts = match attrs.get(&d.field) {
        Some(Value::DateTime(ms)) | Some(Value::Int(ms)) => *ms,
        _ => return d.missing,
    };
    let age = d.origin.saturating_sub(ts).max(0) as f64;
    d.decay.powf((age / d.scale as f64) as f32)
}

/// Apply the ranking expression to one base score: `base − lambda × (1 − factor)`. It
/// **subtracts**; multiplying would be sound only for Cosine and would need a negative-score
/// clamp, whereas a subtraction is a translation every metric survives (nidus-m50.15 #7).
pub(super) fn adjust(rank_by: Option<&RankBy>, base: f32, attrs: &BTreeMap<String, Value>) -> f32 {
    match rank_by {
        None => base,
        Some(RankBy::Decay(d)) => base - d.lambda * (1.0 - factor(d, attrs)),
    }
}

impl Store {
    /// The live attrs of one doc, for a path holding only `(collection, id)`.
    pub(super) fn attrs_of(&self, collection: &str, id: &str) -> Option<&BTreeMap<String, Value>> {
        self.collections
            .get(collection)
            .and_then(|c| c.docs.get(id))
            .map(|e| &e.attrs)
    }

    /// [`adjust`] for a candidate resolved only to `(collection, id)`. `min_score` is compared
    /// against this final number, not the base one, on every path.
    pub(super) fn ranked_score(
        &self,
        rank_by: Option<&RankBy>,
        base: f32,
        collection: &str,
        id: &str,
    ) -> f32 {
        if rank_by.is_none() {
            return base;
        }
        match self.attrs_of(collection, id) {
            Some(attrs) => adjust(rank_by, base, attrs),
            None => base,
        }
    }

    /// The exact scan used when a ranking expression is set: score the row, apply the
    /// expression against that record's own attrs, then bound. Serial by design — the per-row
    /// attrs lookup is the cliff, and generalizing `parallel_topk` waits (nidus-m50.15 #11).
    pub(super) fn rank_scan_expr<'b>(
        &self,
        q: &[f32],
        scan: &[(u64, &'b str, &'b str)],
        score_fn: fn(&[f32], &[f32]) -> f32,
        opts: &SearchOpts,
    ) -> Result<Vec<Hit>> {
        let mut topk: TopK<(&'b str, &'b str)> = TopK::new(opts.top_k);
        for block in scan.chunks(CHECK_EVERY) {
            check()?;
            for &(row, col_name, id) in block {
                let base = score_fn(q, self.data.row(row));
                let score = self.ranked_score(opts.rank_by.as_ref(), base, col_name, id);
                if let Some(min) = opts.min_score
                    && score < min
                {
                    continue;
                }
                topk.offer(score, (col_name, id));
            }
        }
        Ok(self.hits_from_topk(topk, &opts.projection))
    }

    /// Reorder a `list` scan by `order.field`. The **witness** — the first value in the
    /// incoming stable order that compares against itself — fixes the sort's type; everything
    /// else trails in the order it arrived, whichever direction is asked for.
    pub(super) fn order_scan<'b>(
        &'b self,
        scan: &mut [(Option<u64>, &'b str, &'b str)],
        order: &OrderBy,
    ) {
        let key = |&(_, col, id): &(Option<u64>, &'b str, &'b str)| -> Option<&'b Value> {
            self.attrs_of(col, id).and_then(|a| a.get(&order.field))
        };
        let Some(witness) = scan
            .iter()
            .filter_map(key)
            .find(|v| value_cmp(v, v).is_some())
        else {
            return; // nothing orderable at all: the incoming order already is the answer
        };
        let sortable = |v: Option<&Value>| matches!(v, Some(x) if value_cmp(x, witness).is_some());
        scan.sort_by(|a, b| {
            let (va, vb) = (key(a), key(b));
            match (sortable(va), sortable(vb)) {
                (true, true) => {
                    let ord = value_cmp(va.unwrap_or(&Value::Null), vb.unwrap_or(&Value::Null))
                        .unwrap_or(Ordering::Equal);
                    if order.descending { ord.reverse() } else { ord }
                }
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                // A stable sort leaves the trailing bucket in the order `list` built it.
                (false, false) => Ordering::Equal,
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    /// The exact formula, asserted numerically: at one full `scale` the factor is `decay`,
    /// so the penalty is `lambda * (1 - decay)` and it is SUBTRACTED.
    #[test]
    fn one_half_life_subtracts_lambda_times_one_minus_decay() {
        let day = 86_400_000i64;
        let d = Decay::new("ts", 10 * day, day).lambda(0.4);
        let rank = RankBy::Decay(d);
        let fresh = adjust(
            Some(&rank),
            0.9,
            &attrs(&[("ts", Value::DateTime(10 * day))]),
        );
        let aged = adjust(
            Some(&rank),
            0.9,
            &attrs(&[("ts", Value::DateTime(9 * day))]),
        );
        assert!((fresh - 0.9).abs() < 1e-6, "age 0 keeps the base score");
        assert!((aged - (0.9 - 0.4 * 0.5)).abs() < 1e-6, "got {aged}");
    }

    #[test]
    fn a_negative_base_score_stays_ordered_under_the_penalty() {
        let day = 86_400_000i64;
        let rank = RankBy::Decay(Decay::new("ts", 10 * day, day).lambda(1.0));
        let old = adjust(Some(&rank), -4.0, &attrs(&[("ts", Value::DateTime(0))]));
        // Fully decayed: factor ≈ 0, so the whole lambda comes off, sign notwithstanding.
        assert!((old - (-5.0)).abs() < 1e-3, "got {old}");
    }

    #[test]
    fn a_missing_timestamp_is_unpenalized_by_default() {
        let rank = RankBy::Decay(Decay::new("ts", 1_000_000, 1_000));
        assert_eq!(adjust(Some(&rank), 0.7, &attrs(&[])), 0.7);
        assert_eq!(
            adjust(
                Some(&rank),
                0.7,
                &attrs(&[("ts", Value::Str("nope".into()))])
            ),
            0.7
        );
    }

    #[test]
    fn an_int_timestamp_decays_exactly_like_a_datetime_one() {
        let rank = RankBy::Decay(Decay::new("ts", 2_000, 1_000).lambda(1.0));
        let as_int = adjust(Some(&rank), 1.0, &attrs(&[("ts", Value::Int(1_000))]));
        let as_dt = adjust(Some(&rank), 1.0, &attrs(&[("ts", Value::DateTime(1_000))]));
        // Tolerance, not equality: the claim is that both variants take the same code
        // path, and Miri rounds `powf` non-deterministically per call, so two identical
        // inputs can differ by an ULP there while agreeing bit-for-bit natively.
        assert!((as_int - as_dt).abs() < 1e-6, "{as_int} vs {as_dt}");
    }

    #[test]
    fn a_future_timestamp_is_not_boosted() {
        let rank = RankBy::Decay(Decay::new("ts", 1_000, 1_000).lambda(1.0));
        assert_eq!(
            adjust(Some(&rank), 0.5, &attrs(&[("ts", Value::Int(9_999))])),
            0.5
        );
    }

    #[test]
    fn no_expression_leaves_the_score_untouched() {
        assert_eq!(adjust(None, -3.25, &attrs(&[("ts", Value::Int(0))])), -3.25);
    }

    #[test]
    fn validate_refuses_the_degenerate_knobs() {
        let ok = Decay::new("ts", 0, 1_000);
        assert!(validate(Some(&RankBy::Decay(ok.clone()))).is_ok());
        assert!(
            validate(Some(&RankBy::Decay(Decay {
                scale: 0,
                ..ok.clone()
            })))
            .is_err()
        );
        assert!(validate(Some(&RankBy::Decay(ok.clone().decay(1.0)))).is_err());
        assert!(validate(Some(&RankBy::Decay(ok.clone().decay(0.0)))).is_err());
        assert!(validate(Some(&RankBy::Decay(ok.clone().lambda(-1.0)))).is_err());
        assert!(validate(Some(&RankBy::Decay(ok.clone().missing(1.5)))).is_err());
        assert!(validate(Some(&RankBy::Decay(Decay::new("", 0, 1)))).is_err());
        assert!(validate(None).is_ok());
    }
}
