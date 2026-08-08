//! Aggregation and result diversity (nidus-m50.6): `count`/`sum` answered from the in-RAM index
//! without materializing a record, and `limit_per` — a cap on how many hits may carry one
//! attribute value, applied to a deliberately over-fetched ranking. See `SPEC.md` §7.7.

use std::collections::HashMap;

use anyhow::{Result, bail};

use super::Store;
use crate::filter;
use crate::model::{AggregateOpts, Aggregation, Group, Hit, LimitPer, Value};

/// Attributes as the index holds them, so the accumulator reads a record without building one.
type Attrs = std::collections::BTreeMap<String, Value>;

/// How much deeper than a page a capped search ranks, so hits behind a capped group still have
/// room to surface. Larger costs a proportionally deeper scan (nidus-m50.15 #16).
pub(super) const LIMIT_PER_OVERFETCH: usize = 8;

/// Cap on distinct group values tracked while capping. Past it further values pass through
/// uncapped, rather than letting an unbounded-cardinality attribute grow the map without limit.
pub(super) const MAX_GROUPS: usize = 10_000;

/// Reject a cap that cannot be honoured, once per query. `max: 0` would return nothing at all,
/// which is a caller mistake worth naming rather than an empty page worth guessing at.
pub(super) fn validate(limit_per: Option<&LimitPer>) -> Result<()> {
    let Some(cap) = limit_per else { return Ok(()) };
    if cap.field.is_empty() {
        bail!("limit_per needs a field name");
    }
    if cap.max == 0 {
        bail!("limit_per max must be at least 1");
    }
    Ok(())
}

/// Reject a `group_by` naming no attribute. Every record would land in the one "missing"
/// group, which reads as a working query returning a meaningless answer. Marked inline
/// because this one runs from `Nidus`, outside `check_query_opts`'s blanket `.context`.
pub(crate) fn validate_group_by(group_by: Option<&String>) -> Result<()> {
    if group_by.is_some_and(String::is_empty) {
        bail!("{}: group_by needs a field name", super::BAD_QUERY);
    }
    Ok(())
}

/// A collision-free string key for one group value — the variant prefix is what keeps
/// `Int(1)` and `Str("1")` apart. Every record **missing** the attribute maps to the same
/// key, so an absent value cannot slip past the cap (nidus-m50.15 #14).
fn group_key(v: Option<&Value>) -> String {
    match v {
        None => "\u{0}missing".to_string(),
        Some(Value::Null) => "\u{0}null".to_string(),
        Some(Value::Str(s)) => format!("s{s}"),
        Some(Value::Int(i)) => format!("i{i}"),
        Some(Value::Bool(b)) => format!("b{b}"),
        Some(Value::Float(f)) => format!("f{:016x}", f.to_bits()),
        Some(Value::DateTime(t)) => format!("t{t}"),
        Some(Value::List(items)) => format!("l{}", items.join("\u{0}")),
    }
}

/// Running totals over one bucket of records — the whole scope, or one `group_by` value.
struct Acc {
    count: u64,
    /// Per sum field: the integer part in `i128` so a long `i64` run cannot wrap, the float
    /// part separately, and whether any float was seen (which decides the reported type).
    sums: Vec<(i128, f64, bool)>,
}

impl Acc {
    fn new(fields: usize) -> Self {
        Self {
            count: 0,
            sums: vec![(0, 0.0, false); fields],
        }
    }

    fn add(&mut self, attrs: &Attrs, fields: &[String]) {
        self.count += 1;
        for (slot, field) in self.sums.iter_mut().zip(fields) {
            match attrs.get(field) {
                Some(Value::Int(n)) => slot.0 += *n as i128,
                Some(Value::Float(f)) => {
                    slot.1 += *f;
                    slot.2 = true;
                }
                // A missing or non-numeric value is skipped, never counted as zero.
                _ => {}
            }
        }
    }

    fn finish(self, fields: &[String]) -> std::collections::BTreeMap<String, Value> {
        fields
            .iter()
            .cloned()
            .zip(self.sums)
            .map(|(field, (ints, floats, saw_float))| {
                let total = if saw_float {
                    Value::Float(ints as f64 + floats)
                } else {
                    i64::try_from(ints).map_or_else(|_| Value::Float(ints as f64), Value::Int)
                };
                (field, total)
            })
            .collect()
    }
}

impl Store {
    /// Count the filter-matching records across `collections` and sum the requested attributes
    /// in ONE pass over the in-RAM index — no `Record` is built and no vector row is read.
    /// A `group_by` splits the same pass into one [`Group`] per distinct value.
    pub fn aggregate(&self, collections: &[&str], opts: &AggregateOpts) -> Aggregation {
        let fields = opts.sum.len();
        let mut total = Acc::new(fields);
        let mut groups: HashMap<String, (Option<Value>, Acc)> = HashMap::new();
        let mut truncated = false;
        for &col_name in collections {
            let Some(col) = self.collections.get(col_name) else {
                continue;
            };
            for entry in col.docs.values() {
                if !filter::matches(&opts.filter, &entry.attrs) {
                    continue;
                }
                total.add(&entry.attrs, &opts.sum);
                let Some(group_field) = &opts.group_by else {
                    continue;
                };
                let value = entry.attrs.get(group_field);
                let key = group_key(value);
                if let Some((_, acc)) = groups.get_mut(&key) {
                    acc.add(&entry.attrs, &opts.sum);
                } else if groups.len() >= MAX_GROUPS {
                    // Past the cap a NEW value is dropped rather than growing the map without
                    // bound; values already tracked keep accumulating, so their rows stay exact.
                    truncated = true;
                } else {
                    let mut acc = Acc::new(fields);
                    acc.add(&entry.attrs, &opts.sum);
                    groups.insert(key, (value.cloned(), acc));
                }
            }
        }
        // Ordered by size, ties broken by the collision-free key: a `HashMap` has no order of
        // its own, and an answer that reshuffles between identical calls is not reportable.
        let mut keyed: Vec<(String, Group)> = groups
            .into_iter()
            .map(|(key, (value, acc))| {
                let group = Group {
                    value,
                    count: acc.count,
                    sums: acc.finish(&opts.sum),
                };
                (key, group)
            })
            .collect();
        keyed.sort_by(|a, b| b.1.count.cmp(&a.1.count).then_with(|| a.0.cmp(&b.0)));
        Aggregation {
            count: total.count,
            sums: total.finish(&opts.sum),
            groups: keyed.into_iter().map(|(_, g)| g).collect(),
            groups_truncated: truncated,
        }
    }

    /// Drop hits past the per-value cap, walking the ranking in order so the best hits of each
    /// group are the ones kept. The group value is read from the **live record**, so a
    /// projection that excludes the field cannot disable the cap.
    pub(super) fn cap_per_value(&self, hits: Vec<Hit>, cap: &LimitPer) -> Vec<Hit> {
        let mut seen: HashMap<String, usize> = HashMap::new();
        hits.into_iter()
            .filter(|h| {
                let value = self
                    .attrs_of(&h.collection, &h.id)
                    .and_then(|a| a.get(&cap.field));
                let key = group_key(value);
                if let Some(n) = seen.get_mut(&key) {
                    *n += 1;
                    return *n <= cap.max;
                }
                if seen.len() >= MAX_GROUPS {
                    return true;
                }
                seen.insert(key, 1);
                cap.max >= 1
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_group_key_never_collides_across_variants() {
        let keys = [
            group_key(None),
            group_key(Some(&Value::Null)),
            group_key(Some(&Value::Str("1".into()))),
            group_key(Some(&Value::Int(1))),
            group_key(Some(&Value::Bool(true))),
            group_key(Some(&Value::Float(1.0))),
            group_key(Some(&Value::DateTime(1))),
            group_key(Some(&Value::List(vec!["1".into()]))),
        ];
        let unique: std::collections::HashSet<&String> = keys.iter().collect();
        assert_eq!(unique.len(), keys.len(), "{keys:?}");
    }

    #[test]
    fn every_missing_record_shares_one_key() {
        assert_eq!(group_key(None), group_key(None));
        assert_ne!(group_key(None), group_key(Some(&Value::Null)));
    }
}
