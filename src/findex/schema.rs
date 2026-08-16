//! The per-field filter-index declaration. Contract: see the root `SPEC.md` §7.4.

use serde::{Deserialize, Serialize};

use anyhow::{Result, bail};

/// One attribute field to index for the text predicates of `SPEC.md` §7.4/§7.5. Both
/// structures default on: a caller should not have to know which predicate uses which,
/// and turning one off is the tuning step, not the starting point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilterIndexField {
    pub(crate) field: String,
    /// Raw-token postings, serving `ContainsAllTokens`/`ContainsAnyToken`/`ContainsTokenSequence`.
    pub(crate) tokens: bool,
    /// Character-trigram postings, serving `Fuzzy` and `Regex`.
    pub(crate) trigrams: bool,
}

impl FilterIndexField {
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            tokens: true,
            trigrams: true,
        }
    }

    /// Index raw tokens for this field (default `true`).
    pub fn tokens(mut self, on: bool) -> Self {
        self.tokens = on;
        self
    }

    /// Index character trigrams for this field (default `true`).
    pub fn trigrams(mut self, on: bool) -> Self {
        self.trigrams = on;
        self
    }

    pub fn field_name(&self) -> &str {
        &self.field
    }
}

/// Reject a declaration that cannot be honoured, before it reaches the log. A field with
/// both structures off would index nothing while reading as indexed, which is worse than
/// an error: queries would silently keep their un-indexed cost with no way to see why.
pub(crate) fn validate(fields: &[FilterIndexField]) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for f in fields {
        if f.field.is_empty() {
            bail!("filter index field name must not be empty");
        }
        if !f.tokens && !f.trigrams {
            bail!(
                "filter index on `{}` has both `tokens` and `trigrams` disabled, so it would index nothing",
                f.field
            );
        }
        if !seen.insert(f.field.as_str()) {
            bail!("filter index field `{}` declared twice", f.field);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{FilterIndexField, validate};

    #[test]
    fn both_structures_default_on() {
        let f = FilterIndexField::new("text");
        assert!(f.tokens && f.trigrams);
    }

    #[test]
    fn a_field_indexing_nothing_is_rejected() {
        let f = FilterIndexField::new("text").tokens(false).trigrams(false);
        let err = validate(&[f]).unwrap_err().to_string();
        assert!(err.contains("would index nothing"), "{err}");
    }

    #[test]
    fn a_duplicate_field_is_rejected() {
        let fields = [FilterIndexField::new("t"), FilterIndexField::new("t")];
        assert!(validate(&fields).unwrap_err().to_string().contains("twice"));
    }

    #[test]
    fn an_empty_field_name_is_rejected() {
        assert!(validate(&[FilterIndexField::new("")]).is_err());
    }

    #[test]
    fn one_structure_alone_is_fine() {
        assert!(validate(&[FilterIndexField::new("t").trigrams(false)]).is_ok());
        assert!(validate(&[FilterIndexField::new("t").tokens(false)]).is_ok());
    }
}
