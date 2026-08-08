//! The declared shape of one full-text field: BM25 tuning plus its analyzer.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use super::analyzer::{Analyzer, Language};

/// BM25 term-frequency saturation — the conventional default, and what every store
/// written before the parameter was configurable was scored under.
pub const DEFAULT_K1: f32 = 1.2;
/// BM25 length normalization (`0` = none, `1` = full) — the conventional default.
pub const DEFAULT_B: f32 = 0.75;

/// One full-text-indexed attribute field: its name, BM25 tuning, and analyzer. Build with
/// [`FtsField::new`] and override only what you need — the defaults are BM25's textbook
/// `k1 = 1.2` / `b = 0.75` over the US English analyzer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FtsField {
    /// The attribute to index. Its value is indexed as text when it is a `Str`, or as the
    /// space-joined elements when it is a `List`.
    pub field: String,
    /// Term-frequency saturation: larger means repeated terms keep adding score for longer.
    pub k1: f32,
    /// Length normalization: `0` ignores document length, `1` divides fully by it.
    pub b: f32,
    /// How text is turned into terms, at both index and query time.
    pub analyzer: Analyzer,
}

impl Default for FtsField {
    fn default() -> Self {
        Self {
            field: String::new(),
            k1: DEFAULT_K1,
            b: DEFAULT_B,
            analyzer: Analyzer::default(),
        }
    }
}

impl FtsField {
    /// Index `field` with the default BM25 params and the default (US English) analyzer —
    /// exactly the behaviour of every nidus release before these became tunable.
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            ..Self::default()
        }
    }

    /// Set BM25 `k1` (term-frequency saturation).
    pub fn k1(mut self, k1: f32) -> Self {
        self.k1 = k1;
        self
    }

    /// Set BM25 `b` (length normalization).
    pub fn b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }

    /// Replace the whole analyzer.
    pub fn analyzer(mut self, analyzer: Analyzer) -> Self {
        self.analyzer = analyzer;
        self
    }

    /// Set the analyzer language.
    pub fn language(mut self, language: Language) -> Self {
        self.analyzer.language = language;
        self
    }

    /// Fold Latin diacritics to ASCII before stemming, so "café" and "cafe" share a term.
    pub fn ascii_folding(mut self, on: bool) -> Self {
        self.analyzer.ascii_folding = on;
        self
    }

    /// Drop tokens longer than `chars` — the guard against a base64 blob or minified
    /// bundle inflating the term dictionary.
    pub fn max_token_len(mut self, chars: usize) -> Self {
        self.analyzer.max_token_len = Some(chars);
        self
    }
}

impl From<&str> for FtsField {
    fn from(field: &str) -> Self {
        Self::new(field)
    }
}

impl From<String> for FtsField {
    fn from(field: String) -> Self {
        Self::new(field)
    }
}

/// Reject a schema that would be persisted into the op log with values BM25 cannot use.
/// Rejected here rather than clamped, because a silently-corrected `k1` would score
/// differently from what the caller asked for with no way to notice.
pub(crate) fn validate(fields: &[FtsField]) -> Result<()> {
    for f in fields {
        if f.field.is_empty() {
            bail!("FTS field name must not be empty");
        }
        if !f.k1.is_finite() || f.k1 < 0.0 {
            bail!(
                "FTS field `{}`: k1 must be finite and >= 0, got {}",
                f.field,
                f.k1
            );
        }
        if !f.b.is_finite() || !(0.0..=1.0).contains(&f.b) {
            bail!(
                "FTS field `{}`: b must be within [0, 1], got {}",
                f.field,
                f.b
            );
        }
        if f.analyzer.max_token_len == Some(0) {
            bail!("FTS field `{}`: max_token_len must be >= 1", f.field);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_todays_hard_coded_constants() {
        let f = FtsField::new("body");
        assert_eq!(f.field, "body");
        assert_eq!(f.k1, 1.2);
        assert_eq!(f.b, 0.75);
        assert_eq!(f.analyzer.language, Language::English);
        assert!(!f.analyzer.ascii_folding);
        assert_eq!(f.analyzer.max_token_len, None);
    }

    #[test]
    fn builders_compose_without_disturbing_the_rest() {
        let f = FtsField::new("body")
            .k1(1.5)
            .b(0.4)
            .ascii_folding(true)
            .max_token_len(40);
        assert_eq!((f.k1, f.b), (1.5, 0.4));
        assert!(f.analyzer.ascii_folding);
        assert_eq!(f.analyzer.max_token_len, Some(40));
        assert_eq!(f.analyzer.language, Language::English);
    }

    #[test]
    fn from_str_is_the_bare_default_field() {
        assert_eq!(FtsField::from("body"), FtsField::new("body"));
        assert_eq!(FtsField::from("body".to_string()), FtsField::new("body"));
    }

    #[test]
    fn validate_rejects_unusable_params() {
        assert!(validate(&[FtsField::new("body")]).is_ok());
        assert!(validate(&[FtsField::new("")]).is_err());
        assert!(validate(&[FtsField::new("b").k1(-0.1)]).is_err());
        assert!(validate(&[FtsField::new("b").k1(f32::NAN)]).is_err());
        assert!(validate(&[FtsField::new("b").b(1.5)]).is_err());
        assert!(validate(&[FtsField::new("b").b(-0.0001)]).is_err());
        assert!(validate(&[FtsField::new("b").max_token_len(0)]).is_err());
        // The edges are legal: k1 = 0 disables tf saturation, b = 0/1 the two extremes.
        assert!(validate(&[FtsField::new("b").k1(0.0).b(0.0)]).is_ok());
        assert!(validate(&[FtsField::new("b").b(1.0)]).is_ok());
    }

    #[test]
    fn serde_round_trips_through_bincode() {
        let f = FtsField::new("body")
            .k1(1.9)
            .b(0.1)
            .ascii_folding(true)
            .max_token_len(12);
        let back: FtsField = bincode::deserialize(&bincode::serialize(&f).unwrap()).unwrap();
        assert_eq!(f, back);
    }
}
