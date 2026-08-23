//! Result annotations (nidus-m50.5): the opt-in explanation of *why* a hit matched —
//! each fusion leg's own rank and score, each BM25 clause's contribution, and
//! highlighted fragments of the stored text.

use serde::{Deserialize, Serialize};

/// How much text a highlight carries. `fragment_chars` is a **character** budget per
/// fragment (the excerpt is cut on char boundaries, never mid-codepoint).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct HighlightOpts {
    /// Most fragments returned per field.
    pub max_fragments: usize,
    /// Characters per fragment, split as leading context then the match and its tail.
    pub fragment_chars: usize,
}

impl Default for HighlightOpts {
    /// One fragment of 160 characters — a snippet an agent can read, not a second copy
    /// of the document.
    fn default() -> Self {
        Self {
            max_fragments: 1,
            fragment_chars: 160,
        }
    }
}

impl HighlightOpts {
    /// Return at most `n` fragments per field.
    pub fn max_fragments(mut self, n: usize) -> Self {
        self.max_fragments = n;
        self
    }

    /// Budget `n` characters per fragment.
    pub fn fragment_chars(mut self, n: usize) -> Self {
        self.fragment_chars = n;
        self
    }
}

/// One fusion leg's own view of a document: where it ranked in that leg (0-based) and
/// what that leg scored it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LegScore {
    pub rank: usize,
    pub score: f32,
}

/// One BM25 clause's contribution to a hit's text score. Only clauses that actually
/// matched are reported.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClauseScore {
    pub field: String,
    pub score: f32,
    /// For a prefix clause: how many terms matched, and how many were scored after the cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansion: Option<Expansion>,
}

/// A prefix clause's expansion: `matched > scored` means the cap truncated it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Expansion {
    pub matched: usize,
    pub scored: usize,
}

/// An excerpt of a field's stored text plus the byte ranges **within `text`** that a
/// query term matched. The ranges cover the *surface* form, so a stemmed match
/// ("running" for the query "run") highlights the word as the document spells it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Fragment {
    pub text: String,
    pub spans: Vec<(usize, usize)>,
}

/// The fragments found in one full-text field.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Highlight {
    pub field: String,
    pub fragments: Vec<Fragment>,
}

/// Why a hit matched. Every part is opt-in — `SearchOpts::explain` / `HybridOpts::explain`
/// for the scores, `FtsQuery::highlight` for the fragments — so a default query's response
/// is byte-identical to a nidus without annotations.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Annotations {
    /// The vector leg's rank and score, on a hybrid hit the vector leg returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<LegScore>,
    /// The BM25 leg's rank and combined text score, on a hybrid hit that leg returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<LegScore>,
    /// Each matched clause's own BM25 score, in query order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub clauses: Vec<ClauseScore>,
    /// Highlighted fragments, one entry per clause field that had a match.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub highlights: Vec<Highlight>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highlight_defaults_are_one_short_fragment() {
        let d = HighlightOpts::default();
        assert_eq!((d.max_fragments, d.fragment_chars), (1, 160));
        assert_eq!(
            HighlightOpts::default().max_fragments(3).fragment_chars(40),
            HighlightOpts {
                max_fragments: 3,
                fragment_chars: 40
            }
        );
    }

    #[test]
    fn an_annotation_defaults_to_carrying_nothing() {
        // Opting in to one part must not manufacture the other three; the wire form
        // skips every empty one (asserted in `server::dto`, which has serde_json).
        let a = Annotations::default();
        assert!(a.vector.is_none() && a.text.is_none());
        assert!(a.clauses.is_empty() && a.highlights.is_empty());
    }
}
