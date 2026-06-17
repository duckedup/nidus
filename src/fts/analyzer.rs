//! The text analyzer: raw text → normalized terms for BM25 indexing and querying.
//!
//! Pure safe Rust, zero FFI (no ICU / `unicode-segmentation` C deps), Miri-clean. The
//! pipeline for [`Language::English`] is: lowercase → tokenize on Unicode
//! alphanumeric runs → drop stopwords → [Porter stem](stem). The same analyzer runs at
//! index time and query time, so a query term matches a stored term iff they reduce to
//! the same stem.
//!
//! [`Language`] is the seam for more languages: each maps to its own stopword set and
//! stemmer, selected in [`analyze`]. Only English is implemented today.

use serde::{Deserialize, Serialize};

/// The analyzer language for a full-text field. Extensible; only English is implemented
/// today (the variant gates the stopword set + stemmer in [`analyze`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    /// US English: ASCII-folding lowercase, English stopwords, Porter stemming.
    #[default]
    English,
}

/// Analyze `text` into a sequence of normalized terms (in document order, duplicates
/// kept so term frequencies are countable). Empty input → no terms.
pub(crate) fn analyze(text: &str, lang: Language) -> Vec<String> {
    match lang {
        Language::English => tokenize(text)
            .into_iter()
            .filter(|t| !is_stopword(t))
            .map(|t| stem(&t))
            .filter(|t| !t.is_empty())
            .collect(),
    }
}

/// Split `text` into lowercased tokens on runs of Unicode alphanumerics. Everything
/// else (punctuation, whitespace, symbols) is a separator. Lowercasing uses
/// `char::to_lowercase` (std, no FFI), which handles the Latin script we target; a
/// pragmatic stand-in for full UAX #29 segmentation that stays pure and dependency-free.
fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// A common-English stopword (closed-class function words that carry little ranking
/// signal). Matched case-insensitively against an already-lowercased token.
fn is_stopword(token: &str) -> bool {
    STOPWORDS.binary_search(&token).is_ok()
}

/// English stopwords, **sorted** so [`is_stopword`] can binary-search. A compact,
/// conventional list (the classic Snowball English set, trimmed).
static STOPWORDS: &[&str] = &[
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "were",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
];

/// Reduce an English token to its [Porter stem](https://tartarus.org/martin/PorterStemmer/).
/// Pure ASCII-letter words are stemmed; anything else (digits, accented/non-ASCII
/// letters) is returned unchanged, since the classic algorithm is defined over `a–z`.
pub(crate) fn stem(word: &str) -> String {
    // Porter operates on lowercase ASCII letters. Tokens reach here already lowercased;
    // bail out (unchanged) on anything that isn't pure a–z so we never mangle numbers or
    // non-Latin script.
    if word.len() < 3 || !word.bytes().all(|b| b.is_ascii_lowercase()) {
        return word.to_string();
    }
    let mut s = Porter {
        b: word.bytes().collect(),
    };
    s.step1ab();
    s.step1c();
    s.step2();
    s.step3();
    s.step4();
    s.step5();
    String::from_utf8(s.b).expect("ascii in, ascii out")
}

/// The classic Porter stemmer, operating on a buffer of lowercase ASCII bytes. Faithful
/// to M. F. Porter's 1980 algorithm; pure integer/byte arithmetic, no allocation beyond
/// the buffer, so it runs under Miri.
struct Porter {
    b: Vec<u8>,
}

impl Porter {
    /// `b[i]` is a consonant. `y` is a consonant iff the preceding letter is a vowel
    /// (or it starts the word).
    fn is_consonant(&self, i: usize) -> bool {
        match self.b[i] {
            b'a' | b'e' | b'i' | b'o' | b'u' => false,
            b'y' => i == 0 || !self.is_consonant(i - 1),
            _ => true,
        }
    }

    /// The "measure" of `b[0..end]`: the number of consonant→vowel→consonant... VC
    /// sequences, which Porter writes as `m`.
    fn measure(&self, end: usize) -> usize {
        let mut n = 0;
        let mut i = 0;
        // Skip an initial run of consonants.
        while i < end && self.is_consonant(i) {
            i += 1;
        }
        loop {
            // Skip the vowel run.
            while i < end && !self.is_consonant(i) {
                i += 1;
            }
            if i >= end {
                return n;
            }
            n += 1;
            // Skip the consonant run.
            while i < end && self.is_consonant(i) {
                i += 1;
            }
            if i >= end {
                return n;
            }
        }
    }

    /// Any vowel in `b[0..end]`.
    fn has_vowel(&self, end: usize) -> bool {
        (0..end).any(|i| !self.is_consonant(i))
    }

    /// `b[end-1]` and `b[end-2]` are the same consonant (a doubled consonant ending).
    fn ends_double_consonant(&self, end: usize) -> bool {
        end >= 2 && self.b[end - 1] == self.b[end - 2] && self.is_consonant(end - 1)
    }

    /// `b[end-3..end]` is consonant-vowel-consonant and the final consonant is not
    /// `w`, `x`, or `y` (Porter's `*o` condition, used to decide a trailing `e`).
    fn cvc(&self, end: usize) -> bool {
        if end < 3 {
            return false;
        }
        if !self.is_consonant(end - 1) || self.is_consonant(end - 2) || !self.is_consonant(end - 3)
        {
            return false;
        }
        !matches!(self.b[end - 1], b'w' | b'x' | b'y')
    }

    /// Does the buffer end with `suffix`?
    fn ends_with(&self, suffix: &str) -> bool {
        self.b.ends_with(suffix.as_bytes())
    }

    /// Replace the trailing `suffix` with `repl` only if the stem before it has measure
    /// strictly greater than `min_measure`. Returns whether it fired.
    fn replace_if(&mut self, suffix: &str, repl: &str, min_measure: usize) -> bool {
        if !self.ends_with(suffix) {
            return false;
        }
        let stem_len = self.b.len() - suffix.len();
        if self.measure(stem_len) <= min_measure {
            return false;
        }
        self.b.truncate(stem_len);
        self.b.extend_from_slice(repl.as_bytes());
        true
    }

    /// Step 1a: plural-style `-s` endings.
    fn step1ab(&mut self) {
        if self.ends_with("sses") {
            let n = self.b.len() - 2;
            self.b.truncate(n); // sses → ss
        } else if self.ends_with("ies") {
            let n = self.b.len() - 2;
            self.b.truncate(n); // ies → i
        } else if self.b.ends_with(b"s") && !self.b.ends_with(b"ss") {
            self.b.pop(); // s → (drop), but not ss
        }

        // Step 1b: `-ed` / `-ing`. `eed` is matched first and is mutually exclusive with
        // the `ed`/`ing` branch (so "feed", with m==0 before "eed", stays "feed" rather
        // than falling through to the `ed` rule).
        let mut fixup = false;
        if self.ends_with("eed") {
            self.replace_if("eed", "ee", 0); // eed → ee only when m > 0; never a fixup
        } else if self.ends_with("ed") && {
            let stem_len = self.b.len() - 2;
            self.has_vowel(stem_len)
        } {
            let n = self.b.len() - 2;
            self.b.truncate(n);
            fixup = true;
        } else if self.ends_with("ing") && {
            let stem_len = self.b.len() - 3;
            self.has_vowel(stem_len)
        } {
            let n = self.b.len() - 3;
            self.b.truncate(n);
            fixup = true;
        }

        if fixup {
            if self.ends_with("at") || self.ends_with("bl") || self.ends_with("iz") {
                self.b.push(b'e'); // at→ate, bl→ble, iz→ize
            } else if self.ends_double_consonant(self.b.len())
                && !matches!(self.b.last(), Some(b'l') | Some(b's') | Some(b'z'))
            {
                self.b.pop(); // collapse the doubled consonant
            } else if self.measure(self.b.len()) == 1 && self.cvc(self.b.len()) {
                self.b.push(b'e'); // short word: restore trailing e (e.g. fil → file)
            }
        }
    }

    /// Step 1c: terminal `y` → `i` when the stem contains a vowel.
    fn step1c(&mut self) {
        if self.b.ends_with(b"y") {
            let stem_len = self.b.len() - 1;
            if self.has_vowel(stem_len) {
                self.b[stem_len] = b'i';
            }
        }
    }

    /// Step 2: map double suffixes to single ones when `m > 0`.
    fn step2(&mut self) {
        const PAIRS: &[(&str, &str)] = &[
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("bli", "ble"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
            ("logi", "log"),
        ];
        self.first_match(PAIRS, |m| m > 0);
    }

    /// Step 3: strip/shorten `-icate`, `-ative`, … when `m > 0`.
    fn step3(&mut self) {
        const PAIRS: &[(&str, &str)] = &[
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ];
        self.first_match(PAIRS, |m| m > 0);
    }

    /// Step 4: remove `-ant`, `-ence`, … when `m > 1`.
    fn step4(&mut self) {
        // `-ion` only after `s` or `t`; handled specially below.
        const SUFFIXES: &[&str] = &[
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ou",
            "ism", "ate", "iti", "ous", "ive", "ize",
        ];
        for suf in SUFFIXES {
            if self.replace_if(suf, "", 1) {
                return;
            }
        }
        if self.ends_with("ion") {
            let stem_len = self.b.len() - 3;
            if self.measure(stem_len) > 1
                && matches!(self.b.get(stem_len - 1), Some(b's') | Some(b't'))
            {
                self.b.truncate(stem_len);
            }
        }
    }

    /// Step 5a/5b: drop a final `e` (m>1, or m==1 and not `*o`), and collapse `-ll` to
    /// `-l` when `m > 1`.
    fn step5(&mut self) {
        if self.b.ends_with(b"e") {
            let stem_len = self.b.len() - 1;
            let m = self.measure(stem_len);
            if m > 1 || (m == 1 && !self.cvc(stem_len)) {
                self.b.truncate(stem_len);
            }
        }
        if self.b.ends_with(b"ll") && self.measure(self.b.len()) > 1 {
            self.b.pop();
        }
    }

    /// Apply the first matching `(suffix, replacement)` pair whose stem satisfies
    /// `cond(m)`; at most one fires (Porter's step structure).
    fn first_match(&mut self, pairs: &[(&str, &str)], cond: impl Fn(usize) -> bool) {
        for (suf, repl) in pairs {
            if self.ends_with(suf) {
                let stem_len = self.b.len() - suf.len();
                if cond(self.measure(stem_len)) {
                    self.b.truncate(stem_len);
                    self.b.extend_from_slice(repl.as_bytes());
                }
                return; // a suffix matched (fired or not) — Porter stops at the first
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_splits_and_lowercases() {
        assert_eq!(
            tokenize("Hello, World! 123-foo"),
            vec!["hello", "world", "123", "foo"]
        );
        assert!(tokenize("   ").is_empty());
    }

    #[test]
    fn stopwords_are_dropped() {
        let terms = analyze("The quick brown fox and the lazy dog", Language::English);
        // "the", "and" are stopwords; the rest stem to themselves here.
        assert!(!terms.iter().any(|t| t == "the" || t == "and"));
        assert!(terms.contains(&"quick".to_string()));
        assert!(terms.contains(&"brown".to_string()));
    }

    #[test]
    fn porter_canonical_examples() {
        // From Porter's paper / reference vocabulary.
        let cases = [
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("caress", "caress"),
            ("cats", "cat"),
            ("feed", "feed"),
            ("agreed", "agre"),
            ("plastered", "plaster"),
            ("motoring", "motor"),
            ("sing", "sing"),
            ("conflated", "conflat"),
            ("troubling", "troubl"),
            ("sized", "size"),
            ("hopping", "hop"),
            ("falling", "fall"),
            ("hissing", "hiss"),
            ("fizzed", "fizz"),
            ("failing", "fail"),
            ("filing", "file"),
            ("happy", "happi"),
            ("relational", "relat"),
            ("conditional", "condit"),
            ("rational", "ration"),
            ("vileness", "vile"),
            ("analogousli", "analog"),
            ("triplicate", "triplic"),
            ("formative", "form"),
            ("electriciti", "electr"),
            ("hopeful", "hope"),
            ("goodness", "good"),
            ("revival", "reviv"),
            ("allowance", "allow"),
            ("inference", "infer"),
            ("adjustable", "adjust"),
            ("defensible", "defens"),
            ("homologou", "homolog"),
            ("effective", "effect"),
            ("bowdlerize", "bowdler"),
            ("probate", "probat"),
            ("controll", "control"),
            ("roll", "roll"),
        ];
        for (input, want) in cases {
            assert_eq!(stem(input), want, "stem({input})");
        }
    }

    #[test]
    fn stem_leaves_short_and_nonascii_words() {
        assert_eq!(stem("at"), "at"); // too short
        assert_eq!(stem("42"), "42"); // not letters
        assert_eq!(stem("café"), "café"); // non-ascii passes through
    }

    #[test]
    fn running_matches_run_family() {
        // The headline requirement: inflections collapse to a shared stem.
        let r = stem("running");
        assert_eq!(r, stem("run"));
        assert_eq!(stem("runs"), stem("run"));
    }

    #[test]
    fn analyze_is_query_index_symmetric() {
        let doc = analyze("The cats were running quickly", Language::English);
        let query = analyze("run cat", Language::English);
        for q in &query {
            assert!(
                doc.contains(q),
                "query term {q} should match an indexed term"
            );
        }
    }
}
