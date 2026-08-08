//! The text analyzer: raw text → normalized terms for BM25 indexing and querying.

use serde::{Deserialize, Serialize};

use super::fold::fold_ascii;

/// The analyzer language for a full-text field. Extensible; only English is implemented
/// today (the variant gates the stopword set + stemmer in [`analyze`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    /// US English: lowercase, English stopwords, Porter stemming.
    #[default]
    #[serde(alias = "english", alias = "en")]
    English,
}

/// How one full-text field turns text into terms — applied identically at index and query
/// time, so a query term can only match a stored term when both were analyzed the same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Analyzer {
    /// Picks the stopword set and stemmer.
    pub language: Language,
    /// Fold Latin diacritics to ASCII before stemming ("café" → "cafe").
    pub ascii_folding: bool,
    /// Drop tokens longer than this many chars. `None` keeps every token.
    pub max_token_len: Option<usize>,
}

impl Default for Analyzer {
    /// US English, no folding, no length cap — the behaviour of every release before the
    /// analyzer became configurable.
    fn default() -> Self {
        Self {
            language: Language::default(),
            ascii_folding: false,
            max_token_len: None,
        }
    }
}

impl Analyzer {
    /// Set the language.
    pub fn language(mut self, language: Language) -> Self {
        self.language = language;
        self
    }

    /// Turn ASCII folding on or off.
    pub fn ascii_folding(mut self, on: bool) -> Self {
        self.ascii_folding = on;
        self
    }

    /// Drop tokens longer than `chars`.
    pub fn max_token_len(mut self, chars: usize) -> Self {
        self.max_token_len = Some(chars);
        self
    }
}

/// One analyzed term plus the byte range of the **original** text it came from. Stemming
/// and folding mean the term is usually *not* a substring of that range, which is exactly
/// why the range is carried rather than re-found by searching for the term (nidus-m50.5).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TermSpan {
    pub(crate) term: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// Analyze `text` into a sequence of normalized terms (in document order, duplicates
/// kept so term frequencies are countable). Empty input → no terms.
pub(crate) fn analyze(text: &str, cfg: Analyzer) -> Vec<String> {
    analyze_spans(text, cfg)
        .into_iter()
        .map(|t| t.term)
        .collect()
}

/// [`analyze`], keeping each surviving term's byte range in `text`. The one analysis path —
/// `analyze` is this, with the offsets dropped — so a highlighter can never disagree with
/// the index about what was a term.
pub(crate) fn analyze_spans(text: &str, cfg: Analyzer) -> Vec<TermSpan> {
    let tokens = tokenize(text, cfg.ascii_folding, cfg.max_token_len);
    match cfg.language {
        Language::English => tokens
            .into_iter()
            .filter(|t| !is_stopword(&t.term))
            .map(|t| TermSpan {
                term: stem(&t.term),
                ..t
            })
            .filter(|t| !t.term.is_empty())
            .collect(),
    }
}

/// Split `text` into lowercased tokens on runs of Unicode alphanumerics, everything else being a
/// separator. Lowercasing is std's `char::to_lowercase`, which covers the Latin script we target —
/// a pragmatic stand-in for full UAX #29 segmentation that stays dependency-free.
fn tokenize(text: &str, ascii_folding: bool, max_token_len: Option<usize>) -> Vec<TermSpan> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut start = 0usize;
    // The length cap counts the token as the text held it, before folding can expand it
    // ("ß" → "ss"), so the cap means the same thing whether folding is on or off.
    let mut push = |cur: &mut String, start: usize, end: usize| {
        let token = std::mem::take(cur);
        if max_token_len.is_some_and(|max| token.chars().count() > max) {
            return;
        }
        out.push(TermSpan {
            term: if ascii_folding {
                fold_ascii(&token)
            } else {
                token
            },
            start,
            end,
        });
    };
    for (i, ch) in text.char_indices() {
        if ch.is_alphanumeric() {
            if cur.is_empty() {
                start = i;
            }
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            push(&mut cur, start, i);
        }
    }
    if !cur.is_empty() {
        push(&mut cur, start, text.len());
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

    /// Tokenize, keeping only the token text.
    fn terms(text: &str, ascii_folding: bool, max_token_len: Option<usize>) -> Vec<String> {
        tokenize(text, ascii_folding, max_token_len)
            .into_iter()
            .map(|t| t.term)
            .collect()
    }

    /// Tokenize with today's defaults (no folding, no length cap).
    fn tok(text: &str) -> Vec<String> {
        terms(text, false, None)
    }

    #[test]
    fn tokenize_splits_and_lowercases() {
        assert_eq!(
            tok("Hello, World! 123-foo"),
            vec!["hello", "world", "123", "foo"]
        );
        assert!(tok("   ").is_empty());
    }

    #[test]
    fn ascii_folding_is_off_by_default_and_folds_when_on() {
        assert_eq!(tok("Café RÉSUMÉ"), vec!["café", "résumé"]);
        assert_eq!(terms("Café RÉSUMÉ", true, None), vec!["cafe", "resume"]);
    }

    #[test]
    fn folded_and_unfolded_spellings_share_a_term() {
        let cfg = Analyzer::default().ascii_folding(true);
        assert_eq!(analyze("café", cfg), analyze("cafe", cfg));
        // Without folding they stay distinct, which is exactly the pre-existing behaviour.
        assert_ne!(
            analyze("café", Analyzer::default()),
            analyze("cafe", Analyzer::default())
        );
    }

    #[test]
    fn max_token_len_drops_only_the_oversized_tokens() {
        assert_eq!(
            terms("ok aaaaaaaa fine", false, Some(4)),
            vec!["ok", "fine"]
        );
        // The cap counts chars, not bytes, so a 4-char accented token survives a cap of 4.
        assert_eq!(terms("héllo", false, Some(4)), Vec::<String>::new());
        assert_eq!(terms("héll", false, Some(4)), vec!["héll"]);
        // And it is measured before folding expands "ß" into two ASCII chars.
        assert_eq!(terms("straße", true, Some(6)), vec!["strasse"]);
    }

    #[test]
    fn stopwords_are_dropped() {
        let terms = analyze("The quick brown fox and the lazy dog", Analyzer::default());
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
    fn spans_point_at_the_surface_form_not_the_stem() {
        // The property the highlighter rests on: the term is "run", but the range covers
        // the word as the document actually spells it.
        let text = "The developers were running quickly";
        let spans = analyze_spans(text, Analyzer::default());
        let run = spans.iter().find(|t| t.term == "run").unwrap();
        assert_eq!(&text[run.start..run.end], "running");
        let dev = spans.iter().find(|t| t.term == "develop").unwrap();
        assert_eq!(&text[dev.start..dev.end], "developers");
    }

    #[test]
    fn spans_survive_folding_punctuation_and_multibyte_text() {
        let text = "Le café, «RÉSUMÉ» — 42!";
        let cfg = Analyzer::default().ascii_folding(true);
        let spans = analyze_spans(text, cfg);
        let at = |term: &str| {
            let t = spans.iter().find(|t| t.term == term).unwrap();
            &text[t.start..t.end]
        };
        assert_eq!(at("cafe"), "café");
        assert_eq!(at("resum"), "RÉSUMÉ");
        assert_eq!(at("42"), "42");
        // Dropping the offsets reproduces `analyze` exactly.
        let terms: Vec<String> = spans.into_iter().map(|t| t.term).collect();
        assert_eq!(terms, analyze(text, cfg));
    }

    #[test]
    fn analyze_is_query_index_symmetric() {
        let doc = analyze("The cats were running quickly", Analyzer::default());
        let query = analyze("run cat", Analyzer::default());
        for q in &query {
            assert!(
                doc.contains(q),
                "query term {q} should match an indexed term"
            );
        }
    }
}
