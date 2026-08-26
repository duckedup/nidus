//! Pure-Rust text chunking for the memory layer (nidus-lvo.1): split a document into
//! ordered, source-exact spans instead of embedding it as one vector that averages away
//! whatever the caller will later search for. Ungated: no new dependency, so it ships to
//! every `cargo add nidus`, not just a feature-gated edge (`src/tune.rs` sets the
//! precedent).
//!
//! Sizes are **characters**, not bytes or tokens: nidus does not tokenize for a model it
//! does not own. `char` boundaries are not grapheme boundaries — a ZWJ emoji sequence or a
//! combining accent can be split between `char`s. That is accepted: grapheme segmentation
//! needs `unicode-segmentation`, a dependency this module does not take.
//!
//! **The load-bearing invariant**: every [`Chunk`] is an exact char slice of the source.
//! With `src: Vec<char> = text.chars().collect()`, for every emitted `c`:
//! `src[c.char_start .. c.char_start + c.text.chars().count()] == c.text.chars()`.
//! Trimming, where it happens, narrows the span — it never edits characters.

#[cfg(feature = "code")]
pub(crate) mod code;
mod markdown;
mod recursive;
mod sentence;

use anyhow::{Result, bail};

/// How [`chunk_text`] divides a document into [`Chunk`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkStrategy {
    #[default]
    Recursive,
    Markdown,
    Sentence,
    /// AST-aware: one chunk per symbol, from wdpkr-core's tree-sitter chunker.
    #[cfg(feature = "code")]
    Code,
}

/// Chunking parameters. `max_chars`/`overlap_chars` count `char`s, not bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkOpts {
    pub strategy: ChunkStrategy,
    pub max_chars: usize,
    pub overlap_chars: usize,
    /// The language [`ChunkStrategy::Code`] parses `text` as: a `detect_language` name such
    /// as `"rust"`. `None` tries every grammar and keeps the best guess, at one parse per
    /// language (nidus-61d). Other strategies ignore it.
    pub language: Option<&'static str>,
}

impl Default for ChunkOpts {
    fn default() -> Self {
        Self {
            strategy: ChunkStrategy::Recursive,
            max_chars: 1000,
            overlap_chars: 100,
            language: None,
        }
    }
}

/// One emitted span. `text` is an exact char-slice of the source: content is never
/// rewritten, only narrowed. See [`chunk_text`] for the full invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
    /// 0-based, dense, in source order.
    pub index: usize,
    /// Char (not byte) offset into the source.
    pub char_start: usize,
}

/// Splits `text` into ordered [`Chunk`]s per `opts` (see the module docs for the char
/// slice invariant every chunk satisfies). `Err` on invalid options; `Ok(vec![])` for
/// empty or all-whitespace input.
pub fn chunk_text(text: &str, opts: &ChunkOpts) -> Result<Vec<Chunk>> {
    if opts.max_chars == 0 {
        bail!("chunk_text: max_chars must be > 0");
    }
    if opts.overlap_chars >= opts.max_chars {
        let (overlap, max) = (opts.overlap_chars, opts.max_chars);
        bail!(
            "chunk_text: overlap_chars ({overlap}) must be less than max_chars ({max}); at \
             or above the budget, forward progress is zero and the splitter never terminates"
        );
    }
    if text.trim().is_empty() {
        return Ok(vec![]);
    }

    let src: Vec<char> = text.chars().collect();
    let (spans, floors) = match opts.strategy {
        ChunkStrategy::Recursive => (recursive::split(&src, 0, src.len(), opts.max_chars), None),
        ChunkStrategy::Markdown => {
            let (spans, floors) = markdown::split(&src, opts.max_chars);
            (spans, Some(floors))
        }
        ChunkStrategy::Sentence => (sentence::split(&src, opts.max_chars), None),
        #[cfg(feature = "code")]
        ChunkStrategy::Code => (code::split(&src, opts.max_chars, opts.language), None),
    };
    Ok(apply_overlap(
        &src,
        &spans,
        opts.overlap_chars,
        floors.as_deref(),
    ))
}

/// Greedily merges adjacent, contiguous char ranges into groups no larger than
/// `max_chars`. Assumes each input piece already fits within `max_chars` alone.
fn pack(pieces: &[(usize, usize)], max_chars: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut group: Option<(usize, usize)> = None;
    for &(ps, pe) in pieces {
        group = Some(match group {
            None => (ps, pe),
            Some((gs, _)) if pe - gs <= max_chars => (gs, pe),
            Some((gs, ge)) => {
                out.push((gs, ge));
                (ps, pe)
            }
        });
    }
    if let Some(g) = group {
        out.push(g);
    }
    out
}

/// Backward overlap per span after the first, clamped so it never reaches past the
/// preceding chunk's own start. `floors[i]` raises that bound further — markdown passes
/// each span's section start, so overlap never crosses a heading.
fn apply_overlap(
    src: &[char],
    spans: &[(usize, usize)],
    overlap_chars: usize,
    floors: Option<&[usize]>,
) -> Vec<Chunk> {
    let mut out = Vec::with_capacity(spans.len());
    let mut prev_original_start = 0usize;
    for (i, &(s, e)) in spans.iter().enumerate() {
        let floor = floors.and_then(|f| f.get(i).copied()).unwrap_or(0);
        let actual_start = if i == 0 || overlap_chars == 0 {
            s
        } else {
            s.saturating_sub(overlap_chars)
                .max(prev_original_start)
                .max(floor)
        };
        let text: String = src[actual_start..e].iter().collect();
        out.push(Chunk {
            text,
            index: i,
            char_start: actual_start,
        });
        prev_original_start = s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "code"))]
    const STRATEGIES: [ChunkStrategy; 3] = [
        ChunkStrategy::Recursive,
        ChunkStrategy::Markdown,
        ChunkStrategy::Sentence,
    ];
    #[cfg(feature = "code")]
    const STRATEGIES: [ChunkStrategy; 4] = [
        ChunkStrategy::Recursive,
        ChunkStrategy::Markdown,
        ChunkStrategy::Sentence,
        ChunkStrategy::Code,
    ];

    /// `Code` needs a real language grammar to find symbols, so the plain-prose fixtures
    /// these length-sensitive tests share across strategies legitimately yield none from it.
    fn is_code_strategy(_strategy: ChunkStrategy) -> bool {
        #[cfg(feature = "code")]
        {
            _strategy == ChunkStrategy::Code
        }
        #[cfg(not(feature = "code"))]
        {
            false
        }
    }

    fn opts(strategy: ChunkStrategy, max_chars: usize, overlap_chars: usize) -> ChunkOpts {
        ChunkOpts {
            strategy,
            max_chars,
            overlap_chars,
            ..Default::default()
        }
    }

    /// The load-bearing invariant: every chunk is an exact char slice of the source.
    fn assert_char_slice_invariant(text: &str, chunks: &[Chunk]) {
        let src: Vec<char> = text.chars().collect();
        for c in chunks {
            let len = c.text.chars().count();
            assert!(
                c.char_start + len <= src.len(),
                "chunk {c:?} runs past the source"
            );
            let slice: String = src[c.char_start..c.char_start + len].iter().collect();
            assert_eq!(slice, c.text, "chunk {c:?} is not an exact source slice");
        }
    }

    fn assert_dense_ascending(chunks: &[Chunk]) {
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index, i);
        }
        for w in chunks.windows(2) {
            assert!(w[0].char_start <= w[1].char_start);
        }
    }

    #[test]
    fn markdown_overlap_never_crosses_a_heading() {
        let doc = format!(
            "# Heading One\n{}\n\n# Heading Two\n{}",
            "x".repeat(480),
            "y".repeat(480)
        );
        let opts = ChunkOpts {
            strategy: ChunkStrategy::Markdown,
            max_chars: 1000,
            overlap_chars: 100,
            ..Default::default()
        };
        let chunks = chunk_text(&doc, &opts).unwrap();
        assert!(
            chunks.len() >= 2,
            "expected a chunk per section: {chunks:?}"
        );
        let second = &chunks[1];
        assert!(
            !second.text.contains('x'),
            "chunk 2 bled into section one: starts {:?}",
            &second.text[..second.text.len().min(40)]
        );
        assert!(
            second.text.starts_with("# Heading Two"),
            "chunk 2 must begin at its own heading, got {:?}",
            &second.text[..second.text.len().min(40)]
        );
    }

    #[test]
    fn char_slice_invariant_holds_for_every_strategy() {
        let long_words = "a b c d e f g h i j k l m n o p q r s t u v w x y z. ".repeat(20);
        let fixtures = [
            "short",
            long_words.as_str(),
            "# Heading\n\nSome text here.\n\n## Sub\n\nMore text follows along nicely.",
            "Sentence one. Sentence two! Sentence three? Sentence four.",
            "emoji test \u{1F642}\u{1F642}\u{1F642}\u{1F642} caf\u{e9} na\u{ef}ve 日本語 テスト 文字列 chunking",
        ];
        for text in fixtures {
            for strategy in STRATEGIES {
                let o = opts(strategy, 20, 5);
                let chunks = chunk_text(text, &o).unwrap();
                assert_char_slice_invariant(text, &chunks);
                assert_dense_ascending(&chunks);
            }
        }
    }

    #[test]
    fn short_input_is_one_chunk_at_zero() {
        for strategy in STRATEGIES {
            if is_code_strategy(strategy) {
                continue;
            }
            let o = opts(strategy, 1000, 100);
            let chunks = chunk_text("hello world", &o).unwrap();
            assert_eq!(chunks.len(), 1);
            assert_eq!(chunks[0].index, 0);
            assert_eq!(chunks[0].char_start, 0);
            assert_eq!(chunks[0].text, "hello world");
        }
    }

    #[test]
    fn empty_and_whitespace_only_input_yields_no_chunks() {
        for strategy in STRATEGIES {
            let o = opts(strategy, 100, 10);
            assert!(chunk_text("", &o).unwrap().is_empty());
            assert!(chunk_text("   \n\t  ", &o).unwrap().is_empty());
        }
    }

    #[test]
    fn single_char_input() {
        for strategy in STRATEGIES {
            if is_code_strategy(strategy) {
                continue;
            }
            let o = opts(strategy, 100, 10);
            let chunks = chunk_text("x", &o).unwrap();
            assert_eq!(chunks.len(), 1);
            assert_eq!(chunks[0].text, "x");
            assert_eq!(chunks[0].char_start, 0);
        }
    }

    #[test]
    fn zero_max_chars_errors() {
        let o = opts(ChunkStrategy::Recursive, 0, 0);
        assert!(chunk_text("hello", &o).is_err());
    }

    #[test]
    fn overlap_at_or_above_budget_errors() {
        let o = opts(ChunkStrategy::Recursive, 10, 10);
        assert!(chunk_text("hello world this is long enough", &o).is_err());
        let o2 = opts(ChunkStrategy::Recursive, 10, 20);
        assert!(chunk_text("hello world this is long enough", &o2).is_err());
    }

    #[test]
    fn overlap_zero_is_contiguous_non_overlapping() {
        let text = "a".repeat(97);
        for strategy in STRATEGIES {
            if is_code_strategy(strategy) {
                continue;
            }
            let o = opts(strategy, 25, 0);
            let chunks = chunk_text(&text, &o).unwrap();
            assert!(chunks.len() > 1);
            for w in chunks.windows(2) {
                let prev_end = w[0].char_start + w[0].text.chars().count();
                assert_eq!(prev_end, w[1].char_start);
            }
        }
    }

    #[test]
    fn overlap_n_starts_n_chars_before_previous_end() {
        let text = "a".repeat(97);
        for strategy in STRATEGIES {
            if is_code_strategy(strategy) {
                continue;
            }
            let o = opts(strategy, 25, 7);
            let chunks = chunk_text(&text, &o).unwrap();
            assert!(chunks.len() > 1);
            for w in chunks.windows(2) {
                let prev_end = w[0].char_start + w[0].text.chars().count();
                assert_eq!(prev_end.saturating_sub(w[1].char_start), 7);
            }
        }
    }

    #[test]
    fn multibyte_utf8_respects_char_budget() {
        let text = "日本語のテキストです。".repeat(10)
            + "\u{1F642}\u{1F642}\u{1F642}\u{1F642}\u{1F642}\u{1F642}\u{1F642}\u{1F642}";
        for strategy in STRATEGIES {
            let o = opts(strategy, 15, 3);
            let chunks = chunk_text(&text, &o).unwrap();
            assert_char_slice_invariant(&text, &chunks);
            for c in &chunks {
                assert!(c.text.chars().count() <= o.max_chars + o.overlap_chars);
            }
        }
    }

    // ── ChunkStrategy::Code, through the public entry point ────────────────────────────

    #[cfg(feature = "code")]
    #[test]
    fn code_strategy_splits_rust_functions_by_symbol() {
        let src = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n\
                    fn sub(a: i32, b: i32) -> i32 {\n    a - b\n}\n";
        let chunks = chunk_text(src, &opts(ChunkStrategy::Code, 1000, 0)).unwrap();
        assert_eq!(chunks.len(), 2, "chunks: {chunks:?}");
        assert!(
            chunks[0].text.starts_with("fn add"),
            "got {:?}",
            chunks[0].text
        );
        assert!(
            chunks[1].text.starts_with("fn sub"),
            "got {:?}",
            chunks[1].text
        );
        assert_char_slice_invariant(src, &chunks);
        assert_dense_ascending(&chunks);
    }

    // wdpkr-core 0.2.0's `SymbolChunk::body` is the item's own text only; a preceding doc
    // comment lands in the separate `doc_comment` field. Asserts the observed behavior:
    // the chunk still slices exactly and starts at the item, not the comment (nidus-3gm).
    #[cfg(feature = "code")]
    #[test]
    fn code_strategy_symbol_with_doc_comment_still_slices_exactly() {
        let src = "/// Adds two numbers.\nfn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let chunks = chunk_text(src, &opts(ChunkStrategy::Code, 1000, 0)).unwrap();
        assert_eq!(chunks.len(), 1, "chunks: {chunks:?}");
        assert!(
            chunks[0].text.starts_with("fn add"),
            "got {:?}",
            chunks[0].text
        );
        assert_char_slice_invariant(src, &chunks);
    }

    /// nidus-61d: a caller that already knows the language hands it over instead of paying
    /// one parse per grammar. Same input, same chunks — the field is a shortcut, not a
    /// different splitter.
    #[cfg(feature = "code")]
    #[test]
    fn code_strategy_with_a_supplied_language_matches_the_guess() {
        let src = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n\
                    fn sub(a: i32, b: i32) -> i32 {\n    a - b\n}\n";
        let guessed = chunk_text(src, &opts(ChunkStrategy::Code, 1000, 0)).unwrap();
        let told = chunk_text(
            src,
            &ChunkOpts {
                language: Some("rust"),
                ..opts(ChunkStrategy::Code, 1000, 0)
            },
        )
        .unwrap();
        assert_eq!(told, guessed, "told: {told:?}");
        assert_eq!(told.len(), 2, "told: {told:?}");
    }

    /// The assertion that would fail if `split` accepted `language` and then guessed anyway:
    /// Rust source parsed as Python finds no symbols. Without it, every test above would pass
    /// against an implementation that ignored the field entirely.
    #[cfg(feature = "code")]
    #[test]
    fn code_strategy_uses_the_supplied_language_rather_than_the_best_guess() {
        let src = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n\
                    fn sub(a: i32, b: i32) -> i32 {\n    a - b\n}\n";
        assert_eq!(
            chunk_text(src, &opts(ChunkStrategy::Code, 1000, 0))
                .unwrap()
                .len(),
            2,
            "precondition: guessing finds this Rust"
        );
        let chunks = chunk_text(
            src,
            &ChunkOpts {
                language: Some("python"),
                ..opts(ChunkStrategy::Code, 1000, 0)
            },
        )
        .unwrap();
        assert!(
            chunks.is_empty(),
            "a supplied language must be used, not overridden by a better guess: {chunks:?}"
        );
    }

    /// `detect_language` recognises extensions wdpkr ships no grammar for (svelte, for one).
    /// Naming one yields nothing rather than silently falling back to the guess, which would
    /// chunk a Svelte file as whichever language scored highest.
    #[cfg(feature = "code")]
    #[test]
    fn code_strategy_with_a_language_that_has_no_grammar_yields_no_chunks() {
        let src = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let chunks = chunk_text(
            src,
            &ChunkOpts {
                language: Some("svelte"),
                ..opts(ChunkStrategy::Code, 1000, 0)
            },
        )
        .unwrap();
        assert!(chunks.is_empty(), "chunks: {chunks:?}");
    }

    /// `language` is meaningless outside `ChunkStrategy::Code`, and must stay inert there
    /// rather than quietly changing how prose splits.
    #[cfg(feature = "code")]
    #[test]
    fn a_supplied_language_is_ignored_by_the_prose_strategies() {
        let doc = "First sentence here. Second sentence here. Third one too.";
        for strategy in [
            ChunkStrategy::Recursive,
            ChunkStrategy::Markdown,
            ChunkStrategy::Sentence,
        ] {
            let plain = chunk_text(doc, &opts(strategy, 30, 0)).unwrap();
            let with_lang = chunk_text(
                doc,
                &ChunkOpts {
                    language: Some("rust"),
                    ..opts(strategy, 30, 0)
                },
            )
            .unwrap();
            assert_eq!(plain, with_lang, "{strategy:?} must ignore `language`");
        }
    }

    #[cfg(feature = "code")]
    #[test]
    fn code_strategy_with_no_recognizable_grammar_yields_no_chunks() {
        let src = "(define (add a b) (+ a b))\n(display (add 1 2))\n";
        let chunks = chunk_text(src, &opts(ChunkStrategy::Code, 1000, 0)).unwrap();
        assert!(chunks.is_empty(), "chunks: {chunks:?}");
    }

    #[cfg(feature = "code")]
    #[test]
    fn code_strategy_splits_an_oversized_symbol_but_keeps_exact_slices() {
        let body_lines = "    let _ = 1 + 1;\n".repeat(2000);
        let src = format!("fn big() {{\n{body_lines}}}\n");
        let chunks = chunk_text(&src, &opts(ChunkStrategy::Code, 2000, 0)).unwrap();
        assert!(
            chunks.len() > 1,
            "expected the oversized symbol to split: {} chunks",
            chunks.len()
        );
        assert_char_slice_invariant(&src, &chunks);
        assert_dense_ascending(&chunks);
    }
}
