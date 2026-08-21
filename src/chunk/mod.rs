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
}

/// Chunking parameters. `max_chars`/`overlap_chars` count `char`s, not bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkOpts {
    pub strategy: ChunkStrategy,
    pub max_chars: usize,
    pub overlap_chars: usize,
}

impl Default for ChunkOpts {
    fn default() -> Self {
        Self {
            strategy: ChunkStrategy::Recursive,
            max_chars: 1000,
            overlap_chars: 100,
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

    const STRATEGIES: [ChunkStrategy; 3] = [
        ChunkStrategy::Recursive,
        ChunkStrategy::Markdown,
        ChunkStrategy::Sentence,
    ];

    fn opts(strategy: ChunkStrategy, max_chars: usize, overlap_chars: usize) -> ChunkOpts {
        ChunkOpts {
            strategy,
            max_chars,
            overlap_chars,
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
}
