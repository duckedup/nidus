//! Adapter from wdpkr-core's AST-aware symbol chunker to nidus's [`Chunk`] shape. Gated
//! behind the `code` feature; nothing here is reachable, or built, in the default lane.
//!
//! `chunk_text`'s public entry carries only `src` and `max_chars`, no file path, so no
//! [`wdpkr_core::chunk::detect_language`] input reaches this module. `split` tries every
//! grammar wdpkr-core ships and keeps whichever finds the most symbols: real source
//! reliably matches exactly one grammar, and plain prose matches none.

use wdpkr_core::chunk::tree_sitter::TreeSitterChunker;
use wdpkr_core::chunk::{Chunker, SymbolChunk};

use super::recursive;

/// Keep an oversized symbol as one chunk up to this many chars; a generated match arm or a
/// minified blob beyond it falls back to `recursive::split` so no chunk balloons unbounded.
const OVERSIZED_SYMBOL_CAP: usize = 20_000;

/// Every language wdpkr-core's tree-sitter chunker recognises (`languages::get_config`'s
/// keys), minus the `"c"` alias: `get_config` maps it to the same grammar as `"cpp"`.
const LANGUAGES: &[&str] = &[
    "rust",
    "go",
    "typescript",
    "tsx",
    "javascript",
    "python",
    "java",
    "cpp",
    "csharp",
];

/// Splits `src` into per-symbol char ranges. `vec![]` when no grammar recognises the
/// content — an unrecognised language and a parse failure look the same from here.
pub(super) fn split(src: &[char], max_chars: usize) -> Vec<(usize, usize)> {
    let content: String = src.iter().collect();
    let chunker = TreeSitterChunker::new();

    let mut symbols = LANGUAGES
        .iter()
        .filter_map(|&lang| chunker.chunk("", &content, lang).ok())
        .max_by_key(|fc| fc.symbols.len())
        .map(|fc| fc.symbols)
        .unwrap_or_default();
    if symbols.is_empty() {
        return Vec::new();
    }
    symbols.sort_by_key(|s| s.start_line);

    let line_starts = line_start_table(src);
    let mut spans = Vec::with_capacity(symbols.len());
    for sym in &symbols {
        let Some(start) = locate_symbol(src, &line_starts, sym) else {
            continue;
        };
        let end = start + sym.body.chars().count();
        if end - start <= OVERSIZED_SYMBOL_CAP {
            spans.push((start, end));
        } else {
            spans.extend(recursive::split(src, start, end, max_chars));
        }
    }
    spans
}

/// `line_starts[i]` is the char offset where 1-based source line `i + 1` begins.
fn line_start_table(src: &[char]) -> Vec<usize> {
    let mut starts = vec![0];
    starts.extend(
        src.iter()
            .enumerate()
            .filter_map(|(i, &c)| (c == '\n').then_some(i + 1)),
    );
    starts
}

/// Locates `sym.body`'s char offset: finds its first line inside the char range of
/// `sym.start_line`, then verifies the whole body slices there exactly. `None` — skip the
/// symbol, never emit a bad span — if the grammar's line doesn't actually hold the body.
fn locate_symbol(src: &[char], line_starts: &[usize], sym: &SymbolChunk) -> Option<usize> {
    let line_idx = (sym.start_line as usize).checked_sub(1)?;
    let line_start = *line_starts.get(line_idx)?;
    let line_end = line_starts.get(line_idx + 1).copied().unwrap_or(src.len());

    let first_line: Vec<char> = sym.body.split('\n').next().unwrap_or("").chars().collect();
    if first_line.is_empty() {
        return None;
    }
    let window_end = line_end.max(line_start + first_line.len()).min(src.len());
    let offset = src
        .get(line_start..window_end)?
        .windows(first_line.len())
        .position(|w| w == first_line.as_slice())?;
    let start = line_start + offset;

    let body_len = sym.body.chars().count();
    let end = start.checked_add(body_len)?;
    let matches = end <= src.len() && src[start..end].iter().copied().eq(sym.body.chars());
    debug_assert!(
        matches,
        "wdpkr reported line {} for {:?} but the body does not slice there",
        sym.start_line, sym.name
    );
    matches.then_some(start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_grammar_recognises_the_content_yields_no_spans() {
        let src: Vec<char> = "(define (add a b) (+ a b))\n".chars().collect();
        assert!(split(&src, 1000).is_empty());
    }

    #[test]
    fn two_rust_functions_yield_two_exact_spans() {
        let text = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n\
                     fn sub(a: i32, b: i32) -> i32 {\n    a - b\n}\n";
        let src: Vec<char> = text.chars().collect();
        let spans = split(&src, 1000);
        assert_eq!(spans.len(), 2, "spans: {spans:?}");
        for &(s, e) in &spans {
            let slice: String = src[s..e].iter().collect();
            assert!(slice.starts_with("fn "), "slice: {slice:?}");
        }
    }

    #[test]
    fn oversized_symbol_falls_back_to_recursive_split() {
        let body_lines = "    let _ = 1 + 1;\n".repeat(2000);
        let text = format!("fn big() {{\n{body_lines}}}\n");
        let src: Vec<char> = text.chars().collect();
        let spans = split(&src, 2000);
        assert!(
            spans.len() > 1,
            "expected the oversized symbol to split: {} pieces",
            spans.len()
        );
        for &(s, e) in &spans {
            assert!(e <= src.len(), "span {s}..{e} runs past the source");
        }
    }
}
