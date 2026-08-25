//! The `code` engine: per-file chunk-strategy dispatch ([`dispatch`]), the summarize wiring
//! onto wdpkr-core's code-summarization prompts, and file-grouped presentation
//! ([`present`]). **This is the only directory in nidus allowed to `use wdpkr_core`** — see
//! `BLUEPRINT-nidus-3gm.md`.
//!
//! `wdpkr_core::indexer` is never called: `nidus ingest` already owns the walk, the
//! per-file digest skip, provenance, the stale-tail prune and the group-commit barrier
//! (`src/cli/ingest.rs`). Running wdpkr's own walker alongside it would give one corpus two
//! idempotence schemes.

pub mod dispatch;
pub mod present;

use std::collections::BTreeMap;

use anyhow::Result;
use wdpkr_core::chunk::{Chunker, detect_language, tree_sitter::TreeSitterChunker};

use crate::Value;
use crate::chunk::{ChunkOpts, ChunkStrategy, chunk_text};

/// The file this chunk came from, relative to the ingest root.
pub const META_PATH: &str = "code.path";
/// The symbol name (function, struct, …) this chunk covers, when the file was AST-chunked.
pub const META_SYMBOL: &str = "code.symbol";
/// The symbol's normalized kind (`function`, `struct`, `trait`, …).
pub const META_KIND: &str = "code.kind";
/// 1-based first line of the chunk in its source file.
pub const META_START_LINE: &str = "code.start_line";
/// 1-based last line of the chunk in its source file.
pub const META_END_LINE: &str = "code.end_line";
/// The language [`wdpkr_core::chunk::detect_language`] reported for this file.
pub const META_LANGUAGE: &str = "code.language";

/// One chunked span plus the metadata `code search` presents without re-parsing the file
/// (see the `META_*` constants above; units 4, 5 and the SDKs consume these key names).
#[derive(Debug, Clone, PartialEq)]
pub struct CodeChunk {
    pub text: String,
    pub index: usize,
    pub char_start: usize,
    pub attrs: BTreeMap<String, Value>,
}

/// Chunk `text` (`path`'s contents) per [`dispatch::strategy_for`]. A recognised language
/// gets one [`CodeChunk`] per symbol, tagged with its own metadata; everything else goes
/// through [`crate::chunk::chunk_text`] tagged with `code.path` alone.
pub fn chunk_file(path: &str, text: &str, opts: &ChunkOpts) -> Result<Vec<CodeChunk>> {
    match dispatch::strategy_for(path) {
        ChunkStrategy::Code => Ok(code_chunks(
            path,
            text,
            detect_language(path).unwrap_or_default(),
        )),
        strategy => prose_chunks(path, text, strategy, opts),
    }
}

/// The non-code path: nidus's own splitter, tagged with `code.path` only. No symbol
/// metadata, because a markdown/recursive span is not a symbol.
fn prose_chunks(
    path: &str,
    text: &str,
    strategy: ChunkStrategy,
    opts: &ChunkOpts,
) -> Result<Vec<CodeChunk>> {
    let opts = ChunkOpts {
        strategy,
        ..opts.clone()
    };
    Ok(chunk_text(text, &opts)?
        .into_iter()
        .map(|c| {
            let mut attrs = BTreeMap::new();
            attrs.insert(META_PATH.to_string(), Value::Str(path.to_string()));
            CodeChunk {
                text: c.text,
                index: c.index,
                char_start: c.char_start,
                attrs,
            }
        })
        .collect())
}

/// One [`CodeChunk`] per symbol wdpkr's tree-sitter chunker finds in `text` for `lang`, in
/// source order, each carrying its own name/kind/line-span. A parse failure yields no
/// chunks; a symbol whose body cannot be relocated in `text` is skipped, not guessed.
fn code_chunks(path: &str, text: &str, lang: &str) -> Vec<CodeChunk> {
    let mut symbols = match TreeSitterChunker::new().chunk(path, text, lang) {
        Ok(f) => f.symbols,
        Err(_) => return Vec::new(),
    };
    symbols.sort_by_key(|s| s.start_line);

    let mut out = Vec::with_capacity(symbols.len());
    let mut from_byte = 0usize;
    for sym in &symbols {
        let Some(rel) = text[from_byte..].find(sym.body.as_str()) else {
            continue;
        };
        let byte_start = from_byte + rel;
        let char_start = text[..byte_start].chars().count();
        from_byte = byte_start + sym.body.len();

        let mut attrs = BTreeMap::new();
        attrs.insert(META_PATH.to_string(), Value::Str(path.to_string()));
        attrs.insert(META_LANGUAGE.to_string(), Value::Str(lang.to_string()));
        attrs.insert(META_SYMBOL.to_string(), Value::Str(sym.name.clone()));
        attrs.insert(META_KIND.to_string(), Value::Str(sym.kind.clone()));
        attrs.insert(
            META_START_LINE.to_string(),
            Value::Int(i64::from(sym.start_line)),
        );
        attrs.insert(
            META_END_LINE.to_string(),
            Value::Int(i64::from(sym.end_line)),
        );
        out.push(CodeChunk {
            text: sym.body.clone(),
            index: out.len(),
            char_start,
            attrs,
        });
    }
    out
}

/// [`crate::summarize::SummarizeOpts`] carrying wdpkr-core's file prompt (re-exported here,
/// not copied, so a 0.2.x tune arrives with the lockfile bump) for a call through nidus's
/// own [`crate::summarize::AnySummarizer`], never wdpkr's summarizer.
#[cfg(feature = "summarize")]
pub fn file_summarize_opts(
    input: &wdpkr_core::summarize::FileSummaryInput,
) -> crate::summarize::SummarizeOpts {
    // `content` left empty: wdpkr's `file_user_message` bakes the source in, but nidus's
    // `user_message` also appends the real `text` (passed separately to `summarize()`)
    // after `instructions` — a populated `content` here would send the file twice.
    let lead_only = wdpkr_core::summarize::FileSummaryInput {
        content: String::new(),
        ..input.clone()
    };
    crate::summarize::SummarizeOpts {
        system: Some(wdpkr_core::summarize::prompts::SYSTEM_PROMPT.to_string()),
        instructions: Some(wdpkr_core::summarize::prompts::file_user_message(
            &lead_only,
        )),
        max_tokens: None,
    }
}

/// Symbol-level counterpart to [`file_summarize_opts`]: wdpkr's per-symbol prompt, minus the
/// body (appended separately as the `text` argument to `summarize()`, per the same rule).
#[cfg(feature = "summarize")]
pub fn symbol_summarize_opts(
    input: &wdpkr_core::summarize::SymbolSummaryInput,
) -> crate::summarize::SummarizeOpts {
    let lead_only = wdpkr_core::summarize::SymbolSummaryInput {
        body: String::new(),
        ..input.clone()
    };
    crate::summarize::SummarizeOpts {
        system: Some(wdpkr_core::summarize::prompts::SYSTEM_PROMPT.to_string()),
        instructions: Some(wdpkr_core::summarize::prompts::symbol_user_message(
            &lead_only,
        )),
        max_tokens: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> ChunkOpts {
        ChunkOpts {
            strategy: ChunkStrategy::Recursive,
            max_chars: 1000,
            overlap_chars: 100,
        }
    }

    #[test]
    fn markdown_chunk_carries_path_but_no_symbol_metadata() {
        let chunks = chunk_file("README.md", "# Hello\n\nSome body text.", &opts()).unwrap();
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert_eq!(
                c.attrs.get(META_PATH),
                Some(&Value::Str("README.md".to_string()))
            );
            assert!(!c.attrs.contains_key(META_SYMBOL));
            assert!(!c.attrs.contains_key(META_LANGUAGE));
        }
    }

    #[test]
    fn rust_source_chunks_carry_symbol_metadata() {
        let src = "pub fn one() -> i32 {\n    1\n}\n\npub fn two() -> i32 {\n    2\n}\n";
        let chunks = chunk_file("src/lib.rs", src, &opts()).unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(
            chunks[0].attrs.get(META_LANGUAGE),
            Some(&Value::Str("rust".to_string()))
        );
        let names: Vec<&str> = chunks
            .iter()
            .filter_map(|c| match c.attrs.get(META_SYMBOL) {
                Some(Value::Str(s)) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert!(names.contains(&"one"));
        assert!(names.contains(&"two"));
    }

    #[cfg(feature = "summarize")]
    #[test]
    fn file_summarize_opts_carries_wdpkrs_system_prompt_not_nidus_default() {
        let input = wdpkr_core::summarize::FileSummaryInput {
            file_path: "src/lib.rs".into(),
            content: "pub fn f() {}".into(),
            imports: vec![],
            language: "rust".into(),
        };
        let out = file_summarize_opts(&input);
        assert_eq!(
            out.system.as_deref(),
            Some(wdpkr_core::summarize::prompts::SYSTEM_PROMPT)
        );
        assert_ne!(
            out.system.as_deref(),
            Some(crate::summarize::prompts::DEFAULT_SYSTEM_PROMPT)
        );
        let instructions = out.instructions.unwrap();
        assert!(instructions.contains("src/lib.rs"));
        // The real content is never duplicated into the instructions lead-in.
        assert!(!instructions.contains("pub fn f() {}"));
    }
}

#[cfg(all(test, feature = "summarize-anthropic"))]
mod wire_tests {
    use super::*;
    use crate::summarize::{
        AnySummarizer, SummarizeConfig, SummarizeProvider, Summarizer, test_server::serve_once,
    };

    #[tokio::test]
    async fn wdpkrs_prompt_is_what_reaches_the_adapter() {
        let (base, rx) = serve_once(200, r#"{"content":[{"type":"text","text":"ok"}]}"#);
        let summarizer = AnySummarizer::build(
            SummarizeProvider::Anthropic,
            SummarizeConfig::new("claude-haiku-4-5-20251001")
                .api_key("k")
                .base_url(base),
        )
        .await
        .unwrap();

        let input = wdpkr_core::summarize::FileSummaryInput {
            file_path: "src/lib.rs".into(),
            content: "pub fn f() {}".into(),
            imports: vec![],
            language: "rust".into(),
        };
        let opts = file_summarize_opts(&input);
        summarizer.summarize(&input.content, &opts).await.unwrap();

        let req = rx.recv().unwrap();
        // wdpkr's code-summarizer system prompt reached the wire...
        assert!(req.body.contains("code summarizer for a semantic search index"));
        // ...never nidus's own generic default, which someone "simplifying" this back
        // to `SummarizeOpts::default()` would send instead.
        assert!(!req.body.contains("Preserve the key names"));
        assert!(req.body.contains("src/lib.rs"));
        assert!(req.body.contains("pub fn f() {}"));
    }
}
