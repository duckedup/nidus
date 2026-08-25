//! Per-file chunk-strategy dispatch: the mapping that makes the epic's ONE CORPUS claim
//! true (`SPEC.md` and `compact()` land in one store, each chunked for what it is).
//! Code-only, reached only from `nidus code ingest` and the docs-index absorption (unit
//! 11): there is deliberately no public `--strategy auto`.

use crate::chunk::ChunkStrategy;

/// A language wdpkr-core both recognises AND ships a grammar for gets AST-aware chunking;
/// `.md`/`.mdx`/`.markdown` gets the markdown splitter; everything else falls back to the
/// generic recursive splitter. Path-only: UTF-8 validity of `path`'s content is upstream.
///
/// `detect_language` recognises more extensions than `languages::get_config` has grammars
/// for (`.svelte`, for one). Routing one of those to the AST chunker yields ZERO chunks, so
/// the file would vanish from the corpus; `has_grammar` keeps it on the recursive splitter.
pub fn strategy_for(path: &str) -> ChunkStrategy {
    if wdpkr_core::chunk::detect_language(path).is_some_and(crate::chunk::code::has_grammar) {
        return ChunkStrategy::Code;
    }
    match path.rsplit('.').next() {
        Some("md") | Some("mdx") | Some("markdown") => ChunkStrategy::Markdown,
        _ => ChunkStrategy::Recursive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table() {
        let cases: &[(&str, ChunkStrategy)] = &[
            ("src/main.rs", ChunkStrategy::Code),
            ("pkg/handler.go", ChunkStrategy::Code),
            ("src/app.ts", ChunkStrategy::Code),
            ("src/App.tsx", ChunkStrategy::Code),
            ("index.js", ChunkStrategy::Code),
            ("main.py", ChunkStrategy::Code),
            ("Service.java", ChunkStrategy::Code),
            ("main.c", ChunkStrategy::Code),
            ("main.cpp", ChunkStrategy::Code),
            ("Program.cs", ChunkStrategy::Code),
            // The case that would fail if markdown were routed to the AST chunker: `.md`
            // is not a language `detect_language` recognises, so it must land on the
            // markdown splitter, never `ChunkStrategy::Code`.
            ("README.md", ChunkStrategy::Markdown),
            ("docs/guide.mdx", ChunkStrategy::Markdown),
            ("NOTES.markdown", ChunkStrategy::Markdown),
            ("data.json", ChunkStrategy::Recursive),
            ("Makefile", ChunkStrategy::Recursive),
            (".gitignore", ChunkStrategy::Recursive),
            ("SPEC.txt", ChunkStrategy::Recursive),
        ];
        for (path, want) in cases {
            assert_eq!(strategy_for(path), *want, "path: {path}");
        }
    }

    #[test]
    fn nested_paths_still_detect_language() {
        assert_eq!(
            strategy_for("internal/finance/commission/release.go"),
            ChunkStrategy::Code
        );
    }
}
