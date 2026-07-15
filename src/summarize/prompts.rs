//! Prompt building for single-shot summarization.
//!
//! Adapted from the ported code-summarizer's prompt, with all code-specific
//! framing (files, symbols, imports, languages) stripped out — nidus knows
//! nothing about the caller's domain. The target output is dense, search-
//! optimized prose that embeds well and matches natural-language queries.

/// The default system prompt. Turns arbitrary text into dense, retrieval-
/// friendly prose. Overridable per adapter ([`SummarizeConfig::system_prompt`])
/// or per call ([`SummarizeOpts::system`]).
///
/// [`SummarizeConfig::system_prompt`]: super::SummarizeConfig::system_prompt
/// [`SummarizeOpts::system`]: super::SummarizeOpts::system
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a summarizer for a semantic search index. Your summaries will be \
embedded as vectors and matched against natural-language queries.\n\n\
Rules:\n\
- Write dense, specific prose. Every sentence should contain searchable terms.\n\
- Focus on WHAT the content is about, in concrete and meaningful terms.\n\
- Preserve the key names, terms, and identifiers that appear in the source — \
they are the bridge between the source's vocabulary and a searcher's query.\n\
- Do NOT include markdown formatting, bullet points, or headers. Plain prose only.\n\
- Do NOT open with \"This text\" or \"This document\" — start with the subject itself.";

/// The lead-in used for the user message when the caller supplies no
/// per-call instructions.
pub const DEFAULT_INSTRUCTION: &str =
    "Summarize the following text into dense, retrieval-friendly prose:";

/// Assemble the user message from the source `text` and optional caller
/// `instructions`. When `instructions` is present (and non-blank) it becomes
/// the lead-in; otherwise [`DEFAULT_INSTRUCTION`] is used. The text follows,
/// separated by a blank line.
pub fn user_message(text: &str, instructions: Option<&str>) -> String {
    let lead = instructions
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_INSTRUCTION);
    format!("{lead}\n\n{text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_is_generic_and_non_empty() {
        assert!(!DEFAULT_SYSTEM_PROMPT.is_empty());
        assert!(DEFAULT_SYSTEM_PROMPT.contains("semantic search"));
        // No code-specific framing leaked over from the port.
        assert!(!DEFAULT_SYSTEM_PROMPT.contains("code"));
        assert!(!DEFAULT_SYSTEM_PROMPT.contains("file"));
        assert!(!DEFAULT_SYSTEM_PROMPT.contains("function"));
        assert!(!DEFAULT_SYSTEM_PROMPT.contains("import"));
    }

    #[test]
    fn user_message_uses_default_instruction_when_none() {
        let msg = user_message("hello world", None);
        assert!(msg.starts_with(DEFAULT_INSTRUCTION));
        assert!(msg.contains("hello world"));
    }

    #[test]
    fn user_message_uses_custom_instruction() {
        let msg = user_message("body text", Some("Summarize in one sentence."));
        assert!(msg.starts_with("Summarize in one sentence."));
        assert!(msg.contains("body text"));
        assert!(!msg.contains(DEFAULT_INSTRUCTION));
    }

    #[test]
    fn blank_instruction_falls_back_to_default() {
        let msg = user_message("x", Some("   "));
        assert!(msg.starts_with(DEFAULT_INSTRUCTION));
    }
}
