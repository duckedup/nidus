// Pure decisions for the PreToolUse guards. No IO, so selftest.mjs drives them from
// fixtures and a guard that stops firing fails there rather than silently (nidus-gmy.5).

/// Docs big enough that a whole-file read is never what the reader wanted, and the tool
/// that addresses them by section instead.
export const GUARDED = { 'SPEC.md': 'just spec' }


function advice(rel, how) {
  return `Do not read ${rel} whole — use \`${how}\` instead:

  ${how} toc            the section index, with line counts
  ${how} find <words>   which section covers a topic
  ${how} <ref>          print one section (7, 7.4, 7.4.1, or a slug)

A whole-file read spends tens of thousands of tokens to use one section, and every
subagent pays it again. If you need a specific line range, pass Read both offset and limit.`
}

/// A bounded read (both offset and limit) is allowed: that caller already knows the range.
export function specRead({ rel, offset, limit } = {}) {
  if (!rel) return null
  if (offset !== undefined && limit !== undefined) return null
  const how = GUARDED[rel]
  return how ? advice(rel, how) : null
}

// A Bash matcher on command text was tried and removed: it fired on its own test fixture,
// because a command that merely *mentions* a read (a heredoc, a doc example) is not one.
// Structured tool input is the only reliable layer, so the guard covers Read alone.
