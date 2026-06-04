//! Minimal GLOB matcher. Contract: see the root `SPEC.md` §7.1.

/// Returns true if `text` matches the GLOB `pattern`.
///
/// Supports `*` (any run, including empty), `?` (exactly one char), and character
/// classes `[abc]` / ranges `[a-z]` / negation `[!..]` or `[^..]`. A literal `*`,
/// `?`, or `[` is matched only via a class. Operates on Unicode scalar values.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    match_pattern(&pat, &txt)
}

/// Parses a character class starting after the `[`. Returns `(matched, chars_consumed)`
/// where `chars_consumed` is the number of pattern chars consumed (not including the `[`).
/// If the class is unterminated (no `]`), returns `None` so the caller treats `[` literally.
fn match_class(class_chars: &[char], ch: char) -> Option<(bool, usize)> {
    // class_chars starts after `[`
    let mut i = 0;

    // Check for negation
    let negate = if i < class_chars.len() && (class_chars[i] == '!' || class_chars[i] == '^') {
        i += 1;
        true
    } else {
        false
    };

    let mut matched = false;
    let mut found_close = false;

    while i < class_chars.len() {
        if class_chars[i] == ']' {
            found_close = true;
            i += 1;
            break;
        }

        // Check for range: x-y (but not if `-` is at end before `]`, or `]` follows immediately)
        if i + 2 < class_chars.len() && class_chars[i + 1] == '-' && class_chars[i + 2] != ']' {
            let lo = class_chars[i];
            let hi = class_chars[i + 2];
            if ch >= lo && ch <= hi {
                matched = true;
            }
            i += 3;
        } else {
            if class_chars[i] == ch {
                matched = true;
            }
            i += 1;
        }
    }

    if !found_close {
        // Unterminated class — treat `[` as literal
        return None;
    }

    Some((if negate { !matched } else { matched }, i))
}

/// Core recursive matcher operating on slices of chars.
fn match_pattern(pat: &[char], txt: &[char]) -> bool {
    let mut pi = 0;
    let mut ti = 0;

    // We use an iterative approach with backtracking for `*`.
    // `star_pi` and `star_ti` track the last `*` position for backtracking.
    let mut star_pi: Option<usize> = None;
    let mut star_ti: usize = 0;

    loop {
        // If we still have pattern left, try to match current pattern char.
        if pi < pat.len() {
            match pat[pi] {
                '*' => {
                    // Collapse consecutive stars.
                    while pi < pat.len() && pat[pi] == '*' {
                        pi += 1;
                    }
                    // If pattern is exhausted after stars, it matches everything remaining.
                    if pi == pat.len() {
                        return true;
                    }
                    // Record the star position for backtracking.
                    star_pi = Some(pi);
                    star_ti = ti;
                    // Don't advance ti yet; let the non-star branch try to match at ti.
                    continue;
                }
                '?' => {
                    if ti < txt.len() {
                        pi += 1;
                        ti += 1;
                        continue;
                    }
                    // `?` needs a char but text is exhausted — backtrack or fail.
                }
                '[' => {
                    // Try to parse a character class.
                    let class_slice = &pat[pi + 1..];
                    match match_class(class_slice, if ti < txt.len() { txt[ti] } else { '\0' }) {
                        Some((matched, consumed)) if ti < txt.len() && matched => {
                            pi += 1 + consumed; // skip `[` + class content + `]`
                            ti += 1;
                            continue;
                        }
                        Some(_) => {
                            // Class didn't match (or text exhausted) — fall through to backtrack.
                        }
                        None => {
                            // Unterminated class: treat `[` as literal.
                            if ti < txt.len() && txt[ti] == '[' {
                                pi += 1;
                                ti += 1;
                                continue;
                            }
                            // Literal `[` doesn't match current char — fall through to backtrack.
                        }
                    }
                }
                literal => {
                    if ti < txt.len() && txt[ti] == literal {
                        pi += 1;
                        ti += 1;
                        continue;
                    }
                    // Literal mismatch — fall through to backtrack.
                }
            }
        } else {
            // Pattern exhausted.
            if ti == txt.len() {
                return true;
            }
            // Text still has chars — fall through to backtrack.
        }

        // Backtrack: if we have a saved `*`, let it consume one more text char.
        if let Some(spi) = star_pi {
            star_ti += 1;
            if star_ti <= txt.len() {
                pi = spi;
                ti = star_ti;
                continue;
            }
        }

        return false;
    }
}

#[cfg(test)]
mod tests {
    use super::glob_match;

    // ── Literal matches ──────────────────────────────────────────────────────────

    #[test]
    fn literal_exact_match() {
        assert!(glob_match("hello", "hello"));
    }

    #[test]
    fn literal_mismatch() {
        assert!(!glob_match("hello", "world"));
    }

    #[test]
    fn literal_prefix_not_full_match() {
        // Anchored: pattern "a" must not match text "ab"
        assert!(!glob_match("a", "ab"));
    }

    #[test]
    fn literal_suffix_not_full_match() {
        assert!(!glob_match("b", "ab"));
    }

    #[test]
    fn empty_pattern_matches_empty_text() {
        assert!(glob_match("", ""));
    }

    #[test]
    fn empty_pattern_does_not_match_nonempty_text() {
        assert!(!glob_match("", "x"));
    }

    #[test]
    fn nonempty_pattern_does_not_match_empty_text() {
        assert!(!glob_match("a", ""));
    }

    // ── Unicode ──────────────────────────────────────────────────────────────────

    #[test]
    fn unicode_literal_match() {
        assert!(glob_match("héllo", "héllo"));
    }

    #[test]
    fn unicode_literal_mismatch() {
        assert!(!glob_match("héllo", "hello"));
    }

    #[test]
    fn unicode_wildcard() {
        assert!(glob_match("h*o", "héllo"));
    }

    // ── `*` wildcard ─────────────────────────────────────────────────────────────

    #[test]
    fn star_matches_empty() {
        assert!(glob_match("a*", "a"));
    }

    #[test]
    fn star_matches_nonempty() {
        assert!(glob_match("a*", "abc"));
    }

    #[test]
    fn star_at_start() {
        assert!(glob_match("*c", "abc"));
        assert!(!glob_match("*c", "ab"));
    }

    #[test]
    fn star_at_middle() {
        assert!(glob_match("a*c", "ac"));
        assert!(glob_match("a*c", "abc"));
        assert!(glob_match("a*c", "aXYZc"));
        assert!(!glob_match("a*c", "abd"));
    }

    #[test]
    fn star_alone_matches_anything() {
        assert!(glob_match("*", ""));
        assert!(glob_match("*", "abc"));
        assert!(glob_match("*", "x"));
    }

    #[test]
    fn double_star_same_as_single() {
        // Consecutive `**` behaves like `*`
        assert!(glob_match("a**c", "ac"));
        assert!(glob_match("a**c", "aXc"));
        assert!(!glob_match("a**c", "abd"));
    }

    #[test]
    fn multiple_stars() {
        assert!(glob_match("*.*", "foo.rs"));
        assert!(glob_match("*.*", "a.b.c"));
        assert!(!glob_match("*.*", "nodot"));
    }

    #[test]
    fn star_does_not_require_separator() {
        // `*` matches `/` like any other char (no special path semantics)
        assert!(glob_match("src/*", "src/main.rs"));
        assert!(glob_match("src/*", "src/a/b.rs"));
    }

    // ── `?` wildcard ─────────────────────────────────────────────────────────────

    #[test]
    fn question_matches_one_char() {
        assert!(glob_match("a?c", "abc"));
        assert!(glob_match("a?c", "axc"));
    }

    #[test]
    fn question_does_not_match_empty() {
        assert!(!glob_match("a?c", "ac"));
    }

    #[test]
    fn question_does_not_match_two_chars() {
        assert!(!glob_match("a?c", "axyc"));
    }

    #[test]
    fn multiple_questions() {
        assert!(glob_match("???", "abc"));
        assert!(!glob_match("???", "ab"));
        assert!(!glob_match("???", "abcd"));
    }

    // ── Character classes `[...]` ─────────────────────────────────────────────────

    #[test]
    fn class_simple_match() {
        assert!(glob_match("[abc]", "a"));
        assert!(glob_match("[abc]", "b"));
        assert!(glob_match("[abc]", "c"));
        assert!(!glob_match("[abc]", "d"));
    }

    #[test]
    fn class_range_lowercase() {
        assert!(glob_match("[a-z]", "m"));
        assert!(!glob_match("[a-z]", "A"));
        assert!(!glob_match("[a-z]", "0"));
    }

    #[test]
    fn class_range_digits() {
        assert!(glob_match("[0-9]", "5"));
        assert!(!glob_match("[0-9]", "a"));
    }

    #[test]
    fn class_negate_exclamation() {
        assert!(glob_match("[!0-9]", "a"));
        assert!(!glob_match("[!0-9]", "5"));
    }

    #[test]
    fn class_negate_caret() {
        assert!(glob_match("[^abc]", "d"));
        assert!(!glob_match("[^abc]", "a"));
    }

    #[test]
    fn class_mixed() {
        // Digits or underscore
        assert!(glob_match("[0-9_]", "3"));
        assert!(glob_match("[0-9_]", "_"));
        assert!(!glob_match("[0-9_]", "a"));
    }

    #[test]
    fn class_in_pattern() {
        assert!(glob_match("file[0-9].rs", "file3.rs"));
        assert!(!glob_match("file[0-9].rs", "fileX.rs"));
    }

    // ── Unterminated `[` treated as literal ──────────────────────────────────────

    #[test]
    fn unterminated_class_literal_bracket() {
        // No closing `]` → the `[` is a literal.
        assert!(glob_match("[abc", "[abc"));
        assert!(!glob_match("[abc", "a"));
    }

    #[test]
    fn unterminated_class_with_star() {
        // `[abc` treated as literal chars (or as a `[` literal followed by more pattern).
        // The `[` is literal, so pattern "[*" should match "[anything".
        assert!(glob_match("[*", "[hello"));
        assert!(!glob_match("[*", "hello"));
    }

    // ── Anchoring ────────────────────────────────────────────────────────────────

    #[test]
    fn anchored_both_ends() {
        // "ab" pattern must not match "xab" or "abx"
        assert!(!glob_match("ab", "xab"));
        assert!(!glob_match("ab", "abx"));
        assert!(glob_match("ab", "ab"));
    }

    // ── Realistic path patterns ───────────────────────────────────────────────────

    #[test]
    fn path_src_star() {
        assert!(glob_match("src/*", "src/main.rs"));
        assert!(glob_match("src/*", "src/lib.rs"));
        // Different directory
        assert!(!glob_match("src/x/*", "src/y/a.rs"));
    }

    #[test]
    fn path_src_star_rs() {
        assert!(glob_match("src/*.rs", "src/a.rs"));
        assert!(glob_match("src/*.rs", "src/main.rs"));
        assert!(!glob_match("src/*.rs", "src/a.txt"));
    }

    #[test]
    fn path_double_wildcard_deep() {
        // src/* will match src/a/b.rs because * matches anything including /
        assert!(glob_match("src/*", "src/a/b.rs"));
    }

    #[test]
    fn path_exact() {
        assert!(glob_match("src/main.rs", "src/main.rs"));
        assert!(!glob_match("src/main.rs", "src/lib.rs"));
    }

    #[test]
    fn path_extension_glob() {
        assert!(glob_match("*.toml", "Cargo.toml"));
        assert!(glob_match("*.toml", "config.toml"));
        assert!(!glob_match("*.toml", "main.rs"));
    }

    #[test]
    fn path_question_in_name() {
        assert!(glob_match("file?.rs", "file1.rs"));
        assert!(glob_match("file?.rs", "fileA.rs"));
        assert!(!glob_match("file?.rs", "file.rs"));
        assert!(!glob_match("file?.rs", "file12.rs"));
    }

    // ── Edge cases ────────────────────────────────────────────────────────────────

    #[test]
    fn pattern_only_stars_matches_anything() {
        assert!(glob_match("***", ""));
        assert!(glob_match("***", "hello"));
    }

    #[test]
    fn star_at_both_ends() {
        assert!(glob_match("*foo*", "foo"));
        assert!(glob_match("*foo*", "foobar"));
        assert!(glob_match("*foo*", "barfoo"));
        assert!(glob_match("*foo*", "barfoobar"));
        assert!(!glob_match("*foo*", "bar"));
    }

    #[test]
    fn empty_class_not_matchable() {
        // `[]` — immediately closing bracket; in many glob implementations `]`
        // is treated as first char of class. But our spec says unterminated if no `]`
        // after content. `[]` means class with just `]`? Let's test that `[]` followed
        // by `]` closes on the first `]` making an empty set... actually `[]` sees `]`
        // immediately and since found_close=true with 0 members, matches nothing.
        assert!(!glob_match("[]", "a"));
        assert!(!glob_match("[]", "]"));
    }

    #[test]
    fn star_before_literal_backtracking() {
        // Requires backtracking: `*b` on "aab"
        assert!(glob_match("*b", "aab"));
        assert!(glob_match("a*b*c", "aXbYc"));
        assert!(glob_match("a*b*c", "abc"));
        assert!(!glob_match("a*b*c", "axc"));
    }

    #[test]
    fn question_with_star() {
        assert!(glob_match("?*", "a")); // one char, then any
        assert!(glob_match("?*", "ab"));
        assert!(!glob_match("?*", "")); // needs at least one char
    }
}
