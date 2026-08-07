//! ASCII folding: Latin letters with diacritics → their unaccented ASCII form.

/// Fold `text`'s Latin-1 / Latin-Extended-A letters to ASCII, leaving every other char
/// alone. Applied to an already-lowercased token, so every mapping targets lowercase
/// ("Ä" and "ä" both fold to "a"). Returns `text` unchanged when nothing folds.
pub(crate) fn fold_ascii(text: &str) -> String {
    if text.is_ascii() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match folded(ch) {
            Some(s) => out.push_str(s),
            None => out.push(ch),
        }
    }
    out
}

/// The ASCII expansion for one char, or `None` to keep it as-is. Expansions can be
/// multi-char ("æ" → "ae"), which is why this is `&str` and not `char`.
fn folded(ch: char) -> Option<&'static str> {
    let cp = ch as u32;
    match cp {
        0x00C0..=0x00FF => Some(LATIN1[(cp - 0x00C0) as usize]).filter(|s| !s.is_empty()),
        0x0100..=0x017F => Some(LATIN_A[(cp - 0x0100) as usize]),
        _ => None,
    }
}

/// U+00C0–U+00FF in code-point order. `""` marks the two non-letters in the block
/// (× and ÷), which [`folded`] then leaves untouched.
static LATIN1: &[&str; 64] = &[
    "a", "a", "a", "a", "a", "a", "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i", "d", "n", "o",
    "o", "o", "o", "o", "", "o", "u", "u", "u", "u", "y", "th", "ss", "a", "a", "a", "a", "a", "a",
    "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i", "d", "n", "o", "o", "o", "o", "o", "", "o",
    "u", "u", "u", "u", "y", "th", "y",
];

/// U+0100–U+017F (Latin Extended-A) in code-point order.
static LATIN_A: &[&str; 128] = &[
    "a", "a", "a", "a", "a", "a", "c", "c", "c", "c", "c", "c", "c", "c", "d", "d", "d", "d", "e",
    "e", "e", "e", "e", "e", "e", "e", "e", "e", "g", "g", "g", "g", "g", "g", "g", "g", "h", "h",
    "h", "h", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i", "ij", "ij", "j", "j", "k", "k",
    "k", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "n", "n", "n", "n", "n", "n", "n", "n",
    "n", "o", "o", "o", "o", "o", "o", "oe", "oe", "r", "r", "r", "r", "r", "r", "s", "s", "s",
    "s", "s", "s", "s", "s", "t", "t", "t", "t", "t", "t", "u", "u", "u", "u", "u", "u", "u", "u",
    "u", "u", "u", "u", "w", "w", "y", "y", "y", "z", "z", "z", "z", "z", "z", "s",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_passes_through_untouched() {
        assert_eq!(fold_ascii("hello world 42"), "hello world 42");
        assert_eq!(fold_ascii(""), "");
    }

    #[test]
    fn latin_diacritics_fold_to_bare_letters() {
        assert_eq!(fold_ascii("café"), "cafe");
        assert_eq!(fold_ascii("naïve"), "naive");
        assert_eq!(fold_ascii("žluťoučký"), "zlutoucky");
        assert_eq!(fold_ascii("łódź"), "lodz");
    }

    #[test]
    fn ligatures_and_eszett_expand() {
        assert_eq!(fold_ascii("æon"), "aeon");
        assert_eq!(fold_ascii("œuvre"), "oeuvre");
        assert_eq!(fold_ascii("straße"), "strasse");
    }

    #[test]
    fn uppercase_forms_fold_to_lowercase_ascii() {
        // Folding runs after lowercasing, but the table covers the whole block so an
        // uppercase char reaching it still yields a usable term rather than garbage.
        assert_eq!(fold_ascii("ÉCOLE"), "eCOLE");
        assert_eq!(fold_ascii("Ø"), "o");
    }

    #[test]
    fn non_latin_scripts_are_left_alone() {
        assert_eq!(fold_ascii("日本語"), "日本語");
        assert_eq!(fold_ascii("Привет"), "Привет");
        // The two non-letters inside the Latin-1 block survive as themselves.
        assert_eq!(fold_ascii("2×3"), "2×3");
        assert_eq!(fold_ascii("6÷2"), "6÷2");
    }
}
