//! Separator-ladder splitting: try the coarsest separator first (`"\n\n"`, `"\n"`,
//! `". "`, `" "`), recurse into any oversized piece with the next separator down, then
//! pack adjacent pieces up to the char budget. Falling off the ladder means a hard split
//! at a char boundary.

use super::pack;

const SEPARATORS: &[&str] = &["\n\n", "\n", ". ", " "];

/// Splits `src[start..end)` into raw, budget-packed, contiguous char ranges.
pub(super) fn split(
    src: &[char],
    start: usize,
    end: usize,
    max_chars: usize,
) -> Vec<(usize, usize)> {
    split_at(src, start, end, max_chars, 0)
}

fn split_at(
    src: &[char],
    start: usize,
    end: usize,
    max_chars: usize,
    sep_idx: usize,
) -> Vec<(usize, usize)> {
    if start >= end {
        return Vec::new();
    }
    if end - start <= max_chars {
        return vec![(start, end)];
    }
    if sep_idx >= SEPARATORS.len() {
        return hard_split(start, end, max_chars);
    }

    let pieces = tile_on_separator(src, start, end, SEPARATORS[sep_idx]);
    if pieces.len() < 2 {
        return split_at(src, start, end, max_chars, sep_idx + 1);
    }

    let mut refined = Vec::new();
    for (ps, pe) in pieces {
        if pe - ps > max_chars {
            refined.extend(split_at(src, ps, pe, max_chars, sep_idx + 1));
        } else {
            refined.push((ps, pe));
        }
    }
    pack(&refined, max_chars)
}

/// Splits at raw char boundaries every `max_chars`, no separators involved.
fn hard_split(start: usize, end: usize, max_chars: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = start;
    while i < end {
        let j = (i + max_chars).min(end);
        out.push((i, j));
        i = j;
    }
    out
}

/// Tiles `src[start..end)` into contiguous pieces on occurrences of `sep`; each piece
/// ends right after its separator match (the last piece has none). No chars are dropped.
fn tile_on_separator(src: &[char], start: usize, end: usize, sep: &str) -> Vec<(usize, usize)> {
    let sep_chars: Vec<char> = sep.chars().collect();
    let sep_len = sep_chars.len();
    if sep_len == 0 || end - start < sep_len {
        return vec![(start, end)];
    }
    let mut pieces = Vec::new();
    let mut piece_start = start;
    let mut i = start;
    while i + sep_len <= end {
        if src[i..i + sep_len] == sep_chars[..] {
            let piece_end = i + sep_len;
            pieces.push((piece_start, piece_end));
            piece_start = piece_end;
            i = piece_end;
        } else {
            i += 1;
        }
    }
    if piece_start < end {
        pieces.push((piece_start, end));
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_piece() {
        let src: Vec<char> = "hello".chars().collect();
        assert_eq!(split(&src, 0, src.len(), 100), vec![(0, 5)]);
    }

    #[test]
    fn splits_on_paragraph_boundary() {
        let text = "aaaaaaaaaa\n\nbbbbbbbbbb";
        let src: Vec<char> = text.chars().collect();
        let spans = split(&src, 0, src.len(), 12);
        assert_eq!(spans, vec![(0, 12), (12, 22)]);
    }

    #[test]
    fn hard_splits_when_no_separator_fits() {
        let src: Vec<char> = "a".repeat(10).chars().collect();
        let spans = split(&src, 0, src.len(), 4);
        assert_eq!(spans, vec![(0, 4), (4, 8), (8, 10)]);
    }

    #[test]
    fn packs_small_pieces_together() {
        let text = "a\nb\nc\nd\ne\nf\ng\nh";
        let src: Vec<char> = text.chars().collect();
        let spans = split(&src, 0, src.len(), 4);
        for (s, e) in &spans {
            assert!(e - s <= 4);
        }
        let newline_count = text.chars().filter(|&c| c == '\n').count();
        assert!(spans.len() < newline_count + 1);
    }

    #[test]
    fn empty_range_is_no_pieces() {
        let src: Vec<char> = "hello".chars().collect();
        assert!(split(&src, 2, 2, 10).is_empty());
    }
}
