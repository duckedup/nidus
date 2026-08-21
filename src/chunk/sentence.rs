//! Naive sentence-boundary splitting: break after `.`, `!`, or `?` when followed by
//! whitespace or end of input, then pack whole sentences up to the budget. No
//! abbreviation detection ("Dr.", "e.g.") — that needs a dictionary and is out of scope,
//! so a boundary here is naive by design, not by oversight.

use super::{pack, recursive};

pub(super) fn split(src: &[char], max_chars: usize) -> Vec<(usize, usize)> {
    let sentences = tile_sentences(src, 0, src.len());
    let mut refined = Vec::new();
    for (s, e) in sentences {
        if e - s > max_chars {
            refined.extend(recursive::split(src, s, e, max_chars));
        } else {
            refined.push((s, e));
        }
    }
    pack(&refined, max_chars)
}

/// Tiles `src[start..end)` into contiguous sentence-ish pieces; each piece ends right
/// after its terminator (the last piece has none). No chars are dropped.
fn tile_sentences(src: &[char], start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut pieces = Vec::new();
    let mut piece_start = start;
    let mut i = start;
    while i < end {
        let c = src[i];
        if c == '.' || c == '!' || c == '?' {
            let next = i + 1;
            if next >= end || src[next].is_whitespace() {
                pieces.push((piece_start, next));
                piece_start = next;
                i = next;
                continue;
            }
        }
        i += 1;
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
    fn fits_in_one_chunk_under_budget() {
        let text = "One. Two. Three.";
        let spans = split(&text.chars().collect::<Vec<_>>(), 100);
        assert_eq!(spans, vec![(0, text.chars().count())]);
    }

    #[test]
    fn packs_sentences_up_to_budget() {
        let text = "Aa. Bb. Cc. Dd. Ee.";
        let src: Vec<char> = text.chars().collect();
        let spans = split(&src, 8);
        for (s, e) in &spans {
            assert!(e - s <= 8);
        }
        assert!(spans.len() > 1);
    }

    #[test]
    fn oversized_single_sentence_falls_back_to_recursive() {
        let text = "a".repeat(50) + ".";
        let src: Vec<char> = text.chars().collect();
        let spans = split(&src, 10);
        for (s, e) in &spans {
            assert!(e - s <= 10);
        }
    }

    #[test]
    fn no_terminators_is_one_piece_before_packing() {
        let text = "no terminators here at all";
        let spans = tile_sentences(&text.chars().collect::<Vec<_>>(), 0, text.chars().count());
        assert_eq!(spans.len(), 1);
    }
}
