//! Heading-aware splitting: sections break at top-level ATX headings, but never inside a
//! fenced code block. An oversized section falls through to line-level packing that stays
//! fence-safe, then to the recursive splitter for any fence-free run still over budget —
//! passing only that section's own range down, never re-scanning the whole document.
//!
//! Fence rule: an opener is a line with <=3 leading spaces then a run of 3+ backticks or
//! tildes; the closer must use the same char and be at least as long, or the fence stays
//! open. An unterminated fence at EOF leaves the rest of the document inside it, so no
//! later heading counts as a split point.

use super::recursive;

#[derive(Clone, Copy)]
struct Line {
    start: usize,
    end: usize, // exclusive; includes the trailing '\n' when present
    depth_before: u8,
    depth_after: u8,
    is_heading: bool,
}

/// Returns the spans plus, per span, the start of the section it belongs to. That floor is
/// what stops backward overlap reaching across a heading into an unrelated section, which
/// would blend the topics heading-aware splitting exists to keep apart.
pub(super) fn split(src: &[char], max_chars: usize) -> (Vec<(usize, usize)>, Vec<usize>) {
    if src.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let lines = scan_lines(src);
    let mut out = Vec::new();
    let mut floors = Vec::new();
    for (ls, le) in section_ranges(&lines) {
        let sec = &lines[ls..le];
        let sec_start = sec.first().unwrap().start;
        let sec_end = sec.last().unwrap().end;
        if sec_end - sec_start <= max_chars {
            out.push((sec_start, sec_end));
            floors.push(sec_start);
            continue;
        }
        for (gs, ge, has_fence) in pack_section(sec, max_chars) {
            if !has_fence && ge - gs > max_chars {
                let pieces = recursive::split(src, gs, ge, max_chars);
                floors.extend(std::iter::repeat_n(sec_start, pieces.len()));
                out.extend(pieces);
            } else {
                out.push((gs, ge));
                floors.push(sec_start);
            }
        }
    }
    (out, floors)
}

fn scan_lines(src: &[char]) -> Vec<Line> {
    let mut lines = Vec::new();
    let mut depth = 0u8;
    let mut fence_char = '`';
    let mut fence_len = 0usize;
    let n = src.len();
    let mut i = 0usize;
    while i < n {
        let line_start = i;
        while i < n && src[i] != '\n' {
            i += 1;
        }
        let content = &src[line_start..i];
        let line_end = if i < n { i + 1 } else { i };

        let depth_before = depth;
        let mut is_heading = false;
        if depth == 0 {
            if let Some((fc, flen)) = fence_open(content) {
                depth = 1;
                fence_char = fc;
                fence_len = flen;
            } else if is_atx_heading(content) {
                is_heading = true;
            }
        } else if fence_close(content, fence_char, fence_len) {
            depth = 0;
        }

        lines.push(Line {
            start: line_start,
            end: line_end,
            depth_before,
            depth_after: depth,
            is_heading,
        });
        i = line_end;
    }
    lines
}

fn count_leading_spaces(line: &[char]) -> usize {
    line.iter().take_while(|&&c| c == ' ').count()
}

/// A run of 3+ of the same fence char (backtick or tilde) right after <=3 leading spaces.
fn fence_run(line: &[char]) -> Option<(char, usize)> {
    let indent = count_leading_spaces(line);
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let c = *rest.first()?;
    if c != '`' && c != '~' {
        return None;
    }
    let len = rest.iter().take_while(|&&ch| ch == c).count();
    (len >= 3).then_some((c, len))
}

fn fence_open(line: &[char]) -> Option<(char, usize)> {
    fence_run(line)
}

fn fence_close(line: &[char], fence_char: char, fence_len: usize) -> bool {
    matches!(fence_run(line), Some((c, len)) if c == fence_char && len >= fence_len)
}

/// `#{1,6}` at <=3 leading spaces, followed by whitespace or end of line.
fn is_atx_heading(line: &[char]) -> bool {
    let indent = count_leading_spaces(line);
    if indent > 3 {
        return false;
    }
    let rest = &line[indent..];
    let hashes = rest.iter().take_while(|&&c| c == '#').count();
    if hashes == 0 || hashes > 6 {
        return false;
    }
    matches!(rest.get(hashes), None | Some(&' ') | Some(&'\t'))
}

fn section_ranges(lines: &[Line]) -> Vec<(usize, usize)> {
    let mut starts = vec![0];
    for (i, l) in lines.iter().enumerate().skip(1) {
        if l.is_heading {
            starts.push(i);
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(w, &s)| (s, starts.get(w + 1).copied().unwrap_or(lines.len())))
        .collect()
}

/// Greedily packs a section's lines up to `max_chars`, never cutting where the next line
/// starts inside an open fence. Each output group reports whether it touched a fence, so
/// the caller knows not to split it further.
fn pack_section(sec: &[Line], max_chars: usize) -> Vec<(usize, usize, bool)> {
    let mut out = Vec::new();
    let mut gs_idx = 0usize;
    let mut has_fence = false;
    for i in 0..sec.len() {
        let would_size = sec[i].end - sec[gs_idx].start;
        let cut_before_i_is_safe = sec[i].depth_before == 0;
        if i > gs_idx && would_size > max_chars && cut_before_i_is_safe {
            out.push((sec[gs_idx].start, sec[i - 1].end, has_fence));
            gs_idx = i;
            has_fence = false;
        }
        if sec[i].depth_before == 1 || sec[i].depth_after == 1 {
            has_fence = true;
        }
    }
    out.push((sec[gs_idx].start, sec.last().unwrap().end, has_fence));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans_text(text: &str, spans: &[(usize, usize)]) -> Vec<String> {
        let src: Vec<char> = text.chars().collect();
        spans
            .iter()
            .map(|&(s, e)| src[s..e].iter().collect())
            .collect()
    }

    #[test]
    fn heading_looking_line_inside_fence_is_not_a_split_point() {
        let text = "Intro text here.\n```\n# not a heading\n```\nMore text after fence.";
        let spans = split(&text.chars().collect::<Vec<_>>(), 1000).0;
        assert_eq!(spans.len(), 1, "no real heading, so one section");
        assert_eq!(spans_text(text, &spans)[0], text);
    }

    #[test]
    fn unterminated_fence_at_eof_swallows_following_headings() {
        let text = "Body.\n```\ncode\n# Looks like a heading\nmore code\n";
        let spans = split(&text.chars().collect::<Vec<_>>(), 1000).0;
        assert_eq!(
            spans.len(),
            1,
            "fence never closes, so no split ever occurs"
        );
    }

    #[test]
    fn backtick_inside_tilde_fence_is_literal_and_reverse() {
        // Fake headings sit inside each fence type nested in the other; only the two
        // real headings, outside any fence, should produce section boundaries.
        let text = "~~~\n```\n# fake heading A\n~~~\n# Real Heading 1\n```\n~~~\n# fake heading B\n```\n# Real Heading 2\ntail";
        let spans = split(&text.chars().collect::<Vec<_>>(), 1000).0;
        assert_eq!(spans.len(), 3, "only the two real headings split");
    }

    #[test]
    fn fence_indented_three_is_fence_four_is_not() {
        let three = "   ```\ncode\n   ```\n# Heading\nBody";
        let spans = split(&three.chars().collect::<Vec<_>>(), 1000).0;
        assert_eq!(
            spans.len(),
            2,
            "3-space fence closes, heading after it splits"
        );

        let four = "    ```\n# Heading\nBody";
        let spans4 = split(&four.chars().collect::<Vec<_>>(), 1000).0;
        assert_eq!(
            spans4.len(),
            2,
            "4-space indent is not a fence, so the heading still splits"
        );
    }

    #[test]
    fn closing_fence_shorter_than_opener_does_not_close() {
        let text = "````\ncode\n```\nstill in fence\n````\n# Heading\nBody";
        let spans = split(&text.chars().collect::<Vec<_>>(), 1000).0;
        assert_eq!(spans.len(), 2, "short ``` doesn't close the ```` opener");
    }

    #[test]
    fn fence_landing_on_max_chars_boundary_is_not_torn() {
        // The fence spans well past any naive char-count boundary; every emitted span
        // must contain BOTH the opening and closing fence line, never one without the
        // other, however small max_chars is.
        let text = "# H\n```\naaaaaaaaaa\nbbbbbbbbbb\ncccccccccc\n```\ntail text";
        let src: Vec<char> = text.chars().collect();
        for max_chars in 5..40 {
            let spans = split(&src, max_chars).0;
            let open = text.find("```").unwrap();
            let close = text.rfind("```").unwrap();
            for &(s, e) in &spans {
                let contains_open = s <= open && open < e;
                let contains_close = s <= close && close < e;
                assert_eq!(
                    contains_open, contains_close,
                    "fence torn at max_chars={max_chars}: span ({s},{e})"
                );
            }
        }
    }

    #[test]
    fn document_with_no_headings_still_chunks() {
        let text = "just a plain paragraph with no headings or fences at all here";
        let src: Vec<char> = text.chars().collect();
        let spans = split(&src, 20).0;
        assert!(!spans.is_empty());
        for (s, e) in &spans {
            assert!(e - s <= 20);
        }
    }

    #[test]
    fn heading_splits_into_sections() {
        let text = "# One\nbody one\n# Two\nbody two";
        let spans = split(&text.chars().collect::<Vec<_>>(), 1000).0;
        assert_eq!(spans.len(), 2);
    }
}
