//! Tokenizer for the SQL front end. Tokens carry byte offsets into the original source —
//! `at` is the only position [`SqlError`] ever reports (see root blueprint's error model).

use std::ops::Range;

use super::error::{SEC_SYNTAX, SqlError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Ident,
    QuotedIdent,
    Str,
    Int,
    Float,
    Star,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Minus,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Tilde,
    Select,
    From,
    Where,
    Group,
    By,
    Order,
    Limit,
    Offset,
    With,
    And,
    Or,
    Not,
    In,
    Like,
    Ilike,
    Asc,
    Desc,
    Except,
    Sum,
    Fuse,
    Prefix,
    True,
    False,
    Null,
    Eof,
}

/// One token: its kind, the byte offset it starts at, and the byte range of its *content*
/// (identifier chars, or a string/quoted-ident body excluding delimiters) into the source.
#[derive(Clone, Debug)]
pub(super) struct Token {
    pub kind: Kind,
    pub at: usize,
    pub text: Range<usize>,
}

/// Keywords recognized case-insensitively. Every other bare word (function names like `knn`,
/// `contains`, `fuzzy`, and every `WITH` key) lexes as a plain [`Kind::Ident`] and is
/// disambiguated by the parser — keeping the keyword set to only what shapes the grammar.
fn keyword(word_lc: &str) -> Option<Kind> {
    Some(match word_lc {
        "select" => Kind::Select,
        "from" => Kind::From,
        "where" => Kind::Where,
        "group" => Kind::Group,
        "by" => Kind::By,
        "order" => Kind::Order,
        "limit" => Kind::Limit,
        "offset" => Kind::Offset,
        "with" => Kind::With,
        "and" => Kind::And,
        "or" => Kind::Or,
        "not" => Kind::Not,
        "in" => Kind::In,
        "like" => Kind::Like,
        "ilike" => Kind::Ilike,
        "asc" => Kind::Asc,
        "desc" => Kind::Desc,
        "except" => Kind::Except,
        "sum" => Kind::Sum,
        "fuse" => Kind::Fuse,
        "prefix" => Kind::Prefix,
        "true" => Kind::True,
        "false" => Kind::False,
        "null" => Kind::Null,
        _ => return None,
    })
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Bare identifiers routinely carry dots (`nidus.text`), so `.` is a body char, not a
/// separator.
fn is_ident_body(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.'
}

/// Tokenize `src` in full. A malformed token (unterminated string, bad character) is a
/// [`SqlError`] tagged the general-syntax section; the parser assigns more specific tails.
pub(super) fn lex(src: &str) -> Result<Vec<Token>, SqlError> {
    let bytes = src.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < bytes.len() {
        let c = bytes[i] as char;
        match c {
            ' ' | '\t' | '\r' | '\n' => i += 1,
            '*' => {
                out.push(Token {
                    kind: Kind::Star,
                    at: i,
                    text: i..i + 1,
                });
                i += 1;
            }
            '(' => {
                out.push(tok1(Kind::LParen, i));
                i += 1;
            }
            ')' => {
                out.push(tok1(Kind::RParen, i));
                i += 1;
            }
            '[' => {
                out.push(tok1(Kind::LBracket, i));
                i += 1;
            }
            ']' => {
                out.push(tok1(Kind::RBracket, i));
                i += 1;
            }
            ',' => {
                out.push(tok1(Kind::Comma, i));
                i += 1;
            }
            ';' => {
                out.push(tok1(Kind::Semicolon, i));
                i += 1;
            }
            '-' => {
                out.push(tok1(Kind::Minus, i));
                i += 1;
            }
            '~' => {
                out.push(tok1(Kind::Tilde, i));
                i += 1;
            }
            '=' => {
                out.push(tok1(Kind::Eq, i));
                i += 1;
            }
            '!' => {
                if bytes.get(i + 1) == Some(&b'=') {
                    out.push(Token {
                        kind: Kind::Ne,
                        at: i,
                        text: i..i + 2,
                    });
                    i += 2;
                } else {
                    return Err(SqlError::new(i, "expected '=' after '!'", SEC_SYNTAX));
                }
            }
            '<' => {
                if bytes.get(i + 1) == Some(&b'=') {
                    out.push(Token {
                        kind: Kind::Le,
                        at: i,
                        text: i..i + 2,
                    });
                    i += 2;
                } else {
                    out.push(tok1(Kind::Lt, i));
                    i += 1;
                }
            }
            '>' => {
                if bytes.get(i + 1) == Some(&b'=') {
                    out.push(Token {
                        kind: Kind::Ge,
                        at: i,
                        text: i..i + 2,
                    });
                    i += 2;
                } else {
                    out.push(tok1(Kind::Gt, i));
                    i += 1;
                }
            }
            '\'' => {
                let (end, next) = lex_single_quoted(src, i)?;
                out.push(Token {
                    kind: Kind::Str,
                    at: i,
                    text: i + 1..end,
                });
                i = next;
            }
            '"' => {
                let end = find_close(src, i, '"').ok_or_else(|| {
                    SqlError::new(i, "unterminated quoted identifier", SEC_SYNTAX)
                })?;
                out.push(Token {
                    kind: Kind::QuotedIdent,
                    at: i,
                    text: i + 1..end,
                });
                i = end + 1;
            }
            c if c.is_ascii_digit() => {
                let (kind, end) = lex_number(src, i);
                // `1e` / `12abc`: a number running straight into an identifier is a malformed
                // literal, not a number beside a name. Caught here so the offset blames it.
                if end < bytes.len() && is_ident_start(bytes[end] as char) {
                    return Err(SqlError::new(i, "malformed number literal", SEC_SYNTAX));
                }
                out.push(Token {
                    kind,
                    at: i,
                    text: i..end,
                });
                i = end;
            }
            c if is_ident_start(c) => {
                let start = i;
                while i < bytes.len() && is_ident_body(bytes[i] as char) {
                    i += 1;
                }
                let word = &src[start..i];
                let kind = keyword(&word.to_ascii_lowercase()).unwrap_or(Kind::Ident);
                out.push(Token {
                    kind,
                    at: start,
                    text: start..i,
                });
            }
            other => {
                return Err(SqlError::new(
                    i,
                    format!("unexpected character '{other}'"),
                    SEC_SYNTAX,
                ));
            }
        }
    }
    let eof_at = bytes.len();
    out.push(Token {
        kind: Kind::Eof,
        at: eof_at,
        text: eof_at..eof_at,
    });
    Ok(out)
}

fn tok1(kind: Kind, at: usize) -> Token {
    Token {
        kind,
        at,
        text: at..at + 1,
    }
}

/// Find the byte offset of the closing `quote`, starting the scan just past the opener at
/// `open`. No escaping — used for double-quoted identifiers only.
fn find_close(src: &str, open: usize, quote: char) -> Option<usize> {
    src[open + 1..].find(quote).map(|off| open + 1 + off)
}

/// Scan a single-quoted string starting at the opening `'` (byte offset `open`), handling
/// `''` as an escaped quote. Returns (byte offset of the closing `'`, offset just past it).
fn lex_number(src: &str, start: usize) -> (Kind, usize) {
    let bytes = src.as_bytes();
    let mut i = start;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let mut is_float = false;
    if bytes.get(i) == Some(&b'.') && bytes.get(i + 1).is_some_and(u8::is_ascii_digit) {
        is_float = true;
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    if matches!(bytes.get(i), Some(b'e') | Some(b'E')) {
        let mut j = i + 1;
        if matches!(bytes.get(j), Some(b'+') | Some(b'-')) {
            j += 1;
        }
        if bytes.get(j).is_some_and(u8::is_ascii_digit) {
            is_float = true;
            i = j;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
    }
    (if is_float { Kind::Float } else { Kind::Int }, i)
}

fn lex_single_quoted(src: &str, open: usize) -> Result<(usize, usize), SqlError> {
    let bytes = src.as_bytes();
    let mut i = open + 1;
    loop {
        if i >= bytes.len() {
            return Err(SqlError::new(
                open,
                "unterminated string literal",
                SEC_SYNTAX,
            ));
        }
        if bytes[i] == b'\'' {
            if bytes.get(i + 1) == Some(&b'\'') {
                i += 2;
                continue;
            }
            return Ok((i, i + 1));
        }
        i += 1;
    }
}

/// Unescape a single-quoted string body (`''` -> `'`). `raw` is the token's `text` slice,
/// i.e. content only, delimiting quotes already excluded.
pub(super) fn unescape_string(raw: &str) -> String {
    raw.replace("''", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Kind> {
        lex(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn punctuation_and_operators() {
        assert_eq!(
            kinds("[](),;* = != < <= > >= ~ -"),
            vec![
                Kind::LBracket,
                Kind::RBracket,
                Kind::LParen,
                Kind::RParen,
                Kind::Comma,
                Kind::Semicolon,
                Kind::Star,
                Kind::Eq,
                Kind::Ne,
                Kind::Lt,
                Kind::Le,
                Kind::Gt,
                Kind::Ge,
                Kind::Tilde,
                Kind::Minus,
                Kind::Eof,
            ]
        );
    }

    #[test]
    fn keywords_are_case_insensitive() {
        assert_eq!(
            kinds("SeLeCt From WHERE"),
            vec![Kind::Select, Kind::From, Kind::Where, Kind::Eof]
        );
    }

    #[test]
    fn bare_ident_carries_dots() {
        let toks = lex("nidus.text").unwrap();
        assert_eq!(toks[0].kind, Kind::Ident);
        assert_eq!(&"nidus.text"[toks[0].text.clone()], "nidus.text");
    }

    #[test]
    fn function_names_are_plain_idents_not_keywords() {
        assert_eq!(
            kinds("knn contains fuzzy"),
            vec![Kind::Ident; 3]
                .into_iter()
                .chain([Kind::Eof])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn quoted_identifier_holds_spaces_and_dots() {
        let toks = lex(r#""a field.with space""#).unwrap();
        assert_eq!(toks[0].kind, Kind::QuotedIdent);
        assert_eq!(
            &r#""a field.with space""#[toks[0].text.clone()],
            "a field.with space"
        );
    }

    #[test]
    fn unterminated_quoted_identifier_errors() {
        let err = lex(r#""oops"#).unwrap_err();
        assert_eq!(err.at, 0);
    }

    #[test]
    fn single_quoted_string_with_escape() {
        let src = "'it''s here'";
        let toks = lex(src).unwrap();
        assert_eq!(toks[0].kind, Kind::Str);
        let raw = &src[toks[0].text.clone()];
        assert_eq!(unescape_string(raw), "it's here");
    }

    #[test]
    fn unterminated_string_errors_at_the_opening_quote() {
        let err = lex("'oops").unwrap_err();
        assert_eq!(err.at, 0);
    }

    #[test]
    fn integers_and_floats() {
        let toks = lex("42 3.14 1e10 2.5e-3").unwrap();
        assert_eq!(
            toks.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![Kind::Int, Kind::Float, Kind::Float, Kind::Float, Kind::Eof]
        );
    }

    #[test]
    fn bang_alone_is_a_lex_error() {
        let err = lex("!").unwrap_err();
        assert_eq!(err.at, 0);
    }

    #[test]
    fn unexpected_character_is_a_lex_error() {
        let err = lex("field @ 1").unwrap_err();
        assert_eq!(err.at, 6);
    }

    #[test]
    fn every_token_carries_its_byte_offset() {
        let toks = lex("a = 1").unwrap();
        assert_eq!(toks[0].at, 0); // a
        assert_eq!(toks[1].at, 2); // =
        assert_eq!(toks[2].at, 4); // 1
    }
}
