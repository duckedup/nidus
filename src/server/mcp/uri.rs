//! The `nidus://` resource URI codec. Percent-encoded to the unreserved set so a URI
//! survives the `Mcp-Name` header un-base64'd, and so a `/` in a name stays unambiguous.

pub(super) const SCHEME: &str = "nidus://";
pub(super) const COLLECTION_TEMPLATE: &str = "nidus://collections/{collection}";
pub(super) const ENTRY_TEMPLATE: &str = "nidus://collections/{collection}/entries/{id}";

/// What a `nidus://` URI addresses.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Target {
    Collection(String),
    Entry { collection: String, id: String },
}

pub(super) fn collection_uri(collection: &str) -> String {
    format!("{SCHEME}collections/{}", encode(collection))
}

pub(super) fn entry_uri(collection: &str, id: &str) -> String {
    format!(
        "{SCHEME}collections/{}/entries/{}",
        encode(collection),
        encode(id)
    )
}

/// `None` when the URI is not a well-formed `nidus://` resource URI.
pub(super) fn parse(uri: &str) -> Option<Target> {
    let rest = uri.strip_prefix(SCHEME)?;
    let segments: Vec<&str> = rest.split('/').collect();
    match segments.as_slice() {
        ["collections", c] => {
            let collection = decode(c)?;
            (!collection.is_empty()).then_some(Target::Collection(collection))
        }
        ["collections", c, "entries", id] => {
            let collection = decode(c)?;
            let id = decode(id)?;
            (!collection.is_empty() && !id.is_empty()).then_some(Target::Entry { collection, id })
        }
        _ => None,
    }
}

/// Keep `A-Z a-z 0-9 - . _ ~` verbatim; every other UTF-8 byte becomes `%XX` (uppercase hex).
fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Accepts upper- or lower-case hex; `None` on a truncated/non-hex escape or invalid UTF-8.
fn decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 3 > bytes.len() {
                return None;
            }
            let hi = hex_digit(bytes[i + 1])?;
            let lo = hex_digit(bytes[i + 2])?;
            out.push((hi << 4) | lo);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_plain_name() {
        let uri = entry_uri("newsletters", "e1");
        assert_eq!(
            parse(&uri),
            Some(Target::Entry {
                collection: "newsletters".to_string(),
                id: "e1".to_string(),
            })
        );
    }

    #[test]
    fn round_trips_a_name_containing_slash() {
        let uri = entry_uri("a/b", "id/1");
        assert_eq!(
            parse(&uri),
            Some(Target::Entry {
                collection: "a/b".to_string(),
                id: "id/1".to_string(),
            })
        );
    }

    #[test]
    fn round_trips_a_name_containing_percent() {
        let uri = entry_uri("50%", "id");
        assert_eq!(
            parse(&uri),
            Some(Target::Entry {
                collection: "50%".to_string(),
                id: "id".to_string(),
            })
        );
    }

    #[test]
    fn round_trips_a_name_containing_space() {
        let uri = entry_uri("my collection", "an id");
        assert_eq!(
            parse(&uri),
            Some(Target::Entry {
                collection: "my collection".to_string(),
                id: "an id".to_string(),
            })
        );
    }

    #[test]
    fn round_trips_a_non_ascii_name() {
        let uri = entry_uri("réunions", "id");
        assert_eq!(
            parse(&uri),
            Some(Target::Entry {
                collection: "réunions".to_string(),
                id: "id".to_string(),
            })
        );
    }

    /// A `/` in a name must be percent-encoded rather than split the path: the literal
    /// (unencoded) slash count in an entry URI is always 5, regardless of what the
    /// collection/id names contain.
    #[test]
    fn slash_in_a_name_encodes_rather_than_splits_the_path() {
        let plain = entry_uri("plain", "id");
        let with_slash = entry_uri("a/b/c", "id");
        assert_eq!(plain.matches('/').count(), 5);
        assert_eq!(with_slash.matches('/').count(), 5);
    }

    #[test]
    fn emitted_uris_are_pure_ascii_with_no_space_or_control_bytes() {
        for uri in [
            collection_uri("réunions/wéird one"),
            entry_uri("a b", "c\td\ne"),
        ] {
            assert!(uri.is_ascii());
            assert!(!uri.bytes().any(|b| b == b' ' || b.is_ascii_control()));
        }
    }

    #[test]
    fn parse_rejects_malformed_uris() {
        assert_eq!(parse("file:///etc/passwd"), None);
        assert_eq!(parse("nidus://collections"), None);
        assert_eq!(parse("nidus://collections/a/b/c"), None);
        assert_eq!(parse("nidus://collections/a/entries/b/c"), None);
        assert_eq!(parse("nidus://collections/%ZZ"), None);
        assert_eq!(parse("nidus://collections/%4"), None);
    }
}
