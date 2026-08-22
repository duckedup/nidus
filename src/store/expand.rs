//! Neighbour expansion (SPEC §7.9): widen each hit with the surrounding chunks of its own
//! document. Payload only — it writes [`Hit::context`] and never `attrs`, never reorders,
//! and never adds or drops a hit, so the total order §7 defines is untouched.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

use crate::model::{Expand, Hit, Value};

use super::Store;

/// One chunk of a document, as the stitcher needs it.
struct Piece<'a> {
    text: &'a str,
    char_start: Option<i64>,
}

/// Refuse an unusable field name once per query. `radius: 0` is legal: it reports the hit's
/// own text, which is what a caller asking for a context field wants when it widens nothing.
pub(super) fn validate(expand: Option<&Expand>) -> Result<()> {
    let Some(e) = expand else { return Ok(()) };
    for (name, value) in [
        ("parent_field", &e.parent_field),
        ("index_field", &e.index_field),
        ("text_field", &e.text_field),
    ] {
        if value.trim().is_empty() {
            bail!("expand needs a {name}");
        }
    }
    Ok(())
}

/// The chunk coordinates of one hit, read from the **stored** record rather than the hit's
/// projected attrs, so a projected-away field still expands.
fn coords(store: &Store, hit: &Hit, opts: &Expand) -> Option<(String, i64)> {
    let attrs = &store
        .collections
        .get(&hit.collection)?
        .docs
        .get(&hit.id)?
        .attrs;
    let parent = match attrs.get(&opts.parent_field)? {
        Value::Str(s) => s.clone(),
        _ => return None,
    };
    let index = match attrs.get(&opts.index_field)? {
        Value::Int(i) => *i,
        _ => return None,
    };
    Some((parent, index))
}

impl Store {
    /// Attach [`Hit::context`] to every hit that carries chunk coordinates. A record without
    /// them keeps `None` rather than failing the query — a mixed collection must still answer.
    pub(crate) fn expand_hits(&self, hits: &mut [Hit], opts: &Expand) {
        let radius = opts.radius as i64;
        let windows: Vec<Option<(String, i64)>> =
            hits.iter().map(|h| coords(self, h, opts)).collect();
        if windows.iter().all(Option::is_none) {
            return;
        }

        // One pass per involved collection, not one per hit: gather every chunk any window
        // wants, keyed by parent then index so the stitch below is already in source order.
        let mut wanted: BTreeMap<String, BTreeMap<String, ()>> = BTreeMap::new();
        for (hit, w) in hits.iter().zip(&windows) {
            if let Some((parent, _)) = w {
                wanted
                    .entry(hit.collection.clone())
                    .or_default()
                    .insert(parent.clone(), ());
            }
        }
        let mut pieces: BTreeMap<(String, String), BTreeMap<i64, Piece<'_>>> = BTreeMap::new();
        for (name, parents) in &wanted {
            let Some(col) = self.collections.get(name) else {
                continue;
            };
            for entry in col.docs.values() {
                let Some(Value::Str(parent)) = entry.attrs.get(&opts.parent_field) else {
                    continue;
                };
                if !parents.contains_key(parent) {
                    continue;
                }
                let Some(Value::Int(index)) = entry.attrs.get(&opts.index_field) else {
                    continue;
                };
                let Some(Value::Str(text)) = entry.attrs.get(&opts.text_field) else {
                    continue;
                };
                let char_start = match entry.attrs.get(crate::model::META_CHAR_START) {
                    Some(Value::Int(c)) => Some(*c),
                    _ => None,
                };
                pieces
                    .entry((name.clone(), parent.clone()))
                    .or_default()
                    .insert(*index, Piece { text, char_start });
            }
        }

        for (hit, w) in hits.iter_mut().zip(&windows) {
            let Some((parent, index)) = w else { continue };
            let Some(by_index) = pieces.get(&(hit.collection.clone(), parent.clone())) else {
                continue;
            };
            let lo = index.saturating_sub(radius);
            let hi = index.saturating_add(radius);
            hit.context = stitch(by_index.range(lo..=hi).map(|(_, p)| p));
        }
    }
}

/// Join a window in source order, dropping the overlap two adjacent chunks share. Exact when
/// both carry [`crate::model::META_CHAR_START`] (chunks are char slices of the source, so the
/// offsets reconstruct it byte for byte); a blank-line join otherwise.
fn stitch<'a>(window: impl Iterator<Item = &'a Piece<'a>>) -> Option<String> {
    let mut out = String::new();
    // How far into the source `out` already reaches. Monotonic: the offsets come from
    // ordinary attrs, so a hand-written record must not be able to rewind it and have
    // already-written text emitted twice.
    let mut written_to: Option<i64> = None;
    for piece in window {
        let len = piece.text.chars().count() as i64;
        let (text, contiguous) = match (written_to, piece.char_start) {
            // Overlapping slices of one source: skip the part already written.
            (Some(end), Some(start)) if start < end => {
                let skip = (end - start).min(len) as usize;
                (piece.text.chars().skip(skip).collect::<String>(), true)
            }
            (Some(end), Some(start)) => (piece.text.to_string(), start == end),
            _ => (piece.text.to_string(), false),
        };
        if let Some(start) = piece.char_start {
            written_to = Some(written_to.unwrap_or(i64::MIN).max(start + len));
        }
        if text.is_empty() {
            continue;
        }
        if !out.is_empty() && !contiguous {
            out.push_str("\n\n");
        }
        out.push_str(&text);
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Projection, Record, SearchOpts};

    const SOURCE: &str = "alpha beta gamma delta epsilon zeta eta theta";

    /// Three overlapping chunks of `SOURCE`, each an exact char slice, as `remember_chunked`
    /// would stamp them: `(char_start, char_end)`.
    const SPANS: [(usize, usize); 3] = [(0, 20), (15, 34), (29, 45)];

    fn chunked_store(with_offsets: bool) -> Store {
        let mut store = Store::in_memory(2).unwrap();
        let records: Vec<Record> = SPANS
            .iter()
            .enumerate()
            .map(|(i, &(start, end))| {
                let text: String = SOURCE.chars().take(end).skip(start).collect();
                let mut attrs = BTreeMap::new();
                attrs.insert(
                    crate::model::META_PARENT_ID.to_string(),
                    Value::Str("doc".to_string()),
                );
                attrs.insert(
                    crate::model::META_CHUNK_INDEX.to_string(),
                    Value::Int(i as i64),
                );
                attrs.insert(crate::model::META_TEXT.to_string(), Value::Str(text));
                if with_offsets {
                    attrs.insert(
                        crate::model::META_CHAR_START.to_string(),
                        Value::Int(start as i64),
                    );
                }
                Record::new(format!("doc#{i}"), vec![1.0, i as f32], attrs)
            })
            .collect();
        store.upsert("c", &records).unwrap();
        store
    }

    fn hit(store: &Store, id: &str) -> Hit {
        let attrs = store.collections["c"].docs[id].attrs.clone();
        Hit::new("c", id, 1.0, attrs)
    }

    #[test]
    fn a_window_with_offsets_reconstructs_the_source_exactly() {
        let store = chunked_store(true);
        let mut hits = vec![hit(&store, "doc#1")];
        store.expand_hits(&mut hits, &Expand::new(1));
        // The whole document, with neither seam repeating the 5 chars its chunks share.
        assert_eq!(hits[0].context.as_deref(), Some(SOURCE));
    }

    #[test]
    fn a_window_without_offsets_falls_back_to_a_blank_line_join() {
        let store = chunked_store(false);
        let mut hits = vec![hit(&store, "doc#1")];
        store.expand_hits(&mut hits, &Expand::new(1));
        let context = hits[0].context.clone().unwrap();
        assert!(context.contains("\n\n"), "{context}");
        // The overlap survives, which is exactly why the offsets are worth stamping.
        assert!(context.len() > SOURCE.len(), "{context}");
    }

    #[test]
    fn a_window_is_clipped_at_both_ends_of_the_document() {
        let store = chunked_store(true);
        let mut hits = vec![hit(&store, "doc#0"), hit(&store, "doc#2")];
        store.expand_hits(&mut hits, &Expand::new(5));
        // A radius past either end widens to the whole document rather than failing.
        assert_eq!(hits[0].context.as_deref(), Some(SOURCE));
        assert_eq!(hits[1].context.as_deref(), Some(SOURCE));
    }

    #[test]
    fn radius_zero_reports_the_hits_own_text() {
        let store = chunked_store(true);
        let mut hits = vec![hit(&store, "doc#1")];
        store.expand_hits(&mut hits, &Expand::new(0));
        let own: String = SOURCE.chars().take(SPANS[1].1).skip(SPANS[1].0).collect();
        assert_eq!(hits[0].context, Some(own));
    }

    #[test]
    fn expansion_never_crosses_a_document_boundary() {
        let mut store = chunked_store(true);
        let mut attrs = BTreeMap::new();
        attrs.insert(
            crate::model::META_PARENT_ID.to_string(),
            Value::Str("other".to_string()),
        );
        attrs.insert(crate::model::META_CHUNK_INDEX.to_string(), Value::Int(1));
        attrs.insert(
            crate::model::META_TEXT.to_string(),
            Value::Str("CONTAMINANT".to_string()),
        );
        attrs.insert(crate::model::META_CHAR_START.to_string(), Value::Int(0));
        store
            .upsert("c", &[Record::new("other#1", vec![1.0, 9.0], attrs)])
            .unwrap();

        let mut hits = vec![hit(&store, "doc#1")];
        store.expand_hits(&mut hits, &Expand::new(2));
        assert!(!hits[0].context.as_deref().unwrap().contains("CONTAMINANT"));
    }

    #[test]
    fn a_record_without_chunk_coordinates_is_left_alone() {
        let mut store = chunked_store(true);
        let mut attrs = BTreeMap::new();
        attrs.insert(
            crate::model::META_TEXT.to_string(),
            Value::Str("a plain memory".to_string()),
        );
        store
            .upsert("c", &[Record::new("plain", vec![1.0, 0.5], attrs)])
            .unwrap();

        let mut hits = vec![hit(&store, "plain"), hit(&store, "doc#1")];
        store.expand_hits(&mut hits, &Expand::new(1));
        assert_eq!(
            hits[0].context, None,
            "a non-chunked record gets no context"
        );
        assert!(hits[1].context.is_some(), "its neighbour still expands");
    }

    /// The ticket's provable-ordering criterion: expansion is payload-only, so the ranked
    /// `(id, score)` sequence must be identical with it on and off.
    #[test]
    fn expansion_does_not_change_the_ranking() {
        let store = chunked_store(true);
        let q = [1.0, 1.0];
        let base = SearchOpts {
            top_k: 10,
            ..Default::default()
        };
        let plain = store.search(&["c"], &q, &base).unwrap();
        let expanded = store
            .search(
                &["c"],
                &q,
                &SearchOpts {
                    expand: Some(Expand::new(1)),
                    ..base.clone()
                },
            )
            .unwrap();

        let seq = |hits: &[Hit]| -> Vec<(String, f32)> {
            hits.iter().map(|h| (h.id.clone(), h.score)).collect()
        };
        assert_eq!(seq(&plain), seq(&expanded));
        assert!(plain.iter().all(|h| h.context.is_none()));
        assert!(expanded.iter().all(|h| h.context.is_some()));
    }

    /// Coordinates and text are read from the stored record, not the projected hit, so a
    /// caller that projected the body away still gets its context (SPEC §7).
    #[test]
    fn a_projected_away_body_still_expands() {
        let store = chunked_store(true);
        let hits = store
            .search(
                &["c"],
                &[1.0, 1.0],
                &SearchOpts {
                    top_k: 1,
                    projection: Projection::Exclude(vec![
                        crate::model::META_TEXT.to_string(),
                        crate::model::META_PARENT_ID.to_string(),
                        crate::model::META_CHUNK_INDEX.to_string(),
                    ]),
                    expand: Some(Expand::new(1)),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(!hits[0].attrs.contains_key(crate::model::META_TEXT));
        assert!(hits[0].context.is_some(), "the stored text still expands");
    }

    /// The offsets are ordinary attrs, so a hand-written record can claim a `char_start`
    /// behind its predecessor. That must not rewind the stitcher into emitting text twice.
    #[test]
    fn a_rewound_char_start_does_not_duplicate_written_text() {
        let mut store = Store::in_memory(2).unwrap();
        // Chunk 1 is nested inside chunk 0, and chunk 2 starts inside chunk 0 but *after*
        // where chunk 1 ends — so a watermark that follows chunk 1 backwards re-emits it.
        let spans = [(0i64, "AAAAA BBBBB"), (3, "AA"), (6, "BBBBB")];
        let records: Vec<Record> = spans
            .iter()
            .enumerate()
            .map(|(i, &(start, text))| {
                let mut attrs = BTreeMap::new();
                attrs.insert(
                    crate::model::META_PARENT_ID.to_string(),
                    Value::Str("doc".to_string()),
                );
                attrs.insert(
                    crate::model::META_CHUNK_INDEX.to_string(),
                    Value::Int(i as i64),
                );
                attrs.insert(crate::model::META_CHAR_START.to_string(), Value::Int(start));
                attrs.insert(
                    crate::model::META_TEXT.to_string(),
                    Value::Str(text.to_string()),
                );
                Record::new(format!("doc#{i}"), vec![1.0, i as f32], attrs)
            })
            .collect();
        store.upsert("c", &records).unwrap();

        let mut hits = vec![hit(&store, "doc#0")];
        store.expand_hits(&mut hits, &Expand::new(2));
        let context = hits[0].context.clone().unwrap();
        assert_eq!(
            context.matches("BBBBB").count(),
            1,
            "already-written text was re-emitted: {context}"
        );
    }

    #[test]
    fn an_empty_field_name_is_a_query_error() {
        let mut e = Expand::new(1);
        e.parent_field = String::new();
        let err = validate(Some(&e)).unwrap_err().to_string();
        assert!(err.contains("parent_field"), "{err}");
    }
}
