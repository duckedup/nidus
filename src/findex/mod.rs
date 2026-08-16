//! The opt-in filter index behind the text predicates of `SPEC.md` §7.4/§7.5.
//!
//! **The index narrows; it never answers.** Every query here returns a *superset* of the
//! matching documents, and `filter::matches` then decides over the survivors. An
//! over-approximation costs a little scan time; an under-approximation returns wrong
//! answers, so every construct that cannot narrow soundly returns [`Candidates::All`]
//! rather than a smaller set.
//!
//! `fts/` is not reusable here on two independent grounds: its postings are keyed by the
//! *stemmed* term while filter tokens are deliberately unstemmed, and its postings carry no
//! positions, so a phrase predicate could not be served even if the analyzers agreed.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::model::{Predicate, Value};

mod literal;
mod persist;
mod schema;
mod tokens;
mod trigram;

#[cfg(test)]
mod tests;

pub use schema::FilterIndexField;
pub(crate) use schema::validate;

use tokens::TokenPostings;
use trigram::TrigramPostings;

/// The flattened text of attribute `field`: a `Str` directly, a `List` joined by spaces,
/// everything else empty. Joining a list is why a candidate may match tokens spread across
/// two elements, which the caller's verify step then rejects.
fn field_text(attrs: &BTreeMap<String, Value>, field: &str) -> String {
    match attrs.get(field) {
        Some(Value::Str(s)) => s.clone(),
        Some(Value::List(items)) => items.join(" "),
        _ => String::new(),
    }
}

/// One indexed `(collection, field)`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct FieldIndex {
    cfg_tokens: bool,
    cfg_trigrams: bool,
    tokens: TokenPostings,
    trigrams: TrigramPostings,
    /// docnum → owning doc id, `None` once tombstoned.
    docnum_to_id: Vec<Option<String>>,
    id_to_docnum: HashMap<String, u32>,
}

impl FieldIndex {
    fn new(f: &FilterIndexField) -> Self {
        Self {
            cfg_tokens: f.tokens,
            cfg_trigrams: f.trigrams,
            ..Default::default()
        }
    }

    /// Index (or re-index) `id`. Re-indexing tombstones the previous docnum and leaves its
    /// postings dangling — they are skipped at resolve time, exactly as FTS does it.
    fn index(&mut self, id: &str, text: &str) {
        self.tombstone(id);
        let docnum = self.docnum_to_id.len() as u32;
        if self.cfg_tokens {
            self.tokens.index(docnum, text);
        }
        if self.cfg_trigrams {
            self.trigrams.index(docnum, text);
        }
        self.docnum_to_id.push(Some(id.to_string()));
        self.id_to_docnum.insert(id.to_string(), docnum);
    }

    fn tombstone(&mut self, id: &str) {
        if let Some(d) = self.id_to_docnum.remove(id) {
            self.docnum_to_id[d as usize] = None;
        }
    }

    fn live_count(&self) -> usize {
        self.id_to_docnum.len()
    }

    fn heap_bytes(&self) -> usize {
        self.tokens.heap_bytes()
            + self.trigrams.heap_bytes()
            + self
                .docnum_to_id
                .iter()
                .map(|s| s.as_ref().map_or(0, String::len) + size_of::<Option<String>>())
                .sum::<usize>()
    }

    /// Resolve docnums to live ids, dropping tombstoned slots.
    fn ids<'a>(&'a self, docnums: &[u32]) -> Vec<&'a str> {
        docnums
            .iter()
            .filter_map(|d| self.docnum_to_id.get(*d as usize)?.as_deref())
            .collect()
    }
}

/// The whole store's filter index, keyed by `(collection, field)`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct Findex {
    fields: HashMap<(String, String), FieldIndex>,
    schema: HashMap<String, Vec<FilterIndexField>>,
}

impl Findex {
    pub(crate) fn is_active(&self) -> bool {
        !self.schema.is_empty()
    }

    pub(crate) fn schema_for(&self, collection: &str) -> Option<&[FilterIndexField]> {
        self.schema.get(collection).map(Vec::as_slice)
    }

    /// Opaque validity bytes for [`crate::index_cache`]: any schema change flips it, so a
    /// stale cache is rejected and rebuilt. Serialized through a `BTreeMap` so declaration
    /// order cannot change the encoding.
    pub(crate) fn cache_key(&self) -> Vec<u8> {
        let sorted: BTreeMap<&String, &Vec<FilterIndexField>> = self.schema.iter().collect();
        bincode::serialize(&sorted).unwrap_or_default()
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.fields.values().map(FieldIndex::heap_bytes).sum()
    }

    pub(crate) fn set_schema(&mut self, collection: &str, fields: &[FilterIndexField]) {
        self.fields.retain(|(c, _), _| c != collection);
        for f in fields {
            self.fields.insert(
                (collection.to_string(), f.field.clone()),
                FieldIndex::new(f),
            );
        }
        if fields.is_empty() {
            self.schema.remove(collection);
        } else {
            self.schema.insert(collection.to_string(), fields.to_vec());
        }
    }

    pub(crate) fn index_doc(
        &mut self,
        collection: &str,
        id: &str,
        attrs: &BTreeMap<String, Value>,
    ) {
        let Some(fields) = self.schema.get(collection).cloned() else {
            return;
        };
        for f in &fields {
            let text = field_text(attrs, &f.field);
            if let Some(idx) = self
                .fields
                .get_mut(&(collection.to_string(), f.field.clone()))
            {
                idx.index(id, &text);
            }
        }
    }

    pub(crate) fn remove_doc(&mut self, collection: &str, id: &str) {
        for ((c, _), idx) in &mut self.fields {
            if c == collection {
                idx.tombstone(id);
            }
        }
    }

    pub(crate) fn drop_collection(&mut self, collection: &str) {
        self.fields.retain(|(c, _), _| c != collection);
        self.schema.remove(collection);
    }

    /// Forget every posting but keep the declarations — the state a rebuild starts from.
    pub(crate) fn clear_indexes(&mut self) {
        let schema = std::mem::take(&mut self.schema);
        self.fields.clear();
        for (c, fields) in &schema {
            self.set_schema(c, fields);
        }
        self.schema = schema;
    }

    fn index_for(&self, collection: &str, field: &str) -> Option<&FieldIndex> {
        self.fields
            .get(&(collection.to_string(), field.to_string()))
    }

    /// Candidate **ids** for one leaf predicate, or `None` to scan everything.
    ///
    /// `limit` is the point past which narrowing stops paying, and it is checked against a
    /// cheap bound on the posting lists *before* any list is materialised: a hot term
    /// otherwise costs a full candidate build only to be thrown away, which measured 20-30%
    /// *slower* than the unindexed scan.
    pub(crate) fn candidate_ids(
        &self,
        collection: &str,
        pred: &Predicate,
        limit: usize,
    ) -> Option<Vec<String>> {
        let (field, docnums) = self.candidates(collection, pred, limit)?;
        if docnums.len() > limit {
            return None;
        }
        Some(
            field
                .ids(&docnums)
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
    }

    /// The narrowing itself. `None` is [`Candidates::All`] in `Option` form: no sound
    /// subset is available, so the caller must consider every document.
    fn candidates(
        &self,
        collection: &str,
        pred: &Predicate,
        limit: usize,
    ) -> Option<(&FieldIndex, Vec<u32>)> {
        match pred {
            Predicate::ContainsAllTokens(key, query) => {
                let idx = self.index_for(collection, key).filter(|i| i.cfg_tokens)?;
                let want: Vec<&str> = crate::filter::tokens(query).collect();
                if want.is_empty() {
                    return None; // the empty query matches any present text attribute
                }
                let lists: Vec<&[u32]> = want.iter().map(|t| idx.tokens.get(t)).collect();
                if shortest(&lists)? > limit {
                    return None;
                }
                Some((idx, tokens::intersect(&lists)?))
            }
            Predicate::ContainsTokenSequence(key, query) => {
                let idx = self.index_for(collection, key).filter(|i| i.cfg_tokens)?;
                let want: Vec<&str> = crate::filter::tokens(query).collect();
                if want.is_empty() {
                    return None; // vacuously true, so nothing may be excluded
                }
                // Ordering is not indexed; the phrase's tokens must all be present, and
                // `filter::matches` then checks adjacency on the survivors.
                let lists: Vec<&[u32]> = want.iter().map(|t| idx.tokens.get(t)).collect();
                if shortest(&lists)? > limit {
                    return None;
                }
                Some((idx, tokens::intersect(&lists)?))
            }
            Predicate::ContainsAnyToken(key, query) => {
                let idx = self.index_for(collection, key).filter(|i| i.cfg_tokens)?;
                let want: Vec<&str> = crate::filter::tokens(query).collect();
                if want.is_empty() {
                    return Some((idx, Vec::new())); // `Any([])` matches nothing
                }
                let lists: Vec<&[u32]> = want.iter().map(|t| idx.tokens.get(t)).collect();
                if lists.iter().map(|l| l.len()).max()? > limit {
                    return None;
                }
                Some((idx, tokens::union(&lists)))
            }
            Predicate::Fuzzy(key, needle, max_edits) => {
                let idx = self.index_for(collection, key).filter(|i| i.cfg_trigrams)?;
                let threshold = trigram::fuzzy_threshold(needle, *max_edits)?;
                let want = trigram::distinct_trigrams(needle);
                // No cheap pre-bound here on purpose. `sum(len)/threshold` is sound but far
                // too loose over a shared-prefix vocabulary, and applying it gave the whole
                // 66x Fuzzy win back. The build is cheap next to the DP it saves, so this
                // one predicate pays for the count and lets the post-check decide.
                Some((idx, idx.trigrams.at_least(&want, threshold)))
            }
            Predicate::Regex(key, pattern) => {
                let idx = self.index_for(collection, key).filter(|i| i.cfg_trigrams)?;
                let lits = literal::required_literals(pattern);
                // Every required literal's trigrams must all be present. A literal shorter
                // than 3 chars yields none and cannot narrow, so it is skipped rather than
                // treated as satisfied-by-nothing.
                let want: Vec<trigram::Trigram> = lits
                    .iter()
                    .flat_map(|l| trigram::distinct_trigrams(l))
                    .collect();
                if want.is_empty() {
                    return None;
                }
                let mut want = want;
                want.sort_unstable();
                want.dedup();
                let n = want.len();
                let lists: Vec<&[u32]> = want.iter().map(|t| idx.trigrams.get(t)).collect();
                if shortest(&lists)? > limit {
                    return None;
                }
                Some((idx, idx.trigrams.at_least(&want, n)))
            }
            _ => None,
        }
    }

    /// Live docs in `collection`, for the caller's cost guard. `None` when the collection
    /// is not indexed at all.
    pub(crate) fn live_docs(&self, collection: &str) -> Option<usize> {
        self.fields
            .iter()
            .find(|((c, _), _)| c == collection)
            .map(|(_, idx)| idx.live_count())
    }
}

/// The shortest posting list's length — an upper bound on any intersection of them.
/// `None` for an empty set of lists, which means there is nothing to bound.
fn shortest(lists: &[&[u32]]) -> Option<usize> {
    lists.iter().map(|l| l.len()).min()
}

pub(crate) use persist::{load, save};
