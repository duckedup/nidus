//! Opt-in BM25 full-text search index (the FTS leg of SPEC.md §9).
//!
//! A *derived secondary index*, like [`crate::ann`]: built from the documents'
//! text-bearing attrs, rebuildable from the op-log, and cached on disk only as an
//! optimization. Each full-text-indexed `(collection, field)` owns one [`FieldIndex`]
//! — an inverted index (term → postings) plus the per-doc lengths BM25 needs. A query
//! is [analyzed](analyzer) into terms, scored against a field, and ranked by BM25.
//!
//! Identity is an FTS-local **docnum** (a dense `[0, n)` id), not the data-matrix row:
//! FTS indexes only a subset of docs per field and text-only docs have no row at all.
//! A candidate docnum is *hint-verified* against the live `id ↔ docnum` maps before it
//! scores, so deletes/overwrites need no posting rewrite (they leave a tombstone the
//! lookup skips). Pure safe Rust, Miri-clean, zero FFI.

use std::collections::BTreeMap;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::model::Value;

mod analyzer;

pub use analyzer::Language;
pub(crate) use analyzer::analyze;

/// The full text of attribute `field` for FTS purposes: a `Str` directly, a `List`
/// joined by spaces (each element is its own run of terms), everything else empty.
fn field_text(attrs: &BTreeMap<String, Value>, field: &str) -> String {
    match attrs.get(field) {
        Some(Value::Str(s)) => s.clone(),
        Some(Value::List(items)) => items.join(" "),
        _ => String::new(),
    }
}

/// BM25 term-frequency saturation. Larger = term frequency matters more before
/// saturating. The conventional default.
const K1: f32 = 1.2;
/// BM25 length normalization. `0` = none, `1` = full. The conventional default.
const B: f32 = 0.75;

/// One posting: a document's local docnum and the term's frequency in this field.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Posting {
    docnum: u32,
    tf: u32,
}

/// The BM25 inverted index for one `(collection, field)`. `docnum` is dense and
/// FTS-local; `docnum_to_id[d]` is `None` once that doc is tombstoned (deleted or
/// overwritten), and `id_to_docnum` is the authoritative live mapping.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct FieldIndex {
    lang: Language,
    /// term → postings, appended in docnum order.
    postings: HashMap<String, Vec<Posting>>,
    /// docnum → field length in terms (`0` once tombstoned).
    doc_len: Vec<u32>,
    /// docnum → owning doc id, or `None` for a tombstoned slot.
    docnum_to_id: Vec<Option<String>>,
    /// doc id → its live docnum.
    id_to_docnum: HashMap<String, u32>,
    /// Live (non-tombstoned) docs — BM25's `N`.
    doc_count: u64,
    /// Sum of `doc_len` over live docs — `avgdl = total_len / doc_count`.
    total_len: u64,
    /// Tombstoned slots (a compaction-pressure signal).
    tombstones: u32,
}

impl FieldIndex {
    pub(crate) fn new(lang: Language) -> Self {
        Self {
            lang,
            ..Default::default()
        }
    }

    pub(crate) fn language(&self) -> Language {
        self.lang
    }

    /// Index (or re-index) document `id` with this field's `text`. Re-indexing an
    /// existing id tombstones its previous docnum first (lazy delete — the old postings
    /// stay but are skipped via hint-verify). O(terms in `text`).
    pub(crate) fn index(&mut self, id: &str, text: &str) {
        self.tombstone(id);

        let terms = analyze(text, self.lang);
        let len = terms.len() as u32;
        let docnum = self.docnum_to_id.len() as u32;

        // Term frequencies within this doc.
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for t in &terms {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }
        for (term, count) in tf {
            self.postings
                .entry(term.to_string())
                .or_default()
                .push(Posting { docnum, tf: count });
        }

        self.doc_len.push(len);
        self.docnum_to_id.push(Some(id.to_string()));
        self.id_to_docnum.insert(id.to_string(), docnum);
        self.doc_count += 1;
        self.total_len += len as u64;
    }

    /// Tombstone document `id` if present (delete or pre-overwrite). Postings are left
    /// dangling and skipped at query time; live counts are corrected immediately so
    /// `avgdl`/`N` track live docs (compaction later drops the dead postings).
    pub(crate) fn tombstone(&mut self, id: &str) {
        if let Some(docnum) = self.id_to_docnum.remove(id) {
            let d = docnum as usize;
            self.docnum_to_id[d] = None;
            self.total_len -= self.doc_len[d] as u64;
            self.doc_len[d] = 0;
            self.doc_count -= 1;
            self.tombstones += 1;
        }
    }

    /// Average field length over live docs (`1.0` when empty, to avoid a 0/0 in BM25).
    fn avgdl(&self) -> f32 {
        if self.doc_count == 0 {
            1.0
        } else {
            self.total_len as f32 / self.doc_count as f32
        }
    }

    /// Live BM25 score for every doc matching at least one already-analyzed
    /// `query_term`, as `(id, score)`. Unranked (the caller feeds these into the shared
    /// top-k heap so scope/filter/top-k apply uniformly with vector search). Takes
    /// pre-analyzed terms so a multi-collection query is analyzed once, not per field.
    pub(crate) fn score(&self, query_terms: &[String]) -> Vec<(&str, f32)> {
        if query_terms.is_empty() || self.doc_count == 0 {
            return Vec::new();
        }
        let avgdl = self.avgdl();
        let n = self.doc_count as f32;

        // docnum → accumulated score.
        let mut scores: HashMap<u32, f32> = HashMap::new();
        // De-dup query terms: a repeated query term doesn't change BM25 here (we score
        // document term frequency, not query term frequency).
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for term in query_terms {
            if !seen.insert(term.as_str()) {
                continue;
            }
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            // df over **live** postings only, so idf reflects the live corpus.
            let live: Vec<&Posting> = postings
                .iter()
                .filter(|p| self.docnum_to_id[p.docnum as usize].is_some())
                .collect();
            let df = live.len() as f32;
            if df == 0.0 {
                continue;
            }
            // BM25+ idf: the leading `1 +` keeps it positive for all df (defensive even
            // though df ≤ N here), so a common term never drags a score negative.
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for p in live {
                let dl = self.doc_len[p.docnum as usize] as f32;
                let tf = p.tf as f32;
                let norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / avgdl));
                *scores.entry(p.docnum).or_insert(0.0) += idf * norm;
            }
        }

        scores
            .into_iter()
            .filter_map(|(docnum, score)| {
                self.docnum_to_id[docnum as usize]
                    .as_deref()
                    .map(|id| (id, score))
            })
            .collect()
    }

    /// Whether this index currently holds document `id` (live).
    #[cfg(test)]
    fn contains(&self, id: &str) -> bool {
        self.id_to_docnum.contains_key(id)
    }
}

/// All FTS state for a store: the per-`(collection, field)` indexes plus the declared
/// schema (`collection → [(field, language)]`). The schema is the source of truth for
/// which attrs are full-text indexed; it is persisted via the op-log and replayed on
/// open.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct Fts {
    fields: HashMap<(String, String), FieldIndex>,
    schema: HashMap<String, Vec<(String, Language)>>,
}

impl Fts {
    /// Whether any collection has declared full-text fields. When false the store skips
    /// all FTS work on the hot path.
    pub(crate) fn is_active(&self) -> bool {
        !self.schema.is_empty()
    }

    /// The on-disk cache validity key: the cache format version, the BM25 params, and
    /// the full declared schema (deterministically ordered via `BTreeMap`). Any change
    /// to the analyzer params or the schema flips the key, so a stale cache is rejected
    /// by [`crate::index_cache`] and the index is rebuilt.
    pub(crate) fn cache_key(&self) -> Vec<u8> {
        /// Bump when the inverted-index layout or analyzer behaviour changes.
        const FTS_CACHE_VERSION: u8 = 1;
        let mut key = vec![FTS_CACHE_VERSION];
        key.extend_from_slice(&K1.to_le_bytes());
        key.extend_from_slice(&B.to_le_bytes());
        // BTreeMap iterates key-sorted, so the encoding is deterministic. Serializing a
        // BTreeMap of owned/Copy types is infallible; a silent drop here would weaken the
        // cache's validity (two schemas could share a key), so we assert rather than skip.
        let sorted: std::collections::BTreeMap<&String, &Vec<(String, Language)>> =
            self.schema.iter().collect();
        let bytes = bincode::serialize(&sorted).expect("FTS schema serialization is infallible");
        key.extend_from_slice(&bytes);
        key
    }

    /// The declared `(field, language)` list for `collection`, if any.
    pub(crate) fn schema_for(&self, collection: &str) -> Option<&[(String, Language)]> {
        self.schema.get(collection).map(Vec::as_slice)
    }

    /// The analyzer language declared for `collection`.`field`, if it is indexed.
    pub(crate) fn field_language(&self, collection: &str, field: &str) -> Option<Language> {
        self.fields
            .get(&(collection.to_string(), field.to_string()))
            .map(FieldIndex::language)
    }

    /// Fraction of indexed docs that are tombstoned (dead) across all field indexes —
    /// the FTS analog of the dead-row ratio, used to trigger an auto-compact rebuild for
    /// text-only workloads (whose deletes leave no data rows). `0.0` when nothing is
    /// indexed. Reads the per-field `tombstones`/`doc_count`.
    pub(crate) fn tombstone_ratio(&self) -> f32 {
        let mut tomb: u64 = 0;
        let mut live: u64 = 0;
        for idx in self.fields.values() {
            tomb += idx.tombstones as u64;
            live += idx.doc_count;
        }
        let total = tomb + live;
        if total == 0 {
            0.0
        } else {
            tomb as f32 / total as f32
        }
    }

    /// Declare (or redeclare) `collection`'s full-text fields, discarding any existing
    /// field indexes for it. The caller then re-indexes the collection's live docs.
    pub(crate) fn set_schema(&mut self, collection: &str, fields: &[(String, Language)]) {
        self.fields.retain(|(c, _), _| c != collection);
        for (field, lang) in fields {
            self.fields.insert(
                (collection.to_string(), field.clone()),
                FieldIndex::new(*lang),
            );
        }
        self.schema.insert(collection.to_string(), fields.to_vec());
    }

    /// Index document `id`'s text into every declared field of `collection`. A field
    /// with no text (absent / non-string attr) tombstones any prior value for that id,
    /// so a doc only lives in a field's index while it has text there. No-op if the
    /// collection has no FTS schema.
    pub(crate) fn index_doc(
        &mut self,
        collection: &str,
        id: &str,
        attrs: &BTreeMap<String, Value>,
    ) {
        let Fts { fields, schema } = self;
        let Some(decl) = schema.get(collection) else {
            return;
        };
        for (field, _lang) in decl {
            let Some(idx) = fields.get_mut(&(collection.to_string(), field.clone())) else {
                continue;
            };
            let text = field_text(attrs, field);
            if text.is_empty() {
                idx.tombstone(id);
            } else {
                idx.index(id, &text);
            }
        }
    }

    /// Tombstone document `id` across all of `collection`'s field indexes (delete).
    pub(crate) fn remove_doc(&mut self, collection: &str, id: &str) {
        if let Some(decl) = self.schema.get(collection) {
            for (field, _) in decl {
                if let Some(idx) = self
                    .fields
                    .get_mut(&(collection.to_string(), field.clone()))
                {
                    idx.tombstone(id);
                }
            }
        }
    }

    /// Drop `collection`'s schema and field indexes entirely (collection dropped).
    pub(crate) fn drop_collection(&mut self, collection: &str) {
        self.fields.retain(|(c, _), _| c != collection);
        self.schema.remove(collection);
    }

    /// Reset every field index to empty (keeping the declared schema), so the caller can
    /// re-index all live docs from scratch — used on compaction and on open.
    pub(crate) fn clear_indexes(&mut self) {
        for ((_, _), idx) in self.fields.iter_mut() {
            let lang = idx.language();
            *idx = FieldIndex::new(lang);
        }
    }

    /// BM25-score already-analyzed `query_terms` against `collection`.`field`, as
    /// `(id, score)` for live matches. Empty when the field isn't indexed or nothing
    /// matches. The caller analyzes the query once (per [`field_language`]) and reuses
    /// the term list across collections.
    pub(crate) fn score(
        &self,
        collection: &str,
        field: &str,
        query_terms: &[String],
    ) -> Vec<(&str, f32)> {
        match self
            .fields
            .get(&(collection.to_string(), field.to_string()))
        {
            Some(idx) => idx.score(query_terms),
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx_with(docs: &[(&str, &str)]) -> FieldIndex {
        let mut idx = FieldIndex::new(Language::English);
        for (id, text) in docs {
            idx.index(id, text);
        }
        idx
    }

    /// Analyze a query string into terms (the analysis the store does once per query).
    fn q(query: &str) -> Vec<String> {
        analyze(query, Language::English)
    }

    /// `score` ranked descending, for assertions.
    fn ranked(idx: &FieldIndex, query: &str) -> Vec<(String, f32)> {
        let mut v: Vec<(String, f32)> = idx
            .score(&q(query))
            .into_iter()
            .map(|(id, s)| (id.to_string(), s))
            .collect();
        // Score desc, then id asc — a deterministic order despite HashMap iteration.
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        v
    }

    #[test]
    fn ranks_by_relevance_and_stems() {
        let idx = idx_with(&[
            ("d1", "the cat sat on the mat"),
            ("d2", "cats and more cats running with cats"),
            ("d3", "a dog barked"),
        ]);
        let hits = ranked(&idx, "cat");
        // d2 mentions "cats" most → highest; d3 has no cat term → absent.
        assert_eq!(hits[0].0, "d2");
        assert_eq!(hits[1].0, "d1");
        assert!(!hits.iter().any(|(id, _)| id == "d3"));
        assert!(hits.iter().all(|(_, s)| *s > 0.0));
    }

    #[test]
    fn query_is_stemmed_to_match_documents() {
        let idx = idx_with(&[("d1", "developers love running tests")]);
        // "run" (query) stems to the same root as "running" (doc).
        assert_eq!(idx.score(&q("run")).len(), 1);
        assert_eq!(idx.score(&q("RUNNING")).len(), 1);
    }

    #[test]
    fn tombstone_removes_doc_from_results_and_fixes_counts() {
        let mut idx = idx_with(&[("d1", "alpha beta"), ("d2", "alpha gamma")]);
        assert_eq!(idx.doc_count, 2);
        idx.tombstone("d1");
        assert_eq!(idx.doc_count, 1);
        assert!(!idx.contains("d1"));
        let hits = idx.score(&q("alpha"));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "d2");
    }

    #[test]
    fn reindex_overwrites_previous_text() {
        let mut idx = idx_with(&[("d1", "alpha beta")]);
        idx.index("d1", "gamma delta");
        assert_eq!(idx.doc_count, 1); // still one live doc
        assert!(idx.score(&q("alpha")).is_empty(), "old term gone");
        assert_eq!(idx.score(&q("gamma")).len(), 1, "new term present");
    }

    #[test]
    fn idf_stays_positive_when_term_is_in_every_doc() {
        // A term present in all docs has df == N; the BM25+ `1 +` keeps idf > 0, so
        // scores never go negative (which would invert ranking).
        let idx = idx_with(&[("d1", "common"), ("d2", "common"), ("d3", "common")]);
        let hits = idx.score(&q("common"));
        assert_eq!(hits.len(), 3);
        assert!(
            hits.iter().all(|(_, s)| *s > 0.0),
            "idf must stay positive at df==N"
        );
    }

    #[test]
    fn shorter_docs_score_higher_for_same_tf() {
        // Length normalization (b): the same single occurrence is worth more in a short
        // doc than a long one.
        let idx = idx_with(&[
            ("short", "needle"),
            (
                "long",
                "needle and a whole lot of other unrelated padding words here",
            ),
        ]);
        let hits = ranked(&idx, "needle");
        assert_eq!(hits[0].0, "short");
    }

    #[test]
    fn empty_and_unknown_queries_return_nothing() {
        let idx = idx_with(&[("d1", "alpha")]);
        assert!(idx.score(&q("")).is_empty());
        assert!(idx.score(&q("the and of")).is_empty()); // all stopwords
        assert!(idx.score(&q("zzz")).is_empty()); // unknown term
    }

    #[test]
    fn field_index_serde_roundtrips() {
        let idx = idx_with(&[("d1", "alpha beta"), ("d2", "beta gamma")]);
        let bytes = bincode::serialize(&idx).unwrap();
        let restored: FieldIndex = bincode::deserialize(&bytes).unwrap();
        // Compare the ranking (ids), not exact scores: BM25's idf uses `ln`, which Miri
        // deliberately evaluates non-deterministically (last-ULP), so two score
        // computations can differ by an ULP while the ranking is identical.
        let ids = |i: &FieldIndex| {
            ranked(i, "beta")
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(&idx), ids(&restored));
    }
}
