//! Opt-in BM25 full-text search index (the FTS leg of SPEC.md §9).

use std::collections::BTreeMap;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::model::Value;

mod analyzer;
mod fold;
mod highlight;
mod schema;

pub(crate) use analyzer::analyze;
pub use analyzer::{Analyzer, Language};
pub(crate) use highlight::fragments;
pub use schema::FtsField;
pub(crate) use schema::validate;

/// The full text of attribute `field` for FTS purposes: a `Str` directly, a `List`
/// joined by spaces (each element is its own run of terms), everything else empty.
/// Highlight offsets are into **this** string, so a `List` field's spans index the join.
pub(crate) fn field_text(attrs: &BTreeMap<String, Value>, field: &str) -> String {
    match attrs.get(field) {
        Some(Value::Str(s)) => s.clone(),
        Some(Value::List(items)) => items.join(" "),
        _ => String::new(),
    }
}

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
    /// The field's declared BM25 params + analyzer — what this index was built under.
    cfg: FtsField,
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
    pub(crate) fn new(cfg: FtsField) -> Self {
        Self {
            cfg,
            ..Default::default()
        }
    }

    pub(crate) fn analyzer(&self) -> Analyzer {
        self.cfg.analyzer
    }

    /// Postings held for this field, live and tombstoned. Test-only (see `Fts::posting_count`).
    #[cfg(test)]
    pub(crate) fn posting_count(&self) -> usize {
        self.postings.values().map(|v| v.len()).sum()
    }

    /// Index (or re-index) document `id` with this field's `text`. Re-indexing an
    /// existing id tombstones its previous docnum first (lazy delete — the old postings
    /// stay but are skipped via hint-verify). O(terms in `text`).
    pub(crate) fn index(&mut self, id: &str, text: &str) {
        self.tombstone(id);

        let terms = analyze(text, self.cfg.analyzer);
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

    /// Live BM25 score for every doc matching at least one analyzed `query_term`, as `(id, score)`.
    /// Unranked — the caller feeds these into the shared top-k heap so scope/filter apply uniformly
    /// with vector search. Takes pre-analyzed terms so a multi-collection query analyzes once.
    pub(crate) fn score(&self, query_terms: &[String]) -> Vec<(&str, f32)> {
        if query_terms.is_empty() || self.doc_count == 0 {
            return Vec::new();
        }
        let avgdl = self.avgdl();
        let n = self.doc_count as f32;
        let (k1, b) = (self.cfg.k1, self.cfg.b);

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
                let norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avgdl));
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

/// All FTS state for a store: the per-`(collection, field)` indexes plus the declared schema
/// (`collection → [FtsField]`). The schema is the source of truth for which attrs are
/// full-text indexed; it is persisted via the op-log and replayed on open.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct Fts {
    fields: HashMap<(String, String), FieldIndex>,
    schema: HashMap<String, Vec<FtsField>>,
}

impl Fts {
    /// Whether any collection has declared full-text fields. When false the store skips
    /// all FTS work on the hot path.
    pub(crate) fn is_active(&self) -> bool {
        !self.schema.is_empty()
    }

    /// Total postings held, live and tombstoned. Test-only: the observable for "this write
    /// did not re-append", which scores cannot show because `index` corrects live counts.
    #[cfg(test)]
    pub(crate) fn posting_count(&self) -> usize {
        self.fields.values().map(|idx| idx.posting_count()).sum()
    }

    /// The on-disk cache validity key: format version plus the full declared schema —
    /// every field's name, BM25 `k1`/`b`, and analyzer — deterministically ordered. Any
    /// change flips it, so [`crate::index_cache`] rejects the stale cache and rebuilds.
    pub(crate) fn cache_key(&self) -> Vec<u8> {
        /// Bump when the inverted-index layout or analyzer behaviour changes. `2` = per-field
        /// BM25/analyzer params (`1` keyed the two store-wide constants separately).
        const FTS_CACHE_VERSION: u8 = 2;
        let mut key = vec![FTS_CACHE_VERSION];
        // BTreeMap iterates key-sorted, so the encoding is deterministic. Serializing a
        // BTreeMap of owned/Copy types is infallible; a silent drop here would weaken the
        // cache's validity (two schemas could share a key), so we assert rather than skip.
        let sorted: std::collections::BTreeMap<&String, &Vec<FtsField>> =
            self.schema.iter().collect();
        let bytes = bincode::serialize(&sorted).expect("FTS schema serialization is infallible");
        key.extend_from_slice(&bytes);
        key
    }

    /// The declared fields for `collection`, if any.
    pub(crate) fn schema_for(&self, collection: &str) -> Option<&[FtsField]> {
        self.schema.get(collection).map(Vec::as_slice)
    }

    /// The analyzer declared for `collection`.`field`, if it is indexed.
    pub(crate) fn field_analyzer(&self, collection: &str, field: &str) -> Option<Analyzer> {
        self.fields
            .get(&(collection.to_string(), field.to_string()))
            .map(FieldIndex::analyzer)
    }

    /// Fraction of indexed docs tombstoned across all field indexes — the FTS analog of the
    /// dead-row ratio, which triggers an auto-compact rebuild for text-only workloads whose deletes
    /// leave no data rows. `0.0` when nothing is indexed.
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
    pub(crate) fn set_schema(&mut self, collection: &str, fields: &[FtsField]) {
        self.fields.retain(|(c, _), _| c != collection);
        for f in fields {
            self.fields.insert(
                (collection.to_string(), f.field.clone()),
                FieldIndex::new(f.clone()),
            );
        }
        self.schema.insert(collection.to_string(), fields.to_vec());
    }

    /// Index document `id`'s text into every declared field of `collection`. A field with no text
    /// (absent / non-string attr) tombstones any prior value for that id, so a doc only lives in a
    /// field's index while it has text there. No-op if the collection has no FTS schema.
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
        for f in decl {
            let Some(idx) = fields.get_mut(&(collection.to_string(), f.field.clone())) else {
                continue;
            };
            let text = field_text(attrs, &f.field);
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
            for f in decl {
                if let Some(idx) = self
                    .fields
                    .get_mut(&(collection.to_string(), f.field.clone()))
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
            *idx = FieldIndex::new(idx.cfg.clone());
        }
    }

    /// BM25-score already-analyzed `query_terms` against `collection`.`field`, as `(id, score)` for
    /// live matches. Empty when the field isn't indexed or nothing matches. The caller analyzes the
    /// query once (per [`field_language`]) and reuses the term list across collections.
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
        idx_cfg(FtsField::new("body"), docs)
    }

    /// An index over `docs` built under an explicit field configuration.
    fn idx_cfg(cfg: FtsField, docs: &[(&str, &str)]) -> FieldIndex {
        let mut idx = FieldIndex::new(cfg);
        for (id, text) in docs {
            idx.index(id, text);
        }
        idx
    }

    /// Analyze a query string into terms (the analysis the store does once per query).
    fn q(query: &str) -> Vec<String> {
        analyze(query, Analyzer::default())
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

    // ── per-field BM25 tuning ─────────────────────────────────────────────────────

    #[test]
    fn default_params_reproduce_the_frozen_bm25_score() {
        // A regression freeze. Hand-checked against the BM25 formula at k1 = 1.2, b = 0.75:
        // idf = ln(1.6) = 0.470004, and d1 (tf 1, dl 3, avgdl 3) has norm exactly 1.
        let idx = idx_with(&[
            ("d1", "the cat sat on the mat"),
            ("d2", "cats and more cats running with cats"),
            ("d3", "a dog barked"),
        ]);
        let hits = ranked(&idx, "cat");
        let by_id = |id: &str| hits.iter().find(|(i, _)| i == id).unwrap().1;
        // Tolerance, not equality: BM25's idf uses `ln`, which Miri evaluates to within
        // an ULP rather than exactly. A parameter change moves these by ~1e-2.
        assert!((by_id("d1") - 0.470_003_63).abs() < 1e-6, "{hits:?}");
        assert!((by_id("d2") - 0.689_338_7).abs() < 1e-6, "{hits:?}");
    }

    #[test]
    fn raising_k1_raises_the_reward_for_repeated_terms() {
        let docs: &[(&str, &str)] = &[
            ("many", "needle needle needle needle and some padding"),
            ("one", "needle and some padding text goes here"),
        ];
        let gap = |k1: f32| {
            let idx = idx_cfg(FtsField::new("body").k1(k1), docs);
            let hits = ranked(&idx, "needle");
            assert_eq!(hits[0].0, "many");
            hits[0].1 - hits[1].1
        };
        // k1 = 0 saturates tf immediately, so both docs score on length alone.
        assert!(gap(0.0) < gap(1.2), "k1=0 must flatten the tf advantage");
        assert!(gap(1.2) < gap(3.0), "a larger k1 must widen it");
    }

    #[test]
    fn b_zero_drops_the_length_penalty() {
        let docs: &[(&str, &str)] = &[
            ("short", "needle"),
            (
                "long",
                "needle plus a lot of other unrelated padding words here",
            ),
        ];
        let idx = idx_cfg(FtsField::new("body").b(0.0), docs);
        let hits = ranked(&idx, "needle");
        // Same tf, no length normalization → identical scores (the default b ranks
        // "short" strictly first, as `shorter_docs_score_higher_for_same_tf` asserts).
        assert!((hits[0].1 - hits[1].1).abs() < 1e-6, "{hits:?}");
    }

    // ── analyzer configuration ────────────────────────────────────────────────────

    #[test]
    fn ascii_folding_makes_accented_and_bare_spellings_one_term() {
        let docs: &[(&str, &str)] = &[("d1", "the café was open")];
        let folded = FtsField::new("body").ascii_folding(true);
        let idx = idx_cfg(folded.clone(), docs);
        let terms = |text: &str| analyze(text, folded.analyzer);
        assert_eq!(idx.score(&terms("cafe")).len(), 1);
        assert_eq!(idx.score(&terms("café")).len(), 1);
        // Off (the default), the two spellings stay separate terms.
        let plain = idx_cfg(FtsField::new("body"), docs);
        assert!(plain.score(&q("cafe")).is_empty());
        assert_eq!(plain.score(&q("café")).len(), 1);
    }

    #[test]
    fn max_token_len_keeps_long_tokens_out_of_the_index() {
        let blob = "a".repeat(64);
        let docs = [("d1", format!("alpha {blob} omega"))];
        let docs: Vec<(&str, &str)> = docs.iter().map(|(i, t)| (*i, t.as_str())).collect();
        let capped = FtsField::new("body").max_token_len(8);
        let idx = idx_cfg(capped.clone(), &docs);
        assert_eq!(idx.score(&analyze("alpha", capped.analyzer)).len(), 1);
        assert!(idx.score(&analyze(&blob, capped.analyzer)).is_empty());
        // The dropped token doesn't count toward the doc length either.
        assert_eq!(idx.doc_len[0], 2);
    }

    // ── cache invalidation ────────────────────────────────────────────────────────

    /// The cache key a store would carry with `fields` declared on one collection.
    fn key_for(fields: &[FtsField]) -> Vec<u8> {
        let mut fts = Fts::default();
        fts.set_schema("docs", fields);
        fts.cache_key()
    }

    #[test]
    fn cache_key_changes_on_every_schema_parameter() {
        // The highest-consequence property here: a store whose schema changed must
        // rebuild. A key collision would serve postings scored under the old params.
        let base = FtsField::new("body");
        let baseline = key_for(std::slice::from_ref(&base));
        for variant in [
            base.clone().k1(1.201),
            base.clone().b(0.7501),
            base.clone().ascii_folding(true),
            base.clone().max_token_len(40),
            base.clone().language(Language::English).k1(0.0),
            FtsField::new("title"),
        ] {
            assert_ne!(
                baseline,
                key_for(std::slice::from_ref(&variant)),
                "{variant:?}"
            );
        }
        assert_ne!(baseline, key_for(&[base.clone(), FtsField::new("title")]));
        // Redeclaring the same schema must keep the key, or every reopen would rebuild.
        assert_eq!(baseline, key_for(&[base]));
    }

    #[test]
    fn cache_key_is_stable_across_declaration_order_of_collections() {
        let mut a = Fts::default();
        a.set_schema("x", &[FtsField::new("body")]);
        a.set_schema("y", &[FtsField::new("body")]);
        let mut b = Fts::default();
        b.set_schema("y", &[FtsField::new("body")]);
        b.set_schema("x", &[FtsField::new("body")]);
        assert_eq!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn schema_accessors_expose_the_declared_configuration() {
        let mut fts = Fts::default();
        fts.set_schema("docs", &[FtsField::new("body").ascii_folding(true).k1(1.7)]);
        let decl = fts.schema_for("docs").unwrap();
        assert_eq!(decl.len(), 1);
        assert_eq!(decl[0].k1, 1.7);
        assert!(fts.field_analyzer("docs", "body").unwrap().ascii_folding);
        assert!(fts.field_analyzer("docs", "title").is_none());
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
