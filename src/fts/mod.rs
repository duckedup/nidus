//! Opt-in BM25 full-text search index (the FTS leg of SPEC.md §9).

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::model::Value;

mod analyzer;
mod fold;
mod highlight;
mod schema;

pub use analyzer::{Analyzer, Language};
pub(crate) use analyzer::{analyze, analyze_surface, analyze_with_prefix};
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

/// Whether one posting's document may count toward a `df`, by `(docnum, id)`: the docnum tests
/// membership of a head-term intersection (nidus-ucl), the id looks the document's attrs up for a
/// metadata filter (nidus-3j8). `None` everywhere means the unconditioned whole-corpus count.
pub(crate) type Admit<'a> = &'a dyn Fn(u32, &str) -> bool;

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
    /// term → postings, appended in docnum order. Ordered (not a `HashMap`) so prefix
    /// expansion is a range scan rather than a full vocabulary sweep (D1).
    postings: BTreeMap<String, Vec<Posting>>,
    /// folded surface form → its stem, for `suggest` alone. Ordered for the same reason
    /// `postings` is; `postings` cannot serve it, since "nidus" stems to "nidu" and a
    /// stem-keyed scan therefore goes empty exactly as the typist finishes the word.
    surface: BTreeMap<String, String>,
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

        // Surface forms are vocabulary, not per-doc state: a tombstone leaves them behind
        // exactly as it leaves postings keys, and `suggest` drops any whose stem has no
        // live doc left.
        for (surface, stemmed) in analyze_surface(text, self.cfg.analyzer) {
            self.surface.insert(surface, stemmed);
        }

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

    /// Stems to score for `fragment`: those carrying it, unioned with the stems of the *surface
    /// forms* carrying it (nidus-dnm — "running" is indexed as "run", so a stem-only scan misses
    /// both "runn" and the whole word). Commonest first, capped; `matched` reports a truncation.
    pub(crate) fn expand_prefix(&self, fragment: &str) -> (Vec<String>, usize) {
        // Keyed by stem, so a stem both legs reach is one term in the disjunction. Both legs
        // read the same `df_where`, so the two `df`s agree and either may win.
        let mut hits: BTreeMap<&str, usize> = BTreeMap::new();
        for (term, postings) in self
            .postings
            .range(fragment.to_string()..)
            .take_while(|(t, _)| t.starts_with(fragment))
        {
            let df = self.df_where(postings, None);
            if df > 0 {
                hits.insert(term, df);
            }
        }
        for (_, stemmed) in self
            .surface
            .range(fragment.to_string()..)
            .take_while(|(s, _)| s.starts_with(fragment))
        {
            let Some(postings) = self.postings.get(stemmed) else {
                continue;
            };
            let df = self.df_where(postings, None);
            if df > 0 {
                hits.insert(stemmed, df);
            }
        }
        let matched = hits.len();
        // (df desc, term asc): the tie-break is load-bearing, or two equal-df terms
        // truncate in whatever order the sort happened to leave them.
        let mut matches: Vec<(&str, usize)> = hits.into_iter().collect();
        matches.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        matches.truncate(MAX_PREFIX_EXPANSION);
        (
            matches.into_iter().map(|(t, _)| t.to_string()).collect(),
            matched,
        )
    }

    /// Postings whose owning doc is live and, when `admit` is given, admissible — the `df`
    /// every ranking here uses. With no `admit` and no tombstone every posting counts, so the
    /// length is the exact answer and nothing is walked (nidus-clv stage 1).
    fn df_where(&self, postings: &[Posting], admit: Option<Admit<'_>>) -> usize {
        match admit {
            None if self.tombstones == 0 => postings.len(),
            None => postings
                .iter()
                .filter(|p| self.docnum_to_id[p.docnum as usize].is_some())
                .count(),
            Some(admit) => postings
                .iter()
                .filter(|p| {
                    self.docnum_to_id[p.docnum as usize]
                        .as_deref()
                        .is_some_and(|id| admit(p.docnum, id))
                })
                .count(),
        }
    }

    /// Live docnums carrying **every** term in `heads` — the documents a multi-word prefix has
    /// described so far (nidus-ucl). `None` when there are no heads, which is not an empty set:
    /// no heads means unconditioned, an empty set means nothing continues the phrase.
    pub(crate) fn head_docs(&self, heads: &[String]) -> Option<HashSet<u32>> {
        if heads.is_empty() {
            return None;
        }
        // Rarest head first so the intersection shrinks fastest. A head with no postings makes the
        // whole conjunction empty rather than dropping out of it.
        let mut order: Vec<&String> = heads.iter().collect();
        order.sort_unstable_by_key(|t| self.postings.get(*t).map_or(0, Vec::len));
        let mut acc: HashSet<u32> = match self.postings.get(order[0]) {
            Some(ps) => ps
                .iter()
                .filter(|p| self.docnum_to_id[p.docnum as usize].is_some())
                .map(|p| p.docnum)
                .collect(),
            None => return Some(HashSet::new()),
        };
        for term in &order[1..] {
            let Some(ps) = self.postings.get(*term) else {
                return Some(HashSet::new());
            };
            acc = ps
                .iter()
                .map(|p| p.docnum)
                .filter(|d| acc.contains(d))
                .collect();
            if acc.is_empty() {
                break;
            }
        }
        Some(acc)
    }

    /// Surface forms carrying `fragment` as a prefix, each with the `df` of the stem it maps to
    /// counted through `admit`. Un-capped and unranked: the caller merges the scope's collections
    /// first, so the cap and its `matched` count are over the whole scope, not one collection.
    pub(crate) fn suggest_scored(
        &self,
        fragment: &str,
        admit: Option<Admit<'_>>,
    ) -> Vec<(String, usize)> {
        self.surface
            .range(fragment.to_string()..)
            .take_while(|(s, _)| s.starts_with(fragment))
            .filter_map(|(s, stemmed)| {
                let df = self
                    .postings
                    .get(stemmed)
                    .map_or(0, |ps| self.df_where(ps, admit));
                // Dropped, not ranked last: a completion no admissible document carries does not
                // continue the phrase and must not be offered (nidus-ucl DECIDED).
                (df > 0).then(|| (s.clone(), df))
            })
            .collect()
    }

    /// Whether this index currently holds document `id` (live).
    #[cfg(test)]
    fn contains(&self, id: &str) -> bool {
        self.id_to_docnum.contains_key(id)
    }
}

/// The most terms one prefix clause may **score**. Ranking by df means every matching term
/// is still df-scanned first, so this bounds the BM25 disjunction, not the range scan.
/// Truncates rather than erroring: typeahead's first keystroke must still answer.
pub(crate) const MAX_PREFIX_EXPANSION: usize = 256;

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
        /// Bump when the inverted-index layout or analyzer behaviour changes. `4` = the
        /// `surface` map `suggest` scans; `3` = postings became a `BTreeMap`; `2` = per-field
        /// BM25/analyzer params (`1` keyed the two store-wide constants separately).
        const FTS_CACHE_VERSION: u8 = 4;
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

    /// Terms carrying `fragment` as a prefix in `collection`.`field` (see
    /// [`FieldIndex::expand_prefix`]). `(vec![], 0)` when the field isn't indexed.
    pub(crate) fn expand_prefix(
        &self,
        collection: &str,
        field: &str,
        fragment: &str,
    ) -> (Vec<String>, usize) {
        match self
            .fields
            .get(&(collection.to_string(), field.to_string()))
        {
            Some(idx) => idx.expand_prefix(fragment),
            None => (Vec::new(), 0),
        }
    }

    /// Un-capped completions for `fragment` in `collection`.`field` with their conditioned `df`
    /// (see [`FieldIndex::suggest_scored`]). Empty when the field isn't indexed.
    pub(crate) fn suggest(
        &self,
        collection: &str,
        field: &str,
        fragment: &str,
        admit: Option<Admit<'_>>,
    ) -> Vec<(String, usize)> {
        match self
            .fields
            .get(&(collection.to_string(), field.to_string()))
        {
            Some(idx) => idx.suggest_scored(fragment, admit),
            None => Vec::new(),
        }
    }

    /// Live docnums in `collection`.`field` carrying every term in `heads` (see
    /// [`FieldIndex::head_docs`]). `None` when there are no heads or the field isn't indexed.
    pub(crate) fn head_docs(
        &self,
        collection: &str,
        field: &str,
        heads: &[String],
    ) -> Option<HashSet<u32>> {
        self.fields
            .get(&(collection.to_string(), field.to_string()))?
            .head_docs(heads)
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

    // ── prefix expansion ──────────────────────────────────────────────────────────

    #[test]
    fn expand_prefix_returns_every_matching_term_and_nothing_else() {
        let idx = idx_with(&[("d1", "cat car cap dog")]);
        let (mut terms, matched) = idx.expand_prefix("ca");
        terms.sort();
        assert_eq!(terms, vec!["cap", "car", "cat"]);
        assert_eq!(matched, 3);
    }

    #[test]
    fn a_term_whose_postings_are_all_tombstoned_is_not_returned() {
        let mut idx = idx_with(&[("d1", "cat"), ("d2", "car")]);
        idx.tombstone("d1");
        let (terms, matched) = idx.expand_prefix("ca");
        assert_eq!(terms, vec!["car"]);
        assert_eq!(
            matched, 1,
            "the tombstoned \"cat\" must not count as matched"
        );
    }

    /// `n` distinct single-term documents sharing the prefix `zz`, each its own term so
    /// every one has `df == 1` (a controlled tie, since the cap's boundary must not
    /// depend on which terms happen to sort first by relevance).
    fn zz_corpus(n: usize) -> FieldIndex {
        let docs: Vec<(String, String)> = (0..n)
            .map(|i| (format!("d{i}"), format!("zz{i:05}")))
            .collect();
        let refs: Vec<(&str, &str)> = docs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        idx_with(&refs)
    }

    #[test]
    fn exactly_the_cap_scores_every_matching_term() {
        let idx = zz_corpus(MAX_PREFIX_EXPANSION);
        let (terms, matched) = idx.expand_prefix("zz");
        assert_eq!(matched, MAX_PREFIX_EXPANSION);
        assert_eq!(terms.len(), MAX_PREFIX_EXPANSION);
    }

    #[test]
    fn one_past_the_cap_truncates_and_reports_it() {
        let idx = zz_corpus(MAX_PREFIX_EXPANSION + 1);
        let (terms, matched) = idx.expand_prefix("zz");
        assert_eq!(matched, MAX_PREFIX_EXPANSION + 1);
        assert_eq!(terms.len(), MAX_PREFIX_EXPANSION);
        assert!(matched > terms.len(), "truncation must be reportable");
    }

    /// The surface leg resolves to a *stem*, so a fragment carried by one spelling scores the
    /// whole stem family. It fires only where some indexed spelling carries the fragment:
    /// "runs" alone never reaches "run" from "running", since "runs" is the shorter word.
    #[test]
    fn the_surface_leg_scores_the_stem_a_spelling_resolves_to() {
        let idx = idx_with(&[("d1", "running"), ("d2", "runs")]);
        assert_eq!(idx.expand_prefix("running"), (vec!["run".to_string()], 1));
        assert_eq!(idx.score(&["run".to_string()]).len(), 2, "both docs score");

        let alone = idx_with(&[("d2", "runs")]);
        assert_eq!(alone.expand_prefix("running"), (vec![], 0));
        assert_eq!(alone.expand_prefix("run"), (vec!["run".to_string()], 1));
    }

    /// Both legs reach every one of these stems, so the cap and `matched` must count the
    /// union: counting the legs separately would report twice the vocabulary that exists.
    #[test]
    fn the_cap_and_matched_count_the_union_not_the_legs() {
        let docs: Vec<(String, String)> = (0..MAX_PREFIX_EXPANSION + 1)
            .map(|i| (format!("d{i}"), format!("zz{i:05}ing")))
            .collect();
        let refs: Vec<(&str, &str)> = docs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        let idx = idx_with(&refs);
        let (terms, matched) = idx.expand_prefix("zz");
        assert_eq!(matched, MAX_PREFIX_EXPANSION + 1);
        assert_eq!(terms.len(), MAX_PREFIX_EXPANSION);
    }

    #[test]
    fn truncation_is_deterministic_across_repeated_calls() {
        let idx = zz_corpus(MAX_PREFIX_EXPANSION + 1);
        let first = idx.expand_prefix("zz");
        let second = idx.expand_prefix("zz");
        assert_eq!(first, second);
    }

    /// nidus-dnm: the fragment is never stemmed, so before the surface leg was unioned in
    /// neither the half-typed "runn" nor the whole word "running" reached the stem "run".
    #[test]
    fn a_prefix_clause_matches_a_stem_shortened_word() {
        let idx = idx_with(&[("d1", "running")]);
        for typed in ["r", "ru", "run", "runn", "runni", "runnin", "running"] {
            assert_eq!(
                idx.expand_prefix(typed),
                (vec!["run".to_string()], 1),
                "prefix {typed:?} must expand to the indexed stem"
            );
        }
    }

    /// A stem reachable both by the stem scan and through a surface form is ONE term: the
    /// two legs read the same `df`, so a duplicate would double-count it in the disjunction.
    #[test]
    fn a_stem_reachable_by_both_legs_is_returned_once() {
        let idx = idx_with(&[("d1", "run running runs")]);
        assert_eq!(idx.expand_prefix("run"), (vec!["run".to_string()], 1));
    }

    /// The union is a superset of each leg. For "runn": the stem scan alone reaches "runner"
    /// (its own stem) and never "run"; the surface scan alone reaches "run" (via "running")
    /// and never "runner", whose surface form outranks the fragment.
    #[test]
    fn the_union_keeps_terms_only_one_leg_can_reach() {
        let idx = idx_with(&[("d1", "runner"), ("d2", "running")]);
        let (mut terms, matched) = idx.expand_prefix("runn");
        terms.sort();
        assert_eq!(terms, vec!["run", "runner"], "{terms:?}");
        assert_eq!(matched, 2);
    }

    /// A surface form outlives the doc that introduced it (vocabulary, not per-doc state),
    /// so the live-df filter is the only thing keeping a dead term out of the expansion.
    #[test]
    fn a_surface_form_whose_stem_has_no_live_posting_is_not_returned() {
        let mut idx = idx_with(&[("d1", "running"), ("d2", "cat")]);
        idx.tombstone("d1");
        assert_eq!(idx.expand_prefix("runn"), (vec![], 0));
        assert_eq!(idx.expand_prefix("run"), (vec![], 0));
        assert_eq!(idx.expand_prefix("ca"), (vec!["cat".to_string()], 1));
    }

    /// `suggest_scored` ranked the way `Store::suggest` ranks it (df desc, term asc). The cap and
    /// the `matched` count moved to the store with the scope merge, so they are asserted there.
    fn sug(idx: &FieldIndex, fragment: &str) -> Vec<(String, usize)> {
        sug_admit(idx, fragment, None)
    }

    /// [`sug`] with an admissibility predicate — the conditioned `df` a filter or a head-term
    /// intersection produces.
    fn sug_admit(
        idx: &FieldIndex,
        fragment: &str,
        admit: Option<Admit<'_>>,
    ) -> Vec<(String, usize)> {
        let mut v = idx.suggest_scored(fragment, admit);
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }

    // ── suggest_scored (surface-form completions) ──────────────────────────────────

    #[test]
    fn suggest_returns_live_df_per_completion() {
        let idx = idx_with(&[("d1", "cat"), ("d2", "cat"), ("d3", "cat car")]);
        let scored = sug(&idx, "ca");
        assert_eq!(scored, vec![("cat".to_string(), 3), ("car".to_string(), 1)]);
    }

    #[test]
    fn suggest_excludes_a_deleted_docs_only_term() {
        let mut idx = idx_with(&[("d1", "car")]);
        idx.tombstone("d1");
        assert!(sug(&idx, "ca").is_empty());
    }

    #[test]
    fn suggest_df_falls_when_one_of_several_docs_is_deleted() {
        let mut idx = idx_with(&[("d1", "car"), ("d2", "car"), ("d3", "car")]);
        idx.tombstone("d1");
        assert_eq!(sug(&idx, "ca"), vec![("car".to_string(), 2)]);
    }

    #[test]
    fn suggest_ties_break_on_term_ascending() {
        let idx = idx_with(&[("d1", "cat"), ("d2", "car")]);
        let scored = sug(&idx, "ca");
        assert_eq!(scored, vec![("car".to_string(), 1), ("cat".to_string(), 1)]);
    }

    /// The whole reason `surface` exists: "nidus" stems to "nidu", so a stem-keyed scan alone
    /// goes empty on the full word. `suggest` returns the spelling, a clause the stem, and
    /// since nidus-dnm both reach it — the difference is what they rank, not what they find.
    #[test]
    fn suggest_and_a_clause_both_reach_a_word_the_stem_keyed_scan_cannot() {
        let idx = idx_with(&[("d1", "nidus"), ("d2", "nidus")]);

        let scored = sug(&idx, "nidus");
        assert_eq!(scored, vec![("nidus".to_string(), 2)], "{scored:?}");

        assert_eq!(idx.expand_prefix("nidus"), (vec!["nidu".to_string()], 1));
        assert_eq!(idx.expand_prefix("nidu"), (vec!["nidu".to_string()], 1));
    }

    /// Every keystroke answers, which the stem-keyed scan could not do past "nidu".
    #[test]
    fn suggest_answers_at_every_keystroke_of_a_stemmed_word() {
        let idx = idx_with(&[("d1", "running")]);
        for typed in ["r", "ru", "run", "runn", "runni", "runnin", "running"] {
            assert_eq!(
                sug(&idx, typed),
                vec![("running".to_string(), 1)],
                "prefix {typed:?} must still complete to the word"
            );
        }
    }

    /// Two spellings of one stem are two completions, both carrying that stem's df.
    #[test]
    fn suggest_returns_each_surface_form_of_a_shared_stem() {
        let idx = idx_with(&[("d1", "running"), ("d2", "runs")]);
        let scored = sug(&idx, "run");
        assert_eq!(
            scored,
            vec![("running".to_string(), 2), ("runs".to_string(), 2)],
            "{scored:?}"
        );
    }

    // ── df_where / head_docs: the conditioned df (nidus-clv, nidus-3j8, nidus-ucl) ─

    /// nidus-clv stage 1. The length shortcut and the walk must be the same number, or the fast
    /// path is a silent wrong answer on exactly the corpus shape it exists for.
    #[test]
    fn df_of_a_tombstone_free_index_equals_the_walked_count() {
        let idx = idx_with(&[("d1", "cat"), ("d2", "cat"), ("d3", "car")]);
        assert_eq!(idx.tombstones, 0);
        let postings = &idx.postings["cat"];
        let walked = postings
            .iter()
            .filter(|p| idx.docnum_to_id[p.docnum as usize].is_some())
            .count();
        assert_eq!(idx.df_where(postings, None), walked);
        assert_eq!(idx.df_where(postings, None), 2);
    }

    /// And once a tombstone exists the shortcut must NOT fire: the length would over-count.
    #[test]
    fn df_walks_once_the_index_carries_a_tombstone() {
        let mut idx = idx_with(&[("d1", "cat"), ("d2", "cat")]);
        idx.tombstone("d1");
        let postings = &idx.postings["cat"];
        assert_eq!(postings.len(), 2, "the dead posting is still there");
        assert_eq!(idx.df_where(postings, None), 1);
    }

    #[test]
    fn df_counts_only_admissible_postings() {
        let idx = idx_with(&[("d1", "cat"), ("d2", "cat"), ("d3", "cat")]);
        let postings = &idx.postings["cat"];
        let only_d2 = |_: u32, id: &str| id == "d2";
        assert_eq!(idx.df_where(postings, Some(&only_d2)), 1);
        let none = |_: u32, _: &str| false;
        assert_eq!(idx.df_where(postings, Some(&none)), 0);
    }

    #[test]
    fn head_docs_intersects_and_distinguishes_no_heads_from_no_match() {
        let idx = idx_with(&[
            ("d1", "quick brown fox"),
            ("d2", "quick red fox"),
            ("d3", "brown bear"),
        ]);
        // No heads is unconditioned; it is not the same as an empty conjunction.
        assert!(idx.head_docs(&[]).is_none());

        let quick = idx.head_docs(&[q("quick")[0].clone()]).unwrap();
        assert_eq!(quick.len(), 2);

        // AND, not OR: only d1 carries both.
        let both = idx
            .head_docs(&[q("quick")[0].clone(), q("brown")[0].clone()])
            .unwrap();
        assert_eq!(both.len(), 1);

        // A head the corpus never spells empties the whole conjunction rather than dropping out.
        let absent = idx
            .head_docs(&[q("quick")[0].clone(), "zzzz".to_string()])
            .unwrap();
        assert!(absent.is_empty());
    }

    /// nidus-ucl at the index level: the head set, fed in as `admit`, is what narrows the df.
    #[test]
    fn suggest_df_narrows_to_the_head_matching_docs() {
        let idx = idx_with(&[
            ("d1", "quick brown fox"),
            ("d2", "brown bear"),
            ("d3", "brown owl"),
        ]);
        assert_eq!(sug(&idx, "brown"), vec![("brown".to_string(), 3)]);

        let heads = idx.head_docs(&[q("quick")[0].clone()]).unwrap();
        let admit = |docnum: u32, _: &str| heads.contains(&docnum);
        assert_eq!(
            sug_admit(&idx, "brown", Some(&admit)),
            vec![("brown".to_string(), 1)]
        );
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
    fn cache_key_carries_the_bumped_version_for_btreemap_postings() {
        // Guards each bump: a store built under an older `FieldIndex` bincode shape must
        // not be read back as valid by today's.
        let key = key_for(&[FtsField::new("body")]);
        assert_eq!(key[0], 4, "FTS_CACHE_VERSION must be 4, not the old 3");
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
