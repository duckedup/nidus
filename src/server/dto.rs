//! Wire types for the HTTP API and the CLI's JSON I/O.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    AnnConfig, AnnKind, Annotations, Filter, Footprint, FtsClause, FtsCombine, FtsField,
    HighlightOpts, Hit, Language, ListOpts, Projection, Record, Value,
};

/// Body of `POST /collections/{name}/upsert`.
#[derive(Debug, Deserialize)]
pub struct UpsertRequest {
    pub records: Vec<Record>,
}

/// Body of `POST /collections/{name}/delete`. Supply `ids` to delete by id, or
/// `filter` to delete every matching record; `filter` wins if both are present.
#[derive(Debug, Default, Deserialize)]
pub struct DeleteRequest {
    #[serde(default)]
    pub ids: Vec<String>,
    #[serde(default)]
    pub filter: Option<Filter>,
}

/// The default result count. `pub(super)` so the MCP tools share it rather than picking
/// their own, which would answer one query two different ways depending on the surface.
pub(super) fn default_top_k() -> usize {
    10
}

/// The largest `offset + top_k` any request surface accepts. Past this a request is an
/// allocation demand rather than a query — no store returns ten thousand hits usefully — and
/// the bounded top-k kernel would otherwise be handed a `k` it must defend against itself.
pub(super) const MAX_TOP_K: usize = 10_000;

/// Body of `POST /search`. An empty `scope` searches every collection; `offset` skips
/// that many top-ranked hits, for pagination.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
    #[serde(default)]
    pub filter: Filter,
    /// Force the exact scan for this query, bypassing any index and the quantized first pass.
    #[serde(default)]
    pub exact: bool,
    #[serde(default)]
    pub include_attributes: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_attributes: Option<Vec<String>>,
}

/// Resolve a request's projection fields. The two are spelled out on each request rather than
/// `#[serde(flatten)]`ed from a shared struct, because flatten buffers the *whole* body — the
/// query vector included — through `Content` on every search.
pub(super) fn resolve_projection(
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
) -> Result<Projection, &'static str> {
    match (include, exclude) {
        (None, None) => Ok(Projection::All),
        (Some(keys), None) => Ok(Projection::Include(keys)),
        (None, Some(keys)) => Ok(Projection::Exclude(keys)),
        (Some(_), Some(_)) => Err(
            "include_attributes and exclude_attributes are mutually exclusive; send at most one",
        ),
    }
}

/// Derived from [`ListOpts`] so the wire default and the library default cannot drift.
fn default_limit() -> usize {
    ListOpts::default().limit
}

fn default_rrf_k() -> f32 {
    60.0
}

fn default_candidates() -> usize {
    100
}

/// One clause of a multi-field text query: `{"field": "title", "query": "rust"}`.
#[derive(Debug, Deserialize)]
pub struct FtsClauseDto {
    pub field: String,
    pub query: String,
}

/// Body of `POST /text-search` (BM25). An empty `scope` searches every collection. Name the
/// fields either as one `field` + `query` pair or as a `clauses` list — never both.
#[derive(Debug, Deserialize)]
pub struct TextSearchRequest {
    #[serde(default)]
    pub field: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub clauses: Option<Vec<FtsClauseDto>>,
    /// How several clauses fold into one score: `"Sum"` (default) or `"Max"`.
    #[serde(default)]
    pub combine: FtsCombine,
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub offset: usize,
    /// A raw BM25 score floor (not cosine).
    #[serde(default)]
    pub min_score: Option<f32>,
    #[serde(default)]
    pub filter: Filter,
    /// Report each matched clause's own BM25 score on every hit.
    #[serde(default)]
    pub explain: bool,
    /// Return highlighted fragments; `{}` takes the defaults. Highlighting reads the stored
    /// text, so it still works on a field the projection below dropped.
    #[serde(default)]
    pub highlight: Option<HighlightOpts>,
    #[serde(default)]
    pub include_attributes: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_attributes: Option<Vec<String>>,
}

/// Body of `POST /hybrid-search`: fuse a vector query and a BM25 text query (RRF). The text
/// leg takes the same `field`+`text` / `clauses` choice as `/text-search`.
#[derive(Debug, Deserialize)]
pub struct HybridSearchRequest {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub field: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub clauses: Option<Vec<FtsClauseDto>>,
    #[serde(default)]
    pub combine: FtsCombine,
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub filter: Filter,
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
    #[serde(default = "default_candidates")]
    pub candidates: usize,
    /// Report each leg's own rank and score, plus every matched clause's BM25 score.
    #[serde(default)]
    pub explain: bool,
    #[serde(default)]
    pub highlight: Option<HighlightOpts>,
}

/// Resolve a text query's clauses from the two accepted spellings: the single `field` +
/// `text` pair, or a non-empty `clauses` list. Both, neither, or an empty list is a caller
/// error — an empty result would otherwise read as "no matches" rather than "no query".
pub(super) fn resolve_clauses(
    field: Option<String>,
    text: Option<String>,
    clauses: Option<Vec<FtsClauseDto>>,
) -> Result<Vec<FtsClause>, &'static str> {
    match (field, text, clauses) {
        (Some(f), Some(t), None) => Ok(vec![FtsClause::new(f, t)]),
        (None, None, Some(list)) if !list.is_empty() => Ok(list
            .into_iter()
            .map(|c| FtsClause::new(c.field, c.query))
            .collect()),
        (None, None, Some(_)) => Err("clauses must not be empty"),
        (None, None, None) => Err("a text query needs a field plus its text, or a clauses list"),
        (_, _, Some(_)) => Err("field/text and clauses are mutually exclusive; send one form"),
        _ => Err("field and its query text must be sent together"),
    }
}

/// One entry of [`FtsSchemaRequest::fields`]: either a bare field name (everything
/// defaulted) or an object naming the field plus the params to override. The two forms
/// are why this is `untagged` — `{"fields": ["body"]}` must keep working verbatim.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum FtsFieldDto {
    /// `"body"` — default BM25 params, default analyzer.
    Name(String),
    /// `{"field": "body", "k1": 1.5, "ascii_folding": true}` — any subset of the knobs.
    Spec {
        field: String,
        #[serde(default)]
        k1: Option<f32>,
        #[serde(default)]
        b: Option<f32>,
        #[serde(default)]
        language: Option<Language>,
        #[serde(default)]
        ascii_folding: Option<bool>,
        #[serde(default)]
        max_token_len: Option<usize>,
    },
}

impl From<FtsFieldDto> for FtsField {
    fn from(dto: FtsFieldDto) -> Self {
        match dto {
            FtsFieldDto::Name(field) => FtsField::new(field),
            FtsFieldDto::Spec {
                field,
                k1,
                b,
                language,
                ascii_folding,
                max_token_len,
            } => {
                let mut f = FtsField::new(field);
                // Each knob is independent: an absent one keeps the default rather than
                // resetting a sibling, so a partial body means "change only this".
                f.k1 = k1.unwrap_or(f.k1);
                f.b = b.unwrap_or(f.b);
                f.analyzer.language = language.unwrap_or(f.analyzer.language);
                f.analyzer.ascii_folding = ascii_folding.unwrap_or(f.analyzer.ascii_folding);
                f.analyzer.max_token_len = max_token_len.or(f.analyzer.max_token_len);
                f
            }
        }
    }
}

/// Body of `POST /collections/{name}/fts-schema`: the attribute fields to full-text
/// index, each a name or a `{field, k1, b, language, ascii_folding, max_token_len}` object.
#[derive(Debug, Deserialize)]
pub struct FtsSchemaRequest {
    pub fields: Vec<FtsFieldDto>,
}

/// Body of `POST /list`. Metadata-only query (no vector). An empty `scope`
/// lists from every collection. `offset` skips matches for pagination.
#[derive(Debug, Deserialize)]
pub struct ListRequest {
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub filter: Filter,
    #[serde(default)]
    pub include_attributes: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_attributes: Option<Vec<String>>,
}

/// Body of `POST /collections/{name}/remember` (the `memory` feature).
#[cfg(feature = "memory")]
#[derive(Debug, Deserialize)]
pub struct RememberRequest {
    pub id: String,
    pub text: String,
    /// `"raw"` (embed the text as given, the default) or `"summarize"`
    /// (summarize first, embed the summary — requires a summarizer).
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub attrs: BTreeMap<String, Value>,
}

/// Body of `POST /collections/{name}/recall` (the `memory` feature).
#[cfg(feature = "memory")]
#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
    #[serde(default)]
    pub filter: Filter,
}

/// Serializable mirror of [`crate::Hit`] (which carries no serde derive).
#[derive(Debug, Serialize)]
pub struct HitDto {
    pub collection: String,
    pub id: String,
    pub score: f32,
    pub attrs: BTreeMap<String, Value>,
    /// Present only when the query asked to `explain` or to highlight, so an unannotated
    /// response is byte-identical to a nidus without annotations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Annotations>,
}

impl From<Hit> for HitDto {
    fn from(h: Hit) -> Self {
        Self {
            collection: h.collection,
            id: h.id,
            score: h.score,
            attrs: h.attrs,
            annotations: h.annotations,
        }
    }
}

/// Serializable mirror of [`crate::Footprint`].
#[derive(Debug, Serialize)]
pub struct FootprintDto {
    pub rows: u64,
    pub dead_rows: u64,
    pub dimension: usize,
    pub vector_bytes: u64,
    pub doc_count: usize,
}

impl From<Footprint> for FootprintDto {
    fn from(f: Footprint) -> Self {
        Self {
            rows: f.rows,
            dead_rows: f.dead_rows,
            dimension: f.dimension,
            vector_bytes: f.vector_bytes,
            doc_count: f.doc_count,
        }
    }
}

/// Serializable mirror of [`crate::AnnConfig`] for the `stats` surface. Only the
/// knobs that apply to the active [`AnnKind`] are emitted; the inert ones are
/// omitted. `stats` reports `null` when no ANN index is configured (exact search).
#[derive(Debug, Serialize)]
pub struct AnnDto {
    pub kind: String,
    pub overscan: usize,
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub m: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_construction: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_search: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_lists: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_probe: Option<usize>,
}

impl From<AnnConfig> for AnnDto {
    fn from(a: AnnConfig) -> Self {
        let (hnsw, ivf) = match a.kind {
            AnnKind::Hnsw => (true, false),
            AnnKind::Ivf => (false, true),
        };
        Self {
            kind: format!("{:?}", a.kind),
            overscan: a.overscan,
            seed: a.seed,
            m: hnsw.then_some(a.m),
            ef_construction: hnsw.then_some(a.ef_construction),
            ef_search: hnsw.then_some(a.ef_search),
            n_lists: ivf.then_some(a.n_lists),
            n_probe: ivf.then_some(a.n_probe),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::Predicate;

    /// Decode one `fields` entry exactly as the handler does.
    fn field(json: serde_json::Value) -> FtsField {
        serde_json::from_value::<FtsFieldDto>(json).unwrap().into()
    }

    #[test]
    fn a_bare_field_name_still_means_todays_defaults() {
        // The compatibility contract: `{"fields": ["body"]}` must not change meaning.
        let req: FtsSchemaRequest = serde_json::from_str(r#"{"fields":["body"]}"#).unwrap();
        let decoded: Vec<FtsField> = req.fields.into_iter().map(FtsField::from).collect();
        assert_eq!(decoded, vec![FtsField::new("body")]);
    }

    #[test]
    fn a_field_object_overrides_only_what_it_names() {
        let f = field(json!({"field": "body", "k1": 1.5}));
        assert_eq!(f.k1, 1.5);
        assert_eq!(f.b, 0.75, "an absent knob keeps its default");
        assert_eq!(f.analyzer, crate::Analyzer::default());

        let f = field(json!({"field": "body", "ascii_folding": true, "max_token_len": 40}));
        assert_eq!((f.k1, f.b), (1.2, 0.75));
        assert!(f.analyzer.ascii_folding);
        assert_eq!(f.analyzer.max_token_len, Some(40));

        // `{"field": "body"}` is the object spelling of the bare name.
        assert_eq!(field(json!({"field": "body"})), FtsField::new("body"));
    }

    #[test]
    fn the_language_accepts_both_json_spellings() {
        for form in ["English", "english", "en"] {
            let f = field(json!({"field": "body", "language": form}));
            assert_eq!(f.analyzer.language, Language::English);
        }
    }

    #[test]
    fn the_two_field_forms_mix_within_one_request() {
        let req: FtsSchemaRequest =
            serde_json::from_str(r#"{"fields":["title",{"field":"body","b":0.4}]}"#).unwrap();
        let decoded: Vec<FtsField> = req.fields.into_iter().map(FtsField::from).collect();
        assert_eq!(
            decoded,
            vec![FtsField::new("title"), FtsField::new("body").b(0.4)]
        );
    }

    /// Decode a `/text-search` body and resolve its clauses exactly as the handler does.
    fn clauses(json: serde_json::Value) -> Result<Vec<FtsClause>, &'static str> {
        let req: TextSearchRequest = serde_json::from_value(json).unwrap();
        resolve_clauses(req.field, req.query, req.clauses)
    }

    #[test]
    fn the_single_field_body_still_means_one_clause() {
        // The compatibility contract: `{"field": …, "query": …}` must not change meaning.
        assert_eq!(
            clauses(json!({"field": "body", "query": "run"})).unwrap(),
            vec![FtsClause::new("body", "run")]
        );
    }

    #[test]
    fn a_clauses_list_carries_distinct_text_per_field() {
        let got = clauses(json!({"clauses": [
            {"field": "title", "query": "rust"},
            {"field": "body", "query": "async runtime"}
        ]}))
        .unwrap();
        assert_eq!(
            got,
            vec![
                FtsClause::new("title", "rust"),
                FtsClause::new("body", "async runtime")
            ]
        );
    }

    #[test]
    fn an_unusable_clause_spelling_is_refused_rather_than_guessed() {
        for body in [
            json!({"clauses": []}),
            json!({}),
            json!({"field": "body"}),
            json!({"query": "run"}),
            json!({"field": "body", "query": "run", "clauses": [{"field": "t", "query": "x"}]}),
        ] {
            assert!(clauses(body.clone()).is_err(), "{body}");
        }
    }

    #[test]
    fn combine_accepts_both_json_spellings_and_defaults_to_sum() {
        let req: TextSearchRequest = serde_json::from_value(json!({"clauses": []})).unwrap();
        assert_eq!(req.combine, FtsCombine::Sum);
        for (form, want) in [
            ("Sum", FtsCombine::Sum),
            ("sum", FtsCombine::Sum),
            ("Max", FtsCombine::Max),
            ("max", FtsCombine::Max),
        ] {
            let req: TextSearchRequest =
                serde_json::from_value(json!({"clauses": [], "combine": form})).unwrap();
            assert_eq!(req.combine, want, "{form}");
        }
    }

    #[test]
    fn a_hit_without_annotations_serializes_exactly_as_it_always_did() {
        let hit = HitDto::from(Hit::new("docs", "a", 0.5, BTreeMap::new()));
        assert_eq!(
            serde_json::to_value(&hit).unwrap(),
            json!({"collection": "docs", "id": "a", "score": 0.5, "attrs": {}})
        );
    }

    /// The annotation wire spelling every SDK will mirror in nidus-m50.18. Asserted here
    /// because the crate's own round trip is bincode, which would not catch a rename.
    #[test]
    fn annotation_json_spelling_is_stable() {
        let mut hit = Hit::new("docs", "a", 0.5, BTreeMap::new());
        hit.annotations = Some(crate::Annotations {
            vector: Some(crate::LegScore {
                rank: 0,
                score: 0.5,
            }),
            text: Some(crate::LegScore {
                rank: 2,
                score: 1.5,
            }),
            clauses: vec![crate::ClauseScore {
                field: "title".into(),
                score: 1.5,
            }],
            highlights: vec![crate::Highlight {
                field: "body".into(),
                fragments: vec![crate::Fragment {
                    text: "we were running".into(),
                    spans: vec![(8, 15)],
                }],
            }],
        });
        assert_eq!(
            serde_json::to_value(HitDto::from(hit)).unwrap()["annotations"],
            json!({
                "vector": {"rank": 0, "score": 0.5},
                "text": {"rank": 2, "score": 1.5},
                "clauses": [{"field": "title", "score": 1.5}],
                "highlights": [{
                    "field": "body",
                    "fragments": [{"text": "we were running", "spans": [[8, 15]]}]
                }]
            })
        );
    }

    #[test]
    fn highlight_options_come_off_the_wire_with_their_defaults() {
        let req: TextSearchRequest =
            serde_json::from_value(json!({"clauses": [], "highlight": {}})).unwrap();
        assert_eq!(req.highlight, Some(HighlightOpts::default()));
        let req: TextSearchRequest =
            serde_json::from_value(json!({"clauses": [], "highlight": {"max_fragments": 3}}))
                .unwrap();
        assert_eq!(req.highlight.unwrap().max_fragments, 3);
        // Absent means no highlighting, and `explain` is off unless asked.
        let req: TextSearchRequest = serde_json::from_value(json!({"clauses": []})).unwrap();
        assert!(req.highlight.is_none() && !req.explain);
    }

    #[test]
    fn ann_dto_hnsw_emits_only_hnsw_knobs() {
        let v = serde_json::to_value(AnnDto::from(AnnConfig::hnsw())).unwrap();
        assert_eq!(v["kind"], "Hnsw");
        assert!(v.get("m").is_some());
        assert!(v.get("ef_search").is_some());
        // IVF-only knobs are skipped for an HNSW index.
        assert!(v.get("n_lists").is_none());
        assert!(v.get("n_probe").is_none());
    }

    #[test]
    fn ann_dto_ivf_emits_only_ivf_knobs() {
        let v = serde_json::to_value(AnnDto::from(AnnConfig::ivf())).unwrap();
        assert_eq!(v["kind"], "Ivf");
        assert!(v.get("n_probe").is_some());
        // HNSW-only knobs are skipped for an IVF index.
        assert!(v.get("m").is_none());
        assert!(v.get("ef_search").is_none());
    }

    /// The externally-tagged JSON every SDK mirrors by hand. Asserted here because the
    /// crate's own round-trip is bincode, which would not catch a change in the JSON
    /// spelling — and a silent change there breaks three clients at once.
    #[test]
    fn value_json_spelling_is_stable() {
        let cases = [
            (Value::Null, serde_json::json!("Null")),
            (Value::Str("x".into()), serde_json::json!({ "Str": "x" })),
            (Value::Int(42), serde_json::json!({ "Int": 42 })),
            (Value::Bool(true), serde_json::json!({ "Bool": true })),
            (
                Value::List(vec!["a".into()]),
                serde_json::json!({ "List": ["a"] }),
            ),
            (Value::Float(1.5), serde_json::json!({ "Float": 1.5 })),
            (
                Value::DateTime(1_700_000_000_000),
                serde_json::json!({ "DateTime": 1_700_000_000_000i64 }),
            ),
        ];
        for (value, want) in cases {
            let got = serde_json::to_value(&value).unwrap();
            assert_eq!(got, want, "wire spelling changed for {value:?}");
            let back: Value = serde_json::from_value(got).unwrap();
            assert_eq!(back, value);
        }
    }

    /// A whole number sent as `Float` must stay a `Float` — JSON writes 1.0 as `1.0`,
    /// but a client that reads it back as `Int` would break same-type comparison.
    #[test]
    fn a_whole_float_does_not_decode_as_an_int() {
        let json = serde_json::to_string(&Value::Float(1.0)).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&json).unwrap(),
            Value::Float(1.0)
        );
    }

    #[test]
    fn group_predicate_json_spelling_is_stable() {
        let p = Predicate::Not(Box::new(Predicate::Any(vec![Predicate::Contains(
            "tags".into(),
            Value::Str("wip".into()),
        )])));
        assert_eq!(
            serde_json::to_value(&p).unwrap(),
            serde_json::json!({ "Not": { "Any": [{ "Contains": ["tags", { "Str": "wip" }] }] } })
        );
    }
}
