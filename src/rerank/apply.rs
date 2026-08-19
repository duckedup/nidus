//! The async orchestrator every surface (HTTP, CLI, MCP) calls instead of reimplementing
//! the over-fetch/rerank/re-trim window (nidus-4ss). `Nidus`/`Store` stay sync throughout —
//! `src/memory.rs`'s `recall_with` is the precedent for composing an async provider call
//! above a plain sync `db` call, never inside one.

use crate::model::{Hit, HybridOpts, Projection, SearchOpts};
use crate::{FtsQuery, Nidus, Scope};

use super::stage::{self, RerankOpts};
use super::{AnyReranker, Reranker, reranker_identity};

/// What [`finish`] needs once the store has answered: everything derived from the caller's
/// opts that the provider step still has to know.
pub struct Plan {
    query: String,
    text_field: String,
    /// Per-request model override, passed to [`Reranker::rerank`]; `None` uses the
    /// reranker's own configured model.
    model: Option<String>,
    kept: bool,
    offset: usize,
    top_k: usize,
}

/// Plan a vector search: the over-fetched opts for the store, plus the [`Plan`] for
/// [`finish`]. `rr.query` is required here (a raw-vector query has no text of its own), and
/// split from [`finish`] so a caller's lock guard can drop before the provider `.await`.
pub fn plan_search(opts: &SearchOpts, rr: &RerankOpts) -> anyhow::Result<(SearchOpts, Plan)> {
    let query = rr.query.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "rerank.query is required for search(): a raw-vector query has no text of its own"
        )
    })?;
    let text_field = text_field_of(rr);
    let (projection, kept) = widen(&opts.projection, &text_field);
    let deep = SearchOpts {
        top_k: stage::depth(opts.offset, opts.top_k, rr.overscan),
        offset: 0,
        projection,
        ..opts.clone()
    };
    Ok((
        deep,
        Plan {
            query,
            text_field,
            model: rr.model.clone(),
            kept,
            offset: opts.offset,
            top_k: opts.top_k,
        },
    ))
}

/// Plan a full-text search. `rr.query` defaults to the query's own clause text.
pub fn plan_text_search(
    query: &FtsQuery,
    opts: &SearchOpts,
    rr: &RerankOpts,
) -> (SearchOpts, Plan) {
    let query_text = rr.query.clone().unwrap_or_else(|| query_text_of(query));
    let text_field = text_field_of(rr);
    let (projection, kept) = widen(&opts.projection, &text_field);
    let deep = SearchOpts {
        top_k: stage::depth(opts.offset, opts.top_k, rr.overscan),
        offset: 0,
        projection,
        ..opts.clone()
    };
    (
        deep,
        Plan {
            query: query_text,
            text_field,
            model: rr.model.clone(),
            kept,
            offset: opts.offset,
            top_k: opts.top_k,
        },
    )
}

/// Plan a hybrid search. `HybridOpts` carries no `projection` (every fused hit already
/// carries its full attrs — see `Store::hybrid_search`'s `leg_opts`), so there is no
/// widen/re-trim step here, only the depth/offset one.
pub fn plan_hybrid_search(
    text: &FtsQuery,
    opts: &HybridOpts,
    rr: &RerankOpts,
) -> (HybridOpts, Plan) {
    let query_text = rr.query.clone().unwrap_or_else(|| query_text_of(text));
    let deep = HybridOpts {
        top_k: stage::depth(opts.offset, opts.top_k, rr.overscan),
        offset: 0,
        ..opts.clone()
    };
    (
        deep,
        Plan {
            query: query_text,
            text_field: text_field_of(rr),
            model: rr.model.clone(),
            kept: true,
            offset: opts.offset,
            top_k: opts.top_k,
        },
    )
}

/// Search `scope` by vector, then rerank the over-fetched window. For a caller that owns the
/// store outright (the CLI); a lock-holding caller must use [`plan_search`] + [`finish`].
pub async fn search_reranked<'a>(
    db: &Nidus,
    r: &AnyReranker,
    scope: impl Into<Scope<'a>>,
    vector: &[f32],
    opts: &SearchOpts,
    rr: &RerankOpts,
) -> anyhow::Result<Vec<Hit>> {
    let (deep, plan) = plan_search(opts, rr)?;
    let hits = db.search(scope, vector, &deep)?;
    finish(hits, r, &plan).await
}

/// Full-text search `scope`, then rerank the over-fetched window. See [`search_reranked`] on
/// when to use this rather than [`plan_text_search`] + [`finish`].
pub async fn text_search_reranked<'a>(
    db: &Nidus,
    r: &AnyReranker,
    scope: impl Into<Scope<'a>>,
    query: &FtsQuery,
    opts: &SearchOpts,
    rr: &RerankOpts,
) -> anyhow::Result<Vec<Hit>> {
    let (deep, plan) = plan_text_search(query, opts, rr);
    let hits = db.text_search(scope, query, &deep)?;
    finish(hits, r, &plan).await
}

/// Hybrid search `scope`, then rerank the over-fetched window. See [`search_reranked`] on
/// when to use this rather than [`plan_hybrid_search`] + [`finish`].
pub async fn hybrid_search_reranked<'a>(
    db: &Nidus,
    r: &AnyReranker,
    scope: impl Into<Scope<'a>>,
    vector: &[f32],
    text: &FtsQuery,
    opts: &HybridOpts,
    rr: &RerankOpts,
) -> anyhow::Result<Vec<Hit>> {
    let (deep, plan) = plan_hybrid_search(text, opts, rr);
    let hits = db.hybrid_search(scope, vector, text, &deep)?;
    finish(hits, r, &plan).await
}

/// The attr the candidate text is read from, resolving `rr.text_field`'s default.
fn text_field_of(rr: &RerankOpts) -> String {
    rr.text_field
        .clone()
        .unwrap_or_else(|| stage::DEFAULT_TEXT_FIELD.to_string())
}

/// The text scored against the reranker when `rr.query` is unset: every clause's own query
/// text, joined with a space (a cross-encoder wants one string, not a clause list).
fn query_text_of(q: &FtsQuery) -> String {
    q.clauses
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether `p` would already carry `field`.
fn projection_includes(p: &Projection, field: &str) -> bool {
    match p {
        Projection::All => true,
        Projection::Include(keys) => keys.iter().any(|k| k == field),
        Projection::Exclude(keys) => !keys.iter().any(|k| k == field),
    }
}

/// Force-include `field` in `p`. Returns the widened projection and whether the caller's
/// original projection already carried it — closing the hazard where
/// `Store::hits_from_topk` applies the projection before a `Hit` exists (SPEC §9).
fn widen(p: &Projection, field: &str) -> (Projection, bool) {
    if projection_includes(p, field) {
        return (p.clone(), true);
    }
    let widened = match p {
        Projection::All => Projection::All,
        Projection::Include(keys) => {
            let mut keys = keys.clone();
            keys.push(field.to_string());
            Projection::Include(keys)
        }
        Projection::Exclude(keys) => {
            Projection::Exclude(keys.iter().filter(|k| *k != field).cloned().collect())
        }
    };
    (widened, false)
}

/// The shared post-fetch pipeline, identical across all three surfaces: pull candidate
/// texts, call the provider (chunked, skipped entirely if nothing has text), merge scores
/// in, re-trim the projection back to what the caller asked for, then page.
pub async fn finish<R: Reranker>(hits: Vec<Hit>, r: &R, plan: &Plan) -> anyhow::Result<Vec<Hit>> {
    let Plan {
        query,
        text_field,
        model,
        kept,
        offset,
        top_k,
    } = plan;
    let candidates: Vec<(usize, &str)> = hits
        .iter()
        .enumerate()
        .filter_map(|(i, h)| stage::text_of(&h.attrs, text_field).map(|t| (i, t)))
        .collect();
    let passthrough = hits.len() - candidates.len();
    crate::metrics::metrics()
        .rerank_provider_passthrough
        .add(passthrough as u64);

    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
    if !candidates.is_empty() {
        let max_docs = r.max_documents().max(1);
        for chunk in candidates.chunks(max_docs) {
            let docs: Vec<&str> = chunk.iter().map(|(_, t)| *t).collect();
            let scores = r.rerank(query, &docs, model.as_deref()).await?;
            if scores.len() != docs.len() {
                anyhow::bail!(
                    "reranker {} returned {} scores for {} docs",
                    reranker_identity(r),
                    scores.len(),
                    docs.len()
                );
            }
            for (&(idx, _), score) in chunk.iter().zip(scores) {
                scored.push((idx, score));
            }
            crate::metrics::metrics().rerank_provider_calls.inc();
            // Per chunk, not once after the loop: a failure partway through a multi-chunk
            // rerank would otherwise credit the calls but none of the candidates.
            crate::metrics::metrics()
                .rerank_provider_candidates
                .add(docs.len() as u64);
        }
    }

    let mut merged = stage::merge(hits, &scored);
    if !kept {
        for h in &mut merged {
            h.attrs.remove(text_field);
        }
    }
    Ok(stage::page(merged, *offset, *top_k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Value;
    use std::collections::BTreeMap;

    #[test]
    fn widen_include_adds_the_field_once() {
        let p = Projection::include(["a"]);
        let (widened, kept) = widen(&p, "nidus.text");
        assert!(!kept);
        assert_eq!(widened, Projection::include(["a", "nidus.text"]));

        let (still, kept2) = widen(&widened, "nidus.text");
        assert!(kept2);
        assert_eq!(still, widened);
    }

    #[test]
    fn widen_exclude_drops_the_field_from_the_exclusion() {
        let p = Projection::exclude(["nidus.text", "b"]);
        let (widened, kept) = widen(&p, "nidus.text");
        assert!(!kept);
        assert_eq!(widened, Projection::exclude(["b"]));
    }

    #[test]
    fn widen_all_is_a_no_op_and_already_kept() {
        let (widened, kept) = widen(&Projection::All, "nidus.text");
        assert!(kept);
        assert_eq!(widened, Projection::All);
    }

    #[test]
    fn query_text_of_joins_clauses_with_a_space() {
        let q = FtsQuery::multi([
            crate::model::FtsClause::new("title", "cats"),
            crate::model::FtsClause::new("body", "dogs"),
        ]);
        assert_eq!(query_text_of(&q), "cats dogs");
    }

    #[test]
    fn text_field_of_defaults_and_overrides() {
        assert_eq!(
            text_field_of(&RerankOpts::default()),
            stage::DEFAULT_TEXT_FIELD
        );
        let rr = RerankOpts {
            text_field: Some("body".to_string()),
            ..Default::default()
        };
        assert_eq!(text_field_of(&rr), "body");
    }

    fn hit_with_text(id: &str, text: Option<&str>) -> Hit {
        let mut attrs = BTreeMap::new();
        if let Some(t) = text {
            attrs.insert(
                stage::DEFAULT_TEXT_FIELD.to_string(),
                Value::Str(t.to_string()),
            );
        }
        Hit::new("c", id, 0.5, attrs)
    }

    use super::super::RerankError;

    /// Always returns `scores`, asserting the caller sent exactly that many docs.
    struct Fixed(Vec<f32>);
    impl Reranker for Fixed {
        async fn rerank(
            &self,
            _q: &str,
            docs: &[&str],
            _model: Option<&str>,
        ) -> Result<Vec<f32>, RerankError> {
            assert_eq!(docs.len(), self.0.len());
            Ok(self.0.clone())
        }
        fn provider_name(&self) -> &str {
            "test"
        }
        fn model_name(&self) -> &str {
            "fixed"
        }
        fn max_documents(&self) -> usize {
            1000
        }
    }

    /// A `Plan` over the default text field, `kept` as given, unpaginated.
    fn plan(kept: bool) -> Plan {
        Plan {
            query: "q".into(),
            text_field: stage::DEFAULT_TEXT_FIELD.into(),
            model: None,
            kept,
            offset: 0,
            top_k: 10,
        }
    }

    /// Panics if called — proves the empty-docs guard actually short-circuits.
    struct NeverCalled;
    impl Reranker for NeverCalled {
        async fn rerank(
            &self,
            _q: &str,
            _docs: &[&str],
            _model: Option<&str>,
        ) -> Result<Vec<f32>, RerankError> {
            panic!("provider must not be called when no candidate has text");
        }
        fn provider_name(&self) -> &str {
            "never"
        }
        fn model_name(&self) -> &str {
            "never"
        }
        fn max_documents(&self) -> usize {
            1000
        }
    }

    #[tokio::test]
    async fn finish_passes_through_when_no_candidate_has_text() {
        let hits = vec![hit_with_text("a", None), hit_with_text("b", None)];
        let out = finish(hits, &NeverCalled, &plan(true)).await.unwrap();
        assert_eq!(
            out.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[tokio::test]
    async fn finish_reranks_and_reorders() {
        let hits = vec![hit_with_text("a", Some("x")), hit_with_text("b", Some("y"))];
        // "a" has the higher metric score, but the provider ranks "b" first.
        let out = finish(hits, &Fixed(vec![1.0, 9.0]), &plan(true))
            .await
            .unwrap();
        assert_eq!(
            out.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
            vec!["b", "a"]
        );
    }

    #[tokio::test]
    async fn finish_retrims_a_forced_text_field() {
        let hits = vec![hit_with_text("a", Some("x"))];
        let out = finish(hits, &Fixed(vec![1.0]), &plan(false)).await.unwrap();
        assert!(!out[0].attrs.contains_key(stage::DEFAULT_TEXT_FIELD));
    }
}
