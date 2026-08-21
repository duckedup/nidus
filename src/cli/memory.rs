//! `nidus remember` / `nidus recall`: the text-native memory surface as one-shot
//! subcommands, so a shell hook can write or query a fact without standing up a
//! long-lived `nidus serve` (#134).

use std::collections::BTreeMap;

use anyhow::{Context, Result};

use super::{IngestArgs, StoreArgs};
use crate::embed::{AnyEmbedder, Embedder, embedder_identity};
use crate::server::dto::HitDto;
use crate::{Filter, Memory, Nidus, RecallOpts, RememberMode, Value};

/// A stable id for a memory whose caller supplied none, mirroring the MCP surface's
/// derivation so the same text lands on the same id from either entry point.
/// `DefaultHasher` is fixed-key, so the id survives restarts.
fn content_id(text: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    format!("mem-{:016x}", h.finish())
}

/// A current-thread runtime: these subcommands make one or two HTTP calls and exit, so
/// they have nothing to gain from `serve`'s multi-threaded pool.
fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building the async runtime for the embedder")
}

/// Build the embedder these subcommands cannot run without, failing with the flag to
/// reach for when none was configured.
async fn require_embedder(ingest: &IngestArgs) -> Result<AnyEmbedder> {
    ingest.build_embedder().await?.context(
        "no embedder configured: pass --embed-provider (voyage, openai, ollama, cohere, \
         gemini, mistral, jina, openai-compat), or set NIDUS_EMBED_PROVIDER",
    )
}

/// Open the store for a memory subcommand, defaulting the dimension to the embedder's own
/// so a first `remember` into a fresh directory needs no `--dim`. An explicit `--dim` still
/// wins, and either way a store whose header disagrees is refused on open.
fn open_with(mut store: StoreArgs, embedder: &AnyEmbedder, mutating: bool) -> Result<Nidus> {
    if store.dim.is_none() {
        store.dim = Some(embedder.dimension());
    }
    super::open(&store, mutating)
}

/// Parse `--attrs` into the typed attr map, defaulting to empty.
fn parse_attrs(attrs: Option<String>) -> Result<BTreeMap<String, Value>> {
    match attrs {
        Some(s) => serde_json::from_str(&s)
            .with_context(|| format!("--attrs must be a JSON object of typed values, got {s}")),
        None => Ok(BTreeMap::new()),
    }
}

/// `nidus remember`: embed `text` (optionally summarizing first) and upsert it. The
/// reserved `nidus.*` stamping (text, recency, expiry) all happens inside
/// [`Memory::remember`] since #133, so nothing is merged here.
#[allow(clippy::too_many_arguments)]
pub(super) fn remember(
    store: StoreArgs,
    ingest: IngestArgs,
    collection: String,
    text: String,
    id: Option<String>,
    attrs: Option<String>,
    ttl_seconds: Option<i64>,
    dedupe_threshold: Option<f32>,
    #[cfg(feature = "summarize")] summarize: bool,
) -> Result<()> {
    let attrs = parse_attrs(attrs)?;
    let id = id.unwrap_or_else(|| content_id(&text));
    let rt = runtime()?;

    rt.block_on(async move {
        let embedder = require_embedder(&ingest).await?;
        let identity = embedder_identity(&embedder);
        let dimension = embedder.dimension();
        let db = open_with(store, &embedder, true)?;

        #[cfg(feature = "summarize")]
        let mode = if summarize {
            RememberMode::Summarize
        } else {
            RememberMode::Raw
        };
        #[cfg(not(feature = "summarize"))]
        let mode = RememberMode::Raw;

        let mut memory = Memory::new(db, embedder);
        #[cfg(feature = "summarize")]
        if summarize {
            let summarizer = ingest.build_summarizer().await?.context(
                "--summarize needs a summarizer: pass --summarize-provider (anthropic or \
                 openai), or set NIDUS_SUMMARIZE_PROVIDER",
            )?;
            memory = memory.with_summarizer(summarizer);
        }

        let mode_label = format!("{mode:?}");
        let written = memory
            .remember(
                &collection,
                &id,
                &text,
                crate::RememberOpts {
                    mode,
                    attrs,
                    ttl_seconds,
                    dedupe_threshold,
                },
            )
            .await?;
        // Nothing flushes on drop, so a one-shot process must take the barrier itself or
        // `--fsync on-flush` would discard the write it just reported.
        memory.db_mut().flush()?;

        super::print_json(&serde_json::json!({
            "collection": collection,
            "id": written.id,
            "deduped": written.deduped,
            "upserted": written.upserted,
            "mode": mode_label,
            "embedder": identity,
            "dimension": dimension,
        }))
    })
}

/// The query knobs of `nidus recall`, bundled so the entry point stays under clippy's
/// argument bound.
pub(super) struct RecallArgs {
    pub collection: String,
    pub query: String,
    pub top_k: usize,
    pub min_score: Option<f32>,
    pub filter: Option<String>,
    pub diversity: Option<f32>,
}

/// `nidus recall`: embed `query` and print the ranked hits, in the shape `search` prints.
pub(super) fn recall(store: StoreArgs, ingest: IngestArgs, args: RecallArgs) -> Result<()> {
    let RecallArgs {
        collection,
        query,
        top_k,
        min_score,
        filter,
        diversity,
    } = args;
    let filter: Option<Filter> = match filter {
        Some(s) => Some(
            serde_json::from_str(&s)
                .with_context(|| format!("--where must be a JSON filter, got {s}"))?,
        ),
        None => None,
    };
    let rt = runtime()?;

    rt.block_on(async move {
        let embedder = require_embedder(&ingest).await?;
        // Read-only: a recall must run alongside a `nidus serve` holding the writer lock.
        let db = open_with(store, &embedder, false)?;
        let memory = Memory::new(db, embedder);

        let opts = RecallOpts {
            top_k,
            min_score: min_score.unwrap_or(0.0),
            filter,
            diversity,
        };
        let hits = memory.recall(&collection, &query, &opts).await?;
        let out: Vec<HitDto> = hits.into_iter().map(HitDto::from).collect();
        super::print_json(&out)
    })
}

/// Guard against the two ids drifting apart: a memory written over MCP and the same text
/// written from the CLI must collide on one record, not accumulate two.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_id_is_stable_and_text_addressed() {
        assert_eq!(content_id("a fact"), content_id("a fact"));
        assert_ne!(content_id("a fact"), content_id("another fact"));
        let id = content_id("a fact");
        assert!(id.starts_with("mem-"), "{id}");
        assert_eq!(id.len(), "mem-".len() + 16, "{id}");
    }

    #[test]
    fn attrs_parse_from_the_tagged_value_form() {
        let attrs = parse_attrs(Some(r#"{"tag": {"Str": "x"}, "n": {"Int": 3}}"#.into())).unwrap();
        assert_eq!(attrs.get("tag"), Some(&Value::Str("x".into())));
        assert_eq!(attrs.get("n"), Some(&Value::Int(3)));
        assert!(parse_attrs(None).unwrap().is_empty());
        assert!(parse_attrs(Some("not json".into())).is_err());
    }
}
