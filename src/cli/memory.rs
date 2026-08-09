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

/// Caller attrs first, reserved `nidus.*` stamped after so they win a collision — the merge
/// order the MCP and HTTP handlers use. `Memory::remember` does not stamp `META_TEXT` itself,
/// yet it declares an FTS schema over that field, so an unstamped write is unsearchable.
fn merge_reserved(mut attrs: BTreeMap<String, Value>, text: &str) -> BTreeMap<String, Value> {
    crate::memory::strip_reserved_recency(&mut attrs);
    attrs.insert(
        crate::memory::META_TEXT.to_string(),
        Value::Str(text.to_string()),
    );
    attrs
}

/// `nidus remember`: embed `text` (optionally summarizing first) and upsert it.
#[allow(clippy::too_many_arguments)]
pub(super) fn remember(
    store: StoreArgs,
    ingest: IngestArgs,
    collection: String,
    text: String,
    id: Option<String>,
    attrs: Option<String>,
    #[cfg(feature = "summarize")] summarize: bool,
) -> Result<()> {
    let attrs = merge_reserved(parse_attrs(attrs)?, &text);
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

        memory
            .remember(&collection, &id, &text, attrs, mode)
            .await?;
        // Nothing flushes on drop, so a one-shot process must take the barrier itself or
        // `--fsync on-flush` would discard the write it just reported.
        memory.db_mut().flush()?;

        super::print_json(&serde_json::json!({
            "collection": collection,
            "id": id,
            "mode": format!("{mode:?}"),
            "embedder": identity,
            "dimension": dimension,
        }))
    })
}

/// `nidus recall`: embed `query` and print the ranked hits, in the shape `search` prints.
pub(super) fn recall(
    store: StoreArgs,
    ingest: IngestArgs,
    collection: String,
    query: String,
    top_k: usize,
    min_score: Option<f32>,
    filter: Option<String>,
) -> Result<()> {
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
