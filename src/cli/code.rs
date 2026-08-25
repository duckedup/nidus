//! `nidus code ingest` / `nidus code search` (epic nidus-3gm): front doors over the walk,
//! digest-skip, embed and prune steps `nidus ingest` already owns, and the `search`/
//! `text-search` seams, with per-file chunk strategy resolved through
//! [`crate::code::dispatch`] instead of one strategy for the whole walk.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Context, Result};

use super::ingest::{
    META_SOURCE_CHUNKS, META_SOURCE_HASH, META_SOURCE_PATH, fts_schema_differs, prune_gone,
    source_hash, stored_hash, walk,
};
use super::{IngestArgs, StoreArgs};
use crate::chunk::{ChunkOpts, ChunkStrategy};
use crate::code::present::{FileGroup, group_by_file};
use crate::code::{META_PATH as CODE_META_PATH, META_SYMBOL, chunk_file};
use crate::embed::cache::CachedEmbedder;
use crate::embed::{Embedder, embedder_identity};
use crate::memory::{META_TEXT, RememberWrite, commit_remember_chunks, stamp_recency};
use crate::{
    Filter, FtsField, FtsQuery, META_CHAR_START, META_CHUNK_INDEX, META_PARENT_ID, Nidus,
    Predicate, Record, SearchOpts, Value,
};

/// wdpkr-core's pinned minor version (`Cargo.toml`'s `wdpkr-core = "0.2"`), folded into the
/// re-ingest digest (via the `identity` [`source_hash`] hashes) so a grammar/query bump that
/// moves symbol boundaries re-ingests instead of leaving two chunking regimes in one store.
const WDPKR_CORE_VERSION: &str = "0.2.0";

/// What `code search`'s BM25 fallback is declared over: the chunk body, plus the two fields
/// a caller is most likely to name directly.
const CODE_FTS_FIELDS: [&str; 3] = [META_TEXT, CODE_META_PATH, META_SYMBOL];

/// `code search`'s query knobs, bundled so the entry point stays under clippy's argument cap.
pub(super) struct SearchArgs {
    pub query: String,
    pub collections: Vec<String>,
    pub top_k: usize,
    pub vector: bool,
}

/// `nidus code ingest`: same walk/digest-skip/embed/prune pipeline as `nidus ingest`, but
/// each file's chunks come from [`chunk_file`] (AST-aware for code, markdown-aware for
/// docs) rather than one strategy applied to everything, and dot-entries are always walked.
#[allow(clippy::too_many_arguments)]
pub(super) fn ingest(
    store: StoreArgs,
    ingest: IngestArgs,
    path: PathBuf,
    collection: String,
    max_chars: usize,
    overlap_chars: usize,
    prune: bool,
    dry_run: bool,
    no_cache: bool,
    cache_max_entries: usize,
) -> Result<()> {
    let opts = ChunkOpts {
        strategy: ChunkStrategy::Recursive,
        max_chars,
        overlap_chars,
    };
    // Validates the non-code fallback's options; dispatch decides the strategy per file.
    crate::chunk::chunk_text("probe", &opts).context("invalid chunk options")?;

    // A repo scan that cannot reach `.github`/`.claude` is the bug nidus-0fw fixes; `.git`
    // stays skipped regardless.
    let found = walk(&path, true)?;

    let rt = super::memory::runtime()?;
    rt.block_on(async move {
        let embedder = ingest.build_embedder().await?;
        let base_identity = match &embedder {
            Some(e) => embedder_identity(e),
            None => "fts-only:code".to_string(),
        };
        let identity = format!("{base_identity}+wdpkr-{WDPKR_CORE_VERSION}");
        let dimension = embedder.as_ref().map_or(0, |e| e.dimension());
        let mut db = match &embedder {
            Some(e) => super::memory::open_with(store, e, !dry_run)?,
            None => super::memory::open_fts_only(store, !dry_run)?,
        };

        if embedder.is_none() && !dry_run {
            let resolved = db.resolve_alias(&collection);
            let target = resolved.unwrap_or_else(|| collection.clone());
            let fields: Vec<FtsField> = CODE_FTS_FIELDS.into_iter().map(FtsField::new).collect();
            if fts_schema_differs(db.fts_schema(&target), &fields) {
                db.set_fts_schema(&target, &fields)
                    .with_context(|| format!("declaring the fts schema on '{target}'"))?;
            }
        }

        let mut report = Report {
            matched: found.len(),
            ..Default::default()
        };
        let mut seen: HashSet<String> = HashSet::new();
        let cache_slot = db.persistence();
        let cached = embedder.as_ref().map(|e| {
            CachedEmbedder::open(
                e,
                if no_cache { None } else { cache_slot },
                &identity,
                dimension,
                if no_cache { 0 } else { cache_max_entries },
            )
        });

        for file in &found {
            seen.insert(file.rel.clone());
            let text = match std::fs::read_to_string(&file.abs) {
                Ok(t) => t,
                Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                    report.skipped_non_utf8 += 1;
                    continue;
                }
                Err(e) => {
                    return Err(
                        anyhow::Error::new(e).context(format!("reading {}", file.abs.display()))
                    );
                }
            };

            let hash = source_hash(&text, &opts, &identity, dimension);
            let chunks = chunk_file(&file.rel, &text, &opts)
                .with_context(|| format!("chunking {}", file.rel))?;
            let n = chunks.len();
            if n == 0 {
                report.skipped_empty += 1;
                continue;
            }
            if stored_hash(&db, &collection, &file.rel).as_deref() == Some(hash.as_str())
                && db
                    .get(&collection, &format!("{}#{}", file.rel, n - 1))
                    .is_some()
            {
                report.unchanged += 1;
                continue;
            }
            if dry_run {
                report.would_ingest += 1;
                continue;
            }

            let written = match &cached {
                Some(cached) => {
                    ingest_chunks_embedded(
                        &mut db,
                        cached,
                        &collection,
                        &file.rel,
                        chunks,
                        &hash,
                        n,
                    )
                    .await?
                }
                None => ingest_chunks_text_only(&mut db, &collection, &file.rel, chunks, &hash, n)?,
            };
            report.ingested += 1;
            report.chunks += written;
        }

        if prune && !dry_run {
            report.pruned = prune_gone(&mut db, &collection, &seen)?;
        }
        if !dry_run {
            if let Some(cached) = &cached {
                cached.save().context("saving the embedding cache")?;
            }
            db.flush()?;
        }

        let stats = cached.as_ref().map(|c| c.stats()).unwrap_or_default();
        super::print_json(&serde_json::json!({
            "collection": collection,
            "root": path.display().to_string(),
            "matched": report.matched,
            "ingested": report.ingested,
            "unchanged": report.unchanged,
            "skipped_non_utf8": report.skipped_non_utf8,
            "skipped_empty": report.skipped_empty,
            "chunks": report.chunks,
            "pruned": report.pruned,
            "would_ingest": report.would_ingest,
            "dry_run": dry_run,
            "embedder": identity,
            "dimension": dimension,
            "cache": { "hits": stats.hits, "misses": stats.misses, "evicted": stats.evicted },
        }))
    })
}

/// What one `code ingest` run did, printed as JSON.
#[derive(Default)]
struct Report {
    matched: usize,
    ingested: usize,
    unchanged: usize,
    skipped_non_utf8: usize,
    skipped_empty: usize,
    chunks: usize,
    pruned: usize,
    would_ingest: usize,
}

/// One file's chunks, embedded and committed as one group under a single barrier (mirrors
/// [`crate::memory::remember_chunked_with`]), each keeping its own per-symbol attrs rather
/// than the one shared `attrs` map a plain `remember` write uses.
async fn ingest_chunks_embedded<E: Embedder>(
    db: &mut Nidus,
    embedder: &E,
    collection: &str,
    rel: &str,
    chunks: Vec<crate::code::CodeChunk>,
    hash: &str,
    n: usize,
) -> Result<usize> {
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let vectors = embedder
        .embed_batch(&texts)
        .await
        .with_context(|| format!("embedding {} chunks for '{collection}/{rel}'", texts.len()))?;
    if vectors.len() != chunks.len() {
        anyhow::bail!(
            "code ingest: embedder returned {} vectors for {} chunks of '{collection}/{rel}'",
            vectors.len(),
            chunks.len()
        );
    }

    let writes: Vec<(RememberWrite, Vec<f32>, i64, i64)> = chunks
        .into_iter()
        .zip(vectors)
        .enumerate()
        .map(|(i, (chunk, vector))| {
            let mut attrs = chunk.attrs;
            if i == 0 {
                attrs.insert(META_SOURCE_HASH.to_string(), Value::Str(hash.to_string()));
                attrs.insert(META_SOURCE_PATH.to_string(), Value::Str(rel.to_string()));
                attrs.insert(META_SOURCE_CHUNKS.to_string(), Value::Int(n as i64));
            }
            let write = RememberWrite {
                id: format!("{rel}#{i}"),
                text: chunk.text,
                attrs,
                ttl_seconds: None,
                dedupe_threshold: None,
            };
            (write, vector, i as i64, chunk.char_start as i64)
        })
        .collect();

    let n_i64 = n as i64;
    db.deferred(|db| {
        let remembered = commit_remember_chunks(db, embedder, collection, rel, writes)?;
        db.delete_where(
            collection,
            &Filter(vec![Predicate::All(vec![
                Predicate::Eq(META_PARENT_ID.to_string(), Value::Str(rel.to_string())),
                Predicate::Ge(META_CHUNK_INDEX.to_string(), Value::Int(n_i64)),
            ])]),
        )?;
        db.commit()?;
        Ok(remembered.len())
    })
}

/// The no-provider twin of [`ingest_chunks_embedded`]: every chunk is a text-only record
/// under one group-commit barrier, same as [`crate::memory::remember_chunked_text_only`],
/// but built directly since that helper always re-chunks with one strategy.
fn ingest_chunks_text_only(
    db: &mut Nidus,
    collection: &str,
    rel: &str,
    chunks: Vec<crate::code::CodeChunk>,
    hash: &str,
    n: usize,
) -> Result<usize> {
    let now = crate::meta::now_ms();
    let resolved = db.resolve_alias(collection);
    let target = resolved.unwrap_or_else(|| collection.to_string());

    let records: Vec<Record> = chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| {
            let mut attrs = chunk.attrs;
            attrs.insert(META_TEXT.to_string(), Value::Str(chunk.text));
            attrs.insert(META_PARENT_ID.to_string(), Value::Str(rel.to_string()));
            attrs.insert(META_CHUNK_INDEX.to_string(), Value::Int(i as i64));
            attrs.insert(
                META_CHAR_START.to_string(),
                Value::Int(chunk.char_start as i64),
            );
            if i == 0 {
                attrs.insert(META_SOURCE_HASH.to_string(), Value::Str(hash.to_string()));
                attrs.insert(META_SOURCE_PATH.to_string(), Value::Str(rel.to_string()));
                attrs.insert(META_SOURCE_CHUNKS.to_string(), Value::Int(n as i64));
            }
            stamp_recency(&mut attrs, now, None, None);
            Record::text_only(format!("{rel}#{i}"), attrs)
        })
        .collect();

    let count = records.len();
    let n_i64 = n as i64;
    db.deferred(|db| {
        if !db.has_collection(&target) {
            db.create_collection(&target)?;
        }
        db.upsert(&target, &records)?;
        db.delete_where(
            &target,
            &Filter(vec![Predicate::All(vec![
                Predicate::Eq(META_PARENT_ID.to_string(), Value::Str(rel.to_string())),
                Predicate::Ge(META_CHUNK_INDEX.to_string(), Value::Int(n_i64)),
            ])]),
        )?;
        db.commit()
    })?;
    Ok(count)
}

/// `nidus code search`: vector search when an embedder is configured, BM25 otherwise. On a
/// dim-0 (`--fts-only`-ingested) store, an unrequested vector attempt falls back to BM25
/// instead of surfacing the refusal; `--vector` surfaces it unchanged.
pub(super) fn search(store: StoreArgs, ingest: IngestArgs, args: SearchArgs) -> Result<()> {
    let SearchArgs {
        query,
        collections,
        top_k,
        vector,
    } = args;
    let refs: Vec<&str> = collections.iter().map(String::as_str).collect();

    let rt = super::memory::runtime()?;
    rt.block_on(async move {
        let embedder = ingest.build_embedder().await?;
        let db = match &embedder {
            Some(e) => super::memory::open_with(store, e, false)?,
            None => super::open(&store, false)?,
        };

        let hits = match (&embedder, vector) {
            (Some(e), _) => {
                let q = e.embed_query(&query).await.context("embedding the query")?;
                let opts = SearchOpts {
                    top_k,
                    ..Default::default()
                };
                match db.search(scope(&refs), &q, &opts) {
                    Ok(hits) => hits,
                    Err(err) if !vector && err.to_string().contains("dimension 0") => {
                        text_search(&db, &refs, &query, top_k)?
                    }
                    Err(err) => return Err(err),
                }
            }
            (None, true) => anyhow::bail!(
                "--vector needs an embedder: pass --embed-provider (voyage, openai, ollama, \
                 cohere, gemini, mistral, jina, openai-compat), or set NIDUS_EMBED_PROVIDER"
            ),
            (None, false) => text_search(&db, &refs, &query, top_k)?,
        };

        super::print_json(&groups_to_json(&group_by_file(&hits)))
    })
}

fn scope<'a>(refs: &'a [&'a str]) -> crate::Scope<'a> {
    if refs.is_empty() {
        crate::Scope::All
    } else {
        crate::Scope::Collections(refs)
    }
}

fn text_search(db: &Nidus, refs: &[&str], query: &str, top_k: usize) -> Result<Vec<crate::Hit>> {
    let q = FtsQuery::new(META_TEXT, query.to_string());
    let opts = SearchOpts {
        top_k,
        ..Default::default()
    };
    db.text_search(scope(refs), &q, &opts)
}

/// [`FileGroup`]/[`crate::code::present::SymbolHit`] carry no `Serialize`, so this builds the
/// same shape by hand rather than adding a derive to `src/code/` from outside its own unit.
fn groups_to_json(groups: &[FileGroup]) -> serde_json::Value {
    serde_json::Value::Array(
        groups
            .iter()
            .map(|g| {
                let symbols: Vec<serde_json::Value> = g
                    .symbols
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "symbol": s.symbol,
                            "kind": s.kind,
                            "start_line": s.start_line,
                            "end_line": s.end_line,
                            "score": s.score,
                        })
                    })
                    .collect();
                serde_json::json!({ "path": g.path, "symbols": symbols })
            })
            .collect(),
    )
}
