//! `nidus ingest <PATH>` (nidus-lvo.2): walk a tree, chunk each file, embed the chunks and
//! upsert them. Idempotent by construction, which is the whole point — every caller writes
//! this script themselves today and writes it slightly wrong, re-embedding the whole corpus
//! on every run.
//!
//! Two independent skips do that work. A **file-level hash** stamped on the chunk records
//! makes an unchanged file cost nothing: no chunking, no embed call, no write. The
//! **chunk-level cache** underneath ([`CachedEmbedder`], nidus-lvo.3) then makes a file that
//! *did* change cost only its changed chunks.

use std::collections::{BTreeMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::{ChunkStrategyArg, IngestArgs, StoreArgs};
use crate::chunk::ChunkOpts;
use crate::embed::cache::CachedEmbedder;
use crate::embed::{Embedder, embedder_identity};
use crate::{Filter, META_PARENT_ID, Nidus, Predicate, RememberMode, RememberOpts, Value};

/// Attr holding the digest that makes a re-ingest a no-op. Doubles as the marker that a
/// record came from `ingest`, which is what keeps `--prune` off hand-written memories.
pub(super) const META_SOURCE_HASH: &str = "nidus.source_hash";
/// Attr holding the source path relative to the ingest root, for callers filtering by file.
pub(super) const META_SOURCE_PATH: &str = "nidus.source_path";
/// Attr holding how many chunks the file was split into. The skip check needs it: chunk 0
/// carries the digest but is written *first*, so a write torn partway (nidus-lvo.5) would
/// otherwise leave a half-ingested file looking complete for good.
pub(super) const META_SOURCE_CHUNKS: &str = "nidus.source_chunks";

/// One file the walk turned up: its forward-slashed path relative to the root, and where to
/// read it from.
pub(super) struct Found {
    pub(super) rel: String,
    pub(super) abs: PathBuf,
}

/// Recursive `read_dir`, sorted so the order is reproducible. Symlinks are never followed so
/// a cycle cannot hang the walk. `.git` is always skipped at any depth; every other
/// dot-entry is skipped unless `include_hidden` is set (nidus-0fw).
pub(super) fn walk(root: &Path, include_hidden: bool) -> Result<Vec<Found>> {
    let mut out = Vec::new();
    visit(root, root, include_hidden, &mut out)?;
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok(out)
}

fn visit(root: &Path, dir: &Path, include_hidden: bool, out: &mut Vec<Found>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("listing {}", dir.display()))?;
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str == ".git" {
            continue;
        }
        if !include_hidden && name_str.starts_with('.') {
            continue;
        }
        let kind = entry
            .file_type()
            .with_context(|| format!("stat {}", entry.path().display()))?;
        if kind.is_symlink() {
            continue;
        }
        let path = entry.path();
        if kind.is_dir() {
            visit(root, &path, include_hidden, out)?;
        } else if kind.is_file() {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            out.push(Found { rel, abs: path });
        }
    }
    Ok(())
}

/// nidus's `*` crosses `/` (SQL GLOB, SPEC §7.1), so `*.md` is already recursive. The
/// ticket's own example is `**/*.md`, which under those semantics would miss root-level
/// files, so a leading `**/` is treated as optional.
pub(super) fn matches(glob: &str, rel: &str) -> bool {
    if crate::glob::glob_match(glob, rel) {
        return true;
    }
    match glob.strip_prefix("**/") {
        Some(rest) => crate::glob::glob_match(rest, rel),
        None => false,
    }
}

/// The digest a re-ingest compares. Covers the chunk options and the embedder as well as the
/// text, so a `--max-chars` change or a model swap re-ingests instead of leaving vectors from
/// one regime beside another. `DefaultHasher` is fixed-key, so it survives restarts.
pub(super) fn source_hash(
    text: &str,
    opts: &ChunkOpts,
    identity: &str,
    dimension: usize,
) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    format!("{:?}", opts.strategy).hash(&mut h);
    opts.max_chars.hash(&mut h);
    opts.overlap_chars.hash(&mut h);
    identity.hash(&mut h);
    dimension.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Whether `declared` needs replacing by `wanted`. Compared by field name only: the tuning
/// knobs come from `FtsField::new`'s defaults on this path, so a name-set match means the
/// declaration would be a no-op, and `set_fts_schema` reindexes every live doc.
pub(super) fn fts_schema_differs(
    declared: Option<&[crate::FtsField]>,
    wanted: &[crate::FtsField],
) -> bool {
    let Some(declared) = declared else {
        return true;
    };
    let names = |f: &[crate::FtsField]| {
        let mut v: Vec<&str> = f.iter().map(|x| x.field.as_str()).collect();
        v.sort_unstable();
        v.dedup();
        v.into_iter().map(str::to_owned).collect::<Vec<_>>()
    };
    names(declared) != names(wanted)
}

/// The embedder-identity stand-in an `--fts-only` run folds into [`source_hash`]. It names the
/// declared field set, sorted, so changing which attrs are full-text indexed re-ingests instead
/// of leaving chunks indexed under the old schema.
fn fts_identity(fields: &[String]) -> String {
    let mut sorted: Vec<&str> = fields.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    sorted.dedup();
    format!("fts-only:{}", sorted.join(","))
}

/// The `nidus.source_hash` already stored for this file, if any. Chunk 0 carries it, so this
/// is one point lookup per file rather than a scan.
pub(super) fn stored_hash(db: &Nidus, collection: &str, rel: &str) -> Option<String> {
    let record = db.get(collection, &format!("{rel}#0"))?;
    match record.attrs.get(META_SOURCE_HASH) {
        Some(Value::Str(s)) => Some(s.clone()),
        _ => None,
    }
}

/// What the run did, printed as JSON so a CI job has something to assert on.
#[derive(Default)]
struct Report {
    matched: usize,
    ingested: usize,
    unchanged: usize,
    skipped_non_utf8: usize,
    skipped_empty: usize,
    chunks: usize,
    stale_tail_pruned: usize,
    pruned: usize,
    would_ingest: usize,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    store: StoreArgs,
    ingest: IngestArgs,
    path: PathBuf,
    collection: String,
    glob: String,
    strategy: ChunkStrategyArg,
    max_chars: usize,
    overlap_chars: usize,
    prune: bool,
    dry_run: bool,
    no_cache: bool,
    cache_max_entries: usize,
    fts_only: Vec<String>,
) -> Result<()> {
    let opts = ChunkOpts {
        strategy: strategy.into(),
        max_chars,
        overlap_chars,
    };
    // Validate before the first billed call rather than letting chunk_text bail mid-run.
    crate::chunk::chunk_text("probe", &opts).context("invalid chunk options")?;

    let found = walk(&path, ingest.include_hidden)?;
    let matched: Vec<Found> = found
        .into_iter()
        .filter(|f| matches(&glob, &f.rel))
        .collect();

    let fts_fields = fts_only;
    let rt = super::memory::runtime()?;
    rt.block_on(async move {
        // `--fts-only` is the whole no-provider path: no embedder is built, so no API key and
        // no network call, and the identity below stands in for one in the re-ingest digest.
        let embedder = match fts_fields.is_empty() {
            true => Some(super::memory::require_embedder(&ingest).await?),
            false => None,
        };
        let identity = match &embedder {
            Some(e) => embedder_identity(e),
            None => fts_identity(&fts_fields),
        };
        let dimension = embedder.as_ref().map_or(0, |e| e.dimension());
        let mut db = match &embedder {
            Some(e) => super::memory::open_with(store, e, !dry_run)?,
            None => super::memory::open_fts_only(store, !dry_run)?,
        };
        // Declared before the first write so the chunks land already indexed. Resolved
        // through an alias first, because `set_fts_schema` refuses one outright (nidus-klh),
        // and re-declared only when the field set actually changed (a no-op reindexes).
        if !fts_fields.is_empty() && !dry_run {
            let resolved = db.resolve_alias(&collection);
            let target = resolved.unwrap_or_else(|| collection.clone());
            let fields: Vec<crate::FtsField> =
                fts_fields.iter().map(crate::FtsField::new).collect();
            if fts_schema_differs(db.fts_schema(&target), &fields) {
                db.set_fts_schema(&target, &fields)
                    .with_context(|| format!("declaring the fts schema on '{target}'"))?;
            }
        }

        let mut report = Report {
            matched: matched.len(),
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

        for file in &matched {
            seen.insert(file.rel.clone());
            let text = match std::fs::read_to_string(&file.abs) {
                Ok(t) => t,
                Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                    // A binary file in the tree must not abort the ingest.
                    report.skipped_non_utf8 += 1;
                    crate::diag::diag!(
                        crate::diag::Level::Warn,
                        "ingest",
                        "skipping a file that is not utf-8",
                        "path" => file.rel,
                    );
                    continue;
                }
                Err(e) => {
                    return Err(
                        anyhow::Error::new(e).context(format!("reading {}", file.abs.display()))
                    );
                }
            };
            let hash = source_hash(&text, &opts, &identity, dimension);
            // Chunked here only to learn the count the completeness check below needs.
            // Pure in-memory string work, so paying it twice per changed file is nothing
            // next to one embedding call.
            let n = crate::chunk::chunk_text(&text, &opts)
                .with_context(|| format!("chunking {}", file.rel))?
                .len();
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

            let mut attrs = BTreeMap::new();
            attrs.insert(META_SOURCE_HASH.to_string(), Value::Str(hash));
            attrs.insert(META_SOURCE_PATH.to_string(), Value::Str(file.rel.clone()));
            attrs.insert(META_SOURCE_CHUNKS.to_string(), Value::Int(n as i64));
            let remember_opts = RememberOpts {
                mode: RememberMode::Raw,
                attrs,
                ttl_seconds: None,
                dedupe_threshold: None,
            };
            let written = match &cached {
                Some(cached) => crate::memory::remember_chunked_with(
                    &mut db,
                    cached,
                    &collection,
                    &file.rel,
                    &text,
                    &opts,
                    remember_opts,
                )
                .await
                .with_context(|| format!("ingesting {}", file.rel))?,
                None => crate::memory::remember_chunked_text_only(
                    &mut db,
                    &collection,
                    &file.rel,
                    &text,
                    &opts,
                    remember_opts,
                )
                .with_context(|| format!("ingesting {}", file.rel))?,
            };
            report.ingested += 1;
            report.chunks += written.chunks.len();
            report.stale_tail_pruned += written.pruned;
            crate::diag::diag!(
                crate::diag::Level::Debug,
                "ingest",
                "ingested a file",
                "path" => file.rel,
                "chunks" => written.chunks.len(),
            );
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
            "glob": glob,
            "strategy": format!("{:?}", opts.strategy),
            "matched": report.matched,
            "ingested": report.ingested,
            "unchanged": report.unchanged,
            "skipped_non_utf8": report.skipped_non_utf8,
            "skipped_empty": report.skipped_empty,
            "chunks": report.chunks,
            "stale_tail_pruned": report.stale_tail_pruned,
            "pruned": report.pruned,
            "would_ingest": report.would_ingest,
            "dry_run": dry_run,
            "embedder": identity,
            "dimension": dimension,
            "fts_only": fts_fields,
            "cache": { "hits": stats.hits, "misses": stats.misses, "evicted": stats.evicted },
        }))
    })
}

/// Delete records whose source file is gone. Only records carrying `nidus.source_hash` are
/// considered, so a collection also holding hand-written `nidus remember` facts keeps them.
pub(super) fn prune_gone(
    db: &mut Nidus,
    collection: &str,
    seen: &HashSet<String>,
) -> Result<usize> {
    let mut gone: HashSet<String> = HashSet::new();
    for record in db.get_all(collection) {
        if !record.attrs.contains_key(META_SOURCE_HASH) {
            continue;
        }
        if let Some(Value::Str(parent)) = record.attrs.get(META_PARENT_ID)
            && !seen.contains(parent)
        {
            gone.insert(parent.clone());
        }
    }
    let mut removed = 0;
    for parent in gone {
        removed += db.delete_where(
            collection,
            &Filter(vec![Predicate::Eq(
                META_PARENT_ID.to_string(),
                Value::Str(parent),
            )]),
        )?;
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkStrategy;

    fn opts() -> ChunkOpts {
        ChunkOpts {
            strategy: ChunkStrategy::Recursive,
            max_chars: 1000,
            overlap_chars: 100,
        }
    }

    /// The two silent-staleness cases the hash exists to catch: re-chunking under different
    /// options, and a model swap. Either would otherwise leave vectors from one regime
    /// sitting beside another in the same collection.
    #[test]
    fn source_hash_covers_the_chunk_options_and_the_embedder() {
        let text = "the same text throughout";
        let base = source_hash(text, &opts(), "voyage/voyage-3", 1024);

        let mut wider = opts();
        wider.max_chars = 2000;
        assert_ne!(base, source_hash(text, &wider, "voyage/voyage-3", 1024));

        let mut overlapped = opts();
        overlapped.overlap_chars = 200;
        assert_ne!(
            base,
            source_hash(text, &overlapped, "voyage/voyage-3", 1024)
        );

        let mut strategy = opts();
        strategy.strategy = ChunkStrategy::Markdown;
        assert_ne!(base, source_hash(text, &strategy, "voyage/voyage-3", 1024));

        assert_ne!(
            base,
            source_hash(text, &opts(), "openai/text-embedding-3", 1024)
        );
        assert_ne!(base, source_hash(text, &opts(), "voyage/voyage-3", 512));
        assert_ne!(
            base,
            source_hash("different text", &opts(), "voyage/voyage-3", 1024)
        );

        assert_eq!(
            base,
            source_hash(text, &opts(), "voyage/voyage-3", 1024),
            "stable"
        );
    }

    /// nidus's `*` crosses `/`, so `*.md` is recursive already. The ticket's example is
    /// `**/*.md`, which without the leading-`**/` allowance would miss root-level files.
    #[test]
    fn a_leading_double_star_still_matches_at_the_root() {
        assert!(matches("**/*.md", "a.md"), "root file");
        assert!(matches("**/*.md", "deep/nested/a.md"), "nested file");
        assert!(matches("*.md", "a.md"));
        assert!(matches("*.md", "deep/a.md"), "* crosses /");
        assert!(!matches("**/*.md", "a.txt"));
        assert!(!matches("*.txt", "deep/a.md"));
    }

    #[test]
    fn the_default_glob_matches_everything() {
        assert!(matches("*", "a.md"));
        assert!(matches("*", "deep/nested/thing.rs"));
    }

    /// A tree with a dot-directory that has its own nested dot-directory, so the flag-on
    /// half can assert `.claude/rules/x.md` is reached while `.git` stays skipped.
    fn dotted_tree(root: &Path) {
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/HEAD"), "ref: refs/heads/main").unwrap();
        std::fs::create_dir_all(root.join(".claude/rules")).unwrap();
        std::fs::write(root.join(".claude/rules/x.md"), "rule").unwrap();
        std::fs::write(root.join(".hidden"), "x").unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/b.md"), "b").unwrap();
        std::fs::write(root.join("a.md"), "a").unwrap();
        std::fs::write(root.join("c.md"), "c").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(root, root.join("loop")).unwrap();
    }

    /// Without `--include-hidden`, `nidus ingest .` must not walk `.git` or any other
    /// dot-entry, and a symlink cycle must not hang the walk. Unchanged since before nidus-0fw.
    #[test]
    fn walk_skips_dot_entries_and_symlinks_and_sorts_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        dotted_tree(root);

        let found = walk(root, false).unwrap();
        let rels: Vec<&str> = found.iter().map(|f| f.rel.as_str()).collect();
        assert_eq!(
            rels,
            vec!["a.md", "c.md", "sub/b.md"],
            "sorted, no dots, no symlink"
        );
    }

    /// With `--include-hidden`, the walk descends into dot-directories (`.claude/rules/x.md`
    /// is reached) but `.git` stays skipped at any depth — the case that fails if the flag
    /// were implemented as "drop the dot-check" instead of naming `.git` specifically.
    #[test]
    fn walk_with_include_hidden_reaches_dot_directories_but_never_git() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        dotted_tree(root);

        let found = walk(root, true).unwrap();
        let rels: Vec<&str> = found.iter().map(|f| f.rel.as_str()).collect();
        assert!(
            rels.contains(&".claude/rules/x.md"),
            "a dot-directory must be walked: {rels:?}"
        );
        assert!(
            rels.contains(&".hidden"),
            "a dot-file must be walked: {rels:?}"
        );
        assert!(
            !rels.iter().any(|r| r.starts_with(".git")),
            ".git must stay skipped even with the flag on: {rels:?}"
        );
        assert!(!rels.contains(&"loop"), "the symlink is still skipped");
    }

    /// A path with no matches is a report of zero, not an error.
    #[test]
    fn walk_of_an_empty_tree_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(walk(dir.path(), false).unwrap().is_empty());
    }
}
