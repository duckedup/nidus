//! `nidus backup` / `nidus restore`: snapshot a store into a single pure-Rust
//! `.tar.gz` **object**, and extract one back. The archive's destination/source is a
//! [`Persistence`](crate::backend::Persistence) location, so a snapshot is just one
//! named object on any backend — a local file (`./snap.tar.gz`, `file:///b/snap.tar.gz`)
//! today, an object store (`s3://…`) once one lands (SPEC §13.7). It is *exactly*
//! object-granular, which is why every backend does it trivially.
//!
//! A store is a directory of named objects; `data` (the flat `f32` matrix) and `log`
//! (the commit record) are the durable source of truth. `lock` is per-process writer
//! exclusion and the derived caches (`ann`/`fts`) are rebuildable — none belong in a
//! backup. We read `data` then `log` from the source backend.
//!
//! **Why a hot backup is consistent (no writer lock needed).** The writer fsyncs
//! `data` *before* appending committing records to `log`, and every reader ignores log
//! records that reference a row beyond the data object's size (`row ≥ data_len / dim`).
//! So capturing `data` first and `log` second sees exactly what a lock-free reader
//! would: a log record for a row not yet in the captured `data` is simply ignored on
//! restore-open — possibly a hair stale, never torn.

use std::io::Read;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde::Serialize;

use crate::backend::{Persistence, open_object_location};
use crate::{Config, Nidus, OpenMode};

/// Source-of-truth objects that make up the durable, portable state of a store.
const ARCHIVED: [&str; 2] = ["data", "log"];
/// Embedded manifest entry name (informational; restore tolerates its absence).
const MANIFEST: &str = "nidus-backup.json";

/// What a backup recorded, printed as JSON by the CLI.
#[derive(Debug, Serialize)]
pub struct BackupReport {
    pub backup: String,
    pub source: String,
    pub dimension: usize,
    pub distance: String,
    pub data_bytes: u64,
    pub log_bytes: u64,
    pub archive_bytes: u64,
}

/// What a restore produced, printed as JSON by the CLI.
#[derive(Debug, Serialize)]
pub struct RestoreReport {
    pub restored_to: String,
    pub source_archive: String,
    pub dimension: usize,
    pub distance: String,
    pub collections: Vec<String>,
    pub records: usize,
}

/// The small JSON manifest embedded in each archive. Purely informational — it
/// lets a human (or a future tool) see who/what/when made the backup without
/// unpacking the binary `data`/`log`. Restore does not require it.
#[derive(Serialize)]
struct Manifest {
    nidus_version: &'static str,
    created_unix: u64,
    dimension: usize,
    distance: String,
    data_bytes: u64,
    log_bytes: u64,
}

/// Snapshot the store at the persistence location `source` into a gzip-compressed tar
/// **object** at `out_location`. Both are [`open_persistence`](crate::open_persistence)
/// locations — a local path/`file://`, or an `s3://`/`gs://` object store — so a store
/// living on any backend can be snapshotted to any backend.
pub fn backup(source: &str, out_location: &str) -> Result<BackupReport> {
    // Read the source store's durable objects through its backend — `data` first, then
    // `log`, for the consistent lock-free snapshot (see the module docs).
    let src = crate::open_persistence(source)?;
    let data = src
        .get("data")?
        .with_context(|| format!("no nidus store at {source} (no `data` object)"))?;
    let log = src.get("log")?.unwrap_or_default();
    let (dimension, distance) = crate::data::header_from_bytes(&data)
        .with_context(|| format!("{source} has no readable nidus header"))?;

    let created_unix = now_unix();

    // Build the whole gzip-tar archive in memory, then PUT it as one object. A
    // snapshot of a dev/small-scale store fits in RAM comfortably (SPEC §13.7).
    let mut archive: Vec<u8> = Vec::new();
    {
        let gz = GzEncoder::new(&mut archive, Compression::default());
        let mut tar = tar::Builder::new(gz);
        append_bytes(&mut tar, "data", &data, created_unix)?;
        append_bytes(&mut tar, "log", &log, created_unix)?;

        let manifest = Manifest {
            nidus_version: env!("CARGO_PKG_VERSION"),
            created_unix,
            dimension,
            distance: format!("{distance:?}"),
            data_bytes: data.len() as u64,
            log_bytes: log.len() as u64,
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
        append_bytes(&mut tar, MANIFEST, &manifest_bytes, created_unix)?;

        // Finalize tar, then the gzip stream — both flush into `archive`.
        let gz = tar.into_inner().context("failed to finalize tar archive")?;
        gz.finish().context("failed to finish gzip stream")?;
    }

    let (dest, key) = open_object_location(out_location)?;
    dest.put(&key, &archive)
        .with_context(|| format!("failed to write backup to {out_location}"))?;

    Ok(BackupReport {
        backup: out_location.to_string(),
        source: source.to_string(),
        dimension,
        distance: format!("{distance:?}"),
        data_bytes: data.len() as u64,
        log_bytes: log.len() as u64,
        archive_bytes: archive.len() as u64,
    })
}

/// Restore the store in the archive at `in_location` into the persistence location
/// `target` (a local path/`file://`, or an `s3://`/`gs://` object store).
///
/// If `target` already holds a store, the caller must confirm: with
/// `assume_yes == false` we prompt on stderr and read one line from stdin;
/// anything but `y`/`yes` (including EOF / a non-interactive pipe) aborts.
pub fn restore(
    in_location: &str,
    target_location: &str,
    assume_yes: bool,
) -> Result<RestoreReport> {
    // Extract the source-of-truth objects into the target store's backend. `put`
    // validates each key (rejecting any path separators / `..`), so a hand-crafted
    // traversal entry can never escape the store.
    let target = crate::open_persistence(target_location)?;

    if store_present(target.as_ref()) && !assume_yes && !confirm_overwrite(target_location)? {
        bail!("aborted: {target_location} already contains a store (pass -y/--yes to overwrite)");
    }

    let (src, key) = open_object_location(in_location)?;
    let archive = src
        .get(&key)?
        .with_context(|| format!("backup archive not found: {in_location}"))?;
    let mut tar = tar::Archive::new(GzDecoder::new(&archive[..]));
    let mut found_data = false;
    for entry in tar.entries().context("malformed backup archive")? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) if path.components().count() == 1 => n.to_string(),
            _ => continue,
        };
        if ARCHIVED.contains(&name.as_str()) {
            let mut buf = Vec::new();
            entry
                .read_to_end(&mut buf)
                .with_context(|| format!("failed to read `{name}` from archive"))?;
            target
                .put(&name, &buf)
                .with_context(|| format!("failed to write `{name}`"))?;
            if name == "data" {
                found_data = true;
            }
        }
        // The manifest and anything unexpected are ignored.
    }

    if !found_data {
        bail!("backup archive contained no `data` object — not a nidus backup");
    }

    // Leave a clean store: never carry over a stale writer lock.
    let _ = target.delete("lock");

    // Validate by reopening read-only — surfaces a corrupt/incompatible archive
    // instead of silently leaving an unloadable store behind.
    let data = target
        .get("data")?
        .context("restored store has no `data` object")?;
    let (dimension, distance) = crate::data::header_from_bytes(&data)
        .context("restored data has no readable nidus header")?;
    let db = Nidus::open(
        Config::new(".", dimension)
            .distance(distance)
            .persistence(target_location)
            .open_mode(OpenMode::ReadOnly),
    )
    .context("restored store failed to open — the archive may be corrupt")?;

    Ok(RestoreReport {
        restored_to: target_location.to_string(),
        source_archive: in_location.to_string(),
        dimension,
        distance: format!("{distance:?}"),
        collections: db.collections(),
        records: db.footprint().doc_count,
    })
}

/// A sortable default backup object name: `<dir-name>-<unix-secs>.tar.gz` (written to
/// the current directory). Cron users template their own via `--out`.
pub fn default_out_name(dir: &Path) -> String {
    let stem = dir
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("nidus");
    format!("{stem}-{}.tar.gz", now_unix())
}

/// Append an in-memory byte buffer as a tar entry.
fn append_bytes<W: std::io::Write>(
    tar: &mut tar::Builder<W>,
    name: &str,
    bytes: &[u8],
    mtime: u64,
) -> Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_mtime(mtime);
    tar.append_data(&mut header, name, bytes)
        .with_context(|| format!("failed to archive `{name}`"))?;
    Ok(())
}

/// Does the target backend already hold store objects we'd overwrite? Backend errors
/// read as "absent" (the safe direction — the restore then proceeds and surfaces any
/// real failure on `put`).
fn store_present(p: &dyn Persistence) -> bool {
    matches!(p.get("data"), Ok(Some(_))) || matches!(p.get("log"), Ok(Some(_)))
}

/// Prompt on stderr; return `true` only on an explicit yes. EOF or a
/// non-interactive pipe reads as empty → `false` (safe default).
fn confirm_overwrite(target: &str) -> Result<bool> {
    use std::io::Write;
    eprint!("{target} already contains a store; overwrite it? [y/N] ");
    std::io::stderr().flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let answer = line.trim().to_ascii_lowercase();
    Ok(answer == "y" || answer == "yes")
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{Record, Scope, SearchOpts};

    fn rec(id: &str, vector: Vec<f32>) -> Record {
        Record::new(id, vector, BTreeMap::new())
    }

    fn make_store(dir: &Path) {
        let mut db = Nidus::open(Config::new(dir.to_path_buf(), 3)).unwrap();
        db.upsert(
            "docs",
            &[rec("a", vec![1.0, 0.0, 0.0]), rec("b", vec![0.0, 1.0, 0.0])],
        )
        .unwrap();
        db.flush().unwrap();
    }

    #[test]
    fn round_trip_preserves_records() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let arc = tempfile::tempdir().unwrap();
        let archive = arc.path().join("snap.tar.gz");
        make_store(src.path());

        let report = backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();
        assert_eq!(report.dimension, 3);
        assert!(archive.exists());

        // Restore into a fresh (empty) directory.
        let restored = dst.path().join("store");
        let rr = restore(
            &archive.to_string_lossy(),
            &restored.to_string_lossy(),
            true,
        )
        .unwrap();
        assert_eq!(rr.records, 2);
        assert_eq!(rr.collections, vec!["docs".to_string()]);

        // No stale writer lock was carried into the restored store.
        assert!(!restored.join("lock").exists());

        // The restored store answers the same query.
        let db = Nidus::open(Config::new(restored, 3).open_mode(OpenMode::ReadOnly)).unwrap();
        let hits = db
            .search(
                Scope::All,
                &[1.0, 0.0, 0.0],
                &SearchOpts {
                    top_k: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(hits[0].id, "a");
    }

    #[test]
    fn backup_to_file_url_then_restore() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        make_store(src.path());

        // `file://<abs path>` destination exercises the URL-scheme path.
        let archive = src.path().join("via-url.tar.gz");
        let url = format!("file://{}", archive.display());
        backup(&src.path().to_string_lossy(), &url).unwrap();
        assert!(archive.exists());

        let restored = dst.path().join("store");
        let rr = restore(&url, &restored.to_string_lossy(), true).unwrap();
        assert_eq!(rr.records, 2);
    }

    #[test]
    fn restore_into_existing_store_without_yes_aborts() {
        let src = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        let arc = tempfile::tempdir().unwrap();
        let archive = arc.path().join("snap.tar.gz");
        make_store(src.path());
        make_store(target.path()); // target already holds a store

        backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();
        // assume_yes == false with no interactive stdin (EOF) → abort.
        let err = restore(
            &archive.to_string_lossy(),
            &target.path().to_string_lossy(),
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("already contains a store"));
    }

    #[test]
    fn backup_rejects_missing_store() {
        let empty = tempfile::tempdir().unwrap();
        let archive = empty.path().join("snap.tar.gz");
        let err = backup(&empty.path().to_string_lossy(), &archive.to_string_lossy()).unwrap_err();
        assert!(err.to_string().contains("no nidus store"));
    }
}
