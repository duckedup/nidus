//! `nidus backup` / `nidus restore`: snapshot a store directory into a single
//! pure-Rust `.tar.gz`, and extract one back.
//!
//! A store is a directory holding `data` (append-only flat `f32` matrix) and
//! `log` (the commit record); `lock` is per-process writer exclusion and the
//! `.tmp` files are transient compaction scratch — neither belongs in a backup.
//!
//! **Why a hot backup is consistent (no writer lock needed).** The writer
//! fsyncs `data` *before* appending committing records to `log`, and every
//! reader ignores log records that reference a row beyond the data file's size
//! (`row ≥ data_len / dim`). So a snapshot that captures `data` first — at a
//! fixed size — and `log` second sees exactly what a lock-free reader would:
//! possibly a hair stale, never torn. We therefore copy `data` then `log`, each
//! at the length observed when we start streaming it.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde::Serialize;

use crate::{Config, Nidus, OpenMode};

/// Files that make up the durable, portable state of a store. Order matters:
/// `data` is captured before `log` so the snapshot is a consistent lock-free
/// view (see the module docs).
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

/// Snapshot the store at `dir` into a gzip-compressed tar at `out`.
pub fn backup(dir: &Path, out: &Path) -> Result<BackupReport> {
    let data_path = dir.join("data");
    if !data_path.exists() {
        bail!("no nidus store at {} (no `data` file)", dir.display());
    }
    // Read dim/distance for the manifest (and to confirm the header is sane).
    let (dimension, distance) = crate::data::peek_header(&data_path)?
        .with_context(|| format!("{} has no readable nidus header", data_path.display()))?;

    let file = File::create(out)
        .with_context(|| format!("failed to create backup file {}", out.display()))?;
    let gz = GzEncoder::new(file, Compression::default());
    let mut tar = tar::Builder::new(gz);

    let created_unix = now_unix();

    // `data` first, then `log` — the consistent-snapshot ordering.
    let data_bytes = append_fixed(&mut tar, &data_path, "data", created_unix)?;
    let log_bytes = append_fixed(&mut tar, &dir.join("log"), "log", created_unix)?;

    // Manifest last (it is derived from the two above).
    let manifest = Manifest {
        nidus_version: env!("CARGO_PKG_VERSION"),
        created_unix,
        dimension,
        distance: format!("{distance:?}"),
        data_bytes,
        log_bytes,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    append_bytes(&mut tar, MANIFEST, &manifest_bytes, created_unix)?;

    // Finish the tar, then the gzip stream, then make the bytes durable.
    let gz = tar.into_inner().context("failed to finalize tar archive")?;
    let mut file = gz.finish().context("failed to finish gzip stream")?;
    file.flush()?;
    file.sync_all().ok();

    let archive_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);

    Ok(BackupReport {
        backup: out.display().to_string(),
        source: dir.display().to_string(),
        dimension,
        distance: format!("{distance:?}"),
        data_bytes,
        log_bytes,
        archive_bytes,
    })
}

/// Restore the store in archive `archive` into `dir`.
///
/// If `dir` already holds a store, the caller must confirm: with
/// `assume_yes == false` we prompt on stderr and read one line from stdin;
/// anything but `y`/`yes` (including EOF / a non-interactive pipe) aborts.
pub fn restore(archive: &Path, dir: &Path, assume_yes: bool) -> Result<RestoreReport> {
    if store_present(dir) && !assume_yes && !confirm_overwrite(dir)? {
        bail!(
            "aborted: {} already contains a store (pass -y/--yes to overwrite)",
            dir.display()
        );
    }

    let file = File::open(archive)
        .with_context(|| format!("failed to open backup {}", archive.display()))?;
    let mut tar = tar::Archive::new(GzDecoder::new(file));

    std::fs::create_dir_all(dir)
        .with_context(|| format!("failed to create target directory {}", dir.display()))?;

    for entry in tar.entries().context("malformed backup archive")? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        // Only ever write the known, bare filenames — never honour a path with
        // directory components (defends against a hand-crafted traversal entry).
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) if path.components().count() == 1 => n.to_string(),
            _ => continue,
        };
        if ARCHIVED.contains(&name.as_str()) {
            entry
                .unpack(dir.join(&name))
                .with_context(|| format!("failed to extract `{name}`"))?;
        }
        // The manifest and anything unexpected are ignored.
    }

    if !dir.join("data").exists() {
        bail!("backup archive contained no `data` file — not a nidus backup");
    }

    // Leave a clean store: never carry over a stale writer lock or tmp scratch.
    for stale in ["lock", "data.tmp", "log.tmp"] {
        let _ = std::fs::remove_file(dir.join(stale));
    }

    // Validate by reopening read-only — surfaces a corrupt/incompatible archive
    // instead of silently leaving an unloadable store behind.
    let (dimension, distance) = crate::data::peek_header(&dir.join("data"))?
        .context("restored data file has no readable nidus header")?;
    let db = Nidus::open(
        Config::new(dir.to_path_buf(), dimension)
            .distance(distance)
            .open_mode(OpenMode::ReadOnly),
    )
    .context("restored store failed to open — the archive may be corrupt")?;

    Ok(RestoreReport {
        restored_to: dir.display().to_string(),
        source_archive: archive.display().to_string(),
        dimension,
        distance: format!("{distance:?}"),
        collections: db.collections(),
        records: db.footprint().doc_count,
    })
}

/// A sortable default backup filename: `<dir-name>-<unix-secs>.tar.gz`. Cron
/// users template their own via `--out`.
pub fn default_out_name(dir: &Path) -> String {
    let stem = dir
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("nidus");
    format!("{stem}-{}.tar.gz", now_unix())
}

/// Append `src` to the tar at a fixed size: snapshot `len` once, then stream
/// exactly that many bytes. A concurrent append that grows the file mid-copy is
/// ignored (we stop at `len`), so the entry never contains a torn trailing row.
fn append_fixed<W: Write>(
    tar: &mut tar::Builder<W>,
    src: &Path,
    name: &str,
    mtime: u64,
) -> Result<u64> {
    let file = File::open(src).with_context(|| format!("failed to read {}", src.display()))?;
    let len = file.metadata()?.len();
    let mut header = tar::Header::new_gnu();
    header.set_size(len);
    header.set_mode(0o644);
    header.set_mtime(mtime);
    // append_data sets the path and recomputes the checksum for us.
    tar.append_data(&mut header, name, file.take(len))
        .with_context(|| format!("failed to archive `{name}`"))?;
    Ok(len)
}

/// Append an in-memory byte buffer (the manifest) as a tar entry.
fn append_bytes<W: Write>(
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

/// Does `dir` already hold store files we'd overwrite?
fn store_present(dir: &Path) -> bool {
    dir.join("data").exists() || dir.join("log").exists()
}

/// Prompt on stderr; return `true` only on an explicit yes. EOF or a
/// non-interactive pipe reads as empty → `false` (safe default).
fn confirm_overwrite(dir: &Path) -> Result<bool> {
    eprint!(
        "{} already contains a store; overwrite it? [y/N] ",
        dir.display()
    );
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

        let report = backup(src.path(), &archive).unwrap();
        assert_eq!(report.dimension, 3);
        assert!(archive.exists());

        // Restore into a fresh (empty) directory.
        let restored = dst.path().join("store");
        let rr = restore(&archive, &restored, true).unwrap();
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
    fn restore_into_existing_store_without_yes_aborts() {
        let src = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        let arc = tempfile::tempdir().unwrap();
        let archive = arc.path().join("snap.tar.gz");
        make_store(src.path());
        make_store(target.path()); // target already holds a store

        backup(src.path(), &archive).unwrap();
        // assume_yes == false with no interactive stdin (EOF) → abort.
        let err = restore(&archive, target.path(), false).unwrap_err();
        assert!(err.to_string().contains("already contains a store"));
    }

    #[test]
    fn backup_rejects_missing_store() {
        let empty = tempfile::tempdir().unwrap();
        let archive = empty.path().join("snap.tar.gz");
        let err = backup(empty.path(), &archive).unwrap_err();
        assert!(err.to_string().contains("no nidus store"));
    }
}
