//! `nidus backup` / `nidus restore`: snapshot a store into one pure-Rust `.tar.gz` object, and
//! extract it back. The archive's source/destination is a `Persistence` location, so a snapshot is
//! one named object on any backend (SPEC §13.7) — exactly object-granular, hence trivial everywhere.

use std::io::Read;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};

use crate::backend::{Persistence, open_object_location};
use crate::{Config, Nidus, OpenMode};

/// Embedded report entry name (informational; restore tolerates its absence).
const MANIFEST: &str = "nidus-backup.json";

/// Is `name` part of a store's durable object set? (`ann`/`fts` caches and `lock`
/// are rebuildable/transient and deliberately excluded, #130.)
fn is_store_object(name: &str) -> bool {
    name == "data"
        || name == "log"
        || name == crate::manifest::MANIFEST_KEY
        || name.starts_with("seg-")
}

/// What a backup recorded, printed as JSON by the CLI.
#[derive(Debug, Serialize)]
pub struct BackupReport {
    pub backup: String,
    pub source: String,
    pub dimension: usize,
    pub distance: String,
    pub data_bytes: u64,
    pub log_bytes: u64,
    pub segments: usize,
    pub segment_bytes: u64,
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

/// What `verify` found, printed as JSON by the CLI.
#[derive(Debug, Serialize)]
pub struct VerifyReport {
    pub archive: String,
    pub dimension: usize,
    pub distance: String,
    pub collections: Vec<String>,
    pub records: usize,
    pub objects_checked: usize,
    pub archive_bytes: u64,
}

/// A stored object's size and CRC32, checked at `backup()` time and rechecked by `verify`.
#[derive(Serialize, Deserialize)]
struct ObjectSum {
    name: String,
    bytes: u64,
    crc32: u32,
}

/// The small JSON manifest embedded in each archive: who/what/when, readable without
/// unpacking the binary `data`/`log`. `objects` is load-bearing, not descriptive — it is
/// the CRC baseline `restore`/`verify` check. An archive predating it still restores.
#[derive(Serialize, Deserialize)]
struct Manifest {
    nidus_version: String,
    created_unix: u64,
    dimension: usize,
    distance: String,
    data_bytes: u64,
    log_bytes: u64,
    segments: usize,
    #[serde(default)]
    objects: Vec<ObjectSum>,
}

/// Snapshot the store at `source` into a gzip-compressed tar object at `out_location`. Both are
/// [`open_persistence`](crate::open_persistence) locations, so a store on any backend can be
/// snapshotted to any backend.
pub fn backup(source: &str, out_location: &str) -> Result<BackupReport> {
    // Read order gives the lock-free snapshot (§6.2): the store `manifest` (the segment
    // set's commit point) first, then the segments it names (sealed ones are immutable),
    // then `log` last, so replay ignores anything newer than the data we captured.
    let src = crate::open_persistence(source)?;
    let store_manifest = src.get(crate::manifest::MANIFEST_KEY)?;
    let sealed: Vec<String> = match &store_manifest {
        Some(bytes) => crate::manifest::Manifest::decode(bytes)
            .with_context(|| format!("{source} has an unreadable `manifest` object"))?
            .segments
            .into_iter()
            .filter(|name| name != "data")
            .collect(),
        None => Vec::new(),
    };
    let data = src
        .get("data")?
        .with_context(|| format!("no nidus store at {source} (no `data` object)"))?;
    let mut segments: Vec<(String, Vec<u8>)> = Vec::with_capacity(sealed.len());
    for name in &sealed {
        let bytes = src.get(name)?.with_context(|| {
            format!("{source} is torn: the manifest names `{name}` but it does not exist")
        })?;
        segments.push((name.clone(), bytes));
    }
    let log = src.get("log")?.unwrap_or_default();
    let (dimension, distance) = crate::data::header_from_bytes(&data)
        .with_context(|| format!("{source} has no readable nidus header"))?;

    let created_unix = now_unix();
    let segment_bytes: u64 = segments.iter().map(|(_, b)| b.len() as u64).sum();

    // CRCs over the exact byte buffers already read above — race-free by construction,
    // since nothing else can mutate these bytes between the read and the checksum.
    let mut objects: Vec<ObjectSum> = vec![object_sum("data", &data)];
    for (name, bytes) in &segments {
        objects.push(object_sum(name, bytes));
    }
    if let Some(bytes) = &store_manifest {
        objects.push(object_sum(crate::manifest::MANIFEST_KEY, bytes));
    }
    objects.push(object_sum("log", &log));

    // Build the whole gzip-tar archive in memory, then PUT it as one object. A
    // snapshot of a dev/small-scale store fits in RAM comfortably (SPEC §13.7).
    let mut archive: Vec<u8> = Vec::new();
    {
        let gz = GzEncoder::new(&mut archive, Compression::default());
        let mut tar = tar::Builder::new(gz);
        append_bytes(&mut tar, "data", &data, created_unix)?;
        for (name, bytes) in &segments {
            append_bytes(&mut tar, name, bytes, created_unix)?;
        }
        // Verbatim bytes: the manifest is CRC-framed, so re-encoding it would break it.
        if let Some(bytes) = &store_manifest {
            append_bytes(&mut tar, crate::manifest::MANIFEST_KEY, bytes, created_unix)?;
        }
        append_bytes(&mut tar, "log", &log, created_unix)?;

        let manifest = Manifest {
            nidus_version: env!("CARGO_PKG_VERSION").to_string(),
            created_unix,
            dimension,
            distance: format!("{distance:?}"),
            data_bytes: data.len() as u64,
            log_bytes: log.len() as u64,
            segments: segments.len(),
            objects,
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
        segments: segments.len(),
        segment_bytes,
        archive_bytes: archive.len() as u64,
    })
}

/// Restore the store in the archive at `in_location` into the persistence location
/// `target` (a local path/`file://`, or an `s3://`/`gs://` object store).
pub fn restore(
    in_location: &str,
    target_location: &str,
    assume_yes: bool,
) -> Result<RestoreReport> {
    // Extract the source-of-truth objects into the target store's backend. `put`
    // validates each key (rejecting any path separators / `..`), so a hand-crafted
    // traversal entry can never escape the store.
    let target = crate::open_persistence(target_location)?;

    // Fully validate the archive before touching the target: a corrupt one must not
    // leave a half-restored store behind, and must never pass silently (#152).
    let (src, key) = open_object_location(in_location)?;
    let archive = src
        .get(&key)?
        .with_context(|| format!("backup archive not found: {in_location}"))?;
    let (objects, manifest) = read_archive(&archive)?;
    check_baseline(&objects, manifest.as_ref())?;
    if !objects.iter().any(|(name, _)| name == "data") {
        bail!("backup archive contained no `data` object — not a nidus backup");
    }

    if store_present(target.as_ref()) && !assume_yes && !confirm_overwrite(target_location)? {
        bail!("aborted: {target_location} already contains a store (pass -y/--yes to overwrite)");
    }

    // Clear the target's segment state before writing: a pre-existing `manifest` or
    // `seg-*` the archive does not carry would otherwise point at segments that no
    // longer match the restored `data`/`log` (#130).
    let _ = target.delete(crate::manifest::MANIFEST_KEY);
    if let Ok(existing) = target.list() {
        for name in existing.iter().filter(|n| n.starts_with("seg-")) {
            target
                .delete(name)
                .with_context(|| format!("failed to remove stale `{name}` at the target"))?;
        }
    }

    put_objects(target.as_ref(), &objects)?;

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
        // The path arg is unused: a non-empty `persistence(target_location)` drives the
        // open, so `"."` is just a placeholder (see `Store::open`'s location resolution).
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

/// A store object's name and bytes, as carried in an archive.
type Object = (String, Vec<u8>);

/// Every store object in `archive`, plus its embedded report, read in one pass.
/// Shared by `restore` and `verify` so both get the same traversal guard and gzip check.
fn read_archive(archive: &[u8]) -> Result<(Vec<Object>, Option<Manifest>)> {
    let mut tar = tar::Archive::new(GzDecoder::new(archive));
    let mut objects: Vec<Object> = Vec::new();
    let mut manifest = None;
    for entry in tar.entries().context("malformed backup archive")? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) if path.components().count() == 1 => n.to_string(),
            _ => continue,
        };
        if is_store_object(&name) {
            // A repeated name would be checked once but written twice (last wins), so the
            // bytes that land could be ones no CRC ever covered. Refuse instead.
            if objects.iter().any(|(seen, _)| *seen == name) {
                bail!("backup archive carries `{name}` more than once");
            }
            let mut buf = Vec::new();
            entry
                .read_to_end(&mut buf)
                .with_context(|| format!("failed to read `{name}` from archive"))?;
            objects.push((name, buf));
        } else if name == MANIFEST {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            manifest = Some(
                serde_json::from_slice(&buf)
                    .with_context(|| format!("archive's `{MANIFEST}` is not valid JSON"))?,
            );
        }
    }
    // GzDecoder checks its trailer CRC only on a read that hits true EOF (#152).
    let mut rest = tar.into_inner();
    std::io::copy(&mut rest, &mut std::io::sink())
        .context("backup archive failed its gzip integrity check")?;
    Ok((objects, manifest))
}

/// Recheck the archive's own CRC baseline, returning how many objects it covered.
/// An archive written before 0.57 carries none; that is not a failure.
fn check_baseline(objects: &[Object], manifest: Option<&Manifest>) -> Result<usize> {
    let baseline = match manifest {
        Some(m) if !m.objects.is_empty() => &m.objects,
        _ => return Ok(0),
    };
    for sum in baseline {
        let bytes = objects
            .iter()
            .find(|(name, _)| *name == sum.name)
            .map(|(_, bytes)| bytes)
            .with_context(|| {
                format!("archive's baseline names `{}` but it is missing", sum.name)
            })?;
        let got = crc32fast::hash(bytes);
        let got_len = bytes.len() as u64;
        if got != sum.crc32 || got_len != sum.bytes {
            bail!(
                "archive object `{}` is corrupt: expected crc32 {:08x} ({} bytes), got {:08x} ({} bytes)",
                sum.name,
                sum.crc32,
                sum.bytes,
                got,
                got_len
            );
        }
    }
    Ok(baseline.len())
}

/// Write the extracted objects into a store location.
fn put_objects(target: &dyn Persistence, objects: &[Object]) -> Result<()> {
    for (name, bytes) in objects {
        target
            .put(name, bytes)
            .with_context(|| format!("failed to write `{name}`"))?;
    }
    Ok(())
}

/// Prove `in_location` is a restorable backup: drain the gzip stream, recheck every
/// baseline CRC (if the archive carries one), and open the extracted store read-only.
/// Never touches a real store — extraction lands in a `TempDir` cleaned up on every path.
pub fn verify(in_location: &str) -> Result<VerifyReport> {
    let (src, key) = open_object_location(in_location)?;
    let archive = src
        .get(&key)?
        .with_context(|| format!("backup archive not found: {in_location}"))?;

    let (objects, manifest) = read_archive(&archive)?;
    let objects_checked = check_baseline(&objects, manifest.as_ref())?;
    if !objects.iter().any(|(name, _)| name == "data") {
        bail!("backup archive contained no `data` object — not a nidus backup");
    }

    let scratch = tempfile::TempDir::new().context("failed to create scratch dir")?;
    let scratch_target = crate::open_persistence(&scratch.path().to_string_lossy())?;
    put_objects(scratch_target.as_ref(), &objects)?;

    let data = scratch_target
        .get("data")?
        .context("extracted store has no `data` object")?;
    let (dimension, distance) = crate::data::header_from_bytes(&data)
        .context("extracted data has no readable nidus header")?;
    let scratch_location = scratch.path().to_string_lossy().into_owned();
    let db = Nidus::open(
        Config::new(".", dimension)
            .distance(distance)
            .persistence(&scratch_location)
            .open_mode(OpenMode::ReadOnly),
    )
    .context("extracted store failed to open — the archive may be corrupt")?;

    Ok(VerifyReport {
        archive: in_location.to_string(),
        dimension,
        distance: format!("{distance:?}"),
        collections: db.collections(),
        records: db.footprint().doc_count,
        objects_checked,
        archive_bytes: archive.len() as u64,
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

/// Compute a named object's checksum baseline for the embedded manifest.
fn object_sum(name: &str, bytes: &[u8]) -> ObjectSum {
    ObjectSum {
        name: name.to_string(),
        bytes: bytes.len() as u64,
        crc32: crc32fast::hash(bytes),
    }
}

/// Does the target backend already hold store objects we'd overwrite? Backend errors
/// read as "absent" (the safe direction — the restore then proceeds and surfaces any
/// real failure on `put`).
fn store_present(p: &dyn Persistence) -> bool {
    matches!(p.get("data"), Ok(Some(_)))
        || matches!(p.get("log"), Ok(Some(_)))
        || matches!(p.get(crate::manifest::MANIFEST_KEY), Ok(Some(_)))
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
    use std::io::Read;

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

    /// Write enough rows through a small `segment_max_rows` that the store seals
    /// segments, returning each row's id and vector for later parity checks.
    fn make_segmented_store(dir: &Path) -> Vec<(String, Vec<f32>)> {
        let mut db =
            Nidus::open(Config::new(dir.to_path_buf(), 3).segment_max_rows(Some(2))).unwrap();
        let mut rows = Vec::new();
        for i in 0..7u32 {
            let v = vec![1.0 + i as f32, (i % 3) as f32, 0.5];
            let id = format!("r{i}");
            db.upsert("docs", &[rec(&id, v.clone())]).unwrap();
            rows.push((id, v));
        }
        db.flush().unwrap();
        rows
    }

    /// #130: a segmented store's sealed segments and its `manifest` must survive the
    /// round trip — the old two-object archive silently lost every sealed row.
    #[test]
    fn segmented_store_round_trip_is_complete() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let archive = src.path().join("snap.tar.gz");
        let rows = make_segmented_store(src.path());
        assert!(
            src.path().join("manifest").exists()
                && std::fs::read_dir(src.path())
                    .unwrap()
                    .filter_map(|e| e.ok())
                    .any(|e| e.file_name().to_string_lossy().starts_with("seg-")),
            "test premise: the source store must actually be segmented"
        );

        let report = backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();
        assert!(
            report.segments >= 1,
            "sealed segments must be archived: {report:?}"
        );

        let restored = dst.path().join("store");
        let rr = restore(
            &archive.to_string_lossy(),
            &restored.to_string_lossy(),
            true,
        )
        .unwrap();
        assert_eq!(
            rr.records,
            rows.len(),
            "every row survives, sealed ones included"
        );

        // Ranking parity: each row's own vector must rank itself first.
        let db = Nidus::open(
            Config::new(restored, 3)
                .segment_max_rows(Some(2))
                .open_mode(OpenMode::ReadOnly),
        )
        .unwrap();
        for (id, v) in &rows {
            let hits = db
                .search(
                    Scope::All,
                    v,
                    &SearchOpts {
                        top_k: 1,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(&hits[0].id, id, "restored ranking diverged for {id}");
        }
    }

    /// #130 (restore half): restoring an unsegmented archive over a segmented store
    /// must not leave a stale `manifest`/`seg-*` pointing at vanished rows.
    #[test]
    fn restore_over_a_segmented_store_leaves_no_stale_segments() {
        let src = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        let archive = src.path().join("snap.tar.gz");
        make_store(src.path()); // unsegmented, 2 records
        make_segmented_store(target.path()); // 7 records across segments

        backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();
        let rr = restore(
            &archive.to_string_lossy(),
            &target.path().to_string_lossy(),
            true,
        )
        .unwrap();
        assert_eq!(
            rr.records, 2,
            "the restored store is the archive's, not the old one"
        );

        let stale: Vec<String> = std::fs::read_dir(target.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("seg-"))
            .collect();
        assert!(
            stale.is_empty(),
            "stale segments survived the restore: {stale:?}"
        );
    }

    /// Corrupt the first tar entry's *content* and re-gzip losslessly, so every
    /// structural layer stays valid: gzip's trailer, tar's header checksum, and the
    /// deflate stream. Only the embedded CRC baseline can catch this (#152).
    fn corrupt_entry_content(archive: &[u8], content_offset: usize) -> Vec<u8> {
        let mut tar_bytes = Vec::new();
        GzDecoder::new(archive).read_to_end(&mut tar_bytes).unwrap();
        tar_bytes[512 + content_offset] ^= 0xFF;

        let mut out = Vec::new();
        let mut enc = GzEncoder::new(&mut out, Compression::default());
        std::io::Write::write_all(&mut enc, &tar_bytes).unwrap();
        enc.finish().unwrap();
        out
    }

    /// Rebuild `archive` with its embedded `nidus-backup.json` stripped of the
    /// `objects` baseline, simulating an archive written before #152.
    fn strip_objects_baseline(archive: &[u8]) -> Vec<u8> {
        let mut tar = tar::Archive::new(GzDecoder::new(archive));
        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
        for entry in tar.entries().unwrap() {
            let mut entry = entry.unwrap();
            let name = entry
                .path()
                .unwrap()
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap()
                .to_string();
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf).unwrap();
            entries.push((name, buf));
        }

        let mut out = Vec::new();
        {
            let gz = GzEncoder::new(&mut out, Compression::default());
            let mut builder = tar::Builder::new(gz);
            for (name, bytes) in &mut entries {
                if name == MANIFEST {
                    let mut v: serde_json::Value = serde_json::from_slice(bytes).unwrap();
                    v.as_object_mut().unwrap().remove("objects");
                    *bytes = serde_json::to_vec_pretty(&v).unwrap();
                }
                append_bytes(&mut builder, name, bytes, 0).unwrap();
            }
            let gz = builder.into_inner().unwrap();
            gz.finish().unwrap();
        }
        out
    }

    #[test]
    fn pristine_archive_verifies_and_reports_source() {
        let src = tempfile::tempdir().unwrap();
        let arc = tempfile::tempdir().unwrap();
        let archive = arc.path().join("snap.tar.gz");
        make_store(src.path());
        backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();

        let report = verify(&archive.to_string_lossy()).unwrap();
        assert_eq!(report.dimension, 3);
        assert_eq!(report.records, 2);
        assert_eq!(report.collections, vec!["docs".to_string()]);
        assert!(report.objects_checked > 0);
    }

    /// #152: corrupted vector bytes must fail `verify` even when every structural
    /// layer still checks out — the case a bare "does it open?" check cannot see.
    #[test]
    fn corrupted_vector_content_fails_verify() {
        let src = tempfile::tempdir().unwrap();
        let arc = tempfile::tempdir().unwrap();
        let archive = arc.path().join("snap.tar.gz");
        make_store(src.path());
        backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();

        let bytes = std::fs::read(&archive).unwrap();
        std::fs::write(&archive, corrupt_entry_content(&bytes, 70)).unwrap();

        let err = verify(&archive.to_string_lossy()).unwrap_err().to_string();
        assert!(err.contains("data") && err.contains("crc32"), "{err}");
    }

    /// A duplicate entry would be CRC-checked on its first copy and written from its
    /// last, so the persisted bytes could be ones no baseline ever covered.
    #[test]
    fn duplicate_object_entry_is_refused() {
        let src = tempfile::tempdir().unwrap();
        let arc = tempfile::tempdir().unwrap();
        let archive = arc.path().join("snap.tar.gz");
        make_store(src.path());
        backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();

        // Rebuild with `data` appended a second time, corrupted.
        let bytes = std::fs::read(&archive).unwrap();
        let mut tar_bytes = Vec::new();
        GzDecoder::new(&bytes[..])
            .read_to_end(&mut tar_bytes)
            .unwrap();
        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
        for entry in tar::Archive::new(&tar_bytes[..]).entries().unwrap() {
            let mut entry = entry.unwrap();
            let name = entry.path().unwrap().to_string_lossy().into_owned();
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf).unwrap();
            entries.push((name, buf));
        }
        let mut dup = entries.iter().find(|(n, _)| n == "data").unwrap().clone();
        dup.1[70] ^= 0xFF;
        entries.push(dup);

        let mut out = Vec::new();
        {
            let gz = GzEncoder::new(&mut out, Compression::default());
            let mut builder = tar::Builder::new(gz);
            for (name, bytes) in &entries {
                append_bytes(&mut builder, name, bytes, 0).unwrap();
            }
            builder.into_inner().unwrap().finish().unwrap();
        }
        std::fs::write(&archive, out).unwrap();

        let err = verify(&archive.to_string_lossy()).unwrap_err().to_string();
        assert!(err.contains("more than once"), "{err}");
    }

    /// A flip that breaks the gzip stream itself must fail too, via the trailer CRC
    /// the drain now reaches.
    #[test]
    fn bit_flip_in_compressed_body_fails_verify() {
        let src = tempfile::tempdir().unwrap();
        let arc = tempfile::tempdir().unwrap();
        let archive = arc.path().join("snap.tar.gz");
        make_store(src.path());
        backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();

        let mut bytes = std::fs::read(&archive).unwrap();
        let offset = bytes.len() / 3;
        bytes[offset] ^= 0x01;
        std::fs::write(&archive, bytes).unwrap();

        assert!(verify(&archive.to_string_lossy()).is_err());
    }

    /// #152 regression: `restore` on the same corrupted archive must also err, not
    /// silently write corrupt vector bytes into the target store.
    #[test]
    fn restore_on_corrupted_archive_also_errs() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let arc = tempfile::tempdir().unwrap();
        let archive = arc.path().join("snap.tar.gz");
        make_store(src.path());
        backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();

        let bytes = std::fs::read(&archive).unwrap();
        std::fs::write(&archive, corrupt_entry_content(&bytes, 70)).unwrap();

        let restored = dst.path().join("store");
        assert!(
            restore(
                &archive.to_string_lossy(),
                &restored.to_string_lossy(),
                true
            )
            .is_err()
        );
        // Validated before the target is touched, so nothing was half-written.
        assert!(!restored.join("data").exists());
    }

    #[test]
    fn truncated_archive_fails_verify() {
        let src = tempfile::tempdir().unwrap();
        let arc = tempfile::tempdir().unwrap();
        let archive = arc.path().join("snap.tar.gz");
        make_store(src.path());
        backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();

        let mut bytes = std::fs::read(&archive).unwrap();
        bytes.truncate(bytes.len() - 8); // drop the gzip trailer
        std::fs::write(&archive, bytes).unwrap();

        assert!(verify(&archive.to_string_lossy()).is_err());
    }

    #[test]
    fn segmented_store_verifies_clean() {
        let src = tempfile::tempdir().unwrap();
        let arc = tempfile::tempdir().unwrap();
        let archive = arc.path().join("snap.tar.gz");
        let rows = make_segmented_store(src.path());
        backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();

        let report = verify(&archive.to_string_lossy()).unwrap();
        assert_eq!(report.records, rows.len());
        assert!(report.objects_checked > 0);
    }

    /// Older archives (no `objects` baseline) still verify structurally, with
    /// `objects_checked == 0` — nonzero exit is reserved for a real mismatch.
    #[test]
    fn archive_without_objects_baseline_still_verifies() {
        let src = tempfile::tempdir().unwrap();
        let arc = tempfile::tempdir().unwrap();
        let archive = arc.path().join("snap.tar.gz");
        make_store(src.path());
        backup(&src.path().to_string_lossy(), &archive.to_string_lossy()).unwrap();

        let bytes = std::fs::read(&archive).unwrap();
        let stripped = strip_objects_baseline(&bytes);
        std::fs::write(&archive, stripped).unwrap();

        let report = verify(&archive.to_string_lossy()).unwrap();
        assert_eq!(report.objects_checked, 0);
        assert_eq!(report.records, 2);
    }
}
