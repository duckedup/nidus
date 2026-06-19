//! [`LocalFs`]: a [`Persistence`] backend over a local directory — the default, and
//! the impl every other backend's behaviour is checked against. Each object is a file
//! `<dir>/<key>`; whole-object writes are atomic (temp + fsync + rename), the native
//! [`Appender`] is a plain `File` (so the live `data`/`log` path keeps its exact
//! append + fsync + rename discipline with zero overhead), and `try_lock` is the
//! existing O_EXCL [`WriteLock`](crate::lock::WriteLock) — one source, not a copy.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};

use super::{Appender, BackendLock, Persistence, validate_key};
use crate::lock::WriteLock;

/// A persistence backend rooted at a local directory. The directory is created on
/// construction if it does not exist.
pub struct LocalFs {
    dir: PathBuf,
}

impl LocalFs {
    /// Root a backend at `dir`, creating it (and parents) if absent.
    pub fn new(dir: impl Into<PathBuf>) -> Result<LocalFs> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create backend directory {}", dir.display()))?;
        Ok(LocalFs { dir })
    }

    /// The directory this backend is rooted at.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn object_path(&self, key: &str) -> Result<PathBuf> {
        validate_key(key)?;
        Ok(self.dir.join(key))
    }
}

impl Persistence for LocalFs {
    fn local_path(&self, key: &str) -> Option<PathBuf> {
        // A plain file under the root — mappable. Reject malformed keys (the mmap caller
        // then falls back to a RAM load) rather than surfacing an error from a path getter.
        self.object_path(key).ok()
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let path = self.object_path(key)?;
        match std::fs::read(&path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("failed to read {}", path.display())),
        }
    }

    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let path = self.object_path(key)?;
        // Atomic whole-object write: temp + fsync + rename (the same discipline the
        // index-cache codec uses). `<key>.tmp` is a sibling so the rename is atomic.
        let tmp = path.with_extension("tmp");
        {
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)
                .with_context(|| format!("failed to create temp object {}", tmp.display()))?;
            f.write_all(bytes)
                .with_context(|| format!("failed to write temp object {}", tmp.display()))?;
            f.sync_all()
                .with_context(|| format!("failed to fsync temp object {}", tmp.display()))?;
        }
        std::fs::rename(&tmp, &path).with_context(|| {
            format!("failed to rename {} into {}", tmp.display(), path.display())
        })?;
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        let path = self.object_path(key)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e).with_context(|| format!("failed to delete {}", path.display())),
        }
    }

    fn list(&self) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        for entry in std::fs::read_dir(&self.dir)
            .with_context(|| format!("failed to list {}", self.dir.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_file()
                && let Some(name) = entry.file_name().to_str()
            {
                keys.push(name.to_string());
            }
        }
        keys.sort();
        Ok(keys)
    }

    fn appender(&self, key: &str) -> Result<Option<Box<dyn Appender>>> {
        let path = self.object_path(key)?;
        Ok(Some(Box::new(FileAppender::open(&path)?)))
    }

    fn try_lock(&self, key: &str, ttl: Duration) -> Result<Option<Box<dyn BackendLock>>> {
        let path = self.object_path(key)?;
        Ok(WriteLock::try_acquire_path(&path, ttl)?.map(|l| Box::new(l) as Box<dyn BackendLock>))
    }
}

/// The existing O_EXCL writer lock *is* the local backend lock — no second
/// implementation.
impl BackendLock for WriteLock {}

/// A local-file [`Appender`]: a plain `File` opened read+write, positioned at the end.
/// Mirrors the `data`/`log` append + rollback + rewrite logic exactly, so backing
/// those segments with it later is a drop-in with no behavioural change.
pub struct FileAppender {
    path: PathBuf,
    handle: File,
}

impl FileAppender {
    /// Open or create the object at `path` for appending, positioning at the end.
    pub fn open(path: &Path) -> Result<FileAppender> {
        let mut handle = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("failed to open appender at {}", path.display()))?;
        handle
            .seek(SeekFrom::End(0))
            .with_context(|| format!("failed to seek to end of {}", path.display()))?;
        Ok(FileAppender {
            path: path.to_path_buf(),
            handle,
        })
    }
}

impl Appender for FileAppender {
    fn len(&self) -> Result<u64> {
        Ok(self
            .handle
            .metadata()
            .with_context(|| format!("failed to stat {}", self.path.display()))?
            .len())
    }

    fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.handle
            .seek(SeekFrom::Start(offset))
            .with_context(|| format!("failed to seek {} to {offset}", self.path.display()))?;
        self.handle
            .read_exact(buf)
            .with_context(|| format!("failed to read {} at {offset}", self.path.display()))
    }

    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        // Always append at the true end: a preceding `read_exact_at` may have left the
        // cursor mid-file, so don't trust it — seek to the end and capture that as the
        // rollback boundary.
        let start = self
            .handle
            .seek(SeekFrom::End(0))
            .with_context(|| format!("failed to seek to end of {}", self.path.display()))?;
        if let Err(e) = self.handle.write_all(bytes) {
            // Roll back any partial bytes to the boundary the append started at.
            let _ = self.handle.set_len(start);
            let _ = self.handle.seek(SeekFrom::Start(start));
            return Err(anyhow::Error::new(e))
                .with_context(|| format!("failed to append to {}", self.path.display()));
        }
        Ok(())
    }

    fn truncate_to(&mut self, offset: u64) -> Result<()> {
        self.handle
            .set_len(offset)
            .with_context(|| format!("failed to truncate {}", self.path.display()))?;
        self.handle
            .seek(SeekFrom::Start(offset))
            .with_context(|| format!("failed to seek after truncating {}", self.path.display()))?;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.handle
            .sync_all()
            .with_context(|| format!("failed to fsync {}", self.path.display()))
    }

    fn rewrite(&mut self, bytes: &[u8]) -> Result<()> {
        let dir = self
            .path
            .parent()
            .context("appender path has no parent directory")?;
        let tmp = dir.join(format!(
            "{}.tmp",
            self.path
                .file_name()
                .and_then(|n| n.to_str())
                .context("appender path has no file name")?
        ));
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)
                .with_context(|| format!("failed to create temp {}", tmp.display()))?;
            f.write_all(bytes)
                .with_context(|| format!("failed to write temp {}", tmp.display()))?;
            f.sync_all()
                .with_context(|| format!("failed to fsync temp {}", tmp.display()))?;
        }
        std::fs::rename(&tmp, &self.path).with_context(|| {
            format!(
                "failed to rename {} into {}",
                tmp.display(),
                self.path.display()
            )
        })?;
        let mut handle = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .with_context(|| format!("failed to reopen {} after rewrite", self.path.display()))?;
        handle.seek(SeekFrom::End(0)).with_context(|| {
            format!(
                "failed to seek to end of {} after rewrite",
                self.path.display()
            )
        })?;
        self.handle = handle;
        Ok(())
    }
    // `read_to_end` uses the trait's provided impl (fallible-reserve + `read_exact_at`).
}
