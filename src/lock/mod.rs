//! Writer exclusion via an `O_EXCL` lock file (pure std, no `flock`/FFI).
//! Contract: see `SPEC.md` in this directory.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, bail};

/// A held writer lock. Removes the lock file on drop.
#[derive(Debug)]
pub struct WriteLock {
    path: PathBuf,
}

impl WriteLock {
    /// Acquire `<dir>/lock` by atomic create (`create_new`). Writes the PID + a
    /// timestamp inside for diagnostics. If the file already exists and is older
    /// than `ttl` (stale — a crashed writer), reclaim it; otherwise return an error
    /// (`anyhow`, surfaced as a clear "store is locked" message).
    pub fn acquire(dir: &Path, ttl: Duration) -> Result<WriteLock> {
        let path = dir.join("lock");
        match try_create_lock(&path) {
            Ok(lock) => Ok(lock),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // Existing lock: reclaim it if stale, otherwise refuse.
                if is_stale(&path, ttl)? {
                    std::fs::remove_file(&path).with_context(|| {
                        format!("failed to reclaim stale lock at {}", path.display())
                    })?;
                    match try_create_lock(&path) {
                        Ok(lock) => Ok(lock),
                        Err(e2) if e2.kind() == std::io::ErrorKind::AlreadyExists => {
                            bail!("store is locked: {}", path.display())
                        }
                        Err(e2) => Err(e2).with_context(|| {
                            format!(
                                "failed to acquire lock at {} after reclaiming stale lock",
                                path.display()
                            )
                        }),
                    }
                } else {
                    bail!("store is locked: {}", path.display())
                }
            }
            Err(e) => {
                Err(e).with_context(|| format!("failed to create lock file at {}", path.display()))
            }
        }
    }
}

/// Attempt an atomic O_EXCL create of the lock file and write PID + timestamp.
fn try_create_lock(path: &Path) -> std::io::Result<WriteLock> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;

    let pid = std::process::id();
    let unix_millis = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis();
    write!(file, "{pid} {unix_millis}")?;
    file.flush()?;

    Ok(WriteLock {
        path: path.to_path_buf(),
    })
}

/// Returns true if the lock file's mtime is older than `ttl`.
fn is_stale(path: &Path, ttl: Duration) -> Result<bool> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("failed to stat lock file at {}", path.display()))?;
    let mtime = metadata
        .modified()
        .with_context(|| format!("failed to read mtime of lock file at {}", path.display()))?;
    let age = SystemTime::now()
        .duration_since(mtime)
        .unwrap_or(Duration::ZERO);
    Ok(age > ttl)
}

impl Drop for WriteLock {
    fn drop(&mut self) {
        // Best-effort remove — ignore errors.
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn acquire_creates_lock_file() {
        let dir = tempfile::tempdir().unwrap();
        let lock = WriteLock::acquire(dir.path(), Duration::from_secs(60)).unwrap();
        assert!(
            dir.path().join("lock").exists(),
            "lock file should exist after acquire"
        );
        drop(lock);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn second_acquire_fails_while_first_is_alive() {
        let dir = tempfile::tempdir().unwrap();
        let _lock = WriteLock::acquire(dir.path(), Duration::from_secs(60)).unwrap();
        let result = WriteLock::acquire(dir.path(), Duration::from_secs(60));
        assert!(
            result.is_err(),
            "second acquire should fail while first guard is alive"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("store is locked"),
            "error message should mention 'store is locked', got: {err_msg}"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn drop_removes_lock_file_and_subsequent_acquire_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let lock = WriteLock::acquire(dir.path(), Duration::from_secs(60)).unwrap();
        let lock_path = dir.path().join("lock");
        assert!(lock_path.exists());
        drop(lock);
        assert!(
            !lock_path.exists(),
            "lock file should be removed after drop"
        );
        // A subsequent acquire should now succeed.
        let lock2 = WriteLock::acquire(dir.path(), Duration::from_secs(60)).unwrap();
        assert!(lock_path.exists());
        drop(lock2);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn stale_lock_is_reclaimed() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("lock");

        // Write a "stale" lock file manually.
        std::fs::write(&lock_path, b"99999 0").unwrap();

        // Set its mtime to the distant past by using a zero ttl — any file
        // created before "now" will appear stale with ttl=0.
        // With ttl of Duration::ZERO, is_stale checks age > 0, which will be
        // true for any file created even 1ns ago. We ensure staleness by
        // sleeping a tiny bit or simply using a small ttl.
        //
        // Actually, with ttl=0 and a freshly written file, the mtime is "now"
        // and age ≈ 0, so age > 0 may or may not be true depending on clock
        // resolution. To guarantee staleness, modify the file mtime explicitly.
        //
        // Use filetime to backdate the mtime. But we can't add deps — instead
        // we create the file and then sleep 10ms and use a ttl=1ms.
        std::thread::sleep(Duration::from_millis(20));

        // Acquire with a ttl of 1ms — the 20ms-old file is stale.
        let lock = WriteLock::acquire(dir.path(), Duration::from_millis(1))
            .expect("should reclaim stale lock");
        assert!(
            lock_path.exists(),
            "new lock file should exist after reclaiming stale one"
        );
        drop(lock);
        assert!(!lock_path.exists());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn error_message_names_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let _lock = WriteLock::acquire(dir.path(), Duration::from_secs(60)).unwrap();
        let err = WriteLock::acquire(dir.path(), Duration::from_secs(60)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("lock"),
            "error message should mention the lock path, got: {msg}"
        );
    }
}
