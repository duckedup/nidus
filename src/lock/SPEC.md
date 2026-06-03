# `lock` module — spec

Implement `WriteLock` in `mod.rs`: best-effort writer exclusion via an `O_EXCL` lock
file, pure std (no `flock`/FFI). **Do not change the public signatures.** Root
design: `../../SPEC.md` §6.3.

## Behavior
- `acquire(dir: &Path, ttl: Duration) -> Result<WriteLock>`:
  - Lock path is `dir.join("lock")`.
  - Try to create it atomically: `OpenOptions::new().write(true).create_new(true)`.
    On success, write a small diagnostic body (e.g. `"<pid> <unix_millis>"` —
    `std::process::id()` and `SystemTime::now()`), keep the path, return the lock.
  - On `AlreadyExists`: stat the file's modified time. If `now - mtime > ttl`, the
    holder is presumed dead (stale) → remove the file and retry the create **once**.
    If still failing, or the lock is fresh, return an `anyhow` error whose message
    makes clear the store is locked and names the path (callers surface this).
  - Any other IO error → propagate via `anyhow` (`.context()`).
- `Drop`: best-effort remove the lock file (ignore errors). Only the path is needed.

## Constraints
Pure safe Rust (`#![forbid(unsafe_code)]`), `std` + `anyhow` only. No PID-liveness
syscalls (that would be FFI) — staleness is purely `ttl` vs file mtime. Do not block
/ spin waiting for the lock; fail fast.

## Tests (`tests` submodule, file-backed → `#[cfg_attr(miri, ignore)]`)
Use `tempfile::tempdir()`. Verify: first `acquire` succeeds and creates the file; a
second `acquire` on the same dir fails while the first guard is alive; dropping the
guard removes the file and lets a later `acquire` succeed; a lock file older than
`ttl` (set the file's mtime into the past, or use a `ttl` of 0 / tiny duration) is
reclaimed. Keep any pure helpers (e.g. staleness comparison) factored out and
Miri-clean if practical.
