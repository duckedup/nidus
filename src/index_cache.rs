//! Shared on-disk codec for **derived index caches** (the ANN graph/lists, the FTS
//! inverted index, …). A cache is never the source of truth — both indexes are fully
//! reconstructable from the `data`/`log` files — so this codec exists only to let
//! `open` *load* an index instead of rebuilding it. Every load is therefore
//! best-effort: a missing, stale (validity key changed), or CRC-failed file returns
//! `Ok(None)` and the caller rebuilds. It is never fatal.
//!
//! One framing for every index (so ANN and FTS don't each hand-roll it):
//!
//! ```text
//! MAGIC(6) "NIDUS\0" | VERSION(2 LE) | watermark(8 LE) | key_len(2 LE) | key(key_len)
//!   | bincode(payload) | crc32(payload) (4 LE)
//! ```
//!
//! The **key** is opaque validity bytes the caller composes (ANN: dim/metric/params;
//! FTS: schema hash/language) — any change to it invalidates the cache. The
//! **watermark** is *data*, not part of the key: it records how much the cache covers
//! (rows / docs) so the caller can incrementally catch up the rest. Writes are atomic
//! (temp file + fsync + rename) so a crash can't leave a torn cache.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Magic bytes, shared with the `data`/`log` files.
const MAGIC: &[u8; 6] = b"NIDUS\0";
/// Framing version (bump only if this layout itself changes, not per-index).
const VERSION: u16 = 1;
/// Bytes before the variable-length key: magic + version + watermark + key_len.
const FIXED: usize = 6 + 2 + 8 + 2;

/// Build the full framed byte buffer (header + payload + CRC) in memory. Shared by
/// [`save`] (which then writes it atomically) and the tests (which keep it Miri-clean
/// by never touching the filesystem).
pub(crate) fn frame<T: Serialize>(key: &[u8], watermark: u64, payload: &T) -> Result<Vec<u8>> {
    let bytes = bincode::serialize(payload).context("serialize index cache payload")?;
    let crc = crc32fast::hash(&bytes);
    let key_len = u16::try_from(key.len()).context("index cache validity key too long")?;

    let mut out = Vec::with_capacity(FIXED + key.len() + bytes.len() + 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&watermark.to_le_bytes());
    out.extend_from_slice(&key_len.to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&bytes);
    out.extend_from_slice(&crc.to_le_bytes());
    Ok(out)
}

/// Decode a framed buffer, validating magic, version, the caller's `key`, and the CRC.
/// Returns `None` (never an error) on any mismatch — the caller rebuilds. Pure and
/// filesystem-free, so it runs under Miri and is exercised directly in tests.
pub(crate) fn decode<T: DeserializeOwned>(bytes: &[u8], key: &[u8]) -> Option<(T, u64)> {
    if bytes.len() < FIXED {
        return None;
    }
    if &bytes[..6] != MAGIC || bytes[6..8] != VERSION.to_le_bytes() {
        return None;
    }
    let watermark = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let key_len = u16::from_le_bytes(bytes[16..18].try_into().unwrap()) as usize;
    let header_len = FIXED + key_len;
    // Room for the declared key plus the trailing 4-byte CRC.
    if bytes.len() < header_len + 4 {
        return None;
    }
    if &bytes[FIXED..header_len] != key {
        return None;
    }
    let payload = &bytes[header_len..bytes.len() - 4];
    let stored_crc = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
    if crc32fast::hash(payload) != stored_crc {
        return None;
    }
    bincode::deserialize::<T>(payload)
        .ok()
        .map(|p| (p, watermark))
}

/// Save `payload` to `path` atomically, valid only for `key`. `watermark` is how much
/// of the source the cache reflects (rows / docs), so a later [`load`] knows how far to
/// catch up.
pub(crate) fn save<T: Serialize>(
    path: &Path,
    key: &[u8],
    watermark: u64,
    payload: &T,
) -> Result<()> {
    let buf = frame(key, watermark, payload)?;
    let tmp = path.with_extension("tmp");
    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)
            .with_context(|| format!("create index cache temp {tmp:?}"))?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("rename index cache into {path:?}"))?;
    Ok(())
}

/// Load and validate the cache at `path` for `key`. `Ok(None)` — never an error — when
/// the file is absent, stale, or corrupt; on success returns the decoded payload and
/// the watermark it covers.
pub(crate) fn load<T: DeserializeOwned>(path: &Path, key: &[u8]) -> Result<Option<(T, u64)>> {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("open index cache {path:?}")),
    };
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)
        .with_context(|| format!("read index cache {path:?}"))?;
    Ok(decode::<T>(&bytes, key))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A stand-in payload exercising the generic codec independent of any real index.
    type Payload = Vec<(String, u32)>;

    fn sample() -> Payload {
        vec![("alpha".into(), 1), ("beta".into(), 2), ("gamma".into(), 3)]
    }

    #[test]
    fn roundtrips_payload_and_watermark() {
        let key = b"key-v1";
        let bytes = frame(key.as_slice(), 42, &sample()).unwrap();
        let (got, watermark) = decode::<Payload>(&bytes, key).unwrap();
        assert_eq!(got, sample());
        assert_eq!(watermark, 42);
    }

    #[test]
    fn key_mismatch_is_rejected() {
        let bytes = frame(b"key-A".as_slice(), 0, &sample()).unwrap();
        assert!(decode::<Payload>(&bytes, b"key-B").is_none());
        // A key that is a prefix of the stored key must also be rejected (length differs).
        assert!(decode::<Payload>(&bytes, b"key-").is_none());
    }

    #[test]
    fn corrupt_crc_is_rejected() {
        let key = b"k";
        let mut bytes = frame(key.as_slice(), 0, &sample()).unwrap();
        let mid = FIXED + key.len() + 1;
        bytes[mid] ^= 0xFF;
        assert!(decode::<Payload>(&bytes, key).is_none());
    }

    #[test]
    fn truncated_buffers_are_rejected() {
        let key = b"k";
        let bytes = frame(key.as_slice(), 0, &sample()).unwrap();
        // Every short prefix decodes to None, never panics.
        for n in 0..bytes.len() {
            assert!(decode::<Payload>(&bytes[..n], key).is_none(), "len {n}");
        }
        // The full buffer still decodes.
        assert!(decode::<Payload>(&bytes, key).is_some());
    }

    #[test]
    fn wrong_magic_or_version_is_rejected() {
        let key = b"k";
        let mut bytes = frame(key.as_slice(), 0, &sample()).unwrap();
        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(decode::<Payload>(&bad_magic, key).is_none());
        bytes[6] = bytes[6].wrapping_add(1); // bump version
        assert!(decode::<Payload>(&bytes, key).is_none());
    }
}
