//! Bounded, opt-in history of past commit points (nidus-bnf): enough per entry to
//! reconstruct the reader snapshot a manifest publish named. New object keys only —
//! this module tracks the live `manifest` format independently (SPEC §14.2 history
//! subsection).

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::Manifest;
use crate::backend::Persistence;
use crate::model::Distance;
use crate::profile::OpenProfile;

/// History frame format version. v2 appends `aliases` to [`HistoryEntry`]; a v1 entry still
/// loads, lifted with an empty alias map. `pub(crate)` so `store/write.rs` can stamp a
/// freshly-built [`HistoryEntry`]/[`HistoryFloor`] without a second constant.
pub(crate) const HIST_FORMAT_VERSION: u16 = 2;
const HIST_PREFIX: &str = "hist-";
const FLOOR_KEY: &str = "hist-floor";

/// One published commit point, enough to reconstruct the reader snapshot it named:
/// the segment set plus the exact log length at that commit (a row-count bound alone
/// would still replay a later `Delete`/`UpsertText`, which carries no row).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct HistoryEntry {
    pub format_version: u16,
    pub version: u64,
    pub dimension: u64,
    pub distance: Distance,
    pub segments: Vec<String>,
    pub next_id: u64,
    pub row_count: u64,
    pub log_offset: u64,
    pub profile: OpenProfile,
    /// Display only, never addressable (`--at-time` is a purely additive follow-up).
    pub commit_millis: u64,
    /// Alias table as it stood at this commit (nidus-klh). Added in v2; a lifted v1
    /// entry gets an empty map.
    pub aliases: BTreeMap<String, String>,
}

/// The v1 `HistoryEntry` shape (no `aliases`). Frozen: this is v1's exact historical layout,
/// so editing it silently breaks reading history entries written before v2.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct HistoryEntryV1 {
    format_version: u16,
    version: u64,
    dimension: u64,
    distance: Distance,
    segments: Vec<String>,
    next_id: u64,
    row_count: u64,
    log_offset: u64,
    profile: OpenProfile,
    commit_millis: u64,
}

/// The retention floor: no version below this is readable, whatever objects survive —
/// written before `compact()` rewrites the base segment in place, so a crash between the
/// floor write and the rewrite only loses history, never serves wrong bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct HistoryFloor {
    pub format_version: u16,
    pub oldest_readable: u64,
}

impl HistoryEntry {
    /// Reconstruct the [`Manifest`] this entry named, for `Segments::open` /
    /// `OpLog::open_bounded` to rebuild the pinned snapshot against.
    pub(crate) fn manifest(&self) -> Manifest {
        let mut m = Manifest::new(
            self.dimension as usize,
            self.distance,
            self.segments.clone(),
            self.next_id,
            self.version,
            self.profile.clone(),
        );
        m.aliases = self.aliases.clone();
        m
    }

    fn encode(&self) -> Result<Vec<u8>> {
        encode_frame(self)
    }

    /// bincode is positional, not self-describing, so a v1 buffer (no `aliases`) runs out of
    /// bytes before filling that field — decode into the old shape first, then dispatch on
    /// its version, mirroring `Manifest::decode` exactly.
    fn decode(bytes: &[u8]) -> Result<HistoryEntry> {
        let payload = checked_payload(bytes, "history entry")?;
        let v1: HistoryEntryV1 =
            bincode::deserialize(payload).context("deserialize history entry")?;
        match v1.format_version {
            1 => Ok(HistoryEntry {
                format_version: v1.format_version,
                version: v1.version,
                dimension: v1.dimension,
                distance: v1.distance,
                segments: v1.segments,
                next_id: v1.next_id,
                row_count: v1.row_count,
                log_offset: v1.log_offset,
                profile: v1.profile,
                commit_millis: v1.commit_millis,
                aliases: BTreeMap::new(),
            }),
            2 => {
                let entry: HistoryEntry =
                    bincode::deserialize(payload).context("deserialize history entry")?;
                Ok(entry)
            }
            other => bail!(
                "history entry format version {} is not supported (expected {})",
                other,
                HIST_FORMAT_VERSION
            ),
        }
    }
}

impl HistoryFloor {
    fn encode(&self) -> Result<Vec<u8>> {
        encode_frame(self)
    }

    /// Unlike `HistoryEntry`, the floor's shape never changed across v1/v2 — widen the
    /// accepted range instead of freezing a V1 shape, or a v1 floor becomes unreadable
    /// for no reason.
    fn decode(bytes: &[u8]) -> Result<HistoryFloor> {
        let floor: HistoryFloor = decode_frame(bytes, "history floor")?;
        if !(1..=HIST_FORMAT_VERSION).contains(&floor.format_version) {
            bail!(
                "history floor format version {} is not supported (expected {})",
                floor.format_version,
                HIST_FORMAT_VERSION
            );
        }
        Ok(floor)
    }
}

/// CRC32 + bincode, mirroring `Manifest::encode` exactly (§ "Codec discipline").
fn encode_frame<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let payload = bincode::serialize(value).context("serialize history frame")?;
    let crc = crc32fast::hash(&payload);
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// CRC-verify a frame and hand back its payload slice, so a caller that needs to decode into
/// more than one candidate shape (e.g. [`HistoryEntry::decode`]'s v1/v2 dispatch) does the
/// CRC check exactly once.
fn checked_payload<'a>(bytes: &'a [u8], what: &str) -> Result<&'a [u8]> {
    if bytes.len() < 4 {
        bail!(
            "{what} object is truncated: {} bytes (need ≥ 4)",
            bytes.len()
        );
    }
    let stored = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let payload = &bytes[4..];
    let computed = crc32fast::hash(payload);
    if computed != stored {
        bail!(
            "{what} CRC mismatch (stored {stored:#010x}, computed {computed:#010x}) — the \
             {what} object is corrupt"
        );
    }
    Ok(payload)
}

/// CRC32-verified decode shared by [`HistoryFloor::decode`] and the encode side; the
/// format-version check is the caller's, since the two types don't share one.
fn decode_frame<T: for<'de> Deserialize<'de>>(bytes: &[u8], what: &str) -> Result<T> {
    let payload = checked_payload(bytes, what)?;
    bincode::deserialize(payload).context(format!("deserialize {what}"))
}

/// The key one history entry lives under — exactly `hist-` + 20 decimal digits.
fn entry_key(version: u64) -> String {
    format!("{HIST_PREFIX}{version:020}")
}

/// The load-bearing parse rule: a history entry key is `hist-` + EXACTLY 20 digits.
/// `hist-floor` shares the prefix, so a `list()` sweep that skipped this check would
/// try to decode the floor object as an entry.
fn version_from_key(key: &str) -> Option<u64> {
    let digits = key.strip_prefix(HIST_PREFIX)?;
    if digits.len() != 20 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

pub(crate) fn load_entry(p: &dyn Persistence, version: u64) -> Result<Option<HistoryEntry>> {
    match p.get(&entry_key(version))? {
        Some(bytes) => Ok(Some(HistoryEntry::decode(&bytes)?)),
        None => Ok(None),
    }
}

pub(crate) fn load_floor(p: &dyn Persistence) -> Result<Option<HistoryFloor>> {
    match p.get(FLOOR_KEY)? {
        Some(bytes) => Ok(Some(HistoryFloor::decode(&bytes)?)),
        None => Ok(None),
    }
}

pub(crate) fn store_entry(p: &dyn Persistence, entry: &HistoryEntry) -> Result<()> {
    let bytes = entry.encode()?;
    p.put(&entry_key(entry.version), &bytes)
        .context("write history entry object")
}

pub(crate) fn store_floor(p: &dyn Persistence, floor: &HistoryFloor) -> Result<()> {
    let bytes = floor.encode()?;
    p.put(FLOOR_KEY, &bytes)
        .context("write history floor object")
}

pub(crate) fn delete_entry(p: &dyn Persistence, version: u64) -> Result<()> {
    p.delete(&entry_key(version))
}

/// Every recorded history version, ascending. One `list()` call per invocation — callers
/// that need this repeatedly (e.g. `Store::versions`) should not call it from a hot path.
pub(crate) fn list_versions(p: &dyn Persistence) -> Result<Vec<u64>> {
    let mut versions: Vec<u64> = p
        .list()?
        .iter()
        .filter_map(|k| version_from_key(k))
        .collect();
    versions.sort_unstable();
    Ok(versions)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::Duration;

    use super::*;
    use crate::backend::BackendLock;

    /// A trivial in-RAM [`Persistence`] double — pure Rust, no real IO, so these tests stay
    /// Miri-clean. `try_lock` is unused by history but required by the trait.
    #[derive(Default)]
    struct MemBackend {
        objects: Mutex<HashMap<String, Vec<u8>>>,
    }

    impl Persistence for MemBackend {
        fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
            Ok(self.objects.lock().unwrap().get(key).cloned())
        }
        fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
            self.objects
                .lock()
                .unwrap()
                .insert(key.to_string(), bytes.to_vec());
            Ok(())
        }
        fn delete(&self, key: &str) -> Result<()> {
            self.objects.lock().unwrap().remove(key);
            Ok(())
        }
        fn list(&self) -> Result<Vec<String>> {
            Ok(self.objects.lock().unwrap().keys().cloned().collect())
        }
        fn try_lock(&self, _key: &str, _ttl: Duration) -> Result<Option<Box<dyn BackendLock>>> {
            Ok(None)
        }
    }

    fn sample_entry(version: u64) -> HistoryEntry {
        HistoryEntry {
            format_version: HIST_FORMAT_VERSION,
            version,
            dimension: 8,
            distance: Distance::Cosine,
            segments: vec!["data".to_string()],
            next_id: 1,
            row_count: 3,
            log_offset: 42,
            profile: OpenProfile::default(),
            commit_millis: 1_700_000_000_000,
            aliases: BTreeMap::new(),
        }
    }

    #[test]
    fn entry_round_trips_through_store_and_load() {
        let p = MemBackend::default();
        let entry = sample_entry(7);
        store_entry(&p, &entry).unwrap();
        let back = load_entry(&p, 7).unwrap().unwrap();
        assert_eq!(back, entry);
        assert!(load_entry(&p, 8).unwrap().is_none());
    }

    #[test]
    fn entry_round_trips_with_aliases() {
        let p = MemBackend::default();
        let mut entry = sample_entry(9);
        entry
            .aliases
            .insert("docs".to_string(), "docs_v2".to_string());
        store_entry(&p, &entry).unwrap();
        let back = load_entry(&p, 9).unwrap().unwrap();
        assert_eq!(back, entry);
        assert_eq!(
            back.manifest().aliases.get("docs"),
            Some(&"docs_v2".to_string())
        );
    }

    /// A hand-built v1 blob (no `aliases`), decoded WITHOUT ever calling the current
    /// `encode` — the gap that would let a broken lift path go green.
    #[test]
    fn entry_decode_lifts_a_hand_built_v1_blob() {
        let v1 = HistoryEntryV1 {
            format_version: 1,
            version: 4,
            dimension: 8,
            distance: Distance::Cosine,
            segments: vec!["data".to_string()],
            next_id: 1,
            row_count: 2,
            log_offset: 10,
            profile: OpenProfile::default(),
            commit_millis: 1_700_000_000_000,
        };
        let payload = bincode::serialize(&v1).unwrap();
        let crc = crc32fast::hash(&payload);
        let mut bytes = Vec::with_capacity(4 + payload.len());
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&payload);

        let entry = HistoryEntry::decode(&bytes).unwrap();
        assert_eq!(entry.format_version, 1);
        assert_eq!(entry.version, 4);
        assert!(entry.aliases.is_empty());
    }

    /// A hand-built v1 [`HistoryFloor`] blob still loads under the widened v1..=v2 check.
    #[test]
    fn floor_decode_still_loads_a_v1_floor() {
        let floor = HistoryFloor {
            format_version: 1,
            oldest_readable: 3,
        };
        let bytes = floor.encode().unwrap();
        assert_eq!(HistoryFloor::decode(&bytes).unwrap(), floor);
    }

    #[test]
    fn floor_round_trips_through_store_and_load() {
        let p = MemBackend::default();
        assert!(load_floor(&p).unwrap().is_none());
        let floor = HistoryFloor {
            format_version: HIST_FORMAT_VERSION,
            oldest_readable: 5,
        };
        store_floor(&p, &floor).unwrap();
        assert_eq!(load_floor(&p).unwrap().unwrap(), floor);
    }

    #[test]
    fn entry_decode_rejects_crc_corruption() {
        let entry = sample_entry(1);
        let mut bytes = entry.encode().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(HistoryEntry::decode(&bytes).is_err());
    }

    #[test]
    fn entry_decode_rejects_truncated_object() {
        assert!(HistoryEntry::decode(&[0u8; 2]).is_err());
    }

    #[test]
    fn entry_decode_rejects_unknown_format_version() {
        let mut entry = sample_entry(1);
        entry.format_version = 99;
        let bytes = entry.encode().unwrap();
        let err = HistoryEntry::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("format version"), "{err}");
    }

    #[test]
    fn floor_decode_rejects_crc_corruption() {
        let floor = HistoryFloor {
            format_version: HIST_FORMAT_VERSION,
            oldest_readable: 2,
        };
        let mut bytes = floor.encode().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(HistoryFloor::decode(&bytes).is_err());
    }

    #[test]
    fn version_from_key_accepts_a_well_formed_entry_key() {
        assert_eq!(version_from_key(&entry_key(123)), Some(123));
    }

    #[test]
    fn version_from_key_rejects_the_floor_key() {
        assert_eq!(version_from_key(FLOOR_KEY), None);
    }

    #[test]
    fn version_from_key_rejects_a_short_number() {
        assert_eq!(version_from_key("hist-123"), None);
    }

    #[test]
    fn version_from_key_rejects_a_20_char_non_numeric_suffix() {
        assert_eq!(version_from_key("hist-abcdefghijklmnopqrst"), None);
    }

    #[test]
    fn list_versions_sorts_ascending_and_skips_the_floor() {
        let p = MemBackend::default();
        store_entry(&p, &sample_entry(5)).unwrap();
        store_entry(&p, &sample_entry(1)).unwrap();
        store_entry(&p, &sample_entry(3)).unwrap();
        store_floor(
            &p,
            &HistoryFloor {
                format_version: HIST_FORMAT_VERSION,
                oldest_readable: 0,
            },
        )
        .unwrap();
        assert_eq!(list_versions(&p).unwrap(), vec![1, 3, 5]);
    }

    #[test]
    fn delete_entry_removes_only_that_version() {
        let p = MemBackend::default();
        store_entry(&p, &sample_entry(1)).unwrap();
        store_entry(&p, &sample_entry(2)).unwrap();
        delete_entry(&p, 1).unwrap();
        assert!(load_entry(&p, 1).unwrap().is_none());
        assert!(load_entry(&p, 2).unwrap().is_some());
    }
}
