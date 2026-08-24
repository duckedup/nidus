//! The `manifest`: the atomic commit point that names the live segments (SPEC §14.2).

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::backend::Persistence;
use crate::model::Distance;
use crate::profile::OpenProfile;

pub(crate) mod history;

/// The object key the manifest lives under within a store.
pub(crate) const MANIFEST_KEY: &str = "manifest";

/// The name of the first (base) segment — kept as `data` so a single-segment store stays
/// byte-compatible with the pre-segment layout (`peek_header`, snapshot/backup, and legacy
/// stores all keep resolving `data`). Sealed segments mint `seg-NNNNNNNN` names instead.
pub(crate) const BASE_SEGMENT: &str = "data";

/// Manifest frame format version (bumped only on an incompatible payload change).
/// v2 appends `profile`; v3 appends `aliases`. v1/v2 manifests still decode, lifted
/// with an empty profile/alias map respectively.
const FORMAT_VERSION: u16 = 3;

/// The live-segment set + the pins needed to open them. Serialized as the `manifest` object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Manifest {
    /// Frame format version (rejected on open if unknown).
    pub format_version: u16,
    /// The pinned embedding dimension (must match every segment header and the open config).
    pub dimension: u64,
    /// The pinned distance metric.
    pub distance: Distance,
    /// Live segments in global-row order; the **last** is the active (appendable) one.
    pub segments: Vec<String>,
    /// Monotonic counter for minting fresh `seg-NNNNNNNN` names — never reused, so a stale
    /// reader can't confuse an old segment object with a new one of the same name.
    pub next_id: u64,
    /// Monotonic manifest version, bumped on every seal/compaction. Carried now for the
    /// Phase-4 reader-refresh (a reader adopts a newer manifest when this advances); unused
    /// until then.
    pub version: u64,
    /// Recorded open-time defaults (nidus-141). Added in v2; a lifted v1 manifest gets
    /// [`OpenProfile::default`] (nothing recorded).
    pub profile: OpenProfile,
    /// Alias name → concrete collection name (nidus-klh). Added in v3; a lifted v1/v2
    /// manifest gets an empty map. `BTreeMap` for deterministic (CRC'd) encoding.
    pub aliases: BTreeMap<String, String>,
}

/// The v1 manifest shape (six fields, no `profile`), bincode's positional format only lets a
/// v1 buffer be told apart from v2 by first decoding into this shape and reading its version.
/// Frozen: this is v1's exact historical layout, so editing it silently breaks reading old stores.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManifestV1 {
    format_version: u16,
    dimension: u64,
    distance: Distance,
    segments: Vec<String>,
    next_id: u64,
    version: u64,
}

/// The v2 manifest shape (seven fields, no `aliases`). Frozen: this is v2's exact historical
/// layout, so editing it silently breaks reading stores written before v3.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManifestV2 {
    format_version: u16,
    dimension: u64,
    distance: Distance,
    segments: Vec<String>,
    next_id: u64,
    version: u64,
    profile: OpenProfile,
}

impl Manifest {
    /// A fresh single-segment manifest naming the base [`BASE_SEGMENT`] — used to
    /// initialize a brand-new store and to synthesize one for a legacy `data`+`log` store
    /// that predates the manifest (transparent migration).
    pub(crate) fn fresh(dimension: usize, distance: Distance) -> Manifest {
        Manifest {
            format_version: FORMAT_VERSION,
            dimension: dimension as u64,
            distance,
            segments: vec![BASE_SEGMENT.to_string()],
            next_id: 1,
            version: 1,
            profile: OpenProfile::default(),
            aliases: BTreeMap::new(),
        }
    }

    /// Build a manifest from explicit parts (the [`Segments`](crate::data::Segments)
    /// snapshot that compaction/seal persist). `segments` is in global-row order, last
    /// active.
    pub(crate) fn new(
        dimension: usize,
        distance: Distance,
        segments: Vec<String>,
        next_id: u64,
        version: u64,
        profile: OpenProfile,
    ) -> Manifest {
        Manifest {
            format_version: FORMAT_VERSION,
            dimension: dimension as u64,
            distance,
            segments,
            next_id,
            version,
            profile,
            aliases: BTreeMap::new(),
        }
    }

    /// Encode the manifest to its on-disk frame (`crc32` + bincode payload). Always emits
    /// the current [`FORMAT_VERSION`].
    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        let payload = bincode::serialize(self).context("serialize manifest")?;
        let crc = crc32fast::hash(&payload);
        let mut out = Vec::with_capacity(4 + payload.len());
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Decode a manifest frame, verifying the CRC, then dispatching on the format version
    /// (bincode is positional, not self-describing, so a v1 buffer runs out of bytes before
    /// filling `profile` — that must be handled explicitly, not defaulted).
    pub(crate) fn decode(bytes: &[u8]) -> Result<Manifest> {
        if bytes.len() < 4 {
            bail!(
                "manifest object is truncated: {} bytes (need ≥ 4)",
                bytes.len()
            );
        }
        let stored = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let payload = &bytes[4..];
        let computed = crc32fast::hash(payload);
        if computed != stored {
            bail!(
                "manifest CRC mismatch (stored {stored:#010x}, computed {computed:#010x}) \
                 — the manifest object is corrupt"
            );
        }
        let v1: ManifestV1 = bincode::deserialize(payload).context("deserialize manifest")?;
        match v1.format_version {
            1 => Ok(Manifest {
                format_version: v1.format_version,
                dimension: v1.dimension,
                distance: v1.distance,
                segments: v1.segments,
                next_id: v1.next_id,
                version: v1.version,
                profile: OpenProfile::default(),
                aliases: BTreeMap::new(),
            }),
            2 => {
                let v2: ManifestV2 =
                    bincode::deserialize(payload).context("deserialize manifest")?;
                Ok(Manifest {
                    format_version: v2.format_version,
                    dimension: v2.dimension,
                    distance: v2.distance,
                    segments: v2.segments,
                    next_id: v2.next_id,
                    version: v2.version,
                    profile: v2.profile,
                    aliases: BTreeMap::new(),
                })
            }
            3 => {
                let m: Manifest = bincode::deserialize(payload).context("deserialize manifest")?;
                Ok(m)
            }
            other => bail!(
                "manifest format version {} is not supported (expected {})",
                other,
                FORMAT_VERSION
            ),
        }
    }

    /// Read the manifest object from `persistence`. `Ok(None)` when absent (a fresh or a
    /// pre-manifest legacy store — the caller synthesizes one).
    pub(crate) fn load(persistence: &dyn Persistence) -> Result<Option<Manifest>> {
        match persistence.get(MANIFEST_KEY)? {
            Some(bytes) => Ok(Some(Self::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Publish the manifest atomically — the commit point for a seal/compaction.
    pub(crate) fn store(&self, persistence: &dyn Persistence) -> Result<()> {
        let bytes = self.encode()?;
        persistence
            .put(MANIFEST_KEY, &bytes)
            .context("write manifest object")
    }
}

/// Pure alias-shape checks shared by every write surface: empty name, self-reference, and
/// the no-chain rule (a target that is itself an alias). Existence/collision against real
/// collection names needs `Store::collections`, so those two checks live at the call site.
pub(crate) fn validate_alias(
    name: &str,
    target: &str,
    existing: &BTreeMap<String, String>,
) -> Result<()> {
    if name.is_empty() {
        bail!("alias name must not be empty");
    }
    if name == target {
        bail!("alias `{name}` cannot point at itself");
    }
    if existing.contains_key(target) {
        bail!(
            "alias `{name}` cannot point at `{target}`: `{target}` is itself an alias \
             (aliases resolve in one hop, never chained)"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AnnConfig, Quantization};

    #[test]
    fn fresh_names_the_base_segment() {
        let m = Manifest::fresh(8, Distance::Cosine);
        assert_eq!(m.segments, vec![BASE_SEGMENT.to_string()]);
        assert_eq!(m.dimension, 8);
        assert_eq!(m.next_id, 1);
        assert_eq!(m.version, 1);
        assert_eq!(m.profile, OpenProfile::default());
    }

    #[test]
    fn encode_decode_round_trip_empty_profile() {
        let m = Manifest {
            format_version: FORMAT_VERSION,
            dimension: 384,
            distance: Distance::DotProduct,
            segments: vec!["data".into(), "seg-00000001".into(), "seg-00000002".into()],
            next_id: 3,
            version: 7,
            profile: OpenProfile::default(),
            aliases: BTreeMap::new(),
        };
        let bytes = m.encode().unwrap();
        let back = Manifest::decode(&bytes).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn encode_decode_round_trip_populated_profile() {
        let m = Manifest {
            format_version: FORMAT_VERSION,
            dimension: 384,
            distance: Distance::Cosine,
            segments: vec!["data".into()],
            next_id: 1,
            version: 1,
            profile: OpenProfile {
                ann: Some(AnnConfig::hnsw()),
                quantization: Some(Quantization::int8()),
                query_threads: Some(4),
                mmap: Some(true),
            },
            aliases: BTreeMap::new(),
        };
        let bytes = m.encode().unwrap();
        let back = Manifest::decode(&bytes).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn encode_decode_round_trip_with_aliases() {
        let mut aliases = BTreeMap::new();
        aliases.insert("docs".to_string(), "docs_v2".to_string());
        aliases.insert("legacy".to_string(), "archive".to_string());
        let m = Manifest {
            format_version: FORMAT_VERSION,
            dimension: 384,
            distance: Distance::Cosine,
            segments: vec!["data".into()],
            next_id: 1,
            version: 1,
            profile: OpenProfile::default(),
            aliases,
        };
        let bytes = m.encode().unwrap();
        let back = Manifest::decode(&bytes).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.aliases.get("docs"), Some(&"docs_v2".to_string()));
    }

    /// The gap that would let a broken change go green: decode a hand-built v1 byte blob
    /// (no `profile` field at all) and confirm it lifts into a v3 `Manifest` with an empty
    /// profile and alias map, WITHOUT ever calling the current `encode` (which always emits v3).
    #[test]
    fn decode_lifts_a_hand_built_v1_blob() {
        let v1 = ManifestV1 {
            format_version: 1,
            dimension: 8,
            distance: Distance::Cosine,
            segments: vec![BASE_SEGMENT.to_string()],
            next_id: 1,
            version: 1,
        };
        let payload = bincode::serialize(&v1).unwrap();
        let crc = crc32fast::hash(&payload);
        let mut bytes = Vec::with_capacity(4 + payload.len());
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&payload);

        let m = Manifest::decode(&bytes).unwrap();
        assert_eq!(m.format_version, 1);
        assert_eq!(m.dimension, 8);
        assert_eq!(m.segments, vec![BASE_SEGMENT.to_string()]);
        assert_eq!(m.profile, OpenProfile::default());
        assert!(m.aliases.is_empty());
    }

    /// Same gap, one version up: a hand-built v2 blob (has `profile`, no `aliases`) must lift
    /// with an empty alias map, WITHOUT ever calling the current `encode`.
    #[test]
    fn decode_lifts_a_hand_built_v2_blob() {
        let v2 = ManifestV2 {
            format_version: 2,
            dimension: 16,
            distance: Distance::DotProduct,
            segments: vec![BASE_SEGMENT.to_string()],
            next_id: 2,
            version: 3,
            profile: OpenProfile {
                ann: None,
                quantization: Some(Quantization::int8()),
                query_threads: None,
                mmap: None,
            },
        };
        let payload = bincode::serialize(&v2).unwrap();
        let crc = crc32fast::hash(&payload);
        let mut bytes = Vec::with_capacity(4 + payload.len());
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&payload);

        let m = Manifest::decode(&bytes).unwrap();
        assert_eq!(m.format_version, 2);
        assert_eq!(m.dimension, 16);
        assert_eq!(m.profile.quantization, Some(Quantization::int8()));
        assert!(m.aliases.is_empty());
    }

    #[test]
    fn decode_rejects_crc_corruption() {
        let m = Manifest::fresh(4, Distance::Cosine);
        let mut bytes = m.encode().unwrap();
        // Flip a payload byte (after the 4-byte CRC) — the CRC must catch it.
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(Manifest::decode(&bytes).is_err());
    }

    #[test]
    fn decode_rejects_short_object() {
        assert!(Manifest::decode(&[0u8; 2]).is_err());
    }

    #[test]
    fn decode_rejects_unknown_format_version() {
        let mut m = Manifest::fresh(4, Distance::Cosine);
        m.format_version = 99;
        let bytes = m.encode().unwrap();
        let err = Manifest::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("format version"), "{err}");
    }

    #[test]
    fn validate_alias_accepts_a_fresh_pair() {
        let existing = BTreeMap::new();
        assert!(validate_alias("docs", "docs_v2", &existing).is_ok());
    }

    #[test]
    fn validate_alias_rejects_an_empty_name() {
        let existing = BTreeMap::new();
        let err = validate_alias("", "docs_v2", &existing)
            .unwrap_err()
            .to_string();
        assert!(err.contains("alias name must not be empty"), "{err}");
    }

    #[test]
    fn validate_alias_rejects_self_reference() {
        let existing = BTreeMap::new();
        let err = validate_alias("docs", "docs", &existing)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot point at itself"), "{err}");
    }

    #[test]
    fn validate_alias_rejects_a_chain() {
        let mut existing = BTreeMap::new();
        existing.insert("docs".to_string(), "docs_v2".to_string());
        let err = validate_alias("shortcut", "docs", &existing)
            .unwrap_err()
            .to_string();
        assert!(err.contains("is itself an alias"), "{err}");
        assert!(err.contains("never chained"), "{err}");
    }
}
