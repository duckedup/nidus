//! The open-time knobs recorded once in the manifest (SPEC §14.2, nidus-141) so
//! `ann`/`quantization`/`query_threads`/`mmap` need not be repeated on every open.
//!
//! Every field is `Option`, but this is a **different** `None` than
//! [`crate::Config::ann`]'s: here `None` means "nothing recorded, fall back to the
//! built-in default," not "explicitly off." Recording an explicit off is out of
//! scope; clearing a recorded knob (going back to the built-in default) is a
//! profile-management operation the CLI provides, not a third state on this type
//! (so no `Option<Option<_>>`).

use serde::{Deserialize, Serialize};

use crate::model::{AnnConfig, Quantization};

/// Recorded open-time defaults for a store, carried in the manifest. An explicit
/// [`crate::Config`] setter always wins over a recorded value for the same knob.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenProfile {
    /// Recorded ANN default (`None` = no recorded default, i.e. exact search).
    pub ann: Option<AnnConfig>,
    /// Recorded quantization default (`None` = no recorded default, i.e. disabled).
    pub quantization: Option<Quantization>,
    /// Recorded query-thread count default (`None` = no recorded default).
    pub query_threads: Option<usize>,
    /// Recorded mmap default (`None` = no recorded default).
    pub mmap: Option<bool>,
}

impl OpenProfile {
    /// Overlay `newer`'s recorded knobs onto this profile, keeping anything `newer` leaves
    /// unrecorded. This is what makes `configure` accumulate: recording one knob must not
    /// erase knobs an earlier call recorded (nidus-141).
    pub fn overlay(&mut self, newer: &OpenProfile) {
        if newer.ann.is_some() {
            self.ann = newer.ann;
        }
        if newer.quantization.is_some() {
            self.quantization = newer.quantization;
        }
        if newer.query_threads.is_some() {
            self.query_threads = newer.query_threads;
        }
        if newer.mmap.is_some() {
            self.mmap = newer.mmap;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AnnConfig, Quantization};

    #[test]
    fn overlay_keeps_knobs_the_newer_profile_does_not_record() {
        let mut base = OpenProfile {
            ann: Some(AnnConfig::hnsw()),
            query_threads: Some(8),
            ..Default::default()
        };
        base.overlay(&OpenProfile {
            quantization: Some(Quantization::int8()),
            ..Default::default()
        });
        assert!(base.ann.is_some(), "an earlier ann must survive");
        assert_eq!(base.query_threads, Some(8));
        assert_eq!(base.quantization, Some(Quantization::int8()));
    }

    #[test]
    fn overlay_replaces_a_knob_the_newer_profile_does_record() {
        let mut base = OpenProfile {
            query_threads: Some(2),
            ..Default::default()
        };
        base.overlay(&OpenProfile {
            query_threads: Some(9),
            ..Default::default()
        });
        assert_eq!(base.query_threads, Some(9));
    }
}
