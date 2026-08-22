//! How a query ran (nidus-cvz): the path taken, rows scanned, candidate survival, and
//! per-phase timings. Opt-in via `SearchOpts::plan`/`HybridOpts::plan`, returned by the
//! `*_with_plan` sibling methods; `NIDUS_SLOW_QUERY_MS` logs the short form unconditionally.

use std::time::Duration;

use serde::Serialize;

/// How a query was answered.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QueryPlan {
    pub path: QueryPath,
    /// Rows fed to a brute-force scan. `None` on the ANN and segmented paths, where no
    /// full scan happens — the same split `metrics::search_vectors_scanned` keeps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows_scanned: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Candidates>,
    pub narrowing: Narrowing,
    pub timings: Timings,
}

/// Which branch of `Store::search` answered the query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryPath {
    Ann,
    AnnPrefilterFallback,
    Segmented,
    Quantized,
    Exact,
}

/// What an index walk surfaced vs what survived, broken down by why each was dropped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Candidates {
    pub surfaced: u64,
    pub survived: u64,
    pub dropped_out_of_scope: u64,
    pub dropped_stale: u64,
    pub dropped_filtered: u64,
    pub dropped_min_score: u64,
}

/// Whether the opt-in filter index narrowed the scan before it ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum Narrowing {
    /// No collection in scope declares a filter index.
    Inactive,
    /// An index exists but could not answer this filter, so the full scan ran.
    Declined,
    Narrowed {
        candidates: u64,
    },
}

/// Per-phase wall time, in **microseconds**; a phase that did not run is absent from the
/// wire form. `total` always runs, so it alone is not `Option`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Timings {
    #[serde(
        rename = "narrow_us",
        serialize_with = "us",
        skip_serializing_if = "Option::is_none"
    )]
    pub narrow: Option<Duration>,
    #[serde(
        rename = "gather_us",
        serialize_with = "us",
        skip_serializing_if = "Option::is_none"
    )]
    pub gather: Option<Duration>,
    #[serde(
        rename = "walk_us",
        serialize_with = "us",
        skip_serializing_if = "Option::is_none"
    )]
    pub walk: Option<Duration>,
    #[serde(
        rename = "resolve_us",
        serialize_with = "us",
        skip_serializing_if = "Option::is_none"
    )]
    pub resolve: Option<Duration>,
    #[serde(
        rename = "first_pass_us",
        serialize_with = "us",
        skip_serializing_if = "Option::is_none"
    )]
    pub first_pass: Option<Duration>,
    #[serde(
        rename = "rescore_us",
        serialize_with = "us",
        skip_serializing_if = "Option::is_none"
    )]
    pub rescore: Option<Duration>,
    #[serde(
        rename = "score_us",
        serialize_with = "us",
        skip_serializing_if = "Option::is_none"
    )]
    pub score: Option<Duration>,
    #[serde(rename = "total_us", serialize_with = "total_us")]
    pub total: Duration,
}

/// Serialize an optional phase duration as integer microseconds.
fn us<S: serde::Serializer>(d: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_u64(d.map_or(0, |d| d.as_micros() as u64))
}

/// Serialize the non-optional `total` duration as integer microseconds.
fn total_us<S: serde::Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_u64(d.as_micros() as u64)
}
