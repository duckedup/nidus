//! Engine adapters. nidus is always available; DuckDB and LanceDB are feature-gated
//! so the heavy deps they pull compile only when explicitly requested.

pub mod nidus;

/// The same nidus over HTTP — pair it with [`nidus`] to see what `serve` costs.
#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "duckdb")]
pub mod duckdb;

#[cfg(feature = "lancedb")]
pub mod lancedb;
