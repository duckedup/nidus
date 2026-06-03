//! Engine adapters. nidus is always available; DuckDB and LanceDB are feature-gated
//! so the heavy deps they pull compile only when explicitly requested.

pub mod nidus;

#[cfg(feature = "duckdb")]
pub mod duckdb;

#[cfg(feature = "lancedb")]
pub mod lancedb;
