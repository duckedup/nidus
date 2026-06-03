//! Engine adapters. nidus is always available; DuckDB and LanceDB are feature-gated
//! so the heavy deps they pull compile only when explicitly requested.

pub mod nidus;

#[cfg(feature = "duckdb")]
pub mod duckdb;

// The `lancedb` module (and its `#[cfg(feature = "lancedb")] pub mod` declaration) is
// added by its adapter task, together with the source file — rustfmt follows `mod` to a
// file, so the declaration and the file must land together.
