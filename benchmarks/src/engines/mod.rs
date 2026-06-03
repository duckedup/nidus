//! Engine adapters. nidus is always available; DuckDB and LanceDB are feature-gated
//! so the heavy deps they pull compile only when explicitly requested.

pub mod nidus;

// The `duckdb` and `lancedb` modules (and their `#[cfg(feature = ...)] pub mod`
// declarations) are added by their adapter tasks, together with the source files —
// rustfmt follows `mod` to a file, so the declaration and the file must land together.
