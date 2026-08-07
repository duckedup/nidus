//! End-to-end tests that drive a **real `nidus serve` process** over a real socket.
//!
//! The suites in `src/server/mod.rs` call the axum `Router` in-process via
//! `tower::ServiceExt::oneshot`, which never binds a port. That covers handler logic
//! but not the parts a user actually depends on: `serve()`'s bind + graceful shutdown,
//! the CLI-flag → `ServeConfig` wiring, real HTTP framing, concurrency through the
//! `Arc<RwLock<Nidus>>` + `spawn_blocking` seam, and the binary as shipped. These tests
//! spawn that binary and talk to it with an HTTP client.
//!
//! One test binary holds every suite so they share [`harness`] rather than each
//! re-inventing process spawning (the cluster suite lands here too).
#![cfg(feature = "cli")]

mod harness;

mod cluster;
mod hardening;
#[cfg(feature = "mcp")]
mod mcp;
mod scale;
mod server;
