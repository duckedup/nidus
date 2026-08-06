//! End-to-end tests that drive a **real `nidus serve` process** over a real socket.
#![cfg(feature = "cli")]

mod harness;

mod cluster;
mod hardening;
#[cfg(feature = "mcp")]
mod mcp;
mod scale;
mod server;
