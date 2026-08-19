//! End-to-end tests that drive a **real `nidus serve` process** over a real socket.
#![cfg(feature = "cli")]

mod cli;
mod harness;

mod cluster;
mod env_flags;
mod hardening;
#[cfg(feature = "mcp")]
mod mcp;
#[cfg(feature = "mcp")]
mod memory_http;
mod profile;
#[cfg(all(feature = "mcp", feature = "embed-ollama", feature = "rerank-cohere"))]
mod rerank;
mod scale;
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
mod serve_dim;
mod server;
