//! End-to-end tests that drive a **real `nidus serve` process** over a real socket.
#![cfg(feature = "cli")]

mod cli;
mod harness;

mod aliases;
mod cluster;
#[cfg(all(feature = "memory", feature = "code"))]
mod code;
#[cfg(all(feature = "memory", feature = "code"))]
mod docs_index;
mod env_flags;
mod hardening;
#[cfg(feature = "embed-ollama")]
mod ingest;
#[cfg(feature = "memory")]
mod ingest_fts;
#[cfg(feature = "mcp")]
mod mcp;
#[cfg(feature = "mcp")]
mod memory_http;
mod plan;
mod profile;
#[cfg(all(feature = "mcp", feature = "embed-ollama", feature = "rerank-cohere"))]
mod rerank;
mod scale;
#[cfg(all(feature = "mcp", feature = "embed-ollama"))]
mod serve_dim;
mod server;
mod tune;
