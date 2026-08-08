//! MCP over stdio for a client that spawns its own server (`nidus mcp --dir …`). Hands the
//! SAME [`NidusMcp`] the HTTP transport uses to rmcp's stdio transport — no second handler.
//! Deliberately without `limits.rs`/`metrics.rs`: both are axum `.layer()`-only.

use anyhow::Context;
use rmcp::ServiceExt;
use rmcp::transport::io::stdio;

use super::AppState;
use super::NidusMcp;

pub(super) async fn serve(state: AppState) -> anyhow::Result<()> {
    NidusMcp::new(state)
        .serve(stdio())
        .await
        .context("starting MCP over stdio")?
        .waiting()
        .await
        .context("serving MCP over stdio")?;
    Ok(())
}
