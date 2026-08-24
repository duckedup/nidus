---
paths:
  - "src/cli/**"
  - "src/server/**"
  - "src/bin/**"
---

# The `cli` / `serve` / `mcp` feature gates

The binary (CLI plus `nidus serve`, an axum/tokio HTTP wrapper, SPEC §9) is gated behind the
**non-default `cli` feature**, so the core recipes (`just test`, `ci`, `lint`, Miri) build ONLY
the pure library and `cargo add nidus` keeps its seconds-long build (D0011).

```bash
just ci-cli        # fmt-check + clippy + test, all with --features cli
just test-cli      # cargo test --features cli
just build-cli     # release build of the nidus binary
just serve DIR DIM # cargo run --features cli -- serve --dir DIR --dim DIM
just install       # cargo install --path . --features cli
```

**When you touch these directories, gate it on the feature and verify with `just ci-cli`** —
the core `just ci` does not compile them. Do NOT move these deps into the default feature set
or use them from a library module: that breaks the pure `cargo add nidus` install, and
`nidus-check laws` checks for it.

The binary's deps (`clap`, `tokio`, `axum`, `tower`, `serde_json`, `tar`, `flate2` — all pure
Rust, zero FFI) compile only under `--features cli`. The AI ingest layer
(`embed`/`summarize`/`memory`/`mcp`, which add `reqwest` + `rmcp`) is likewise off by default.

**The binary adapts to the library, never the reverse.** Wire DTOs mirror `Hit` and `Footprint`
in `src/server/dto.rs`.

## MCP (`src/server/mcp/`)

`nidus serve` answers MCP `2026-07-28` at `/mcp` behind the `mcp` feature, folded into `serve`.
It is an *adapter*: every tool routes through the same `run_read`/`run_write` helpers the HTTP
handlers use, and the service is `nest_service`'d **inside** the middleware stack so it
inherits the body limit, backpressure, bearer auth, and metrics rather than reimplementing any
of them.

Two things there are load-bearing and easy to break:

- **Text-native surface** — no tool may take a raw `vector`, because a model cannot emit one.
  `tests/e2e/mcp/` asserts it. Resource content and prompt messages carry `{id, attrs}` too,
  never a vector.
- **Hand-written JSON schemas**, never `schemars`-derived, because the descriptions drive
  tool-selection quality.

Verify with `cargo clippy --all-targets --features mcp -- -D warnings` and `just test-e2e`.
