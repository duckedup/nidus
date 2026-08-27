---
paths:
  - "src/cli/**"
  - "src/server/**"
  - "src/bin/**"
---

# The `cli` / `serve` / `mcp` feature gates

The binary (CLI plus `nidus serve`, an axum/tokio HTTP wrapper, SPEC §9) is gated behind the
`cli` feature, part of `default = ["serve"]` (D0015): a bare `cargo install nidus` ships the
whole binary. `--no-default-features` remains the supported lean library build, and the core
recipes (`just test`, `ci`, `lint`, Miri) build ONLY that lean library, still reached only via
`--no-default-features` (D0011, superseded in part by D0015).

```bash
just ci-cli        # fmt-check + clippy + test, --no-default-features --features cli
just test-cli      # cargo test --no-default-features --features cli
just build-cli     # release build of the nidus binary, cli slice only
just serve DIR DIM # cargo run --no-default-features --features cli -- serve --dir DIR --dim DIM
just install       # cargo install --path . --features serve (redundant now, kept as intent)
```

**When you touch these directories, gate it on the feature and verify with `just ci-cli`** —
the core `just ci` does not compile them. The gates themselves are unchanged and still matter:
do NOT use these deps from a library module, because `--no-default-features` must still yield
the lean tree, and `nidus-check laws` checks for it. **Any lane that means a narrower slice
than the default must pass `--no-default-features` explicitly** — `--features X` alone is
additive to the default set, not a replacement, so a flag meant to isolate `cli` (or `code`,
`embed-all`, …) silently re-tests the full default build instead if it omits that flag.

The binary's deps (`clap`, `tokio`, `axum`, `tower`, `serde_json`, `tar`, `flate2` — all pure
Rust, zero FFI) sit behind `feature = "cli"`, part of the default build. The AI ingest layer
(`embed`/`summarize`/`memory`/`mcp`, which add `reqwest` + `rmcp`) is gated the same way, also
in by default; `--no-default-features` is what excludes all of it.

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
