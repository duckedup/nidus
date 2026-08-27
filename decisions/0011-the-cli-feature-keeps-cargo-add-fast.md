# D0011 — The binary is gated behind the non-default `cli` feature

**Status:** accepted
**Superseded in part by [D0015](0015-the-default-build-ships-the-whole-binary.md):** the `cli`/`serve` deps are now in the default set. Everything else here still holds: the gates stay, the lean library build is still `--no-default-features`, and library modules still must not import binary-only crates.
**Rule:** Gate anything in `src/cli/`, `src/server/`, or `src/bin/` on the `cli` feature and verify with `just ci-cli`. Never move those deps into the default set, and never use them from a library module.

## Why

The crate ships an optional binary: the CLI plus `nidus serve`, an axum/tokio HTTP wrapper
(SPEC §9). It is gated exactly the way the benchmarks are a separate member, so the core
recipes (`just test`, `ci`, `lint`, Miri) build ONLY the pure library and `cargo add nidus`
keeps the seconds-long build path intact.

The binary's deps (`clap`, `tokio`, `axum`, `tower`, `serde_json`, `tar`, `flate2` — all pure
Rust, zero FFI) compile only under `--features cli`. The AI ingest layer
(`embed`/`summarize`/`memory`/`mcp`, which add `reqwest` + `rmcp`) is likewise off by default.
Using any of them from a library module breaks the pure install, which is why `nidus-check
laws` checks for it.

Direction of adaptation is fixed: the binary adapts to the library. Wire DTOs mirror `Hit`
and `Footprint` in `src/server/dto.rs`, never the reverse.

`nidus serve` also answers MCP `2026-07-28` at `/mcp` behind the `mcp` feature, folded into
`serve`. `src/server/mcp/` is an adapter: every tool routes through the same
`run_read`/`run_write` helpers the HTTP handlers use, and the service is `nest_service`'d
*inside* the middleware stack so it inherits the body limit, backpressure, bearer auth, and
metrics rather than reimplementing them.

Two things there are load-bearing and easy to break. The tool surface is **text-native**: no
tool may take a raw `vector`, because a model cannot emit one, and `tests/e2e/mcp/` asserts
it. Tool schemas are **hand-written JSON**, never `schemars`-derived, because the descriptions
drive tool-selection quality. The same holds for resources and prompts: resource content and
prompt messages carry `{id, attrs}`, never a vector.

## Evidence

- `lib/laws.mjs` — `featureGating` and `modGating`.
- `cargo binstall nidus` fetches prebuilt binaries via `[package.metadata.binstall]`;
  `cargo install nidus --features cli` builds from source.
