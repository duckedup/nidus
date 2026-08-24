# D0013 — Retrieval over the repo's own docs is derived, and never committed

**Status:** accepted
**Rule:** `bin/spec` over the tracked markdown is the retrieval floor and needs no setup. Any index built on top lives in `target/`, is gitignored, and is never a prerequisite.

## Why

SPEC.md is 2577 lines. `spec.workflow.js` told every research and implement agent to read it
whole, and a non-fork subagent also loads the entire CLAUDE.md hierarchy at startup, so a
five-agent fan-out paid for both five times to use one section of each.

`bin/spec` addresses the doc instead of loading it: `toc` for the index, `find <words>` for
which section covers a topic, `<ref>` to print one. It is pure text over tracked markdown —
no store, no embedder, no network — so it works in a bare clone, in CI, and in every
subagent. That property is the point: retrieval must never be something a teammate has to
set up before the rules are reachable.

A ranked index on top is optional, and if one is built it belongs in `target/`:

- Committing a nidus store recreates D0001's failure shape. It is binary, rewritten on every
  doc edit, unmergeable on conflict, and regenerated per branch from local state.
- `target/` is already gitignored, is per-worktree (each checkout has its own, which is why
  two sessions must never share one), and `cargo clean` removing it is correct.
- Not `docs/`: that path triggers the docs site deploy, so a decision record or index there
  would rebuild and publish the site on every edit. It is also why these records live in
  `decisions/` at the repo root.

The blocker on a zero-setup ranked index is real: every `EmbedProvider` is a network provider,
and `src/cli/ingest.rs` calls `require_embedder`, so `nidus ingest` fails on a clone with no
API key and no local ollama. `nidus text-search` needs no embedder, but nothing can populate a
store for it. Closing that gap is nidus-gmy.6.

## Evidence

- nidus-gmy.1 — `bin/spec` and the workflow rewire.
- #83 via D0001 — the tracked-derived-artifact failure this avoids.
- `.github/workflows/docs.yml` — `docs/**` triggers the site deploy.
