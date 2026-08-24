# D0012 — A feature ships whole, in one PR

**Status:** accepted
**Rule:** Core, HTTP, CLI, MCP, all three SDKs, docs. Do not split the SDKs into a follow-up and do not ask whether they are in scope: they are.

## Why

nidus has one capability surface, not six that drift. A change that adds a server route also
adds its `sdks/js`, `sdks/go` and `sdks/python` method, its MCP tool where the text-native
rule allows one, its CLI subcommand, and its entry in the HTTP/CLI/MCP reference.

A partial surface is the failure this exists to prevent, because the missing half is
invisible from inside the PR that shipped the first half.

If a leg genuinely cannot ship — a surface where the capability makes no sense — say which
and why in the PR body, rather than leaving it silently absent.

The SDKs need no separate version bump: each is released at `Cargo.toml`'s version by
`release.yml` calling its release workflow via `workflow_call` (see D0007). So an SDK-only fix
still ships by bumping `Cargo.toml`. Tag namespaces are distinct from the crate's `v*`:
`js-v*`, `py-v*`, and `sdks/go/v*` — that last form is forced, since Go resolves a module in
a repo subdirectory only via a `<subdir>/v<semver>` tag.
