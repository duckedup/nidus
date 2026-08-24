---
paths:
  - "Cargo.toml"
  - "charts/**"
  - "sdks/**"
  - "docs/**"
  - "README.md"
---

# Releases, versions, SDKs, and the docs site

**Bump `Cargo.toml` `version` in every PR** with a user-visible or behavioural change (semver:
patch for fixes/refactors, minor for new features, major for breaking API). `release.yml` runs
on push to `main`, reads `version`, and releases only if that tag does not already exist — so a
PR that does not bump ships **nothing**, silently. Pure-internal no-op churn is the only
exception (D0007).

**Do NOT hand-edit the derived version files.** `charts/nidus/Chart.yaml` (`version` and
`appVersion`), `sdks/js/package.json`, and `sdks/python/src/nidus/_version.py` are stamped from
`Cargo.toml` at release time. The Go SDK has no version file — a Go module's version *is* its
tag. Tag namespaces: `js-v*`, `py-v*`, `sdks/go/v*` (that last form is forced, since Go
resolves a subdirectory module only via `<subdir>/v<semver>`).

**Known caveat: the stamp-back has never landed.** `main`'s ruleset rejects the bot push, so
those files on `main` lag what was really published — but only the repo is wrong, since the
stamp is applied *before* publish. Do not "fix" it by hand: a hand-stamp asserts a version that
may never have been published (D0007).

**Hand-maintained, and checked:** on a `major.minor` bump, update the install snippet in BOTH
`README.md` and `docs/src/content/docs/getting-started.md` to match (e.g. `nidus = "0.3"`). A
patch bump leaves `nidus = "0.12"` correct.

**A feature ships whole, in one PR** — core, HTTP, CLI, MCP, all three SDKs, docs. A change
that adds a server route also adds its `sdks/js`, `sdks/go` and `sdks/python` method, its MCP
tool where the text-native rule allows one, its CLI subcommand, and its HTTP/CLI/MCP reference
entry. If a leg genuinely cannot ship, say which and why in the PR body rather than leaving it
silently absent (D0012).

## Docs site

`docs/` is an Astro + Starlight site (`just docs` / `docs-build` / `docs-preview`), deployed to
**nidus.duckedup.org** by `.github/workflows/docs.yml` on push to `main`.

- **NO em dashes in user-facing prose** (docs site, README, SDK READMEs). Reword with a period,
  comma, colon, or parentheses. A sweep removed them all; do not reintroduce them. En dashes in
  numeric ranges are fine.
- **Positioning:** nidus is a vector store **for development and small-scale use**. Keep the
  framing open — do NOT pin it to "an embeddable library" (or "a library, not a server") and do
  NOT promise future modes ("server planned / on the roadmap"). Describe what it does today.
