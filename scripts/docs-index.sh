#!/usr/bin/env bash
# Build the ranked docs+code index (nidus-3gm unit 11). Everything it writes lives in
# target/: gitignored, per-worktree, and `cargo clean`-able (D0013). Never a prerequisite
# for anything -- `bin/spec` falls back to its own text search when this has not been run.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
cd "$repo"
store="target/docs-index"
features="cli,memory,code"

# The digest that tells `spec find` whether the index still describes the tree. `git
# ls-files` names the whole tracked corpus the ingest below walks; the content hash comes
# from the WORKING TREE, because `-s` would report the staged blob and an unstaged edit
# would leave the index silently stale.
digest=$(git ls-files -z | xargs -0 shasum 2>/dev/null | shasum | cut -d' ' -f1)

if [ "${1:-}" = "--digest" ]; then echo "$digest"; exit 0; fi

# Same feature set the ingest call below needs, checked here so a missing `code` feature
# fails with this message instead of `code ingest` surfacing as a clap "unrecognized
# subcommand" error further down.
if ! cargo run --quiet --features "$features" --bin nidus -- --version >/dev/null 2>&1; then
  echo "docs-index: cannot build a nidus binary with --features $features (needs the" \
    "'code' feature: cargo build --features $features)" >&2
  exit 1
fi

echo "building the docs+code index → $store"
rm -rf "$store"

# ONE walk, dot-entries included so it reaches .claude/rules and decisions/ from the repo
# root; .git is always skipped regardless. Per-file dispatch chunks markdown by heading and
# source by AST -- what the old three-ingest-plus-staging-copy dance was faking by hand.
cargo run --quiet --features "$features" --bin nidus -- code ingest . \
  --dir "$store" --include-hidden --max-chars 1200 --overlap-chars 100 >/dev/null

# In the store, as a sentinel record: it survives a copy of the directory and is visible to
# every nidus query surface, which a sidecar file beside `manifest` would not be.
printf '[{"id":"docs-index.digest","attrs":{"digest":{"Str":"%s"}}}]' "$digest" \
  | cargo run --quiet --features "$features" --bin nidus -- upsert --dir "$store" meta >/dev/null

echo "docs-index: ready. \`spec find <words>\` now ranks; \`just docs-index\` to refresh."
