#!/usr/bin/env bash
# Build the ranked docs index (nidus-gmy.7). Everything it writes lives in target/:
# gitignored, per-worktree, and `cargo clean`-able (D0013). Never a prerequisite for
# anything — `bin/spec` falls back to its own text search when this has not been run.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
cd "$repo"
store="target/docs-index"
staged="target/docs-index-src/root"

# The digest that tells `spec find` whether the index still describes the tree. `git ls-files`
# names the tracked corpus; the content hash comes from the WORKING TREE, because `-s` would
# report the staged blob and an unstaged doc edit would leave the index silently stale.
digest=$(git ls-files -z SPEC.md CLAUDE.md .claude/rules decisions \
  | xargs -0 shasum 2>/dev/null | shasum | cut -d' ' -f1)

if [ "${1:-}" = "--digest" ]; then echo "$digest"; exit 0; fi

echo "building the docs index → $store"
rm -rf "$store" "$staged"
mkdir -p "$staged"
# `ingest` walks directories, not files, so the two root docs are staged into one.
cp SPEC.md CLAUDE.md "$staged/"

bin=$(cargo run --quiet --features cli,memory --bin nidus -- --version >/dev/null 2>&1 && echo ok || echo no)
[ "$bin" = "ok" ] || { echo "docs-index: cannot build the nidus binary" >&2; exit 1; }

ingest() { # <root> <collection>
  cargo run --quiet --features cli,memory --bin nidus -- ingest "$1" \
    --collection "$2" --dir "$store" --glob '*.md' --strategy markdown \
    --max-chars 1200 --overlap-chars 100 \
    --fts-only nidus.text --fts-only nidus.source_path >/dev/null
}

ingest "$staged" root
ingest .claude/rules rules
ingest decisions decisions

# In the store, as a sentinel record: it survives a copy of the directory and is visible to
# every nidus query surface, which a sidecar file beside `manifest` would not be.
printf '[{"id":"docs-index.digest","attrs":{"digest":{"Str":"%s"}}}]' "$digest" \
  | cargo run --quiet --features cli,memory --bin nidus -- upsert --dir "$store" meta >/dev/null

echo "docs-index: ready. \`spec find <words>\` now ranks; \`just docs-index\` to refresh."
