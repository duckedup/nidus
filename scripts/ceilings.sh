#!/usr/bin/env bash
# Bound the two claims about the lean tree that nothing was checking: the crate count
# Cargo.toml's comment names ("four crates plus the backends") and the shipped binary's
# size. `build-budget`/`build-budget-default` already enforce the build-TIME thesis
# (D0014/D0015); this is the same discipline for its SHAPE. Shared by `just ceilings` and
# the `ceilings` CI job so the two cannot drift (the scripts/e2e-services.sh pattern,
# .claude/rules/ci.md).
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
cd "$repo"
# shellcheck source=scripts/ceilings.env
source scripts/ceilings.env

# ── Lean dependency count ────────────────────────────────────────────────────
# Exact command committed in ceilings.env's comment: `--edges normal` drops
# build-dependencies, which is why this is 117 and D0015's 120 (`-e no-dev`) differs by
# three. Changing the command here without updating that comment is its own bug.
lean_crates=$(cargo tree -p nidus --no-default-features --edges normal --prefix none |
    sed 's/ (\*)$//' | awk '{print $1}' | sort -u | wc -l | tr -d ' ')

echo "lean dependency count: ${lean_crates} (ceiling: ${LEAN_CRATES_MAX})"

crates_over=false
if [ "$lean_crates" -gt "$LEAN_CRATES_MAX" ]; then
    echo "::error::lean dependency count is ${lean_crates}, over the ceiling of ${LEAN_CRATES_MAX} (D0005, D0015). A new dependency in the lean tree is a design change, not an implementation detail."
    crates_over=true
fi

# ── Shipped binary size ──────────────────────────────────────────────────────
# Strip a COPY, never the shipped artifact: adding `strip = true` to [profile.release]
# would change what actually ships, a release-behaviour change riding along on a CI
# ticket, which is exactly what this unit must not do.
cargo build --release --quiet
tmp_bin=$(mktemp)
trap 'rm -f "$tmp_bin"' EXIT
cp target/release/nidus "$tmp_bin"
strip "$tmp_bin"
# `wc -c` rather than `stat -c%s`/`stat -f%z`: one portable spelling instead of a
# Linux/macOS split.
binary_bytes=$(wc -c <"$tmp_bin" | tr -d ' ')

os=$(uname -s)
arch=$(uname -m)
echo "stripped release binary: ${binary_bytes} bytes on ${os}/${arch} (ceiling: ${BINARY_BYTES_MAX}, Linux x86_64 only)"

size_over=false
if [ "$os" = "Linux" ] && [ "$arch" = "x86_64" ]; then
    if [ "$binary_bytes" -gt "$BINARY_BYTES_MAX" ]; then
        echo "::error::stripped binary is ${binary_bytes} bytes, over the ceiling of ${BINARY_BYTES_MAX} (Linux x86_64). A new dependency likely grew the shipped binary; this is a design change, not an implementation detail."
        size_over=true
    fi
else
    echo "binary-size ceiling is asserted on Linux x86_64 only (CI's ubuntu-latest); ${os}/${arch} is reported here, not gated."
fi

if [ "$crates_over" = true ] || [ "$size_over" = true ]; then
    exit 1
fi
