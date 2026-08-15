#!/usr/bin/env bash
# Set up the beads tracker in a fresh clone: recover the issue database from
# the repo's own `refs/dolt/data` ref, then wire the remote for push/pull.
#
# Called by `just bd-setup`. This does the job `bd bootstrap` advertises but
# cannot do here (nidus-1oq): bootstrap reads the tracked `sync.remote`, rejects
# its `git+ssh://` form as "not a Dolt remote", and so never reaches its own
# refs/dolt/data branch — leaving a fresh clone with an EMPTY tracker and no
# error naming the cause. An empty database then reads as divergent history, and
# the recovery `bd dolt pull` offers is `bd dolt push --force`, which would
# force-push nothing over everyone's issues.
#
# The ref holds a raw Dolt NBS store (a `manifest` plus `.darc` table files),
# which is exactly the layout of a `file://` Dolt remote — so it can be cloned
# directly once extracted.
set -euo pipefail

die() { echo "bd-setup: $*" >&2; exit 1; }

command -v git >/dev/null || die "git is required"
command -v bd >/dev/null || die "bd is required: https://github.com/steveyegge/beads"
command -v dolt >/dev/null || die "dolt CLI is required to clone the database: brew install dolt"

ROOT=$(git rev-parse --show-toplevel) || die "not inside a git repository"
cd "$ROOT"

[ -f .beads/metadata.json ] || die ".beads/metadata.json missing — is this the nidus repo?"

# Both come from the git-tracked config, so a fresh clone already knows them.
DB=$(sed -n 's/.*"dolt_database"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' .beads/metadata.json)
DB=${DB:-nidus}

# bd REWRITES this file on first use: the flat `sync.remote:` key gets commented
# out and reappears as a nested `sync:`/`remote:` block. Accept both spellings or
# a re-run reads no URL at all and silently rebuilds a mangled one.
REMOTE=$(sed -n \
  -e 's/^[[:space:]]*sync\.remote:[[:space:]]*"\{0,1\}\([^"[:space:]]*\)"\{0,1\}.*/\1/p' \
  -e 's/^[[:space:]][[:space:]]*remote:[[:space:]]*"\{0,1\}\([^"[:space:]]*\)"\{0,1\}.*/\1/p' \
  .beads/config.yaml 2>/dev/null | head -1)
if [ -z "$REMOTE" ]; then
  origin=$(git remote get-url origin) || die "no git origin and no sync.remote to fall back on"
  REMOTE=$(printf '%s' "$origin" \
    | sed -e 's|^git@\([^:]*\):|git+ssh://git@\1/|' -e 's|^https://|git+ssh://git@|')
fi
case "$REMOTE" in
  *://*) ;;
  *) die "could not determine the Dolt remote URL (got '$REMOTE')" ;;
esac

TARGET=".beads/embeddeddolt/$DB"

# Never clobber an existing database: it may hold work that was never pushed,
# and this script's job is the fresh-clone path, not repair. A present-but-empty
# database is the divergent case (nidus-1oq) and needs a human, so say so rather
# than guessing.
if [ -d "$TARGET/.dolt" ]; then
  echo "bd-setup: database already present at $TARGET — leaving it alone."
  configure_only=1
else
  configure_only=0
  echo "bd-setup: no database yet; recovering from refs/dolt/data ..."

  git ls-remote --exit-code origin refs/dolt/data >/dev/null 2>&1 \
    || die "origin has no refs/dolt/data — nothing to recover from"

  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT

  git fetch -f -q origin 'refs/dolt/data:refs/dolt/data'
  mkdir -p "$TMP/remote" "$TMP/clone"
  git archive refs/dolt/data | tar -x -C "$TMP/remote"
  [ -f "$TMP/remote/manifest" ] || die "refs/dolt/data is not a Dolt store (no manifest)"

  ( cd "$TMP/clone" && dolt clone "file://$TMP/remote" "$DB" >/dev/null 2>&1 ) \
    || die "dolt clone from the extracted ref failed"

  mkdir -p .beads/embeddeddolt
  cp -R "$TMP/clone/$DB" "$TARGET"
fi

git config beads.role maintainer
chmod 700 .beads

# The clone inherits the temp `file://` path as its origin; leaving that in place
# would silently push issue state into a deleted temp dir instead of GitHub.
bd dolt remote remove origin >/dev/null 2>&1 || true
bd dolt remote add origin "$REMOTE" --allow-git-origin >/dev/null

if [ "$configure_only" = 0 ]; then
  bd dolt pull >/dev/null 2>&1 || die "recovered database does not agree with the remote — do NOT 'bd dolt push --force'; ask for help"
fi

echo "bd-setup: ready — $(bd stats 2>/dev/null | sed -n 's/.*Total Issues:[[:space:]]*//p') issues, remote $REMOTE"
