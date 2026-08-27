#!/usr/bin/env bash
# Measure how many single-bit archive corruptions `nidus restore` accepts silently.
#
# This is the measurement that found #152: before the fix, 114 of 204 sampled offsets
# (55.9%) restored with exit 0 and a correct-looking record/collection count, and 111
# of those produced byte-different vector data. After the fix the silent count must be 0.
#
# The deterministic regression tests live in src/cli/backup.rs and tests/e2e/cli.rs;
# this script is the population-level check those cannot express. Run it by hand when
# touching the archive format or the restore path — it is not wired into CI.
#
# Usage: scripts/backup-corruption-sweep.sh [stride]
set -euo pipefail

STRIDE="${1:-41}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/target/debug/nidus"
[ -x "$BIN" ] || { echo "build first: cargo build" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/src"

"$BIN" create --dir "$WORK/src" --dim 8 docs >/dev/null
python3 - <<'EOF' > "$WORK/seed.json"
import json, random
random.seed(7)
print(json.dumps([
    {"id": f"r{i}", "vector": [round(random.uniform(-1, 1), 4) for _ in range(8)], "attrs": {}}
    for i in range(200)
]))
EOF
"$BIN" upsert --dir "$WORK/src" docs < "$WORK/seed.json" >/dev/null
"$BIN" backup --dir "$WORK/src" -o "$WORK/good.tar.gz" >/dev/null

SIZE=$(wc -c < "$WORK/good.tar.gz")
silent=0; corrupt_data=0; caught=0; total=0

for ((off = 0; off < SIZE; off += STRIDE)); do
    total=$((total + 1))
    python3 -c "
import sys
b = bytearray(open('$WORK/good.tar.gz','rb').read())
b[$off] ^= 0x01
open('$WORK/bad.tar.gz','wb').write(b)
"
    rm -rf "$WORK/dst"; mkdir "$WORK/dst"
    if out=$("$BIN" restore -i "$WORK/bad.tar.gz" --dir "$WORK/dst" --yes 2>/dev/null); then
        recs=$(python3 -c "import json,sys; print(json.loads(sys.stdin.read())['records'])" <<<"$out")
        if [ "$recs" = "200" ]; then
            silent=$((silent + 1))
            cmp -s "$WORK/src/data" "$WORK/dst/data" || corrupt_data=$((corrupt_data + 1))
            continue
        fi
    fi
    caught=$((caught + 1))
done

echo "archive bytes:            $SIZE (stride $STRIDE)"
echo "offsets tested:           $total"
echo "SILENT (exit 0, n=200):   $silent"
echo "  with corrupted vectors: $corrupt_data"
echo "caught:                   $caught"

if [ "$silent" -ne 0 ]; then
    echo "FAIL: $silent corruptions restored silently (#152 regressed)" >&2
    exit 1
fi
echo "OK: every corruption was caught"
