#!/usr/bin/env bash
# Persist a release version stamp back to main.
#
# One source of truth for sdk-js-release.yml, sdk-py-release.yml and helm-publish.yml,
# which each carried their own copy of this logic — and each carried the same bug
# (see nidus #82): a blanket `continue-on-error: true` that swallowed a PERMANENT
# misconfiguration as if it were the transient race the flag was written for.
#
# Two failure modes, deliberately distinguished:
#
#   * a race with a concurrent push to main  -> retry, then warn. Genuinely transient.
#   * branch protection rejecting the bot    -> loud error annotation + step summary.
#     Never succeeds on retry, so retrying is pointless and silence is harmful: the
#     package IS published, and it is the repository that ends up lying about it.
#
# Exit status is 0 in both cases by default: the artifact is already published by the
# time this runs, so failing here would misreport a release that actually succeeded.
# Set STAMP_BACK_STRICT=1 to exit non-zero on a protection rejection instead — worth
# turning on once the ruleset grants the Actions app a bypass, so a silent regression
# back to the #82 state fails loudly.
#
# Usage: scripts/stamp-back.sh <label> <version> <file>...
#   label    short name for messages, e.g. "sdk-js", "sdk-py", "chart"
#   version  the released version, used in the commit subject
#   file...  the stamped paths to commit

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <label> <version> <file>..." >&2
    exit 2
fi

label=$1
version=$2
shift 2
files=("$@")

readonly MAX_ATTEMPTS=3

# GitHub's rejection vocabulary for "a rule says you may not push here". GH006 is the
# legacy protected-branch code, GH013 the rulesets one; the prose lines are matched too
# because the codes are not guaranteed to appear on every rejection path.
readonly PROTECTION_RE='GH006|GH013|protected branch|Repository rule violations|required status check|Changes must be made through a pull request'

# A rebase content conflict is NOT the race the retry loop exists for: `rebase --abort`
# restores the same pre-conflict state, so every later attempt reproduces it identically.
readonly CONFLICT_RE='CONFLICT|could not apply|Automatic merge failed|Resolve all conflicts'

note() { echo "stamp-back($label): $*"; }

summary() {
    [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] || return 0
    cat >>"$GITHUB_STEP_SUMMARY"
}

if git diff --quiet -- "${files[@]}"; then
    note "already at $version — nothing to commit"
    exit 0
fi

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -- "${files[@]}"

# [skip ci] because this commit is a derived stamp, not a change worth re-testing.
git commit -m "🪺 $label: stamp version $version [skip ci]"

push_output=""
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    if push_output=$(
        {
            git pull --rebase origin main &&
                git push origin HEAD:main
        } 2>&1
    ); then
        note "stamped $version onto main (attempt $attempt)"
        exit 0
    fi

    echo "$push_output"

    if grep -qEi "$PROTECTION_RE" <<<"$push_output" || grep -qEi "$CONFLICT_RE" <<<"$push_output"; then
        git rebase --abort 2>/dev/null || true
        break
    fi

    # A failed `pull --rebase` can stop mid-rebase; without this every later attempt
    # dies on "rebase in progress" and the retry loop is decoration.
    git rebase --abort 2>/dev/null || true

    if [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; then
        note "push failed (attempt $attempt/$MAX_ATTEMPTS), retrying — looks like a race"
        sleep $((attempt * 3))
    else
        note "push failed (attempt $attempt/$MAX_ATTEMPTS), giving up"
    fi
done

printf -v file_list '`%s`, ' "${files[@]}"
file_list=${file_list%, }

if grep -qEi "$PROTECTION_RE" <<<"$push_output"; then
    echo "::error title=Stamp-back rejected by branch protection::main still holds a stale $label version. Published $version, but ${files[*]} was not updated. This is a repo-settings problem, not a race — it will fail identically on every release until fixed."

    summary <<EOF
### ⚠️ Stamp-back to \`main\` rejected — $label

**Published \`$version\` successfully.** The repository was *not* updated to match.

$file_list on \`main\` now disagrees with what shipped. A clone will report a stale
version. Nothing about the published artifact is wrong.

Branch protection on \`main\` rejected the \`github-actions[bot]\` push. This is
permanent — it fails identically on every release. Retrying will not help.

**To fix (one of):**
1. Grant the GitHub Actions app a bypass on \`main\`'s ruleset. Preserves the
   derive-and-stamp design; these commits already carry \`[skip ci]\`, so no loop.
2. Have this step open a stamp PR with auto-merge instead of pushing directly.
3. Drop stamp-back and treat the manifests as non-authoritative placeholders,
   documenting that the tag is the real version.

Once fixed, set \`STAMP_BACK_STRICT=1\` so a regression fails the job instead of
warning.

<sub>See nidus #82.</sub>
EOF

    if [[ "${STAMP_BACK_STRICT:-}" == "1" ]]; then
        note "STAMP_BACK_STRICT=1 — failing the job"
        exit 1
    fi
    exit 0
fi

if grep -qEi "$CONFLICT_RE" <<<"$push_output"; then
    echo "::error title=Stamp-back hit a rebase conflict::main still holds a stale $label version. Published $version, but ${files[*]} conflicts with main. Retrying cannot resolve this — re-run the job, or stamp by hand."

    summary <<EOF
### ⚠️ Stamp-back to \`main\` conflicted — $label

**Published \`$version\` successfully.** The repository was *not* updated to match.

$file_list conflicts with \`main\`, so the rebase could not replay the stamp commit.
The published artifact is correct; only the repo is stale.

Unlike a push race, **retrying in-place cannot fix this** — \`rebase --abort\` restores
the same state, so every attempt reproduces the same conflict. Most likely two releases
overlapped and both stamped the same file. Re-running the whole job (fresh checkout of
the now-updated \`main\`) usually clears it; otherwise stamp the file by hand.
EOF

    if [[ "${STAMP_BACK_STRICT:-}" == "1" ]]; then
        note "STAMP_BACK_STRICT=1 — failing the job"
        exit 1
    fi
    exit 0
fi

echo "::warning title=Stamp-back failed::$label $version published, but ${files[*]} could not be pushed to main after $MAX_ATTEMPTS attempts. main holds a stale version."

summary <<EOF
### Stamp-back to \`main\` failed — $label

Published \`$version\`, but could not push the stamp after $MAX_ATTEMPTS attempts.
$file_list on \`main\` is stale. The published artifact is correct.

This matched neither a branch-protection rejection nor a rebase conflict, so it is most
likely a transient race with concurrent pushes to \`main\`. Re-running the job should
clear it.
EOF

exit 0
