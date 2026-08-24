# D0007 — Chart and SDK versions are stamped from `Cargo.toml`, never hand-edited

**Status:** accepted, with a known unlanded half
**Rule:** Bump `Cargo.toml` only. Leave `charts/nidus/Chart.yaml`, `sdks/js/package.json`, and `sdks/python/src/nidus/_version.py` alone.

## Why

Releases are automatic: `release.yml` runs on push to `main`, reads `version` from
`Cargo.toml`, and releases only if that tag does not already exist. So a PR that does not
bump `version` ships nothing, silently. Every PR with a user-visible or behavioural change
bumps it; pure-internal no-op churn is the only exception.

Every downstream version string is derived from that one. The chart's `version` and
`appVersion` are stamped by `helm-publish.yml`; the SDKs ship at the crate's version so
"which nidus does this client speak to" is answerable from the version alone. The Go SDK has
no version file at all, since a Go module's version *is* its tag. This used to be a
hand-edit enforced by a CI assertion, which fired on essentially every PR — that is why it
is derived now. A PR that bumps the crate and also edits the chart just creates a conflict
with the bot commit.

**The known caveat: the stamp-back has never actually landed.** `main`'s ruleset rejects the
`github-actions[bot]` push with `GH013`, so those three files on `main` lag what was really
published. Only the repo is wrong — the stamp is applied *before* publish, so everything
that did publish (PyPI, the OCI chart) got the correct version. npm is the exception, for an
unrelated reason: #81 has blocked every JS publish since 0.2.0, so that job fails at the
publish step and never reaches stamp-back at all.

Do not "fix" the drift by hand. A hand-stamp asserts a version that may never have been
published, and `sdks/js/package.json` is exactly that case: stamping it to the crate version
would claim an npm release that does not exist. The mechanism is right; it needs a one-time
bypass on `main`'s ruleset. Until then `scripts/stamp-back.sh` makes each failure loud in the
release run instead of silently green, and `STAMP_BACK_STRICT=1` turns it into a hard job
failure once the bypass is in place.

One thing is still hand-maintained: on a `major.minor` bump, the install snippets in
`README.md` and `docs/src/content/docs/getting-started.md` must match. A patch bump leaves
`nidus = "0.12"` correct. `nidus-check laws` checks this one.

## Evidence

- #82 — the stamp-back push rejected by the ruleset.
- #81 — the npm publish blocked since 0.2.0.
- nidus-yap — the release job committing the stamp back.
