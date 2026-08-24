# D0001 — The tracker is beads, and its JSONL export stays untracked

**Status:** accepted · 2026-08-10
**Rule:** Never let `.beads/issues.jsonl` become tracked. The shared state is the Dolt ref.

## Why

GitHub Issues was retired wholesale on 2026-08-10. Every issue came across keeping its
number (GitHub `#186` is `nidus-186`) and the GitHub issues were closed pointing here, so
the closed history is still searchable and a rejected idea is still findable.

Issue state lives in a Dolt database under `.beads/`, which is local and gitignored, and is
shared by pushing to `refs/dolt/data` on `origin` — this repo, not a separate service. So
`bd dolt push` is as load-bearing as `git push`: without it your issue changes exist on your
machine only. What *is* tracked is the handful of config files a fresh clone needs to find
that database (`.beads/config.yaml`, `.beads/metadata.json`, `.beads/.gitignore`,
`.beads/hooks/`); everything else under `.beads/` is runtime.

The export is gitignored, with `export.git-add` off, because tracking it recreates a bug we
already shipped. The retired pre-migration tracker committed that file and rewrote it from
each branch's *local* database, so any branch could silently revert another branch's closes.
One did.

That failure shape generalises: a tracked artifact regenerated from local state, where each
writer overwrites the others and nothing looks wrong in the diff. It is the reason the docs
index in D0013 is not committed either.

## Evidence

- #83 — a branch reverting another branch's closes through the committed JSONL export.
- `.gitignore` — `.beads/` is deliberately not ignored wholesale; `.beads/.gitignore`
  governs what inside it is runtime.
