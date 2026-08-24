# D0002 — A fresh clone runs `just bd-setup`, never `bd bootstrap` or `bd init`

**Status:** accepted
**Rule:** `just bd-setup` in a fresh clone. Never `bd bootstrap`, never `bd init`. A worktree needs neither.

## Why

`scripts/bd-setup.sh` reads the tracked config, recovers the database from `refs/dolt/data`,
and wires the remote. It needs the `dolt` CLI on PATH (`brew install dolt`); `bd` alone can
push but cannot clone. It is safe to re-run: an existing database is left untouched, because
it may hold work that was never pushed.

`bd bootstrap` cannot do this job. It reads the tracked `sync.remote`, rejects its
`git+ssh://` form as "not a Dolt remote", and so never reaches the repo's own
`refs/dolt/data` branch. The result is a fresh clone with an **empty** tracker and no error
naming the cause. That silence is the danger, not the failure: an empty database reads as
divergent history, and the recovery `bd dolt pull` then offers is `bd dolt push --force`,
which would force-push nothing over everyone's issues. If you ever see that prompt, stop.

`bd init` is worse: it mints a new identity and can force-push over everyone else's history
(`bd help init-safety`).

A git worktree needs none of this. `bd` finds the main clone's database through the shared
git common directory, so peers under `.claude/worktrees/` read and write one database and
see each other's claims and closes immediately. `bd dolt push`/`pull` is for crossing clones
and machines, not for peers on this one.

## Evidence

- nidus-1oq — bootstrap leaving a fresh clone with an empty tracker and no error.
- `just bd-sync` is pull-then-push, for finishing a session.
