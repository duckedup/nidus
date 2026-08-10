# Agent Instructions

This project tracks work in **beads**, via the `bd` CLI. GitHub Issues is retired as of
2026-08-10 — do not file there.

> **Architecture in one line:** issue state lives in a Dolt database under `.beads/`,
> which is local and gitignored, and is shared by pushing it to `refs/dolt/data` on
> this repo's own `origin`. So `bd dolt push` is as load-bearing as `git push` — your
> commit does not carry your issue changes.
>
> A fresh clone runs **`bd bootstrap`** to pull the database down (needs the `dolt` CLI:
> `brew install dolt`). A git worktree needs nothing — it shares the main clone's
> database automatically.
>
> An earlier version of this tracker committed its JSONL export and rewrote it from each
> branch's local database, so any branch could silently revert another's closes, and one
> did (#83). That is why `.beads/issues.jsonl` is gitignored and `export.git-add` is off:
> the shared state is the Dolt ref, never a tracked file.

## Quick Reference

```bash
bd ready                              # Find available work (open, nothing blocking it)
bd show nidus-186                     # View issue details
bd update nidus-186 --claim           # Claim work
bd close nidus-186 --reason "…"       # Complete work
bd create "title" -t task -p 2        # File new work
bd dolt push                          # Publish issue changes (do NOT skip)
```

Issues kept their GitHub numbers through the migration: `#186` is `nidus-186`. Priority
is a first-class field (`bd priority`) and also a `p0`–`p4` label; type is
`epic`/`bug`/`feature`/`task`/`decision`. Epics use dependency links —
`bd dep add <child> <parent> --type parent-child`, then `bd children <epic>`.

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

## Close the ticket yourself when the PR merges

**Nothing auto-closes any more.** A `Closes nidus-186` line in a PR body is
documentation only — GitHub cannot close a bead, so a trailer that used to do the work
now merely looks like it did. Closing is `bd close nidus-186 --reason "…"` followed by
`bd dolt push`, as part of shipping.

This law has been broken in both directions and they are the same failure. The old
tracker hid finished-but-open tickets from every routine check; GitHub's auto-close then
introduced the reverse lie, firing on merge whether or not the work survived review, so
a gutted PR still closed its issue. Both are the tracker asserting what the tree does not
support. The manual close makes the first direction the live risk again: before you
close, confirm the merged diff actually finishes that issue.

(Full rationale, with the incident that prompted it, is in `CLAUDE.md` §Conventions.)

## Rules

- Use beads for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Durable knowledge goes in the issue that owns it, or in `SPEC.md` — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   bd dolt push                 # issue state — a separate ref, NOT carried by git push
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
   Issue state is not in your commit, so `git push` does not carry it: closing a ticket
   and pushing the branch still leaves the close on your machine. Push both.
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
