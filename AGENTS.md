# Agent Instructions

This project tracks work in **GitHub Issues** on `duckedup/nidus`, via the `gh` CLI.

> **Architecture in one line:** issue state lives on GitHub, not in the repo, so a
> checkout carries no tracker data and there is nothing to sync, export, or import.
> `gh` is the only interface.
>
> This replaced an embedded beads/Dolt tracker whose exporter rewrote the whole issue
> file from each branch's local database — so any branch could silently revert
> another's closes, and one did (#83). It is fully retired: never reinstall it or its
> git hooks. The pre-migration issues remain in this repository's git history.

## Quick Reference

```bash
gh issue list --state open            # Find available work
gh issue view <n>                     # View issue details
gh issue edit <n> --add-assignee @me  # Claim work
gh issue close <n>                    # Complete work
gh issue create --title=… --body=…    # File new work
```

Priority is a `p0`–`p4` label; type is an `epic`/`bug`/`feature`/`task`/`decision`
label. A child of an epic names its parent in the body (`Part of #12`).

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

## Close the ticket in the PR that ships it

`Closes #<n>` belongs in the PR body that ships the work — never a later cleanup pass.
If the PR ships the fix, close it there; reopen if review changes the outcome.

The hazard runs the opposite way from the one that made this a law. The old tracker
hid finished-but-open tickets from every routine check; GitHub hides nothing, so that
direction nags on its own. What `Closes` adds is the reverse lie — it fires on merge
whether or not the work survived review, so a gutted PR still closes its issue. Both
are the tracker asserting what the tree does not support. Before opening a PR, re-read
every `Closes` line and confirm the diff actually finishes that issue; downgrade it to
`Refs #<n>` if it does not.

(Full rationale, with the incident that prompted it, is in `CLAUDE.md` §Conventions.)

## Rules

- Use GitHub Issues for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Durable knowledge goes in the issue that owns it, or in `SPEC.md` — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
   Issue state lives on GitHub, not in the repo, so nothing extra ships with the
   commit — but a `Closes #<n>` line only fires when the PR merges, so close
   anything the PR does not itself close.
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
