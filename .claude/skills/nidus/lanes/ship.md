## Ship

1. `nidus-check laws --strict`, always — it is cheap and the diff has moved since Implement.
   Lanes are NOT re-run wholesale here: the merged tree already passed them at Implement
   step 6, and Review step 7 re-ran whatever its fixes touched. Re-run only the lanes for
   files changed since the last green run; an unchanged green tree proves nothing twice.
   `ci`-class lanes (Miri) stay with the PR's required checks. Do not ship red.
2. **Version.** Every user-visible or behavioural change bumps `Cargo.toml` `version`
   (patch for fixes/refactors, minor for features, major for breaks) — `release.yml` cuts a
   release only when the `v<version>` tag is new, so an un-bumped PR silently ships nothing.
   If `major.minor` changed, update the `nidus = "M.m"` snippet in `README.md` and
   `docs/src/content/docs/getting-started.md`. Never touch `charts/nidus/Chart.yaml` or the
   SDK version files — CI stamps those and commits them back.
3. **State the close, then perform it.** A bundled PR carries one `Closes` line **per
   ticket**, each audited separately: it is normal for a bundle to close two tickets and only
   `Refs` a third. Put `Closes nidus-<n>` in the PR body — but that
   line no longer *does* anything, because GitHub cannot close a bead. Re-read every such
   line against the diff and downgrade it to `Refs nidus-<n>` if this change does not
   finish the issue. Then, once the PR merges, actually close it:
   `bd close nidus-<n> --reason "shipped in #<pr>"` followed by `bd dolt push`. A PR that
   merges with the bead left open is the failure this step exists to prevent.
4. **Commit.** Subject `🪺 <area>: <terse description>` — an emoji prefix and the area, no
   issue or PR number (the squash merge appends `(#<pr>)`; a second `(#<n>)` for the issue
   would be unreadable next to it). The issue ref belongs in the PR body.
   **Keep the body short or omit it**: of the last 25 commits on `main`, 4 have a body at all.
   GitHub's squash default concatenates the branch's commit messages into the merge message,
   so a long body becomes the text the maintainer has to edit at merge. Reasoning belongs in
   the PR body, which is durable and is not concatenated into anything. (Trailers your harness
   stamps on every commit are its business, not this repo's — this is about body length.)
   Counting `git log` yourself is right, but count **25+ top-level commits**: a small window
   over a squash-merge repo shows the sub-commits preserved *inside* one PR's squash message,
   which reads exactly like a convention and is not one.
5. **Push both.** `git pull --rebase` then `git push -u origin <branch>`, and `bd dolt push`
   for the issue state. Nothing tracker-related rides along in the commit, so a `git push`
   alone leaves every claim and close on your machine.
6. **Offer the PR** with `AskUserQuestion` — open it, or stop with the branch pushed. When
   opening: `gh pr create --assignee @me`, title = the commit subject verbatim, body = what
   changed and why plus `Closes nidus-<n>`. Then `gh pr merge --auto --squash`: the queue
   still retests, and the PR merges the moment checks go green instead of waiting for
   someone to come back and press the button. Print the URL.
   Auto-merge makes step 3's bead close easy to orphan — nobody is watching when it lands —
   so close it at your next touch of the repo rather than assuming someone saw it merge.

Ask before the commit. Never commit or push without the user choosing to.

