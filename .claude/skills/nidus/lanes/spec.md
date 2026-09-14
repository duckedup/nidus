## Spec

1. Run the research workflow. It fans out four fixed lenses (modules, tests, laws, prior art)
   and returns a proposed directory partition plus the scope forks it could not settle:
   `Workflow({ scriptPath: ".claude/skills/nidus/spec.workflow.js", args: { id, ask } })`
   Ask something before you launch it only if the ambiguity would send the *research* somewhere
   useless — the ticket names no surface at all, or two readings of it share no files.
   Otherwise research first: a question asked from the code is a better question than the same
   question asked from the title.
2. **Falsify the criteria you inherited, before the gate.** Take each acceptance criterion the
   ticket already states and ask what would have to be true for it to fail. A criterion that
   holds with and without the fix is not a weaker check, it is no check, and it is the one
   defect the rest of this pipeline cannot catch: it propagates into the blueprint, then into
   an implementation agent, then into the criteria pass that grades the work against it. The
   research above is what makes this answerable — nidus-g4h's "assert the reopened store
   returns the upserted rows through a filtered query" reads fine on the ticket and is
   unfalsifiable in the code, and only tracing every reader of the flag showed it. When one has
   no failing mode, carry it into the scope gate as a question with the criterion you would
   ship instead, and fix the ticket (`bd update <id> --description`) once the user agrees.
3. **The scope gate — ask before you write, not after.** `partition.scope_questions` is the
   seed. Drop any the ticket already answers, add anything the research surfaced that the
   partition missed, and put what is left in **one** `AskUserQuestion` (four maximum), each
   with concrete options and what each one adds to or drops from the change. Lead with the 2–3
   sentence understanding summary, so the answers land against your reading rather than theirs.

   The rule is positional, not advisory: **no `BLUEPRINT-*.md` exists on disk until these are
   answered.** Once a blueprint is written the question silently changes from "what should this
   be" to "is this wrong" — a worse question, asked later, against work already done. And a
   scope assumption is not local: it is baked into every sub-blueprint, so walking one back
   means rewriting all of them. If `scope_questions` comes back empty and you agree the ask is
   unambiguous, say so in one line and go to step 4 — an invented question is its own noise.
4. **You** write the blueprints from the research and the answers — do not delegate this. The
   gate the user approves must be yours.

   **Several tickets are ONE blueprint set and ONE PR, never one per ticket.** When the target
   names more than one issue, they share a branch, a root blueprint, a version bump and a PR
   whose body carries a `Closes` line each. Splitting them is the expensive default: N branches
   racing for one `Cargo.toml` version (of which the second to merge releases nothing, silently),
   N review cycles over one tree, and N chances to leave a bead open. `<id>` is then the tickets
   joined by `-` (`BLUEPRINT-nidus-hzi-61d.md`). One root blueprint covers all of them: one
   summary, one file list, one lane set, and a per-ticket section for its own root cause,
   acceptance criteria and tests. Split into separate PRs only when the user asks, or when one
   ticket is blocked and the rest should not wait for it — say which and why.

   A ticket that turns out to need **no code** still gets its section, recording what was
   measured and why it closes; it just contributes no files.
   - `BLUEPRINT-<id>.md` in **each directory** that will change.
   - `BLUEPRINT-<id>.md` at the **repo root**: summary, the table of sub-blueprints, complete
     file create/modify/remove list, group ordering and why, and the CI jobs that will cover
     it, from `nidus-check lanes` — a coverage map for the reader, not commands to run.
   - **Exception: never write one inside `docs/src/content/docs/`.** Starlight's `docsLoader()`
     schema-validates every `.md` under that root, so a blueprint there fails `just docs-build`
     with an error pointing at the blueprint. Put that slice's file at `docs/BLUEPRINT-<id>.md`.
   - Never name these `SPEC-*.md` — `SPEC.md` at the root is nidus's product spec.
   - Each sub-blueprint carries: context, files to modify/create/remove, concrete code
     patterns to mirror (path + line range + snippet, so the agent never re-explores), the
     test pattern for that area, acceptance criteria, the CI jobs that cover it, and a scope
     boundary naming the files it may NOT touch.
5. **The plan gate.** One `AskUserQuestion`: what you are about to build in 2–3 sentences, the
   unit list, and the file create/modify/remove count. Options: approve / refine (they edit,
   then re-ask) / reject (delete the blueprints, stop). Scope was settled at step 3 — do not
   re-ask it here. Carry a decision into this gate only if writing the blueprints surfaced a
   fork the research did not; small reversible details belong in the blueprint's open
   questions instead.

**Do not implement anything until the user picks approve.**

