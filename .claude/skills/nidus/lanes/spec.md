## Spec

1. Run the research workflow. It fans out four fixed lenses (modules, tests, laws, prior art)
   and returns a proposed directory partition plus the scope forks it could not settle:
   `Workflow({ scriptPath: ".claude/skills/nidus/spec.workflow.js", args: { id, ask } })`
   Ask something before you launch it only if the ambiguity would send the *research* somewhere
   useless — the ticket names no surface at all, or two readings of it share no files.
   Otherwise research first: a question asked from the code is a better question than the same
   question asked from the title.
2. **The scope gate — ask before you write, not after.** `partition.scope_questions` is the
   seed. Drop any the ticket already answers, add anything the research surfaced that the
   partition missed, and put what is left in **one** `AskUserQuestion` (four maximum), each
   with concrete options and what each one adds to or drops from the change. Lead with the 2–3
   sentence understanding summary, so the answers land against your reading rather than theirs.

   The rule is positional, not advisory: **no `BLUEPRINT-*.md` exists on disk until these are
   answered.** Once a blueprint is written the question silently changes from "what should this
   be" to "is this wrong" — a worse question, asked later, against work already done. And a
   scope assumption is not local: it is baked into every sub-blueprint, so walking one back
   means rewriting all of them. If `scope_questions` comes back empty and you agree the ask is
   unambiguous, say so in one line and go to step 3 — an invented question is its own noise.
3. **You** write the blueprints from the research and the answers — do not delegate this. The
   gate the user approves must be yours.
   - `BLUEPRINT-<id>.md` in **each directory** that will change.
   - `BLUEPRINT-<id>.md` at the **repo root**: summary, the table of sub-blueprints, complete
     file create/modify/remove list, group ordering and why, and the global verification lanes
     from `nidus-check lanes`.
   - **Exception: never write one inside `docs/src/content/docs/`.** Starlight's `docsLoader()`
     schema-validates every `.md` under that root, so a blueprint there fails `just docs-build`
     with an error pointing at the blueprint. Put that slice's file at `docs/BLUEPRINT-<id>.md`.
   - Never name these `SPEC-*.md` — `SPEC.md` at the root is nidus's product spec.
   - Each sub-blueprint carries: context, files to modify/create/remove, concrete code
     patterns to mirror (path + line range + snippet, so the agent never re-explores), the
     test pattern for that area, acceptance criteria, its exact `verify` lanes, and a scope
     boundary naming the files it may NOT touch.
4. **The plan gate.** One `AskUserQuestion`: what you are about to build in 2–3 sentences, the
   unit list, and the file create/modify/remove count. Options: approve / refine (they edit,
   then re-ask) / reject (delete the blueprints, stop). Scope was settled at step 2 — do not
   re-ask it here. Carry a decision into this gate only if writing the blueprints surfaced a
   fork the research did not; small reversible details belong in the blueprint's open
   questions instead.

**Do not implement anything until the user picks approve.**

