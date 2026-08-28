## Optimize

Staff-level performance work: make nidus do the same thing faster or cheaper, and **prove it
with a number**. Build time and dependency weight count as performance here, because Core
Foundation §1 makes them CI-asserted budgets (D0014, D0015).

The target is a scope, not a ticket: a path (`src/store`), a subsystem in words ("the write
path"), or nothing at all, which means the whole repository.

**The two laws of this lane.** *Results do not move* — same recall, same ordering, same
tie-breaking, same errors; a change to any of those is a design change and needs an issue, not
a blueprint. And *an unmeasured win did not happen* — every unit names the benchmark that shows
it, and the number is recorded before and after. A refactor that "should be faster" and never
appeared in a benchmark is the failure mode this lane exists to prevent.

Off the table without an issue first: weakening the durability order (append vectors, fsync
data, append committing log records, fsync log), a second `allow(unsafe_code)` (D0006), a
dependency that costs build time (D0005), a non-additive on-disk format change.

1. **Preflight, then measure. You measure, not an agent.** SKILL.md's preflight has already
   run. Now record the control, on this tree, before anything changes:
   - `just bench-crit --save-baseline pre` — the criterion regression suite, and the one that
     later `bench-crit` runs diff against automatically;
   - whichever of `just bench`, `just bench-server`, `just bench-quant`, `just bench-ann`,
     `just bench-write` covers the scope, output kept in a `mktemp -d`;
   - `cargo build --timings` or `just deps` if the scope touches Cargo.toml or feature gates;
   - `benchmarks/baselines/` for what was already recorded, and `benchmarks/README.md` for what
     each harness actually measures.
   Run these **alone**. Nothing else may be building or testing on the machine, or the control
   is noise. This is why the sweep's agents are forbidden to run benchmarks: five of them
   timing a contended machine produce five confident wrong numbers.
2. **Sweep.** `Workflow({ scriptPath: ".claude/skills/nidus/sweep.workflow.js",
   args: { mode: "optimize", scope, baseline: "<the numbers from step 1>", perLens: 2 } })`
   Five opus lenses read in parallel — the per-query and per-row kernels, read/write
   amplification, allocations and copies, the library-to-server gap, and build cost — and each
   lens's top candidates are corroborated by a fresh sonnet agent that must answer *is this
   site actually hot, which existing benchmark shows it, has the compiler already done it, does
   it change results, what does it cost*. Pass `baseline` so every lens argues against real
   numbers instead of guessing. The default shape is 16 agents (5 opus + 10 sonnet + 1
   partition); `perLens: 1` halves the corroboration half, and narrowing `scope` is the
   cheaper lever than either. Tell the user to watch with `/workflows`.
3. **The candidate gate.** **Spec**'s scope gate, and its positional rule holds: **no
   `BLUEPRINT-*.md` exists on disk until it is answered.** Present the shortlist as
   *win / benchmark that shows it / what it costs in readability*, then one `AskUserQuestion`
   (four maximum) from `partition.scope_questions` plus the shortlist/deferred split. Two
   things to surface rather than decide: any candidate whose benchmark does not exist yet
   (writing one is real work and belongs in the plan or in a separate ticket), and any
   candidate that trades legibility for speed, which in a codebase this small is a product
   decision.
4. **File the work, both halves.** One bead for what is being taken now, then the sweep's
   `deferred` and `design_changes` as their own beads carrying sites, evidence and the measured
   or estimated win. A perf idea nobody filed is one somebody re-derives next quarter.
   Then branch per SKILL.md preflight step 4.
5. **You write the blueprints** — `lanes/spec.md` step 3 governs the format. Every sub-blueprint
   in this lane also carries, in its acceptance criteria:
   - **the benchmark command and the control number from step 1**, and the delta that would
     count as success. "Faster" is not an acceptance criterion; "p95 at n=100k, dim=768, from
     X ms to under Y ms, shown by `just bench-quant n=100000 dim=768`" is.
   - the invariant that results do not move, and which existing test proves it.
6. **The plan gate.** `lanes/spec.md` step 4, unchanged.
7. **Implement** — `lanes/implement.md`, unchanged. Its step 5 (fails-without-fix) has a direct
   analogue here that you owe at step 8, so read it as written and apply it there instead.
8. **Re-measure, alone, and compare.** Re-run exactly step 1's commands on the merged tree,
   with nothing else running, and put the before and after side by side.
   - `just bench-crit` prints the regression against the `pre` baseline for free — read every
     line of it, not just the ones you expected to move. A win in one kernel paid for by a loss
     in another is the normal shape of a bad optimization.
   - **Then revert the change and watch the number go back.** This is Implement step 5's
     discipline in its measurement form, and it is the only thing that separates a real win
     from build noise, a warm page cache, or a thermal accident. Once observed, say so; a
     speedup claimed but never reverted-and-remeasured is an unverified claim.
   - A unit whose number did not move is not done. Report it, do not narrate around it: the
     honest outcomes are "shipped, here is the delta", "no measurable win, reverted", and "the
     win was real but smaller than predicted, here is the actual number".
   - Correctness first, always: the lanes from `nidus-check lanes`, including `just ci-cli`,
     the SDK lanes and `just test-e2e`. A faster wrong answer is a bug.
9. **Review** — `lanes/review.md`, unchanged, with `issues` set to the bead from step 4. Its
   criteria pass will re-run your benchmark acceptance criteria from a context that did not
   write the code, which is exactly the check a performance claim needs. Then **Ship**
   (`lanes/ship.md`), with the measured before/after in the PR body, and record any lasting
   number in `benchmarks/baselines/` if the harness supports it (`just bench-write json=…`).
