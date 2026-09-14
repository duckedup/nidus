# D0017 — SQL is syntax over §7, not an engine

**Status:** accepted
**Rule:** a `SELECT` statement (`SPEC.md` §7.12) is a compiled front end over the existing
`search`/`text_search`/`hybrid_search`/`list`/`aggregate` entry points, hand-rolled and
dependency-free in the lean core. A planner, joins, DML (`INSERT`/`UPDATE`/`DELETE`),
subqueries, or transactions are a design change and need an issue first, exactly as any of
`SPEC.md` §2's other non-goals would.

## Why

nidus-yq9p.2/.3/.6 asked for "SQL-shaped read syntax" without deciding two things up front:
whether adding SQL means adding a query engine (a planner, a second execution path, a real
dependency), and how to parse it without paying the build-and-ship cost the whole project
exists to avoid (SPEC §1, D0005). Both were settled at the scope gate, and this record is
why, so the next person proposing "just add `sqlparser`" or "let's support joins while we're
in here" has the argument rather than re-deriving it.

**SQL as syntax, not an engine.** `src/model.rs`, `src/store/`, and `src/plan.rs` are
untouched by this feature. A `SELECT` lexes, parses, and compiles to the same typed
`SearchOpts`/`HybridOpts`/`ListOpts`/`AggregateOpts`/`Filter`/`FtsQuery` values a typed caller
already builds by hand, then runs through the same `Store` methods those callers already
call. There is no second execution path to drift from the first, and no planner to keep
correct: a SQL query and its typed equivalent produce byte-identical `QueryPlan`s (§7.11) and
identical ordered ids, because they are the same call underneath. This is why `INSERT`/
`UPDATE`/`DELETE`, subqueries, joins, and transactions are excluded rather than deferred: a
vector store's collections share one embedding space and are unioned in scope, never joined;
writes already have `upsert`/`delete`/`delete_where` and stay outside a read-only front end;
subqueries and transactions need a planner and a multi-statement execution model this design
deliberately has neither of. Reversing SPEC §2's SQL non-goal is scoped to exactly this: read
syntax over existing search, not a database.

**Why the hand-rolled parser won.** Three candidates were measured, clean offline debug
build, `rm -rf target`, pinned 1.98 toolchain, one machine:

| Candidate | Transitive crates | Wall | Native code |
|---|---|---|---|
| `sqlparser` 0.63, default features | +18 | 13.25s | yes — `psm`/`stacker` via `cc` |
| `sqlparser` 0.63, `default-features = false, features = ["std"]` | +2 | 5.78s | none, but no recursion protection |
| hand-rolled (chosen) | 0 | 0s | none |

`sqlparser`'s default build adds 18 transitive crates and a `cc`-compiled native dependency
(`psm`/`stacker`, for its own recursion-safety trampoline) — the exact shape of build cost
D0005 exists to keep out, for a grammar this project's whole surface fits in one file. The
cheap `default-features = false` spelling avoids the native code, but it does so by **dropping
`recursive-protection`** — `sqlparser`'s own stack-depth guard against deeply nested input.
Without it, a deeply nested `WHERE (((...)))` clause overflows the parser's call stack and
**aborts the process**, rather than returning an error. That is not a slower path to the same
correctness; it is a crash on attacker- or accident-supplied text, which fails the Core
Foundation *Stable* commitment (SPEC §1: graceful resource exhaustion, not a preference to
trade against build time. A hand-rolled recursive-descent parser costs nothing to build
(it is source already compiled as part of `src/sql/`, not a new crate) and carries its own
explicit depth cap, `MAX_NEST_DEPTH = 128` in `src/sql/mod.rs`, matching `serde_json`'s de
facto recursion cap on the JSON spelling of the same values, so the two spellings accept
the same shapes and both fail the same way: an error, never a crash.

**Reversing SPEC §2's non-goal is deliberate, and scoped.** §2 listed "SQL" among a store's
non-goals alongside a planner and transactions; this feature ships syntax, not an engine,
so that non-goal's wording is corrected in place (`SPEC.md` §2) rather than left to
contradict §7.12 on the same page. The non-goal that survives, unchanged, is a SQL *engine*:
no planner, no joins, no DML, no subqueries, no transactions. Any of those remains a design
change needing an issue first, the same as it always was.

## Evidence

- `src/sql/mod.rs` — `MAX_NEST_DEPTH`, the module doc recording "no feature gate, no new
  crate," and the `Compiled` enum whose variants are exactly the existing `Store` entry
  points.
- `SPEC.md` §7.12 — the grammar, the `WITH`-key table, the dispatch table, the per-surface
  reachability column, and this same measured table restated for readers who start there.
- `cargo tree --no-default-features` — zero new crates from this feature; `tests/build_thesis.rs`
  asserts it in CI.
