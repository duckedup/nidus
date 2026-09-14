# The conformance corpus

`queries.json` is the one shared source of query conformance cases for nidus-yq9p.2: the
Rust e2e runner (`tests/e2e/corpus.rs`) reads it, and the Python/JS/Go SDK integration suites
(`sdks/*/tests?/…`, a later unit) read the same file rather than each hand-writing their own
query list. Add a case here and it runs everywhere at once — that is the property this file
exists to guarantee.

## Shape

```jsonc
{
  "fixture": {
    "dim": 4,
    "fts": { "<collection>": ["<field>", ...] },
    "collections": {
      "<collection>": [
        { "id": "...", "vector": [...], "attrs": { "<field>": <tagged value>, ... } }
      ]
    }
  },
  "cases": [
    {
      "name": "...",
      "section": "7.1",
      "sql": "SELECT ...",
      "dsl": { "endpoint": "/search", "body": { ... } },
      "expect": { "ids": ["..."] },
      "surfaces": ["lib", "http", "cli", "mcp", "python", "js", "go"],
      "note": "why a surface is missing from the list above, when one is"
    }
  ]
}
```

- **`attrs` values are the tagged wire form** nidus already uses everywhere else on the wire
  (`{"Str": "rust"}`, `{"Int": 120}`, `{"List": ["a","b"]}`, `{"Bool": true}`) — the same shape
  `Filter`/`Predicate`/`Value` serialize to, and the same shape `dsl.body.filter` is already
  written in. One convention, no second translation layer.
- **Seeded and deterministic.** Small vectors, integer scores where possible, no wall clock. A
  case that needs "now" passes an explicit origin (`SPEC.md` §7.6's `decay` reads no clock).
- **`section`** must cover every `SPEC.md` §7 subsection from §7.1 to §7.11 at least once —
  that is nidus-yq9p.2's bar, and §7.12's mapping table (a later unit) is written against this
  same list, so the two must agree.
- **`dsl`** is the typed spelling of the same query: `endpoint` is the HTTP route it would hit,
  `body` is that route's own request body. Every case with a `dsl` runs **both** spellings and
  requires identical ordered ids — that is what proves SQL is a front end, not a second engine.
  `dsl` is `null` only for `7.9` (`;`-batching): no single typed endpoint runs a script, so a
  batch case is checked directly against `expect.batch_ids` instead (see below).
- **`expect`** is one of:
  - `{"ids": [...]}` — the ordered ids a `Hits`-dispatch statement (search / text_search /
    hybrid / list) must return.
  - `{"aggregation": {"count", "sums", "groups": [{"value","count","sums"}, ...]}}` — for a
    `GROUP BY` statement. `groups_truncated` is not compared (it is `false` in every case here).
  - `{"batch_ids": [[...], [...]]}` — one ordered id list per `;`-separated statement, for the
    one `dsl: null` batching case.
- **`surfaces`** is how a case opts out where a surface genuinely cannot express it. The
  **only** legitimate opt-out today is `knn(...)` on `mcp` — MCP's `query` tool is text-native
  and refuses a literal vector (`.claude/rules/cli-feature.md`). A `;`-batched script is also
  out of scope for `mcp` (that surface calls `Nidus::query`, which is single-statement by
  design). Any other omission needs a `note` explaining why; an undocumented opt-out is how a
  corpus quietly stops guarding.
- `python`/`js`/`go` are read by the SDK integration suites (a later unit); the Rust runner
  here ignores those two tokens (it has no such surfaces of its own).

## Adding a case

1. Pick (or extend) a fixture record so the case is decidable by hand — work out the expected
   ids/aggregation yourself before writing them down, the way this file's existing cases were
   built. Vector rankings here are chosen so cosine similarity is unambiguous by inspection
   (no near-ties); BM25-ranked cases either have one matching row or a clear term-frequency
   gap, so the order does not depend on knowing BM25's exact constants.
2. Write the `sql` spelling and, unless it is `7.9` batching, the equivalent typed `dsl` body.
3. Add `surfaces`, narrowing only for the one legitimate `mcp` exclusion above (with a `note`)
   or another documented reason.
4. Run `just test-e2e corpus` (plus `just sdk-*-test-all` once the SDK suites exist) and watch
   it pass. If you are adding coverage for a surface that does not yet implement the feature,
   the corpus should go **red on that surface by name** first — that is the guard actually
   working; fix the surface, then it goes green.

## Cross-reference

`SPEC.md` §7.12 (a later unit) restates this same `section` coverage in its SQL-clause →
§7-section mapping table. If you add a `section` here that table does not yet mention, that
table is now behind this file, not the other way around.
