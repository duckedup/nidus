---
title: Query with SQL
description: "SELECT statements as read syntax over nidus's search: the grammar, what each clause compiles to, and where it runs (library, HTTP, CLI, MCP)."
---

A `SELECT` statement is a **front end, not a second query engine**. It lexes, parses, and
compiles to the same `SearchOpts`/`HybridOpts`/`ListOpts`/`AggregateOpts`/`Filter`/`FtsQuery`
values a typed caller builds by hand, then runs through the same [`search`](/guides/vector-search/),
[`text_search`](/guides/full-text-search/), [`hybrid_search`](/guides/hybrid-search/),
[`list`](/guides/filters/), and [`aggregate`](/guides/filters/#aggregation) this store already
has. There is no planner and no second execution path: a query and its typed equivalent
produce identical ordered ids and a byte-identical [query plan](/reference/http-api/#query-plans-how-a-query-ran)
when one is asked for.

## The same query, both spellings

```rust
use nidus::{Filter, Predicate, SearchOpts};

let opts = SearchOpts {
    filter: Filter(vec![Predicate::Glob("lang".into(), "r*".into())]),
    top_k: 3,
    ..Default::default()
};
let hits = db.search("notes", &query_vector, &opts)?;
```

```sql
SELECT * FROM notes WHERE lang LIKE 'r*' ORDER BY knn([1, 0, 0, 0]) LIMIT 3
```

Both return the same ordered ids. Run the SQL form with [`Nidus::query`](/reference/api/#nidus),
[`POST /query`](/reference/http-api/#post-query), [`nidus query`](/reference/cli/#query), or
MCP's `query` tool (minus the vector literal; see [Reachability](#reachability) below).

## Grammar

```
script     := statement (';' statement)* [';']              -- multi-query batching
statement  := SELECT projection FROM scope
              [WHERE predicate]
              [GROUP BY field]                               -- aggregation
              [ORDER BY ranking]
              [LIMIT n] [OFFSET n]
              [WITH '(' option (',' option)* ')']            -- everything else below

projection := '*' | '*' EXCEPT '(' field, ... ')' | field (',' field)*
scope      := '*' | ident (',' ident)*

predicate  := or_expr
or_expr    := and_expr (OR and_expr)*
and_expr   := not_expr (AND not_expr)*
not_expr   := [NOT] primary
primary    := '(' predicate ')' | comparison | fn_predicate
comparison := field ('=' | '!=' | '<' | '<=' | '>' | '>=') literal
            | field [NOT] IN '(' literal, ... ')'
            | field [NOT] LIKE  string                       -- glob
            | field ILIKE string                             -- case-insensitive glob
            | field '~' string                                -- regex
fn_predicate := contains '(' field ',' literal ')'
            | not_contains '(' field ',' literal ')'
            | contains_any '(' field ',' literal, ... ')'
            | fuzzy '(' field ',' string ',' int ')'
            | match_all '(' field ',' string ')'
            | match_any '(' field ',' string ')'
            | phrase '(' field ',' string ')'

ranking    := knn '(' vector ')' [decay_tail]
            | match '(' clause (',' clause)* ')'
            | knn '(' vector ')' FUSE match '(' ... ')'
            | field [ASC | DESC]
decay_tail := '-' decay '(' field ',' origin ',' scale [',' decay [',' lambda]] ')'
clause     := field ',' string [PREFIX]
```

`FROM *` and an omitted `FROM` both mean "every collection," matching an empty `scope` on the
typed API and HTTP surfaces. `WITH (...)` is a named-option bag, not a keyword per feature:

| `WITH` key | Compiles to |
|---|---|
| `annotations` | `explain: true` |
| `plan` | `plan: true` |
| `exact` | `exact: true` |
| `min_score = f` | `min_score` |
| `diversity = f` | MMR `diversity` |
| `limit_per = (field, n)` | result-diversity cap per distinct value of `field` |
| `context = (radius n [, parent f, index f, text f])` | parent rollup / neighbour expansion |
| `rerank = ([overscan n] [, text f])` | cross-encoder rerank options |
| `candidates = n`, `rrf_k = f`, `weights = (v, t)` | hybrid fusion knobs |

## What each clause reaches

| Reference | SQL | Typed equivalent |
|---|---|---|
| [Filters & metadata](/guides/filters/) | `WHERE lang LIKE 'r*'` | `Predicate::Glob` |
| [Filters & metadata](/guides/filters/) | `WHERE contains(tags, 'cli')` | `Predicate::Contains` |
| [Filters & metadata](/guides/filters/) | `WHERE (a = 1 OR b = 2) AND NOT c` | `Predicate::Any` / `Predicate::All` / `Predicate::Not` |
| [Filters & metadata](/guides/filters/) | `WHERE fuzzy(f, 'x', 1) AND match_all(f, 'a b')` | `Predicate::Fuzzy` / `Predicate::ContainsAllTokens` |
| [Filters & metadata](/guides/filters/) | `WHERE field ~ '^the.*fox$'` | `Predicate::Regex` |
| [Vector](/guides/vector-search/) / [full-text](/guides/full-text-search/) / [hybrid](/guides/hybrid-search/) search | `ORDER BY knn([...])`, `ORDER BY match(f, 'q')`, `ORDER BY knn([...]) FUSE match(f, 'q')` | `search` / `text_search` / `hybrid_search` |
| [Filters & metadata](/guides/filters/#aggregation) | `SELECT sum(n), count(*) [GROUP BY f]`; `WITH (limit_per = (f, n))`; `WITH (diversity = d)` | `AggregateOpts`; `LimitPer`; `diversity` |
| [Query annotations](/reference/http-api/#annotations-why-a-hit-matched) | `WITH (annotations)` | `explain: true` |
| Batching (below) | `SELECT ...; SELECT ...` | `Nidus::query_batch` |
| Parent rollup & expansion | `WITH (context = (radius 1, parent "p", index "i", text "t"))` | `Expand` |
| [Query plans](/reference/http-api/#query-plans-how-a-query-ran) | `WITH (plan)` | `plan: true` |

## Aggregates and timestamps

`sum(...)` and `count(*)` go in the projection, where SQL puts them. A `GROUP BY` is
optional: without one you get whole-scope totals.

```sql
SELECT sum(bytes), count(*) FROM files
SELECT sum(bytes) FROM files GROUP BY lang
```

Timestamps are written with a `timestamp` prefix, which is what makes a value a
`DateTime` rather than a plain integer. Either spelling works, and both are UTC:

```sql
SELECT * FROM notes WHERE created > timestamp '2026-09-13T00:00:00Z'
SELECT * FROM notes WHERE created > timestamp 1757721600000
```

Neither `count` nor `timestamp` is a reserved word. `count` is a function only directly
before `(`, and `timestamp` only directly before a quoted instant or a number, so an
attribute may still be named either one.

## Multi-statement scripts

`;`-separate several statements to run them as a script, each answered in order:

```sql
SELECT * FROM notes WHERE lang = 'rust' ORDER BY bytes;
SELECT * FROM notes WHERE lang = 'go' ORDER BY bytes
```

Run this form with [`Nidus::query_batch`](/reference/api/#nidus) or `POST /query`; a single
statement uses [`Nidus::query`](/reference/api/#nidus), which errors if it finds more than one.

## Reachability

Every clause above runs on the library, HTTP, and CLI surfaces. MCP's `query` tool narrows
one case on purpose: it refuses a literal `ORDER BY knn([...])` vector, because that surface
is text-native and gives a caller no way to type a vector argument. Use `ORDER BY match(field,
'text')` for keyword ranking, a `WHERE` filter for metadata, or the `recall`/`hybrid_search`
tools for vector search. A `;`-batched script is likewise out of scope on MCP, since its
`query` tool calls the single-statement `Nidus::query`.

This mirrors a narrowing MCP already had: its other filter-taking tools expose a curated 13
of `Predicate`'s 21 variants, because the full set is too large to be a usable
tool-selection prompt. SQL's `WHERE` clause compiles the same 21 variants everywhere else.

## Errors

Every surface reports a parse or compile failure the same way: a byte offset into the source
text, what went wrong, and the reference section that owns the rule.

```
sql parse error at byte 17: expected a value after '=' (§7.3 boolean composition)
```

Over HTTP this is a `400`, the same status a malformed typed request gets.

## Excluded, and why

nidus is a vector store with no engine to run these, not a case where they were merely hard
to parse:

| Excluded | Why |
|---|---|
| Joins | Collections share one embedding space and are unioned in scope, never joined. |
| `INSERT` / `UPDATE` / `DELETE` | Writes go through `upsert`/`delete`/`delete_where`; this syntax is read-only by construction. |
| Subqueries | Sequencing and materializing intermediate results needs a planner, and there is none. |
| Transactions | Multi-operation transactions are a store-wide non-goal, unrelated to read syntax. |

`SPEC.md` §7.12 carries this same grammar with a per-surface reachability table and the
measured build numbers behind the parser choice;
[D0017](https://github.com/duckedup/nidus/blob/main/decisions/0017-sql-is-syntax-not-an-engine.md)
records why a hand-rolled parser won over a parser crate.
