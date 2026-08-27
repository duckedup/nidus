---
title: JavaScript / TypeScript SDK
description: "@duckedup/nidus: the official JavaScript/TypeScript client for nidus. Connect to a local or remote nidus server over HTTP, upsert, and search."
---

[`@duckedup/nidus`](https://www.npmjs.com/package/@duckedup/nidus) is the official
JavaScript/TypeScript client for nidus. It drives a running
[`nidus serve`](/guides/http-server/) instance over HTTP, local or remote.

It has **zero runtime dependencies** and is built on the platform-global `fetch`, so it
runs unchanged on Node 18+, Deno, Bun, Cloudflare Workers, and in the browser.

```sh
npm install @duckedup/nidus
```

The SDK is versioned in lockstep with nidus: a given `@duckedup/nidus` version is the
client for the identically-numbered nidus release. Match the two and the wire contract
lines up.

## Connecting

"Local vs remote" is just the base URL: point the client at a local `nidus serve` or
any reachable host. When the server was started with a [token](/guides/http-server/),
pass it as `token`.

```ts
import { NidusClient } from "@duckedup/nidus";

// Local
const db = new NidusClient({ baseUrl: "http://127.0.0.1:7700" });

// Remote, with bearer-token auth
const db = new NidusClient({
  baseUrl: "https://nidus.internal.example.com",
  token: process.env.NIDUS_TOKEN,
  timeoutMs: 5000, // optional per-request timeout
});
```

## Upserting and searching

`attrs` accept plain JS values (strings, integers, booleans, string arrays, `Date`, and
`null`) and are normalized to nidus's typed values for you. Results come back with
`attrs` decoded to plain JS values.

```ts
await db.createCollection("docs");

await db.upsert("docs", [
  { id: "a", vector: [0.1, 0.2, 0.3], attrs: { lang: "rust", year: 2024 } },
  { id: "b", vector: [0.4, 0.5, 0.6], attrs: { lang: "go", year: 2023 } },
  // text-only doc: omit the vector
  { id: "c", attrs: { body: "vector stores are neat" } },
]);

const hits = await db.search({ query: [0.1, 0.2, 0.3], topK: 5 });
for (const hit of hits) {
  console.log(hit.id, hit.score, hit.attrs.lang);
}
```

JS has one `number` type, so a plain number is normalized to `Int` or `Float` by
`Number.isInteger`: a whole-numbered measurement lands as an `Int`, and a `Float` range
filter then skips exactly those records. For an explicit type, use the `v.*` helpers
(`v.str`, `v.int`, `v.float`, `v.bool`, `v.list`, `v.datetime`, `v.nil`):

```ts
import { v } from "@duckedup/nidus";
await db.upsert("docs", [
  {
    id: "d",
    attrs: {
      tags: v.list(["a", "b"]),
      score: v.float(7), // pinned to Float even though it is whole-numbered
      seen: v.datetime(new Date()), // or a raw epoch-millisecond number
    },
  },
]);
```

`v.nil()` is the explicit `Null` value ("set, and empty"), a different fact from an
absent key ("not set / not indexed"). The SDK keeps the two apart in both directions:
a decoded `Date` (from `v.datetime`) re-encodes to the same `DateTime`, never a plain
number, so a round trip through `attrs` never demotes an instant to an `Int`.

## Similar records ("more like this")

`searchSimilar` runs a search using the vector already stored at a record, instead of
a query you supply yourself:

```ts
const hits = await db.searchSimilar("docs", "a", { topK: 5 });
```

Takes the same options as `search` (`topK`, `offset`, `minScore`, `filter`, `exact`,
`includeAttributes`/`excludeAttributes`, `rankBy`, `limitPer`), plus `scope`: which
collections to search, defaulting to the source record's own collection rather than
every collection in the store the way a plain `search`'s omitted scope does.

The source record is never in its own results, dropped by id after ranking, not by a
score cutoff, so a genuine duplicate of it (also scoring near 1.0) still comes back. A
`collection`/id pair naming no record, or a record with no stored vector (a text-only
entry), throws a `NidusError` naming the id and the reason, not an empty result.

## Filtering

Build an AND-filter with the `f.*` helpers. Each predicate is a positive assertion about
a **present** attribute: an absent key matches nothing (including the negative
predicates). See [Search & filters](/guides/search/) for the full semantics.

```ts
import { f } from "@duckedup/nidus";

const hits = await db.search({
  query: [0.1, 0.2, 0.3],
  topK: 10,
  filter: f.and(
    f.eq("lang", "rust"),
    f.ge("year", 2020),
    f.in("status", ["published", "draft"]),
    f.glob("path", "src/*"),
  ),
});
```

Predicates: `eq`, `ne`, `glob`, `iglob`, `in`, `notIn`, `lt`, `le`, `gt`, `ge`, `contains`,
`notContains`, `containsAny`, `all`, `any`, `not`. `iglob` is `glob` with ASCII case
folded on both sides.

## Full-text and hybrid search

```ts
await db.setFtsSchema("docs", ["body"]);

// …or tune BM25 / the analyzer per field; an omitted knob keeps the server default
await db.setFtsSchema("docs", [
  "title",
  { field: "body", k1: 1.5, asciiFolding: true },
]);

// BM25 text search
const text = await db.textSearch({ field: "body", query: "vector store", topK: 10 });

// Fuse vector + text via reciprocal rank fusion
const hybrid = await db.hybridSearch({
  vector: [0.1, 0.2, 0.3],
  field: "body",
  text: "vector store",
  topK: 10,
});

// Prefix match for typeahead: only the final word of `query` expands
const typeahead = await db.textSearch({
  field: "title",
  query: "quick br",
  prefix: true,
  topK: 10,
});
```

Both accept `clauses` (a `TextClause[]`, each `{ field, query, prefix? }`) instead of a
single `field`/`query`, folded by `combine` (`"Sum"` by default, or `"Max"`). Naming the
field both ways at once is a `400`.

`prefix` (default `false`, omitted from the request body when unset) expands only the
final word of that field's query text to every indexed term carrying it as a prefix, for
autocomplete as a caller is still typing; earlier words still match exactly. It is capped
at 256 expansions, past which the commonest completions win rather than the call failing.
Set it on the shorthand `field`/`query` call, or per-entry inside `clauses`.

## Suggesting completions

`suggest` completes a partial word from an indexed field's vocabulary, ranked by document
frequency (the commonest term first) rather than the idf-based ranking `textSearch` uses
for documents. That makes it a separate method built for an autocomplete dropdown: `"nid"`
should surface a common term like `"nidus"` above a rare one like `"nidification"`, the
opposite of how a prefix clause would rank the documents containing them.

```ts
const { suggestions, matched } = await db.suggest({
  scope: ["docs"], // omit for every collection
  field: "body",
  prefix: "vec",
  limit: 10, // the server default; capped at 256 matching terms overall
});
for (const { term, df } of suggestions) console.log(term, df);
```

`matched` counts every term the prefix matched before the 256-term cap, so
`matched > suggestions.length` means the cap truncated the list. Unlike `prefix` above,
which matches stems, `suggest` matches surface forms, so completions are real words:
every keystroke of `"running"` completes to `"running"`.

Each `df` is a conditioned count. `filter` narrows it to the matching documents, so a
dropdown offers only vocabulary the caller can see and a completion whose only documents are
filtered out is absent rather than present with a corpus-wide count. The words before the
final token narrow it too, so send the whole phrase typed so far:

```ts
// "brown" is the commonest br* here, but no document says both "quick" and "brown"
const { suggestions } = await db.suggest({
  scope: ["docs"],
  field: "body",
  prefix: "quick br",
  filter: [{ Eq: ["tenant", { Str: "acme" }] }],
});
// suggestions: [{ term: "bracket", df: 1 }] ("brown" is not offered)
```

A single-token prefix, or one whose earlier words are all stopwords, has no head terms and
behaves exactly as the bare fragment does.

If the exact match finds nothing at all, `suggest` retries with a short edit-distance budget
before giving up, so a mistyped fragment like `"runing"` still completes to `"running"`. This
is on by default; set `fuzzy: false` to opt out.

## Remembering and recalling

When the server is started with an embedder
([`nidus serve --embed-provider …`](/guides/remember-and-recall/)) you can send **text**
and let the server embed it: no need to compute vectors client-side. `remember` embeds
and upserts; `recall` embeds the query and vector-searches.

```ts
// Embed "the quick brown fox" and store it under id "a"
await db.remember("notes", "a", "the quick brown fox", { attrs: { tag: "x" } });

// Expire after an hour, and fold near-duplicates into the closest existing entry.
// On a dedupe match, result.deduped is true and result.id names the entry the
// write actually landed on, which may differ from the id you passed.
const result = await db.remember("notes", "a2", "the quick brown fox!", {
  ttlSeconds: 3600,
  dedupeThreshold: 0.95,
});

// Summarize first, then embed the summary (the server also needs --summarize-provider).
// The stored record additionally carries the `nidus.summary` attr, with the raw
// input in `nidus.text`.
await db.remember("notes", "b", longArticle, { mode: "summarize" });

// Embed the query text and search, best first
const hits = await db.recall("notes", "quick fox", {
  topK: 5,
  minScore: 0.2,
  filter: f.and(f.eq("tag", "x")),
});

// Reinforce: stamp nidus.access_count / nidus.last_accessed on every hit
// returned, and push an existing expiry forward. Off by default, so a plain
// recall stays a pure read.
const reinforced = await db.recall("notes", "quick fox", {
  topK: 5,
  reinforce: true,
  extendTtlSeconds: 86400,
});
```

Setting `reinforce` makes the call a write: it takes the server's writer lock to apply
the stamp, and against a server started `--read-only` the stamp is skipped with a
warning rather than failing the recall. `extendTtlSeconds` only applies with
`reinforce` set, and only pushes an **existing** `nidus.expires_at` forward; it never
gives an expiry to an entry that had none. See
[reinforcement](/guides/remember-and-recall/#reinforcement).

Against a server started **without** an embedder both throw `NidusError` with status
`400`, and the message names `--embed-provider`; `mode: "summarize"` without a
summarizer configured is likewise a `400`. The client only ever sends text; the
embedding always happens server-side. Read `result.id` off the return value rather than
assuming the id you passed, since `dedupeThreshold` can redirect the write.

## Batch search and fusion

`batchSearch` answers several vector queries in one round trip (16 max), and can fuse
them into a single ranking via the same Reciprocal Rank Fusion `hybridSearch` runs
across a vector and a text leg, here across N query legs instead.

```ts
// One ranking per query, in request order
const [rustHits, goHits] = await db.batchSearch({
  queries: [
    { query: rustVec, topK: 5, filter: f.and(f.eq("lang", "rust")) },
    { query: goVec, topK: 5, filter: f.and(f.eq("lang", "go")) },
  ],
});

// Fuse the two legs into ONE ranking instead
const [fused] = await db.batchSearch({
  queries: [{ query: rustVec, topK: 20 }, { query: goVec, topK: 20 }],
  fuse: { rrfK: 60, weights: [2, 1], topK: 10 },
});
```

`batchSearch` always returns `Hit[][]`: without `fuse` it is one array per query in
request order; with `fuse` it is a single-entry array holding the one fused ranking, so
the return shape stays uniform either way. The server validates the whole batch before
running any leg, so a malformed query fails the call rather than returning a partial
answer that cannot be told apart from a real one. `weights` must be empty or exactly as
long as `queries`.

## Aggregating

`aggregate` counts the records matching a filter and sums named attributes, answered
straight from the in-RAM index: no record is built and no vector is read.

```ts
const totals = await db.aggregate({
  scope: ["docs"],
  filter: f.and(f.eq("lang", "rust")),
  sum: ["stars"],
});
// { count, sums: { stars } }

// One row per distinct value of an attribute, alongside the whole-scope totals
const byLang = await db.aggregate({
  scope: ["docs"],
  sum: ["stars"],
  groupBy: "lang",
});
// { count, sums, groups: [{ value, count, sums }, ...], groupsTruncated? }
```

`groups[].value` is `null` for the records *missing* the `groupBy` attribute, a
different bucket from records holding an explicit `Null`. `groupsTruncated` is present
(and `true`) only when distinct values outran the server's cap and later ones were
dropped.

## Pagination, ranking, and projections

These knobs are shared across the search family (`SearchOptions`, `TextSearchOptions`,
`ListOptions`) wherever they apply:

```ts
const hits = await db.search({
  query: [0.1, 0.2, 0.3],
  topK: 10,
  offset: 10, // skip this many top-ranked hits, for pagination (offset + topK ≤ 10000)
  exact: true, // force the exact scan for this query, bypassing any ANN index
  includeAttributes: ["lang", "year"], // or excludeAttributes; sending both is a 400
  rankBy: {
    decay: {
      field: "updatedAt", // a DateTime attr, or an Int of epoch ms
      origin: new Date(),
      scale: 7 * 24 * 60 * 60 * 1000, // half-life of one week
    },
  },
  limitPer: { field: "sourceFile", max: 2 }, // at most 2 hits per sourceFile
});

// list() sorts by an attribute instead of storage order
await db.list({ scope: ["docs"], orderBy: { field: "year", descending: true } });

// explain and highlight are text/hybrid-search only
const explained = await db.textSearch({
  field: "body",
  query: "vector store",
  explain: true, // report each leg's / clause's own score in hit.annotations
  highlight: { maxFragments: 2, fragmentChars: 120 }, // or `true` for the defaults
});
console.log(explained[0].annotations?.highlights);
```

`rankBy`'s `decay` subtracts a recency penalty from the base score
(`score = base - lambda * (1 - decay ^ (age / scale))`), so it stays meaningful even for
a metric whose scores are negative or unbounded. `limitPer` thins an already-ranked
result rather than searching deeper to refill the cap, so it is approximate. `highlight`
reads the stored text, so it still works on a field a projection dropped.

## Running fully in the browser

`@duckedup/nidus` also ships a separate, **ESM-only** subpath, `@duckedup/nidus/wasm`,
that runs nidus itself inside the browser via WebAssembly and stores data in the
browser's Origin Private File System, rather than talking to a `nidus serve` over
HTTP. It is a distinct entry point from the `NidusClient` documented above, kept out
of the default import so its wasm payload never lands in a bundle that only wanted
the HTTP client. Import it lazily, from wherever your app actually runs in the
browser:

```ts
import { acquireOpfsPool } from "@duckedup/nidus/wasm";
```

See [Running in the browser](/guides/wasm/) for the full walkthrough, including the
dedicated-worker requirement OPFS imposes.

## The rest of the API

Every data-plane endpoint of the [HTTP API](/reference/http-api/) has a typed method.
The three SDKs are kept in lockstep on purpose, stated as policy in the Go client's own
header comment (`sdks/go/client.go`):

> The surface mirrors the JavaScript SDK (sdks/js/src/client.ts) endpoint for
> endpoint, deliberately: the SDKs are meant to be reviewable side by side, so a
> method exists here if and only if it exists there, which is why /ready, /cluster
> and /refresh shipped to all three SDKs together. /metrics remains the one route
> still absent, out of scope until it moves the same way.

So `ready()`, `cluster()`, and `refresh()` are each a typed method today, same as every
other endpoint. `/metrics` is the sole deliberate exception across all three SDKs: it
answers Prometheus text, not JSON, which is a scraper's format rather than something
application code parses. `ready()` returns a verdict rather than throwing when the
server reports not-ready, so a `503` is something you branch on, not something you
catch.

```ts
await db.collections();                  // string[]
await db.stats();                        // dimension, distance, ANN config, footprint
await db.list({ scope: ["docs"], filter: f.and(f.eq("lang", "rust")) });
await db.records("docs");                // every record, attrs decoded
await db.getMeta("docs"); await db.setMeta("docs", { owner: "search-team" });
await db.delete("docs", { ids: ["a"] });
await db.deleteWhere("docs", f.and(f.lt("year", 2000)));
await db.flush(); await db.compact();
await db.dropCollection("docs");
await db.aliases();                      // { "docs": "docs_v2" }
await db.setAlias("docs", "docs_v2");    // create or repoint; target must already exist
await db.dropAlias("docs");              // removes the alias, not the records
await db.health();                       // boolean
const r = await db.ready();              // { ready, role, staleness_secs }
if (!r.ready) console.warn(r.reason);    // a 503 is an answer, not a throw
await db.cluster();                      // role, lease state, fencing, commit version
await db.refresh();                      // boolean: did it adopt newer state
```

## Errors

A failed request throws a `NidusError` carrying the HTTP status the server reported, so
you can tell a client fault from a server fault:

```ts
import { NidusError } from "@duckedup/nidus";

try {
  await db.upsert("docs", records);
} catch (err) {
  if (err instanceof NidusError) {
    if (err.isBadRequest) {/* e.g. vector dimension mismatch (400) */}
    if (err.isReadOnly) {/* the store is read-only (403) */}
    if (err.isLocked) {/* the writer lock is held elsewhere (409) */}
    if (err.isOutOfCapacity) {/* max_vector_bytes exceeded, or OOM (507) */}
    console.error(err.status, err.message);
  }
}
```

A status of `0` means a transport-level failure: the server was unreachable, or the
request exceeded `timeoutMs`.
