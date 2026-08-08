# @duckedup/nidus

The JavaScript/TypeScript client for [nidus](https://nidus.duckedup.org) — a small,
fast vector store. This package connects to a running `nidus serve` instance over
HTTP, whether it's on your laptop or a remote host.

It is a **remote client**: zero runtime dependencies, built on the platform-global
`fetch`, so it runs unchanged on Node 18+, Deno, Bun, Cloudflare Workers, and in the
browser.

```sh
npm install @duckedup/nidus
```

This package is versioned in lockstep with nidus itself: a given `@duckedup/nidus`
version is the client for the identically-numbered nidus release. Match the two and the
wire contract lines up.

## Connecting

"Local vs remote" is just the base URL — point the client at a local `nidus serve`
or any reachable host.

```ts
import { NidusClient } from "@duckedup/nidus";

// Local
const db = new NidusClient({ baseUrl: "http://127.0.0.1:7700" });

// Remote, with the bearer token the server was started with (`nidus serve --token`)
const db = new NidusClient({
  baseUrl: "https://nidus.internal.example.com",
  token: process.env.NIDUS_TOKEN,
});
```

## Upserting and searching

`attrs` accept plain JS values — strings, numbers, booleans, string arrays, `Date`s,
and `null` — and are normalized to nidus's typed values for you. (For an explicit
type, use the `v.*` helpers.)

```ts
await db.createCollection("docs");

await db.upsert("docs", [
  { id: "a", vector: [0.1, 0.2, 0.3], attrs: { lang: "rust", year: 2024 } },
  { id: "b", vector: [0.4, 0.5, 0.6], attrs: { lang: "go", year: 2023 } },
  // a text-only doc — omit the vector
  { id: "c", attrs: { body: "vector stores are neat" } },
]);

const hits = await db.search({ query: [0.1, 0.2, 0.3], topK: 5 });
for (const hit of hits) {
  console.log(hit.id, hit.score, hit.attrs.lang); // attrs decoded to plain JS values
}
```

nidus has separate `Int` and `Float` attribute types and compares them same-type only,
but JS has one `number` and `1.0 === 1` — so a plain number becomes an `Int` when
`Number.isInteger` says so and a `Float` otherwise. That means a whole-numbered
measurement lands as an `Int` in whichever records it came out round, and a `Float`
range filter then skips exactly those. Pin such a field with `v.float`:

```ts
import { v } from "@duckedup/nidus";

await db.upsert("docs", [
  {
    id: "d",
    attrs: {
      score: v.float(1), // a Float even though the value is whole
      ratio: 0.75, // already a Float — not an integer
      year: 2024, // an Int
      seen: new Date(), // a DateTime: a UTC instant, epoch milliseconds
    },
  },
]);
```

A `DateTime` carries no timezone and has millisecond resolution; it decodes back to a
`Date`, so a decoded `attrs` map re-encodes to what it came from. `NaN` and `Infinity`
throw — JSON has no spelling for them. The Go and Python SDKs have the numeric types JS
lacks and decide from those instead, so a Python `2.0` or a Go `float64(2)` is a
`Float` where a bare `2` here is an `Int`.

## Filtering

Build an AND-filter with the `f.*` helpers. Each predicate is a positive assertion
about a present attribute (an absent key matches nothing). Comparisons are same-type
only, so an operand must encode to the attribute's type — `f.ge("score", v.float(2))`,
not `f.ge("score", 2)`, for a `Float` attribute.

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

Beyond the comparisons there are text predicates — approximate, token-wise, and
regular-expression matching over a plain attribute (no full-text index required):

```ts
f.fuzzy("title", "vecter store", 2);      // within 2 Levenshtein edits (max 8)
f.containsAllTokens("body", "vector store"); // both tokens, any order
f.containsAnyToken("body", "vector store");
f.containsTokenSequence("body", "vector store"); // as a phrase, in order
f.regex("path", "src/.*\\.rs");           // anchored at both ends, like f.glob
```

`f.regex` uses Rust's `regex` syntax, not JavaScript's: no backreferences and no
lookaround. Prefix it with `(?i)` for case-insensitive matching.

## Full-text and hybrid search

```ts
await db.setFtsSchema("docs", ["body"]);
// Per-field tuning: await db.setFtsSchema("docs", [{ field: "body", k1: 1.5 }]);

// BM25 text search
const text = await db.textSearch({ field: "body", query: "vector store", topK: 10 });

// Fuse vector + text via reciprocal rank fusion
const hybrid = await db.hybridSearch({
  vector: [0.1, 0.2, 0.3],
  field: "body",
  text: "vector store",
  topK: 10,
});
```

A query can search several fields at once, each with its own text, by sending `clauses`
instead of the single field — folded by `combine`, `"Sum"` (a doc hitting title *and*
body outranks one hitting either) or `"Max"` (a long body cannot out-accumulate a
precise title match). Weight the two hybrid legs with `vectorWeight`/`textWeight`.

```ts
const hits = await db.textSearch({
  clauses: [
    { field: "title", query: "rust" },
    { field: "body", query: "async runtime" },
  ],
  combine: "Max",
  topK: 10,
});
```

## Explaining and highlighting a hit

`explain` reports what each leg and each matched clause contributed; `highlight` returns
excerpts of the stored text (so it works even on a field the projection dropped). Both
land on `hit.annotations`, which is absent unless you asked for one of them.

```ts
const hits = await db.hybridSearch({
  vector: [0.1, 0.2, 0.3],
  field: "body",
  text: "vector store",
  explain: true,
  highlight: true, // or { maxFragments: 3, fragmentChars: 120 }
});

for (const { field, fragments } of hits[0]?.annotations?.highlights ?? []) {
  for (const fragment of fragments) {
    for (const span of fragment.spans) {
      console.log(field, fragment.text.slice(...span));
    }
  }
}
```

nidus reports a span as a **UTF-8 byte** range, but a JS string is indexed in UTF-16
code units — so a raw span slices the wrong text out of any non-ASCII excerpt. This SDK
converts them for you: `fragment.spans` are JS string indices, and `fragment.text.slice`
is the matched term. If you compare them against the raw HTTP response, expect the
numbers to differ wherever the excerpt is not ASCII.

## Ranking, grouping, ordering, and aggregating

```ts
// Prefer recent hits: subtract a penalty that halves every `scale` ms of age.
const recent = await db.search({
  query: [0.1, 0.2, 0.3],
  rankBy: {
    decay: { field: "updated_at", origin: Date.now(), scale: 7 * 86_400_000 },
  },
  // …and keep at most 2 hits from any one file
  limitPer: { field: "path", max: 2 },
});

// Sort a listing by an attribute instead of storage order
await db.list({ orderBy: { field: "updated_at", descending: true } });

// Count matches and sum attributes, without reading a single vector
const { count, sums } = await db.aggregate({
  filter: f.and(f.eq("lang", "rust")),
  sum: ["bytes"],
});
```

Ages are measured back from `origin`, never the wall clock, so the same query against an
unchanged store ranks the same way twice. The penalty is *subtracted* from the score, so
it stays meaningful for a metric whose scores are negative or unbounded, and a record
with no usable timestamp is not penalized at all (`missing` defaults to `1`).

## Remembering and recalling (text-native)

When the server is started with an embedder (`nidus serve --embed-provider …`), you
can send **text** and let the server embed it — no need to compute vectors client-side.
`remember` embeds and upserts; `recall` embeds the query and vector-searches.

```ts
// Embed "the quick brown fox" and store it under id "a"
await db.remember("notes", "a", "the quick brown fox", { attrs: { tag: "x" } });

// Summarize first, then embed the summary (server also needs --summarize-provider).
// The stored record additionally carries `nidus.summary` and `nidus.source` attrs.
await db.remember("notes", "b", longArticle, { mode: "summarize" });

// Embed the query text and search, best-first (attrs decoded to plain JS values)
const hits = await db.recall("notes", "quick fox", {
  topK: 5,
  minScore: 0.2,
  filter: f.and(f.eq("tag", "x")),
});
```

Both throw a `NidusError` with status `400` if the server has no embedder configured
(the message names `--embed-provider`); `mode: "summarize"` without a summarizer is
likewise a `400`.

## Everything else

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
```

## Errors

A failed request throws a `NidusError` carrying the HTTP status the server reported,
so you can tell a client fault from a server fault:

```ts
import { NidusError } from "@duckedup/nidus";

try {
  await db.upsert("docs", records);
} catch (err) {
  if (err instanceof NidusError) {
    if (err.isBadRequest) {/* e.g. vector dimension mismatch */}
    if (err.isLocked) {/* the writer lock is held elsewhere (409) */}
    console.error(err.status, err.message);
  }
}
```

A status of `0` means a transport-level failure (the server was unreachable, or the
request timed out — configure `timeoutMs` on the client).

## License

MIT
