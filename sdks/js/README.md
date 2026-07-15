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

`attrs` accept plain JS values — strings, integers, booleans, string arrays, and
`null` — and are normalized to nidus's typed values for you. (For an explicit type,
use the `v.*` helpers.)

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

## Filtering

Build an AND-filter with the `f.*` helpers. Each predicate is a positive assertion
about a present attribute (an absent key matches nothing).

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

## Full-text and hybrid search

```ts
await db.setFtsSchema("docs", ["body"]);

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
