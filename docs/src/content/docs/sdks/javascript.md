---
title: JavaScript / TypeScript SDK
description: "@duckedup/nidus — the official JavaScript/TypeScript client for nidus. Connect to a local or remote nidus server over HTTP, upsert, and search."
---

[`@duckedup/nidus`](https://www.npmjs.com/package/@duckedup/nidus) is the official
JavaScript/TypeScript client for nidus. It drives a running
[`nidus serve`](/guides/http-server/) instance over HTTP — local or remote.

It has **zero runtime dependencies** and is built on the platform-global `fetch`, so it
runs unchanged on Node 18+, Deno, Bun, Cloudflare Workers, and in the browser.

```sh
npm install @duckedup/nidus
```

## Connecting

"Local vs remote" is just the base URL — point the client at a local `nidus serve` or
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

`attrs` accept plain JS values — strings, integers, booleans, string arrays, and `null`
— and are normalized to nidus's typed values for you. Results come back with `attrs`
decoded to plain JS values.

```ts
await db.createCollection("docs");

await db.upsert("docs", [
  { id: "a", vector: [0.1, 0.2, 0.3], attrs: { lang: "rust", year: 2024 } },
  { id: "b", vector: [0.4, 0.5, 0.6], attrs: { lang: "go", year: 2023 } },
  // text-only doc — omit the vector
  { id: "c", attrs: { body: "vector stores are neat" } },
]);

const hits = await db.search({ query: [0.1, 0.2, 0.3], topK: 5 });
for (const hit of hits) {
  console.log(hit.id, hit.score, hit.attrs.lang);
}
```

For an explicit attribute type, use the `v.*` helpers (`v.str`, `v.int`, `v.bool`,
`v.list`, `v.nil`) — useful to disambiguate, e.g., an integer from a float-free number.

```ts
import { v } from "@duckedup/nidus";
await db.upsert("docs", [
  { id: "d", attrs: { tags: v.list(["a", "b"]), score: v.int(7) } },
]);
```

## Filtering

Build an AND-filter with the `f.*` helpers. Each predicate is a positive assertion about
a **present** attribute — an absent key matches nothing (including the negative
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

Predicates: `eq`, `ne`, `glob`, `in`, `notIn`, `lt`, `le`, `gt`, `ge`.

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

## The rest of the API

Every endpoint of the [HTTP API](/reference/http-api/) has a typed method:

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
await db.health();                       // boolean
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
    if (err.isLocked) {/* the writer lock is held elsewhere (409) */}
    console.error(err.status, err.message);
  }
}
```

A status of `0` means a transport-level failure — the server was unreachable, or the
request exceeded `timeoutMs`.
