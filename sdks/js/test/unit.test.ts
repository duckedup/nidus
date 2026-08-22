import { describe, expect, it } from "vitest";

import { NidusClient, NidusError, f, v } from "../src/index.js";
import { decodeValue, encodeValue } from "../src/values.js";

/** A fetch double that records the last call and returns a canned JSON response. */
function mockFetch(body: unknown, status = 200) {
  const calls: { url: string; init: RequestInit; json: unknown }[] = [];
  const fn = async (url: string, init: RequestInit = {}) => {
    calls.push({
      url,
      init,
      json: init.body ? JSON.parse(init.body as string) : undefined,
    });
    return new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    });
  };
  return { fn, calls };
}

describe("value encoding", () => {
  it("maps plain JS scalars to the externally-tagged wire shape", () => {
    expect(encodeValue("rust")).toEqual({ Str: "rust" });
    expect(encodeValue(2024)).toEqual({ Int: 2024 });
    expect(encodeValue(true)).toEqual({ Bool: true });
    expect(encodeValue(["a", "b"])).toEqual({ List: ["a", "b"] });
    expect(encodeValue(1.5)).toEqual({ Float: 1.5 });
    expect(encodeValue(new Date(1700000000000))).toEqual({
      DateTime: 1700000000000,
    });
    expect(encodeValue(null)).toBe("Null");
  });

  it("passes an already-tagged value through unchanged", () => {
    expect(encodeValue(v.str("x"))).toEqual({ Str: "x" });
    expect(encodeValue(v.float(2))).toEqual({ Float: 2 });
    expect(encodeValue(v.datetime(0))).toEqual({ DateTime: 0 });
    expect(encodeValue(v.nil())).toBe("Null");
  });

  it("splits Int from Float by Number.isInteger, since JS has no int type", () => {
    // `1.0 === 1` in JS, so the *value* is all there is to go on. Go and Python decide
    // from the static type instead, which is why `2.0` is a Float there and an Int here.
    expect(encodeValue(1.5)).toEqual({ Float: 1.5 });
    expect(encodeValue(2.0)).toEqual({ Int: 2 });
    expect(encodeValue(-0)).toEqual({ Int: -0 });
    // v.float is the escape hatch: it pins a whole-numbered field to Float, which is
    // what keeps a Float range filter from skipping the records that came out round.
    expect(v.float(2)).toEqual({ Float: 2 });
    expect(v.int(2)).toEqual({ Int: 2 });
    expect(() => v.int(1.5)).toThrow(TypeError);
  });

  it("rejects the numbers JSON cannot spell", () => {
    // JSON.stringify writes these as `null`, which serde then refuses to read as an f64.
    for (const bad of [NaN, Infinity, -Infinity]) {
      expect(() => encodeValue(bad)).toThrow(TypeError);
      expect(() => v.float(bad)).toThrow(TypeError);
    }
  });

  it("encodes a Date as epoch milliseconds in UTC", () => {
    const when = new Date("2023-11-14T22:13:20.000Z");
    expect(encodeValue(when)).toEqual({ DateTime: 1700000000000 });
    expect(v.datetime(when)).toEqual({ DateTime: 1700000000000 });
    // The raw millisecond form is accepted too, for a caller who already holds one.
    expect(v.datetime(1700000000000)).toEqual({ DateTime: 1700000000000 });
    expect(() => v.datetime(new Date("not a date"))).toThrow(TypeError);
  });

  it("round-trips through decode", () => {
    expect(decodeValue(encodeValue("rust") as never)).toBe("rust");
    expect(decodeValue(encodeValue(7) as never)).toBe(7);
    expect(decodeValue(encodeValue(null) as never)).toBe(null);
    expect(decodeValue(encodeValue(["a"]) as never)).toEqual(["a"]);
    expect(decodeValue(encodeValue(1.5) as never)).toBe(1.5);
    // A DateTime decodes to a Date, not a number, so re-encoding reproduces the tag
    // rather than demoting the instant to an Int.
    const when = new Date("2023-11-14T22:13:20.000Z");
    const back = decodeValue(encodeValue(when) as never);
    expect(back).toEqual(when);
    expect(encodeValue(back as Date)).toEqual({ DateTime: 1700000000000 });
  });
});

describe("filter builder", () => {
  it("produces the bare predicate-array wire shape", () => {
    const filter = f.and(
      f.eq("lang", "rust"),
      f.ge("year", 2020),
      f.in("tag", ["a", "b"]),
      f.glob("path", "src/*"),
    );
    expect(filter).toEqual([
      { Eq: ["lang", { Str: "rust" }] },
      { Ge: ["year", { Int: 2020 }] },
      { In: ["tag", [{ Str: "a" }, { Str: "b" }]] },
      { Glob: ["path", "src/*"] },
    ]);
  });

  it("carries the operand's encoded type into the predicate", () => {
    // Same-type-only comparison makes this the difference between a range that matches
    // and one that silently matches nothing, and the operand is where it is decided.
    expect(f.ge("score", 1.5)).toEqual({ Ge: ["score", { Float: 1.5 }] });
    expect(f.ge("score", v.float(2))).toEqual({ Ge: ["score", { Float: 2 }] });
    expect(f.ge("year", 2)).toEqual({ Ge: ["year", { Int: 2 }] });
    expect(f.ge("seen", new Date(1700000000000))).toEqual({
      Ge: ["seen", { DateTime: 1700000000000 }],
    });
  });

  it("tags iglob distinctly from glob, sharing the bare-string operand", () => {
    const pred = f.iglob("path", "Src/*") as { IGlob: [string, string] };
    expect(pred).toEqual({ IGlob: ["path", "Src/*"] });
    expect(typeof pred.IGlob[1]).toBe("string");
  });

  it("encodes the containment predicates over a list attribute", () => {
    expect(f.contains("tags", "rust")).toEqual({
      Contains: ["tags", { Str: "rust" }],
    });
    expect(f.notContains("tags", "wip")).toEqual({
      NotContains: ["tags", { Str: "wip" }],
    });
    expect(f.containsAny("tags", ["rust", "go"])).toEqual({
      ContainsAny: ["tags", [{ Str: "rust" }, { Str: "go" }]],
    });
  });

  it("encodes combinators outside the key/value tuple shape", () => {
    // all/any wrap a bare array of predicates; not wraps a single one.
    expect(f.any(f.eq("a", 1), f.eq("b", 2))).toEqual({
      Any: [{ Eq: ["a", { Int: 1 }] }, { Eq: ["b", { Int: 2 }] }],
    });
    expect(f.all(f.eq("a", 1))).toEqual({ All: [{ Eq: ["a", { Int: 1 }] }] });
    expect(f.not(f.eq("a", 1))).toEqual({ Not: { Eq: ["a", { Int: 1 }] } });
  });

  it("emits empty groups as [] so the identities survive deserialization", () => {
    expect(f.all()).toEqual({ All: [] });
    expect(f.any()).toEqual({ Any: [] });
  });

  it("nests groups arbitrarily", () => {
    expect(f.not(f.any(f.contains("tags", "wip")))).toEqual({
      Not: { Any: [{ Contains: ["tags", { Str: "wip" }] }] },
    });
  });

  it("encodes fuzzy as the one three-element predicate", () => {
    // Every other leaf is a [key, operand] pair; Fuzzy carries the edit budget too.
    const pred = f.fuzzy("title", "nidus", 2) as { Fuzzy: [string, string, number] };
    expect(pred).toEqual({ Fuzzy: ["title", "nidus", 2] });
    expect(pred.Fuzzy).toHaveLength(3);
    expect(typeof pred.Fuzzy[2]).toBe("number");
  });

  it("encodes the token and regex predicates with a bare-string operand", () => {
    // The query text is raw, not a tagged Value — the server tokenizes it itself.
    expect(f.containsAllTokens("body", "quick fox")).toEqual({
      ContainsAllTokens: ["body", "quick fox"],
    });
    expect(f.containsAnyToken("body", "quick fox")).toEqual({
      ContainsAnyToken: ["body", "quick fox"],
    });
    expect(f.containsTokenSequence("body", "quick brown fox")).toEqual({
      ContainsTokenSequence: ["body", "quick brown fox"],
    });
    expect(f.regex("path", "^src/.*\\.rs$")).toEqual({
      Regex: ["path", "^src/.*\\.rs$"],
    });
  });

  it("distinguishes f.all (a predicate) from f.and (a filter)", () => {
    // Only f.all nests: f.and returns the top-level array.
    expect(Array.isArray(f.and(f.eq("a", 1)))).toBe(true);
    expect(Array.isArray(f.all(f.eq("a", 1)))).toBe(false);
  });
});

describe("NidusClient request shaping", () => {
  it("sends upsert with normalized attrs and an omitted vector for text-only docs", async () => {
    const { fn, calls } = mockFetch({ upserted: 2 });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const n = await db.upsert("docs", [
      { id: "a", vector: [1, 0, 0], attrs: { lang: "rust", year: 2024 } },
      { id: "b", attrs: { body: v.str("text only") } },
    ]);
    expect(n).toBe(2);
    expect(calls[0]!.url).toBe("http://x/collections/docs/upsert");
    expect(calls[0]!.json).toEqual({
      records: [
        { id: "a", vector: [1, 0, 0], attrs: { lang: { Str: "rust" }, year: { Int: 2024 } } },
        { id: "b", attrs: { body: { Str: "text only" } } },
      ],
    });
  });

  it("omits offset when unset and sends it on every paginated search when set", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });

    // Unset: byte-identical to a client that predates pagination.
    await db.search({ query: [1, 0, 0], topK: 5 });
    expect(calls[0]!.json).toEqual({ query: [1, 0, 0], scope: [], top_k: 5, filter: [] });

    await db.search({ query: [1, 0, 0], topK: 5, offset: 10 });
    expect(calls[1]!.json).toMatchObject({ top_k: 5, offset: 10 });
    await db.textSearch({ field: "body", query: "fox", offset: 3 });
    expect(calls[2]!.json).toMatchObject({ offset: 3 });
    await db.hybridSearch({ vector: [1, 0, 0], field: "body", text: "fox", offset: 3 });
    expect(calls[3]!.json).toMatchObject({ offset: 3 });
  });

  it("omits the projection and exact knobs unless asked, and maps them to snake_case", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });

    // Unset: byte-identical to a client that predates projection.
    await db.search({ query: [1, 0, 0], topK: 5 });
    expect(calls[0]!.json).toEqual({ query: [1, 0, 0], scope: [], top_k: 5, filter: [] });
    await db.list({ limit: 5 });
    expect(calls[1]!.json).toEqual({ scope: [], limit: 5, filter: [] });

    await db.search({ query: [1, 0, 0], exact: true, includeAttributes: ["title"] });
    expect(calls[2]!.json).toMatchObject({ exact: true, include_attributes: ["title"] });
    await db.search({ query: [1, 0, 0], excludeAttributes: ["body"] });
    expect(calls[3]!.json).toMatchObject({ exclude_attributes: ["body"] });
    await db.list({ includeAttributes: ["lang"] });
    expect(calls[4]!.json).toMatchObject({ include_attributes: ["lang"] });
  });

  it("sends search with camelCase mapped to snake_case and decodes hit attrs", async () => {
    const { fn, calls } = mockFetch([
      { collection: "docs", id: "a", score: 0.9, attrs: { lang: { Str: "rust" } } },
    ]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const hits = await db.search({
      query: [1, 0, 0],
      topK: 5,
      minScore: 0.1,
      filter: f.and(f.eq("lang", "rust")),
    });
    expect(calls[0]!.json).toEqual({
      query: [1, 0, 0],
      scope: [],
      top_k: 5,
      min_score: 0.1,
      filter: [{ Eq: ["lang", { Str: "rust" }] }],
    });
    expect(hits[0]).toEqual({
      collection: "docs",
      id: "a",
      score: 0.9,
      attrs: { lang: "rust" },
    });
  });

  it("sends searchSimilar to /search/similar with camelCase mapped to snake_case", async () => {
    const { fn, calls } = mockFetch([
      { collection: "docs", id: "b", score: 0.8, attrs: { lang: { Str: "rust" } } },
    ]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const hits = await db.searchSimilar({
      collection: "docs",
      id: "a",
      topK: 5,
      minScore: 0.1,
      filter: f.and(f.eq("lang", "rust")),
    });
    expect(calls[0]!.url).toBe("http://x/search/similar");
    expect(calls[0]!.json).toEqual({
      collection: "docs",
      id: "a",
      scope: [],
      top_k: 5,
      min_score: 0.1,
      filter: [{ Eq: ["lang", { Str: "rust" }] }],
    });
    expect(hits[0]).toEqual({
      collection: "docs",
      id: "b",
      score: 0.8,
      attrs: { lang: "rust" },
    });
  });

  it("sends a bare fts field name unchanged", async () => {
    const { fn, calls } = mockFetch({ ok: true });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.setFtsSchema("docs", ["body"]);
    expect(calls[0]!.url).toBe("http://x/collections/docs/fts-schema");
    expect(calls[0]!.json).toEqual({ fields: ["body"] });
  });

  it("maps fts field tuning to snake_case and omits the unset knobs", async () => {
    const { fn, calls } = mockFetch({ ok: true });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.setFtsSchema("docs", [
      "title",
      { field: "body", k1: 1.5, asciiFolding: true, maxTokenLen: 40 },
    ]);
    expect(calls[0]!.json).toEqual({
      fields: [
        "title",
        { field: "body", k1: 1.5, ascii_folding: true, max_token_len: 40 },
      ],
    });
  });

  it("sends a bare filter-index field name unchanged", async () => {
    const { fn, calls } = mockFetch({ ok: true });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.setFilterIndex("docs", ["body"]);
    expect(calls[0]!.url).toBe("http://x/collections/docs/filter-index");
    expect(calls[0]!.json).toEqual({ fields: ["body"] });
  });

  it("omits unset filter-index knobs, since the server defaults them to true", async () => {
    const { fn, calls } = mockFetch({ ok: true });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.setFilterIndex("docs", [
      "title",
      { field: "body", trigrams: false },
    ]);
    expect(calls[0]!.json).toEqual({
      fields: ["title", { field: "body", trigrams: false }],
    });
  });

  it("sends an empty filter-index declaration to drop the index", async () => {
    const { fn, calls } = mockFetch({ ok: true });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.setFilterIndex("docs", []);
    expect(calls[0]!.json).toEqual({ fields: [] });
  });

  it("sends diversity only when set, and keeps a zero lambda", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });

    // Absent, not null: an unset knob must leave the request bytes unchanged.
    await db.search({ query: [1, 0, 0] });
    expect("diversity" in (calls[0]!.json as object)).toBe(false);

    // Zero is a meaningful lambda (pure variety), so it must survive pruning.
    await db.search({ query: [1, 0, 0], diversity: 0 });
    expect(calls[1]!.json).toMatchObject({ diversity: 0 });

    await db.searchSimilar({ collection: "docs", id: "d1", diversity: 0.3 });
    expect(calls[2]!.json).toMatchObject({ diversity: 0.3 });
    await db.textSearch({ field: "body", query: "alpha", diversity: 0.5 });
    expect(calls[3]!.json).toMatchObject({ diversity: 0.5 });
    await db.recall("notes", "why", { diversity: 1 });
    expect(calls[4]!.json).toMatchObject({ diversity: 1 });
    await db.recall("notes", "why");
    expect("diversity" in (calls[5]!.json as object)).toBe(false);

    // The batch route builds its per-query objects inline rather than through `prune`, so
    // absence there depends on JSON.stringify dropping `undefined`, not on the pruner.
    await db.batchSearch({ queries: [{ query: [1, 0, 0] }, { query: [0, 1, 0], diversity: 0 }] });
    const sent = JSON.parse(JSON.stringify(calls[6]!.json)) as {
      queries: Record<string, unknown>[];
    };
    expect("diversity" in sent.queries[0]!).toBe(false);
    expect(sent.queries[1]!.diversity).toBe(0);
  });

  it("maps expand and rollup to snake_case, and omits them unless asked", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });

    // Unset: byte-identical to a client that predates expansion.
    await db.search({ query: [1, 0, 0] });
    expect(calls[0]!.json).toEqual({ query: [1, 0, 0], scope: [], filter: [] });

    // A bare radius sends only a radius; the server fills the reserved attr names.
    await db.search({ query: [1, 0, 0], expand: { radius: 2 } });
    expect(calls[1]!.json).toMatchObject({ expand: { radius: 2 } });
    expect(
      Object.keys((calls[1]!.json as { expand: object }).expand),
    ).toEqual(["radius"]);

    await db.search({
      query: [1, 0, 0],
      expand: { radius: 1, parentField: "doc", textField: "body" },
    });
    expect(calls[2]!.json).toMatchObject({
      expand: { radius: 1, parent_field: "doc", text_field: "body" },
    });

    await db.recall("notes", "hi", { rollup: { neighbours: 1 } });
    expect(calls[3]!.json).toMatchObject({ rollup: { neighbours: 1 } });
  });

  it("carries a hit's context through when the server sends one", async () => {
    const { fn, calls } = mockFetch([
      { collection: "c", id: "d#1", score: 0.9, attrs: {}, context: "widened" },
      { collection: "c", id: "d#2", score: 0.8, attrs: {} },
    ]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const hits = await db.search({ query: [1, 0, 0], expand: { radius: 1 } });
    expect(calls[0]!.json).toMatchObject({ expand: { radius: 1 } });
    expect(hits[0]!.context).toBe("widened");
    // Absent stays absent, not `undefined` — the shape an unexpanded hit always had.
    expect("context" in hits[1]!).toBe(false);
  });

  it("omits the ranking knobs unless asked, and maps them to snake_case", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });

    // Unset: byte-identical to a client that predates ranking.
    await db.search({ query: [1, 0, 0] });
    expect(calls[0]!.json).toEqual({ query: [1, 0, 0], scope: [], filter: [] });
    await db.list();
    expect(calls[1]!.json).toEqual({ scope: [], filter: [] });

    await db.search({
      query: [1, 0, 0],
      rankBy: { decay: { field: "ts", origin: new Date(1700000000000) } },
      limitPer: { field: "path", max: 2 },
    });
    expect(calls[2]!.json).toMatchObject({
      // A Date origin becomes epoch ms, and the knobs left unset keep the server's.
      rank_by: { Decay: { field: "ts", origin: 1700000000000 } },
      limit_per: { field: "path", max: 2 },
    });
    expect(
      Object.keys(
        (calls[2]!.json as { rank_by: { Decay: object } }).rank_by.Decay,
      ),
    ).toEqual(["field", "origin"]);

    await db.search({
      query: [1, 0, 0],
      rankBy: {
        decay: {
          field: "ts",
          origin: 1700000000000,
          scale: 86400000,
          decay: 0.9,
          lambda: 0.5,
          missing: 0.2,
        },
      },
    });
    expect(calls[3]!.json).toMatchObject({
      rank_by: {
        Decay: {
          field: "ts",
          origin: 1700000000000,
          scale: 86400000,
          decay: 0.9,
          lambda: 0.5,
          missing: 0.2,
        },
      },
    });

    await db.list({ orderBy: { field: "ts", descending: true } });
    expect(calls[4]!.json).toMatchObject({
      order_by: { field: "ts", descending: true },
    });
  });

  it("keeps the single-field text query spelling and adds the clause list", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });

    // The compatibility contract: one field plus its query, exactly as before.
    await db.textSearch({ field: "body", query: "fox" });
    expect(calls[0]!.json).toEqual({
      field: "body",
      query: "fox",
      scope: [],
      filter: [],
    });

    await db.textSearch({
      clauses: [
        { field: "title", query: "rust" },
        { field: "body", query: "async runtime" },
      ],
      combine: "Max",
    });
    expect(calls[1]!.json).toEqual({
      clauses: [
        { field: "title", query: "rust" },
        { field: "body", query: "async runtime" },
      ],
      combine: "Max",
      scope: [],
      filter: [],
    });

    // `combine` unset leaves the server's "Sum", and never names a `field` alongside.
    await db.textSearch({ clauses: [{ field: "title", query: "rust" }] });
    expect(calls[2]!.json).toEqual({
      clauses: [{ field: "title", query: "rust" }],
      scope: [],
      filter: [],
    });
  });

  it("spells the hybrid text leg as field+text or clauses, and weights each leg", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });

    await db.hybridSearch({ vector: [1, 0, 0], field: "body", text: "fox" });
    expect(calls[0]!.json).toEqual({
      vector: [1, 0, 0],
      field: "body",
      text: "fox",
      scope: [],
      filter: [],
    });

    // The clause form spells the text `query`, matching /text-search.
    await db.hybridSearch({
      vector: [1, 0, 0],
      clauses: [{ field: "title", query: "rust" }],
      combine: "Sum",
    });
    expect(calls[1]!.json).toEqual({
      vector: [1, 0, 0],
      clauses: [{ field: "title", query: "rust" }],
      combine: "Sum",
      scope: [],
      filter: [],
    });

    await db.hybridSearch({
      vector: [1, 0, 0],
      field: "body",
      text: "fox",
      vectorWeight: 2,
      textWeight: 0.5,
    });
    expect(calls[2]!.json).toMatchObject({ vector_weight: 2, text_weight: 0.5 });
  });

  it("omits explain and highlight unless asked, sending `true` as the defaults object", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });

    await db.textSearch({ field: "body", query: "fox", highlight: false });
    expect(calls[0]!.json).toEqual({
      field: "body",
      query: "fox",
      scope: [],
      filter: [],
    });

    // `true` is `{}` — the wire spelling for "highlight with every default".
    await db.textSearch({ field: "body", query: "fox", explain: true, highlight: true });
    expect(calls[1]!.json).toMatchObject({ explain: true, highlight: {} });

    await db.textSearch({
      field: "body",
      query: "fox",
      highlight: { maxFragments: 3, fragmentChars: 40 },
    });
    expect(calls[2]!.json).toMatchObject({
      highlight: { max_fragments: 3, fragment_chars: 40 },
    });

    // Only the named knob travels; the other keeps the server's default.
    await db.hybridSearch({
      vector: [1, 0, 0],
      field: "body",
      text: "fox",
      highlight: { maxFragments: 2 },
    });
    expect(calls[3]!.json).toMatchObject({ highlight: { max_fragments: 2 } });
    expect(
      (calls[3]!.json as { highlight: object }).highlight,
    ).not.toHaveProperty("fragment_chars");
  });

  it("aggregates a count and tagged sums, decoding the sums to numbers", async () => {
    const { fn, calls } = mockFetch({
      count: 12,
      sums: { bytes: { Int: 40960 }, ratio: { Float: 1.5 } },
    });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.aggregate({
      scope: ["docs"],
      sum: ["bytes", "ratio"],
      filter: f.and(f.eq("lang", "rust")),
    });
    expect(calls[0]!.url).toBe("http://x/aggregate");
    expect(calls[0]!.json).toEqual({
      scope: ["docs"],
      filter: [{ Eq: ["lang", { Str: "rust" }] }],
      sum: ["bytes", "ratio"],
    });
    expect(out).toEqual({ count: 12, sums: { bytes: 40960, ratio: 1.5 } });
  });

  it("aggregates store-wide with no options at all", async () => {
    const { fn, calls } = mockFetch({ count: 3, sums: {} });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    expect(await db.aggregate()).toEqual({ count: 3, sums: {} });
    expect(calls[0]!.json).toEqual({ scope: [], filter: [], sum: [] });
  });

  it("groups an aggregate, decoding each group's value and sums", async () => {
    const { fn, calls } = mockFetch({
      count: 3,
      sums: { bytes: { Int: 8 } },
      groups: [
        { value: { Str: "rust" }, count: 2, sums: { bytes: { Int: 8 } } },
        { value: null, count: 1, sums: { bytes: { Int: 0 } } },
      ],
    });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.aggregate({ sum: ["bytes"], groupBy: "lang" });
    expect(calls[0]!.json).toEqual({
      scope: [],
      filter: [],
      sum: ["bytes"],
      group_by: "lang",
    });
    expect(out.groups).toEqual([
      { value: "rust", count: 2, sums: { bytes: 8 } },
      // The records missing `lang` entirely — a different group from a present null.
      { value: null, count: 1, sums: { bytes: 0 } },
    ]);
    expect(out.groupsTruncated).toBeUndefined();
  });

  it("leaves an ungrouped aggregate exactly the shape it always was", async () => {
    const { fn, calls } = mockFetch({ count: 3, sums: {} });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.aggregate({ sum: [] });
    expect(calls[0]!.json).toEqual({ scope: [], filter: [], sum: [] });
    expect("groups" in out).toBe(false);
  });

  it("batches several queries into one request, one ranking per query", async () => {
    const { fn, calls } = mockFetch({
      results: [[{ collection: "docs", id: "a", score: 1, attrs: {} }], []],
    });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.batchSearch({
      queries: [{ query: [1, 0, 0], topK: 1 }, { query: [0, 1, 0] }],
    });
    expect(calls[0]!.url).toBe("http://x/search/batch");
    expect(calls[0]!.json).toEqual({
      queries: [
        { query: [1, 0, 0], scope: [], top_k: 1, filter: [] },
        { query: [0, 1, 0], scope: [], filter: [] },
      ],
    });
    expect(out).toHaveLength(2);
    expect(out[0]![0]!.id).toBe("a");
    expect(out[1]).toEqual([]);
  });

  it("returns a fused batch as a single ranking, so the shape is uniform", async () => {
    const { fn, calls } = mockFetch({
      fused: [{ collection: "docs", id: "a", score: 0.5, attrs: {} }],
    });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.batchSearch({
      queries: [{ query: [1, 0, 0] }, { query: [0, 1, 0] }],
      fuse: { rrfK: 60, weights: [1, 0.5], topK: 5 },
    });
    const sent = calls[0]!.json as { fuse: unknown };
    expect(sent.fuse).toEqual({ rrf_k: 60, weights: [1, 0.5], top_k: 5 });
    expect(out).toHaveLength(1);
    expect(out[0]!.map((h) => h.id)).toEqual(["a"]);
  });

  it("sends fuse even with no knobs, since fuse is what picks the response shape", async () => {
    const { fn, calls } = mockFetch({ fused: [] });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.batchSearch({ queries: [{ query: [1, 0, 0] }], fuse: {} });
    expect((calls[0]!.json as { fuse: unknown }).fuse).toEqual({});
  });

  it("attaches a bearer token when configured", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn, token: "sekret" });
    await db.list();
    expect((calls[0]!.init.headers as Record<string, string>).authorization).toBe(
      "Bearer sekret",
    );
  });

  it("strips a trailing slash from the base URL", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x/", fetch: fn });
    await db.collections();
    expect(calls[0]!.url).toBe("http://x/collections");
  });
});

describe("hit annotations", () => {
  /** A one-hit response carrying whatever annotations the case is about. */
  function annotated(annotations?: unknown) {
    return mockFetch([
      {
        collection: "docs",
        id: "a",
        score: 0.5,
        attrs: { lang: { Str: "rust" } },
        ...(annotations ? { annotations } : {}),
      },
    ]);
  }

  it("leaves the key absent on an unannotated hit", async () => {
    const { fn } = annotated();
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const [hit] = await db.textSearch({ field: "body", query: "fox" });
    expect(hit).toEqual({
      collection: "docs",
      id: "a",
      score: 0.5,
      attrs: { lang: "rust" },
    });
    expect("annotations" in hit!).toBe(false);
  });

  it("decodes every leg, clause, and highlight the server sent", async () => {
    const { fn } = annotated({
      vector: { rank: 0, score: 0.98 },
      text: { rank: 1, score: 1.1 },
      clauses: [{ field: "title", score: 0.49 }],
      highlights: [
        { field: "body", fragments: [{ text: "we were running", spans: [[8, 15]] }] },
      ],
    });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const [hit] = await db.hybridSearch({
      vector: [1, 0, 0],
      field: "body",
      text: "run",
      explain: true,
      highlight: true,
    });
    expect(hit!.annotations).toEqual({
      vector: { rank: 0, score: 0.98 },
      text: { rank: 1, score: 1.1 },
      clauses: [{ field: "title", score: 0.49 }],
      highlights: [
        { field: "body", fragments: [{ text: "we were running", spans: [[8, 15]] }] },
      ],
    });
    // The parts the server omitted stay omitted rather than becoming empty arrays.
    const { fn: fn2 } = annotated({ clauses: [{ field: "title", score: 0.49 }] });
    const db2 = new NidusClient({ baseUrl: "http://x", fetch: fn2 });
    const [only] = await db2.textSearch({ field: "title", query: "run", explain: true });
    expect(only!.annotations).toEqual({ clauses: [{ field: "title", score: 0.49 }] });
  });

  it("converts highlight spans from UTF-8 bytes to JS string indices", async () => {
    // The trap this guards: the server counts bytes, a JS string counts UTF-16 code
    // units, so an unconverted span slices the wrong text out of any non-ASCII excerpt.
    const { fn } = annotated({
      highlights: [
        {
          field: "body",
          fragments: [
            { text: "café au lait", spans: [[9, 13]] },
            { text: "🦆 duck", spans: [[5, 9]] },
          ],
        },
      ],
    });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const [hit] = await db.textSearch({ field: "body", query: "lait", highlight: true });
    const [latin, emoji] = hit!.annotations!.highlights![0]!.fragments;
    expect(latin!.spans).toEqual([[8, 12]]);
    expect(latin!.text.slice(...latin!.spans[0]!)).toBe("lait");
    // A 4-byte codepoint is 2 UTF-16 units, so the shift is 2, not 3.
    expect(emoji!.spans).toEqual([[3, 7]]);
    expect(emoji!.text.slice(...emoji!.spans[0]!)).toBe("duck");
  });

  it("passes an all-ASCII fragment's spans through untouched", async () => {
    const { fn } = annotated({
      highlights: [
        { field: "body", fragments: [{ text: "the quick fox", spans: [[4, 9]] }] },
      ],
    });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const [hit] = await db.textSearch({ field: "body", query: "quick", highlight: true });
    const fragment = hit!.annotations!.highlights![0]!.fragments[0]!;
    expect(fragment.spans).toEqual([[4, 9]]);
    expect(fragment.text.slice(...fragment.spans[0]!)).toBe("quick");
  });
});

describe("memory (remember/recall)", () => {
  it("sends remember with the id/text body and normalized attrs, omitting mode", async () => {
    const { fn, calls } = mockFetch({ ok: true, upserted: 1 });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.remember("notes", "a", "the quick brown fox", {
      attrs: { tag: "x", year: 2024 },
    });
    expect(calls[0]!.url).toBe("http://x/collections/notes/remember");
    expect(calls[0]!.json).toEqual({
      id: "a",
      text: "the quick brown fox",
      attrs: { tag: { Str: "x" }, year: { Int: 2024 } },
    });
    // `mode`, `ttl_seconds`, and `dedupe_threshold` are omitted so the server's own
    // defaults ("raw", never expire, no dedupe) apply.
    const body = calls[0]!.json as Record<string, unknown>;
    expect(body.mode).toBeUndefined();
    expect(body.ttl_seconds).toBeUndefined();
    expect(body.dedupe_threshold).toBeUndefined();
    // A response that echoes no `id` leaves the requested one standing.
    expect(out).toEqual({ id: "a", upserted: 1, deduped: false });
  });

  it("maps ttlSeconds/dedupeThreshold to their snake_case wire keys", async () => {
    const { fn, calls } = mockFetch({ ok: true, upserted: 1, id: "a", deduped: false });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.remember("notes", "a", "the quick brown fox", {
      ttlSeconds: 3600,
      dedupeThreshold: 0.95,
    });
    expect(calls[0]!.json).toEqual({
      id: "a",
      text: "the quick brown fox",
      ttl_seconds: 3600,
      dedupe_threshold: 0.95,
    });
  });

  it("sends a zero ttl and a zero dedupe threshold rather than pruning them", async () => {
    const { fn, calls } = mockFetch({ ok: true, upserted: 1, id: "a", deduped: false });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.remember("notes", "a", "t", { ttlSeconds: 0, dedupeThreshold: 0 });
    expect(calls[0]!.json).toEqual({
      id: "a",
      text: "t",
      ttl_seconds: 0,
      dedupe_threshold: 0,
    });
  });

  it("reports the id a dedupe match redirected the write onto", async () => {
    const { fn } = mockFetch({ ok: true, upserted: 1, id: "older", deduped: true });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.remember("notes", "newer", "the quick brown fox", {
      dedupeThreshold: 0.9,
    });
    expect(out).toEqual({ id: "older", upserted: 1, deduped: true });
  });

  it("sends remember with mode:summarize and no attrs", async () => {
    const { fn, calls } = mockFetch({ ok: true, upserted: 1 });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.remember("notes", "b", "a long article", { mode: "summarize" });
    expect(calls[0]!.json).toEqual({
      id: "b",
      text: "a long article",
      mode: "summarize",
    });
  });

  it("sends recall with camelCase mapped to snake_case and decodes hit attrs", async () => {
    const { fn, calls } = mockFetch([
      { collection: "notes", id: "a", score: 0.99, attrs: { tag: { Str: "x" } } },
    ]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const hits = await db.recall("notes", "quick fox", {
      topK: 5,
      minScore: 0.2,
      filter: f.and(f.eq("tag", "x")),
    });
    expect(calls[0]!.url).toBe("http://x/collections/notes/recall");
    expect(calls[0]!.json).toEqual({
      query: "quick fox",
      top_k: 5,
      min_score: 0.2,
      filter: [{ Eq: ["tag", { Str: "x" }] }],
    });
    expect(hits[0]).toEqual({
      collection: "notes",
      id: "a",
      score: 0.99,
      attrs: { tag: "x" },
    });
  });

  it("sends recall with defaults: an empty filter and omitted top_k/min_score", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.recall("notes", "hello");
    expect(calls[0]!.json).toEqual({ query: "hello", filter: [] });
  });

  it("surfaces a 400 (no embedder configured) as a NidusError", async () => {
    const { fn } = mockFetch(
      { error: "nidus serve was started without an embedder; pass --embed-provider …" },
      400,
    );
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const err = (await db.recall("notes", "hi").then(
      () => null,
      (e) => e,
    )) as NidusError;
    expect(err).toBeInstanceOf(NidusError);
    expect(err.status).toBe(400);
    expect(err.isBadRequest).toBe(true);
    expect(err.message).toContain("--embed-provider");
  });
});

describe("rerank", () => {
  it("rerank is sent on all four search methods, renaming textAttr to text_attr", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const rerank = { query: "how do users sign in", overscan: 4, textAttr: "body" };
    const wire = { query: "how do users sign in", overscan: 4, text_attr: "body" };

    await db.search({ query: [1, 0, 0], rerank });
    expect(calls[0]!.json).toMatchObject({ rerank: wire });

    await db.textSearch({ field: "body", query: "fox", rerank });
    expect(calls[1]!.json).toMatchObject({ rerank: wire });

    await db.hybridSearch({ vector: [1, 0, 0], field: "body", text: "fox", rerank });
    expect(calls[2]!.json).toMatchObject({ rerank: wire });

    await db.recall("notes", "hello", { rerank });
    expect(calls[3]!.json).toMatchObject({ rerank: wire });
  });

  it("rerank is absent from the body when not asked for", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.textSearch({ field: "body", query: "fox" });
    expect("rerank" in (calls[0]!.json as object)).toBe(false);
  });

  it("an empty rerank object is sent as an empty object", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.recall("notes", "hello", { rerank: {} });
    expect((calls[0]!.json as { rerank: unknown }).rerank).toEqual({});
    await db.textSearch({ field: "body", query: "fox", rerank: {} });
    expect((calls[1]!.json as { rerank: unknown }).rerank).toEqual({});
  });
});

describe("error handling", () => {
  it("throws NidusError carrying the server status and message", async () => {
    const { fn } = mockFetch({ error: "store is locked: /tmp/s/lock" }, 409);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await expect(db.flush()).rejects.toMatchObject({
      name: "NidusError",
      status: 409,
      message: "store is locked: /tmp/s/lock",
    });
    const err = (await db.flush().then(
      () => null,
      (e) => e,
    )) as NidusError;
    expect(err.isLocked).toBe(true);
  });

  it("reports a transport failure as status 0", async () => {
    const fn = async () => {
      throw new Error("ECONNREFUSED");
    };
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const err = (await db.stats().then(
      () => null,
      (e) => e,
    )) as NidusError;
    expect(err).toBeInstanceOf(NidusError);
    expect(err.status).toBe(0);
  });
});

describe("ops surface (ready/cluster/refresh)", () => {
  it("decodes a 200 ready response and hits GET /ready", async () => {
    const { fn, calls } = mockFetch({ ready: true, role: "Leader", staleness_secs: 0 });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const res = await db.ready();
    expect(calls[0]!.url).toBe("http://x/ready");
    expect(calls[0]!.init.method).toBe("GET");
    expect(res).toEqual({ ready: true, role: "Leader", staleness_secs: 0 });
  });

  it("resolves (not rejects) on a 503, returning ready:false and the error message", async () => {
    const { fn } = mockFetch({ error: "store is stale" }, 503);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const res = await db.ready();
    expect(res).toEqual({ ready: false, reason: "store is stale" });
  });

  it("still throws NidusError for a non-503 non-2xx status", async () => {
    const { fn: fn500 } = mockFetch({ error: "internal error" }, 500);
    const db500 = new NidusClient({ baseUrl: "http://x", fetch: fn500 });
    await expect(db500.ready()).rejects.toMatchObject({
      name: "NidusError",
      status: 500,
      message: "internal error",
    });

    const { fn: fn401 } = mockFetch({ error: "unauthorized" }, 401);
    const db401 = new NidusClient({ baseUrl: "http://x", fetch: fn401 });
    await expect(db401.ready()).rejects.toMatchObject({
      name: "NidusError",
      status: 401,
    });
  });

  it("decodes all eight cluster fields, including the nullable ones", async () => {
    const { fn, calls } = mockFetch({
      role: "Follower",
      cluster: true,
      holds_writer_handle: false,
      fenced: false,
      lease_owner: null,
      commit_version: 42,
      staleness_secs: 3,
      max_staleness_secs: null,
    });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const status = await db.cluster();
    expect(calls[0]!.url).toBe("http://x/cluster");
    expect(status).toEqual({
      role: "Follower",
      cluster: true,
      holds_writer_handle: false,
      fenced: false,
      lease_owner: null,
      commit_version: 42,
      staleness_secs: 3,
      max_staleness_secs: null,
    });
  });

  it("decodes all four versions fields, including the nullable ones", async () => {
    const { fn, calls } = mockFetch({
      commit_version: 42,
      oldest_readable: 40,
      pinned: null,
      readable: [40, 41, 42],
    });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const versions = await db.versions();
    expect(calls[0]!.url).toBe("http://x/versions");
    expect(calls[0]!.init.method).toBe("GET");
    expect(calls[0]!.json).toBeUndefined();
    expect(versions).toEqual({
      commit_version: 42,
      oldest_readable: 40,
      pinned: null,
      readable: [40, 41, 42],
    });
  });

  it("posts to /refresh and returns the adopted boolean", async () => {
    const { fn: fnTrue, calls: callsTrue } = mockFetch({ adopted: true });
    const dbTrue = new NidusClient({ baseUrl: "http://x", fetch: fnTrue });
    expect(await dbTrue.refresh()).toBe(true);
    expect(callsTrue[0]!.url).toBe("http://x/refresh");
    expect(callsTrue[0]!.init.method).toBe("POST");
    expect(callsTrue[0]!.json).toEqual({});

    const { fn: fnFalse } = mockFetch({ adopted: false });
    const dbFalse = new NidusClient({ baseUrl: "http://x", fetch: fnFalse });
    expect(await dbFalse.refresh()).toBe(false);
  });
});
