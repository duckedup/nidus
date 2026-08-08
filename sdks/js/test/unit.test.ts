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

describe("memory (remember/recall)", () => {
  it("sends remember with the id/text body and normalized attrs, omitting mode", async () => {
    const { fn, calls } = mockFetch({ ok: true, upserted: 1 });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.remember("notes", "a", "the quick brown fox", {
      attrs: { tag: "x", year: 2024 },
    });
    expect(out).toBeUndefined();
    expect(calls[0]!.url).toBe("http://x/collections/notes/remember");
    expect(calls[0]!.json).toEqual({
      id: "a",
      text: "the quick brown fox",
      attrs: { tag: { Str: "x" }, year: { Int: 2024 } },
    });
    // `mode` is omitted so the server default ("raw") applies.
    expect((calls[0]!.json as Record<string, unknown>).mode).toBeUndefined();
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
