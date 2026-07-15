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
    expect(encodeValue(null)).toBe("Null");
  });

  it("passes an already-tagged value through unchanged", () => {
    expect(encodeValue(v.str("x"))).toEqual({ Str: "x" });
    expect(encodeValue(v.nil())).toBe("Null");
  });

  it("rejects a non-integer number (no float attribute type)", () => {
    expect(() => encodeValue(1.5)).toThrow(TypeError);
    expect(() => v.int(1.5)).toThrow(TypeError);
  });

  it("round-trips through decode", () => {
    expect(decodeValue(encodeValue("rust") as never)).toBe("rust");
    expect(decodeValue(encodeValue(7) as never)).toBe(7);
    expect(decodeValue(encodeValue(null) as never)).toBe(null);
    expect(decodeValue(encodeValue(["a"]) as never)).toEqual(["a"]);
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
