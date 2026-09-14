import { describe, expect, it } from "vitest";

import { NidusClient, NidusError } from "../src/index.js";

/** A fetch double that records every call and returns one canned JSON response. */
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

/** A fetch double answering one response per call, in order — for a client that only
 * ever issues one request per `query`/`compile` call, used here to prove it does. */
function mockFetchSequence(bodies: { body: unknown; status?: number }[]) {
  const calls: { url: string; init: RequestInit; json: unknown }[] = [];
  let i = 0;
  const fn = async (url: string, init: RequestInit = {}) => {
    calls.push({
      url,
      init,
      json: init.body ? JSON.parse(init.body as string) : undefined,
    });
    const next = bodies[i++];
    if (!next) throw new Error(`unexpected extra request to ${url}`);
    return new Response(JSON.stringify(next.body), {
      status: next.status ?? 200,
      headers: { "content-type": "application/json" },
    });
  };
  return { fn, calls };
}

describe("query() — transport", () => {
  it("posts {sql} to /query and nothing else", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await db.query("SELECT * FROM docs");
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe("http://x/query");
    expect(calls[0]!.json).toEqual({ sql: "SELECT * FROM docs" });
  });

  it("attaches a bearer token like every other method", async () => {
    const { fn, calls } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn, token: "sekret" });
    await db.query("SELECT * FROM docs");
    expect((calls[0]!.init.headers as Record<string, string>).authorization).toBe(
      "Bearer sekret",
    );
  });

  it("reports a transport failure as status 0, same as search()", async () => {
    const fn = async () => {
      throw new Error("ECONNREFUSED");
    };
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const err = (await db.query("SELECT * FROM docs").then(
      () => null,
      (e) => e,
    )) as NidusError;
    expect(err).toBeInstanceOf(NidusError);
    expect(err.status).toBe(0);
  });
});

describe("query() — decode parity with search()", () => {
  it("decodes a bare hits answer identically to search()'s own hit decoder", async () => {
    const wire = [
      { collection: "docs", id: "a", score: 0.9, attrs: { lang: { Str: "rust" } } },
    ];
    const { fn: searchFn } = mockFetch(wire);
    const searchDb = new NidusClient({ baseUrl: "http://x", fetch: searchFn });
    const viaSearch = await searchDb.search({ query: [1, 0, 0] });

    const { fn: queryFn } = mockFetch(wire);
    const queryDb = new NidusClient({ baseUrl: "http://x", fetch: queryFn });
    const viaQuery = await queryDb.query("SELECT * FROM docs ORDER BY knn([1,0,0])");

    expect(viaQuery).toEqual(viaSearch);
    expect(viaQuery).toEqual([
      { collection: "docs", id: "a", score: 0.9, attrs: { lang: "rust" } },
    ]);
  });

  it("decodes {hits, plan} for a WITH (plan) statement, field by field", async () => {
    const { fn } = mockFetch({
      hits: [{ collection: "docs", id: "a", score: 0.5, attrs: {} }],
      plan: {
        path: "exact",
        narrowing: { state: "inactive" },
        timings: { total_us: 10 },
      },
    });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.query("SELECT * FROM docs ORDER BY knn([1,0,0]) WITH (plan)");
    expect(out).toEqual({
      hits: [{ collection: "docs", id: "a", score: 0.5, attrs: {} }],
      plan: { path: "exact", narrowing: { state: "inactive" }, timings: { totalUs: 10 } },
    });
  });
});

describe("query() — aggregation answers", () => {
  it("decodes count/sums/groups, a missing-value (null) group, and a float sum", async () => {
    const { fn } = mockFetch({
      count: 3,
      sums: { bytes: { Float: 12.5 } },
      groups: [
        { value: { Str: "rust" }, count: 2, sums: { bytes: { Float: 10 } } },
        // The records missing the grouped attribute entirely, not a present `Null`.
        { value: null, count: 1, sums: { bytes: { Float: 2.5 } } },
      ],
    });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.query("SELECT * FROM docs GROUP BY lang, SUM(bytes)");
    expect(out).toEqual({
      count: 3,
      sums: { bytes: 12.5 },
      groups: [
        { value: "rust", count: 2, sums: { bytes: 10 } },
        { value: null, count: 1, sums: { bytes: 2.5 } },
      ],
    });
  });

  it("leaves an ungrouped aggregation exactly the shape it always was", async () => {
    const { fn } = mockFetch({ count: 3, sums: {} });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.query("SELECT * FROM docs GROUP BY lang");
    expect("groups" in (out as object)).toBe(false);
  });
});

describe("query() — batch ordering (§7.9)", () => {
  it("returns three answers in the statements' own order", async () => {
    const { fn } = mockFetch([
      [{ collection: "docs", id: "a", score: 1, attrs: {} }],
      { count: 1, sums: {} },
      [{ collection: "docs", id: "b", score: 1, attrs: {} }],
    ]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = (await db.query(
      "SELECT * FROM docs WHERE id = 'a' ORDER BY id; SELECT * FROM docs GROUP BY lang; SELECT * FROM docs WHERE id = 'b' ORDER BY id",
    )) as unknown[];
    expect(out).toHaveLength(3);
    expect((out[0] as { id: string }[])[0]!.id).toBe("a");
    expect(out[1]).toEqual({ count: 1, sums: {} });
    expect((out[2] as { id: string }[])[0]!.id).toBe("b");
  });

  it("an empty single-statement hits answer stays a bare empty array, not a batch", async () => {
    const { fn } = mockFetch([]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.query("SELECT * FROM docs WHERE id = 'nope'");
    expect(out).toEqual([]);
  });
});

describe("compile() — introspection only, executes nothing", () => {
  it("sends compile_only:true and issues exactly one request", async () => {
    const { fn, calls } = mockFetchSequence([
      {
        body: { kind: "list", collections: ["docs"], opts: { offset: null, limit: null } },
      },
    ]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.compile("SELECT * FROM docs");
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe("http://x/query");
    expect(calls[0]!.json).toEqual({ sql: "SELECT * FROM docs", compile_only: true });
    expect(out).toEqual([
      { kind: "list", collections: ["docs"], opts: { offset: null, limit: null } },
    ]);
  });

  it("always returns an array, even for a single statement", async () => {
    const { fn } = mockFetch({ kind: "aggregate", collections: [], opts: {} });
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.compile("SELECT * FROM docs GROUP BY lang");
    expect(Array.isArray(out)).toBe(true);
    expect(out).toHaveLength(1);
  });

  it("decodes a multi-statement compile as one entry per statement, in order", async () => {
    const { fn } = mockFetch([
      { kind: "list", collections: ["a"], opts: {} },
      { kind: "aggregate", collections: ["b"], opts: {} },
    ]);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const out = await db.compile("SELECT * FROM a; SELECT * FROM b GROUP BY lang");
    expect(out.map((c) => c.kind)).toEqual(["list", "aggregate"]);
  });
});

describe("query() — error surface", () => {
  it("raises NidusError carrying the server's message verbatim, byte offset and §7 tail intact", async () => {
    const message =
      "sql parse error at byte 17: expected a value after '=' (§7.3 boolean composition)";
    const { fn } = mockFetch({ error: message }, 400);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    const err = (await db.query("SELECT * FROM docs WHERE x =").then(
      () => null,
      (e) => e,
    )) as NidusError;
    expect(err).toBeInstanceOf(NidusError);
    expect(err.message).toBe(message);
    expect(err.status).toBe(400);
    expect(err.isBadRequest).toBe(true);
  });

  it("does not confuse a parse error's 400 with a 500", async () => {
    const { fn: fn400 } = mockFetch({ error: "sql parse error at byte 0: bad" }, 400);
    const db400 = new NidusClient({ baseUrl: "http://x", fetch: fn400 });
    const err400 = (await db400.query("nonsense").then(
      () => null,
      (e) => e,
    )) as NidusError;
    expect(err400.status).toBe(400);
    expect(err400.isBadRequest).toBe(true);

    const { fn: fn500 } = mockFetch({ error: "internal error" }, 500);
    const db500 = new NidusClient({ baseUrl: "http://x", fetch: fn500 });
    const err500 = (await db500.query("SELECT * FROM docs").then(
      () => null,
      (e) => e,
    )) as NidusError;
    expect(err500.status).toBe(500);
    expect(err500.isBadRequest).toBe(false);
  });

  it("compile()'s parse error carries the same verbatim message", async () => {
    const message = "sql parse error at byte 3: unexpected token (§7.12 SQL syntax)";
    const { fn } = mockFetch({ error: message }, 400);
    const db = new NidusClient({ baseUrl: "http://x", fetch: fn });
    await expect(db.compile("bad sql")).rejects.toMatchObject({
      name: "NidusError",
      status: 400,
      message,
    });
  });
});
