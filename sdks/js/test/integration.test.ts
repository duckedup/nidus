import { spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { NidusClient, NidusError, f, v } from "../src/index.js";

// End-to-end against a real `nidus serve`. Mirrors the server's own
// `full_lifecycle_over_http` test, but driven entirely through the SDK.
//
// The binary is located at $NIDUS_BIN, else `target/release/nidus` in the repo
// root (build it with `just build-cli`). If neither exists, the suite is skipped
// so a contributor without the Rust toolchain can still run the unit tests.

const repoRoot = fileURLToPath(new URL("../../..", import.meta.url));
const binary = process.env.NIDUS_BIN ?? join(repoRoot, "target/release/nidus");

let binaryExists = false;
try {
  // Resolve lazily; spawn will fail loudly if it's wrong.
  binaryExists = (await import("node:fs")).existsSync(binary);
} catch {
  binaryExists = false;
}

const PORT = 7799;
const baseUrl = `http://127.0.0.1:${PORT}`;

describe.skipIf(!binaryExists)("lifecycle over a real nidus serve", () => {
  let server: ChildProcess;
  let dir: string;
  const db = new NidusClient({ baseUrl, timeoutMs: 5000 });

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "nidus-sdk-it-"));
    server = spawn(
      binary,
      ["serve", "--dir", dir, "--dim", "3", "--addr", `127.0.0.1:${PORT}`],
      { stdio: "ignore" },
    );
    // Poll /ready, not /health (#121): health is liveness and answers before the
    // store finishes opening, so a health gate can hand tests a server that 503s.
    const deadline = Date.now() + 5000;
    let last = "";
    while (Date.now() < deadline) {
      try {
        const res = await fetch(`${baseUrl}/ready`);
        if (res.status === 200) return;
        last = `/ready answered ${res.status}`;
      } catch (e) {
        last = String(e);
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    throw new Error(`nidus serve did not become ready in time (${last})`);
  });

  afterAll(() => {
    server?.kill("SIGTERM");
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  it("create → upsert → search → stats", async () => {
    await db.createCollection("docs");
    expect(await db.collections()).toContain("docs");

    const n = await db.upsert("docs", [
      { id: "a", vector: [1, 0, 0], attrs: { lang: "rust" } },
      { id: "b", vector: [0, 1, 0], attrs: { lang: "go" } },
    ]);
    expect(n).toBe(2);

    const hits = await db.search({ query: [1, 0, 0], topK: 1 });
    expect(hits[0]!.id).toBe("a");
    expect(hits[0]!.attrs.lang).toBe("rust");

    const stats = await db.stats();
    expect(stats.dimension).toBe(3);
    expect(stats.footprint.doc_count).toBe(2);
  });

  it("filters, text search, and hybrid search", async () => {
    await db.setFtsSchema("notes", ["body"]);
    await db.upsert("notes", [
      { id: "x", vector: [1, 0, 0], attrs: { body: v.str("the quick brown fox"), kind: "a" } },
      { id: "y", attrs: { body: v.str("foxes are running quickly"), kind: "b" } },
    ]);

    const listed = await db.list({
      scope: ["notes"],
      filter: f.and(f.eq("kind", "a")),
    });
    expect(listed.map((h) => h.id)).toEqual(["x"]);

    const text = await db.textSearch({ scope: ["notes"], field: "body", query: "run", topK: 5 });
    expect(text[0]!.id).toBe("y");

    const hybrid = await db.hybridSearch({
      scope: ["notes"],
      vector: [1, 0, 0],
      field: "body",
      text: "fox",
      topK: 5,
    });
    const ids = hybrid.map((h) => h.id);
    expect(ids).toContain("x");
    expect(ids).toContain("y");
  });

  it("carries the ranking, annotation, and aggregate knobs end to end", async () => {
    const day = 86_400_000;
    const origin = 1_700_000_000_000;
    await db.createCollection("m50");
    await db.setFtsSchema("m50", ["title", "body"]);
    await db.upsert("m50", [
      {
        id: "p",
        vector: [1, 0, 0],
        attrs: {
          title: "rust vectors",
          body: "the quick brown fox",
          path: "src/a.rs",
          ts: new Date(origin),
          bytes: 100,
        },
      },
      {
        id: "q",
        vector: [0, 1, 0],
        attrs: {
          title: "go vectors",
          body: "foxes run quickly",
          path: "src/a.rs",
          ts: new Date(origin - 30 * day),
          bytes: 200,
        },
      },
    ]);

    const hits = await db.textSearch({
      scope: ["m50"],
      clauses: [
        { field: "title", query: "rust" },
        { field: "body", query: "fox" },
      ],
      combine: "Sum",
      explain: true,
      highlight: { maxFragments: 1, fragmentChars: 40 },
    });
    expect(hits[0]!.id).toBe("p");
    const annotations = hits[0]!.annotations!;
    expect(annotations.clauses!.map((c) => c.field)).toContain("title");
    const body = annotations.highlights!.find((h) => h.field === "body")!;
    const fragment = body.fragments[0]!;
    expect(fragment.text.slice(...fragment.spans[0]!)).toBe("fox");

    const scoped = { scope: ["m50"] };
    const ids = async (filter: ReturnType<typeof f.and>) =>
      (await db.list({ ...scoped, filter })).map((h) => h.id);
    expect(await ids(f.and(f.fuzzy("title", "rast vectors", 1)))).toEqual(["p"]);
    expect(await ids(f.and(f.containsTokenSequence("body", "brown fox")))).toEqual(["p"]);
    expect(await ids(f.and(f.containsAnyToken("body", "fox")))).toEqual(["p"]);
    expect(await ids(f.and(f.regex("path", "src/.*\\.rs")))).toEqual(["p", "q"]);

    const ordered = await db.list({
      ...scoped,
      orderBy: { field: "bytes", descending: true },
    });
    expect(ordered.map((h) => h.id)).toEqual(["q", "p"]);

    // Both records share one `path`, so a cap of 1 keeps only the better-scoring one.
    const capped = await db.search({
      ...scoped,
      query: [1, 0, 0],
      limitPer: { field: "path", max: 1 },
    });
    expect(capped.map((h) => h.id)).toEqual(["p"]);

    // Equally similar, but `q` is 30 days older — decay is what separates them.
    const decayed = await db.search({
      ...scoped,
      query: [1, 1, 0],
      rankBy: { decay: { field: "ts", origin, scale: 7 * day } },
    });
    expect(decayed[0]!.id).toBe("p");

    expect(await db.aggregate({ ...scoped, sum: ["bytes"] })).toEqual({
      count: 2,
      sums: { bytes: 300 },
    });

    const hybrid = await db.hybridSearch({
      ...scoped,
      vector: [1, 0, 0],
      clauses: [{ field: "body", query: "fox" }],
      textWeight: 2,
      explain: true,
    });
    expect(hybrid[0]!.annotations!.text).toBeDefined();
  });

  it("deletes and reflects the change in stats", async () => {
    expect(await db.delete("docs", { ids: ["b"] })).toBe(1);
    const remaining = await db.records("docs");
    expect(remaining.map((r) => r.id)).toEqual(["a"]);
  });

  // 404 when the binary lacks the `memory` feature (what `just build-cli` builds, so
  // the usual case here), 400 when the routes exist but no `--embed-provider` was
  // given. Either way the new options travel and the call fails visibly, with a status.
  it("fails visibly with a status when the server has no embedder", async () => {
    const err = (await db
      .remember("notes", "a", "the quick brown fox", {
        ttlSeconds: 3600,
        dedupeThreshold: 0.95,
      })
      .then(
        () => null,
        (e) => e,
      )) as NidusError;
    expect(err).toBeInstanceOf(NidusError);
    expect([400, 404]).toContain(err.status);
  });
});
