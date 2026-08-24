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

// SIGTERM, then SIGKILL if it will not go — mirroring the Go suite's 5s escalation
// and Python's `wait(timeout=10)`. Returns only once the child has actually exited.
async function stopServer(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const exited = new Promise<void>((resolve) => child.once("exit", () => resolve()));
  child.kill("SIGTERM");
  let timer: ReturnType<typeof setTimeout> | undefined;
  await Promise.race([
    exited,
    new Promise<void>((resolve) => {
      timer = setTimeout(resolve, 5000);
    }),
  ]);
  if (timer) clearTimeout(timer);
  if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
  await exited;
}

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

  afterAll(async () => {
    // Wait for exit before removing the directory: a graceful shutdown persists the
    // ann/fts caches into it, so tearing it down mid-write races the writer and fails
    // with ENOTEMPTY. The Go and Python suites already wait; this one did not.
    if (server) await stopServer(server);
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  it("reports readiness, cluster status, and refresh over the real server", async () => {
    const readiness = await db.ready();
    expect(readiness.ready).toBe(true);
    expect(typeof readiness.role).toBe("string");
    expect(readiness.role!.length).toBeGreaterThan(0);
    expect(typeof readiness.staleness_secs).toBe("number");

    const status = await db.cluster();
    expect(typeof status.role).toBe("string");
    expect(typeof status.cluster).toBe("boolean");
    expect(typeof status.holds_writer_handle).toBe("boolean");
    expect(typeof status.fenced).toBe("boolean");
    expect(typeof status.commit_version).toBe("number");
    expect(typeof status.staleness_secs).toBe("number");

    expect(typeof (await db.refresh())).toBe("boolean");

    const versions = await db.versions();
    expect(typeof versions.commit_version).toBe("number");
    expect(versions.pinned).toBeNull();
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

  it("searchSimilar finds a near neighbour and excludes the source record", async () => {
    await db.upsert("docs", [
      { id: "c", vector: [0.9, 0.1, 0], attrs: { lang: "rust-ish" } },
    ]);

    const hits = await db.searchSimilar({ collection: "docs", id: "a", topK: 10 });
    expect(hits.some((h) => h.id === "a")).toBe(false);
    expect(hits.some((h) => h.id === "c")).toBe(true);

    // This suite shares one server, and a later case asserts the exact contents
    // of `docs`, so the fixture must not outlive the test that added it.
    expect(await db.delete("docs", { ids: ["c"] })).toBe(1);
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

  // A truncated query matches only with `prefix: true` — asserting solely the positive case
  // would pass against a client that drops the field entirely.
  it("prefix expands a truncated clause's final term, on both spellings", async () => {
    await db.createCollection("m51");
    await db.setFtsSchema("m51", ["title"]);
    await db.upsert("m51", [
      { id: "a", vector: [1, 0, 0], attrs: { title: "running quickly" } },
    ]);

    const noPrefix = await db.textSearch({
      scope: ["m51"],
      field: "title",
      query: "ru",
    });
    expect(noPrefix.map((h) => h.id)).toEqual([]);

    const shorthand = await db.textSearch({
      scope: ["m51"],
      field: "title",
      query: "ru",
      prefix: true,
    });
    expect(shorthand.map((h) => h.id)).toEqual(["a"]);

    const clauseForm = await db.textSearch({
      scope: ["m51"],
      clauses: [{ field: "title", query: "ru", prefix: true }],
    });
    expect(clauseForm.map((h) => h.id)).toEqual(["a"]);
  });

  // "run" and "runner" stem to themselves (Porter leaves both alone), so their df
  // ordering is asserted exactly rather than hedged around stemming.
  it("suggest ranks completions by document frequency", async () => {
    await db.createCollection("m52");
    await db.setFtsSchema("m52", ["body"]);
    await db.upsert("m52", [
      { id: "a", vector: [1, 0, 0], attrs: { body: "I run every morning" } },
      { id: "b", vector: [0, 1, 0], attrs: { body: "run run run" } },
      { id: "c", vector: [0, 0, 1], attrs: { body: "they run too" } },
      { id: "d", vector: [1, 1, 0], attrs: { body: "a runner races" } },
    ]);

    const result = await db.suggest({ scope: ["m52"], field: "body", prefix: "run", limit: 10 });
    expect(result.suggestions.map((s) => s.term)).toEqual(["run", "runner"]);
    expect(result.suggestions.map((s) => s.df)).toEqual([3, 1]);
    expect(result.matched).toBe(2);

    // The words before the fragment narrow it: only "runner" shares a document with "races".
    const phrase = await db.suggest({ scope: ["m52"], field: "body", prefix: "races run" });
    expect(phrase.suggestions.map((s) => s.term)).toEqual(["runner"]);

    // And a filter narrows each completion's df to the matching documents.
    const filtered = await db.suggest({
      scope: ["m52"],
      field: "body",
      prefix: "run",
      filter: [{ Eq: ["id", { Str: "b" }] }],
    });
    expect(filtered.suggestions).toEqual([]);
  });

  it("sets an alias and searches through it to the concrete collection", async () => {
    await db.createCollection("docs_v2");
    await db.upsert("docs_v2", [{ id: "z", vector: [1, 0, 0], attrs: { lang: "rust-v2" } }]);
    await db.setAlias("docs_alias", "docs_v2");
    expect(await db.aliases()).toMatchObject({ docs_alias: "docs_v2" });

    const hits = await db.search({ scope: ["docs_alias"], query: [1, 0, 0], topK: 1 });
    expect(hits[0]!.id).toBe("z");
    expect(hits[0]!.collection).toBe("docs_v2");

    await db.dropAlias("docs_alias");
    expect(await db.aliases()).not.toHaveProperty("docs_alias");
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

  // This binary is built with `--features cli` only (no embedder), so `reinforce`
  // and `extendTtlSeconds` cannot be proven end-to-end here (that needs `memory`
  // plus `--embed-provider`, which this harness does not wire up). What this proves:
  // the two new options travel to the server without a client-side error, and the
  // call still fails visibly, with a status, exactly like a plain recall does above.
  it("recall with reinforce fails visibly with a status when the server has no embedder", async () => {
    const err = (await db
      .recall("notes", "the quick brown fox", { reinforce: true, extendTtlSeconds: 3600 })
      .then(
        () => null,
        (e) => e,
      )) as NidusError;
    expect(err).toBeInstanceOf(NidusError);
    expect([400, 404]).toContain(err.status);
  });
});
