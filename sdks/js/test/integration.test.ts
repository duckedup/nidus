import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { NidusClient, NidusError, decodeAttrs, decodeValue, f, v } from "../src/index.js";
import type { Value } from "../src/index.js";

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

// The lane that runs this (`just sdk-js-test-all`, CI's sdk-integration job) builds
// `--features serve`, which includes `code`, so the subcommand must be there. Detected once
// up front — synchronously, so it can fail loudly below rather than silently skipping: a
// skipped contract test reads as a pass, which is how a broken route ships (nidus-3gm).
let codeFeatureAvailable = false;
if (binaryExists) {
  try {
    codeFeatureAvailable =
      spawnSync(binary, ["code", "--help"], { stdio: "ignore" }).status === 0;
  } catch {
    codeFeatureAvailable = false;
  }
}
if (binaryExists && !codeFeatureAvailable) {
  throw new Error(
    `${binary} has no \`code\` subcommand, so the codeSearch contract cannot be tested. ` +
      "Build it with `--features serve` (just build-serve-bin) and point NIDUS_BIN at it. " +
      "This throws rather than skipping: a skipped contract test reads as a pass.",
  );
}
const CODE_PORT = 7798;
const codeBaseUrl = `http://127.0.0.1:${CODE_PORT}`;

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

  // Declaring names, then searching both, reduces a multi-vector record to **one** hit
  // whose score is the `Sum`-pooled combination, different from either name's score alone.
  it("named vectors reduce to one hit, scored by pooling", async () => {
    await db.createCollection("named_docs");
    await db.setVectorNames("named_docs", ["title", "body"]);
    await db.upsert("named_docs", [
      { id: "r1", vectors: { title: [1, 0, 0], body: [0, 1, 0] }, attrs: {} },
    ]);

    const query = [0.6, 0.8, 0];
    const scope = ["named_docs"];
    const titleOnly = await db.search({ scope, query, topK: 5, names: ["title"] });
    const bodyOnly = await db.search({ scope, query, topK: 5, names: ["body"] });
    const both = await db.search({
      scope,
      query,
      topK: 5,
      names: ["title", "body"],
      pool: "Sum",
    });

    expect(both.length).toBe(1);
    expect(both[0]!.id).toBe("r1");
    const titleScore = titleOnly[0]!.score;
    const bodyScore = bodyOnly[0]!.score;
    const pooledScore = both[0]!.score;
    expect(Math.abs(pooledScore - (titleScore + bodyScore))).toBeLessThan(1e-4);
    expect(pooledScore).not.toBe(titleScore);
    expect(pooledScore).not.toBe(bodyScore);
  });

  // The same query at two different per-name weightings flips which record ranks first,
  // proving the weights are actually applied rather than dropped.
  it("per-name weights change the top hit", async () => {
    await db.createCollection("named_weights");
    await db.setVectorNames("named_weights", ["title", "body"]);
    await db.upsert("named_weights", [
      { id: "titled", vectors: { title: [1, 0, 0] }, attrs: {} },
      { id: "bodied", vectors: { body: [1, 0, 0] }, attrs: {} },
    ]);

    const scope = ["named_weights"];
    const titleHeavy = await db.search({
      scope,
      query: [1, 0, 0],
      topK: 1,
      names: ["title", "body"],
      nameWeights: { title: 2, body: 1 },
    });
    expect(titleHeavy[0]!.id).toBe("titled");

    const bodyHeavy = await db.search({
      scope,
      query: [1, 0, 0],
      topK: 1,
      names: ["title", "body"],
      nameWeights: { title: 1, body: 2 },
    });
    expect(bodyHeavy[0]!.id).toBe("bodied");
  });

  // A search naming no vectors must still rank on `vector` (the reserved `default`)
  // alone, unaffected by a record that also carries named vectors.
  it("a search naming no vectors ignores named vectors on the same record", async () => {
    await db.createCollection("named_default");
    await db.setVectorNames("named_default", ["title"]);
    await db.upsert("named_default", [
      { id: "r1", vector: [1, 0, 0], vectors: { title: [0, 1, 0] }, attrs: {} },
    ]);

    const scope = ["named_default"];
    const matching = await db.search({ scope, query: [1, 0, 0], topK: 5 });
    expect(matching[0]!.id).toBe("r1");
    expect(Math.abs(matching[0]!.score - 1.0)).toBeLessThan(1e-4);

    // Orthogonal to `vector` but exactly matches the named `title` vector: a search
    // naming no vectors must not fall back to it.
    const orthogonal = await db.search({ scope, query: [0, 1, 0], topK: 5 });
    expect(orthogonal[0]!.id).toBe("r1");
    expect(Math.abs(orthogonal[0]!.score)).toBeLessThan(1e-4);
  });

  // Upserting a name that was never declared is a 400 naming the offending name, not a
  // silent write.
  it("upserting an undeclared vector name is refused, naming it", async () => {
    await db.createCollection("named_undeclared");
    const err = (await db
      .upsert("named_undeclared", [{ id: "r1", vectors: { nope: [1, 0, 0] }, attrs: {} }])
      .then(
        () => null,
        (e) => e,
      )) as NidusError;
    expect(err).toBeInstanceOf(NidusError);
    expect(err.status).toBe(400);
    expect(err.message).toContain("nope");
  });

  // nidus-29ui: `limitPer` on `hybridSearch` caps hits per attribute value, returning a
  // **different hit set** (not just a shorter page) than the same query unset.
  it("hybridSearch limitPer returns a different hit set, not just a shorter page", async () => {
    await db.createCollection("hybrid_limit");
    await db.setFtsSchema("hybrid_limit", ["body"]);
    await db.upsert("hybrid_limit", [
      { id: "a1", vector: [1, 0, 0], attrs: { src: "A", body: "cats and foxes" } },
      { id: "a2", vector: [0.9, 0.1, 0], attrs: { src: "A", body: "cats and foxes" } },
      { id: "a3", vector: [0.8, 0.2, 0], attrs: { src: "A", body: "cats and foxes" } },
      { id: "b1", vector: [0.7, 0.3, 0], attrs: { src: "B", body: "cats and foxes" } },
      { id: "b2", vector: [0.6, 0.4, 0], attrs: { src: "B", body: "cats and foxes" } },
    ]);

    const scope = ["hybrid_limit"];
    const unset = await db.hybridSearch({
      scope,
      vector: [1, 0, 0],
      field: "body",
      text: "cats",
      topK: 5,
    });
    expect(unset.length).toBe(5);

    const capped = await db.hybridSearch({
      scope,
      vector: [1, 0, 0],
      field: "body",
      text: "cats",
      topK: 5,
      limitPer: { field: "src", max: 1 },
    });
    expect(capped.length).toBe(2);
    expect(capped.map((h) => h.id)).not.toEqual(unset.map((h) => h.id));
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

  it("fuzzy fallback completes a mistyped prefix by default, and fuzzy: false opts out", async () => {
    await db.createCollection("m53");
    await db.setFtsSchema("m53", ["body"]);
    await db.upsert("m53", [
      { id: "a", vector: [1, 0, 0], attrs: { body: "running quickly" } },
    ]);

    // "runing" has no exact-prefix match at all, so the fuzzy fallback fires by default.
    const byDefault = await db.suggest({ scope: ["m53"], field: "body", prefix: "runing" });
    expect(byDefault.suggestions.map((s) => s.term)).toEqual(["running"]);

    const optedOut = await db.suggest({
      scope: ["m53"],
      field: "body",
      prefix: "runing",
      fuzzy: false,
    });
    expect(optedOut.suggestions).toEqual([]);
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

// A known fixture: one `pub fn add`, so the AST chunker's start/end line span is
// predictable (the doc comment sits outside the `function_item` node, so the span starts
// at `pub fn`, not at `///`).
const CODE_FIXTURE = `/// Adds two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
`;

describe.skipIf(!codeFeatureAvailable)("codeSearch over a code-featured nidus serve", () => {
  let server: ChildProcess;
  let sourceDir: string;
  let storeDir: string;
  const db = new NidusClient({ baseUrl: codeBaseUrl, timeoutMs: 5000 });

  beforeAll(async () => {
    sourceDir = mkdtempSync(join(tmpdir(), "nidus-sdk-code-src-"));
    writeFileSync(join(sourceDir, "sample.rs"), CODE_FIXTURE);
    storeDir = mkdtempSync(join(tmpdir(), "nidus-sdk-code-store-"));

    // `nidus code ingest` opens the store itself (no embedder configured, so it ingests
    // for BM25 only); it must finish, and close its handle, before `serve` opens the same
    // directory below.
    const ingested = spawnSync(
      binary,
      ["code", "ingest", sourceDir, "--dir", storeDir, "--collection", "code"],
      { encoding: "utf8" },
    );
    if (ingested.status !== 0) {
      throw new Error(`nidus code ingest failed (${ingested.status}): ${ingested.stderr}`);
    }

    server = spawn(binary, ["serve", "--dir", storeDir, "--addr", `127.0.0.1:${CODE_PORT}`], {
      stdio: "ignore",
    });
    const deadline = Date.now() + 5000;
    let last = "";
    while (Date.now() < deadline) {
      try {
        const res = await fetch(`${codeBaseUrl}/ready`);
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
    if (server) await stopServer(server);
    if (storeDir) rmSync(storeDir, { recursive: true, force: true });
    if (sourceDir) rmSync(sourceDir, { recursive: true, force: true });
  });

  // The load-bearing assertion (#172): a 200 with an empty `files` array would also pass
  // a shape check, so this proves the round trip through a real `code ingest` and a real
  // `/code-search` by asserting the exact symbol name and line span.
  it("finds the ingested symbol by name, with its file, language and line span", async () => {
    const files = await db.codeSearch({ collection: "code", query: "add", vector: false });

    const file = files.find((f) => f.path === "sample.rs");
    expect(file).toBeDefined();
    expect(file!.language).toBe("rust");

    const symbol = file!.symbols.find((s) => s.symbol === "add");
    expect(symbol).toBeDefined();
    expect(symbol!.kind).toBe("function");
    expect(symbol!.startLine).toBe(2);
    expect(symbol!.endLine).toBe(4);
  });
});

// ── SQL conformance corpus (nidus-yq9p.2) ───────────────────────────────────
//
// Replays `tests/corpus/queries.json` (shape and workflow: `tests/corpus/README.md`) over
// this SDK's own `query()`, and asserts it returns identical ordered ids to the equivalent
// typed `dsl` request run directly over HTTP. Read the file; never transcribe the cases.

interface CorpusFixtureRecord {
  id: string;
  vector: number[];
  attrs: Record<string, unknown>;
}

interface CorpusCase {
  name: string;
  section: string;
  sql: string;
  dsl: { endpoint: string; body: unknown } | null;
  expect: {
    ids?: string[];
    aggregation?: {
      count: number;
      sums: Record<string, Value>;
      groups?: { value: Value | null; count: number; sums: Record<string, Value> }[];
    };
    batch_ids?: string[][];
  };
  surfaces: string[];
  note?: string;
}

interface Corpus {
  fixture: {
    dim: number;
    fts: Record<string, string[]>;
    collections: Record<string, CorpusFixtureRecord[]>;
  };
  cases: CorpusCase[];
}

const corpus: Corpus = JSON.parse(
  readFileSync(join(repoRoot, "tests/corpus/queries.json"), "utf8"),
);
const jsCases = corpus.cases.filter((c) => c.surfaces.includes("js"));

const CORPUS_PORT = 7797;
const corpusBaseUrl = `http://127.0.0.1:${CORPUS_PORT}`;

/** A ranked/listed answer's ordered ids, whichever of the two `/query` shapes it took. */
function idsFrom(answer: unknown): string[] {
  if (Array.isArray(answer)) return (answer as { id: string }[]).map((h) => h.id);
  const withPlan = answer as { hits?: { id: string }[] };
  if (withPlan.hits) return withPlan.hits.map((h) => h.id);
  throw new Error(`not a hits answer: ${JSON.stringify(answer)}`);
}

/** Normalize a raw (still tagged-`Value`) `/aggregate` response for comparison, dropping
 * `groups_truncated` per the corpus README (every case here keeps it `false`). */
function normalizeAggregation(raw: {
  count: number;
  sums: Record<string, Value>;
  groups?: { value: Value | null; count: number; sums: Record<string, Value> }[];
}) {
  return {
    count: raw.count,
    sums: decodeAttrs(raw.sums) as Record<string, number>,
    groups: (raw.groups ?? []).map((g) => ({
      value: g.value === null ? null : decodeValue(g.value),
      count: g.count,
      sums: decodeAttrs(g.sums) as Record<string, number>,
    })),
  };
}

describe.skipIf(!binaryExists || jsCases.length === 0)("SQL conformance corpus", () => {
  let server: ChildProcess;
  let dir: string;
  const db = new NidusClient({ baseUrl: corpusBaseUrl, timeoutMs: 5000 });

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "nidus-sdk-sql-"));
    server = spawn(
      binary,
      [
        "serve",
        "--dir",
        dir,
        "--dim",
        String(corpus.fixture.dim),
        "--addr",
        `127.0.0.1:${CORPUS_PORT}`,
      ],
      { stdio: "ignore" },
    );
    const deadline = Date.now() + 5000;
    let last = "";
    while (Date.now() < deadline) {
      try {
        const res = await fetch(`${corpusBaseUrl}/ready`);
        if (res.status === 200) break;
        last = `/ready answered ${res.status}`;
      } catch (e) {
        last = String(e);
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    if (Date.now() >= deadline) throw new Error(`nidus serve did not become ready in time (${last})`);

    // Seed every fixture collection over raw HTTP, attrs passed through as the corpus's own
    // tagged wire form — the one write path the whole corpus uses (mirrors `tests/e2e/corpus.rs`).
    for (const [name, records] of Object.entries(corpus.fixture.collections)) {
      await fetch(`${corpusBaseUrl}/collections/${name}`, { method: "POST", body: "{}" });
      const fields = corpus.fixture.fts[name];
      if (fields) {
        await fetch(`${corpusBaseUrl}/collections/${name}/fts-schema`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ fields }),
        });
      }
      const res = await fetch(`${corpusBaseUrl}/collections/${name}/upsert`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ records }),
      });
      if (!res.ok) {
        throw new Error(`seeding ${name} failed: ${await res.text()}`);
      }
    }
  });

  afterAll(async () => {
    if (server) await stopServer(server);
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  for (const c of jsCases) {
    it(`${c.section} ${c.name}`, async () => {
      if (c.expect.batch_ids) {
        // §7.9 multi-query batching: no typed endpoint runs a script, so this is checked
        // directly against `expect.batch_ids` rather than a `dsl` twin.
        const answers = (await db.query(c.sql)) as unknown[];
        expect(answers.map((a) => idsFrom(a))).toEqual(c.expect.batch_ids);
        return;
      }

      const sqlAnswer = await db.query(c.sql);
      if (!c.dsl) throw new Error(`case ${c.name} has no dsl and is not a batch case`);
      const dslRes = await fetch(`${corpusBaseUrl}${c.dsl.endpoint}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(c.dsl.body),
      });
      const dslBody = await dslRes.json();
      if (!dslRes.ok) {
        throw new Error(`dsl twin for ${c.name} failed: ${JSON.stringify(dslBody)}`);
      }

      if (c.expect.aggregation) {
        const sqlAgg = sqlAnswer as {
          count: number;
          sums: Record<string, number>;
          groups?: unknown[];
        };
        expect({ ...sqlAgg, groups: sqlAgg.groups ?? [] }).toEqual(
          normalizeAggregation(dslBody),
        );
        expect({ ...sqlAgg, groups: sqlAgg.groups ?? [] }).toEqual(
          normalizeAggregation(c.expect.aggregation!),
        );
        return;
      }

      const sqlIds = idsFrom(sqlAnswer);
      const dslIds = idsFrom(dslBody);
      expect(sqlIds).toEqual(dslIds);
      if (c.expect.ids) expect(sqlIds).toEqual(c.expect.ids);
    });
  }
});
