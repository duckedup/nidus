import { spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { NidusClient, f, v } from "../src/index.js";

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
    // Poll /health until the server is up (or give up after ~5s).
    const deadline = Date.now() + 5000;
    while (Date.now() < deadline) {
      if (await db.health()) return;
      await new Promise((r) => setTimeout(r, 100));
    }
    throw new Error("nidus serve did not become healthy in time");
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

  it("deletes and reflects the change in stats", async () => {
    expect(await db.delete("docs", { ids: ["b"] })).toBe(1);
    const remaining = await db.records("docs");
    expect(remaining.map((r) => r.id)).toEqual(["a"]);
  });
});
