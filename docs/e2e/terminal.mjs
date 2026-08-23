#!/usr/bin/env bun
// nidus-7pj: drives the REAL docs terminal over raw W3C WebDriver and asserts a
// live wasm search ranking, not just that the page rendered. See
// docs/BLUEPRINT-nidus-7pj.md. Group 3b (scripts/e2e-wasm.sh) serves the site
// and invokes: `bun docs/e2e/terminal.mjs <base-url>`.

import { spawn } from "node:child_process";
import assert from "node:assert/strict";

// Observed against the seeded corpus for query "token rotation" (docs/src/terminal/corpus.js
// EXAMPLE_OUTPUT). Declared as literals, not recomputed, so this test can catch a real
// ranking regression instead of re-deriving the answer with the code under test.
// A query that shares NO words with its top hit: it ranks through the concept map, so
// this assertion fails if the vectors, the normalisation or the ranking regress.
// Observed values, not recomputed here: recomputing with the function under test would
// assert nothing.
const EXPECTED_TOP = "durability";
const EXPECTED_SECOND = "recovery";
const QUERY = "how do I keep data safe if the process dies";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(fn, { timeoutMs, label, intervalMs = 250 }) {
  const start = Date.now();
  let lastErr;
  while (Date.now() - start < timeoutMs) {
    try {
      const result = await fn();
      if (result) return result;
    } catch (e) {
      lastErr = e;
    }
    await sleep(intervalMs);
  }
  const suffix = lastErr ? `: ${lastErr.message}` : "";
  throw new Error(`timed out after ${timeoutMs}ms waiting for ${label}${suffix}`);
}

function detectDriver() {
  const candidates = [
    { env: "SAFARIDRIVER", kind: "safari" },
    { env: "GECKODRIVER", kind: "firefox" },
    { env: "CHROMEDRIVER", kind: "chrome" },
  ];
  for (const { env, kind } of candidates) {
    const path = process.env[env];
    if (path) return { kind, path };
  }
  throw new Error(
    "no driver set: export one of SAFARIDRIVER, GECKODRIVER, CHROMEDRIVER to a driver binary path",
  );
}

function driverArgs(kind, port) {
  if (kind === "safari") return ["-p", String(port)];
  if (kind === "firefox") return ["--port", String(port)];
  return [`--port=${port}`]; // chromedriver wants --port=PORT, not a space
}

function capabilitiesFor(kind) {
  if (kind === "chrome") {
    return { browserName: "chrome", "goog:chromeOptions": { args: ["--headless=new", "--no-sandbox"] } };
  }
  if (kind === "firefox") {
    return { browserName: "firefox", "moz:firefoxOptions": { args: ["-headless"] } };
  }
  return { browserName: "safari" }; // no headless mode; local-macOS path only
}

async function driverReady(port) {
  try {
    const res = await fetch(`http://127.0.0.1:${port}/status`);
    if (!res.ok) return false;
    const body = await res.json();
    return body?.value?.ready !== false;
  } catch {
    return false;
  }
}

async function startDriver({ kind, path }) {
  for (let attempt = 0; attempt < 5; attempt++) {
    const port = 9200 + Math.floor(Math.random() * 900);
    const proc = spawn(path, driverArgs(kind, port), { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    proc.stderr?.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    const exited = new Promise((resolve) => proc.once("exit", (code) => resolve(code)));
    try {
      await waitFor(() => driverReady(port), { timeoutMs: 10000, label: `${kind} driver on port ${port}` });
      return { proc, port, kind };
    } catch (e) {
      proc.kill("SIGKILL");
      await Promise.race([exited, sleep(500)]);
      if (attempt === 4) {
        throw new Error(`could not start ${kind} driver after 5 attempts: ${e.message}\n${stderr}`);
      }
    }
  }
  throw new Error("unreachable");
}

function elementRef(value) {
  const key = Object.keys(value).find((k) => k.startsWith("element-")) ?? "ELEMENT";
  return value[key];
}

class Session {
  constructor(driverUrl) {
    this.driverUrl = driverUrl;
    this.sessionId = null;
  }

  async call(method, path, body) {
    const res = await fetch(`${this.driverUrl}${path}`, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const text = await res.text();
    let json = {};
    try {
      json = text ? JSON.parse(text) : {};
    } catch {
      // non-JSON body; fall through with res.ok check below
    }
    if (!res.ok) {
      throw new Error(json?.value?.message || text || `HTTP ${res.status} on ${method} ${path}`);
    }
    return json;
  }

  async open(capabilities) {
    const res = await this.call("POST", "/session", { capabilities: { alwaysMatch: capabilities } });
    this.sessionId = res.value.sessionId;
    return this.sessionId;
  }

  get base() {
    return `/session/${this.sessionId}`;
  }

  async close() {
    if (!this.sessionId) return;
    try {
      await this.call("DELETE", this.base);
    } catch {
      // best-effort teardown
    }
  }

  navigate(url) {
    return this.call("POST", `${this.base}/url`, { url });
  }

  async findElement(selector) {
    const res = await this.call("POST", `${this.base}/element`, { using: "css selector", value: selector });
    return elementRef(res.value);
  }

  async findElements(selector) {
    const res = await this.call("POST", `${this.base}/elements`, { using: "css selector", value: selector });
    return res.value.map(elementRef);
  }

  click(el) {
    return this.call("POST", `${this.base}/element/${el}/click`, {});
  }

  sendKeys(el, text) {
    return this.call("POST", `${this.base}/element/${el}/value`, { text });
  }

  async getAttribute(el, name) {
    const res = await this.call("GET", `${this.base}/element/${el}/attribute/${name}`);
    return res.value;
  }

  async getText(el) {
    const res = await this.call("GET", `${this.base}/element/${el}/text`);
    return res.value;
  }
}

async function main() {
  const baseUrl = process.argv[2];
  if (!baseUrl) {
    throw new Error("usage: bun docs/e2e/terminal.mjs <base-url>");
  }

  const driverInfo = detectDriver();
  const { proc, port, kind } = await startDriver(driverInfo);
  const session = new Session(`http://127.0.0.1:${port}`);

  try {
    await session.open(capabilitiesFor(kind));
    await session.navigate(baseUrl);

    const runBtn = await waitFor(() => session.findElement("#nd-term-run"), {
      timeoutMs: 15000,
      label: "#nd-term-run to appear",
    });
    await session.click(runBtn);

    // The wasm module is ~485 KB plus compile; give it a generous budget on a shared runner.
    const modeEl = await waitFor(() => session.findElement("#nd-term-mode"), {
      timeoutMs: 35000,
      label: "#nd-term-mode after clicking run (wasm load)",
    });
    const mode = await session.getAttribute(modeEl, "data-mode");
    console.log(`mode: ${mode}`);
    assert.ok(mode === "opfs" || mode === "memory", `unexpected data-mode: ${mode}`);

    const inputEl = await waitFor(() => session.findElement("#nd-term-input"), {
      timeoutMs: 5000,
      label: "#nd-term-input to appear after boot",
    });
    await session.click(inputEl);
    const enterKey = String.fromCharCode(0xe007); // WebDriver's normalized "Enter" key value
    await session.sendKeys(inputEl, `search ${QUERY}${enterKey}`);

    const rows = await waitFor(
      async () => {
        const found = await session.findElements("#nd-term-output .nd-term__hit");
        return found.length >= 2 ? found : null;
      },
      { timeoutMs: 10000, label: "search results (>=2 hits) in #nd-term-output" },
    );

    const ids = await Promise.all(rows.map((r) => session.getAttribute(r, "data-id")));
    const scoreEls = await session.findElements("#nd-term-output .nd-term__score");
    const scoreTexts = await Promise.all(scoreEls.map((s) => session.getText(s)));
    const scores = scoreTexts.map((t) => parseFloat(t));

    let outputText = "";
    try {
      const outputEl = await session.findElement("#nd-term-output");
      outputText = await session.getText(outputEl);
    } catch {
      // best-effort diagnostic only
    }

    try {
      assert.ok(ids.length >= 2, `expected at least 2 hits, got ${ids.length}`);
      assert.equal(ids[0], EXPECTED_TOP, `top hit: expected ${EXPECTED_TOP}, got ${ids[0]}`);
      assert.equal(ids[1], EXPECTED_SECOND, `second hit: expected ${EXPECTED_SECOND}, got ${ids[1]}`);
      for (let i = 1; i < scores.length; i++) {
        assert.ok(
          scores[i] <= scores[i - 1],
          `scores not descending at index ${i}: ${scores[i - 1]} then ${scores[i]}`,
        );
      }
    } catch (e) {
      console.error(`FAIL: ${e.message}`);
      console.error(`ids: ${JSON.stringify(ids)}`);
      console.error(`scores: ${JSON.stringify(scores)}`);
      console.error(`terminal output:\n${outputText}`);
      throw e;
    }

    // `similar <id>` searches by a stored record's own vector, and must never return
    // that record: an off-by-one in the exclusion would silently make it hit #1.
    await session.click(inputEl);
    await session.sendKeys(inputEl, `similar ${EXPECTED_TOP}${enterKey}`);
    const simRows = await waitFor(
      async () => {
        const found = await session.findElements("#nd-term-output .nd-term__hit");
        return found.length > rows.length ? found.slice(rows.length) : null;
      },
      { timeoutMs: 10000, label: `hits for "similar ${EXPECTED_TOP}"` },
    );
    const simIds = await Promise.all(simRows.map((r) => session.getAttribute(r, "data-id")));
    if (!simIds.length || simIds.includes(EXPECTED_TOP)) {
      console.error(`FAIL: similar ${EXPECTED_TOP} returned ${JSON.stringify(simIds)}`);
      throw new Error("similar: expected neighbours, and never the queried record itself");
    }
    console.log(`similar ${EXPECTED_TOP}: ${JSON.stringify(simIds)}`);

    // The headline claim: reload, reopen, and the records are still there. Only
    // meaningful on the OPFS path, so the in-memory fallback skips it rather than
    // failing for a reason the ticket explicitly permits.
    if (mode === "opfs") {
      await session.navigate(baseUrl);
      const runAgain = await waitFor(() => session.findElement("#nd-term-run").catch(() => null), {
        timeoutMs: 15000,
        label: "#nd-term-run after reload",
      });
      await session.click(runAgain);
      const restoredEl = await waitFor(
        () => session.findElement(".nd-term__restored").catch(() => null),
        { timeoutMs: 35000, label: ".nd-term__restored after reload (persistence)" },
      );
      const restoredText = await session.getText(restoredEl);
      const restoredCount = parseInt(restoredText.match(/(\d+) record/)?.[1] ?? "0", 10);
      if (!(restoredCount > 0)) {
        console.error(`FAIL: nothing survived the reload: ${JSON.stringify(restoredText)}`);
        throw new Error("persistence: expected a positive restored count after reload");
      }
      console.log(`persisted across reload: ${restoredCount} records`);
    } else {
      console.log("skipped the reload check: store fell back to in-memory");
    }

    console.log(`PASS: top hits ${JSON.stringify(ids.slice(0, 2))}, scores ${JSON.stringify(scores)}`);
  } finally {
    await session.close();
    proc.kill("SIGTERM");
    await sleep(200);
    if (proc.exitCode === null) proc.kill("SIGKILL");
  }
}

main().catch((e) => {
  console.error(`FAIL: ${e.message}`);
  process.exitCode = 1;
});
