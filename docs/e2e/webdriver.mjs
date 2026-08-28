// Raw W3C WebDriver plumbing shared by the docs e2e scripts (terminal.mjs,
// navigation.mjs). One copy, for the same reason scripts/e2e-wasm.sh is one
// copy: driver detection and session handling cannot be allowed to drift
// between two suites that must agree on which browser CI actually drove.

import { spawn } from "node:child_process";

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

  back() {
    return this.call("POST", `${this.base}/back`, {});
  }

  async currentUrl() {
    const res = await this.call("GET", `${this.base}/url`);
    return res.value;
  }

  /** Runs in the page and returns the (JSON-serialisable) result. */
  async execute(script, args = []) {
    const res = await this.call("POST", `${this.base}/execute/sync`, { script, args });
    return res.value;
  }

  /** Best-effort: some drivers refuse to resize, and the caller can live with that. */
  async setWindowSize(width, height) {
    try {
      await this.call("POST", `${this.base}/window/rect`, { width, height, x: 0, y: 0 });
    } catch {
      // headless drivers occasionally reject a resize; assertions that need the
      // width will say so themselves.
    }
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

  /**
   * Click an element after centring what `selector` matches in the viewport.
   *
   * The driver's own click scrolls the element only just into view, which on a
   * page with a `position: sticky` header parks it underneath that header and
   * gets the click intercepted ("Element is not clickable at point"). Centring
   * first is what a person does without thinking about it, and the click itself
   * stays a real driver click rather than a dispatched event.
   *
   * Scrolling by selector rather than by element handle keeps the script's only
   * argument a string, so there is no element-reference encoding to get wrong on
   * one driver and right on another. `behavior: "instant"` is load-bearing: the
   * page sets `scroll-behavior: smooth`, and the driver would otherwise click
   * where the element was before the animation started.
   */
  async clickInView(el, selector) {
    await this.execute(
      "document.querySelector(arguments[0])?.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'instant' }); return true;",
      [selector],
    );
    return this.click(el);
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

export { sleep, waitFor, detectDriver, capabilitiesFor, startDriver, Session };
