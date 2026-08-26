#!/usr/bin/env bun
// nidus-5os: the docs site runs Astro's <ClientRouter />, and Starlight ships no
// view-transition support of its own — its <site-search> mounts Pagefind from a
// `DOMContentLoaded` listener that never fires again after a client-side swap.
// Left alone, one click on the home icon killed search until a hard reload. This
// drives the REAL built site over raw W3C WebDriver and asserts search still
// works *after* navigating, which is the only place the bug ever showed.
// scripts/e2e-wasm.sh serves the site and invokes:
//   bun docs/e2e/navigation.mjs <base-url>

import assert from "node:assert/strict";
import {
  waitFor,
  detectDriver,
  capabilitiesFor,
  startDriver,
  Session,
} from "./webdriver.mjs";

// A word that appears in the prose of more than one page, so a single stale
// index cannot satisfy it by accident.
const QUERY = "storage";

// Records every uncaught error on `window`. The window survives a view
// transition, so a listener installed once sees the whole session — which is
// how the leaked-listener regression shows up: each discarded <site-search>
// left a keydown handler bound to a detached <dialog>, and ⌘K then threw
// `InvalidStateError` once per navigation made so far.
const INSTALL_ERROR_TRAP = `
  if (!window.__ndErrors) {
    window.__ndErrors = [];
    window.addEventListener("error", (e) => {
      window.__ndErrors.push(String(e.message || e.error));
    });
  }
  return true;
`;

const BADGE_VISIBLE = `
  const kbd = document.querySelector("site-search button kbd");
  if (!kbd) return "no badge in the header at all";
  const style = getComputedStyle(kbd);
  if (style.display === "none") return "display: none — the reveal script never re-ran";
  if (style.visibility === "hidden") return "visibility: hidden";
  return kbd.getBoundingClientRect().width > 0 || "zero width";
`;

// A view transition paints over the page while it runs, and Safari refuses to
// click through that overlay. Waiting for the header's own search button to be
// the topmost thing at its own centre is the settled state we actually need.
const SEARCH_BUTTON_READY = `
  const b = document.querySelector("site-search button[data-open-modal]");
  if (!b) return false;
  const r = b.getBoundingClientRect();
  const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
  return !!hit && (hit === b || b.contains(hit));
`;

const OPEN_SEARCH = `
  const b = document.querySelector("site-search button[data-open-modal]");
  if (!b) return "no search button in the header";
  if (b.disabled) return "the search button is still disabled";
  b.click();
  return document.querySelector("site-search dialog")?.open === true;
`;

const TYPE_QUERY = `
  const input = document.querySelector(".pagefind-ui__search-input");
  if (!input) return "the Pagefind input is not in the dialog";
  input.value = arguments[0];
  input.dispatchEvent(new Event("input", { bubbles: true }));
  return true;
`;

const PRESS_CMD_K = `
  window.dispatchEvent(
    new KeyboardEvent("keydown", { key: "k", metaKey: true, ctrlKey: true, bubbles: true }),
  );
  return window.__ndErrors.slice();
`;

// safaridriver drops a synthetic click on a position: fixed header often enough
// to matter, and it reports success when it does, so the only reliable signal is
// whether the navigation actually happened. Retry until it does.
async function clickUntilAt(session, selector, pathname, what) {
  await waitFor(
    async () => {
      if (new URL(await session.currentUrl()).pathname === pathname) return true;
      try {
        await session.click(await session.findElement(selector));
      } catch {
        return null;
      }
      return null;
    },
    { timeoutMs: 20000, label: `${pathname} after clicking ${what}`, intervalMs: 500 },
  );
}

async function main() {
  const baseUrl = process.argv[2];
  if (!baseUrl) {
    throw new Error("usage: bun docs/e2e/navigation.mjs <base-url>");
  }
  const root = baseUrl.replace(/\/$/, "");

  const step = (m) => console.log(`  ${m}`);

  const { proc, port, kind } = await startDriver(detectDriver());
  const session = new Session(`http://127.0.0.1:${port}`);

  try {
    await session.open(capabilitiesFor(kind));
    // Starlight hides the ⌘K badge below 50rem, so assert at a desktop width or
    // the badge check would pass for the wrong reason.
    await session.setWindowSize(1280, 900);

    // Start on a docs page, so the first navigation is the one users reported:
    // a click on the home icon in the header.
    await session.navigate(`${root}/getting-started/`);
    await waitFor(() => session.findElement("site-search").catch(() => null), {
      timeoutMs: 15000,
      label: "the header search element on first load",
    });
    await session.execute(INSTALL_ERROR_TRAP);

    step("clicking the home icon");
    await clickUntilAt(session, "header a.site-title", "/", "the site logo");
    await waitFor(() => session.execute(SEARCH_BUTTON_READY), {
      timeoutMs: 10000,
      label: "the header to settle on the home page",
    });

    // And back into the docs, so the assertions below cover a round trip rather
    // than only the splash page. Back is a client-side navigation too (Astro
    // handles popstate), and unlike a link in the page body it cannot be
    // knocked out of reach by the splash layout.
    step("navigating back into the docs");
    await session.back();
    await waitFor(
      async () => new URL(await session.currentUrl()).pathname === "/getting-started/",
      { timeoutMs: 10000, label: "the getting-started page after navigating back" },
    );
    await waitFor(() => session.execute(SEARCH_BUTTON_READY), {
      timeoutMs: 10000,
      label: "the header to settle after navigating back",
    });

    // 1. The real thing: open search, type, and get Pagefind results back.
    // Before the fix the dialog opened with no input in it at all.
    step("opening search");
    // Dispatched in the page rather than through the driver's synthetic click:
    // after a view transition safaridriver's click lands on a stale hit-test
    // region and silently misses, intermittently. This is still the real
    // button, the real listener and the real dialog — only the input device is
    // simulated, and the assertions below are all about what Pagefind does.
    assert.equal(
      await session.execute(OPEN_SEARCH),
      true,
      "the search button did not open its dialog",
    );
    await waitFor(
      () => session.findElement(".pagefind-ui__search-input").catch(() => null),
      { timeoutMs: 15000, label: "the Pagefind input after navigating (the bug)" },
    );
    step("typing the query");
    // Same reason as the click above, and Pagefind searches off the `input`
    // event either way, so this drives exactly the code a keystroke would.
    assert.equal(
      await session.execute(TYPE_QUERY, [QUERY]),
      true,
      "could not type into the search input",
    );

    const results = await waitFor(
      async () => {
        const found = await session.findElements(".pagefind-ui__result");
        return found.length > 0 ? found : null;
      },
      { timeoutMs: 15000, label: `Pagefind results for "${QUERY}"` },
    );

    // 2. The ⌘K badge. Starlight reveals it from a run-once inline script that
    // Astro's swap deduplicates by content, so a rebuilt header leaves it
    // stuck at `display: none`. Measured in the page rather than through
    // /element/{id}/displayed, which safaridriver does not implement.
    step("checking the shortcut badge");
    assert.equal(
      await session.execute(BADGE_VISIBLE),
      true,
      "the ⌘K badge is hidden after navigating: the header was rebuilt, not persisted",
    );

    // 3. ⌘K itself throws nothing. It closes the dialog opened above; a leaked
    // handler per navigation would log an InvalidStateError instead.
    step("pressing the keyboard shortcut");
    const errors = await session.execute(PRESS_CMD_K);
    assert.deepEqual(
      errors,
      [],
      `⌘K threw after navigating: ${JSON.stringify(errors)}`,
    );

    console.log(`PASS: ${results.length} result(s) for "${QUERY}" after two client-side navigations`);
  } finally {
    await session.close();
    proc.kill("SIGTERM");
    if (proc.exitCode === null) proc.kill("SIGKILL");
  }
}

main().catch((e) => {
  console.error(`FAIL: ${e.message}`);
  process.exitCode = 1;
});
