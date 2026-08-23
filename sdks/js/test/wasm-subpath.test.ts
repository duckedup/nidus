import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

// Runs on a fresh checkout with no wasm artifact present, so it reads package.json off
// disk (never `dist`) and never instantiates the wasm module itself.
const pkgPath = fileURLToPath(new URL("../package.json", import.meta.url));
const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));

describe("./wasm subpath contract", () => {
  it("declares types and import, but no require condition", () => {
    const wasmExport = pkg.exports["./wasm"];
    expect(wasmExport).toBeDefined();
    expect(wasmExport.types).toBeTypeOf("string");
    expect(wasmExport.import).toBeTypeOf("string");
    expect(wasmExport.require).toBeUndefined();
  });

  it("points at dist/wasm/", () => {
    const wasmExport = pkg.exports["./wasm"];
    expect(wasmExport.types).toMatch(/^\.\/dist\/wasm\//);
    expect(wasmExport.import).toMatch(/^\.\/dist\/wasm\//);
  });

  it("keeps the wasm subpath out of the default entry", () => {
    const indexSrc = readFileSync(fileURLToPath(new URL("../src/index.ts", import.meta.url)), "utf8");
    expect(indexSrc).not.toMatch(/wasm/i);
  });
});
