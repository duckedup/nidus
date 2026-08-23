#!/usr/bin/env node
// Copies the generated `bindings/wasm/pkg` artifact into `dist/wasm/` verbatim, never a
// tsup/tsc input (nidus-3hc). Default: a missing pkg is a no-op, so a fresh checkout
// stays green. NIDUS_WASM_REQUIRED=1 (the release lane, the CI wasm job) is a hard error.
import { existsSync, mkdirSync, copyFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const pkgDir = join(scriptDir, "..", "..", "..", "bindings", "wasm", "pkg");
const outDir = join(scriptDir, "..", "dist", "wasm");
const required = process.env.NIDUS_WASM_REQUIRED === "1";

const files = ["nidus_wasm.js", "nidus_wasm.d.ts", "nidus_wasm_bg.wasm", "nidus_wasm_bg.wasm.d.ts"];

if (!existsSync(pkgDir)) {
  const msg = "bindings/wasm/pkg is absent. Run `just build-wasm-binding` to generate it.";
  if (required) {
    console.error(`error: ${msg}`);
    process.exit(1);
  }
  console.log(`note: ${msg} Skipping the ./wasm subpath for this build.`);
  process.exit(0);
}

mkdirSync(outDir, { recursive: true });
for (const file of files) {
  copyFileSync(join(pkgDir, file), join(outDir, file));
}
console.log(`copied ${files.length} wasm artifact files into dist/wasm/`);
