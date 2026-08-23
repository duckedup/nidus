import { defineConfig } from "tsup";

// Two builds: the main dual ESM+CJS package and the `./wasm` subpath (ESM-only,
// nidus-3hc). NEITHER may set `clean`: tsup runs array configs concurrently, so the main
// build wiping `dist/` could delete the wasm build's `dist/wasm` output. The npm script
// clears `dist/` once, up front, instead.
export default defineConfig([
  {
    entry: ["src/index.ts"],
    format: ["esm", "cjs"],
    dts: true,
    clean: false,
    sourcemap: true,
    minify: false,
    target: "es2022",
    outExtension({ format }) {
      return { js: format === "cjs" ? ".cjs" : ".js" };
    },
  },
  {
    entry: { index: "src/wasm-helper.ts" },
    format: ["esm"],
    dts: true,
    clean: false,
    sourcemap: true,
    minify: false,
    target: "es2022",
    outDir: "dist/wasm",
  },
]);
