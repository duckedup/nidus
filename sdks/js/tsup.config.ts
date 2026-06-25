import { defineConfig } from "tsup";

// Dual ESM + CJS bundles with type declarations. The CJS output uses a `.cjs`
// extension so it resolves under `"type": "module"`.
export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  minify: false,
  target: "es2022",
  outExtension({ format }) {
    return { js: format === "cjs" ? ".cjs" : ".js" };
  },
});
