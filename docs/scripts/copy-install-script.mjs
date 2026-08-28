#!/usr/bin/env node
// Publishes the repo's own install.sh at https://nidus.duckedup.org/install.sh so the
// landing page's call to action fits on one line:
//
//   curl -fsSL https://nidus.duckedup.org/install.sh | sh
//
// It is copied at build time rather than checked in twice, so the served script can
// never drift from the one in the repo root. A missing source is a hard error: shipping
// the site without it would leave that command 404ing.
import { copyFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const src = join(scriptDir, "..", "..", "install.sh");
const dest = join(scriptDir, "..", "public", "install.sh");

if (!existsSync(src)) {
  console.error(`error: ${src} is absent; the site's install command would 404.`);
  process.exit(1);
}

copyFileSync(src, dest);
console.log("copied install.sh into public/");
