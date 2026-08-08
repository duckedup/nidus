// Changed paths → the verification lanes that actually cover them.
// `just ci` compiles the pure library ONLY: it does not touch src/cli, src/server,
// src/bin, or any feature-gated layer, so a blanket `just ci` proves very little.

// Ordered cheapest-first. `manual` lanes need services or a release build, so they
// are reported as advisory rather than run automatically.
const RULES = [
  {
    recipe: 'just deps',
    why: 'Cargo.toml changed — assert the dependency tree stayed minimal',
    match: [/^Cargo\.toml$/],
  },
  {
    recipe: 'just ci',
    why: 'library source changed — fmt-check + clippy + test on the pure library',
    match: [/^src\//, /^tests\/(integration|build_thesis)\.rs$/, /^Cargo\.toml$/, /^rust-toolchain\.toml$/],
  },
  {
    recipe: 'just ci-cli',
    why: 'binary/server code changed — `just ci` does NOT compile src/cli, src/server or src/bin',
    match: [/^src\/cli\//, /^src\/bin\//, /^src\/server\//],
  },
  {
    recipe: 'cargo clippy --all-targets --features mcp -- -D warnings',
    why: 'the MCP surface changed — it compiles only under the `mcp` feature',
    match: [/^src\/server\/mcp\.rs$/, /^tests\/e2e\/mcp\.rs$/],
  },
  {
    recipe: 'just ci-embed',
    why: 'embedder adapters changed — cfg-gated out of the default build',
    match: [/^src\/embed\//, /^src\/providers\.rs$/],
  },
  {
    recipe: 'just ci-summarize',
    why: 'summarizer adapters changed — cfg-gated out of the default build',
    match: [/^src\/summarize\//],
  },
  {
    recipe: 'just ci-serve',
    why: 'the memory surface changed — remember/recall compile only under `serve`',
    match: [/^src\/memory\.rs$/],
  },
  {
    recipe: 'just miri',
    why: 'pure-logic codec/kernel code changed — Miri covers all of it',
    match: [
      /^src\/log\//, /^src\/glob\//, /^src\/filter\//, /^src\/search\//,
      /^src\/data\//, /^src\/ann\//, /^src\/fts\//, /^src\/manifest\//,
      /^src\/lock\//, /^src\/model\.rs$/, /^src\/(fuse|annotate)\.rs$/,
      /^src\/store\/(scoring|quant|rank|aggregate|text)\.rs$/,
    ],
  },
  {
    recipe: 'just test-e2e',
    why: 'end-to-end tests changed — they drive the real `nidus serve` binary',
    match: [/^tests\/e2e\/(?!cluster\.rs)/, /^src\/server\/mcp\.rs$/],
  },
  {
    recipe: 'just docs-build',
    why: 'the docs site changed — the Astro build is the only gate on it',
    match: [/^docs\//],
  },
  {
    recipe: 'cd sdks/js && npm ci && npm run typecheck && npm run test:unit',
    why: 'the JS SDK changed',
    match: [/^sdks\/js\//],
    exclude: [/\.md$/, /^sdks\/js\/LICENSE$/],
  },
  {
    recipe: 'cd sdks/go && go test ./...',
    why: 'the Go SDK changed',
    match: [/^sdks\/go\//],
    exclude: [/\.md$/, /^sdks\/go\/LICENSE$/],
  },
  {
    recipe: 'cd sdks/python && python -m pytest tests -k "not integration"',
    why: 'the Python SDK changed',
    match: [/^sdks\/python\//],
    exclude: [/\.md$/, /^sdks\/python\/LICENSE$/],
  },
  {
    recipe: 'just e2e-services-up && just test-e2e-cluster',
    why: 'cluster/object-store/tier code changed — needs real minio + valkey',
    manual: true,
    match: [
      /^tests\/e2e\/cluster\.rs$/,
      /^src\/backend\/(s3|gcs|redis|cloud|object)\.rs$/,
      /^src\/store\/memtier\.rs$/,
    ],
  },
  {
    recipe: 'just bench',
    why: 'benchmarks changed — release build, run deliberately',
    manual: true,
    match: [/^benchmarks\//],
  },
  {
    recipe: 'helm lint charts/nidus',
    why: 'the chart changed (note: versions there are bot-stamped, not hand-edited)',
    manual: true,
    match: [/^charts\//],
  },
]

// Paths that need no build lane at all — reported so the caller can say why
// nothing ran, instead of silently returning an empty set.
const INERT = [/^\.claude\//, /^\.github\//, /^\.beads\//, /^[^/]*\.md$/, /^LICENSE$/, /^\.gitignore$/, /LICENSE$/, /^sdks\/[^/]+\/.*\.md$/]

const covers = (rule, f) => rule.match.some(re => re.test(f)) && !(rule.exclude || []).some(re => re.test(f))

export function lanes(paths) {
  const files = (paths || []).filter(Boolean)
  const hit = []
  for (const rule of RULES) {
    const cause = files.find(f => covers(rule, f))
    if (cause) hit.push({ recipe: rule.recipe, why: rule.why, cause, manual: !!rule.manual })
  }
  const covered = new Set(hit.flatMap(h => files.filter(f => covers(RULES.find(r => r.recipe === h.recipe), f))))
  const inert = files.filter(f => !covered.has(f) && INERT.some(re => re.test(f)))
  const unmatched = files.filter(f => !covered.has(f) && !inert.includes(f))
  return {
    run: hit.filter(h => !h.manual),
    manual: hit.filter(h => h.manual),
    inert,
    unmatched,
  }
}

export function formatLanes(result) {
  const out = []
  if (result.run.length) {
    out.push('Run these:')
    for (const l of result.run) out.push(`  ${l.recipe}\n      ↳ ${l.why} (${l.cause})`)
  } else {
    out.push('No automated lane applies.')
  }
  if (result.manual.length) {
    out.push('', 'Needs a deliberate run (services or release build):')
    for (const l of result.manual) out.push(`  ${l.recipe}\n      ↳ ${l.why} (${l.cause})`)
  }
  if (result.unmatched.length) {
    out.push('', `Unmapped paths (no lane knows about these — check by hand): ${result.unmatched.join(', ')}`)
  }
  return out.join('\n')
}
