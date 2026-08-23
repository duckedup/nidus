// Changed paths → the verification lanes that actually cover them.
// `just ci` compiles the pure library ONLY: it does not touch src/cli, src/server,
// src/bin, or any feature-gated layer, so a blanket `just ci` proves very little.

// Ordered cheapest-first. `manual` lanes need services or a release build, so they
// are reported as advisory rather than run automatically. `ci` lanes are enforced
// by a required check on every PR and are too slow to gate the local loop.
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
    match: [/^src\/server\/mcp(\.rs$|\/)/, /^tests\/e2e\/mcp(\.rs$|\/)/],
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
    // The interpreter is orders of magnitude slower than `cargo test` and the same
    // suite is a required PR check (ci.yml `Miri`), so a local run buys nothing.
    recipe: 'just miri',
    why: 'pure-logic codec/kernel code changed — CI enforces Miri on the PR',
    ci: true,
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
    match: [/^tests\/e2e\/(?!cluster\.rs)/, /^src\/server\/mcp(\.rs$|\/)/],
  },
  {
    recipe: 'just docs-build',
    why: 'the docs site changed — the Astro build is the only gate on it',
    match: [/^docs\//],
  },
  // Each SDK lane mirrors its CI job step for step (ci.yml `sdk-js`/`sdk-go`/`sdk-python`,
  // as of 2026-08-09). A lane that runs a subset is a subset wearing a total: it reports
  // green for a change CI will reject. Re-check this mapping when those jobs change.
  {
    recipe: 'cd sdks/js && npm ci && npm run typecheck && npm run test:unit && npm run build',
    why: 'the JS SDK changed',
    match: [/^sdks\/js\//],
    exclude: [/\.md$/, /^sdks\/js\/LICENSE$/],
  },
  {
    // `gofmt -l` lists offending files and still exits 0, so a bare call in a && chain
    // passes silently. Test its output for emptiness, exactly as the CI job does.
    recipe: 'cd sdks/go && test -z "$(gofmt -l .)" && go vet ./... && go test ./...',
    why: 'the Go SDK changed',
    match: [/^sdks\/go\//],
    exclude: [/\.md$/, /^sdks\/go\/LICENSE$/],
  },
  {
    recipe: 'cd sdks/python && ruff check && ruff format --check && mypy src && pytest --ignore=tests/test_integration.py',
    why: 'the Python SDK changed',
    match: [/^sdks\/python\//],
    exclude: [/\.md$/, /^sdks\/python\/LICENSE$/],
  },
  {
    // The SDK↔server contract (CLAUDE.md §2). The unit lanes above run against a mocked
    // transport, so they pass just as happily against a shape the server never emits;
    // this is the only lane that can tell. Manual: it needs a release build and a server.
    recipe: 'cargo build --release --features cli && export NIDUS_BIN=$PWD/target/release/nidus'
      + ' && (cd sdks/js && npm run test:integration)'
      + ' && (cd sdks/go && go test -tags integration ./...)'
      + ' && (cd sdks/python && pytest tests/test_integration.py)',
    why: 'an SDK changed — only a real `nidus serve` proves the wire contract (#172)',
    manual: true,
    match: [/^sdks\//],
    exclude: [/\.md$/, /LICENSE$/],
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
    // Same services-class shape as the cluster lane above, but the "service" is a
    // headless browser driver rather than minio/valkey — see scripts/e2e-wasm.sh.
    // `justfile` counts because the recipe itself lives there (nidus-3hc).
    recipe: 'just test-wasm-e2e',
    why: 'the wasm_opfs browser suite or its justfile recipe changed — needs a real headless browser',
    manual: true,
    match: [/^tests\/wasm_opfs\//, /^justfile$/],
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

// Which changed paths exercise each heavy CI job. The guard step in ci.yml /
// integration.yml skips a job's WORK when nothing here matches — never its report,
// so required checks still report (the QUEUE_LITE pattern). Keys are ci.yml job ids.
const RUST = [
  /^src\//, /^tests\//, /^examples\//, /^benchmarks\//,
  /^Cargo\.(toml|lock)$/, /^rust-toolchain\.toml$/, /^\.cargo\//,
  /^\.github\/workflows\/ci\.yml$/,
]
export const CI_JOBS = {
  'test': RUST,
  'test-extended': RUST,
  'release': RUST,
  'miri': RUST,
  'miri-integration': RUST,
  'build-budget': RUST,
  'bench-compiles': RUST,
  'build-thesis': RUST,
  'sdk-integration': [...RUST, /^sdks\//],
  'e2e': [...RUST.slice(0, -1), /^scripts\/e2e-services\.sh$/, /^\.github\/workflows\/integration\.yml$/],
  // wasm32 (nidus-y67). `justfile` counts because the recipes ARE the lane, and
  // `bindings/` because the binding is the only consumer of the browser backend.
  // `sdks/js/` (nidus-3hc): the `wasm` job now builds and packs the ./wasm subpath.
  'wasm': [...RUST, /^justfile$/, /^bindings\//, /^sdks\/js\//],
  'wasm-e2e': [
    ...RUST.slice(0, -1), /^justfile$/, /^bindings\//,
    /^scripts\/e2e-wasm\.sh$/, /^\.github\/workflows\/integration\.yml$/,
  ],
}

// Fail open twice over: an empty file list runs everything (a guard that saw
// nothing must not skip), and an unknown job throws (a renamed job fails loud).
export function ciGuard(job, paths) {
  const rules = CI_JOBS[job]
  if (!rules) throw new Error(`ci-guard: unknown job '${job}' — add it to CI_JOBS in lanes.mjs`)
  const files = (paths || []).filter(Boolean)
  const cause = files.find(f => rules.some(re => re.test(f))) || null
  return { job, run: !files.length || !!cause, cause, examined: files.length }
}

// Paths that need no build lane at all — reported so the caller can say why
// nothing ran, instead of silently returning an empty set.
const INERT = [/^\.claude\//, /^\.github\//, /^[^/]*\.md$/, /^LICENSE$/, /^\.gitignore$/, /LICENSE$/, /^sdks\/[^/]+\/.*\.md$/]

const covers = (rule, f) => rule.match.some(re => re.test(f)) && !(rule.exclude || []).some(re => re.test(f))

export function lanes(paths) {
  const files = (paths || []).filter(Boolean)
  const hit = []
  for (const rule of RULES) {
    const cause = files.find(f => covers(rule, f))
    if (cause) hit.push({ recipe: rule.recipe, why: rule.why, cause, manual: !!rule.manual, ci: !!rule.ci })
  }
  const covered = new Set(hit.flatMap(h => files.filter(f => covers(RULES.find(r => r.recipe === h.recipe), f))))
  const inert = files.filter(f => !covered.has(f) && INERT.some(re => re.test(f)))
  const unmatched = files.filter(f => !covered.has(f) && !inert.includes(f))
  return {
    // What the answer is *about*. Without it, an empty file list and a change that
    // genuinely needs no lane are the same output, and the first reads as the second.
    examined: files.length,
    run: hit.filter(h => !h.manual && !h.ci),
    manual: hit.filter(h => h.manual),
    ci: hit.filter(h => h.ci),
    inert,
    unmatched,
  }
}

export function formatLanes(result) {
  const out = [`Examined ${result.examined ?? 0} file(s).`]
  if (!result.examined) {
    out.push('Nothing to check — this target has no files, so no lane could have applied.')
    out.push('Commit first, or name the files with --paths.')
    return out.join('\n')
  }
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
  if (result.ci?.length) {
    out.push('', 'CI enforces these on the PR — skip locally unless debugging that lane:')
    for (const l of result.ci) out.push(`  ${l.recipe}\n      ↳ ${l.why} (${l.cause})`)
  }
  if (result.unmatched.length) {
    out.push('', `Unmapped paths (no lane knows about these — check by hand): ${result.unmatched.join(', ')}`)
  }
  return out.join('\n')
}
