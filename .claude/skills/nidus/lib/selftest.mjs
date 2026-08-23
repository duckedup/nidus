// Fixture suite for the detectors. Pure — no repository, no git, no network — so a
// broken detector fails here instead of silently passing a real review.

import * as laws from './laws.mjs'
import * as fleet from './fleet.mjs'
import * as pre from './preflight.mjs'
import { lanes, formatLanes, ciGuard } from './lanes.mjs'

const cases = []
const test = (name, fn) => cases.push({ name, fn })
const ids = fs => fs.map(f => f.id).sort()

function eq(actual, expected, what) {
  const a = JSON.stringify(actual)
  const e = JSON.stringify(expected)
  if (a !== e) throw new Error(`${what}: expected ${e}, got ${a}`)
}

// ── comment cap ────────────────────────────────────────────────────────────

test('comment cap: a 3-line block is fine', () => {
  const src = `// one\n// two\n// three\nfn f() {}\n`
  eq(ids(laws.commentCap(src, null, 'src/x.rs')), [], 'findings')
})

test('comment cap: a 4-line block is flagged', () => {
  const src = `// one\n// two\n// three\n// four\nfn f() {}\n`
  const found = laws.commentCap(src, null, 'src/x.rs')
  eq(ids(found), ['comment-cap'], 'findings')
  eq(found[0].line, 1, 'line')
})

test('comment cap: /// blank separators count toward the cap', () => {
  const src = `/// prose\n///\n/// more prose\n///\n/// even more\npub fn f() {}\n`
  eq(ids(laws.commentCap(src, null, 'src/x.rs')), ['comment-cap'], 'findings')
})

test('comment cap: a doc-example fence does not count', () => {
  const src = [
    '/// Opens a store.',
    '/// ```',
    '/// let n = Nidus::open("dir", 3)?;',
    '/// let hits = n.search(&q, 5)?;',
    '/// assert_eq!(hits.len(), 5);',
    '/// ```',
    'pub fn open() {}',
  ].join('\n')
  eq(ids(laws.commentCap(src, null, 'src/x.rs')), [], 'findings')
})

test('comment cap: prose around a fence still counts', () => {
  const src = [
    '/// one',
    '/// two',
    '/// three',
    '/// four',
    '/// ```',
    '/// code();',
    '/// ```',
    'pub fn open() {}',
  ].join('\n')
  eq(ids(laws.commentCap(src, null, 'src/x.rs')), ['comment-cap'], 'findings')
})

test('comment cap: //! module docs are exempt at any length', () => {
  const src = [
    '//! # thing',
    '//!',
    '//! A long module doc that would blow the cap as `///` but is the published',
    '//! rustdoc landing page, not commentary on code.',
    '//!',
    '//! More prose still.',
    'use std::fmt;',
  ].join('\n')
  eq(ids(laws.commentCap(src, null, 'src/x.rs')), [], 'findings')
})

test('comment cap: a //! line exempts only its own block', () => {
  const src = [
    '//! module doc',
    '//! second line',
    '//! third line',
    '//! fourth line',
    '',
    '/// one',
    '/// two',
    '/// three',
    '/// four',
    'pub fn f() {}',
  ].join('\n')
  const found = laws.commentCap(src, null, 'src/x.rs')
  eq(ids(found), ['comment-cap'], 'findings')
  eq(found[0].line, 6, 'line')
})

// Regression: the first cut of the //! exemption tainted the whole contiguous block,
// and blocks break only on a blank/code line — so one //! glued to a /// doc carried it
// over the cap. A stray //! was a one-line way to dodge the rule entirely.
test('comment cap: a //! glued to a /// doc does not exempt it', () => {
  const src = [
    '//! module doc',
    '/// one',
    '/// two',
    '/// three',
    '/// four',
    'pub fn f() {}',
  ].join('\n')
  const found = laws.commentCap(src, null, 'src/x.rs')
  eq(ids(found), ['comment-cap'], 'findings')
  eq(found[0].line, 2, 'anchors at the first counted line, not the //!')
})

test('comment cap: //! lines do not count toward an adjacent block', () => {
  const src = ['//! module doc', '/// one', '/// two', '/// three', 'pub fn f() {}'].join('\n')
  eq(ids(laws.commentCap(src, null, 'src/x.rs')), [], 'findings')
})

test('comment cap: separate blocks are counted separately', () => {
  const src = `// a\n// b\nfn f() {}\n\n// c\n// d\nfn g() {}\n`
  eq(ids(laws.commentCap(src, null, 'src/x.rs')), [], 'findings')
})

test('comment cap: untouched violations are skipped when scoped to added lines', () => {
  const src = `// one\n// two\n// three\n// four\nfn f() {}\n`
  eq(ids(laws.commentCap(src, new Set([5]), 'src/x.rs')), [], 'findings')
  eq(ids(laws.commentCap(src, new Set([2]), 'src/x.rs')), ['comment-cap'], 'findings')
})

// ── unsafe ─────────────────────────────────────────────────────────────────

test('unsafe: flagged in a normal module', () => {
  eq(ids(laws.unsafeUse('unsafe { ptr.read() }\n', null, 'src/store/read.rs')), ['unsafe-code'], 'findings')
})

test('unsafe: the sanctioned mmap module is exempt', () => {
  eq(ids(laws.unsafeUse('unsafe { Mmap::map(&f)? }\n', null, 'src/data/mmap.rs')), [], 'findings')
})

test('unsafe: the word inside a comment is not a use', () => {
  eq(ids(laws.unsafeUse('// this would be unsafe { } to do\n', null, 'src/x.rs')), [], 'findings')
})

test('unsafe: crate attribute must survive', () => {
  eq(ids(laws.crateAttrWeakened('#![deny(unsafe_code)]\n')), [], 'deny is fine')
  eq(ids(laws.crateAttrWeakened('#![forbid(unsafe_code)]\n')), [], 'forbid is fine')
  eq(ids(laws.crateAttrWeakened('//! nidus\n')), ['unsafe-attr'], 'missing is flagged')
})

// ── version bump + docs sync ───────────────────────────────────────────────

const cargo = v => `[package]\nname = "nidus"\nversion = "${v}"\n`

test('version bump: required when src changed', () => {
  eq(ids(laws.versionBump(cargo('0.43.0'), cargo('0.43.0'), ['src/store/read.rs'])), ['version-bump'], 'findings')
})

test('version bump: satisfied by an actual bump', () => {
  eq(ids(laws.versionBump(cargo('0.43.0'), cargo('0.43.1'), ['src/store/read.rs'])), [], 'findings')
})

test('version bump: not required for skill-only churn', () => {
  eq(ids(laws.versionBump(cargo('0.43.0'), cargo('0.43.0'), ['.claude/skills/nidus/SKILL.md'])), [], 'findings')
})

test('docs sync: stale snippet flagged on a minor bump', () => {
  const found = laws.docsVersionSync(cargo('0.43.0'), cargo('0.44.0'), {
    'README.md': 'nidus = "0.43"\n',
    'docs/src/content/docs/getting-started.md': 'nidus = "0.44"\n',
  })
  eq(ids(found), ['docs-version-sync'], 'findings')
  eq(found[0].file, 'README.md', 'file')
})

test('docs sync: a patch bump leaves the snippet correct', () => {
  eq(ids(laws.docsVersionSync(cargo('0.43.0'), cargo('0.43.1'), { 'README.md': 'nidus = "0.43"\n' })), [], 'findings')
})

// ── bot-stamped files ──────────────────────────────────────────────────────

test('bot-stamped: hand-edited chart version flagged', () => {
  const diff = '--- a\n+++ b\n-version: 0.43.0\n+version: 0.44.0\n'
  eq(ids(laws.botStamped(['charts/nidus/Chart.yaml'], { 'charts/nidus/Chart.yaml': diff })), ['bot-stamped'], 'findings')
})

test('bot-stamped: a non-version chart edit is fine', () => {
  const diff = '--- a\n+++ b\n+  description: a vector store\n'
  eq(ids(laws.botStamped(['charts/nidus/Chart.yaml'], { 'charts/nidus/Chart.yaml': diff })), [], 'findings')
})

// ── dependencies ───────────────────────────────────────────────────────────

const manifest = (version, deps) =>
  `[package]\nname = "nidus"\nversion = "${version}"\n\n[dependencies]\n${deps.map(d => `${d} = "1"`).join('\n')}\n`

test('deps: a bundled-C dep is a hard error', () => {
  eq(ids(laws.newDeps(manifest('0.43.0', ['serde']), manifest('0.43.0', ['serde', 'duckdb']))), ['forbidden-dep'], 'findings')
})

test('deps: aws-lc and *-sys are hard errors', () => {
  const found = laws.newDeps(manifest('0.43.0', []), manifest('0.43.0', ['aws-lc-rs', 'foo-sys']))
  eq(ids(found), ['forbidden-dep', 'forbidden-dep'], 'findings')
})

test('deps: an ordinary new dep is a warning', () => {
  eq(ids(laws.newDeps(manifest('0.43.0', ['serde']), manifest('0.43.0', ['serde', 'bytes']))), ['new-dep'], 'findings')
})

// Regression: a real run read `+version = "0.44.0"` from [package] as a new dep.
test('deps: a package version bump is not a new dependency', () => {
  eq(ids(laws.newDeps(manifest('0.43.0', ['serde']), manifest('0.44.0', ['serde']))), [], 'findings')
})

test('deps: an unchanged manifest adds nothing', () => {
  eq(ids(laws.newDeps(manifest('0.43.0', ['serde', 'anyhow']), manifest('0.43.0', ['serde', 'anyhow']))), [], 'findings')
})

test('deps: [dependencies.foo] table form is seen', () => {
  const head = '[package]\nversion = "1"\n\n[dependencies.rmcp]\nversion = "0.1"\n'
  eq(ids(laws.newDeps('[package]\nversion = "1"\n', head)), ['new-dep'], 'findings')
})

// ── test placement ─────────────────────────────────────────────────────────

test('test placement: a binary-driving tests/*.rs is an error', () => {
  const found = laws.testPlacement(['tests/smoke.rs'], () => 'let bin = env!("CARGO_BIN_EXE_nidus");')
  eq(ids(found), ['test-placement'], 'findings')
  eq(found[0].severity, 'error', 'severity')
})

test('test placement: modules under tests/e2e are fine', () => {
  eq(ids(laws.testPlacement(['tests/e2e/token.rs'], () => 'CARGO_BIN_EXE_nidus')), [], 'findings')
})

// ── miri ignores ───────────────────────────────────────────────────────────

test('miri: ignoring a pure-logic test is flagged', () => {
  const src = '#[cfg_attr(miri, ignore)]\n#[test]\nfn cosine_is_symmetric() {\n  assert_eq!(dot(&a, &b), dot(&b, &a));\n}\n'
  eq(ids(laws.miriIgnore(src, null, 'src/search/mod.rs')), ['miri-ignore'], 'findings')
})

test('miri: ignoring an fsync test is fine', () => {
  const src = '#[cfg_attr(miri, ignore)]\n#[test]\nfn flush_syncs() {\n  file.sync_all().unwrap();\n}\n'
  eq(ids(laws.miriIgnore(src, null, 'src/log/mod.rs')), [], 'findings')
})

// Regression: a full-tree sweep flagged 22 of these. CLAUDE.md sanctions them —
// they are the localhost-mock round-trips, which hit the network, not the disk.
test('miri: ignoring a localhost-mock round-trip is fine', () => {
  const src = '#[cfg_attr(miri, ignore)]\n#[test]\nfn round_trip_against_mock_s3() {\n  let server = mock::MockS3::start();\n}\n'
  eq(ids(laws.miriIgnore(src, null, 'src/backend/s3.rs')), [], 'findings')
})

// Regression (nidus #92): the rule's own remedy is "say why in a comment", so a
// documented ignore must stop warning — otherwise the check can never be resolved
// and reads as ambient noise. Seven already-documented ignores were still firing.
test('miri: a documented ignore is accepted', () => {
  const src = '#[cfg_attr(miri, ignore)] // N=2000 build is too slow under Miri.\n#[test]\nfn hnsw_recall() {\n  assert!(recall > 0.9);\n}\n'
  eq(ids(laws.miriIgnore(src, null, 'src/store/tests.rs')), [], 'findings')
})

// Regression (nidus #99): the codebase's existing style puts the reason on the line
// ABOVE the attribute. Reading only the trailing form reported those as bare, and two
// correctly-ignored float-ULP tests were deleted on the strength of it.
test('miri: a reason on the line above the attribute counts', () => {
  const src = [
    '#[test]',
    "// BM25's `idf` calls `ln`, which Miri evaluates non-deterministically.",
    '#[cfg_attr(miri, ignore)]',
    'fn hybrid_scores_are_exact() {',
    '  assert_eq!(hit.score.to_bits(), 0x3f2a_1b3c);',
    '}',
  ].join('\n')
  eq(ids(laws.miriIgnore(src, null, 'src/store/tests.rs')), [], 'findings')
})

test('miri: an attribute directly under another attribute is still bare', () => {
  const src = '#[test]\n#[cfg_attr(miri, ignore)]\nfn cosine_is_symmetric() {\n  assert_eq!(dot(&a, &b), dot(&b, &a));\n}\n'
  eq(ids(laws.miriIgnore(src, null, 'src/search/mod.rs')), ['miri-ignore'], 'findings')
})

test('miri: an empty trailing comment does not count as documentation', () => {
  const src = '#[cfg_attr(miri, ignore)] //\n#[test]\nfn cosine_is_symmetric() {\n  assert_eq!(dot(&a, &b), dot(&b, &a));\n}\n'
  eq(ids(laws.miriIgnore(src, null, 'src/search/mod.rs')), ['miri-ignore'], 'findings')
})

// ── feature gating ─────────────────────────────────────────────────────────

test('gating: a library module importing tokio is an error', () => {
  eq(ids(laws.featureGating('src/store/read.rs', 'use tokio::fs;\n')), ['feature-gating'], 'findings')
})

test('gating: src/server may import axum', () => {
  eq(ids(laws.featureGating('src/server/mod.rs', 'use axum::Router;\n')), [], 'findings')
})

test('gating: ungated mod declaration flagged', () => {
  eq(ids(laws.modGating('mod store;\npub mod cli;\n')), ['mod-gating'], 'findings')
  eq(ids(laws.modGating('mod store;\n#[cfg(feature = "cli")]\npub mod cli;\n')), [], 'findings')
})

// ── stale tickets ──────────────────────────────────────────────────────────

test('tickets: a mentioned issue with no Closes line warns', () => {
  const found = laws.unclosedTickets(new Set(['#42']), new Set(), { '#42': 'a thing' })
  eq(ids(found), ['stale-ticket'], 'findings')
  eq(found[0].severity, 'warn', 'severity')
})

test('tickets: a Closes line clears the warning', () => {
  eq(ids(laws.unclosedTickets(new Set(['#42']), new Set(['#42']), { '#42': 'a thing' })), [], 'findings')
})

// Only OPEN issues reach titles, so an already-closed or cross-repo ref is not noise.
test('tickets: an issue with no resolvable open title is not flagged', () => {
  eq(ids(laws.unclosedTickets(new Set(['#7']), new Set(), {})), [], 'findings')
})

// ── lanes ──────────────────────────────────────────────────────────────────

test('lanes: a store change runs the pure-library gate', () => {
  const r = lanes(['src/store/write.rs'])
  eq(r.run.map(l => l.recipe), ['just ci'], 'recipes')
})

// `just ci-cli` runs `test-cli`, which already includes the e2e suite — so a plain
// server change needs no separate test-e2e lane. The MCP path does: `just test-e2e`
// is the only recipe that builds with `--features cli,mcp`.
test('lanes: a server change adds the cli gate that `just ci` skips', () => {
  const r = lanes(['src/server/mod.rs'])
  eq(r.run.map(l => l.recipe), ['just ci', 'just ci-cli'], 'recipes')
})

test('lanes: the MCP surface pulls its own feature build', () => {
  const r = lanes(['src/server/mcp.rs'])
  eq(r.run.map(l => l.recipe).includes('cargo clippy --all-targets --features mcp -- -D warnings'), true, 'mcp clippy')
  eq(r.run.map(l => l.recipe).includes('just test-e2e'), true, 'e2e')
})

// The same lane must survive `mcp.rs` becoming `mcp/` (nidus-k28) — a path-pinned
// detector would silently drop the only gate that compiles the MCP surface.
test('lanes: the MCP surface pulls its feature build as a directory too', () => {
  for (const p of ['src/server/mcp/stdio.rs', 'tests/e2e/mcp/filters.rs']) {
    const r = lanes([p]).run.map(l => l.recipe)
    eq(r.includes('cargo clippy --all-targets --features mcp -- -D warnings'), true, `mcp clippy: ${p}`)
    eq(r.includes('just test-e2e'), true, `e2e: ${p}`)
  }
})

// Miri is deferred to CI (nidus-a9l): the required `Miri` job runs the suite on every
// PR, and the interpreter is too slow to gate the local loop. Named, never in `run`.
test('lanes: codec changes name Miri as CI-covered, not a local run', () => {
  const r = lanes(['src/log/mod.rs'])
  eq(r.ci.map(l => l.recipe), ['just miri'], 'ci-covered')
  eq(r.run.map(l => l.recipe).includes('just miri'), false, 'not local')
})

// The search-parity epic (#75) added these pure-logic kernels; the lane map missed
// them on day one, which is the whole reason it is tested rather than described.
test('lanes: the ranking/fusion/BM25 kernels are Miri-covered too', () => {
  for (const f of ['src/fuse.rs', 'src/annotate.rs', 'src/store/rank.rs', 'src/store/aggregate.rs', 'src/store/text.rs', 'src/filter/pattern.rs']) {
    eq(lanes([f]).ci.map(l => l.recipe).includes('just miri'), true, `miri for ${f}`)
  }
})

// A deferred lane that vanished from the report entirely would read as "no Miri
// needed" — the section must still name it so the PR author expects the CI lane.
test('lanes: the CI-covered section is printed, not silently dropped', () => {
  const out = formatLanes(lanes(['src/log/mod.rs']))
  eq(/CI enforces these on the PR/.test(out), true, `ci section present in: ${out}`)
  eq(/just miri/.test(out), true, 'names the recipe')
})

test('lanes: cluster tests are manual, not automatic', () => {
  const r = lanes(['tests/e2e/cluster.rs'])
  eq(r.run.map(l => l.recipe), [], 'nothing automatic')
  eq(r.manual.map(l => l.recipe), ['just e2e-services-up && just test-e2e-cluster'], 'manual')
})

test('lanes: skill-only changes are inert', () => {
  const r = lanes(['.claude/skills/nidus/SKILL.md'])
  eq(r.run.length, 0, 'no lanes')
  eq(r.inert, ['.claude/skills/nidus/SKILL.md'], 'inert')
})

test('lanes: an unmapped path is reported, not swallowed', () => {
  eq(lanes(['weird/place.txt']).unmatched, ['weird/place.txt'], 'unmatched')
})

test('lanes: an SDK README does not run that SDK suite', () => {
  const r = lanes(['sdks/go/README.md'])
  eq(r.run.length, 0, 'no lanes')
  eq(r.inert, ['sdks/go/README.md'], 'inert')
})

test('lanes: each SDK gets its own lane', () => {
  eq(lanes(['sdks/go/client.go']).run.length, 1, 'go')
  eq(lanes(['sdks/python/src/nidus/client.py']).run.length, 1, 'py')
  eq(lanes(['sdks/js/src/client.ts']).run.length, 1, 'js')
})

// An SDK change is the one case where the unit lanes are worth least: they run against a
// mocked transport, so they pass just as happily against a shape the server never emits.
test('lanes: an SDK change reports the real-server contract lane (#172)', () => {
  for (const f of ['sdks/js/src/client.ts', 'sdks/go/client.go', 'sdks/python/src/nidus/client.py']) {
    const manual = lanes([f]).manual.map(l => l.recipe).join('\n')
    eq(/test:integration/.test(manual), true, `js integration suite named for ${f}`)
    eq(/-tags integration/.test(manual), true, `go integration suite named for ${f}`)
    eq(/test_integration\.py/.test(manual), true, `python integration suite named for ${f}`)
  }
})

// The lane table's claim is "run this and you have run what CI runs". Each SDK's CI job
// has steps beyond its test command, and a lane missing them is a subset wearing a total.
test('lanes: each SDK lane runs every step its CI job runs (#172)', () => {
  const recipeFor = f => lanes([f]).run.map(l => l.recipe).join('\n')
  const js = recipeFor('sdks/js/src/client.ts')
  for (const step of ['npm run typecheck', 'npm run test:unit', 'npm run build']) {
    eq(js.includes(step), true, `js lane runs ${step}`)
  }
  const go = recipeFor('sdks/go/client.go')
  for (const step of ['gofmt', 'go vet', 'go test']) {
    eq(go.includes(step), true, `go lane runs ${step}`)
  }
  const py = recipeFor('sdks/python/src/nidus/client.py')
  for (const step of ['ruff check', 'ruff format --check', 'mypy src', 'pytest']) {
    eq(py.includes(step), true, `python lane runs ${step}`)
  }
})

// ── ci-guard: the per-job skip oracle (nidus-0bs) ───────────────────────────
// A wrong `skip` silently unguards a required check's work, so the failure modes
// are pinned: docs-only skips, src runs, empty runs, unknown job throws.

test('ci-guard: a docs/skill-only change skips the Rust jobs', () => {
  for (const job of ['test', 'test-extended', 'miri', 'miri-integration', 'release', 'build-budget']) {
    eq(ciGuard(job, ['docs/src/content/docs/api.md', '.claude/skills/nidus/SKILL.md', 'README.md']).run, false, `${job} skips`)
  }
})

test('ci-guard: any src or manifest change runs every Rust job', () => {
  for (const f of ['src/lib.rs', 'Cargo.toml', 'Cargo.lock', 'tests/integration.rs', 'rust-toolchain.toml']) {
    eq(ciGuard('miri', ['docs/x.md', f]).run, true, `miri runs for ${f}`)
  }
})

test('ci-guard: an SDK change runs sdk-integration but not miri', () => {
  eq(ciGuard('sdk-integration', ['sdks/js/src/client.ts']).run, true, 'sdk-integration runs')
  eq(ciGuard('miri', ['sdks/js/src/client.ts']).run, false, 'miri skips')
})

test('ci-guard: a workflow edit runs its own jobs', () => {
  eq(ciGuard('test', ['.github/workflows/ci.yml']).run, true, 'ci.yml runs test')
  eq(ciGuard('e2e', ['.github/workflows/integration.yml']).run, true, 'integration.yml runs e2e')
  eq(ciGuard('e2e', ['scripts/e2e-services.sh']).run, true, 'services script runs e2e')
})

test('ci-guard: the wasm lanes (nidus-y67) key off src, the justfile and the binding', () => {
  for (const job of ['wasm', 'wasm-e2e']) {
    eq(ciGuard(job, ['src/backend/opfs.rs']).run, true, `${job} runs for the backend`)
    // The recipes ARE the lane, so a justfile edit must not skip it.
    eq(ciGuard(job, ['justfile']).run, true, `${job} runs for the justfile`)
    eq(ciGuard(job, ['bindings/wasm/src/lib.rs']).run, true, `${job} runs for the binding`)
    eq(ciGuard(job, ['docs/src/content/docs/guides/wasm.md']).run, false, `${job} skips docs`)
  }
})

test('ci-guard: the browser bring-up script runs wasm-e2e only', () => {
  eq(ciGuard('wasm-e2e', ['scripts/e2e-wasm.sh']).run, true, 'wasm-e2e runs')
  // ci.yml's cheap lane never invokes the script, so it must not be woken by it.
  eq(ciGuard('wasm', ['scripts/e2e-wasm.sh']).run, false, 'wasm skips')
  eq(ciGuard('test', ['scripts/e2e-wasm.sh']).run, false, 'test skips')
})

test('ci-guard: an empty diff runs everything — a guard that saw nothing must not skip', () => {
  eq(ciGuard('test', []).run, true, 'fail open')
})

test('ci-guard: an unknown job is an error, never a skip', () => {
  let threw = false
  try { ciGuard('renamed-job', ['src/lib.rs']) } catch { threw = true }
  eq(threw, true, 'throws')
})

// ── #173: a check must not report success without having run ───────────────

test('laws: an empty changeset is itself a finding, not a clean result', () => {
  const out = laws.emptyScope([], 'range')
  eq(out.length, 1, 'one finding')
  eq(out[0].id, 'empty-scope', 'id')
})

test('laws: a non-empty changeset reports no empty-scope finding', () => {
  eq(laws.emptyScope(['src/lib.rs'], 'range').length, 0, 'silent when there is work')
})

// The asymmetry this closes: the scope disclosure used to be printed only when there
// were no findings, so any finding hid it. Assert both together rather than inferring it.
test('laws: the empty scope is reported even when another finding is present', () => {
  const other = laws.commentCap('// a\n// b\n// c\n// d\nfn f() {}\n', null, 'src/x.rs')
  eq(other.length > 0, true, 'fixture really does produce a finding')
  const all = [...laws.emptyScope([], 'range'), ...other]
  eq(all.filter(f => f.id === 'empty-scope').length, 1, 'empty-scope survives alongside it')
})

// Three distinct versions on purpose: base 0.58, head 0.59, origin/main 0.60. A
// two-version fixture passes under both the old and the new logic and proves nothing.
const CARGO = v => `[package]\nname = "nidus"\nversion = "${v}"\n`

test('laws: a version below origin/main is caught (#173)', () => {
  const out = laws.versionBackwards(CARGO('0.58.0'), CARGO('0.59.0'), CARGO('0.60.0'), ['Cargo.toml'])
  eq(out.length, 1, 'one finding')
  eq(out[0].id, 'version-backwards', 'id')
  eq(out[0].severity, 'error', 'error, not a warning')
})

// The case both existing laws pass clean: base != head, so versionBump is satisfied and
// docsVersionSync then checks the snippet against head, which matches. Neither can see it.
test('laws: the backwards case slips past versionBump and docsVersionSync (#173)', () => {
  const base = CARGO('0.60.0'), head = CARGO('0.59.0')
  eq(laws.versionBump(base, head, ['src/lib.rs']).length, 0, 'versionBump sees a bump')
  eq(laws.docsVersionSync(base, head, { 'README.md': 'nidus = "0.59"' }).length, 0, 'snippet matches head')
  eq(laws.versionBackwards(base, head, CARGO('0.60.0'), ['Cargo.toml']).length, 1, 'only the new law catches it')
})

test('laws: only a version strictly above origin/main is fine', () => {
  eq(laws.versionBackwards(CARGO('0.60.0'), CARGO('0.61.0'), CARGO('0.60.0'), ['Cargo.toml']).length, 0, 'ahead')
  eq(laws.versionBackwards(CARGO('0.9.0'), CARGO('0.10.0'), CARGO('0.9.0'), ['Cargo.toml']).length, 0, 'numeric, not lexical')
})

// nidus-7nk: #213 bumped 0.67->0.68 while #212 took 0.68 and merged first, so v0.68.0
// already existed and release.yml skipped every publish job. The law ran in CI and passed,
// because it allowed equality. Equality is the collision, not a safe no-op.
test('laws: a version equal to origin/main is caught (nidus-7nk)', () => {
  const out = laws.versionBackwards(CARGO('0.67.0'), CARGO('0.68.0'), CARGO('0.68.0'), ['Cargo.toml'])
  eq(out.length, 1, 'one finding')
  eq(out[0].id, 'version-backwards', 'id')
  eq(out[0].severity, 'error', 'error, not a warning')
})

// nidus-zin: the residual gap. A version ABOVE origin/main whose tag exists still ships
// nothing, and versionBackwards passes it clean — it never sees a tag.
test('laws: a version ahead of main but already tagged is caught (nidus-zin)', () => {
  const base = CARGO('0.74.0'), head = CARGO('0.75.0')
  const tags = new Set(['v0.74.0', 'v0.75.0'])
  eq(laws.versionBackwards(base, head, CARGO('0.74.0'), ['Cargo.toml']).length, 0,
    'versionBackwards passes it: 0.75.0 IS above main')
  const out = laws.versionAlreadyTagged(base, head, tags, ['Cargo.toml', 'src/lib.rs'])
  eq(out.length, 1, 'one finding')
  eq(out[0].id, 'version-already-tagged', 'id')
  eq(out[0].severity, 'error', 'blocks: claiming a released version always ships nothing')
  eq(/v0\.75\.0/.test(out[0].summary), true, 'names the tag it collides with')
})

// What lets the finding above be a hard error rather than a warning: an unreadable or
// absent tag list must produce NO finding, so a fresh or offline clone never sees a false one.
test('laws: an unreadable tag list produces no version-already-tagged finding', () => {
  const base = CARGO('0.74.0'), head = CARGO('0.75.0'), ch = ['Cargo.toml']
  eq(laws.versionAlreadyTagged(base, head, new Set(), ch).length, 0, 'empty set (fresh clone)')
  eq(laws.versionAlreadyTagged(base, head, null, ch).length, 0, 'null')
  eq(laws.versionAlreadyTagged(base, head, undefined, ch).length, 0, 'undefined')
  eq(laws.versionAlreadyTagged(base, head, {}, ch).length, 0, 'not a Set')
})

test('laws: an untagged bump, and an unchanged version, stay clean', () => {
  const tags = new Set(['v0.74.0']), ch = ['Cargo.toml']
  eq(laws.versionAlreadyTagged(CARGO('0.74.0'), CARGO('0.75.0'), tags, ch).length, 0, 'tag is free')
  // No bump at all is versionBump's business, not this law's.
  eq(laws.versionAlreadyTagged(CARGO('0.74.0'), CARGO('0.74.0'), tags, ch).length, 0, 'unchanged')
})

// Only when nidus itself is changing, same exemption versionBump applies. A skill/docs/CI
// branch is not competing for a release, and a stale two-dot base can make the version look
// changed on a branch that never touched Cargo.toml — that must not fire on someone else's bump.
test('laws: version-already-tagged fires only when Cargo.toml itself changed', () => {
  const base = CARGO('0.74.0'), head = CARGO('0.75.0'), tags = new Set(['v0.75.0'])
  eq(laws.versionAlreadyTagged(base, head, tags, ['.claude/skills/nidus/SKILL.md']).length, 0,
    'skill-only branch, version diff came from a stale base')
  eq(laws.versionAlreadyTagged(base, head, tags, ['docs/src/content/docs/guides/search.md']).length, 0,
    'docs-only')
  eq(laws.versionAlreadyTagged(base, head, tags, ['src/lib.rs']).length, 0,
    'src changed but Cargo.toml did not: versionBump owns that, not this law')
  eq(laws.versionAlreadyTagged(base, head, tags, []).length, 0, 'no changed list at all')
  eq(laws.versionAlreadyTagged(base, head, tags, ['Cargo.toml']).length, 1, 'a real bump fires')
})

// The reason equality was originally allowed: a branch may edit Cargo.toml for a
// dependency and never claim a version. Gating on base !== head keeps that clean.
test('laws: a dependency-only Cargo.toml edit at the same version is fine (nidus-7nk)', () => {
  eq(laws.versionBackwards(CARGO('0.68.0'), CARGO('0.68.0'), CARGO('0.68.0'), ['Cargo.toml']).length, 0, 'version never claimed')
})

// A branch that never touches Cargo.toml cannot move the version backwards: the merge
// keeps main's value. Firing there would nag every skill-only PR into a false bump.
test('laws: an untouched Cargo.toml is not a backwards version', () => {
  eq(laws.versionBackwards(CARGO('0.58.0'), CARGO('0.59.0'), CARGO('0.60.0'), ['.claude/skills/nidus/SKILL.md']).length, 0, 'not touched')
})

// The lanes half of #173, and the more dangerous half: a missing lane costs the
// verification itself, not just a misread report. Structural, so --json carries it too.
test('lanes: the result states how many files it examined (#173)', () => {
  eq(lanes([]).examined, 0, 'nothing examined')
  eq(lanes(['src/lib.rs', 'src/store/read.rs']).examined, 2, 'counts what it was given')
})

// "No automated lane applies" over zero files and over a file that genuinely needs no
// lane must not print the same thing — one is an answer, the other is an empty question.
test('lanes: an empty scope does not read as "no lane applies" (#173)', () => {
  const empty = formatLanes(lanes([]))
  const inert = formatLanes(lanes(['LICENSE']))
  eq(/examined 0 file/i.test(empty), true, `empty scope is disclosed: ${empty}`)
  eq(empty === inert, false, 'the two answers are distinguishable')
})

// ── stale-ticket: intent, not every mention ────────────────────────────────

const TITLES = { '#42': 'a real open issue', '#43': 'another open issue' }

test('stale-ticket: an unaddressed issue still warns', () => {
  eq(ids(laws.unclosedTickets(new Set(['#42']), new Set(), TITLES)), ['stale-ticket'], 'findings')
})

test('stale-ticket: a Closes line suppresses it', () => {
  eq(ids(laws.unclosedTickets(new Set(['#42']), new Set(['#42']), TITLES)), [], 'findings')
})

// The finding's own remediation says "or leave it as Refs", but Refs was not accepted
// by anything — following the printed advice left the warning exactly where it was.
test('stale-ticket: an acknowledged ref suppresses it without claiming closure', () => {
  eq(ids(laws.unclosedTickets(new Set(['#42']), new Set(), TITLES, new Set(['#42']))), [], 'findings')
})

test('stale-ticket: acknowledging one issue does not silence another', () => {
  const found = laws.unclosedTickets(new Set(['#42', '#43']), new Set(), TITLES, new Set(['#42']))
  eq(ids(found), ['stale-ticket'], 'one finding')
  eq(found[0].summary.startsWith('#43'), true, 'the un-acknowledged one')
})

// ── fleet ──────────────────────────────────────────────────────────────────

const SELF = { dir: '/r/nidus', commonDir: '/r/nidus/.git', remote: 'git@github.com:duckedup/nidus.git', mainSha: 'aaa', login: 'austin' }
const wt = (name, slug, over = {}) => ({
  name, dir: `/r/nidus/.claude/worktrees/${slug}`, isRepo: true, commonDir: '/r/nidus/.git',
  remote: SELF.remote, branch: slug, dirty: false, mainSha: 'aaa', ...over,
})

test('fleet: worktrees off one clone are the clean case', () => {
  eq(ids(fleet.treeFindings([wt('a', 'x'), wt('b', 'y')], SELF)), [], 'findings')
})

test('fleet: two peers in one directory is an error', () => {
  const found = fleet.treeFindings([wt('a', 'x'), wt('b', 'x')], SELF)
  eq(ids(found), ['fleet-shared-tree'], 'findings')
})

test('fleet: a peer in the coordinator own tree is an error', () => {
  const found = fleet.treeFindings([wt('a', 'x', { dir: '/r/nidus' })], SELF)
  eq(found.filter(f => f.id === 'fleet-shared-tree').length, 1, 'shared')
})

test('fleet: the coordinator own row is not a collision with itself', () => {
  const me = wt('coordinator', 'x', { dir: '/r/nidus', commonDir: '/r/nidus/.git', branch: 'austin/141', self: true })
  eq(ids(fleet.treeFindings([me, wt('a', 'y')], SELF)), [], 'findings')
})

test('fleet: a trailing slash does not hide a shared tree', () => {
  const found = fleet.treeFindings([wt('a', 'x'), wt('b', 'x', { dir: '/r/nidus/.claude/worktrees/x/' })], SELF)
  eq(ids(found), ['fleet-shared-tree'], 'findings')
})

test('fleet: a separate clone of the same remote is a warning, not an error', () => {
  const found = fleet.treeFindings([wt('a', 'x', { dir: '/r/n2', commonDir: '/r/n2/.git' })], SELF)
  eq(ids(found), ['fleet-separate-clone'], 'findings')
  eq(found[0].severity, 'warn', 'severity')
})

test('fleet: a foreign remote is an error', () => {
  const found = fleet.treeFindings([wt('a', 'x', { commonDir: '/r/other/.git', remote: 'git@github.com:someone/fork.git' })], SELF)
  eq(found.some(f => f.id === 'fleet-foreign-remote'), true, 'foreign')
})

test('fleet: ssh and https spellings of one remote match', () => {
  const found = fleet.treeFindings([wt('a', 'x', { remote: 'https://github.com/duckedup/nidus' })], SELF)
  eq(ids(found), [], 'findings')
})

test('fleet: a peer on main, dirty, or behind is warned about', () => {
  eq(ids(fleet.treeFindings([wt('a', 'x', { branch: 'main' })], SELF)), ['fleet-on-main'], 'on main')
  eq(ids(fleet.treeFindings([wt('a', 'x', { dirty: true })], SELF)), ['fleet-dirty-tree'], 'dirty')
  eq(ids(fleet.treeFindings([wt('a', 'x', { mainSha: 'bbb' })], SELF)), ['fleet-stale-main'], 'stale')
})

test('fleet: a peer with no known cwd cannot be dispatched', () => {
  eq(ids(fleet.treeFindings([{ name: 'a', dir: null }], SELF)), ['fleet-no-tree'], 'findings')
})

test('fleet: a parked queue has no tree and that is fine', () => {
  eq(ids(fleet.treeFindings([{ name: 'backlog', dir: null, unassigned: true }], SELF)), [], 'findings')
})

test('fleet: a parked queue still has its tickets checked', () => {
  const peers = [{ name: 'backlog', dir: null, unassigned: true, queue: [9] }]
  eq(ids(fleet.issueFindings(peers, { 9: { state: 'CLOSED', linkedPrs: [] } })), ['fleet-issue-closed'], 'findings')
})

const OPEN = { state: 'OPEN', assignees: [], linkedPrs: [] }

test('fleet: an open unclaimed queue is clear', () => {
  const peers = [{ name: 'a', queue: [141] }]
  eq(ids(fleet.issueFindings(peers, { 141: OPEN }, { login: 'austin' })), [], 'findings')
})

test('fleet: the same ticket in two queues is an error', () => {
  const peers = [{ name: 'a', queue: [141] }, { name: 'b', queue: [141] }]
  eq(ids(fleet.issueFindings(peers, { 141: OPEN })), ['fleet-double-assigned'], 'findings')
})

test('fleet: a closed or missing ticket is an error', () => {
  eq(ids(fleet.issueFindings([{ name: 'a', queue: [9] }], { 9: { ...OPEN, state: 'CLOSED' } })), ['fleet-issue-closed'], 'closed')
  eq(ids(fleet.issueFindings([{ name: 'a', queue: [9] }], {})), ['fleet-issue-missing'], 'missing')
})

test('fleet: a ticket already assigned elsewhere warns, but not against yourself', () => {
  const taken = { 9: { ...OPEN, assignees: ['someone'] } }
  eq(ids(fleet.issueFindings([{ name: 'a', queue: [9] }], taken, { login: 'austin' })), ['fleet-issue-taken'], 'taken')
  const mine = { 9: { ...OPEN, assignees: ['austin'] } }
  eq(ids(fleet.issueFindings([{ name: 'a', queue: [9] }], mine, { login: 'austin' })), [], 'mine')
})

test('fleet: an open PR already closing the ticket warns', () => {
  const issues = { 9: { ...OPEN, linkedPrs: [{ number: 50, state: 'OPEN' }] } }
  eq(ids(fleet.issueFindings([{ name: 'a', queue: [9] }], issues)), ['fleet-issue-has-pr'], 'findings')
})

test('fleet: a merged PR on the ticket does not warn', () => {
  const issues = { 9: { ...OPEN, linkedPrs: [{ number: 50, state: 'MERGED' }] } }
  eq(ids(fleet.issueFindings([{ name: 'a', queue: [9] }], issues)), [], 'findings')
})

const MAIN = { path: '/r/nidus', branch: 'main', isMain: true, dirty: false, hasCommits: false }
const agentWt = (id, over = {}) => ({
  path: `/r/nidus/.claude/worktrees/agent-${id}`, branch: `worktree-agent-${id}`,
  isMain: false, dirty: false, hasCommits: false, ...over,
})

test('fleet: the main tree and declared peer trees are never orphans', () => {
  const peer = { path: '/r/nidus/.claude/worktrees/x', branch: 'austin/141', isMain: false, dirty: false, hasCommits: false }
  eq(ids(fleet.orphanFindings([MAIN, peer], [{ name: 'a', dir: '/r/nidus/.claude/worktrees/x' }], SELF)), [], 'findings')
})

test('fleet: an agent worktree nobody claims is an orphan', () => {
  const found = fleet.orphanFindings([MAIN, agentWt('a0c1')], [], SELF)
  eq(ids(found), ['fleet-orphan-agent-worktree'], 'findings')
  eq(/prune/.test(found[0].detail), true, 'prune suggested when clean')
})

test('fleet: an orphan carrying commits says prune will not reclaim it', () => {
  const found = fleet.orphanFindings([MAIN, agentWt('a0c1', { hasCommits: true })], [], SELF)
  eq(/--force/.test(found[0].detail), true, 'force')
  eq(/will not reclaim/.test(found[0].detail), true, 'explains why prune fails')
})

test('fleet: a non-agent worktree outside the plan is reported separately', () => {
  const stray = { path: '/r/nidus/.claude/worktrees/old', branch: 'austin/99', isMain: false, dirty: false, hasCommits: false }
  eq(ids(fleet.orphanFindings([MAIN, stray], [], SELF)), ['fleet-unaccounted-worktree'], 'findings')
})

test('fleet: two peers claiming one file is flagged for sequencing', () => {
  const peers = [
    { name: 'a', surface: { 139: ['src/cli/mod.rs'] } },
    { name: 'b', surface: { 141: ['src/cli/mod.rs', 'src/store/mod.rs'] } },
  ]
  const found = fleet.overlapFindings(peers)
  eq(ids(found), ['fleet-file-overlap'], 'findings')
  eq(found[0].subject, 'src/cli/mod.rs', 'subject')
})

test('fleet: one peer holding a file across its own tickets is not an overlap', () => {
  const peers = [{ name: 'a', surface: { 139: ['src/cli/mod.rs'], 140: ['src/cli/mod.rs'] } }]
  eq(ids(fleet.overlapFindings(peers)), [], 'findings')
})

test('fleet: rehydrate derives shipped / in-review / in-flight / queued', () => {
  const peers = [{ name: 'own-138', queue: [138, 139, 140, 141] }]
  const issues = {
    138: { state: 'CLOSED', assignees: [], linkedPrs: [{ number: 60, state: 'MERGED' }] },
    139: { state: 'OPEN', assignees: [], linkedPrs: [{ number: 61, state: 'OPEN' }] },
    140: { state: 'OPEN', assignees: [], linkedPrs: [] },
    141: { state: 'OPEN', assignees: [], linkedPrs: [] },
  }
  const trees = [MAIN, { path: '/r/nidus/.claude/worktrees/x', branch: 'austin/140-sweep', isMain: false }]
  const rows = fleet.rehydrate(peers, issues, trees)
  eq(rows.map(r => `${r.issue}:${r.state}`), ['138:shipped', '139:in-review', '140:in-flight', '141:queued'], 'states')
  eq(rows[0].pr, 60, 'merged pr surfaced')
})

test('fleet: rehydrate does not confuse #14 with #141', () => {
  const trees = [MAIN, { path: '/r/nidus/.claude/worktrees/y', branch: 'austin/141-profile', isMain: false }]
  const rows = fleet.rehydrate([{ name: 'a', queue: [14] }], { 14: { state: 'OPEN', linkedPrs: [] } }, trees)
  eq(rows[0].state, 'queued', 'no false branch match')
})

test('fleet: a pushed remote branch counts as in-flight without a local worktree', () => {
  const rows = fleet.rehydrate([{ name: 'peer', queue: [138] }], { 138: { state: 'OPEN', linkedPrs: [] } },
    [MAIN], ['origin/austin/138-backup-verify'])
  eq(rows[0].state, 'in-flight', 'state')
  eq(rows[0].branch, 'origin/austin/138-backup-verify', 'branch')
})

test('fleet: a remote branch for #14 does not mark #141 in-flight', () => {
  const rows = fleet.rehydrate([{ name: 'peer', queue: [141] }], { 141: { state: 'OPEN', linkedPrs: [] } },
    [MAIN], ['origin/austin/14-something'])
  eq(rows[0].state, 'queued', 'no false match')
})

test('fleet: a bundled sibling inherits the in-flight state', () => {
  const peers = [{ name: 'peer', queue: [138, 152], bundles: [[138, 152]] }]
  const issues = { 138: { state: 'OPEN', linkedPrs: [] }, 152: { state: 'OPEN', linkedPrs: [] } }
  const rows = fleet.rehydrate(peers, issues, [MAIN], ['origin/austin/138-backup-verify'])
  eq(rows.map(r => r.state), ['in-flight', 'in-flight'], 'both')
  eq(rows[1].via, '138', 'names what it inherited from')
})

test('fleet: bundling does not invent progress when nothing has started', () => {
  const peers = [{ name: 'peer', queue: [138, 152], bundles: [[138, 152]] }]
  const issues = { 138: { state: 'OPEN', linkedPrs: [] }, 152: { state: 'OPEN', linkedPrs: [] } }
  eq(fleet.rehydrate(peers, issues, [MAIN], []).map(r => r.state), ['queued', 'queued'], 'both queued')
})

test('fleet: an issue closed with no PR still reads as shipped', () => {
  const rows = fleet.rehydrate([{ name: 'a', queue: [9] }], { 9: { state: 'CLOSED', linkedPrs: [] } }, [MAIN])
  eq(rows[0].state, 'shipped', 'state')
})

test('fleet: two branches claiming one version is an error', () => {
  const inflight = [{ ref: 'origin/a', version: '0.57.0' }, { ref: 'origin/b', version: '0.57.0' }]
  const found = fleet.versionFindings(inflight, '0.56.1', new Set(['v0.56.1']))
  eq(ids(found), ['fleet-version-collision'], 'findings')
})

const BEHAV = [/^src\//, /^sdks\//, /^Cargo\.toml$/]

test('fleet: a skill-only branch owes no bump and collides with nothing', () => {
  const b = [{ ref: 'origin/s1', version: '0.56.1', changed: ['.claude/skills/nidus/SKILL.md'] },
             { ref: 'origin/s2', version: '0.56.1', changed: ['docs/x.md'] }]
  eq(ids(fleet.versionFindings(b, '0.56.1', new Set(['v0.56.1']), new Set(['s1', 's2']), BEHAV)), [], 'exempt')
})

test('fleet: a src-touching branch at a tagged version still fires', () => {
  const b = [{ ref: 'origin/c', version: '0.56.1', changed: ['src/lib.rs'] }]
  eq(ids(fleet.versionFindings(b, '0.56.1', new Set(['v0.56.1']), new Set(['c']), BEHAV)), ['fleet-version-released'], 'fires')
})

test('fleet: an exempt branch does not collide with a releasing one', () => {
  const b = [{ ref: 'origin/s', version: '0.57.0', changed: ['docs/x.md'] },
             { ref: 'origin/c', version: '0.57.0', changed: ['src/lib.rs'] }]
  eq(ids(fleet.versionFindings(b, '0.56.1', new Set(), new Set(), BEHAV)), [], 'no phantom collision')
})

test('fleet: a long-landed branch below main is not in flight', () => {
  const stale = [{ ref: 'origin/old-a', version: '0.12.0' }, { ref: 'origin/old-b', version: '0.12.0' }]
  eq(ids(fleet.versionFindings(stale, '0.56.1', new Set())), [], 'squash-merged branches stay quiet')
})

test('fleet: a branch with an open PR is in flight even at main version', () => {
  const b = [{ ref: 'origin/live', version: '0.56.1' }]
  eq(ids(fleet.versionFindings(b, '0.56.1', new Set(), new Set(['live']))), ['fleet-version-stale'], 'findings')
})

test('fleet: distinct versions ahead of main are clean', () => {
  const inflight = [{ ref: 'origin/a', version: '0.57.0' }, { ref: 'origin/b', version: '0.58.0' }]
  eq(ids(fleet.versionFindings(inflight, '0.56.1', new Set(['v0.56.1']))), [], 'findings')
})

test('fleet: a branch with a PR but no bump ships no release', () => {
  const pr = new Set(['a'])
  eq(ids(fleet.versionFindings([{ ref: 'origin/a', version: '0.56.1' }], '0.56.1', new Set(), pr)), ['fleet-version-stale'], 'equal')
  eq(ids(fleet.versionFindings([{ ref: 'origin/a', version: '0.56.0' }], '0.56.1', new Set(), pr)), ['fleet-version-stale'], 'behind')
})

test('fleet: an already-tagged version is caught before the stale check', () => {
  const found = fleet.versionFindings([{ ref: 'origin/a', version: '0.56.1' }], '0.56.1', new Set(['v0.56.1']), new Set(['a']))
  eq(ids(found), ['fleet-version-released'], 'findings')
})

test('fleet: version compare is numeric, not lexical', () => {
  eq(ids(fleet.versionFindings([{ ref: 'origin/a', version: '0.10.0' }], '0.9.0', new Set())), [], '0.10 > 0.9 and alone')
  const two = [{ ref: 'origin/a', version: '0.10.0' }, { ref: 'origin/b', version: '0.9.1' }]
  eq(ids(fleet.versionFindings(two, '0.9.0', new Set())), [], 'both ahead, both distinct')
})

// ── preflight (nidus-3xx) ──────────────────────────────────────────────────

const clean = { fetched: true, branch: 'austin/x', onMain: false, dirty: false, behind: 0 }

test('preflight: a fresh branch off a fetched main is clear', () => {
  eq(ids(pre.preflight(clean)), [], 'findings')
})

test('preflight: --no-fetch is itself the finding', () => {
  eq(ids(pre.preflight({ ...clean, fetched: false })), ['preflight-no-fetch'], 'findings')
})

test('preflight: behind origin/main blocks, and names the count', () => {
  const found = pre.preflight({ ...clean, behind: 12 })
  eq(ids(found), ['preflight-stale-base'], 'findings')
  eq(found[0].summary.includes('12'), true, 'the count is in the summary')
})

test('preflight: on main reports that instead of staleness', () => {
  eq(ids(pre.preflight({ ...clean, onMain: true, branch: 'main', behind: 3 })), ['preflight-on-main'], 'one finding, not two')
})

test('preflight: a dirty tree warns but does not block', () => {
  const found = pre.preflight({ ...clean, dirty: true })
  eq(found.map(f => f.severity), ['warn'], 'severity')
})

test('preflight: no --issue means no ticket findings at all', () => {
  eq(ids(pre.preflight({ ...clean, issue: null })), [], 'findings')
})

test('preflight: a closed ticket shipped by a merged PR blocks on both counts', () => {
  const issue = { number: 'lvo.1', state: 'CLOSED', assignees: [], linkedPrs: [{ number: 218, state: 'MERGED' }] }
  eq(ids(pre.preflight({ ...clean, issue })), ['preflight-ticket-closed', 'preflight-ticket-shipped'], 'findings')
})

test('preflight: an open PR on the ticket blocks a second branch', () => {
  const issue = { number: 'tx2', state: 'OPEN', assignees: [], linkedPrs: [{ number: 219, state: 'OPEN' }] }
  const found = pre.preflight({ ...clean, issue })
  eq(ids(found), ['preflight-ticket-in-pr'], 'findings')
  eq(found[0].summary.includes('#219'), true, 'the PR number is in the summary')
})

test('preflight: assigned to me is not taken; assigned to someone else is', () => {
  const mine = { number: '7', state: 'OPEN', assignees: ['Austin Riendeau'], linkedPrs: [] }
  eq(ids(pre.preflight({ ...clean, issue: mine, me: ['ariendeau', 'Austin Riendeau'] })), [], 'mine')
  eq(ids(pre.preflight({ ...clean, issue: mine, me: ['someone-else'] })), ['preflight-ticket-taken'], 'theirs')
})

test('preflight: a ticket bd cannot resolve warns rather than being invented', () => {
  eq(ids(pre.preflight({ ...clean, issue: { number: 'zzz', unknown: true } })), ['preflight-ticket-unknown'], 'findings')
})

test('preflight: an existing remote branch for the ticket is not this branch', () => {
  const issue = { number: '7', state: 'OPEN', assignees: [], linkedPrs: [] }
  eq(ids(pre.preflight({ ...clean, issue, issueBranches: ['origin/austin/x'] })), [], 'our own branch')
  eq(ids(pre.preflight({ ...clean, issue, issueBranches: ['origin/someone/7-thing'] })), ['preflight-branch-exists'], 'a foreign one')
})

test('preflight: next free version skips main, in-flight branches and released tags', () => {
  const claimed = [{ ref: 'origin/a', version: '0.73.0' }]
  eq(pre.nextFreeVersion('0.72.0', claimed, new Set(['v0.72.0'])), '0.74.0', 'past the in-flight claim')
  eq(pre.nextFreeVersion('0.72.0', [], new Set(['v0.72.0', 'v0.73.0'])), '0.74.0', 'past a released tag with no branch')
  eq(pre.nextFreeVersion('0.9.0', [{ ref: 'origin/a', version: '0.10.0' }], new Set()), '0.11.0', 'numeric, not lexical')
})

test('preflight: unpushed commits on local main warn, naming the count', () => {
  const found = pre.preflight({ ...clean, mainAhead: 2 })
  eq(ids(found), ['preflight-unpushed-main'], 'findings')
  eq(found[0].severity, 'warn', 'severity — it does not invalidate this branch\'s diff')
  eq(found[0].summary.includes('2'), true, 'the count is in the summary')
})

test('preflight: a main that is up to date says nothing about being ahead', () => {
  eq(ids(pre.preflight({ ...clean, mainAhead: 0 })), [], 'zero')
  eq(ids(pre.preflight(clean)), [], 'absent')
})

test('preflight: a diverged main reports both directions', () => {
  const found = pre.preflight({ ...clean, behind: 3, mainAhead: 2 })
  eq(ids(found), ['preflight-stale-base', 'preflight-unpushed-main'], 'both, not one')
})

test('preflight: on main, unpushed commits are still reported alongside', () => {
  const found = pre.preflight({ ...clean, onMain: true, branch: 'main', mainAhead: 2 })
  eq(ids(found), ['preflight-on-main', 'preflight-unpushed-main'], 'findings')
})

// ── stale base (nidus-qko) ─────────────────────────────────────────────────

test('stale base: a local ref behind its remote is flagged, with both counts', () => {
  const found = laws.staleBase({ ref: 'main', hasRemote: true, behind: 12, examined: 89, examinedFresh: 10 })
  eq(ids(found), ['stale-base'], 'findings')
  eq(found[0].severity, 'warn', 'severity')
  eq(found[0].detail.includes('89') && found[0].detail.includes('10'), true, 'both counts named')
})

test('stale base: an up-to-date ref is silent', () => {
  eq(ids(laws.staleBase({ ref: 'main', hasRemote: true, behind: 0, examined: 10, examinedFresh: 10 })), [], 'findings')
})

test('stale base: a ref with no remote counterpart is silent', () => {
  eq(ids(laws.staleBase({ ref: 'local-only', hasRemote: false, behind: 0 })), [], 'findings')
})

test('stale base: equal counts do not claim a difference that is not there', () => {
  const found = laws.staleBase({ ref: 'main', hasRemote: true, behind: 2, examined: 10, examinedFresh: 10 })
  eq(ids(found), ['stale-base'], 'still flagged — the range is still wrong')
  eq(found[0].detail.includes('examined 10 file(s)'), false, 'no fabricated count comparison')
})

export function selftest({ json = false } = {}) {
  const failures = []
  for (const c of cases) {
    try { c.fn() } catch (e) { failures.push({ name: c.name, error: e.message }) }
  }
  if (json) {
    console.log(JSON.stringify({ total: cases.length, failed: failures.length, failures }, null, 2))
  } else {
    for (const f of failures) console.log(`✗ ${f.name}\n    ${f.error}`)
    console.log(`${cases.length - failures.length}/${cases.length} detector tests passed`)
  }
  return failures.length ? 1 : 0
}
