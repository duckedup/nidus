// Fixture suite for the detectors. Pure — no repository, no git, no network — so a
// broken detector fails here instead of silently passing a real review.

import * as laws from './laws.mjs'
import { lanes } from './lanes.mjs'

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

test('lanes: codec changes pull Miri', () => {
  eq(lanes(['src/log/mod.rs']).run.map(l => l.recipe).includes('just miri'), true, 'miri')
})

// The search-parity epic (#75) added these pure-logic kernels; the lane map missed
// them on day one, which is the whole reason it is tested rather than described.
test('lanes: the ranking/fusion/BM25 kernels pull Miri too', () => {
  for (const f of ['src/fuse.rs', 'src/annotate.rs', 'src/store/rank.rs', 'src/store/aggregate.rs', 'src/store/text.rs', 'src/filter/pattern.rs']) {
    eq(lanes([f]).run.map(l => l.recipe).includes('just miri'), true, `miri for ${f}`)
  }
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
  eq(lanes(['sdks/go/client.go']).run.map(l => l.recipe), ['cd sdks/go && go test ./...'], 'go')
  eq(lanes(['sdks/python/src/nidus/client.py']).run.map(l => l.recipe), ['cd sdks/python && python -m pytest tests -k "not integration"'], 'py')
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
