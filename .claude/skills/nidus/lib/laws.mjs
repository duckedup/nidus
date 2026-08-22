// Deterministic detectors for nidus's repo laws — the rules in CLAUDE.md that a
// reviewer restates as prose and an LLM reviewer then forgets. Every check here is
// a pure function over text so lib/selftest.mjs can drive it from fixtures.

const finding = (id, severity, file, line, summary, detail) => ({ id, severity, file, line, summary, detail })

// ── 1. The 3-line comment cap ───────────────────────────────────────────────
// Counts a whole block: // and /// plus the /// blank separators between them.
// A ``` doc-example fence is test code, not commentary, so its lines do not count.
// `//!` is exempt: a module/crate doc is the published rustdoc landing page, not
// commentary on code (CLAUDE.md §Conventions).

const COMMENT = /^\s*\/\/(\/|!)?/
const FENCE = /^\s*\/\/[\/!]?\s*```/
const INNER_DOC = /^\s*\/\/!/

export function commentCap(text, addedLines = null, file = '') {
  const lines = text.split('\n')
  const out = []
  let block = null
  const flush = () => {
    if (block && block.counted > 3) {
      const touches = !addedLines || block.lines.some(n => addedLines.has(n))
      if (touches) {
        out.push(finding('comment-cap', 'error', file, block.firstCounted ?? block.start,
          `comment block is ${block.counted} lines — the cap is 3`,
          'CLAUDE.md: a comment earns its place by saying what the code cannot. Rationale longer than three lines belongs in the commit message, the PR, SPEC.md, or a GitHub issue.'))
      }
    }
    block = null
  }
  let inFence = false
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i]
    const lineNo = i + 1
    if (!COMMENT.test(raw)) { inFence = false; flush(); continue }
    if (!block) block = { start: lineNo, counted: 0, lines: [], firstCounted: null }
    block.lines.push(lineNo)
    if (FENCE.test(raw)) { inFence = !inFence; continue }
    if (inFence) continue
    // Exempt per LINE, never per block: blocks break only on a blank/code line, so
    // tainting the block would let one `//!` carry an adjacent `///` over the cap.
    if (INNER_DOC.test(raw)) continue
    if (block.firstCounted === null) block.firstCounted = lineNo
    block.counted++
  }
  flush()
  return out
}

// ── 2. unsafe code ──────────────────────────────────────────────────────────
// The crate is `deny(unsafe_code)` (not `forbid`) for ONE sanctioned mmap call.

const UNSAFE_EXEMPT = /^src\/data\/mmap\.rs$/
const UNSAFE_USE = /\bunsafe\s*(\{|fn\b|impl\b|trait\b)/

export function unsafeUse(text, addedLines, file) {
  if (UNSAFE_EXEMPT.test(file)) return []
  const lines = text.split('\n')
  const out = []
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i]
    if (COMMENT.test(raw) || !UNSAFE_USE.test(raw)) continue
    if (addedLines && !addedLines.has(i + 1)) continue
    out.push(finding('unsafe-code', 'error', file, i + 1,
      'new `unsafe` outside the sanctioned mmap call',
      'src/lib.rs is #![deny(unsafe_code)]; src/data/mmap.rs is the only place allowed to opt out (SPEC §9/§14.6).'))
  }
  return out
}

export function crateAttrWeakened(libText) {
  if (/#!\[(deny|forbid)\(unsafe_code\)\]/.test(libText)) return []
  return [finding('unsafe-attr', 'error', 'src/lib.rs', 1,
    'the crate-level unsafe_code attribute is gone',
    'src/lib.rs must keep #![deny(unsafe_code)] (forbid is too strong only because of the mmap opt-out).')]
}

// ── 3/4. Version bump + the docs snippets that pin major.minor ──────────────

export const BEHAVIOURAL = [/^src\//, /^sdks\//, /^Cargo\.toml$/, /^Dockerfile$/, /^install\.sh$/]

export const versionOf = t => (t.match(/^version\s*=\s*"([^"]+)"/m) || [])[1] || null
const majorMinor = v => (v || '').split('.').slice(0, 2).join('.')

export function versionBump(baseCargo, headCargo, changed) {
  const touched = changed.filter(f => BEHAVIOURAL.some(re => re.test(f)))
  if (!touched.length) return []
  const base = versionOf(baseCargo)
  const head = versionOf(headCargo)
  if (base && head && base !== head) return []
  return [finding('version-bump', 'error', 'Cargo.toml', 1,
    `version stayed at ${head || '?'} despite behavioural changes`,
    `release.yml only cuts a release when the v<version> tag does not already exist, so an un-bumped PR ships NOTHING. Changed: ${touched.slice(0, 6).join(', ')}${touched.length > 6 ? '…' : ''}`)]
}

// A run that examined nothing must not read as a pass. Reported as a finding rather than
// a console line so it survives --json and cannot be hidden by another finding — the
// exact asymmetry that let a vacuous run mask a real comment-cap error (#173).
// `nidus-check laws --base <local ref>` over a ref behind its remote counterpart examines
// a range nobody meant, and reports it as thoroughness rather than as the wrong range
// (nidus-qko). The two file counts are the tell that distinguishes the two.
export function staleBase({ ref, hasRemote, behind = 0, examined = null, examinedFresh = null } = {}) {
  if (!ref || !hasRemote || behind <= 0) return []
  const counts = examined !== null && examinedFresh !== null && examined !== examinedFresh
    ? ` This run examined ${examined} file(s); against origin/${ref} it is ${examinedFresh}.`
    : ''
  return [finding('stale-base', 'warn', ref, 1,
    `local ${ref} is ${behind} commit(s) behind origin/${ref}, so this range is not the one you meant`,
    `A green result over the wrong range says nothing about your work.${counts} Re-run with --base origin/${ref}.`)]
}

export function emptyScope(changed, kind = 'range') {
  if ((changed || []).length) return []
  return [finding('empty-scope', 'warn', 'scope', 1,
    `no files examined — this ${kind} contains no changes, so no law could have failed`,
    'A clean result here says nothing about your work. Commit first, or widen the target with --base/--pr/--path.')]
}

const cmpVersion = (a, b) => {
  const pa = String(a).split('.').map(Number)
  const pb = String(b).split('.').map(Number)
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const d = (pa[i] || 0) - (pb[i] || 0)
    if (d) return d
  }
  return 0
}

// versionBump and docsVersionSync both compare merge-base to head, so neither can see a
// branch claiming a version origin/main already has or has passed — the second merge
// silently releases nothing. Gate on the version having *changed*, not on Cargo.toml
// being touched, so a dependency-only edit at the same version stays clean.
export function versionBackwards(baseCargo, headCargo, originCargo, changed) {
  if (!(changed || []).includes('Cargo.toml')) return []
  const base = versionOf(baseCargo)
  const head = versionOf(headCargo)
  const origin = versionOf(originCargo)
  if (!head || !origin || base === head) return []
  if (cmpVersion(head, origin) > 0) return []
  const how = cmpVersion(head, origin) === 0 ? `is already origin/main's` : `is behind origin/main's`
  return [finding('version-backwards', 'error', 'Cargo.toml', 1,
    `version ${head} ${how} ${origin}`,
    'release.yml only tags a version it has not seen, so merging this ships NOTHING, silently. Pick a version strictly above origin/main.')]
}

// The residual gap versionBackwards cannot see (nidus-zin): a version strictly ABOVE
// origin/main whose tag already exists, so release.yml skips it and the merge ships
// nothing. Reachable when a tag was cut and main reverted, when one was pushed by hand,
// or when a branch is cut from a point a release has since moved past.
//
// An error: claiming a released version ALWAYS ships nothing, so there is no legitimate
// case to soften. The offline/fresh-clone risk is handled by the tag guard below, not by
// severity — no readable tag list means no finding at all.
//
// Gated on this branch actually changing nidus, exactly as versionBump is. Two reasons:
// a skill/docs/CI-only branch is not competing for a release at all, and a stale two-dot
// base makes the version *look* changed on a branch that never touched Cargo.toml, which
// would fire this on someone else's bump (the hole versionBackwards' own gate closes).
export function versionAlreadyTagged(baseCargo, headCargo, tags, changed) {
  const touched = (changed || []).filter(f => BEHAVIOURAL.some(re => re.test(f)))
  if (!touched.includes('Cargo.toml')) return []
  const base = versionOf(baseCargo)
  const head = versionOf(headCargo)
  if (!head || base === head) return []
  if (!tags || typeof tags.has !== 'function' || tags.size === 0) return []
  if (!tags.has(`v${head}`)) return []
  return [finding('version-already-tagged', 'error', 'Cargo.toml', 1,
    `version ${head} is already released as v${head}`,
    'release.yml cuts a release only when the tag is new, so merging this ships NOTHING even though the version is ahead of main. Bump past the tag.')]
}

const SNIPPET_FILES = ['README.md', 'docs/src/content/docs/getting-started.md']

export function docsVersionSync(baseCargo, headCargo, texts) {
  const base = majorMinor(versionOf(baseCargo))
  const head = majorMinor(versionOf(headCargo))
  if (!head || base === head) return []
  const out = []
  for (const file of SNIPPET_FILES) {
    const text = texts[file]
    if (text == null) continue
    if (text.includes(`nidus = "${head}"`)) continue
    const stale = (text.match(/nidus = "([^"]+)"/) || [])[1]
    out.push(finding('docs-version-sync', 'error', file, 1,
      `install snippet still says nidus = "${stale ?? '?'}" but the crate is ${head}`,
      'CLAUDE.md: on a major.minor bump, the snippet in README.md and getting-started.md must match the released crate.'))
  }
  return out
}

// ── 5. Bot-stamped version files must not be hand-edited ────────────────────

const STAMPED = [
  { re: /^charts\/nidus\/Chart\.yaml$/, by: 'helm-publish.yml stamps version and appVersion from Cargo.toml' },
  { re: /^sdks\/js\/package\.json$/, by: 'release.yml stamps the SDK version from Cargo.toml and commits it back' },
  { re: /^sdks\/python\/src\/nidus\/_version\.py$/, by: 'release.yml stamps the SDK version from Cargo.toml and commits it back' },
]

export function botStamped(changed, diffs) {
  const out = []
  for (const f of changed) {
    const rule = STAMPED.find(s => s.re.test(f))
    if (!rule) continue
    const diff = diffs[f] || ''
    if (!/^[+-].*version/im.test(diff)) continue
    out.push(finding('bot-stamped', 'error', f, 1,
      'hand-edited a version that CI stamps',
      `${rule.by} — editing it here just conflicts with the bot commit.`))
  }
  return out
}

// ── 6. New dependencies vs. the build-and-ship budget ──────────────────────

const FORBIDDEN_DEP = /(libduckdb|duckdb|aws-lc|openssl-sys|vendored-openssl|arrow|datafusion|[-_]sys$)/i
const DEP_TABLE = /^\[(?:target\.[^\]]+\.)?(dependencies|dev-dependencies|build-dependencies)(?:\.([A-Za-z0-9_-]+))?\]/

// Names from the dependency tables only. Reading the two Cargo.toml versions beats
// scanning the diff: `+version = "0.44.0"` in [package] is not a new dependency.
export function depNames(cargo) {
  const names = new Set()
  let inTable = false
  for (const line of (cargo || '').split('\n')) {
    const header = line.match(/^\[/) ? line.match(DEP_TABLE) : null
    if (line.startsWith('[')) {
      // Inside `[dependencies.foo]` the following lines are foo's fields, not deps.
      inTable = !!header && !header[2]
      if (header && header[2]) names.add(header[2])
      continue
    }
    if (!inTable) continue
    const m = line.match(/^\s*([A-Za-z0-9_-]+)\s*=/)
    if (m) names.add(m[1])
  }
  return names
}

export function newDeps(baseCargo, headCargo) {
  const out = []
  const before = depNames(baseCargo)
  const added = [...depNames(headCargo)].filter(n => !before.has(n))
  for (const name of added) {
    if (FORBIDDEN_DEP.test(name)) {
      out.push(finding('forbidden-dep', 'error', 'Cargo.toml', 1,
        `new dependency \`${name}\` looks like a bundled-C / heavy tree`,
        'CLAUDE.md forbids bundled C/C++, vendored OpenSSL, aws-lc-sys, and Arrow+DataFusion. This is a design change — raise an issue first.'))
    } else {
      out.push(finding('new-dep', 'warn', 'Cargo.toml', 1,
        `new dependency \`${name}\` — confirm the clean build stays well under a minute`,
        'The guardrail is empirical (measured ~7s, asserted in CI). Judge it by compile time, toolchain weight, and binary bloat.'))
    }
  }
  return out
}

// ── 7. Test placement ──────────────────────────────────────────────────────

export function testPlacement(addedFiles, read) {
  const out = []
  for (const f of addedFiles) {
    if (!/^tests\/[^/]+\.rs$/.test(f)) continue
    const text = read(f) || ''
    const drivesBinary = text.includes('CARGO_BIN_EXE_nidus')
    out.push(finding('test-placement', drivesBinary ? 'error' : 'warn', f, 1,
      'new top-level tests/*.rs file',
      drivesBinary
        ? 'This drives the real binary, so it belongs as a module under tests/e2e/ — each tests/*.rs is its own crate, so a new file means a second copy of the harness.'
        : 'Confirm this belongs here: pure-logic tests go inline per module, file-backed behaviour in tests/, binary-driving suites as modules under tests/e2e/.'))
  }
  return out
}

// ── 8. Miri ignores that are not earning it ────────────────────────────────

// The rule is "do not ignore PURE-LOGIC tests", so flag only a body that touches
// nothing outside the process. Filesystem, network (the sanctioned localhost-mock
// round-trips), subprocesses, clocks and env all legitimately earn the ignore.
const SYSCALLY = new RegExp([
  'sync_all', 'sync_data', 'fsync', 'File::', 'OpenOptions', 'std::fs', 'fs::',
  'tempdir', 'TempDir', 'remove_dir', 'create_dir', 'mmap', 'Nidus::open', 'Store::open',
  'TcpListener', 'TcpStream', 'UdpSocket', 'bind\\(', 'connect\\(', 'mock', 'Mock',
  'spawn', 'Command::', 'thread::', 'SystemTime', 'Instant::', 'env::', 'sleep',
].join('|'))

// A reason may sit after the attribute or on the line directly above it. Recognising only
// the trailing form reported already-documented ignores as bare, which is how #99 came to
// delete two whose reason was written above them (nidus #92).
const DOCUMENTED_IGNORE = /#\[cfg_attr\(miri,\s*ignore\)\]\s*\/\/\s*\S/
const REASON_ABOVE = /^\s*\/\/\/?\s*\S/

export function miriIgnore(text, addedLines, file) {
  const lines = text.split('\n')
  const out = []
  for (let i = 0; i < lines.length; i++) {
    if (!/#\[cfg_attr\(miri,\s*ignore\)\]/.test(lines[i])) continue
    if (addedLines && !addedLines.has(i + 1)) continue
    if (DOCUMENTED_IGNORE.test(lines[i])) continue
    if (i > 0 && REASON_ABOVE.test(lines[i - 1])) continue
    const body = lines.slice(i, i + 40).join('\n')
    if (SYSCALLY.test(body)) continue
    out.push(finding('miri-ignore', 'warn', file, i + 1,
      'test is ignored under Miri but touches nothing outside the process',
      'CLAUDE.md: pure-logic tests (cosine math, glob, filters, codec round-trips) MUST run under Miri — ignore only syscall paths. If the ignore is right (runtime cost, float ULP, or IO the scan cannot see through a helper), record why as `#[cfg_attr(miri, ignore)] // <reason>`; otherwise drop it.'))
  }
  return out
}

// ── 9. Feature gating: binary-only deps must not reach the library ─────────

const BINARY_ONLY = /^\s*use\s+(clap|tokio|axum|tower|http_body|rmcp)\b/m
const LIB_EXEMPT = /^src\/(cli|server|bin)\//

export function featureGating(file, text) {
  if (LIB_EXEMPT.test(file) || !file.startsWith('src/')) return []
  const m = text.match(BINARY_ONLY)
  if (!m) return []
  const line = text.slice(0, text.indexOf(m[0])).split('\n').length
  return [finding('feature-gating', 'error', file, line,
    `library module imports the binary-only crate \`${m[1]}\``,
    'CLAUDE.md: those deps compile only under `cli`/`serve`. Using them from a library module breaks the pure `cargo add nidus` install.')]
}

export function modGating(libText) {
  const out = []
  const lines = libText.split('\n')
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^(?:pub )?mod (cli|server|memory|embed|summarize);/)
    if (!m) continue
    if (/#\[cfg\(/.test(lines[i - 1] || '')) continue
    out.push(finding('mod-gating', 'error', 'src/lib.rs', i + 1,
      `mod ${m[1]} is declared without a #[cfg(feature = …)] gate`,
      'The default build must pull none of the binary/async-edge layers.'))
  }
  return out
}

// ── 10. Issues this change ships but does not close ────────────────────────
// A bare mention does not close on merge, so the issue silently outlives the work
// that finished it (PR #63 found ten such tickets under the previous tracker).
// `acknowledged` is a deliberate third state, not a widening of `closing`: `Refs`/`Part of`
// answer "have you addressed this issue's disposition" without claiming to close it, which
// is what the finding's own remediation text has always advised.
export function unclosedTickets(mentioned = new Set(), closing = new Set(), titles = {}, acknowledged = new Set()) {
  return Array.from(mentioned)
    .filter(ref => !closing.has(ref) && !acknowledged.has(ref) && titles[ref])
    .map(ref => finding('stale-ticket', 'warn', 'PR body', 1,
      `${ref} is worked by this change but nothing states it will be closed`,
      `"${titles[ref]}". CLAUDE.md: close the ticket that ships. Add "Closes nidus-${ref.slice(1)}" to the PR body, or "Refs" if this change does not finish it. Nothing auto-closes — on merge, run bd close nidus-${ref.slice(1)} and bd dolt push.`))
}

export const LAW_IDS = [
  'stale-base',
  'comment-cap', 'unsafe-code', 'unsafe-attr', 'version-bump', 'docs-version-sync',
  'bot-stamped', 'forbidden-dep', 'new-dep', 'test-placement', 'miri-ignore',
  'feature-gating', 'mod-gating', 'stale-ticket',
]
