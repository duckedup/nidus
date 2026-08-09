// Entry point for bin/nidus-check. Wires the pure detectors in laws.mjs and the
// lane map in lanes.mjs to real git/gh IO.

import { lanes, formatLanes } from './lanes.mjs'
import * as laws from './laws.mjs'
import * as fleet from './fleet.mjs'
import * as git from './git.mjs'
import { selftest } from './selftest.mjs'
import { readFileSync } from 'node:fs'

const argv = process.argv.slice(2)
const cmd = argv[0]
const flag = name => {
  const i = argv.indexOf(`--${name}`)
  return i === -1 ? null : (argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : true)
}
const asJson = argv.includes('--json')

function target() {
  return git.resolveTarget({
    base: flag('base') === true ? null : flag('base'),
    head: flag('head') === true ? null : flag('head'),
    pr: flag('pr') === true ? null : flag('pr'),
    path: flag('path') === true ? null : flag('path'),
  })
}

function runLanes() {
  const explicit = flag('paths')
  const files = typeof explicit === 'string' ? explicit.split(',').map(s => s.trim()) : git.changedFiles(target())
  const result = lanes(files)
  if (asJson) { console.log(JSON.stringify({ files, ...result }, null, 2)); return 0 }
  console.log(formatLanes(result))
  return 0
}

const RS = f => f.endsWith('.rs')

function runLaws() {
  const t = target()
  const changed = git.changedFiles(t)
  const added = git.addedFiles(t)
  const addedLines = git.addedLineMap(t)
  const findings = []

  for (const f of changed.filter(RS)) {
    const text = git.readAt(t, f)
    if (text == null) continue
    const lines = addedLines.get(f) || null
    findings.push(...laws.commentCap(text, lines, f))
    findings.push(...laws.unsafeUse(text, lines, f))
    findings.push(...laws.miriIgnore(text, lines, f))
    findings.push(...laws.featureGating(f, text))
  }

  const libText = git.readAt(t, 'src/lib.rs')
  if (libText) {
    findings.push(...laws.crateAttrWeakened(libText))
    findings.push(...laws.modGating(libText))
  }

  // A --path sweep has no base commit, so the laws that compare two revisions
  // (version bump, docs sync, new deps, bot-stamped files) simply do not apply.
  if (t.base) {
    const headCargo = git.readAt(t, 'Cargo.toml') || ''
    const baseCargo = git.readBase(t, 'Cargo.toml') || ''
    findings.push(...laws.versionBump(baseCargo, headCargo, changed))
    findings.push(...laws.docsVersionSync(baseCargo, headCargo, {
      'README.md': git.readAt(t, 'README.md'),
      'docs/src/content/docs/getting-started.md': git.readAt(t, 'docs/src/content/docs/getting-started.md'),
    }))
    const diffs = {}
    for (const f of changed) diffs[f] = git.diffFor(t, f)
    findings.push(...laws.botStamped(changed, diffs))
    if (changed.includes('Cargo.toml')) findings.push(...laws.newDeps(baseCargo, headCargo))
  }

  findings.push(...laws.testPlacement(added, f => git.readAt(t, f)))
  const mentioned = git.mentionedIssues(t)
  findings.push(...laws.unclosedTickets(mentioned, git.closingIssues(t), git.issueTitles(mentioned)))

  const errors = findings.filter(f => f.severity === 'error')
  if (asJson) {
    console.log(JSON.stringify({ target: { kind: t.kind, base: t.base, head: t.head }, changed, findings }, null, 2))
  } else {
    if (!findings.length) console.log(`No law violations. (${changed.length} changed file(s), ${t.kind})`)
    for (const f of findings) {
      console.log(`${f.severity === 'error' ? '✗' : '!'} [${f.id}] ${f.file}:${f.line} — ${f.summary}\n    ${f.detail}`)
    }
    if (findings.length) console.log(`\n${errors.length} error(s), ${findings.length - errors.length} warning(s)`)
  }
  return errors.length || (argv.includes('--strict') && findings.length) ? 1 : 0
}

function runFleet() {
  const planPath = flag('plan')
  if (typeof planPath !== 'string') { console.error('fleet: --plan <file.json> is required'); return 1 }
  const plan = JSON.parse(readFileSync(planPath, 'utf8'))
  const self = git.selfFacts()

  const peers = (plan.peers || []).map(p => ({ ...p, ...(p.dir ? git.treeFacts(p.dir) : {}), name: p.name }))
  const queued = [...new Set(peers.flatMap(p => (p.queue || []).map(String)))]
  const issues = queued.length ? git.issueFacts(queued) : {}

  const findings = [
    ...fleet.treeFindings(peers, self),
    ...fleet.issueFindings(peers, issues, { login: self.login }),
    ...fleet.overlapFindings(peers),
  ]

  if (asJson) console.log(JSON.stringify({ self, peers, issues, findings }, null, 2))
  else console.log(fleet.formatFleet(findings))
  return findings.some(f => f.severity === 'error') || (argv.includes('--strict') && findings.length) ? 1 : 0
}

const USAGE = `nidus-check — deterministic checks for this repo's laws and verification lanes

  nidus-check lanes  [--base <ref>] [--pr <n>] [--paths a,b] [--json]
      Which just recipes actually cover the change. Core \`just ci\` does not
      compile src/cli, src/server or src/bin, so this is not one blanket gate.

  nidus-check laws   [--base <ref>] [--pr <n>] [--json] [--strict]
      The CLAUDE.md rules as detectors: 3-line comment cap, unsafe, version bump,
      stale install snippets, bot-stamped files, heavy deps, test placement,
      bogus Miri ignores, feature gating, tickets left in_progress.

  nidus-check fleet  --plan <file.json> [--json] [--strict]
      Is this dispatch safe? Shared working trees, foreign remotes, dirty or stale
      peer clones, tickets that are closed/taken/already-PR'd or queued twice, and
      files two peers both claim. The plan is
      {"peers":[{"name":…,"dir":…,"queue":[…],"surface":{"<issue>":["path"]}}]}.

  nidus-check selftest
      Run the fixture suite for the detectors.

With no --base/--pr, both compare the working tree against HEAD.`

const exit = (() => {
  switch (cmd) {
    case 'lanes': return runLanes()
    case 'laws': return runLaws()
    case 'fleet': return runFleet()
    case 'selftest': return selftest({ json: asJson })
    default: console.log(USAGE); return cmd ? 1 : 0
  }
})()

process.exit(exit)
