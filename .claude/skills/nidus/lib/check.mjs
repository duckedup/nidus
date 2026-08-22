// Entry point for bin/nidus-check. Wires the pure detectors in laws.mjs and the
// lane map in lanes.mjs to real git/gh IO.

import { lanes, formatLanes } from './lanes.mjs'
import * as laws from './laws.mjs'
import * as fleet from './fleet.mjs'
import * as pre from './preflight.mjs'
import * as git from './git.mjs'
import { selftest } from './selftest.mjs'
import { readFileSync, existsSync } from 'node:fs'

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
  // nidus-qko: a --base over a stale local ref examines a range nobody meant and reports
  // it as thoroughness. The two file counts are the tell, so compute both.
  const drift = t.kind === 'range' ? git.refDrift(flag('base') === true ? null : flag('base') || 'main') : { behind: 0 }
  const driftFindings = drift.behind > 0
    ? laws.staleBase({ ...drift, examined: changed.length, examinedFresh: git.changedFiles(git.resolveTarget({ base: `origin/${drift.ref}`, head: flag('head') === true ? null : flag('head') })).length })
    : []
  const added = git.addedFiles(t)
  const addedLines = git.addedLineMap(t)
  const findings = []
  findings.push(...driftFindings)
  findings.push(...laws.emptyScope(changed, t.kind))

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
    // Against where origin/main is NOW, not the merge base — the backwards case is
    // invisible from inside the branch, which is why no existing law catches it.
    const originCargo = git.readAtRef('origin/main', 'Cargo.toml')
    if (originCargo) findings.push(...laws.versionBackwards(baseCargo, headCargo, originCargo, changed))
    // nidus-zin: the case above cannot see, since a tagged version can still be ahead
    // of main. Degrades to silence when the tag list is empty (fresh or offline clone).
    findings.push(...laws.versionAlreadyTagged(baseCargo, headCargo, git.releasedTags(), changed))
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
  findings.push(...laws.unclosedTickets(
    mentioned, git.closingIssues(t), git.issueTitles(mentioned), git.acknowledgedIssues(t),
  ))

  const errors = findings.filter(f => f.severity === 'error')
  if (asJson) {
    console.log(JSON.stringify({ target: { kind: t.kind, base: t.base, head: t.head }, changed, findings }, null, 2))
  } else {
    // Printed unconditionally: when this was tied to the no-findings branch, any finding
    // hid what had been examined, so a run over nothing read exactly like a clean one.
    const stale = driftFindings.length ? ` — WARNING: base ${drift.ref} is ${drift.behind} commit(s) behind origin/${drift.ref}` : ''
    console.log(`Examined ${changed.length} changed file(s) (${t.kind})${stale}.`)
    if (!findings.length) console.log('No law violations.')
    for (const f of findings) {
      console.log(`${f.severity === 'error' ? '✗' : '!'} [${f.id}] ${f.file}:${f.line} — ${f.summary}\n    ${f.detail}`)
    }
    if (findings.length) console.log(`\n${errors.length} error(s), ${findings.length - errors.length} warning(s)`)
  }
  return errors.length || (argv.includes('--strict') && findings.length) ? 1 : 0
}

// The plan is the coordinator's only un-derivable state, so it lives on disk rather
// than in a context that cannot clear itself.
const PLAN = '.claude/fleet-plan.json'

const mainVersion_ = () => (git.sh('git show origin/main:Cargo.toml', { allowFail: true }).match(/^version\s*=\s*"([^"]+)"/m) || [])[1] || null

function runFleet() {
  const explicit = flag('plan')
  const planPath = typeof explicit === 'string' ? explicit : PLAN
  if (!existsSync(planPath)) {
    console.error(`fleet: no plan at ${planPath}. Write one, or pass --plan <file.json>.`)
    return 1
  }
  const plan = JSON.parse(readFileSync(planPath, 'utf8'))
  const self = git.selfFacts()

  const peers = (plan.peers || []).map(p => ({ ...p, ...(p.dir ? git.treeFacts(p.dir) : {}), name: p.name }))
  const queued = [...new Set(peers.flatMap(p => (p.queue || []).map(String)))]
  const issues = queued.length ? git.issueFacts(queued) : {}
  const trees = git.worktrees()

  const findings = [
    ...fleet.treeFindings(peers, self),
    ...fleet.issueFindings(peers, issues, { login: self.login }),
    ...fleet.overlapFindings(peers),
    ...fleet.orphanFindings(trees, peers, self),
    ...fleet.versionFindings(git.inflightVersions(), mainVersion_(), git.releasedTags(), git.openPrRefs(), laws.BEHAVIOURAL),
  ]
  const state = fleet.rehydrate(peers, issues, trees, git.inflightVersions().map(b => b.ref))

  if (asJson) console.log(JSON.stringify({ plan: planPath, self, peers, issues, state, findings }, null, 2))
  else if (argv.includes('--status')) console.log(fleet.formatRehydrate(state))
  else console.log(`${fleet.formatRehydrate(state)}\n\n${fleet.formatFleet(findings)}`)
  return findings.some(f => f.severity === 'error') || (argv.includes('--strict') && findings.length) ? 1 : 0
}


function runPreflight() {
  const noFetch = argv.includes('--no-fetch')
  const fetched = noFetch ? false : git.fetchOrigin()
  const self = git.treeFacts(process.cwd())
  const behind = git.behindMain()
  const mainAhead = git.refDrift('main').ahead
  const raw = flag('issue')
  const id = typeof raw === 'string' ? raw.replace(new RegExp(`^(?:#|${git.BEAD_PREFIX}-)`), '') : null

  let issue = null
  let issueBranches = []
  if (id) {
    const facts = git.issueFacts([id])
    issue = facts[String(id)] || { number: id, unknown: true }
    issueBranches = git.branchesForIssue(id)
  }

  const mainVersion = mainVersion_()
  const claimed = git.inflightVersions()
  const released = git.releasedTags()
  const nextVersion = pre.nextFreeVersion(mainVersion, claimed, released)

  const findings = pre.preflight({
    fetched, branch: self.branch, onMain: self.branch === 'main',
    dirty: self.dirty, behind, mainAhead, issue, issueBranches,
    me: git.identities(),
  })
  const info = { branch: self.branch, behind, mainAhead, mainVersion, nextVersion, fetched }

  if (asJson) console.log(JSON.stringify({ info, claimed, issue, issueBranches, findings }, null, 2))
  else console.log(pre.formatPreflight(findings, info))
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

  nidus-check fleet  [--plan <file.json>] [--status] [--json] [--strict]
      Is this dispatch safe? Shared working trees, foreign remotes, dirty or stale
      peer clones, tickets that are closed/taken/already-PR'd or queued twice,
      files two peers both claim, worktrees left behind by finished agents, and
      two in-flight branches claiming one Cargo.toml version.
      Defaults to .claude/fleet-plan.json, which is the coordinator's durable
      state: everything else is derived, so a cleared session rehydrates with one
      run. --status prints just that derived state. The plan is
      {"peers":[{"name":…,"dir":…,"self":true?,"queue":[…],"surface":{"<n>":["path"]}}]}.

  nidus-check preflight [--issue <id>] [--no-fetch] [--json] [--strict]
      Run this FIRST, before evaluating anything. Fetches origin, then reports
      whether this tree is fit to reason from: behind origin/main, on main,
      dirty, and — with --issue — whether the ticket is already closed, already
      carried by a merged or open PR, assigned to someone else, or already has a
      remote branch. Also prints the next free Cargo.toml version. Errors block:
      a stale base makes every judgement below it, the ticket's state included,
      a statement about a main that has since moved.

  nidus-check selftest
      Run the fixture suite for the detectors.

With no --base/--pr, both compare the working tree against HEAD.`

const exit = (() => {
  switch (cmd) {
    case 'lanes': return runLanes()
    case 'laws': return runLaws()
    case 'fleet': return runFleet()
    case 'preflight': return runPreflight()
    case 'selftest': return selftest({ json: asJson })
    default: console.log(USAGE); return cmd ? 1 : 0
  }
})()

process.exit(exit)
