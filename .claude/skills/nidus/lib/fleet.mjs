// Dispatch safety as detectors. Pure functions over facts the caller gathered, so
// the selftest needs no peers, no clones and no network.

const err = (id, subject, summary, detail) => ({ id, severity: 'error', subject, summary, detail })
const warn = (id, subject, summary, detail) => ({ id, severity: 'warn', subject, summary, detail })

// Two sessions in one checkout is the hazard that outranks every other: they share
// a branch, an index and a target dir, so one agent's checkout silently rewrites
// another's tree. Everything else here is recoverable.
export function treeFindings(peers, self = {}) {
  const findings = []
  const byDir = new Map()

  for (const p of peers) {
    const dir = normalize(p.dir)
    if (!dir) { findings.push(err('fleet-no-tree', p.name, 'no working directory known', 'Ask the peer for its cwd before dispatching; a peer with no tree of its own has nowhere to work.')); continue }

    const COLLIDE = 'Two sessions in one checkout share a HEAD, an index and a target/, so one agent\'s checkout silently rewrites the other\'s tree. Give each its own worktree.'
    if (!p.self && self.dir && dir === normalize(self.dir)) {
      findings.push(err('fleet-shared-tree', p.name, `works in the coordinator's own tree (${dir})`, COLLIDE))
    }
    const prior = byDir.get(dir)
    if (prior) {
      findings.push(err('fleet-shared-tree', `${prior} + ${p.name}`, `share the working tree ${dir}`, COLLIDE))
    } else byDir.set(dir, p.name)

    if (p.isRepo === false) {
      findings.push(err('fleet-not-a-repo', p.name, `${dir} is not a git repository`, 'Clone the repo there before dispatching.'))
      continue
    }
    if (self.remote && p.remote && !sameRemote(p.remote, self.remote)) {
      findings.push(err('fleet-foreign-remote', p.name, `origin is ${p.remote}, not ${self.remote}`, 'The peer would push to a different repository. Point it at the same remote.'))
    }
    // A separate clone works, it is just the expensive way to buy isolation a
    // worktree gives for the cost of a checkout against the same object store.
    if (!p.self && self.commonDir && p.commonDir && p.commonDir !== self.commonDir && sameRemote(p.remote || '', self.remote || '')) {
      findings.push(warn('fleet-separate-clone', p.name, `${dir} is its own clone, not a worktree`, `Prefer \`git worktree add ${normalize(self.dir)}/.claude/worktrees/<slug> -b <branch> origin/main\` and have the peer EnterWorktree into it. One clone, one object store, still fully isolated.`))
    }
    if (p.branch === 'main') {
      findings.push(warn('fleet-on-main', p.name, 'sits on main', 'The peer must branch before working; the skill forbids committing to main.'))
    }
    if (p.dirty) {
      findings.push(warn('fleet-dirty-tree', p.name, `${dir} has uncommitted changes`, 'The peer will branch on top of them. Have it stash or commit first so the diff under review is only the ticket.'))
    }
    if (self.mainSha && p.mainSha && p.mainSha !== self.mainSha) {
      findings.push(warn('fleet-stale-main', p.name, 'origin/main differs from the coordinator', 'Tell the peer to fetch before branching, or its PR opens against a stale base.'))
    }
  }
  return findings
}

// A queue is only dispatchable if every ticket in it is real, open, unclaimed and
// not already shipped. Dispatching a closed or already-PR'd issue is the tracker
// asserting something the tree does not support, the failure CLAUDE.md names.
export function issueFindings(peers, issues = {}, opts = {}) {
  const findings = []
  const claimedBy = new Map()
  const me = opts.login || null

  for (const p of peers) {
    for (const n of p.queue || []) {
      const key = String(n)
      const prior = claimedBy.get(key)
      if (prior && prior !== p.name) {
        findings.push(err('fleet-double-assigned', `#${key}`, `queued for both ${prior} and ${p.name}`, 'Two peers would open competing PRs for one ticket. Assign it once.'))
      } else claimedBy.set(key, p.name)

      const meta = issues[key]
      if (!meta) {
        findings.push(err('fleet-issue-missing', `#${key}`, 'no such issue', 'gh could not resolve it. Do not dispatch a ticket whose contents you would have to invent.'))
        continue
      }
      if (meta.state && meta.state !== 'OPEN') {
        findings.push(err('fleet-issue-closed', `#${key}`, `is ${String(meta.state).toLowerCase()}`, 'Already done or rejected. Re-read it before dispatching; do not reopen work the tree already has.'))
      }
      const others = (meta.assignees || []).filter(a => a !== me && a !== p.login)
      if (others.length) {
        findings.push(warn('fleet-issue-taken', `#${key}`, `already assigned to ${others.join(', ')}`, 'Someone else may be on it. Confirm before handing it to this peer.'))
      }
      const openPr = (meta.linkedPrs || []).find(pr => pr.state === 'OPEN')
      if (openPr) {
        findings.push(warn('fleet-issue-has-pr', `#${key}`, `PR #${openPr.number} is already open for it`, 'The work may be in flight. Review that PR instead of starting a second one.'))
      }
    }
  }
  return findings
}

// Tickets that share a file will be individually green and jointly broken (#149).
// Only the peers can name their real surface, so this flags the pairs to ask about.
export function overlapFindings(peers) {
  const findings = []
  const byFile = new Map()
  for (const p of peers) {
    for (const [n, files] of Object.entries(p.surface || {})) {
      for (const f of files) {
        if (!byFile.has(f)) byFile.set(f, [])
        byFile.get(f).push({ peer: p.name, issue: n })
      }
    }
  }
  for (const [file, holders] of byFile) {
    const peers_ = new Set(holders.map(h => h.peer))
    if (peers_.size < 2) continue
    const who = holders.map(h => `#${h.issue} (${h.peer})`).join(', ')
    findings.push(warn('fleet-file-overlap', file, `claimed by ${who}`, 'Sequence these rather than running them concurrently: land one, then rebase the other onto its final shape.'))
  }
  return findings
}

// Implementation agents get a worktree each, siblings of the peer's in the one
// registry. Git stops them colliding; nothing stops them accumulating.
export function orphanFindings(list = [], peers = [], self = {}) {
  const claimed = new Set([normalize(self.dir), ...peers.map(p => normalize(p.dir))].filter(Boolean))
  const findings = []

  for (const w of list) {
    if (w.isMain || claimed.has(normalize(w.path))) continue
    const agent = /^worktree-agent-/.test(w.branch || '') || /\/agent-[0-9a-f]+$/.test(w.path)
    const stuck = w.dirty || w.hasCommits
    const how = stuck
      ? `\`git worktree remove --force ${w.path}\` then \`git branch -D ${w.branch || '<branch>'}\` — it carries ${w.hasCommits ? 'commits' : 'uncommitted edits'}, so \`prune\` will not reclaim it.`
      : `\`git worktree remove ${w.path}\`, or \`git worktree prune\` once the directory is gone.`
    findings.push(warn(
      agent ? 'fleet-orphan-agent-worktree' : 'fleet-unaccounted-worktree',
      w.path,
      agent ? 'implementation-agent worktree left behind' : `belongs to no peer in this plan (branch ${w.branch || 'detached'})`,
      how,
    ))
  }
  return findings
}

export function formatFleet(findings) {
  if (!findings.length) return 'Dispatch is clear. No tree, ticket or overlap findings.'
  const errors = findings.filter(f => f.severity === 'error')
  const lines = findings.map(f => `${f.severity === 'error' ? '✗' : '!'} [${f.id}] ${f.subject} — ${f.summary}\n    ${f.detail}`)
  lines.push(`\n${errors.length} error(s), ${findings.length - errors.length} warning(s)`)
  return lines.join('\n')
}

const normalize = d => (d ? String(d).replace(/\/+$/, '') : null)

// git@host:o/r.git, https://host/o/r.git and https://host/o/r are one remote.
function sameRemote(a, b) {
  const canon = u => String(u).trim().replace(/\.git$/, '').replace(/^git@([^:]+):/, 'https://$1/').replace(/\/+$/, '').toLowerCase()
  return canon(a) === canon(b)
}
