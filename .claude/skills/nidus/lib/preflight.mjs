// Is this tree fit to reason from? Pure detectors over a facts object, so the selftest
// drives them with no repository, no git and no network. The IO lives in git.mjs.

const err = (id, subject, summary, detail) => ({ id, severity: 'error', subject, summary, detail })
const warn = (id, subject, summary, detail) => ({ id, severity: 'warn', subject, summary, detail })

const cmp = (a, b) => {
  const pa = String(a).split('.').map(Number)
  const pb = String(b).split('.').map(Number)
  for (let i = 0; i < 3; i++) if ((pa[i] || 0) !== (pb[i] || 0)) return (pa[i] || 0) - (pb[i] || 0)
  return 0
}

// The lowest minor above every version already spoken for: main's, each in-flight
// branch's, and every released tag. Gaps are harmless, so this never reuses one.
export function nextFreeVersion(mainVersion, claimed = [], released = new Set()) {
  if (!mainVersion) return null
  const taken = new Set([...claimed.map(c => c.version)])
  let best = mainVersion
  for (const v of taken) if (cmp(v, best) > 0) best = v
  const [maj, min] = String(best).split('.').map(Number)
  for (let m = (min || 0) + 1; m < (min || 0) + 100; m++) {
    const cand = `${maj || 0}.${m}.0`
    if (!taken.has(cand) && !released.has(`v${cand}`)) return cand
  }
  return null
}

// Every reason to stop before evaluating. Ordered most-fundamental first: a stale or
// un-fetched base invalidates every judgement below it, including the ticket's own state.
export function preflight(facts = {}) {
  const {
    fetched, branch, onMain, dirty, behind = 0,
    issue, issueBranches = [], me = [],
  } = facts
  const findings = []
  const here = branch || 'HEAD'

  if (!fetched) {
    findings.push(err('preflight-no-fetch', here,
      'origin was not fetched, so every judgement below is against a stale main',
      'A ticket reads as unshipped, a dependency as unmerged, and a version as free purely because this clone has not looked. Drop --no-fetch.'))
  }
  if (onMain) {
    findings.push(err('preflight-on-main', here, 'this is main', 'Branch first. Work committed here cannot be reviewed and cannot be reverted cleanly.'))
  } else if (behind > 0) {
    findings.push(err('preflight-stale-base', here,
      `${behind} commit(s) behind origin/main`,
      `Whatever landed in those ${behind} commits is invisible from this tree, so a dependency can read as unmerged and a conflict cannot be seen. Rebase onto origin/main before evaluating.`))
  }
  if (dirty) {
    findings.push(warn('preflight-dirty', here, 'uncommitted changes present',
      'Decide before touching them: stash and branch, commit here first, or stop. Do not fold someone else\'s work-in-progress into this ticket.'))
  }

  findings.push(...ticketFindings(issue, issueBranches, me, branch))
  return findings
}

// A ticket already finished, already claimed, or already carried by a PR. Split out
// because the tree checks above are meaningful with no ticket at all.
function ticketFindings(issue, issueBranches, me, branch) {
  const findings = []
  if (issue === null || issue === undefined) return findings
  const subject = `nidus-${issue.number}`

  if (issue.unknown) {
    findings.push(warn('preflight-ticket-unknown', subject, 'bd could not resolve it',
      'Ask whether to proceed from the description alone. Do not invent the issue\'s contents.'))
    return findings
  }
  if (issue.state === 'CLOSED') {
    findings.push(err('preflight-ticket-closed', subject, 'already closed',
      'Re-deriving a decided ticket is the cost this check exists to avoid. Read the close reason first.'))
  }
  if (issue.state === 'DEFERRED') {
    findings.push(warn('preflight-ticket-deferred', subject, 'deferred, not queued',
      'It was parked on a trigger. Confirm the trigger fired rather than restarting it by accident.'))
  }
  const merged = (issue.linkedPrs || []).filter(p => p.state === 'MERGED')
  const open = (issue.linkedPrs || []).filter(p => p.state === 'OPEN')
  if (merged.length) {
    findings.push(err('preflight-ticket-shipped', subject,
      `carried by merged PR ${merged.map(p => `#${p.number}`).join(', ')}`,
      'The work is on main. Verify against the merged diff, then close the bead rather than rebuilding it.'))
  }
  if (open.length) {
    findings.push(err('preflight-ticket-in-pr', subject,
      `already claimed by open PR ${open.map(p => `#${p.number}`).join(', ')}`,
      'Two branches on one ticket is the collision this blocks. Review that PR or pick another ticket.'))
  }
  const others = (issue.assignees || []).filter(a => a && !me.includes(a))
  if (others.length) {
    findings.push(warn('preflight-ticket-taken', subject, `assigned to ${others.join(', ')}`,
      'Settle who has it before writing code, not after both of you have a diff.'))
  }
  const foreign = issueBranches.filter(b => b !== branch && b !== `origin/${branch}`)
  if (foreign.length) {
    findings.push(warn('preflight-branch-exists', subject, `a remote branch already names it: ${foreign.join(', ')}`,
      'Someone started this. Continue that branch or find out why it was abandoned.'))
  }
  return findings
}

export function formatPreflight(findings, info = {}) {
  const lines = []
  const { branch, behind, mainVersion, nextVersion, fetched } = info
  lines.push(`Branch ${branch || '(detached)'} — ${fetched ? 'origin fetched' : 'origin NOT fetched'}, ${behind || 0} commit(s) behind origin/main.`)
  if (mainVersion) lines.push(`origin/main is ${mainVersion}; next free version to claim: ${nextVersion || '(none found)'}.`)
  if (!findings.length) {
    lines.push('Clear to evaluate. Fresh base, ticket unclaimed and unshipped.')
    return lines.join('\n')
  }
  const errors = findings.filter(f => f.severity === 'error')
  for (const f of findings) lines.push(`${f.severity === 'error' ? '✗' : '!'} [${f.id}] ${f.subject} — ${f.summary}\n    ${f.detail}`)
  lines.push(`\n${errors.length} error(s), ${findings.length - errors.length} warning(s)`)
  return lines.join('\n')
}

export const PREFLIGHT_IDS = [
  'preflight-no-fetch', 'preflight-on-main', 'preflight-stale-base', 'preflight-dirty',
  'preflight-ticket-unknown', 'preflight-ticket-closed', 'preflight-ticket-deferred',
  'preflight-ticket-shipped', 'preflight-ticket-in-pr', 'preflight-ticket-taken',
  'preflight-branch-exists',
]
