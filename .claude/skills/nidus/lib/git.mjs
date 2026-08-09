// Git/gh IO for the checker. Kept apart from lanes.mjs and laws.mjs so those stay
// pure functions over text and the selftest never needs a repository.

import { execSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'

export function sh(cmd, { allowFail = false } = {}) {
  try {
    return execSync(cmd, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 64 * 1024 * 1024 })
  } catch (e) {
    if (allowFail) return ''
    throw new Error(`command failed: ${cmd}\n${e.stderr || e.message}`)
  }
}

// A target is what we compare: a base commit, a head (a ref or the working tree),
// and how to read a file's full text at that head.
export function resolveTarget({ base, head, pr, path } = {}) {
  if (pr) {
    const meta = JSON.parse(sh(`gh pr view ${pr} --json baseRefName,headRefOid,number,title,body,state,isDraft,url`))
    sh(`git fetch --quiet origin pull/${pr}/head`)
    const headOid = meta.headRefOid
    sh(`git fetch --quiet origin ${meta.baseRefName}`, { allowFail: true })
    const baseRef = sh(`git merge-base origin/${meta.baseRefName} ${headOid}`, { allowFail: true }).trim() || `origin/${meta.baseRefName}`
    return { kind: 'pr', base: baseRef, head: headOid, pr: meta }
  }
  if (head || base) {
    const b = base || 'main'
    const h = head || 'HEAD'
    const mb = sh(`git merge-base ${b} ${h}`, { allowFail: true }).trim() || b
    return { kind: 'range', base: mb, head: h }
  }
  if (path) return { kind: 'path', base: null, head: null, path }
  return { kind: 'worktree', base: 'HEAD', head: null }
}

export function changedFiles(t) {
  if (t.kind === 'path') {
    const listed = sh(`git ls-files ${t.path}`, { allowFail: true }).trim()
    return listed ? listed.split('\n') : []
  }
  const range = t.head ? `${t.base}...${t.head}` : t.base
  const tracked = sh(`git diff --name-only --diff-filter=ACMR ${range}`, { allowFail: true }).trim()
  const files = tracked ? tracked.split('\n') : []
  if (!t.head) {
    const untracked = sh('git ls-files --others --exclude-standard', { allowFail: true }).trim()
    if (untracked) files.push(...untracked.split('\n'))
  }
  return [...new Set(files)]
}

export function addedFiles(t) {
  if (t.kind === 'path') return []
  const range = t.head ? `${t.base}...${t.head}` : t.base
  const out = sh(`git diff --name-only --diff-filter=A ${range}`, { allowFail: true }).trim()
  const files = out ? out.split('\n') : []
  if (!t.head) {
    const untracked = sh('git ls-files --others --exclude-standard', { allowFail: true }).trim()
    if (untracked) files.push(...untracked.split('\n'))
  }
  return [...new Set(files)]
}

// Added-line numbers per file, from a zero-context diff. Used to scope findings to
// what this change actually touched instead of every pre-existing violation.
export function addedLineMap(t) {
  const map = new Map()
  if (t.kind === 'path') return map
  const range = t.head ? `${t.base}...${t.head}` : t.base
  const diff = sh(`git diff -U0 ${range}`, { allowFail: true })
  let file = null
  for (const line of diff.split('\n')) {
    const f = line.match(/^\+\+\+ b\/(.+)$/)
    if (f) { file = f[1]; if (!map.has(file)) map.set(file, new Set()); continue }
    const h = line.match(/^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@/)
    if (h && file) {
      const start = Number(h[1])
      const count = h[2] === undefined ? 1 : Number(h[2])
      for (let i = 0; i < count; i++) map.get(file).add(start + i)
    }
  }
  return map
}

export function readAt(t, path) {
  if (t.head) {
    const out = sh(`git show ${t.head}:${path}`, { allowFail: true })
    return out === '' ? null : out
  }
  return existsSync(path) ? readFileSync(path, 'utf8') : null
}

export function readBase(t, path) {
  if (!t.base) return null
  const out = sh(`git show ${t.base}:${path}`, { allowFail: true })
  return out === '' ? null : out
}

export function diffFor(t, path) {
  if (t.kind === 'path') return ''
  const range = t.head ? `${t.base}...${t.head}` : t.base
  return sh(`git diff ${range} -- ${path}`, { allowFail: true })
}

// Issue ids this change claims to ship — from the branch name, its commit subjects,
// and the PR title/body. Lets the ticket check tell "close this before merge" apart
// from "unrelated backlog rot".
// A branch is austin/<n>-<slug>, so its leading number is an issue ref too.
export function mentionedIssues(t) {
  const branch = sh('git branch --show-current', { allowFail: true })
  let text = branch.replace(/(?:^|\/)(\d+)-/g, ' #$1 ')
  if (t.base) {
    const range = t.head ? `${t.base}..${t.head}` : `${t.base}..HEAD`
    text += '\n' + sh(`git log --format=%s%n%b ${range}`, { allowFail: true })
  }
  if (t.pr) text += `\n${t.pr.title || ''}\n${t.pr.body || ''}`
  return new Set(text.match(/#\d+/g) || [])
}

// GitHub's own closing keywords; anything else is a mention that will NOT close.
export function closingIssues(t) {
  let text = t.pr ? t.pr.body || '' : ''
  if (t.base) {
    const range = t.head ? `${t.base}..${t.head}` : `${t.base}..HEAD`
    text += '\n' + sh(`git log --format=%b ${range}`, { allowFail: true })
  }
  const re = /\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)/gi
  return new Set(Array.from(text.matchAll(re), m => `#${m[1]}`))
}

// ── fleet IO ───────────────────────────────────────────────────────────────

const inDir = (dir, cmd) => sh(`git -C ${JSON.stringify(dir)} ${cmd}`, { allowFail: true }).trim()

export function treeFacts(dir) {
  if (!existsSync(dir)) return { dir, isRepo: false }
  const top = inDir(dir, 'rev-parse --show-toplevel')
  if (!top) return { dir, isRepo: false }
  // commonDir is the shared object store: equal across worktrees of one clone,
  // distinct across separate clones. That is what tells the two layouts apart.
  return {
    dir: top,
    isRepo: true,
    commonDir: inDir(dir, 'rev-parse --path-format=absolute --git-common-dir') || null,
    isWorktree: inDir(dir, 'rev-parse --is-inside-work-tree') === 'true' && inDir(dir, 'rev-parse --git-dir') !== inDir(dir, 'rev-parse --git-common-dir'),
    remote: inDir(dir, 'remote get-url origin') || null,
    branch: inDir(dir, 'branch --show-current') || null,
    dirty: inDir(dir, 'status --porcelain --untracked-files=no') !== '',
    mainSha: inDir(dir, 'rev-parse origin/main') || null,
  }
}

export function worktrees(dir = process.cwd()) {
  const out = inDir(dir, 'worktree list --porcelain')
  const list = []
  for (const block of out.split('\n\n')) {
    const path = block.match(/^worktree (.+)$/m)?.[1]
    if (!path) continue
    list.push({ path, branch: block.match(/^branch refs\/heads\/(.+)$/m)?.[1] || null, detached: /^detached$/m.test(block) })
  }
  return list
}

export function selfFacts() {
  return {
    ...treeFacts(process.cwd()),
    login: JSON.parse(sh('gh api user --jq "{login:.login}"', { allowFail: true }) || '{}').login || null,
  }
}

// gh has no linked-PR field on an issue, so derive it the way GitHub does: an open
// PR whose title or body carries a closing keyword for that number.
export function issueFacts(numbers) {
  const out = {}
  const raw = sh('gh pr list --state open --limit 200 --json number,title,body,state', { allowFail: true })
  let prs = []
  try { prs = JSON.parse(raw || '[]') } catch { prs = [] }

  for (const n of numbers) {
    const meta = sh(`gh issue view ${n} --json number,state,assignees`, { allowFail: true })
    if (!meta) continue
    try {
      const j = JSON.parse(meta)
      const re = new RegExp(`\\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\\s+#${n}\\b`, 'i')
      out[String(n)] = {
        number: j.number,
        state: j.state,
        assignees: (j.assignees || []).map(a => a.login),
        linkedPrs: prs.filter(p => re.test(`${p.title}\n${p.body || ''}`)).map(p => ({ number: p.number, state: p.state })),
      }
    } catch { /* unparseable — the missing-issue detector reports it */ }
  }
  return out
}

export function issueTitles(refs) {
  const out = {}
  for (const r of refs) {
    const raw = sh(`gh issue view ${r.slice(1)} --json title,state`, { allowFail: true })
    try {
      const j = JSON.parse(raw)
      if (j.state === 'OPEN') out[r] = j.title
    } catch { /* closed, missing, or gh unavailable — not a finding */ }
  }
  return out
}
