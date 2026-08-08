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
    const meta = JSON.parse(sh(`gh pr view ${pr} --json baseRefName,headRefOid,number,title,state,isDraft,url`))
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
export function mentionedIssues(t) {
  let text = sh('git branch --show-current', { allowFail: true })
  if (t.base) {
    const range = t.head ? `${t.base}..${t.head}` : `${t.base}..HEAD`
    text += '\n' + sh(`git log --format=%s%n%b ${range}`, { allowFail: true })
  }
  if (t.pr) text += `\n${t.pr.title || ''}`
  return new Set(text.match(/\bnidus-[a-z0-9]+(?:\.\d+)?\b/gi) || [])
}

export function inProgressIssues() {
  const raw = sh('bd list --status in_progress --json', { allowFail: true })
  try {
    const parsed = JSON.parse(raw)
    const list = Array.isArray(parsed) ? parsed : parsed.issues || []
    return list.map(i => ({ id: i.id, title: i.title, type: i.issue_type }))
  } catch {
    return []
  }
}
