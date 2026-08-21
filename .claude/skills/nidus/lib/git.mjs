// Git/gh/bd IO for the checker. Kept apart from lanes.mjs and laws.mjs so those stay
// pure functions over text and the selftest never needs a repository.

import { execSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'

// Beads kept the GitHub numbers, so `#186` and `nidus-186` are the same ticket and the
// checker still canonicalises refs as `#<n>`.
export const BEAD_PREFIX = 'nidus'

// Only `closed` is done. `in_progress`/`blocked` are open work, and `deferred` reports
// as itself so a dispatch against one says "is deferred" rather than "is closed".
const beadState = s => (s === 'closed' ? 'CLOSED' : s === 'deferred' ? 'DEFERRED' : 'OPEN')

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
    const meta = JSON.parse(sh(`gh pr view ${pr} --json baseRefName,headRefName,headRefOid,number,title,body,state,isDraft,url`))
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

// Where a ref *currently* is, independent of this branch's merge base — which is the only
// way to see a version that has moved on origin/main since the branch was cut (#173).
export function readAtRef(ref, path) {
  const out = sh(`git show ${ref}:${path}`, { allowFail: true })
  return out === '' ? null : out
}

export function diffFor(t, path) {
  if (t.kind === 'path') return ''
  const range = t.head ? `${t.base}...${t.head}` : t.base
  return sh(`git diff ${range} -- ${path}`, { allowFail: true })
}

// Issue ids this change declares itself to BE — branch name, commit subjects, PR title.
// Deliberately not the bodies: a `#n` in prose is context, and cross-referencing related
// issues is good practice, so scraping bodies made the law fire on exactly that.
// A branch is austin/<n>-<slug>, so its leading number is an issue ref too.
export function mentionedIssues(t) {
  // The PR's own head ref, not the local checkout: targeting a PR from an unrelated
  // branch used to attribute that branch's issue to it, and CI checks out a detached
  // merge ref where `--show-current` is empty, losing the signal altogether.
  const branch = t.pr ? (t.pr.headRefName || '') : sh('git branch --show-current', { allowFail: true })
  let text = branch.replace(/(?:^|\/)(\d+)-/g, ' #$1 ')
  if (t.base) {
    const range = t.head ? `${t.base}..${t.head}` : `${t.base}..HEAD`
    text += '\n' + sh(`git log --format=%s ${range}`, { allowFail: true })
  }
  if (t.pr) text += `\n${t.pr.title || ''}`
  const refs = Array.from(text.matchAll(/(?:#|nidus-)(\d+)/g), m => `#${m[1]}`)
  return new Set(refs)
}

// Refs/Part of/See: the author has stated this issue's disposition without claiming to
// close it. Read from the bodies, where such a trailer is actually written.
const ACK_RE = /\b(?:refs?|part of|see)\s+(?:#|nidus-)(\d+)/gi

export function acknowledgedIssues(t) {
  let text = t.pr ? t.pr.body || '' : ''
  if (t.base) {
    const range = t.head ? `${t.base}..${t.head}` : `${t.base}..HEAD`
    text += '\n' + sh(`git log --format=%b ${range}`, { allowFail: true })
  }
  return new Set(Array.from(text.matchAll(ACK_RE), m => `#${m[1]}`))
}

// A closing keyword now only states intent — GitHub cannot close a bead, so these are
// the claims `unclosedTickets` holds the author to at `bd close` time.
export function closingIssues(t) {
  let text = t.pr ? t.pr.body || '' : ''
  if (t.base) {
    const range = t.head ? `${t.base}..${t.head}` : `${t.base}..HEAD`
    text += '\n' + sh(`git log --format=%b ${range}`, { allowFail: true })
  }
  const re = /\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+(?:#|nidus-)(\d+)/gi
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

// `prune` only reclaims worktrees whose directory is gone. One carrying commits or
// edits survives it, so report both — they decide whether removal needs --force.
export function worktrees(dir = process.cwd()) {
  const out = inDir(dir, 'worktree list --porcelain')
  const main = inDir(dir, 'rev-parse --path-format=absolute --git-common-dir').replace(/\/\.git\/?$/, '')
  const list = []
  for (const block of out.split('\n\n')) {
    const path = block.match(/^worktree (.+)$/m)?.[1]
    if (!path) continue
    const isMain = path.replace(/\/+$/, '') === main.replace(/\/+$/, '')
    list.push({
      path,
      branch: block.match(/^branch refs\/heads\/(.+)$/m)?.[1] || null,
      detached: /^detached$/m.test(block),
      isMain,
      dirty: isMain ? false : inDir(path, 'status --porcelain --untracked-files=no') !== '',
      hasCommits: isMain ? false : inDir(path, 'rev-list --count origin/main..HEAD') !== '0',
    })
  }
  return list
}

export function selfFacts() {
  return {
    ...treeFacts(process.cwd()),
    login: JSON.parse(sh('gh api user --jq "{login:.login}"', { allowFail: true }) || '{}').login || null,
  }
}

// Every name that means "me". beads records a display name and GitHub a login, so a
// single identity would report a ticket assigned to its own holder as taken.
export function identities() {
  const gh = JSON.parse(sh('gh api user --jq "{login:.login}"', { allowFail: true }) || '{}').login || null
  const name = sh('git config user.name', { allowFail: true }).trim() || null
  const email = sh('git config user.email', { allowFail: true }).trim() || null
  return [gh, name, email].filter(Boolean)
}

// Issues live in beads; PRs still live on GitHub. `bd show --json` returns an array.
const bead = n => {
  try {
    const raw = sh(`bd show ${BEAD_PREFIX}-${n} --json`, { allowFail: true })
    const j = JSON.parse(raw)
    return Array.isArray(j) ? (j[0] || null) : (j && j.id ? j : null)
  } catch { return null }
}

// Nothing links a bead to a PR, so derive it the way GitHub used to: a PR whose title
// or body carries a closing keyword for that ticket, in either `#186` or `nidus-186` form.
export function issueFacts(numbers) {
  const out = {}
  // --state all, because a merged PR is how a cleared coordinator learns a ticket
  // already shipped. The open-PR detector filters for OPEN itself.
  const raw = sh('gh pr list --state all --limit 200 --json number,title,body,state', { allowFail: true })
  let prs = []
  try { prs = JSON.parse(raw || '[]') } catch { prs = [] }

  for (const n of numbers) {
    const j = bead(n)
    if (!j) continue
    const re = new RegExp(`\\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\\s+(?:#${n}|${BEAD_PREFIX}-${n})\\b`, 'i')
    out[String(n)] = {
      number: /^\d+$/.test(String(n)) ? Number(n) : String(n),
      state: beadState(j.status),
      status: j.status,
      assignees: j.assignee ? [j.assignee] : [],
      linkedPrs: prs.filter(p => re.test(`${p.title}\n${p.body || ''}`)).map(p => ({ number: p.number, state: p.state })),
    }
  }
  return out
}

export function issueTitles(refs) {
  const out = {}
  for (const r of refs) {
    const j = bead(r.slice(1))
    if (j && beadState(j.status) === 'OPEN') out[r] = j.title
  }
  return out
}

// Branches that are ahead of main, with the version each one claims. Two claiming
// the same version is invisible from inside either tree.
export function inflightVersions(dir = process.cwd()) {
  const refs = inDir(dir, "for-each-ref --format='%(refname:short)' refs/remotes/origin").split('\n').filter(Boolean)
  const out = []
  for (const ref of refs) {
    if (/\/(HEAD|main)$/.test(ref)) continue
    if (inDir(dir, `merge-base --is-ancestor ${ref} origin/main && echo merged`) === 'merged') continue
    const cargo = inDir(dir, `show ${ref}:Cargo.toml`)
    const v = cargo.match(/^version\s*=\s*"([^"]+)"/m)
    if (!v) continue
    const changed = inDir(dir, `diff --name-only origin/main...${ref}`)
    out.push({ ref, version: v[1], changed: changed ? changed.split('\n') : [] })
  }
  return out
}

export function releasedTags(dir = process.cwd()) {
  return new Set(inDir(dir, "tag -l v*").split('\n').filter(Boolean))
}

export function openPrRefs() {
  const raw = sh('gh pr list --state open --limit 200 --json headRefName', { allowFail: true })
  try { return new Set(JSON.parse(raw || '[]').map(p => p.headRefName)) } catch { return new Set() }
}

// ── preflight IO ───────────────────────────────────────────────────────────

// The whole point of preflight: judge against origin as it is now, not as this clone
// last saw it. Returns false when the fetch failed, so the caller can say so.
export function fetchOrigin(dir = process.cwd()) {
  try {
    sh(`git -C ${JSON.stringify(dir)} fetch --quiet --prune origin`)
    return true
  } catch { return false }
}

// How many commits origin/main has that HEAD does not. Non-zero means work landed that
// this tree cannot see, so a dependency here can read as unmerged when it has shipped.
export function behindMain(dir = process.cwd()) {
  const n = inDir(dir, 'rev-list --count HEAD..origin/main')
  return /^\d+$/.test(n) ? Number(n) : 0
}

// Is `ref` a local branch behind its own remote counterpart? The nidus-qko case: a
// `--base main` over a stale local main examines a range nobody meant.
export function refDrift(ref, dir = process.cwd()) {
  if (!ref || /^origin\//.test(ref)) return { ref, hasRemote: false, behind: 0, ahead: 0 }
  const short = ref.replace(/^refs\/heads\//, '')
  const hasRemote = inDir(dir, `rev-parse --verify --quiet origin/${short}`) !== ''
  if (!hasRemote) return { ref: short, hasRemote: false, behind: 0, ahead: 0 }
  const count = spec => {
    const n = inDir(dir, `rev-list --count ${spec}`)
    return /^\d+$/.test(n) ? Number(n) : 0
  }
  // Ahead matters as much as behind (nidus-1jb): unpushed commits on a shared branch are
  // work stranded on one machine, and they widen a `--base <ref>` range nobody meant.
  const local = inDir(dir, `rev-parse --verify --quiet ${short}`) !== ''
  return {
    ref: short,
    hasRemote: true,
    behind: count(`${short}..origin/${short}`),
    ahead: local ? count(`origin/${short}..${short}`) : 0,
  }
}

// Remote branches whose name carries this issue's number, so a ticket someone else
// already started does not get picked up twice.
export function branchesForIssue(number, dir = process.cwd()) {
  const refs = inDir(dir, "for-each-ref --format='%(refname:short)' refs/remotes/origin").split('\n').filter(Boolean)
  const re = new RegExp(`(^|[^0-9a-z])${String(number).replace(/\./g, '\\.')}([^0-9a-z]|$)`, 'i')
  return refs.filter(r => !/\/(HEAD|main)$/.test(r) && re.test(r.replace(/^origin\//, '')))
}
