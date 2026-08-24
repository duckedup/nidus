// PostToolUse gate for the two laws a script can settle at edit time: the comment cap and
// an undocumented Miri ignore. Catching them here removes a whole /nidus review round-trip.

import { readFileSync, existsSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { resolve, relative } from 'node:path'

import * as laws from '../skills/nidus/lib/laws.mjs'

const repo = resolve(new URL('../..', import.meta.url).pathname)

function stdin() {
  try { return JSON.parse(readFileSync(0, 'utf8')) } catch { return null }
}

// Only lines this edit could have introduced: a file barely touched should not be blocked by
// a violation that was already there. An untracked file has no base, so all of it is new.
function addedLines(rel) {
  let diff
  try {
    diff = execFileSync('git', ['diff', '-U0', '--', rel], { cwd: repo, encoding: 'utf8' })
  } catch { return null }
  if (!diff.trim()) return null
  const out = new Set()
  for (const m of diff.matchAll(/^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@/gm)) {
    const start = Number(m[1])
    for (let i = 0; i < (m[2] === undefined ? 1 : Number(m[2])); i++) out.add(start + i)
  }
  return out
}

const input = stdin()
const path = input?.tool_input?.file_path
if (!path || !path.endsWith('.rs')) process.exit(0)
if (!existsSync(path)) process.exit(0)

const rel = relative(repo, resolve(path))
if (rel.startsWith('..')) process.exit(0)

const text = readFileSync(path, 'utf8')
const scope = addedLines(rel)
const findings = [
  ...laws.commentCap(text, scope, rel),
  ...laws.miriIgnore(text, scope, rel),
  ...laws.unsafeUse(text, scope, rel),
]

// Only an error blocks. A warn (an undocumented Miri ignore) rides along when something else
// already blocked, rather than stopping an edit on a judgement call.
const errors = findings.filter(f => f.severity === 'error')
if (!errors.length) process.exit(0)
for (const f of findings) console.error(`${f.file}:${f.line} — ${f.summary}\n    ${f.detail}`)
console.error('\nFix these before continuing (.claude/rules/rust-style.md; nidus-check laws is the same detector).')
process.exit(2)
