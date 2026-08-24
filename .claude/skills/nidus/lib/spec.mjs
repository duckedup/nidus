// Entry point for bin/spec. Wires the pure section addressing in specdoc.mjs to file IO,
// so an agent fetches §7.4 instead of reading all 177KB of SPEC.md.

import { readFileSync, existsSync } from 'node:fs'
import { resolve, dirname } from 'node:path'

import { headings, label, title, locate, section, search } from './specdoc.mjs'

const REPO = resolve(dirname(new URL(import.meta.url).pathname), '../../../..')

const argv = process.argv.slice(2)
const asJson = argv.includes('--json')
const rest = argv.filter(a => a !== '--json')

function flag(name) {
  const i = rest.indexOf(`--${name}`)
  if (i === -1) return null
  const v = rest[i + 1]
  const has = v && !v.startsWith('--')
  rest.splice(i, has ? 2 : 1)
  return has ? v : true
}

const docFlag = flag('file')
const doc = typeof docFlag === 'string' ? docFlag : 'SPEC.md'

function load() {
  const path = resolve(REPO, doc)
  if (!existsSync(path)) {
    console.error(`spec: no such file: ${doc}`)
    process.exit(2)
  }
  return readFileSync(path, 'utf8').split('\n')
}

function runToc(lines) {
  const hs = headings(lines)
  if (asJson) {
    console.log(JSON.stringify(hs.map(h => ({ ref: h.num || h.slug, lines: h.end - h.line + 1, ...h })), null, 2))
    return 0
  }
  console.log(`${doc} — ${lines.length} lines. Fetch a section with: spec <ref>\n`)
  const width = Math.min(20, Math.max(...hs.map(h => label(h).length + 2 * (h.level - 2))))
  for (const h of hs) {
    const pad = '  '.repeat(h.level - 2)
    console.log(`${pad}${label(h).padEnd(width - pad.length)}  ${title(h)}  (${h.end - h.line + 1} lines)`)
  }
  return 0
}

function runGet(lines, ref) {
  const h = locate(headings(lines), ref)
  if (!h) {
    console.error(`spec: no section '${ref}' in ${doc}. Run 'spec toc' for the index.`)
    return 2
  }
  console.log(`${doc}:${h.line}-${h.end}`)
  console.log(section(lines, h).join('\n'))
  return 0
}

function runFind(lines, words) {
  if (!words.length) {
    console.error('spec find: give at least one word')
    return 2
  }
  const found = search(lines, words)
  if (asJson) {
    console.log(JSON.stringify(found, null, 2))
    return found.length ? 0 : 1
  }
  if (!found.length) {
    console.log(`no section of ${doc} mentions all of: ${words.join(', ')}`)
    return 1
  }
  for (const f of found.slice(0, 10)) {
    const n = f.hits.length
    console.log(`${f.ref}  ${f.title}  (${n} line${n === 1 ? '' : 's'}) — fetch: spec ${f.ref.replace(/^[§#]/, '')}`)
    for (const l of f.hits.slice(0, 2)) console.log(`    ${l.n}: ${l.text.slice(0, 110)}`)
    if (n > 2) console.log(`    … ${n - 2} more`)
  }
  if (found.length > 10) console.log(`\n… ${found.length - 10} more sections. Add a word to narrow.`)
  return 0
}

const USAGE = `spec — fetch one section of a repo doc instead of reading the whole file.

  spec toc                    the heading index, with line counts
  spec <ref>                  print one section: 7, 7.4, 7.4.1, or a slug
  spec find <words…>          which sections mention all of these words
  spec --file CLAUDE.md toc   any tracked markdown (default: SPEC.md)

  --json                      machine-readable output for toc and find`

const cmd = rest[0]
if (!cmd || cmd === '--help' || cmd === '-h') { console.log(USAGE); process.exit(0) }
const lines = load()
if (cmd === 'toc') process.exit(runToc(lines))
else if (cmd === 'find' || cmd === 'grep') process.exit(runFind(lines, rest.slice(1)))
else process.exit(runGet(lines, cmd))
