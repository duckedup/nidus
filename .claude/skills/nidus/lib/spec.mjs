// Entry point for bin/spec. Wires the pure section addressing in specdoc.mjs to file IO,
// so an agent fetches §7.4 instead of reading all 177KB of SPEC.md.

import { readFileSync, existsSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { resolve, dirname } from 'node:path'

import {
  headings, label, title, locate, section, search,
  docForHit, lineForOffset, refForLine, dedupeRanked, indexIsFresh,
} from './specdoc.mjs'

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

const INDEX = resolve(REPO, 'target/docs-index')

/// The prebuilt binary, or null. Deliberately not `cargo run`: `find` is called constantly,
/// and a build check per invocation would cost more than the ranking saves.
function nidusBin() {
  for (const p of ['target/debug/nidus', 'target/release/nidus']) {
    const abs = resolve(REPO, p)
    if (existsSync(abs)) return abs
  }
  return null
}

function currentDigest() {
  try {
    return execFileSync(resolve(REPO, 'scripts/docs-index.sh'), ['--digest'], {
      cwd: REPO, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
  } catch { return null }
}

/// The digest the index recorded, read back out of the store itself (the `meta` sentinel
/// record `just docs-index` writes). Any failure — no store, no record, no binary — is a
/// missing digest, which reads as stale.
function recordedDigest() {
  try {
    const out = execFileSync(nidusBin(), ['get', 'meta', '--dir', INDEX], {
      cwd: REPO, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'],
    })
    const rec = JSON.parse(out).find(r => r.id === 'docs-index.digest')
    return rec?.attrs?.digest?.Str ?? null
  } catch { return null }
}

/// Why the ranked tier is unavailable, or null when it is usable. Never throws and never
/// prompts: a missing index is the normal state of a fresh clone (D0013).
function rankedUnavailable() {
  if (!existsSync(INDEX)) return 'no docs index yet'
  if (!nidusBin()) return 'the nidus binary is not built'
  if (!indexIsFresh(recordedDigest(), currentDigest())) return 'the docs index is stale'

  return null
}

/// BM25 hits → ranked section refs. Each hit names a doc and a char offset; the section is
/// whatever `spec <ref>` would print for that line, so the ref is directly fetchable.
function rankedFind(words) {
  const out = execFileSync(nidusBin(), [
    'text-search', '--dir', INDEX, 'nidus.text', words.join(' '), '--top-k', '30',
  ], { cwd: REPO, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
  const rows = []
  for (const hit of JSON.parse(out)) {
    const attrs = hit.attrs || {}
    const doc = docForHit(hit.collection, attrs['nidus.source_path']?.Str)
    if (!doc || !existsSync(resolve(REPO, doc))) continue
    const text = readFileSync(resolve(REPO, doc), 'utf8')
    const lines = text.split('\n')
    const h = refForLine(lines, lineForOffset(text, attrs['nidus.char_start']?.Int ?? 0))
    // No enclosing `##`+ heading — a doc titled with a bare `#`, or a preamble. Dropping the
    // hit made whole files invisible and the query answer "nothing found"; name the file.
    rows.push(h
      ? { doc, ref: label(h), title: title(h), score: hit.score, line: h.line, end: h.end }
      : { doc, ref: null, title: '(whole file)', score: hit.score, line: 1, end: lines.length })
  }
  return dedupeRanked(rows)
}

const USAGE = `spec — fetch one section of a repo doc instead of reading the whole file.

  spec toc                    the heading index, with line counts
  spec <ref>                  print one section: 7, 7.4, 7.4.1, or a slug
  spec find <words…>          which sections mention all of these words
  spec --file CLAUDE.md toc   any tracked markdown (default: SPEC.md)

  --json                      machine-readable output for toc and find`

/// Ranked output. One row per section, across every corpus, so the answer names the doc
/// as well as the ref — `spec find` over SPEC.md alone cannot point at a rule or an ADR.
function runRanked(rows, words) {
  if (asJson) {
    console.log(JSON.stringify(rows, null, 2))
    return rows.length ? 0 : 1
  }
  for (const r of rows.slice(0, 10)) {
    const file = r.doc === 'SPEC.md' ? '' : ` --file ${r.doc}`
    const fetch = r.ref ? `spec${file} ${r.ref.replace(/^[§#]/, '')}` : `spec${file} toc`
    const ref = r.ref || '—'
    console.log(`${r.doc}  ${ref}  ${r.title}  (${r.end - r.line + 1} lines) — fetch: ${fetch}`)
  }
  if (rows.length > 10) console.log(`\n… ${rows.length - 10} more. Add a word to narrow.`)
  return 0
}

const cmd = rest[0]
if (!cmd || cmd === '--help' || cmd === '-h') { console.log(USAGE); process.exit(0) }

// The ranked tier applies to an unqualified `find` only: `--file` names one doc on purpose,
// and the index spans all four. Any reason it is unavailable is one stderr line, never an
// error and never a prompt — the text search below is the floor and always works.
if ((cmd === 'find' || cmd === 'grep') && docFlag === null && rest.length > 1) {
  const why = rankedUnavailable()
  if (why) {
    console.error(`spec: ${why}; using text search. \`just docs-index\` builds the ranked one.`)
  } else {
    try {
      const rows = rankedFind(rest.slice(1))
      // Empty is not an answer: the ranked tier must never be worse than the floor it
      // replaces, so fall through to the text search rather than reporting nothing.
      if (rows.length) process.exit(runRanked(rows, rest.slice(1)))
    } catch (e) {
      console.error(`spec: the docs index could not be queried (${e.message}); using text search.`)
    }
  }
}

const lines = load()
if (cmd === 'toc') process.exit(runToc(lines))
else if (cmd === 'find' || cmd === 'grep') process.exit(runFind(lines, rest.slice(1)))
else process.exit(runGet(lines, cmd))
