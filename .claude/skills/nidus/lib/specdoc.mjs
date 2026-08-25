// Pure section addressing for a markdown doc: parse headings, slice one out, and answer
// "which section should I fetch". No IO, so selftest covers it without a repo (nidus-gmy.1).

/// A heading's `num` is its leading dotted number when it has one (`### 7.4 Fuzzy …`),
/// which is how the spec is cited; unnumbered headings are addressable by slug only.
export function headings(lines) {
  const out = []
  let fence = false
  lines.forEach((line, i) => {
    if (/^\s*```/.test(line)) fence = !fence
    if (fence) return
    const m = /^(#{2,4})\s+(.*)$/.exec(line)
    if (!m) return
    const text = m[2].trim()
    const num = /^(\d+(?:\.\d+)*)\.?\s+/.exec(text)
    out.push({
      level: m[1].length,
      line: i + 1,
      text,
      num: num ? num[1] : null,
      slug: text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''),
    })
  })
  out.forEach((h, i) => {
    const next = out.slice(i + 1).find(n => n.level <= h.level)
    h.end = next ? next.line - 1 : lines.length
  })
  return out
}

export function label(h) {
  return h.num ? `§${h.num}` : `#${h.slug}`
}

/// The ref already carries the number, so a title that repeats it just eats width.
export function title(h) {
  return h.num ? h.text.replace(/^\d+(?:\.\d+)*\.?\s+/, '') : h.text
}

/// Exact number beats exact slug beats slug substring, so `spec 7` cannot be hijacked
/// by a later heading that merely mentions 7 in its title.
export function locate(hs, ref) {
  const want = String(ref).replace(/^[§#]/, '').toLowerCase()
  return hs.find(h => h.num === want)
    || hs.find(h => h.slug === want)
    || hs.find(h => h.slug.includes(want))
    || null
}

export function section(lines, h) {
  return lines.slice(h.line - 1, h.end)
}

/// A section matches when every word appears somewhere inside it, not necessarily on one
/// line: the question is "which ref do I fetch", not "which line matches".
export function search(lines, words) {
  const hs = headings(lines)
  const needles = words.map(w => String(w).toLowerCase()).filter(Boolean)
  if (!needles.length) return []
  const found = []
  for (const h of hs) {
    const body = section(lines, h)
    const low = body.join('\n').toLowerCase()
    if (!needles.every(w => low.includes(w))) continue
    const hits = body
      .map((text, i) => ({ n: h.line + i, text: text.trim() }))
      .filter(l => needles.some(w => l.text.toLowerCase().includes(w)))
    found.push({ ref: label(h), title: title(h), hits, line: h.line, end: h.end })
  }
  // Report the smallest ref that satisfies the query: an outer section whose match is
  // already covered by a matching child is noise, and §7 is 669 lines of it.
  return found.filter(h => !found.some(o => o !== h && o.line > h.line && o.end <= h.end))
}

// ── The ranked tier (nidus-gmy.7) ────────────────────────────────────────────────────
// Pure translation between a BM25 hit and a section ref, so `spec find` can promote a
// ranked result without the fallback path knowing an index exists.

/// The four corpora `just docs-index` builds, as `collection → repo-relative prefix`.
/// `ingest` skips dot-entries, so `.claude/rules` is reachable only as its own root, and
/// SPEC.md/CLAUDE.md are staged because `ingest` walks directories, not files.
export const CORPORA = {
  root: '',
  rules: '.claude/rules/',
  decisions: 'decisions/',
}

/// A hit's repo-relative doc path, or null when it names a collection we did not build.
export function docForHit(collection, sourcePath) {
  const prefix = CORPORA[collection]
  if (prefix === undefined || !sourcePath) return null
  return `${prefix}${sourcePath}`
}

/// The 1-based line a character offset falls on. `ingest` stamps `nidus.char_start` per
/// chunk, which is the only thing tying a BM25 hit back to a place in the file.
export function lineForOffset(text, charStart) {
  const upto = [...text].slice(0, Math.max(0, charStart)).join('')
  return upto.split('\n').length
}

/// The innermost section containing `line`, matching what `spec <ref>` would print. Falls
/// back to the whole doc when the hit lands above the first heading (a preamble).
export function refForLine(lines, line) {
  const hs = headings(lines)
  const containing = hs.filter(h => h.line <= line && line <= h.end)
  if (!containing.length) return null
  return containing[containing.length - 1]
}

/// Collapse ranked chunk hits into one row per (doc, section), keeping the best score and
/// the order BM25 put them in. Several chunks of one section is one answer, not three.
export function dedupeRanked(rows) {
  const seen = new Map()
  for (const r of rows) {
    const key = `${r.doc}\u0000${r.ref}`
    const prior = seen.get(key)
    if (!prior || r.score > prior.score) seen.set(key, { ...r, score: Math.max(r.score, prior?.score ?? r.score) })
  }
  return [...seen.values()].sort((a, b) => b.score - a.score)
}

/// Whether a built index still describes the working tree. The digest is whatever the
/// caller can compute cheaply over the corpus (git blob ids); equality is the whole rule,
/// so a missing or unreadable digest is stale rather than an error.
export function indexIsFresh(recorded, current) {
  return Boolean(recorded) && Boolean(current) && recorded === current
}
