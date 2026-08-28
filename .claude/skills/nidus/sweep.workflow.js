export const meta = {
  name: 'nidus-sweep',
  description: 'Sweep the codebase for duplication and complexity, or for measurable performance wins, corroborate every candidate against the real code, and propose the directory partition the blueprints will follow',
  phases: [
    { title: 'Scan', model: 'opus' },
    { title: 'Corroborate' },
    { title: 'Partition' },
  ],
}

// args (from /nidus simplify and /nidus optimize):
// {
//   mode: 'simplify' | 'optimize',
//   scope: "src/store, src/search" | "the whole repository",
//   perLens: 2,                 // candidates corroborated per lens; the rest come back unverified
//   only: ['duplication'],      // narrow the lens set
//   baseline: "<the numbers the COORDINATOR measured before launching>",
// }
//
// One workflow, two lens tables. The skeleton — scan wide, corroborate each claim against
// the real code, dedupe across lenses, then partition by directory — is the same job in
// both modes, and a second copy of it is exactly what /nidus simplify exists to find.
//
// The coordinator measures, not this workflow. Several agents run at once on one machine
// and the coordinator may be running a lane beside them, so any timing taken in here is
// taken under contention; `baseline` carries the numbers in instead.
const cfg = typeof args === 'string' ? JSON.parse(args) : (args || {})
const mode = cfg.mode === 'optimize' ? 'optimize' : 'simplify'
const scope = cfg.scope || 'the whole repository'
const perLens = Number.isInteger(cfg.perLens) ? cfg.perLens : 2

const CONTEXT = `nidus is an embeddable pure-Rust vector store: dense vectors plus typed metadata in one
on-disk directory, exact brute-force cosine search, with opt-in ANN, quantisation, an HTTP
server, an MCP surface, and three client SDKs. Read CLAUDE.md first. SPEC.md is 2577 lines —
do NOT read it whole: \`.claude/skills/nidus/bin/spec toc\` is the index, \`spec find <words>\`
says which section covers a topic, and \`spec <ref>\` prints just that one.

The module map matters to this sweep: src/ is the library, src/cli, src/server,
src/server/mcp and src/bin are feature-gated binaries, sdks/js sdks/go sdks/python wrap the
HTTP surface, tests/ and tests/e2e/ are separate crates, benchmarks/ is a quarantined crate,
docs/ is the site. \`just ci\` compiles the PURE LIBRARY ONLY — it does not see src/cli,
src/server, src/bin, the SDKs, the benchmarks or the e2e suite, which is why a caller living
in one of those is the one nobody notices.`

// nidus-jni: every agent here shares ONE checkout with the coordinator, which may be
// running the verification lanes right now. A temporary edit that is reverted still fails
// that run and leaves nothing behind to diagnose it by.
const NO_MUTATION = `YOU SHARE ONE CHECKOUT with the coordinator, which may be running \`just ci\`, \`just ci-cli\`,
\`just test-e2e\` or a benchmark right now. Never write a tracked file in it — not even
temporarily and reverted. A skeptic once added a \`#[test]\`, ran it and reverted it; the
concurrent lane failed on a test attributable to nothing anybody had changed, and by the time
someone looked the file was byte-identical to HEAD again.

Fine here: reading anything, \`cargo check\`/\`cargo test\`/\`cargo clippy\` of the code AS IT
STANDS, and \`gh\`/\`bd\`/\`git\`/\`rg\` queries. To run MODIFIED code, cut your own worktree
(\`git worktree add /tmp/sw-<unique> HEAD\`), work there, and remove it when done.`

// What a sweep may never quietly trade away. Both modes get this: the whole point of both
// lanes is that they are not allowed to change what nidus does.
const NON_NEGOTIABLE = `THIS SWEEP MAY NOT CHANGE WHAT NIDUS DOES. Not the public API, not the CLI output, not the
HTTP or MCP response shapes, not the on-disk bytes, not error text a test asserts on, not
search results. A candidate that changes any of those is a DESIGN CHANGE: report it as such,
with what it would change, and let the developer file an issue. Do not propose it as work.

These need an issue before anyone writes code, per CLAUDE.md and decisions/:
- a dependency that costs build time (D0005) — the lean build stays under a minute,
- a second \`allow(unsafe_code)\` (D0006) — src/data/mmap.rs is the only sanctioned one,
- a non-additive on-disk format change,
- weakening durability: append vectors, fsync data, append committing log records, fsync
  log; readers are lock-free and must never observe a torn state.

And the test suite is the control, not the variable. If a candidate would require editing or
deleting an existing test to pass, that is a behaviour change wearing a refactor's clothes.
Say so; do not propose the test edit.`

const MODES = {
  simplify: {
    charter: `You are sweeping nidus for DUPLICATION and needless COMPLEXITY, at staff level. The
deliverable is a smaller, clearer codebase that does exactly what it does today. Shorter is
not the goal and neither is DRY for its own sake: the goal is that a competent reader has
fewer things to hold in their head. Two straight-line functions can beat one helper with five
boolean parameters, and deleting an abstraction is as much a win as adding one.`,
    lenses: [
      {
        key: 'duplication',
        prompt: `Find code that exists more than once. Near-identical functions or blocks in sibling
modules, the same match arms written twice, error mapping or validation repeated at every call
site, a helper reimplemented because nobody knew the first one existed, and magic values —
meta keys, header names, env vars, limits, error strings — spelled out in several files where
one constant belongs. nidus splits modules by concern into sibling files, so the duplication
that costs the most is usually ACROSS directories (src/cli vs src/server vs src/server/mcp),
not inside one file. For every candidate, name every site with path and line range and quote
enough of each to show they really are the same thing.`,
      },
      {
        key: 'surface-parity',
        prompt: `nidus ships one feature across core, CLI, HTTP, MCP and three SDKs, in one PR (CLAUDE.md:
"a feature ships whole"). That is six places to say the same thing, and it is where this
codebase accumulates its most expensive duplication. Find behaviour re-implemented per surface
instead of shared: argument validation written differently in src/cli and src/server, DTO
conversion hand-written in src/server/dto.rs where the library type already knows the answer,
an MCP tool re-deriving what the HTTP handler computed, a default spelled out separately in
each surface so the four can drift apart. Then look for the inverse defect, which is worse: a
surface that ALREADY drifted and now disagrees with its siblings. For each, say which layer
the shared code belongs in and which surfaces would call it.`,
      },
      {
        key: 'altitude',
        prompt: `Find code at the wrong altitude, in both directions. Too low: a function doing three
things whose caller must know all three, an inlined loop that is really a named operation,
a caller reaching through two layers to touch a field. Too high: an indirection with one
implementation and one caller, a generic parameter with one instantiation, a trait nobody
else implements, a builder for a two-field struct, a config knob every caller sets to the
same value, an enum with one live variant, a wrapper type that only forwards. Premature
abstraction is the commoner defect and DELETING it is simplification, not loss. For each,
say what the code looks like at the right altitude and how many lines and concepts go away.`,
      },
      {
        key: 'vestigial',
        prompt: `Find code that no longer earns its place: unreachable branches, \`pub\` items with no
caller, a path superseded by a newer one that still compiles, a feature gate nothing enables,
a struct field written and never read, a workaround for something already fixed, a shim for a
format version nobody writes any more.

THE BAR IS PROOF, NOT SUSPICION. Before proposing any removal, grep the WHOLE repo — src/,
tests/, tests/e2e/, benchmarks/, sdks/js, sdks/go, sdks/python, docs/, charts/ and the
justfile — and say where you looked, because \`just ci\` compiles the pure library only and a
caller in src/cli or in an SDK is invisible to it. A \`pub\` item is also part of the published
API surface: removing one is a breaking change, so flag it as a decision for the developer
rather than a candidate. Anything you cannot prove dead goes in \`questions\`, not in
\`candidates\`.`,
      },
      {
        key: 'test-scaffolding',
        prompt: `Read the tests, where duplication concentrates and nobody looks: tests/, tests/e2e/, the
inline \`#[cfg(test)]\` modules, and the SDK suites. Find repeated fixture construction (the
same store built row by row in twenty tests), hand-rolled temp directories and port allocation
where a helper already exists, assertion blocks copied with one value changed that a
table-driven test would say once, and setup that has drifted so two tests claim to build the
same fixture and do not. Say which existing helper each site should use, or what the missing
helper is.

Do NOT propose deleting or weakening any assertion. Fewer lines of test is not the goal, and a
test that stops proving what it proved is a regression. Consolidating twenty setups behind one
helper is in scope; making twenty tests into five is not, unless the fifteen assert nothing the
five do not.`,
      },
    ],
  },
  optimize: {
    charter: `You are sweeping nidus for PERFORMANCE WINS, at staff level, and results must not move. The
deliverable is the same answers, faster or cheaper. Every candidate must name the benchmark
that would show the win, because a win nobody can measure did not happen; and every candidate
must say what it costs in readability, because nidus is a small codebase whose legibility is
part of the product.`,
    lenses: [
      {
        key: 'hot-path',
        prompt: `Profile by reading: find the work nidus does per query and per row. The kernels are
src/store/scoring.rs, src/store/quant.rs, src/store/rank.rs, src/store/aggregate.rs, plus
src/search/, src/ann/, src/filter/, src/glob/ and src/fts/. Look for a redundant pass over the
candidate set, a full sort where a partial select would do, per-row work that is
loop-invariant, iterator adapters and bounds checks in the innermost loop where a slice would
compile better, a distance kernel recomputing a norm it could cache, an early exit present in
one path and missing from its sibling. Include algorithmic wins: a caller-supplied glob, regex
or fuzzy term evaluated superlinearly (SPEC §7.4/§7.5), a structure scanned linearly once per
row. For each, say at what input size it starts to matter — a constant-factor win on a
thousand rows is not a candidate.`,
      },
      {
        key: 'io',
        prompt: `Find read and write amplification: the fsync count per batch and whether it is per row or
per commit, the log replay cost at open, mmap access patterns and page-cache behaviour, a
header re-read per operation, a manifest rewritten whole for a one-field change, a file opened
per call, a seek pattern that defeats readahead.

The durability order is NOT available to you: append vectors, fsync data, append committing
log records, fsync log. Any candidate that reorders, removes or weakens an fsync, or that lets
a lock-free reader observe a torn state, is a design change — report it as one, do not propose
it. Everything that keeps the order intact is fair game, and batching work WITHIN a commit
usually is.`,
      },
      {
        key: 'alloc',
        prompt: `Count the allocations and copies on the per-request and per-row paths. Vec and String
growth without a \`reserve\` where the size is known, \`format!\` or \`to_string\` on a path
taken per row, a \`collect()\` into a temporary that is immediately iterated, \`clone()\` or
\`to_owned()\` where a borrow would do, an \`Arc\` cloned per item, a HashMap rebuilt per call,
an error value constructed on the success path, serde materialising a \`Value\` where a typed
struct would deserialize straight through. For each site, say roughly how many allocations it
costs per query or per row of a realistic workload — that number is what decides whether it is
worth doing at all.`,
      },
      {
        key: 'server',
        prompt: `Find what lives in the gap between the library and the server. \`just bench-server\` exists
to expose exactly this: the same dataset in-process and over HTTP, so the difference between
the two rows IS the server's overhead. Look at JSON encode and decode (src/server/dto.rs), the
RwLock plus spawn_blocking hop, per-request setup that could be hoisted into shared state,
middleware doing work on every request that only some requests need, a handler cloning a whole
result to reshape it, an MCP tool re-deriving what the HTTP handler already computed, a
response built as a String before it is serialised. Read benchmarks/baselines/ and
benchmarks/README.md for what has already been measured, and say which recorded number your
candidate would move.`,
      },
      {
        key: 'build-cost',
        prompt: `Build time and dependency weight ARE performance here, and they are CI-asserted (D0014,
D0015): the lean library build (\`--no-default-features\`) stays under a minute and the default
build under two. Find: a generic function monomorphised over many types where an inner
non-generic \`fn\` would compile once, a heavy derive or macro on a large type, a dependency
reaching the lean build that only a feature-gated layer needs, a dependency taken with default
features where one feature would do, a module compiled in the lean build but reachable only
under a feature. \`cargo build --timings\` and \`just deps\` are your instruments. Say which of
the two budgets each win moves, and by roughly how much.`,
      },
    ],
  },
}

const M = MODES[mode]

const CANDIDATE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'candidates', 'questions', 'design_changes'],
  properties: {
    lens: { type: 'string' },
    candidates: {
      type: 'array',
      description: 'Ordered best payoff first — the first few are the ones that get corroborated',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'kind', 'sites', 'evidence', 'proposal', 'payoff', 'risk', 'effort'],
        properties: {
          title: { type: 'string', description: 'One line naming the specific thing, not the category' },
          kind: { type: 'string', description: 'e.g. duplication, dead-code, over-abstraction, allocation, io' },
          sites: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['path', 'lines', 'what'],
              properties: { path: { type: 'string' }, lines: { type: 'string' }, what: { type: 'string' } },
            },
          },
          evidence: { type: 'string', description: 'The quoted code, measurement or grep that proves it' },
          proposal: { type: 'string', description: 'What the code becomes, concretely' },
          payoff: { type: 'string', description: 'What improves and by how much: lines and concepts removed, or the metric moved' },
          expected_win: { type: 'string', description: 'optimize only: the metric and rough magnitude' },
          benchmark: { type: 'string', description: 'optimize only: the just recipe that would show it' },
          risk: { type: 'string', description: 'How this could change observable behaviour, and what stops it' },
          effort: { enum: ['small', 'medium', 'large'] },
        },
      },
    },
    questions: {
      type: 'array',
      description: 'Things you suspect but could not prove — NOT candidates',
      items: { type: 'string' },
    },
    design_changes: {
      type: 'array',
      description: 'Wins that would change behaviour, a format, durability, deps or unsafe: for an issue, not for this PR',
      items: { type: 'string' },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['confidence', 'preserves_behaviour', 'objections', 'reason'],
  properties: {
    confidence: { type: 'integer', description: '0-100 per the rubric' },
    preserves_behaviour: { type: 'boolean' },
    objections: {
      type: 'array',
      description: 'Every real difference between the sites, or every reason the win is not where it is claimed',
      items: { type: 'string' },
    },
    reason: { type: 'string' },
  },
}

const PARTITION_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['summary', 'shortlist', 'deferred', 'groups', 'ordering', 'scope_questions', 'open_questions'],
  properties: {
    summary: { type: 'string', description: '2-3 sentences: what this sweep found and what one PR should take' },
    shortlist: {
      type: 'array',
      description: 'What to do NOW, in order: one coherent, reviewable PR and no more',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'why_now', 'files'],
        properties: {
          title: { type: 'string' },
          why_now: { type: 'string' },
          files: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    deferred: {
      type: 'array',
      description: 'Real candidates that should be FILED AS BEADS rather than dropped, with why they wait',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'why_later'],
        properties: { title: { type: 'string' }, why_later: { type: 'string' } },
      },
    },
    groups: {
      type: 'array',
      description: 'Ordered groups covering the shortlist; everything in one group is file-disjoint and parallel-safe',
      items: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['dir', 'scope', 'verify'],
          properties: {
            dir: { type: 'string' },
            scope: { type: 'string', description: 'What changes here' },
            verify: { type: 'array', items: { type: 'string' } },
          },
        },
      },
    },
    ordering: { type: 'array', items: { type: 'string' } },
    scope_questions: {
      type: 'array',
      description: 'Forks that change WHICH files change, so they must be settled before any blueprint is written',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['question', 'forks', 'options'],
        properties: {
          question: { type: 'string' },
          forks: { type: 'string' },
          options: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['label', 'implication'],
              properties: { label: { type: 'string' }, implication: { type: 'string' } },
            },
          },
          recommendation: { type: 'string' },
        },
      },
    },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

const baselineBlock = cfg.baseline
  ? `\nTHE COORDINATOR MEASURED THIS TREE BEFORE LAUNCHING YOU. These numbers are the control:\n${cfg.baseline}\n`
  : ''

function scanPrompt(l) {
  return `${CONTEXT}

${M.charter}

SCOPE OF THIS SWEEP: ${scope}
${baselineBlock}
YOUR LENS — ${l.key}:
${l.prompt}

${NON_NEGOTIABLE}

${NO_MUTATION}
${mode === 'optimize' ? `
DO NOT RUN BENCHMARKS. Four other agents are working on this machine and the coordinator may
be running one beside you, so any timing you take is taken under contention and is worse than
no timing at all. Read the recorded baselines, read the code, and name the benchmark that
WOULD show your win. The coordinator runs it once, alone, afterwards.
` : ''}
Read real files and quote real paths and line ranges — a candidate without a site is not a
candidate. Order your candidates best payoff first, because only the first ${perLens} of them get
independently corroborated and the rest are reported unverified. Do not pre-filter for
importance beyond that ordering: a separate agent scores each one and weak ones are dropped
automatically. Do not edit anything and do not propose an implementation in detail; this is
research another agent turns into a plan.`
}

const RUBRIC = mode === 'simplify'
  ? `Score 0-100 how confident you are that this is REAL, WORTH DOING and BEHAVIOUR-PRESERVING:
- 0: not real. The sites are not equivalent, the code is not reachable the way the reviewer
  thinks, or the "simplification" is only shorter.
- 25: real duplication, but merging it costs more clarity than it buys.
- 50: real and worth doing, but small.
- 75: real, worth doing, and you have checked every caller and every test.
- 100: certain, and you can name the exact shape of the result.
Default LOW when uncertain. Your job is to REFUTE, not to agree.`
  : `Score 0-100 how confident you are that this is a REAL, MEASURABLE win that does not change results:
- 0: not real. The site is not hot, the win is imaginary, or the compiler already does it.
- 25: real but on a path taken rarely, or unmeasurable with any benchmark that exists.
- 50: real and measurable, but small or costly in readability.
- 75: real, on a demonstrably hot path, and an existing benchmark would show it.
- 100: certain, with the benchmark and the expected magnitude both named.
Default LOW when uncertain. Your job is to REFUTE, not to agree.`

function corroboratePrompt(c) {
  const common = `You are the corroborator. A scanner claims this candidate in nidus:

TITLE: ${c.title}
KIND: ${c.kind}
SITES:
${(c.sites || []).map(s => `  ${s.path}:${s.lines} — ${s.what}`).join('\n')}
EVIDENCE OFFERED: ${c.evidence}
PROPOSAL: ${c.proposal}
CLAIMED PAYOFF: ${c.payoff}${c.expected_win ? `\nCLAIMED WIN: ${c.expected_win}` : ''}${c.benchmark ? `\nCLAIMED BENCHMARK: ${c.benchmark}` : ''}

Go and read the real code at every site. Do not take the quoted evidence on trust.`

  const questions = mode === 'simplify'
    ? `Answer these, and put each real one in \`objections\`:
1. **Are the sites actually equivalent?** Enumerate EVERY difference, however small: a different
   error message, an extra guard, a different default, a different feature gate, one handling an
   edge case the other does not, a different lock held. A merge erases those differences. Say
   which of them matter and to whom.
2. **Does anything depend on the difference?** Grep for callers and tests across src/, tests/,
   tests/e2e/, benchmarks/, sdks/ and docs/ — remember \`just ci\` compiles the pure library
   only, so a caller in src/cli or an SDK is invisible to it.
3. **Would any observable behaviour change?** Public API, CLI output, HTTP or MCP response
   shape, on-disk bytes, error text a test asserts on. If yes, \`preserves_behaviour\` is false
   and the confidence is low, whatever the payoff.
4. **Would any existing test have to be edited or deleted to make this pass?** If yes, this is a
   behaviour change wearing a refactor's clothes. Say so.
5. **Is the result actually simpler, or only shorter?** A helper with five boolean parameters is
   not simpler than two straight-line functions. Fewer concepts is the test, not fewer lines.`
    : `Answer these, and put each real one in \`objections\`:
1. **Is this site actually hot?** What executes it, how many times per query or per row, and at
   what input size does it start to show? Trace the call path and say it out loud. A win on a
   path taken once at open time is not a win.
2. **Which EXISTING benchmark would show it?** \`just bench\`, \`bench-server\`, \`bench-quant\`,
   \`bench-ann\`, \`bench-write\`, \`bench-crit\`, or a criterion bench in benchmarks/benches/. If
   none of them would, say so — the developer cannot accept a win nobody can measure, and a new
   bench is itself work to be planned.
3. **Has the compiler already done it?** Check for an existing \`#[inline]\`, an iterator chain
   LLVM already fuses, a \`reserve\` upstream, a bounds check already elided. Read the code, not
   the folklore.
4. **Does it change results at all?** Recall, ordering, tie-breaking, rounding, error text.
   That is a behaviour change, not an optimization: \`preserves_behaviour\` false.
5. **What does it cost?** Readability, a law (unsafe, a dependency, the build budget, the
   durability order), or a new invariant a future reader must hold. Name it.

DO NOT RUN BENCHMARKS — you are one of several agents on a contended machine, and a timing
taken here is worse than no timing. Read, trace, and grep instead.`

  return `${CONTEXT}

${common}

${questions}

${NON_NEGOTIABLE}

${NO_MUTATION}

${RUBRIC}`
}

phase('Scan')

const scanners = (cfg.only && cfg.only.length ? M.lenses.filter(l => cfg.only.includes(l.key)) : M.lenses)

// Each lens's candidates are corroborated the moment that lens lands, so a slow scanner
// never holds up verification of a fast one.
const swept = await pipeline(
  scanners,
  l => agent(scanPrompt(l), {
    label: `scan:${l.key}`,
    phase: 'Scan',
    model: 'opus',
    schema: CANDIDATE_SCHEMA,
  }).then(r => ({
    key: l.key,
    candidates: (r && r.candidates) || [],
    questions: (r && r.questions) || [],
    design_changes: (r && r.design_changes) || [],
  })),
  (r) => {
    const found = r.candidates
    const take = found.slice(0, perLens)
    // A cap that says nothing reads as "that lens found nothing more".
    if (found.length > take.length) {
      log(`${r.key}: corroborating ${take.length} of ${found.length}, ${found.length - take.length} reported UNVERIFIED`)
    }
    const rest = found.slice(take.length).map(c => ({ ...c, lens: r.key, confidence: null, objections: [], verdict_reason: 'not corroborated (over the per-lens cap)' }))
    return parallel(take.map(c => () => agent(corroboratePrompt(c), {
      label: `check:${c.title.slice(0, 40)}`,
      phase: 'Corroborate',
      model: 'sonnet',
      schema: VERDICT_SCHEMA,
    }).then(v => ({
      ...c,
      lens: r.key,
      confidence: v ? v.confidence : null,
      preserves_behaviour: v ? v.preserves_behaviour : null,
      objections: (v && v.objections) || [],
      verdict_reason: (v && v.reason) || 'corroborator returned nothing',
    })))).then(done => ({ ...r, verified: [...done.filter(Boolean), ...rest] }))
  },
)

const lensReports = swept.filter(Boolean)

// Lenses overlap by design — `altitude` and `duplication` both find a one-caller helper —
// so the same candidate can arrive twice with different scores. Same shape as the review
// workflow's deduper, and for the same reason.
const words = s => new Set(String(s || '').toLowerCase().match(/[a-z_]{4,}/g) || [])
function similar(a, b) {
  const A = words(a), B = words(b)
  if (!A.size || !B.size) return false
  return [...A].filter(w => B.has(w)).length / Math.min(A.size, B.size) >= 0.6
}
const siteKey = c => (c.sites || []).map(s => s.path).sort().join('|')

const all = []
for (const c of lensReports.flatMap(r => r.verified || [])) {
  const dup = all.find(o => siteKey(o) === siteKey(c) && similar(o.title, c.title))
  if (!dup) { all.push({ ...c, lenses: [c.lens] }); continue }
  dup.lenses.push(c.lens)
  if ((c.confidence || 0) > (dup.confidence || 0)) Object.assign(dup, c, { lenses: dup.lenses })
}

const score = c => (c.confidence == null ? -1 : c.confidence)
const corroborated = all.filter(c => score(c) >= 70 && c.preserves_behaviour !== false).sort((a, b) => score(b) - score(a))
const weak = all.filter(c => !corroborated.includes(c))

log(`${all.length} candidate(s) after dedupe, ${corroborated.length} corroborated at >=70 and behaviour-preserving`)

phase('Partition')

const partition = await agent(
  `${CONTEXT}

${M.charter}

SCOPE OF THIS SWEEP: ${scope}
${baselineBlock}
CORROBORATED CANDIDATES (each independently checked against the real code):
${JSON.stringify(corroborated, null, 2)}

NOT CORROBORATED — weak, refuted, behaviour-changing, or over the per-lens cap. Do NOT put any
of these in the shortlist; the good ones belong in \`deferred\` so they can be filed:
${JSON.stringify(weak.map(c => ({ title: c.title, lens: c.lens, confidence: c.confidence, objections: c.objections, sites: (c.sites || []).map(s => s.path) })), null, 2)}

OPEN QUESTIONS AND DESIGN CHANGES the scanners surfaced:
${JSON.stringify({ questions: lensReports.flatMap(r => r.questions), design_changes: lensReports.flatMap(r => r.design_changes) }, null, 2)}

Your job is to turn this into ONE PR's worth of work, and to say clearly what is being left.

1. **Shortlist ruthlessly.** A sweep of a whole codebase returns more than anybody can review.
   A 40-file refactor does not get merged, it sits. Pick the coherent slice with the best
   payoff-to-risk ratio that one reviewer can hold in their head, and put everything else in
   \`deferred\` with a reason — the coordinator files those as beads, so nothing is wasted.
   Prefer candidates that are independently valuable over ones that only pay off together.
2. **Partition the shortlist by directory.** One unit owns one directory or file set; units in
   the same group MUST be file-disjoint, because they are implemented in parallel worktrees.
   Put a unit in a later group only when it genuinely needs an earlier group's code to exist —
   for this kind of work that is usually "introduce the shared helper" then "move the callers
   onto it", and getting that order wrong is the main way a sweep breaks the build.
3. **Each unit's \`verify\` is the exact just recipes that cover it.** \`just ci\` does NOT
   compile src/cli, src/server or src/bin — those need \`just ci-cli\`; the MCP surface needs
   \`--features mcp\`; the SDKs and the e2e suite have their own lanes. Never list \`just miri\`:
   CI's required Miri job covers codec and kernel changes on the PR.${mode === 'optimize' ? `
   For this mode also name the benchmark that proves each unit's win, separately from the
   correctness lanes. A unit with no benchmark is a unit whose win nobody can confirm.` : ''}
4. **Split what you do not know into two piles.** \`scope_questions\` are forks the developer
   must settle BEFORE any blueprint is written, because each changes which units exist or which
   files they touch — how far the shortlist should reach, whether a \`pub\` item may be removed
   (it is a breaking change), whether a surface should be brought into line or left alone,
   whether to take the risky-but-valuable one. Give concrete options with their consequences.
   If the sweep is genuinely unambiguous, return an empty array rather than inventing a
   question. Everything cheap and reversible — a name, an ordering — is an \`open_question\`
   and goes in a blueprint instead.

Remember what this lane may not do: no behaviour change, no API change, no test edited to fit,
no new dependency, no unsafe, no format change. If the best candidate needs one of those, it
goes in \`deferred\` with that named as the reason.`,
  { label: 'partition', phase: 'Partition', schema: PARTITION_SCHEMA },
)

return {
  mode,
  scope,
  corroborated,
  weak: weak.map(c => ({ title: c.title, lens: c.lens, confidence: c.confidence, preserves_behaviour: c.preserves_behaviour, objections: c.objections, why: c.verdict_reason })),
  questions: lensReports.flatMap(r => r.questions),
  design_changes: lensReports.flatMap(r => r.design_changes),
  partition,
}
