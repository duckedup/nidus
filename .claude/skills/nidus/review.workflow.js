export const meta = {
  name: 'nidus-review',
  description: 'Review a nidus change across repo-specific dimensions, then adversarially verify every finding',
  phases: [
    { title: 'Review' },
    { title: 'Verify' },
  ],
}

// args (from the /nidus review path):
// {
//   ref: "PR #73" | "austin/foo vs main" | "working tree",  // human label
//   diffCmd: "gh pr diff 73" | "git diff main...HEAD",       // how an agent reads the diff
//   changed: ["src/store/read.rs", ...],
//   laws: [ ...nidus-check laws findings... ],               // deterministic, already true
//   effort: "medium" | "high",
// }
const cfg = typeof args === 'string' ? JSON.parse(args) : (args || {})

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['file', 'line', 'category', 'summary', 'failure_scenario', 'evidence'],
        properties: {
          file: { type: 'string' },
          line: { type: 'integer' },
          category: { type: 'string' },
          summary: { type: 'string', description: 'One sentence stating the defect' },
          failure_scenario: { type: 'string', description: 'Concrete inputs/state → wrong output or crash' },
          evidence: { type: 'string', description: 'The code, CLAUDE.md line, or history that proves it' },
        },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['confidence', 'reason'],
  properties: {
    confidence: { type: 'integer', description: '0-100 per the rubric' },
    reason: { type: 'string' },
  },
}

// Ported from the official code-review command — these are the things that make a
// review report noise rather than signal.
const FALSE_POSITIVES = `Do NOT report:
- Pre-existing issues on lines this change did not touch.
- Anything the compiler, clippy or rustfmt rejects outright — CI runs those separately.
- Pedantic nitpicks a senior engineer would not raise.
- Missing test coverage, general security posture, or documentation gaps, unless CLAUDE.md demands them explicitly.
- Style rules explicitly silenced in the code (an #[allow] with a reason).
- Intentional behaviour changes that are the point of this change.

DO report, because these are the defects that survive review here:
- Code ADDED by this change that is wrong, including TEST code. A new test that can fail
  spuriously — it reads process-global state, depends on timing, ordering, or on other tests
  not running concurrently — is a real defect, not a nitpick. cargo runs lib tests in
  parallel threads in one process, so any test asserting on a \`static\` counter, an env var,
  a shared temp path, or a global registry is suspect.
- A test that passes for the wrong reason, or asserts something weaker than it claims to.`

const HOUSE_RULES = `nidus's laws (CLAUDE.md), for context — a deterministic checker already covers the
mechanical ones, so do not re-report those; use them to judge intent:
- Pure-library build stays fast and FFI-light. src/cli, src/server, src/bin and the
  embed/summarize/memory layers are feature-gated and must never be reachable from the
  default build. \`just ci\` does NOT compile them.
- #![deny(unsafe_code)]; src/data/mmap.rs is the single sanctioned exception.
- Durability: append vectors → fsync data → append committing log records → fsync log.
  A crash may lose the in-flight batch and nothing else. Appends are atomic and roll back
  to a row/frame boundary; upsert is all-or-nothing. Readers are lock-free and must never
  observe a torn state: replay the log and ignore rows beyond the data file's size.
- All on-disk encoding is little-endian, length-prefixed, CRC32-checked.
- anyhow everywhere; no hand-rolled error enum.
- Comments cap at 3 lines and must say what the code cannot.
- The MCP tool surface is text-native: no tool may take a raw vector.
- Wire DTOs in src/server/dto.rs mirror the library's Hit/Footprint — the binary adapts to
  the library, never the reverse.`

const DIMENSIONS = [
  {
    key: 'durability',
    prompt: `Review this change for DURABILITY and CRASH-CONSISTENCY defects: fsync ordering, the
commit record, torn-tail recovery, rollback on a partial append, all-or-nothing upsert,
try_reserve/OOM paths, and the lock-free cross-process reader snapshot rule. A finding here
must describe a concrete crash or interleaving that leaves the store wrong or unreadable.`,
  },
  {
    key: 'concurrency',
    prompt: `Review this change for CONCURRENCY defects: the single-writer lock, group commit, the
cluster writer lease (CAS gating, staleness, fencing), shared metrics counters, and any
assumption that two processes cannot interleave. Describe the actual interleaving.`,
  },
  {
    key: 'build-thesis',
    prompt: `Review this change against nidus's BUILD-AND-SHIP THESIS: does anything reachable from
the DEFAULT build pull a binary-only or async-edge dependency (clap, tokio, axum, tower,
reqwest, rmcp)? Are new modules correctly #[cfg(feature = …)]-gated in src/lib.rs? Would a
plain \`cargo add nidus\` still compile in seconds with no FFI? Check tests/build_thesis.rs
still asserts what it claims to.`,
  },
  {
    key: 'api-contract',
    prompt: `Review this change for API and WIRE-CONTRACT defects: do the DTOs in src/server/dto.rs
still mirror the library types; does the MCP surface stay text-native (no tool taking a raw
vector); do the three SDKs (sdks/js, sdks/go, sdks/python) still agree with the HTTP surface
they wrap; is anything a breaking change that the version bump does not reflect?`,
  },
  {
    key: 'bugs',
    prompt: `Read ONLY the diff and scan for outright bugs: wrong index or bound, inverted condition,
mishandled error, resource leak, unwrap on a fallible path, incorrect arithmetic in a
distance kernel or quantisation step. Do not read wider context; do not report nitpicks.`,
  },
  {
    key: 'history',
    prompt: `Use git history as the lens. For the regions this change touches, read \`git log -p\` and
\`git blame\`, and look at review comments on earlier PRs that touched the same files
(\`gh pr list --state merged\` then \`gh pr view <n> --comments\`). Flag anything that
reintroduces a bug previously fixed, contradicts a decision recorded in a commit message,
or repeats something a reviewer already objected to. Cite the commit or PR.`,
  },
]

const RUBRIC = `Score your confidence 0-100 that this is a REAL defect worth reporting:
- 0: false positive; does not survive light scrutiny, or is pre-existing.
- 25: might be real, but you could not verify it.
- 50: verified real, but a nitpick or rare in practice.
- 75: verified, very likely hit in practice, and the current code is insufficient — or it is
  a rule CLAUDE.md states explicitly.
- 100: certain; the evidence directly confirms it.
Default LOW when uncertain. Your job is to REFUTE the finding, not to agree with it.`

const lawsBlock = (cfg.laws || []).length
  ? `A deterministic checker already confirmed these — do NOT re-report them, but they may hint
at intent:\n${cfg.laws.map(l => `- [${l.id}] ${l.file}:${l.line} ${l.summary}`).join('\n')}`
  : 'The deterministic checker found no law violations.'

function finderPrompt(d) {
  return `You are reviewing a nidus change: ${cfg.ref}.

Read the diff with: ${cfg.diffCmd}
Changed files: ${(cfg.changed || []).join(', ') || '(see the diff)'}

${d.prompt}

${HOUSE_RULES}

${lawsBlock}

${FALSE_POSITIVES}

Report every defect you can substantiate from the code, and do NOT pre-filter for
importance — an independent skeptic scores each finding afterwards and anything weak is
discarded automatically. Suppressing a real defect because it "might not be worth raising"
is the failure mode here; that is the skeptic's call to make, not yours. Return an empty
list only if you genuinely found nothing.`
}

phase('Review')

// `only` narrows the lens set — used to re-run one lens against a known defect when
// tuning the prompts, instead of paying for all six.
const active = cfg.only && cfg.only.length ? DIMENSIONS.filter(d => cfg.only.includes(d.key)) : DIMENSIONS

const reviewed = await pipeline(
  active,
  d => agent(finderPrompt(d), {
    label: `review:${d.key}`,
    phase: 'Review',
    model: 'sonnet',
    schema: FINDINGS_SCHEMA,
  }).then(r => ({ key: d.key, findings: (r && r.findings) || [] })),
  // Each dimension's findings are verified as soon as that dimension lands, so a slow
  // finder never holds up verification of a fast one.
  (r) => parallel((r.findings || []).slice(0, 6).map(f => () =>
    agent(
      `You are the skeptic. A reviewer claims this defect in ${cfg.ref}:

FILE: ${f.file}:${f.line}
CLAIM: ${f.summary}
FAILS WHEN: ${f.failure_scenario}
EVIDENCE OFFERED: ${f.evidence}

Read the actual code (diff: ${cfg.diffCmd}) and try to REFUTE it. Consider: is it pre-existing?
Is it already handled elsewhere (a caller, a guard, a cfg gate, a test)? Did the reviewer
misread the control flow? Would a compiler or clippy already catch it?

${RUBRIC}`,
      { label: `verify:${f.file.split('/').pop()}`, phase: 'Verify', model: 'sonnet', schema: VERDICT_SCHEMA },
    ).then(v => ({ ...f, dimension: r.key, confidence: (v && v.confidence) || 0, verdict_reason: v && v.reason })),
  )),
)

// Lenses overlap, so one defect can arrive twice with different scores — without this
// the same bug lands in `confirmed` and in `rejected` at once.
const words = s => new Set(String(s || '').toLowerCase().match(/[a-z_]{4,}/g) || [])
function similar(a, b) {
  const A = words(a), B = words(b)
  if (!A.size || !B.size) return false
  const shared = [...A].filter(w => B.has(w)).length
  return shared / Math.min(A.size, B.size) >= 0.6
}

const all = []
for (const f of reviewed.flat().filter(Boolean)) {
  const dup = all.find(o => o.file === f.file && similar(o.summary, f.summary))
  if (!dup) { all.push({ ...f, dimensions: [f.dimension] }); continue }
  dup.dimensions.push(f.dimension)
  // Keep the best-argued version of a defect two lenses both found.
  if (f.confidence > dup.confidence) Object.assign(dup, f, { dimensions: dup.dimensions })
}

const confirmed = all.filter(f => f.confidence >= 80).sort((a, b) => b.confidence - a.confidence)

log(`${all.length} candidate finding(s), ${confirmed.length} survived verification at >=80 confidence`)

return {
  ref: cfg.ref,
  confirmed,
  rejected: all.filter(f => f.confidence < 80).map(f => ({ file: f.file, summary: f.summary, confidence: f.confidence, why: f.verdict_reason })),
  laws: cfg.laws || [],
}
