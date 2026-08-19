export const meta = {
  name: 'nidus-review',
  description: 'Review a nidus change across repo-specific dimensions, then adversarially verify every finding',
  phases: [
    { title: 'Review' },
    { title: 'Verify' },
    { title: 'Criteria' },
  ],
}

// args (from the /nidus review path):
// {
//   ref: "PR #73" | "austin/foo vs main" | "working tree",  // human label
//   diffCmd: "gh pr diff 73" | "git diff main...HEAD",       // how an agent reads the diff
//   changed: ["src/store/read.rs", ...],
//   laws: [ ...nidus-check laws findings... ],               // deterministic, already true
//   issues: ["nidus-8ki", ...],   // the ticket(s) this change claims to finish, if any
//   effort: "medium" | "high",    // high = 3 diverse skeptics per finding, majority rules
//   only: ["durability", ...],    // narrow the lens set
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

const issues = (cfg.issues || []).filter(Boolean)
const issueList = issues.join(', ')

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
    key: 'scope',
    needs: 'issues',
    prompt: `Review this change against the TICKET it claims to finish. Read the ticket yourself:
\`bd show <id>\` and \`bd comments <id>\` for each of ${issueList}.

Report three kinds of drift, and nothing else:
- **Unmet**: an acceptance criterion, or an explicit requirement in the description, that the
  diff does not deliver. Quote the criterion and name what is missing.
- **Undeclared**: behaviour the diff adds that the ticket did not ask for — a new flag, a
  changed default, a second fix riding along. Say why it is not incidental to the fix.
- **Out of scope**: a hunk unrelated to the ticket that survived into this diff (a stray
  refactor, a debug line, an unrelated file).
Silence on all three is the right answer for a change that does what it said it would.`,
  },
  {
    key: 'seams',
    prompt: `Review the SEAMS this change crosses. nidus is implemented by per-directory agents that
each saw one directory, so a contract that changed on one side of a boundary and not the
other is the defect none of them could see, and \`just ci\` compiles only the pure library.

For every function, type, field, meta key, on-disk field, or invariant the diff changed,
find its callers OUTSIDE the changed directory and check each still holds:
- \`src/cli/\`, \`src/server/\`, \`src/server/mcp/\` and \`src/bin/\` (built only under features:
  \`just ci\` does NOT compile them, so a stale caller here is invisible to the core lane),
- the SDKs under \`sdks/\` and the docs under \`docs/\` that describe the old behaviour,
- \`benchmarks/\` and \`tests/\` (a separate crate each),
- the reverse direction: a library change that leaves \`src/server/dto.rs\` mirroring a shape
  that no longer exists.
Grep for the identifier rather than assuming; name the caller you found and what breaks.`,
  },
  {
    key: 'security',
    prompt: `Review this change for SECURITY defects reachable from untrusted input. nidus's untrusted
edges are the HTTP surface, the MCP tool surface, and the bytes on disk or in a backend:
- authentication and the body/concurrency limits: does anything added bypass the middleware
  stack (a route mounted outside it, an MCP tool routed around \`run_read\`/\`run_write\`)?
- a caller-supplied glob, regex, or fuzzy term that costs superlinear time (§7.4/§7.5) —
  algorithmic denial of service from one request;
- credentials or tokens reaching a log line, an error message, a metric label, or a
  \`diag!\` field (S3/GCS/redis URLs carry them in userinfo and query strings);
- path or URI handling that escapes the store prefix (\`src/server/mcp/uri.rs\`, backup and
  restore paths, a collection name used as a path component);
- decoding attacker-influenced bytes: a length prefix trusted before it is bounded, an
  allocation sized from input ahead of the \`max_vector_bytes\` guard.
A finding must name the request or file that triggers it. Not a posture review: no
"consider adding rate limiting" without a concrete reachable defect.`,
  },
  {
    key: 'test-efficacy',
    prompt: `Judge the TESTS this change adds or edits, against the law that every behaviour claim is
backed by a test that would fail without the change (CLAUDE.md, "Testing"):
- **Vacuous**: would this test pass with the fix reverted? Walk the assertion against the
  OLD code and say so explicitly. A bug fix whose test cannot fail is the defect.
- **Weaker than it claims**: asserts a status code but not the body, asserts \`is_ok()\`
  where the point was the value, or asserts a substring so short anything matches.
- **Wrong lane**: a claim only a real binary can prove (CLI flag wiring, sockets,
  cross-process locking, restart) tested in-process; or a pure-logic test parked in
  \`tests/\` where Miri never sees it.
- **Flaky by construction**: \`cargo\` runs lib tests as parallel threads in ONE process, so
  a test touching a \`static\`, an env var, the process CWD, a fixed port, or a shared temp
  path can fail when a sibling runs beside it.
- **Miri**: a new \`#[cfg_attr(miri, ignore)]\` whose stated reason is not one of the three
  sanctioned ones (unimplemented syscall, runtime cost, float ULP), or a pure-logic test
  ignored for no reason at all.`,
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
// tuning the prompts, instead of paying for all of them. A lens declaring `needs: 'issues'`
// has nothing to read without a ticket, so it drops out rather than guessing the intent.
const active = (cfg.only && cfg.only.length ? DIMENSIONS.filter(d => cfg.only.includes(d.key)) : DIMENSIONS)
  .filter(d => d.needs !== 'issues' || issues.length)

// One skeptic reads a claim one way. At high effort three read it from angles that fail
// differently — misread control flow, an existing guard, and "does it actually reproduce" —
// and the median score decides, so a single generous verifier cannot carry a weak finding.
const SKEPTIC_ANGLES = [
  `Attack the READING: did the reviewer misread the control flow, the types, the feature
gates, or which branch runs? Re-derive the behaviour from the code yourself.`,
  `Attack the NOVELTY: is it pre-existing on lines this change did not touch, or already
handled elsewhere — a caller's guard, a validate() pass, a cfg gate, an existing test,
or something clippy/the compiler rejects outright?`,
  `Attack the REPRODUCTION: construct the concrete input, state, or interleaving that
triggers it and follow it through the real code. If you cannot construct one, say so —
an unreproducible defect scores low however plausible the story is.`,
]

// The single skeptic asks all three questions at once rather than only the first, or the
// default review would stop asking "is this already handled?" — the most common refutation.
const ONE_SKEPTIC = `Consider: is it pre-existing on lines this change did not touch? Is it already
handled elsewhere (a caller, a guard, a cfg gate, a test)? Did the reviewer misread the
control flow? Would a compiler or clippy already catch it? Can you construct the concrete
input or interleaving that triggers it?`

const deep = ['high', 'max'].includes(cfg.effort)
const perLens = deep ? 8 : 6
const angles = deep ? SKEPTIC_ANGLES : [ONE_SKEPTIC]

const median = xs => {
  const v = [...xs].sort((a, b) => a - b)
  if (!v.length) return 0
  const m = v.length >> 1
  return v.length % 2 ? v[m] : Math.round((v[m - 1] + v[m]) / 2)
}

function skepticPrompt(f, angle) {
  return `You are the skeptic. A reviewer claims this defect in ${cfg.ref}:

FILE: ${f.file}:${f.line}
CLAIM: ${f.summary}
FAILS WHEN: ${f.failure_scenario}
EVIDENCE OFFERED: ${f.evidence}

Read the actual code (diff: ${cfg.diffCmd}) and try to REFUTE it.

${angle}

${RUBRIC}`
}

const reviewed = await pipeline(
  active,
  d => agent(finderPrompt(d), {
    label: `review:${d.key}`,
    phase: 'Review',
    model: 'sonnet',
    schema: FINDINGS_SCHEMA,
  }).then(r => ({ key: d.key, findings: (r && r.findings) || [] })),
  // Each dimension's findings are verified as soon as that dimension lands, so a slow
  // finder never holds up verification of a fast one. A lens over the cap says what it
  // dropped: a silent truncation reads as "that lens found nothing more".
  (r) => {
    const found = r.findings || []
    if (found.length > perLens) log(`${r.key}: verifying ${perLens} of ${found.length} findings, ${found.length - perLens} dropped unverified`)
    return parallel(found.slice(0, perLens).map(f => () =>
    parallel(angles.map((angle, i) => () =>
      agent(skepticPrompt(f, angle), {
        label: `verify:${f.file.split('/').pop()}${angles.length > 1 ? `:${i + 1}` : ''}`,
        phase: 'Verify',
        model: 'sonnet',
        schema: VERDICT_SCHEMA,
      }),
    )).then(vs => {
      const votes = vs.filter(Boolean)
      return {
        ...f,
        dimension: r.key,
        confidence: median(votes.map(v => v.confidence || 0)),
        verdict_reason: votes.map(v => v.reason).filter(Boolean).join(' | '),
      }
    }),
    ))
  },
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

// The lenses above all read code. This one RUNS it: whatever wrote the change is the worst
// judge of whether it meets the ticket, and a criterion nobody demonstrated is not done.
const CRITERIA_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['issue', 'criteria'],
  properties: {
    issue: { type: 'string' },
    criteria: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['criterion', 'demonstrated', 'how'],
        properties: {
          criterion: { type: 'string', description: 'Quoted from the ticket' },
          demonstrated: { type: 'boolean' },
          how: { type: 'string', description: 'The command actually run, or why it could not be' },
          evidence: { type: 'string', description: 'The output that shows it, trimmed' },
        },
      },
    },
  },
}

phase('Criteria')

const criteria = issues.length
  ? (await parallel(issues.map(id => () => agent(
      `Verify the change in ${cfg.ref} against ticket ${id}'s acceptance criteria. You did not write
this code; do not take its word for anything.

1. Read the ticket: \`bd show ${id}\`, and \`bd comments ${id}\`.
2. Read what actually landed: ${cfg.diffCmd}.
3. Take the acceptance criteria one at a time and DEMONSTRATE each by running something —
   the test that covers it (\`cargo test --features <needed> <filter>\`), the real binary
   (\`cargo run --features cli -- …\`), or \`just test-e2e <filter>\`. Quote the criterion,
   give the command you ran, and paste the few lines of output that show the result.
   Pay particular attention to the criteria about what must NOT happen — a refusal, an
   error, a flag that must not be honoured. Those are the ones a happy-path test misses.
4. A criterion you cannot demonstrate is \`demonstrated: false\` with the reason. Do not
   mark one demonstrated because the code looks like it would work, and do not soften a
   criterion to fit what shipped.

Build with the features the code needs (\`--features cli\`, \`memory\`, \`mcp\`) — \`just ci\`
compiles the pure library only, so a criterion about the binary needs its own feature flag.`,
      { label: `criteria:${id}`, phase: 'Criteria', schema: CRITERIA_SCHEMA },
    ))))
      .filter(Boolean)
  : []

const unmet = criteria.flatMap(c => (c.criteria || []).filter(x => !x.demonstrated).map(x => ({ issue: c.issue, ...x })))
if (issues.length) {
  log(`${criteria.flatMap(c => c.criteria || []).length} acceptance criterion/criteria checked, ${unmet.length} not demonstrated`)
}

return {
  ref: cfg.ref,
  confirmed,
  rejected: all.filter(f => f.confidence < 80).map(f => ({ file: f.file, summary: f.summary, confidence: f.confidence, why: f.verdict_reason })),
  laws: cfg.laws || [],
  criteria,
  unmet,
}
