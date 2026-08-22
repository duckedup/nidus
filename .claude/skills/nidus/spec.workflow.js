export const meta = {
  name: 'nidus-spec',
  description: 'Research a nidus change from four fixed lenses and propose the directory partition the blueprints will follow',
  phases: [
    { title: 'Research' },
    { title: 'Partition' },
  ],
}

// args (from the /nidus spec path):
// { id: "nidus-oes" | null, ask: "<ticket text or description>" }
//
// Returns research only. The MAIN thread writes the blueprints from it — the gate the
// user approves should be Opus-authored, not a sonnet summary.
const cfg = typeof args === 'string' ? JSON.parse(args) : (args || {})

const RESEARCH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'files', 'findings', 'risks'],
  properties: {
    lens: { type: 'string' },
    files: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['path', 'role'],
        properties: {
          path: { type: 'string' },
          role: { type: 'string', description: 'What this file does and why it matters here' },
          symbols: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    findings: { type: 'array', items: { type: 'string' } },
    patterns: {
      type: 'array',
      description: 'Concrete code to mirror: path, line range, and what the pattern is',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['path', 'lines', 'what'],
        properties: { path: { type: 'string' }, lines: { type: 'string' }, what: { type: 'string' } },
      },
    },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const PARTITION_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['summary', 'groups', 'ordering', 'scope_questions', 'open_questions'],
  properties: {
    summary: { type: 'string' },
    groups: {
      type: 'array',
      description: 'Ordered groups; everything in one group can be implemented in parallel',
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
    ordering: { type: 'array', items: { type: 'string' }, description: 'Why the groups are ordered that way' },
    scope_questions: {
      type: 'array',
      description: 'Forks that change WHICH files exist, so they must be settled before any blueprint is written',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['question', 'forks', 'options'],
        properties: {
          question: { type: 'string', description: 'Asked of the developer, answerable without reading the code' },
          forks: { type: 'string', description: 'What changes in the unit list and file list depending on the answer' },
          options: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['label', 'implication'],
              properties: {
                label: { type: 'string', description: 'Four words or fewer' },
                implication: { type: 'string', description: 'What this answer adds to or drops from the change' },
              },
            },
          },
          recommendation: { type: 'string', description: 'Which option this partition would pick, and one line of why' },
        },
      },
    },
    open_questions: {
      type: 'array',
      description: 'Small reversible details that belong in a blueprint, NOT questions for the developer',
      items: { type: 'string' },
    },
  },
}

const CONTEXT = `nidus is an embeddable pure-Rust vector store: dense vectors plus typed metadata in one
on-disk directory, exact brute-force cosine search, with opt-in ANN, quantisation, an HTTP
server, an MCP surface, and three client SDKs. Read CLAUDE.md and SPEC.md first — SPEC.md §9
records which seams are deliberately deferred, so check there before proposing something new.`

const LENSES = [
  {
    key: 'modules',
    prompt: `Map the code this change must touch. Which modules own the behaviour today, what are the
seams, and where would a new concern go? nidus splits modules by concern into sibling files
(src/store/ is the worked example), so say which FILE each piece belongs in, not just which
directory. Note anything already feature-gated.`,
  },
  {
    key: 'tests',
    prompt: `Map how this area is tested and verified. Inline pure-logic tests, file-backed tests in
tests/, e2e modules under tests/e2e/ that drive the real binary, the cluster suite behind
services, Miri discipline (what must run under it, what may be #[cfg_attr(miri, ignore)]d),
and which \`just\` recipe is the real gate. Give concrete examples to copy.`,
  },
  {
    key: 'laws',
    prompt: `Find the repo laws and release surface this change will collide with: feature gating and
the pure-default-build thesis, the dependency budget, the version bump plus the README and
docs install snippets, the bot-stamped chart and SDK version files, and whether the docs site
or the SDKs need a matching change. Quote the CLAUDE.md lines that apply.`,
  },
  {
    key: 'prior-art',
    prompt: `Find prior art. Search the tracker for related issues, open and closed
(\`bd search "<terms>"\`, \`bd list --all\`; the closed history came across from GitHub, so
\`#186\` is \`nidus-186\`), read SPEC.md for the relevant section (especially §9's deferred seams), and read git history
(\`git log --oneline\`, then \`git show\`) for earlier attempts or decisions about this area.
Report what was already decided and why, so this change does not relitigate it.`,
  },
]

phase('Research')

const research = await parallel(LENSES.map(l => () => agent(
  `${CONTEXT}

THE ASK${cfg.id ? ` (${cfg.id})` : ''}:
${cfg.ask}

YOUR LENS — ${l.key}:
${l.prompt}

Read real files and quote real paths and line numbers. Do not propose an implementation and do
not edit anything; this is research that another agent will build a plan from.`,
  { label: `research:${l.key}`, phase: 'Research', model: 'sonnet', schema: RESEARCH_SCHEMA },
)))

const bundle = research.filter(Boolean)

phase('Partition')

const partition = await agent(
  `${CONTEXT}

THE ASK${cfg.id ? ` (${cfg.id})` : ''}:
${cfg.ask}

Four researchers reported:
${JSON.stringify(bundle, null, 2)}

Propose how to split the implementation into directory-scoped units that can be built in
parallel without touching each other's files. Rules:
- One unit owns one directory (or one file set); units in the same group MUST be file-disjoint.
- Put a unit in a later group only when it genuinely depends on an earlier one's code.
- Each unit's \`verify\` is the exact just recipes that cover it. Remember \`just ci\` does NOT
  compile src/cli, src/server or src/bin — those need \`just ci-cli\`; the MCP surface needs
  \`--features mcp\`; codec and kernel changes need \`just miri\`.
- Flag anything that must stay in ONE unit because splitting it would break the build.

Then split what you do not know into two piles, because they are consumed differently.
\`scope_questions\` are forks the developer must settle BEFORE any blueprint is written: each one
changes which units exist or which files they touch, and answering it wrong is expensive to walk
back. Which surfaces are in scope (core / HTTP / CLI / MCP / the three SDKs / docs), whether an
on-disk or wire format changes, whether an existing default moves, whether this supersedes or
sits beside something that already ships. Ask only what the ticket and the research do not
already answer, phrase each so it can be answered without reading the code, and give concrete
options with their consequences. If the ask is genuinely unambiguous, return an empty array
rather than inventing a question. Everything else — a name, a constant, an ordering that is
cheap to revise — is an \`open_question\`, and goes in a blueprint rather than to the developer.`,
  { label: 'partition', phase: 'Partition', schema: PARTITION_SCHEMA },
)

return { id: cfg.id, research: bundle, partition }
