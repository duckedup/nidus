export const meta = {
  name: 'nidus-implement',
  description: 'Implement each blueprint in an isolated worktree and return source patches for the main thread to merge and verify',
  phases: [
    { title: 'Implement' },
  ],
}

// args (from the /nidus implement path):
// {
//   id: "nidus-oes",
//   scratchDir: "/abs/path",              // where agents write their patches
//   groups: [                              // ordered; group N starts after N-1 finishes
//     [ { dir, content, path }, ... ],     // path: ABSOLUTE path to the blueprint file
//   ],
// }
//
// `content` is captured at launch for every group, so pass `path` as well: the agent reads
// it at unit start, which is what makes a mid-run blueprint edit reach later groups (#175).
//
// Workers do NOT build. Each owns one isolated slice and returns a patch; the merging
// thread runs the lanes once against the merged tree. N workers would otherwise mean N
// cold Rust builds to prove something that only holds after the merge anyway.
const cfg = typeof args === 'string' ? JSON.parse(args) : (args || {})

const IMPL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dir', 'status', 'patch_file', 'files_changed', 'blockers', 'notes'],
  properties: {
    dir: { type: 'string' },
    status: { enum: ['success', 'partial', 'blocked'] },
    patch_file: { type: 'string', description: 'Absolute path to the written patch, or "" if no changes' },
    files_changed: { type: 'array', items: { type: 'string' } },
    blockers: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}

// The group/index prefix is the real key: slug() alone maps src/store, src.store and
// src-store to one name, and two agents would then overwrite one patch in the shared
// scratch dir — invisible to git, since each is isolated in its own worktree.
const slug = dir => dir.replace(/[^A-Za-z0-9]+/g, '_')
const key = spec => `g${spec._g}-${spec._i}-${slug(spec.dir)}`

// A patch is cut with `git add -A`, so anything the agent touched outside its blueprint
// rides along. Surface that to the merging thread instead of letting it merge silently.
const outOfScope = spec => (files) =>
  (files || []).filter(f => f !== spec.dir && !f.startsWith(spec.dir.replace(/\/*$/, '/')))

function implPrompt(spec, prior, upstream = []) {
  const patchFile = `${cfg.scratchDir}/${key(spec)}.patch`
  const deps = upstream.length ? `
FIRST, lay down the earlier groups' work. Your worktree is cut from the branch commit and does
NOT contain it; your blueprint is in a later group precisely because it builds on this. Apply
it and commit it, so that your own patch below contains only your changes:
${upstream.map(f => `    git apply --whitespace=nowarn ${f}`).join('\n')}
    git add -A && git commit -q -m "upstream: prior groups"
That commit is local bookkeeping and is never pushed — it is what keeps your patch clean.
If a patch fails to apply, stop and report 'blocked' naming it. Do not proceed against a tree
missing the code you depend on, and do not re-implement it yourself.
` : ''
  // Read at unit start, not captured at launch, so a blueprint edited mid-run reaches the
  // agents that have not started yet. The path is absolute into the coordinating checkout:
  // BLUEPRINT-*.md is gitignored, so it is never in this agent's own worktree (#175).
  const authoritative = spec.path ? `
YOUR BLUEPRINT IS THE FILE AT ${spec.path}. Read it FIRST, with an absolute path — it is NOT in
your worktree (gitignored) and it is authoritative. It may have been edited since this run
started, in which case it supersedes the copy below. Use the copy only if the file is missing.
` : ''
  return `You are implementing ONE blueprint for ${cfg.id} in an isolated git worktree.
${deps}${authoritative}

BLUEPRINT (implement exactly this, nothing outside its scope):
${spec.content}

Rules:
- Follow the repo's CLAUDE.md. Comments cap at 3 lines. Errors are anyhow. On-disk encoding is
  little-endian, length-prefixed and CRC32-checked.
- Only touch files under this blueprint's scope (${spec.dir}). Another agent owns the rest.
- Do NOT commit your own work, do NOT switch branches, do NOT push. (The upstream commit
  above, if you were given one, is the sole exception and is bookkeeping only.)
- **Do NOT build, test, lint or run any lane.** You own one slice of a larger change and your
  worktree does not contain the others, so a green run here would prove nothing and a red one
  is probably someone else's missing half. The thread that merges every patch runs the lanes
  once, against the merged tree, where the answer is real. Write the code and stop.
- Read whatever you need to get the slice right, and say so in notes if the blueprint looks
  wrong or depends on a sibling slice. Reporting a suspicion costs nothing; guessing costs a
  merge conflict.
- Track work in beads (\`bd\`), never with markdown checklists.
${prior ? `\nYOUR PRIOR ATTEMPT FAILED — fix exactly this, do not start over:\n${prior}\n` : ''}
When the blueprint is implemented and its lanes pass, write your patch:
    git add -A && git diff --cached > ${patchFile}
Then return the structured result with patch_file set to "${patchFile}" (or "" if you changed nothing).
Set files_changed to exactly what that patch contains — read it back with
    git diff --cached --name-only
and copy the list verbatim. That patch is cut with \`git add -A\`, so it carries everything in
this worktree, not only ${spec.dir}. If any file in it sits outside ${spec.dir}, say so in notes
and why it was unavoidable. Do not hide it and do not hand-edit the patch to remove it.
Report status 'success' when the blueprint is fully implemented, 'partial' when some of it is
implemented and you have said in notes exactly what is missing, 'blocked' when you could not
make the change at all.`
}

const MAX_ATTEMPTS = 3 // 1 initial + 2 retries

async function implementOne(spec, upstream) {
  let prior = null
  let last = null
  let lastBlockers = null
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const r = await agent(implPrompt(spec, prior, upstream), {
      phase: 'Implement',
      model: 'sonnet',
      isolation: 'worktree',
      schema: IMPL_SCHEMA,
      label: attempt === 1 ? `impl:${spec.dir}` : `impl:${spec.dir} (retry ${attempt - 1})`,
    })
    last = r
    // No lane to fail on any more, so a retry is only worth it when the worker produced
    // nothing usable. 'partial' comes back with its gap named and the merger decides.
    const bad = !r || r.status === 'blocked' || (r.status === 'success' && !r.patch_file && !(r.files_changed || []).length)
    if (!bad) return { spec, result: r, attempts: attempt }
    if (attempt >= MAX_ATTEMPTS) break
    // An identical failure twice means the retry is adding no information — usually the
    // tree cannot satisfy the blueprint at all. Stop rather than spend the budget.
    const signature = JSON.stringify([r && r.status, r && r.blockers])
    if (signature === lastBlockers) { log(`${spec.dir}: identical failure twice, not retrying`); break }
    lastBlockers = signature
    prior = JSON.stringify({ status: r && r.status, blockers: r && r.blockers, notes: r && r.notes })
    log(`retry ${attempt} for ${spec.dir}`)
  }
  return { spec, result: last, attempts: MAX_ATTEMPTS, failed: true }
}

const groups = cfg.groups || []
const results = []
// Groups sequence state, not just timing: a later group is handed every earlier patch,
// because that dependency is the only reason it is a later group.
const landed = []
for (let g = 0; g < groups.length; g++) {
  log(`group ${g + 1}/${groups.length}: ${groups[g].length} blueprint(s)${landed.length ? `, on ${landed.length} upstream patch(es)` : ''}`)
  const specs = groups[g].map((spec, i) => ({ ...spec, _g: g, _i: i }))
  const upstream = [...landed]
  const done = await parallel(specs.map(spec => () => implementOne(spec, upstream)))
  results.push(...done)
  for (const r of done) if (r && !r.failed && r.result && r.result.patch_file) landed.push(r.result.patch_file)
}

const ok = results.filter(r => r && !r.failed && r.result)
const failed = results.filter(r => !r || r.failed)

return {
  id: cfg.id,
  // The main thread applies these itself — no consolidation agent, so one Opus context
  // has seen every patch before it reviews the merged result.
  patches: ok.filter(r => r.result.patch_file).map(r => ({
    dir: r.result.dir,
    patch_file: r.result.patch_file,
    files_changed: r.result.files_changed || [],
    // Self-reported, so treat it as a pointer for the merging thread to verify against
    // the patch itself, never as proof the rest of the patch is in scope.
    out_of_scope: outOfScope(r.spec)(r.result.files_changed),
    attempts: r.attempts,
  })),
  no_change: ok.filter(r => !r.result.patch_file).map(r => r.result.dir),
  failures: failed.map(r => ({
    dir: r.spec.dir,
    status: r.result && r.result.status,
    blockers: (r.result && r.result.blockers) || [],
    notes: r.result && r.result.notes,
  })),
}
