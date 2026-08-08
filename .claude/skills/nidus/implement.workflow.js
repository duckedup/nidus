export const meta = {
  name: 'nidus-implement',
  description: 'Implement each blueprint in an isolated worktree and return verified source patches for the main thread to merge',
  phases: [
    { title: 'Implement' },
  ],
}

// args (from the /nidus implement path):
// {
//   id: "nidus-oes",
//   scratchDir: "/abs/path",              // where agents write patches + verify logs
//   groups: [                              // ordered; group N starts after N-1 finishes
//     [ { dir, content, verify: ["just ci", ...] }, ... ],
//   ],
// }
//
// Unlike a codegen repo, a nidus worktree is a COMPLETE checkout: `just ci` really does
// build and test there. So an agent's own verification is authoritative, not advisory,
// and a failed lane is a real reason to retry.
const cfg = typeof args === 'string' ? JSON.parse(args) : (args || {})

const IMPL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dir', 'status', 'patch_file', 'verification', 'blockers', 'notes'],
  properties: {
    dir: { type: 'string' },
    status: { enum: ['success', 'partial', 'blocked'] },
    patch_file: { type: 'string', description: 'Absolute path to the written patch, or "" if no changes' },
    files_changed: { type: 'array', items: { type: 'string' } },
    verification: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['command', 'result'],
        properties: {
          command: { type: 'string' },
          result: { enum: ['pass', 'fail'] },
          log_excerpt: { type: 'string' },
        },
      },
    },
    blockers: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}

const slug = dir => dir.replace(/[^A-Za-z0-9]+/g, '_')

function implPrompt(spec, prior) {
  const patchFile = `${cfg.scratchDir}/${slug(spec.dir)}.patch`
  const logFile = `${cfg.scratchDir}/${slug(spec.dir)}.verify.log`
  return `You are implementing ONE blueprint for ${cfg.id} in an isolated git worktree.

BLUEPRINT (implement exactly this, nothing outside its scope):
${spec.content}

Rules:
- Follow the repo's CLAUDE.md. Comments cap at 3 lines. Errors are anyhow. On-disk encoding is
  little-endian, length-prefixed and CRC32-checked.
- Only touch files under this blueprint's scope (${spec.dir}). Another agent owns the rest.
- Do NOT commit, do NOT switch branches, do NOT push.
- This worktree is a complete checkout — your verification commands genuinely work here, so a
  failure is a real failure. Run each of ${JSON.stringify(spec.verify || [])} and tee it:
      <command> 2>&1 | tee -a ${logFile}
- Track work with bd, never with markdown checklists.
${prior ? `\nYOUR PRIOR ATTEMPT FAILED — fix exactly this, do not start over:\n${prior}\n` : ''}
When the blueprint is implemented and its lanes pass, write your patch:
    git add -A && git diff --cached > ${patchFile}
Then return the structured result with patch_file set to "${patchFile}" (or "" if you changed nothing).
Report status 'success' only when the blueprint is fully implemented AND its verification passed;
'partial' when implemented but a lane still fails; 'blocked' when you could not make the change.`
}

const MAX_ATTEMPTS = 3 // 1 initial + 2 retries

async function implementOne(spec) {
  let prior = null
  let last = null
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const r = await agent(implPrompt(spec, prior), {
      phase: 'Implement',
      model: 'sonnet',
      isolation: 'worktree',
      schema: IMPL_SCHEMA,
      label: attempt === 1 ? `impl:${spec.dir}` : `impl:${spec.dir} (retry ${attempt - 1})`,
    })
    last = r
    const failedLane = r && (r.verification || []).some(v => v.result === 'fail')
    const bad = !r || r.status === 'blocked' || failedLane
    if (!bad) return { spec, result: r, attempts: attempt }
    if (attempt >= MAX_ATTEMPTS) break
    prior = JSON.stringify({ status: r && r.status, blockers: r && r.blockers, notes: r && r.notes, verification: r && r.verification })
    log(`retry ${attempt} for ${spec.dir}`)
  }
  return { spec, result: last, attempts: MAX_ATTEMPTS, failed: true }
}

const groups = cfg.groups || []
const results = []
for (let g = 0; g < groups.length; g++) {
  log(`group ${g + 1}/${groups.length}: ${groups[g].length} blueprint(s)`)
  results.push(...await parallel(groups[g].map(spec => () => implementOne(spec))))
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
    attempts: r.attempts,
  })),
  no_change: ok.filter(r => !r.result.patch_file).map(r => r.result.dir),
  failures: failed.map(r => ({
    dir: r.spec.dir,
    status: r.result && r.result.status,
    blockers: (r.result && r.result.blockers) || [],
    notes: r.result && r.result.notes,
    verification: (r.result && r.result.verification) || [],
  })),
}
