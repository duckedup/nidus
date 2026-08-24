// PreToolUse gate: an unbounded read of a doc `bin/spec` can address costs ~44k tokens to
// use one section of. Instructions to use `spec` are advisory; this is not.

import { readFileSync } from 'node:fs'
import { resolve, relative } from 'node:path'

import { specRead } from '../skills/nidus/lib/guards.mjs'

const repo = resolve(new URL('../..', import.meta.url).pathname)

let input
try { input = JSON.parse(readFileSync(0, 'utf8')) } catch { process.exit(0) }

const args = input?.tool_input ?? {}
const message = args.file_path
  ? specRead({ rel: relative(repo, resolve(args.file_path)), offset: args.offset, limit: args.limit })
  : null

if (!message) process.exit(0)
console.error(message)
process.exit(2)
