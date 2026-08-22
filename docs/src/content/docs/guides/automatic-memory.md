---
title: Automatic memory
description: Wire a nidus store into an agent so memories are written and recalled without the model having to remember to ask. A session-start hook recalls context; a stop hook writes back what was learned.
---

Every tool in the [MCP surface](/guides/mcp/) still only fires when the model
*chooses* to call it. A memory the model forgets to write is the exact failure a
memory store exists to prevent.

What closes that gap is host-side wiring: a hook at session start that recalls
relevant context and puts it in front of the model, and a hook at session end that
writes back what was learned. None of it is nidus code (it is configuration in your
agent host), but nidus is shaped to make it a few lines rather than a project.

This page walks from an empty store to working automatic capture, using
[Claude Code](https://claude.ai/code) hooks as the worked example. The shape
transfers to any host with equivalent lifecycle events.

## Start a server, not a stdio session

The obvious first move (register `nidus mcp` over stdio and add hooks beside it)
**does not work**, and it is worth knowing why before you build on it.

A stdio session holds the store's writer lock for its whole lifetime. A hook that
tried to write while that session was live would find the store locked. There is also
no `nidus remember` subcommand for a hook to shell out to: the memory surface is
reachable through MCP tools or the HTTP routes, and nothing else.

So run one `nidus serve`, and let both the model and the hooks talk to it:

```bash
nidus serve --dir ~/.nidus --dim 1024 \
  --addr 127.0.0.1:7700 \
  --embed-provider voyage --embed-model voyage-4
```

Point the agent's MCP client at that server's `/mcp` endpoint rather than spawning
its own process:

```bash
claude mcp add --transport http nidus http://127.0.0.1:7700/mcp
```

Now the model has `remember`/`recall` as tools, and your hooks have
`POST /collections/{name}/remember` and `POST /collections/{name}/recall` against the
same store, the same embedder, and the same lock.

:::note
The server needs the `memory` routes, which ship in the `serve` umbrella:
`cargo install nidus --features serve`. Without them `/remember` and `/recall` are
absent and only the raw vector endpoints answer.
:::

## You do not have to create anything

There is no setup call. The first `remember` into a collection creates it **and**
declares a default full-text schema over `nidus.text`, so a client that only ever
speaks to the server goes from an empty directory to a working
[`hybrid_search`](/guides/search/) without a CLI invocation or a provisioning
request.

That is what makes the hooks below safe to run on a machine where the store does not
exist yet: the first session creates it.

## The conventions that make later filtering work

Every write stamps a reserved `nidus.*` namespace. These are ordinary attributes
(filterable, projectable, sortable); they simply arrive without you setting them:

| Attr | Type | When |
|---|---|---|
| `nidus.text` | `Str` | **always**, both modes: the raw text exactly as given. This is the field the default full-text schema indexes, so it is what you `text_search` against. |
| `nidus.created_at` | `DateTime` | always; UTC epoch milliseconds. Survives a re-`remember` of the same id. |
| `nidus.updated_at` | `DateTime` | always; moves on every write. |
| `nidus.expires_at` | `DateTime` | only when `ttl_seconds` was passed. |
| `nidus.summary` | `Str` | summarize mode only: the generated summary that was embedded. |

Everything else in `attrs` is yours, and **the conventions you pick at write time are
the only ones you can filter on later**. A memory written with no dimensions is a
memory you can only find by meaning. Stamp what you will want to slice by:

```json
{
  "project":  "nidus",
  "repo":     "duckedup/nidus",
  "branch":   "main",
  "kind":     "decision",
  "session":  "0f2c…"
}
```

`kind` is the one worth thinking about hardest. A store where everything is `note`
recalls badly, because a decision, a gotcha and a preference all compete on raw
cosine. Splitting them lets a recall ask for the kind it actually needs.

Use collections for hard boundaries you would never want mixed (one per project, say)
and attrs for everything else: a filter is cheap, and all collections share one
embedding space, so a cross-collection search is one ranking rather than a merge.

## The session-start hook: recall before the model asks

A `SessionStart` hook's stdout is injected into the model's context. So the hook
recalls against the current working directory and prints what it finds:

```bash
#!/usr/bin/env bash
# ~/.claude/hooks/nidus-recall.sh
set -euo pipefail

NIDUS=${NIDUS_URL:-http://127.0.0.1:7700}
PROJECT=$(basename "$PWD")

hits=$(curl -sS --max-time 5 \
  "$NIDUS/collections/memories/recall" \
  -H 'content-type: application/json' \
  -d "$(jq -nc --arg q "context for working on $PROJECT" --arg p "$PROJECT" '{
        query: $q,
        top_k: 8,
        min_score: 0.35,
        filter: [ { Eq: ["project", { Str: $p }] } ]
      }')" 2>/dev/null) || exit 0

echo "$hits" | jq -r '
  if length == 0 then empty else
  "## Remembered context\n",
  (.[] | "- [\(.attrs["kind"].Str // "note")] \(.attrs["nidus.text"].Str)")
  end' || exit 0
```

Note the response is a **bare JSON array**, not an object: `/recall` returns
`[{ "collection", "id", "score", "attrs" }, …]` with no `hits` wrapper, the same shape
as `/search` and `/text-search`. Indexing it as `.hits` is a jq type error, which under
`set -euo pipefail` kills the hook on every run.

Two deliberate choices. It **fails open** (note the `|| exit 0` on *both* the curl and
the jq) because a memory store being down should degrade a session, never block one.
And it sets a `min_score` floor, since recall always returns *something*; without a
floor a fresh store pours unrelated text into every session.

Register it:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume",
        "hooks": [
          { "type": "command", "command": "~/.claude/hooks/nidus-recall.sh" }
        ]
      }
    ]
  }
}
```

## The stop hook: write back what was learned

A `Stop` hook receives JSON on stdin including `transcript_path`. The honest version
of this hook does not try to summarize the transcript in bash. It stores a durable
fact you extract deliberately:

```bash
#!/usr/bin/env bash
# ~/.claude/hooks/nidus-remember.sh
set -euo pipefail

NIDUS=${NIDUS_URL:-http://127.0.0.1:7700}
input=$(cat)
# `|| exit 0` on every jq: under `set -e` a malformed payload would otherwise kill the
# hook here, before the guard below that exists to handle exactly that.
transcript=$(jq -r '.transcript_path // empty' <<<"$input") || exit 0
session=$(jq -r '.session_id // "unknown"' <<<"$input") || exit 0
[ -n "$transcript" ] && [ -f "$transcript" ] || exit 0

# Whatever you choose to persist. Keep it small and factual.
text=$(tail -n 200 "$transcript" | jq -rs 'map(select(.type=="assistant")) | last.message.content[0].text // empty' 2>/dev/null) || exit 0
[ -n "$text" ] || exit 0

curl -sS --max-time 5 -o /dev/null \
  "$NIDUS/collections/memories/remember" \
  -H 'content-type: application/json' \
  -d "$(jq -nc --arg id "session-$session" --arg t "$text" --arg p "$(basename "$PWD")" '{
        id: $id,
        text: $t,
        mode: "raw",
        attrs: { project: { Str: $p }, kind: { Str: "session" } },
        ttl_seconds: 7776000,
        dedupe_threshold: 0.95
      }')" || true
```

`mode: "raw"` embeds the text as given. `"summarize"` summarizes first and embeds the
summary, which is better for a long transcript, but it **requires the server to have been
started with `--summarize-provider`** (anthropic or openai, plus that provider's key).
The `nidus serve` command above configures only an embedder, so a `"summarize"` write
against it fails with *"nidus serve was started without a summarizer"*. Add the flag
before switching the mode.

The last two arguments are what keep the store from rotting:

- **`ttl_seconds`**: here 90 days. An entry past its TTL stops surfacing in `recall`,
  `text_search`, `hybrid_search`, `browse` and `get`.
- **`dedupe_threshold`**: at or above this cosine similarity to an existing entry, the
  write **updates that entry in place** instead of inserting a competitor. This is the
  single most valuable setting for automatic capture, which otherwise stores the same
  fact under a new id every session until a top-10 is five copies of one thing. Attrs
  are merged rather than replaced, and the matched entry's `nidus.created_at` carries
  forward, so an entry re-learned ten times keeps its original age.

The response tells you which happened (`{"ok":…,"upserted":…,"id":…,"deduped":true}`),
and `id` is the record actually written, which on a dedupe match is **not** the id you
sent. Log it if you care which entry moved.

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [{ "type": "command", "command": "~/.claude/hooks/nidus-remember.sh" }] }
    ]
  }
}
```

## A worked recall

With a few sessions captured, ask the store directly what the hook would inject:

```bash
curl -sS http://127.0.0.1:7700/collections/memories/recall \
  -H 'content-type: application/json' \
  -d '{
    "query": "why does the release workflow stamp versions instead of hand-editing them?",
    "top_k": 3,
    "min_score": 0.4,
    "filter": [
      { "Eq": ["project", { "Str": "nidus" }] },
      { "Eq": ["kind",    { "Str": "decision" }] }
    ]
  }' | jq '.[] | {score, id, text: .attrs["nidus.text"].Str}'
```

```json
{
  "score": 0.71,
  "id": "session-0f2c",
  "text": "Chart.yaml version and appVersion are stamped from Cargo.toml at release time because a CI assertion that they agree fired on essentially every PR."
}
```

A `filter` is a JSON **array** of predicates, AND-combined, not an object. Nest
`All`/`Any`/`Not` *inside* that list when you need other boolean shapes.

The filter is doing real work here: `kind = decision` is why a question about *why*
returns a decision rather than the three times someone edited that file. That is the
payoff for stamping conventions at write time.

To search the literal words instead of the meaning, `text_search` against
`nidus.text`, the field the auto-declared schema indexes:

```bash
curl -sS http://127.0.0.1:7700/text-search \
  -H 'content-type: application/json' \
  -d '{"scope":["memories"],"field":"nidus.text","query":"appVersion","top_k":5}'
```

## Two limits worth knowing

**A TTL hides an entry; it does not reclaim the row.** Expiry is evaluated at read
time. nidus runs no background threads, so nothing sweeps expired entries on its own
and the bytes stay on disk until you reclaim them deliberately. One call finds every
lapsed entry, deletes it, and compacts the store to reclaim the rows:

```bash
curl -sS -X POST http://127.0.0.1:7700/compact \
  -H 'content-type: application/json' \
  -d '{"expired": true}'
```

Reclaim through the **running server**, not the CLI, when one is running: `nidus
compact` opens the store read-write and would block on the writer lock `nidus serve`
already holds, for exactly the reason the stdio transport does. `nidus compact
--expired` is the same operation for a store with no server running (and still blocks
on the writer lock if one turns out to be running after all).

**TTL is enforced on every memory read, not the raw store routes.** `recall`,
`text_search`, `hybrid_search`, `browse` and `get` over MCP filter expired entries
for you, and so does `POST /recall`: the examples above need no extra predicate. The
plain `/search` and `/list` routes are raw store access and do not, so a hook querying
those two directly will see expired memories unless it adds the `nidus.expires_at`
predicate itself. This is the exact predicate the memory reads AND into every query:

```json
[ { "Not": { "Le": ["nidus.expires_at", { "DateTime": 1765200000000 }] } } ]
```

:::caution
Write it as `Not(Le(…))`, not as `Gt(…)`. Every predicate requires the key to be
**present**, so a bare `Gt` on `nidus.expires_at` is false for an entry that never got
a TTL, silently hiding every never-expiring memory you have. `Not(Le(…))` is true both
when the key is absent and when it is in the future, which is the behaviour you want.

This is read-time filtering only, and it deliberately reads the opposite of the sweep
above. The sweep that reclaims rows (`{"expired": true}` / `--expired`) matches on the
**un-negated** `Le`: it wants exactly the entries that have a TTL and have passed it,
never touching one with no `expires_at` at all. A read filter wants the reverse: keep
everything except those same lapsed entries, including the ones with no TTL. Same
field, opposite predicate, because the two are answering opposite questions.
:::

## Where to next

- [MCP (agent memory)](/guides/mcp/): the tool surface the model sees, and the stdio
  transport for the single-client case.
- [Remember & recall](/guides/remember-and-recall/): the same layer from Rust, and
  what an embedder needs from you. Note the attr table above describes what the
  **server** stamps; the in-process Rust `Memory::remember` does not currently write
  `nidus.text`, so `text_search` against it needs you to set that attr yourself.
- [Search & filters](/guides/search/): the filter grammar the hooks above use.
