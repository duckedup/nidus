---
title: Give an agent memory
description: Save something worth keeping, then find it again weeks later by meaning rather than filename. MCP means an agent reaches it with no glue code.
---

An agent's context resets every session; a project does not. Save something worth
keeping once, and find it again weeks later by meaning rather than by filename or
timestamp. Over MCP, an agent reaches the store directly, with no glue code to
maintain on your side.

## What you need

- [Remember & recall](/guides/remember-and-recall/): embed and store a piece of
  text, then recall the closest ones back by meaning.
- [MCP](/guides/mcp/): the tools an agent calls directly, no SDK and no server code
  of your own.
- [Automatic memory](/guides/automatic-memory/): the session-start/stop hook
  pattern that recalls and writes back without the model having to ask.

## Doing it

```sh
nidus remember notes "Users authenticate with a bearer token issued at login." \
  --dir ./store

nidus recall notes "how do users sign in?" --dir ./store
```

The same two operations are `remember`/`recall` over MCP, so an agent calls them
directly instead of shelling out.

## What to tune

- [Reinforcement](/guides/remember-and-recall/#reinforcement): an entry recalled
  often decays slower than one nobody asks for again.
- [Filters & metadata](/guides/filters/): scope memories to a project or a user,
  and combine that with recency-decay ranking so old notes fade rather than pile up.

## Where it stops

One writer holds the store at a time. nidus stores what you tell it to remember and
ranks it on recall; it does not decide on its own what is worth keeping.
