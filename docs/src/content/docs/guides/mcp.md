---
title: MCP (agent memory)
description: nidus serve answers the Model Context Protocol at /mcp, so any MCP client can use a store as long-term memory — remember text, recall it by meaning, with no glue code.
---

`nidus serve` speaks the [Model Context Protocol](https://modelcontextprotocol.io)
at `/mcp`. Point an MCP client at it and the store becomes long-term memory for
an agent: it can `remember` text worth keeping and `recall` it later by meaning,
without you writing any integration code.

This is the [memory layer](/guides/remember-and-recall/) with a different front
door. Everything MCP exposes goes through the same store, the same embedder, and
the same locking as the [HTTP API](/reference/http-api/) — write over HTTP and
read over MCP, or the reverse, and you are talking to one store.

## Turning it on

The endpoint ships in the `mcp` feature, which is part of the `serve` umbrella —
so a binary built for the memory layer already has it:

```bash
cargo install nidus --features serve
```

It needs an embedder, because the useful tools take **text** and embed it for
you:

```bash
nidus serve --dir ./store --dim 1024 \
  --embed-provider voyage --embed-model voyage-3 \
  --token "$NIDUS_TOKEN"
```

Then register it with your client. For anything that reads the standard MCP
config shape:

```json
{
  "mcpServers": {
    "nidus": {
      "type": "http",
      "url": "http://127.0.0.1:7700/mcp",
      "headers": { "Authorization": "Bearer YOUR_TOKEN" }
    }
  }
}
```

`/mcp` sits behind the same bearer token as every other route, so if you started
the server with `--token` the client has to send it. Drop the `headers` block if
you did not.

## The tools

| Tool | What it does |
| --- | --- |
| `remember` | Store text in a collection. Embedded server-side; optionally summarized first. |
| `recall` | Search by meaning and get the closest entries back with scores. |
| `text_search` | Search by keyword (BM25) instead, for when exact wording matters. |
| `hybrid_search` | Both at once, fused — semantic intent plus a term that must appear. |
| `list_collections` | List the collections in the store. |
| `stats` | Dimension, distance metric, collections, and memory footprint. |

Every one of them takes **natural language, never vectors**. That is deliberate:
a model cannot write a 1024-float array as a tool argument, so the raw
vector-taking endpoints (`POST /search`, and the `vector` field on
`POST /hybrid-search`) have no MCP equivalent. `recall` and `hybrid_search`
embed your query text server-side to get there instead.

`remember` derives a content-based id when you do not pass one, which makes it
idempotent — remembering the same sentence twice replaces the entry rather than
accumulating near-duplicates that then compete for the same results. Pass an
explicit `id` when you want to update a specific memory later.

## What it does not do

The surface is deliberately small. There are no MCP resources, prompts,
subscriptions, or tasks: every nidus operation is a fast synchronous call, so
there is nothing to subscribe to and nothing long-running to hand back a task
handle for. Store maintenance (`compact`, `flush`, collection deletion) is not
exposed either — those are operator actions, and an agent should not be reaching
for them.

Authorization is the server's existing bearer token rather than MCP's OAuth
flows. For a store you run yourself, a token over loopback is the honest security
model; if you expose one on a real interface, `nidus serve` warns you when it has
no token set.

## Protocol version

nidus implements the `2026-07-28` revision, which is stateless: no
`initialize` handshake, no session ids, and method and tool names carried in the
`Mcp-Method` and `Mcp-Name` headers so a gateway can route without reading the
body. Statelessness is what lets `/mcp` sit behind an ordinary round-robin load
balancer alongside [multi-box deployments](/guides/multi-box/) — there is no
session to pin a client to an instance.

Older clients still work. The server negotiates down, and `server/discover`
reports every revision it supports:

```bash
curl -s localhost:7700/mcp \
  -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -H 'mcp-protocol-version: 2026-07-28' \
  -H 'mcp-method: server/discover' \
  -d '{"jsonrpc":"2.0","id":1,"method":"server/discover","params":{
        "_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28"}}}'
```
