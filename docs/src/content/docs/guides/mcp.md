---
title: MCP (agent memory)
description: "nidus speaks the Model Context Protocol over stdio or at nidus serve's /mcp, so any MCP client can use a store as long-term memory: remember text, recall it by meaning, with no glue code."
---

nidus speaks the [Model Context Protocol](https://modelcontextprotocol.io) over
two transports: standalone over **stdio** (`nidus mcp`), for a client that spawns
its own server process, or nested inside `nidus serve`'s HTTP stack at **`/mcp`**,
for a client that talks to a long-lived server. Either way, an MCP client sees the
store as long-term memory: it can `remember` text worth keeping and `recall` it
later by meaning, without you writing any integration code.

This is the [memory layer](/guides/remember-and-recall/) with a different front
door. Everything MCP exposes goes through the same store, the same embedder, and
the same locking that the [HTTP API](/reference/http-api/) uses. Write over
HTTP and read over MCP, or the reverse, and you are talking to one store.

## Over stdio

The `mcp` subcommand speaks MCP on stdin/stdout: no address, no token, nothing
to bind. This is the shape most local clients expect:

```bash
claude mcp add nidus -- nidus mcp --dir ~/.nidus --dim 1024 \
  --embed-provider voyage --embed-model voyage-4
```

`--dim` is optional here as long as `~/.nidus` does not exist yet: with an
embedder configured, nidus reads the dimension from it instead. Point the same
command at a store that already exists and the on-disk header wins regardless,
so a `--dim` that disagrees with it is still a hard error.

A stdio session opens the store and **holds the writer lock for its lifetime**.
There is no listener standing by to keep answering while a second client waits,
so another `nidus mcp` (or `nidus serve`) pointed at the same directory fails
immediately, naming the lock. Run one stdio session per store, or use `nidus
serve` below when several clients need to share one.

Pass `--read-only` to open without taking that lock: any number of read-only
sessions can run alongside a writer. `recall`, `text_search`, `hybrid_search`,
`get`, `browse`, `related`, and `suggest` all work; `remember` and `forget`
fail, since they write.

## Over HTTP (`nidus serve`)

The endpoint ships in the `mcp` feature, which is part of the `serve` umbrella,
so a binary built for the memory layer already has it:

```bash
cargo install nidus --features serve
```

It needs an embedder, because the useful tools take **text** and embed it for
you:

```bash
nidus serve --dir ./store --dim 1024 \
  --embed-provider voyage --embed-model voyage-4 \
  --token "$NIDUS_TOKEN"
```

As with `nidus mcp`, `--dim` here is only needed if `./store` does not already
exist and no embedder is configured; with `--embed-provider` set, a not-yet-created
store infers its dimension from the embedder instead.

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
you did not. Unlike stdio, several clients can share one `nidus serve` process.
There is one writer lock either way, but the server holds it and applies every
client's writes rather than each client racing for its own.

## The tools

| Tool | What it does |
| --- | --- |
| `remember` | Store text in a collection. Embedded server-side; optionally summarized first. |
| `recall` | Search by meaning and get the closest entries back with scores. |
| `text_search` | Search by keyword (BM25) instead, for when exact wording matters. |
| `hybrid_search` | Both at once, fused: semantic intent plus a term that must appear. |
| `list_collections` | List the collections in the store. |
| `stats` | Dimension, distance metric, collections, and memory footprint. |
| `forget` | Remove memories by id or by metadata filter. Irreversible. |
| `get` | Fetch one memory by id: its id and attrs, never its vector. |
| `browse` | List a collection's contents, bounded and optionally filtered, without a query. |
| `related` | Find entries like one you already have, using its own stored vector as the query. |
| `suggest` | Complete a partial word from a field's indexed vocabulary, ranked commonest-first. Takes a `filter`, and the words before the final token narrow it. |
| `code_search`* | Search a chunked code/docs corpus, grouped by file with each hit's matching symbols: name, kind, line span. Never returns source; read the file at the given lines instead. |

\* Needs the off-by-default `code` feature on top of `mcp`. See the
[code search guide](/guides/code/) for indexing a repo with `nidus code ingest` first.

Every one of them takes **natural language, never vectors**. That is deliberate:
a model cannot write a 1024-float array as a tool argument, so the raw
vector-taking endpoints (`POST /search`, and the `vector` field on
`POST /hybrid-search`) have no MCP equivalent. `recall` and `hybrid_search`
embed your query text server-side to get there instead.

`remember` derives a content-based id when you do not pass one, which makes it
idempotent: remembering the same sentence twice replaces the entry rather than
accumulating near-duplicates that then compete for the same results. Pass an
explicit `id` when you want to update a specific memory later. It also takes an
optional `attrs` object: structured metadata stored alongside the text (e.g.
`{"project": {"Str": "nidus"}, "tags": {"List": ["mcp"]}}`), each value tagged
by type, so a memory can later be found by filter as well as by meaning.

`recall`, `text_search`, and `hybrid_search` all take an optional `filter`: a
JSON array of metadata predicates, AND-combined, with `Any`/`Not` for OR and
negation. It narrows a search to the records that match before scoring, the
same filters `forget` and `browse` use to scope which records they touch.

### Reading a chunked corpus

A collection written by [`nidus ingest`](/guides/ingest/) holds chunks, not whole
documents, so a plain `recall` returns several fragments of the same file competing
for the window. `recall`, `text_search` and `related` all take an optional `rollup`
to fix that:

```json
{"collection": "docs", "query": "how does the writer lock work",
 "rollup": {"neighbours": 1}}
```

`per_parent` (default 1) is how many chunks survive per document; `neighbours` is
how many chunks are stitched either side of each survivor. Each result then carries
a `context` string: the passage the chunk came from, with the overlap between
adjacent chunks dropped rather than repeated. Read `context` when it is there and
`nidus.text` when it is not.

`rollup` only changes the payload. The results, their scores and their order are
exactly what the same call returns without it.

### Typeahead with `suggest`

`suggest` completes the last word of a `prefix` against a field's indexed
vocabulary, returning words with their document counts, not entries, ranked
commonest-first (the opposite of the ranking a prefix search itself uses to
rank documents). Call `text_search` with a completion once you have one to get
the entries themselves. Completions are real words: surface forms are indexed
alongside stems, so every keystroke of `running` completes to `running` rather
than to the stem `run`.

Send the whole phrase typed so far, not just the fragment: the words before the
final token narrow the completions to entries that also contain them, so
`"quick br"` completes against the entries that say "quick". `filter` narrows each
completion's count the same way, and a completion no matching entry carries is not
offered at all. Like every MCP tool, `suggest` takes one `collection`.

This is the only spelling MCP offers. The HTTP and CLI surfaces also take a raw
`expand` naming the chunk attrs directly, which matters for a corpus chunked by
something other than nidus; a model asking for one result per document does not
need it.

### Prefix matching (search as you type)

`text_search` and `hybrid_search` both take an optional `prefix` boolean. When set,
the final word of `query` is treated as a partial word and matched against any
indexed term starting with it, so a partial last word still finds documents while a
caller is typing; earlier words must still match exactly. It is off by default:

```json
{"collection": "docs", "field": "title", "query": "quick br", "prefix": true}
```

The match expands to at most 256 terms; past that it keeps the commonest
completions rather than failing the call. Both tools return matching documents, not a
list of candidate words to complete with.

### No setup step

`remember` provisions what it needs on first write: it creates the collection if
it is missing and declares a full-text schema over the stored text, so
`text_search` and `hybrid_search` work on a collection that a client created
purely over MCP. Going from an empty store to a working hybrid search takes no
CLI invocation and no HTTP call.

### Reserved attrs

`remember` stamps a few keys of its own alongside whatever `attrs` you pass.
They are ordinary attrs (filterable, and usable as the timestamp field for
recency-decay ranking), so they are worth knowing by name:

| Key | What it holds |
| --- | --- |
| `nidus.text` | The text you remembered, verbatim. This is the field the default full-text schema indexes. |
| `nidus.created_at` | When the entry was first written, as a `DateTime` (UTC epoch ms). Preserved when you re-remember the same id. |
| `nidus.updated_at` | When the entry was last written. |
| `nidus.expires_at` | Set only when you pass `ttl_seconds`; after this instant the entry stops surfacing. |
| `nidus.access_count` | How many `recall` calls with `reinforce` set have returned this entry. Absent means never reinforced. |
| `nidus.last_accessed` | When the entry was last returned by a reinforced `recall`, as a `DateTime` (UTC epoch ms). |

Reserved keys win a collision, so an attr you pass under one of these names is
overwritten rather than silently changing what the store relies on. In summarize
mode `nidus.summary` holds the generated summary (the text that was actually
embedded) while `nidus.text` still holds your original.

### Reinforcement

`recall` takes two optional arguments that let a model mark which entries were
actually useful: **`reinforce`** stamps `nidus.access_count` and
`nidus.last_accessed` on every entry the call returns, and **`extend_ttl_seconds`**
additionally pushes an existing `nidus.expires_at` forward by that many seconds
(only with `reinforce` set, and only on entries that already expire). Rank on
`nidus.access_count` with the [count-decay knobs](/guides/search/#ranking-by-reinforcement)
so memories that keep getting recalled float up and memories nothing ever recalls
sink. Setting `reinforce` makes the call a write, so against a store opened read-only
the tool call is refused rather than answered as though the stamp happened.

### Expiry and duplicates

`remember` takes two optional arguments that shape what accumulates:

- **`ttl_seconds`** gives the entry a lifetime. Past it, the entry stops coming
  back from `recall`, `text_search`, `hybrid_search`, `browse`, `get`, and `related`.
- **`dedupe_threshold`** (0–1) turns on a similarity check at write time. If an
  existing entry in the collection scores above the threshold against the new
  text, that entry is updated in place instead of a competing near-duplicate
  being inserted, and the response tells you which happened. Attrs already on the
  matched entry that your call did not supply are **kept**, not dropped; the
  attrs you do supply win. Expired entries are never dedupe candidates, so a
  write is never merged onto a record that has already lapsed. Leave it out and
  the check never runs, which is worth knowing, because the check costs a scan
  of the collection while the write lock is held, so on a large collection an
  opted-in `remember` is meaningfully slower than one without it.

Correcting or removing what you stored is `forget`'s job: pass `ids` to
remove specific memories, or `filter` to remove every match at once. At least
one of them is required: a call with neither is refused rather than treated
as "remove everything in the collection". Before writing, `get` and `browse`
let an agent check what is already there: `get` looks up one id directly,
and `browse` lists a collection's contents (optionally filtered) so a model
can spot a near-duplicate before adding one.

`related` takes a `collection` and an `id` instead of query text: it searches
with the vector already stored at that entry, in the same collection, and the
source entry is never included in its own results. A genuine near-duplicate of
the source **is** still returned, since exclusion is by id rather than by
score, which makes `related` a way to surface a duplicate that a `dedupe_threshold`
write missed. An expired source, or an id naming no entry, is refused rather
than answered with an empty list.

## Resources and prompts

A tool call is something the model decides to make on its own. A resource is
something a user or client can browse and mention directly instead, so a
stored memory can be pulled into context by name rather than found by search.

### Resources

Two kinds of resource, both under one URI scheme:

```
nidus://collections/<collection>
nidus://collections/<collection>/entries/<id>
```

Both `<collection>` and `<id>` are percent-encoded, so a name containing a
slash stays one path segment. A URI is stable: it names the collection and
the record id rather than a row offset, so it keeps working across restarts
and `compact`.

Reading a collection returns a bounded page of its entries, as a JSON object
with an `entries` array and a `truncated` flag. Each listed entry carries its
own URI, so you can go straight from a collection to one of its entries
without building the URI yourself. When `truncated` is true there are more
entries than fit on the page, and `browse` pages further from there. Reading
an entry returns its id and attributes. Neither read ever returns the vector,
matching every tool on this page.

Expiry applies here too: an expired entry is absent from a collection read,
and its entry URI does not resolve, the same as every other memory surface
(see "Expiry and duplicates" above).

### The `recall_then_answer` prompt

`recall_then_answer` takes a `question` and a `collection` (and an optional
`top_k`), runs the recall server-side, and hands back the matching memories
already assembled into a message with an instruction to answer from them and
cite the ids used. It needs the server to have an embedder, the same as
`recall`, so start it with `--embed-provider` set.

### Reading a resource

This mirrors the `server/discover` example below, but with the method set to
`resources/read` and the `Mcp-Name` header carrying the URI: that header is
mandatory for a resource read, and a request without it is rejected.

```bash
curl -s localhost:7700/mcp \
  -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -H 'mcp-protocol-version: 2026-07-28' \
  -H 'mcp-method: resources/read' \
  -H 'mcp-name: nidus://collections/notes/entries/falcon' \
  -d '{"jsonrpc":"2.0","id":1,"method":"resources/read","params":{
        "uri":"nidus://collections/notes/entries/falcon",
        "_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28",
                 "io.modelcontextprotocol/clientCapabilities":{}}}}'
```

Every request carries `_meta` with both a `protocolVersion` and a
`clientCapabilities` object. There is no handshake over HTTP, so a request
missing either is rejected with `-32602`.

## What it does not do

The surface is deliberately small. There are no MCP subscriptions or tasks:
every nidus operation is a fast synchronous call, so there is nothing to
subscribe to and nothing long-running to hand back a task handle for.
Record-level hygiene is exposed (`forget`, `get`, `browse`), and
`remember` provisions a collection on first write, but the rest of
collection-level lifecycle (reconfiguring or dropping one) and store
maintenance (`compact`, `flush`) are not: those remain operator actions,
deliberately out of an agent's reach.

**A TTL hides an entry; it does not reclaim its row.** Expiry is applied when
you read, by excluding expired entries from results. The underlying row stays
until something explicitly deletes it and compacts the store, and nidus runs no
background threads, so nothing sweeps on a timer. A long-lived store with heavy
TTL churn grows until an operator compacts it.

**Expiry applies to every memory read, not to the raw store routes.** The MCP
tools and `POST /recall` all hide expired entries automatically. `POST /search`
and `/list` are the general-purpose store API whose callers pass their own
filters; those two return expired entries unless you filter `nidus.expires_at`
yourself.

Authorization is the server's existing bearer token rather than MCP's OAuth
flows. For a store you run yourself, a token over loopback is the honest security
model; if you expose one on a real interface, `nidus serve` warns you when it has
no token set.

## Protocol version

nidus implements the `2026-07-28` revision. Over stdio that means the ordinary
session-based handshake: a client sends `initialize`, nidus answers with its
negotiated version and server info, the client confirms with
`notifications/initialized`, and the session proceeds. Over HTTP the same
revision is used statelessly instead: no `initialize` handshake, no session ids,
and method and tool names carried in the `Mcp-Method` and `Mcp-Name` headers so a
gateway can route without reading the body. Statelessness is what lets `/mcp` sit
behind an ordinary round-robin load balancer alongside
[multi-box deployments](/guides/multi-box/): there is no session to pin a client
to an instance, which stdio (one process, one client) never needed anyway.

Older clients still work. The server negotiates down, and `server/discover`
reports every revision it supports:

```bash
curl -s localhost:7700/mcp \
  -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -H 'mcp-protocol-version: 2026-07-28' \
  -H 'mcp-method: server/discover' \
  -d '{"jsonrpc":"2.0","id":1,"method":"server/discover","params":{
        "_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28",
                 "io.modelcontextprotocol/clientCapabilities":{}}}}'
```
