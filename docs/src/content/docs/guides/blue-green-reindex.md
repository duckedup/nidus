---
title: Blue/green reindexing
description: Rebuild a collection under a new name and repoint a fixed alias at it atomically, so a re-embed or a schema change never takes a reader offline.
---

A nidus store pins one dimension for its whole life: reopening with a different dimension
is a hard error. That is exactly right for keeping every collection in one comparable
embedding space, but it means
switching embedding models, changing chunking, or rebuilding an index from scratch cannot
happen **in place**. Deleting and re-upserting a live collection makes it briefly wrong or
empty for anyone reading it mid-rebuild.

**Aliases** solve this without a special mode: an alias is an indirect collection name
that resolves to a concrete collection in one hop. Build the replacement beside the
original under a new name, verify it, then atomically repoint the alias your callers
already use. A reader already open picks the repoint up on its next
[`refresh()`](/guides/storage/#refreshing-a-reader); a search that lands mid-repoint
sees the manifest as it stood a moment before or a moment after, never a partial state, so
it returns hits from the **old** target rather than nothing.

## The sequence

The walkthrough below starts from the steady state this pattern assumes: your callers
query the alias `docs`, which currently resolves to the concrete collection `docs_v1`.
(See [Starting from a concrete collection](#starting-from-a-concrete-collection) if you
are not there yet.)

1. Create the new collection and ingest into it under a name nobody queries yet
   (`docs_v2`), leaving `docs_v1` untouched and serving traffic through the alias.
2. Verify the new collection with a real search before it goes live.
3. Repoint the alias your application actually queries (`docs`) at `docs_v2`. This is one
   atomic manifest publish: no reader ever sees a state where `docs` names neither
   collection.
4. Once the alias no longer points at `docs_v1`, drop it to reclaim the rows.

### CLI

```bash
# 1. build the replacement
nidus create --dir ./store docs_v2
nidus upsert --dir ./store docs_v2 --file docs_v2.json

# 2. verify it directly, before anything points at it
nidus search --dir ./store docs_v2 --query-file query.json -k 5

# 3. atomically repoint the alias callers use
nidus set-alias --dir ./store docs docs_v2
# → {"alias": "docs", "target": "docs_v2"}

# confirm: queries against `docs` now resolve to docs_v2
nidus aliases --dir ./store
# → {"docs": "docs_v2"}

# 4. the alias points at docs_v2 now, so the old collection can go
nidus drop --dir ./store docs_v1
```

### Starting from a concrete collection

An alias name and a collection name share one namespace, so while a concrete collection
called `docs` exists, `set-alias docs ...` is refused. Getting to the steady state above
is therefore a one-time move, and it is the only step in this guide with a gap in it:

```bash
nidus create --dir ./store docs_v1
nidus upsert --dir ./store docs_v1 --file docs.json   # copy the live data across
nidus drop   --dir ./store docs                       # frees the name
nidus set-alias --dir ./store docs docs_v1            # docs is now an alias
```

Between the third and fourth commands the name `docs` resolves to nothing, so do this
once, deliberately, while you can tolerate it. Every later re-embed is then the
gap-free sequence above. The cheapest option, if you are designing a new store, is to
never point callers at a concrete name in the first place: create `docs_v1` and alias
`docs` at it on day one.

### HTTP

```bash
# 1. build the replacement
curl -s -X POST localhost:7700/collections/docs_v2
curl -s localhost:7700/collections/docs_v2/upsert \
  -H 'content-type: application/json' -d @docs_v2.json

# 2. verify it directly
curl -s localhost:7700/search \
  -H 'content-type: application/json' \
  -d '{"scope": ["docs_v2"], "query": [0.1, 0.2, 0.3], "top_k": 5}'

# 3. atomically repoint the alias
curl -s -X PUT localhost:7700/aliases/docs \
  -H 'content-type: application/json' \
  -d '{"target": "docs_v2"}'
# → {"alias": "docs", "target": "docs_v2"}

curl -s localhost:7700/aliases   # → {"docs": "docs_v2"}

# 4. the alias points at docs_v2 now, so the old collection can go
curl -s -X DELETE localhost:7700/collections/docs_v1
```

## What resolves through an alias, and what refuses one

Data verbs resolve an alias to its concrete target transparently: `upsert`, `delete`,
`get`/`get_all`, `get_meta`, and `set_meta` all accept an alias in place of a collection
name. Every `Hit` a search returns still reports the **concrete** collection, never the
alias, since ids are only unique within one collection.

Structural verbs refuse an alias outright: `drop_collection`, `set_fts_schema`,
`set_filter_index`, and `create_collection_with_fts` all require a concrete name. This
keeps "which collection has this schema" unambiguous, and it is what makes step 4 above
safe: you always drop the collection by its own name, never through whatever alias
happens to point at it right now, and `drop_collection` itself refuses while any alias
still names it.

## Constraints worth knowing before you rely on this

- **One hop, never chained.** An alias may not point at another alias; `set_alias`
  rejects that at write time. This keeps resolution O(1) and keeps "what does `docs`
  point at" a single, unambiguous lookup.
- **Shared namespace.** An alias name and a collection name can never collide: creating
  an alias with a collection's name fails, and creating a collection with an alias's name
  fails, in both directions.
- **No dangling aliases.** `set_alias` requires the target collection to exist, and
  `drop_collection` refuses while an alias still points at it. An alias always resolves
  to something.
- **Point-in-time reads see period-correct aliases.** A pinned open
  (`Config::at_version` / `--at-version`) resolves aliases as they stood at that commit,
  not against whatever the alias points at live.
