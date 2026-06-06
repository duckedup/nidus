---
title: Running across a few boxes
description: A user-orchestrated recipe for spreading nidus over a handful of machines — one instance per box, query fan-out and top-k merge on the client. Not a managed cluster.
---

Sometimes one machine isn't quite enough — you've got a couple of spare Mac Minis,
or a desktop and a NUC, and you'd like to spread the vectors across them for more
RAM and more parallel scanning. You can, and it's straightforward, **but read this
first**: what follows is a *deployment recipe you assemble*, not a feature nidus
ships.

:::caution[This is a recipe, not a cluster]
nidus carries **no coordinator, no replication, no rebalancing, and no
fault-tolerance machinery**. Becoming a managed distributed database is exactly
what nidus is *not* — that path leads back to the heavyweight systems it exists to
avoid. The pieces below run entirely on your side. If you outgrow them, you've
outgrown nidus, and that's fine — reach for a purpose-built distributed store.
:::

## The idea

Run **one independent `nidus serve` per box**, each holding a shard of your data.
At query time, your client sends the *same* query to every box, collects each
box's local top-k, and merges those lists into one global top-k.

```
              query q, top_k = 10
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   box A:7700   box B:7700   box C:7700     ← one `nidus serve` each
   local top-10  local top-10  local top-10
        └────────────┼────────────┘
                     ▼
        client merges → global top-10
```

This works because **every nidus store shares one embedding space**. A cosine
(or dot, or Euclidean) score from box A is directly comparable to one from box B,
so merging is just "concatenate the hits and re-rank." There is no global index to
keep consistent — each box answers independently and the client does the rest.

The one rule: **every box must use the same embedding dimension and the same
distance metric**, since their scores have to be comparable. (The dimension and
metric are pinned in each store's header, so `nidus serve` enforces them per box;
it's on you to keep them identical across boxes.)

## Set up each box

On every machine, build a shard and serve it. How you split records across boxes
is your choice — by source repository, by hash of the id, by tenant, whatever
keeps shards roughly even. The simplest split is "different collections, or
different documents, on different boxes."

```bash
# On each box (here dimension 768, cosine):
nidus create --dir ./shard --dim 768 docs
cat shard-records.json | nidus upsert --dir ./shard docs

# Serve it on the box's LAN address. --read-only if a separate writer owns the shard.
# --token requires `Authorization: Bearer <token>` on every request (see below).
nidus serve --dir ./shard --addr 0.0.0.0:7700 --token "$NIDUS_TOKEN"
```

:::danger[Binding `0.0.0.0` exposes the store to your whole network]
Once you leave `127.0.0.1`, anyone who can reach the port can read **and write** —
upsert, delete, drop collections, compact. The server is unauthenticated by
default. On a shared or untrusted LAN, set a shared secret with `--token` (or the
`NIDUS_TOKEN` env var) on every box and send it from the client as
`Authorization: Bearer <token>`; `--read-only` shards additionally refuse all
mutations. `/health` stays open so liveness checks need no credential.
:::

## Fan out and merge on the client

Send the query to each box and merge. For **cosine** and **dot** higher scores are
better, so the global top-k is the highest-scoring hits across all boxes; for
**Euclidean**, lower is better, so flip the sort.

Here's the whole client in `curl` + [`jq`](https://jqlang.github.io/jq/) — fan out
to three boxes, concatenate the hit arrays, sort by score, and keep the top 10:

```bash
QUERY='{"query":[/* 768 floats */],"scope":["docs"],"top_k":10}'
BOXES="box-a.lan box-b.lan box-c.lan"

for host in $BOXES; do
  curl -s "http://$host:7700/search" \
    -H 'content-type: application/json' \
    -H "authorization: Bearer $NIDUS_TOKEN" \
    -d "$QUERY"
done \
| jq -s 'add | sort_by(-.score) | .[0:10]'    # cosine/dot: highest score wins
#         └ for Euclidean, use `sort_by(.score)` instead (lowest distance wins)
```

`jq -s 'add'` slurps the per-box result arrays and flattens them into one; the
sort-and-slice is the merge. Each hit already carries its `collection`, `id`,
`score`, and `attrs`, so the merged list is everything you need — the originating
box doesn't matter, because the scores are comparable.

For a real application you'd do the same thing in your language of choice: issue
the N requests **concurrently** (they're independent), gather the hit lists, and
take the top-k by score. The merge is a few lines; the network fan-out is the only
part worth making parallel.

### Filters and `min_score` carry through

Anything you can pass to a single `/search` — a metadata `filter`, a `min_score`
floor, a `scope` of specific collections — works unchanged per box, because each
box is just a normal nidus server. Apply the same options to every request so the
shards stay comparable.

## What you're responsible for

Because nidus does none of this for you, the recipe leaves these in your hands:

- **Sharding & placement** — deciding which records live on which box, and
  re-sharding if a box fills up. Nothing rebalances automatically.
- **Fault tolerance** — if a box is down, its shard simply isn't in the results.
  There's no replication; a missing box means missing hits until you bring it back
  (or skip it and accept partial results).
- **Writes** — route each upsert to the box that owns that shard. There is no
  cross-box transaction; each box commits its own batch durably and independently.
- **Consistency** — there's nothing to keep consistent *across* boxes (no shared
  index), but each box is individually crash-safe per the
  [storage & durability](/guides/storage/) contract.

## When you've outgrown the recipe

If you find yourself wanting automatic rebalancing, replication for availability,
or a single endpoint that hides the fan-out, you've reached the edge of what nidus
is for. That's the signal to move to a system built for managed distribution —
nidus deliberately stops here so it can stay small, pure-Rust, and fast to build.
For development and small-scale use across a few boxes, the recipe above is the
whole story.
