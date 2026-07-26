---
title: Running across a few boxes
description: Spread a corpus over several machines by running one independent nidus instance per box and fanning queries out from the client — no coordinator, no replication, nothing to operate.
---

Suppose one machine is no longer enough — a handful of Mac Minis, a couple of old servers,
some spare VMs. You do not need a distributed system to use them. You need a **shard map and
a merge**, and both live in your client.

Run one ordinary nidus instance per box, each owning its own slice of the corpus. To query,
ask every box and merge the answers. That is the whole recipe. It works because **all nidus
stores of the same dimension share one embedding space**: a score from box 3 means exactly
what a score from box 1 means, so ranking across them is just sorting.

This is a **deployment pattern, not a product feature**. nidus ships nothing to support it,
and that is deliberate — see [what this is not](#what-this-is-not).

## When to use this instead of cluster mode

nidus has [two](/guides/storage-backends/#cooperating-instances-cluster) shapes for more than
one process, and they solve opposite problems. Picking the wrong one is the main way to make
this harder than it is.

| | Several boxes, fanned out *(this guide)* | [Cluster mode](/guides/storage-backends/#cooperating-instances-cluster) |
| --- | --- | --- |
| Stores | **N independent** stores, one per box | **One** shared store on S3/GCS |
| Solves | **Capacity** — more RAM and more cores than one box has | **Read scale-out and failover** over one dataset |
| Needs | Nothing. Local disks. | A shared object store **and** a shared memory tier |
| Query | Client asks every box, merges | Any instance answers in full |
| Who knows the layout | Your client | Nobody has to |

If the corpus fits on one box and you want more searchers or a standby writer, use cluster
mode. If the corpus does *not* fit on one box, fan out. They compose: each shard in a fan-out
can itself be a cluster, though that is a lot of machinery for a small setup.

## Set up the boxes

Nothing special. Each box runs a normal server over its own store:

```bash
# On each box, with its own slice of the data.
nidus serve --dir /var/lib/nidus/store --dim 768 --addr 0.0.0.0:7700
```

Every box must use the **same `--dim`**. That is the one hard requirement, and getting it
wrong is caught immediately: a mismatched vector is rejected with a `400`.

Two things worth doing up front, because they are annoying to retrofit:

- **Give each box a token** (`--token`) and keep the fleet on a private network. Fan-out means
  every box is reachable from wherever your client runs. See
  [securing a deployment](/guides/http-server/#securing-a-deployment).
- **Write down the shard map** — which ids live where — somewhere your client and your
  ingest job both read. It is the only piece of state this design has, and losing track of it
  is the only way to lose data you cannot recover by re-indexing.

## Decide what goes where

Any rule works, as long as it is a **function of the document id** so that a re-index sends
each document back to the same box:

```python
BOXES = ["box1:7700", "box2:7700", "box3:7700"]

def box_for(doc_id: str) -> str:
    import hashlib
    h = hashlib.blake2b(doc_id.encode(), digest_size=8).digest()
    return BOXES[int.from_bytes(h, "big") % len(BOXES)]
```

Hash the id when you have no better idea — it spreads evenly and needs no bookkeeping.
Prefer a *natural* boundary when you have one (per tenant, per repository, per year): it
keeps related documents together, which means most filtered queries can skip boxes entirely,
and it makes "drop that customer" a single `DELETE`.

Adding a box changes a hash-based map's answer for most ids. There is no rebalancer, so a
re-index is how you grow — plan for the corpus to be re-indexable, or use a natural
boundary and assign new boundaries to new boxes.

## Query: fan out, then merge

Ask every box for the top `k`, concatenate, sort by score, keep `k`.

**Taking `k` from each box and keeping the best `k` overall is exact** — not an
approximation. A document can only be in the global top `k` if it is in its own box's top
`k`, so nothing that belongs in the answer can be missed. (This holds for the whole-store
ranking too, and for the same reason.)

With `curl` and `jq`, which is enough to try it out:

```bash
QUERY='{"query": [0.1, 0.2, 0.3], "top_k": 10}'

for box in box1 box2 box3; do
  curl -sS "http://$box:7700/search" \
    -H 'content-type: application/json' \
    -H "authorization: Bearer $NIDUS_TOKEN" \
    -d "$QUERY" &
done | jq -s 'add | sort_by(-.score) | .[:10]'
```

`jq -s` slurps the concatenated arrays into one, `add` flattens them, and the sort takes it
from there. Note the `&`: the boxes are queried **concurrently**, so latency is the slowest
box rather than the sum. Doing this serially is the one mistake that makes fan-out feel slow.

The same thing in Python, with the error handling a real client needs:

```python
import concurrent.futures, requests

def search(box, body, token):
    r = requests.post(f"http://{box}/search", json=body,
                      headers={"authorization": f"Bearer {token}"}, timeout=2.0)
    r.raise_for_status()
    return r.json()

def fan_out(boxes, body, token):
    with concurrent.futures.ThreadPoolExecutor(len(boxes)) as pool:
        results, failed = [], []
        for box, fut in [(b, pool.submit(search, b, body, token)) for b in boxes]:
            try:
                results.extend(fut.result())
            except Exception:
                failed.append(box)          # partial answer — say so, do not hide it
    results.sort(key=lambda h: -h["score"])
    return results[: body.get("top_k", 10)], failed
```

That `failed` list is the part worth writing yourself rather than borrowing. With no
replication, a box that is down means its slice is **absent from the results**, and an
answer missing a third of the corpus looks exactly like an answer where nothing matched.
Decide explicitly which you want — fail the query, or return it flagged as partial — and
make the choice visible to whatever consumes it.

Everything else fans out the same way. `POST /list` merges by concatenation (there is no
score to sort by, so paginate per box or gather and re-page). `POST /text-search` and
`POST /hybrid-search` return scores that are **only comparable within one box** — BM25 is
scored against the local corpus statistics, and hybrid search fuses local ranks — so merging
those by score is not sound the way vector search is. Fan them out per box and combine by
rank if you need to.

## Ingest and delete

Route each document to `box_for(id)` and upsert there as usual. Two properties keep this
simple:

- **`upsert` is idempotent per id**, so a retry after a timeout is safe.
- **The shard map is a pure function**, so re-ingesting a document lands it on the same box
  and overwrites in place rather than duplicating across the fleet.

Deletes go to the owning box. If you have lost track of which box owns an id, sending the
delete to every box is harmless — a delete of an absent id is a no-op.

Give each box's write path room to work: writes to one instance are
[group-committed](/guides/how-it-works/#group-commit), so a batch ingest that keeps a few
connections busy per box costs far less than one at a time.

## Backups

Each box backs itself up independently — `nidus backup` produces one archive per store (see
[backup & restore](/guides/cli-and-server/#backup--restore)). Snapshot them on the same
schedule and keep them together with the shard map: an archive restored onto the wrong box
is not wrong exactly, but every query for its documents will go somewhere else.

## What this is not

This recipe is entirely yours to operate. nidus contributes no code to it, and the following
are explicitly out of scope — not "not yet", but **not planned**, because building them is
how a small embeddable store turns into the managed cluster it exists as an alternative to:

- **No coordinator or service discovery.** The list of boxes is a constant in your client.
- **No replication.** One copy of each shard. A dead box takes its slice offline until it is
  restored from backup or re-indexed.
- **No fault tolerance in queries.** Partial results are a thing you have to notice and
  decide about (see above). nidus cannot do it for you: it does not know the other boxes
  exist.
- **No rebalancing.** Changing the box count is a re-index, not an operation.
- **No cross-box transactions.** Each upsert is atomic on its own box, and nothing spans two.
- **No global BM25 statistics.** Full-text scores are per-box, as noted above.

If you want any of those, you want a different kind of system, and you should reach for one
rather than growing this into it. What this pattern buys is real, though, and it is the thing
people usually actually need: the RAM and cores of several cheap machines, for the cost of a
hash function and a `sort_by`.
