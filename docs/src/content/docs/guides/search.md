---
title: Vector search
description: "Scoped vector search across nidus collections: three distance metrics, exact by default, optional HNSW/IVF approximate indexing, int8/binary quantization, recency and reinforcement ranking, per-attribute caps, and MMR diversity."
---

Search in nidus runs over a scope you choose, using one of three distance metrics,
optionally narrowed by a metadata filter and a score floor. It is **exact by
default** (every in-scope vector is scored) and can opt into an
[approximate index](#approximate-search-ann) (HNSW or IVF) when a full scan is more
than you want to pay.

## Distance metrics

The distance metric is set at store creation via `Config::distance` and pinned
in the data header. Reopening with a different metric is an error.

| Metric | Normalization | Score | Range | Best for |
| --- | --- | --- | --- | --- |
| `Cosine` (default) | Vectors unit-normalized on insert | `dot(q, v)` | \[−1, 1\] | Embedding similarity |
| `Euclidean` | Stored as-is | `−‖q − v‖²` | (−∞, 0\] | Spatial distance |
| `DotProduct` | Stored as-is | `dot(q, v)` | (−∞, ∞) | When magnitude matters |

For all metrics, **higher score = more relevant**, so top-k, `min_score`, and
ranking all work the same way regardless of which metric you choose.

```rust
use nidus::{Config, Distance, Nidus};

// Cosine (default, same as before)
let db = Nidus::open(Config::new("./store", 384))?;

// Euclidean distance
let db = Nidus::open(Config::new("./store-l2", 384).distance(Distance::Euclidean))?;

// Raw dot product (magnitude carries signal)
let db = Nidus::open(Config::new("./store-dot", 384).distance(Distance::DotProduct))?;
# anyhow::Ok(())
```

## Scope

Every search runs over a [`Scope`](/reference/api/#scope): one collection, a
named subset of collections, or the whole store. Results from a multi-collection
scope are **merged into a single ranking**.

```rust
use nidus::Scope;

db.search("code", &q, &opts)?;                                // one collection
db.search(Scope::Collections(&["code", "docs"]), &q, &opts)?; // a named subset
db.search(Scope::All, &q, &opts)?;                            // the whole store
# anyhow::Ok(())
```

This is sound because **all collections share one embedding space**. The
dimension is pinned at store creation, so a vector in `code` and a vector in
`docs` are directly comparable: one ranking over both is meaningful, not a
category error.

## More like this

`search_similar` runs an ordinary search using the vector already stored at a
record, instead of a query you supply yourself:

```rust
use nidus::{Scope, SearchOpts};

let opts = SearchOpts { top_k: 10, ..Default::default() };
let hits = db.search_similar(Scope::Collections(&["docs"]), "docs", "a1", &opts)?;
# anyhow::Ok(())
```

A few things about it are worth knowing up front, because they are easy to get
surprised by otherwise:

- **The source record is never in its own results.** It is dropped by
  `(collection, id)` identity, after ranking, not by a score threshold.
- **A genuine duplicate is still returned.** Exclusion is by id, so a second
  record that happens to hold the same vector as the source scores near 1.0 and
  comes back like any other neighbour.
- **An omitted `scope` searches the source's own collection**, not the whole
  store, which is the one place this call differs from a plain `search`.
- **A text-only record (no vector) is an error**, naming the id and the reason,
  not an empty result: there is nothing to search with.
- It searches with the stored, unit-scaled vector, so only its direction
  matters, exactly like any other search.

Every other `SearchOpts` field (`filter`, `min_score`, `exact`, projections,
`rank_by`, `limit_per`, `diversity`) works the same as it does for `search`.

## Scoring

`SearchOpts` controls the ranking:

```rust
use nidus::SearchOpts;

let opts = SearchOpts {
    top_k: 10,             // keep at most this many hits
    offset: 0,             // skip this many top-ranked hits (pagination)
    min_score: Some(0.5),  // drop anything below this score (None = no floor)
    ..Default::default()
};
# anyhow::Ok(())
```

`top_k` is enforced by a bounded heap, so memory stays flat regardless of how
many rows are scored.

Results are ordered by **score descending, then `collection`, then `id`**. That
tie-break is a guarantee, not a coincidence: it makes a query against an unchanged
store return the same ranking every time, which is what lets you page through it.

## Paginating a search

`offset` skips that many top-ranked hits, so successive pages tile one ranking with
no gap and no overlap. It works the same on `search`, `text_search`, and
`hybrid_search`.

```rust
use nidus::SearchOpts;

let query = vec![0.1_f32; 384];
let page1 = db.search("code", &query, &SearchOpts { top_k: 20, ..Default::default() })?;
let page2 = db.search(
    "code",
    &query,
    &SearchOpts { top_k: 20, offset: 20, ..Default::default() },
)?;
# anyhow::Ok(())
```

Things worth knowing:

- The ranking is computed `offset + top_k` deep, so a later page costs a little more
  than the first. `offset + top_k` may not exceed **10 000** over HTTP: past that a
  request is refused with a `400` rather than quietly shortened.
- An `offset` past the end returns an **empty** list, not an error. That is the signal
  to stop walking.
- For `hybrid_search` the page is cut on the *fused* ranking, never on one leg.
- A page is stable only against an **unchanging** store. Concurrent upserts and deletes
  shift the ranking, so a document can move between pages during a paged walk.

## Choosing the attrs a hit carries

By default a `Hit` carries every attr of the matched record. When the records hold long
text bodies, a top-50 search ships fifty full documents to answer a question about ids
and scores. `SearchOpts::projection` (and `ListOpts::projection`) trims that:

```rust
use nidus::{Projection, SearchOpts};

let query = vec![0.1_f32; 384];
// Only these attrs come back: nothing else is even cloned.
let lean = db.search(
    "code",
    &query,
    &SearchOpts {
        top_k: 10,
        projection: Projection::include(["path", "lang"]),
        ..Default::default()
    },
)?;

// Or: everything except the expensive one.
let no_body = db.search(
    "code",
    &query,
    &SearchOpts {
        top_k: 10,
        projection: Projection::exclude(["body"]),
        ..Default::default()
    },
)?;
# anyhow::Ok(())
```

- The default is `Projection::All`: every attr, exactly as before.
- Projection is applied where the hit is built, so an excluded attr is never copied.
- An included attr the record does not have is simply absent from the hit.
- Ranking, scores, and `top_k` are unaffected: this changes the payload, not the answer.
- Over HTTP the two are `include_attributes` and `exclude_attributes`. Sending both in
  one request is a `400`, since there is no precedence rule to remember.

## Forcing an exact search

A store configured with an [ANN index](#approximate-search-ann) or
[quantization](#quantization) answers every query approximately. `exact: true` opts one
query out of that, running the exact brute-force scan instead, useful when a caller
needs a guaranteed-exact answer over a small filtered subset but wants to keep the index
for everything else.

```rust
use nidus::SearchOpts;

let query = vec![0.1_f32; 384];
let certain = db.search(
    "code",
    &query,
    &SearchOpts { top_k: 10, exact: true, ..Default::default() },
)?;
# anyhow::Ok(())
```

It bypasses the ANN walk, the per-segment index fan-out, and the quantized first pass
alike. The default is `false`, so the store's configured path is unchanged for anyone who
does not ask.

## Ranking by recency

On pure similarity a two-year-old note beats a fresh one that says the same thing.
`SearchOpts::rank_by` layers a **recency decay** over the store's metric: an age penalty
computed from a timestamp attribute and subtracted from each hit's score.

```rust
use nidus::{Decay, RankBy, SearchOpts};

let query = vec![0.1_f32; 384];
let now = 1_770_000_000_000_i64; // epoch milliseconds
let week = 7 * 24 * 60 * 60 * 1000;

let hits = db.search(
    "notes",
    &query,
    &SearchOpts {
        top_k: 10,
        // Halve the decay factor every week; a fully-decayed hit gives up 0.2 of score.
        rank_by: Some(RankBy::Decay(Decay::new("updated_at", now, week).lambda(0.2))),
        ..Default::default()
    },
)?;
# anyhow::Ok(())
```

The timestamp attribute may be a `Value::DateTime` or a `Value::Int`, both **epoch
milliseconds**. `origin` is "now" as *you* supply it, not the wall clock, so the same query
against an unchanged store ranks the same way twice. The score is:

```text
age    = max(0, origin − attrs[field])   // a future timestamp is simply un-aged
factor = decay ^ (age / scale)           // `decay` (default 0.5) at exactly one `scale`
score  = base − lambda × (1 − factor)
```

The penalty **subtracts** rather than multiplying. That is what makes it valid for
`Euclidean` (which scores in (−∞, 0]) and `DotProduct` (which scores anywhere at all), not
just cosine, and for the unbounded BM25 scores of `text_search`, where `rank_by` works
identically. The cost is that `lambda` is in score units, so pick it against the metric in
use.

Two behaviours worth knowing:

- **A record with no usable timestamp is not penalized** (`missing` defaults to `1.0`), so
  switching decay on never silently buries data written before the field existed. Pass
  `.missing(0.0)` for the opposite.
- **`rank_by` does not force an exact search.** [`exact`](#forcing-an-exact-search) is the
  knob for that. Over an ANN or quantized result set the candidates were selected on the
  base score, so decay reorders within an approximate set. A ranked scan also runs
  single-threaded (it needs each record's attrs, which the parallel scan kernels do not
  see), so a `rank_by` query gives up `query_threads`.

`min_score` is compared against the final, decayed score on every path.

### Ranking by reinforcement

`Decay` also takes a `count_field`, so a hit that has been [reinforced](/guides/remember-and-recall/#reinforcement)
pays a smaller penalty than one nothing ever recalls:

```rust
use nidus::{Decay, RankBy, SearchOpts};

let query = vec![0.1_f32; 384];
let now = 1_770_000_000_000_i64;
let week = 7 * 24 * 60 * 60 * 1000;

let hits = db.search(
    "notes",
    &query,
    &SearchOpts {
        top_k: 10,
        rank_by: Some(RankBy::Decay(
            Decay::new("updated_at", now, week)
                .count_field("nidus.access_count")
                .count_scale(10.0)
                .count_lambda(1.0),
        )),
        ..Default::default()
    },
)?;
# anyhow::Ok(())
```

The count term is a second, independent penalty, computed and subtracted alongside the
recency one:

```text
count_factor = n / (n + count_scale)          // n read from `count_field`; 0 if absent
score        = base − lambda × (1 − factor) − count_lambda × (1 − count_factor)
```

`count_scale` (default `10.0`) is the count at which the term is half spent; `count_lambda`
(default `1.0`) is the penalty an entirely un-reinforced record pays. That last part is the
point, not a bug: a record with a high count pays almost nothing, and a record with no
count at all pays the full `count_lambda`, so memories nothing ever recalls sink over time.
`count_field` defaults to `None`, so a `Decay` that never sets it ranks exactly as it did
before this existed. `field` may be left empty when only the count term is wanted.

## Capping hits per attribute value

One verbose document can fill an entire recall window. `limit_per` caps how many hits may
carry any one value of an attribute:

```rust
use nidus::{LimitPer, SearchOpts};

let query = vec![0.1_f32; 384];
let hits = db.search(
    "code",
    &query,
    &SearchOpts {
        top_k: 20,
        limit_per: Some(LimitPer::new("path", 2)), // at most 2 hits per file
        ..Default::default()
    },
)?;
# anyhow::Ok(())
```

The group value is read from the stored record, so excluding the field with a
[projection](#choosing-the-attrs-a-hit-carries) does not lift the cap, and records
**missing** the attribute all share one group: otherwise dropping the attribute would be a
way to opt out.

It is a deliberately **approximate** cap. A capped search ranks eight pages deep, applies the
cap in rank order, then cuts the page, so a page can come back shorter than `top_k` even
though more uncapped matches exist further down. What is guaranteed is the upper bound: a
returned page never carries more than `max` hits for one value.

## Spreading near-duplicates apart

`limit_per` diversifies by attribute value. It cannot help when five near-identical passages
carry the same attributes, or when they are five chunks of one document. `diversity` handles
that case, in vector space: it is a Maximal Marginal Relevance lambda, and over an
over-fetched candidate window it greedily picks the hit maximising

```text
lambda * score - (1 - lambda) * max_cosine_similarity_to_what_is_already_picked
```

```rust
use nidus::SearchOpts;

let query = vec![0.1_f32; 384];
let hits = db.search(
    "code",
    &query,
    &SearchOpts {
        top_k: 10,
        diversity: Some(0.5), // 1.0 is pure relevance, 0.0 pure variety
        ..Default::default()
    },
)?;
# anyhow::Ok(())
```

Things worth knowing:

- **Rank 1 never moves.** With nothing selected yet there is no redundancy to penalise, so
  the best hit is always the best hit. Only what follows it is reshaped.
- **Similarity is cosine**, computed from the stored vectors' real norms, so it means the
  same thing on a dot-product or Euclidean store where vectors are not normalized.
- **Lambda is relative, not a threshold.** A duplicate scoring 0.9996 against 0.9998 is
  barely less relevant, so it only loses its slot once lambda tips away from relevance.
  Around 0.5 balances; lower favours variety.
- **The window is bounded** (512 candidates). MMR needs pairwise similarity, which is
  quadratic, so a deeper page keeps its score order past that point rather than paying
  an unbounded cost.
- **Order of operations.** `limit_per` runs first (a hard cap), then `diversity` reorders
  the survivors, then the page is cut. So both compose, and `offset` still tiles one
  ranking.
- **A text-only record** has no vector and so can never be measurably redundant; it is
  carried on its score alone.
- **Unset is unset.** `None` skips the pass and does not deepen the scan, so an existing
  query is unchanged to the byte.

## Widening a chunked hit with its neighbours

A chunked corpus stores fragments, but a fragment is rarely what you want to read. `expand`
widens each hit with the chunks around it, keyed on the chunk attrs that
[`nidus ingest`](/guides/ingest/) stamps:

```rust
use nidus::{Expand, LimitPer, SearchOpts};

let query = vec![0.1_f32; 384];
let hits = db.search(
    "docs",
    &query,
    &SearchOpts {
        top_k: 10,
        // The best chunk per document, widened by one chunk either side.
        limit_per: Some(LimitPer::new("nidus.parent_id", 1)),
        expand: Some(Expand::new(1)),
        ..Default::default()
    },
)?;
for hit in &hits {
    println!("{}", hit.context.as_deref().unwrap_or_default());
}
# anyhow::Ok(())
```

`Expand::new(radius)` uses the reserved attrs (`nidus.parent_id`, `nidus.chunk_index`,
`nidus.text`). Set the three fields yourself for a corpus you chunked with your own attrs.

Things worth knowing:

- **It is payload only.** The window lands in `Hit::context` and nowhere else, so the ids,
  the scores and the order are identical to the same query with `expand` unset.
- **It runs last**, after reranking, `diversity`, `limit_per`, `min_score` and pagination,
  which is what makes the point above true rather than merely intended.
- **A projected-away body still expands.** Coordinates and text are read from the stored
  record, the same rule [highlighting](/guides/full-text-search/) follows.
- **A record without chunk attrs is left alone**, with no `context`, so a collection holding
  both chunked documents and plain memories still answers.
- **The overlap is dropped, not repeated.** Chunks written by nidus carry their source
  offset, so the window is the source once. A corpus upserted raw (or chunked before 0.75.0)
  has no offsets, and its window is joined with a blank line instead.
- **`radius: 0` is legal** and reports the hit's own text, so code that asks for a context
  field always gets one.

On the memory API the same pair has one text-native spelling, `rollup`, described in the
[ingest guide](/guides/ingest/#reading-a-chunked-corpus-back).

## Quantization

For larger collections, enable quantization to speed up search. The store keeps a
compressed copy of every vector in RAM and search runs in two passes: a fast quantized
first pass selects candidates, overscanning by the `rescore` factor, then the original
f32 vectors re-rank those candidates for an exact final ranking.

Two schemes are available:

| Scheme | Size vs f32 | Default `rescore` | Metrics |
| --- | --- | --- | --- |
| `Quantization::int8()` | 4× smaller | 4 | any |
| `Quantization::binary()` | 32× smaller | 16 | **cosine only** |

int8 is global **symmetric** scalar quantization: one shared scale, no per-dimension
offset, so the int8 score stays monotonic with the true score. Binary keeps only each
dimension's sign bit and scores a Hamming distance; sign codes approximate *angular*
similarity and discard magnitude, which is why they are not a sound ranking proxy for
`DotProduct` or `Euclidean`. Binary is the coarser proxy, hence its larger default
overscan.

```rust
use nidus::{Config, Nidus, Quantization};

// int8: valid for any distance metric.
let db = Nidus::open(Config::new("./store", 768).quantization(Some(Quantization::int8())))?;

// Binary sign bits, with a wider candidate net (cosine stores only).
let db2 = Nidus::open(
    Config::new("./store2", 768).quantization(Some(Quantization::binary().rescore(24)))
)?;
# anyhow::Ok(())
```

The `rescore` factor trades recall for speed: `rescore: 4` means the first pass keeps
`top_k * 4` candidates before the f32 re-rank. Higher values widen the candidate net
(better recall, more f32 work).

**What to expect (int8).** In the `just bench-quant` sweep (uniform-random vectors,
a near-worst case for quantization recall), the two-pass search returns
essentially the exact neighbours (**~100% recall@10 at `rescore` ≥ 2**) for a
**~1.4× query speedup** at 1M × 768, in exchange for **~25% more RAM** (the int8
copy sits alongside the f32 matrix, which the re-rank still needs). The speedup
is bounded by the pure-safe-Rust scalar int8 kernel; the larger theoretical win
would need SIMD int8 dot-product intrinsics, which are `unsafe` and outside
nidus's zero-FFI design. Run `just bench-quant` to measure on your own shapes.

Quantization is purely a runtime optimization: it doesn't change the on-disk
format, and a store opened without it produces identical results (just slower
for large scans). Reach for it when search latency matters more than the extra
RAM. Vectors quantize incrementally on upsert, so adding records stays cheap.

## Approximate search (ANN)

By default every search is **exact**: it scores every in-scope vector. When a
collection grows past the point where a full scan is more than you want to pay,
`Config::ann` opts into an **approximate** index that walks a much smaller candidate
set instead. It is purely a runtime choice: nothing about the on-disk format changes,
and a store opened without it behaves exactly as before.

```rust
use nidus::{Config, Nidus, AnnConfig};

// HNSW: a navigable small-world graph (the default ANN index).
let db = Nidus::open(Config::new("./store", 768).ann(Some(AnnConfig::hnsw())))?;

// or IVF: k-means inverted lists.
let db2 = Nidus::open(Config::new("./store2", 768).ann(Some(AnnConfig::ivf())))?;
# anyhow::Ok(())
```

Both index types work the same way at query time: the index selects an over-fetched
candidate set (`top_k × overscan`), then nidus applies your scope, metadata filter,
and `min_score` to those candidates and ranks the survivors by the **exact** f32
score. So the *final ordering is always exact*; only the candidate *selection* is
approximate.

**Choosing an index.** `AnnConfig::hnsw()` gives high recall and supports cheap
incremental `upsert` (new vectors are inserted into the graph directly); it is the
right default. `AnnConfig::ivf()` uses less memory for its links but fits its k-means
partition from the data present at build time, so heavy incremental growth drifts the
partition until the next `compact()` rebuilds it.

**Tuning.** Each has builder setters: HNSW exposes `m`, `ef_construction`, and
`ef_search` (higher = better recall, more work); IVF exposes `n_lists` and `n_probe`
(more probed lists = better recall, slower). Both share `overscan` (how many
candidates to fetch before the post-filter) and a `seed` for reproducible builds.

```rust
use nidus::AnnConfig;
let cfg = AnnConfig::hnsw().m(32).ef_search(128).overscan(8);
# let _ = cfg;
```

**Persisting the index.** The graph is in-RAM and rebuilt from the vectors on
`open()`, and for HNSW that build is the expensive part. Call
[`db.persist_index()`](/reference/api/#nidus) to write it to an `ann` cache file so
the next `open()` *loads* it instead of rebuilding. The same call also writes the
full-text index's `fts` cache when a collection declares one, so it is worth calling
even on a store with no ANN index at all. It's an explicit, out-of-band operation
(also triggered by `compact()`), never on the `upsert`/`flush` write path,
so storage stays fast. `nidus serve` and `nidus mcp` call it for you on a clean
shutdown, so only an embedded handle needs the explicit call before it is dropped. `open()`
incrementally catches up any rows added since the cache was written, and silently
rebuilds if the cache is missing, stale, or for a different config. The caches are
derived data: deleting the `ann` or `fts` file only costs a one-time rebuild.

**Trade-offs to know.**

- **Approximate recall.** ANN may miss a true neighbour. Raise `ef_search`/`n_probe`
  and `overscan` to trade speed for recall.
- **Selective filters are handled exactly.** The filter is applied *after* the walk, so
  a very selective filter or a narrow collection-subset scope could starve the candidate
  set. nidus detects this case: when a narrowed query's survivor population drops below
  what the overscanned walk can reliably surface, it gathers the filter-passing rows
  directly and scores them exactly instead of walking the index. Selective filtered
  queries stay recall-complete; you pay an exact scan over the survivors, which is small
  by definition in exactly the case that triggers it.
- **Deletes.** A deleted vector leaves a stale node in the index that is skipped at
  query time and fully reclaimed on the next `compact()`.
- **Combine with quantization.** `ann` and `quantization` can be enabled together: the
  index walk scores quantized codes (a cheaper candidate selection) and the exact f32
  rerank over those candidates restores accuracy. Recall runs a little below the
  exact-walk index, so widen `ef_search`/`n_probe` and `overscan` if you need it back.

### Measuring recall against your own data

"Raise `ef_search`/`n_probe` and `overscan`" is easy advice to give and hard to act
on without a number: raise them how far, for what recall, at what latency cost?
`nidus tune --dir <DIR>` answers that against the store you actually have, instead
of a synthetic dataset. It samples queries from the store's own vectors, runs each
one twice (once exact, once through the configured ANN/quantization path), and
reports recall@k and latency for every setting in the sweep, ending in a
recommended `Config`.

```bash
nidus tune --dir ./my-store --ef-search 32,64,128,256 --target-recall 0.95
```

Because a sampled query is itself a stored vector, it has a guaranteed distance-0
match against itself; `tune` drops that self-hit before scoring on both legs, so
the reported recall is not flattered by it. See the [`tune` reference](/reference/cli/#tune)
for the full flag list, and `nidus configure` for persisting the recommendation.

## Per-segment indexing at scale

`Config::ann` above is a **single global** index over every row. There is a second,
size-driven way to index that keeps the freshest data exact: when a store is split into
[segments](/guides/storage/#segments), nidus can IVF-index only the **cold, immutable**
segments and leave the recent write tail exhaustive.

```rust
use nidus::{Config, Nidus};

let db = Nidus::open(
    Config::new("./store", 768)
        .segment_max_rows(Some(100_000))        // seal a segment every 100k rows
        .segment_index_min_rows(Some(100_000)), // IVF-index each sealed segment
)?;
# anyhow::Ok(())
```

With [`segment_index_min_rows`](/reference/configuration/#segment_index_min_rows) set, a
sealed segment of at least that many rows gets its own IVF index (built when it seals and
rebuilt on each `open()`; unlike the `ann`/`fts` caches, per-segment indexes are not
persisted);
the **active** segment (everything written since the last seal) and any smaller segment
stay brute-forced. A search then **fans out**: it scans the exhaustive tail exactly and
walks each cold segment's index for candidates, merging both into one ranking with an exact
f32 rerank. So "exact vs approximate" follows size automatically: the fresh data is always
exact, only the cold bulk is approximate.

This is **off by default** (`segment_index_min_rows = None` → every segment brute-forced →
100% recall, zero knobs), and it is ignored when a global `ann` index is set (that index
already covers every row). The same approximate-recall and deleted-row notes above apply to
the cold segments.

## Explaining a hit

A `Hit` carries one score and the record's attrs, which does not tell you *why* it matched.
`Hit::annotations` does. It is `None` unless the query asks, so nothing about the default
response changes.

```rust
use nidus::{FtsQuery, HighlightOpts, SearchOpts};

let q = FtsQuery::new("body", "run").highlight(HighlightOpts::default());
let hits = db.text_search(
    "docs",
    &q,
    &SearchOpts { top_k: 10, explain: true, ..Default::default() },
)?;

for hit in &hits {
    let a = hit.annotations.as_ref().unwrap();
    for clause in &a.clauses {
        println!("{} scored {}", clause.field, clause.score);
    }
    for hl in &a.highlights {
        for frag in &hl.fragments {
            for &(start, end) in &frag.spans {
                println!("{}: …{}…  matched {:?}", hl.field, frag.text, &frag.text[start..end]);
            }
        }
    }
}
# anyhow::Ok(())
```

- **`explain: true`** reports each *matched* clause's own BM25 score, in query order (a
  clause that did not match is absent, not a zero). A clause carries a **score only, no
  rank**: clauses are summed or maxed into one text score, so there is no separate ranking
  per clause for a rank to refer to. On `hybrid_search`, `HybridOpts::explain` additionally
  reports each fusion leg's own `(rank, score)` (the legs *are* ranked independently, which
  is exactly what RRF fuses), so you can see whether a document rode in on the vector leg,
  the text leg, or both. Vector `search` ignores `explain`: it has a single score to report.
- **`highlight`** returns fragments of the field's stored text with the byte ranges that
  matched. The offsets index the **original** text, not the analyzed token stream: a query
  for `run` highlights the word a document spells `running`, which no substring search would
  find. `HighlightOpts { max_fragments, fragment_chars }` bounds the output; fragments are
  snapped to word boundaries so they never open or close mid-word, and `spans` are byte
  offsets *within the fragment*.
- **Highlighting and [projection](#choosing-the-attrs-a-hit-carries) are independent.**
  Fragments are cut from the stored text, so dropping a 10 KB body from the payload and
  keeping only its snippet is the supported combination, not a silent no-op.
- Annotations are computed on the returned page, after pagination, so the cost scales with
  `top_k` rather than with the candidate set.

Full-text and hybrid search are a runtime feature over the same store. Like ANN and
quantization, they change nothing about the on-disk vector format, and a store opened
without declaring any FTS schema behaves exactly as before.
