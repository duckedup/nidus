---
title: Full-text search (BM25)
description: "BM25 full-text search over nidus collections: declaring a field schema, searching several fields at once, prefix matching for search-as-you-type, per-field tuning, and what gets indexed."
---

Sometimes you know the exact words. A collection can declare **full-text-indexed
fields** and be queried by keyword with
[BM25](https://en.wikipedia.org/wiki/Okapi_BM25) ranking, with no embedder and no
model anywhere in the path. It reuses the same `Hit` results, `Filter`, scope, and
`top_k` heap as [vector search](/guides/search/); only the scoring differs, so
everything you know about filtering and scoping carries straight over.

To search by meaning rather than spelling, see [vector search](/guides/search/). To
run both legs and fuse them into one ranking, see
[hybrid search](/guides/hybrid-search/).

Declare which attribute fields are full-text indexed. You can do it up front at
collection creation (the recommended path: indexing is incremental from the first
upsert) or any time afterward (it indexes the docs already stored):

```rust
use nidus::{Config, FtsField, Nidus};

let mut db = Nidus::open(Config::new("./store", 384))?;

// Up front (recommended):
db.create_collection_with_fts("docs", &[FtsField::new("body")])?;

// …or declare/redeclare later on an existing collection:
db.set_fts_schema("docs", &[FtsField::new("title")])?;
# anyhow::Ok(())
```

Then query a field with [`text_search`](/reference/api/#nidus):

```rust
use nidus::{FtsQuery, SearchOpts};

let hits = db.text_search(
    "docs",
    &FtsQuery::new("body", "running quickly"),
    &SearchOpts { top_k: 10, ..Default::default() },
)?;
# anyhow::Ok(())
```

### Searching several fields at once

A query is a **list of clauses**, and each clause carries its own text, so a record with
both a title and a body can be searched across both in one query, with different words per
field:

```rust
use nidus::{FtsClause, FtsCombine, FtsQuery};

let q = FtsQuery::multi([
    FtsClause::new("title", "rust"),
    FtsClause::new("body", "async runtime"),
]);
let hits = db.text_search("docs", &q, &SearchOpts { top_k: 10, ..Default::default() })?;

// …or take the strongest single clause instead of adding them up:
let q = q.combine(FtsCombine::Max);
# anyhow::Ok(())
```

- **`FtsCombine::Sum`** (the default) adds every matched clause's BM25 score, so a document
  that hits the title *and* the body outranks one that hits either alone.
- **`FtsCombine::Max`** takes the strongest clause, so a long body cannot out-accumulate a
  precise title match.
- A clause naming a field the collection does not full-text index simply contributes
  nothing. An **empty clause list is an error** (over HTTP a `400`) because an empty
  result would otherwise read as "no matches" rather than "you sent no query".
- `FtsQuery::new(field, text)` is the one-clause shorthand, and a single clause scores
  exactly the same under either combine mode. `min_score` applies to the combined score.

### Prefix matching (search as you type)

A clause can match its final term as a **prefix** instead of a complete word, so a
partial last word still finds documents while a caller is typing. Set `prefix` on the
clause:

```rust
use nidus::{FtsClause, FtsQuery};

let q = FtsQuery::multi([FtsClause::new("title", "quick br").prefix()]);
let hits = db.text_search("docs", &q, &SearchOpts { top_k: 10, ..Default::default() })?;
# anyhow::Ok(())
```

Only the **final** term of the clause's text expands: `"quick br"` with `prefix` set
matches any indexed term starting with `br` (`brown`, `bread`, …), while `quick` still
has to match exactly. Each expanded term keeps its own idf and scores as its own
disjunct, so a rare completion can outrank a common one rather than all completions
scoring identically.

The expansion is capped at 256 terms. Past the cap, the match keeps the most common
completions (highest document frequency first) rather than erroring: a caller typing
one character expects *something* back, not a rejection. When `explain: true` is set,
each hit's `ClauseScore` carries an `expansion: {matched, scored}` so a caller can tell
when the cap has truncated a broad prefix.

**This returns documents, not completions.** A prefix clause ranks matching *records*,
the same as any other clause. It does not hand back a list of candidate words to show in
an autocomplete dropdown. For that, use [`Nidus::suggest`](/reference/api/#suggestopts-suggestion--suggestions):
it reuses the same range scan a prefix clause runs, but ranks the terms themselves by
document frequency (commonest first) rather than folding them into a document ranking.

```rust
use nidus::SuggestOpts;

let opts = SuggestOpts { limit: 10, ..Default::default() };
let got = db.suggest("docs", "body", "nid", &opts)?;
// got.suggestions: [{ term: "nidus", df: 42 }, { term: "nidification", df: 3 }, ...]
```

It takes a `Scope`, like `text_search`, so one dropdown can complete from several
collections at once (a completion two of them share is one row whose `df` is the sum).

The prefix fragment is folded (lowercased, optionally ASCII-folded) but not stemmed, while
the index holds stems. So the fragment is matched two ways and the results are unioned:
against the stems directly, and against the field's surface forms (the spellings those stems
came from). That is what lets `"runn"`, and the whole word `"running"`, match a document
containing "running" even though the indexed term is `run`.

A surface form resolves to its *stem*, so a fragment one document spells reaches the whole
stem family: with "running" in the corpus, typing `"running"` also ranks a document that only
says "runs", exactly as a non-prefix clause for `"running"` would. Nothing is invented from
spellings the corpus does not contain, so `"running"` finds nothing in a corpus whose only
word is "runs".

`suggest` scans the same surface forms, so the two surfaces agree on what a fragment reaches.
They differ in what they hand back: `suggest` gives you the words, a prefix clause gives you
the documents.

#### Narrowing a dropdown

The `df` on each completion is a conditioned count, which matters for two things a typeahead
surface usually needs.

**A filter, so the dropdown only offers vocabulary the caller can see.** A completion whose
only documents the filter excludes is not returned at all, rather than returned with a
corpus-wide count (that count describes documents the caller cannot retrieve, so it is
disclosure in its own right).

```rust
let opts = SuggestOpts {
    limit: 10,
    filter: Filter(vec![Predicate::Eq("tenant".into(), Value::Str("acme".into()))]),
};
let got = db.suggest("docs", "body", "nid", &opts)?;
```

**The words already typed, so completions continue the phrase.** Only the final token is
completed, but the earlier words are not thrown away: a completion's `df` counts only
documents that also carry them.

```rust
// "brown" is the commonest br* in the corpus, but no document says both "quick" and "brown"
let got = db.suggest("docs", "body", "quick br", &opts)?;
// got.suggestions: [{ term: "bracket", df: 1 }] ("brown" is not offered at all)
```

Pass the whole phrase typed so far and this happens on its own. A single-token prefix, or one
whose earlier words are all stopwords (`"the br"`), has no head terms and behaves exactly as
`"br"` does.

#### Typo tolerance

If the exact prefix match finds nothing at all, `suggest` retries the fragment against a short
edit-distance budget before giving up:

```rust
let opts = SuggestOpts { limit: 10, ..Default::default() };
let got = db.suggest("docs", "body", "runing", &opts)?;
// got.suggestions: [{ term: "running", df: 12 }, ...]
```

`"runing"` has no exact completion, so the fallback finds `"running"` one edit away and returns
it. This only engages when the exact match is empty: a fragment that already completes costs
nothing extra.

How much tolerance a fragment earns depends on its length: none below four characters, one edit
at four to seven, two at eight or more. Very short fragments get none deliberately, because at
three characters most of the vocabulary is one edit away and the completions would be noise.

It is on by default; turn it off with `fuzzy: false`:

```rust
let opts = SuggestOpts { limit: 10, fuzzy: false, ..Default::default() };
let got = db.suggest("docs", "body", "runing", &opts)?;
// got.suggestions: []
```

### Tuning a field

Each declared field is an `FtsField`, and every knob has a default that reproduces
nidus's original scoring exactly (`FtsField::new("body")` is the untuned field):

```rust
use nidus::{Analyzer, FtsField, Language};

db.set_fts_schema("docs", &[
    // BM25: k1 is term-frequency saturation (default 1.2), b is length
    // normalization from 0 (off) to 1 (full; default 0.75).
    FtsField::new("body").k1(1.5).b(0.3),
    // Analyzer: fold Latin diacritics so "café" and "cafe" are one term, and drop
    // absurd tokens (a base64 blob, a minified bundle) before they reach the index.
    FtsField::new("title").ascii_folding(true).max_token_len(40),
    // The whole analyzer at once, if you prefer.
    FtsField::new("tags").analyzer(Analyzer::default().language(Language::English)),
])?;
# anyhow::Ok(())
```

Tuning is **per field**, not per store: `body` and `title` can score differently in the
same collection. Redeclaring a schema rebuilds the affected field indexes under the new
parameters: the parameters are part of the index cache's validity key, so a reopened
store never serves results scored under a schema you have since changed.

Over HTTP (and in the SDKs) the same knobs ride in the `fts-schema` body, where a bare
field name still means "all defaults":

```json
{"fields": ["title", {"field": "body", "k1": 1.5, "b": 0.3, "ascii_folding": true}]}
```

### What gets indexed, and how a query behaves

- **Analyzer.** US English today (`Language::English`): lowercase → Unicode word
  tokenization → English stopword removal → Porter stemming. Stemming means a query for
  `run` matches documents containing `running`, `runs`, or `ran`. The same analysis runs
  at index and query time, per field, so a field's own tuning applies to both. The
  `Language` enum is the seam for further languages.
- **What gets indexed.** `Str` attrs are indexed directly; `List` attrs are indexed
  per element. A document only lives in a field's index while it has text there.
- **`SearchOpts`.** `top_k`, `offset`, `filter`, `projection`, `rank_by`, `limit_per`, and
  `diversity` all work exactly as for vector search; only `min_score` differs, being a **raw BM25**
  floor rather than a cosine one. Results are tie-broken by `(collection, id)` for
  determinism, the same total order [pagination](/guides/search/#paginating-a-search) relies on.
- **Text-only documents.** A `Record` may carry no vector (`Record::text_only`), a
  pure full-text document. It is found by `text_search` and never by vector `search`.
  Vector-bearing and text-only docs coexist in one collection.
