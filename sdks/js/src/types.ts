//! Wire types for the nidus HTTP API.
//
// These mirror `src/server/dto.rs` and the serde-derived core types in
// `src/model.rs`. The SDK adapts to the server's wire contract — never the
// reverse — so the shapes here are the source of truth for what travels on the
// wire, and the ergonomic helpers in `values.ts` / `filter.ts` produce them.

/**
 * A typed attribute value, externally tagged exactly as `nidus` serde-encodes
 * `Value` on the wire: `{ Str }`, `{ Int }`, `{ Bool }`, `{ List }`, `{ Float }`,
 * `{ DateTime }` (epoch milliseconds, UTC), or the bare string `"Null"`.
 *
 * `Null` is distinct from an absent key: absence means "not set / not indexed",
 * `Null` means "set, and empty/none".
 */
export type Value =
  | { Str: string }
  | { Int: number }
  | { Bool: boolean }
  | { List: string[] }
  | { Float: number }
  | { DateTime: number }
  | "Null";

/**
 * What callers may pass anywhere a {@link Value} is expected: either an
 * explicitly-tagged `Value` (from the `v.*` helpers) or a plain JS scalar that
 * the SDK normalizes — `string → Str`, `boolean → Bool`, `string[] → List`,
 * `Date → DateTime`, `null → Null`, and a `number` to `Int` or `Float` by
 * `Number.isInteger` (JS has no int type, so the value has to decide; use
 * {@link v.float} to pin a whole-numbered field to `Float`).
 */
export type AttrInput =
  | Value
  | string
  | number
  | boolean
  | string[]
  | Date
  | null;

/** A document: caller-supplied `id`, an optional embedding, and typed metadata. */
export interface NidusRecord {
  id: string;
  /** Omit for a text-only doc (indexed by FTS/metadata only, never by vector search). */
  vector?: number[];
  attrs: Record<string, Value>;
}

/** Like {@link NidusRecord} but accepts plain JS values in `attrs` (auto-normalized). */
export interface RecordInput {
  id: string;
  vector?: number[];
  attrs: Record<string, AttrInput>;
}

/** A record read back from the server, with `attrs` decoded to plain JS values. */
export interface DecodedRecord {
  id: string;
  vector?: number[];
  attrs: Record<string, DecodedValue>;
}

/** A single attribute predicate, externally tagged as `nidus` encodes `Predicate`. */
export type Predicate =
  | { Eq: [string, Value] }
  | { Ne: [string, Value] }
  | { Glob: [string, string] }
  | { IGlob: [string, string] }
  | { In: [string, Value[]] }
  | { NotIn: [string, Value[]] }
  | { Lt: [string, Value] }
  | { Le: [string, Value] }
  | { Gt: [string, Value] }
  | { Ge: [string, Value] }
  | { Contains: [string, Value] }
  | { NotContains: [string, Value] }
  | { ContainsAny: [string, Value[]] }
  | { All: Predicate[] }
  | { Any: Predicate[] }
  | { Not: Predicate }
  /** The one three-element leaf: key, text, and the edit budget. */
  | { Fuzzy: [string, string, number] }
  | { ContainsAllTokens: [string, string] }
  | { ContainsAnyToken: [string, string] }
  | { ContainsTokenSequence: [string, string] }
  | { Regex: [string, string] };

/**
 * A conjunction (AND) of predicates. On the wire `Filter` is a newtype over
 * `Vec<Predicate>`, so it serializes as a bare array — an empty array matches
 * everything.
 */
export type Filter = Predicate[];

/** A search/list result row, decoded so `attrs` holds plain JS values. */
export interface Hit {
  collection: string;
  id: string;
  score: number;
  attrs: Record<string, DecodedValue>;
  /** Why this hit matched — present only when the query asked to `explain` or highlight. */
  annotations?: Annotations;
}

/** One fusion leg's own view of a hit: its rank in that leg (0-based) and that leg's score. */
export interface LegScore {
  rank: number;
  score: number;
}

/** One BM25 clause's contribution to a hit's text score. Only matched clauses appear. */
export interface ClauseScore {
  field: string;
  score: number;
}

/**
 * An excerpt of a field's stored text plus the ranges a query term matched. The server
 * reports UTF-8 **byte** offsets; the SDK converts them to JS string indices (UTF-16 code
 * units), so `text.slice(...span)` is the matched term even when the excerpt is not ASCII.
 */
export interface Fragment {
  text: string;
  spans: [number, number][];
}

/** The fragments found in one full-text field. */
export interface Highlight {
  field: string;
  fragments: Fragment[];
}

/**
 * Why a hit matched. Every part is opt-in and absent when it carries nothing, so a hit
 * annotated by `explain` alone has no `highlights` key.
 */
export interface Annotations {
  /** The vector leg's rank and score, on a hybrid hit that leg returned. */
  vector?: LegScore;
  /** The BM25 leg's rank and combined text score, on a hybrid hit that leg returned. */
  text?: LegScore;
  /** Each matched clause's own BM25 score, in query order. */
  clauses?: ClauseScore[];
  /** Highlighted fragments, one entry per clause field that had a match. */
  highlights?: Highlight[];
}

/**
 * A {@link Value} decoded back to a plain JS value. A `DateTime` comes back as a
 * `Date`, not a number, so a decoded `attrs` map re-encodes to what it came from.
 */
export type DecodedValue = string | number | boolean | string[] | Date | null;

/** On-disk footprint, mirroring `FootprintDto`. */
export interface Footprint {
  rows: number;
  dead_rows: number;
  dimension: number;
  vector_bytes: number;
  doc_count: number;
}

/** Active ANN-index configuration, mirroring `AnnDto` (`null` when exact search). */
export interface AnnInfo {
  kind: string;
  overscan: number;
  seed: number;
  m?: number;
  ef_construction?: number;
  ef_search?: number;
  n_lists?: number;
  n_probe?: number;
}

/** Store-wide introspection, mirroring the `/stats` response. */
export interface Stats {
  dimension: number;
  distance: string;
  ann: AnnInfo | null;
  collections: string[];
  footprint: Footprint;
}

/**
 * Which attrs the returned hits carry. Omit both for every attr (the default).
 * Sending both is a `400` — the server refuses rather than picking one.
 */
export interface ProjectionOptions {
  /** Return only these attrs. A named attr the record lacks is simply absent. */
  includeAttributes?: string[];
  /** Return every attr but these. */
  excludeAttributes?: string[];
}

/**
 * Recency decay over a timestamp attribute. The penalty is *subtracted* from the base
 * score — `score = base - lambda * (1 - decay ^ (age / scale))` — so it stays meaningful
 * for a metric whose scores are negative or unbounded (Euclidean, dot product, BM25).
 */
export interface Decay {
  /** The timestamp attribute: a `DateTime`, or an `Int` of epoch milliseconds. */
  field: string;
  /**
   * "Now". Ages are measured back from here rather than from the wall clock, so the same
   * query against an unchanged store ranks the same way twice. A `Date` or epoch ms.
   */
  origin: Date | number;
  /** Age in milliseconds at which the factor equals `decay` (default: 7 days). */
  scale?: number;
  /** Factor reached at exactly `scale` old, in `(0, 1)`; the default `0.5` makes it a half-life. */
  decay?: number;
  /** Score a fully-decayed hit gives up (default `1`). */
  lambda?: number;
  /** Factor for a record whose `field` is missing or not a timestamp. Defaults to `1` — no penalty. */
  missing?: number;
}

/**
 * A ranking expression layered over the store's distance metric. Omitting it is the bare
 * metric — the ranking nidus has always returned.
 */
export type RankBy = { decay: Decay };

/**
 * Cap how many hits may carry any one value of an attribute — "at most 2 hits per file".
 * Records *missing* the attribute form one shared group, so an absent value cannot evade
 * the cap. Approximate: it thins the ranking rather than searching deeper to refill it.
 */
export interface LimitPer {
  field: string;
  max: number;
}

/**
 * Sort a {@link NidusClient.list} by an attribute instead of storage order. Values of
 * another type, unorderable ones (`null`/lists/`NaN`), and records missing the attribute
 * sort into one trailing bucket, which stays trailing in both directions.
 */
export interface OrderBy {
  field: string;
  descending?: boolean;
}

/** Ranking knobs shared by {@link NidusClient.search} and {@link NidusClient.textSearch}. */
export interface RankingOptions {
  rankBy?: RankBy;
  limitPer?: LimitPer;
}

/** How several {@link TextClause}s fold into one text score. */
export type FtsCombine = "Sum" | "Max";

/** One clause of a multi-field text query: an indexed field and the query text for it. */
export interface TextClause {
  field: string;
  query: string;
}

/** How much text a highlight carries. `fragmentChars` is a character budget, not bytes. */
export interface HighlightOptions {
  /** Most fragments returned per field (default `1`). */
  maxFragments?: number;
  /** Characters per fragment (default `160`): leading context, then the match and its tail. */
  fragmentChars?: number;
}

/** Annotation knobs shared by {@link NidusClient.textSearch} and {@link NidusClient.hybridSearch}. */
export interface AnnotationOptions {
  /** Report each leg's and each matched clause's own score in a hit's `annotations`. */
  explain?: boolean;
  /**
   * Return highlighted fragments; `true` takes the defaults. Highlighting reads the
   * stored text, so it still works on a field the projection dropped.
   */
  highlight?: boolean | HighlightOptions;
}

/** Options for {@link NidusClient.search}. An empty/omitted `scope` searches every collection. */
export interface SearchOptions extends ProjectionOptions, RankingOptions {
  query: number[];
  scope?: string[];
  topK?: number;
  /** Skip this many top-ranked hits, for pagination. `offset + topK` may not exceed 10000. */
  offset?: number;
  minScore?: number;
  filter?: Filter;
  /**
   * Force the exact scan for this query, bypassing any ANN index and the
   * quantized first pass. The index stays in place for every other query.
   */
  exact?: boolean;
}

/**
 * The two accepted spellings of a text query: one `field` plus its `query`, or a list of
 * `clauses` each carrying its own text. Sending both, or an empty list, is a `400` — an
 * empty result would otherwise read as "no matches" rather than "no query".
 */
export type TextQuerySpelling =
  | { field: string; query: string; clauses?: never; combine?: never }
  | {
      clauses: TextClause[];
      combine?: FtsCombine;
      field?: never;
      query?: never;
    };

/** The knobs of {@link TextSearchOptions} that do not name what to search. */
export interface TextSearchBase
  extends ProjectionOptions,
    RankingOptions,
    AnnotationOptions {
  scope?: string[];
  topK?: number;
  /** Skip this many top-ranked hits, for pagination. */
  offset?: number;
  /** A raw BM25 score floor (not cosine). */
  minScore?: number;
  filter?: Filter;
}

/** Options for {@link NidusClient.textSearch} (BM25). */
export type TextSearchOptions = TextSearchBase & TextQuerySpelling;

/**
 * One entry of {@link NidusClient.setFtsSchema}'s `fields`: the attribute to index
 * plus any BM25/analyzer knobs to override. Every knob is optional — omit them all
 * (or pass the bare field name instead) for the server's defaults, `k1 = 1.2`,
 * `b = 0.75`, US English, no folding, no token-length cap.
 */
export interface FtsField {
  /** The attribute to full-text index. */
  field: string;
  /** BM25 term-frequency saturation (default `1.2`). */
  k1?: number;
  /** BM25 length normalization, `0`–`1` (default `0.75`). */
  b?: number;
  /** Analyzer language; `"english"` is the only one today. */
  language?: string;
  /** Fold Latin diacritics to ASCII, so `café` and `cafe` share a term. */
  asciiFolding?: boolean;
  /** Drop tokens longer than this many characters (default: no cap). */
  maxTokenLen?: number;
}

/** {@link TextQuerySpelling} for hybrid search, whose single form spells the text `text`. */
export type HybridQuerySpelling =
  | { field: string; text: string; clauses?: never; combine?: never }
  | {
      clauses: TextClause[];
      combine?: FtsCombine;
      field?: never;
      text?: never;
    };

/** The knobs of {@link HybridSearchOptions} that do not name what the text leg searches. */
export interface HybridSearchBase extends AnnotationOptions {
  vector: number[];
  scope?: string[];
  topK?: number;
  /** Skip this many hits of the *fused* ranking, for pagination. */
  offset?: number;
  filter?: Filter;
  rrfK?: number;
  candidates?: number;
  /** Weight on the vector leg's RRF contribution. Both weights at `1` is plain fusion. */
  vectorWeight?: number;
  /** Weight on the BM25 leg's RRF contribution (default `1`). */
  textWeight?: number;
}

/** Options for {@link NidusClient.hybridSearch} (vector + BM25 fused via RRF). */
export type HybridSearchOptions = HybridSearchBase & HybridQuerySpelling;

/** Options for {@link NidusClient.list} (metadata-only, paginated). */
export interface ListOptions extends ProjectionOptions {
  scope?: string[];
  offset?: number;
  limit?: number;
  filter?: Filter;
  /** Sort by an attribute instead of storage order. */
  orderBy?: OrderBy;
}

/** Options for {@link NidusClient.aggregate}. An empty/omitted `scope` covers every collection. */
export interface AggregateOptions {
  scope?: string[];
  filter?: Filter;
  /** Attributes to sum. A missing or non-numeric value is skipped, not counted as zero. */
  sum?: string[];
}

/** What {@link NidusClient.aggregate} answers: the match count plus one sum per named field. */
export interface Aggregation {
  count: number;
  /** One entry per requested `sum` field, decoded from its tagged `Int`/`Float`. */
  sums: Record<string, number>;
}

/**
 * Options for {@link NidusClient.remember} (text-native ingest). The server
 * embeds the text and upserts; `mode: "summarize"` summarizes it first (and
 * requires the server to have been started with a summarizer).
 */
export interface RememberOptions {
  /**
   * `"raw"` (embed the text as given, the default) or `"summarize"` (summarize
   * first, then embed the summary — stamps `nidus.summary`/`nidus.source` attrs).
   */
  mode?: "raw" | "summarize";
  /** Typed metadata to stamp on the stored record (plain JS values auto-normalized). */
  attrs?: Record<string, AttrInput>;
}

/** Options for {@link NidusClient.recall} (embed the query text, then vector-search). */
export interface RecallOptions {
  topK?: number;
  /** Cosine-similarity floor; hits below it are dropped. */
  minScore?: number;
  filter?: Filter;
}
