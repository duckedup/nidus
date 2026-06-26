//! Wire types for the nidus HTTP API.
//
// These mirror `src/server/dto.rs` and the serde-derived core types in
// `src/model.rs`. The SDK adapts to the server's wire contract — never the
// reverse — so the shapes here are the source of truth for what travels on the
// wire, and the ergonomic helpers in `values.ts` / `filter.ts` produce them.

/**
 * A typed attribute value, externally tagged exactly as `nidus` serde-encodes
 * `Value` on the wire: `{ Str }`, `{ Int }`, `{ Bool }`, `{ List }`, or the bare
 * string `"Null"`.
 *
 * `Null` is distinct from an absent key: absence means "not set / not indexed",
 * `Null` means "set, and empty/none".
 */
export type Value =
  | { Str: string }
  | { Int: number }
  | { Bool: boolean }
  | { List: string[] }
  | "Null";

/**
 * What callers may pass anywhere a {@link Value} is expected: either an
 * explicitly-tagged `Value` (from the `v.*` helpers) or a plain JS scalar that
 * the SDK normalizes — `string → Str`, `boolean → Bool`, integer `number → Int`,
 * `string[] → List`, `null → Null`. A non-integer number throws (the store has no
 * float attribute type; floats belong in the vector, not in attrs).
 */
export type AttrInput = Value | string | number | boolean | string[] | null;

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
  | { In: [string, Value[]] }
  | { NotIn: [string, Value[]] }
  | { Lt: [string, Value] }
  | { Le: [string, Value] }
  | { Gt: [string, Value] }
  | { Ge: [string, Value] };

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
}

/** A {@link Value} decoded back to a plain JS value. */
export type DecodedValue = string | number | boolean | string[] | null;

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

/** Options for {@link NidusClient.search}. An empty/omitted `scope` searches every collection. */
export interface SearchOptions {
  query: number[];
  scope?: string[];
  topK?: number;
  minScore?: number;
  filter?: Filter;
}

/** Options for {@link NidusClient.textSearch} (BM25). */
export interface TextSearchOptions {
  field: string;
  query: string;
  scope?: string[];
  topK?: number;
  /** A raw BM25 score floor (not cosine). */
  minScore?: number;
  filter?: Filter;
}

/** Options for {@link NidusClient.hybridSearch} (vector + BM25 fused via RRF). */
export interface HybridSearchOptions {
  vector: number[];
  field: string;
  text: string;
  scope?: string[];
  topK?: number;
  filter?: Filter;
  rrfK?: number;
  candidates?: number;
}

/** Options for {@link NidusClient.list} (metadata-only, paginated). */
export interface ListOptions {
  scope?: string[];
  offset?: number;
  limit?: number;
  filter?: Filter;
}
