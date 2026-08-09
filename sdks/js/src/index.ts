//! `@duckedup/nidus` — the JavaScript/TypeScript client for nidus.
//
// A zero-dependency, cross-runtime remote client over the `nidus serve` HTTP API.
// Point a {@link NidusClient} at a local or remote server, then upsert and search.

export { NidusClient } from "./client.js";
export type { FetchLike, NidusClientOptions } from "./client.js";
export { NidusError } from "./errors.js";
export { f } from "./filter.js";
export { decodeAttrs, decodeValue, encodeAttrs, encodeValue, v } from "./values.js";
export type {
  AggregateOptions,
  Aggregation,
  AnnInfo,
  AnnotationOptions,
  Annotations,
  AttrInput,
  ClauseScore,
  ClusterStatus,
  Decay,
  DecodedRecord,
  DecodedValue,
  Filter,
  Footprint,
  Fragment,
  FtsCombine,
  FtsField,
  Highlight,
  HighlightOptions,
  Hit,
  HybridQuerySpelling,
  HybridSearchBase,
  HybridSearchOptions,
  LegScore,
  LimitPer,
  ListOptions,
  NidusRecord,
  OrderBy,
  Predicate,
  ProjectionOptions,
  RankBy,
  RankingOptions,
  Readiness,
  RecallOptions,
  RecordInput,
  RememberOptions,
  RememberResult,
  SearchOptions,
  Stats,
  TextClause,
  TextQuerySpelling,
  TextSearchBase,
  TextSearchOptions,
  Value,
} from "./types.js";
