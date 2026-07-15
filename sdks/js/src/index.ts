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
  AnnInfo,
  AttrInput,
  DecodedRecord,
  DecodedValue,
  Filter,
  Footprint,
  Hit,
  HybridSearchOptions,
  ListOptions,
  NidusRecord,
  Predicate,
  RecallOptions,
  RecordInput,
  RememberOptions,
  SearchOptions,
  Stats,
  TextSearchOptions,
  Value,
} from "./types.js";
