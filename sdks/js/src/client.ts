//! `NidusClient` — a remote client over the `nidus serve` HTTP API.
//
// One method per endpoint (`src/server/mod.rs`). "Local vs remote" is just the
// base URL: point at a local `nidus serve` or any reachable host. Built on the
// platform-global `fetch`, so it runs unchanged on Node 18+, Deno, Bun, Cloudflare
// Workers, and browsers — with no runtime dependencies.

import { decodeAnnotations, type WireAnnotations } from "./annotations.js";
import { NidusError } from "./errors.js";
import type {
  AggregateOptions,
  Aggregation,
  BatchSearchOptions,
  ClusterStatus,
  DecodedRecord,
  Filter,
  FilterIndexField,
  FtsField,
  HighlightOptions,
  Hit,
  HybridSearchOptions,
  ListOptions,
  NidusRecord,
  RankBy,
  Readiness,
  RecallOptions,
  RecordInput,
  RememberOptions,
  RememberResult,
  SearchOptions,
  Stats,
  TextSearchOptions,
  Value,
} from "./types.js";
import { decodeAttrs, decodeValue, encodeAttrs } from "./values.js";

/** Minimal `fetch` signature the client needs — satisfied by the platform global. */
export type FetchLike = (
  input: string,
  init?: RequestInit,
) => Promise<Response>;

/** Construction options for {@link NidusClient}. */
export interface NidusClientOptions {
  /** Base URL of the server, e.g. `http://127.0.0.1:7700`. Trailing slash optional. */
  baseUrl: string;
  /** Bearer token, when the server was started with `--token`. */
  token?: string;
  /** Override the `fetch` implementation (defaults to `globalThis.fetch`). */
  fetch?: FetchLike;
  /** Per-request timeout in milliseconds. Omit (or `0`) to disable. */
  timeoutMs?: number;
  /** Extra headers sent on every request. */
  headers?: Record<string, string>;
}

export class NidusClient {
  private readonly baseUrl: string;
  private readonly token?: string;
  private readonly doFetch: FetchLike;
  private readonly timeoutMs: number;
  private readonly extraHeaders: Record<string, string>;

  constructor(options: NidusClientOptions) {
    if (!options.baseUrl) {
      throw new TypeError("NidusClient requires a baseUrl");
    }
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.token = options.token;
    this.timeoutMs = options.timeoutMs ?? 0;
    this.extraHeaders = options.headers ?? {};
    const f = options.fetch ?? globalThis.fetch;
    if (typeof f !== "function") {
      throw new TypeError(
        "no fetch available; pass options.fetch (Node < 18, or a custom runtime)",
      );
    }
    // Bind so a passed `globalThis.fetch` keeps its `this`.
    this.doFetch = f === globalThis.fetch ? f.bind(globalThis) : f;
  }

  // ── Admin / introspection ─────────────────────────────────────────────────

  /** Liveness check. Returns `true` when the server answers `/health`. */
  async health(): Promise<boolean> {
    try {
      const res = await this.raw("GET", "/health");
      return res.ok;
    } catch {
      return false;
    }
  }

  /** Store-wide introspection: dimension, distance, ANN config, collections, footprint. */
  stats(): Promise<Stats> {
    return this.request<Stats>("GET", "/stats");
  }

  /**
   * Readiness: whether this instance can serve. A `503` is the negative answer, not an
   * error, so a poll loop branches on `ready` instead of catching. Other failures throw.
   */
  async ready(): Promise<Readiness> {
    const res = await this.raw("GET", "/ready");
    const text = await res.text();
    if (res.status === 503) return { ready: false, reason: extractError(text, 503) };
    if (!res.ok) throw new NidusError(extractError(text, res.status), res.status);
    return JSON.parse(text) as Readiness;
  }

  /** Cluster role, writer-handle state, fencing token, commit counter, staleness. */
  cluster(): Promise<ClusterStatus> {
    return this.request<ClusterStatus>("GET", "/cluster");
  }

  /** List every collection name. */
  collections(): Promise<string[]> {
    return this.request<string[]>("GET", "/collections");
  }

  /** Create a collection. Idempotent on the server side. */
  async createCollection(name: string): Promise<void> {
    await this.request("POST", `/collections/${enc(name)}`, {});
  }

  /** Drop a collection and all its records. */
  async dropCollection(name: string): Promise<void> {
    await this.request("DELETE", `/collections/${enc(name)}`);
  }

  /** Read a collection's free-form string metadata. */
  getMeta(name: string): Promise<Record<string, string>> {
    return this.request<Record<string, string>>(
      "GET",
      `/collections/${enc(name)}/meta`,
    );
  }

  /** Replace a collection's free-form string metadata. */
  async setMeta(name: string, meta: Record<string, string>): Promise<void> {
    await this.request("PUT", `/collections/${enc(name)}/meta`, meta);
  }

  // ── Data ──────────────────────────────────────────────────────────────────

  /**
   * Insert or replace records (idempotent on `id` within the collection).
   * `attrs` accept plain JS values or `v.*` helpers; they are normalized for you.
   * Returns the number of records upserted.
   */
  async upsert(name: string, records: RecordInput[]): Promise<number> {
    const wire: NidusRecord[] = records.map((r) => ({
      id: r.id,
      ...(r.vector !== undefined ? { vector: r.vector } : {}),
      attrs: encodeAttrs(r.attrs),
    }));
    const res = await this.request<{ upserted: number }>(
      "POST",
      `/collections/${enc(name)}/upsert`,
      { records: wire },
    );
    return res.upserted;
  }

  /** Delete records by id. Returns the number deleted. */
  async delete(name: string, opts: { ids: string[] }): Promise<number> {
    const res = await this.request<{ deleted: number }>(
      "POST",
      `/collections/${enc(name)}/delete`,
      { ids: opts.ids },
    );
    return res.deleted;
  }

  /** Delete every record matching `filter`. Returns the number deleted. */
  async deleteWhere(name: string, filter: Filter): Promise<number> {
    const res = await this.request<{ deleted: number }>(
      "POST",
      `/collections/${enc(name)}/delete`,
      { filter },
    );
    return res.deleted;
  }

  /** Fetch every record in a collection (attrs decoded to plain JS values). */
  async records(name: string): Promise<DecodedRecord[]> {
    const recs = await this.request<NidusRecord[]>(
      "GET",
      `/collections/${enc(name)}/records`,
    );
    return recs.map((r) => ({
      id: r.id,
      ...(r.vector !== undefined ? { vector: r.vector } : {}),
      attrs: decodeAttrs(r.attrs),
    }));
  }

  /**
   * Declare the full-text-indexed attribute fields for a collection. A bare string
   * takes the server's BM25/analyzer defaults; an {@link FtsField} object tunes `k1`,
   * `b`, and the analyzer for that field alone.
   */
  async setFtsSchema(
    name: string,
    fields: (string | FtsField)[],
  ): Promise<void> {
    await this.request("POST", `/collections/${enc(name)}/fts-schema`, {
      fields: fields.map(encodeFtsField),
    });
  }

  /**
   * Declare which attribute fields are indexed for the text predicates (`Fuzzy`,
   * `ContainsAllTokens`, `ContainsAnyToken`, `ContainsTokenSequence`, `Regex`). Fields
   * already written are indexed as part of applying the declaration.
   *
   * This changes how fast those predicates run, never what they return: the index
   * proposes candidate documents and the predicate itself still decides. The cost is
   * paid at write time and in memory. Pass an empty array to drop the declaration.
   */
  async setFilterIndex(
    name: string,
    fields: (string | FilterIndexField)[],
  ): Promise<void> {
    await this.request("POST", `/collections/${enc(name)}/filter-index`, {
      fields: fields.map(encodeFilterIndexField),
    });
  }

  // ── Search ──────────────────────────────────────────────────────────────

  /** Vector (cosine) nearest-neighbour search. Empty `scope` searches all collections. */
  search(opts: SearchOptions): Promise<Hit[]> {
    return this.searchRequest("/search", {
      query: opts.query,
      scope: opts.scope ?? [],
      top_k: opts.topK,
      offset: opts.offset,
      min_score: opts.minScore,
      filter: opts.filter ?? [],
      exact: opts.exact,
      include_attributes: opts.includeAttributes,
      exclude_attributes: opts.excludeAttributes,
      rank_by: encodeRankBy(opts.rankBy),
      limit_per: opts.limitPer,
    });
  }

  /**
   * BM25 full-text search over one indexed field, or over a `clauses` list folded by
   * `combine` (`"Sum"` unless said otherwise). Naming the fields both ways is a `400`.
   */
  textSearch(opts: TextSearchOptions): Promise<Hit[]> {
    return this.searchRequest("/text-search", {
      ...(opts.clauses
        ? { clauses: opts.clauses, combine: opts.combine }
        : { field: opts.field, query: opts.query }),
      scope: opts.scope ?? [],
      top_k: opts.topK,
      offset: opts.offset,
      min_score: opts.minScore,
      filter: opts.filter ?? [],
      explain: opts.explain,
      highlight: encodeHighlight(opts.highlight),
      include_attributes: opts.includeAttributes,
      exclude_attributes: opts.excludeAttributes,
      rank_by: encodeRankBy(opts.rankBy),
      limit_per: opts.limitPer,
    });
  }

  /**
   * Hybrid search: fuse a vector query and a BM25 text query via RRF. The text leg takes
   * the same single-field / `clauses` choice as {@link NidusClient.textSearch}.
   */
  hybridSearch(opts: HybridSearchOptions): Promise<Hit[]> {
    return this.searchRequest("/hybrid-search", {
      vector: opts.vector,
      ...(opts.clauses
        ? { clauses: opts.clauses, combine: opts.combine }
        : { field: opts.field, text: opts.text }),
      scope: opts.scope ?? [],
      top_k: opts.topK,
      offset: opts.offset,
      filter: opts.filter ?? [],
      rrf_k: opts.rrfK,
      candidates: opts.candidates,
      explain: opts.explain,
      highlight: encodeHighlight(opts.highlight),
      vector_weight: opts.vectorWeight,
      text_weight: opts.textWeight,
    });
  }

  /** Metadata-only listing (no vector), paginated by `offset`/`limit`. */
  list(opts: ListOptions = {}): Promise<Hit[]> {
    return this.searchRequest("/list", {
      scope: opts.scope ?? [],
      offset: opts.offset,
      limit: opts.limit,
      filter: opts.filter ?? [],
      include_attributes: opts.includeAttributes,
      exclude_attributes: opts.excludeAttributes,
      order_by: opts.orderBy,
    });
  }

  /**
   * Count the records matching a filter and sum the named attributes. Answered from the
   * in-RAM index alone — no record is built and no vector is read.
   */
  async aggregate(opts: AggregateOptions = {}): Promise<Aggregation> {
    const res = await this.request<RawAggregation>(
      "POST",
      "/aggregate",
      prune({
        scope: opts.scope ?? [],
        filter: opts.filter ?? [],
        sum: opts.sum ?? [],
        group_by: opts.groupBy,
      }),
    );
    // Every sum is an `Int` or a `Float`, both of which decode to a JS number.
    return {
      count: res.count,
      sums: decodeAttrs(res.sums) as Record<string, number>,
      // Kept absent, not `undefined`, so an ungrouped answer is the shape it always was.
      ...(res.groups
        ? {
            groups: res.groups.map((g) => ({
              value: g.value === null ? null : decodeValue(g.value),
              count: g.count,
              sums: decodeAttrs(g.sums) as Record<string, number>,
            })),
          }
        : {}),
      ...(res.groups_truncated ? { groupsTruncated: true } : {}),
    };
  }

  /**
   * Answer several vector queries in one round-trip (16 max). Returns one ranking per
   * query in request order, or — with `opts.fuse` — a single array holding the one fused
   * ranking, so the return shape is uniform either way.
   *
   * The server validates the whole batch before running any leg, so a malformed query
   * fails the call rather than returning a partial answer that cannot be told apart.
   */
  async batchSearch(opts: BatchSearchOptions): Promise<Hit[][]> {
    const body = prune({
      queries: opts.queries.map((q) => ({
        query: q.query,
        scope: q.scope ?? [],
        top_k: q.topK,
        offset: q.offset,
        min_score: q.minScore,
        filter: q.filter ?? [],
        exact: q.exact,
        include_attributes: q.includeAttributes,
        exclude_attributes: q.excludeAttributes,
        rank_by: encodeRankBy(q.rankBy),
        limit_per: q.limitPer,
      })),
      fuse: opts.fuse
        ? prune({
            rrf_k: opts.fuse.rrfK,
            weights: opts.fuse.weights,
            top_k: opts.fuse.topK,
          })
        : undefined,
    });
    const res = await this.request<RawBatchSearch>(
      "POST",
      "/search/batch",
      body,
    );
    return (res.fused ? [res.fused] : (res.results ?? [])).map((hits) =>
      hits.map((h) => this.decodeHit(h)),
    );
  }

  // ── Memory (text-native) ──────────────────────────────────────────────────
  //
  // Available only when `nidus serve` was started with an embedder
  // (`--embed-provider …`); otherwise these answer `400`. The server embeds the
  // text/query — the client only sends strings.

  /**
   * Embed `text` and upsert it under `id` in `collection` (idempotent on `id`).
   * With `opts.mode === "summarize"` the server summarizes first, embeds the
   * summary, and stamps a `nidus.summary` attr (requires the server to have a
   * summarizer). The raw text is always stored under `nidus.text`. `opts.attrs` accept plain JS values or `v.*`
   * helpers; they are normalized for you.
   *
   * Read `id` off the result rather than assuming the one you passed:
   * `opts.dedupeThreshold` can redirect the write onto a near-duplicate.
   */
  async remember(
    collection: string,
    id: string,
    text: string,
    opts: RememberOptions = {},
  ): Promise<RememberResult> {
    const res = await this.request<Partial<RememberResult> | undefined>(
      "POST",
      `/collections/${enc(collection)}/remember`,
      prune({
        id,
        text,
        mode: opts.mode,
        attrs: opts.attrs ? encodeAttrs(opts.attrs) : undefined,
        ttl_seconds: opts.ttlSeconds,
        dedupe_threshold: opts.dedupeThreshold,
      }),
    );
    // A server predating the echoed fields answers `{ok, upserted}`; falling back to the
    // requested id keeps that case honest instead of reporting `undefined` as the target.
    return {
      id: res?.id ?? id,
      upserted: res?.upserted ?? 0,
      deduped: res?.deduped ?? false,
    };
  }

  /**
   * Embed `query` and vector-search `collection`, best-first (attrs decoded to
   * plain JS values). Refused with a cross-model guard if the collection was
   * written with a different embedder than the server's.
   */
  recall(
    collection: string,
    query: string,
    opts: RecallOptions = {},
  ): Promise<Hit[]> {
    return this.searchRequest(`/collections/${enc(collection)}/recall`, {
      query,
      top_k: opts.topK,
      min_score: opts.minScore,
      filter: opts.filter ?? [],
    });
  }

  // ── Maintenance ───────────────────────────────────────────────────────────

  /** Force a durability flush. */
  async flush(): Promise<void> {
    await this.request("POST", "/flush", {});
  }

  /** Compact the store (reclaim space from deleted/overwritten rows). */
  async compact(): Promise<void> {
    await this.request("POST", "/compact", {});
  }

  /** Adopt a writer's newer committed state. Returns whether anything was adopted. */
  async refresh(): Promise<boolean> {
    const res = await this.request<{ adopted: boolean }>("POST", "/refresh", {});
    return res.adopted;
  }

  // ── Internals ─────────────────────────────────────────────────────────────

  /** Run a search-family request and decode the resulting hits' attrs. */
  private async searchRequest(
    path: string,
    body: Record<string, unknown>,
  ): Promise<Hit[]> {
    const hits = await this.request<RawHit[]>("POST", path, prune(body));
    return hits.map((h) => this.decodeHit(h));
  }

  /** One wire hit into a {@link Hit}. Shared so every search surface decodes identically. */
  private decodeHit(h: RawHit): Hit {
    return {
      collection: h.collection,
      id: h.id,
      score: h.score,
      attrs: decodeAttrs(h.attrs),
      // Kept absent, not `undefined`, so an unannotated hit is the shape it always was.
      ...(h.annotations
        ? { annotations: decodeAnnotations(h.annotations) }
        : {}),
    };
  }

  /** Issue a request and parse a JSON body, mapping a non-2xx to {@link NidusError}. */
  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const res = await this.raw(method, path, body);
    const text = await res.text();
    if (!res.ok) {
      throw new NidusError(extractError(text, res.status), res.status);
    }
    return (text ? JSON.parse(text) : undefined) as T;
  }

  /** The bare transport: headers, auth, timeout, and transport-error mapping. */
  private async raw(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<Response> {
    const headers: Record<string, string> = { ...this.extraHeaders };
    if (this.token) headers.authorization = `Bearer ${this.token}`;
    let payload: string | undefined;
    if (body !== undefined) {
      headers["content-type"] = "application/json";
      payload = JSON.stringify(body);
    }

    const controller = this.timeoutMs > 0 ? new AbortController() : undefined;
    const timer =
      controller && this.timeoutMs > 0
        ? setTimeout(() => controller.abort(), this.timeoutMs)
        : undefined;
    try {
      return await this.doFetch(`${this.baseUrl}${path}`, {
        method,
        headers,
        body: payload,
        signal: controller?.signal,
      });
    } catch (err) {
      const reason =
        (controller?.signal.aborted ?? false)
          ? `request to ${path} timed out after ${this.timeoutMs}ms`
          : `request to ${path} failed: ${(err as Error).message}`;
      throw new NidusError(reason, 0);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
}

/** A hit as it arrives on the wire, before attrs are decoded. */
interface RawHit {
  collection: string;
  id: string;
  score: number;
  attrs: Record<string, Value>;
  annotations?: WireAnnotations;
}

/** An `/aggregate` response, whose sums arrive as tagged {@link Value}s. */
interface RawAggregation {
  count: number;
  sums: Record<string, Value>;
  groups?: {
    value: Value | null;
    count: number;
    sums: Record<string, Value>;
  }[];
  groups_truncated?: boolean;
}

/** A `/search/batch` response: exactly one of the two fields is present. */
interface RawBatchSearch {
  results?: RawHit[][];
  fused?: RawHit[];
}

/** Encode `rankBy` to its externally-tagged wire form, dropping the knobs left unset. */
function encodeRankBy(rank: RankBy | undefined): unknown {
  if (!rank) return undefined;
  const d = rank.decay;
  return {
    Decay: prune({
      field: d.field,
      origin: d.origin instanceof Date ? d.origin.getTime() : d.origin,
      scale: d.scale,
      decay: d.decay,
      lambda: d.lambda,
      missing: d.missing,
    }),
  };
}

/** Encode `highlight`: `true` is the empty object the server reads as "all defaults". */
function encodeHighlight(h: boolean | HighlightOptions | undefined): unknown {
  if (h === undefined || h === false) return undefined;
  if (h === true) return {};
  return prune({
    max_fragments: h.maxFragments,
    fragment_chars: h.fragmentChars,
  });
}

/** Path-segment encode a collection name (allows slashes/spaces in names). */
function enc(name: string): string {
  return encodeURIComponent(name);
}

/**
 * Encode one `setFtsSchema` field. A string passes through as the server's bare-name
 * form; an object becomes the snake_case body, pruned so an unset knob keeps the
 * server's default rather than being sent as `undefined`.
 */
function encodeFtsField(f: string | FtsField): unknown {
  if (typeof f === "string") return f;
  return prune({
    field: f.field,
    k1: f.k1,
    b: f.b,
    language: f.language,
    ascii_folding: f.asciiFolding,
    max_token_len: f.maxTokenLen,
  });
}

/**
 * Encode one `setFilterIndex` field, on the same bare-name-or-object rule as
 * {@link encodeFtsField}. Pruning matters here: the server defaults both structures to
 * `true`, so sending an explicit `undefined` would be indistinguishable from `false`.
 */
function encodeFilterIndexField(f: string | FilterIndexField): unknown {
  if (typeof f === "string") return f;
  return prune({
    field: f.field,
    tokens: f.tokens,
    trigrams: f.trigrams,
  });
}

/** Drop `undefined` fields so server `#[serde(default)]`s apply instead. */
function prune(body: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, val] of Object.entries(body)) {
    if (val !== undefined) out[k] = val;
  }
  return out;
}

/** Pull the `{ "error": … }` message out of a failed response, or fall back. */
function extractError(text: string, status: number): string {
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed.error === "string") return parsed.error;
  } catch {
    // not JSON — fall through
  }
  return text || `HTTP ${status}`;
}
