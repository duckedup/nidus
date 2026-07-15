//! `NidusClient` — a remote client over the `nidus serve` HTTP API.
//
// One method per endpoint (`src/server/mod.rs`). "Local vs remote" is just the
// base URL: point at a local `nidus serve` or any reachable host. Built on the
// platform-global `fetch`, so it runs unchanged on Node 18+, Deno, Bun, Cloudflare
// Workers, and browsers — with no runtime dependencies.

import { NidusError } from "./errors.js";
import type {
  DecodedRecord,
  Filter,
  Hit,
  HybridSearchOptions,
  ListOptions,
  NidusRecord,
  RecallOptions,
  RecordInput,
  RememberOptions,
  SearchOptions,
  Stats,
  TextSearchOptions,
  Value,
} from "./types.js";
import { decodeAttrs, encodeAttrs } from "./values.js";

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

  /** Declare the full-text-indexed attribute fields for a collection. */
  async setFtsSchema(name: string, fields: string[]): Promise<void> {
    await this.request("POST", `/collections/${enc(name)}/fts-schema`, {
      fields,
    });
  }

  // ── Search ──────────────────────────────────────────────────────────────

  /** Vector (cosine) nearest-neighbour search. Empty `scope` searches all collections. */
  search(opts: SearchOptions): Promise<Hit[]> {
    return this.searchRequest("/search", {
      query: opts.query,
      scope: opts.scope ?? [],
      top_k: opts.topK,
      min_score: opts.minScore,
      filter: opts.filter ?? [],
    });
  }

  /** BM25 full-text search over one indexed field. */
  textSearch(opts: TextSearchOptions): Promise<Hit[]> {
    return this.searchRequest("/text-search", {
      field: opts.field,
      query: opts.query,
      scope: opts.scope ?? [],
      top_k: opts.topK,
      min_score: opts.minScore,
      filter: opts.filter ?? [],
    });
  }

  /** Hybrid search: fuse a vector query and a BM25 text query via RRF. */
  hybridSearch(opts: HybridSearchOptions): Promise<Hit[]> {
    return this.searchRequest("/hybrid-search", {
      vector: opts.vector,
      field: opts.field,
      text: opts.text,
      scope: opts.scope ?? [],
      top_k: opts.topK,
      filter: opts.filter ?? [],
      rrf_k: opts.rrfK,
      candidates: opts.candidates,
    });
  }

  /** Metadata-only listing (no vector), paginated by `offset`/`limit`. */
  list(opts: ListOptions = {}): Promise<Hit[]> {
    return this.searchRequest("/list", {
      scope: opts.scope ?? [],
      offset: opts.offset,
      limit: opts.limit,
      filter: opts.filter ?? [],
    });
  }

  // ── Memory (text-native) ──────────────────────────────────────────────────
  //
  // Available only when `nidus serve` was started with an embedder
  // (`--embed-provider …`); otherwise these answer `400`. The server embeds the
  // text/query — the client only sends strings.

  /**
   * Embed `text` and upsert it under `id` in `collection` (idempotent on `id`).
   * With `opts.mode === "summarize"` the server summarizes first, embeds the
   * summary, and stamps `nidus.summary`/`nidus.source` attrs (requires the
   * server to have a summarizer). `opts.attrs` accept plain JS values or `v.*`
   * helpers; they are normalized for you.
   */
  async remember(
    collection: string,
    id: string,
    text: string,
    opts: RememberOptions = {},
  ): Promise<void> {
    await this.request(
      "POST",
      `/collections/${enc(collection)}/remember`,
      prune({
        id,
        text,
        mode: opts.mode,
        attrs: opts.attrs ? encodeAttrs(opts.attrs) : undefined,
      }),
    );
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

  // ── Internals ─────────────────────────────────────────────────────────────

  /** Run a search-family request and decode the resulting hits' attrs. */
  private async searchRequest(
    path: string,
    body: Record<string, unknown>,
  ): Promise<Hit[]> {
    const hits = await this.request<RawHit[]>("POST", path, prune(body));
    return hits.map((h) => ({
      collection: h.collection,
      id: h.id,
      score: h.score,
      attrs: decodeAttrs(h.attrs),
    }));
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

    const controller =
      this.timeoutMs > 0 ? new AbortController() : undefined;
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
        controller?.signal.aborted ?? false
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
}

/** Path-segment encode a collection name (allows slashes/spaces in names). */
function enc(name: string): string {
  return encodeURIComponent(name);
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
