// Client — one method per endpoint of a running `nidus serve`, plus its options.
//
// The surface mirrors the JavaScript SDK (sdks/js/src/client.ts) endpoint for
// endpoint, deliberately: the SDKs are meant to be reviewable side by side, so a
// method exists here if and only if it exists there. That is also why /ready,
// /cluster, /refresh and /metrics are absent even though the server routes them —
// wrapping them is a change to all the SDKs at once, not a favour this one does
// alone.
//
// The one place this SDK diverges in *shape*: every method takes a context.Context
// first and returns (T, error). Cancellation and deadlines belong to the caller, and
// idiomatic Go beats a literal transliteration of a Promise API.
//
// "Local vs remote" is just the base URL — a `nidus serve` on your laptop and one
// across the network are the same code.

package nidus

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// A Client is a handle on a nidus server.
//
// It is safe for concurrent use: it carries no per-request state, and an http.Client
// is itself concurrent. One Client per server, shared, is the intended shape — that is
// what keeps connections pooled.
type Client struct {
	baseURL string
	token   string
	hc      *http.Client
	timeout time.Duration
	headers map[string]string
}

// An Option configures a [Client] at construction. See [WithToken],
// [WithHTTPClient], [WithTimeout], and [WithHeader].
type Option func(*Client)

// NewClient returns a client for the nidus server at baseURL, e.g.
// "http://127.0.0.1:7700". Trailing slashes are stripped, so a base URL pasted from a
// browser does not produce "//health".
//
// It fails on a base URL that is empty or that no request could be built from — a
// missing scheme is the common one ("127.0.0.1:7700" parses, but as a URL with the
// scheme "127.0.0.1"). Failing here rather than on the first call means the mistake
// surfaces where it was made.
//
// A query string or fragment is rejected for the same reason, and it is the subtler
// mistake: the base URL is concatenated with each endpoint's path, so
// "http://h/?x=1" + "/health" parses as the path "/" with the query "x=1/health" —
// every request for the client's whole lifetime addresses the wrong route, and the
// server's 404 reads at the call site as "no such collection".
func NewClient(baseURL string, opts ...Option) (*Client, error) {
	trimmed := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if trimmed == "" {
		return nil, fmt.Errorf("nidus: NewClient needs a base URL, e.g. http://127.0.0.1:7700")
	}
	parsed, err := url.Parse(trimmed)
	if err != nil {
		return nil, fmt.Errorf("nidus: base URL %q is not a URL: %w", baseURL, err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf(
			"nidus: base URL %q needs a scheme and a host, e.g. http://127.0.0.1:7700", baseURL,
		)
	}
	// Name the offending part: "not a base URL" alone leaves the caller staring at a
	// string that looks perfectly fine.
	switch {
	case parsed.RawQuery != "" || parsed.ForceQuery:
		return nil, fmt.Errorf(
			"nidus: base URL %q carries a query string (%q); a base URL is scheme, host and "+
				"optional path only — each endpoint's path is appended to it",
			baseURL, parsed.RawQuery,
		)
	case parsed.Fragment != "":
		return nil, fmt.Errorf(
			"nidus: base URL %q carries a fragment (%q); a base URL is scheme, host and "+
				"optional path only", baseURL, parsed.Fragment,
		)
	case parsed.Opaque != "":
		return nil, fmt.Errorf(
			"nidus: base URL %q is not hierarchical (opaque part %q); want e.g. "+
				"http://127.0.0.1:7700", baseURL, parsed.Opaque,
		)
	}

	c := &Client{
		baseURL: trimmed,
		// A client of our own, never http.DefaultClient: a library that reaches into
		// the process-global client — to set a Timeout, say — changes the behaviour of
		// every other package in the binary, from a place nobody thinks to look. The
		// zero-value http.Client still uses http.DefaultTransport, so this shares the
		// standard connection pool instead of opening a second one; pooling is a
		// process-wide concern, policy is not. No hc.Timeout either: the per-request
		// timeout is a context deadline (transport.go), which composes with a caller's.
		hc: &http.Client{},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c, nil
}

// WithToken authenticates every request with a bearer token — the value passed to
// `nidus serve --token`.
func WithToken(token string) Option {
	return func(c *Client) { c.token = token }
}

// WithHTTPClient supplies the http.Client to use, for callers who need their own
// transport, proxy, TLS configuration, retry wrapper, or instrumentation. A nil
// argument is ignored rather than stored, since a nil client can serve no request and
// the resulting panic would point at the wrong line.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		if hc != nil {
			c.hc = hc
		}
	}
}

// WithTimeout bounds each request. It is applied as a context deadline per request, so
// it composes with a caller's own context rather than overriding it: whichever
// deadline is earlier wins. Zero or negative means no SDK-imposed timeout.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) { c.timeout = d }
}

// WithHeader adds a header sent on every request — a trace id, a gateway's own
// credential, a tenant tag. Call it once per header.
//
// It cannot displace Authorization or Content-Type: the SDK sets those last, on
// purpose, so its contract with the server survives a typo here.
func WithHeader(key, value string) Option {
	return func(c *Client) {
		if c.headers == nil {
			c.headers = make(map[string]string)
		}
		c.headers[key] = value
	}
}

// ── Admin and introspection ─────────────────────────────────────────────────

// Ping calls /health and returns why it failed, or nil when the server is up.
//
// It is the diagnosing half of [Client.Health]: a typo'd port, a wrong scheme, a TLS
// failure, a cancelled context and a genuine 503 are all "not up", but they are not
// the same problem, and a readiness loop that gives up should be able to say which
// one it was. The error is an [*Error] like every other method's, so Status
// distinguishes "nidus said no" from "nidus was not reachable".
//
// /health needs no token — the server exempts it, so an orchestrator does not read a
// 401 as "not ready".
func (c *Client) Ping(ctx context.Context) error {
	return c.request(ctx, http.MethodGet, "/health", nil, nil)
}

// Health reports whether the server answers /health.
//
// It returns a bare bool rather than (bool, error), matching the JS SDK, because a
// caller asking "is it up" has one thing to branch on and nothing it would do
// differently between an unreachable host and a 503. Use [Client.Ping] when the
// reason matters — this is deliberately the convenient wrapper over it, not a place
// where a diagnosis is thrown away for want of anywhere to put it.
func (c *Client) Health(ctx context.Context) bool {
	return c.Ping(ctx) == nil
}

// Stats reads store-wide introspection: dimension, distance metric, ANN
// configuration, collection names, and the footprint. Stats.Ann is nil when the store
// does exact brute-force search, which is the default rather than a fault.
func (c *Client) Stats(ctx context.Context) (*Stats, error) {
	var out Stats
	if err := c.request(ctx, http.MethodGet, "/stats", nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Collections lists every collection name.
func (c *Client) Collections(ctx context.Context) ([]string, error) {
	var out []string
	if err := c.request(ctx, http.MethodGet, "/collections", nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// CreateCollection creates a collection. It is idempotent — creating one that already
// exists succeeds and changes nothing — so callers need no exists-check race.
func (c *Client) CreateCollection(ctx context.Context, name string) error {
	// An empty object, not an absent body: the route is a POST, and sending the
	// Content-Type with `{}` keeps it a well-formed JSON request everywhere in between.
	return c.request(ctx, http.MethodPost, collPath(name, ""), struct{}{}, nil)
}

// DropCollection removes a collection and every record in it.
func (c *Client) DropCollection(ctx context.Context, name string) error {
	return c.request(ctx, http.MethodDelete, collPath(name, ""), nil, nil)
}

// GetMeta reads a collection's free-form string metadata.
func (c *Client) GetMeta(ctx context.Context, name string) (map[string]string, error) {
	var out map[string]string
	if err := c.request(ctx, http.MethodGet, collPath(name, "/meta"), nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// SetMeta replaces a collection's metadata wholesale — it is not a merge, so read,
// change, and write back if you mean to keep the other keys.
func (c *Client) SetMeta(ctx context.Context, name string, meta map[string]string) error {
	// A nil map would marshal to `null`, which the server's map field rejects outright
	// (serde reads null as a type error, not as "empty"). Send `{}` so "clear the
	// metadata" is expressible rather than a 400.
	if meta == nil {
		meta = map[string]string{}
	}
	return c.request(ctx, http.MethodPut, collPath(name, "/meta"), meta, nil)
}

// ── Data ────────────────────────────────────────────────────────────────────

// Upsert inserts or replaces records, keyed by [Record.ID] within the collection, and
// returns how many were written. Re-sending the same id overwrites rather than
// duplicating, so a retried batch is safe.
//
// A Record with a nil Vector is a text-only document: it occupies no row in the vector
// matrix and never appears in a vector search, but is findable by [Client.TextSearch]
// and [Client.List].
//
// A Record with a non-nil but *empty* Vector is refused, because Go's `omitempty`
// cannot encode it: an empty slice marshals byte-identically to an absent one, so it
// would silently become a text-only document instead of the dimension mismatch the
// server would have reported. The realistic source is an embedder that returned an
// empty slice, and the last thing that caller wants is a stored document invisible to
// every vector search.
func (c *Client) Upsert(ctx context.Context, name string, records []Record) (int, error) {
	for i, r := range records {
		if r.Vector != nil && len(r.Vector) == 0 {
			return 0, fmt.Errorf(
				"nidus: record %d (id %q) has an empty vector: an empty vector is neither an "+
					"embedding nor a text-only document — pass a nil Vector for text-only", i, r.ID,
			)
		}
	}
	// Nil becomes an empty slice for the same reason as SetMeta's map: `records: null`
	// is a deserialization error on the server, where `[]` is a lawful no-op.
	if records == nil {
		records = []Record{}
	}
	body := struct {
		Records []Record `json:"records"`
	}{records}

	var out struct {
		Upserted int `json:"upserted"`
	}
	if err := c.request(ctx, http.MethodPost, collPath(name, "/upsert"), body, &out); err != nil {
		return 0, err
	}
	return out.Upserted, nil
}

// Delete removes records by id and returns how many were deleted. Ids that are not
// present are not an error; they simply do not count.
func (c *Client) Delete(ctx context.Context, name string, ids []string) (int, error) {
	if ids == nil {
		ids = []string{}
	}
	return c.deleted(ctx, name, struct {
		IDs []string `json:"ids"`
	}{ids})
}

// DeleteWhere removes every record matching filter and returns how many were deleted.
//
// Note what an empty filter means: it matches everything, so DeleteWhere with no
// predicates empties the collection. That is the server's semantics and this SDK does
// not second-guess it — but it is worth a guard at the call site.
func (c *Client) DeleteWhere(ctx context.Context, name string, filter Filter) (int, error) {
	// Delete-by-id and delete-by-filter are the same endpoint; the server takes the
	// filter branch whenever the field is present, which is why these are two methods
	// rather than one struct with both fields set.
	return c.deleted(ctx, name, struct {
		Filter Filter `json:"filter"`
	}{filter})
}

// Records fetches every record in a collection. Vector is nil for a text-only
// document — the server omits the field entirely, keeping "no embedding"
// distinguishable from "an empty one".
func (c *Client) Records(ctx context.Context, name string) ([]Record, error) {
	var out []Record
	if err := c.request(ctx, http.MethodGet, collPath(name, "/records"), nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// SetFtsSchema declares which attribute fields are full-text indexed, which is what
// makes [Client.TextSearch] and [Client.HybridSearch] able to see them. Fields already
// written are indexed as part of applying the schema.
//
// Every field takes the server's BM25 and analyzer defaults. Use
// [Client.SetFtsFields] to tune k1, b, or the analyzer per field.
func (c *Client) SetFtsSchema(ctx context.Context, name string, fields []string) error {
	if fields == nil {
		fields = []string{}
	}
	body := struct {
		Fields []string `json:"fields"`
	}{fields}
	return c.request(ctx, http.MethodPost, collPath(name, "/fts-schema"), body, nil)
}

// SetFtsFields is [Client.SetFtsSchema] with per-field BM25 and analyzer tuning. It
// hits the same endpoint: the server accepts a bare field name or a field object, and
// an [FtsField] whose knobs are all unset encodes to the same defaults.
func (c *Client) SetFtsFields(ctx context.Context, name string, fields []FtsField) error {
	if fields == nil {
		fields = []FtsField{}
	}
	body := struct {
		Fields []FtsField `json:"fields"`
	}{fields}
	return c.request(ctx, http.MethodPost, collPath(name, "/fts-schema"), body, nil)
}

// ── Search ──────────────────────────────────────────────────────────────────

// Search runs a vector nearest-neighbour query, best-first.
//
// An empty [SearchRequest.Scope] searches every collection and merges the results into
// one ranking — sound because a store has a single embedding space. Leave TopK zero to
// take the server's default rather than asking for zero results; see the note in
// types.go on the omit-vs-zero trap. Offset skips that many top-ranked hits, so
// successive pages tile the ranking; Offset+TopK may not exceed 10000.
func (c *Client) Search(ctx context.Context, req SearchRequest) ([]Hit, error) {
	return c.hits(ctx, "/search", req)
}

// TextSearch runs a BM25 full-text query over one field declared with
// [Client.SetFtsSchema]. Scores are raw BM25, not cosine: unbounded above and not
// comparable between queries.
func (c *Client) TextSearch(ctx context.Context, req TextSearchRequest) ([]Hit, error) {
	return c.hits(ctx, "/text-search", req)
}

// HybridSearch fuses a vector query and a BM25 text query with reciprocal rank
// fusion, so a document that ranks well on either leg surfaces. The returned score is
// the fused RRF score — a rank-derived number, not a similarity, and not comparable
// with a [Client.Search] score.
func (c *Client) HybridSearch(ctx context.Context, req HybridSearchRequest) ([]Hit, error) {
	return c.hits(ctx, "/hybrid-search", req)
}

// List returns records by metadata alone — no query vector — paginated by Offset and
// Limit, in storage order unless ListRequest.OrderBy says otherwise. Hit.Score is not
// meaningful here; there is nothing being scored.
func (c *Client) List(ctx context.Context, req ListRequest) ([]Hit, error) {
	return c.hits(ctx, "/list", req)
}

// Aggregate counts the records a filter matches and sums the attributes named in
// AggregateRequest.Sum. It is answered from the server's in-RAM index alone — no
// record is materialized and no vector is read — so it stays cheap over a whole store.
//
// The zero request counts every record in every collection.
func (c *Client) Aggregate(ctx context.Context, req AggregateRequest) (*Aggregation, error) {
	var out Aggregation
	if err := c.request(ctx, http.MethodPost, "/aggregate", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ── Memory (text in, text out) ──────────────────────────────────────────────
//
// These two need a server that both *has* the routes and was started with an
// embedder, and the two failures look different on the wire:
//
//   - 404 with no body, when the binary was built without the `memory` feature. The
//     routes are registered behind #[cfg(feature = "memory")] (src/server/mod.rs), so
//     a plain `cli` build — which is what `just build-cli` and the SDK's own
//     integration harness produce — does not have them at all.
//   - 400, when the routes exist but the server was started without
//     --embed-provider. The message names the flag.
//
// Both are the server's business, so these are wrapped unconditionally and the error
// is left to surface — an SDK that hid them would have to know a server's build
// features and launch flags to be correct.

// Remember embeds text server-side and upserts it under id, idempotent on id. The
// client sends only strings; the embedding never crosses the wire.
//
// With opts.Mode == "summarize" the server summarizes first, embeds the summary, and
// stamps nidus.summary/nidus.source attrs so a later recall is explainable back to the
// source text — that path additionally needs a server started with a summarizer.
func (c *Client) Remember(
	ctx context.Context, collection, id, text string, opts RememberOptions,
) error {
	return c.request(
		ctx, http.MethodPost, collPath(collection, "/remember"), opts.wire(id, text), nil,
	)
}

// Recall embeds query server-side and vector-searches the collection, best-first.
//
// A collection written with one embedding model and recalled against a server
// configured with another is refused (409) rather than silently answered with
// meaningless scores.
func (c *Client) Recall(
	ctx context.Context, collection, query string, opts RecallOptions,
) ([]Hit, error) {
	return c.hits(ctx, collPath(collection, "/recall"), opts.wire(query))
}

// ── Maintenance ─────────────────────────────────────────────────────────────

// Flush forces a durability flush. Writes are already durable per batch, so this is
// for the moments where a caller wants that guarantee at a point of their choosing —
// before a snapshot, or on a clean shutdown path.
func (c *Client) Flush(ctx context.Context) error {
	return c.request(ctx, http.MethodPost, "/flush", struct{}{}, nil)
}

// Compact rewrites the store to reclaim the rows left behind by deletes and
// overwrites (Stats.Footprint.DeadRows counts them). It is a whole-store operation
// that holds the writer, so schedule it rather than calling it per request.
func (c *Client) Compact(ctx context.Context) error {
	return c.request(ctx, http.MethodPost, "/compact", struct{}{}, nil)
}

// ── Internals ───────────────────────────────────────────────────────────────

// hits runs a search-family request and decodes the ranked rows.
//
// Unlike the JS SDK there is no attr-decoding step here: [Hit] keeps typed [Value]s,
// which [Value.UnmarshalJSON] produces directly. See the note on [Attrs] for why the
// Go SDK stops there instead of flattening to a loose map.
func (c *Client) hits(ctx context.Context, path string, body any) ([]Hit, error) {
	var out []Hit
	if err := c.request(ctx, http.MethodPost, path, body, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// deleted posts a body to a collection's /delete route and returns the count, shared
// by [Client.Delete] and [Client.DeleteWhere] since only the body differs.
func (c *Client) deleted(ctx context.Context, name string, body any) (int, error) {
	var out struct {
		Deleted int `json:"deleted"`
	}
	if err := c.request(ctx, http.MethodPost, collPath(name, "/delete"), body, &out); err != nil {
		return 0, err
	}
	return out.Deleted, nil
}
