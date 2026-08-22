// Tests for the client and its single request path, against net/http/httptest.
//
// No network, no binary: every test here drives a real HTTP round trip over a
// loopback listener, which is enough to exercise the parts of transport.go that a
// hand-rolled fake would paper over — URL escaping as the http package actually
// serializes it, header precedence, status handling, body draining.
//
// The assertions on request bodies are on *bytes* rather than on a re-decoded struct,
// deliberately. The bug this SDK is most exposed to is an omit-vs-zero mistake: the
// server fills unset fields from #[serde(default)] (top_k = 10, limit = 100), so
// sending "top_k": 0 is a request for zero results. A test that decodes the body back
// into a struct cannot see the difference between an absent field and a zero one —
// which is the only thing that matters — so it has to read the JSON.
package nidus

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// ── Harness ─────────────────────────────────────────────────────────────────

// capture is a fake nidus server: it records what it received and replies with a
// canned status and body. One instance serves one test.
type capture struct {
	mu     sync.Mutex
	calls  int
	method string
	path   string // EscapedPath, so a percent-escaped collection name stays visible
	body   []byte
	header http.Header

	status int    // 0 means 200
	reply  string // "" means an empty body
}

func (c *capture) handler(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	c.mu.Lock()
	c.calls++
	c.method, c.path = r.Method, r.URL.EscapedPath()
	c.body, c.header = body, r.Header.Clone()
	status, reply := c.status, c.reply
	c.mu.Unlock()

	if status == 0 {
		status = http.StatusOK
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if reply != "" {
		_, _ = io.WriteString(w, reply)
	}
}

// serve starts the fake server and returns a client pointed at it. The listener and
// the client are torn down by t.Cleanup, so no test has to remember to.
func serve(t *testing.T, c *capture, opts ...Option) *Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(c.handler))
	t.Cleanup(srv.Close)
	db, err := NewClient(srv.URL, opts...)
	if err != nil {
		t.Fatalf("NewClient(%q) failed: %v", srv.URL, err)
	}
	return db
}

// recorded is a copy of what the fake saw — the one way a test reads those fields.
//
// A copy rather than a locked read at each assertion, and a separate type rather than
// a copy of capture, because capture holds the mutex and `go vet`'s copylocks check is
// right to refuse copying that. Every field below is written under c.mu by the handler
// while the test goroutine reads, so an unlocked read is a data race even where
// net/http happens to supply a happens-before edge today; one accessor means no test
// can quietly opt out of the lock.
type recorded struct {
	calls  int
	method string
	path   string
	body   []byte
	header http.Header
}

func (c *capture) snapshot() recorded {
	c.mu.Lock()
	defer c.mu.Unlock()
	return recorded{
		calls:  c.calls,
		method: c.method,
		path:   c.path,
		body:   c.body,
		header: c.header,
	}
}

// sentBody returns the request body the fake server saw, as a string, failing the
// test if nothing was sent at all.
func (c *capture) sentBody(t *testing.T) string {
	t.Helper()
	got := c.snapshot()
	if got.calls == 0 {
		t.Fatal("no request reached the server")
	}
	return string(got.body)
}

// f32 and iptr make the pointers the request structs take for the knobs whose zero the
// server treats as a real value.
func f32(v float32) *float32 { return &v }
func iptr(v int) *int        { return &v }
func i64(v int64) *int64     { return &v }

// ── Batch search and grouped aggregation ────────────────────────────────────

// TestBatchSearchFusedReturnsTheMergedList pins the one place BatchSearch's return shape
// depends on the request: the server answers a fused batch with "fused", not "results", so
// reading the wrong key would hand the caller an empty slice and no error.
func TestBatchSearchFusedReturnsTheMergedList(t *testing.T) {
	cap := &capture{reply: `{"fused":[{"collection":"docs","id":"a","score":0.5}]}`}
	db := serve(t, cap)

	out, err := db.BatchSearch(context.Background(), BatchSearchRequest{
		Queries: []SearchRequest{{Query: []float32{1, 0, 0}}, {Query: []float32{0, 1, 0}}},
		Fuse:    &BatchFuse{TopK: 5},
	})
	if err != nil {
		t.Fatalf("BatchSearch: %v", err)
	}
	if len(out) != 1 || len(out[0]) != 1 || out[0][0].ID != "a" {
		t.Fatalf("a fused batch must return one merged ranking, got %#v", out)
	}
	body := cap.sentBody(t)
	if !strings.Contains(body, `"fuse"`) || !strings.Contains(body, `"queries"`) {
		t.Fatalf("body must carry queries and fuse, got %s", body)
	}
}

// TestBatchSearchUnfusedReturnsOneListPerQuery is the other half: without Fuse the server
// answers with "results", one ranking per query, in request order.
func TestBatchSearchUnfusedReturnsOneListPerQuery(t *testing.T) {
	cap := &capture{reply: `{"results":[[{"collection":"docs","id":"a","score":1}],[]]}`}
	db := serve(t, cap)

	out, err := db.BatchSearch(context.Background(), BatchSearchRequest{
		Queries: []SearchRequest{{Query: []float32{1, 0, 0}}, {Query: []float32{0, 1, 0}}},
	})
	if err != nil {
		t.Fatalf("BatchSearch: %v", err)
	}
	if len(out) != 2 || len(out[0]) != 1 || len(out[1]) != 0 {
		t.Fatalf("want one list per query in order, got %#v", out)
	}
	if body := cap.sentBody(t); strings.Contains(body, "fuse") {
		t.Fatalf("an unfused batch must not send fuse, got %s", body)
	}
}

// TestAggregateGroupBySendsAndDecodesGroups covers the round trip: group_by is only sent
// when set (so an ungrouped call keeps the body it always had), and a null group Value
// decodes as nil — the records missing the attribute, distinct from a present null.
func TestAggregateGroupBySendsAndDecodesGroups(t *testing.T) {
	cap := &capture{reply: `{"count":3,"sums":{},"groups":[` +
		`{"value":{"Str":"rust"},"count":2,"sums":{"bytes":{"Int":8}}},` +
		`{"value":null,"count":1,"sums":{"bytes":{"Int":0}}}]}`}
	db := serve(t, cap)

	out, err := db.Aggregate(context.Background(), AggregateRequest{GroupBy: "lang"})
	if err != nil {
		t.Fatalf("Aggregate: %v", err)
	}
	if body := cap.sentBody(t); !strings.Contains(body, `"group_by":"lang"`) {
		t.Fatalf("group_by must reach the wire, got %s", body)
	}
	if len(out.Groups) != 2 {
		t.Fatalf("want 2 groups, got %#v", out.Groups)
	}
	if out.Groups[0].Value == nil {
		t.Fatal("first group must carry its value")
	}
	if got, ok := out.Groups[0].Value.Str(); !ok || got != "rust" {
		t.Fatalf("want Str(rust), got %v", out.Groups[0].Value)
	}
	if out.Groups[1].Value != nil {
		t.Fatalf("the attribute-less group must decode as nil, got %#v", out.Groups[1].Value)
	}
}

// TestAggregateWithoutGroupByOmitsIt keeps the ungrouped body byte-identical to what every
// pre-grouping release sent.
func TestAggregateWithoutGroupByOmitsIt(t *testing.T) {
	cap := &capture{reply: `{"count":0,"sums":{}}`}
	db := serve(t, cap)

	if _, err := db.Aggregate(context.Background(), AggregateRequest{}); err != nil {
		t.Fatalf("Aggregate: %v", err)
	}
	if body := cap.sentBody(t); strings.Contains(body, "group_by") {
		t.Fatalf("an ungrouped request must not send group_by, got %s", body)
	}
}

// ── Routing ─────────────────────────────────────────────────────────────────

// TestClientMethodsHitTheRightRoute drives every public method against the fake
// server and checks the HTTP method and path. A wrong path is the kind of mistake
// that only shows up as a 404 from a real server — which reads as "collection not
// found" and sends a caller looking in the wrong place entirely.
//
// The paths here are transcribed from the router in src/server/mod.rs; the reply
// column is whatever shape that endpoint's response decoder expects.
func TestClientMethodsHitTheRightRoute(t *testing.T) {
	ctx := context.Background()

	cases := []struct {
		method     string // the Client method this row covers
		reply      string
		wantMethod string
		wantPath   string
		call       func(*Client) error
	}{
		{"Health", `{}`, http.MethodGet, "/health", func(c *Client) error {
			if !c.Health(ctx) {
				return errors.New("Health reported down against a 200")
			}
			return nil
		}},
		{"Ping", `{}`, http.MethodGet, "/health", func(c *Client) error {
			return c.Ping(ctx)
		}},
		{"Ready", `{"ready":true,"role":"Solo","staleness_secs":0}`, http.MethodGet, "/ready", func(c *Client) error {
			_, err := c.Ready(ctx)
			return err
		}},
		{"Cluster", `{"role":"Solo","cluster":false,"holds_writer_handle":true,"fenced":false,"lease_owner":null,"commit_version":1,"staleness_secs":0,"max_staleness_secs":null}`, http.MethodGet, "/cluster", func(c *Client) error {
			_, err := c.Cluster(ctx)
			return err
		}},
		{"Versions", `{"commit_version":1,"oldest_readable":null,"pinned":null,"readable":[1]}`, http.MethodGet, "/versions", func(c *Client) error {
			_, err := c.Versions(ctx)
			return err
		}},
		{"Stats", `{"dimension":3}`, http.MethodGet, "/stats", func(c *Client) error {
			_, err := c.Stats(ctx)
			return err
		}},
		{"Collections", `["docs"]`, http.MethodGet, "/collections", func(c *Client) error {
			_, err := c.Collections(ctx)
			return err
		}},
		{"CreateCollection", `{"ok":true}`, http.MethodPost, "/collections/docs", func(c *Client) error {
			return c.CreateCollection(ctx, "docs")
		}},
		{"DropCollection", `{"ok":true}`, http.MethodDelete, "/collections/docs", func(c *Client) error {
			return c.DropCollection(ctx, "docs")
		}},
		{"GetMeta", `{"k":"v"}`, http.MethodGet, "/collections/docs/meta", func(c *Client) error {
			_, err := c.GetMeta(ctx, "docs")
			return err
		}},
		{"SetMeta", `{"ok":true}`, http.MethodPut, "/collections/docs/meta", func(c *Client) error {
			return c.SetMeta(ctx, "docs", map[string]string{"k": "v"})
		}},
		{"Upsert", `{"upserted":1}`, http.MethodPost, "/collections/docs/upsert", func(c *Client) error {
			_, err := c.Upsert(ctx, "docs", []Record{{ID: "a"}})
			return err
		}},
		{"Delete", `{"deleted":1}`, http.MethodPost, "/collections/docs/delete", func(c *Client) error {
			_, err := c.Delete(ctx, "docs", []string{"a"})
			return err
		}},
		{"DeleteWhere", `{"deleted":1}`, http.MethodPost, "/collections/docs/delete", func(c *Client) error {
			_, err := c.DeleteWhere(ctx, "docs", And(Eq("lang", "rust")))
			return err
		}},
		{"Records", `[]`, http.MethodGet, "/collections/docs/records", func(c *Client) error {
			_, err := c.Records(ctx, "docs")
			return err
		}},
		{"SetFtsSchema", `{"ok":true}`, http.MethodPost, "/collections/docs/fts-schema", func(c *Client) error {
			return c.SetFtsSchema(ctx, "docs", []string{"body"})
		}},
		{"SetFilterIndex", `{"ok":true}`, http.MethodPost, "/collections/docs/filter-index", func(c *Client) error {
			return c.SetFilterIndex(ctx, "docs", []string{"body"})
		}},
		{"SetFilterIndexFields", `{"ok":true}`, http.MethodPost, "/collections/docs/filter-index", func(c *Client) error {
			return c.SetFilterIndexFields(ctx, "docs", []FilterIndexField{{Field: "body"}})
		}},
		{"SetFtsFields", `{"ok":true}`, http.MethodPost, "/collections/docs/fts-schema", func(c *Client) error {
			return c.SetFtsFields(ctx, "docs", []FtsField{{Field: "body"}})
		}},
		{"Search", `[]`, http.MethodPost, "/search", func(c *Client) error {
			_, err := c.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}})
			return err
		}},
		{"BatchSearch", `{"results":[[]]}`, http.MethodPost, "/search/batch", func(c *Client) error {
			_, err := c.BatchSearch(ctx, BatchSearchRequest{
				Queries: []SearchRequest{{Query: []float32{1, 0, 0}}},
			})
			return err
		}},
		{"TextSearch", `[]`, http.MethodPost, "/text-search", func(c *Client) error {
			_, err := c.TextSearch(ctx, TextSearchRequest{Field: "body", Query: "fox"})
			return err
		}},
		{"HybridSearch", `[]`, http.MethodPost, "/hybrid-search", func(c *Client) error {
			_, err := c.HybridSearch(ctx, HybridSearchRequest{
				Vector: []float32{1, 0, 0}, Field: "body", Text: "fox",
			})
			return err
		}},
		{"SearchSimilar", `[]`, http.MethodPost, "/search/similar", func(c *Client) error {
			_, err := c.SearchSimilar(ctx, SimilarRequest{Collection: "docs", ID: "a"})
			return err
		}},
		{"List", `[]`, http.MethodPost, "/list", func(c *Client) error {
			_, err := c.List(ctx, ListRequest{Scope: []string{"docs"}})
			return err
		}},
		{"Aggregate", `{"count":0,"sums":{}}`, http.MethodPost, "/aggregate", func(c *Client) error {
			_, err := c.Aggregate(ctx, AggregateRequest{Sum: []string{"bytes"}})
			return err
		}},
		{"Remember", `{"ok":true}`, http.MethodPost, "/collections/docs/remember", func(c *Client) error {
			_, err := c.Remember(ctx, "docs", "a", "some text", RememberOptions{})
			return err
		}},
		{"Recall", `[]`, http.MethodPost, "/collections/docs/recall", func(c *Client) error {
			_, err := c.Recall(ctx, "docs", "some text", RecallOptions{})
			return err
		}},
		{"Flush", `{"ok":true}`, http.MethodPost, "/flush", func(c *Client) error {
			return c.Flush(ctx)
		}},
		{"Compact", `{"ok":true}`, http.MethodPost, "/compact", func(c *Client) error {
			return c.Compact(ctx)
		}},
		{"Refresh", `{"adopted":true}`, http.MethodPost, "/refresh", func(c *Client) error {
			_, err := c.Refresh(ctx)
			return err
		}},
	}

	covered := make(map[string]bool, len(cases))
	for _, tc := range cases {
		covered[tc.method] = true
		t.Run(tc.method, func(t *testing.T) {
			fake := &capture{reply: tc.reply}
			db := serve(t, fake)
			if err := tc.call(db); err != nil {
				t.Fatalf("%s failed: %v", tc.method, err)
			}
			got := fake.snapshot()
			if got.calls != 1 {
				t.Fatalf("server saw %d requests, want exactly 1", got.calls)
			}
			if got.method != tc.wantMethod {
				t.Errorf("HTTP method = %s, want %s", got.method, tc.wantMethod)
			}
			if got.path != tc.wantPath {
				t.Errorf("path = %s, want %s", got.path, tc.wantPath)
			}
		})
	}

	// A new endpoint should not be able to ship without a row above. reflect on
	// *Client reports exactly its exported methods, so the two lists must agree.
	clientType := reflect.TypeOf((*Client)(nil))
	for i := range clientType.NumMethod() {
		name := clientType.Method(i).Name
		if !covered[name] {
			t.Errorf("Client.%s has no row in this table — add one so its route is pinned", name)
		}
	}
}

// TestCollectionNameIsPathEscaped — a collection name is an opaque string that may
// hold a slash ("notes/2024") or a space. Unescaped, a slash silently addresses a
// different route: /collections/notes/2024/upsert is not a route the server has, so
// the request 404s and the caller reads it as "no such collection".
func TestCollectionNameIsPathEscaped(t *testing.T) {
	cases := []struct {
		name     string
		wantPath string
	}{
		{"docs", "/collections/docs/upsert"},
		{"notes/2024", "/collections/notes%2F2024/upsert"},
		{"my notes", "/collections/my%20notes/upsert"},
		{"notes/2024 archive", "/collections/notes%2F2024%20archive/upsert"},
		{"a?b#c", "/collections/a%3Fb%23c/upsert"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &capture{reply: `{"upserted":0}`}
			db := serve(t, fake)
			if _, err := db.Upsert(context.Background(), tc.name, nil); err != nil {
				t.Fatalf("Upsert failed: %v", err)
			}
			if got := fake.snapshot().path; got != tc.wantPath {
				t.Errorf("path = %s, want %s", got, tc.wantPath)
			}
		})
	}
}

// TestBaseURLTrailingSlashIsNormalized — a base URL pasted from a browser ends in a
// slash, and "http://host//health" is a different path to a strict router.
func TestBaseURLTrailingSlashIsNormalized(t *testing.T) {
	for _, suffix := range []string{"", "/", "//", "///"} {
		fake := &capture{reply: `{}`}
		srv := httptest.NewServer(http.HandlerFunc(fake.handler))
		t.Cleanup(srv.Close)

		db, err := NewClient(srv.URL + suffix)
		if err != nil {
			t.Fatalf("NewClient(%q) failed: %v", srv.URL+suffix, err)
		}
		if err := db.Ping(context.Background()); err != nil {
			t.Fatalf("Ping failed for base URL %q: %v", srv.URL+suffix, err)
		}
		if got := fake.snapshot().path; got != "/health" {
			t.Errorf("base URL %q produced path %s, want /health", srv.URL+suffix, got)
		}
	}
}

// TestNewClientRejectsABadBaseURL — failing at construction puts the error where the
// mistake was made, rather than on the first call from somewhere else entirely. The
// bare host:port case is the common one: it parses, but with "127.0.0.1" as the
// scheme.
//
// The query and fragment cases are the subtle ones, and the reason they are here rather
// than left to the first request: the base URL is concatenated with each endpoint path,
// so "http://h/?x=1" + "/health" parses as path "/" with query "x=1/health", and
// "http://h#frag" + "/health" has no path at all. Every call then misroutes for the
// client's whole lifetime, and the 404 reads at the call site as "no such collection".
func TestNewClientRejectsABadBaseURL(t *testing.T) {
	bad := []string{
		"", "   ", "/", "///", "127.0.0.1:7700", "http://", "localhost",
		"http://127.0.0.1:7700/?x=1",
		"http://127.0.0.1:7700?x=1",
		"http://127.0.0.1:7700/#frag",
		"http://127.0.0.1:7700#frag",
		"http://127.0.0.1:7700/api?tenant=acme#top",
	}
	for _, bad := range bad {
		if db, err := NewClient(bad); err == nil {
			t.Errorf("NewClient(%q) succeeded (baseURL %q), want an error", bad, db.baseURL)
		}
	}
	for _, good := range []string{"http://127.0.0.1:7700", "https://nidus.example.com/api"} {
		if _, err := NewClient(good); err != nil {
			t.Errorf("NewClient(%q) failed: %v", good, err)
		}
	}
}

// ── Request bodies: the omit-vs-zero contract ───────────────────────────────

// TestSearchOmitsZeroTopK is the omit-vs-zero trap, asserted on the marshalled bytes.
// The server's default top_k is 10; sending "top_k": 0 asks for zero results, which
// is a silent empty-result bug rather than a visible failure.
func TestSearchOmitsZeroTopK(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	body := fake.sentBody(t)
	if body != `{"query":[1,0,0]}` {
		t.Errorf("body = %s, want only the query field", body)
	}
	if strings.Contains(body, "top_k") {
		t.Errorf("body = %s, must not mention top_k when TopK is 0", body)
	}

	// A set TopK travels, so omitting zero is not the same as never sending it.
	if _, err := db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}, TopK: 5}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"query":[1,0,0],"top_k":5}` {
		t.Errorf("body = %s, want top_k:5", body)
	}
}

// TestSearchPaginationOffsetIsAdditive pins the new knob against the promise that a
// client which never sets it sends byte-identical requests: a zero Offset is omitted
// (the server's own default), and a set one travels in the server's spelling.
func TestSearchPaginationOffsetIsAdditive(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}, TopK: 5}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); strings.Contains(body, "offset") {
		t.Errorf("body = %s, must not mention offset when Offset is 0", body)
	}

	if _, err := db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}, TopK: 5, Offset: 10}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"query":[1,0,0],"top_k":5,"offset":10}` {
		t.Errorf("body = %s, want offset:10", body)
	}

	if _, err := db.TextSearch(ctx, TextSearchRequest{Field: "body", Query: "fox", Offset: 3}); err != nil {
		t.Fatalf("TextSearch failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"field":"body","query":"fox","offset":3}` {
		t.Errorf("body = %s, want offset:3", body)
	}

	if _, err := db.HybridSearch(ctx, HybridSearchRequest{
		Vector: []float32{1, 0, 0}, Field: "body", Text: "fox", Offset: 3,
	}); err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"vector":[1,0,0],"field":"body","text":"fox","offset":3}` {
		t.Errorf("body = %s, want offset:3", body)
	}
}

// TestExactAndProjectionAreAdditive — both knobs must be invisible until asked for, so a
// client that never sets them keeps sending byte-identical bodies, and must travel in the
// server's spelling when set. The embedded Projection's fields promote to the top level.
func TestExactAndProjectionAreAdditive(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}, TopK: 5}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"query":[1,0,0],"top_k":5}` {
		t.Errorf("body = %s, want no exact/projection keys", body)
	}

	if _, err := db.Search(ctx, SearchRequest{
		Query:      []float32{1, 0, 0},
		Exact:      true,
		Projection: Projection{IncludeAttributes: []string{"title"}},
	}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"query":[1,0,0],"exact":true,"include_attributes":["title"]}` {
		t.Errorf("body = %s, want exact + include_attributes", body)
	}

	if _, err := db.List(ctx, ListRequest{
		Limit:      5,
		Projection: Projection{ExcludeAttributes: []string{"body"}},
	}); err != nil {
		t.Fatalf("List failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"limit":5,"exclude_attributes":["body"]}` {
		t.Errorf("body = %s, want exclude_attributes", body)
	}
}

// TestSearchSimilarSendsIDNotQuery pins the one thing that distinguishes SimilarRequest
// from SearchRequest on the wire: no query vector, but collection and id instead. Zero-
// valued options must stay omitted, exactly as they do on Search.
func TestSearchSimilarSendsIDNotQuery(t *testing.T) {
	fake := &capture{reply: `[{"collection":"docs","id":"b","score":0.9,"attrs":{}}]`}
	db := serve(t, fake)
	ctx := context.Background()

	hits, err := db.SearchSimilar(ctx, SimilarRequest{Collection: "docs", ID: "a"})
	if err != nil {
		t.Fatalf("SearchSimilar failed: %v", err)
	}
	if snap := fake.snapshot(); snap.method != http.MethodPost || snap.path != "/search/similar" {
		t.Errorf("request = %s %s, want POST /search/similar", snap.method, snap.path)
	}
	if body := fake.sentBody(t); body != `{"collection":"docs","id":"a"}` {
		t.Errorf("body = %s, want only collection and id", body)
	}
	if len(hits) != 1 || hits[0].ID != "b" {
		t.Fatalf("hits = %+v, want one hit for id b", hits)
	}

	if _, err := db.SearchSimilar(ctx, SimilarRequest{
		Collection: "docs", ID: "a", Scope: []string{"docs", "notes"}, TopK: 5, Offset: 2,
		MinScore: f32(0), Exact: true,
	}); err != nil {
		t.Fatalf("SearchSimilar failed: %v", err)
	}
	want := `{"collection":"docs","id":"a","scope":["docs","notes"],"top_k":5,"offset":2,` +
		`"min_score":0,"exact":true}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}
}

// TestSetFtsFieldsOmitsUnsetKnobs — an FtsField carrying only a name must marshal to
// the same defaults the bare-string form gets, and an explicit zero must survive
// `omitempty` (which is why the knobs are pointers).
func TestSetFtsFieldsOmitsUnsetKnobs(t *testing.T) {
	fake := &capture{reply: `{"ok":true}`}
	db := serve(t, fake)
	ctx := context.Background()

	if err := db.SetFtsFields(ctx, "docs", []FtsField{{Field: "body"}}); err != nil {
		t.Fatalf("SetFtsFields failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"fields":[{"field":"body"}]}` {
		t.Errorf("body = %s, want only the field name", body)
	}

	zero := float32(0)
	folding := true
	cap40 := 40
	err := db.SetFtsFields(ctx, "docs", []FtsField{
		{Field: "body", B: &zero, AsciiFolding: &folding, MaxTokenLen: &cap40},
	})
	if err != nil {
		t.Fatalf("SetFtsFields failed: %v", err)
	}
	want := `{"fields":[{"field":"body","b":0,"ascii_folding":true,"max_token_len":40}]}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}

	// A nil slice is the lawful empty schema, as for SetFtsSchema.
	if err := db.SetFtsFields(ctx, "docs", nil); err != nil {
		t.Fatalf("SetFtsFields failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"fields":[]}` {
		t.Errorf("body = %s, want an empty fields array", body)
	}
}

// TestListOmitsZeroLimit — same trap on /list, where the server's default limit is
// 100. Offset's default is 0 so omitting a zero offset is harmless, but it is
// asserted here too so the whole body is pinned rather than half of it.
func TestListOmitsZeroLimit(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.List(ctx, ListRequest{}); err != nil {
		t.Fatalf("List failed: %v", err)
	}
	body := fake.sentBody(t)
	if body != `{}` {
		t.Errorf("body = %s, want {} — every field defaults on the server", body)
	}
	for _, field := range []string{"limit", "offset", "scope", "filter"} {
		if strings.Contains(body, field) {
			t.Errorf("body = %s, must not mention %s when it is unset", body, field)
		}
	}

	if _, err := db.List(ctx, ListRequest{Scope: []string{"docs"}, Offset: 10, Limit: 25}); err != nil {
		t.Fatalf("List failed: %v", err)
	}
	want := `{"scope":["docs"],"offset":10,"limit":25}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}
}

// TestZeroValuedRequestFieldsAreOmitted sweeps the rest of the defaulted fields —
// hybrid search's rrf_k and candidates, text search's top_k — so no request type is
// left with an unasserted zero.
func TestZeroValuedRequestFieldsAreOmitted(t *testing.T) {
	cases := []struct {
		name string
		want string
		call func(context.Context, *Client) error
	}{
		{
			"TextSearch", `{"field":"body","query":"fox"}`,
			func(ctx context.Context, c *Client) error {
				_, err := c.TextSearch(ctx, TextSearchRequest{Field: "body", Query: "fox"})
				return err
			},
		},
		{
			"HybridSearch", `{"vector":[1,0,0],"field":"body","text":"fox"}`,
			func(ctx context.Context, c *Client) error {
				_, err := c.HybridSearch(ctx, HybridSearchRequest{
					Vector: []float32{1, 0, 0}, Field: "body", Text: "fox",
				})
				return err
			},
		},
		{
			"HybridSearch with tuning", `{"vector":[1,0,0],"field":"body","text":"fox","top_k":5,"rrf_k":40,"candidates":200}`,
			func(ctx context.Context, c *Client) error {
				_, err := c.HybridSearch(ctx, HybridSearchRequest{
					Vector: []float32{1, 0, 0}, Field: "body", Text: "fox",
					TopK: 5, RRFK: f32(40), Candidates: iptr(200),
				})
				return err
			},
		},
		{
			"Recall", `{"query":"some text"}`,
			func(ctx context.Context, c *Client) error {
				_, err := c.Recall(ctx, "docs", "some text", RecallOptions{})
				return err
			},
		},
		// Remember's own zero-vs-omitted cases live in
		// TestRememberOmitsUnsetKnobsAndSendsZeroes: it decodes an object rather than the
		// search family's `[]`, so it cannot share this table's fake reply.
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &capture{reply: `[]`}
			db := serve(t, fake)
			if err := tc.call(context.Background(), db); err != nil {
				t.Fatalf("call failed: %v", err)
			}
			if body := fake.sentBody(t); body != tc.want {
				t.Errorf("body = %s, want %s", body, tc.want)
			}
		})
	}
}

// TestMinScoreNilIsOmittedAndZeroIsSent — the case a plain float32 could not express.
// nil means "no floor"; &0.0 means "a floor of exactly zero", which for cosine drops
// everything orthogonal or worse and is a real thing to ask for.
func TestMinScoreNilIsOmittedAndZeroIsSent(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.Search(ctx, SearchRequest{Query: []float32{1}}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); strings.Contains(body, "min_score") {
		t.Errorf("body = %s, must omit min_score when MinScore is nil", body)
	}

	for _, floor := range []float32{0, 0.5, -1} {
		if _, err := db.Search(ctx, SearchRequest{Query: []float32{1}, MinScore: f32(floor)}); err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		want := fmt.Sprintf(`{"query":[1],"min_score":%v}`, floor)
		if body := fake.sentBody(t); body != want {
			t.Errorf("body = %s, want %s", body, want)
		}
	}

	// The pointer fields on the other request types behave the same way.
	if _, err := db.TextSearch(ctx, TextSearchRequest{
		Field: "body", Query: "fox", MinScore: f32(0),
	}); err != nil {
		t.Fatalf("TextSearch failed: %v", err)
	}
	if body := fake.sentBody(t); !strings.Contains(body, `"min_score":0`) {
		t.Errorf("body = %s, want min_score:0", body)
	}
	if _, err := db.Recall(ctx, "docs", "q", RecallOptions{MinScore: f32(0)}); err != nil {
		t.Fatalf("Recall failed: %v", err)
	}
	if body := fake.sentBody(t); !strings.Contains(body, `"min_score":0`) {
		t.Errorf("body = %s, want min_score:0", body)
	}
}

// TestHybridZeroKnobsAreSentNotOmitted — the other half of the omit-vs-zero rule, and
// the half a plain float32/int gets wrong. Zero is a real request for both of these:
// the server fuses with 1/(rrf_k + rank + 1), so rrf_k = 0 is the maximally top-heavy
// weighting (verified against a running server: 2.0 / 1.0 / 0.333 for three docs,
// versus 0.0328 / 0.0323 / 0.0159 at the default 60), and candidates = 0 is clamped up
// to top_k — "fuse exactly top_k deep". Value fields with `omitempty` would silently
// substitute 60 and 100, which is also where the JS SDK's prune() and this SDK would
// have disagreed: `rrfK: 0` travels there.
func TestHybridZeroKnobsAreSentNotOmitted(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	req := HybridSearchRequest{
		Vector: []float32{1, 0, 0}, Field: "body", Text: "fox",
		RRFK: f32(0), Candidates: iptr(0),
	}
	if _, err := db.HybridSearch(ctx, req); err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}
	want := `{"vector":[1,0,0],"field":"body","text":"fox","rrf_k":0,"candidates":0}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s — an explicit zero must travel", body, want)
	}

	// nil is how the server's default is requested, so those keys must be absent.
	if _, err := db.HybridSearch(ctx, HybridSearchRequest{
		Vector: []float32{1, 0, 0}, Field: "body", Text: "fox",
	}); err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}
	body := fake.sentBody(t)
	for _, field := range []string{"rrf_k", "candidates"} {
		if strings.Contains(body, field) {
			t.Errorf("body = %s, must omit %s when it is nil", body, field)
		}
	}
}

// TestEmptyFilterIsOmittedFromASearchBody — an unset Filter must be absent, so the
// server's #[serde(default)] applies, rather than sent as [] (which means the same
// thing but restates a default the SDK should not own).
func TestEmptyFilterIsOmittedFromASearchBody(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.Search(ctx, SearchRequest{Query: []float32{1}}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); strings.Contains(body, "filter") {
		t.Errorf("body = %s, must omit an empty filter", body)
	}

	if _, err := db.Search(ctx, SearchRequest{
		Query:  []float32{1},
		Filter: And(Glob("path", "src/*")),
	}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	want := `{"query":[1],"filter":[{"Glob":["path","src/*"]}]}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}
}

// TestNilSlicesAndMapsBecomeEmptyCollections — `records: null`, `ids: null` and a
// `null` metadata map are deserialization errors on the server, not empty ones. The
// SDK sends the lawful empty shape so a no-op call is a no-op rather than a 400.
func TestNilSlicesAndMapsBecomeEmptyCollections(t *testing.T) {
	cases := []struct {
		name string
		want string
		call func(context.Context, *Client) error
	}{
		{"Upsert", `{"records":[]}`, func(ctx context.Context, c *Client) error {
			_, err := c.Upsert(ctx, "docs", nil)
			return err
		}},
		{"Delete", `{"ids":[]}`, func(ctx context.Context, c *Client) error {
			_, err := c.Delete(ctx, "docs", nil)
			return err
		}},
		{"DeleteWhere", `{"filter":[]}`, func(ctx context.Context, c *Client) error {
			_, err := c.DeleteWhere(ctx, "docs", nil)
			return err
		}},
		{"SetMeta", `{}`, func(ctx context.Context, c *Client) error {
			return c.SetMeta(ctx, "docs", nil)
		}},
		{"SetFtsSchema", `{"fields":[]}`, func(ctx context.Context, c *Client) error {
			return c.SetFtsSchema(ctx, "docs", nil)
		}},
		{"SetFtsFields", `{"fields":[]}`, func(ctx context.Context, c *Client) error {
			return c.SetFtsFields(ctx, "docs", nil)
		}},
		// A bodyless POST still sends {}, so the request is well-formed JSON all the
		// way through whatever sits between the client and the server.
		{"CreateCollection", `{}`, func(ctx context.Context, c *Client) error {
			return c.CreateCollection(ctx, "docs")
		}},
		{"Flush", `{}`, func(ctx context.Context, c *Client) error {
			return c.Flush(ctx)
		}},
		{"Compact", `{}`, func(ctx context.Context, c *Client) error {
			return c.Compact(ctx)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &capture{reply: `{"upserted":0,"deleted":0}`}
			db := serve(t, fake)
			if err := tc.call(context.Background(), db); err != nil {
				t.Fatalf("call failed: %v", err)
			}
			if body := fake.sentBody(t); body != tc.want {
				t.Errorf("body = %s, want %s", body, tc.want)
			}
		})
	}
}

// TestUpsertBodyShape — attrs is `{}` and not `null` for a record built without any
// (the server's attrs field has no serde default), and the vector is omitted for a
// text-only doc so "no embedding" stays distinguishable from "an empty one".
func TestUpsertBodyShape(t *testing.T) {
	fake := &capture{reply: `{"upserted":2}`}
	db := serve(t, fake)

	n, err := db.Upsert(context.Background(), "docs", []Record{
		{ID: "a", Vector: []float32{1, 0, 0}, Attrs: Attrs{"lang": Str("rust")}},
		{ID: "b"},
	})
	if err != nil {
		t.Fatalf("Upsert failed: %v", err)
	}
	if n != 2 {
		t.Errorf("Upsert returned %d, want 2 (the server's count)", n)
	}
	want := `{"records":[` +
		`{"id":"a","vector":[1,0,0],"attrs":{"lang":{"Str":"rust"}}},` +
		`{"id":"b","attrs":{}}` +
		`]}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}
}

// TestUpsertRefusesAnEmptyVector — the case `omitempty` cannot encode. A non-nil
// zero-length Vector marshals byte-identically to an absent one, so sending it would
// turn a vector-bearing upsert into a text-only document that no vector search can ever
// see — silently, with no error anywhere in the stack. The realistic source is an
// embedder that returned an empty slice, so the SDK refuses it at the call site instead
// of encoding an ambiguity, and nothing is sent.
func TestUpsertRefusesAnEmptyVector(t *testing.T) {
	cases := []struct {
		name    string
		records []Record
		wantErr bool
	}{
		{"nil vector is a text-only doc", []Record{{ID: "a"}}, false},
		{"empty vector", []Record{{ID: "a", Vector: []float32{}}}, true},
		{
			"empty vector among good ones",
			[]Record{
				{ID: "a", Vector: []float32{1, 0, 0}},
				{ID: "b", Vector: []float32{}},
			},
			true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &capture{reply: `{"upserted":0}`}
			db := serve(t, fake)

			_, err := db.Upsert(context.Background(), "docs", tc.records)
			if tc.wantErr != (err != nil) {
				t.Fatalf("Upsert error = %v, wantErr %v", err, tc.wantErr)
			}
			if !tc.wantErr {
				return
			}
			// All-or-nothing: a batch holding an unencodable record is not partially sent.
			if got := fake.snapshot(); got.calls != 0 {
				t.Errorf("the server saw %d requests, body = %s; nothing should have been sent",
					got.calls, got.body)
			}
			if !strings.Contains(err.Error(), "empty vector") {
				t.Errorf("error = %q, want it to name the empty vector", err)
			}
			// A caller mistake, not a server or transport failure, so not an *Error.
			var nerr *Error
			if errors.As(err, &nerr) {
				t.Errorf("error is an *Error with status %d", nerr.Status)
			}
		})
	}
}

// TestDeleteAndDeleteWhereAreTheSameEndpoint — id-delete and filter-delete differ
// only in the body, and the server takes the filter branch whenever that field is
// present. Two methods rather than one struct with both fields is what keeps a caller
// from accidentally sending both and getting whichever the server prefers.
func TestDeleteAndDeleteWhereAreTheSameEndpoint(t *testing.T) {
	fake := &capture{reply: `{"deleted":3}`}
	db := serve(t, fake)
	ctx := context.Background()

	n, err := db.Delete(ctx, "docs", []string{"a", "b"})
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if n != 3 {
		t.Errorf("Delete returned %d, want 3", n)
	}
	if body := fake.sentBody(t); body != `{"ids":["a","b"]}` {
		t.Errorf("Delete body = %s, want only ids", body)
	}
	if strings.Contains(fake.sentBody(t), "filter") {
		t.Error("Delete sent a filter field; the server would take the filter branch")
	}

	if _, err := db.DeleteWhere(ctx, "docs", And(Eq("lang", "go"))); err != nil {
		t.Fatalf("DeleteWhere failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"filter":[{"Eq":["lang",{"Str":"go"}]}]}` {
		t.Errorf("DeleteWhere body = %s, want only a filter", body)
	}
	if strings.Contains(fake.sentBody(t), `"ids"`) {
		t.Error("DeleteWhere sent an ids field")
	}
}

// TestUnencodableFilterFailsBeforeSending — a predicate built from a value the store
// has no attribute type for must surface as an ordinary error from the call, with no
// request made. Not a panic, and above all not a request with a mangled body: a
// filter that silently lost a predicate would delete or return the wrong records.
func TestUnencodableFilterFailsBeforeSending(t *testing.T) {
	cases := []struct {
		name string
		call func(context.Context, *Client) error
	}{
		{"Search", func(ctx context.Context, c *Client) error {
			_, err := c.Search(ctx, SearchRequest{
				Query:  []float32{1, 0, 0},
				Filter: And(Eq("lang", "rust"), Ge("score", []int{1})),
			})
			return err
		}},
		{"List", func(ctx context.Context, c *Client) error {
			_, err := c.List(ctx, ListRequest{Filter: And(Eq("score", []int{1}))})
			return err
		}},
		{"DeleteWhere", func(ctx context.Context, c *Client) error {
			_, err := c.DeleteWhere(ctx, "docs", And(Eq("score", []int{1})))
			return err
		}},
		{"TextSearch", func(ctx context.Context, c *Client) error {
			_, err := c.TextSearch(ctx, TextSearchRequest{
				Field: "body", Query: "fox", Filter: And(Eq("score", []int{1})),
			})
			return err
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &capture{reply: `[]`}
			db := serve(t, fake)

			err := tc.call(context.Background(), db)
			if err == nil {
				t.Fatal("call succeeded; a filter holding an unencodable value must not be sent")
			}
			if got := fake.snapshot(); got.calls != 0 {
				t.Errorf("the server saw %d requests; nothing should have been sent, and "+
					"body = %s", got.calls, got.body)
			}
			if !strings.Contains(err.Error(), "cannot use") {
				t.Errorf("error = %q, want it to name the value that could not be encoded", err)
			}
			// Not an *Error: nothing was sent, so there is no status, and Status 0
			// ("never got an answer") would misattribute a caller mistake to the network.
			var nerr *Error
			if errors.As(err, &nerr) {
				t.Errorf("error is an *Error with status %d; an encode failure is not a "+
					"transport or server failure", nerr.Status)
			}
		})
	}
}

// ── Responses ───────────────────────────────────────────────────────────────

// TestStatsAnnNullDecodesToNil — the server sends "ann": null for a store doing exact
// brute-force search, which is the default rather than a fault.
func TestStatsAnnNullDecodesToNil(t *testing.T) {
	fake := &capture{reply: `{
		"dimension": 384,
		"distance": "Cosine",
		"ann": null,
		"collections": ["docs", "notes"],
		"footprint": {
			"rows": 12, "dead_rows": 2, "dimension": 384,
			"vector_bytes": 18432, "doc_count": 10
		}
	}`}
	db := serve(t, fake)

	stats, err := db.Stats(context.Background())
	if err != nil {
		t.Fatalf("Stats failed: %v", err)
	}
	if stats.Ann != nil {
		t.Errorf("Ann = %+v, want nil for an exact-search store", stats.Ann)
	}
	if stats.Dimension != 384 || stats.Distance != "Cosine" {
		t.Errorf("dimension/distance = %d/%q", stats.Dimension, stats.Distance)
	}
	if !reflect.DeepEqual(stats.Collections, []string{"docs", "notes"}) {
		t.Errorf("collections = %v", stats.Collections)
	}
	want := Footprint{Rows: 12, DeadRows: 2, Dimension: 384, VectorBytes: 18432, DocCount: 10}
	if stats.Footprint != want {
		t.Errorf("footprint = %+v, want %+v", stats.Footprint, want)
	}
}

// TestStatsHnswAnnDecodesOnlyTheHnswKnobs — the server omits the knobs that do not
// apply to the active index (AnnDto's skip_serializing_if), so the IVF fields must
// come back nil rather than as a plausible zero. A plain int could not tell "this
// knob does not apply" from "n_probe is 0".
func TestStatsHnswAnnDecodesOnlyTheHnswKnobs(t *testing.T) {
	fake := &capture{reply: `{
		"dimension": 3,
		"distance": "Cosine",
		"ann": {
			"kind": "Hnsw", "overscan": 2, "seed": 42,
			"m": 16, "ef_construction": 200, "ef_search": 64
		},
		"collections": [],
		"footprint": {"rows":0,"dead_rows":0,"dimension":3,"vector_bytes":0,"doc_count":0}
	}`}
	db := serve(t, fake)

	stats, err := db.Stats(context.Background())
	if err != nil {
		t.Fatalf("Stats failed: %v", err)
	}
	ann := stats.Ann
	if ann == nil {
		t.Fatal("Ann = nil, want the HNSW configuration")
	}
	if ann.Kind != "Hnsw" || ann.Overscan != 2 || ann.Seed != 42 {
		t.Errorf("kind/overscan/seed = %q/%d/%d", ann.Kind, ann.Overscan, ann.Seed)
	}
	if ann.M == nil || *ann.M != 16 {
		t.Errorf("m = %v, want 16", ann.M)
	}
	if ann.EfConstruction == nil || *ann.EfConstruction != 200 {
		t.Errorf("ef_construction = %v, want 200", ann.EfConstruction)
	}
	if ann.EfSearch == nil || *ann.EfSearch != 64 {
		t.Errorf("ef_search = %v, want 64", ann.EfSearch)
	}
	if ann.NLists != nil || ann.NProbe != nil {
		t.Errorf("IVF knobs decoded as %v/%v, want nil for an HNSW index", ann.NLists, ann.NProbe)
	}
}

// TestStatsIvfAnnDecodesOnlyTheIvfKnobs — the mirror image, so neither direction can
// regress unnoticed.
func TestStatsIvfAnnDecodesOnlyTheIvfKnobs(t *testing.T) {
	fake := &capture{reply: `{"dimension":3,"ann":{"kind":"Ivf","overscan":2,"seed":1,` +
		`"n_lists":64,"n_probe":8},"collections":[],` +
		`"footprint":{"rows":0,"dead_rows":0,"dimension":3,"vector_bytes":0,"doc_count":0}}`}
	db := serve(t, fake)

	stats, err := db.Stats(context.Background())
	if err != nil {
		t.Fatalf("Stats failed: %v", err)
	}
	ann := stats.Ann
	if ann == nil {
		t.Fatal("Ann = nil, want the IVF configuration")
	}
	if ann.NLists == nil || *ann.NLists != 64 || ann.NProbe == nil || *ann.NProbe != 8 {
		t.Errorf("n_lists/n_probe = %v/%v, want 64/8", ann.NLists, ann.NProbe)
	}
	if ann.M != nil || ann.EfConstruction != nil || ann.EfSearch != nil {
		t.Error("HNSW knobs decoded non-nil for an IVF index")
	}
}

// TestRecordWithoutAVectorStaysAbsent — a text-only doc has no "vector" key, and it
// must survive a decode → encode round trip without acquiring an empty one. `null` or
// `[]` there would be a dimension mismatch on the way back in, and worse, would turn
// a text-only document into a vector-bearing one.
func TestRecordWithoutAVectorStaysAbsent(t *testing.T) {
	const wire = `[{"id":"a","attrs":{"body":{"Str":"text only"}}},` +
		`{"id":"b","vector":[1,0,0],"attrs":{}}]`
	fake := &capture{reply: wire}
	db := serve(t, fake)

	recs, err := db.Records(context.Background(), "notes")
	if err != nil {
		t.Fatalf("Records failed: %v", err)
	}
	if len(recs) != 2 {
		t.Fatalf("decoded %d records, want 2", len(recs))
	}
	if recs[0].Vector != nil {
		t.Errorf("text-only record decoded with Vector = %v, want nil", recs[0].Vector)
	}
	if !reflect.DeepEqual(recs[1].Vector, []float32{1, 0, 0}) {
		t.Errorf("vector = %v, want [1 0 0]", recs[1].Vector)
	}

	encoded, err := json.Marshal(recs)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(encoded) != wire {
		t.Errorf("round trip produced %s, want the original %s", encoded, wire)
	}
	if strings.Contains(string(encoded), `"vector":null`) {
		t.Error("a text-only record re-encoded with a null vector")
	}
}

// TestHitAttrsDecodeToTypedValues — hits keep [Value]s rather than a loose map (see
// the note on Attrs), and the i64 precision guard has to survive the whole response
// path, not just a direct Value decode.
func TestHitAttrsDecodeToTypedValues(t *testing.T) {
	fake := &capture{reply: `[{"collection":"docs","id":"a","score":0.98,` +
		`"attrs":{"lang":{"Str":"rust"},"id":{"Int":9007199254740993},"none":"Null"}}]`}
	db := serve(t, fake)

	hits, err := db.Search(context.Background(), SearchRequest{Query: []float32{1, 0, 0}, TopK: 1})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("decoded %d hits, want 1", len(hits))
	}
	hit := hits[0]
	if hit.Collection != "docs" || hit.ID != "a" || hit.Score != 0.98 {
		t.Errorf("hit = %+v", hit)
	}
	if lang, ok := hit.Attrs["lang"].Str(); !ok || lang != "rust" {
		t.Errorf("lang = (%q, %v)", lang, ok)
	}
	if id, ok := hit.Attrs["id"].Int(); !ok || id != 9007199254740993 {
		t.Errorf("id = (%d, %v), want the exact i64 through the response path", id, ok)
	}
	if hit.Attrs["none"].Kind() != KindNull {
		t.Errorf("none kind = %v, want KindNull", hit.Attrs["none"].Kind())
	}
}

// TestSuccessWithAnEmptyBodyIsNotAnError — a 200 with no body and a decoder waiting
// for one is a lawful answer (some proxies strip bodies, and 204 is legal), so it must
// not turn into a decode error.
func TestSuccessWithAnEmptyBodyIsNotAnError(t *testing.T) {
	for _, status := range []int{http.StatusOK, http.StatusNoContent} {
		fake := &capture{status: status}
		db := serve(t, fake)
		got, err := db.Collections(context.Background())
		if err != nil {
			t.Errorf("Collections against a %d with no body failed: %v", status, err)
		}
		if got != nil {
			t.Errorf("Collections = %v, want nil", got)
		}
	}
}

// TestMalformedSuccessBodyIsNotAnAPIError — a 2xx whose body will not decode is a
// contract problem, not a server error, so it must not masquerade as an *Error with a
// status a caller might retry on.
func TestMalformedSuccessBodyIsNotAnAPIError(t *testing.T) {
	fake := &capture{reply: `{"dimension": "three"}`}
	db := serve(t, fake)

	_, err := db.Stats(context.Background())
	if err == nil {
		t.Fatal("Stats succeeded against a malformed body")
	}
	var nerr *Error
	if errors.As(err, &nerr) {
		t.Errorf("error is an *Error with status %d, want a plain decode error", nerr.Status)
	}
	if !strings.Contains(err.Error(), "/stats") {
		t.Errorf("error = %q, want it to name the endpoint", err)
	}
}

// ── Errors ──────────────────────────────────────────────────────────────────

// TestErrorMessageExtraction — the thing on the other end of the socket is not always
// the server. A reverse proxy returns HTML, axum's body-limit rejection is plain text,
// and a dropped upstream returns nothing at all; the caller needs the most useful
// message available in each case rather than an error-path failure.
func TestErrorMessageExtraction(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		reply   string
		wantMsg string
	}{
		{
			"the server's JSON envelope", http.StatusBadRequest,
			`{"error":"dimension mismatch: expected 3, got 4"}`,
			"dimension mismatch: expected 3, got 4",
		},
		{
			// A proxy's HTML, or axum's plain-text body-limit rejection: not JSON, so
			// the raw body is the best message there is.
			"a non-JSON body", http.StatusBadGateway,
			"<html><body>502 Bad Gateway</body></html>",
			"<html><body>502 Bad Gateway</body></html>",
		},
		{
			"plain text", http.StatusRequestEntityTooLarge,
			"length limit exceeded",
			"length limit exceeded",
		},
		{
			// Nothing to go on, so the status is the message. Anything else would be
			// invented.
			"an empty body", http.StatusServiceUnavailable, "", "HTTP 503",
		},
		{
			"whitespace only", http.StatusInternalServerError, "   \n  ", "HTTP 500",
		},
		{
			// Valid JSON with no error key — the envelope is absent, so fall back to the
			// body rather than reporting an empty message.
			"JSON without an error key", http.StatusBadRequest,
			`{"detail":"something else"}`, `{"detail":"something else"}`,
		},
		{
			"an empty error string", http.StatusBadRequest, `{"error":""}`, `{"error":""}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &capture{status: tc.status, reply: tc.reply}
			db := serve(t, fake)

			err := db.Flush(context.Background())
			if err == nil {
				t.Fatalf("Flush succeeded against a %d", tc.status)
			}
			var nerr *Error
			if !errors.As(err, &nerr) {
				t.Fatalf("error is %T, want *nidus.Error", err)
			}
			if nerr.Status != tc.status {
				t.Errorf("Status = %d, want %d", nerr.Status, tc.status)
			}
			if nerr.Message != tc.wantMsg {
				t.Errorf("Message = %q, want %q", nerr.Message, tc.wantMsg)
			}
			// Error() carries both parts, since that is what lands in a log line.
			if !strings.Contains(err.Error(), tc.wantMsg) ||
				!strings.Contains(err.Error(), fmt.Sprint(tc.status)) {
				t.Errorf("Error() = %q, want the message and the status", err)
			}
		})
	}
}

// TestErrorClassifiers — the status is the part a caller acts on (409 and 503 are
// worth retrying, 400 and 422 never are), so the helpers must map to the statuses the
// server actually chooses.
//
// 422 is the row that matters most, because it is the one a reader of the HTTP contract
// does not expect: axum's Json extractor answers 422, not 400, for a body it cannot
// deserialize (a negative top_k, a wrong-shaped Value), and 400 only for a JSON syntax
// error. A retry loop written as `if !nerr.IsBadRequest() { retry() }` would otherwise
// spin forever on a request that can never succeed.
func TestErrorClassifiers(t *testing.T) {
	cases := []struct {
		status int
		checks map[string]bool // the classifier that must report true; the rest false
	}{
		{0, map[string]bool{"transport": true}},
		{400, map[string]bool{"bad request": true}},
		{401, map[string]bool{"unauthorized": true}},
		{403, map[string]bool{"read only": true}},
		{409, map[string]bool{"locked": true}},
		{422, map[string]bool{"bad request": true}},
		{503, map[string]bool{"unavailable": true}},
		{507, map[string]bool{"out of capacity": true}},
		// A 500 is none of them: a caller has no specific recovery for it.
		{500, nil},
		// Nor is a 404, which is what a memory route answers on a build without the
		// `memory` feature. Nothing about the request is malformed.
		{404, nil},
	}
	for _, tc := range cases {
		e := &Error{Message: "boom", Status: tc.status}
		got := map[string]bool{
			"transport":       e.IsTransport(),
			"bad request":     e.IsBadRequest(),
			"unauthorized":    e.IsUnauthorized(),
			"read only":       e.IsReadOnly(),
			"locked":          e.IsLocked(),
			"unavailable":     e.IsUnavailable(),
			"out of capacity": e.IsOutOfCapacity(),
		}
		for name, val := range got {
			want := tc.checks[name]
			if val != want {
				t.Errorf("status %d: Is%s = %v, want %v", tc.status, name, val, want)
			}
		}
	}

	// Status 0 renders without a status suffix, because "(HTTP 0)" is noise.
	if got := (&Error{Message: "unreachable"}).Error(); got != "nidus: unreachable" {
		t.Errorf("Error() = %q, want no HTTP suffix for status 0", got)
	}
	if got := (&Error{Message: "nope", Status: 400}).Error(); got != "nidus: nope (HTTP 400)" {
		t.Errorf("Error() = %q", got)
	}
}

// TestReadyDecodesA200 — a 200 is the ordinary case: Ready decodes the body straight
// through and asks GET /ready.
func TestReadyDecodesA200(t *testing.T) {
	fake := &capture{reply: `{"ready":true,"role":"Solo","staleness_secs":5}`}
	db := serve(t, fake)

	got, err := db.Ready(context.Background())
	if err != nil {
		t.Fatalf("Ready failed: %v", err)
	}
	want := &Readiness{Ready: true, Role: "Solo", StalenessSecs: 5}
	if *got != *want {
		t.Errorf("Ready = %+v, want %+v", *got, *want)
	}
	if snap := fake.snapshot(); snap.method != http.MethodGet || snap.path != "/ready" {
		t.Errorf("request = %s %s, want GET /ready", snap.method, snap.path)
	}
}

// TestReadyOn503IsNotAnError is the decision this ticket fixes in place: a 503 from
// /ready is the negative readiness answer, not a failure, so it must come back as
// (Readiness{Ready:false}, nil) rather than making a poll loop branch on an error.
func TestReadyOn503IsNotAnError(t *testing.T) {
	fake := &capture{status: http.StatusServiceUnavailable, reply: `{"error":"store not open"}`}
	db := serve(t, fake)

	got, err := db.Ready(context.Background())
	if err != nil {
		t.Fatalf("Ready returned an error for a 503, want (Readiness, nil): %v", err)
	}
	want := &Readiness{Ready: false, Reason: "store not open"}
	if *got != *want {
		t.Errorf("Ready = %+v, want %+v", *got, *want)
	}
}

// TestReadyOn500IsAnError — only 503 gets the special treatment; every other status
// is still the ordinary error path, with a nil *Readiness.
func TestReadyOn500IsAnError(t *testing.T) {
	fake := &capture{status: http.StatusInternalServerError, reply: `{"error":"boom"}`}
	db := serve(t, fake)

	got, err := db.Ready(context.Background())
	if got != nil {
		t.Errorf("Ready = %+v, want nil on a 500", got)
	}
	var nerr *Error
	if !errors.As(err, &nerr) {
		t.Fatalf("error is %T, want *nidus.Error", err)
	}
	if nerr.Status != http.StatusInternalServerError {
		t.Errorf("Status = %d, want 500", nerr.Status)
	}
}

// TestClusterDecodesAllFields — LeaseOwner and MaxStalenessSecs come back nil when
// the server sends null, since a single-instance store has neither.
func TestClusterDecodesAllFields(t *testing.T) {
	fake := &capture{reply: `{"role":"Solo","cluster":false,"holds_writer_handle":true,` +
		`"fenced":false,"lease_owner":null,"commit_version":7,"staleness_secs":0,` +
		`"max_staleness_secs":null}`}
	db := serve(t, fake)

	got, err := db.Cluster(context.Background())
	if err != nil {
		t.Fatalf("Cluster failed: %v", err)
	}
	want := &ClusterStatus{
		Role: "Solo", Cluster: false, HoldsWriterHandle: true, Fenced: false,
		LeaseOwner: nil, CommitVersion: 7, StalenessSecs: 0, MaxStalenessSecs: nil,
	}
	if *got != *want {
		t.Errorf("Cluster = %+v, want %+v", *got, *want)
	}
}

// TestVersionsDecodesAllFields — OldestReadable and Pinned come back nil when the
// server sends null, since a fresh, unpinned store has neither.
func TestVersionsDecodesAllFields(t *testing.T) {
	fake := &capture{reply: `{"commit_version":3,"oldest_readable":null,"pinned":null,` +
		`"readable":[1,2,3]}`}
	db := serve(t, fake)

	got, err := db.Versions(context.Background())
	if err != nil {
		t.Fatalf("Versions failed: %v", err)
	}
	want := &StoreVersions{
		CommitVersion: 3, OldestReadable: nil, Pinned: nil, Readable: []uint64{1, 2, 3},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Versions = %+v, want %+v", *got, *want)
	}
}

// TestRefreshReportsAdopted covers both outcomes: a newer manifest was picked up, or
// this snapshot was already current.
func TestRefreshReportsAdopted(t *testing.T) {
	for _, adopted := range []bool{true, false} {
		fake := &capture{reply: fmt.Sprintf(`{"adopted":%v}`, adopted)}
		db := serve(t, fake)

		got, err := db.Refresh(context.Background())
		if err != nil {
			t.Fatalf("Refresh failed: %v", err)
		}
		if got != adopted {
			t.Errorf("Refresh = %v, want %v", got, adopted)
		}
		if snap := fake.snapshot(); snap.method != http.MethodPost || snap.path != "/refresh" {
			t.Errorf("request = %s %s, want POST /refresh", snap.method, snap.path)
		}
	}
}

// TestTransportFailureIsStatusZero — a request that never got an answer is an *Error
// with Status 0, so a caller has one error type to check and can still tell "nidus
// said no" from "nidus was not reachable".
func TestTransportFailureIsStatusZero(t *testing.T) {
	// Take a real port and then close the listener, so the address is well-formed and
	// nothing is listening — connection refused rather than a DNS or routing timeout.
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	addr := srv.URL
	srv.Close()

	db, err := NewClient(addr)
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}

	_, err = db.Stats(context.Background())
	if err == nil {
		t.Fatal("Stats succeeded against a closed port")
	}
	var nerr *Error
	if !errors.As(err, &nerr) {
		t.Fatalf("error is %T, want *nidus.Error", err)
	}
	if nerr.Status != 0 {
		t.Errorf("Status = %d, want 0 for a request that never got an answer", nerr.Status)
	}
	if !nerr.IsTransport() {
		t.Error("IsTransport() = false")
	}
	if !strings.Contains(nerr.Message, "/stats") {
		t.Errorf("Message = %q, want it to name the endpoint", nerr.Message)
	}

	// Health collapses every failure to false — "is it up" has one answer here — while
	// Ping keeps the diagnosis that got the answer.
	if db.Health(context.Background()) {
		t.Error("Health reported up against a closed port")
	}
	perr := db.Ping(context.Background())
	if perr == nil {
		t.Fatal("Ping reported no error against a closed port")
	}
	if !errors.As(perr, &nerr) || nerr.Status != 0 {
		t.Errorf("Ping error = %v (%T), want an *Error with status 0", perr, perr)
	}
	if !strings.Contains(perr.Error(), "/health") {
		t.Errorf("Ping error = %q, want it to name the endpoint it could not reach", perr)
	}
}

// TestErrorBodyIsBounded — the error path reads a bounded prefix, because the body only
// ever becomes a short message and the thing on the other end of the socket is not
// necessarily nidus. A gateway streaming a huge error document must not decide how much
// memory this client allocates; the success path is deliberately still unbounded, since
// Records returns a whole collection.
func TestErrorBodyIsBounded(t *testing.T) {
	const bodySize = 4 * errorBodyLimit
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		// Not JSON, so extractError falls back to the raw body — which is exactly the
		// case where an unbounded read would have kept all of it.
		_, _ = w.Write(bytes.Repeat([]byte("x"), bodySize))
	}))
	t.Cleanup(srv.Close)

	db, err := NewClient(srv.URL)
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	err = db.Flush(context.Background())
	if err == nil {
		t.Fatal("Flush succeeded against a 502")
	}
	var nerr *Error
	if !errors.As(err, &nerr) {
		t.Fatalf("error is %T, want *nidus.Error", err)
	}
	if nerr.Status != http.StatusBadGateway {
		t.Errorf("Status = %d, want 502", nerr.Status)
	}
	if len(nerr.Message) > errorBodyLimit {
		t.Errorf("Message is %d bytes from a %d-byte body, want at most %d",
			len(nerr.Message), bodySize, errorBodyLimit)
	}
	if len(nerr.Message) == 0 {
		t.Error("Message is empty; the prefix that was read is still the best message there is")
	}

	// The client is still usable afterwards, which is what the bounded drain is for.
	if err := db.Flush(context.Background()); err == nil {
		t.Error("the second Flush succeeded against a 502")
	}
}

// TestCancelledContextIsATransportError — a cancelled or expired context is reported
// as Status 0 too, and the message says which, because a caller chasing a phantom
// network problem is a real cost.
func TestCancelledContextIsATransportError(t *testing.T) {
	fake := &capture{reply: `{}`}
	db := serve(t, fake)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := db.Flush(ctx)
	if err == nil {
		t.Fatal("Flush succeeded with a cancelled context")
	}
	var nerr *Error
	if !errors.As(err, &nerr) || nerr.Status != 0 {
		t.Fatalf("error = %v (%T), want an *Error with status 0", err, err)
	}
	if !strings.Contains(nerr.Message, "cancelled") {
		t.Errorf("Message = %q, want it to say the request was cancelled", nerr.Message)
	}
}

// TestWithTimeoutBoundsARequestAndSaysSo — the SDK timeout is applied as a context
// deadline so it composes with the caller's, and it is named in the message only when
// it was the deadline that could fire first.
func TestWithTimeoutBoundsARequestAndSaysSo(t *testing.T) {
	slow := func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(2 * time.Second):
		case <-r.Context().Done():
		}
	}
	srv := httptest.NewServer(http.HandlerFunc(slow))
	t.Cleanup(srv.Close)

	db, err := NewClient(srv.URL, WithTimeout(30*time.Millisecond))
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}

	start := time.Now()
	err = db.Flush(context.Background())
	if err == nil {
		t.Fatal("Flush succeeded against a hanging server")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("the request took %s; the timeout did not fire", elapsed)
	}
	var nerr *Error
	if !errors.As(err, &nerr) || nerr.Status != 0 {
		t.Fatalf("error = %v (%T), want an *Error with status 0", err, err)
	}
	if !strings.Contains(nerr.Message, "timed out after 30ms") {
		t.Errorf("Message = %q, want it to name the configured timeout", nerr.Message)
	}

	// When the caller's own deadline is the earlier one, the message must not blame
	// the SDK's timeout — that would send them looking in the wrong place.
	loose, err := NewClient(srv.URL, WithTimeout(10*time.Second))
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	err = loose.Flush(ctx)
	if err == nil {
		t.Fatal("Flush succeeded against a hanging server")
	}
	if strings.Contains(err.Error(), "10s") {
		t.Errorf("error = %q, must not blame the SDK timeout for the caller's deadline", err)
	}
}

// ── Options and headers ─────────────────────────────────────────────────────

// TestWithTokenSendsABearerHeader — and no Authorization header at all when there is
// no token, since an empty bearer is the sort of thing an intermediary rejects while
// `nidus serve` without --token would have accepted anything.
func TestWithTokenSendsABearerHeader(t *testing.T) {
	fake := &capture{reply: `{}`}
	db := serve(t, fake, WithToken("s3cret"))
	if err := db.Flush(context.Background()); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	if got := fake.snapshot().header.Get("Authorization"); got != "Bearer s3cret" {
		t.Errorf("Authorization = %q, want %q", got, "Bearer s3cret")
	}

	plain := &capture{reply: `{}`}
	db = serve(t, plain)
	if err := db.Flush(context.Background()); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	header := plain.snapshot().header
	if _, present := header["Authorization"]; present {
		t.Errorf("Authorization = %q with no token configured, want the header absent",
			header.Get("Authorization"))
	}
}

// TestWithHeaderCannotDisplaceTheSDKsOwn — a caller's extra headers travel, but the
// SDK sets Authorization and Content-Type last so its contract with the server
// survives a typo in a WithHeader call.
func TestWithHeaderCannotDisplaceTheSDKsOwn(t *testing.T) {
	fake := &capture{reply: `{}`}
	db := serve(t, fake,
		WithToken("real"),
		WithHeader("X-Trace-Id", "abc123"),
		WithHeader("X-Tenant", "acme"),
		WithHeader("Authorization", "Bearer forged"),
		WithHeader("Content-Type", "text/plain"),
	)
	if err := db.Flush(context.Background()); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	header := fake.snapshot().header
	if got := header.Get("X-Trace-Id"); got != "abc123" {
		t.Errorf("X-Trace-Id = %q, want abc123", got)
	}
	if got := header.Get("X-Tenant"); got != "acme" {
		t.Errorf("X-Tenant = %q, want acme", got)
	}
	if got := header.Get("Authorization"); got != "Bearer real" {
		t.Errorf("Authorization = %q; the configured token must win", got)
	}
	if got := header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q; the SDK's must win", got)
	}
}

// TestContentTypeOnlyAccompaniesABody — a bodyless request that claims
// application/json is a lie, and some proxies act on it.
func TestContentTypeOnlyAccompaniesABody(t *testing.T) {
	cases := []struct {
		name     string
		wantType string
		call     func(context.Context, *Client) error
	}{
		{"GET has no body", "", func(ctx context.Context, c *Client) error {
			_, err := c.Collections(ctx)
			return err
		}},
		{"DELETE has no body", "", func(ctx context.Context, c *Client) error {
			return c.DropCollection(ctx, "docs")
		}},
		{"POST with a body", "application/json", func(ctx context.Context, c *Client) error {
			return c.Flush(ctx)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &capture{reply: `[]`}
			db := serve(t, fake)
			if err := tc.call(context.Background(), db); err != nil {
				t.Fatalf("call failed: %v", err)
			}
			got := fake.snapshot()
			if ct := got.header.Get("Content-Type"); ct != tc.wantType {
				t.Errorf("Content-Type = %q, want %q", ct, tc.wantType)
			}
			if tc.wantType == "" && len(got.body) != 0 {
				t.Errorf("sent a body (%s) on a request that should have none", got.body)
			}
		})
	}
}

// TestWithHTTPClientIsUsedAndNilIsIgnored — a caller's transport is honoured, and a
// nil one is dropped rather than stored, since a nil client can serve no request and
// the resulting panic would point at the wrong line.
func TestWithHTTPClientIsUsedAndNilIsIgnored(t *testing.T) {
	fake := &capture{reply: `{}`}

	var seen int
	custom := &http.Client{Transport: countingTransport{&seen, http.DefaultTransport}}
	db := serve(t, fake, WithHTTPClient(custom))
	if err := db.Flush(context.Background()); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	if seen != 1 {
		t.Errorf("the supplied http.Client served %d requests, want 1", seen)
	}

	nilled := serve(t, &capture{reply: `{}`}, WithHTTPClient(nil))
	if nilled.hc == nil {
		t.Fatal("WithHTTPClient(nil) stored a nil client")
	}
	if err := nilled.Flush(context.Background()); err != nil {
		t.Errorf("Flush failed after WithHTTPClient(nil): %v", err)
	}
}

// countingTransport counts round trips so a test can prove which client was used.
type countingTransport struct {
	n    *int
	next http.RoundTripper
}

func (t countingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	*t.n++
	return t.next.RoundTrip(r)
}

// TestNewClientDoesNotTouchTheDefaultClient — a library that sets a Timeout on
// http.DefaultClient changes the behaviour of every other package in the binary, from
// a place nobody thinks to look.
func TestNewClientDoesNotTouchTheDefaultClient(t *testing.T) {
	db, err := NewClient("http://127.0.0.1:7700", WithTimeout(time.Second))
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	if db.hc == http.DefaultClient {
		t.Error("the client is http.DefaultClient; SDK policy would leak process-wide")
	}
	if http.DefaultClient.Timeout != 0 {
		t.Errorf("http.DefaultClient.Timeout = %s, want it untouched", http.DefaultClient.Timeout)
	}
	// The timeout is a per-request context deadline, not client policy, so it composes
	// with a caller's own context instead of overriding it.
	if db.hc.Timeout != 0 {
		t.Errorf("hc.Timeout = %s, want 0 (the deadline is applied per request)", db.hc.Timeout)
	}
}

// TestHealthCollapsesEveryFailureToFalse — "is it up" has one answer, and /health
// needs no token, so a false is never merely an auth problem.
func TestHealthCollapsesEveryFailureToFalse(t *testing.T) {
	cases := []struct {
		name   string
		status int
		want   bool
	}{
		{"200", http.StatusOK, true},
		{"401", http.StatusUnauthorized, false},
		{"503 store not open", http.StatusServiceUnavailable, false},
		{"500", http.StatusInternalServerError, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &capture{status: tc.status, reply: `{"status":"ok"}`}
			db := serve(t, fake)
			if got := db.Health(context.Background()); got != tc.want {
				t.Errorf("Health() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestConcurrentUseIsSafe — one Client per server, shared, is the intended shape;
// this is the assertion that says so, and `go test -race` is what enforces it.
func TestConcurrentUseIsSafe(t *testing.T) {
	// A handler of its own here rather than the shared capture, whose recorded fields
	// are last-write-wins by design.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[]`)
	}))
	t.Cleanup(srv.Close)

	db, err := NewClient(srv.URL, WithToken("t"), WithHeader("X-Trace-Id", "abc"))
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}

	var wg sync.WaitGroup
	errs := make(chan error, 32)
	for i := range 32 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := db.Search(context.Background(), SearchRequest{
				Query: []float32{1, 0, 0},
				TopK:  i + 1,
			})
			if err != nil {
				errs <- err
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent Search failed: %v", err)
	}
}

// TestDiversityIsAdditiveAndKeepsZero — an unset diversity must leave the key out of the
// body entirely (not null), and &0 is a meaningful lambda that omitempty would drop if the
// field were a bare float32. RecallOptions is checked too: its wire form is hand-copied,
// so a new field there can silently never reach the server.
func TestDiversityIsAdditiveAndKeepsZero(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.Search(ctx, SearchRequest{Query: []float32{1}}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"query":[1]}` {
		t.Errorf("body = %s, want no diversity key at all", body)
	}

	if _, err := db.Search(ctx, SearchRequest{Query: []float32{1}, Diversity: f32(0)}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"query":[1],"diversity":0}` {
		t.Errorf("body = %s, want a zero lambda on the wire", body)
	}

	_, err := db.SearchSimilar(ctx, SimilarRequest{
		Collection: "docs", ID: "d1", Diversity: f32(0.3),
	})
	if err != nil {
		t.Fatalf("SearchSimilar failed: %v", err)
	}
	want := `{"collection":"docs","id":"d1","diversity":0.3}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}

	_, err = db.TextSearch(ctx, TextSearchRequest{
		Field: "body", Query: "alpha", Diversity: f32(0.5),
	})
	if err != nil {
		t.Fatalf("TextSearch failed: %v", err)
	}
	want = `{"field":"body","query":"alpha","diversity":0.5}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}

	if _, err := db.Recall(ctx, "notes", "why", RecallOptions{}); err != nil {
		t.Fatalf("Recall failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"query":"why"}` {
		t.Errorf("body = %s, want no diversity key at all", body)
	}
	_, err = db.Recall(ctx, "notes", "why", RecallOptions{Diversity: f32(1)})
	if err != nil {
		t.Fatalf("Recall failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"query":"why","diversity":1}` {
		t.Errorf("body = %s, want the diversity to reach recallWire", body)
	}
}

// TestExpandAndRollupAreAdditive — an unexpanded request stays byte-identical to a client
// that predates expansion, and a set one carries only the fields the caller named. Recall
// takes the text-native Rollup, never the raw attr-name Expand.
func TestExpandAndRollupAreAdditive(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.Search(ctx, SearchRequest{Query: []float32{1}}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); strings.Contains(body, "expand") {
		t.Errorf("body = %s, want no expand key at all", body)
	}

	// A bare radius sends only a radius; the server fills the reserved chunk attrs.
	if _, err := db.Search(ctx, SearchRequest{
		Query: []float32{1}, Expand: &Expand{Radius: 2},
	}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body, want := fake.sentBody(t), `"expand":{"radius":2}`; !strings.Contains(body, want) {
		t.Errorf("body = %s, want it to contain %s", body, want)
	}

	if _, err := db.HybridSearch(ctx, HybridSearchRequest{
		Vector: []float32{1}, Field: "body", Text: "fox",
		Expand: &Expand{Radius: 1, ParentField: "doc", TextField: "body"},
	}); err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}
	want := `"expand":{"radius":1,"parent_field":"doc","text_field":"body"}`
	if body := fake.sentBody(t); !strings.Contains(body, want) {
		t.Errorf("body = %s, want it to contain %s", body, want)
	}

	if _, err := db.Recall(ctx, "notes", "why", RecallOptions{
		Rollup: &Rollup{Neighbours: 1},
	}); err != nil {
		t.Fatalf("Recall failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"query":"why","rollup":{"neighbours":1}}` {
		t.Errorf("body = %s, want the rollup to reach recallWire", body)
	}
}

// TestAHitCarriesItsContext — Context is nil unless the server sent one, so an unexpanded
// hit is the object it always was.
func TestAHitCarriesItsContext(t *testing.T) {
	fake := &capture{reply: `[{"collection":"c","id":"d#1","score":0.9,"attrs":{},"context":"widened"},` +
		`{"collection":"c","id":"d#2","score":0.8,"attrs":{}}]`}
	db := serve(t, fake)
	hits, err := db.Search(context.Background(), SearchRequest{
		Query: []float32{1}, Expand: &Expand{Radius: 1},
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if hits[0].Context == nil || *hits[0].Context != "widened" {
		t.Errorf("hits[0].Context = %v, want \"widened\"", hits[0].Context)
	}
	if hits[1].Context != nil {
		t.Errorf("hits[1].Context = %v, want nil", hits[1].Context)
	}
}

// TestRankingKnobsAreAdditive — rank_by, limit_per and order_by must be absent from a
// request that does not use them, so today's bodies stay byte-identical, and must
// carry only the sub-knobs the caller actually named when they are used.
func TestRankingKnobsAreAdditive(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.Search(ctx, SearchRequest{Query: []float32{1}}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"query":[1]}` {
		t.Errorf("body = %s, want no ranking keys at all", body)
	}

	// A Decay naming only what it changes: scale, decay, lambda and missing all default
	// on the server, so sending them would restate a default this SDK does not own.
	_, err := db.Search(ctx, SearchRequest{
		Query:  []float32{1},
		RankBy: DecayRank(Decay{Field: "ts", Origin: 1700000000000}),
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	want := `{"query":[1],"rank_by":{"Decay":{"field":"ts","origin":1700000000000}}}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}

	// Every knob set, including the two whose zero is a real request.
	_, err = db.Search(ctx, SearchRequest{
		Query: []float32{1},
		RankBy: DecayRank(Decay{
			Field: "ts", Origin: 1700000000000, Scale: 604800000, Decay: 0.9,
			Lambda: f32(2), Missing: f32(0),
		}),
		LimitPer: &LimitPer{Field: "path", Max: 2},
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	want = `{"query":[1],"rank_by":{"Decay":{"field":"ts","origin":1700000000000,` +
		`"scale":604800000,"decay":0.9,"lambda":2,"missing":0}},` +
		`"limit_per":{"field":"path","max":2}}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}

	// order_by rides on /list, and ascending is the wire default.
	if _, err := db.List(ctx, ListRequest{OrderBy: &OrderBy{Field: "ts"}}); err != nil {
		t.Fatalf("List failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"order_by":{"field":"ts"}}` {
		t.Errorf("body = %s, want an ascending order_by", body)
	}
	if _, err := db.List(ctx, ListRequest{
		OrderBy: &OrderBy{Field: "ts", Descending: true},
	}); err != nil {
		t.Fatalf("List failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"order_by":{"field":"ts","descending":true}}` {
		t.Errorf("body = %s, want a descending order_by", body)
	}
}

// TestRankByNamingNoExpressionIsAnEncodeError — RankBy is a tagged union, so an empty
// one would travel as {} and come back as a serde message about an unknown variant.
// Failing in the encoder keeps the mistake at the call site that made it.
func TestRankByNamingNoExpressionIsAnEncodeError(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)

	_, err := db.Search(context.Background(), SearchRequest{
		Query: []float32{1}, RankBy: &RankBy{},
	})
	if err == nil {
		t.Fatal("Search succeeded with an empty RankBy, want an encode error")
	}
	if !strings.Contains(err.Error(), "DecayRank") {
		t.Errorf("error = %q, want it to point at the builder", err)
	}
	if got := fake.snapshot(); got.calls != 0 {
		t.Errorf("server saw %d requests; the body must fail before sending", got.calls)
	}
}

// TestTextQuerySpellings — the single field+query pair and the clauses list are
// mutually exclusive on the server, so the old spelling must still travel exactly as it
// did and the new one must travel alone.
func TestTextQuerySpellings(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	// The compatibility contract: unchanged bytes for the single-field form.
	if _, err := db.TextSearch(ctx, TextSearchRequest{Field: "body", Query: "fox"}); err != nil {
		t.Fatalf("TextSearch failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"field":"body","query":"fox"}` {
		t.Errorf("body = %s, want the single-field spelling unchanged", body)
	}

	// The clauses form must not drag an empty field/query along, which the server reads
	// as "both spellings at once" and refuses.
	_, err := db.TextSearch(ctx, TextSearchRequest{
		Clauses: []FtsClause{{Field: "title", Query: "rust"}, {Field: "body", Query: "async"}},
		Combine: CombineMax,
	})
	if err != nil {
		t.Fatalf("TextSearch failed: %v", err)
	}
	want := `{"clauses":[{"field":"title","query":"rust"},{"field":"body","query":"async"}],` +
		`"combine":"Max"}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}

	// Sum is the server's default, so leaving Combine empty must omit the key.
	_, err = db.TextSearch(ctx, TextSearchRequest{
		Clauses: []FtsClause{{Field: "title", Query: "rust"}},
	})
	if err != nil {
		t.Fatalf("TextSearch failed: %v", err)
	}
	if body := fake.sentBody(t); strings.Contains(body, "combine") {
		t.Errorf("body = %s, must omit combine when it is unset", body)
	}

	// Hybrid takes the same choice, spelled `text` rather than `query`.
	_, err = db.HybridSearch(ctx, HybridSearchRequest{
		Vector:  []float32{1, 0, 0},
		Clauses: []FtsClause{{Field: "title", Query: "rust"}},
		Combine: CombineSum,
	})
	if err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}
	want = `{"vector":[1,0,0],"clauses":[{"field":"title","query":"rust"}],"combine":"Sum"}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}
}

// TestExplainAndHighlightAreAdditive — both are off unless asked for, and an empty
// HighlightOpts is the request for the server's defaults rather than for zero
// fragments of zero characters.
func TestExplainAndHighlightAreAdditive(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.TextSearch(ctx, TextSearchRequest{Field: "body", Query: "fox"}); err != nil {
		t.Fatalf("TextSearch failed: %v", err)
	}
	body := fake.sentBody(t)
	for _, field := range []string{"explain", "highlight"} {
		if strings.Contains(body, field) {
			t.Errorf("body = %s, must omit %s when it is unset", body, field)
		}
	}

	_, err := db.TextSearch(ctx, TextSearchRequest{
		Field: "body", Query: "fox", Explain: true, Highlight: &HighlightOpts{},
	})
	if err != nil {
		t.Fatalf("TextSearch failed: %v", err)
	}
	want := `{"field":"body","query":"fox","explain":true,"highlight":{}}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}

	_, err = db.HybridSearch(ctx, HybridSearchRequest{
		Vector: []float32{1}, Field: "body", Text: "fox",
		Explain:   true,
		Highlight: &HighlightOpts{MaxFragments: 3, FragmentChars: 40},
	})
	if err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}
	want = `{"vector":[1],"field":"body","text":"fox","explain":true,` +
		`"highlight":{"max_fragments":3,"fragment_chars":40}}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}
}

// TestHybridWeightsOmitNilAndSendZero — the omit-vs-zero rule again, and the case a
// plain float32 gets wrong: both weights default to 1.0 on the server, so a zero-valued
// field would silently ask for "weight this leg at nothing" on every unweighted query.
func TestHybridWeightsOmitNilAndSendZero(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.HybridSearch(ctx, HybridSearchRequest{
		Vector: []float32{1}, Field: "body", Text: "fox",
	}); err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}
	body := fake.sentBody(t)
	for _, field := range []string{"vector_weight", "text_weight"} {
		if strings.Contains(body, field) {
			t.Errorf("body = %s, must omit %s when it is nil", body, field)
		}
	}

	// Dropping the vector leg entirely is a real request, and the only way to spell it.
	_, err := db.HybridSearch(ctx, HybridSearchRequest{
		Vector: []float32{1}, Field: "body", Text: "fox",
		VectorWeight: f32(0), TextWeight: f32(2.5),
	})
	if err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}
	want := `{"vector":[1],"field":"body","text":"fox","vector_weight":0,"text_weight":2.5}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s — an explicit zero must travel", body, want)
	}
}

// TestTextSearchProjectionAndRanking — /text-search takes the projection and the
// ranking knobs too, and each must stay absent until asked for.
func TestTextSearchProjectionAndRanking(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)

	_, err := db.TextSearch(context.Background(), TextSearchRequest{
		Field: "body", Query: "fox",
		RankBy:     DecayRank(Decay{Field: "ts", Origin: 1}),
		LimitPer:   &LimitPer{Field: "path", Max: 1},
		Projection: Projection{ExcludeAttributes: []string{"body"}},
	})
	if err != nil {
		t.Fatalf("TextSearch failed: %v", err)
	}
	want := `{"field":"body","query":"fox","rank_by":{"Decay":{"field":"ts","origin":1}},` +
		`"limit_per":{"field":"path","max":1},"exclude_attributes":["body"]}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}
}

// TestRerankIsSentOnEverySearchRoute — all four rerankable requests carry Rerank and send
// the frozen wire shape. Recall is the leg recallWire can silently drop a new field on, so
// it must be in this table rather than covered separately.
func TestRerankIsSentOnEverySearchRoute(t *testing.T) {
	rerank := &RerankOptions{Query: "fox", Overscan: iptr(4), TextAttr: "body"}
	want := `"rerank":{"query":"fox","overscan":4,"text_attr":"body"}`

	cases := []struct {
		name string
		run  func(db *Client, ctx context.Context) error
	}{
		{"Search", func(db *Client, ctx context.Context) error {
			_, err := db.Search(ctx, SearchRequest{Query: []float32{1}, Rerank: rerank})
			return err
		}},
		{"TextSearch", func(db *Client, ctx context.Context) error {
			_, err := db.TextSearch(ctx, TextSearchRequest{Field: "body", Query: "fox", Rerank: rerank})
			return err
		}},
		{"HybridSearch", func(db *Client, ctx context.Context) error {
			_, err := db.HybridSearch(ctx, HybridSearchRequest{
				Vector: []float32{1}, Field: "body", Text: "fox", Rerank: rerank,
			})
			return err
		}},
		{"Recall", func(db *Client, ctx context.Context) error {
			_, err := db.Recall(ctx, "docs", "q", RecallOptions{Rerank: rerank})
			return err
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &capture{reply: `[]`}
			db := serve(t, fake)
			if err := tc.run(db, context.Background()); err != nil {
				t.Fatalf("%s failed: %v", tc.name, err)
			}
			if body := fake.sentBody(t); !strings.Contains(body, want) {
				t.Errorf("body = %s, want it to contain %s", body, want)
			}
		})
	}
}

// TestRerankOmittedIsAbsentFromTheBody — the additive-wire guarantee: not setting Rerank
// leaves the request byte-identical to a client that predates this field.
func TestRerankOmittedIsAbsentFromTheBody(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)

	if _, err := db.Search(context.Background(), SearchRequest{Query: []float32{1}}); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if body := fake.sentBody(t); strings.Contains(body, "rerank") {
		t.Errorf("body = %s, must omit rerank when it is nil", body)
	}
}

// TestRerankEmptyStructIsSentAsEmptyObject — &RerankOptions{} is the valid minimal form
// (query defaults to the request's own text server-side), pinned so a later omitempty
// change on a sub-field cannot silently drop it.
func TestRerankEmptyStructIsSentAsEmptyObject(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)

	_, err := db.TextSearch(context.Background(), TextSearchRequest{
		Field: "body", Query: "fox", Rerank: &RerankOptions{},
	})
	if err != nil {
		t.Fatalf("TextSearch failed: %v", err)
	}
	want := `{"field":"body","query":"fox","rerank":{}}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}
}

// TestHitAnnotationsDecode — a hit carrying annotations, and one that does not. The
// second case is the common one and the one that must not regress: the server omits
// the key entirely, and the decoded Hit has to be indistinguishable from today's.
func TestHitAnnotationsDecode(t *testing.T) {
	annotated := `[{"collection":"docs","id":"a","score":0.9,"attrs":{},"annotations":{` +
		`"vector":{"rank":0,"score":0.98},"text":{"rank":1,"score":1.1},` +
		`"clauses":[{"field":"title","score":0.49}],` +
		`"highlights":[{"field":"body","fragments":[` +
		`{"text":"we were running","spans":[[8,15]]}]}]}}]`
	fake := &capture{reply: annotated}
	db := serve(t, fake)
	ctx := context.Background()

	hits, err := db.HybridSearch(ctx, HybridSearchRequest{
		Vector: []float32{1}, Field: "body", Text: "run", Explain: true,
	})
	if err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}
	a := hits[0].Annotations
	if a == nil {
		t.Fatal("Annotations = nil, want the decoded annotations")
	}
	if a.Vector == nil || a.Vector.Rank != 0 || a.Vector.Score != 0.98 {
		t.Errorf("Vector = %+v, want rank 0 score 0.98", a.Vector)
	}
	if a.Text == nil || a.Text.Rank != 1 || a.Text.Score != 1.1 {
		t.Errorf("Text = %+v, want rank 1 score 1.1", a.Text)
	}
	if len(a.Clauses) != 1 || a.Clauses[0].Field != "title" || a.Clauses[0].Score != 0.49 {
		t.Errorf("Clauses = %+v, want one title clause scoring 0.49", a.Clauses)
	}
	if len(a.Highlights) != 1 || a.Highlights[0].Field != "body" {
		t.Fatalf("Highlights = %+v, want one over body", a.Highlights)
	}
	frag := a.Highlights[0].Fragments[0]
	// The spans are byte offsets into the fragment's own text, so slicing it directly
	// is the whole point of decoding them into a named pair.
	if got := frag.Text[frag.Spans[0].Start:frag.Spans[0].End]; got != "running" {
		t.Errorf("span %v covers %q, want %q", frag.Spans[0], got, "running")
	}

	plain := &capture{reply: `[{"collection":"docs","id":"a","score":0.9,"attrs":{}}]`}
	hits, err = serve(t, plain).Search(ctx, SearchRequest{Query: []float32{1}})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if hits[0].Annotations != nil {
		t.Errorf("Annotations = %+v on an unannotated hit, want nil", hits[0].Annotations)
	}
}

// TestSpanRejectsAMalformedPair — a span that is not a two-element array is an error
// rather than a zero span, which would silently highlight the wrong text (or nothing).
func TestSpanRejectsAMalformedPair(t *testing.T) {
	for _, raw := range []string{`{"start":1,"end":2}`, `[1]`, `[1,2,3]`, `"1,2"`} {
		var s Span
		if err := json.Unmarshal([]byte(raw), &s); err == nil {
			t.Errorf("Unmarshal(%s) succeeded, want an error", raw)
		}
	}
	// And the round trip, since the marshalled form is what an SDK test fixture writes.
	encoded, err := json.Marshal(Span{Start: 8, End: 15})
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(encoded) != `[8,15]` {
		t.Errorf("Marshal = %s, want [8,15]", encoded)
	}
}

// TestAggregateBodyAndResponse — the zero request counts everything, and the sums come
// back as tagged Values so an Int total does not arrive having passed through a float.
func TestAggregateBodyAndResponse(t *testing.T) {
	fake := &capture{reply: `{"count":12,"sums":{"bytes":{"Int":40960},"ratio":{"Float":1.5}}}`}
	db := serve(t, fake)
	ctx := context.Background()

	if _, err := db.Aggregate(ctx, AggregateRequest{}); err != nil {
		t.Fatalf("Aggregate failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{}` {
		t.Errorf("body = %s, want {} — scope, filter and sum all default", body)
	}

	out, err := db.Aggregate(ctx, AggregateRequest{
		Scope:  []string{"docs"},
		Filter: And(Eq("lang", "rust")),
		Sum:    []string{"bytes", "ratio"},
	})
	if err != nil {
		t.Fatalf("Aggregate failed: %v", err)
	}
	want := `{"scope":["docs"],"filter":[{"Eq":["lang",{"Str":"rust"}]}],"sum":["bytes","ratio"]}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}

	if out.Count != 12 {
		t.Errorf("Count = %d, want 12", out.Count)
	}
	if n, ok := out.Sums["bytes"].Int(); !ok || n != 40960 {
		t.Errorf("Sums[bytes] = %v, want Int(40960)", out.Sums["bytes"])
	}
	// A Float sum must stay a Float: it is the tag that says the addends were not all
	// integers, and collapsing it would lose that.
	if f, ok := out.Sums["ratio"].Float(); !ok || f != 1.5 {
		t.Errorf("Sums[ratio] = %v, want Float(1.5)", out.Sums["ratio"])
	}
}

// ── Remember: the TTL and dedupe knobs ──────────────────────────────────────

// TestRememberOmitsUnsetKnobsAndSendsZeroes covers every RememberOptions field's effect
// on the body, and with it the omit-vs-zero rule on the two pointer knobs: a TTL of 0
// expires the entry immediately and a dedupe floor of 0 matches any entry at all, so
// neither zero may be dropped in favour of the server's default (never expire / no dedupe).
func TestRememberOmitsUnsetKnobsAndSendsZeroes(t *testing.T) {
	cases := []struct {
		name string
		opts RememberOptions
		want string
	}{
		{"defaults", RememberOptions{}, `{"id":"a","text":"t"}`},
		{
			"mode and attrs",
			RememberOptions{Mode: "summarize", Attrs: Attrs{"src": Str("x")}},
			`{"id":"a","text":"t","mode":"summarize","attrs":{"src":{"Str":"x"}}}`,
		},
		{
			"ttl and dedupe",
			RememberOptions{TTLSeconds: i64(3600), DedupeThreshold: f32(0.95)},
			`{"id":"a","text":"t","ttl_seconds":3600,"dedupe_threshold":0.95}`,
		},
		{
			"zeroed ttl and dedupe are sent, not omitted",
			RememberOptions{TTLSeconds: i64(0), DedupeThreshold: f32(0)},
			`{"id":"a","text":"t","ttl_seconds":0,"dedupe_threshold":0}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &capture{reply: `{"ok":true,"upserted":1,"id":"a","deduped":false}`}
			if _, err := serve(t, fake).Remember(
				context.Background(), "notes", "a", "t", tc.opts,
			); err != nil {
				t.Fatalf("Remember failed: %v", err)
			}
			if body := fake.sentBody(t); body != tc.want {
				t.Errorf("body = %s, want %s", body, tc.want)
			}
		})
	}
}

// TestRememberResultReportsTheRecordActuallyWritten pins the reason the response is
// decoded at all: on a dedupe match the server writes a *different* record than the one
// asked for, and ID is the caller's only way to learn which.
func TestRememberResultReportsTheRecordActuallyWritten(t *testing.T) {
	ctx := context.Background()

	fake := &capture{reply: `{"ok":true,"upserted":1,"id":"older","deduped":true}`}
	out, err := serve(t, fake).Remember(ctx, "notes", "newer", "t", RememberOptions{
		DedupeThreshold: f32(0.9),
	})
	if err != nil {
		t.Fatalf("Remember failed: %v", err)
	}
	if out.ID != "older" || !out.Deduped || out.Upserted != 1 {
		t.Errorf("result = %+v, want {ID:older Upserted:1 Deduped:true}", out)
	}

	// A server predating the echoed fields answers {ok, upserted}; reporting an empty ID
	// there would be a lie about which record changed.
	fake = &capture{reply: `{"ok":true,"upserted":1}`}
	out, err = serve(t, fake).Remember(ctx, "notes", "a", "t", RememberOptions{})
	if err != nil {
		t.Fatalf("Remember failed: %v", err)
	}
	if out.ID != "a" || out.Deduped {
		t.Errorf("result = %+v, want the requested id and Deduped false", out)
	}
}

// TestSetFilterIndexFieldsOmitsUnsetKnobs — the server defaults both structures to true,
// so an unset knob must be absent from the body rather than sent as false, which would
// silently turn a structure off.
func TestSetFilterIndexFieldsOmitsUnsetKnobs(t *testing.T) {
	fake := &capture{reply: `{"ok":true}`}
	db := serve(t, fake)
	ctx := context.Background()

	if err := db.SetFilterIndexFields(ctx, "docs", []FilterIndexField{{Field: "body"}}); err != nil {
		t.Fatalf("SetFilterIndexFields failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"fields":[{"field":"body"}]}` {
		t.Errorf("body = %s, want only the field name", body)
	}

	off := false
	err := db.SetFilterIndexFields(ctx, "docs", []FilterIndexField{
		{Field: "body", Trigrams: &off},
	})
	if err != nil {
		t.Fatalf("SetFilterIndexFields failed: %v", err)
	}
	want := `{"fields":[{"field":"body","trigrams":false}]}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}

	// A nil slice is the lawful empty declaration, which drops the index.
	if err := db.SetFilterIndex(ctx, "docs", nil); err != nil {
		t.Fatalf("SetFilterIndex failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"fields":[]}` {
		t.Errorf("body = %s, want an empty field list", body)
	}
}

// TestRecallOmitsReinforceWhenUnset — the compatibility promise: a recall that does not
// ask to reinforce must send no reinforce/extend_ttl_seconds keys at all, so a server
// predating them sees exactly the body it always has.
func TestRecallOmitsReinforceWhenUnset(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)

	if _, err := db.Recall(context.Background(), "notes", "why", RecallOptions{}); err != nil {
		t.Fatalf("Recall failed: %v", err)
	}
	if body := fake.sentBody(t); body != `{"query":"why"}` {
		t.Errorf("body = %s, want no reinforce keys at all", body)
	}
}

// TestRecallSendsReinforceAndExtendTTL asserts the exact body once both knobs are set.
func TestRecallSendsReinforceAndExtendTTL(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)

	_, err := db.Recall(context.Background(), "notes", "why", RecallOptions{
		Reinforce: true, ExtendTTLSeconds: i64(3600),
	})
	if err != nil {
		t.Fatalf("Recall failed: %v", err)
	}
	want := `{"query":"why","reinforce":true,"extend_ttl_seconds":3600}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}
}

// TestDecayCountKnobsMarshal asserts the reinforcement sub-knobs travel under Decay
// when the caller sets them.
func TestDecayCountKnobsMarshal(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)

	_, err := db.Search(context.Background(), SearchRequest{
		Query: []float32{1},
		RankBy: DecayRank(Decay{
			Field: "ts", Origin: 1700000000000,
			CountField: "nidus.access_count", CountScale: 20, CountLambda: 0.5,
		}),
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	want := `{"query":[1],"rank_by":{"Decay":{"field":"ts","origin":1700000000000,` +
		`"count_field":"nidus.access_count","count_scale":20,"count_lambda":0.5}}}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s", body, want)
	}
}

// TestDecayWithoutCountKnobsIsUnchanged is the assertion that makes generalising Decay
// instead of adding a variant safe: no count_* key travels unless CountField is set.
func TestDecayWithoutCountKnobsIsUnchanged(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)

	_, err := db.Search(context.Background(), SearchRequest{
		Query:  []float32{1},
		RankBy: DecayRank(Decay{Field: "ts", Origin: 1700000000000}),
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	want := `{"query":[1],"rank_by":{"Decay":{"field":"ts","origin":1700000000000}}}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want no count_* keys", body)
	}
}

// A RankBy on a recall must reach the wire as the same tagged union /search takes, and must
// be absent when unset: a recall that names no ranking expression is the plain metric.
func TestRecallRankByMarshalsAndIsOmittedWhenUnset(t *testing.T) {
	withRankBy := RecallOptions{
		RankBy: DecayRank(Decay{CountField: "nidus.access_count", CountLambda: 10}),
	}.wire("hello")
	b, err := json.Marshal(withRankBy)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	rb, ok := got["rank_by"].(map[string]any)
	if !ok {
		t.Fatalf("rank_by missing or not an object: %s", b)
	}
	d, ok := rb["Decay"].(map[string]any)
	if !ok {
		t.Fatalf("rank_by is not the Decay variant: %s", b)
	}
	if d["count_field"] != "nidus.access_count" {
		t.Errorf("count_field = %v, want nidus.access_count", d["count_field"])
	}

	plain, err := json.Marshal(RecallOptions{}.wire("hello"))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if bytes.Contains(plain, []byte("rank_by")) {
		t.Errorf("a recall naming no ranking expression must omit rank_by: %s", plain)
	}
}
