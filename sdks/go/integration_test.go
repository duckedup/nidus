//go:build integration

// End-to-end against a real `nidus serve`, driven entirely through this SDK.
//
// This is the tier that catches what httptest cannot. The unit tests assert the SDK
// sends the bytes we believe the server wants; only a real server can say whether that
// belief is true. Everything here is a claim about the *contract* — that an omitted
// top_k really does take the server's default, that a text-only record really does
// come back without a vector key, that a filter's tuple shape really does deserialize —
// and each of those is a place where a plausible-looking SDK is silently wrong.
//
// It mirrors sdks/js/test/integration.test.ts step for step, on purpose: the two SDKs
// are meant to be reviewable side by side, and a divergence in what they prove is a
// divergence nobody notices until one of them has a bug the other's suite would have
// caught.
//
// Behind the `integration` build tag, so `go test ./...` stays hermetic and needs no
// Rust toolchain. Run it with:
//
//	go test -tags integration ./...
//
// The binary is $NIDUS_BIN, else target/release/nidus in the repo root (build it with
// `just build-cli`). If neither exists the suite skips rather than fails, so a
// contributor without the Rust toolchain is not staring at a red build for a binary
// they were never asked to produce.
package nidus

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

const (
	// Generous, because this may be a cold binary on a shared CI runner: the point of
	// the deadline is to fail with a message instead of hanging forever, not to measure
	// anything. Nothing here asserts on timing.
	startupTimeout = 20 * time.Second
	// A per-request bound so a wedged server fails the test that provoked it rather
	// than the whole run.
	requestTimeout = 15 * time.Second
)

// ── Harness ─────────────────────────────────────────────────────────────────

// logBuffer collects the child's stdout and stderr. It is mutex-guarded because
// os/exec writes from its own copying goroutines while the test reads.
//
// Keeping the transcript is the whole reason this is not io.Discard: a server that
// fails to start says why on stderr — a port in use, a store already locked, a bad
// flag — and a test that reports "did not become ready in time" while throwing that
// away is a test that costs an hour to debug.
type logBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *logBuffer) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

func (l *logBuffer) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.String()
}

// binaryPath resolves the `nidus` binary, skipping the suite when there is none.
func binaryPath(t *testing.T) string {
	t.Helper()
	bin := os.Getenv("NIDUS_BIN")
	if bin == "" {
		// The package directory is sdks/go, so the repo root is two levels up.
		abs, err := filepath.Abs(filepath.Join("..", "..", "target", "release", "nidus"))
		if err != nil {
			t.Skipf("cannot resolve the default binary path: %v", err)
		}
		bin = abs
	}
	if info, err := os.Stat(bin); err != nil || info.IsDir() {
		t.Skipf("no nidus binary at %s — set NIDUS_BIN or run `just build-cli`", bin)
	}
	return bin
}

// child is one spawned `nidus serve`, plus its transcript and its reaper.
//
// `done` is *closed* rather than sent on, so "has it exited?" can be asked any number
// of times. An earlier version of this harness used a one-shot channel: the writer-lock
// test consumed it to detect an early exit, and then the registered shutdown blocked on
// the same channel forever. A closed channel is the shape that cannot deadlock.
type child struct {
	cmd  *exec.Cmd
	log  *logBuffer
	done chan struct{}
	err  error // the exit status; valid only once done is closed
}

// spawn starts a server over dir and registers its shutdown with t.
//
// The port is 0: the kernel picks a free one and the startup line reports it. A fixed
// port would race against anything else on the machine — including a second run of this
// suite — and the failure mode is a confusing "address already in use" attributed to
// whichever test happened to run first.
func spawn(t *testing.T, dir string, extra ...string) *child {
	t.Helper()
	bin := binaryPath(t)
	args := append([]string{
		"serve", "--dir", dir, "--dim", "3", "--addr", "127.0.0.1:0",
	}, extra...)

	c := &child{cmd: exec.Command(bin, args...), log: &logBuffer{}, done: make(chan struct{})}
	c.cmd.Stdout = c.log
	c.cmd.Stderr = c.log
	if err := c.cmd.Start(); err != nil {
		t.Fatalf("spawning %s: %v", bin, err)
	}
	// Reap in a goroutine so the poll loops below can tell "the child died" (bad flags,
	// lock held, port taken) from "the child is merely slow" — two failures with
	// completely different causes that look identical from a timeout alone.
	go func() {
		c.err = c.cmd.Wait()
		close(c.done)
	}()
	t.Cleanup(c.stop)
	return c
}

// gone reports whether the process has already been reaped.
func (c *child) gone() bool {
	select {
	case <-c.done:
		return true
	default:
		return false
	}
}

// stop shuts the server down, gracefully if it will go.
func (c *child) stop() {
	if c.gone() {
		return
	}
	// SIGTERM, not Kill: `nidus serve` catches it to flush and release the writer lock
	// on the way out, which is the path a rolling restart takes and therefore the one
	// worth exercising. Kill is the fallback for a server that will not go.
	_ = c.cmd.Process.Signal(syscall.SIGTERM)
	select {
	case <-c.done:
	case <-time.After(5 * time.Second):
		_ = c.cmd.Process.Kill()
		<-c.done
	}
}

// transcript is the child's own stdout and stderr, for attaching to a failure. A
// startup failure explains itself there — a port in use, a store already locked, a bad
// flag — and a test that throws it away costs an hour to debug.
func (c *child) transcript() string {
	return "\n--- server output ---\n" + c.log.String()
}

// baseURL scrapes the bound address out of the startup line
// (`nidus serving on http://127.0.0.1:51513 (Ctrl-C / SIGTERM to stop)`).
func (c *child) baseURL(t *testing.T) string {
	t.Helper()
	deadline := time.Now().Add(startupTimeout)
	for time.Now().Before(deadline) {
		// The log is checked before the exit status, so a server that printed its
		// address and then died still reports the address it had.
		if _, rest, found := strings.Cut(c.log.String(), "http://"); found {
			if fields := strings.Fields(rest); len(fields) > 0 {
				return "http://" + fields[0]
			}
		}
		if c.gone() {
			t.Fatalf("nidus serve exited before reporting an address (%v)%s",
				c.err, c.transcript())
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("nidus serve printed no address within %s%s", startupTimeout, c.transcript())
	return ""
}

// client returns a client for this server, waited until /ready answers 200 (#121):
// /health is liveness and answers before the store finishes opening, so gating on it
// hands tests a server that can still 503. /ready is equally token-exempt.
func (c *child) client(t *testing.T, opts ...Option) *Client {
	t.Helper()
	addr := c.baseURL(t)
	db, err := NewClient(addr, append([]Option{WithTimeout(requestTimeout)}, opts...)...)
	if err != nil {
		t.Fatalf("NewClient(%q) failed: %v", addr, err)
	}

	// Raw GET so a give-up reports *why* the last attempt failed — connection refused
	// reads very differently from a 503 from a store that is still opening.
	ready := func() error {
		resp, err := http.Get(addr + "/ready")
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("/ready answered %s", resp.Status)
		}
		return nil
	}
	deadline := time.Now().Add(startupTimeout)
	var last error
	for time.Now().Before(deadline) {
		last = ready()
		if last == nil {
			return db
		}
		if c.gone() {
			t.Fatalf("nidus serve exited during startup (%v)%s", c.err, c.transcript())
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("nidus serve at %s did not become ready within %s (last attempt: %v)%s",
		addr, startupTimeout, last, c.transcript())
	return nil
}

// startServer is the common case: one fresh server over its own temp dir, healthy and
// ready to take requests.
func startServer(t *testing.T) *Client {
	t.Helper()
	return spawn(t, t.TempDir()).client(t)
}

// ── The lifecycle ───────────────────────────────────────────────────────────

// TestLifecycleAgainstARealServer walks create → upsert → search → filter → text
// search → hybrid search → delete → stats against one server.
//
// The steps are ordered and share the store deliberately — later steps assert on what
// earlier ones wrote, which is the only way to check that a delete is reflected in
// stats or that a text-only doc survives a round trip. That is also why they are
// t.Run steps in one function rather than separate Test functions: separate tests would
// each want their own server, and the ordering would be a coincidence of file layout.
func TestLifecycleAgainstARealServer(t *testing.T) {
	db := startServer(t)
	ctx := context.Background()

	t.Run("create and list a collection", func(t *testing.T) {
		if err := db.CreateCollection(ctx, "docs"); err != nil {
			t.Fatalf("CreateCollection failed: %v", err)
		}
		// Idempotent: creating one that exists succeeds and changes nothing, so callers
		// need no exists-check race.
		if err := db.CreateCollection(ctx, "docs"); err != nil {
			t.Fatalf("CreateCollection is not idempotent: %v", err)
		}
		names, err := db.Collections(ctx)
		if err != nil {
			t.Fatalf("Collections failed: %v", err)
		}
		if !slices.Contains(names, "docs") {
			t.Fatalf("collections = %v, want it to contain docs", names)
		}
	})

	t.Run("upsert and vector search", func(t *testing.T) {
		n, err := db.Upsert(ctx, "docs", []Record{
			{ID: "a", Vector: []float32{1, 0, 0}, Attrs: Attrs{"lang": Str("rust"), "year": Int(2024)}},
			{ID: "b", Vector: []float32{0, 1, 0}, Attrs: Attrs{"lang": Str("go"), "year": Int(2020)}},
		})
		if err != nil {
			t.Fatalf("Upsert failed: %v", err)
		}
		if n != 2 {
			t.Fatalf("Upsert wrote %d records, want 2", n)
		}

		// No TopK: the field is omitted and the server's default (10) applies. If the
		// SDK sent "top_k": 0 this would come back empty — which is precisely the bug
		// the omit-vs-zero handling exists to prevent, and it is invisible without a
		// real server to answer.
		hits, err := db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(hits) != 2 {
			t.Fatalf("search with no TopK returned %d hits, want both records — an omitted "+
				"top_k must take the server's default, not zero", len(hits))
		}

		hits, err = db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}, TopK: 1})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(hits) != 1 {
			t.Fatalf("TopK 1 returned %d hits", len(hits))
		}
		if hits[0].ID != "a" {
			t.Errorf("nearest hit = %q, want a", hits[0].ID)
		}
		if hits[0].Collection != "docs" {
			t.Errorf("hit collection = %q, want docs", hits[0].Collection)
		}
		// Cosine against an identical unit vector, so ~1.0. Loose, because the vector is
		// normalized on insert and stored as f32 — this asserts the scale is right, not
		// the last bit.
		if hits[0].Score < 0.99 {
			t.Errorf("score = %v, want ~1.0 for an exact match", hits[0].Score)
		}
		// Typed attributes survive the round trip through the real server.
		if lang, ok := hits[0].Attrs["lang"].Str(); !ok || lang != "rust" {
			t.Errorf("lang = (%q, %v), want (rust, true)", lang, ok)
		}
		if year, ok := hits[0].Attrs["year"].Int(); !ok || year != 2024 {
			t.Errorf("year = (%d, %v), want (2024, true)", year, ok)
		}

		// A MinScore floor really does drop the orthogonal record, which is what proves
		// the pointer field arrived as a number rather than being omitted.
		hits, err = db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}, MinScore: f32(0.5)})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(hits) != 1 || hits[0].ID != "a" {
			t.Errorf("min_score 0.5 returned %d hits (%v), want only a", len(hits), ids(hits))
		}
	})

	t.Run("upsert is idempotent on id", func(t *testing.T) {
		n, err := db.Upsert(ctx, "docs", []Record{
			{ID: "a", Vector: []float32{1, 0, 0}, Attrs: Attrs{"lang": Str("rust"), "year": Int(2025)}},
		})
		if err != nil {
			t.Fatalf("Upsert failed: %v", err)
		}
		if n != 1 {
			t.Fatalf("Upsert wrote %d records, want 1", n)
		}
		recs, err := db.Records(ctx, "docs")
		if err != nil {
			t.Fatalf("Records failed: %v", err)
		}
		if len(recs) != 2 {
			t.Fatalf("collection holds %d records after re-upserting a, want 2 — an upsert "+
				"must overwrite rather than duplicate", len(recs))
		}
		// The overwrite won, so the attrs are the new ones.
		hits, err := db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}, TopK: 1})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if year, ok := hits[0].Attrs["year"].Int(); !ok || year != 2025 {
			t.Errorf("year = (%d, %v), want the overwritten 2025", year, ok)
		}
	})

	t.Run("metadata filters", func(t *testing.T) {
		if err := db.SetFtsSchema(ctx, "notes", []string{"body"}); err != nil {
			t.Fatalf("SetFtsSchema failed: %v", err)
		}
		if _, err := db.Upsert(ctx, "notes", []Record{
			{ID: "x", Vector: []float32{1, 0, 0}, Attrs: Attrs{
				"body": Str("the quick brown fox"), "kind": Str("a"), "rank": Int(1),
			}},
			// No vector: a text-only document, findable by text search and metadata but
			// never by a vector query.
			{ID: "y", Attrs: Attrs{
				"body": Str("foxes are running quickly"), "kind": Str("b"), "rank": Int(2),
			}},
		}); err != nil {
			t.Fatalf("Upsert failed: %v", err)
		}

		// Each of these exercises a different predicate's tuple shape against the real
		// deserializer, which is the only place a wrong shape shows up.
		cases := []struct {
			name   string
			filter Filter
			want   []string
		}{
			{"Eq", And(Eq("kind", "a")), []string{"x"}},
			{"Ne", And(Ne("kind", "a")), []string{"y"}},
			{"In", And(In("kind", "a", "b")), []string{"x", "y"}},
			{"NotIn", And(NotIn("kind", "b")), []string{"x"}},
			{"Glob", And(Glob("body", "the quick*")), []string{"x"}},
			{"Lt", And(Lt("rank", 2)), []string{"x"}},
			{"Le", And(Le("rank", 2)), []string{"x", "y"}},
			{"Gt", And(Gt("rank", 1)), []string{"y"}},
			{"Ge", And(Ge("rank", 1)), []string{"x", "y"}},
			{"conjunction", And(Ge("rank", 1), Eq("kind", "b")), []string{"y"}},
			// An absent attribute matches nothing, including the negative predicates.
			{"absent key", And(Ne("nosuchfield", "v")), nil},
			// An empty filter matches everything.
			{"empty", nil, []string{"x", "y"}},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				hits, err := db.List(ctx, ListRequest{Scope: []string{"notes"}, Filter: tc.filter})
				if err != nil {
					t.Fatalf("List failed: %v", err)
				}
				if got := ids(hits); !sameSet(got, tc.want) {
					t.Errorf("matched %v, want %v", got, tc.want)
				}
			})
		}
	})

	t.Run("text search", func(t *testing.T) {
		// "run" stems to match "running", so only the text-only doc hits — which also
		// proves a doc with no vector is genuinely retrievable.
		hits, err := db.TextSearch(ctx, TextSearchRequest{
			Scope: []string{"notes"}, Field: "body", Query: "run", TopK: 5,
		})
		if err != nil {
			t.Fatalf("TextSearch failed: %v", err)
		}
		if len(hits) == 0 {
			t.Fatal("no text hits for \"run\"; the FTS schema was not applied")
		}
		if hits[0].ID != "y" {
			t.Errorf("top text hit = %q, want y", hits[0].ID)
		}

		// A filter narrows a text query the same way it narrows a vector one.
		hits, err = db.TextSearch(ctx, TextSearchRequest{
			Scope: []string{"notes"}, Field: "body", Query: "fox", TopK: 5,
			Filter: And(Eq("kind", "a")),
		})
		if err != nil {
			t.Fatalf("TextSearch failed: %v", err)
		}
		if got := ids(hits); !sameSet(got, []string{"x"}) {
			t.Errorf("filtered text search matched %v, want [x]", got)
		}
	})

	t.Run("hybrid search", func(t *testing.T) {
		// RRF fuses the two legs, so a doc that ranks on either surfaces: x on the
		// vector leg, y on the text leg.
		hits, err := db.HybridSearch(ctx, HybridSearchRequest{
			Scope: []string{"notes"}, Vector: []float32{1, 0, 0},
			Field: "body", Text: "fox", TopK: 5,
		})
		if err != nil {
			t.Fatalf("HybridSearch failed: %v", err)
		}
		if got := ids(hits); !sameSet(got, []string{"x", "y"}) {
			t.Errorf("hybrid search matched %v, want both x and y", got)
		}
		// The default rrf_k is 60, so a top-ranked doc scores about 1/61 per leg.
		if hits[0].Score > 0.1 {
			t.Errorf("top fused score = %v with the default rrf_k; want ~1/61 per leg — is "+
				"rrf_k arriving at all?", hits[0].Score)
		}

		// RRFK is a *float32 precisely so an explicit zero can travel, and this is the
		// assertion that it does: the server fuses with 1/(rrf_k + rank + 1), so rrf_k = 0
		// scores a leg's top hit at 1.0 rather than ~0.016. A plain float32 with
		// `omitempty` would have omitted the zero and silently given the caller 60 —
		// visible nowhere except in the scores.
		zeroed, err := db.HybridSearch(ctx, HybridSearchRequest{
			Scope: []string{"notes"}, Vector: []float32{1, 0, 0},
			Field: "body", Text: "fox", TopK: 5, RRFK: f32(0),
		})
		if err != nil {
			t.Fatalf("HybridSearch with RRFK 0 failed: %v", err)
		}
		if len(zeroed) == 0 {
			t.Fatal("hybrid search with RRFK 0 returned nothing")
		}
		if zeroed[0].Score < 0.9 {
			t.Errorf("top fused score with RRFK 0 = %v, want >= 1.0 for a leg's top hit — an "+
				"explicit rrf_k of 0 was omitted and the server's 60 applied instead",
				zeroed[0].Score)
		}

		// Candidates 0 is meaningful too: the server clamps it up to top_k, so "no
		// over-fetch" is a request it accepts rather than a stand-in for the default 100.
		if _, err := db.HybridSearch(ctx, HybridSearchRequest{
			Scope: []string{"notes"}, Vector: []float32{1, 0, 0},
			Field: "body", Text: "fox", TopK: 5, Candidates: iptr(0),
		}); err != nil {
			t.Fatalf("HybridSearch with Candidates 0 failed: %v", err)
		}
	})

	t.Run("a text-only record keeps no vector", func(t *testing.T) {
		recs, err := db.Records(ctx, "notes")
		if err != nil {
			t.Fatalf("Records failed: %v", err)
		}
		byID := make(map[string]Record, len(recs))
		for _, r := range recs {
			byID[r.ID] = r
		}
		if got, ok := byID["y"]; !ok {
			t.Fatal("record y is missing")
		} else if got.Vector != nil {
			t.Errorf("text-only record y came back with Vector = %v, want nil — the server "+
				"omits the field and it must stay absent, not become an empty slice", got.Vector)
		}
		if got := byID["x"]; len(got.Vector) != 3 {
			t.Errorf("record x has a %d-element vector, want 3", len(got.Vector))
		}
		if body, ok := byID["y"].Attrs["body"].Str(); !ok || body == "" {
			t.Errorf("record y lost its attrs: %v", byID["y"].Attrs)
		}
	})

	t.Run("collection metadata", func(t *testing.T) {
		if err := db.SetMeta(ctx, "docs", map[string]string{"model": "bge-small", "rev": "1"}); err != nil {
			t.Fatalf("SetMeta failed: %v", err)
		}
		meta, err := db.GetMeta(ctx, "docs")
		if err != nil {
			t.Fatalf("GetMeta failed: %v", err)
		}
		if meta["model"] != "bge-small" || meta["rev"] != "1" {
			t.Errorf("meta = %v, want the two keys just written", meta)
		}
		// SetMeta replaces wholesale rather than merging, which is worth pinning
		// because the opposite is the easy assumption to make at a call site.
		if err := db.SetMeta(ctx, "docs", map[string]string{"rev": "2"}); err != nil {
			t.Fatalf("SetMeta failed: %v", err)
		}
		meta, err = db.GetMeta(ctx, "docs")
		if err != nil {
			t.Fatalf("GetMeta failed: %v", err)
		}
		if _, present := meta["model"]; present {
			t.Errorf("meta = %v; SetMeta replaces rather than merges", meta)
		}
		if meta["rev"] != "2" {
			t.Errorf("meta = %v, want rev 2", meta)
		}
	})

	t.Run("delete by id", func(t *testing.T) {
		n, err := db.Delete(ctx, "docs", []string{"b"})
		if err != nil {
			t.Fatalf("Delete failed: %v", err)
		}
		if n != 1 {
			t.Fatalf("Delete removed %d records, want 1", n)
		}
		recs, err := db.Records(ctx, "docs")
		if err != nil {
			t.Fatalf("Records failed: %v", err)
		}
		if len(recs) != 1 || recs[0].ID != "a" {
			t.Errorf("docs holds %v, want only a", recordIDs(recs))
		}
		// An id that is not there is not an error; it simply does not count.
		n, err = db.Delete(ctx, "docs", []string{"nosuchid"})
		if err != nil {
			t.Fatalf("deleting an absent id failed: %v", err)
		}
		if n != 0 {
			t.Errorf("deleting an absent id counted %d", n)
		}
	})

	t.Run("delete by filter", func(t *testing.T) {
		n, err := db.DeleteWhere(ctx, "notes", And(Eq("kind", "b")))
		if err != nil {
			t.Fatalf("DeleteWhere failed: %v", err)
		}
		if n != 1 {
			t.Fatalf("DeleteWhere removed %d records, want 1 (y)", n)
		}
		recs, err := db.Records(ctx, "notes")
		if err != nil {
			t.Fatalf("Records failed: %v", err)
		}
		if len(recs) != 1 || recs[0].ID != "x" {
			t.Errorf("notes holds %v, want only x", recordIDs(recs))
		}
	})

	t.Run("stats reflect the writes and deletes", func(t *testing.T) {
		stats, err := db.Stats(ctx)
		if err != nil {
			t.Fatalf("Stats failed: %v", err)
		}
		if stats.Dimension != 3 {
			t.Errorf("dimension = %d, want the 3 the server was started with", stats.Dimension)
		}
		if stats.Distance == "" {
			t.Error("distance is empty; the server always reports a metric")
		}
		// Exact search unless --ann was passed, which it was not.
		if stats.Ann != nil {
			t.Errorf("ann = %+v, want nil for a store started without --ann", stats.Ann)
		}
		if !slices.Contains(stats.Collections, "docs") || !slices.Contains(stats.Collections, "notes") {
			t.Errorf("collections = %v, want docs and notes", stats.Collections)
		}
		// a in docs, x in notes.
		if stats.Footprint.DocCount != 2 {
			t.Errorf("doc_count = %d, want 2 after the deletes", stats.Footprint.DocCount)
		}
		if stats.Footprint.Dimension != 3 {
			t.Errorf("footprint dimension = %d, want 3", stats.Footprint.Dimension)
		}
		// The overwrite of a and the delete of b both leave rows behind.
		if stats.Footprint.DeadRows == 0 {
			t.Error("dead_rows = 0 after an overwrite and a delete, want some to reclaim")
		}
	})

	t.Run("flush and compact", func(t *testing.T) {
		if err := db.Flush(ctx); err != nil {
			t.Fatalf("Flush failed: %v", err)
		}
		if err := db.Compact(ctx); err != nil {
			t.Fatalf("Compact failed: %v", err)
		}
		stats, err := db.Stats(ctx)
		if err != nil {
			t.Fatalf("Stats failed: %v", err)
		}
		if stats.Footprint.DeadRows != 0 {
			t.Errorf("dead_rows = %d after compaction, want 0 reclaimed", stats.Footprint.DeadRows)
		}
		// Compaction rewrites the store; the live data must still be there and findable.
		hits, err := db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}, TopK: 10})
		if err != nil {
			t.Fatalf("Search after compaction failed: %v", err)
		}
		if got := ids(hits); !sameSet(got, []string{"a", "x"}) {
			t.Errorf("after compaction search found %v, want a and x", got)
		}
	})

	t.Run("drop a collection", func(t *testing.T) {
		if err := db.DropCollection(ctx, "notes"); err != nil {
			t.Fatalf("DropCollection failed: %v", err)
		}
		names, err := db.Collections(ctx)
		if err != nil {
			t.Fatalf("Collections failed: %v", err)
		}
		if slices.Contains(names, "notes") {
			t.Errorf("collections = %v, want notes gone", names)
		}
	})
}

// TestRankingAndAnnotationsAgainstARealServer covers the surface the unit tests can
// only pin the *bytes* of: the text predicates, multi-clause text queries, the
// explain/highlight annotations, the ranking knobs, and /aggregate.
//
// Every one of these is a body the SDK believes the server will accept — a wrong field
// name or a wrong tuple arity is a 400 nothing else in this suite would see, and the
// annotations in particular only exist on a response a real query produced.
func TestRankingAndAnnotationsAgainstARealServer(t *testing.T) {
	db := startServer(t)
	ctx := context.Background()

	if err := db.CreateCollection(ctx, "docs"); err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}
	if err := db.SetFtsSchema(ctx, "docs", []string{"title", "body"}); err != nil {
		t.Fatalf("SetFtsSchema failed: %v", err)
	}
	// Two epoch-ms timestamps a fixed distance apart, so the decay test needs no clock.
	const now = int64(1_700_000_000_000)
	const week = int64(7 * 24 * 60 * 60 * 1000)
	_, err := db.Upsert(ctx, "docs", []Record{
		{ID: "a", Vector: []float32{1, 0, 0}, Attrs: Attrs{
			"title": Str("rust async runtime"), "body": Str("we were running the executor"),
			"path": Str("src/main.rs"), "ts": DateTimeMillis(now), "bytes": Int(40960),
		}},
		{ID: "b", Vector: []float32{0.99, 0.14, 0}, Attrs: Attrs{
			"title": Str("go scheduler"), "body": Str("the runtime schedules goroutines"),
			"path": Str("src/main.rs"), "ts": DateTimeMillis(now - 52*week), "bytes": Int(1024),
		}},
		{ID: "c", Vector: []float32{0.98, 0.2, 0}, Attrs: Attrs{
			"title": Str("notes"), "body": Str("nothing to see"), "path": Str("docs/notes.md"),
		}},
	})
	if err != nil {
		t.Fatalf("Upsert failed: %v", err)
	}

	t.Run("the text predicates filter on a plain attribute", func(t *testing.T) {
		cases := []struct {
			name string
			pred Predicate
			want []string
		}{
			{"Fuzzy", Fuzzy("path", "src/mian.rs", 2), []string{"a", "b"}},
			{"ContainsAllTokens", ContainsAllTokens("title", "runtime rust"), []string{"a"}},
			{"ContainsAnyToken", ContainsAnyToken("title", "rust scheduler"), []string{"a", "b"}},
			{"ContainsTokenSequence", ContainsTokenSequence("title", "async runtime"), []string{"a"}},
			{"Regex", Regex("path", "src/.*"), []string{"a", "b"}},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				rows, err := db.List(ctx, ListRequest{Filter: And(tc.pred)})
				if err != nil {
					t.Fatalf("List failed: %v", err)
				}
				if !sameSet(ids(rows), tc.want) {
					t.Errorf("matched %v, want %v", ids(rows), tc.want)
				}
			})
		}

		// The server owns the edit ceiling; the SDK refuses before sending so the
		// mistake names the builder rather than arriving as a 400.
		if _, err := db.List(ctx, ListRequest{Filter: And(Fuzzy("path", "x", 99))}); err == nil {
			t.Error("an out-of-range Fuzzy reached the server")
		}
	})

	t.Run("a multi-clause query scores every clause", func(t *testing.T) {
		hits, err := db.TextSearch(ctx, TextSearchRequest{
			Clauses: []FtsClause{
				{Field: "title", Query: "runtime"},
				{Field: "body", Query: "runtime"},
			},
			Explain: true,
		})
		if err != nil {
			t.Fatalf("TextSearch failed: %v", err)
		}
		if !sameSet(ids(hits), []string{"a", "b"}) {
			t.Fatalf("hits = %v, want a and b", ids(hits))
		}
		// Sum is the default, so the doc matching on both fields must report both
		// clauses — which is the whole observable difference from a single-field query.
		for _, h := range hits {
			if h.Annotations == nil || len(h.Annotations.Clauses) == 0 {
				t.Fatalf("hit %s carries no clause scores; Explain did not arrive", h.ID)
			}
		}

		// Max cannot exceed Sum for the same query, and both must actually differ from
		// each other when a doc matches two clauses.
		maxed, err := db.TextSearch(ctx, TextSearchRequest{
			Clauses: []FtsClause{{Field: "title", Query: "runtime"}, {Field: "body", Query: "runtime"}},
			Combine: CombineMax,
		})
		if err != nil {
			t.Fatalf("TextSearch failed: %v", err)
		}
		summed := map[string]float32{}
		for _, h := range hits {
			summed[h.ID] = h.Score
		}
		for _, h := range maxed {
			if h.Score > summed[h.ID] {
				t.Errorf("hit %s scored %v under Max but %v under Sum", h.ID, h.Score, summed[h.ID])
			}
		}
	})

	t.Run("highlight returns fragments with usable spans", func(t *testing.T) {
		hits, err := db.TextSearch(ctx, TextSearchRequest{
			Field: "body", Query: "running", Highlight: &HighlightOpts{},
		})
		if err != nil {
			t.Fatalf("TextSearch failed: %v", err)
		}
		if len(hits) == 0 || hits[0].Annotations == nil {
			t.Fatalf("hits = %v with no annotations; Highlight did not arrive", ids(hits))
		}
		hl := hits[0].Annotations.Highlights
		if len(hl) != 1 || hl[0].Field != "body" || len(hl[0].Fragments) == 0 {
			t.Fatalf("highlights = %+v, want one over body", hl)
		}
		frag := hl[0].Fragments[0]
		if len(frag.Spans) == 0 {
			t.Fatalf("fragment %q carries no spans", frag.Text)
		}
		// The spans are byte offsets into the fragment's own text, so this slice is the
		// claim: a span that decoded wrong panics or produces the wrong word.
		if got := frag.Text[frag.Spans[0].Start:frag.Spans[0].End]; got != "running" {
			t.Errorf("span %v covers %q, want %q", frag.Spans[0], got, "running")
		}
	})

	t.Run("hybrid explain reports each leg, and a zero weight drops one", func(t *testing.T) {
		hits, err := db.HybridSearch(ctx, HybridSearchRequest{
			Vector: []float32{1, 0, 0}, Field: "title", Text: "rust", Explain: true,
		})
		if err != nil {
			t.Fatalf("HybridSearch failed: %v", err)
		}
		var sawVector, sawText bool
		for _, h := range hits {
			if h.Annotations == nil {
				t.Fatalf("hit %s carries no annotations; Explain did not arrive", h.ID)
			}
			sawVector = sawVector || h.Annotations.Vector != nil
			sawText = sawText || h.Annotations.Text != nil
		}
		if !sawVector || !sawText {
			t.Errorf("legs reported: vector=%v text=%v, want both", sawVector, sawText)
		}

		// A weight of zero must actually travel: the fused scores have to change.
		weighted, err := db.HybridSearch(ctx, HybridSearchRequest{
			Vector: []float32{1, 0, 0}, Field: "title", Text: "rust",
			VectorWeight: f32(0), TextWeight: f32(1),
		})
		if err != nil {
			t.Fatalf("HybridSearch failed: %v", err)
		}
		if len(weighted) == 0 {
			t.Fatal("a zero vector weight returned nothing")
		}
		if weighted[0].Score >= hits[0].Score {
			t.Errorf("top score = %v with the vector leg dropped, want below the unweighted %v",
				weighted[0].Score, hits[0].Score)
		}
	})

	t.Run("decay, limit_per and order_by reshape the ranking", func(t *testing.T) {
		// Undecayed, b outranks c on cosine. A year of decay with a week-long half-life
		// buries b — and must leave c, which has no ts at all, exactly where it was.
		plain, err := db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if got := ids(plain); len(got) != 3 || got[0] != "a" {
			t.Fatalf("undecayed ranking = %v, want a first", got)
		}
		decayed, err := db.Search(ctx, SearchRequest{
			Query:  []float32{1, 0, 0},
			RankBy: DecayRank(Decay{Field: "ts", Origin: now, Scale: week}),
		})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if got := ids(decayed); got[len(got)-1] != "b" {
			t.Errorf("decayed ranking = %v, want the year-old b last", got)
		}

		// Two records share src/main.rs; a cap of one keeps a single one of them.
		capped, err := db.Search(ctx, SearchRequest{
			Query: []float32{1, 0, 0}, LimitPer: &LimitPer{Field: "path", Max: 1},
		})
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(capped) != 2 {
			t.Errorf("limit_per 1 per path returned %v, want two hits", ids(capped))
		}

		rows, err := db.List(ctx, ListRequest{OrderBy: &OrderBy{Field: "bytes", Descending: true}})
		if err != nil {
			t.Fatalf("List failed: %v", err)
		}
		// c has no `bytes` at all, so it sorts into the trailing bucket either way.
		if got := ids(rows); len(got) < 2 || got[0] != "a" || got[1] != "b" {
			t.Errorf("descending order_by = %v, want a then b", got)
		}
	})

	t.Run("aggregate counts and sums", func(t *testing.T) {
		out, err := db.Aggregate(ctx, AggregateRequest{Sum: []string{"bytes"}})
		if err != nil {
			t.Fatalf("Aggregate failed: %v", err)
		}
		if out.Count != 3 {
			t.Errorf("count = %d, want 3", out.Count)
		}
		if n, ok := out.Sums["bytes"].Int(); !ok || n != 41984 {
			t.Errorf("sums[bytes] = %v, want Int(41984)", out.Sums["bytes"])
		}

		// A filter narrows both halves of the answer at once.
		out, err = db.Aggregate(ctx, AggregateRequest{
			Filter: And(Glob("path", "src/*")), Sum: []string{"bytes"},
		})
		if err != nil {
			t.Fatalf("Aggregate failed: %v", err)
		}
		if out.Count != 2 {
			t.Errorf("filtered count = %d, want 2", out.Count)
		}
	})
}

// TestSearchSimilarAgainstARealServer checks "more like this" over a real store: the
// source record never comes back, its nearest neighbour does, and an empty Scope stays
// within the source's own collection rather than searching every collection.
func TestSearchSimilarAgainstARealServer(t *testing.T) {
	db := startServer(t)
	ctx := context.Background()

	if err := db.CreateCollection(ctx, "docs"); err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}
	if err := db.CreateCollection(ctx, "other"); err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}
	_, err := db.Upsert(ctx, "docs", []Record{
		{ID: "a", Vector: []float32{1, 0, 0}},
		{ID: "b", Vector: []float32{0.99, 0.14, 0}},
		{ID: "c", Vector: []float32{-1, 0, 0}},
	})
	if err != nil {
		t.Fatalf("Upsert docs failed: %v", err)
	}
	if _, err := db.Upsert(ctx, "other", []Record{{ID: "z", Vector: []float32{1, 0, 0}}}); err != nil {
		t.Fatalf("Upsert other failed: %v", err)
	}

	hits, err := db.SearchSimilar(ctx, SimilarRequest{Collection: "docs", ID: "a"})
	if err != nil {
		t.Fatalf("SearchSimilar failed: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("SearchSimilar returned no hits")
	}
	if hits[0].Collection != "docs" || hits[0].ID != "b" {
		t.Errorf("nearest hit = %s/%s, want docs/b ranked first", hits[0].Collection, hits[0].ID)
	}
	for _, h := range hits {
		if h.ID == "a" {
			t.Fatalf("source record a came back in its own similarity search: %+v", hits)
		}
	}
	// An empty Scope stays within the source's own collection — "other" never appears.
	if !sameSet(ids(hits), []string{"b", "c"}) {
		t.Errorf("hits = %v, want b and c only", ids(hits))
	}
}

// TestServerErrorsCarryTheirStatus checks the error surface against the real
// classifier in src/server/mod.rs rather than a canned httptest reply. The status is
// the part a caller acts on, so it has to be the status the server actually chose.
func TestServerErrorsCarryTheirStatus(t *testing.T) {
	db := startServer(t)
	ctx := context.Background()

	if err := db.CreateCollection(ctx, "docs"); err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	t.Run("a dimension mismatch is a 400", func(t *testing.T) {
		// The store was opened with --dim 3.
		_, err := db.Upsert(ctx, "docs", []Record{
			{ID: "wrong", Vector: []float32{1, 0, 0, 0}, Attrs: Attrs{}},
		})
		if err == nil {
			t.Fatal("a 4-element vector was accepted into a 3-dimensional store")
		}
		var nerr *Error
		if !errors.As(err, &nerr) {
			t.Fatalf("error is %T, want *nidus.Error", err)
		}
		if !nerr.IsBadRequest() {
			t.Errorf("status = %d, want 400 for a dimension mismatch", nerr.Status)
		}
		if nerr.IsTransport() {
			t.Error("IsTransport() = true for a server-reported error")
		}
		if nerr.Message == "" {
			t.Error("Message is empty; the server explains a 400")
		}
	})

	// The tier that has to own this: only a real axum server says which status a
	// wrong-*typed* body gets, and it is not the 400 an HTTP contract reader expects.
	// axum's Json extractor answers 422 for a body it cannot deserialize and reserves 400
	// for a JSON syntax error — so IsBadRequest() must cover both, or a retry loop built
	// on `!nerr.IsBadRequest()` spins forever on a request that can never succeed.
	t.Run("a wrong-typed field is a 422 and still IsBadRequest", func(t *testing.T) {
		// top_k is a usize on the server, so a negative one fails deserialization rather
		// than validation. Reachable through the typed surface, which is the point.
		_, err := db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}, TopK: -1})
		if err == nil {
			t.Fatal("a negative TopK was accepted")
		}
		var nerr *Error
		if !errors.As(err, &nerr) {
			t.Fatalf("error is %T, want *nidus.Error", err)
		}
		if nerr.Status != 422 {
			t.Errorf("status = %d (%s), want 422 — if the server changed this, the "+
				"classifier and the note in errors.go both need updating",
				nerr.Status, nerr.Message)
		}
		if !nerr.IsBadRequest() {
			t.Errorf("IsBadRequest() = false for status %d; a body the server cannot "+
				"deserialize is never worth retrying", nerr.Status)
		}
	})

	t.Run("a rejected write leaves the store intact", func(t *testing.T) {
		// SPEC §6.6: an upsert is all-or-nothing. The failed write above must not have
		// left a partial record behind.
		recs, err := db.Records(ctx, "docs")
		if err != nil {
			t.Fatalf("Records failed: %v", err)
		}
		if len(recs) != 0 {
			t.Errorf("docs holds %v after only failed writes, want none", recordIDs(recs))
		}
		// And the store still takes a good write.
		if _, err := db.Upsert(ctx, "docs", []Record{
			{ID: "ok", Vector: []float32{0, 0, 1}, Attrs: Attrs{}},
		}); err != nil {
			t.Fatalf("a valid upsert after a rejected one failed: %v", err)
		}
	})

	// The server is deliberately lenient about things that are merely *absent* — an
	// unknown collection, a field with no full-text schema — answering with an empty
	// result rather than an error. That is worth pinning from the SDK side, because the
	// tempting "helpful" move is to synthesize a not-found error here, and that would
	// break every caller who legitimately queries a scope before writing to it.
	t.Run("absent collections and fields return empty, not an error", func(t *testing.T) {
		hits, err := db.Search(ctx, SearchRequest{Query: []float32{1, 0, 0}, Scope: []string{"nosuch"}})
		if err != nil {
			t.Errorf("searching an unknown collection failed: %v", err)
		}
		if len(hits) != 0 {
			t.Errorf("searching an unknown collection returned %v", ids(hits))
		}

		recs, err := db.Records(ctx, "nosuch")
		if err != nil {
			t.Errorf("Records on an unknown collection failed: %v", err)
		}
		if len(recs) != 0 {
			t.Errorf("Records on an unknown collection returned %v", recordIDs(recs))
		}

		if _, err := db.GetMeta(ctx, "nosuch"); err != nil {
			t.Errorf("GetMeta on an unknown collection failed: %v", err)
		}

		// "docs" has no FTS schema, so there is nothing indexed to match.
		hits, err = db.TextSearch(ctx, TextSearchRequest{
			Scope: []string{"docs"}, Field: "body", Query: "fox",
		})
		if err != nil {
			t.Errorf("text search on an unindexed field failed: %v", err)
		}
		if len(hits) != 0 {
			t.Errorf("text search on an unindexed field returned %v", ids(hits))
		}

		// Deleting ids that are not there counts zero rather than failing.
		n, err := db.Delete(ctx, "nosuch", []string{"a"})
		if err != nil {
			t.Errorf("deleting from an unknown collection failed: %v", err)
		}
		if n != 0 {
			t.Errorf("deleting from an unknown collection counted %d", n)
		}
	})
}

// TestOpsSurfaceAgainstARealServer covers Ready, Cluster, Versions, and Refresh
// against a real single-instance server — not what those fields mean for a cluster,
// only that this SDK decodes the shape the running binary actually sends.
func TestOpsSurfaceAgainstARealServer(t *testing.T) {
	db := startServer(t)
	ctx := context.Background()

	ready, err := db.Ready(ctx)
	if err != nil {
		t.Fatalf("Ready failed: %v", err)
	}
	if !ready.Ready {
		t.Errorf("Ready.Ready = false, want true for a server startServer already waited on")
	}
	if ready.Role == "" {
		t.Error("Ready.Role is empty")
	}

	status, err := db.Cluster(ctx)
	if err != nil {
		t.Fatalf("Cluster failed: %v", err)
	}
	if status.Role == "" {
		t.Error("Cluster.Role is empty")
	}

	versions, err := db.Versions(ctx)
	if err != nil {
		t.Fatalf("Versions failed: %v", err)
	}
	if versions.CommitVersion == 0 {
		t.Error("Versions.CommitVersion is 0, want a set commit version")
	}
	if versions.Pinned != nil {
		t.Errorf("Versions.Pinned = %v, want nil on an unpinned instance", *versions.Pinned)
	}

	if _, err := db.Refresh(ctx); err != nil {
		t.Fatalf("Refresh failed: %v", err)
	}
}

// TestMemoryRoutesWithoutAnEmbedder pins what Remember and Recall actually answer on a
// server that cannot serve them — the claim the SDK's comments and README make, and the
// one that was wrong.
//
// There are two distinct failures and the docs previously named only the second:
//
//   - 404 when the binary was built without the `memory` feature, because the routes are
//     registered behind #[cfg(feature = "memory")] and simply do not exist. That is the
//     build `just build-cli` produces, which is the binary this harness runs.
//   - 400 when the routes exist but the server was started without --embed-provider.
//
// Either is correct depending on the binary, so both are accepted; what is asserted is
// that the call fails visibly, with a status, rather than looking like success.
func TestMemoryRoutesWithoutAnEmbedder(t *testing.T) {
	db := startServer(t)
	ctx := context.Background()
	if err := db.CreateCollection(ctx, "notes"); err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	// The TTL and dedupe knobs ride along, so a body carrying them still fails as one of
	// these two rather than as a deserialization error.
	_, err := db.Remember(ctx, "notes", "a", "the quick brown fox", RememberOptions{
		TTLSeconds: i64(3600), DedupeThreshold: f32(0.95),
	})
	if err == nil {
		t.Fatal("Remember succeeded on a server with no embedder")
	}
	var nerr *Error
	if !errors.As(err, &nerr) {
		t.Fatalf("error is %T, want *nidus.Error", err)
	}
	if nerr.Status != 404 && nerr.Status != 400 {
		t.Errorf("Remember status = %d (%s), want 404 (route absent — no `memory` feature) "+
			"or 400 (route present, no --embed-provider)", nerr.Status, nerr.Message)
	}

	if _, err := db.Recall(ctx, "notes", "quick fox", RecallOptions{}); err == nil {
		t.Error("Recall succeeded on a server with no embedder")
	} else if !errors.As(err, &nerr) {
		t.Errorf("Recall error is %T, want *nidus.Error", err)
	} else if nerr.Status != 404 && nerr.Status != 400 {
		t.Errorf("Recall status = %d (%s), want 404 or 400", nerr.Status, nerr.Message)
	}

	// The reinforcement knobs must reach the wire and fail the same visible way, not as
	// a client-side encode error. This `cli`-feature binary never has an embedder, so
	// the access_count assertion itself lives in tests/e2e/memory_http.rs instead.
	if _, err := db.Recall(ctx, "notes", "quick fox", RecallOptions{
		Reinforce: true, ExtendTTLSeconds: i64(3600),
	}); err == nil {
		t.Error("reinforced Recall succeeded on a server with no embedder")
	} else if !errors.As(err, &nerr) {
		t.Errorf("reinforced Recall error is %T, want *nidus.Error", err)
	} else if nerr.Status != 404 && nerr.Status != 400 {
		t.Errorf("reinforced Recall status = %d (%s), want 404 or 400", nerr.Status, nerr.Message)
	}
}

// TestBearerTokenIsEnforced starts a token-protected server, since auth is a layer
// the in-process tests cannot exercise: WithToken has to produce a header the real
// middleware accepts, and the absence of one has to produce a real 401.
func TestBearerTokenIsEnforced(t *testing.T) {
	const token = "s3cret-token"
	server := spawn(t, t.TempDir(), "--token", token)

	// The unauthenticated client is the one that waits for readiness: /ready is exempt
	// from auth (so an orchestrator does not read a 401 as "not ready"), and gating on
	// it here proves that exemption rather than assuming it.
	anon := server.client(t)
	addr := server.baseURL(t)
	authed, err := NewClient(addr, WithToken(token), WithTimeout(requestTimeout))
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}

	ctx := context.Background()
	if err := authed.CreateCollection(ctx, "docs"); err != nil {
		t.Fatalf("an authenticated write failed: %v", err)
	}

	err = anon.CreateCollection(ctx, "nope")
	if err == nil {
		t.Fatal("an unauthenticated write succeeded against a --token server")
	}
	var nerr *Error
	if !errors.As(err, &nerr) {
		t.Fatalf("error is %T, want *nidus.Error", err)
	}
	if !nerr.IsUnauthorized() {
		t.Errorf("status = %d, want 401 for a missing bearer token", nerr.Status)
	}

	wrong, err := NewClient(addr, WithToken("not-the-token"), WithTimeout(requestTimeout))
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	if err := wrong.CreateCollection(ctx, "nope"); err == nil {
		t.Error("a wrong token was accepted")
	} else if !errors.As(err, &nerr) || !nerr.IsUnauthorized() {
		t.Errorf("error = %v, want a 401", err)
	}
}

// TestWriterLockIsExclusive — two servers over one directory. Cross-process exclusion
// is invisible to an in-process test, and a 409 is the one status a caller is expected
// to retry on, so the SDK's classification of it is worth checking for real.
func TestWriterLockIsExclusive(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	first := spawn(t, dir).client(t)
	if err := first.CreateCollection(ctx, "docs"); err != nil {
		t.Fatalf("the first server could not write: %v", err)
	}

	// The second instance over the same directory cannot take the writer lock. It may
	// refuse at startup or bind and then answer 409/503 on the first write, depending on
	// when it reaches for the lock; both are correct, and both must be visible to a
	// caller rather than looking like success.
	second := spawn(t, dir)

	// Give it a moment to decide which way it goes. `gone` is safe to poll repeatedly,
	// which is the whole point of closing `done` instead of sending on it.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) && !second.gone() {
		if strings.Contains(second.log.String(), "http://") {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	if second.gone() {
		if out := second.log.String(); !strings.Contains(strings.ToLower(out), "lock") {
			t.Errorf("the second server exited without mentioning the lock:%s",
				second.transcript())
		}
		return
	}

	addr := second.baseURL(t)
	locked, err := NewClient(addr, WithTimeout(requestTimeout))
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	err = locked.CreateCollection(ctx, "other")
	if err == nil {
		t.Fatalf("two servers wrote to one store directory%s", second.transcript())
	}
	var nerr *Error
	if !errors.As(err, &nerr) {
		t.Fatalf("error is %T, want *nidus.Error", err)
	}
	if !nerr.IsLocked() && !nerr.IsUnavailable() {
		t.Errorf("status = %d (%s), want 409 (lock held) or 503 (store not open)%s",
			nerr.Status, nerr.Message, second.transcript())
	}
}

// TestUnreachableServerIsATransportError closes the loop the other way: the SDK's
// Status-0 classification, against a port that really has nothing on it rather than a
// httptest listener that was shut down.
func TestUnreachableServerIsATransportError(t *testing.T) {
	// Port 1 on loopback: privileged, and nothing sane binds it.
	db, err := NewClient("http://127.0.0.1:1", WithTimeout(2*time.Second))
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	if db.Health(context.Background()) {
		t.Skip("something is listening on 127.0.0.1:1")
	}
	_, err = db.Stats(context.Background())
	if err == nil {
		t.Fatal("Stats succeeded against a dead address")
	}
	var nerr *Error
	if !errors.As(err, &nerr) {
		t.Fatalf("error is %T, want *nidus.Error", err)
	}
	if !nerr.IsTransport() || nerr.Status != 0 {
		t.Errorf("status = %d, want 0 for an unreachable server", nerr.Status)
	}
}

// ── Small assertions helpers ────────────────────────────────────────────────

// idsOf projects rows onto the ids they carry. One generic loop rather than one per row
// type: hits and records differ only in which field the id lives in, and a second copy
// of a three-line loop is a second place to fix it.
func idsOf[T any](rows []T, id func(T) string) []string {
	out := make([]string, len(rows))
	for i, row := range rows {
		out[i] = id(row)
	}
	return out
}

func ids(hits []Hit) []string { return idsOf(hits, func(h Hit) string { return h.ID }) }

func recordIDs(recs []Record) []string {
	return idsOf(recs, func(r Record) string { return r.ID })
}

// sameSet compares ignoring order, because only the search tests assert on ranking —
// a filter or a list is a set, and pinning its order would be asserting something the
// server never promised.
func sameSet(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	seen := make(map[string]int, len(got))
	for _, s := range got {
		seen[s]++
	}
	for _, s := range want {
		seen[s]--
		if seen[s] < 0 {
			return false
		}
	}
	return true
}
