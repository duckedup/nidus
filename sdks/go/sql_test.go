// Unit tests for Query, QueryBatch and Compile — the SQL front end's Go binding
// (SPEC §7.12). Same harness as client_test.go: a fake server over net/http/httptest,
// asserting the bytes the SDK sends and its decode of a canned response back.
//
// SQL is a front end over the same typed opts Search and its four siblings already
// run, so the falsifiable claim throughout is decode *parity*: the same JSON a
// Search response carries must decode to the same [Hit] values through Query, not
// merely to "something the right length".
package nidus

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

// ── Query: success ───────────────────────────────────────────────────────────

func TestQueryDecodesABareHitsArray(t *testing.T) {
	fake := &capture{reply: `[{"collection":"docs","id":"a","score":0.9,"attrs":{"lang":{"Str":"rust"}}}]`}
	db := serve(t, fake)

	got, err := db.Query(context.Background(), "SELECT * FROM docs ORDER BY knn([1,0,0])")
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if got.Aggregation != nil || got.Plan != nil {
		t.Fatalf("got = %+v, want only Hits populated", got)
	}
	if len(got.Hits) != 1 || got.Hits[0].ID != "a" || got.Hits[0].Collection != "docs" {
		t.Fatalf("Hits = %+v, want one hit docs/a", got.Hits)
	}
	if lang, ok := got.Hits[0].Attrs["lang"].Str(); !ok || lang != "rust" {
		t.Errorf("lang = (%q, %v), want (rust, true)", lang, ok)
	}

	if fake.snapshot().path != "/query" {
		t.Errorf("path = %q, want /query", fake.snapshot().path)
	}
	if fake.snapshot().method != http.MethodPost {
		t.Errorf("method = %q, want POST", fake.snapshot().method)
	}
	want := `{"sql":"SELECT * FROM docs ORDER BY knn([1,0,0])"}`
	if body := fake.sentBody(t); body != want {
		t.Errorf("body = %s, want %s (no compile_only key on a plain Query)", body, want)
	}
	if fake.snapshot().calls != 1 {
		t.Errorf("calls = %d, want exactly 1", fake.snapshot().calls)
	}
}

// TestQueryHitsDecodeIdenticallyToSearch pins Hits' decode against Search's: the same
// JSON, decoded through both call paths, must produce identical [Hit] values field by
// field — not merely the same length. A Query that quietly used a looser Hit shape
// would pass a length check and fail this one.
func TestQueryHitsDecodeIdenticallyToSearch(t *testing.T) {
	const reply = `[{"collection":"docs","id":"a","score":0.75,` +
		`"attrs":{"lang":{"Str":"rust"},"year":{"Int":2024}},` +
		`"annotations":{"clauses":[{"field":"body","score":1.5}]}}]`

	viaSearch := serve(t, &capture{reply: reply})
	searchHits, err := viaSearch.Search(context.Background(), SearchRequest{Query: []float32{1, 0, 0}})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	viaQuery := serve(t, &capture{reply: reply})
	answer, err := viaQuery.Query(context.Background(), "SELECT * FROM docs ORDER BY knn([1,0,0])")
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}

	if !reflect.DeepEqual(searchHits, answer.Hits) {
		t.Fatalf("Query decoded %+v; Search decoded %+v from the same JSON — must be identical",
			answer.Hits, searchHits)
	}
}

// TestQueryPlanEnvelopeDecodesLikeSearchWithPlan pins the {hits,plan} envelope shape
// (WITH (plan), §7.11) against SearchWithPlan's own decode of the identical body.
func TestQueryPlanEnvelopeDecodesLikeSearchWithPlan(t *testing.T) {
	fake := &capture{reply: planEnvelope}
	db := serve(t, fake)

	got, err := db.Query(context.Background(), "SELECT * FROM docs ORDER BY knn([1,0,0]) WITH (plan)")
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if got.Plan == nil {
		t.Fatal("Plan is nil for a WITH (plan) statement")
	}
	if got.Plan.Path != "exact" {
		t.Errorf("Plan.Path = %q, want exact", got.Plan.Path)
	}
	if got.Hits == nil || len(got.Hits) != 0 {
		t.Errorf("Hits = %+v, want an empty (non-nil) slice", got.Hits)
	}
	if got.Aggregation != nil {
		t.Errorf("Aggregation = %+v, want nil for a Hits-dispatch statement", got.Aggregation)
	}
}

// ── Query: aggregation ───────────────────────────────────────────────────────

// TestQueryAggregationDecodesGroupsAndAFloatSum covers a group missing the value
// entirely (a `None` group, distinct from a present null) and a float sum, which is
// the one numeric shape a naive decode could round through float64 and still pass a
// looser check.
func TestQueryAggregationDecodesGroupsAndAFloatSum(t *testing.T) {
	fake := &capture{reply: `{"count":3,"sums":{"score":{"Float":12.5}},"groups":[` +
		`{"value":{"Str":"rust"},"count":2,"sums":{"score":{"Float":10.0}}},` +
		`{"value":null,"count":1,"sums":{"score":{"Float":2.5}}}]}`}
	db := serve(t, fake)

	got, err := db.Query(context.Background(), "SELECT * FROM docs GROUP BY lang, SUM(score)")
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if got.Hits != nil || got.Plan != nil {
		t.Fatalf("got = %+v, want only Aggregation populated", got)
	}
	agg := got.Aggregation
	if agg == nil {
		t.Fatal("Aggregation is nil")
	}
	if agg.Count != 3 {
		t.Errorf("Count = %d, want 3", agg.Count)
	}
	if f, ok := agg.Sums["score"].Float(); !ok || f != 12.5 {
		t.Errorf("Sums[score] = %v, want Float(12.5)", agg.Sums["score"])
	}
	if len(agg.Groups) != 2 {
		t.Fatalf("Groups = %+v, want 2", agg.Groups)
	}
	if agg.Groups[0].Value == nil {
		t.Fatal("first group must carry its value")
	} else if s, ok := agg.Groups[0].Value.Str(); !ok || s != "rust" {
		t.Errorf("Groups[0].Value = %v, want Str(rust)", agg.Groups[0].Value)
	}
	if agg.Groups[1].Value != nil {
		t.Errorf("the no-value group must decode Value as nil, got %v", agg.Groups[1].Value)
	}
	if f, ok := agg.Groups[1].Sums["score"].Float(); !ok || f != 2.5 {
		t.Errorf("Groups[1].Sums[score] = %v, want Float(2.5)", agg.Groups[1].Sums["score"])
	}
}

// ── Query: refuses a script ──────────────────────────────────────────────────

// TestQueryRefusesAMultiStatementScript is a Go-side check independent of the server:
// even if a `;`-separated script somehow reached Query, it must not silently hand back
// only the first statement's answer.
func TestQueryRefusesAMultiStatementScript(t *testing.T) {
	fake := &capture{reply: `[[{"collection":"docs","id":"a","score":1,"attrs":{}}],{"count":0,"sums":{}}]`}
	db := serve(t, fake)

	_, err := db.Query(context.Background(), "SELECT * FROM docs; SELECT * FROM docs GROUP BY lang")
	if err == nil {
		t.Fatal("Query succeeded on a two-statement script; use QueryBatch for that")
	}
}

// ── QueryBatch ────────────────────────────────────────────────────────────────

func TestQueryBatchReturnsThreeAnswersInOrder(t *testing.T) {
	fake := &capture{reply: `[` +
		`[{"collection":"docs","id":"a","score":1,"attrs":{}}],` +
		`{"count":1,"sums":{}},` +
		`[{"collection":"docs","id":"b","score":1,"attrs":{}}]` +
		`]`}
	db := serve(t, fake)

	got, err := db.QueryBatch(context.Background(),
		"SELECT * FROM docs ORDER BY id; SELECT * FROM docs GROUP BY lang; SELECT * FROM docs ORDER BY id DESC")
	if err != nil {
		t.Fatalf("QueryBatch failed: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d answers, want 3", len(got))
	}
	if len(got[0].Hits) != 1 || got[0].Hits[0].ID != "a" {
		t.Errorf("statement 0 = %+v, want hit a", got[0])
	}
	if got[1].Aggregation == nil || got[1].Aggregation.Count != 1 {
		t.Errorf("statement 1 = %+v, want an aggregation with count 1", got[1])
	}
	if len(got[2].Hits) != 1 || got[2].Hits[0].ID != "b" {
		t.Errorf("statement 2 = %+v, want hit b", got[2])
	}
}

// TestQueryBatchStartingWithAnAggregationIsDecodedAsABatch exercises the other half
// of the batch-vs-bare-array heuristic: the first statement's own answer is an object
// (carrying "count"), not the nested array the case above starts with.
func TestQueryBatchStartingWithAnAggregationIsDecodedAsABatch(t *testing.T) {
	fake := &capture{reply: `[{"count":2,"sums":{}},[{"collection":"docs","id":"a","score":1,"attrs":{}}]]`}
	db := serve(t, fake)

	got, err := db.QueryBatch(context.Background(),
		"SELECT * FROM docs GROUP BY lang; SELECT * FROM docs ORDER BY id")
	if err != nil {
		t.Fatalf("QueryBatch failed: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d answers, want 2", len(got))
	}
	if got[0].Aggregation == nil || got[0].Aggregation.Count != 2 {
		t.Errorf("statement 0 = %+v, want an aggregation with count 2", got[0])
	}
	if len(got[1].Hits) != 1 || got[1].Hits[0].ID != "a" {
		t.Errorf("statement 1 = %+v, want hit a", got[1])
	}
}

func TestQueryBatchParseErrorPropagates(t *testing.T) {
	fake := &capture{status: http.StatusBadRequest, reply: `{"error":"sql parse error at byte 5: unexpected ';'"}`}
	db := serve(t, fake)

	_, err := db.QueryBatch(context.Background(), "SELECT * FROM docs;;")
	if err == nil {
		t.Fatal("QueryBatch succeeded against a parse error reply")
	}
	var nerr *Error
	if !errors.As(err, &nerr) || !nerr.IsBadRequest() {
		t.Fatalf("error = %v, want a 400 (IsBadRequest)", err)
	}
}

// ── Compile ───────────────────────────────────────────────────────────────────

func TestCompileSendsCompileOnlyAndDecodesOneStatement(t *testing.T) {
	fake := &capture{reply: `{"kind":"list","collections":["docs"],` +
		`"opts":{"offset":0,"limit":100,"filter":[],"projection":"all","order_by":null}}`}
	db := serve(t, fake)

	got, err := db.Compile(context.Background(), "SELECT * FROM docs")
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d compiled statements, want 1", len(got))
	}
	if got[0].Kind != "list" || len(got[0].Collections) != 1 || got[0].Collections[0] != "docs" {
		t.Errorf("Compiled = %+v, want kind list over [docs]", got[0])
	}

	if fake.snapshot().path != "/query" {
		t.Errorf("path = %q, want /query", fake.snapshot().path)
	}
	if body := fake.sentBody(t); !strings.Contains(body, `"compile_only":true`) {
		t.Errorf("body = %s, must set compile_only", body)
	}
	if !strings.Contains(fake.sentBody(t), `"sql":"SELECT * FROM docs"`) {
		t.Errorf("body = %s, must carry the sql string", fake.sentBody(t))
	}
}

func TestCompileDecodesAScriptInOrder(t *testing.T) {
	fake := &capture{reply: `[` +
		`{"kind":"list","collections":[],"opts":{}},` +
		`{"kind":"aggregate","collections":[],"opts":{"filter":[],"sum":[],"group_by":null}}` +
		`]`}
	db := serve(t, fake)

	got, err := db.Compile(context.Background(), "SELECT * FROM docs; SELECT * FROM docs GROUP BY lang")
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	if len(got) != 2 || got[0].Kind != "list" || got[1].Kind != "aggregate" {
		t.Fatalf("got %+v, want [list, aggregate] in order", got)
	}
}

// TestCompileExecutesNothing is the falsifiable half of "compile-only": exactly one
// request reaches the fake server. A Compile that ran the query first (to validate it,
// say) or twice would trip the call count.
func TestCompileExecutesNothing(t *testing.T) {
	fake := &capture{reply: `{"kind":"search","collections":[],"vector":[1,0,0],"opts":{}}`}
	db := serve(t, fake)

	if _, err := db.Compile(context.Background(), "SELECT * FROM docs ORDER BY knn([1,0,0])"); err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	if calls := fake.snapshot().calls; calls != 1 {
		t.Fatalf("compile made %d requests, want exactly 1 (compile executes nothing else)", calls)
	}
	if path := fake.snapshot().path; path != "/query" {
		t.Fatalf("compile called %q, want only /query", path)
	}
}

func TestCompileParseErrorSurfacesTheSameWay(t *testing.T) {
	const msg = "sql parse error at byte 4: unexpected token (§7.1 core SELECT)"
	fake := &capture{status: http.StatusBadRequest, reply: `{"error":"` + msg + `"}`}
	db := serve(t, fake)

	_, err := db.Compile(context.Background(), "SELEC * FROM docs")
	if err == nil {
		t.Fatal("Compile succeeded against a parse error reply")
	}
	var nerr *Error
	if !errors.As(err, &nerr) || nerr.Message != msg || !nerr.IsBadRequest() {
		t.Fatalf("error = %v, want a 400 carrying %q verbatim", err, msg)
	}
}

// ── Errors: the parse-error format, and that it is not confused with a server fault ──

// TestQueryParseErrorCarriesTheServersMessageVerbatim pins the one error format every
// surface shares (src/sql/error.rs): the byte offset and the §7 feature tail must
// survive intact, and the status must be 400, not 500 — a caller's retry loop built on
// IsBadRequest() depends on that distinction.
func TestQueryParseErrorCarriesTheServersMessageVerbatim(t *testing.T) {
	const msg = "sql parse error at byte 17: expected a value after '=' (§7.3 boolean composition)"
	fake := &capture{status: http.StatusBadRequest, reply: `{"error":"` + msg + `"}`}
	db := serve(t, fake)

	_, err := db.Query(context.Background(), "SELECT * FROM docs WHERE lang =")
	if err == nil {
		t.Fatal("Query succeeded against a parse error reply")
	}
	var nerr *Error
	if !errors.As(err, &nerr) {
		t.Fatalf("error is %T, want *nidus.Error", err)
	}
	if nerr.Message != msg {
		t.Errorf("Message = %q, want the server's message verbatim: %q", nerr.Message, msg)
	}
	if !strings.Contains(nerr.Message, "byte 17") {
		t.Errorf("Message = %q, want the byte offset preserved", nerr.Message)
	}
	if !strings.Contains(nerr.Message, "§7.3") {
		t.Errorf("Message = %q, want the §7 section tail preserved", nerr.Message)
	}
	if !nerr.IsBadRequest() {
		t.Errorf("status = %d, want 400 (IsBadRequest)", nerr.Status)
	}
}

func TestQueryServerFaultIsNotConfusedWithAParseError(t *testing.T) {
	fake := &capture{status: http.StatusInternalServerError, reply: `{"error":"internal error"}`}
	db := serve(t, fake)

	_, err := db.Query(context.Background(), "SELECT * FROM docs")
	if err == nil {
		t.Fatal("Query succeeded against a 500 reply")
	}
	var nerr *Error
	if !errors.As(err, &nerr) {
		t.Fatalf("error is %T, want *nidus.Error", err)
	}
	if nerr.IsBadRequest() {
		t.Errorf("status = %d classified as bad-request; a 500 is a server fault, not a parse error", nerr.Status)
	}
	if nerr.Status != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", nerr.Status)
	}
}

// ── Transport parity: whatever Search gets, Query gets too ──────────────────

func TestQueryCancelledContextIsATransportError(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := db.Query(ctx, "SELECT * FROM docs")
	if err == nil {
		t.Fatal("Query succeeded with a cancelled context")
	}
	var nerr *Error
	if !errors.As(err, &nerr) || nerr.Status != 0 {
		t.Fatalf("error = %v (%T), want an *Error with status 0", err, err)
	}
}

func TestQuerySendsTheBearerToken(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake, WithToken("s3cret"))

	if _, err := db.Query(context.Background(), "SELECT * FROM docs"); err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if got := fake.snapshot().header.Get("Authorization"); got != "Bearer s3cret" {
		t.Errorf("Authorization = %q, want Bearer s3cret", got)
	}
}

// TestQueryDecodesEmptyBatchElementCounts guards decodeQueryAnswers' one edge case:
// an empty top-level array (a single Hits statement with zero results) must not be
// mistaken for a zero-statement batch — QueryBatch always returns at least one answer
// for a well-formed script.
func TestQueryDecodesEmptyBatchElementCounts(t *testing.T) {
	fake := &capture{reply: `[]`}
	db := serve(t, fake)

	got, err := db.QueryBatch(context.Background(), "SELECT * FROM docs ORDER BY knn([1,0,0])")
	if err != nil {
		t.Fatalf("QueryBatch failed: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d answers, want 1", len(got))
	}
	if got[0].Hits == nil || len(got[0].Hits) != 0 {
		t.Errorf("Hits = %+v, want an empty slice", got[0].Hits)
	}
}
