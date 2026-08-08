// Tests for the predicate and filter wire shapes.
//
// These assert on marshalled bytes throughout. A filter that encodes to the wrong
// shape does not fail loudly — the server answers 400 with a serde message about a
// tuple variant, which tells a caller nothing about which builder was wrong — so the
// shape is worth pinning exactly. Three shapes in particular have no symmetry to fall
// back on: the text predicates' second element is a bare string where every other
// predicate's is a tagged Value, Fuzzy is a 3-tuple where every other leaf is a pair,
// and a Filter is a bare array rather than an object wrapping one.
package nidus

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"
)

// TestPredicateWireShapes pins every variant's encoding. The key is always the first
// tuple element and the operand the second, matching serde's externally-tagged tuple
// form for the crate's Predicate enum — Fuzzy adds a third, the combinators none.
func TestPredicateWireShapes(t *testing.T) {
	cases := []struct {
		name string
		pred Predicate
		want string
	}{
		{"Eq string", Eq("lang", "rust"), `{"Eq":["lang",{"Str":"rust"}]}`},
		{"Eq int", Eq("year", 2024), `{"Eq":["year",{"Int":2024}]}`},
		{"Eq bool", Eq("draft", false), `{"Eq":["draft",{"Bool":false}]}`},
		{"Eq null", Eq("summary", nil), `{"Eq":["summary","Null"]}`},
		{"Eq list", Eq("tags", []string{"a", "b"}), `{"Eq":["tags",{"List":["a","b"]}]}`},
		{"Eq float", Eq("score", 1.5), `{"Eq":["score",{"Float":1.5}]}`},
		// A range over instants: the operand is epoch milliseconds, so the comparison the
		// server does is integer arithmetic on an absolute point in time.
		{
			"Ge datetime",
			Ge("seen", time.UnixMilli(1700000000000).UTC()),
			`{"Ge":["seen",{"DateTime":1700000000000}]}`,
		},
		{"Ne", Ne("lang", "go"), `{"Ne":["lang",{"Str":"go"}]}`},
		{"Lt", Lt("year", 2024), `{"Lt":["year",{"Int":2024}]}`},
		{"Le", Le("year", 2024), `{"Le":["year",{"Int":2024}]}`},
		{"Gt", Gt("year", 2024), `{"Gt":["year",{"Int":2024}]}`},
		{"Ge", Ge("year", 2024), `{"Ge":["year",{"Int":2024}]}`},
		{"In", In("lang", "rust", "go"), `{"In":["lang",[{"Str":"rust"},{"Str":"go"}]]}`},
		{"In mixed types", In("k", "a", 1), `{"In":["k",[{"Str":"a"},{"Int":1}]]}`},
		{"NotIn", NotIn("lang", "js"), `{"NotIn":["lang",[{"Str":"js"}]]}`},
		// A nil set must still be `[]`: the server's is a Vec, and `null` would be a
		// deserialization error rather than "an empty set, matching nothing".
		{"In with no values", In("lang"), `{"In":["lang",[]]}`},
		{"NotIn with no values", NotIn("lang"), `{"NotIn":["lang",[]]}`},
		{"Glob", Glob("path", "src/*.rs"), `{"Glob":["path","src/*.rs"]}`},
		// IGlob shares Glob's bare-string second element, and only the tag differs.
		{"IGlob", IGlob("path", "Src/*.RS"), `{"IGlob":["path","Src/*.RS"]}`},
		// A key with JSON-significant characters must be escaped as a string, not
		// interpolated — the key is caller data.
		{"quoted key", Eq(`a"b`, "v"), `{"Eq":["a\"b",{"Str":"v"}]}`},
		// A Value passed straight through, rather than a plain Go value to normalize.
		{"pre-built Value", Eq("n", Int(9007199254740993)), `{"Eq":["n",{"Int":9007199254740993}]}`},
		// Containment: the two unary forms share the leaf tuple shape, ContainsAny
		// takes an array exactly as In does.
		{"Contains", Contains("tags", "rust"), `{"Contains":["tags",{"Str":"rust"}]}`},
		{"NotContains", NotContains("tags", "wip"), `{"NotContains":["tags",{"Str":"wip"}]}`},
		{
			"ContainsAny",
			ContainsAny("tags", "rust", "go"),
			`{"ContainsAny":["tags",[{"Str":"rust"},{"Str":"go"}]]}`,
		},
		{"ContainsAny with no values", ContainsAny("tags"), `{"ContainsAny":["tags",[]]}`},
		// The token predicates take the query text as a bare string, exactly as Glob
		// takes its pattern — the operand is text to analyze, never a tagged Value.
		{
			"ContainsAllTokens",
			ContainsAllTokens("body", "async runtime"),
			`{"ContainsAllTokens":["body","async runtime"]}`,
		},
		{
			"ContainsAnyToken",
			ContainsAnyToken("body", "async runtime"),
			`{"ContainsAnyToken":["body","async runtime"]}`,
		},
		{
			"ContainsTokenSequence",
			ContainsTokenSequence("body", "async runtime"),
			`{"ContainsTokenSequence":["body","async runtime"]}`,
		},
		{"Regex", Regex("path", "^src/.*"), `{"Regex":["path","^src/.*"]}`},
		// Fuzzy is the one leaf predicate with three tuple elements.
		{"Fuzzy", Fuzzy("title", "serach", 2), `{"Fuzzy":["title","serach",2]}`},
		{"Fuzzy with no edits", Fuzzy("title", "search", 0), `{"Fuzzy":["title","search",0]}`},
		// The combinators break the key/value tuple shape entirely.
		{
			"Any",
			Any(Eq("project", "nidus"), Eq("project", "beads")),
			`{"Any":[{"Eq":["project",{"Str":"nidus"}]},{"Eq":["project",{"Str":"beads"}]}]}`,
		},
		{"All", All(Eq("a", 1)), `{"All":[{"Eq":["a",{"Int":1}]}]}`},
		{"Not wraps a single predicate", Not(Eq("a", 1)), `{"Not":{"Eq":["a",{"Int":1}]}}`},
		// Empty groups must be `[]`, not `null`: the server's field is a Vec, and the
		// identities (All=true, Any=false) only hold if it deserializes at all.
		{"All with no predicates", All(), `{"All":[]}`},
		{"Any with no predicates", Any(), `{"Any":[]}`},
		// Nesting is the whole point: a group holding a group.
		{
			"nested groups",
			Not(Any(Contains("tags", "wip"))),
			`{"Not":{"Any":[{"Contains":["tags",{"Str":"wip"}]}]}}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.pred.Err(); err != nil {
				t.Fatalf("Err() = %v, want nil", err)
			}
			got, err := json.Marshal(tc.pred)
			if err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}
			if string(got) != tc.want {
				t.Errorf("Marshal = %s, want %s", got, tc.want)
			}
		})
	}
}

// TestGlobSecondElementIsABareString states the asymmetry on its own, because it is
// the wire format and not a choice — a Glob pattern wrapped in {"Str":…} is rejected
// by the server, and it is the mistake a reader of the other predicates would make.
func TestGlobSecondElementIsABareString(t *testing.T) {
	encoded, err := json.Marshal(Glob("path", "src/*"))
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if want := `{"Glob":["path","src/*"]}`; string(encoded) != want {
		t.Fatalf("Marshal = %s, want %s", encoded, want)
	}

	// Decode it back generically and check the second element really is a JSON string
	// and not an object, so this cannot pass by coincidence of formatting.
	var tuple map[string][]json.RawMessage
	if err := json.Unmarshal(encoded, &tuple); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	second := string(tuple["Glob"][1])
	if !strings.HasPrefix(second, `"`) {
		t.Errorf("Glob's pattern encoded as %s, want a bare JSON string", second)
	}

	// Every other predicate's second element IS a tagged object — the contrast is the
	// point of this test.
	other, err := json.Marshal(Eq("path", "src/*"))
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if err := json.Unmarshal(other, &tuple); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if second := string(tuple["Eq"][1]); !strings.HasPrefix(second, "{") {
		t.Errorf("Eq's operand encoded as %s, want a tagged Value object", second)
	}
}

// TestFilterMarshalsAsBareArray — the crate's Filter is a newtype over
// Vec<Predicate>, so it serializes as a plain array. An object wrapping one would be
// rejected.
func TestFilterMarshalsAsBareArray(t *testing.T) {
	f := And(Eq("lang", "rust"), Ge("year", 2020), Glob("path", "src/*"))
	got, err := json.Marshal(f)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	want := `[{"Eq":["lang",{"Str":"rust"}]},{"Ge":["year",{"Int":2020}]},{"Glob":["path","src/*"]}]`
	if string(got) != want {
		t.Errorf("Marshal = %s, want %s", got, want)
	}
	// Predicate order is preserved. It does not change the result (they AND), but a
	// reordering would mean the encoder is round-tripping through a map somewhere.
	if got[1] != '{' || !strings.Contains(string(got), `"Eq"`) {
		t.Errorf("unexpected array shape: %s", got)
	}
}

// TestEmptyFilterMarshalsAsEmptyArray — both a nil Filter and a zero-length one. The
// nil case is the one that matters: `null` is a deserialization error on the server,
// not an empty filter, so a caller who left the field alone would get a 400.
func TestEmptyFilterMarshalsAsEmptyArray(t *testing.T) {
	var nilFilter Filter
	got, err := json.Marshal(nilFilter)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(got) != "[]" {
		t.Errorf("nil Filter marshalled as %s, want []", got)
	}

	got, err = json.Marshal(Filter{})
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(got) != "[]" {
		t.Errorf("empty Filter marshalled as %s, want []", got)
	}

	// And() with no arguments is the same thing, since callers reach for it first.
	got, err = json.Marshal(And())
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(got) != "[]" {
		t.Errorf("And() marshalled as %s, want []", got)
	}
}

// TestAndCollectsPredicates — And is sugar over a slice literal, so it must not
// reorder, drop, or wrap.
func TestAndCollectsPredicates(t *testing.T) {
	f := And(Eq("a", 1), Eq("b", 2))
	if len(f) != 2 {
		t.Fatalf("And produced %d predicates, want 2", len(f))
	}
	if f[0].key != "a" || f[1].key != "b" {
		t.Errorf("And reordered: %q then %q", f[0].key, f[1].key)
	}
	// A Filter is a slice, so a literal is equivalent — the sugar must not diverge.
	direct, _ := json.Marshal(Filter{Eq("a", 1), Eq("b", 2)})
	viaAnd, _ := json.Marshal(f)
	if string(direct) != string(viaAnd) {
		t.Errorf("And() = %s but the slice literal = %s", viaAnd, direct)
	}
}

// TestPredicateCarriesANormalizationError — the builders take `any` so that
// nidus.Eq("year", 2024) reads naturally, which means a bad value has no error to
// return at the call site. The failure is carried on the Predicate and surfaced from
// MarshalJSON rather than panicking or silently sending a wrong-but-valid body.
func TestPredicateCarriesANormalizationError(t *testing.T) {
	cases := []struct {
		name string
		pred Predicate
		want string // a substring the error must contain
	}{
		{"Eq slice", Eq("score", []int{1}), `Eq("score")`},
		{"Ne slice", Ne("score", []int{1}), `Ne("score")`},
		{"Lt slice", Lt("score", []int{1}), `Lt("score")`},
		{"Le slice", Le("score", []int{1}), `Le("score")`},
		{"Gt slice", Gt("score", []int{1}), `Gt("score")`},
		{"Ge slice", Ge("score", []int{1}), `Ge("score")`},
		{"In slice", In("score", "ok", []int{1}), `In("score") value 1`},
		{"NotIn slice", NotIn("score", []int{1}), `NotIn("score") value 0`},
		{"Eq unsupported type", Eq("k", struct{ A int }{1}), `Eq("k")`},
		// A NaN is the other shape that cannot travel: it is a float64, so the type
		// check passes, and JSON has no spelling for it.
		{"Eq NaN", Eq("score", math.NaN()), `Eq("score")`},
		{"Contains slice", Contains("tags", []int{1}), `Contains("tags")`},
		{"NotContains slice", NotContains("tags", []int{1}), `NotContains("tags")`},
		{"ContainsAny slice", ContainsAny("tags", "ok", []int{1}), `ContainsAny("tags") value 1`},
		// A broken leaf must not be able to hide inside a group: the combinators
		// propagate it, or the request ships a body missing a condition entirely.
		{"error inside Any", Any(Eq("a", 1), Eq("score", []int{1})), `Eq("score")`},
		{"error inside All", All(Eq("score", []int{1})), `Eq("score")`},
		{"error inside Not", Not(Eq("score", []int{1})), `Eq("score")`},
		{"error nested two deep", Not(Any(All(Eq("score", []int{1})))), `Eq("score")`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.pred.Err()
			if err == nil {
				t.Fatal("Err() = nil, want the normalization failure")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("Err() = %q, want it to name %s", err, tc.want)
			}
			// The same error must come back from the encoder, because that is the path a
			// caller who never checks Err() takes.
			if _, err := json.Marshal(tc.pred); err == nil {
				t.Error("Marshal succeeded; a broken predicate must not encode")
			}
		})
	}
}

// TestFilterErrReportsTheFirstFailure — checking a whole filter at once, for callers
// who want the error before they make the request.
func TestFilterErrReportsTheFirstFailure(t *testing.T) {
	if err := And(Eq("a", 1), Ge("b", "x")).Err(); err != nil {
		t.Errorf("a valid filter reported %v", err)
	}
	var nilFilter Filter
	if err := nilFilter.Err(); err != nil {
		t.Errorf("nil Filter reported %v", err)
	}

	f := And(Eq("ok", 1), Eq("first", []int{1}), Eq("second", []int{2}))
	err := f.Err()
	if err == nil {
		t.Fatal("Err() = nil, want the first failure")
	}
	if !strings.Contains(err.Error(), "first") {
		t.Errorf("Err() = %q, want the first failing predicate", err)
	}
	// A filter holding a broken predicate must not encode at all — a partial body
	// would be worse than an error, since the server would happily run the wrong query.
	if _, err := json.Marshal(f); err == nil {
		t.Error("Marshal succeeded on a filter with a broken predicate")
	}
}

// TestZeroPredicateIsAnEncodeError — Predicate{} names no operation, so it cannot be
// encoded. The message points at the builders, since a zero predicate almost always
// means a struct literal was used where a builder was meant.
func TestZeroPredicateIsAnEncodeError(t *testing.T) {
	_, err := json.Marshal(Predicate{})
	if err == nil {
		t.Fatal("Marshal(Predicate{}) succeeded, want an error")
	}
	if !strings.Contains(err.Error(), "Eq") {
		t.Errorf("error = %q, want it to point at the builders", err)
	}
	if _, err := json.Marshal(Filter{Predicate{}}); err == nil {
		t.Error("a Filter holding a zero Predicate encoded successfully")
	}
}

// TestGlobNeedsNoNormalization — Glob takes a typed string, so it can never carry a
// normalization error, and an empty pattern is a legitimate (if useless) request
// rather than something to reject client-side.
func TestGlobNeedsNoNormalization(t *testing.T) {
	p := Glob("path", "")
	if err := p.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
	got, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if want := `{"Glob":["path",""]}`; string(got) != want {
		t.Errorf("Marshal = %s, want %s", got, want)
	}
}

// TestFuzzyIsAThreeElementTuple states Fuzzy's arity on its own, the way
// TestGlobSecondElementIsABareString states Glob's. Every other leaf predicate is a
// [key, operand] pair, and a Fuzzy encoded as one is a serde error about a tuple
// variant that names neither the builder nor the field.
func TestFuzzyIsAThreeElementTuple(t *testing.T) {
	encoded, err := json.Marshal(Fuzzy("title", "serach", 2))
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	var tuple map[string][]json.RawMessage
	if err := json.Unmarshal(encoded, &tuple); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	got := tuple["Fuzzy"]
	if len(got) != 3 {
		t.Fatalf("Fuzzy encoded %d tuple elements, want 3: %s", len(got), encoded)
	}
	// The edit budget is a bare number, not a tagged Value and not a string: it is a
	// usize on the server, so anything else fails to deserialize.
	if third := string(got[2]); third != "2" {
		t.Errorf("Fuzzy's third element = %s, want the bare number 2", third)
	}
	// The contrast: a sibling text predicate is a 2-tuple.
	if err := json.Unmarshal([]byte(`{"Regex":["p","x"]}`), &tuple); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if len(tuple["Regex"]) != 2 {
		t.Errorf("Regex encoded %d tuple elements, want 2", len(tuple["Regex"]))
	}
}

// TestFuzzyRejectsAnUnusableEditBudget — the edit count is a usize on the wire, so a
// negative one cannot travel at all, and the server refuses anything above 8 rather
// than clamping it. Both are carried on the predicate like any other normalization
// failure, so they surface from the call that used the filter.
func TestFuzzyRejectsAnUnusableEditBudget(t *testing.T) {
	for _, edits := range []int{-1, 9, 1 << 40} {
		p := Fuzzy("title", "serach", edits)
		err := p.Err()
		if err == nil {
			t.Fatalf("Fuzzy(…, %d).Err() = nil, want the range failure", edits)
		}
		if !strings.Contains(err.Error(), `Fuzzy("title")`) {
			t.Errorf("error = %q, want it to name the builder and key", err)
		}
		if _, err := json.Marshal(p); err == nil {
			t.Errorf("Fuzzy(…, %d) encoded successfully; it must not", edits)
		}
	}

	// The whole accepted range still builds, boundary included.
	for _, edits := range []int{0, 1, 8} {
		if err := Fuzzy("title", "serach", edits).Err(); err != nil {
			t.Errorf("Fuzzy(…, %d).Err() = %v, want nil", edits, err)
		}
	}

	// And a broken Fuzzy cannot hide inside a group, like any other broken leaf.
	if err := Not(Any(Fuzzy("title", "x", -1))).Err(); err == nil {
		t.Error("a nested Fuzzy error did not propagate out of the group")
	}
}

// TestTextPredicatesNeedNoNormalization — like [Glob], the token predicates and Regex
// take typed strings, so they can never carry a normalization error and an empty
// operand is a legitimate request rather than something to reject client-side.
func TestTextPredicatesNeedNoNormalization(t *testing.T) {
	cases := []struct {
		pred Predicate
		want string
	}{
		{ContainsAllTokens("body", ""), `{"ContainsAllTokens":["body",""]}`},
		{ContainsAnyToken("body", ""), `{"ContainsAnyToken":["body",""]}`},
		{ContainsTokenSequence("body", ""), `{"ContainsTokenSequence":["body",""]}`},
		{Regex("path", ""), `{"Regex":["path",""]}`},
	}
	for _, tc := range cases {
		if err := tc.pred.Err(); err != nil {
			t.Errorf("Err() = %v, want nil", err)
			continue
		}
		got, err := json.Marshal(tc.pred)
		if err != nil {
			t.Fatalf("Marshal failed: %v", err)
		}
		if string(got) != tc.want {
			t.Errorf("Marshal = %s, want %s", got, tc.want)
		}
	}
}
