// Tests for the Value codec — the wire shapes, and the two silent-failure modes
// value.go exists to prevent.
//
// Most of this file is about precision and forward compatibility rather than the
// happy path, because those are the bugs that do not announce themselves. A Value
// that loses the last digit of a snowflake id, or that drops a variant a newer
// server sent, produces valid JSON and a plausible record; nothing throws. So the
// assertions here are on *bytes* wherever the byte matters, not on a decoded
// float64 that has already lost the argument.
//
// These tests live in package nidus (not nidus_test) deliberately: Value's fields
// are unexported precisely so a caller cannot build an invalid one, which means the
// only place a malformed Value can be constructed — to check the encoder rejects it
// — is inside the package.
package nidus

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
)

// TestValueRoundTripEveryKind pins the externally-tagged wire form of every variant
// in both directions. Note the Null row: serde encodes a fieldless variant as a bare
// string, so it is "Null" and not {"Null":null} — the one shape that does not look
// like the others.
func TestValueRoundTripEveryKind(t *testing.T) {
	cases := []struct {
		name string
		val  Value
		json string
	}{
		{"Null", Null(), `"Null"`},
		{"Str", Str("rust"), `{"Str":"rust"}`},
		{"StrEmpty", Str(""), `{"Str":""}`},
		{"Int", Int(2024), `{"Int":2024}`},
		{"IntNegative", Int(-7), `{"Int":-7}`},
		{"IntMax", Int(math.MaxInt64), `{"Int":9223372036854775807}`},
		{"IntMin", Int(math.MinInt64), `{"Int":-9223372036854775808}`},
		{"BoolTrue", Bool(true), `{"Bool":true}`},
		{"BoolFalse", Bool(false), `{"Bool":false}`},
		{"List", List("a", "b"), `{"List":["a","b"]}`},
		{"ListEmpty", List(), `{"List":[]}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded, err := json.Marshal(tc.val)
			if err != nil {
				t.Fatalf("Marshal(%v) failed: %v", tc.val, err)
			}
			if string(encoded) != tc.json {
				t.Errorf("Marshal(%v) = %s, want %s", tc.val, encoded, tc.json)
			}

			var back Value
			if err := json.Unmarshal([]byte(tc.json), &back); err != nil {
				t.Fatalf("Unmarshal(%s) failed: %v", tc.json, err)
			}
			if back.Kind() != tc.val.Kind() {
				t.Errorf("Unmarshal(%s) kind = %v, want %v", tc.json, back.Kind(), tc.val.Kind())
			}
			// Re-encoding must reproduce the input byte for byte, which is the property
			// a caller who reads a record and writes it back depends on.
			again, err := json.Marshal(back)
			if err != nil {
				t.Fatalf("re-Marshal failed: %v", err)
			}
			if string(again) != tc.json {
				t.Errorf("round trip of %s produced %s", tc.json, again)
			}
		})
	}
}

// TestValueNullDecodesFromBareString states the Null shape on its own, since it is
// the single asymmetric encoding and a reader of this file should not have to infer
// it from a table row.
func TestValueNullDecodesFromBareString(t *testing.T) {
	var v Value
	if err := json.Unmarshal([]byte(`"Null"`), &v); err != nil {
		t.Fatalf("Unmarshal(\"Null\") failed: %v", err)
	}
	if v.Kind() != KindNull {
		t.Fatalf("kind = %v, want KindNull", v.Kind())
	}
	if v.Any() != nil {
		t.Errorf("Any() = %v, want nil", v.Any())
	}
	// {"Null":null} is NOT how serde writes it, so it must not be silently accepted
	// as a Null — it is an unrecognised shape, preserved rather than reinterpreted.
	var obj Value
	if err := json.Unmarshal([]byte(`{"Null":null}`), &obj); err != nil {
		t.Fatalf("Unmarshal({\"Null\":null}) failed: %v", err)
	}
	if obj.Kind() != KindUnknown {
		t.Errorf("{\"Null\":null} decoded as %v; the wire form of Null is the bare string", obj.Kind())
	}
}

// TestValueIntKeepsFullInt64Precision is the float64 trap.
//
// 9007199254740993 is 2^53+1, the smallest integer float64 cannot represent. Decoding
// a JSON number to float64 — which is what encoding/json does by default — turns it
// into 9007199254740992, and the record comes back looking fine while holding the
// wrong id. The test asserts the corruption is real (so the guard is not cargo cult)
// and then that the decoder avoids it.
func TestValueIntKeepsFullInt64Precision(t *testing.T) {
	const exact int64 = 9007199254740993
	if int64(float64(exact)) == exact {
		t.Fatal("premise broken: this platform's float64 can hold 2^53+1")
	}

	var v Value
	if err := json.Unmarshal([]byte(`{"Int":9007199254740993}`), &v); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	got, ok := v.Int()
	if !ok {
		t.Fatalf("kind = %v, want KindInt", v.Kind())
	}
	if got != exact {
		t.Errorf("decoded %d, want %d (lost to float64 rounding)", got, exact)
	}

	encoded, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(encoded) != `{"Int":9007199254740993}` {
		t.Errorf("re-encoded as %s, want {\"Int\":9007199254740993}", encoded)
	}

	// The extremes of the range matter too: i64::MAX is well past float64's exact
	// integers, so a naive decoder mangles it in both directions.
	for _, lit := range []string{`{"Int":9223372036854775807}`, `{"Int":-9223372036854775808}`} {
		var edge Value
		if err := json.Unmarshal([]byte(lit), &edge); err != nil {
			t.Fatalf("Unmarshal(%s) failed: %v", lit, err)
		}
		again, err := json.Marshal(edge)
		if err != nil {
			t.Fatalf("Marshal failed: %v", err)
		}
		if string(again) != lit {
			t.Errorf("round trip of %s produced %s", lit, again)
		}
	}
}

// TestValueIntegerValuedFloatDecodesAsInt covers a number that arrived through some
// encoder that writes integers with a trailing .0. JSON cannot say "I meant an
// integer", so an all-zero fractional part is accepted — textually, so a big value
// written that way still keeps its last digit.
func TestValueIntegerValuedFloatDecodesAsInt(t *testing.T) {
	cases := []struct {
		json string
		want int64
	}{
		{`{"Int":2024.0}`, 2024},
		{`{"Int":2024.000}`, 2024},
		{`{"Int":-7.0}`, -7},
		{`{"Int":0.0}`, 0},
		// The precision guard has to hold on this path too, not just the plain-integer
		// one: going through float64 here would round the final 3 to a 2.
		{`{"Int":9007199254740993.0}`, 9007199254740993},
	}
	for _, tc := range cases {
		var v Value
		if err := json.Unmarshal([]byte(tc.json), &v); err != nil {
			t.Fatalf("Unmarshal(%s) failed: %v", tc.json, err)
		}
		got, ok := v.Int()
		if !ok {
			t.Fatalf("%s decoded as %v, want KindInt", tc.json, v.Kind())
		}
		if got != tc.want {
			t.Errorf("%s decoded as %d, want %d", tc.json, got, tc.want)
		}
	}
}

// TestValueFractionalIntRejected — 2024.5 is an error, never a truncation to 2024.
// There is no float attribute type, so dropping the fraction would be silent data
// loss on the way in.
func TestValueFractionalIntRejected(t *testing.T) {
	for _, lit := range []string{`{"Int":2024.5}`, `{"Int":-0.5}`, `{"Int":1e-3}`} {
		var v Value
		err := json.Unmarshal([]byte(lit), &v)
		if err == nil {
			got, _ := v.Int()
			t.Errorf("Unmarshal(%s) succeeded as %d; a fractional Int must be an error", lit, got)
			continue
		}
		if !strings.Contains(err.Error(), "not an integer") {
			t.Errorf("Unmarshal(%s) error = %q, want it to say the value is not an integer", lit, err)
		}
	}
}

// TestValueIntOverflowRejected — a number past i64 has nowhere to go, and wrapping it
// would be worse than failing.
func TestValueIntOverflowRejected(t *testing.T) {
	for _, lit := range []string{`{"Int":1e300}`, `{"Int":-1e300}`} {
		var v Value
		if err := json.Unmarshal([]byte(lit), &v); err == nil {
			t.Errorf("Unmarshal(%s) succeeded; it overflows int64", lit)
		}
	}
}

// TestValueOfRejectsFloats — floats belong in the vector. Note that an
// integer-valued float64 is rejected too: on the way *in* the caller's type says
// float, and honouring it would establish an attribute type the store does not have.
// (Contrast TestValueIntegerValuedFloatDecodesAsInt, where a JSON number's type is
// genuinely ambiguous and there is no caller intent to respect.)
func TestValueOfRejectsFloats(t *testing.T) {
	for _, x := range []any{float32(1.5), float64(2.5), float32(2024), float64(2024)} {
		v, err := ValueOf(x)
		if err == nil {
			t.Errorf("ValueOf(%v) succeeded as %v; attributes have no float type", x, v)
			continue
		}
		if !strings.Contains(err.Error(), "float") {
			t.Errorf("ValueOf(%v) error = %q, want it to mention floats", x, err)
		}
	}
}

// TestValueOfNormalizes covers every accepted input type. The uint8 row is the one
// worth reading twice: a []byte would arrive as []uint8 and is NOT a list of
// strings, so it must fall through to the error case rather than becoming a List.
func TestValueOfNormalizes(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want Value
	}{
		{"Value", Str("passthrough"), Str("passthrough")},
		{"nil", nil, Null()},
		{"string", "rust", Str("rust")},
		{"bool", true, Bool(true)},
		{"int", 2024, Int(2024)},
		{"int8", int8(-8), Int(-8)},
		{"int16", int16(-16), Int(-16)},
		{"int32", int32(-32), Int(-32)},
		{"int64", int64(math.MaxInt64), Int(math.MaxInt64)},
		{"uint", uint(1), Int(1)},
		{"uint8", uint8(8), Int(8)},
		{"uint16", uint16(16), Int(16)},
		{"uint32", uint32(32), Int(32)},
		{"uint64", uint64(math.MaxInt64), Int(math.MaxInt64)},
		{"[]string", []string{"a", "b"}, List("a", "b")},
		{"[]any of strings", []any{"a", "b"}, List("a", "b")},
		{"[]any empty", []any{}, List()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ValueOf(tc.in)
			if err != nil {
				t.Fatalf("ValueOf(%#v) failed: %v", tc.in, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("ValueOf(%#v) = %v (%v), want %v (%v)",
					tc.in, got, got.Kind(), tc.want, tc.want.Kind())
			}
		})
	}
}

// TestValueOfRejectsUnsupported — anything the store has no type for, including a
// mixed []any, a nested map, and a u64 past i64.
func TestValueOfRejectsUnsupported(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want string // a substring the error must contain
	}{
		{"mixed list", []any{"a", 1}, "element 1"},
		{"nested list", []any{[]string{"a"}}, "element 0"},
		{"map", map[string]any{"k": "v"}, "cannot use"},
		{"[]int", []int{1, 2}, "cannot use"},
		{"[]byte", []byte("bytes"), "cannot use"},
		{"struct", struct{ A int }{1}, "cannot use"},
		// The server's Int is an i64, so a u64 past its max has nowhere to go and must
		// not wrap into a negative.
		{"uint64 overflow", uint64(math.MaxUint64), "overflows"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v, err := ValueOf(tc.in)
			if err == nil {
				t.Fatalf("ValueOf(%#v) succeeded as %v, want an error", tc.in, v)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to contain %q", err, tc.want)
			}
		})
	}
}

// TestMustValueOfPanicsOnUnsupported — the literal-friendly wrapper is allowed to
// panic, but only on input ValueOf rejects.
func TestMustValueOfPanicsOnUnsupported(t *testing.T) {
	if got := MustValueOf(2024); !reflect.DeepEqual(got, Int(2024)) {
		t.Errorf("MustValueOf(2024) = %v, want Int(2024)", got)
	}
	defer func() {
		if recover() == nil {
			t.Error("MustValueOf(1.5) did not panic")
		}
	}()
	MustValueOf(1.5)
}

// TestValueUnknownKindRoundTripsByteIdentical is the forward-compatibility contract:
// a variant a newer server added must survive decode → encode through this SDK
// untouched, so an older client that reads a record and writes it back does not
// silently drop the field.
func TestValueUnknownKindRoundTripsByteIdentical(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		{"unknown tag with a string payload", `{"Blob":"AAECAw=="}`},
		{"unknown tag with an object payload", `{"Geo":{"lat":1,"lon":2}}`},
		{"unknown tag with a float payload", `{"F64":1.5}`},
		{"unknown fieldless variant", `"Timestamp"`},
		{"several keys", `{"Str":"a","Int":1}`},
		{"no keys", `{}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var v Value
			if err := json.Unmarshal([]byte(tc.json), &v); err != nil {
				t.Fatalf("Unmarshal(%s) failed: %v; an unknown tag is version skew, not an error",
					tc.json, err)
			}
			if v.Kind() != KindUnknown {
				t.Fatalf("kind = %v, want KindUnknown", v.Kind())
			}
			encoded, err := json.Marshal(v)
			if err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}
			if string(encoded) != tc.json {
				t.Errorf("re-marshalled as %s, want the original %s byte for byte", encoded, tc.json)
			}
			// Any() hands the raw JSON back so the data stays inspectable rather than
			// vanishing behind an opaque kind.
			if raw, ok := v.Any().(json.RawMessage); !ok || string(raw) != tc.json {
				t.Errorf("Any() = %#v, want the raw JSON %s", v.Any(), tc.json)
			}
		})
	}
}

// TestValueUnknownRawIsCopied guards the buffer-reuse hazard: encoding/json may hand
// UnmarshalJSON a slice into a buffer it reuses for the next value, so a Value that
// kept the slice instead of copying would end up pointing at whatever was decoded
// after it. Decoding two unknown values from one document is what exposes it.
func TestValueUnknownRawIsCopied(t *testing.T) {
	var vals []Value
	if err := json.Unmarshal([]byte(`[{"Blob":"one"},{"Geo":"two"}]`), &vals); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if len(vals) != 2 {
		t.Fatalf("decoded %d values, want 2", len(vals))
	}
	if got := string(vals[0].raw); got != `{"Blob":"one"}` {
		t.Errorf("first value's preserved JSON = %s, want {\"Blob\":\"one\"}", got)
	}
	if got := string(vals[1].raw); got != `{"Geo":"two"}` {
		t.Errorf("second value's preserved JSON = %s, want {\"Geo\":\"two\"}", got)
	}
}

// TestValueUnknownWithoutRawIsAnEncodeError — a KindUnknown with nothing preserved
// cannot be built through the public API; if one ever appears, the encoder must say
// so rather than emit `null` and corrupt a record.
func TestValueUnknownWithoutRawIsAnEncodeError(t *testing.T) {
	if _, err := json.Marshal(Value{kind: KindUnknown}); err == nil {
		t.Error("marshalling a KindUnknown with no preserved JSON succeeded, want an error")
	}
}

// TestValueUnmarshalRejectsMalformedKnownTags — version skew is tolerated, a
// contract violation is not. A known tag carrying the wrong payload shape means the
// server and this SDK disagree about a type they both claim to know.
func TestValueUnmarshalRejectsMalformedKnownTags(t *testing.T) {
	cases := []string{
		`{"Str":5}`,
		`{"Int":"2024"}`,
		`{"Int":true}`,
		`{"Bool":"yes"}`,
		`{"List":[1,2]}`,
		`{"List":"a"}`,
		``,
		`   `,
		`[1,2]`,
		`5`,
		`true`,
	}
	for _, tc := range cases {
		var v Value
		if err := json.Unmarshal([]byte(tc), &v); err == nil {
			t.Errorf("Unmarshal(%q) succeeded as %v, want an error", tc, v)
		}
	}
}

// TestValueUnmarshalOverwritesTheReceiver — decoding into a Value that already holds
// something must replace it wholesale, leaving no payload from the previous variant
// behind for an accessor to read.
func TestValueUnmarshalOverwritesTheReceiver(t *testing.T) {
	v := List("stale", "items")
	if err := json.Unmarshal([]byte(`{"Int":1}`), &v); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if items, ok := v.List(); ok {
		t.Errorf("List() still reports %v after decoding an Int", items)
	}
	if got, ok := v.Int(); !ok || got != 1 {
		t.Errorf("Int() = (%d, %v), want (1, true)", got, ok)
	}
}

// TestValueAccessorsAreCommaOK — reading an attribute as the wrong type yields a
// testable false, not a panic and not a plausible-looking zero.
func TestValueAccessorsAreCommaOK(t *testing.T) {
	str, num, boolean, list, null := Str("s"), Int(1), Bool(true), List("a"), Null()

	if got, ok := str.Str(); !ok || got != "s" {
		t.Errorf("Str.Str() = (%q, %v), want (\"s\", true)", got, ok)
	}
	if got, ok := num.Str(); ok || got != "" {
		t.Errorf("Int.Str() = (%q, %v), want (\"\", false)", got, ok)
	}
	if got, ok := num.Int(); !ok || got != 1 {
		t.Errorf("Int.Int() = (%d, %v), want (1, true)", got, ok)
	}
	if got, ok := str.Int(); ok || got != 0 {
		t.Errorf("Str.Int() = (%d, %v), want (0, false)", got, ok)
	}
	if got, ok := boolean.Bool(); !ok || !got {
		t.Errorf("Bool.Bool() = (%v, %v), want (true, true)", got, ok)
	}
	if got, ok := num.Bool(); ok || got {
		t.Errorf("Int.Bool() = (%v, %v), want (false, false)", got, ok)
	}
	if got, ok := list.List(); !ok || !reflect.DeepEqual(got, []string{"a"}) {
		t.Errorf("List.List() = (%v, %v), want ([a], true)", got, ok)
	}
	if got, ok := str.List(); ok || got != nil {
		t.Errorf("Str.List() = (%v, %v), want (nil, false)", got, ok)
	}
	// Null answers false to all of them: it is set-but-empty, not a string.
	if _, ok := null.Str(); ok {
		t.Error("Null.Str() reported ok")
	}
	if null.Any() != nil {
		t.Errorf("Null.Any() = %v, want nil", null.Any())
	}
}

// TestValueListIsImmutableFromOutside — List copies in, List() copies out, so a
// caller cannot mutate a Value they have already handed to the SDK (nor one the SDK
// handed them).
func TestValueListIsImmutableFromOutside(t *testing.T) {
	items := []string{"a", "b"}
	v := List(items...)
	items[0] = "mutated"
	got, _ := v.List()
	if got[0] != "a" {
		t.Errorf("writing the caller's slice changed the Value: %v", got)
	}

	got[1] = "mutated"
	again, _ := v.List()
	if again[1] != "b" {
		t.Errorf("writing the returned slice changed the Value: %v", again)
	}
}

// TestValueOfCopiesSliceInput — the []string branch of ValueOf must copy for the
// same reason List does.
func TestValueOfCopiesSliceInput(t *testing.T) {
	items := []string{"a"}
	v, err := ValueOf(items)
	if err != nil {
		t.Fatalf("ValueOf failed: %v", err)
	}
	items[0] = "mutated"
	got, _ := v.List()
	if got[0] != "a" {
		t.Errorf("ValueOf did not copy its input: %v", got)
	}
}

// TestValueStringIsForHumans — String() is a debugging aid, distinct from the wire
// form; the test exists so nobody starts relying on it as a codec.
func TestValueStringIsForHumans(t *testing.T) {
	cases := []struct {
		val  Value
		want string
	}{
		{Null(), "Null"},
		{Str("rust"), `"rust"`},
		{Int(-2024), "-2024"},
		{Bool(false), "false"},
		{List("a", "b"), `["a", "b"]`},
		{List(), "[]"},
	}
	for _, tc := range cases {
		if got := tc.val.String(); got != tc.want {
			t.Errorf("String() = %q, want %q", got, tc.want)
		}
	}

	var unknown Value
	if err := json.Unmarshal([]byte(`{"Blob":1}`), &unknown); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if got := unknown.String(); got != `{"Blob":1}` {
		t.Errorf("unknown String() = %q, want the raw JSON", got)
	}
}

// TestKindString — the kind names, including the default branch, since these show up
// in error messages a caller has to act on.
func TestKindString(t *testing.T) {
	cases := []struct {
		kind Kind
		want string
	}{
		{KindNull, "Null"},
		{KindStr, "Str"},
		{KindInt, "Int"},
		{KindBool, "Bool"},
		{KindList, "List"},
		{KindUnknown, "Unknown"},
		{Kind(99), "Kind(99)"},
	}
	for _, tc := range cases {
		if got := tc.kind.String(); got != tc.want {
			t.Errorf("Kind(%d).String() = %q, want %q", tc.kind, got, tc.want)
		}
	}
	// KindNull is the zero value, so an unset Value is a valid Null rather than a
	// broken half-built thing.
	if (Value{}).Kind() != KindNull {
		t.Error("the zero Value is not a Null")
	}
}

// TestAttrsMarshalsNilAsEmptyObject — the server's attrs field has no serde default,
// so `null` there is a deserialization error and a record built without attrs would
// be rejected for a reason the caller cannot see.
func TestAttrsMarshalsNilAsEmptyObject(t *testing.T) {
	var nilAttrs Attrs
	encoded, err := json.Marshal(nilAttrs)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(encoded) != "{}" {
		t.Errorf("nil Attrs marshalled as %s, want {}", encoded)
	}

	encoded, err = json.Marshal(Attrs{"lang": Str("rust"), "year": Int(2024)})
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	// encoding/json sorts map keys, so this is stable.
	if want := `{"lang":{"Str":"rust"},"year":{"Int":2024}}`; string(encoded) != want {
		t.Errorf("Attrs marshalled as %s, want %s", encoded, want)
	}
}

// TestAttrsRoundTrip — an attrs object decodes back to typed Values.
func TestAttrsRoundTrip(t *testing.T) {
	const doc = `{"lang":{"Str":"rust"},"n":{"Int":9007199254740993},"tags":{"List":["a"]},"empty":"Null"}`
	var attrs Attrs
	if err := json.Unmarshal([]byte(doc), &attrs); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if got, ok := attrs["lang"].Str(); !ok || got != "rust" {
		t.Errorf("lang = (%q, %v)", got, ok)
	}
	if got, ok := attrs["n"].Int(); !ok || got != 9007199254740993 {
		t.Errorf("n = (%d, %v), want the exact i64", got, ok)
	}
	if got, ok := attrs["tags"].List(); !ok || !reflect.DeepEqual(got, []string{"a"}) {
		t.Errorf("tags = (%v, %v)", got, ok)
	}
	if attrs["empty"].Kind() != KindNull {
		t.Errorf("empty kind = %v, want KindNull", attrs["empty"].Kind())
	}
	// An absent key is distinct from a Null one: the zero Value reads as Null, so the
	// difference lives in the map, which is exactly where the store keeps it too.
	if _, present := attrs["missing"]; present {
		t.Error("a key that was never sent is present in the map")
	}
}

// TestAttrsOfNamesTheFailingKey — "cannot use float64" alone is not enough to find
// the offending field in a wide document.
func TestAttrsOfNamesTheFailingKey(t *testing.T) {
	got, err := AttrsOf(map[string]any{"lang": "rust", "score": 1.5})
	if err == nil {
		t.Fatalf("AttrsOf succeeded as %v, want an error for the float", got)
	}
	if !strings.Contains(err.Error(), `"score"`) {
		t.Errorf("error = %q, want it to name the score attribute", err)
	}

	ok, err := AttrsOf(map[string]any{"lang": "rust", "year": 2024, "tags": []string{"a"}, "none": nil})
	if err != nil {
		t.Fatalf("AttrsOf failed: %v", err)
	}
	want := Attrs{"lang": Str("rust"), "year": Int(2024), "tags": List("a"), "none": Null()}
	if !reflect.DeepEqual(ok, want) {
		t.Errorf("AttrsOf = %v, want %v", ok, want)
	}
	if empty, err := AttrsOf(nil); err != nil || len(empty) != 0 {
		t.Errorf("AttrsOf(nil) = (%v, %v), want an empty Attrs", empty, err)
	}
}

// TestAttrsDecode — the loose-map view, for callers who prefer the JS SDK's shape to
// the typed accessors.
func TestAttrsDecode(t *testing.T) {
	got := Attrs{
		"lang": Str("rust"),
		"year": Int(2024),
		"ok":   Bool(true),
		"tags": List("a", "b"),
		"none": Null(),
	}.Decode()

	want := map[string]any{
		"lang": "rust",
		"year": int64(2024),
		"ok":   true,
		"tags": []string{"a", "b"},
		"none": nil,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Decode() = %#v, want %#v", got, want)
	}
	// int64, not float64: a caller type-asserting the result must get the value the
	// server sent, at full precision.
	if _, ok := got["year"].(int64); !ok {
		t.Errorf("year decoded as %T, want int64", got["year"])
	}
}
