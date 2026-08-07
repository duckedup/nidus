// Value and Attrs — the store's typed attribute value, and its JSON codec.
//
// On the wire a Value is the externally-tagged serde encoding of the crate's
// `Value` enum (src/model.rs): {"Str":…}, {"Int":…}, {"Bool":…}, {"List":[…]},
// {"Float":…}, {"DateTime":…}, and the unit variant as the bare string "Null".
// Callers should never hand-write those shapes; that is what the constructors are for.
//
// Go has no sum types, so Value is an opaque struct with a kind discriminant and
// unexported payload fields. The alternative — an `any` — would let a caller build
// an attribute nidus cannot store (a []int, a struct) and only find out when the
// server answered 400. With unexported fields an invalid Value is unrepresentable
// from outside the package: every path in is a constructor or ValueOf, and both
// reject what the store has no type for.
//
// Two properties of this file matter more than the rest, because both are silent
// when they go wrong:
//
//   - Integers keep full i64 precision. The server's Int is an i64, but JSON has one
//     number type and encoding/json decodes it to float64 by default, which rounds
//     anything past 2^53 — squarely where snowflake ids and nanosecond timestamps
//     live. Decoding goes through json.Number (the literal digits) instead.
//   - An unrecognised tag survives a round trip. A newer server may add a Value
//     variant; KindUnknown keeps the raw JSON verbatim and re-marshals it unchanged,
//     so an older SDK reading and rewriting a record does not silently drop data it
//     did not understand.

package nidus

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"
)

// Kind discriminates which variant a [Value] holds.
type Kind uint8

// The Value variants. KindNull is the zero value, so a Value{} is a valid Null
// rather than a broken half-built thing.
const (
	KindNull Kind = iota
	KindStr
	KindInt
	KindBool
	KindList
	KindFloat
	KindDateTime
	// KindUnknown is a tag this SDK version does not know, preserved verbatim.
	KindUnknown
)

// String names the kind, for error messages and debugging.
func (k Kind) String() string {
	switch k {
	case KindNull:
		return "Null"
	case KindStr:
		return "Str"
	case KindInt:
		return "Int"
	case KindBool:
		return "Bool"
	case KindList:
		return "List"
	case KindFloat:
		return "Float"
	case KindDateTime:
		return "DateTime"
	case KindUnknown:
		return "Unknown"
	default:
		return "Kind(" + strconv.Itoa(int(k)) + ")"
	}
}

// A Value is one typed metadata value attached to a [Record].
//
// Null is distinct from an absent key: absence means "not set / not indexed", Null
// means "set, and empty/none". The store relies on that difference (it is how a
// caller tells not-computed apart from computed-empty), so never collapse one into
// the other on either side of the wire.
//
// Comparisons on the server are same-type only, which is what makes Int and Float two
// variants rather than one number: an Int attribute never compares against a Float
// operand, so keep a given attribute's Go type uniform across every record you write.
type Value struct {
	kind Kind
	s    string
	i    int64 // KindInt, and KindDateTime as epoch milliseconds
	b    bool
	f    float64
	l    []string
	raw  json.RawMessage // KindUnknown only: the bytes exactly as they arrived
}

// Str is a string attribute.
func Str(s string) Value { return Value{kind: KindStr, s: s} }

// Int is an integer attribute (an i64 on the server).
func Int(i int64) Value { return Value{kind: KindInt, i: i} }

// Bool is a boolean attribute.
func Bool(b bool) Value { return Value{kind: KindBool, b: b} }

// Float is a double attribute. NaN and ±Inf are refused by the encoder rather than
// here, since JSON has no spelling for them — prefer [ValueOf], which catches one at
// the call site.
func Float(f float64) Value { return Value{kind: KindFloat, f: f} }

// DateTime is a UTC instant. It travels as epoch milliseconds, so sub-millisecond
// precision is truncated (time.Time.UnixMilli rounds toward negative infinity) and the
// location is dropped — an instant is absolute, and rendering it is the caller's job.
func DateTime(t time.Time) Value { return Value{kind: KindDateTime, i: t.UnixMilli()} }

// DateTimeMillis is [DateTime] built straight from an epoch-millisecond count, for a
// caller who already holds one and would otherwise round-trip through time.UnixMilli.
func DateTimeMillis(ms int64) Value { return Value{kind: KindDateTime, i: ms} }

// List is a list-of-strings attribute. The items are copied, so a later write to
// the caller's slice cannot mutate a Value that has already been built.
func List(items ...string) Value { return listValue(slices.Clone(items)) }

// listValue is the only place a KindList Value is built: it takes ownership of items
// and normalizes nil to an empty slice.
//
// The invariant it holds is that a List's slice is never nil, because the server's List
// is a Vec and `{"List": null}` fails to deserialize there rather than meaning "empty".
// Two call sites used to open-code the struct literal and re-establish that separately,
// which is exactly the kind of rule that is only correct until the third path forgets
// it. The copy-vs-own distinction lives at the boundary instead: exported List clones
// its variadic argument, callers that just built a slice hand it over.
func listValue(items []string) Value {
	if items == nil {
		items = []string{}
	}
	return Value{kind: KindList, l: items}
}

// Null is the explicit empty value — set-but-empty, distinct from an absent key.
func Null() Value { return Value{kind: KindNull} }

// Kind reports which variant v holds.
func (v Value) Kind() Kind { return v.kind }

// Str returns the string and true when v is a [KindStr], else the zero value and
// false. The comma-ok shape means a caller reading an attribute of the wrong type
// gets a testable false rather than a panic or a plausible-looking empty string.
func (v Value) Str() (string, bool) { return v.s, v.kind == KindStr }

// Int returns the integer and true when v is a [KindInt], else 0 and false.
func (v Value) Int() (int64, bool) { return v.i, v.kind == KindInt }

// Bool returns the boolean and true when v is a [KindBool], else false and false.
func (v Value) Bool() (bool, bool) { return v.b, v.kind == KindBool }

// Float returns the double and true when v is a [KindFloat], else 0 and false. It does
// not widen an [Int]: the two are separate types on the server and reading one as the
// other here would hide that from the caller.
func (v Value) Float() (float64, bool) { return v.f, v.kind == KindFloat }

// DateTime returns the instant in UTC and true when v is a [KindDateTime], else the
// zero Time and false.
func (v Value) DateTime() (time.Time, bool) {
	if v.kind != KindDateTime {
		return time.Time{}, false
	}
	return time.UnixMilli(v.i).UTC(), true
}

// List returns a copy of the items and true when v is a [KindList], else nil and
// false. The copy keeps the Value immutable from the caller's side.
func (v Value) List() ([]string, bool) {
	if v.kind != KindList {
		return nil, false
	}
	out := make([]string, len(v.l))
	copy(out, v.l)
	return out, true
}

// Any returns the value as a plain Go value: string, int64, bool, []string, float64,
// time.Time, or nil for Null. A [KindUnknown] value comes back as its
// [json.RawMessage] — the same "hand it back untouched" behaviour the JS SDK has, so
// forward-compatible data stays inspectable instead of vanishing.
func (v Value) Any() any {
	switch v.kind {
	case KindStr:
		return v.s
	case KindInt:
		return v.i
	case KindBool:
		return v.b
	case KindList:
		items, _ := v.List()
		return items
	case KindFloat:
		return v.f
	case KindDateTime:
		t, _ := v.DateTime()
		return t
	case KindUnknown:
		return v.raw
	default:
		return nil
	}
}

// String renders the value for logs and test failures. It is not the wire format —
// use [Value.MarshalJSON] for that.
func (v Value) String() string {
	switch v.kind {
	case KindStr:
		return strconv.Quote(v.s)
	case KindInt:
		return strconv.FormatInt(v.i, 10)
	case KindBool:
		return strconv.FormatBool(v.b)
	case KindList:
		quoted := make([]string, len(v.l))
		for i, item := range v.l {
			quoted[i] = strconv.Quote(item)
		}
		return "[" + strings.Join(quoted, ", ") + "]"
	case KindFloat:
		return strconv.FormatFloat(v.f, 'g', -1, 64)
	case KindDateTime:
		return time.UnixMilli(v.i).UTC().Format(time.RFC3339Nano)
	case KindUnknown:
		return string(v.raw)
	default:
		return "Null"
	}
}

// MarshalJSON writes the externally-tagged wire form.
func (v Value) MarshalJSON() ([]byte, error) {
	switch v.kind {
	case KindStr:
		return json.Marshal(struct {
			Str string `json:"Str"`
		}{v.s})
	case KindInt:
		return json.Marshal(struct {
			Int int64 `json:"Int"`
		}{v.i})
	case KindBool:
		return json.Marshal(struct {
			Bool bool `json:"Bool"`
		}{v.b})
	case KindList:
		// listValue owns the "never nil" invariant, so v.l is non-nil for every List
		// this package can build. This is the encode-side belt on it: a nil slice would
		// marshal as `null`, which the server's Vec rejects rather than reading as
		// "empty", and the zero value of Value is a Null so it can never reach here.
		items := v.l
		if items == nil {
			items = []string{}
		}
		return json.Marshal(struct {
			List []string `json:"List"`
		}{items})
	case KindFloat:
		if err := finiteFloat(v.f); err != nil {
			return nil, err
		}
		return json.Marshal(struct {
			Float float64 `json:"Float"`
		}{v.f})
	case KindDateTime:
		return json.Marshal(struct {
			DateTime int64 `json:"DateTime"`
		}{v.i})
	case KindUnknown:
		if v.raw == nil {
			return nil, fmt.Errorf("nidus: KindUnknown Value has no preserved JSON")
		}
		out := make([]byte, len(v.raw))
		copy(out, v.raw)
		return out, nil
	default:
		// The unit variant serde-encodes as a bare string, not an object.
		return []byte(`"Null"`), nil
	}
}

// UnmarshalJSON reads the externally-tagged wire form.
//
// An unrecognised tag is not an error: it becomes a [KindUnknown] holding the exact
// bytes, so a newer server's variant survives decode → encode through this SDK.
// A *known* tag carrying the wrong payload shape is an error, loudly — that is a
// contract violation, not a version skew.
func (v *Value) UnmarshalJSON(b []byte) error {
	trimmed := bytes.TrimSpace(b)
	if len(trimmed) == 0 {
		return fmt.Errorf("nidus: empty JSON where a Value was expected")
	}

	// "Null" is a bare string because serde encodes a fieldless variant that way.
	// Any other bare string is a fieldless variant from a future server, so it is
	// preserved rather than rejected.
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return fmt.Errorf("nidus: bad Value string %s: %w", preview(trimmed), err)
		}
		if s == "Null" {
			*v = Null()
			return nil
		}
		v.setUnknown(trimmed)
		return nil
	}

	var tagged map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &tagged); err != nil {
		return fmt.Errorf(
			"nidus: %s is not a Value (want %q or a single-key tagged object): %w",
			preview(trimmed), "Null", err,
		)
	}
	if len(tagged) != 1 {
		// Zero or several tags is not a shape this SDK can interpret; keep it whole.
		v.setUnknown(trimmed)
		return nil
	}
	var tag string
	var payload json.RawMessage
	for k, p := range tagged {
		tag, payload = k, p
	}

	switch tag {
	case "Str":
		var s string
		if err := json.Unmarshal(payload, &s); err != nil {
			return fmt.Errorf("nidus: Str attribute is not a string: %w", err)
		}
		*v = Str(s)
	case "Int":
		n, err := numberPayload("Int", payload)
		if err != nil {
			return err
		}
		i, err := intFromJSON("Int", n)
		if err != nil {
			return err
		}
		*v = Int(i)
	case "Float":
		n, err := numberPayload("Float", payload)
		if err != nil {
			return err
		}
		// ParseFloat over the literal, not a json.Unmarshal into a float64: that would
		// read a `null` payload as a silent 0.0, where an empty json.Number fails here.
		f, err := strconv.ParseFloat(n.String(), 64)
		if err != nil {
			return fmt.Errorf("nidus: %q is not a valid Float attribute: %w", n.String(), err)
		}
		*v = Float(f)
	case "DateTime":
		n, err := numberPayload("DateTime", payload)
		if err != nil {
			return err
		}
		ms, err := intFromJSON("DateTime", n)
		if err != nil {
			return err
		}
		*v = DateTimeMillis(ms)
	case "Bool":
		var b bool
		if err := json.Unmarshal(payload, &b); err != nil {
			return fmt.Errorf("nidus: Bool attribute is not a boolean: %w", err)
		}
		*v = Bool(b)
	case "List":
		var items []string
		if err := json.Unmarshal(payload, &items); err != nil {
			return fmt.Errorf("nidus: List attribute is not a list of strings: %w", err)
		}
		// listValue, not List(items...): the slice was just decoded and is ours already,
		// so a second copy would buy nothing. It also normalizes the `{"List": null}`
		// case, where json.Unmarshal leaves items nil.
		*v = listValue(items)
	default:
		v.setUnknown(trimmed)
	}
	return nil
}

// setUnknown stashes a copy of the bytes. The copy matters: encoding/json reuses
// the buffer it hands to UnmarshalJSON, so keeping the slice would leave the Value
// pointing at whatever gets decoded next.
func (v *Value) setUnknown(raw []byte) {
	cp := make(json.RawMessage, len(raw))
	copy(cp, raw)
	*v = Value{kind: KindUnknown, raw: cp}
}

// ValueOf normalizes a plain Go value into a [Value]:
//
//	Value          → itself, untouched
//	nil            → Null
//	string         → Str
//	bool           → Bool
//	int, int8…int64, uint, uint8…uint64 (in i64 range) → Int
//	float32, float64 (finite) → Float
//	time.Time      → DateTime
//	[]string       → List
//	[]any          → List, if every element is a string
//
// The static type decides Int vs Float, so float64(2024) is a Float. JS has no such
// type and decides by Number.isInteger instead, so an attribute written from both SDKs
// needs `v.float` on the JS side to stay one type. Anything else here is an error.
func ValueOf(x any) (Value, error) {
	switch t := x.(type) {
	case Value:
		return t, nil
	case nil:
		return Null(), nil
	case string:
		return Str(t), nil
	case bool:
		return Bool(t), nil
	case int:
		return Int(int64(t)), nil
	case int8:
		return Int(int64(t)), nil
	case int16:
		return Int(int64(t)), nil
	case int32:
		return Int(int64(t)), nil
	case int64:
		return Int(t), nil
	case uint:
		return uintValue(uint64(t))
	case uint8:
		return Int(int64(t)), nil
	case uint16:
		return Int(int64(t)), nil
	case uint32:
		return Int(int64(t)), nil
	case uint64:
		return uintValue(t)
	case float32:
		// Widened, not reinterpreted: 0.1 as a float32 is 0.10000000149011612 as a
		// float64, which is the number the caller has been holding all along.
		return floatValue(float64(t))
	case float64:
		return floatValue(t)
	case time.Time:
		return DateTime(t), nil
	case []string:
		return List(t...), nil
	case []any:
		// Callers who got their attrs out of some other JSON decode hold []any, so
		// accept it — but only when every element really is a string, since List is
		// a Vec<String> on the server and there is no mixed-type list.
		items := make([]string, len(t))
		for i, e := range t {
			s, ok := e.(string)
			if !ok {
				return Value{}, fmt.Errorf(
					"nidus: a List attribute holds only strings, but element %d is %T", i, e,
				)
			}
			items[i] = s
		}
		return listValue(items), nil
	default:
		return Value{}, fmt.Errorf("nidus: cannot use %T as an attribute value", x)
	}
}

// MustValueOf is [ValueOf] for literals whose type is known at the call site —
// tests, examples, docs. It panics on a value the store cannot hold, so production
// code that normalizes caller input should use ValueOf and handle the error.
func MustValueOf(x any) Value {
	v, err := ValueOf(x)
	if err != nil {
		panic(err)
	}
	return v
}

// uintValue guards the one unsigned case that does not fit: the server's Int is an
// i64, so a u64 past its max has nowhere to go and must not wrap into a negative.
func uintValue(u uint64) (Value, error) {
	if u > math.MaxInt64 {
		return Value{}, fmt.Errorf("nidus: %d overflows the int64 an Int attribute holds", u)
	}
	return Int(int64(u)), nil
}

// floatValue is the ValueOf half of the finiteness rule, so a NaN in a filter fails at
// the call site rather than from the request that carries it.
func floatValue(f float64) (Value, error) {
	if err := finiteFloat(f); err != nil {
		return Value{}, err
	}
	return Float(f), nil
}

// finiteFloat refuses the three doubles JSON cannot spell. serde_json writes them as
// `null` and then refuses to read one back, so a NaN that reached the wire would come
// home as a 400 naming the server's parser rather than the attribute that caused it.
func finiteFloat(f float64) error {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return fmt.Errorf("nidus: %v cannot be a Float attribute; JSON has no NaN or Infinity", f)
	}
	return nil
}

// Attrs is a record's typed metadata, keyed by attribute name.
//
// Deliberate deviation from the JS SDK: values stay as [Value] on [Hit] and
// [Record] instead of being decoded to a loose map. In a statically typed language
// the typed accessor is the better surface — hit.Attrs["lang"].Str() beats an `any`
// type assertion — and [Attrs.Decode] is right here for callers who do want the
// loose map. Please do not "fix" this toward JS.
type Attrs map[string]Value

// AttrsOf normalizes a plain map into [Attrs], running [ValueOf] over each entry.
// The failing key is named in the error, since "cannot use []int" alone is not
// enough to find the offending field in a wide document.
func AttrsOf(m map[string]any) (Attrs, error) {
	out := make(Attrs, len(m))
	for k, raw := range m {
		v, err := ValueOf(raw)
		if err != nil {
			return nil, fmt.Errorf("nidus: attribute %q: %w", k, err)
		}
		out[k] = v
	}
	return out, nil
}

// Decode returns the attributes as plain Go values via [Value.Any] — the shape the
// JS SDK hands back, for callers who prefer it to the typed accessors.
func (a Attrs) Decode() map[string]any {
	out := make(map[string]any, len(a))
	for k, v := range a {
		out[k] = v.Any()
	}
	return out
}

// MarshalJSON emits `{}` rather than `null` for a nil map. The server's `attrs`
// field has no serde default, so `null` there is a deserialization error and a
// record built without attrs would be rejected for a reason the caller cannot see.
func (a Attrs) MarshalJSON() ([]byte, error) {
	if a == nil {
		return []byte("{}"), nil
	}
	return json.Marshal(map[string]Value(a))
}

// numberPayload decodes a numeric variant's payload to its literal digits.
//
// The quoted-string check comes first because encoding/json accepts a JSON *string*
// holding a number literal into a json.Number ("2024" → 2024), so the decode below
// would let a wrong-shaped payload through and silently rewrite it as a bare number on
// the way out. serde emits and accepts a bare JSON number only, so a string here is a
// contract violation, not something to repair on the server's behalf.
func numberPayload(tag string, payload json.RawMessage) (json.Number, error) {
	if p := bytes.TrimSpace(payload); len(p) > 0 && p[0] == '"' {
		return "", fmt.Errorf(
			"nidus: %s attribute %s is a string; it travels as a bare JSON number",
			tag, preview(p),
		)
	}
	var n json.Number
	if err := json.Unmarshal(payload, &n); err != nil {
		return "", fmt.Errorf("nidus: %s attribute is not a number: %w", tag, err)
	}
	return n, nil
}

// intFromJSON turns a JSON number into the i64 behind the server's Int and DateTime
// variants (a DateTime is epoch milliseconds, so it has the same range and the same
// precision hazard).
//
// It reads the literal text rather than a float64 on purpose: encoding/json's
// default number type is float64, which silently rounds integers above 2^53 — the
// range ids and timestamps live in. json.Number keeps every digit, so ParseInt sees
// the value the server actually sent.
//
// A fractional part is tolerated only when it is all zeros ("2024.0" → 2024),
// because some other encoder may have written an integer that way and JSON cannot
// tell us it meant an integer. "2024.5" is an error, never a truncation: a fractional
// number is a Float attribute, and dropping the .5 would be silent data loss.
func intFromJSON(tag string, n json.Number) (int64, error) {
	s := n.String()
	if i, err := strconv.ParseInt(s, 10, 64); err == nil {
		return i, nil
	}
	if mantissa, frac, ok := strings.Cut(s, "."); ok && allZeros(frac) {
		// Handled textually so a big integer written as "9007199254740993.0" keeps
		// its last digit; going through float64 here would round it away.
		if i, err := strconv.ParseInt(mantissa, 10, 64); err == nil {
			return i, nil
		}
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("nidus: %s is not a valid %s attribute: %w", s, tag, err)
	}
	if f != math.Trunc(f) {
		return 0, fmt.Errorf(
			"nidus: %s attribute %s is not an integer; a fractional number is a Float", tag, s,
		)
	}
	// float64(1<<63) is exactly 2^63, the first value an int64 cannot hold.
	if f < math.MinInt64 || f >= float64(1<<63) {
		return 0, fmt.Errorf("nidus: %s attribute %s overflows int64", tag, s)
	}
	return int64(f), nil
}

func allZeros(s string) bool {
	return s != "" && strings.Trim(s, "0") == ""
}

// preview truncates a JSON snippet for an error message, so a malformed multi-KB
// body does not end up quoted in full.
func preview(b []byte) string {
	const max = 48
	if len(b) <= max {
		return string(b)
	}
	return string(b[:max]) + "…"
}
