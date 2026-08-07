// Predicate and Filter — metadata predicates and the bare-array wire shape.
//
// A Filter is AND-combined predicates. On the wire the crate's Filter is a newtype
// over Vec<Predicate> (src/model.rs), so it serializes as a plain JSON array — never
// an object wrapping one. An empty array matches everything.
//
// Each Predicate is a positive assertion about a *present* attribute: an absent key
// matches nothing, including the negative predicates (Ne, NotIn) and the ranges.
// Comparisons are same-type only — Int↔Int numerically, Str↔Str lexically,
// Bool↔Bool — which is why the builders normalize through [ValueOf] rather than
// coercing types to make a comparison "work".
//
// Two asymmetries to keep straight, because they are the wire format and not a choice:
// Glob's and IGlob's second tuple element is a bare string (the pattern), while every
// other leaf predicate's is a tagged Value, and In/NotIn/ContainsAny's is an array of
// them. And the combinators are not key/value tuples at all — All and Any wrap a bare
// array of predicates, Not wraps a single one.

package nidus

import (
	"encoding/json"
	"fmt"
)

// The externally-tagged variant names, exactly as serde spells them.
const (
	opEq          = "Eq"
	opNe          = "Ne"
	opGlob        = "Glob"
	opIGlob       = "IGlob"
	opIn          = "In"
	opNotIn       = "NotIn"
	opLt          = "Lt"
	opLe          = "Le"
	opGt          = "Gt"
	opGe          = "Ge"
	opContains    = "Contains"
	opNotContains = "NotContains"
	opContainsAny = "ContainsAny"
	opAll         = "All"
	opAny         = "Any"
	opNot         = "Not"
)

// A Predicate is one attribute condition. Build it with [Eq], [Ne], [Glob], [IGlob],
// [In], [NotIn], [Lt], [Le], [Gt], or [Ge]; the fields are unexported so a predicate
// the server cannot parse cannot be constructed.
type Predicate struct {
	op   string
	key  string
	val  Value       // Eq, Ne, Lt, Le, Gt, Ge, Contains, NotContains
	vals []Value     // In, NotIn, ContainsAny
	pat  string      // Glob, IGlob
	subs []Predicate // All, Any, Not (Not holds exactly one)
	err  error       // a normalization failure, surfaced from MarshalJSON
}

// A Filter is a conjunction (AND) of predicates. The zero value — a nil Filter —
// matches everything, same as an empty one.
type Filter []Predicate

// The builders take `any` so that nidus.Eq("year", 2024) reads the way a caller
// expects, which means normalization can fail inside a function with no error to
// return. The three ways out are: panic (hostile in a library), swallow the error and
// send a wrong-but-valid body (silent data bugs), or carry the error. We carry it:
// the Predicate remembers the failure and returns it from MarshalJSON, so it lands
// as an ordinary error from the Search call that used the filter, at a call site the
// caller is already checking. Use [Predicate.Err] or [Filter.Err] to check earlier.

// Eq matches records where attrs[key] equals v.
func Eq(key string, v any) Predicate { return unary(opEq, key, v) }

// Ne matches records where attrs[key] is present and does not equal v.
func Ne(key string, v any) Predicate { return unary(opNe, key, v) }

// Glob matches records where attrs[key] is a string matching the pattern (*, ?,
// [..]). The pattern travels as a bare string, not a Value — Glob is the one
// asymmetric variant on the wire.
func Glob(key, pattern string) Predicate {
	return Predicate{op: opGlob, key: key, pat: pattern}
}

// IGlob is [Glob] ignoring ASCII case on both sides: IGlob("path", "Src/*") matches
// "src/main.rs". Non-ASCII is not folded, so É does not match é.
func IGlob(key, pattern string) Predicate {
	return Predicate{op: opIGlob, key: key, pat: pattern}
}

// In matches records where attrs[key] equals one of vals.
func In(key string, vals ...any) Predicate { return set(opIn, key, vals) }

// NotIn matches records where attrs[key] is present and equals none of vals.
func NotIn(key string, vals ...any) Predicate { return set(opNotIn, key, vals) }

// Lt matches records where attrs[key] < v (same-type, orderable).
func Lt(key string, v any) Predicate { return unary(opLt, key, v) }

// Le matches records where attrs[key] <= v (same-type, orderable).
func Le(key string, v any) Predicate { return unary(opLe, key, v) }

// Gt matches records where attrs[key] > v (same-type, orderable).
func Gt(key string, v any) Predicate { return unary(opGt, key, v) }

// Ge matches records where attrs[key] >= v (same-type, orderable).
func Ge(key string, v any) Predicate { return unary(opGe, key, v) }

// Contains matches records where attrs[key] is a list containing v. Matching is
// whole-element, not substring: Contains("tags", "rust") does not match ["rustacean"].
func Contains(key string, v any) Predicate { return unary(opContains, key, v) }

// NotContains matches records where attrs[key] is a present list not containing v.
// Like [Ne], it requires the attribute to exist and be a list.
func NotContains(key string, v any) Predicate { return unary(opNotContains, key, v) }

// ContainsAny matches records where attrs[key] is a list sharing at least one element
// with vals. An empty set matches nothing. "Contains all of" is [All] over [Contains].
func ContainsAny(key string, vals ...any) Predicate { return set(opContainsAny, key, vals) }

// All matches records where every sub-predicate holds. All() with no arguments is
// true, the identity for AND.
func All(preds ...Predicate) Predicate { return group(opAll, preds) }

// Any matches records where at least one sub-predicate holds. Any() with no arguments
// is false, the identity for OR.
func Any(preds ...Predicate) Predicate { return group(opAny, preds) }

// Not matches records where pred does not hold. It differs from [Ne] on an absent key:
// Not(Eq(k, v)) matches a record with no k at all, whereas Ne(k, v) does not, because
// Ne asserts a present-and-different attribute. Use Ne/NotIn/NotContains to require
// the attribute exist, and Not for genuine complement.
func Not(pred Predicate) Predicate { return group(opNot, []Predicate{pred}) }

// And collects predicates into a [Filter]. It is sugar — predicates already AND —
// but it reads better at a call site than a slice literal. For a nested conjunction
// *inside* another group, use [All], which is a Predicate rather than a Filter.
func And(preds ...Predicate) Filter { return Filter(preds) }

// group builds a boolean combinator, propagating the first sub-predicate error so a
// malformed leaf cannot hide inside a nested group and ship a wrong-but-valid body.
func group(op string, preds []Predicate) Predicate {
	for _, p := range preds {
		if p.err != nil {
			return Predicate{op: op, err: p.err}
		}
	}
	return Predicate{op: op, subs: preds}
}

func unary(op, key string, v any) Predicate {
	val, err := ValueOf(v)
	if err != nil {
		return Predicate{op: op, key: key, err: fmt.Errorf("nidus: %s(%q): %w", op, key, err)}
	}
	return Predicate{op: op, key: key, val: val}
}

func set(op, key string, raw []any) Predicate {
	vals := make([]Value, len(raw))
	for i, r := range raw {
		v, err := ValueOf(r)
		if err != nil {
			return Predicate{
				op:  op,
				key: key,
				err: fmt.Errorf("nidus: %s(%q) value %d: %w", op, key, i, err),
			}
		}
		vals[i] = v
	}
	return Predicate{op: op, key: key, vals: vals}
}

// Err reports a value that could not be normalized when the predicate was built
// (for example a float, which is not an attribute type). It is nil for a usable
// predicate. Checking it is optional: the same error comes back from the request
// that carries the filter.
func (p Predicate) Err() error { return p.err }

// Err returns the first predicate error in the filter, or nil.
func (f Filter) Err() error {
	for _, p := range f {
		if err := p.err; err != nil {
			return err
		}
	}
	return nil
}

// MarshalJSON writes the externally-tagged 2-tuple form, e.g.
// {"Eq":["lang",{"Str":"rust"}]} or {"Glob":["path","src/*"]}.
func (p Predicate) MarshalJSON() ([]byte, error) {
	if p.err != nil {
		return nil, p.err
	}
	if p.op == "" {
		return nil, fmt.Errorf(
			"nidus: zero-value Predicate; build one with Eq, Ne, Glob, IGlob, In, NotIn, " +
				"Lt, Le, Gt, Ge, Contains, NotContains, ContainsAny, All, Any or Not",
		)
	}
	// The combinators are not [key, value] 2-tuples: All/Any wrap a bare array of
	// predicates and Not wraps a single one, so they marshal before the tuple path.
	switch p.op {
	case opAll, opAny:
		subs := p.subs
		if subs == nil {
			subs = []Predicate{}
		}
		return json.Marshal(map[string][]Predicate{p.op: subs})
	case opNot:
		if len(p.subs) != 1 {
			return nil, fmt.Errorf("nidus: Not takes exactly one predicate, got %d", len(p.subs))
		}
		return json.Marshal(map[string]Predicate{p.op: p.subs[0]})
	}
	var second any
	switch p.op {
	case opGlob, opIGlob:
		second = p.pat
	case opIn, opNotIn, opContainsAny:
		// A nil set must still be `[]`: the server's is a Vec, and `null` would fail
		// to deserialize rather than mean "an empty set, matching nothing".
		vals := p.vals
		if vals == nil {
			vals = []Value{}
		}
		second = vals
	default:
		second = p.val
	}
	return json.Marshal(map[string][]any{p.op: {p.key, second}})
}

// MarshalJSON writes the bare array. A nil Filter is `[]` rather than `null`,
// because the server deserializes this field into a Vec and `null` is an error
// there, not an empty filter.
func (f Filter) MarshalJSON() ([]byte, error) {
	if f == nil {
		return []byte("[]"), nil
	}
	return json.Marshal([]Predicate(f))
}
