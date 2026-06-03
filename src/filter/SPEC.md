# `filter` module — spec

Implement `pub fn matches(filter: &Filter, attrs: &BTreeMap<String, Value>) -> bool`
in `mod.rs`. **Do not change the signature.** Root design: `../../SPEC.md` §7.
`Filter`, `Predicate`, `Value` live in `crate::model` (already defined). This module
depends on `crate::glob::glob_match` — it is already implemented.

## Semantics
`matches` returns true iff **every** predicate in `filter.0` holds (AND). An empty
filter (`Filter(vec![])`) matches everything. Per predicate, given `key`:
- `Predicate::Eq(key, value)` — true iff `attrs.get(key) == Some(value)` (exact
  `Value` equality, including `Null` vs absent: absent key never equals `Null`).
- `Predicate::Glob(key, pattern)` — true iff `attrs.get(key)` is
  `Some(Value::Str(s))` and `glob_match(pattern, s)` is true. Non-string or absent →
  false.
- `Predicate::In(key, set)` — true iff `attrs.get(key)` is `Some(v)` and `set`
  contains a `Value` equal to `v`. Absent → false. Empty `set` → false.

A predicate whose `key` is absent fails (so the whole filter fails) — except note
`Eq(key, Value::Null)` requires the key to be *present and equal to* `Null`.

## Constraints
Pure safe Rust (`#![forbid(unsafe_code)]`). No new dependencies. No IO. Keep it
small and allocation-free where reasonable.

## Tests (inline, Miri-clean)
Empty filter matches; single and multiple predicates (AND); `Eq` exact incl.
`Null`-present vs absent; `Glob` only on `Str` and via real patterns
(`"src/*"`); `In` membership incl. empty set = false; absent key fails for each
predicate kind; a multi-predicate filter where one fails → overall false.
