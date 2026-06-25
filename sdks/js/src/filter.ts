//! Filter builder producing the bare predicate-array wire shape.
//
// A `Filter` is AND-combined predicates; on the wire it is a plain array. Each
// predicate is a *positive assertion about a present attribute* — an absent key
// matches nothing, including the negative predicates (`ne`/`notIn`) and ranges.
// Comparisons are same-type only (Int↔Int numeric, Str↔Str lexical, Bool↔Bool).

import type { AttrInput, Filter, Predicate, Value } from "./types.js";
import { encodeValue } from "./values.js";

/**
 * Predicate constructors. Each accepts a plain JS value (auto-normalized) or an
 * explicit `v.*` {@link Value}. Combine results into a {@link Filter} array, or
 * use {@link f.and} for readability.
 */
export const f = {
  /** `attrs[key] === value`. */
  eq: (key: string, value: AttrInput): Predicate => ({
    Eq: [key, encodeValue(value)],
  }),
  /** `attrs[key]` is present and `!== value`. */
  ne: (key: string, value: AttrInput): Predicate => ({
    Ne: [key, encodeValue(value)],
  }),
  /** `attrs[key]` is a `Str` matching the glob pattern (`*`, `?`, `[..]`). */
  glob: (key: string, pattern: string): Predicate => ({ Glob: [key, pattern] }),
  /** `attrs[key]` equals one of `values`. */
  in: (key: string, values: AttrInput[]): Predicate => ({
    In: [key, values.map(encodeValue)],
  }),
  /** `attrs[key]` is present and equals none of `values`. */
  notIn: (key: string, values: AttrInput[]): Predicate => ({
    NotIn: [key, values.map(encodeValue)],
  }),
  /** `attrs[key] < value` (same-type, orderable). */
  lt: (key: string, value: AttrInput): Predicate => ({
    Lt: [key, encodeValue(value)],
  }),
  /** `attrs[key] <= value` (same-type, orderable). */
  le: (key: string, value: AttrInput): Predicate => ({
    Le: [key, encodeValue(value)],
  }),
  /** `attrs[key] > value` (same-type, orderable). */
  gt: (key: string, value: AttrInput): Predicate => ({
    Gt: [key, encodeValue(value)],
  }),
  /** `attrs[key] >= value` (same-type, orderable). */
  ge: (key: string, value: AttrInput): Predicate => ({
    Ge: [key, encodeValue(value)],
  }),
  /** Collect predicates into a {@link Filter} (purely sugar — they already AND). */
  and: (...preds: Predicate[]): Filter => preds,
} as const;

// Aliases for the comparison operators, for callers who prefer them.
export type { Filter, Predicate, Value };
