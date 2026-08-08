//! Ergonomic constructors and decoders for the externally-tagged `Value` wire type.
//
// Callers should never hand-write `{ Str: "x" }`. Use `v.str("x")`, `v.int(5)`,
// etc., or just pass plain JS values into `attrs` — `encodeValue` normalizes them.
//
// One rule is JS's alone. The store's `Int` and `Float` are separate types compared
// same-type only, but JS has one `number` and `1.0 === 1`, so `Number.isInteger` has to
// decide. A whole-numbered measurement therefore lands as an `Int` in whichever records
// it came out round, and a `Float` range filter then skips exactly those: write such a
// field with `v.float`. Go and Python have the types JS lacks and decide from them.

import type { AttrInput, DecodedValue, Value } from "./types.js";

/** Every tag this SDK version knows, in the order `decodeValue` tries them. */
const TAGS = ["Str", "Int", "Bool", "List", "Float", "DateTime"] as const;

/**
 * Value constructors mirroring the `Value` variants. `v.int` requires a safe
 * integer and `v.float` a finite number — `NaN`/`Infinity` have no JSON spelling,
 * and `JSON.stringify` would quietly write `null`.
 */
export const v = {
  str: (s: string): Value => ({ Str: s }),
  int: (n: number): Value => {
    if (!Number.isInteger(n)) {
      throw new TypeError(`v.int expects an integer, got ${n}`);
    }
    return { Int: n };
  },
  float: (n: number): Value => {
    if (typeof n !== "number" || !Number.isFinite(n)) {
      throw new TypeError(`v.float expects a finite number, got ${n}`);
    }
    return { Float: n };
  },
  bool: (b: boolean): Value => ({ Bool: b }),
  list: (items: string[]): Value => ({ List: items }),
  /**
   * A UTC instant, from a `Date` or a raw epoch-millisecond count. Milliseconds is
   * the wire type, so there is no sub-millisecond precision and no timezone.
   */
  datetime: (when: Date | number): Value => {
    const ms = when instanceof Date ? when.getTime() : when;
    if (!Number.isSafeInteger(ms)) {
      throw new TypeError(
        `v.datetime expects a valid Date or epoch ms, got ${when}`,
      );
    }
    return { DateTime: ms };
  },
  /** The explicit `Null` value — set-but-empty, distinct from an absent key. */
  nil: (): Value => "Null",
} as const;

/** True if `x` is already a wire-tagged {@link Value}. */
function isValue(x: unknown): x is Value {
  if (x === "Null") return true;
  if (typeof x !== "object" || x === null) return false;
  return TAGS.some((tag) => tag in x);
}

/**
 * Normalize a caller-supplied {@link AttrInput} into the wire {@link Value} shape.
 * Plain scalars map by type; an already-tagged `Value` passes through unchanged.
 * Throws on a non-finite number, an invalid `Date`, or a non-string list element.
 */
export function encodeValue(input: AttrInput): Value {
  if (isValue(input)) return input;
  if (input === null) return "Null";
  switch (typeof input) {
    case "string":
      return { Str: input };
    case "boolean":
      return { Bool: input };
    case "number":
      return Number.isInteger(input) ? v.int(input) : v.float(input);
    case "object":
      // Date before the array check: both are objects, only one is a list.
      if (input instanceof Date) return v.datetime(input);
      if (Array.isArray(input)) {
        if (!input.every((e) => typeof e === "string")) {
          throw new TypeError("a List attribute must contain only strings");
        }
        return { List: input };
      }
    // falls through
    default:
      throw new TypeError(`cannot encode attribute value: ${String(input)}`);
  }
}

/** Normalize a whole `attrs` map of {@link AttrInput} into wire {@link Value}s. */
export function encodeAttrs(
  attrs: Record<string, AttrInput>,
): Record<string, Value> {
  const out: Record<string, Value> = {};
  for (const [k, val] of Object.entries(attrs)) {
    out[k] = encodeValue(val);
  }
  return out;
}

/** Decode a wire {@link Value} back to a plain JS value. */
export function decodeValue(value: Value): DecodedValue {
  if (value === "Null") return null;
  if ("Str" in value) return value.Str;
  if ("Int" in value) return value.Int;
  if ("Bool" in value) return value.Bool;
  if ("List" in value) return value.List;
  if ("Float" in value) return value.Float;
  // A Date, not the raw number: a number would demote every instant to an Int
  // when a decoded attrs map is written back.
  if ("DateTime" in value) return new Date(value.DateTime);
  // Unknown tag (forward-compat): hand it back untouched.
  return value as unknown as DecodedValue;
}

/** Decode a whole wire `attrs` map back to plain JS values. */
export function decodeAttrs(
  attrs: Record<string, Value>,
): Record<string, DecodedValue> {
  const out: Record<string, DecodedValue> = {};
  for (const [k, val] of Object.entries(attrs)) {
    out[k] = decodeValue(val);
  }
  return out;
}
