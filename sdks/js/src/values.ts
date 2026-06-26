//! Ergonomic constructors and decoders for the externally-tagged `Value` wire type.
//
// Callers should never hand-write `{ Str: "x" }`. Use `v.str("x")`, `v.int(5)`,
// etc., or just pass plain JS values into `attrs` — `encodeValue` normalizes them.

import type { AttrInput, DecodedValue, Value } from "./types.js";

/**
 * Value constructors mirroring the `Value` variants. `v.int` requires a safe
 * integer (the store's attribute integer is an `i64`; a non-integer would be a
 * silent type error since there is no float attribute).
 */
export const v = {
  str: (s: string): Value => ({ Str: s }),
  int: (n: number): Value => {
    if (!Number.isInteger(n)) {
      throw new TypeError(`v.int expects an integer, got ${n}`);
    }
    return { Int: n };
  },
  bool: (b: boolean): Value => ({ Bool: b }),
  list: (items: string[]): Value => ({ List: items }),
  /** The explicit `Null` value — set-but-empty, distinct from an absent key. */
  nil: (): Value => "Null",
} as const;

/** True if `x` is already a wire-tagged {@link Value}. */
function isValue(x: unknown): x is Value {
  if (x === "Null") return true;
  if (typeof x !== "object" || x === null) return false;
  return (
    "Str" in x || "Int" in x || "Bool" in x || "List" in x
  );
}

/**
 * Normalize a caller-supplied {@link AttrInput} into the wire {@link Value} shape.
 * Plain scalars map by type; an already-tagged `Value` passes through unchanged.
 * Throws on a non-integer number or a non-string list element.
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
      return v.int(input);
    case "object":
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
