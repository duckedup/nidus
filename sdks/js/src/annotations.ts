//! Decoding a hit's optional annotations — the opt-in "why did this match".
//
// One rule here is JS's alone. The server reports a highlight span as a **UTF-8 byte**
// range into the fragment text, but a JS string is indexed in UTF-16 code units, so
// `text.slice(...span)` on a raw span is wrong for any non-ASCII excerpt. Converted here,
// once, so a caller's obvious slice is the right one.

import type { Annotations, Fragment, Highlight } from "./types.js";

/** A fragment as it arrives: `spans` are UTF-8 byte offsets into `text`. */
interface WireFragment {
  text: string;
  spans: [number, number][];
}

interface WireHighlight {
  field: string;
  fragments: WireFragment[];
}

/** A hit's annotations as they arrive, before the span offsets are converted. */
export interface WireAnnotations {
  vector?: { rank: number; score: number };
  text?: { rank: number; score: number };
  clauses?: { field: string; score: number }[];
  highlights?: WireHighlight[];
}

/**
 * Decode a hit's annotations, converting every highlight span to JS string indices. The
 * parts the server omitted stay omitted rather than becoming empty arrays.
 */
export function decodeAnnotations(a: WireAnnotations): Annotations {
  const out: Annotations = {};
  if (a.vector) out.vector = a.vector;
  if (a.text) out.text = a.text;
  if (a.clauses) out.clauses = a.clauses;
  if (a.highlights) out.highlights = a.highlights.map(decodeHighlight);
  return out;
}

function decodeHighlight(h: WireHighlight): Highlight {
  return { field: h.field, fragments: h.fragments.map(decodeFragment) };
}

function decodeFragment(fr: WireFragment): Fragment {
  return { text: fr.text, spans: toStringIndices(fr.text, fr.spans) };
}

/**
 * Convert UTF-8 byte ranges into `text` to JS string indices (UTF-16 code units), so
 * `text.slice(...span)` yields the matched term. An all-ASCII excerpt needs no conversion
 * and is the common case, so it is detected before any table is built.
 */
export function toStringIndices(
  text: string,
  spans: [number, number][],
): [number, number][] {
  if (spans.length === 0 || isAscii(text)) return spans;
  const index = byteToUnit(text);
  const at = (b: number): number =>
    index[Math.min(Math.max(b, 0), index.length - 1)]!;
  return spans.map(([start, end]): [number, number] => [at(start), at(end)]);
}

/** One entry per byte of `text` (plus its end), holding that byte's UTF-16 index. */
function byteToUnit(text: string): number[] {
  const index: number[] = [];
  let unit = 0;
  // A byte *inside* a codepoint maps to that codepoint's start. Spans land on token
  // boundaries, so this only keeps a malformed offset from landing mid-surrogate.
  for (const ch of text) {
    for (let n = utf8Len(ch.codePointAt(0)!); n > 0; n--) index.push(unit);
    unit += ch.length;
  }
  index.push(unit);
  return index;
}

function utf8Len(codePoint: number): number {
  if (codePoint < 0x80) return 1;
  if (codePoint < 0x800) return 2;
  return codePoint < 0x10000 ? 3 : 4;
}

/** True when every code unit is ASCII, i.e. byte offsets already *are* string indices. */
function isAscii(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    if (text.charCodeAt(i) > 0x7f) return false;
  }
  return true;
}
