// Seeded corpus + a deterministic pseudo-vector, shared by the worker (the live path)
// and Terminal.astro (the static pre-click fallback) so the two can never drift (nidus-7pj).
//
// NOT an embedding. Two honest, hand-built signals stand in for one:
//   1. character trigrams, so "rotation"/"rotate" and typos share dimensions
//   2. CONCEPTS below, a hand-written concept map, so a query can match wording it
//      does not literally share
// Both are disclosed on screen. See Terminal.astro's disclosure line.

export const DIM = 256;

// FNV-1a, 32-bit. Pure and host-independent: the same text hashes identically in the
// worker, the component, and the E2E harness.
function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

// A hand-written stand-in for what an embedding model learns. Each line is one concept;
// every word on it lands in that concept's dimensions, so a query sharing no words with
// a record still scores against it. It only knows what is written here, which is exactly
// the limitation the on-screen disclosure names.
const CONCEPTS = [
  ["crash", "safe", "safety", "durable", "durability", "fsync", "torn", "recover", "recovery", "lose", "lost", "power", "failure", "atomic", "commit"],
  ["store", "storage", "disk", "file", "directory", "folder", "segment", "manifest", "append", "persist", "persistence", "save", "write"],
  ["cost", "ram", "size", "bytes", "usage", "footprint", "rows", "count", "big", "large", "occupy", "consume", "overhead"],
  ["search", "query", "find", "lookup", "nearest", "neighbour", "neighbor", "similar", "similarity", "rank", "ranking", "score", "relevant", "recall", "match"],
  ["vector", "embedding", "embed", "dimension", "cosine", "dot", "euclidean", "distance", "semantic"],
  ["filter", "narrow", "where", "predicate", "range", "glob", "regex", "fuzzy", "equality", "metadata", "attribute", "attrs"],
  ["fast", "faster", "speed", "quick", "quicker", "performance", "latency", "throughput", "index", "hnsw", "ivf", "approximate", "quantization", "quantize", "int8", "binary", "compress", "shrink", "smaller"],
  ["memory", "remember", "recall", "agent", "text", "language", "natural", "summarize", "provider"],
  ["server", "http", "network", "remote", "client", "sdk", "api", "endpoint", "route", "mcp"],
  ["auth", "token", "bearer", "credential", "secret", "key", "rotation", "rotate", "security", "secure", "permission", "access"],
  ["cloud", "s3", "gcs", "bucket", "redis", "valkey", "tier", "shared", "distributed", "instance", "scale"],
  ["browser", "wasm", "web", "opfs", "client", "offline", "local"],
  ["build", "compile", "dependency", "rust", "crate", "install", "toolchain", "cargo"],
];

// Crude suffix stripping so "faster" reaches "fast" and "rotating" reaches "rotate".
// Not a real stemmer; it only has to be consistent between query and document.
function stem(word) {
  for (const suffix of ["ing", "est", "ed", "er", "ly", "s"]) {
    if (word.length - suffix.length >= 4 && word.endsWith(suffix)) {
      return word.slice(0, -suffix.length);
    }
  }
  return word;
}

// stem -> the concept dimensions it belongs to, built once at module load.
const CONCEPT_DIMS = new Map();
CONCEPTS.forEach((words, i) => {
  // Two dimensions per concept, so distinct concepts rarely collide in DIM buckets.
  const dims = [fnv1a(`concept-${i}-a`) % DIM, fnv1a(`concept-${i}-b`) % DIM];
  for (const w of words) {
    for (const form of [w, stem(w)]) {
      CONCEPT_DIMS.set(form, (CONCEPT_DIMS.get(form) || []).concat(dims));
    }
  }
});

const CONCEPT_KEYS = [...CONCEPT_DIMS.keys()];

// True when one edit apart, counting a transposition as one: "rotaiton" is a typo of
// "rotation", which plain Levenshtein would score as two.
function withinOneEdit(a, b) {
  if (Math.abs(a.length - b.length) > 1) return false;
  if (a === b) return true;
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  let j = 0;
  while (j < a.length - i && j < b.length - i && a[a.length - 1 - j] === b[b.length - 1 - j]) j++;
  const ra = a.length - i - j;
  const rb = b.length - i - j;
  if (ra <= 1 && rb <= 1) return true;
  return ra === 2 && rb === 2 && a[i] === b[i + 1] && a[i + 1] === b[i];
}

// A concept hit for a word the map does not know exactly: the nearest concept key one
// edit away. Only tried on a miss, so an exact hit never pays for it.
function fuzzyDims(word) {
  if (word.length < 5) return null;
  for (const key of CONCEPT_KEYS) {
    if (withinOneEdit(word, key)) return CONCEPT_DIMS.get(key);
  }
  return null;
}

// Concepts dominate: trigrams are a fuzzy tiebreak for wording the map does not know,
// not the main signal. Weighted equally, common English trigrams drown the concepts.
const CONCEPT_WEIGHT = 8.0;
const TRIGRAM_WEIGHT = 1.0;

// Function words carry no topic and would otherwise supply most of the trigram overlap
// between any two English sentences.
const STOP = new Set(("a an the and or but if then this that these those is are was were be been "
  + "do does did doing done can could will would shall should may might must have has had "
  + "i you he she it we they them his her its our your their of in on at to for with by from "
  + "as into over under about up down out off no not so than too very just how what when where "
  + "which who why all any each few more most other some such only own same s t don now").split(" "));

function tokens(text) {
  return text.toLowerCase().split(/[^a-z0-9]+/).filter(w => w && !STOP.has(w));
}

/**
 * A deterministic pseudo-vector: character trigrams plus hand-written concept
 * dimensions. Left unnormalized; nidus unit-normalizes on insert and scores by dot.
 */
export function hashVector(text) {
  const vec = new Array(DIM).fill(0);
  for (const token of tokens(text)) {
    // Trigrams only for words long enough to carry meaning; short ones are noise.
    if (token.length >= 4) {
      const padded = ` ${token} `;
      for (let i = 0; i + 3 <= padded.length; i++) {
        vec[fnv1a(padded.slice(i, i + 3)) % DIM] += TRIGRAM_WEIGHT;
      }
    }
    const st = stem(token);
    const dims = CONCEPT_DIMS.get(token) || CONCEPT_DIMS.get(st) || fuzzyDims(st) || [];
    for (const d of dims) {
      vec[d] += CONCEPT_WEIGHT;
    }
  }
  return vec;
}

export const SEEDS = [
  {
    id: "durability",
    text: "Every batch is fsync'd in order: append the vector, fsync it, then append and fsync the committing log record, so a crash loses only the in flight write.",
    attrs: { topic: "durability" },
  },
  {
    id: "storage",
    text: "A nidus store is one append only directory: segments, a manifest, a log, and a lock file, fsync'd per batch so a crash never tears a write.",
    attrs: { topic: "storage" },
  },
  {
    id: "recovery",
    text: "A torn tail repairs itself the moment you reopen the store: every record is length prefixed and CRC checked, so a half written frame is detected and dropped.",
    attrs: { topic: "durability" },
  },
  {
    id: "search",
    text: "Search is exact brute force cosine by default, one hundred percent recall, with an optional HNSW or IVF index when you want more speed.",
    attrs: { topic: "search" },
  },
  {
    id: "distance",
    text: "Rank by cosine, dot product, or Euclidean distance. Vectors are unit normalized on insert, so a cosine score is a plain dot product at query time.",
    attrs: { topic: "search" },
  },
  {
    id: "filters",
    text: "Filter records before they score: equality, ranges, globs, sets, list containment, and fuzzy or regex text matching, composed with boolean logic.",
    attrs: { topic: "filters" },
  },
  {
    id: "quantization",
    text: "Int8 and binary quantization shrink each stored vector on disk, and a quantized two pass search keeps the ranking close to exact.",
    attrs: { topic: "quantization" },
  },
  {
    id: "ann-index",
    text: "The approximate index trades a little recall for speed: HNSW builds a navigable graph, IVF partitions into lists you probe a few of.",
    attrs: { topic: "search" },
  },
  {
    id: "memory",
    text: "The memory layer embeds natural language for you with the provider you choose, then recalls the closest remembered text by similarity.",
    attrs: { topic: "memory" },
  },
  {
    id: "fulltext",
    text: "The full text index folds and analyzes words, then ranks hits with BM25 alongside the vector score in one fused ranking.",
    attrs: { topic: "search" },
  },
  {
    id: "server",
    text: "Run nidus as an HTTP server behind a bearer token, or point any MCP client at it as agent memory over the same store.",
    attrs: { topic: "server" },
  },
  {
    id: "auth-rotation",
    text: "The HTTP server's bearer auth supports token rotation: add the new token to the list, redeploy, then remove the old token once every client has moved.",
    attrs: { topic: "server" },
  },
  {
    id: "sdks",
    text: "Reach the same server from JavaScript, Go, or Python: each SDK mirrors the HTTP surface method for method, and ships at the crate's own version.",
    attrs: { topic: "server" },
  },
  {
    id: "backends",
    text: "Move a store from a local folder to S3 or GCS by changing one URL, and add a shared Redis memory tier for a warm working set.",
    attrs: { topic: "backends" },
  },
  {
    id: "browser",
    text: "Compiled to wasm, nidus runs entirely in a browser tab and keeps its data in the Origin Private File System, with no server round trip at all.",
    attrs: { topic: "browser" },
  },
  {
    id: "build-cost",
    text: "Pure Rust core with no bundled C++ tree: the default build adds tree-sitter's small C parser for code search, and `--no-default-features` gives the storage-and-search core alone.",
    attrs: { topic: "build" },
  },
  {
    id: "footprint",
    text: "Ask the store what it costs: live rows, reclaimable dead rows, the pinned dimension, and the bytes the vector matrix occupies right now.",
    attrs: { topic: "storage" },
  },
];

// The static pre-click example. Computed against SEEDS above, never invented: this is
// the real ranking, and it shares no words with its top hit, which is the point.
export const EXAMPLE_QUERY = "how do I keep data safe if the process dies";
export const EXAMPLE_OUTPUT = [
  { id: "durability", score: 0.848 },
  { id: "recovery", score: 0.651 },
  { id: "storage", score: 0.284 },
];
