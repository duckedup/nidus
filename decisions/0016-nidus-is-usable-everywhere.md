# D0016 — Positioning describes shipped modes, not a size ceiling

**Status:** accepted
**Rule:** Describe nidus by what it is and the modes it ships, never by a deployment size or
an audience: a pure-Rust vector store with full-text search that runs anywhere Rust runs, in
process as a library, behind `nidus serve` over HTTP, as an MCP server, or in a browser on
wasm, with its bytes on local disk or in object storage and an optional shared memory tier.
Describe what ships today; do not pin the framing to "an embeddable library" and do not
promise future modes.

## Why

The framing predates §13 and §14. The front page said "a vector store for development and
small-scale use", exact search "fast at the target scale (≤ a few million vectors, comfortably
in RAM)", while `src/backend/s3.rs`, `gcs.rs` and `redis.rs` were shipping and `src/manifest/`
was already versioning segments for a store whose bytes live in an object store. A reader
deciding whether nidus fits their problem was answering from the prose, not the architecture.

**What stays true.** Exact brute-force scan is still the default and its cost still scales with
rows scanned. ANN, quantization, mmap and segments are still opt-in. Single-writer is still the
model (§13.7, §14.6) and this record does not change it. Removing a size ceiling from the prose
is not a claim that any size has been measured.

**The costs, stated plainly and without softening:**
- "Usable everywhere" is broader than anything currently measured: the object-storage
  cold-start and retrieval-quality benchmarks that would back it are nidus-yq9p.4 and
  nidus-yq9p.5, both still open, so this record widens the claim ahead of the evidence and
  says so.
- A broader positioning invites comparison with managed vector databases on axes
  (replication, multi-writer, a query protocol) that SPEC §2 still lists as non-goals.
- The old framing was a cheap way to deflect feature requests; without it, each one now needs
  the §2 non-goals argument made explicitly.

## Evidence

- `nidus-yq9p.1`, the `nidus-yq9p` epic, SPEC §13 and §14.
- D0005 — the dependency bar is build-and-ship speed, which is the thing that actually
  constrains nidus, not store size.
- D0012 — a feature ships whole, in one PR.
