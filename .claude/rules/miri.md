---
paths:
  - "src/**/*.rs"
  - "tests/**/*.rs"
---

# Miri discipline

`just miri` runs the suite under Miri, against `--no-default-features` (D0015). Miri cannot
instrument foreign code, and `code` (default since D0015) pulls tree-sitter's C via
`wdpkr-core`; the lean lane is the only one Miri can run at all. **All of nidus's own logic**
runs under it — codecs, search kernels, filters, local file IO. Only the network paths in the
S3/GCS backends are out of reach, and even there the unit tests (presigned-URL and request
construction) are pure and DO run; only the localhost-mock round-trips are ignored. Miri runs
with `-Zmiri-disable-isolation` so file-backed tests can touch a temp dir.

**Do NOT ignore** pure-logic tests: cosine math, glob matching, filter evaluation, op-log and
value codec round-trips. Prefer testing a codec against `Vec<u8>` rather than a real file so
coverage stays Miri-clean.

**Every `#[cfg_attr(miri, ignore)]` must name which of three reasons applies**, either trailing
the attribute (`#[cfg_attr(miri, ignore)] // <why>`) or on the line directly above — the
checker reads both positions:

- **Syscalls Miri does not implement** — `File::sync_all`/`sync_data` (fsync) and friends. Keep
  these in the file-backed integration tests.
- **Runtime cost** — the interpreter is orders of magnitude slower, so an N=2000 ANN build or a
  threaded scan will not finish. Say so; do not leave it looking like reason one.
- **Float ULP** — Miri's libm differs in the last bit, so a test asserting *exact* `f32` score
  bits through a transcendental (BM25's IDF uses `ln`) fails under Miri while the ranking is
  identical. Compare ranks, or ignore with this reason named.

The comment is load-bearing, not decoration: `nidus-check laws` treats a bare ignore as
unresolved and a documented one as settled, and the reason is also the only thing standing
between a correct ignore and someone deleting it (D0008).
