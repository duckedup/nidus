# D0008 — A `#[cfg_attr(miri, ignore)]` must name which of three reasons applies

**Status:** accepted
**Rule:** Every Miri ignore carries its reason, either trailing the attribute or on the line directly above. There are only three: unimplemented syscalls, runtime cost, float ULP.

## Why

All of nidus's own logic runs under Miri — codecs, search kernels, filters, local file IO.
Only the network paths in the S3/GCS backends are outside its reach, and even there the unit
tests (presigned-URL and request construction) are pure and do run under it; the
localhost-mock round-trips are the ignored ones. Miri runs with `-Zmiri-disable-isolation`
so file-backed tests can touch a temp dir.

The three legitimate reasons:

- **Syscalls Miri does not implement** — `File::sync_all`/`sync_data` and friends. Keep
  these in the file-backed integration tests.
- **Runtime cost** — the interpreter is orders of magnitude slower, so an N=2000 ANN build
  or a threaded scan will not finish. Say so, rather than leaving it looking like reason one.
- **Float ULP** — Miri's libm differs from the host in the last bit, so a test asserting
  exact `f32` score bits through a transcendental (BM25's IDF uses `ln`) fails under Miri
  while the ranking is identical. Compare ranks, or ignore with this reason named.

The comment is load-bearing, not decoration, and for two independent reasons. `nidus-check
laws` treats a bare ignore as unresolved and a documented one as settled, so an undocumented
ignore nags forever and a documented one stops: that is the difference between an actionable
check and ambient noise. It also cuts the other way — the reason is the only thing standing
between a correct ignore and someone deleting it, which is exactly what happened to two
float-ULP tests here when the checker read only one of the two comment positions.

Do NOT ignore pure-logic tests (cosine math, glob matching, filter evaluation, op-log and
value codec round-trips). Prefer testing a codec against `Vec<u8>` rather than a real file so
coverage stays Miri-clean.

## Evidence

- `lib/laws.mjs` — `miriIgnore` reads both comment positions.
- The two float-ULP tests deleted when it read only one.
