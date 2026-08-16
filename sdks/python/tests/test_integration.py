"""End-to-end against a real ``nidus serve``, driven entirely through the SDK.

Mirrors ``sdks/js/test/integration.test.ts`` step for step, because the point of these two
files is that the SDKs are demonstrably interchangeable: same flow, same expectations, same
server. What this tier catches that the stub tier structurally cannot is everything on the
far side of the socket — that the paths really route, that the pruned bodies really hit the
server's ``#[serde(default)]``, that ``snake_case`` really deserializes, and that a hit's
score survives the JSON round trip.

The binary comes from ``$NIDUS_BIN``, else ``target/release/nidus`` at the repo root (build it
with ``just build-cli``). **When it is absent the whole module skips**, so a contributor
without the Rust toolchain still gets the unit tier — the same bargain the JS suite strikes,
and the reason CI can run the unit tests with no Rust step at all.

Two implementation notes worth stating, since both are the second thing you would try:

* The port is ``0``, and the *bound* address is scraped from the server's own startup line.
  A hardcoded port collides with a developer's other nidus, and a pre-chosen "free" port is a
  race; letting the kernel choose and then reading back what it chose is neither.
* The child's output goes to a **file**, not a pipe. A pipe holds 64 KiB and then blocks the
  writer, so a server that logs per request would eventually wedge on its own stderr — and
  that would look like a hung test, not a full pipe. The file doubles as the transcript
  attached to a startup failure.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nidus import NidusClient, NidusError, f, rank, v

# tests → python → sdks → repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
BINARY = Path(os.environ.get("NIDUS_BIN") or REPO_ROOT / "target" / "release" / "nidus")

# Generous: a debug-built binary on a loaded machine is slow to bind, and a flaky skip-or-fail
# boundary is worse than a slow one. Nothing here is a timing assertion.
STARTUP_TIMEOUT = 20.0

pytestmark = pytest.mark.skipif(
    not (BINARY.is_file() and os.access(BINARY, os.X_OK)),
    reason=f"no nidus binary at {BINARY} (set $NIDUS_BIN, or run `just build-cli`)",
)


@pytest.fixture()
def server(tmp_path: Path) -> Iterator[str]:
    """A real ``nidus serve`` on an ephemeral port over a fresh store; yields its base URL.

    Function-scoped, so each test gets a store nothing else has touched — the alternative is
    tests that pass only in the order they happen to be written.
    """
    log = tmp_path / "server.log"
    store = tmp_path / "store"
    with log.open("wb") as sink:
        # stdout folded into the same sink: the startup line is on stderr today, and a test
        # that breaks if that changes would be testing the logger, not the SDK.
        child = subprocess.Popen(
            [
                str(BINARY),
                "serve",
                "--dir",
                str(store),
                "--dim",
                "3",
                "--addr",
                "127.0.0.1:0",
            ],
            stdout=sink,
            stderr=sink,
        )
    try:
        base_url = _await_base_url(child, log)
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while not _ready(base_url):
            if time.monotonic() > deadline:
                pytest.fail(f"nidus serve never became ready\n{_transcript(log)}")
            time.sleep(0.05)
        yield base_url
    finally:
        # SIGTERM rather than SIGKILL: the graceful path flushes and releases the writer lock,
        # which is also what the server's own shutdown handling is for.
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - only if shutdown hangs
            child.kill()
            child.wait()


def _ready(base_url: str) -> bool:
    """``/ready``, not ``/health`` (#121): health is liveness and answers before the
    store finishes opening, so a health gate can hand tests a server that still 503s."""
    try:
        with urllib.request.urlopen(f"{base_url}/ready", timeout=1.0) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, OSError):
        return False


def _await_base_url(child: subprocess.Popen[bytes], log: Path) -> str:
    """Read the bound address out of ``nidus serving on http://127.0.0.1:<port> …``.

    Distinguishes "the child died" from "the child is slow" so a failure names the real cause
    — bad flags and a locked store both look like a timeout otherwise.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        found = re.search(r"http://\d+\.\d+\.\d+\.\d+:\d+", _transcript(log))
        if found:
            return found.group(0)
        if child.poll() is not None:
            pytest.fail(
                f"nidus serve exited ({child.returncode}) before reporting an address\n"
                f"{_transcript(log)}"
            )
        time.sleep(0.05)
    pytest.fail(f"nidus serve printed no address within {STARTUP_TIMEOUT}s\n{_transcript(log)}")


def _transcript(log: Path) -> str:
    """The server's own output, attached to any startup failure so it is diagnosable."""
    try:
        return log.read_text(errors="replace")
    except OSError:  # pragma: no cover - the file exists unless the temp dir vanished
        return "<no server output>"


def test_full_lifecycle(server: str) -> None:
    """create → upsert → search → filter → text search → hybrid search → delete → stats.

    One test rather than several, because the steps are one story: each assertion is about the
    state the previous step left behind, and splitting them would either hide that dependency
    or pay for a fresh server per step.
    """
    with NidusClient(server, timeout=10.0) as db:
        # ── create ───────────────────────────────────────────────────────────────────
        db.create_collection("docs")
        assert "docs" in db.collections()

        # ── upsert ───────────────────────────────────────────────────────────────────
        assert (
            db.upsert(
                "docs",
                [
                    {"id": "a", "vector": [1.0, 0.0, 0.0], "attrs": {"lang": "rust", "year": 2024}},
                    {"id": "b", "vector": [0.0, 1.0, 0.0], "attrs": {"lang": "go", "year": 2020}},
                ],
            )
            == 2
        )

        # ── search ───────────────────────────────────────────────────────────────────
        hits = db.search(query=[1.0, 0.0, 0.0], top_k=1)
        assert [h.id for h in hits] == ["a"]
        assert hits[0].collection == "docs"
        # Attrs come back decoded, and the int survived the JSON round trip as an int.
        assert hits[0].attrs == {"lang": "rust", "year": 2024}
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

        # An unset top_k must reach the server's default rather than being sent as 0 — if the
        # pruning were wrong this returns nothing at all, which is the bug it guards.
        assert len(db.search(query=[1.0, 0.0, 0.0])) == 2

        # ── filter ───────────────────────────────────────────────────────────────────
        assert [h.id for h in db.search(query=[1.0, 0.0, 0.0], filter=[f.eq("lang", "go")])] == [
            "b"
        ]
        assert [h.id for h in db.list(scope=["docs"], filter=f.and_(f.ge("year", 2024)))] == ["a"]
        # `Glob`'s bare-string operand, exercised against the real matcher.
        assert [h.id for h in db.list(scope=["docs"], filter=[f.glob("lang", "ru*")])] == ["a"]
        # And the negative/range forms, which only match a *present* key.
        assert [h.id for h in db.list(scope=["docs"], filter=[f.ne("lang", "rust")])] == ["b"]
        assert db.list(scope=["docs"], filter=[f.eq("missing", "x")]) == []

        # ── filter index ─────────────────────────────────────────────────────────────
        # Declaring one must change nothing a caller can observe except speed, so the
        # assertion is that the same predicate answers the same way with it in place.
        before = [h.id for h in db.list(scope=["docs"], filter=[f.contains_all_tokens("lang", "rust")])]
        db.set_filter_index("docs", ["lang"])
        after = [h.id for h in db.list(scope=["docs"], filter=[f.contains_all_tokens("lang", "rust")])]
        assert before == after
        assert db.stats().footprint.filter_index_bytes > 0
        # Per-field structures and the empty-list drop both reach the server.
        db.set_filter_index("docs", [{"field": "lang", "trigrams": False}])
        db.set_filter_index("docs", [])
        assert db.stats().footprint.filter_index_bytes == 0

        # ── text search ──────────────────────────────────────────────────────────────
        db.set_fts_schema("notes", ["body"])
        assert (
            db.upsert(
                "notes",
                [
                    {
                        "id": "x",
                        "vector": [1.0, 0.0, 0.0],
                        "attrs": {"body": v.str("the quick brown fox"), "kind": "a"},
                    },
                    # No vector: a text-only doc, which must round-trip as `None`.
                    {"id": "y", "attrs": {"body": v.str("foxes are running quickly"), "kind": "b"}},
                ],
            )
            == 2
        )
        text = db.text_search(scope=["notes"], field="body", query="run", top_k=5)
        assert [h.id for h in text] == ["y"]

        # ── hybrid search ────────────────────────────────────────────────────────────
        hybrid = db.hybrid_search(
            scope=["notes"], vector=[1.0, 0.0, 0.0], field="body", text="fox", top_k=5
        )
        assert {h.id for h in hybrid} == {"x", "y"}

        # A text-only doc has no vector on the wire, and that absence must survive.
        records = {r.id: r for r in db.records("notes")}
        assert records["x"].vector == pytest.approx([1.0, 0.0, 0.0])
        assert records["y"].vector is None
        assert records["y"].attrs == {"body": "foxes are running quickly", "kind": "b"}

        # ── metadata ─────────────────────────────────────────────────────────────────
        db.set_meta("docs", {"owner": "austin"})
        assert db.get_meta("docs") == {"owner": "austin"}

        # ── delete ───────────────────────────────────────────────────────────────────
        assert db.delete("docs", ["b"]) == 1
        assert [r.id for r in db.records("docs")] == ["a"]
        assert db.delete_where("notes", f.and_(f.eq("kind", "b"))) == 1
        assert [r.id for r in db.records("notes")] == ["x"]

        # ── stats ────────────────────────────────────────────────────────────────────
        db.flush()
        stats = db.stats()
        assert stats.dimension == 3
        # The store does exact brute-force search unless an ANN index was configured.
        assert stats.ann is None
        assert set(stats.collections) >= {"docs", "notes"}
        assert stats.footprint.doc_count == 2
        assert stats.footprint.dimension == 3


def test_a_server_error_arrives_as_a_nidus_error(server: str) -> None:
    """The error path over a real socket: the server's own message, with its status."""
    with NidusClient(server, timeout=10.0) as db:
        with pytest.raises(NidusError) as caught:
            # Four components against a dim-3 store: the server rejects it with a 400 and an
            # explanatory message, which is what the SDK must surface rather than flatten.
            db.upsert("docs", [{"id": "bad", "vector": [1.0, 0.0, 0.0, 0.0]}])
        assert caught.value.status == 400
        assert caught.value.is_bad_request
        assert caught.value.message


def test_remember_with_ttl_and_dedupe_fails_visibly_without_an_embedder(server: str) -> None:
    """404 when the binary lacks the ``memory`` feature (what ``just build-cli`` builds, so
    the usual case here), 400 when the routes exist but no ``--embed-provider`` was given.
    Either way the two knobs reach the wire and the call fails with a status, not silently.
    """
    with NidusClient(server, timeout=10.0) as db:
        with pytest.raises(NidusError) as caught:
            db.remember(
                "notes", "a", "the quick brown fox", ttl_seconds=3600, dedupe_threshold=0.95
            )
        assert caught.value.status in (400, 404)


def test_the_guarded_slips_never_reach_the_server(server: str) -> None:
    """The data-loss cases, against the real server that used to accept them.

    Each of these was confirmed end-to-end before the guards existed, every one a 200: a
    bare id string deleted the record named by one of its characters and returned a
    reassuring count; a bare scope string searched five collections that do not exist and
    returned "no matches"; an empty ``delete_where`` filter emptied the collection. The
    assertion that matters is the one after each raise — the store is untouched.
    """
    with NidusClient(server, timeout=10.0) as db:
        db.create_collection("docs")
        db.upsert(
            "docs",
            [
                {"id": "x1", "vector": [1.0, 0.0, 0.0], "attrs": {"lang": "rust"}},
                {"id": "1", "vector": [0.0, 1.0, 0.0], "attrs": {"lang": "go"}},
            ],
        )

        with pytest.raises(TypeError):
            db.delete("docs", "x1")  # type: ignore[arg-type]
        assert sorted(r.id for r in db.records("docs")) == ["1", "x1"]

        with pytest.raises(TypeError):
            db.search(query=[1.0, 0.0, 0.0], scope="docs")  # type: ignore[arg-type]
        # The list form is what was meant, and it does find something.
        assert db.search(query=[1.0, 0.0, 0.0], scope=["docs"], top_k=1)[0].id == "x1"

        with pytest.raises(ValueError, match="drop_collection"):
            db.delete_where("docs", [])
        assert len(db.records("docs")) == 2

        # A non-empty filter still deletes exactly what it matches.
        assert db.delete_where("docs", [f.eq("lang", "go")]) == 1
        assert [r.id for r in db.records("docs")] == ["x1"]


def test_the_ranking_and_annotation_surface(server: str) -> None:
    """Multi-clause text, annotations, the text predicates, ranking, and ``/aggregate``.

    Every one of these is a *new key on an existing body*, and serde ignores what it does not
    recognise — so a misspelled ``limit_per`` returns a perfectly good unfiltered ranking, and
    a clause list under the wrong key returns "no matches". Only a real server tells those
    apart from working, which is why this tier exists at all.
    """
    now = datetime.now(timezone.utc)
    with NidusClient(server, timeout=10.0) as db:
        db.set_fts_schema("posts", ["title", "body"])
        # Same vector on every doc, so the base ranking is a tie and each knob below is the
        # only thing deciding the order it produces.
        assert (
            db.upsert(
                "posts",
                [
                    {
                        "id": "a",
                        "vector": [1.0, 0.0, 0.0],
                        "attrs": {
                            "title": v.str("rust in anger"),
                            "body": v.str("we were running an async runtime"),
                            "path": "src/a.rs",
                            "bytes": 100,
                            "ts": v.datetime(now),
                        },
                    },
                    {
                        "id": "b",
                        "vector": [1.0, 0.0, 0.0],
                        "attrs": {
                            "title": v.str("rust revisited"),
                            "body": v.str("async runtime internals"),
                            "path": "src/a.rs",
                            "bytes": 250,
                            "ts": v.datetime(now - timedelta(days=30)),
                        },
                    },
                    {
                        "id": "c",
                        "vector": [1.0, 0.0, 0.0],
                        "attrs": {
                            "title": v.str("go concurrency"),
                            "body": v.str("goroutines and channels"),
                            "path": "src/c.rs",
                            "bytes": 40,
                            "ts": v.datetime(now),
                        },
                    },
                ],
            )
            == 3
        )

        # ── multi-clause text, explained and highlighted ──────────────────────────────
        hits = db.text_search(
            scope=["posts"],
            clauses=[{"field": "title", "query": "rust"}, {"field": "body", "query": "run"}],
            combine="Sum",
            explain=True,
            highlight=True,
            top_k=5,
        )
        assert {h.id for h in hits} == {"a", "b"}
        # "Sum" rewards the doc matching in both fields, so it must lead.
        assert hits[0].id == "a"
        annotations = hits[0].annotations
        assert annotations is not None
        assert {c.field for c in annotations.clauses} == {"title", "body"}
        fragment = annotations.highlights[0].fragments[0]
        # The span is a byte range into the fragment's own text, pointing at the *surface*
        # form — "running" for the stemmed query "run".
        start, end = fragment.spans[0]
        assert fragment.text.encode()[start:end] in {b"rust", b"running"}
        # Unannotated by default: the same query without the two knobs carries nothing.
        assert db.text_search(scope=["posts"], field="title", query="rust")[0].annotations is None

        # ── the text predicates, against the real matchers ────────────────────────────
        def ids(*filter_: object) -> list[str]:
            return sorted(h.id for h in db.list(scope=["posts"], filter=list(filter_)))

        assert ids(f.regex("path", "src/.*[.]rs")) == ["a", "b", "c"]
        assert ids(f.regex("path", "src/a[.]rs")) == ["a", "b"]
        assert ids(f.fuzzy("title", "go concurrancy", 1)) == ["c"]
        assert ids(f.contains_token_sequence("body", "async runtime")) == ["a", "b"]
        assert ids(f.contains_all_tokens("body", "runtime async")) == ["a", "b"]
        assert ids(f.contains_any_token("body", "goroutines running")) == ["a", "c"]

        # ── ordering, capping, and decay ──────────────────────────────────────────────
        assert [h.id for h in db.list(scope=["posts"], order_by={"field": "bytes"})] == [
            "c",
            "a",
            "b",
        ]
        assert [
            h.id for h in db.list(scope=["posts"], order_by={"field": "bytes", "descending": True})
        ] == ["b", "a", "c"]
        # At most one hit per path collapses a and b, which share one.
        capped = db.search(
            query=[1.0, 0.0, 0.0], scope=["posts"], limit_per={"field": "path", "max": 1}
        )
        assert len(capped) == 2
        assert {h.id for h in capped} <= {"a", "b", "c"}
        # Decay breaks the tie: b is a month old and gives up a full point of score.
        aged = db.search(
            query=[1.0, 0.0, 0.0],
            scope=["posts"],
            rank_by=rank.decay("ts", now, scale=timedelta(days=7), lambda_=1.0),
        )
        assert [h.id for h in aged][-1] == "b"

        # ── aggregate ─────────────────────────────────────────────────────────────────
        totals = db.aggregate(scope=["posts"], sum=["bytes"])
        assert totals.count == 3
        assert totals.sums == {"bytes": 390}
        assert db.aggregate(scope=["posts"], filter=[f.gt("bytes", 50)]).count == 2
        # A field nothing carries still answers, and the count alone needs no `sum` at all.
        assert db.aggregate(scope=["posts"], sum=["absent"]).sums == {"absent": 0}


def test_ready_cluster_and_refresh_against_a_real_server(server: str) -> None:
    """Shape, not specific values: a standalone server is ready, decodes, and adopts nothing."""
    with NidusClient(server, timeout=10.0) as db:
        readiness = db.ready()
        assert readiness.ready is True
        assert readiness.role

        status = db.cluster()
        assert status.role
        assert isinstance(status.cluster, bool)
        assert isinstance(status.holds_writer_handle, bool)
        assert isinstance(status.fenced, bool)
        assert isinstance(status.commit_version, int)
        assert isinstance(status.staleness_secs, int)

        # A standalone instance has no writer lease to adopt from.
        assert isinstance(db.refresh(), bool)


def test_a_collection_name_with_a_slash_and_a_space_round_trips(server: str) -> None:
    """Path escaping, proven against the real router rather than against a string assertion."""
    name = "a/b c"
    with NidusClient(server, timeout=10.0) as db:
        db.create_collection(name)
        assert name in db.collections()
        assert db.upsert(name, [{"id": "1", "vector": [0.0, 0.0, 1.0], "attrs": {"k": "v"}}]) == 1
        assert [h.id for h in db.list(scope=[name])] == ["1"]
        db.set_meta(name, {"note": "escaped"})
        assert db.get_meta(name) == {"note": "escaped"}
        db.drop_collection(name)
        assert name not in db.collections()


async def test_the_async_client_drives_the_same_server(server: str) -> None:
    """The async twin against the real thing, so ``httpx``'s own URL handling is covered too.

    Deliberately short: the full flow is already asserted above and the two clients share
    ``_wire``, so what is left to prove here is that the ``httpx`` transport reaches the same
    endpoints — including a percent-escaped collection name, which ``httpx`` normalizes itself
    and could plausibly re-encode.
    """
    pytest.importorskip("httpx", reason="the async client needs the nidus[async] extra")
    from nidus.aio import AsyncNidusClient

    async with AsyncNidusClient(server, timeout=10.0) as db:
        assert await db.health() is True
        await db.create_collection("async docs")
        assert (
            await db.upsert(
                "async docs",
                [
                    {"id": "a", "vector": [1.0, 0.0, 0.0], "attrs": {"lang": "rust"}},
                    {"id": "b", "vector": [0.0, 1.0, 0.0], "attrs": {"lang": "go"}},
                ],
            )
            == 2
        )
        hits = await db.search(query=[1.0, 0.0, 0.0], scope=["async docs"], top_k=1)
        assert [h.id for h in hits] == ["a"]
        assert hits[0].attrs == {"lang": "rust"}
        assert [h.id for h in await db.list(scope=["async docs"], filter=[f.eq("lang", "go")])] == [
            "b"
        ]
        assert await db.delete("async docs", ["b"]) == 1
        stats = await db.stats()
        assert stats.dimension == 3
        assert stats.footprint.doc_count == 1
        assert (await db.ready()).ready is True
        assert (await db.cluster()).role
        assert isinstance(await db.refresh(), bool)
