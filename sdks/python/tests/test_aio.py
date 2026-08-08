"""Tests for ``AsyncNidusClient``, mirroring ``test_client`` against a mock ``httpx``.

The async client's whole job is transport: paths, bodies, pruning, decoding and error
extraction all come from ``nidus._wire``, already pinned in ``test_wire``. So what is worth
asserting here is that the *awaited* path reaches the same wire — same verbs, same URLs, same
bytes, same ``NidusError`` — because "the async client silently omits something the sync one
sends" is exactly the drift the shared ``_wire`` layer exists to prevent, and only a test can
show the clients actually use it.

``httpx.MockTransport`` is the async counterpart of the sync client's ``transport=`` callable:
each takes the natural extension point of its own HTTP stack, so neither client needs a
bespoke test double. The module skips cleanly when ``httpx`` is absent — that is the whole
point of the ``nidus[async]`` extra, and a contributor without it must still be able to run
the rest of the suite.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

import pytest

httpx = pytest.importorskip("httpx", reason="the async client needs the nidus[async] extra")

from nidus import NidusError, f, rank, v  # noqa: E402 - must follow the importorskip guard
from nidus.aio import AsyncNidusClient  # noqa: E402 - same

STATS_PAYLOAD = {
    "dimension": 3,
    "distance": "Cosine",
    "ann": None,
    "collections": ["docs"],
    "footprint": {
        "rows": 2,
        "dead_rows": 0,
        "dimension": 3,
        "vector_bytes": 24,
        "doc_count": 2,
    },
}


class MockServer:
    """A recording ``httpx.MockTransport``: canned response out, every request kept.

    ``raises`` covers the case a canned response cannot express — a transport that never
    produces a response at all, which is what ``status 0`` is the sentinel for.
    """

    def __init__(
        self,
        payload: Any = None,
        status: int = 200,
        text: str | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.text = text if text is not None else json.dumps(payload)
        self.status = status
        self.raises = raises
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        # `read()` before storing: the body is consumed here, and a test asserting on it
        # afterwards must not have to know that.
        request.read()
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        return httpx.Response(self.status, content=self.text)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    @property
    def last(self) -> httpx.Request:
        assert self.requests, "no request was made"
        return self.requests[-1]

    @property
    def json(self) -> Any:
        """The last request body, parsed back from the bytes actually sent."""
        return json.loads(self.last.content)


def client(mock: MockServer, **kwargs: Any) -> AsyncNidusClient:
    return AsyncNidusClient("http://x", transport=mock.transport(), **kwargs)


# ── Every method hits the right verb and path ────────────────────────────────────────

ENDPOINTS: list[tuple[str, Callable[[AsyncNidusClient], Any], str, str, Any]] = [
    ("health", lambda db: db.health(), "GET", "/health", "ok"),
    ("stats", lambda db: db.stats(), "GET", "/stats", STATS_PAYLOAD),
    ("collections", lambda db: db.collections(), "GET", "/collections", ["docs"]),
    ("create_collection", lambda db: db.create_collection("docs"), "POST", "/collections/docs", {}),
    ("drop_collection", lambda db: db.drop_collection("docs"), "DELETE", "/collections/docs", {}),
    ("get_meta", lambda db: db.get_meta("docs"), "GET", "/collections/docs/meta", {}),
    ("set_meta", lambda db: db.set_meta("docs", {"a": "b"}), "PUT", "/collections/docs/meta", {}),
    (
        "upsert",
        lambda db: db.upsert("docs", [{"id": "a", "vector": [1.0, 0.0, 0.0]}]),
        "POST",
        "/collections/docs/upsert",
        {"upserted": 1},
    ),
    (
        "delete",
        lambda db: db.delete("docs", ["a"]),
        "POST",
        "/collections/docs/delete",
        {"deleted": 1},
    ),
    (
        "delete_where",
        lambda db: db.delete_where("docs", [f.eq("lang", "go")]),
        "POST",
        "/collections/docs/delete",
        {"deleted": 1},
    ),
    ("records", lambda db: db.records("docs"), "GET", "/collections/docs/records", []),
    (
        "set_fts_schema",
        lambda db: db.set_fts_schema("docs", ["body"]),
        "POST",
        "/collections/docs/fts-schema",
        {"ok": True},
    ),
    ("search", lambda db: db.search(query=[1.0, 0.0, 0.0]), "POST", "/search", []),
    (
        "text_search",
        lambda db: db.text_search(field="body", query="fox"),
        "POST",
        "/text-search",
        [],
    ),
    (
        "hybrid_search",
        lambda db: db.hybrid_search(vector=[1.0], field="body", text="fox"),
        "POST",
        "/hybrid-search",
        [],
    ),
    ("list", lambda db: db.list(), "POST", "/list", []),
    ("aggregate", lambda db: db.aggregate(), "POST", "/aggregate", {"count": 0, "sums": {}}),
    (
        "remember",
        lambda db: db.remember("notes", "a", "hello"),
        "POST",
        "/collections/notes/remember",
        {"ok": True},
    ),
    ("recall", lambda db: db.recall("notes", "hello"), "POST", "/collections/notes/recall", []),
    ("flush", lambda db: db.flush(), "POST", "/flush", {"ok": True}),
    ("compact", lambda db: db.compact(), "POST", "/compact", {"ok": True}),
]


@pytest.mark.parametrize(
    ("call", "method", "path", "payload"),
    [(row[1], row[2], row[3], row[4]) for row in ENDPOINTS],
    ids=[row[0] for row in ENDPOINTS],
)
async def test_each_method_uses_the_right_verb_and_path(
    call: Callable[[AsyncNidusClient], Any], method: str, path: str, payload: Any
) -> None:
    """The awaited surface addresses exactly what the sync one does."""
    mock = MockServer(payload)
    async with client(mock) as db:
        await call(db)
    assert len(mock.requests) == 1
    assert mock.last.method == method
    assert str(mock.last.url) == f"http://x{path}"


def test_the_async_surface_matches_the_sync_one() -> None:
    """Every public sync method has an async twin — checked, not assumed.

    Cheap insurance against the two clients drifting by *omission*: a method added to one and
    forgotten on the other is invisible to every other test in both files.
    """
    from nidus import NidusClient

    def public(cls: type) -> set[str]:
        return {n for n in vars(cls) if not n.startswith("_")}

    sync_only = public(NidusClient) - public(AsyncNidusClient)
    async_only = public(AsyncNidusClient) - public(NidusClient)
    # The lifetime hooks are the one deliberate difference: `close` versus `aclose`.
    assert sync_only == {"close"}
    assert async_only == {"aclose"}


def test_the_async_methods_take_the_same_arguments() -> None:
    """Same *names* is not enough — a knob added to one client only is the same drift.

    Every optional is keyword-only on both, so the parameter list is the whole call
    contract: if ``search`` grows a ``rank_by`` here and not there, the two clients can no
    longer send the same request, which no other test in either file would notice.
    """
    from nidus import NidusClient

    shared = {n for n in vars(NidusClient) if not n.startswith("_")} & {
        n for n in vars(AsyncNidusClient) if not n.startswith("_")
    }
    for name in sorted(shared):
        sync = inspect.signature(getattr(NidusClient, name))
        asynchronous = inspect.signature(getattr(AsyncNidusClient, name))
        assert list(sync.parameters) == list(asynchronous.parameters), name


# ── Request bodies as they land on the wire ──────────────────────────────────────────


async def test_upsert_sends_normalized_attrs_and_omits_a_text_only_vector() -> None:
    mock = MockServer({"upserted": 2})
    async with client(mock) as db:
        n = await db.upsert(
            "docs",
            [
                {"id": "a", "vector": [1.0, 0.0, 0.0], "attrs": {"lang": "rust", "draft": False}},
                {"id": "b", "attrs": {"body": v.str("text only")}},
            ],
        )
    assert n == 2
    assert mock.json == {
        "records": [
            {
                "id": "a",
                "vector": [1.0, 0.0, 0.0],
                # Still `Bool`, not `Int` — the shared codec, reached by the other client.
                "attrs": {"lang": {"Str": "rust"}, "draft": {"Bool": False}},
            },
            {"id": "b", "attrs": {"body": {"Str": "text only"}}},
        ]
    }


async def test_the_argument_guards_are_on_the_async_path_too() -> None:
    """The guards live in ``_wire``, so both clients get them — assert it rather than assume.

    The failure they prevent is silent (a well-formed request the server answers wrongly),
    so an async client that skipped them would look identical to one that did not.
    """
    mock = MockServer({"deleted": 1})
    async with client(mock) as db:
        with pytest.raises(TypeError, match="did you mean"):
            await db.delete("docs", "x1")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="drop_collection"):
            await db.delete_where("docs", [])
    assert mock.requests == []


async def test_search_omits_an_unset_top_k_but_sends_an_explicit_zero() -> None:
    """The omit-vs-zero rule holds on the async path too, asserted on the request bytes."""
    mock = MockServer([])
    async with client(mock) as db:
        await db.search(query=[1.0, 0.0, 0.0])
        assert mock.json == {"query": [1.0, 0.0, 0.0], "scope": [], "filter": []}
        await db.search(query=[1.0], top_k=0, min_score=0.0)
        assert mock.json["top_k"] == 0
        assert mock.json["min_score"] == 0.0


async def test_list_omits_an_unset_limit_but_sends_an_explicit_zero() -> None:
    mock = MockServer([])
    async with client(mock) as db:
        await db.list()
        assert "limit" not in mock.json
        await db.list(limit=0, offset=0)
        assert mock.json == {"scope": [], "offset": 0, "limit": 0, "filter": []}


async def test_search_sends_snake_case_and_decodes_hit_attrs() -> None:
    mock = MockServer(
        [{"collection": "docs", "id": "a", "score": 0.9, "attrs": {"lang": {"Str": "rust"}}}]
    )
    async with client(mock) as db:
        hits = await db.search(
            query=[1.0, 0.0, 0.0], top_k=5, min_score=0.1, filter=f.and_(f.eq("lang", "rust"))
        )
    assert mock.json == {
        "query": [1.0, 0.0, 0.0],
        "scope": [],
        "top_k": 5,
        "min_score": 0.1,
        "filter": [{"Eq": ["lang", {"Str": "rust"}]}],
    }
    assert hits[0].id == "a"
    assert hits[0].attrs == {"lang": "rust"}


async def test_the_multi_clause_and_ranking_knobs_reach_the_same_wire() -> None:
    """The awaited path sends the m50 knobs byte for byte with the sync client's.

    Asserted here rather than trusted: the shared ``_wire`` layer is what makes it true, and
    a client that forgot to forward one argument would still compile and still return hits.
    """
    mock = MockServer([])
    async with client(mock) as db:
        await db.text_search(
            clauses=[{"field": "title", "query": "rust"}, {"field": "body", "query": "async"}],
            combine="Max",
            explain=True,
            highlight={"max_fragments": 2},
        )
        assert mock.json == {
            "clauses": [
                {"field": "title", "query": "rust"},
                {"field": "body", "query": "async"},
            ],
            "combine": "Max",
            "scope": [],
            "filter": [],
            "explain": True,
            "highlight": {"max_fragments": 2},
        }
        await db.search(
            query=[1.0],
            rank_by=rank.decay("ts", 1700000000000),
            limit_per={"field": "path", "max": 2},
        )
        assert mock.json["rank_by"] == {"Decay": {"field": "ts", "origin": 1700000000000}}
        assert mock.json["limit_per"] == {"field": "path", "max": 2}
        await db.list(order_by={"field": "ts"})
        assert mock.json["order_by"] == {"field": "ts"}
        with pytest.raises(ValueError, match="mutually exclusive"):
            await db.hybrid_search(
                vector=[1.0], field="body", clauses=[{"field": "t", "query": "x"}]
            )


async def test_aggregate_and_annotations_decode_on_the_async_path_too() -> None:
    """The two new response shapes, reached through the awaited client."""
    mock = MockServer({"count": 2, "sums": {"bytes": {"Int": 40}, "cost": {"Float": 1.5}}})
    async with client(mock) as db:
        out = await db.aggregate(scope=["docs"], sum=["bytes", "cost"])
    assert mock.json == {"scope": ["docs"], "filter": [], "sum": ["bytes", "cost"]}
    assert (out.count, out.sums) == (2, {"bytes": 40, "cost": 1.5})

    mock = MockServer(
        [
            {
                "collection": "docs",
                "id": "a",
                "score": 1.0,
                "attrs": {},
                "annotations": {
                    "vector": {"rank": 0, "score": 0.9},
                    "text": {"rank": 1, "score": 2.0},
                },
            }
        ]
    )
    async with client(mock) as db:
        hits = await db.hybrid_search(vector=[1.0], field="body", text="fox", explain=True)
    assert hits[0].annotations is not None
    assert hits[0].annotations.vector is not None
    assert hits[0].annotations.vector.rank == 0
    assert hits[0].annotations.text is not None
    assert hits[0].annotations.text.score == pytest.approx(2.0)


async def test_remember_and_recall_bodies() -> None:
    mock = MockServer({"ok": True})
    async with client(mock) as db:
        await db.remember("notes", "a", "the quick brown fox", attrs={"tag": "x"})
        assert mock.json == {
            "id": "a",
            "text": "the quick brown fox",
            "attrs": {"tag": {"Str": "x"}},
        }
        assert "mode" not in mock.json
    mock = MockServer([])
    async with client(mock) as db:
        await db.recall("notes", "hello")
        assert mock.json == {"query": "hello", "filter": []}


async def test_a_bodyless_write_sends_an_empty_json_object() -> None:
    mock = MockServer({"ok": True})
    async with client(mock) as db:
        await db.flush()
    assert mock.last.content == b"{}"
    assert mock.last.headers["content-type"] == "application/json"


async def test_a_bodyless_read_sends_no_body_and_no_content_type() -> None:
    mock = MockServer(STATS_PAYLOAD)
    async with client(mock) as db:
        await db.stats()
    assert mock.last.content == b""
    assert "content-type" not in mock.last.headers


# ── Responses ────────────────────────────────────────────────────────────────────────


async def test_stats_decodes_a_null_ann_to_none() -> None:
    mock = MockServer(STATS_PAYLOAD)
    async with client(mock) as db:
        stats = await db.stats()
    assert stats.ann is None
    assert stats.dimension == 3


async def test_stats_decodes_an_hnsw_ann_with_the_ivf_knobs_left_none() -> None:
    mock = MockServer(
        {
            **STATS_PAYLOAD,
            "ann": {
                "kind": "Hnsw",
                "overscan": 2,
                "seed": 42,
                "m": 16,
                "ef_construction": 200,
                "ef_search": 64,
            },
        }
    )
    async with client(mock) as db:
        ann = (await db.stats()).ann
    assert ann is not None
    assert (ann.kind, ann.m, ann.ef_search) == ("Hnsw", 16, 64)
    assert ann.n_lists is None
    assert ann.n_probe is None


async def test_records_keeps_an_absent_vector_as_none() -> None:
    mock = MockServer([{"id": "a", "attrs": {"body": {"Str": "text only"}}}])
    async with client(mock) as db:
        records = await db.records("docs")
    assert records[0].vector is None
    assert records[0].attrs == {"body": "text only"}


@pytest.mark.parametrize("status", [200, 204, 299])
async def test_health_is_true_for_any_2xx(status: int) -> None:
    mock = MockServer(None, status=status, text="ok")
    async with client(mock) as db:
        assert await db.health() is True


@pytest.mark.parametrize("status", [404, 503])
async def test_health_is_false_rather_than_raising(status: int) -> None:
    mock = MockServer({"error": "unhealthy"}, status=status)
    async with client(mock) as db:
        assert await db.health() is False


# ── Errors ───────────────────────────────────────────────────────────────────────────


async def test_a_non_2xx_raises_nidus_error_with_the_servers_message_and_status() -> None:
    mock = MockServer({"error": "store is locked: /tmp/s/lock"}, status=409)
    async with client(mock) as db:
        with pytest.raises(NidusError) as caught:
            await db.flush()
    assert caught.value.status == 409
    assert caught.value.message == "store is locked: /tmp/s/lock"
    assert caught.value.is_locked


async def test_a_non_json_error_body_is_reported_verbatim() -> None:
    mock = MockServer(None, status=502, text="<html>502 Bad Gateway</html>")
    async with client(mock) as db:
        with pytest.raises(NidusError) as caught:
            await db.stats()
    assert caught.value.message == "<html>502 Bad Gateway</html>"


async def test_an_empty_error_body_falls_back_to_the_status() -> None:
    mock = MockServer(None, status=404, text="")
    async with client(mock) as db:
        with pytest.raises(NidusError) as caught:
            await db.get_meta("nope")
    assert caught.value.message == "HTTP 404"


async def test_a_transport_failure_is_status_zero() -> None:
    """``httpx.HTTPError`` covers the whole transport family; all of it means "no answer"."""
    mock = MockServer(raises=httpx.ConnectError("connection refused"))
    async with client(mock) as db:
        with pytest.raises(NidusError) as caught:
            await db.stats()
        assert caught.value.status == 0
        assert caught.value.is_transport_error
        assert "/stats" in caught.value.message
        # And `health` swallows it rather than propagating.
        assert await db.health() is False


# ── Configuration and lifetime ───────────────────────────────────────────────────────


@pytest.mark.parametrize("base", ["http://x/", "http://x//", "http://x"])
async def test_a_trailing_slash_in_the_base_url_is_stripped(base: str) -> None:
    mock = MockServer(["docs"])
    async with AsyncNidusClient(base, transport=mock.transport()) as db:
        await db.collections()
    assert str(mock.last.url) == "http://x/collections"


@pytest.mark.parametrize(
    ("name", "escaped"),
    [("a/b c", "a%2Fb%20c"), ("with space", "with%20space"), ("q?x#y", "q%3Fx%23y")],
)
async def test_a_collection_name_is_percent_escaped_into_one_path_segment(
    name: str, escaped: str
) -> None:
    """``httpx`` must not re-interpret the escapes we produced, or the route changes."""
    mock = MockServer({"upserted": 0})
    async with client(mock) as db:
        await db.upsert(name, [])
    assert str(mock.last.url) == f"http://x/collections/{escaped}/upsert"


async def test_a_bearer_token_and_extra_headers_are_sent() -> None:
    mock = MockServer([])
    async with client(mock, token="sekret", headers={"x-trace": "1"}) as db:
        await db.list()
    assert mock.last.headers["authorization"] == "Bearer sekret"
    assert mock.last.headers["x-trace"] == "1"


async def test_aclose_is_idempotent() -> None:
    """``async with`` closes the pool; calling it again must not be an error."""
    mock = MockServer(["docs"])
    db = AsyncNidusClient("http://x", transport=mock.transport())
    assert await db.collections() == ["docs"]
    await db.aclose()
    await db.aclose()


async def test_an_empty_base_url_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="base_url"):
        AsyncNidusClient("")
