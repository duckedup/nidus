"""Tests for the synchronous ``NidusClient``, driven through the ``transport=`` seam.

``transport=`` exists so a caller can plug in a pooled HTTP stack (``urllib`` opens a fresh
connection per request). Injecting a stub here is the *other* thing it buys: the full
endpoint surface, every request body, every header, and every error path get exercised with
no server, no socket, and no port. That is why these assertions look at the raw bytes the
transport received — the claim is about what goes on the wire, and a Python object one layer
up can be right while the JSON is wrong.

What the stub deliberately does **not** cover is a transport that fails to produce a
response at all: the ``status 0`` sentinel comes from ``urllib`` raising, so that one test
points the real client at a genuinely closed port. Faking it would test the fake.

The last two tests here are about install shapes rather than requests: ``import nidus`` must
work with ``httpx`` absent (the sync client is standard-library only), and ``import
nidus.aio`` without ``httpx`` must fail with an ``ImportError`` naming ``nidus[async]``
rather than an opaque ``ModuleNotFoundError``. Both simulate the missing module through a
``sys.meta_path`` finder, because uninstalling ``httpx`` to test that would also disable
``test_aio``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import json
import socket
import sys
from collections.abc import Iterator, Sequence
from datetime import timedelta
from typing import Any, Callable, NamedTuple

import pytest

from nidus import (
    ClusterStatus,
    NidusClient,
    NidusError,
    Readiness,
    RememberResult,
    StoreVersions,
    f,
    rank,
    v,
)

# Canned response payloads, one per shape the client has to decode. Kept beside the
# endpoint table below so a row and its response read together.
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

CLUSTER_PAYLOAD = {
    "role": "Leader",
    "cluster": True,
    "holds_writer_handle": True,
    "fenced": False,
    "lease_owner": "node-a",
    "commit_version": 42,
    "staleness_secs": 0,
    "max_staleness_secs": 30,
}

VERSIONS_PAYLOAD = {
    "commit_version": 9,
    "oldest_readable": 3,
    "pinned": 5,
    "readable": [3, 4, 5, 6, 7, 8, 9],
}


class Call(NamedTuple):
    """One request as the transport saw it — raw, before any decoding."""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None
    timeout: float | None

    @property
    def json(self) -> Any:
        """The request body parsed back from the bytes actually sent."""
        assert self.body is not None, "expected a request body"
        return json.loads(self.body)


class StubTransport:
    """A recording ``Transport``: canned ``(status, text)`` out, every call kept.

    Matches the real transport contract — a non-2xx is *returned*, never raised, since only
    a failure to get a response at all is exceptional.
    """

    def __init__(self, payload: Any = None, status: int = 200, text: str | None = None) -> None:
        # `text` wins when given, so a test can hand over a body that is not JSON at all.
        self.text = text if text is not None else json.dumps(payload)
        self.status = status
        self.calls: list[Call] = []
        self.closed = 0

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float | None,
    ) -> tuple[int, str]:
        self.calls.append(Call(method, url, dict(headers), body, timeout))
        return self.status, self.text

    def close(self) -> None:
        """The optional hook ``NidusClient.close`` looks for on a pooled transport."""
        self.closed += 1

    @property
    def last(self) -> Call:
        assert self.calls, "no request was made"
        return self.calls[-1]


def client(transport: StubTransport, **kwargs: Any) -> NidusClient:
    """A client wired to a stub, at a base URL short enough to assert on whole."""
    return NidusClient("http://x", transport=transport, **kwargs)


# ── Every method hits the right verb and path ────────────────────────────────────────
#
# Parametrized rather than one test per endpoint: the *interesting* content of each row is
# three strings, and a table makes a missing or misrouted method obvious at a glance.

ENDPOINTS: list[tuple[str, Callable[[NidusClient], Any], str, str, Any]] = [
    ("health", lambda db: db.health(), "GET", "/health", "ok"),
    (
        "ready",
        lambda db: db.ready(),
        "GET",
        "/ready",
        {"ready": True, "role": "Leader", "staleness_secs": 0},
    ),
    ("cluster", lambda db: db.cluster(), "GET", "/cluster", CLUSTER_PAYLOAD),
    ("versions", lambda db: db.versions(), "GET", "/versions", VERSIONS_PAYLOAD),
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
        "search_similar",
        lambda db: db.search_similar(collection="notes", id="a"),
        "POST",
        "/search/similar",
        [],
    ),
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
    ("refresh", lambda db: db.refresh(), "POST", "/refresh", {"adopted": True}),
]


@pytest.mark.parametrize(
    ("call", "method", "path", "payload"),
    [(row[1], row[2], row[3], row[4]) for row in ENDPOINTS],
    ids=[row[0] for row in ENDPOINTS],
)
def test_each_method_uses_the_right_verb_and_path(
    call: Callable[[NidusClient], Any], method: str, path: str, payload: Any
) -> None:
    """One request, the documented verb, the documented path."""
    stub = StubTransport(payload)
    call(client(stub))
    assert len(stub.calls) == 1
    assert stub.last.method == method
    assert stub.last.url == f"http://x{path}"


@pytest.mark.parametrize(
    ("send", "payload"),
    [
        (lambda db: db.stats(), STATS_PAYLOAD),
        (lambda db: db.collections(), ["docs"]),
        (lambda db: db.records("docs"), []),
        (lambda db: db.get_meta("docs"), {}),
        (lambda db: db.drop_collection("docs"), {"dropped": "docs"}),
    ],
    ids=["stats", "collections", "records", "get_meta", "drop_collection"],
)
def test_a_bodyless_request_sends_no_body_and_no_content_type(
    send: Callable[[NidusClient], Any], payload: Any
) -> None:
    """``body=None`` means *no body*, not a JSON ``null`` — so there is nothing to describe."""
    stub = StubTransport(payload)
    send(client(stub))
    assert stub.last.body is None
    assert "content-type" not in stub.last.headers


@pytest.mark.parametrize(
    "send",
    [
        lambda db: db.create_collection("docs"),
        lambda db: db.flush(),
        lambda db: db.compact(),
    ],
    ids=["create_collection", "flush", "compact"],
)
def test_a_bodyless_write_sends_an_empty_json_object(send: Callable[[NidusClient], Any]) -> None:
    """``{}`` rather than nothing: the handlers extract a body, so one has to be there."""
    stub = StubTransport({"ok": True})
    send(client(stub))
    assert stub.last.body == b"{}"
    assert stub.last.headers["content-type"] == "application/json"


# ── Request bodies as they land on the wire ──────────────────────────────────────────


def test_upsert_sends_normalized_attrs_and_omits_a_text_only_vector() -> None:
    """Mirrors the JS SDK's equivalent assertion, byte for byte."""
    stub = StubTransport({"upserted": 2})
    n = client(stub).upsert(
        "docs",
        [
            {"id": "a", "vector": [1.0, 0.0, 0.0], "attrs": {"lang": "rust", "year": 2024}},
            {"id": "b", "attrs": {"body": v.str("text only")}},
        ],
    )
    assert n == 2
    assert stub.last.url == "http://x/collections/docs/upsert"
    assert stub.last.json == {
        "records": [
            {
                "id": "a",
                "vector": [1.0, 0.0, 0.0],
                "attrs": {"lang": {"Str": "rust"}, "year": {"Int": 2024}},
            },
            {"id": "b", "attrs": {"body": {"Str": "text only"}}},
        ]
    }


def test_a_boolean_attr_reaches_the_wire_as_bool_not_int() -> None:
    """The ``bool``-is-an-``int`` trap, asserted on the JSON the server would parse."""
    stub = StubTransport({"upserted": 1})
    client(stub).upsert("docs", [{"id": "a", "attrs": {"draft": True, "shipped": False}}])
    assert stub.last.json["records"][0]["attrs"] == {
        "draft": {"Bool": True},
        "shipped": {"Bool": False},
    }


def test_search_omits_an_unset_top_k_and_min_score() -> None:
    """Asserted on the request bytes: the keys are absent, so the server's defaults apply."""
    stub = StubTransport([])
    client(stub).search(query=[1.0, 0.0, 0.0])
    body = stub.last.json
    assert body == {"query": [1.0, 0.0, 0.0], "scope": [], "filter": []}
    assert "top_k" not in body
    assert "min_score" not in body


def test_search_sends_an_explicit_zero_top_k() -> None:
    """``top_k=0`` is a request for zero results and must survive as ``0``.

    This is the pair that makes ``top_k: int | None = None`` the right default: if the SDK
    restated the server's ``10``, "the default" and "none, please" would be the same call.
    """
    stub = StubTransport([])
    client(stub).search(query=[1.0], top_k=0, min_score=0.0)
    body = stub.last.json
    assert body["top_k"] == 0
    assert body["min_score"] == 0.0


def test_list_omits_an_unset_limit_and_sends_an_explicit_zero() -> None:
    """Same rule for ``limit``/``offset``, which have their own server-side defaults."""
    stub = StubTransport([])
    db = client(stub)
    db.list()
    assert "limit" not in stub.last.json
    assert "offset" not in stub.last.json
    db.list(limit=0, offset=0)
    assert stub.last.json == {"scope": [], "offset": 0, "limit": 0, "filter": []}


def test_search_sends_snake_case_and_decodes_hit_attrs() -> None:
    """The public API is idiomatic Python; the mapping happens at the serialization edge."""
    stub = StubTransport(
        [{"collection": "docs", "id": "a", "score": 0.9, "attrs": {"lang": {"Str": "rust"}}}]
    )
    hits = client(stub).search(
        query=[1.0, 0.0, 0.0], top_k=5, min_score=0.1, filter=f.and_(f.eq("lang", "rust"))
    )
    assert stub.last.json == {
        "query": [1.0, 0.0, 0.0],
        "scope": [],
        "top_k": 5,
        "min_score": 0.1,
        "filter": [{"Eq": ["lang", {"Str": "rust"}]}],
    }
    assert len(hits) == 1
    assert hits[0].id == "a"
    assert hits[0].attrs == {"lang": "rust"}


def test_search_similar_sends_the_source_and_decodes_hit_attrs() -> None:
    """The source is named by ``collection``/``id``, never by a query vector."""
    stub = StubTransport(
        [{"collection": "notes", "id": "b", "score": 0.95, "attrs": {"lang": {"Str": "rust"}}}]
    )
    hits = client(stub).search_similar(
        collection="notes", id="a", top_k=5, filter=f.and_(f.eq("lang", "rust"))
    )
    assert stub.last.json == {
        "collection": "notes",
        "id": "a",
        "scope": [],
        "top_k": 5,
        "filter": [{"Eq": ["lang", {"Str": "rust"}]}],
    }
    assert len(hits) == 1
    assert hits[0].id == "b"
    assert hits[0].attrs == {"lang": "rust"}


def test_a_multi_clause_text_search_reaches_the_wire_with_its_combine_rule() -> None:
    """The clause spelling, asserted on the bytes: one text per field, plus how they fold."""
    stub = StubTransport([])
    client(stub).text_search(
        clauses=[{"field": "title", "query": "rust"}, {"field": "body", "query": "async"}],
        combine="Max",
        explain=True,
        highlight={"max_fragments": 2},
    )
    assert stub.last.json == {
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


def test_diversity_reaches_the_wire_on_every_search_route() -> None:
    """The kwarg has to be plumbed on all four routes, and ``0.0`` must survive the prune."""
    stub = StubTransport([])
    db = client(stub)
    db.search(query=[1.0])
    assert "diversity" not in stub.last.json
    db.search(query=[1.0], diversity=0.0)
    assert stub.last.json["diversity"] == 0.0
    db.search_similar(collection="docs", id="d1", diversity=0.3)
    assert stub.last.json["diversity"] == 0.3
    db.text_search(field="body", query="fox", diversity=0.5)
    assert stub.last.json["diversity"] == 0.5
    db.recall("notes", "why", diversity=1.0)
    assert stub.last.json["diversity"] == 1.0


def test_the_ranking_knobs_reach_the_wire() -> None:
    """``rank_by``/``limit_per``/``order_by``/the hybrid weights, in the server's spelling."""
    stub = StubTransport([])
    db = client(stub)
    db.search(
        query=[1.0],
        rank_by=rank.decay("updated_at", 1700000000000, scale=timedelta(days=7)),
        limit_per={"field": "path", "max": 2},
    )
    assert stub.last.json["rank_by"] == {
        "Decay": {"field": "updated_at", "origin": 1700000000000, "scale": 604800000}
    }
    assert stub.last.json["limit_per"] == {"field": "path", "max": 2}
    db.list(order_by={"field": "updated_at", "descending": True})
    assert stub.last.json["order_by"] == {"field": "updated_at", "descending": True}
    db.hybrid_search(vector=[1.0], field="body", text="fox", vector_weight=2.0, text_weight=0.5)
    assert stub.last.json["vector_weight"] == 2.0
    assert stub.last.json["text_weight"] == 0.5


def test_rerank_reaches_the_wire_on_every_rerankable_method() -> None:
    """``rerank`` on ``search``/``text_search``/``hybrid_search``/``recall``, sync client."""
    stub = StubTransport([])
    db = client(stub)
    opts = {"query": "sign in", "overscan": 4, "text_attr": "body"}
    db.search(query=[1.0], rerank=opts)
    assert stub.last.json["rerank"] == opts
    db.text_search(field="body", query="fox", rerank=opts)
    assert stub.last.json["rerank"] == opts
    db.hybrid_search(vector=[1.0], field="body", text="fox", rerank=opts)
    assert stub.last.json["rerank"] == opts
    db.recall("docs", "fox", rerank=opts)
    assert stub.last.json["rerank"] == opts


def test_a_search_refuses_a_query_it_cannot_spell_unambiguously() -> None:
    """Both text spellings at once, or an empty clause list, never reach the server."""
    stub = StubTransport([])
    db = client(stub)
    with pytest.raises(ValueError, match="mutually exclusive"):
        db.text_search(field="body", query="fox", clauses=[{"field": "t", "query": "x"}])
    with pytest.raises(ValueError, match="must not be empty"):
        db.hybrid_search(vector=[1.0], clauses=[])
    assert stub.calls == []


def test_annotations_decode_onto_the_hit_that_carries_them() -> None:
    """``explain``/``highlight`` answer on the hit itself, and stay ``None`` when unasked."""
    stub = StubTransport(
        [
            {
                "collection": "docs",
                "id": "a",
                "score": 1.5,
                "attrs": {},
                "annotations": {
                    "clauses": [{"field": "body", "score": 1.5}],
                    "highlights": [
                        {
                            "field": "body",
                            "fragments": [{"text": "a quick fox", "spans": [[8, 11]]}],
                        }
                    ],
                },
            },
            {"collection": "docs", "id": "b", "score": 0.5, "attrs": {}},
        ]
    )
    hits = client(stub).text_search(field="body", query="fox", explain=True, highlight=True)
    assert stub.last.json["highlight"] == {}
    assert hits[0].annotations is not None
    assert hits[0].annotations.clauses[0].field == "body"
    assert hits[0].annotations.highlights[0].fragments[0].spans == [(8, 11)]
    assert hits[1].annotations is None


def test_aggregate_sends_its_scope_and_sums_and_decodes_the_answer() -> None:
    """A count plus one decoded sum per field — ``Int`` as ``int``, ``Float`` as ``float``."""
    stub = StubTransport({"count": 3, "sums": {"bytes": {"Int": 4096}, "cost": {"Float": 1.5}}})
    out = client(stub).aggregate(
        scope=["docs"], filter=[f.eq("lang", "rust")], sum=["bytes", "cost"]
    )
    assert stub.last.json == {
        "scope": ["docs"],
        "filter": [{"Eq": ["lang", {"Str": "rust"}]}],
        "sum": ["bytes", "cost"],
    }
    assert out.count == 3
    assert out.sums == {"bytes": 4096, "cost": 1.5}
    assert isinstance(out.sums["bytes"], int)


def test_aggregate_group_by_sends_the_field_and_decodes_the_rows() -> None:
    """One row per distinct value, and a ``null`` value means "missing", not a present Null."""
    stub = StubTransport(
        {
            "count": 3,
            "sums": {"bytes": {"Int": 8}},
            "groups": [
                {"value": {"Str": "rust"}, "count": 2, "sums": {"bytes": {"Int": 8}}},
                {"value": None, "count": 1, "sums": {"bytes": {"Int": 0}}},
            ],
        }
    )
    out = client(stub).aggregate(sum=["bytes"], group_by="lang")
    assert stub.last.json == {"scope": [], "filter": [], "sum": ["bytes"], "group_by": "lang"}
    assert [g.count for g in out.groups] == [2, 1]
    assert out.groups[0].value == "rust"
    assert out.groups[1].value is None
    assert out.groups[0].sums == {"bytes": 8}
    assert out.groups_truncated is False


def test_an_ungrouped_aggregate_body_is_unchanged() -> None:
    """No ``group_by`` key at all, so the request is byte-identical to the pre-grouping one."""
    stub = StubTransport({"count": 0, "sums": {}})
    out = client(stub).aggregate()
    assert stub.last.json == {"scope": [], "filter": []}
    assert out.groups == []


def test_batch_search_sends_every_query_and_returns_one_ranking_each() -> None:
    """Each leg is built by the same ``search_body``, so a batched query is shaped as a solo one."""
    stub = StubTransport(
        {"results": [[{"collection": "docs", "id": "a", "score": 1.0, "attrs": {}}], []]}
    )
    out = client(stub).batch_search(
        [{"query": [1.0, 0.0], "top_k": 1}, {"query": [0.0, 1.0], "filter": [f.eq("k", "v")]}]
    )
    assert stub.last.url.endswith("/search/batch")
    assert stub.last.json == {
        "queries": [
            {"query": [1.0, 0.0], "scope": [], "top_k": 1, "filter": []},
            {"query": [0.0, 1.0], "scope": [], "filter": [{"Eq": ["k", {"Str": "v"}]}]},
        ]
    }
    assert len(out) == 2
    assert out[0][0].id == "a"
    assert out[1] == []


def test_a_fused_batch_returns_one_ranking_in_a_one_element_list() -> None:
    """The shape does not change with ``fuse``: still a list of rankings, just one of them."""
    stub = StubTransport({"fused": [{"collection": "docs", "id": "a", "score": 0.5, "attrs": {}}]})
    out = client(stub).batch_search(
        [{"query": [1.0, 0.0]}, {"query": [0.0, 1.0]}],
        fuse=True,
        rrf_k=60.0,
        weights=[1.0, 0.5],
        top_k=5,
    )
    assert stub.last.json["fuse"] == {"rrf_k": 60.0, "weights": [1.0, 0.5], "top_k": 5}
    assert len(out) == 1
    assert [h.id for h in out[0]] == ["a"]


def test_a_fused_batch_sends_fuse_even_with_no_knobs() -> None:
    """``fuse`` is what picks the response shape, so it must survive the usual pruning."""
    stub = StubTransport({"fused": []})
    client(stub).batch_search([{"query": [1.0, 0.0]}], fuse=True)
    assert stub.last.json["fuse"] == {}


def test_set_meta_sends_the_map_as_the_whole_body() -> None:
    """The handler deserializes a bare ``BTreeMap``, not an object wrapping one."""
    stub = StubTransport({"ok": True})
    client(stub).set_meta("docs", {"owner": "austin"})
    assert stub.last.json == {"owner": "austin"}


def test_delete_and_delete_where_share_a_path_but_not_a_body() -> None:
    """One route, two forms — the body is what selects which."""
    stub = StubTransport({"deleted": 1})
    db = client(stub)
    assert db.delete("docs", ["a", "b"]) == 1
    assert stub.last.json == {"ids": ["a", "b"]}
    assert db.delete_where("docs", f.and_(f.eq("lang", "go"))) == 1
    assert stub.last.json == {"filter": [{"Eq": ["lang", {"Str": "go"}]}]}


def test_the_client_refuses_the_calls_that_would_be_silently_wrong() -> None:
    """The guards belong to ``_wire``; this is the proof they are on the caller's path.

    Each of these type-checks under ``mypy --strict`` and, unguarded, produced a request
    the server accepts and answers wrongly — a bare id string deleting the record named by
    one of its characters, a bare scope string searching collections that do not exist, and
    an empty filter deleting the whole collection with a 200. No request must be made at
    all, so the transport must never have been called.
    """
    stub = StubTransport({"deleted": 1})
    db = client(stub)
    with pytest.raises(TypeError, match="did you mean"):
        db.delete("docs", "x1")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="did you mean"):
        db.search(query=[1.0, 0.0, 0.0], scope="docs")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="did you mean"):
        db.set_fts_schema("docs", "body")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="drop_collection"):
        db.delete_where("docs", [])
    assert stub.calls == []


def test_remember_omits_mode_and_attrs_when_unset() -> None:
    """Both are ``#[serde(default)]`` server-side, so absent means "the server's default"."""
    stub = StubTransport({"ok": True, "upserted": 1})
    db = client(stub)
    db.remember("notes", "a", "the quick brown fox", attrs={"tag": "x", "year": 2024})
    assert stub.last.json == {
        "id": "a",
        "text": "the quick brown fox",
        "attrs": {"tag": {"Str": "x"}, "year": {"Int": 2024}},
    }
    assert "mode" not in stub.last.json
    db.remember("notes", "b", "a long article", mode="summarize")
    assert stub.last.json == {"id": "b", "text": "a long article", "mode": "summarize"}
    assert "attrs" not in stub.last.json
    assert "ttl_seconds" not in stub.last.json
    assert "dedupe_threshold" not in stub.last.json


def test_remember_sends_ttl_and_dedupe_and_returns_what_was_written() -> None:
    """The knobs reach the wire under their snake_case names, and the result names the
    record that changed — which a dedupe match makes a *different* one."""
    stub = StubTransport({"ok": True, "upserted": 1, "id": "older", "deduped": True})
    out = client(stub).remember(
        "notes", "newer", "the quick brown fox", ttl_seconds=3600, dedupe_threshold=0.95
    )
    assert stub.last.json == {
        "id": "newer",
        "text": "the quick brown fox",
        "ttl_seconds": 3600,
        "dedupe_threshold": 0.95,
    }
    assert out == RememberResult(id="older", upserted=1, deduped=True)


def test_recall_defaults_to_an_empty_filter_and_omitted_bounds() -> None:
    stub = StubTransport([])
    client(stub).recall("notes", "hello")
    assert stub.last.json == {"query": "hello", "filter": []}


def test_recall_sends_reinforce_and_extend_ttl_seconds() -> None:
    stub = StubTransport([])
    client(stub).recall("notes", "hello", reinforce=True, extend_ttl_seconds=3600)
    assert stub.last.json == {
        "query": "hello",
        "filter": [],
        "reinforce": True,
        "extend_ttl_seconds": 3600,
    }


def test_recall_omits_reinforce_when_false() -> None:
    stub = StubTransport([])
    client(stub).recall("notes", "hello", reinforce=False)
    assert stub.last.json == {"query": "hello", "filter": []}


def test_recall_sends_rank_by_and_omits_it_when_unset() -> None:
    stub = StubTransport([])
    client(stub).recall(
        "notes",
        "hello",
        rank_by=rank.decay(field="", origin=0, count_field="nidus.access_count"),
    )
    assert stub.last.json == {
        "query": "hello",
        "filter": [],
        "rank_by": {"Decay": {"field": "", "origin": 0, "count_field": "nidus.access_count"}},
    }
    client(stub).recall("notes", "hello")
    assert "rank_by" not in stub.last.json


# ── Responses ────────────────────────────────────────────────────────────────────────


def test_stats_decodes_a_null_ann_to_none() -> None:
    """``ann: null`` is the server saying "exact search", and must not become an object."""
    stub = StubTransport(STATS_PAYLOAD)
    stats = client(stub).stats()
    assert stats.ann is None
    assert stats.dimension == 3
    assert stats.footprint.doc_count == 2


def test_stats_decodes_an_hnsw_ann_with_the_ivf_knobs_left_none() -> None:
    """The server omits the knobs that do not apply, so they arrive as ``None``."""
    stub = StubTransport(
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
    ann = client(stub).stats().ann
    assert ann is not None
    assert ann.kind == "Hnsw"
    assert (ann.m, ann.ef_search) == (16, 64)
    assert ann.n_lists is None
    assert ann.n_probe is None


def test_records_keeps_an_absent_vector_as_none() -> None:
    """A record with no ``vector`` key is a text-only doc — ``None``, never ``[]``."""
    stub = StubTransport(
        [
            {"id": "a", "vector": [1.0, 0.0, 0.0], "attrs": {}},
            {"id": "b", "attrs": {"body": {"Str": "text only"}}},
        ]
    )
    records = client(stub).records("docs")
    assert records[0].vector == [1.0, 0.0, 0.0]
    assert records[1].vector is None
    assert records[1].attrs == {"body": "text only"}


@pytest.mark.parametrize("status", [200, 201, 204, 299])
def test_health_is_true_for_any_2xx(status: int) -> None:
    stub = StubTransport(None, status=status, text="ok")
    assert client(stub).health() is True


@pytest.mark.parametrize("status", [404, 500, 503])
def test_health_is_false_rather_than_raising(status: int) -> None:
    """A health check that throws when the thing is unhealthy is useless for its one job."""
    stub = StubTransport({"error": "unhealthy"}, status=status)
    assert client(stub).health() is False


def test_health_is_false_when_the_transport_itself_blows_up() -> None:
    """Including a broken custom transport: still an answer, still ``False``."""

    def exploding(*_args: Any) -> tuple[int, str]:
        raise RuntimeError("boom")

    db = NidusClient("http://x", transport=exploding)
    assert db.health() is False


# ── Readiness / cluster / refresh ────────────────────────────────────────────────────


def test_ready_decodes_a_200_into_a_positive_readiness() -> None:
    stub = StubTransport({"ready": True, "role": "Leader", "staleness_secs": 3})
    out = client(stub).ready()
    assert stub.last.method == "GET"
    assert stub.last.url == "http://x/ready"
    assert out == Readiness(ready=True, role="Leader", staleness_secs=3)


def test_ready_on_503_returns_a_negative_readiness_and_does_not_raise() -> None:
    """The decision under test: a 503 is an answer, not a fault."""
    stub = StubTransport({"error": "store not open yet"}, status=503)
    out = client(stub).ready()  # must not raise
    assert out == Readiness(ready=False, reason="store not open yet")


def test_ready_on_500_still_raises_nidus_error() -> None:
    """Every OTHER non-2xx keeps raising, exactly as every other method does."""
    stub = StubTransport({"error": "boom"}, status=500)
    with pytest.raises(NidusError) as caught:
        client(stub).ready()
    assert caught.value.status == 500


def test_cluster_decodes_every_field_including_the_nullable_ones() -> None:
    stub = StubTransport(CLUSTER_PAYLOAD)
    out = client(stub).cluster()
    assert out == ClusterStatus(
        role="Leader",
        cluster=True,
        holds_writer_handle=True,
        fenced=False,
        lease_owner="node-a",
        commit_version=42,
        staleness_secs=0,
        max_staleness_secs=30,
    )


def test_cluster_decodes_null_lease_owner_and_max_staleness_as_none() -> None:
    stub = StubTransport(
        {
            "role": "Follower",
            "cluster": False,
            "holds_writer_handle": False,
            "fenced": True,
            "lease_owner": None,
            "commit_version": 7,
            "staleness_secs": 12,
            "max_staleness_secs": None,
        }
    )
    out = client(stub).cluster()
    assert out.lease_owner is None
    assert out.max_staleness_secs is None


def test_versions_decodes_every_field_including_the_nullable_ones() -> None:
    stub = StubTransport(VERSIONS_PAYLOAD)
    out = client(stub).versions()
    assert stub.last.method == "GET"
    assert stub.last.url == "http://x/versions"
    assert out == StoreVersions(
        commit_version=9,
        oldest_readable=3,
        pinned=5,
        readable=[3, 4, 5, 6, 7, 8, 9],
    )


def test_versions_decodes_null_oldest_readable_and_pinned_as_none() -> None:
    stub = StubTransport(
        {"commit_version": 1, "oldest_readable": None, "pinned": None, "readable": [1]}
    )
    out = client(stub).versions()
    assert out.oldest_readable is None
    assert out.pinned is None


@pytest.mark.parametrize("adopted", [True, False])
def test_refresh_posts_and_returns_the_adopted_bool(adopted: bool) -> None:
    stub = StubTransport({"adopted": adopted})
    out = client(stub).refresh()
    assert stub.last.method == "POST"
    assert stub.last.url == "http://x/refresh"
    assert out is adopted


# ── Errors ───────────────────────────────────────────────────────────────────────────


def test_a_non_2xx_raises_nidus_error_with_the_servers_message_and_status() -> None:
    """The ``{"error": …}`` body is where the useful text lives."""
    stub = StubTransport({"error": "store is locked: /tmp/s/lock"}, status=409)
    with pytest.raises(NidusError) as caught:
        client(stub).flush()
    assert caught.value.status == 409
    assert caught.value.message == "store is locked: /tmp/s/lock"
    assert caught.value.is_locked
    assert not caught.value.is_transport_error


def test_a_non_json_error_body_is_reported_verbatim() -> None:
    """A proxy's HTML page says more than "HTTP 502" does."""
    stub = StubTransport(None, status=502, text="<html>502 Bad Gateway</html>")
    with pytest.raises(NidusError) as caught:
        client(stub).stats()
    assert caught.value.status == 502
    assert caught.value.message == "<html>502 Bad Gateway</html>"


def test_an_empty_error_body_falls_back_to_the_status() -> None:
    """Nothing to quote, so the status is the whole message."""
    stub = StubTransport(None, status=404, text="")
    with pytest.raises(NidusError) as caught:
        client(stub).get_meta("nope")
    assert caught.value.status == 404
    assert caught.value.message == "HTTP 404"


def test_a_transport_failure_is_status_zero() -> None:
    """Driven through the real ``urllib`` transport against a port nothing is listening on.

    Status ``0`` is the cross-SDK sentinel for "no response at all", and it is produced by
    catching what ``urllib`` raises — so a stub that returns a status could never test it.
    """
    with socket.socket() as probe:
        # Bind, read the port, close: the kernel hands out a port that is then free, so the
        # connect below is refused rather than hanging on a firewalled address.
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    db = NidusClient(f"http://127.0.0.1:{port}", timeout=2.0)
    with pytest.raises(NidusError) as caught:
        db.stats()
    assert caught.value.status == 0
    assert caught.value.is_transport_error
    assert "/stats" in caught.value.message
    # And `health` swallows the same failure instead of propagating it.
    assert db.health() is False


# ── Configuration ────────────────────────────────────────────────────────────────────


def test_a_trailing_slash_in_the_base_url_is_stripped() -> None:
    """Otherwise every URL is ``http://x//collections``."""
    for base in ("http://x/", "http://x//", "http://x"):
        stub = StubTransport(["docs"])
        NidusClient(base, transport=stub).collections()
        assert stub.last.url == "http://x/collections"


def test_a_base_url_with_a_path_prefix_is_preserved() -> None:
    """A client behind a reverse proxy mount point still addresses the right routes."""
    stub = StubTransport(["docs"])
    NidusClient("http://x/nidus/", transport=stub).collections()
    assert stub.last.url == "http://x/nidus/collections"


@pytest.mark.parametrize(
    ("name", "escaped"),
    [
        ("a/b c", "a%2Fb%20c"),
        ("with space", "with%20space"),
        ("nested/path/name", "nested%2Fpath%2Fname"),
        ("q?x#y", "q%3Fx%23y"),
    ],
)
def test_a_collection_name_is_percent_escaped_into_one_path_segment(
    name: str, escaped: str
) -> None:
    """A slash must not change which route is addressed, and a space must not break the URL."""
    stub = StubTransport({"upserted": 0})
    client(stub).upsert(name, [])
    assert stub.last.url == f"http://x/collections/{escaped}/upsert"


def test_a_bearer_token_is_attached_to_every_request() -> None:
    stub = StubTransport([])
    db = client(stub, token="sekret")
    db.list()
    db.collections()
    assert all(c.headers["authorization"] == "Bearer sekret" for c in stub.calls)


def test_extra_headers_are_sent_and_cannot_unset_the_token() -> None:
    stub = StubTransport([])
    client(stub, token="sekret", headers={"x-trace": "1", "authorization": "Bearer other"}).list()
    assert stub.last.headers["x-trace"] == "1"
    assert stub.last.headers["authorization"] == "Bearer sekret"


def test_headers_are_copied_at_construction() -> None:
    """A later mutation of the caller's dict must not change our auth handling mid-run."""
    stub = StubTransport([])
    headers = {"x-trace": "1"}
    db = client(stub, headers=headers)
    headers["x-trace"] = "2"
    db.list()
    assert stub.last.headers["x-trace"] == "1"


def test_the_timeout_is_handed_to_the_transport() -> None:
    """``None`` means "no timeout" and is passed through as such, not replaced."""
    stub = StubTransport([])
    client(stub, timeout=1.5).list()
    assert stub.last.timeout == 1.5
    stub = StubTransport([])
    client(stub).list()
    assert stub.last.timeout is None


def test_close_forwards_to_a_transport_that_has_one() -> None:
    """The default transport is connectionless; a pooled one gets its sockets closed."""
    stub = StubTransport([])
    with client(stub) as db:
        db.list()
    assert stub.closed == 1
    db.close()
    assert stub.closed == 2


def test_close_is_a_no_op_for_a_transport_without_one() -> None:
    """A plain function is a perfectly good transport and must not have to grow a hook."""

    def bare(*_args: Any) -> tuple[int, str]:
        return 200, "[]"

    with NidusClient("http://x", transport=bare) as db:
        assert db.list() == []


def test_an_empty_base_url_is_rejected_at_construction() -> None:
    """Fail where the mistake is, not on the first request."""
    with pytest.raises(ValueError, match="base_url"):
        NidusClient("")


# ── Install shapes: httpx is genuinely optional ──────────────────────────────────────


class _MissingHttpx(importlib.abc.MetaPathFinder):
    """A finder that makes ``httpx`` unimportable, simulating an install without the extra.

    A finder rather than an uninstall (or ``sys.modules["httpx"] = None``) because it is the
    honest simulation — ``import httpx`` raises the same ``ModuleNotFoundError`` it would on
    a machine that never had it — and because it is reversible, so ``test_aio`` still runs.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: Any | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == "httpx" or fullname.startswith("httpx."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@pytest.fixture()
def httpx_absent() -> Iterator[None]:
    """Run the body as if ``httpx`` were never installed, then put everything back.

    ``nidus`` itself is evicted too, so the import under test is a real cold import rather
    than a cache hit — and the whole ``sys.modules`` snapshot is restored afterwards so the
    freshly-imported duplicate modules cannot leak into any later test.
    """
    saved = dict(sys.modules)
    finder = _MissingHttpx()
    sys.meta_path.insert(0, finder)
    for name in list(sys.modules):
        if name.split(".")[0] in {"httpx", "nidus"}:
            del sys.modules[name]
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.clear()
        sys.modules.update(saved)


def test_import_nidus_works_with_httpx_absent(httpx_absent: None) -> None:
    """The headline property: ``pip install nidus`` pulls nothing, so this must hold."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("httpx")

    fresh = importlib.import_module("nidus")
    assert fresh.NidusClient is not None
    # The sync client is fully usable, not merely importable.
    stub = StubTransport(["docs"])
    assert fresh.NidusClient("http://x", transport=stub).collections() == ["docs"]


def test_importing_nidus_aio_without_httpx_names_the_fix(httpx_absent: None) -> None:
    """A bare ``ModuleNotFoundError`` would say a module is missing but not how to fix it."""
    with pytest.raises(ImportError) as caught:
        importlib.import_module("nidus.aio")
    assert "nidus[async]" in str(caught.value)
    assert "httpx" in str(caught.value)

    # The lazy `nidus.AsyncNidusClient` attribute goes through the same import, so it
    # surfaces the same guidance rather than an AttributeError.
    fresh = importlib.import_module("nidus")
    with pytest.raises(ImportError, match=r"nidus\[async\]"):
        _ = fresh.AsyncNidusClient


def test_star_import_works_with_httpx_absent(httpx_absent: None) -> None:
    """``from nidus import *`` must not drag in the async client.

    ``import *`` resolves every name in ``__all__`` eagerly, lazy ``__getattr__`` included,
    so listing ``AsyncNidusClient`` there made a star-import of the *sync* client fail with
    an ImportError about httpx — turning the optional dependency into a de facto hard one.
    The plain-import test above cannot see that, which is why this one exists.
    """
    namespace: dict[str, Any] = {}
    # `exec` because a star-import is only legal at module level — and the import shape
    # *is* what is under test, so it cannot be spelled any other way.
    exec("from nidus import *", namespace)
    assert namespace["NidusClient"] is not None
    assert "AsyncNidusClient" not in namespace
    assert "httpx" not in sys.modules

    # Still reachable by its two documented spellings, which do opt into httpx.
    fresh = importlib.import_module("nidus")
    with pytest.raises(ImportError, match=r"nidus\[async\]"):
        _ = fresh.AsyncNidusClient


def test_nidus_has_no_attribute_for_an_unknown_name() -> None:
    """The lazy ``__getattr__`` must not turn every typo into an import attempt."""
    import nidus

    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        _ = nidus.Nope  # type: ignore[attr-defined]
