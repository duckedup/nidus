"""Tests for ``NidusClient.query``/``compile`` — the SQL-shaped read syntax's SDK surface
(nidus-yq9p.2 / .3 / .6, SPEC §7.12), against the stub transport.

``POST /query`` itself is exercised end to end, against a real server, by
``test_integration.py``'s ``test_corpus_case_against_a_real_server``, which replays
``tests/corpus/queries.json``. This file is the no-server tier: every answer shape
``/query`` can send back, that a ``;``-script decodes to its answers in order, that
``compile`` never issues anything but the compile call itself, and that a parse error
surfaces exactly like every other ``NidusError`` this SDK raises — the same properties
``test_client.py`` pins for every other method, so ``query``/``compile`` get no exemption.

The bar is falsifiability: every test here names the edit that would turn it red. One that
would still pass with ``query``/``compile`` deleted (a bare "a response came back") does
not belong.
"""

from __future__ import annotations

import json
from typing import Any, NamedTuple, Optional

import pytest

from nidus import Hit, NidusClient, NidusError

# ── A local, self-contained stub (each test file keeps its own — `test_aio.py` already
# duplicates `test_client.py`'s rather than importing it, since these files have no shared
# `conftest.py` to hang a fixture on) ──────────────────────────────────────────────────


class Call(NamedTuple):
    """One request as the transport saw it — raw, before any decoding."""

    method: str
    url: str
    headers: dict[str, str]
    body: Optional[bytes]
    timeout: Optional[float]

    @property
    def json(self) -> Any:
        assert self.body is not None, "expected a request body"
        return json.loads(self.body)


class StubTransport:
    """A recording ``Transport``: one canned ``(status, text)`` out, every call kept."""

    def __init__(self, payload: Any = None, status: int = 200, text: Optional[str] = None) -> None:
        self.text = text if text is not None else json.dumps(payload)
        self.status = status
        self.calls: list[Call] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Optional[bytes],
        timeout: Optional[float],
    ) -> tuple[int, str]:
        self.calls.append(Call(method, url, dict(headers), body, timeout))
        return self.status, self.text

    @property
    def last(self) -> Call:
        assert self.calls, "no request was made"
        return self.calls[-1]


def client(stub: StubTransport, **kwargs: Any) -> NidusClient:
    return NidusClient("http://x", transport=stub, **kwargs)


SQL = "SELECT * FROM notes WHERE lang = 'go' ORDER BY bytes"


# ── `query`: request shape ───────────────────────────────────────────────────────────


def test_query_posts_sql_alone_to_query_with_no_compile_only_key() -> None:
    """A plain ``query`` call omits ``compile_only`` so the server's own default (run it)
    applies — the same "an unset optional is omitted" rule every other body follows.
    """
    stub = StubTransport([])
    client(stub).query(SQL)
    assert stub.last.method == "POST"
    assert stub.last.url == "http://x/query"
    assert stub.last.json == {"sql": SQL}


def test_query_is_the_only_call_it_makes() -> None:
    """Exactly one request per ``query`` call — no other route, no follow-up request."""
    stub = StubTransport([])
    client(stub).query(SQL)
    assert len(stub.calls) == 1


# ── `query`: every answer shape ──────────────────────────────────────────────────────


def test_a_hits_answer_decodes_exactly_like_search() -> None:
    """Decode parity: the same wire payload decodes to the same ``Hit`` objects whether it
    arrives from ``/search`` or from ``/query`` — field by field, not just by length.
    """
    payload = [
        {
            "collection": "notes",
            "id": "n3",
            "score": 0.75,
            "attrs": {"lang": {"Str": "go"}, "bytes": {"Int": 200}},
            "annotations": {
                "clauses": [{"field": "body", "score": 1.5}],
                "highlights": [{"field": "body", "fragments": [{"text": "go", "spans": [[0, 2]]}]}],
            },
        }
    ]
    from_query = client(StubTransport(payload)).query(SQL)
    from_search = client(StubTransport(payload)).search(query=[1.0, 0.0])
    assert from_query == from_search
    assert len(from_query) == 1
    hit = from_query[0]
    assert isinstance(hit, Hit)
    assert hit.collection == "notes"
    assert hit.id == "n3"
    assert hit.score == pytest.approx(0.75)
    assert hit.attrs == {"lang": "go", "bytes": 200}
    assert hit.annotations is not None
    assert hit.annotations.clauses[0].field == "body"
    assert hit.annotations.clauses[0].score == pytest.approx(1.5)
    assert hit.annotations.highlights[0].fragments[0].text == "go"
    assert hit.annotations.highlights[0].fragments[0].spans == [(0, 2)]


def test_a_with_plan_answer_decodes_to_a_hits_plan_pair_like_search_with_plan() -> None:
    """``WITH (plan)``/``WITH (annotations)`` wraps the hits in ``{"hits", "plan"}``,
    mirroring exactly what ``search_with_plan`` decodes for the typed request.
    """
    payload = {
        "hits": [{"collection": "notes", "id": "n3", "score": 0.5, "attrs": {}}],
        "plan": {"path": "exact", "narrowing": {"state": "inactive"}, "timings": {"total_us": 42}},
    }
    hits, plan = client(StubTransport(payload)).query(SQL)
    assert [h.id for h in hits] == ["n3"]
    assert plan.path == "exact"
    assert plan.timings.total_us == 42
    assert plan.narrowing.state == "inactive"


def test_an_aggregation_answer_decodes_with_a_none_group_and_a_float_sum() -> None:
    """``GROUP BY`` decodes exactly like ``/aggregate``'s own answer: a missing-attribute
    group is ``value=None`` (distinct from a present ``Null``), and a float sum stays a
    Python ``float`` rather than being coerced to an ``int``.
    """
    payload = {
        "count": 3,
        "sums": {"amount": {"Float": 12.5}},
        "groups": [
            {"value": {"Str": "rust"}, "count": 2, "sums": {"amount": {"Float": 10.0}}},
            {"value": None, "count": 1, "sums": {"amount": {"Float": 2.5}}},
        ],
    }
    agg = client(StubTransport(payload)).query("SELECT * FROM notes GROUP BY lang, SUM(amount)")
    assert agg.count == 3
    assert agg.sums == {"amount": 12.5}
    assert isinstance(agg.sums["amount"], float)
    assert len(agg.groups) == 2
    assert agg.groups[0].value == "rust"
    assert agg.groups[0].count == 2
    assert agg.groups[0].sums == {"amount": 10.0}
    assert agg.groups[1].value is None
    assert agg.groups[1].count == 1
    assert agg.groups[1].sums == {"amount": 2.5}


def test_a_three_statement_script_returns_three_answers_in_order() -> None:
    """Batch ordering: a ``;``-separated script's answers come back as a list, positionally
    matching the statements as written — not merged, not reordered.
    """
    payload = [
        [{"collection": "notes", "id": "a", "score": 1.0, "attrs": {}}],
        {"count": 1, "sums": {}, "groups": []},
        [
            {"collection": "notes", "id": "c", "score": 1.0, "attrs": {}},
            {"collection": "notes", "id": "b", "score": 0.5, "attrs": {}},
        ],
    ]
    script = (
        "SELECT * FROM notes; SELECT * FROM notes GROUP BY lang; SELECT * FROM notes ORDER BY x"
    )
    answers = client(StubTransport(payload)).query(script)
    assert len(answers) == 3
    assert [h.id for h in answers[0]] == ["a"]
    assert answers[1].count == 1
    assert [h.id for h in answers[2]] == ["c", "b"]


def test_a_single_statement_hits_answer_with_one_hit_is_not_mistaken_for_a_batch() -> None:
    """A one-element bare hits array (one statement, one hit) must decode to that one
    ``Hit`` — not to a one-statement "batch" whose single answer is itself the hit object.
    """
    payload = [{"collection": "notes", "id": "solo", "score": 1.0, "attrs": {}}]
    hits = client(StubTransport(payload)).query(SQL)
    assert [h.id for h in hits] == ["solo"]


def test_an_empty_hits_answer_is_not_mistaken_for_a_batch() -> None:
    """Zero hits from one statement is a bare empty list, not an empty batch."""
    assert client(StubTransport([])).query(SQL) == []


# ── `compile` ─────────────────────────────────────────────────────────────────────────


def test_compile_sends_compile_only_true() -> None:
    stub = StubTransport({"kind": "list", "collections": ["notes"], "opts": {}})
    client(stub).compile(SQL)
    assert stub.last.url == "http://x/query"
    assert stub.last.json == {"sql": SQL, "compile_only": True}


def test_compile_returns_the_servers_rendered_form_unchanged() -> None:
    """``compile`` is introspection only: the SDK renders nothing of its own, it just hands
    back the JSON the server already rendered — bare for one statement, a list for a script.
    """
    compiled = {"kind": "search", "collections": ["notes"], "vector": [1.0], "opts": {"top_k": 10}}
    assert client(StubTransport(compiled)).compile(SQL) == compiled

    compiled_script = [compiled, {"kind": "aggregate", "collections": ["notes"], "opts": {}}]
    assert client(StubTransport(compiled_script)).compile(SQL + "; " + SQL) == compiled_script


def test_compile_executes_nothing() -> None:
    """The one call ``compile`` makes is the compile call itself — no search/list/aggregate
    request follows it, and the compile call carries no route but ``/query``.
    """
    stub = StubTransport({"kind": "list", "collections": [], "opts": {}})
    client(stub).compile(SQL)
    assert len(stub.calls) == 1
    assert stub.calls[0].url == "http://x/query"


# ── Errors ────────────────────────────────────────────────────────────────────────────


def test_a_parse_error_raises_nidus_error_with_the_servers_message_verbatim() -> None:
    """The offset and the §7 feature name are the server's own words, not re-worded here."""
    message = "sql parse error at byte 17: expected a value after '=' (§7.3 boolean composition)"
    stub = StubTransport({"error": message}, status=400)
    with pytest.raises(NidusError) as caught:
        client(stub).query("SELECT * FROM notes WHERE lang =")
    assert caught.value.status == 400
    assert caught.value.message == message
    assert "byte 17" in caught.value.message
    assert "§7.3" in caught.value.message


def test_a_parse_error_is_a_400_not_a_500() -> None:
    """400 (a caller mistake) and 500 (a server fault) must not be confused with each other."""
    stub = StubTransport({"error": "sql parse error at byte 0: ..."}, status=400)
    with pytest.raises(NidusError) as caught:
        client(stub).query("SELECT")
    assert caught.value.status == 400
    assert caught.value.is_bad_request

    stub = StubTransport({"error": "internal error"}, status=500)
    with pytest.raises(NidusError) as caught:
        client(stub).query(SQL)
    assert caught.value.status == 500
    assert not caught.value.is_bad_request


def test_compiles_own_parse_error_is_the_same_shape() -> None:
    """``compile`` fails the same way ``query`` does: same status, same verbatim message."""
    message = "sql parse error at byte 5: unexpected end of input"
    stub = StubTransport({"error": message}, status=400)
    with pytest.raises(NidusError) as caught:
        client(stub).compile("SELECT *")
    assert caught.value.status == 400
    assert "byte 5" in caught.value.message


# ── Parity with the neighbouring methods (auth, timeout) ────────────────────────────────


def test_query_carries_the_bearer_token_and_the_configured_timeout() -> None:
    """`query` reaches the wire through the same `_request`/`_send` every other method
    does, so it must carry auth and timeout exactly as `search`/`list` already do.
    """
    stub = StubTransport([])
    client(stub, token="sekret", timeout=2.5).query(SQL)
    assert stub.last.headers["authorization"] == "Bearer sekret"
    assert stub.last.timeout == 2.5


# ── The async client mirrors the sync one exactly ────────────────────────────────────


async def test_the_async_client_decodes_query_and_compile_the_same_way() -> None:
    """``aio.query``/``aio.compile`` are expected to stay in lockstep with the sync
    client's — same decode, same shape rule — so this is that lockstep, proven once.
    """
    httpx = pytest.importorskip("httpx", reason="the async client needs the nidus[async] extra")
    from nidus.aio import AsyncNidusClient

    hits_payload = [{"collection": "notes", "id": "n3", "score": 1.0, "attrs": {}}]

    def handle(request: Any) -> Any:
        request.read()
        assert request.url.path == "/query"
        body = json.loads(request.content)
        if body.get("compile_only"):
            return httpx.Response(200, json={"kind": "list", "collections": [], "opts": {}})
        return httpx.Response(200, json=hits_payload)

    async with AsyncNidusClient("http://x", transport=httpx.MockTransport(handle)) as db:
        hits = await db.query(SQL)
        assert [h.id for h in hits] == ["n3"]
        compiled = await db.compile(SQL)
        assert compiled == {"kind": "list", "collections": [], "opts": {}}
