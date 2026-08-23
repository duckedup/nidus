"""Tests for ``nidus._wire`` — the whole HTTP contract, minus the HTTP.

``_wire`` exists so the sync and async clients cannot drift, and the happy side effect is
that nearly every wire behaviour worth testing is a pure function here: paths, bodies,
pruning, decoding, error extraction. So this file is where the contract is pinned, and
``test_client``/``test_aio`` only have to prove that each client actually *calls* it.

The load-bearing cases, all of which are silent when wrong:

* **Omit vs zero.** The server supplies ``top_k = 10`` / ``limit = 100`` / ``rrf_k = 60.0``
  / ``candidates = 100`` via ``#[serde(default)]``. An unset optional must be *absent* from
  the JSON, while an explicit ``0`` must be *sent* — ``top_k=0`` is a legitimate request for
  zero results, so ``0`` cannot double as "unset". Restating the server's numbers in Python
  would fork the contract the day one of them changes.
* **``scope`` and ``filter`` are never pruned**, because an empty scope ("everywhere") and an
  empty filter ("everything") are the real values, not missing ones.
* **Path escaping.** ``quote(name, safe="")`` — without ``safe=""``, a collection named
  ``a/b`` addresses a route that does not exist.
* **Absence survives decoding.** A record with no ``vector`` is a text-only doc (``None``),
  not an empty one; ``ann: null`` means exact search, not "ANN with defaults".
* **A bare string is not a sequence of strings.** ``str`` satisfies ``Sequence[str]``, so
  ``ids="x1"`` / ``scope="docs"`` / ``fields="body"`` type-check and iterate per character
  into a request the server accepts and answers wrongly. Every such parameter is asserted
  to raise instead.
* **A vector element is coerced with ``float()``.** A Python caller's embedding comes from
  numpy, and ``np.float32`` is not a ``float`` subclass, so an uncoerced element dies inside
  ``json.dumps`` with a message naming neither nidus nor the argument.

Testing ``_wire`` directly rather than only through a client is deliberate: a private module
that carries this much of the contract is worth asserting on at its own boundary, where a
failure names the function instead of the endpoint.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from decimal import Decimal

import pytest

from nidus import NidusError, RememberResult, _wire, f, rank, v

# A `/stats` payload for a store doing exact brute-force search: `ann` is null.
EXACT_STATS = {
    "dimension": 3,
    "distance": "Cosine",
    "ann": None,
    "collections": ["docs", "notes"],
    "footprint": {
        "rows": 4,
        "dead_rows": 1,
        "dimension": 3,
        "vector_bytes": 48,
        "doc_count": 3,
    },
}

# The same payload with an HNSW index. Note which knobs are present: the server's `AnnDto`
# skips the ones that do not apply to the active kind, so IVF's are simply absent.
HNSW_STATS = {
    **EXACT_STATS,
    "ann": {
        "kind": "Hnsw",
        "overscan": 2,
        "seed": 42,
        "m": 16,
        "ef_construction": 200,
        "ef_search": 64,
    },
}

IVF_STATS = {
    **EXACT_STATS,
    "ann": {"kind": "Ivf", "overscan": 2, "seed": 7, "n_lists": 64, "n_probe": 8},
}


# ── prune: the omit-don't-guess rule ─────────────────────────────────────────────────


def test_prune_drops_only_none_valued_keys() -> None:
    """``None`` means "omit"; every other falsy value is a real value and stays."""
    assert _wire.prune({"a": 1, "b": None}) == {"a": 1}
    assert _wire.prune({"zero": 0, "empty": "", "list": [], "false": False}) == {
        "zero": 0,
        "empty": "",
        "list": [],
        "false": False,
    }
    assert _wire.prune({}) == {}


# ── Paths ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("builder", "expected"),
    [
        (_wire.collection_path, "/collections/docs"),
        (_wire.meta_path, "/collections/docs/meta"),
        (_wire.upsert_path, "/collections/docs/upsert"),
        (_wire.delete_path, "/collections/docs/delete"),
        (_wire.records_path, "/collections/docs/records"),
        (_wire.fts_schema_path, "/collections/docs/fts-schema"),
        (_wire.filter_index_path, "/collections/docs/filter-index"),
        (_wire.remember_path, "/collections/docs/remember"),
        (_wire.recall_path, "/collections/docs/recall"),
        (_wire.suggest_path, "/collections/docs/suggest"),
    ],
)
def test_collection_paths(builder: Callable[[str], str], expected: str) -> None:
    """Every per-collection path hangs off ``/collections/{name}``."""
    assert builder("docs") == expected


def test_collection_names_are_escaped_to_a_single_path_segment() -> None:
    """A slash or a space in a name must not change which route is addressed."""
    assert _wire.collection_path("a/b c") == "/collections/a%2Fb%20c"
    assert _wire.upsert_path("a/b") == "/collections/a%2Fb/upsert"
    # `?` and `#` would otherwise start a query string or a fragment.
    assert _wire.collection_path("q?x#y") == "/collections/q%3Fx%23y"
    # Non-ASCII is percent-encoded UTF-8, which is what the server's path decoder expects.
    assert _wire.collection_path("nøtes") == "/collections/n%C3%B8tes"


def test_suggest_path_escapes_the_collection_name() -> None:
    assert _wire.suggest_path("a/b c") == "/collections/a%2Fb%20c/suggest"


# ── Request bodies ───────────────────────────────────────────────────────────────────


def test_upsert_body_normalizes_attrs_and_omits_an_absent_vector() -> None:
    """Plain attrs are tagged; a text-only doc simply has no ``vector`` key."""
    body = _wire.upsert_body(
        [
            {"id": "a", "vector": [1.0, 0.0, 0.0], "attrs": {"lang": "rust", "year": 2024}},
            {"id": "b", "attrs": {"body": v.str("text only")}},
            {"id": "c"},
        ]
    )
    assert body == {
        "records": [
            {
                "id": "a",
                "vector": [1.0, 0.0, 0.0],
                "attrs": {"lang": {"Str": "rust"}, "year": {"Int": 2024}},
            },
            {"id": "b", "attrs": {"body": {"Str": "text only"}}},
            {"id": "c", "attrs": {}},
        ]
    }
    # `attrs` carries no `#[serde(default)]` on the server's `Record`, so it is always
    # emitted — even empty. Omitting it would be a deserialization failure, not a default.
    assert "attrs" in body["records"][2]
    assert "vector" not in body["records"][1]


def test_delete_bodies_pick_the_form_by_which_key_is_present() -> None:
    """One route, two bodies: ``ids`` or ``filter`` (the server lets ``filter`` win)."""
    assert _wire.delete_ids_body(["a", "b"]) == {"ids": ["a", "b"]}
    assert _wire.delete_where_body(f.and_(f.eq("lang", "go"))) == {
        "filter": [{"Eq": ["lang", {"Str": "go"}]}]
    }
    assert "filter" not in _wire.delete_ids_body(["a"])
    assert "ids" not in _wire.delete_where_body(f.and_(f.eq("lang", "go")))


def test_delete_where_body_refuses_an_empty_filter() -> None:
    """``[]`` is "match everything" on the server, so this would delete the collection.

    Verified against the real binary before the guard existed: a collection holding two
    records answered ``delete_where(name, []) -> 2`` with a 200. The shape that produces it
    is a filter list assembled from optional conditions that all turned out absent, which
    is ordinary Python — so the SDK, the last layer that can tell that apart from a
    deliberate delete-all, refuses it and names ``drop_collection``.
    """
    with pytest.raises(ValueError, match="drop_collection"):
        _wire.delete_where_body([])


def test_search_body_omits_unset_optionals_but_keeps_scope_and_filter() -> None:
    """The default case: only ``query``, plus the two empties that mean "everything"."""
    assert _wire.search_body([1.0, 0.0, 0.0]) == {
        "query": [1.0, 0.0, 0.0],
        "scope": [],
        "filter": [],
    }


def test_search_body_sends_an_explicit_zero_top_k() -> None:
    """``top_k=None`` is omitted; ``top_k=0`` is sent as ``0``.

    These are different requests — "use the server's default" versus "give me nothing" — and
    conflating them is exactly the bug that a ``top_k: int = 10`` default would introduce.
    """
    assert "top_k" not in _wire.search_body([1.0], top_k=None)
    assert _wire.search_body([1.0], top_k=0)["top_k"] == 0
    assert _wire.search_body([1.0], top_k=5)["top_k"] == 5


def test_search_body_sends_an_explicit_zero_min_score() -> None:
    """``min_score`` is ``Option<f32>`` on the server: omit when unset, never coerce to 0."""
    assert "min_score" not in _wire.search_body([1.0], min_score=None)
    assert _wire.search_body([1.0], min_score=0.0)["min_score"] == 0.0
    assert _wire.search_body([1.0], min_score=0.25)["min_score"] == 0.25


def test_search_body_paginates_with_an_optional_offset() -> None:
    """``offset`` is additive: omitted, the body is byte-identical to the pre-pagination one."""
    assert "offset" not in _wire.search_body([1.0], top_k=5)
    assert _wire.search_body([1.0], top_k=5, offset=0)["offset"] == 0
    assert _wire.search_body([1.0], top_k=5, offset=20)["offset"] == 20
    assert "offset" not in _wire.text_search_body("body", "fox")
    assert _wire.text_search_body("body", "fox", offset=3)["offset"] == 3
    assert "offset" not in _wire.hybrid_search_body([1.0], "body", "fox")
    assert _wire.hybrid_search_body([1.0], "body", "fox", offset=3)["offset"] == 3


def test_search_body_full() -> None:
    """Every option set, in the server's ``snake_case`` spelling."""
    assert _wire.search_body(
        [1.0, 0.0, 0.0],
        scope=["docs"],
        top_k=5,
        offset=10,
        min_score=0.1,
        filter=f.and_(f.eq("lang", "rust")),
    ) == {
        "query": [1.0, 0.0, 0.0],
        "scope": ["docs"],
        "top_k": 5,
        "offset": 10,
        "min_score": 0.1,
        "filter": [{"Eq": ["lang", {"Str": "rust"}]}],
    }


def test_projection_and_exact_are_omitted_unless_asked_for() -> None:
    """Both knobs are additive: unset, the body is what a pre-projection client sent."""
    body = _wire.search_body([1.0], top_k=5)
    assert "exact" not in body
    assert "include_attributes" not in body
    assert "exclude_attributes" not in body
    assert _wire.search_body([1.0], exact=True)["exact"] is True
    assert _wire.search_body([1.0], include_attributes=["title"])["include_attributes"] == ["title"]
    assert _wire.search_body([1.0], exclude_attributes=["body"])["exclude_attributes"] == ["body"]

    assert "include_attributes" not in _wire.list_body()
    assert _wire.list_body(include_attributes=["lang"])["include_attributes"] == ["lang"]
    assert _wire.list_body(exclude_attributes=["body"])["exclude_attributes"] == ["body"]


def test_both_projection_lists_at_once_is_refused_client_side() -> None:
    """The server answers 400; failing here instead names the argument at the call site."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _wire.search_body([1.0], include_attributes=["a"], exclude_attributes=["b"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        _wire.list_body(include_attributes=["a"], exclude_attributes=["b"])


def test_a_bare_string_projection_is_refused() -> None:
    """A ``str`` is a ``Sequence[str]``; unguarded it would project one attr per character."""
    with pytest.raises(TypeError, match="include_attributes"):
        _wire.search_body([1.0], include_attributes="title")  # type: ignore[arg-type]


def test_similar_body_omits_unset_optionals_but_keeps_scope_and_filter() -> None:
    """The default case: only ``collection``/``id``, plus the two empties."""
    assert _wire.similar_body("notes", "a") == {
        "collection": "notes",
        "id": "a",
        "scope": [],
        "filter": [],
    }


def test_similar_body_full() -> None:
    """Every option set, in the server's ``snake_case`` spelling."""
    assert _wire.similar_body(
        "notes",
        "a",
        scope=["notes"],
        top_k=5,
        offset=10,
        min_score=0.1,
        filter=f.and_(f.eq("lang", "rust")),
        exact=True,
    ) == {
        "collection": "notes",
        "id": "a",
        "scope": ["notes"],
        "top_k": 5,
        "offset": 10,
        "min_score": 0.1,
        "filter": [{"Eq": ["lang", {"Str": "rust"}]}],
        "exact": True,
    }


def test_similar_body_sends_an_explicit_zero_top_k() -> None:
    """``top_k=None`` is omitted; ``top_k=0`` is sent as ``0``, same rule as ``search_body``."""
    assert "top_k" not in _wire.similar_body("notes", "a", top_k=None)
    assert _wire.similar_body("notes", "a", top_k=0)["top_k"] == 0


def test_text_search_body() -> None:
    """BM25 over one field; ``min_score`` here is a raw BM25 floor, still optional."""
    assert _wire.text_search_body("body", "fox") == {
        "field": "body",
        "query": "fox",
        "scope": [],
        "filter": [],
    }
    full = _wire.text_search_body(
        "body", "fox", scope=["notes"], top_k=0, min_score=0.0, filter=[f.eq("kind", "a")]
    )
    assert full == {
        "field": "body",
        "query": "fox",
        "scope": ["notes"],
        "top_k": 0,
        "min_score": 0.0,
        "filter": [{"Eq": ["kind", {"Str": "a"}]}],
    }


def test_hybrid_search_body_has_no_min_score() -> None:
    """The score is a fused RRF rank, not a similarity, so the server offers no floor."""
    body = _wire.hybrid_search_body([1.0, 0.0, 0.0], "body", "fox")
    assert body == {
        "vector": [1.0, 0.0, 0.0],
        "field": "body",
        "text": "fox",
        "scope": [],
        "filter": [],
    }
    assert "min_score" not in body
    assert "rrf_k" not in body
    assert "candidates" not in body

    tuned = _wire.hybrid_search_body(
        [1.0], "body", "fox", scope=["notes"], top_k=5, rrf_k=30.0, candidates=200
    )
    assert tuned["rrf_k"] == 30.0
    assert tuned["candidates"] == 200


def test_ranking_knobs_are_omitted_unless_asked_for() -> None:
    """``rank_by``/``limit_per`` are additive: unset, the body is the pre-ranking one."""
    body = _wire.search_body([1.0], top_k=5)
    assert "rank_by" not in body
    assert "limit_per" not in body
    assert _wire.search_body([1.0], rank_by=rank.decay("ts", 1700000000000))["rank_by"] == {
        "Decay": {"field": "ts", "origin": 1700000000000}
    }
    assert _wire.search_body([1.0], limit_per={"field": "path", "max": 2})["limit_per"] == {
        "field": "path",
        "max": 2,
    }
    # And on the text route, which takes the same two.
    assert "rank_by" not in _wire.text_search_body("body", "fox")
    assert _wire.text_search_body("body", "fox", limit_per={"field": "path", "max": 1})[
        "limit_per"
    ] == {"field": "path", "max": 1}


def test_diversity_is_omitted_unless_asked_for_and_keeps_zero() -> None:
    """Absent, not null: an unset lambda leaves the request bytes unchanged. ``0.0`` is a
    meaningful lambda (pure variety), so ``prune`` must not drop it as falsy."""
    assert "diversity" not in _wire.search_body([1.0], top_k=5)
    assert "diversity" not in _wire.similar_body("docs", "d1")
    assert "diversity" not in _wire.text_search_body("body", "fox")
    assert "diversity" not in _wire.recall_body("why")
    assert _wire.search_body([1.0], diversity=0.0)["diversity"] == 0.0
    assert _wire.similar_body("docs", "d1", diversity=0.3)["diversity"] == 0.3
    assert _wire.text_search_body("body", "fox", diversity=0.5)["diversity"] == 0.5
    assert _wire.recall_body("why", diversity=1.0)["diversity"] == 1.0


def test_expand_is_omitted_unless_asked_for_and_defaults_its_field_names() -> None:
    """A bare ``radius`` sends only a radius; the server fills the reserved chunk attrs."""
    assert "expand" not in _wire.search_body([1.0], top_k=5)
    assert "expand" not in _wire.similar_body("docs", "d1")
    assert "expand" not in _wire.text_search_body("body", "fox")
    assert "expand" not in _wire.hybrid_search_body([1.0], "body", "fox")
    assert _wire.search_body([1.0], expand={"radius": 2})["expand"] == {"radius": 2}
    # ``0`` is a meaningful radius (the hit's own text as context), so prune must keep it.
    assert _wire.search_body([1.0], expand={"radius": 0})["expand"] == {"radius": 0}
    assert _wire.hybrid_search_body([1.0], "body", "fox", expand={"radius": 1})["expand"] == {
        "radius": 1
    }
    assert _wire.text_search_body(
        "body", "fox", expand={"radius": 1, "parent_field": "doc", "text_field": "body"}
    )["expand"] == {"radius": 1, "parent_field": "doc", "text_field": "body"}


def test_rollup_is_the_recall_spelling_and_is_omitted_unless_asked_for() -> None:
    """``recall`` takes the text-native ``rollup``, never the raw attr-name ``expand``."""
    assert "rollup" not in _wire.recall_body("why")
    assert _wire.recall_body("why", rollup={"neighbours": 1})["rollup"] == {"neighbours": 1}
    assert _wire.recall_body("why", rollup={"per_parent": 2, "neighbours": 0})["rollup"] == {
        "per_parent": 2,
        "neighbours": 0,
    }


def test_an_expand_must_name_a_radius_and_no_unknown_fields() -> None:
    """A misspelled field name would be *ignored* by serde and stitch the wrong window."""
    with pytest.raises(TypeError, match="missing required key"):
        _wire.search_body([1.0], expand={"parent_field": "doc"})  # type: ignore[typeddict-item]
    with pytest.raises(TypeError, match="unknown key"):
        _wire.search_body([1.0], expand={"radius": 1, "parent": "doc"})  # type: ignore[typeddict-unknown-key]
    with pytest.raises(TypeError, match="unknown key"):
        _wire.recall_body("q", rollup={"per_parents": 1})  # type: ignore[typeddict-unknown-key]


def test_a_hit_carries_its_context_only_when_the_server_sends_one() -> None:
    """Absent means ``None``, so an unexpanded hit is the object it always was."""
    hits = _wire.decode_hits(
        [
            {"collection": "c", "id": "d#1", "score": 0.9, "attrs": {}, "context": "widened"},
            {"collection": "c", "id": "d#2", "score": 0.8, "attrs": {}},
        ]
    )
    assert hits[0].context == "widened"
    assert hits[1].context is None


def test_a_limit_per_must_name_both_keys_and_no_others() -> None:
    """A missing ``max`` is a 400; a misspelled key would be *ignored* by serde instead."""
    with pytest.raises(TypeError, match="missing required key"):
        _wire.search_body([1.0], limit_per={"field": "path"})  # type: ignore[typeddict-item]
    with pytest.raises(TypeError, match="unknown key"):
        _wire.search_body([1.0], limit_per={"field": "path", "maximum": 2})  # type: ignore[typeddict-unknown-key]
    with pytest.raises(TypeError, match="limit_per"):
        _wire.search_body([1.0], limit_per="path")  # type: ignore[arg-type]


def test_rerank_is_sent_on_every_search_body() -> None:
    """The four rerankable builders all carry the same option shape under ``rerank``."""
    opts = {"query": "how do users sign in", "overscan": 4, "text_attr": "body"}
    assert _wire.search_body([1.0], rerank=opts)["rerank"] == opts
    assert _wire.text_search_body("body", "fox", rerank=opts)["rerank"] == opts
    assert _wire.hybrid_search_body([1.0], "body", "fox", rerank=opts)["rerank"] == opts
    assert _wire.recall_body("q", rerank=opts)["rerank"] == opts


def test_rerank_omitted_leaves_no_key() -> None:
    """Unset ``rerank`` is pruned, so an old server sees a byte-identical body."""
    assert "rerank" not in _wire.search_body([1.0])
    assert "rerank" not in _wire.text_search_body("body", "fox")
    assert "rerank" not in _wire.hybrid_search_body([1.0], "body", "fox")
    assert "rerank" not in _wire.recall_body("q")


def test_empty_rerank_is_sent_as_an_empty_dict() -> None:
    """``rerank={}`` is the valid minimal form; if ``prune`` collapsed it, this would fail."""
    assert _wire.search_body([1.0], rerank={})["rerank"] == {}
    assert _wire.text_search_body("body", "fox", rerank={})["rerank"] == {}
    assert _wire.hybrid_search_body([1.0], "body", "fox", rerank={})["rerank"] == {}
    assert _wire.recall_body("q", rerank={})["rerank"] == {}


def test_rerank_rejects_an_unknown_key() -> None:
    """The stale ``text_field`` spelling from the earlier blueprint draft must not pass."""
    with pytest.raises(TypeError, match="unknown key"):
        _wire.search_body([1.0], rerank={"text_field": "body"})  # type: ignore[typeddict-unknown-key]


def test_text_search_body_takes_several_clauses_with_a_combine_rule() -> None:
    """The multi-field spelling: one text per field, folded by ``combine``."""
    assert _wire.text_search_body(
        clauses=[{"field": "title", "query": "rust"}, {"field": "body", "query": "async runtime"}],
        combine="Max",
    ) == {
        "clauses": [
            {"field": "title", "query": "rust"},
            {"field": "body", "query": "async runtime"},
        ],
        "combine": "Max",
        "scope": [],
        "filter": [],
    }
    # `combine` is the server's when unset, and a clause's text key is `query` on both routes.
    assert "combine" not in _wire.text_search_body(clauses=[{"field": "t", "query": "x"}])
    assert _wire.hybrid_search_body([1.0], clauses=[{"field": "t", "query": "x"}])["clauses"] == [
        {"field": "t", "query": "x"}
    ]


def test_the_single_field_spelling_is_byte_identical_to_the_one_before_clauses() -> None:
    """The compatibility contract: an existing call must not change what it sends."""
    assert _wire.text_search_body("body", "fox") == {
        "field": "body",
        "query": "fox",
        "scope": [],
        "filter": [],
    }
    assert _wire.hybrid_search_body([1.0], "body", "fox") == {
        "vector": [1.0],
        "field": "body",
        "text": "fox",
        "scope": [],
        "filter": [],
    }


@pytest.mark.parametrize(
    "build",
    [
        lambda **kw: _wire.text_search_body(**kw),
        lambda **kw: _wire.hybrid_search_body([1.0], **kw),
    ],
    ids=["text_search", "hybrid_search"],
)
def test_a_text_query_must_use_exactly_one_of_the_two_spellings(
    build: Callable[..., object],
) -> None:
    """Both, neither, half of one, or an empty list are refused before the request.

    The server refuses them too — it must, since it answers other clients — but an empty
    result would otherwise read as "no matches" when it means "no query", and failing here
    names the argument at the call site instead of returning a 400 about the body.
    """
    one = [{"field": "t", "query": "x"}]
    with pytest.raises(ValueError, match="mutually exclusive"):
        build(field="body", clauses=one)
    with pytest.raises(ValueError, match="must not be empty"):
        build(clauses=[])
    with pytest.raises(ValueError, match="clauses list"):
        build()
    with pytest.raises(ValueError, match="sent together"):
        build(field="body")


def test_prefix_is_omitted_from_the_body_unless_set() -> None:
    """An unset ``prefix`` never serializes as ``false`` — it stays absent on both routes."""
    assert "prefix" not in _wire.text_search_body("body", "ru")
    assert "prefix" not in _wire.hybrid_search_body([1.0], "body", "ru")
    assert _wire.text_search_body("body", "ru", prefix=True)["prefix"] is True
    assert _wire.hybrid_search_body([1.0], "body", "ru", prefix=True)["prefix"] is True


def test_prefix_on_a_clause_survives_the_body_builder_untouched() -> None:
    """A clause's own ``prefix`` passes through ``_spec`` like any other optional key."""
    body = _wire.text_search_body(clauses=[{"field": "title", "query": "ru", "prefix": True}])
    assert body["clauses"] == [{"field": "title", "query": "ru", "prefix": True}]
    # The top-level shorthand `prefix` plays no part once `clauses` is sent.
    assert "prefix" not in _wire.hybrid_search_body(
        [1.0], clauses=[{"field": "title", "query": "ru"}]
    )


def test_a_clause_must_be_a_mapping_naming_a_field_and_its_query() -> None:
    """``text`` is the *single-field* key on the hybrid route; a clause always says ``query``."""
    with pytest.raises(TypeError, match="unknown key"):
        _wire.hybrid_search_body([1.0], clauses=[{"field": "t", "text": "x"}])  # type: ignore[typeddict-unknown-key]
    with pytest.raises(TypeError, match="missing required key"):
        _wire.text_search_body(clauses=[{"field": "t"}])  # type: ignore[typeddict-item]
    with pytest.raises(TypeError, match="expects a mapping"):
        _wire.text_search_body(clauses=["body"])  # type: ignore[list-item]


def test_explain_and_highlight_are_omitted_unless_asked_for() -> None:
    """Both are additive, and ``highlight=True`` is the empty object that means "defaults"."""
    body = _wire.text_search_body("body", "fox")
    assert "explain" not in body
    assert "highlight" not in body
    assert _wire.text_search_body("body", "fox", explain=True)["explain"] is True
    assert _wire.text_search_body("body", "fox", highlight=True)["highlight"] == {}
    assert _wire.text_search_body("body", "fox", highlight={"max_fragments": 3})["highlight"] == {
        "max_fragments": 3
    }
    # `False` means the same as unset — no annotations were asked for.
    assert "highlight" not in _wire.text_search_body("body", "fox", highlight=False)
    assert _wire.hybrid_search_body([1.0], "body", "fox", explain=True, highlight=True) == {
        "vector": [1.0],
        "field": "body",
        "text": "fox",
        "scope": [],
        "filter": [],
        "explain": True,
        "highlight": {},
    }


def test_a_misspelled_highlight_knob_is_refused() -> None:
    """The server ignores unknown keys, so a typo would silently return default fragments."""
    with pytest.raises(TypeError, match="unknown key"):
        _wire.text_search_body("body", "fox", highlight={"maxFragments": 3})  # type: ignore[typeddict-unknown-key]


def test_hybrid_search_body_weights_each_leg_only_when_asked() -> None:
    """Both weights default to 1.0 server-side, which is the unweighted fusion exactly."""
    body = _wire.hybrid_search_body([1.0], "body", "fox")
    assert "vector_weight" not in body
    assert "text_weight" not in body
    weighted = _wire.hybrid_search_body([1.0], "body", "fox", vector_weight=2.0, text_weight=0.0)
    assert weighted["vector_weight"] == 2.0
    # An explicit zero mutes the leg, and is a real value rather than "unset".
    assert weighted["text_weight"] == 0.0


def test_text_search_body_projects_attributes_like_search_does() -> None:
    """``/text-search`` grew the same two projection lists, with the same exclusivity rule."""
    assert _wire.text_search_body("body", "fox", include_attributes=["title"])[
        "include_attributes"
    ] == ["title"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        _wire.text_search_body("body", "fox", include_attributes=["a"], exclude_attributes=["b"])


def test_list_body_orders_by_an_attribute_only_when_asked() -> None:
    """``descending`` is optional (ascending by default) and must survive as ``False``."""
    assert "order_by" not in _wire.list_body()
    assert _wire.list_body(order_by={"field": "ts"})["order_by"] == {"field": "ts"}
    assert _wire.list_body(order_by={"field": "ts", "descending": True})["order_by"] == {
        "field": "ts",
        "descending": True,
    }
    assert _wire.list_body(order_by={"field": "ts", "descending": False})["order_by"] == {
        "field": "ts",
        "descending": False,
    }
    # `desc` would be ignored by serde and sort ascending with a 200.
    with pytest.raises(TypeError, match="unknown key"):
        _wire.list_body(order_by={"field": "ts", "desc": True})  # type: ignore[typeddict-unknown-key]


def test_aggregate_body() -> None:
    """``scope`` and ``filter`` are always sent; ``sum`` is pruned, which means "count only"."""
    assert _wire.aggregate_body() == {"scope": [], "filter": []}
    assert _wire.aggregate_body(
        scope=["docs"], filter=[f.eq("lang", "rust")], sum=["bytes", "lines"]
    ) == {
        "scope": ["docs"],
        "filter": [{"Eq": ["lang", {"Str": "rust"}]}],
        "sum": ["bytes", "lines"],
    }
    # An explicitly empty sum list is a real (if inert) value and is sent.
    assert _wire.aggregate_body(sum=[])["sum"] == []


def test_list_body_omits_an_unset_limit_and_sends_an_explicit_zero() -> None:
    """Same omit-vs-zero rule as ``top_k``, for ``limit`` and ``offset``."""
    assert _wire.list_body() == {"scope": [], "filter": []}
    assert "limit" not in _wire.list_body(offset=10)
    assert _wire.list_body(limit=0)["limit"] == 0
    assert _wire.list_body(offset=0)["offset"] == 0
    assert _wire.list_body(scope=["docs"], offset=10, limit=50, filter=[f.eq("k", "a")]) == {
        "scope": ["docs"],
        "offset": 10,
        "limit": 50,
        "filter": [{"Eq": ["k", {"Str": "a"}]}],
    }


def test_remember_body_prunes_mode_and_attrs() -> None:
    """Unlike a record's, ``remember``'s ``attrs`` *is* ``#[serde(default)]`` server-side.

    So omitting it is well-defined here, and omitting ``mode`` leaves the server's ``"raw"``.
    """
    assert _wire.remember_body("a", "the quick brown fox") == {
        "id": "a",
        "text": "the quick brown fox",
    }
    assert _wire.remember_body("b", "a long article", mode="summarize") == {
        "id": "b",
        "text": "a long article",
        "mode": "summarize",
    }
    assert _wire.remember_body("c", "t", attrs={"tag": "x", "year": 2024}) == {
        "id": "c",
        "text": "t",
        "attrs": {"tag": {"Str": "x"}, "year": {"Int": 2024}},
    }
    # An explicitly empty attrs map is a real (if inert) value and is sent as `{}`.
    assert _wire.remember_body("d", "t", attrs={})["attrs"] == {}


def test_remember_body_ttl_and_dedupe_prune_on_none_but_send_a_zero() -> None:
    """Zero is a real request for both: expire immediately, and match any entry at all."""
    assert _wire.remember_body("a", "t", ttl_seconds=3600, dedupe_threshold=0.95) == {
        "id": "a",
        "text": "t",
        "ttl_seconds": 3600,
        "dedupe_threshold": 0.95,
    }
    assert _wire.remember_body("a", "t", ttl_seconds=0, dedupe_threshold=0.0) == {
        "id": "a",
        "text": "t",
        "ttl_seconds": 0,
        "dedupe_threshold": 0.0,
    }
    body = _wire.remember_body("a", "t")
    assert "ttl_seconds" not in body
    assert "dedupe_threshold" not in body


def test_decode_remember_reports_the_record_actually_written() -> None:
    """On a dedupe match the server writes a different record than the one asked for."""
    assert _wire.decode_remember(
        {"ok": True, "upserted": 1, "id": "older", "deduped": True}, "newer"
    ) == RememberResult(id="older", upserted=1, deduped=True)
    # A server predating the echoed fields answers `{ok, upserted}`; naming the record it
    # did write beats reporting an empty id.
    assert _wire.decode_remember({"ok": True, "upserted": 1}, "a") == RememberResult(
        id="a", upserted=1, deduped=False
    )


def test_recall_body() -> None:
    """Query text in; the usual optional trio pruned, ``filter`` always present."""
    assert _wire.recall_body("quick fox") == {"query": "quick fox", "filter": []}
    assert _wire.recall_body("q", top_k=5, min_score=0.2, filter=[f.eq("tag", "x")]) == {
        "query": "q",
        "top_k": 5,
        "min_score": 0.2,
        "filter": [{"Eq": ["tag", {"Str": "x"}]}],
    }


def test_recall_body_omits_reinforce_when_false() -> None:
    """The compatibility promise: an unset (or explicitly ``False``) ``reinforce`` must
    leave the key absent, not send ``False`` — a byte-identical body for an old server."""
    assert "reinforce" not in _wire.recall_body("why")
    assert "reinforce" not in _wire.recall_body("why", reinforce=False)
    assert "extend_ttl_seconds" not in _wire.recall_body("why")
    assert _wire.recall_body("why", reinforce=True)["reinforce"] is True
    assert _wire.recall_body("why", extend_ttl_seconds=3600)["extend_ttl_seconds"] == 3600


def test_fts_schema_body() -> None:
    assert _wire.fts_schema_body(["body", "title"]) == {"fields": ["body", "title"]}


def test_fts_schema_body_carries_per_field_tuning() -> None:
    """A mapping travels as an object; the two forms mix in one call."""
    assert _wire.fts_schema_body(
        ["title", {"field": "body", "k1": 1.5, "ascii_folding": True, "max_token_len": 40}]
    ) == {
        "fields": [
            "title",
            {"field": "body", "k1": 1.5, "ascii_folding": True, "max_token_len": 40},
        ]
    }
    # An unset knob is omitted, so a knob-less mapping is the bare name in object form.
    assert _wire.fts_schema_body([{"field": "body"}]) == {"fields": [{"field": "body"}]}
    # An explicit zero is a real value and must survive (b = 0 disables normalization).
    assert _wire.fts_schema_body([{"field": "body", "b": 0.0}]) == {
        "fields": [{"field": "body", "b": 0.0}]
    }


def test_fts_schema_body_refuses_a_misspelled_knob() -> None:
    """The server ignores unknown keys, so a typo would index with folding silently off."""
    with pytest.raises(TypeError, match="asciiFolding"):
        _wire.fts_schema_body([{"field": "body", "asciiFolding": True}])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="'field' key"):
        _wire.fts_schema_body([{"k1": 1.5}])  # type: ignore[list-item]


def test_suggest_body_carries_field_and_prefix() -> None:
    assert _wire.suggest_body("body", "nid") == {"field": "body", "prefix": "nid"}


def test_suggest_body_omits_limit_when_unset() -> None:
    """The client must not fork the server's own default of 10."""
    body = _wire.suggest_body("body", "nid")
    assert "limit" not in body
    assert _wire.suggest_body("body", "nid", limit=5) == {
        "field": "body",
        "prefix": "nid",
        "limit": 5,
    }


def test_decode_suggestions_decodes_terms_and_df() -> None:
    result = _wire.decode_suggestions({"suggestions": [{"term": "nidus", "df": 3}], "matched": 1})
    assert result.matched == 1
    assert len(result.suggestions) == 1
    assert result.suggestions[0].term == "nidus"
    assert result.suggestions[0].df == 3


def test_filter_index_body() -> None:
    assert _wire.filter_index_body(["body", "title"]) == {"fields": ["body", "title"]}
    assert _wire.filter_index_body([]) == {"fields": []}


def test_filter_index_body_carries_per_field_structures() -> None:
    """A mapping travels as an object; the two forms mix in one call."""
    assert _wire.filter_index_body(["title", {"field": "body", "trigrams": False}]) == {
        "fields": ["title", {"field": "body", "trigrams": False}]
    }
    # An unset knob is omitted, so a knob-less mapping is the bare name in object form.
    assert _wire.filter_index_body([{"field": "body"}]) == {"fields": [{"field": "body"}]}
    # Explicit False must survive: both structures default to on server side, so sending
    # nothing and sending False mean opposite things.
    assert _wire.filter_index_body([{"field": "body", "tokens": False}]) == {
        "fields": [{"field": "body", "tokens": False}]
    }


def test_filter_index_body_refuses_a_misspelled_knob() -> None:
    """Both knobs default to on, so a typo would leave the structure enabled and say ok."""
    with pytest.raises(TypeError, match="trigram"):
        _wire.filter_index_body([{"field": "body", "trigram": False}])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="'field' key"):
        _wire.filter_index_body([{"tokens": True}])  # type: ignore[list-item]


def test_meta_body_is_the_bare_map() -> None:
    """The meta handler deserializes a bare ``BTreeMap<String, String>``, not a wrapper."""
    assert _wire.meta_body({"owner": "search-team"}) == {"owner": "search-team"}
    # A copy, so a later mutation of the caller's mapping cannot change what was sent.
    caller = {"owner": "a"}
    body = _wire.meta_body(caller)
    caller["owner"] = "b"
    assert body == {"owner": "a"}


# ── The bare-string slip ─────────────────────────────────────────────────────────────
#
# `str` IS a `Sequence[str]`, so every one of these calls passes `mypy --strict` and, before
# the guards, produced a well-formed request with the wrong contents that the server
# answered with a 200. All four were confirmed against the real binary: `delete("docs",
# "x1")` deleted the record whose id is `1` and returned a reassuring 1; `scope="docs2"`
# searched five collections that do not exist and returned `[]`; `fields="body"` indexed
# four one-character fields, after which `text_search(field="body")` returned nothing
# forever. There is no error anywhere in that stack, which is why these must raise.


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda: _wire.delete_ids_body("x1"), "delete(name, ids)"),
        (lambda: _wire.fts_schema_body("body"), "set_fts_schema(name, fields)"),
        (lambda: _wire.search_body([1.0], scope="docs"), "scope"),
        (lambda: _wire.text_search_body("body", "q", scope="docs"), "scope"),
        (lambda: _wire.hybrid_search_body([1.0], "body", "q", scope="docs"), "scope"),
        (lambda: _wire.list_body(scope="docs"), "scope"),
        (lambda: _wire.aggregate_body(scope="docs"), "scope"),
        (lambda: _wire.aggregate_body(sum="bytes"), "aggregate(sum=...)"),
    ],
)
def test_a_bare_string_is_refused_where_a_sequence_is_meant(
    call: Callable[[], object], expected: str
) -> None:
    """The message names the parameter and shows the one-line fix at the call site."""
    with pytest.raises(TypeError, match=re.escape(expected)) as caught:
        call()
    assert "did you mean [" in str(caught.value)


def test_a_real_sequence_of_one_is_of_course_fine() -> None:
    """The guard is about the *type*, not the length — one-element lists are the norm."""
    assert _wire.delete_ids_body(["x1"]) == {"ids": ["x1"]}
    assert _wire.fts_schema_body(["body"]) == {"fields": ["body"]}
    assert _wire.list_body(scope=["docs"])["scope"] == ["docs"]
    # A tuple is a sequence too, and JSON has no tuple, so it is copied to a list.
    assert _wire.delete_ids_body(("a", "b")) == {"ids": ["a", "b"]}


# ── Vector coercion ──────────────────────────────────────────────────────────────────


class _Float32:
    """A stand-in for ``numpy.float32``: converts to ``float``, is not a ``float``.

    Exactly the property that makes this bite — ``np.float64`` *is* a ``float`` subclass and
    serializes fine, ``np.float32`` is not and does not, and ``list(np.asarray(x,
    dtype=np.float32))`` hands you the latter. Hand-rolled because numpy is not (and must
    not become) a test dependency of a zero-dependency SDK.
    """

    def __init__(self, value: float) -> None:
        self._value = value

    def __float__(self) -> float:
        return self._value


@pytest.mark.parametrize(
    "build",
    [
        lambda vec: _wire.search_body(vec),
        lambda vec: _wire.hybrid_search_body(vec, "body", "q"),
        lambda vec: _wire.upsert_body([{"id": "a", "vector": vec}]),
    ],
)
def test_vector_elements_are_coerced_so_json_can_serialize_them(
    build: Callable[[object], object],
) -> None:
    """Without the coercion this dies inside ``json.dumps``, naming neither nidus nor the
    argument — and the fix (``.tolist()``) is undiscoverable from that traceback."""
    for vec in ([_Float32(0.5), _Float32(-1.5)], [Decimal("0.5"), Decimal("-1.5")]):
        body = build(vec)
        # The real assertion is that this survives the trip to bytes, not just to a dict.
        assert b"0.5" in _wire.encode_body(body)


def test_a_non_numeric_vector_element_names_itself() -> None:
    """A ``TypeError`` naming the offending value beats one naming the json encoder."""
    with pytest.raises(TypeError, match="search"):
        _wire.search_body(["not a number"])
    with pytest.raises(TypeError, match="record 'a' vector"):
        _wire.upsert_body([{"id": "a", "vector": [object()]}])


def test_empty_body_is_a_fresh_dict_each_call() -> None:
    """A shared module constant could be mutated by one caller and seen by the next."""
    first = _wire.empty_body()
    assert first == {}
    first["mutated"] = True
    assert _wire.empty_body() == {}


def test_encode_body_is_compact_utf8_json() -> None:
    """One canonical serialization, so both clients put identical bytes on the wire."""
    assert _wire.encode_body({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'
    assert _wire.encode_body({"k": "nøte"}) == b'{"k":"n\\u00f8te"}'


# ── Response decoding ────────────────────────────────────────────────────────────────


def test_is_success_is_the_whole_2xx_rule() -> None:
    """One place decides what an error is, so the two clients cannot disagree."""
    assert _wire.is_success(200)
    assert _wire.is_success(204)
    assert _wire.is_success(299)
    assert not _wire.is_success(199)
    assert not _wire.is_success(300)
    assert not _wire.is_success(400)
    assert not _wire.is_success(0)


def test_decode_response_parses_a_success_and_raises_the_servers_own_message() -> None:
    """The sync and async ``_request`` are both this function plus their own transport."""
    assert _wire.decode_response(200, '{"upserted":2}') == {"upserted": 2}
    assert _wire.decode_response(200, "") is None
    with pytest.raises(NidusError) as caught:
        _wire.decode_response(400, json.dumps({"error": "dimension mismatch"}))
    assert caught.value.status == 400
    assert caught.value.message == "dimension mismatch"


def test_parse_body_treats_an_empty_body_as_none() -> None:
    """Several write endpoints answer with nothing anybody reads; that is not an error."""
    assert _wire.parse_body("") is None
    assert _wire.parse_body('{"ok":true}') == {"ok": True}


def test_decode_hits_decodes_attrs_to_plain_values() -> None:
    """Callers get ``"rust"``, not ``{"Str": "rust"}`` — matching the JS SDK."""
    hits = _wire.decode_hits(
        [
            {
                "collection": "docs",
                "id": "a",
                "score": 0.9,
                "attrs": {"lang": {"Str": "rust"}, "year": {"Int": 2024}, "note": "Null"},
            }
        ]
    )
    assert len(hits) == 1
    assert hits[0].collection == "docs"
    assert hits[0].id == "a"
    assert hits[0].score == pytest.approx(0.9)
    assert hits[0].attrs == {"lang": "rust", "year": 2024, "note": None}


def test_decode_hits_tolerates_an_empty_or_missing_payload() -> None:
    """No hits is the common case, and ``None`` shows up when a body was empty."""
    assert _wire.decode_hits([]) == []
    assert _wire.decode_hits(None) == []
    # A hit with no attrs map at all decodes to an empty one rather than raising.
    assert _wire.decode_hits([{"collection": "d", "id": "a", "score": 1.0}])[0].attrs == {}


def test_decode_hits_leaves_annotations_none_when_the_server_sent_none() -> None:
    """The default response has no ``annotations`` key at all, and must decode to ``None``.

    ``None`` rather than an empty :class:`~nidus.Annotations`, because "nothing was
    explained" and "explained, and every part came back empty" are different facts.
    """
    hits = _wire.decode_hits([{"collection": "d", "id": "a", "score": 1.0, "attrs": {}}])
    assert hits[0].annotations is None
    assert _wire.decode_annotations(None) is None


def test_decode_hits_reads_every_part_of_an_annotation() -> None:
    """The whole annotated shape, spelled as ``src/annotate.rs`` serializes it."""
    hits = _wire.decode_hits(
        [
            {
                "collection": "docs",
                "id": "a",
                "score": 0.5,
                "attrs": {},
                "annotations": {
                    "vector": {"rank": 0, "score": 0.98},
                    "text": {"rank": 2, "score": 1.5},
                    "clauses": [{"field": "title", "score": 0.49}],
                    "highlights": [
                        {
                            "field": "body",
                            "fragments": [{"text": "we were running", "spans": [[8, 15]]}],
                        }
                    ],
                },
            }
        ]
    )
    annotations = hits[0].annotations
    assert annotations is not None
    assert annotations.vector is not None
    assert (annotations.vector.rank, annotations.vector.score) == (0, pytest.approx(0.98))
    assert annotations.text is not None
    assert annotations.text.rank == 2
    assert annotations.clauses[0].field == "title"
    assert annotations.clauses[0].score == pytest.approx(0.49)
    fragment = annotations.highlights[0].fragments[0]
    assert annotations.highlights[0].field == "body"
    assert fragment.text == "we were running"
    # Tuples, and byte offsets into `text` — so this is how the matched run is recovered.
    assert fragment.spans == [(8, 15)]
    assert fragment.text.encode()[8:15] == b"running"


def test_decode_annotations_tolerates_a_partial_object() -> None:
    """A text search has one leg and may have no highlights; the absent parts stay empty.

    The server skips ``clauses``/``highlights`` when empty and both legs unless the query
    was hybrid, so every part has to survive being missing rather than raising.
    """
    annotations = _wire.decode_annotations({"clauses": [{"field": "body", "score": 1.25}]})
    assert annotations is not None
    assert annotations.vector is None
    assert annotations.text is None
    assert annotations.highlights == []
    assert annotations.clauses[0].score == pytest.approx(1.25)
    # And a highlight with no fragments is a field that matched nothing, not an error.
    empty = _wire.decode_annotations({"highlights": [{"field": "body"}]})
    assert empty is not None
    assert empty.highlights[0].fragments == []


def test_decode_aggregation_keeps_the_servers_int_versus_float() -> None:
    """A sum is a tagged ``Value``, so the Python type it decodes to is the server's own."""
    out = _wire.decode_aggregation(
        {"count": 12, "sums": {"bytes": {"Int": 40960}, "seconds": {"Float": 1.5}}}
    )
    assert out.count == 12
    assert out.sums == {"bytes": 40960, "seconds": 1.5}
    assert isinstance(out.sums["bytes"], int)
    # `2.0` must stay a float: `Int` and `Float` are separate types the store compares apart.
    whole = _wire.decode_aggregation({"count": 1, "sums": {"x": {"Float": 2.0}}})
    assert isinstance(whole.sums["x"], float)


def test_decode_aggregation_without_sums_is_just_a_count() -> None:
    """``sum: []`` (or no ``sum`` at all) still answers with the count."""
    assert _wire.decode_aggregation({"count": 0, "sums": {}}) == _wire.decode_aggregation(
        {"count": 0}
    )


def test_decode_aggregation_names_a_missing_body_instead_of_raising_a_key_error() -> None:
    """Same rule as ``/stats``: a count of nothing and "no answer" are different facts."""
    with pytest.raises(NidusError, match="/aggregate") as caught:
        _wire.decode_aggregation(None)
    assert caught.value.is_transport_error


def test_decode_records_keeps_an_absent_vector_as_none() -> None:
    """``None`` (text-only doc) must stay distinguishable from ``[]``."""
    records = _wire.decode_records(
        [
            {"id": "a", "vector": [1.0, 0.0, 0.0], "attrs": {"lang": {"Str": "rust"}}},
            {"id": "b", "attrs": {"body": {"Str": "text only"}}},
            {"id": "c", "vector": [], "attrs": {}},
        ]
    )
    assert records[0].vector == [1.0, 0.0, 0.0]
    assert records[1].vector is None
    assert records[2].vector == []
    assert records[1].attrs == {"body": "text only"}


def test_decode_stats_maps_a_null_ann_to_none() -> None:
    """``ann: null`` means exact brute-force search — not "an index with defaults"."""
    stats = _wire.decode_stats(EXACT_STATS)
    assert stats.ann is None
    assert stats.dimension == 3
    assert stats.distance == "Cosine"
    assert stats.collections == ["docs", "notes"]
    assert stats.footprint.rows == 4
    assert stats.footprint.dead_rows == 1
    assert stats.footprint.vector_bytes == 48
    assert stats.footprint.doc_count == 3


def test_decode_stats_hnsw_sets_only_the_hnsw_knobs() -> None:
    """The server omits the inert knobs, so the absent ones decode to ``None``."""
    ann = _wire.decode_stats(HNSW_STATS).ann
    assert ann is not None
    assert ann.kind == "Hnsw"
    assert (ann.m, ann.ef_construction, ann.ef_search) == (16, 200, 64)
    assert ann.n_lists is None
    assert ann.n_probe is None


def test_decode_stats_ivf_sets_only_the_ivf_knobs() -> None:
    ann = _wire.decode_stats(IVF_STATS).ann
    assert ann is not None
    assert ann.kind == "Ivf"
    assert (ann.n_lists, ann.n_probe) == (64, 8)
    assert ann.m is None
    assert ann.ef_construction is None
    assert ann.ef_search is None


def test_decode_stats_names_a_missing_body_instead_of_leaking_an_attribute_error() -> None:
    """Every ``Stats`` field is required, so there is nothing honest to build from ``None``.

    ``parse_body`` maps an empty body to ``None`` (a proxy or a stripped 204 can produce
    one). The list-shaped decoders tolerate that; this one cannot, so it must fail as the
    SDK's own error type rather than as an ``AttributeError`` from a private module.
    """
    with pytest.raises(NidusError, match="/stats") as caught:
        _wire.decode_stats(None)
    assert caught.value.is_transport_error


def test_decode_versions_names_a_missing_body_instead_of_leaking_an_attribute_error() -> None:
    """Every ``StoreVersions`` field is required, so a non-object payload is malformed."""
    with pytest.raises(NidusError, match="/versions") as caught:
        _wire.decode_versions(None)
    assert caught.value.is_transport_error


def test_decode_collections_and_meta() -> None:
    assert _wire.decode_collections(["docs", "notes"]) == ["docs", "notes"]
    assert _wire.decode_collections(None) == []
    assert _wire.decode_meta({"owner": "austin"}) == {"owner": "austin"}
    assert _wire.decode_meta(None) == {}


def test_write_counts_are_decoded_by_name_and_default_to_zero() -> None:
    """One decoder per response key, so the key names live here and not in two clients."""
    assert _wire.decode_upserted({"upserted": 2}) == 2
    assert _wire.decode_deleted({"deleted": 1}) == 1
    # A body that says nothing about the count reads as 0 rather than raising.
    assert _wire.decode_upserted({}) == 0
    assert _wire.decode_deleted(None) == 0


# ── Error extraction ─────────────────────────────────────────────────────────────────


def test_extract_error_pulls_the_message_out_of_the_error_body() -> None:
    """The normal case: the server's own ``{"error": …}``, which is the useful text."""
    body = json.dumps({"error": "dimension mismatch: expected 3, got 4"})
    assert _wire.extract_error(body, 400) == "dimension mismatch: expected 3, got 4"


def test_extract_error_falls_back_to_a_non_json_body_verbatim() -> None:
    """A proxy's HTML error page is more useful shown than replaced with a status line."""
    assert (
        _wire.extract_error("<html>502 Bad Gateway</html>", 502) == "<html>502 Bad Gateway</html>"
    )
    # JSON that simply is not the error shape is handed back the same way.
    assert _wire.extract_error("[1,2,3]", 500) == "[1,2,3]"
    assert _wire.extract_error('{"detail":"nope"}', 500) == '{"detail":"nope"}'


def test_extract_error_falls_back_to_the_status_for_an_empty_body() -> None:
    """With no body there is nothing left but the status, so say that and no more."""
    assert _wire.extract_error("", 404) == "HTTP 404"
    assert _wire.extract_error("", 0) == "HTTP 0"


# ── Connection-independent request setup ─────────────────────────────────────────────


def test_prepare_pairs_the_encoded_body_with_the_headers_that_describe_it() -> None:
    """``content-type`` is a function of whether there is a body, so the two go together.

    ``body=None`` means *no body*, not a JSON ``null``: no endpoint accepts a bare null, and
    the bodyless writes send ``{}`` explicitly.
    """
    payload, headers = _wire.prepare("t", {"x-trace": "1"}, {"a": 1})
    assert payload == b'{"a":1}'
    assert headers == {
        "x-trace": "1",
        "authorization": "Bearer t",
        "content-type": "application/json",
    }

    payload, headers = _wire.prepare(None, None, None)
    assert payload is None
    assert headers == {}

    # `{}` is a real (empty) body, so it is encoded and does get a content-type.
    payload, headers = _wire.prepare(None, None, _wire.empty_body())
    assert payload == b"{}"
    assert headers == {"content-type": "application/json"}


def test_transport_error_is_the_status_zero_sentinel_worded_once() -> None:
    """The wording is a cross-SDK contract, so it lives here rather than in two clients."""
    err = _wire.transport_error("/search", OSError("Connection refused"))
    assert isinstance(err, NidusError)
    assert err.status == 0
    assert err.is_transport_error
    assert str(err) == "request to /search failed: Connection refused"


def test_normalize_base_url_strips_trailing_slashes() -> None:
    """Otherwise path concatenation produces ``http://x//stats``."""
    assert _wire.normalize_base_url("http://x/") == "http://x"
    assert _wire.normalize_base_url("http://x///") == "http://x"
    assert _wire.normalize_base_url("http://x") == "http://x"
    assert _wire.normalize_base_url("http://x/prefix/") == "http://x/prefix"


def test_normalize_base_url_rejects_an_empty_base_url() -> None:
    """A silently-relative URL would fail much later and much less clearly."""
    with pytest.raises(ValueError, match="base_url"):
        _wire.normalize_base_url("")


def test_request_headers_owns_auth_and_content_type() -> None:
    """Caller extras go first so they cannot unset the token or the content type."""
    assert _wire.request_headers(None, None, has_body=False) == {}
    assert _wire.request_headers("sekret", None, has_body=False) == {
        "authorization": "Bearer sekret"
    }
    assert _wire.request_headers(None, None, has_body=True) == {"content-type": "application/json"}
    assert _wire.request_headers("t", {"x-trace": "1"}, has_body=True) == {
        "x-trace": "1",
        "authorization": "Bearer t",
        "content-type": "application/json",
    }
    # A caller trying to override auth loses, deliberately.
    assert _wire.request_headers("t", {"authorization": "Bearer other"}, has_body=False) == {
        "authorization": "Bearer t"
    }


def test_request_headers_does_not_alias_the_caller_mapping() -> None:
    """The returned dict is ours to add to; the caller's must not grow a token."""
    extra = {"x-trace": "1"}
    _wire.request_headers("t", extra, has_body=True)
    assert extra == {"x-trace": "1"}


# ── Query plans (`*_with_plan`) ──────────────────────────────────────────────────────


def test_plan_is_omitted_from_every_body_unless_asked_for() -> None:
    """The three plan-capable bodies never send the key unless a ``_with_plan`` method asks."""
    assert "plan" not in _wire.search_body([1.0])
    assert "plan" not in _wire.similar_body("notes", "a")
    assert "plan" not in _wire.hybrid_search_body([1.0], "body", "fox")
    assert _wire.search_body([1.0], plan=True)["plan"] is True
    assert _wire.similar_body("notes", "a", plan=True)["plan"] is True
    assert _wire.hybrid_search_body([1.0], "body", "fox", plan=True)["plan"] is True


PLAN_PAYLOAD = {
    "path": "ann_prefilter_fallback",
    "rows_scanned": 1234,
    "candidates": {
        "surfaced": 100,
        "survived": 12,
        "dropped_out_of_scope": 0,
        "dropped_stale": 0,
        "dropped_filtered": 88,
        "dropped_min_score": 0,
    },
    "narrowing": {"state": "narrowed", "candidates": 42},
    "timings": {
        "narrow_us": 12,
        "gather_us": 300,
        "walk_us": 900,
        "resolve_us": 50,
        "score_us": 20,
        "total_us": 1300,
    },
}


def test_decode_plan_decodes_every_part() -> None:
    plan = _wire.decode_plan(PLAN_PAYLOAD)
    assert plan.path == "ann_prefilter_fallback"
    assert plan.rows_scanned == 1234
    assert plan.candidates is not None
    assert plan.candidates.surfaced == 100
    assert plan.candidates.dropped_filtered == 88
    assert plan.narrowing.state == "narrowed"
    assert plan.narrowing.candidates == 42
    assert plan.timings.total_us == 1300
    assert plan.timings.walk_us == 900
    assert plan.timings.first_pass_us is None
    assert plan.timings.rescore_us is None


def test_decode_plan_treats_an_unknown_path_as_an_open_string() -> None:
    """A newer server may add a path this SDK has never heard of; it must not raise."""
    plan = _wire.decode_plan({**PLAN_PAYLOAD, "path": "some_future_path"})
    assert plan.path == "some_future_path"


def test_decode_plan_omits_rows_scanned_and_candidates_when_absent() -> None:
    """``rows_scanned``/``candidates`` are absent, not ``null``, on the ``ann`` path."""
    payload = {
        "path": "ann",
        "narrowing": {"state": "inactive"},
        "timings": {"total_us": 50},
    }
    plan = _wire.decode_plan(payload)
    assert plan.rows_scanned is None
    assert plan.candidates is None
    assert plan.narrowing.state == "inactive"
    assert plan.narrowing.candidates is None


def test_decode_hits_and_plan_splits_the_wrapped_response() -> None:
    hits, plan = _wire.decode_hits_and_plan(
        {
            "hits": [{"collection": "c", "id": "i", "score": 0.9, "attrs": {}}],
            "plan": PLAN_PAYLOAD,
        }
    )
    assert len(hits) == 1
    assert hits[0].id == "i"
    assert plan.path == "ann_prefilter_fallback"


def test_decode_hits_and_plan_rejects_a_bare_array() -> None:
    """A bare hits array is what ``plan`` omitted looks like — not what was asked for here."""
    with pytest.raises(NidusError):
        _wire.decode_hits_and_plan([{"collection": "c", "id": "i", "score": 0.9, "attrs": {}}])
