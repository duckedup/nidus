"""Tests for the predicate builders and the shape a ``Filter`` takes on the wire.

Two things here are easy to get wrong and impossible to notice from the client side, since
a mis-shaped filter is rejected (or worse, silently matches nothing) on the server:

* ``Glob`` is the one asymmetric variant — its second element is a **bare string**, because
  the server's ``Predicate::Glob`` holds a ``String`` while every other variant holds a
  ``Value``. So the test asserts on both sides of that asymmetry together, in one place, or
  the reader has to take it on faith.
* ``Filter`` is a newtype over ``Vec<Predicate>``, so it is a **plain JSON array**. Wrapping
  it in an object would be a natural mistake and a hard failure, and ``[]`` — matching
  everything — has to survive as an empty array rather than being pruned away.
* ``f.in_``/``f.not_in`` are the only constructors taking a *set* of values, and a bare
  string is a legal ``Sequence[AttrInput]``, so passing one value instead of a set is both
  the natural slip for ``in`` and silently accepted by the server.

Assertions run against ``json.dumps`` where the *serialized* form is the actual claim; a
Python ``list`` and a JSON array are only the same thing until someone changes a builder.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

import pytest

from nidus import AttrInput, Filter, Predicate, f, v


def test_every_leaf_predicate_variant_has_the_two_tuple_shape() -> None:
    """Seventeen leaf variants, each externally tagged over ``[key, operand]``.

    The combinators deliberately break this shape, and so does ``fuzzy`` — see the group
    tests and the three-element test below.
    """
    assert f.eq("lang", "rust") == {"Eq": ["lang", {"Str": "rust"}]}
    assert f.ne("lang", "go") == {"Ne": ["lang", {"Str": "go"}]}
    assert f.glob("path", "src/*") == {"Glob": ["path", "src/*"]}
    assert f.iglob("path", "Src/*") == {"IGlob": ["path", "Src/*"]}
    assert f.in_("status", ["published", "draft"]) == {
        "In": ["status", [{"Str": "published"}, {"Str": "draft"}]]
    }
    assert f.not_in("status", ["draft"]) == {"NotIn": ["status", [{"Str": "draft"}]]}
    assert f.lt("year", 2020) == {"Lt": ["year", {"Int": 2020}]}
    assert f.le("year", 2020) == {"Le": ["year", {"Int": 2020}]}
    assert f.gt("year", 2020) == {"Gt": ["year", {"Int": 2020}]}
    assert f.ge("year", 2020) == {"Ge": ["year", {"Int": 2020}]}
    assert f.contains("tags", "rust") == {"Contains": ["tags", {"Str": "rust"}]}
    assert f.not_contains("tags", "wip") == {"NotContains": ["tags", {"Str": "wip"}]}
    assert f.contains_any("tags", ["rust", "go"]) == {
        "ContainsAny": ["tags", [{"Str": "rust"}, {"Str": "go"}]]
    }
    assert f.contains_all_tokens("body", "async runtime") == {
        "ContainsAllTokens": ["body", "async runtime"]
    }
    assert f.contains_any_token("body", "async runtime") == {
        "ContainsAnyToken": ["body", "async runtime"]
    }
    assert f.contains_token_sequence("body", "async runtime") == {
        "ContainsTokenSequence": ["body", "async runtime"]
    }
    assert f.regex("path", "src/.*[.]rs") == {"Regex": ["path", "src/.*[.]rs"]}


def test_fuzzy_is_the_one_predicate_with_a_three_element_operand() -> None:
    """The edit budget is part of the operand, so ``Fuzzy`` breaks the 2-tuple shape.

    Asserted on the serialized form as well: the server's ``Fuzzy(String, String, usize)``
    deserializes from a 3-element array and from nothing else, so a client that dropped the
    budget (or nested it) would fail every fuzzy query with a deserialization error.
    """
    predicate = f.fuzzy("title", "levenshtein", 2)
    assert predicate == {"Fuzzy": ["title", "levenshtein", 2]}
    (operand,) = predicate.values()
    assert len(operand) == 3
    assert json.dumps(predicate, separators=(",", ":")) == '{"Fuzzy":["title","levenshtein",2]}'


def test_the_text_predicates_take_a_bare_string_like_glob_does() -> None:
    """Five more variants holding a ``String``, not a tagged ``Value`` — the ``glob`` shape."""
    for predicate in (
        f.fuzzy("k", "s", 1),
        f.contains_all_tokens("k", "s"),
        f.contains_any_token("k", "s"),
        f.contains_token_sequence("k", "s"),
        f.regex("k", "s"),
    ):
        (operand,) = predicate.values()
        assert operand[0] == "k"
        assert operand[1] == "s"
        assert isinstance(operand[1], str)


def test_contains_any_refuses_a_bare_string() -> None:
    """A `str` is a `Sequence[str]`, so this type-checks and would encode one predicate
    per CHARACTER and match nothing — the same trap `in_`/`not_in` already guard."""
    with pytest.raises(TypeError, match="expects a sequence"):
        f.contains_any("tags", "rust")
    assert f.contains_any("tags", ["rust"]) == {"ContainsAny": ["tags", [{"Str": "rust"}]]}


def test_the_combinators_are_not_key_value_tuples() -> None:
    """``all_``/``any_`` wrap a bare list of predicates, ``not_`` a single one."""
    assert f.any_(f.eq("a", 1), f.eq("b", 2)) == {
        "Any": [{"Eq": ["a", {"Int": 1}]}, {"Eq": ["b", {"Int": 2}]}]
    }
    assert f.all_(f.eq("a", 1)) == {"All": [{"Eq": ["a", {"Int": 1}]}]}
    assert f.not_(f.eq("a", 1)) == {"Not": {"Eq": ["a", {"Int": 1}]}}


def test_empty_groups_are_empty_lists_not_null() -> None:
    """The identities (``All`` true, ``Any`` false) only hold if they deserialize."""
    assert f.all_() == {"All": []}
    assert f.any_() == {"Any": []}


def test_groups_nest() -> None:
    """A group holding a group — the whole point of the combinators."""
    nested = f.not_(f.any_(f.contains("tags", "wip")))
    assert nested == {"Not": {"Any": [{"Contains": ["tags", {"Str": "wip"}]}]}}
    assert json.loads(json.dumps(nested)) == nested


def test_all_is_a_predicate_while_and_is_a_filter() -> None:
    """The distinction that decides which one a caller needs: only ``all_`` nests."""
    assert isinstance(f.and_(f.eq("a", 1)), list)
    assert isinstance(f.all_(f.eq("a", 1)), dict)
    # So a conjunction can sit inside a disjunction only via all_.
    assert f.any_(f.all_(f.eq("a", 1))) == {"Any": [{"All": [{"Eq": ["a", {"Int": 1}]}]}]}


def test_glob_takes_a_bare_string_while_the_others_take_a_value() -> None:
    """The asymmetry, asserted from both sides so it cannot drift unnoticed."""
    glob = f.glob("path", "src/*")
    assert glob["Glob"][1] == "src/*"
    assert isinstance(glob["Glob"][1], str)
    iglob = f.iglob("path", "src/*")
    assert isinstance(iglob["IGlob"][1], str)

    # Every other variant tags its operand. `f.eq("path", "src/*")` on the same inputs is
    # the contrast: same key, same string, different wire shape.
    eq = f.eq("path", "src/*")
    assert eq["Eq"][1] == {"Str": "src/*"}
    assert isinstance(eq["Eq"][1], dict)

    # And In/NotIn take an *array* of tagged values, not a tagged array.
    assert f.in_("tag", ["a"])["In"][1] == [{"Str": "a"}]


@pytest.mark.parametrize("build", [f.in_, f.not_in])
def test_in_and_not_in_refuse_a_single_string_in_place_of_a_set(
    build: Callable[[str, Sequence[AttrInput]], Predicate],
) -> None:
    """``f.in_("lang", "rust")`` used to encode four one-character predicates.

    It type-checks under ``mypy --strict`` (``str`` is a ``Sequence[str]`` and ``str`` is an
    ``AttrInput``), ``Predicate::In(String, Vec<Value>)`` accepts the array without
    complaint, and the filter then matches nothing with no 400 and no exception anywhere.
    The JS SDK cannot express the mistake, and ``v.list`` already refuses it, so these two
    were the only way a filter could be silently wrong on the wire.
    """
    with pytest.raises(TypeError, match="did you mean"):
        build("lang", "rust")
    # The set-of-one it was meant to be still works.
    assert build("lang", ["rust"]) in (
        {"In": ["lang", [{"Str": "rust"}]]},
        {"NotIn": ["lang", [{"Str": "rust"}]]},
    )


@pytest.mark.parametrize(
    "predicate",
    [
        f.eq("k", "s"),
        f.ne("k", 1),
        f.lt("k", 1),
        f.le("k", 1),
        f.gt("k", 1),
        f.ge("k", 1),
        f.in_("k", ["s"]),
        f.not_in("k", ["s"]),
        f.glob("k", "s*"),
    ],
)
def test_a_predicate_is_a_single_tagged_key(predicate: Predicate) -> None:
    """Externally tagged means exactly one key, whose value is the 2-element tuple."""
    assert len(predicate) == 1
    (operand,) = predicate.values()
    assert isinstance(operand, list)
    assert len(operand) == 2
    assert operand[0] == "k"


def test_predicates_accept_plain_values_and_explicit_v_helpers_alike() -> None:
    """Normalization happens in the builder, so both spellings produce one wire form."""
    assert f.eq("lang", "rust") == f.eq("lang", v.str("rust"))
    assert f.ge("year", 2020) == f.ge("year", v.int(2020))
    assert f.eq("draft", False) == f.eq("draft", v.bool(False))
    assert f.eq("note", None) == f.eq("note", v.nil())
    assert f.eq("note", None) == {"Eq": ["note", "Null"]}


def test_a_boolean_operand_is_not_encoded_as_an_int() -> None:
    """The ``bool``-is-an-``int`` trap reaches filters too, via ``encode_value``."""
    assert f.eq("draft", True) == {"Eq": ["draft", {"Bool": True}]}
    assert f.in_("flags", [True, False]) == {"In": ["flags", [{"Bool": True}, {"Bool": False}]]}


def test_a_float_operand_keeps_its_type_through_the_predicate() -> None:
    """Comparisons are same-type only, so ``2020`` and ``2020.0`` are different filters.

    The predicate builders normalize through ``encode_value``, which means the Python type
    of the operand is what picks ``Int`` or ``Float`` — a range over a ``Float`` attribute
    written with an ``int`` operand matches nothing, silently.
    """
    assert f.gt("score", 2020.5) == {"Gt": ["score", {"Float": 2020.5}]}
    assert f.gt("score", 2020.0) == {"Gt": ["score", {"Float": 2020.0}]}
    assert f.gt("year", 2020) == {"Gt": ["year", {"Int": 2020}]}
    assert f.in_("score", [1.5, 2]) == {"In": ["score", [{"Float": 1.5}, {"Int": 2}]]}


def test_a_datetime_operand_encodes_as_epoch_milliseconds() -> None:
    """A range over instants: both operand forms reach the same wire number."""
    when = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
    assert f.ge("seen", when) == {"Ge": ["seen", {"DateTime": 1700000000000}]}
    assert f.ge("seen", v.datetime(1700000000000)) == f.ge("seen", when)


def test_an_unencodable_operand_is_rejected_at_the_call_site() -> None:
    """A value with no attribute type fails here, not as a 400 naming serde."""
    with pytest.raises(TypeError, match="set"):
        f.gt("year", {2020})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        f.gt("score", float("nan"))
    with pytest.raises(ValueError, match="aware datetime"):
        f.ge("seen", datetime(2023, 11, 14))


def test_filter_serializes_as_a_bare_array() -> None:
    """A ``Filter`` is a JSON array of predicates, never an object wrapping one."""
    filt: Filter = f.and_(
        f.eq("lang", "rust"),
        f.ge("year", 2020),
        f.in_("tag", ["a", "b"]),
        f.glob("path", "src/*"),
    )
    assert filt == [
        {"Eq": ["lang", {"Str": "rust"}]},
        {"Ge": ["year", {"Int": 2020}]},
        {"In": ["tag", [{"Str": "a"}, {"Str": "b"}]]},
        {"Glob": ["path", "src/*"]},
    ]
    encoded = json.dumps(filt)
    assert encoded.startswith("[")
    assert encoded.endswith("]")


def test_an_empty_filter_is_an_empty_array() -> None:
    """``[]`` matches everything, and it is a real value — not something to prune away."""
    assert f.and_() == []
    assert json.dumps(f.and_()) == "[]"


def test_and_is_sugar_because_predicates_already_conjoin() -> None:
    """``f.and_`` only collects: a bare list of predicates is the same filter."""
    assert f.and_(f.eq("a", 1), f.eq("b", 2)) == [f.eq("a", 1), f.eq("b", 2)]
    # It copies rather than aliasing, so a caller mutating the result cannot reach back
    # into anything the SDK holds.
    preds = [f.eq("a", 1)]
    collected = f.and_(*preds)
    collected.append(f.eq("b", 2))
    assert len(preds) == 1
