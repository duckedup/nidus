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

import pytest

from nidus import AttrInput, Filter, Predicate, f, v


def test_every_predicate_variant_has_the_two_tuple_shape() -> None:
    """Nine variants, each externally tagged over ``[key, operand]``."""
    assert f.eq("lang", "rust") == {"Eq": ["lang", {"Str": "rust"}]}
    assert f.ne("lang", "go") == {"Ne": ["lang", {"Str": "go"}]}
    assert f.glob("path", "src/*") == {"Glob": ["path", "src/*"]}
    assert f.in_("status", ["published", "draft"]) == {
        "In": ["status", [{"Str": "published"}, {"Str": "draft"}]]
    }
    assert f.not_in("status", ["draft"]) == {"NotIn": ["status", [{"Str": "draft"}]]}
    assert f.lt("year", 2020) == {"Lt": ["year", {"Int": 2020}]}
    assert f.le("year", 2020) == {"Le": ["year", {"Int": 2020}]}
    assert f.gt("year", 2020) == {"Gt": ["year", {"Int": 2020}]}
    assert f.ge("year", 2020) == {"Ge": ["year", {"Int": 2020}]}


def test_glob_takes_a_bare_string_while_the_others_take_a_value() -> None:
    """The asymmetry, asserted from both sides so it cannot drift unnoticed."""
    glob = f.glob("path", "src/*")
    assert glob["Glob"][1] == "src/*"
    assert isinstance(glob["Glob"][1], str)

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


def test_a_float_operand_is_rejected() -> None:
    """No float attribute type, so no float comparisons either — fail at the call site."""
    with pytest.raises(TypeError, match="no float attribute type"):
        f.gt("year", 2020.5)


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
