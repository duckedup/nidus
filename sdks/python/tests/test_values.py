"""Tests for the ``Value`` codec — the file where Python's type system fights the wire.

Four of these tests exist because of a language quirk rather than a design choice, and
each one guards a bug that would ship *silently*:

* ``bool`` is a subclass of ``int``, so a reversed ``isinstance`` order turns every boolean
  attribute into ``{"Int": 1}``. The JSON stays valid, the server accepts it, and the
  attribute is quietly the wrong type forever. That is the single most likely bug in
  ``values.py``, so it gets its own named test rather than a line inside a round-trip.
* Python integers are unbounded and the store's ``Int`` is an ``i64``, so the ceiling has
  to be enforced client-side or a caller learns about it as a puzzling 400.
* ``Int`` and ``Float`` are distinct server types compared same-type-only, so the rule that
  ``2`` is one and ``2.0`` is the other has to hold exactly; a value-based rule would give
  one attribute two types and a range filter would quietly skip half the records.
* A naive ``datetime`` names a wall clock, not an instant. Guessing UTC (or local) for one
  shifts it by hours in valid-looking JSON, so it is refused rather than interpreted.

The round-trip test covers every variant including the odd one out — ``Null`` is the bare
JSON string ``"Null"``, not an object — and the unknown-tag test pins the forward-
compatibility promise: a client built today keeps working against a server that grows a
new ``Value`` variant tomorrow.

No network, no server, no fixtures: this is pure function-in/value-out.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import cast

import pytest

from nidus import (
    AttrInput,
    Value,
    decode_attrs,
    decode_value,
    encode_attrs,
    encode_value,
    v,
)


@pytest.mark.parametrize(
    ("plain", "wire"),
    [
        ("rust", {"Str": "rust"}),
        ("", {"Str": ""}),
        (2024, {"Int": 2024}),
        (0, {"Int": 0}),
        (-1, {"Int": -1}),
        (True, {"Bool": True}),
        (False, {"Bool": False}),
        (["a", "b"], {"List": ["a", "b"]}),
        ([], {"List": []}),
        (1.5, {"Float": 1.5}),
        (2.0, {"Float": 2.0}),
        (
            datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc),
            {"DateTime": 1700000000000},
        ),
        (None, "Null"),
    ],
)
def test_encode_decode_round_trip_for_every_value_kind(plain: AttrInput, wire: Value) -> None:
    """Every variant encodes to its wire shape and decodes back to the plain value."""
    assert encode_value(plain) == wire
    assert decode_value(wire) == plain
    # And the tagged form is a fixed point: re-encoding an already-tagged value is a no-op,
    # which is what lets `v.*` helpers and plain values be mixed in one attrs map.
    assert encode_value(wire) == wire


def test_null_is_a_bare_string_on_the_wire() -> None:
    """``Null`` serializes as the JSON *string* ``"Null"``, not as an object."""
    assert encode_value(None) == "Null"
    assert v.nil() == "Null"
    assert json.dumps(encode_value(None)) == '"Null"'
    assert decode_value("Null") is None


def test_bool_attr_encodes_as_bool_not_int() -> None:
    """A ``True``/``False`` attr is ``{"Bool": …}``.

    The named guard for the ``bool``-is-an-``int`` trap: with the ``isinstance`` order
    reversed this produces ``{"Int": 1}``, which is valid JSON, accepted by the server, and
    wrong in a way nothing else in the stack notices.
    """
    assert encode_value(True) == {"Bool": True}
    assert encode_value(False) == {"Bool": False}
    assert "Int" not in encode_value(True)
    assert "Int" not in encode_value(False)
    # Through the whole-map path too, which is what `upsert` actually calls.
    assert encode_attrs({"ok": True, "shipped": False}) == {
        "ok": {"Bool": True},
        "shipped": {"Bool": False},
    }
    # `is True` rather than `== True`, since `1 == True` would let a wrong decode pass.
    assert decode_value(encode_value(True)) is True
    assert decode_value(encode_value(False)) is False


def test_v_int_refuses_a_bool() -> None:
    """``v.int(True)`` is a mistake worth naming — ``v.bool`` exists for that."""
    with pytest.raises(TypeError):
        v.int(True)


def test_the_python_type_decides_int_versus_float() -> None:
    """``2`` is an ``Int`` and ``2.0`` is a ``Float``, whatever the value happens to be.

    The rule is the *type*, not the number: deciding from the value would make one
    attribute an ``Int`` in the records where it landed on a round number and a ``Float``
    everywhere else, and the server compares same-type only.
    """
    assert encode_value(2) == {"Int": 2}
    assert encode_value(2.0) == {"Float": 2.0}
    assert encode_value(1.5) == {"Float": 1.5}
    assert encode_value(-0.0) == {"Float": -0.0}
    # The explicit constructors say the same thing, and `v.float` takes an int so a
    # whole-numbered measurement can still be written as a Float.
    assert v.int(2) == {"Int": 2}
    assert v.float(2) == {"Float": 2.0}
    assert v.float(2.5) == {"Float": 2.5}
    with pytest.raises(TypeError):
        v.int(1.5)
    with pytest.raises(TypeError):
        v.float(True)


def test_a_non_finite_float_is_rejected() -> None:
    """``json.dumps`` writes ``NaN``/``Infinity``, which is not JSON and which serde refuses."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            encode_value(bad)
        with pytest.raises(ValueError, match="finite"):
            v.float(bad)


def test_datetime_encodes_as_epoch_milliseconds_in_utc() -> None:
    """An instant, not a wall clock: the same moment in any zone is the same number."""
    utc = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
    tokyo = utc.astimezone(timezone(timedelta(hours=9)))
    assert encode_value(utc) == {"DateTime": 1700000000000}
    assert encode_value(tokyo) == encode_value(utc)
    assert v.datetime(utc) == {"DateTime": 1700000000000}
    # The raw millisecond form is accepted too, for a caller who already holds one.
    assert v.datetime(1700000000000) == {"DateTime": 1700000000000}
    # Sub-millisecond precision is truncated: milliseconds is the wire type.
    assert encode_value(utc.replace(microsecond=999)) == {"DateTime": 1700000000000}
    # Before the epoch is an ordinary negative count, not an error.
    assert encode_value(datetime(1969, 12, 31, 23, 59, 59, tzinfo=timezone.utc)) == {
        "DateTime": -1000
    }


def test_a_naive_datetime_is_rejected() -> None:
    """Refused rather than assumed to be UTC — the wrong guess is off by hours, silently."""
    with pytest.raises(ValueError, match="aware datetime"):
        encode_value(datetime(2023, 11, 14, 22, 13, 20))
    with pytest.raises(ValueError, match="aware datetime"):
        v.datetime(datetime(2023, 11, 14))
    # A `date` has no time at all, so it is not a DateTime either — just an unknown type.
    with pytest.raises(TypeError, match="date"):
        encode_value(date(2023, 11, 14))  # type: ignore[arg-type]


def test_datetime_round_trips_back_to_an_aware_datetime() -> None:
    """Decoding to an ``int`` would demote every instant to an ``Int`` on re-encode."""
    utc = datetime(2023, 11, 14, 22, 13, 20, 123000, tzinfo=timezone.utc)
    wire = encode_value(utc)
    back = decode_value(wire)
    assert back == utc
    assert isinstance(back, datetime) and back.tzinfo is timezone.utc
    assert encode_value(cast(AttrInput, back)) == wire


def test_non_integer_types_are_rejected() -> None:
    """Anything outside the five variants is a ``TypeError``, naming the offending type."""
    with pytest.raises(TypeError, match="dict"):
        encode_value({"not": "a tagged value"})
    with pytest.raises(TypeError, match="set"):
        encode_value({"a", "b"})
    with pytest.raises(TypeError, match="bytes"):
        encode_value(b"raw")


def test_out_of_i64_range_int_is_rejected() -> None:
    """Python ints are unbounded; the store's ``Int`` is an ``i64``, so check the edges."""
    assert encode_value(2**63 - 1) == {"Int": 2**63 - 1}
    assert encode_value(-(2**63)) == {"Int": -(2**63)}
    with pytest.raises(ValueError, match="i64"):
        encode_value(2**63)
    with pytest.raises(ValueError, match="i64"):
        encode_value(-(2**63) - 1)
    with pytest.raises(ValueError, match="i64"):
        v.int(2**64)


def test_list_with_a_non_string_element_is_rejected() -> None:
    """``List`` is a list of strings only — the sole element type the store has."""
    with pytest.raises(TypeError, match="only strings"):
        encode_value(["a", 2])
    with pytest.raises(TypeError, match="only strings"):
        encode_value([None])
    with pytest.raises(TypeError, match="only strings"):
        v.list(["ok", 1.5])


def test_a_single_string_is_not_a_list_attribute() -> None:
    """``v.list("rust")`` would ship ``["r", "u", "s", "t"]`` — a ``str`` is a sequence."""
    with pytest.raises(TypeError, match="did you mean"):
        v.list("rust")
    # `bytes` iterates too (into ints), so it is refused by the same guard.
    with pytest.raises(TypeError, match="did you mean"):
        v.list(b"rust")  # type: ignore[arg-type]


def test_an_object_with_a_hostile_eq_still_gets_the_documented_type_error() -> None:
    """The ``"Null"`` test must narrow before it compares, or the caller's ``__eq__`` decides.

    A numpy array in ``attrs`` is a realistic slip in a vector-store SDK, and its
    broadcasting ``__eq__`` turns ``x == "Null"`` into a ``ValueError`` about the truth value
    of an array — a message naming neither nidus nor attrs, so the caller goes looking in the
    wrong place. This file's contract is a ``TypeError`` naming the unsupported type.
    """

    class Broadcasting:
        def __eq__(self, other: object) -> bool:
            raise ValueError("The truth value of an array with more than one element ...")

        __hash__ = None  # type: ignore[assignment]

    with pytest.raises(TypeError, match="Broadcasting"):
        encode_value(Broadcasting())  # type: ignore[arg-type]


def test_unknown_tag_decodes_to_itself_unchanged() -> None:
    """Forward compatibility: a variant this client has never heard of passes through.

    A newer server growing a ``Value`` variant must not break an older client, so decoding
    hands the raw tagged object back rather than raising.
    """
    future = {"Float64": 1.5}
    assert decode_value(future) == future
    assert decode_attrs({"score": future}) == {"score": future}


def test_v_constructors_match_the_wire_shape() -> None:
    """The ``v.*`` helpers are just named spellings of the tagged objects."""
    assert v.str("rust") == {"Str": "rust"}
    assert v.int(2024) == {"Int": 2024}
    assert v.bool(True) == {"Bool": True}
    assert v.list(["a", "b"]) == {"List": ["a", "b"]}
    assert v.float(1.5) == {"Float": 1.5}
    assert v.datetime(1700000000000) == {"DateTime": 1700000000000}
    assert v.nil() == "Null"
    # A tuple is accepted and copied to a list, since JSON has no tuple.
    assert v.list(("a", "b")) == {"List": ["a", "b"]}


def test_a_string_attr_whose_text_is_null_needs_the_explicit_constructor() -> None:
    """The wire's own ambiguity, resolved the same way as in the JS SDK.

    ``"Null"`` *is* a tagged value, so a plain ``"Null"`` string cannot also mean the text
    "Null" — ``v.str("Null")`` says that, and it survives the round trip.
    """
    assert encode_value("Null") == "Null"
    assert encode_value(v.str("Null")) == {"Str": "Null"}
    assert decode_value(v.str("Null")) == "Null"


def test_attrs_maps_round_trip_as_a_whole() -> None:
    """``encode_attrs``/``decode_attrs`` are the per-key codec applied across a map."""
    plain: dict[str, AttrInput] = {
        "lang": "rust",
        "year": 2024,
        "draft": False,
        "tags": ["a", "b"],
        "score": 0.75,
        "seen": datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc),
        "note": None,
    }
    wire = encode_attrs(plain)
    assert wire == {
        "lang": {"Str": "rust"},
        "year": {"Int": 2024},
        "draft": {"Bool": False},
        "tags": {"List": ["a", "b"]},
        "score": {"Float": 0.75},
        "seen": {"DateTime": 1700000000000},
        "note": "Null",
    }
    assert decode_attrs(wire) == plain


def test_null_and_an_absent_key_stay_distinct() -> None:
    """Set-and-empty is a different fact from not-set, in both directions."""
    assert encode_attrs({"note": None}) == {"note": "Null"}
    assert encode_attrs({}) == {}
    assert decode_attrs({"note": "Null"}) == {"note": None}
    assert "note" not in decode_attrs({})
