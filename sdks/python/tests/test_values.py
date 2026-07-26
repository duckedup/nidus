"""Tests for the ``Value`` codec — the file where Python's type system fights the wire.

Three of these tests exist because of a language quirk rather than a design choice, and
each one guards a bug that would ship *silently*:

* ``bool`` is a subclass of ``int``, so a reversed ``isinstance`` order turns every boolean
  attribute into ``{"Int": 1}``. The JSON stays valid, the server accepts it, and the
  attribute is quietly the wrong type forever. That is the single most likely bug in
  ``values.py``, so it gets its own named test rather than a line inside a round-trip.
* Python integers are unbounded and the store's ``Int`` is an ``i64``, so the ceiling has
  to be enforced client-side or a caller learns about it as a puzzling 400.
* There is no float attribute type at all, so a ``float`` must fail loudly here instead of
  being truncated somewhere downstream.

The round-trip test covers every variant including the odd one out — ``Null`` is the bare
JSON string ``"Null"``, not an object — and the unknown-tag test pins the forward-
compatibility promise: a client built today keeps working against a server that grows a
new ``Value`` variant tomorrow.

No network, no server, no fixtures: this is pure function-in/value-out.
"""

from __future__ import annotations

import json

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


def test_float_is_rejected_at_encode_time() -> None:
    """There is no float attribute type; a float fails here, not silently downstream."""
    with pytest.raises(TypeError, match="no float attribute type"):
        encode_value(1.5)
    # A float that happens to be integral is rejected too: accepting 2.0 and refusing 2.5
    # would make the rule depend on the value rather than the type.
    with pytest.raises(TypeError, match="no float attribute type"):
        encode_value(2.0)
    with pytest.raises(TypeError):
        v.int(1.5)


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
        "note": None,
    }
    wire = encode_attrs(plain)
    assert wire == {
        "lang": {"Str": "rust"},
        "year": {"Int": 2024},
        "draft": {"Bool": False},
        "tags": {"List": ["a", "b"]},
        "note": "Null",
    }
    assert decode_attrs(wire) == plain


def test_null_and_an_absent_key_stay_distinct() -> None:
    """Set-and-empty is a different fact from not-set, in both directions."""
    assert encode_attrs({"note": None}) == {"note": "Null"}
    assert encode_attrs({}) == {}
    assert decode_attrs({"note": "Null"}) == {"note": None}
    assert "note" not in decode_attrs({})
