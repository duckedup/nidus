"""Constructors and codecs for the externally-tagged ``Value`` wire type.

Callers should never hand-write ``{"Str": "x"}``: use ``v.str("x")`` / ``v.int(5)`` /
… , or just put plain Python values in ``attrs`` and let :func:`encode_value` normalize
them. ``Value`` is deliberately a type *alias* over the wire JSON rather than a wrapper
class — a wrapper would mean callers converting in both directions for no gain, and the
tagged dict is what actually travels.

Four things about Python make this the highest-risk file in the SDK, so all four are
handled once, here, instead of at each call site:

* ``bool`` is a subclass of ``int``, so the ``bool`` test **must** precede the ``int``
  test. Get that order wrong and every boolean attribute silently ships as
  ``{"Int": 1}`` — a bug no type checker catches and no server rejects.
* Python integers are unbounded; the store's ``Int`` is an ``i64``. Range-checking here
  turns a caller mistake into a local ``ValueError`` instead of a puzzling 400.
* ``Int`` and ``Float`` are separate types on the server and comparisons are same-type
  only, so the **Python type decides**: ``2`` is an ``Int``, ``2.0`` is a ``Float``. A
  value-based rule would give one attribute two types across records.
* A naive ``datetime`` is rejected. ``DateTime`` is a UTC instant, and guessing a
  timezone for a naive one silently shifts it by hours — a wrong answer that looks right.

Absence and ``Null`` are different facts and stay different: an absent key means "not
set / not indexed", the bare string ``"Null"`` means "set, and empty". Never collapse
one into the other.
"""

from __future__ import annotations

import builtins
import datetime as _dt
import math
from collections.abc import Mapping, Sequence
from typing import Any, Union, cast

from . import _guards

#: A typed attribute value exactly as nidus serde-encodes ``Value`` on the wire:
#: ``{"Str": …}``, ``{"Int": …}``, ``{"Bool": …}``, ``{"List": [...]}``, ``{"Float": …}``,
#: ``{"DateTime": …}``, or the bare JSON string ``"Null"`` (a string, not an object —
#: that asymmetry is the wire's).
Value = Union[dict[str, Any], str]

#: What callers may pass anywhere a :data:`Value` is expected: an already-tagged
#: ``Value`` (from the ``v.*`` helpers) or a plain Python value the SDK normalizes.
AttrInput = Union[Value, str, int, float, bool, list[str], _dt.datetime, None]

#: A :data:`Value` decoded back to a plain Python value. ``DateTime`` decodes to an
#: aware ``datetime``, not an ``int``, so a decoded map re-encodes to what it came from.
DecodedValue = Union[str, int, float, bool, list[str], _dt.datetime, None]

# The explicit `Null` value. A bare string on the wire, so it is a constant, not a dict.
# Module-private, like `_TAGS`: `v.nil()` is the public spelling, and the JS SDK exports no
# such constant either, so a public `NULL` would be surface nobody asked for.
_NULL: Value = "Null"

# The known tags. An unknown tag is *not* accepted for encoding (a stray dict is far more
# likely a caller mistake than a value from the future) but *is* passed through on decode,
# so a newer server growing a `Value` variant does not break an older client.
_TAGS = ("Str", "Int", "Bool", "List", "Float", "DateTime")

# The store's `Int` is an i64; Python's int has no such ceiling.
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1

# Epoch milliseconds are counted from here. Arithmetic against a `timedelta` is exact
# integer arithmetic, where `timestamp()`/`fromtimestamp()` would route a large instant
# through a float and lose sub-second digits.
_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
_ONE_MS = _dt.timedelta(milliseconds=1)


class v:
    """Value constructors, named for the ``Value`` variants.

    The names mirror ``sdks/js/src/values.ts`` one for one so the SDKs read alike side by
    side, which is why they shadow builtins as attribute names. Annotations below qualify
    the builtins (``builtins.str``, not ``str``) because inside this class body a bare
    ``str``/``int``/``bool``/``list`` resolves to the staticmethod of that name rather
    than to the type.

    The annotations are the contract for what to pass; the runtime checks live only where
    a wrong type would be *silently* wrong on the wire — the ``bool``/``int`` overlap, the
    ``i64`` range, and list element types.
    """

    @staticmethod
    def str(s: builtins.str) -> Value:
        """A string attribute."""
        return {"Str": s}

    @staticmethod
    def int(n: builtins.int) -> Value:
        """An integer attribute. Must be a true integer within ``i64``."""
        return {"Int": _checked_int(n, "v.int")}

    @staticmethod
    def float(x: Union[builtins.float, builtins.int]) -> Value:
        """A double attribute. ``nan`` and ``inf`` are refused — JSON cannot spell them.

        An ``int`` is accepted so a whole-numbered measurement is still a ``Float``.
        """
        return {"Float": _checked_float(x, "v.float")}

    @staticmethod
    def bool(b: builtins.bool) -> Value:
        """A boolean attribute."""
        return {"Bool": b}

    @staticmethod
    def datetime(x: Union[_dt.datetime, builtins.int]) -> Value:
        """A UTC instant, from an **aware** ``datetime`` or a raw epoch-millisecond count.

        A naive ``datetime`` is a ``ValueError``: it names a wall clock, not an instant,
        and picking a timezone for it would silently shift the value by hours.
        """
        return {"DateTime": _checked_datetime(x, "v.datetime")}

    @staticmethod
    def list(items: Sequence[builtins.str]) -> Value:
        """A list-of-strings attribute (the only element type the store has)."""
        return {"List": _checked_list(items)}

    @staticmethod
    def nil() -> Value:
        """The explicit ``Null`` value — set-but-empty, distinct from an absent key."""
        return _NULL


def _is_value(x: Any) -> bool:
    """True if ``x`` is already a wire-tagged :data:`Value`.

    The bare string ``"Null"`` *is* a tagged value, so an attribute whose text happens to
    be ``"Null"`` has to be written ``v.str("Null")``. The wire format owns that
    ambiguity, and the JS SDK resolves it the same way.

    Both branches narrow with ``isinstance`` *before* comparing. Reaching `x == "Null"` on
    an arbitrary object hands the decision to that object's ``__eq__`` — a numpy array in
    ``attrs`` (a realistic slip in a vector-store SDK) broadcasts the comparison and blows
    up with "truth value of an array is ambiguous", which names neither nidus nor attrs.
    Narrowing first lets `encode_value` reach its own ``TypeError`` instead.
    """
    if isinstance(x, str):
        return x == _NULL
    if not isinstance(x, dict):
        return False
    return any(tag in x for tag in _TAGS)


def encode_value(value: AttrInput) -> Value:
    """Normalize a caller-supplied :data:`AttrInput` into the wire :data:`Value` shape.

    Raises ``TypeError`` for a non-string list element or any other unsupported type;
    ``ValueError`` for an integer outside ``i64``, a non-finite float, or a naive
    ``datetime``.
    """
    # The annotation says what a caller *should* pass. A dynamically-typed caller can pass
    # anything, and refusing the rest loudly is this function's whole job, so widen
    # locally — otherwise the defensive branches read as statically dead code.
    raw: Any = value
    if _is_value(raw):
        return cast(Value, raw)
    if raw is None:
        return _NULL
    # bool BEFORE int: `isinstance(True, int)` is True, so the reverse order would turn
    # every boolean attribute into {"Int": 1}.
    if isinstance(raw, bool):
        return {"Bool": raw}
    if isinstance(raw, str):
        return {"Str": raw}
    if isinstance(raw, int):
        return {"Int": _checked_int(raw, "an Int attribute")}
    # The Python type decides Int vs Float, so `2.0` is a Float and `2` is an Int. Deciding
    # from the value instead would give one attribute two server types across records, and
    # comparisons there are same-type only.
    if isinstance(raw, float):
        return {"Float": _checked_float(raw, "a Float attribute")}
    if isinstance(raw, _dt.datetime):
        return {"DateTime": _checked_datetime(raw, "a DateTime attribute")}
    if isinstance(raw, (list, tuple)):
        return {"List": _checked_list(raw)}
    raise TypeError(f"cannot encode attribute value of type {type(raw).__name__}: {raw!r}")


def encode_attrs(attrs: Mapping[str, AttrInput]) -> dict[str, Value]:
    """Normalize a whole ``attrs`` map of :data:`AttrInput` into wire :data:`Value` s."""
    return {k: encode_value(val) for k, val in attrs.items()}


def decode_value(value: Value) -> DecodedValue:
    """Decode a wire :data:`Value` back to a plain Python value.

    ``DateTime`` comes back as a UTC-aware ``datetime`` rather than the raw millisecond
    count, so that re-encoding a decoded ``attrs`` map reproduces it — an ``int`` there
    would silently demote every instant to an ``Int`` on the way back.
    """
    if isinstance(value, str) and value == _NULL:
        return None
    if isinstance(value, dict):
        if "DateTime" in value:
            return _EPOCH + _dt.timedelta(milliseconds=value["DateTime"])
        for tag in _TAGS:
            if tag in value:
                return cast(DecodedValue, value[tag])
    # Unknown tag: hand it back untouched rather than raising, so a client built against
    # today's server keeps working against a newer one. The cast is the honest admission
    # that the static type cannot describe a variant we do not know yet.
    return cast(DecodedValue, value)


def decode_attrs(attrs: Mapping[str, Value]) -> dict[str, DecodedValue]:
    """Decode a whole wire ``attrs`` map back to plain Python values."""
    return {k: decode_value(val) for k, val in attrs.items()}


def _checked_int(n: Any, who: str) -> int:
    """Reject non-integers and out-of-``i64`` integers, naming the caller in the error."""
    # bool first again: `v.int(True)` is all but certainly a mistake, and `v.bool` exists.
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"{who} expects an integer, got {type(n).__name__}: {n!r}")
    if not _I64_MIN <= n <= _I64_MAX:
        raise ValueError(f"{who} expects an i64, got {n!r} (out of range)")
    # `int(n)` rather than `n`: an int *subclass* (an IntEnum, say) serializes as itself
    # today but has no guarantee to, so narrow it to a plain int on the way to the wire.
    return int(n)


def _checked_float(x: Any, who: str) -> float:
    """Reject a non-number and the three doubles JSON has no spelling for."""
    # bool first, as everywhere here: `True` is an `int`, and `v.float(True)` is a mistake.
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"{who} expects a number, got {type(x).__name__}: {x!r}")
    out = float(x)
    if not math.isfinite(out):
        # `json.dumps` writes `NaN`/`Infinity`, which is not JSON and which serde refuses,
        # so the failure would otherwise land as a 400 about the server's parser.
        raise ValueError(f"{who} expects a finite number, got {x!r}")
    return out


def _checked_datetime(x: Any, who: str) -> int:
    """Convert an aware ``datetime`` (or a raw millisecond count) to epoch milliseconds."""
    # `int` first, and through `_checked_int`, which is also what refuses a `bool`.
    if isinstance(x, int):
        return _checked_int(x, who)
    if not isinstance(x, _dt.datetime):
        raise TypeError(f"{who} expects a datetime or epoch milliseconds, got {x!r}")
    if x.tzinfo is None or x.tzinfo.utcoffset(x) is None:
        raise ValueError(
            f"{who} expects an aware datetime, got the naive {x!r}; a DateTime is an "
            "absolute instant, so attach a timezone (e.g. datetime.timezone.utc)"
        )
    # Integer division of timedeltas, not `.timestamp() * 1000`: the latter routes a large
    # instant through a float. Sub-millisecond precision is truncated — ms is the wire type.
    return _checked_int((x - _EPOCH) // _ONE_MS, who)


def _checked_list(items: Any) -> list[str]:
    """Copy a list attribute, refusing any non-string element."""
    # A bare string first, and with its own message: `str` is a `Sequence[str]`, so
    # `v.list("rust")` type-checks and would otherwise ship four one-character elements.
    _guards.reject_bare_string(items, "a List attribute")
    if not isinstance(items, (list, tuple)):
        raise TypeError(f"a List attribute expects a sequence of str, got {type(items).__name__}")
    out = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"a List attribute must contain only strings, got {item!r}")
        out.append(item)
    return out
