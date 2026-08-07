"""Filter builders producing the bare predicate-array wire shape.

A ``Filter`` is a conjunction of predicates; on the wire it is a newtype over
``Vec<Predicate>`` and therefore a **plain JSON array**, never an object wrapping one.
``[]`` matches everything.

Each predicate is a *positive assertion about a present attribute* — an absent key
matches nothing, including the negative predicates (``ne``/``not_in``) and the ranges.
Comparisons are same-type only (Int↔Int numeric, Str↔Str lexical, Bool↔Bool), which is
the server's rule (``src/filter/``), restated here because it is the thing callers get
wrong when a filter mysteriously returns no rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import _guards
from .values import AttrInput, encode_value

#: A single attribute predicate, externally tagged over a 2-tuple exactly as nidus
#: encodes ``Predicate``: ``{"Eq": ["lang", {"Str": "rust"}]}``.
Predicate = dict[str, Any]

#: A conjunction (AND) of predicates. A bare array on the wire; ``[]`` matches everything.
Filter = list[Predicate]


class f:
    """Predicate constructors, mirroring ``sdks/js/src/filter.ts`` name for name.

    Each accepts a plain Python value (auto-normalized) or an explicit ``v.*``
    :data:`~nidus.values.Value`. Three names carry a trailing underscore because ``in``
    and ``and`` are reserved words in Python — ``f.in_``, ``f.not_in``, and ``f.and_``
    are JS's ``f.in``, ``f.notIn``, and ``f.and``. Nothing else deviates.
    """

    @staticmethod
    def eq(key: str, value: AttrInput) -> Predicate:
        """``attrs[key]`` is present and equals ``value``."""
        return {"Eq": [key, encode_value(value)]}

    @staticmethod
    def ne(key: str, value: AttrInput) -> Predicate:
        """``attrs[key]`` is present and does not equal ``value``."""
        return {"Ne": [key, encode_value(value)]}

    @staticmethod
    def glob(key: str, pattern: str) -> Predicate:
        """``attrs[key]`` is a ``Str`` matching the glob pattern (``*``, ``?``, ``[..]``).

        The only asymmetric variant: the second element is a **bare string**, not a
        tagged ``Value``, because the server's ``Predicate::Glob`` holds a ``String``.
        """
        return {"Glob": [key, pattern]}

    @staticmethod
    def iglob(key: str, pattern: str) -> Predicate:
        """:meth:`glob`, ignoring **ASCII** case on both sides.

        ``f.iglob("path", "Src/*")`` matches ``"src/main.rs"``. Non-ASCII is not folded,
        so ``É`` does not match ``é``. Second element is a bare string, as with ``glob``.
        """
        return {"IGlob": [key, pattern]}

    # `in_`/`not_in` are the only constructors taking a *set* of values, and a `str` is a
    # `Sequence[str]` while `str` is also an `AttrInput` — so `f.in_("lang", "rust")` type-
    # checks under `mypy --strict`, encodes four one-character predicates, and the server
    # accepts the filter and matches nothing. Passing one value instead of a set is the
    # natural slip for `in`; the JS SDK cannot express it, so Python has to refuse it here.

    @staticmethod
    def in_(key: str, values: Sequence[AttrInput]) -> Predicate:
        """``attrs[key]`` equals one of ``values``. (JS: ``f.in`` — ``in`` is reserved.)"""
        _guards.reject_bare_string(values, "f.in_(key, values)")
        return {"In": [key, [encode_value(x) for x in values]]}

    @staticmethod
    def not_in(key: str, values: Sequence[AttrInput]) -> Predicate:
        """``attrs[key]`` is present and equals none of ``values``. (JS: ``f.notIn``.)"""
        _guards.reject_bare_string(values, "f.not_in(key, values)")
        return {"NotIn": [key, [encode_value(x) for x in values]]}

    @staticmethod
    def lt(key: str, value: AttrInput) -> Predicate:
        """``attrs[key] < value`` (same-type, orderable)."""
        return {"Lt": [key, encode_value(value)]}

    @staticmethod
    def le(key: str, value: AttrInput) -> Predicate:
        """``attrs[key] <= value`` (same-type, orderable)."""
        return {"Le": [key, encode_value(value)]}

    @staticmethod
    def gt(key: str, value: AttrInput) -> Predicate:
        """``attrs[key] > value`` (same-type, orderable)."""
        return {"Gt": [key, encode_value(value)]}

    @staticmethod
    def ge(key: str, value: AttrInput) -> Predicate:
        """``attrs[key] >= value`` (same-type, orderable)."""
        return {"Ge": [key, encode_value(value)]}

    @staticmethod
    def and_(*preds: Predicate) -> Filter:
        """Collect predicates into a :data:`Filter` — sugar, since they already AND.

        (JS: ``f.and`` — ``and`` is reserved.)
        """
        return list(preds)
