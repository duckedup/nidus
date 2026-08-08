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
    :data:`~nidus.values.Value`. Some names carry a trailing underscore where JS's does
    not, because the bare word is reserved (``in``, ``and``, ``not``) or shadows a
    builtin (``all``, ``any``): ``f.in_``, ``f.not_in``, ``f.and_``, ``f.all_``,
    ``f.any_``, ``f.not_``. Nothing else deviates.
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
    def contains(key: str, value: AttrInput) -> Predicate:
        """``attrs[key]`` is a list containing ``value``.

        Whole-element, not substring: ``contains("tags", "rust")`` does not match
        ``["rustacean"]``. Use :meth:`glob` for substrings on a plain string.
        """
        return {"Contains": [key, encode_value(value)]}

    @staticmethod
    def not_contains(key: str, value: AttrInput) -> Predicate:
        """``attrs[key]`` is a present list *not* containing ``value``.

        Like :meth:`ne`, it requires the attribute to exist and be a list.
        """
        return {"NotContains": [key, encode_value(value)]}

    @staticmethod
    def contains_any(key: str, values: Sequence[AttrInput]) -> Predicate:
        """``attrs[key]`` is a list sharing at least one element with ``values``.

        An empty set matches nothing. "Contains all of" is :meth:`all_` over several
        :meth:`contains`.
        """
        _guards.reject_bare_string(values, "f.contains_any(key, values)")
        return {"ContainsAny": [key, [encode_value(v) for v in values]]}

    # The text predicates below take a **bare string** as their second element, like
    # `glob`/`iglob` and unlike every other leaf: the server's variants hold a `String`. Each
    # reads any text the attribute carries, so a `List` matches when a single element does.

    @staticmethod
    def fuzzy(key: str, text: str, max_edits: int) -> Predicate:
        """``attrs[key]`` is within ``max_edits`` Levenshtein edits of ``text``.

        The only **3**-element predicate — ``{"Fuzzy": ["k", "text", 2]}`` — because the
        edit budget is part of the operand. Above 8 edits the server errors rather than
        clamping, since that far out the predicate matches most of the store.
        """
        return {"Fuzzy": [key, text, max_edits]}

    @staticmethod
    def contains_all_tokens(key: str, text: str) -> Predicate:
        """Every token of ``text`` appears among ``attrs[key]``'s tokens, in any order.

        Tokens are ASCII-case-folded runs of alphanumerics here and in the two below, so
        case and punctuation do not count — unlike :meth:`glob`, which is character-literal.
        """
        return {"ContainsAllTokens": [key, text]}

    @staticmethod
    def contains_any_token(key: str, text: str) -> Predicate:
        """At least one token of ``text`` appears in ``attrs[key]``. Empty text matches nothing."""
        return {"ContainsAnyToken": [key, text]}

    @staticmethod
    def contains_token_sequence(key: str, text: str) -> Predicate:
        """``text``'s tokens appear consecutively and in order in ``attrs[key]`` — a phrase."""
        return {"ContainsTokenSequence": [key, text]}

    @staticmethod
    def regex(key: str, pattern: str) -> Predicate:
        """``attrs[key]`` matches the regular expression, **anchored at both ends**.

        Anchored like :meth:`glob`, so ``"src"`` matches only the whole value and
        ``".*src.*"`` is the substring search. Case folding is the pattern's own ``(?i)``,
        and an unparseable pattern comes back as a server-side error.
        """
        return {"Regex": [key, pattern]}

    @staticmethod
    def all_(*preds: Predicate) -> Predicate:
        """Every sub-predicate holds. ``all_()`` is true, the identity for AND.

        (JS: ``f.all`` — ``all`` shadows a builtin.) Unlike :meth:`and_` this returns a
        *predicate*, so it can nest inside another group.
        """
        return {"All": list(preds)}

    @staticmethod
    def any_(*preds: Predicate) -> Predicate:
        """At least one sub-predicate holds. ``any_()`` is false, the identity for OR.

        (JS: ``f.any`` — ``any`` shadows a builtin.)
        """
        return {"Any": list(preds)}

    @staticmethod
    def not_(pred: Predicate) -> Predicate:
        """The sub-predicate does not hold. (JS: ``f.not`` — ``not`` is reserved.)

        Differs from :meth:`ne` on an absent key: ``not_(eq(k, v))`` matches a record
        with no ``k`` at all, whereas ``ne(k, v)`` does not.
        """
        return {"Not": pred}

    @staticmethod
    def and_(*preds: Predicate) -> Filter:
        """Collect predicates into a :data:`Filter` — sugar, since they already AND.

        (JS: ``f.and`` — ``and`` is reserved.) For a conjunction *inside* a group use
        :meth:`all_`, which is a predicate rather than a filter.
        """
        return list(preds)
