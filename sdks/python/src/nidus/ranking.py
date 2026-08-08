"""Builders for ``rank_by`` — a ranking expression layered over the distance metric.

``RankBy`` is externally tagged like :data:`~nidus.filter.Predicate` (``{"Decay": {…}}``),
so callers get a constructor rather than the tagged dict, exactly as ``f.*`` and ``v.*``
do. One variant exists today.

Decay **subtracts** its penalty from the base score — ``score - lambda * (1 - factor)``,
where the factor falls from 1 to ``decay`` as a record ages ``scale`` past ``origin``. It
subtracts rather than multiplies so it stays meaningful where scores are negative or
unbounded (Euclidean, dot product, BM25), and ages are measured back from ``origin``
rather than from the wall clock, so the same query against an unchanged store ranks the
same way twice.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Optional, Union

# The same conversions `v.datetime`/`v.int` apply, reused rather than re-derived: the
# epoch-millisecond arithmetic and the naive-datetime refusal must not fork.
from .values import _ONE_MS, _checked_datetime, _checked_int

#: A ranking expression as nidus encodes ``RankBy``: ``{"Decay": {"field": …, …}}``.
RankBy = dict[str, Any]


class rank:  # noqa: N801 - a lowercase namespace, matching `f` and `v`
    """Ranking-expression constructors. Currently one: :meth:`decay`."""

    @staticmethod
    def decay(
        field: str,
        origin: Union[int, _dt.datetime],
        *,
        scale: Optional[Union[int, _dt.timedelta]] = None,
        decay: Optional[float] = None,
        lambda_: Optional[float] = None,
        missing: Optional[float] = None,
    ) -> RankBy:
        """Penalize a hit by the age of its ``field`` timestamp, measured back from ``origin``.

        ``origin`` takes an aware ``datetime`` or raw epoch milliseconds, and ``scale`` a
        ``timedelta`` or milliseconds — the wire unit is milliseconds either way, and a
        naive ``datetime`` is refused as it is everywhere else in this SDK.

        Every knob but the two positional ones is omitted when unset, leaving the server's
        default: ``scale`` a week, ``decay`` 0.5 (so ``scale`` is a half-life), ``lambda_``
        1.0 (the score a fully-decayed hit gives up), and ``missing`` 1.0 — a record whose
        timestamp is absent or unusable is **not** penalized. ``lambda_`` carries the
        underscore because ``lambda`` is a reserved word; it travels as ``lambda``.
        """
        body = {
            "field": field,
            "origin": _checked_datetime(origin, "rank.decay(origin=...)"),
            "scale": _millis(scale),
            "decay": decay,
            "lambda": lambda_,
            "missing": missing,
        }
        return {"Decay": {k: val for k, val in body.items() if val is not None}}


def _millis(scale: Optional[Union[int, _dt.timedelta]]) -> Optional[int]:
    """A ``scale`` as whole milliseconds; a ``timedelta`` divided exactly, never via float."""
    if scale is None:
        return None
    if isinstance(scale, _dt.timedelta):
        return scale // _ONE_MS
    return _checked_int(scale, "rank.decay(scale=...)")
