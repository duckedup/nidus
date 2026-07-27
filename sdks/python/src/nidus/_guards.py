"""Argument checks for the mistakes Python's type system cannot express.

This module exists for one Python-specific hazard, and it has a file of its own because
three modules need the same rule and a copy in each is how they drift:

**A ``str`` IS a ``Sequence[str]``.** So every ``ids`` / ``scope`` / ``fields`` /
``values`` parameter in this SDK statically accepts a bare string, and iterating it
yields one character per element. ``mypy --strict`` reports nothing, and the request that
comes out is *well-formed* — ``delete("docs", "x1")`` asks the server to delete ids
``"x"`` and ``"1"``, ``scope="docs"`` searches five collections that do not exist. The
server answers 200. The JS SDK cannot express this mistake at all (a TS ``string`` is not
a ``string[]``), which is exactly why the Python SDK has to catch it at runtime: the
alternative is a silent wrong answer with no error anywhere in the stack.

The second helper covers the other end of the same asymmetry. In Python an embedding
almost always arrives from ``numpy``, and ``list(np.asarray(x, dtype=np.float32))`` yields
``np.float32`` scalars, which — unlike ``np.float64`` — are **not** a ``float`` subclass,
so ``json.dumps`` refuses them with a message naming neither nidus nor the argument. So
vectors are coerced with ``float()`` on the way out (as the response decoder already does
on the way in), which handles numpy and torch scalars and ``Decimal`` in one step.

Every message names the offending parameter and shows the fix, because the whole point is
to convert a silent wrong answer into a one-line correction at the call site.
"""

from __future__ import annotations

from typing import Any


def reject_bare_string(items: Any, who: str) -> None:
    """Refuse a single ``str``/``bytes`` where a sequence of them is meant."""
    if isinstance(items, (str, bytes)):
        raise TypeError(
            f"{who} expects a sequence, not a single {type(items).__name__} — "
            f"did you mean [{items!r}]?"
        )


def str_sequence(items: Any, who: str) -> list[str]:
    """Copy a sequence of strings, refusing a bare string passed in its place."""
    reject_bare_string(items, who)
    return [str(x) for x in items]


def float_sequence(values: Any, who: str) -> list[float]:
    """Copy a vector as plain ``float``s, coercing numpy/torch scalars and ``Decimal``.

    A genuinely non-numeric element fails here, naming the value, rather than four frames
    deeper inside the json encoder.
    """
    reject_bare_string(values, who)
    out: list[float] = []
    for x in values:
        try:
            out.append(float(x))
        except (TypeError, ValueError) as err:
            raise TypeError(f"{who} expects numbers, got {x!r}") from err
    return out
