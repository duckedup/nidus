"""The 3.9 floor, enforced rather than asserted.

``pyproject.toml`` declares ``requires-python = ">=3.9"`` and a 3.9 classifier, and
``py.typed`` says the annotations are the contract callers see. Those two promises together
mean every public annotation has to be **evaluable** on 3.9, not merely parseable:
``from __future__ import annotations`` defers the expression, but ``typing.get_type_hints``
still runs it, and on 3.9 ``list[float] | None`` raises ``TypeError: unsupported operand
type(s) for |``.

That matters because resolving hints at runtime is ordinary: a pydantic or cattrs adapter
over ``Record``, a dataclass-to-JSON helper, a FastAPI response model, Sphinx autodoc with
typehints. Each of those crashes on the import of *its own* model, not on anything the
caller wrote — and mypy is perfectly happy with PEP 604 under the future import, so nothing
else in CI notices. Hence this file: it does what those tools do.

It is a real regression test only when the interpreter running it is 3.9 (``just
sdk-py-test`` builds its venv from whatever ``python3`` is on PATH, and CI's matrix covers
the floor). On a newer interpreter it still passes, and still catches an annotation naming a
type that does not resolve at all.
"""

from __future__ import annotations

import types
import typing
from typing import Any

import pytest

import nidus
from nidus import (
    Aggregation,
    AnnInfo,
    Annotations,
    ClauseScore,
    Footprint,
    Fragment,
    FtsClause,
    FtsField,
    Highlight,
    HighlightOpts,
    Hit,
    LegScore,
    LimitPer,
    NidusClient,
    OrderBy,
    PlanCandidates,
    PlanNarrowing,
    PlanTimings,
    QueryPlan,
    Record,
    RecordInput,
    RememberResult,
    Stats,
)

# Every public response shape, plus the TypedDicts callers pass in — whose hints the same
# tools resolve the same way.
PUBLIC_SHAPES = [
    Hit,
    Record,
    Footprint,
    AnnInfo,
    Stats,
    Aggregation,
    Annotations,
    RememberResult,
    LegScore,
    ClauseScore,
    Highlight,
    Fragment,
    RecordInput,
    FtsField,
    FtsClause,
    HighlightOpts,
    LimitPer,
    OrderBy,
    PlanCandidates,
    PlanNarrowing,
    PlanTimings,
    QueryPlan,
]

# Every method of the sync client — the whole documented call surface, plus the dunders a
# `with` block goes through. The async twin's signatures are asserted to match it
# method-for-method in `test_aio`, so covering this one covers both without making `httpx` a
# requirement of this file.
CLIENT_METHODS = [
    name for name, member in vars(NidusClient).items() if isinstance(member, types.FunctionType)
]


@pytest.mark.parametrize("shape", PUBLIC_SHAPES, ids=lambda s: str(s.__name__))
def test_public_shapes_resolve_their_type_hints(shape: Any) -> None:
    """What a pydantic/cattrs adapter or Sphinx autodoc does to these dataclasses."""
    hints = typing.get_type_hints(shape)
    assert hints, f"{shape.__name__} has no annotations at all — the test is not looking"


@pytest.mark.parametrize("name", CLIENT_METHODS)
def test_client_signatures_resolve_their_type_hints(name: str) -> None:
    """Same for the call surface: ``help()``, autodoc, and any decorator reading hints."""
    typing.get_type_hints(getattr(NidusClient, name))


def test_the_async_client_is_not_reachable_by_a_star_import() -> None:
    """``__all__`` must not name a lazily-resolved symbol; see ``nidus/__init__.py``.

    Pinned here as well as in ``test_client`` (which proves the runtime behaviour with
    ``httpx`` absent) because this is the one-line invariant that keeps it true: anything
    added to ``__all__`` is resolved eagerly by ``import *``.
    """
    assert "AsyncNidusClient" not in nidus.__all__
    # Every *other* name in `__all__` must exist as a real attribute.
    for name in nidus.__all__:
        assert getattr(nidus, name, None) is not None, name
