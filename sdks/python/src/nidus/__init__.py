"""nidus — the Python client for the ``nidus serve`` HTTP API.

Everything a caller needs is re-exported here, so nothing has to reach into a private
module::

    from nidus import NidusClient, f, v

    with NidusClient("http://127.0.0.1:7700") as db:
        db.create_collection("docs")
        db.upsert("docs", [{"id": "a", "vector": [0.1, 0.2, 0.3], "attrs": {"lang": "rust"}}])
        hits = db.search(query=[0.1, 0.2, 0.3], top_k=5, filter=[f.eq("lang", "rust")])

``NidusClient`` is standard-library only — installing this package pulls **nothing**.

The async twin is imported **lazily**: ``nidus.AsyncNidusClient`` resolves on first
attribute access (PEP 562 ``__getattr__``), which is what keeps ``httpx`` a genuinely
optional dependency instead of one that every ``import nidus`` pays for. Both spellings
work and mean the same thing::

    from nidus.aio import AsyncNidusClient   # explicit
    import nidus; nidus.AsyncNidusClient     # lazy, resolved on access

Either way it needs ``pip install nidus[async]``; without ``httpx`` the failure is an
``ImportError`` that names that fix. ``import nidus`` itself always works — including
``from nidus import *``, which is why ``AsyncNidusClient`` is deliberately absent from
``__all__`` (see below).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import __version__
from .client import NidusClient, Transport
from .errors import NidusError
from .filter import Filter, Predicate, f
from .ranking import RankBy, rank
from .types import (
    Aggregation,
    AnnInfo,
    Annotations,
    Batch,
    ClauseScore,
    ClusterStatus,
    FilterIndexField,
    Footprint,
    Fragment,
    FtsClause,
    FtsField,
    Group,
    Highlight,
    HighlightOpts,
    Hit,
    Hits,
    LegScore,
    LimitPer,
    OrderBy,
    Readiness,
    Record,
    RecordInput,
    RememberResult,
    Stats,
)
from .values import (
    AttrInput,
    DecodedValue,
    Value,
    decode_attrs,
    decode_value,
    encode_attrs,
    encode_value,
    v,
)

if TYPE_CHECKING:
    # For type checkers and IDEs only: the name is real, but resolving it at runtime is
    # deferred to `__getattr__` below so an `import nidus` never touches httpx. The
    # `X as X` spelling marks it an explicit re-export, which is what keeps `mypy
    # --strict` (no_implicit_reexport) happy about `from nidus import AsyncNidusClient`
    # now that `__all__` does not list it.
    from .aio import AsyncNidusClient as AsyncNidusClient

# `AsyncNidusClient` is NOT here on purpose. `from nidus import *` resolves every name in
# `__all__` eagerly — including through the lazy `__getattr__` — so listing it would make
# a star-import of the *sync* client fail with an ImportError about httpx on a plain
# `pip install nidus`, turning the optional dependency into a de facto hard one. The name
# still works as `nidus.AsyncNidusClient` and as `from nidus.aio import AsyncNidusClient`.
__all__ = [
    "Aggregation",
    "AnnInfo",
    "Annotations",
    "AttrInput",
    "Batch",
    "ClauseScore",
    "ClusterStatus",
    "DecodedValue",
    "Filter",
    "FilterIndexField",
    "Footprint",
    "Fragment",
    "FtsClause",
    "FtsField",
    "Group",
    "Highlight",
    "HighlightOpts",
    "Hit",
    "Hits",
    "LegScore",
    "LimitPer",
    "NidusClient",
    "NidusError",
    "OrderBy",
    "Predicate",
    "RankBy",
    "Readiness",
    "Record",
    "RecordInput",
    "RememberResult",
    "Stats",
    "Transport",
    "Value",
    "__version__",
    "decode_attrs",
    "decode_value",
    "encode_attrs",
    "encode_value",
    "f",
    "rank",
    "v",
]


def __getattr__(name: str) -> Any:
    """Resolve ``nidus.AsyncNidusClient`` on first access (PEP 562).

    A plain top-level ``from .aio import AsyncNidusClient`` would make ``httpx`` a de facto
    hard dependency: every ``import nidus`` would fail without it, extra or no extra. Doing
    the import here moves that cost to the one caller who actually asks for the async
    client — and if ``httpx`` is missing, :mod:`nidus.aio` raises an ``ImportError``
    naming the fix, which propagates from here unchanged.
    """
    if name == "AsyncNidusClient":
        from .aio import AsyncNidusClient

        return AsyncNidusClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
