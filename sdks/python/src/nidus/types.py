"""The response shapes, as frozen dataclasses mirroring ``src/server/dto.rs``.

Responses are dataclasses rather than raw dicts because these shapes are stable and
worth naming: attribute access is checked, ``repr`` is readable in a REPL, and a typo
fails at the call site instead of returning ``None`` from a ``.get``. They are frozen
because they describe a snapshot the server already took — mutating a ``Stats`` would
only ever be a way to lie to later code.

Two distinctions here are load-bearing and must survive the round trip:

* ``Stats.ann is None`` means the store does exact brute-force search — not "an ANN index
  with default settings". The server sends ``null``.
* ``Record.vector is None`` means a text-only document (indexed by FTS/metadata only,
  never by vector search); the server *omits* the field. That is a different fact from an
  empty list, so ``None`` and ``[]`` are kept apart.

``AnnInfo``'s per-algorithm knobs are optional for the same reason: the server emits only
the ones that apply to the active index kind (HNSW's ``m``/``ef_*``, IVF's ``n_*``) and
skips the inert ones, so absent means "does not apply here".

Following the JS SDK, ``attrs`` on both ``Hit`` and ``Record`` hold **decoded** plain
Python values, not wire ``Value`` dicts — callers get ``"rust"``, not ``{"Str": "rust"}``.

**Why ``Optional[X]`` and not ``X | None``** — here and throughout the package. These
dataclasses are the shapes callers hold, so they are exactly what a runtime type
introspector reaches for: a pydantic/cattrs adapter, a dataclass-to-JSON helper, a FastAPI
response model, Sphinx autodoc with typehints. ``from __future__ import annotations``
defers an annotation but does not make it *evaluable*, and on 3.9 — this package's declared
floor — ``typing.get_type_hints(Stats)`` still has to run ``AnnInfo | None`` and raises
``TypeError: unsupported operand type(s) for |``. mypy accepts PEP 604 under the future
import, so nothing in CI would have caught it; ``tests/test_typing.py`` pins it instead,
and ruff's ``keep-runtime-typing`` stops the linter rewriting it back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional, TypedDict

from .values import AttrInput, DecodedValue


@dataclass(frozen=True)
class Hit:
    """A search/list result row, with ``attrs`` decoded to plain Python values."""

    collection: str
    id: str
    score: float
    attrs: dict[str, DecodedValue] = field(default_factory=dict)


#: What every search-family call returns. It has a name because both clients carry a
#: ``list()`` method (JS parity), which shadows the builtin ``list`` *inside their class
#: bodies* — so a bare ``list[Hit]`` annotation on any method declared after it would
#: resolve to the method rather than to the type. Naming the type once is cheaper than
#: renaming a method the other SDKs also call ``list``.
Hits = list[Hit]


@dataclass(frozen=True)
class Record:
    """A record read back from the server, with ``attrs`` decoded to plain values."""

    id: str
    #: ``None`` for a text-only doc (the field is absent on the wire), distinct from ``[]``.
    vector: Optional[list[float]] = None
    attrs: dict[str, DecodedValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Footprint:
    """On-disk footprint, mirroring ``FootprintDto``."""

    rows: int
    dead_rows: int
    dimension: int
    vector_bytes: int
    doc_count: int


@dataclass(frozen=True)
class AnnInfo:
    """Active ANN-index configuration, mirroring ``AnnDto``.

    ``kind`` is the server's own spelling of the variant (``"Hnsw"`` / ``"Ivf"``); it is
    passed through verbatim rather than normalized, so a future kind needs no SDK change.
    """

    kind: str
    overscan: int
    seed: int
    # HNSW only.
    m: Optional[int] = None
    ef_construction: Optional[int] = None
    ef_search: Optional[int] = None
    # IVF only.
    n_lists: Optional[int] = None
    n_probe: Optional[int] = None


@dataclass(frozen=True)
class Stats:
    """Store-wide introspection, mirroring the ``/stats`` response."""

    dimension: int
    distance: str
    #: ``None`` when the store does exact search.
    ann: Optional[AnnInfo]
    collections: list[str]
    footprint: Footprint


class _RecordRequired(TypedDict):
    """The one field every record must carry (split out so the rest can be optional)."""

    id: str


class RecordInput(_RecordRequired, total=False):
    """A record to upsert: ``id``, an optional ``vector``, and ``attrs``.

    ``attrs`` takes plain Python values (or ``v.*`` helpers) and the SDK normalizes them.
    Omitting ``vector`` stores a text-only doc. Omitting ``attrs`` is allowed here even
    though the wire field is mandatory — the request builder sends ``{}`` for you.
    """

    vector: Sequence[float]
    attrs: Mapping[str, AttrInput]


class _FtsFieldRequired(TypedDict):
    """The one key every FTS field must carry (split out so the rest can be optional)."""

    field: str


class FtsField(_FtsFieldRequired, total=False):
    """One entry of a :meth:`~nidus.NidusClient.set_fts_schema` schema: the attribute to
    full-text index, plus any BM25 or analyzer knobs to override for it.

    Every knob is optional and omitted when unset, so ``{"field": "body"}`` means exactly
    what the bare string ``"body"`` means: the server's defaults (``k1 = 1.2``,
    ``b = 0.75``, US English, no ASCII folding, no token-length cap).
    """

    k1: float
    b: float
    language: str
    ascii_folding: bool
    max_token_len: int
