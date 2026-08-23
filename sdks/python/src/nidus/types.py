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
class LegScore:
    """One fusion leg's own view of a hit: its rank there (0-based) and its score there."""

    rank: int
    score: float


@dataclass(frozen=True)
class Expansion:
    """A prefix clause's expansion: ``matched > scored`` means the cap truncated it."""

    matched: int
    scored: int


@dataclass(frozen=True)
class ClauseScore:
    """One text clause's own BM25 contribution. Only clauses that matched are reported."""

    field: str
    score: float
    #: Present only for a prefix clause; the server omits it for an exact one.
    expansion: Optional[Expansion] = None


@dataclass(frozen=True)
class Fragment:
    """An excerpt of a field's stored text, plus the ranges within it that a term matched.

    ``spans`` are ``(start, end)`` **byte** offsets into this fragment's ``text``, because
    the server indexes Rust strings — so ``text.encode()[start:end]`` is the matched run,
    while ``text[start:end]`` is only the same thing while the fragment stays ASCII. They
    cover the *surface* form, so a stemmed match highlights "running" for the query "run".
    """

    text: str
    spans: list[tuple[int, int]]


@dataclass(frozen=True)
class Highlight:
    """The fragments found in one full-text field."""

    field: str
    fragments: list[Fragment]


@dataclass(frozen=True)
class Annotations:
    """Why a hit matched, mirroring the server's ``Annotations``.

    Every part is opt-in — ``explain=True`` for the scores, ``highlight=`` for the
    fragments — and the server omits what was not asked for, so an absent part is an empty
    list (or ``None``), never a zero score. ``vector``/``text`` are the two hybrid fusion
    legs and stay ``None`` on a pure text search, which has only one leg to report.
    """

    vector: Optional[LegScore] = None
    text: Optional[LegScore] = None
    clauses: list[ClauseScore] = field(default_factory=list)
    highlights: list[Highlight] = field(default_factory=list)


@dataclass(frozen=True)
class Hit:
    """A search/list result row, with ``attrs`` decoded to plain Python values."""

    collection: str
    id: str
    score: float
    attrs: dict[str, DecodedValue] = field(default_factory=dict)
    #: ``None`` unless the query asked to ``explain`` or to highlight; the server omits it.
    annotations: Optional[Annotations] = None
    #: The hit's chunk widened with its neighbours. ``None`` unless the query asked to
    #: ``expand`` (or, on recall, to ``rollup``); the server omits the key entirely.
    context: Optional[str] = None


@dataclass(frozen=True)
class PlanCandidates:
    """What an index walk surfaced vs what survived, mirroring the server's ``Candidates``.

    Present only when a walk actually ran (absent on the ``ann``/``segmented`` paths, which
    skip the walk entirely).
    """

    surfaced: int
    survived: int
    dropped_out_of_scope: int
    dropped_stale: int
    dropped_filtered: int
    dropped_min_score: int


@dataclass(frozen=True)
class PlanNarrowing:
    """Whether the opt-in filter index narrowed the scan before it ran.

    ``state`` is a plain ``str`` (``"inactive"``, ``"declined"``, or ``"narrowed"``), not an
    enum, so a value a newer server invents does not raise here. ``candidates`` is present
    only when ``state == "narrowed"``.
    """

    state: str
    candidates: Optional[int] = None


@dataclass(frozen=True)
class PlanTimings:
    """Per-phase wall time in **microseconds**; a phase that did not run is ``None``.

    ``total_us`` is the only key every plan carries.
    """

    total_us: int
    narrow_us: Optional[int] = None
    gather_us: Optional[int] = None
    walk_us: Optional[int] = None
    resolve_us: Optional[int] = None
    first_pass_us: Optional[int] = None
    rescore_us: Optional[int] = None
    score_us: Optional[int] = None


@dataclass(frozen=True)
class QueryPlan:
    """How a query was answered, returned alongside the hits by a ``*_with_plan`` method.

    ``path`` is a plain ``str`` (``"ann"``, ``"ann_prefilter_fallback"``, ``"segmented"``,
    ``"quantized"``, ``"exact"``), not an enum, so an unknown value from a newer server
    decodes rather than raising. ``rows_scanned`` is absent on the ``ann``/``segmented``
    paths, where no full scan happens.
    """

    path: str
    narrowing: PlanNarrowing
    timings: PlanTimings
    rows_scanned: Optional[int] = None
    candidates: Optional[PlanCandidates] = None


@dataclass(frozen=True)
class Aggregation:
    """The answer to :meth:`~nidus.NidusClient.aggregate`: a count plus one sum per field.

    ``sums`` holds decoded plain values as ``attrs`` do, and the Python type carries the
    server's own: a run of ``Int`` s sums to an ``int``, a run that met one ``Float`` sums
    to a ``float``. Every requested field gets an entry — a missing or non-numeric value is
    skipped rather than counted as zero, so a field nothing matched sums to ``0``.
    """

    count: int
    sums: dict[str, DecodedValue] = field(default_factory=dict)
    #: One row per distinct ``group_by`` value, largest first. Empty when none was asked for.
    groups: list[Group] = field(default_factory=list)
    #: Distinct values outran the server's cap and later ones were dropped.
    groups_truncated: bool = False


@dataclass(frozen=True)
class Group:
    """One distinct ``group_by`` value with the aggregates over just its records.

    ``value`` is ``None`` for the records missing the attribute entirely — a different group
    from those holding a present ``Null``, matching how the filter predicates treat the two.
    """

    value: Optional[DecodedValue]
    count: int
    sums: dict[str, DecodedValue] = field(default_factory=dict)


#: What every search-family call returns. It has a name because both clients carry a
#: ``list()`` method (JS parity), which shadows the builtin ``list`` *inside their class
#: bodies* — so a bare ``list[Hit]`` annotation on any method declared after it would
#: resolve to the method rather than to the type. Naming the type once is cheaper than
#: renaming a method the other SDKs also call ``list``.
Hits = list[Hit]

#: What :meth:`~nidus.NidusClient.batch_search` returns: one :data:`Hits` per query. Named for
#: the same reason as :data:`Hits` — a bare ``list[Hits]`` inside a client class body would
#: resolve to that class's own ``list()`` method rather than to the builtin.
Batch = list[Hits]


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
    #: Memory held by the opt-in filter index; ``0`` when none is declared. Defaulted so a
    #: client stays compatible with a server that predates the field.
    filter_index_bytes: int = 0


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


@dataclass(frozen=True)
class Readiness:
    """The ``/ready`` verdict. A ``503`` is an answer here, not a fault."""

    ready: bool
    role: Optional[str] = None
    staleness_secs: Optional[int] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class ClusterStatus:
    """Role and lease state, mirroring the ``/cluster`` response."""

    role: str
    cluster: bool
    holds_writer_handle: bool
    fenced: bool
    lease_owner: Optional[str]
    commit_version: int
    staleness_secs: int
    max_staleness_secs: Optional[int]


@dataclass(frozen=True)
class StoreVersions:
    """Readable commit points and this instance's pin, mirroring the ``/versions`` response."""

    commit_version: int
    oldest_readable: Optional[int]
    pinned: Optional[int]
    readable: list[int]


@dataclass(frozen=True)
class RememberResult:
    """What a :meth:`~nidus.NidusClient.remember` actually wrote.

    ``id`` is the record that changed, which is *not* the requested id when ``deduped``: a
    ``dedupe_threshold`` match redirects the write onto the entry it matched, so this is
    the only way to learn which memory now holds the text.
    """

    id: str
    upserted: int
    deduped: bool


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


class _FilterIndexFieldRequired(TypedDict):
    """The one key every filter-index field must carry (so the rest can be optional)."""

    field: str


class FilterIndexField(_FilterIndexFieldRequired, total=False):
    """One entry of a :meth:`~nidus.NidusClient.set_filter_index` declaration: the attribute
    to index for the text predicates (``Fuzzy``, ``ContainsAllTokens``, ``ContainsAnyToken``,
    ``ContainsTokenSequence``, ``Regex``), plus which structures to build for it.

    Both structures default to on, so ``{"field": "body"}`` means exactly what the bare
    string ``"body"`` means. Declaring an index changes how fast those predicates run,
    never what they return.
    """

    tokens: bool
    trigrams: bool


class _FtsClauseRequired(TypedDict):
    """The two keys every clause must carry (split out so the rest can be optional)."""

    field: str
    query: str


class FtsClause(_FtsClauseRequired, total=False):
    """One clause of a multi-field text query: an indexed field and *its own* query text.

    ``field``/``query`` are required, and the text key is ``query`` in every clause,
    including :meth:`~nidus.NidusClient.hybrid_search`'s single-field spelling (``text``).
    ``prefix`` matches the clause's final term as a prefix; unset stays absent on the wire.
    """

    prefix: bool


class HighlightOpts(TypedDict, total=False):
    """How much text a highlight carries. ``{}`` (or ``True``) takes the server's defaults.

    ``fragment_chars`` is a **character** budget per fragment, cut on char boundaries.
    """

    max_fragments: int
    fragment_chars: int


class RerankOpts(TypedDict, total=False):
    """Opt into the hosted cross-encoder stage.

    The server ranks ``(offset + top_k) * overscan`` deep, scores each candidate's text
    against ``query``, and returns the caller's page of that. Needs a server started with
    ``--rerank-provider``; without one the request is a 400, never a silent pass-through.

    ``query`` is required on :meth:`~nidus.NidusClient.search` and
    :meth:`~nidus.NidusClient.hybrid_search`, and defaults to the request's own text on
    :meth:`~nidus.NidusClient.recall` and on the single-field spelling of
    :meth:`~nidus.NidusClient.text_search`. ``overscan`` defaults to 10, ``text_attr`` to
    ``"nidus.text"``.
    """

    query: str
    overscan: int
    text_attr: str


class LimitPer(TypedDict):
    """Cap how many hits may carry any one value of an attribute — "2 hits per file".

    Records *missing* the attribute form one shared group, so an absent value cannot bypass
    the cap. Both keys are required; ``max`` must be at least 1.
    """

    field: str
    max: int


class _ExpandRequired(TypedDict):
    """The one key every ``expand`` must carry (split out so the fields can default)."""

    radius: int


class Expand(_ExpandRequired, total=False):
    """Widen each hit with the neighbouring chunks of its own document.

    Payload only: it fills :attr:`~nidus.types.Hit.context` and changes nothing about the
    ranking. Every key but ``radius`` defaults to the reserved attrs ``nidus ingest`` stamps
    (``nidus.parent_id``, ``nidus.chunk_index``, ``nidus.text``), so ``{"radius": 1}`` is the
    whole option a chunked corpus needs.
    """

    parent_field: str
    index_field: str
    text_field: str


class Rollup(TypedDict, total=False):
    """Read a chunked corpus as documents rather than fragments.

    ``per_parent`` chunks are kept per document (default 1, the best-matching chunk), each
    widened with ``neighbours`` chunks either side. The text-native spelling of
    ``limit_per`` plus :class:`Expand`.
    """

    per_parent: int
    neighbours: int


class _OrderByRequired(TypedDict):
    """The one key every ``order_by`` must carry (split out so ``descending`` can default)."""

    field: str


class OrderBy(_OrderByRequired, total=False):
    """Sort a :meth:`~nidus.NidusClient.list` by an attribute instead of storage order.

    ``descending`` defaults to ascending. Values of a different type than the first
    orderable one, unorderable values (``Null``/list/``nan``), and records missing the
    attribute sort into one trailing bucket, which stays trailing in either direction.
    """

    descending: bool
