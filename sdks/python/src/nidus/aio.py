"""``AsyncNidusClient`` — the same surface as :class:`~nidus.client.NidusClient`, awaited.

Importing this module is what opts into ``httpx``. That is the whole reason the async
client lives in its own file: ``pip install nidus`` must pull nothing (the sync client is
:mod:`urllib.request` only), so the one third-party dependency is quarantined behind an
extra — ``pip install nidus[async]`` — and behind an import nobody pays for unless they
ask for it. :mod:`nidus`'s ``__init__`` therefore never imports this module eagerly.

``httpx`` rather than a hand-rolled ``asyncio`` HTTP client because the interesting part of
an async client is connection pooling, HTTP/1.1 keep-alive, and timeout handling done
correctly — none of which is worth reimplementing, and all of which is why an async caller
reaches for this client in the first place.

Like the sync client, this file owns **transport only**. Paths, bodies, pruning, decoding,
and error extraction all come from :mod:`nidus._wire`, so the two clients cannot drift:
diff them and you should see ``await`` and ``httpx``, not two copies of the wire contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Optional, Union

try:
    import httpx
except ModuleNotFoundError as err:  # pragma: no cover - depends on the install shape
    # A bare ModuleNotFoundError here tells the caller a module is missing but not that
    # they hit an intentional, documented boundary or how to cross it. Name the fix.
    raise ImportError(
        "nidus.aio requires httpx, an optional dependency of nidus. "
        "Install it with: pip install nidus[async]  (the synchronous nidus.NidusClient "
        "needs nothing beyond the standard library)"
    ) from err

from . import _wire
from .filter import Filter
from .ranking import RankBy
from .types import (
    Aggregation,
    Batch,
    ClusterStatus,
    FtsClause,
    FilterIndexField,
    FtsField,
    HighlightOpts,
    Hits,
    LimitPer,
    OrderBy,
    Readiness,
    Record,
    RecordInput,
    RememberResult,
    Stats,
)
from .values import AttrInput

# `Optional[X]` rather than `X | None`, as everywhere else here: on the 3.9 floor a PEP 604
# annotation is not evaluable at runtime (see the note atop `types.py`).


class AsyncNidusClient:
    """An ``asyncio`` client for one ``nidus serve`` instance.

    Mirrors :class:`~nidus.client.NidusClient` method for method, with ``async def`` and
    :meth:`aclose`. Usable as ``async with AsyncNidusClient(...) as db:``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        """Configure a client and its underlying ``httpx.AsyncClient``.

        :param base_url: e.g. ``http://127.0.0.1:7700``; trailing slashes are stripped.
        :param token: bearer token, when the server was started with ``--token``.
        :param timeout: per-request timeout in **seconds**; ``None`` means no timeout.
        :param headers: extra headers sent on every request.
        :param transport: an ``httpx`` transport — a pre-tuned pool, or an
            ``httpx.MockTransport`` so a test needs no server. This is the async
            counterpart of the sync client's ``transport=`` callable; each takes the
            natural extension point of its own HTTP stack rather than inventing one.

        Note ``timeout=None`` means *no* timeout, matching the sync client, rather than
        ``httpx``'s own 5-second default — one client, one documented meaning.
        """
        self._base_url = _wire.normalize_base_url(base_url)
        self._token = token
        self._headers = dict(headers or {})
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    # ── Admin / introspection ────────────────────────────────────────────────────────

    async def health(self) -> bool:
        """Liveness check. ``True`` when the server answers ``/health``; never raises."""
        try:
            status, _ = await self._send("GET", _wire.HEALTH, None)
        except Exception:
            return False
        return _wire.is_success(status)

    async def ready(self) -> Readiness:
        """Whether this instance can serve: store open, not fenced, not stale.

        A ``503`` is the negative answer rather than an exception, so a poll loop branches on
        ``.ready`` instead of catching. Any other failure still raises ``NidusError``.
        """
        status, text = await self._send("GET", _wire.READY, None)
        return _wire.decode_readiness(status, text)

    async def cluster(self) -> ClusterStatus:
        """Role, writer-handle state, fencing token, commit counter, staleness."""
        return _wire.decode_cluster(await self._request("GET", _wire.CLUSTER))

    async def stats(self) -> Stats:
        """Store-wide introspection: dimension, distance, ANN config, collections, footprint."""
        return _wire.decode_stats(await self._request("GET", _wire.STATS))

    async def collections(self) -> list[str]:
        """List every collection name."""
        return _wire.decode_collections(await self._request("GET", _wire.COLLECTIONS))

    async def create_collection(self, name: str) -> None:
        """Create a collection. Idempotent on the server side."""
        await self._request("POST", _wire.collection_path(name), _wire.empty_body())

    async def drop_collection(self, name: str) -> None:
        """Drop a collection and all its records."""
        await self._request("DELETE", _wire.collection_path(name))

    async def get_meta(self, name: str) -> dict[str, str]:
        """Read a collection's free-form string metadata."""
        return _wire.decode_meta(await self._request("GET", _wire.meta_path(name)))

    async def set_meta(self, name: str, meta: Mapping[str, str]) -> None:
        """Replace a collection's free-form string metadata (a whole-map replace)."""
        await self._request("PUT", _wire.meta_path(name), _wire.meta_body(meta))

    # ── Data ─────────────────────────────────────────────────────────────────────────

    async def upsert(self, name: str, records: Sequence[RecordInput]) -> int:
        """Insert or replace records (idempotent on ``id``). Returns the number upserted."""
        payload = await self._request("POST", _wire.upsert_path(name), _wire.upsert_body(records))
        return _wire.decode_upserted(payload)

    async def delete(self, name: str, ids: Sequence[str]) -> int:
        """Delete records by id. Returns the number deleted."""
        payload = await self._request("POST", _wire.delete_path(name), _wire.delete_ids_body(ids))
        return _wire.decode_deleted(payload)

    async def delete_where(self, name: str, filter: Filter) -> int:  # noqa: A002
        """Delete every record matching ``filter``. Returns the number deleted.

        An **empty** filter raises ``ValueError`` rather than deleting the whole collection
        — the server reads ``[]`` as "match everything". Same rule as the sync client; see
        :meth:`nidus.client.NidusClient.delete_where`.
        """
        payload = await self._request(
            "POST", _wire.delete_path(name), _wire.delete_where_body(filter)
        )
        return _wire.decode_deleted(payload)

    async def records(self, name: str) -> list[Record]:
        """Fetch every record in a collection, with ``attrs`` decoded to plain values."""
        return _wire.decode_records(await self._request("GET", _wire.records_path(name)))

    async def set_fts_schema(self, name: str, fields: Sequence[Union[str, FtsField]]) -> None:
        """Declare which attribute fields are full-text indexed for a collection.

        A bare name takes the server's BM25/analyzer defaults; an :class:`~nidus.FtsField`
        mapping tunes ``k1``, ``b``, or the analyzer for that field alone.
        """
        await self._request("POST", _wire.fts_schema_path(name), _wire.fts_schema_body(fields))

    async def set_filter_index(
        self, name: str, fields: Sequence[Union[str, FilterIndexField]]
    ) -> None:
        """Declare which attribute fields are indexed for the text predicates.

        Covers ``Fuzzy``, ``ContainsAllTokens``, ``ContainsAnyToken``,
        ``ContainsTokenSequence`` and ``Regex``. This changes how fast those predicates run,
        never what they return. Pass an empty sequence to drop the declaration.
        """
        await self._request(
            "POST", _wire.filter_index_path(name), _wire.filter_index_body(fields)
        )

    # ── Search ───────────────────────────────────────────────────────────────────────
    #
    # As in the sync client, every optional defaults to `None` = "omit the key", leaving
    # `top_k`/`limit`/`rrf_k`/`candidates` to the server's `#[serde(default)]`.

    async def search(
        self,
        *,
        query: Sequence[float],
        scope: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        offset: Optional[int] = None,
        min_score: Optional[float] = None,
        filter: Optional[Filter] = None,  # noqa: A002
        exact: Optional[bool] = None,
        include_attributes: Optional[Sequence[str]] = None,
        exclude_attributes: Optional[Sequence[str]] = None,
        rank_by: Optional[RankBy] = None,
        limit_per: Optional[LimitPer] = None,
    ) -> Hits:
        """Vector (cosine) nearest-neighbour search. An empty ``scope`` searches everything.

        ``offset`` skips that many top-ranked hits, so successive pages tile one ranking;
        the server refuses ``offset + top_k`` above 10000. ``exact=True`` forces the exact
        scan past any index; the projection arguments are mutually exclusive. ``rank_by``
        and ``limit_per`` are as in :meth:`nidus.client.NidusClient.search`.
        """
        return await self._search(
            _wire.SEARCH,
            _wire.search_body(
                query,
                scope=scope,
                top_k=top_k,
                offset=offset,
                min_score=min_score,
                filter=filter,
                exact=exact,
                include_attributes=include_attributes,
                exclude_attributes=exclude_attributes,
                rank_by=rank_by,
                limit_per=limit_per,
            ),
        )

    async def text_search(
        self,
        *,
        field: Optional[str] = None,
        query: Optional[str] = None,
        scope: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        offset: Optional[int] = None,
        min_score: Optional[float] = None,
        filter: Optional[Filter] = None,  # noqa: A002
        clauses: Optional[Sequence[FtsClause]] = None,
        combine: Optional[str] = None,
        explain: Optional[bool] = None,
        highlight: Optional[Union[bool, HighlightOpts]] = None,
        include_attributes: Optional[Sequence[str]] = None,
        exclude_attributes: Optional[Sequence[str]] = None,
        rank_by: Optional[RankBy] = None,
        limit_per: Optional[LimitPer] = None,
    ) -> Hits:
        """BM25 full-text search, paginated by ``offset``.

        One field (``field`` + ``query``) or several (``clauses`` + ``combine``), never
        both; ``explain``/``highlight`` fill ``hit.annotations``. Same rules as
        :meth:`nidus.client.NidusClient.text_search`.
        """
        return await self._search(
            _wire.TEXT_SEARCH,
            _wire.text_search_body(
                field,
                query,
                scope=scope,
                top_k=top_k,
                offset=offset,
                min_score=min_score,
                filter=filter,
                clauses=clauses,
                combine=combine,
                explain=explain,
                highlight=highlight,
                include_attributes=include_attributes,
                exclude_attributes=exclude_attributes,
                rank_by=rank_by,
                limit_per=limit_per,
            ),
        )

    async def hybrid_search(
        self,
        *,
        vector: Sequence[float],
        field: Optional[str] = None,
        text: Optional[str] = None,
        scope: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        offset: Optional[int] = None,
        filter: Optional[Filter] = None,  # noqa: A002
        rrf_k: Optional[float] = None,
        candidates: Optional[int] = None,
        clauses: Optional[Sequence[FtsClause]] = None,
        combine: Optional[str] = None,
        explain: Optional[bool] = None,
        highlight: Optional[Union[bool, HighlightOpts]] = None,
        vector_weight: Optional[float] = None,
        text_weight: Optional[float] = None,
    ) -> Hits:
        """Hybrid search: fuse a vector query and a BM25 text query via RRF.

        ``offset`` pages the *fused* ranking, never a leg — a leg's rank is an input to
        the fused score. The text leg, the weights, and ``explain`` behave exactly as in
        :meth:`nidus.client.NidusClient.hybrid_search`.
        """
        return await self._search(
            _wire.HYBRID_SEARCH,
            _wire.hybrid_search_body(
                vector,
                field,
                text,
                scope=scope,
                top_k=top_k,
                offset=offset,
                filter=filter,
                rrf_k=rrf_k,
                candidates=candidates,
                clauses=clauses,
                combine=combine,
                explain=explain,
                highlight=highlight,
                vector_weight=vector_weight,
                text_weight=text_weight,
            ),
        )

    async def list(
        self,
        *,
        scope: Optional[Sequence[str]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        filter: Optional[Filter] = None,  # noqa: A002
        include_attributes: Optional[Sequence[str]] = None,
        exclude_attributes: Optional[Sequence[str]] = None,
        order_by: Optional[OrderBy] = None,
    ) -> Hits:
        """Metadata-only listing (no vector query), paginated by ``offset``/``limit``.

        ``order_by`` sorts by an attribute instead of storage order, with the unorderable
        and the absent in one trailing bucket.
        """
        return await self._search(
            _wire.LIST,
            _wire.list_body(
                scope=scope,
                offset=offset,
                limit=limit,
                filter=filter,
                include_attributes=include_attributes,
                exclude_attributes=exclude_attributes,
                order_by=order_by,
            ),
        )

    async def aggregate(
        self,
        *,
        scope: Optional[Sequence[str]] = None,
        filter: Optional[Filter] = None,  # noqa: A002
        sum: Optional[Sequence[str]] = None,  # noqa: A002
        group_by: Optional[str] = None,
    ) -> Aggregation:
        """Count the records a filter matches, and sum the attributes named in ``sum``.

        Answered from the in-RAM index alone — no record is built and no vector is read.
        ``group_by`` adds one :class:`~nidus.Group` per distinct value beside the totals.
        """
        return _wire.decode_aggregation(
            await self._request(
                "POST",
                _wire.AGGREGATE,
                _wire.aggregate_body(scope=scope, filter=filter, sum=sum, group_by=group_by),
            )
        )

    async def batch_search(
        self,
        queries: Sequence[Mapping[str, Any]],
        *,
        fuse: bool = False,
        rrf_k: Optional[float] = None,
        weights: Optional[Sequence[float]] = None,
        top_k: Optional[int] = None,
    ) -> Batch:
        """Answer several vector queries in one round-trip (16 max).

        Returns one ranking per query in request order, or — with ``fuse=True`` — a
        one-element list holding the single RRF-fused ranking.
        """
        bodies = [_wire.search_body(**dict(q)) for q in queries]
        return _wire.decode_batch(
            await self._request(
                "POST",
                _wire.SEARCH_BATCH,
                _wire.batch_search_body(
                    bodies, rrf_k=rrf_k, weights=weights, top_k=top_k, fuse=fuse
                ),
            )
        )

    # ── Memory (text-native) ─────────────────────────────────────────────────────────
    #
    # Present only when the server was started with an embedder; otherwise it answers 400.
    # Wrapped unconditionally, exactly as in the sync client — the server owns that call.

    async def remember(
        self,
        collection: str,
        id: str,  # noqa: A002
        text: str,
        *,
        mode: Optional[str] = None,
        attrs: Optional[Mapping[str, AttrInput]] = None,
        ttl_seconds: Optional[int] = None,
        dedupe_threshold: Optional[float] = None,
    ) -> RememberResult:
        """Embed ``text`` and upsert it under ``id`` (idempotent on ``id``).

        ``ttl_seconds`` expires the memory that long after the write. ``dedupe_threshold``
        folds it onto a near-duplicate above that cosine floor instead — which needs the
        server's embedder, never matches an expired entry, and makes the returned ``id``
        the entry written rather than the one asked for. See the sync twin for the detail.
        """
        return _wire.decode_remember(
            await self._request(
                "POST",
                _wire.remember_path(collection),
                _wire.remember_body(id, text, mode, attrs, ttl_seconds, dedupe_threshold),
            ),
            id,
        )

    async def recall(
        self,
        collection: str,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter: Optional[Filter] = None,  # noqa: A002
    ) -> Hits:
        """Embed ``query`` and vector-search ``collection``, best first."""
        return await self._search(
            _wire.recall_path(collection),
            _wire.recall_body(query, top_k=top_k, min_score=min_score, filter=filter),
        )

    # ── Maintenance ──────────────────────────────────────────────────────────────────

    async def flush(self) -> None:
        """Force a durability flush."""
        await self._request("POST", _wire.FLUSH, _wire.empty_body())

    async def compact(self) -> None:
        """Compact the store, reclaiming space from deleted and overwritten rows."""
        await self._request("POST", _wire.COMPACT, _wire.empty_body())

    async def refresh(self) -> bool:
        """Adopt a writer's newer committed state. Returns whether anything was adopted."""
        return _wire.decode_refresh(await self._request("POST", _wire.REFRESH, _wire.empty_body()))

    # ── Lifetime ─────────────────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Close the connection pool. Idempotent; the client is unusable afterwards."""
        await self._client.aclose()

    async def __aenter__(self) -> AsyncNidusClient:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self.aclose()

    # ── Internals ────────────────────────────────────────────────────────────────────

    async def _search(self, path: str, body: Mapping[str, Any]) -> Hits:
        """Run a search-family POST and decode the hits (attrs to plain Python values)."""
        return _wire.decode_hits(await self._request("POST", path, body))

    async def _request(self, method: str, path: str, body: Any = None) -> Any:
        """Issue a request and hand the raw ``(status, body)`` to ``_wire`` to interpret."""
        status, text = await self._send(method, path, body)
        return _wire.decode_response(status, text)

    async def _send(self, method: str, path: str, body: Any) -> tuple[int, str]:
        """The transport boundary, and the only code here that is not ``_wire``'s call."""
        payload, headers = _wire.prepare(self._token, self._headers, body)
        try:
            response = await self._client.request(
                method, f"{self._base_url}{path}", content=payload, headers=headers
            )
        except httpx.HTTPError as err:
            # `httpx.HTTPError` covers the whole transport family (connect, read, pool,
            # timeout, protocol). A non-2xx *response* never lands here — httpx only
            # raises for status if asked to, and we never ask, so the status flows through
            # to `_request` and becomes a NidusError carrying the server's own message.
            # The status-0 sentinel for "no response at all" and its wording are `_wire`'s,
            # shared verbatim with the sync client.
            raise _wire.transport_error(path, err) from err
        return response.status_code, response.text
