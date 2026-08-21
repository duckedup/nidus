"""``NidusClient`` — the synchronous client over the ``nidus serve`` HTTP API.

One method per endpoint (``src/server/mod.rs``). "Local vs remote" is just the base URL:
point it at a ``nidus serve`` on this machine or at any reachable host.

This file is deliberately thin. Every *decision* about the wire — which path, which body
keys, what gets pruned, which statuses are errors, how a payload decodes, how an error
message is dug out — lives in :mod:`nidus._wire`, shared with the async twin in
:mod:`nidus.aio`, which is also where the :class:`~nidus.errors.NidusError` for an
unreachable server is minted. What is left here is transport: put bytes on a socket and
read bytes back.

**Standard library only.** The transport is :mod:`urllib.request`, so ``pip install nidus``
pulls nothing — the same zero-dependency property the JS SDK has from the platform
``fetch``. The cost is real and worth stating plainly: ``urllib`` opens a fresh connection
per request, which shows up as measurable overhead on a long run of sequential upserts.
The escape hatch is ``transport=`` (JS's ``options.fetch``): hand in a callable backed by
``httpx``/``requests`` and you get pooling, retries, or instrumentation without the SDK
taking on a dependency for everyone. The same seam is what lets the unit tests exercise
the full endpoint surface with no server and no network.

The transport contract is small on purpose: it returns ``(status, text)`` for **any**
response the server produced, including 4xx/5xx — only a genuine failure to get a response
raises. ``urllib`` is the odd one out (it raises ``HTTPError`` for non-2xx), so the default
transport converts that back into a return value; ``httpx``/``requests`` already behave
this way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from http.client import HTTPException
from types import TracebackType
from typing import Any, Callable, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import _wire
from .filter import Filter
from .ranking import RankBy
from .types import (
    Aggregation,
    Batch,
    ClusterStatus,
    FilterIndexField,
    FtsClause,
    FtsField,
    HighlightOpts,
    Hits,
    LimitPer,
    OrderBy,
    Readiness,
    Record,
    RecordInput,
    RememberResult,
    RerankOpts,
    Stats,
)
from .values import AttrInput

# `Optional[X]` rather than `X | None` everywhere in this package, deliberately: see the
# note at the top of `types.py` — on the 3.9 floor a PEP 604 annotation is not evaluable,
# so `typing.get_type_hints` on any of these signatures would raise.

#: A pluggable HTTP transport: ``(method, url, headers, body, timeout) -> (status, text)``.
#:
#: Small and boring by design — it is a seam, not an abstraction layer. ``body`` is
#: ``None`` for a bodyless request and ``timeout`` is ``None`` for "no timeout". A
#: transport must **return** non-2xx statuses rather than raise on them; raising is
#: reserved for "no response at all".
Transport = Callable[
    [str, str, dict[str, str], Optional[bytes], Optional[float]],
    tuple[int, str],
]


class NidusClient:
    """A client for one ``nidus serve`` instance.

    Usable as a context manager (``with NidusClient(...) as db:``); :meth:`close` is
    there for the symmetric case where the caller manages the lifetime itself.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        transport: Optional[Transport] = None,
    ) -> None:
        """Configure a client. Nothing is opened until the first request.

        :param base_url: e.g. ``http://127.0.0.1:7700``; trailing slashes are stripped.
        :param token: bearer token, when the server was started with ``--token``.
        :param timeout: per-request timeout in **seconds**; ``None`` means no timeout.
        :param headers: extra headers sent on every request.
        :param transport: replace the ``urllib`` transport (pooling, retries, tests).
        """
        self._base_url = _wire.normalize_base_url(base_url)
        self._token = token
        self._timeout = timeout
        # Copied so a later mutation of the caller's dict cannot silently change our auth
        # or content-type handling mid-run.
        self._headers = dict(headers or {})
        self._transport: Transport = transport if transport is not None else _urllib_transport

    # ── Admin / introspection ────────────────────────────────────────────────────────

    def health(self) -> bool:
        """Liveness check. ``True`` when the server answers ``/health``.

        Never raises — a health check that throws when the thing is unhealthy is useless
        for the one job it has (deciding whether to bother trying). Any failure, including
        a broken custom transport, is ``False``. Matches the JS SDK exactly.
        """
        try:
            status, _ = self._send("GET", _wire.HEALTH, None)
        except Exception:
            return False
        return _wire.is_success(status)

    def ready(self) -> Readiness:
        """Whether this instance can serve: store open, not fenced, not stale.

        A ``503`` is the negative answer rather than an exception, so a poll loop branches on
        ``.ready`` instead of catching. Any other failure still raises ``NidusError``.
        """
        status, text = self._send("GET", _wire.READY, None)
        return _wire.decode_readiness(status, text)

    def cluster(self) -> ClusterStatus:
        """Role, writer-handle state, fencing token, commit counter, staleness."""
        return _wire.decode_cluster(self._request("GET", _wire.CLUSTER))

    def stats(self) -> Stats:
        """Store-wide introspection: dimension, distance, ANN config, collections, footprint."""
        return _wire.decode_stats(self._request("GET", _wire.STATS))

    def collections(self) -> list[str]:
        """List every collection name."""
        return _wire.decode_collections(self._request("GET", _wire.COLLECTIONS))

    def create_collection(self, name: str) -> None:
        """Create a collection. Idempotent on the server side."""
        self._request("POST", _wire.collection_path(name), _wire.empty_body())

    def drop_collection(self, name: str) -> None:
        """Drop a collection and all its records."""
        self._request("DELETE", _wire.collection_path(name))

    def get_meta(self, name: str) -> dict[str, str]:
        """Read a collection's free-form string metadata."""
        return _wire.decode_meta(self._request("GET", _wire.meta_path(name)))

    def set_meta(self, name: str, meta: Mapping[str, str]) -> None:
        """Replace a collection's free-form string metadata (a whole-map replace)."""
        self._request("PUT", _wire.meta_path(name), _wire.meta_body(meta))

    # ── Data ─────────────────────────────────────────────────────────────────────────

    def upsert(self, name: str, records: Sequence[RecordInput]) -> int:
        """Insert or replace records (idempotent on ``id`` within the collection).

        ``attrs`` take plain Python values or ``v.*`` helpers and are normalized for you.
        Returns the number of records upserted.
        """
        payload = self._request("POST", _wire.upsert_path(name), _wire.upsert_body(records))
        return _wire.decode_upserted(payload)

    def delete(self, name: str, ids: Sequence[str]) -> int:
        """Delete records by id. Returns the number deleted."""
        payload = self._request("POST", _wire.delete_path(name), _wire.delete_ids_body(ids))
        return _wire.decode_deleted(payload)

    def delete_where(self, name: str, filter: Filter) -> int:  # noqa: A002
        """Delete every record matching ``filter``. Returns the number deleted.

        An **empty** filter raises ``ValueError`` rather than deleting the whole
        collection: the server treats ``[]`` as "match everything", and a filter list that
        collapsed to empty because every optional condition was absent is far likelier than
        a deliberate delete-all. Use :meth:`drop_collection` when that *is* the intent.
        """
        payload = self._request("POST", _wire.delete_path(name), _wire.delete_where_body(filter))
        return _wire.decode_deleted(payload)

    def records(self, name: str) -> list[Record]:
        """Fetch every record in a collection, with ``attrs`` decoded to plain values."""
        return _wire.decode_records(self._request("GET", _wire.records_path(name)))

    def set_fts_schema(self, name: str, fields: Sequence[Union[str, FtsField]]) -> None:
        """Declare which attribute fields are full-text indexed for a collection.

        A bare name takes the server's BM25/analyzer defaults; an :class:`~nidus.FtsField`
        mapping tunes ``k1``, ``b``, or the analyzer for that field alone.
        """
        self._request("POST", _wire.fts_schema_path(name), _wire.fts_schema_body(fields))

    def set_filter_index(self, name: str, fields: Sequence[Union[str, FilterIndexField]]) -> None:
        """Declare which attribute fields are indexed for the text predicates.

        Covers ``Fuzzy``, ``ContainsAllTokens``, ``ContainsAnyToken``,
        ``ContainsTokenSequence`` and ``Regex``. This changes how fast those predicates run,
        never what they return: the index proposes candidates and the predicate still
        decides. Pass an empty sequence to drop the declaration.
        """
        self._request("POST", _wire.filter_index_path(name), _wire.filter_index_body(fields))

    # ── Search ───────────────────────────────────────────────────────────────────────
    #
    # Every optional defaults to `None`, which means "omit the key" so the server's
    # `#[serde(default)]` supplies the value. That is why `top_k` is not defaulted to 10
    # here: `top_k=0` is a legitimate request for zero results, so `0` cannot double as
    # "unset", and restating the server's numbers would fork the contract the day one
    # changes.

    def search(
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
        diversity: Optional[float] = None,
        rerank: Optional[RerankOpts] = None,
    ) -> Hits:
        """Vector (cosine) nearest-neighbour search. An empty ``scope`` searches everything.

        ``offset`` skips that many top-ranked hits, so successive pages tile one ranking;
        the server refuses ``offset + top_k`` above 10000. ``exact=True`` forces the exact
        scan past any index; the projection arguments are mutually exclusive.

        ``rank_by`` layers a ranking expression over the metric (``rank.decay(...)``), and
        ``limit_per={"field": "path", "max": 2}`` caps how many hits share one attribute
        value — the cap is applied to the ranking, so it thins results rather than deepening
        the search. ``diversity`` is a Maximal Marginal Relevance lambda spreading hits apart
        in vector space so near-duplicates stop filling a page: ``1.0`` is pure relevance,
        ``0.0`` pure variety, and omitting it leaves the ranking untouched. ``rerank`` is
        documented on :class:`~nidus.RerankOpts`.
        """
        return self._search(
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
                diversity=diversity,
                rerank=rerank,
            ),
        )

    def search_similar(
        self,
        *,
        collection: str,
        id: str,  # noqa: A002
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
        diversity: Optional[float] = None,
    ) -> Hits:
        """Records most like the one already stored at ``collection``/``id``.

        The source record itself is never in the results; a true duplicate of it is. An
        omitted ``scope`` searches the source's own collection, not every collection — the
        one place this differs from :meth:`search`. Every other argument behaves exactly as
        it does there.
        """
        return self._search(
            _wire.SIMILAR,
            _wire.similar_body(
                collection,
                id,
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
                diversity=diversity,
            ),
        )

    def text_search(
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
        diversity: Optional[float] = None,
        rerank: Optional[RerankOpts] = None,
    ) -> Hits:
        """BM25 full-text search, paginated by ``offset``.

        Name the query either as ``field`` + ``query`` (one field) or as ``clauses``, a list
        of ``{"field": …, "query": …}`` each carrying its own text — never both, and never
        an empty list. ``combine`` folds several clauses into one score: ``"Sum"`` (the
        default) rewards a document matching in two fields, ``"Max"`` takes the strongest
        clause so a long body cannot out-accumulate a precise title match.

        ``explain=True`` reports each matched clause's own score, and ``highlight=True``
        (or a ``{"max_fragments": …, "fragment_chars": …}`` mapping) returns fragments of
        the stored text; both land on ``hit.annotations``. Highlighting reads the stored
        text, so it still works on a field the projection dropped. ``rerank`` is documented
        on :class:`~nidus.RerankOpts`; on the single-field spelling its ``query`` defaults
        to this method's own ``query``.
        """
        return self._search(
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
                diversity=diversity,
                rerank=rerank,
            ),
        )

    def hybrid_search(
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
        rerank: Optional[RerankOpts] = None,
    ) -> Hits:
        """Hybrid search: fuse a vector query and a BM25 text query via RRF.

        ``offset`` pages the *fused* ranking, never a leg — a leg's rank is an input to
        the fused score. The text leg takes ``field`` + ``text`` or a ``clauses`` list, on
        the same either/or rule as :meth:`text_search`; a clause spells its text ``query``
        on both routes.

        ``vector_weight``/``text_weight`` scale each leg's contribution to the fused score
        (both default to 1.0, which is the unweighted fusion exactly). With ``explain=True``
        each hit reports both legs' own rank and score in ``hit.annotations``, which is the
        only way to see a leg's rank — the returned score is the fused one. ``rerank`` is
        documented on :class:`~nidus.RerankOpts`; its ``query`` is required here.
        """
        return self._search(
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
                rerank=rerank,
            ),
        )

    def list(
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

        ``order_by={"field": "updated_at", "descending": True}`` sorts by an attribute
        instead of storage order; records missing it — or holding an unorderable value —
        sort into one bucket that stays trailing in either direction.
        """
        return self._search(
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

    def aggregate(
        self,
        *,
        scope: Optional[Sequence[str]] = None,
        filter: Optional[Filter] = None,  # noqa: A002
        sum: Optional[Sequence[str]] = None,  # noqa: A002
        group_by: Optional[str] = None,
    ) -> Aggregation:
        """Count the records a filter matches, and sum the attributes named in ``sum``.

        Answered from the in-RAM index alone — no record is built and no vector is read — so
        it is the cheap way to ask "how many, and how big" without paging through
        :meth:`list`. A missing or non-numeric value is skipped rather than counted as zero.

        ``group_by`` additionally reports one :class:`~nidus.Group` per distinct value of that
        attribute, in the same pass and beside the unchanged whole-scope totals.
        """
        return _wire.decode_aggregation(
            self._request(
                "POST",
                _wire.AGGREGATE,
                _wire.aggregate_body(scope=scope, filter=filter, sum=sum, group_by=group_by),
            )
        )

    def batch_search(
        self,
        queries: Sequence[Mapping[str, Any]],
        *,
        fuse: bool = False,
        rrf_k: Optional[float] = None,
        weights: Optional[Sequence[float]] = None,
        top_k: Optional[int] = None,
    ) -> Batch:
        """Answer several vector queries in one round-trip (16 max), saving a hop per query.

        Each entry of ``queries`` takes the same keys as :meth:`search` and is validated the
        same way; the server checks the whole batch before running any leg, so a malformed
        query fails the call rather than returning a partial answer.

        Returns one ranking per query, in request order. With ``fuse=True`` the legs are
        merged by Reciprocal Rank Fusion and the result is a **one-element** list holding
        that single ranking, so indexing does not change shape with the flag. ``weights``
        must be empty or exactly as long as ``queries``.
        """
        bodies = [_wire.search_body(**dict(q)) for q in queries]
        return _wire.decode_batch(
            self._request(
                "POST",
                _wire.SEARCH_BATCH,
                _wire.batch_search_body(
                    bodies, rrf_k=rrf_k, weights=weights, top_k=top_k, fuse=fuse
                ),
            )
        )

    # ── Memory (text-native) ─────────────────────────────────────────────────────────
    #
    # Present only when `nidus serve` was started with an embedder; otherwise the server
    # answers 400. That is the server's call to make, so these are wrapped
    # unconditionally and the error is left to surface. The client only ever sends text —
    # the embedding happens server-side.

    def remember(
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

        With ``mode="summarize"`` the server summarizes first, embeds the summary, and
        stamps a ``nidus.summary`` attr (needs a summarizer configured). The raw text is
        always stored under ``nidus.text``, whichever mode is used.

        ``ttl_seconds`` expires the memory that long after the write; ``None`` never
        expires, and ``0`` expires it immediately. ``dedupe_threshold`` is a
        cosine-similarity floor above which this write updates the nearest existing entry
        instead of inserting a competing near-duplicate; ``None`` disables dedupe. Dedupe
        is a vector search server-side, so it needs the same embedder ``remember`` does,
        and an already-expired entry is never a candidate — a lapsed TTL cannot be revived
        by a later near-duplicate.

        Read :attr:`~nidus.RememberResult.id` off the result rather than assuming the one
        passed in: a dedupe match redirects the write onto the entry it matched.
        """
        return _wire.decode_remember(
            self._request(
                "POST",
                _wire.remember_path(collection),
                _wire.remember_body(id, text, mode, attrs, ttl_seconds, dedupe_threshold),
            ),
            id,
        )

    def recall(
        self,
        collection: str,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter: Optional[Filter] = None,  # noqa: A002
        diversity: Optional[float] = None,
        rerank: Optional[RerankOpts] = None,
    ) -> Hits:
        """Embed ``query`` and vector-search ``collection``, best first.

        ``rerank`` is documented on :class:`~nidus.RerankOpts`; its ``query`` defaults to
        this method's own ``query`` when omitted.
        """
        return self._search(
            _wire.recall_path(collection),
            _wire.recall_body(
                query,
                top_k=top_k,
                min_score=min_score,
                filter=filter,
                diversity=diversity,
                rerank=rerank,
            ),
        )

    # ── Maintenance ──────────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Force a durability flush."""
        self._request("POST", _wire.FLUSH, _wire.empty_body())

    def compact(self) -> None:
        """Compact the store, reclaiming space from deleted and overwritten rows."""
        self._request("POST", _wire.COMPACT, _wire.empty_body())

    def refresh(self) -> bool:
        """Adopt a writer's newer committed state. Returns whether anything was adopted."""
        return _wire.decode_refresh(self._request("POST", _wire.REFRESH, _wire.empty_body()))

    # ── Lifetime ─────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release anything the transport holds.

        The default ``urllib`` transport is connectionless, so this is a no-op for it. It
        exists so the sync and async clients have the same shape, and so a caller who
        plugs in a *pooled* transport gets its sockets closed: if the transport exposes a
        ``close``, it is called. Idempotent.
        """
        closer = getattr(self._transport, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> NidusClient:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # ── Internals ────────────────────────────────────────────────────────────────────

    def _search(self, path: str, body: Mapping[str, Any]) -> Hits:
        """Run a search-family POST and decode the hits (attrs to plain Python values)."""
        return _wire.decode_hits(self._request("POST", path, body))

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        """Issue a request and hand the raw ``(status, body)`` to ``_wire`` to interpret."""
        status, text = self._send(method, path, body)
        return _wire.decode_response(status, text)

    def _send(self, method: str, path: str, body: Any) -> tuple[int, str]:
        """The transport boundary, and the only code here that is not ``_wire``'s call."""
        payload, headers = _wire.prepare(self._token, self._headers, body)
        try:
            return self._transport(
                method, f"{self._base_url}{path}", headers, payload, self._timeout
            )
        except HTTPError as err:
            # Belt and braces: the default transport already converts this, but a custom
            # urllib-based transport may let it through, and reporting a real 404 as an
            # unreachable server (status 0) would be a lie.
            return _http_error_response(err)
        except (URLError, TimeoutError, OSError, HTTPException) as err:
            # No response at all — refused, DNS failure, unreachable, timed out, or a
            # truncated reply. `socket.timeout` is `TimeoutError` on 3.10+ and an
            # `OSError` subclass on 3.9, so it is covered here too. The status-0 sentinel
            # and the wording are `_wire`'s, shared with the async client.
            raise _wire.transport_error(path, err) from err


def _urllib_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: Optional[bytes],
    timeout: Optional[float],
) -> tuple[int, str]:
    """The default transport: one ``urllib`` request, standard library only.

    ``timeout=None`` is passed straight through and means "block indefinitely", which is
    the documented meaning of an unset timeout on the client.
    """
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except HTTPError as err:
        return _http_error_response(err)


def _http_error_response(err: HTTPError) -> tuple[int, str]:
    """Turn urllib's ``HTTPError`` back into the response it actually is.

    ``HTTPError`` is a response object that urllib chose to raise: it carries the status
    *and* the server's body, which is where the ``{"error": …}`` message lives. Reading
    that body is the difference between reporting "dimension mismatch: expected 3, got 4"
    and reporting a bare "HTTP 400".

    The reads are guarded because an ``HTTPError`` does not always *have* a body stream:
    urllib builds some of them with no file object, and a test constructing one by hand
    almost always does. Whatever goes wrong while fetching the body, the status is already
    in hand and is worth more than a second exception thrown over the top of the first, so
    a failed read degrades to an empty body and lets ``extract_error`` fall back to
    ``HTTP <status>``.
    """
    try:
        raw = err.read()
    except Exception:
        raw = b""
    finally:
        with suppress(Exception):
            err.close()
    return int(err.code), raw.decode("utf-8", "replace")
