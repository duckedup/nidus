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
from typing import Any, Optional

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
from .types import Hits, Record, RecordInput, Stats
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

    async def set_fts_schema(self, name: str, fields: Sequence[str]) -> None:
        """Declare which attribute fields are full-text indexed for a collection."""
        await self._request("POST", _wire.fts_schema_path(name), _wire.fts_schema_body(fields))

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
        min_score: Optional[float] = None,
        filter: Optional[Filter] = None,  # noqa: A002
    ) -> Hits:
        """Vector (cosine) nearest-neighbour search. An empty ``scope`` searches everything."""
        return await self._search(
            _wire.SEARCH,
            _wire.search_body(query, scope=scope, top_k=top_k, min_score=min_score, filter=filter),
        )

    async def text_search(
        self,
        *,
        field: str,
        query: str,
        scope: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter: Optional[Filter] = None,  # noqa: A002
    ) -> Hits:
        """BM25 full-text search over one indexed field."""
        return await self._search(
            _wire.TEXT_SEARCH,
            _wire.text_search_body(
                field, query, scope=scope, top_k=top_k, min_score=min_score, filter=filter
            ),
        )

    async def hybrid_search(
        self,
        *,
        vector: Sequence[float],
        field: str,
        text: str,
        scope: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        filter: Optional[Filter] = None,  # noqa: A002
        rrf_k: Optional[float] = None,
        candidates: Optional[int] = None,
    ) -> Hits:
        """Hybrid search: fuse a vector query and a BM25 text query via RRF."""
        return await self._search(
            _wire.HYBRID_SEARCH,
            _wire.hybrid_search_body(
                vector,
                field,
                text,
                scope=scope,
                top_k=top_k,
                filter=filter,
                rrf_k=rrf_k,
                candidates=candidates,
            ),
        )

    async def list(
        self,
        *,
        scope: Optional[Sequence[str]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        filter: Optional[Filter] = None,  # noqa: A002
    ) -> Hits:
        """Metadata-only listing (no vector query), paginated by ``offset``/``limit``."""
        return await self._search(
            _wire.LIST, _wire.list_body(scope=scope, offset=offset, limit=limit, filter=filter)
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
    ) -> None:
        """Embed ``text`` and upsert it under ``id`` (idempotent on ``id``)."""
        await self._request(
            "POST", _wire.remember_path(collection), _wire.remember_body(id, text, mode, attrs)
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
