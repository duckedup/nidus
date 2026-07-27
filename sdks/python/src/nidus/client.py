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
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import _wire
from .filter import Filter
from .types import Hits, Record, RecordInput, Stats
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

    def set_fts_schema(self, name: str, fields: Sequence[str]) -> None:
        """Declare which attribute fields are full-text indexed for a collection."""
        self._request("POST", _wire.fts_schema_path(name), _wire.fts_schema_body(fields))

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
        min_score: Optional[float] = None,
        filter: Optional[Filter] = None,  # noqa: A002
    ) -> Hits:
        """Vector (cosine) nearest-neighbour search. An empty ``scope`` searches everything."""
        return self._search(
            _wire.SEARCH,
            _wire.search_body(query, scope=scope, top_k=top_k, min_score=min_score, filter=filter),
        )

    def text_search(
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
        return self._search(
            _wire.TEXT_SEARCH,
            _wire.text_search_body(
                field, query, scope=scope, top_k=top_k, min_score=min_score, filter=filter
            ),
        )

    def hybrid_search(
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
        return self._search(
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

    def list(
        self,
        *,
        scope: Optional[Sequence[str]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        filter: Optional[Filter] = None,  # noqa: A002
    ) -> Hits:
        """Metadata-only listing (no vector query), paginated by ``offset``/``limit``."""
        return self._search(
            _wire.LIST, _wire.list_body(scope=scope, offset=offset, limit=limit, filter=filter)
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
    ) -> None:
        """Embed ``text`` and upsert it under ``id`` (idempotent on ``id``).

        With ``mode="summarize"`` the server summarizes first, embeds the summary, and
        stamps ``nidus.summary``/``nidus.source`` attrs (needs a summarizer configured).
        """
        self._request(
            "POST", _wire.remember_path(collection), _wire.remember_body(id, text, mode, attrs)
        )

    def recall(
        self,
        collection: str,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter: Optional[Filter] = None,  # noqa: A002
    ) -> Hits:
        """Embed ``query`` and vector-search ``collection``, best first."""
        return self._search(
            _wire.recall_path(collection),
            _wire.recall_body(query, top_k=top_k, min_score=min_score, filter=filter),
        )

    # ── Maintenance ──────────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Force a durability flush."""
        self._request("POST", _wire.FLUSH, _wire.empty_body())

    def compact(self) -> None:
        """Compact the store, reclaiming space from deleted and overwritten rows."""
        self._request("POST", _wire.COMPACT, _wire.empty_body())

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
