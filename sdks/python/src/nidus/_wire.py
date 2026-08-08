"""Everything about the HTTP contract except the HTTP call itself.

Two clients ship in this package — sync (:mod:`nidus.client`, ``urllib``) and async
(:mod:`nidus.aio`, ``httpx``) — and a sync/async pair is the textbook place for two
near-identical copies of the same knowledge to drift apart. So the split here is
deliberate and strict: this module owns every *decision* (which path, which body keys,
which fields are pruned, which statuses count as success, how a payload decodes, how an
error message is dug out) and the clients own only the *transport* (open a connection,
send bytes, read bytes back). If a reviewer can diff ``client.py`` against ``aio.py`` and
find duplicated logic rather than duplicated transport, whatever they found belongs in
here instead. That is why the success/failure rule (:func:`is_success`,
:func:`decode_response`) and the transport-failure message (:func:`transport_error`) live
here too, even though they construct a :class:`~nidus.errors.NidusError`: "which statuses
are errors" and "what a client says when it never got an answer" are wire contract, and
the status-``0`` sentinel is a cross-SDK promise that must not be reworded in one client
and not the other.

Everything below is a pure function of its arguments — no sockets, no clock, no globals
— which is also why nearly the whole wire contract is unit-testable with no server at
all. ``urllib.parse.quote`` is the single ``urllib`` import and it is pure string work
(percent-escaping), not the ``urllib.request`` transport.

The rule that governs the body builders: **an unset optional is omitted, never guessed.**
The server supplies ``top_k = 10``, ``limit = 100``, ``rrf_k = 60.0``,
``candidates = 100`` and ``offset = 0`` via ``#[serde(default)]``; restating any of those
numbers in Python would fork the contract the day the server changes one. ``prune`` drops
the ``None``-valued keys so the server's default applies — and it is why every optional
defaults to ``None`` rather than to a number: ``top_k=0`` is a legitimate request for zero
results, so ``0`` cannot double as "unset".

Two things the builders check rather than pass through, both from :mod:`nidus._guards`
and both cases where an unchecked value produces a *well-formed* request with the wrong
contents (see that module for why Python needs this and TypeScript does not): a bare
``str`` handed to a ``Sequence[str]`` parameter, and a vector element that ``json`` cannot
serialize (``numpy.float32`` is not a ``float`` subclass, and numpy is where a Python
caller's embeddings come from). Vectors are coerced with ``float()`` on the way out for the
same reason :func:`decode_records` coerces on the way in — do not "simplify" either back
to a bare ``list()``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, Union
from urllib.parse import quote

from . import _guards
from .errors import NidusError
from .filter import Filter
from .types import AnnInfo, Footprint, FtsField, Hit, Record, RecordInput, Stats
from .values import AttrInput, Value, decode_attrs, encode_attrs

# ── Paths ────────────────────────────────────────────────────────────────────────────
#
# Fixed paths are constants; the per-collection ones are builders because the name has to
# be escaped. `safe=""` is the load-bearing argument: `quote` leaves `/` alone by default,
# so a collection literally named "a/b" would otherwise address a route that does not
# exist. Escaping it as %2F keeps a name with slashes or spaces a single path segment.

HEALTH = "/health"
STATS = "/stats"
COLLECTIONS = "/collections"
SEARCH = "/search"
TEXT_SEARCH = "/text-search"
HYBRID_SEARCH = "/hybrid-search"
LIST = "/list"
FLUSH = "/flush"
COMPACT = "/compact"


def collection_path(name: str) -> str:
    """``/collections/{name}`` — create (POST) or drop (DELETE)."""
    return f"{COLLECTIONS}/{quote(name, safe='')}"


def meta_path(name: str) -> str:
    """``/collections/{name}/meta`` — read (GET) or replace (PUT)."""
    return f"{collection_path(name)}/meta"


def upsert_path(name: str) -> str:
    return f"{collection_path(name)}/upsert"


def delete_path(name: str) -> str:
    """One route for both delete forms; the body decides which (``filter`` wins)."""
    return f"{collection_path(name)}/delete"


def records_path(name: str) -> str:
    return f"{collection_path(name)}/records"


def fts_schema_path(name: str) -> str:
    return f"{collection_path(name)}/fts-schema"


def remember_path(name: str) -> str:
    return f"{collection_path(name)}/remember"


def recall_path(name: str) -> str:
    return f"{collection_path(name)}/recall"


# ── Request bodies ───────────────────────────────────────────────────────────────────


def prune(body: Mapping[str, Any]) -> dict[str, Any]:
    """Drop ``None``-valued keys so the server's ``#[serde(default)]`` applies instead."""
    return {k: val for k, val in body.items() if val is not None}


def empty_body() -> dict[str, Any]:
    """``{}`` — what the bodyless writes (create-collection, flush, compact) send.

    They need *a* JSON body because the handlers extract one; a function rather than a
    module constant so no caller can mutate a shared dict.
    """
    return {}


def upsert_body(records: Iterable[RecordInput]) -> dict[str, Any]:
    """Body for ``POST /collections/{name}/upsert``, with attrs normalized."""
    wire: list[dict[str, Any]] = []
    for rec in records:
        # `attrs` carries no `#[serde(default)]` on the server's `Record`, so it is always
        # emitted — an omitted map is a deserialization failure, not an empty map.
        out: dict[str, Any] = {"id": rec["id"], "attrs": encode_attrs(rec.get("attrs") or {})}
        vector = rec.get("vector")
        # Absent (or None) means a text-only doc, so the key is left out entirely; `[]` is
        # passed through untouched and will fail the server's dimension check, which is
        # the honest outcome for an empty vector.
        if vector is not None:
            out["vector"] = _guards.float_sequence(vector, f"record {rec['id']!r} vector")
        wire.append(out)
    return {"records": wire}


def meta_body(meta: Mapping[str, str]) -> dict[str, str]:
    """Body for ``PUT /collections/{name}/meta``.

    The map *is* the body: the handler deserializes a bare ``BTreeMap<String, String>``,
    not an object wrapping one. Rebuilt as a plain ``dict`` of ``str`` because ``json``
    serializes only real dicts, and because a caller's ``Mapping`` may hold anything.
    """
    return {str(k): str(val) for k, val in meta.items()}


def delete_ids_body(ids: Sequence[str]) -> dict[str, Any]:
    """Body for delete-by-id."""
    return {"ids": _guards.str_sequence(ids, "delete(name, ids)")}


def delete_where_body(filter: Filter) -> dict[str, Any]:  # noqa: A002
    """Body for delete-by-filter. Sent without ``ids``, since ``filter`` takes precedence.

    An **empty** filter is refused. The server would accept it — ``DeleteRequest.filter``
    is an ``Option<Filter>``, so ``[]`` arrives as ``Some(Filter([]))``, and an empty
    filter matches everything — and delete every record in the collection with a 200. The
    shape that produces it is ordinary Python: a filter list built from optional conditions
    that all turned out to be absent. This is the last layer that can tell "delete
    everything" from "my conditions collapsed", and nobody writes it deliberately when
    :meth:`drop_collection` exists.
    """
    if not filter:
        raise ValueError(
            "delete_where with an empty filter would delete every record in the "
            "collection; pass at least one predicate, or call drop_collection(name) if "
            "deleting everything is the intent"
        )
    return {"filter": list(filter)}


#: The knobs an :class:`~nidus.FtsField` may carry, in the order the server documents them.
_FTS_FIELD_KNOBS = ("k1", "b", "language", "ascii_folding", "max_token_len")


def fts_schema_body(fields: Sequence[Union[str, FtsField]]) -> dict[str, Any]:
    """Body for the FTS schema. A bare name and a knob-less mapping mean the same thing.

    An unknown key is a ``TypeError`` rather than a silently dropped one: the server
    ignores what it does not recognise, so a misspelled ``asciiFolding`` would otherwise
    index the field with folding off and report success.
    """
    _guards.reject_bare_string(fields, "set_fts_schema(name, fields)")
    return {"fields": [_fts_field(f) for f in fields]}


def _fts_field(spec: Union[str, FtsField]) -> Union[str, dict[str, Any]]:
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, Mapping) or "field" not in spec:
        raise TypeError(
            "set_fts_schema(name, fields) expects each field to be a name or a mapping "
            f"with a 'field' key, got {spec!r}"
        )
    # As a plain Mapping: a TypedDict cannot be indexed by a variable key.
    knobs: Mapping[str, Any] = spec
    unknown = set(knobs) - {"field", *_FTS_FIELD_KNOBS}
    if unknown:
        raise TypeError(
            f"set_fts_schema(name, fields): unknown key(s) {sorted(unknown)} — "
            f"expected any of {sorted(_FTS_FIELD_KNOBS)}"
        )
    body: dict[str, Any] = {"field": str(knobs["field"])}
    body.update({k: knobs[k] for k in _FTS_FIELD_KNOBS if k in knobs})
    return body


def search_body(
    query: Sequence[float],
    scope: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
    offset: Optional[int] = None,
    min_score: Optional[float] = None,
    filter: Optional[Filter] = None,  # noqa: A002
) -> dict[str, Any]:
    """Body for ``POST /search`` (vector nearest-neighbour).

    ``scope`` and ``filter`` are always sent, as ``[]`` when unset — an empty scope means
    "every collection" and an empty filter means "match everything", so the empty array is
    the real value, not a missing one. ``top_k``/``offset``/``min_score`` are pruned when
    unset, so an omitted ``offset`` is byte-identical to the pre-pagination request.
    """
    return prune(
        {
            "query": _guards.float_sequence(query, "search(query=...)"),
            "scope": _scope(scope),
            "top_k": top_k,
            "offset": offset,
            "min_score": min_score,
            "filter": list(filter) if filter is not None else [],
        }
    )


def text_search_body(
    field: str,
    query: str,
    scope: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
    offset: Optional[int] = None,
    min_score: Optional[float] = None,
    filter: Optional[Filter] = None,  # noqa: A002
) -> dict[str, Any]:
    """Body for ``POST /text-search`` (BM25). ``min_score`` here is a raw BM25 floor."""
    return prune(
        {
            "field": field,
            "query": query,
            "scope": _scope(scope),
            "top_k": top_k,
            "offset": offset,
            "min_score": min_score,
            "filter": list(filter) if filter is not None else [],
        }
    )


def hybrid_search_body(
    vector: Sequence[float],
    field: str,
    text: str,
    scope: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
    offset: Optional[int] = None,
    filter: Optional[Filter] = None,  # noqa: A002
    rrf_k: Optional[float] = None,
    candidates: Optional[int] = None,
) -> dict[str, Any]:
    """Body for ``POST /hybrid-search`` (vector + BM25 fused via RRF).

    Note there is no ``min_score``: the score is a fused RRF rank, not a similarity, so
    the server offers no floor for it. ``offset`` pages the *fused* ranking.
    """
    return prune(
        {
            "vector": _guards.float_sequence(vector, "hybrid_search(vector=...)"),
            "field": field,
            "text": text,
            "scope": _scope(scope),
            "top_k": top_k,
            "offset": offset,
            "filter": list(filter) if filter is not None else [],
            "rrf_k": rrf_k,
            "candidates": candidates,
        }
    )


def list_body(
    scope: Optional[Sequence[str]] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    filter: Optional[Filter] = None,  # noqa: A002
) -> dict[str, Any]:
    """Body for ``POST /list`` (metadata-only, paginated)."""
    return prune(
        {
            "scope": _scope(scope),
            "offset": offset,
            "limit": limit,
            "filter": list(filter) if filter is not None else [],
        }
    )


def remember_body(
    id: str,  # noqa: A002
    text: str,
    mode: Optional[str] = None,
    attrs: Optional[Mapping[str, AttrInput]] = None,
) -> dict[str, Any]:
    """Body for ``POST /collections/{name}/remember`` (text in, server embeds).

    ``attrs`` *is* pruned when unset here — unlike a record's, the server declares this
    one ``#[serde(default)]``, so omitting it is well-defined.
    """
    return prune(
        {
            "id": id,
            "text": text,
            "mode": mode,
            "attrs": encode_attrs(attrs) if attrs is not None else None,
        }
    )


def recall_body(
    query: str,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    filter: Optional[Filter] = None,  # noqa: A002
) -> dict[str, Any]:
    """Body for ``POST /collections/{name}/recall`` (query text in, hits out)."""
    return prune(
        {
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
            "filter": list(filter) if filter is not None else [],
        }
    )


def encode_body(body: Any) -> bytes:
    """Serialize a request body to the bytes both clients put on the wire."""
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


# ── Response decoding ────────────────────────────────────────────────────────────────


def is_success(status: int) -> bool:
    """Whether a status counts as success. The whole rule, in one place.

    Written once because it is a *decision*, not transport: if the SDK ever has to treat a
    3xx or a bodyless 204 specially, there must be exactly one place that changes, or the
    sync and async clients would quietly disagree about what an error is.
    """
    return 200 <= status < 300


def decode_response(status: int, text: str) -> Any:
    """Turn a raw ``(status, body)`` into a parsed payload, or raise ``NidusError``.

    The failure branch keeps the server's own message (``{"error": …}``) and its status, so
    a caller can branch on ``err.status`` exactly as the other SDKs let them.
    """
    if not is_success(status):
        raise NidusError(extract_error(text, status), status)
    return parse_body(text)


def parse_body(text: str) -> Any:
    """Parse a successful response body; an empty body decodes to ``None``.

    Several write endpoints answer with a small JSON object nobody reads, and ``/health``
    may answer with nothing at all, so "no body" is normal rather than an error.
    """
    return json.loads(text) if text else None


def decode_hits(payload: Any) -> list[Hit]:
    """Decode a search-family response, turning each hit's wire attrs into plain values."""
    return [
        Hit(
            collection=str(h["collection"]),
            id=str(h["id"]),
            score=float(h["score"]),
            attrs=decode_attrs(_attrs_of(h)),
        )
        for h in payload or ()
    ]


def decode_records(payload: Any) -> list[Record]:
    """Decode ``GET /collections/{name}/records``.

    An absent ``vector`` stays ``None`` (text-only doc) and is never coerced to ``[]``.
    """
    out: list[Record] = []
    for r in payload or ():
        vector = r.get("vector")
        out.append(
            Record(
                id=str(r["id"]),
                vector=None if vector is None else [float(x) for x in vector],
                attrs=decode_attrs(_attrs_of(r)),
            )
        )
    return out


def decode_stats(payload: Any) -> Stats:
    """Decode ``GET /stats``. ``ann`` is ``null`` when the store does exact search.

    Unlike the list-shaped decoders, this one cannot fall back to an empty value: every
    field of :class:`~nidus.types.Stats` is required, so there is nothing honest to build
    from a missing body. ``parse_body`` maps an empty body to ``None`` (a proxy or a
    stripped 204 can produce one), and letting that through would surface an
    ``AttributeError`` from a private module instead of the one exception type the SDK
    promises to raise — so it is named as the malformed response it is.

    Status ``0``, because a 2xx with nothing usable in it is the same fact for the caller as
    never having got an answer: the store's state is unknown and the request has to be made
    again. That is exactly what ``is_transport_error`` is for.
    """
    if not isinstance(payload, Mapping):
        raise NidusError(f"/stats returned no JSON object (got {payload!r})", 0)
    ann = payload.get("ann")
    return Stats(
        dimension=int(payload["dimension"]),
        distance=str(payload["distance"]),
        ann=None if ann is None else decode_ann(ann),
        collections=[str(c) for c in payload.get("collections") or ()],
        footprint=decode_footprint(payload["footprint"]),
    )


def decode_ann(payload: Any) -> AnnInfo:
    """Decode ``AnnDto``; the knobs that do not apply to the active kind are absent."""
    return AnnInfo(
        kind=str(payload["kind"]),
        overscan=int(payload["overscan"]),
        seed=int(payload["seed"]),
        m=_opt_int(payload.get("m")),
        ef_construction=_opt_int(payload.get("ef_construction")),
        ef_search=_opt_int(payload.get("ef_search")),
        n_lists=_opt_int(payload.get("n_lists")),
        n_probe=_opt_int(payload.get("n_probe")),
    )


def decode_footprint(payload: Any) -> Footprint:
    return Footprint(
        rows=int(payload["rows"]),
        dead_rows=int(payload["dead_rows"]),
        dimension=int(payload["dimension"]),
        vector_bytes=int(payload["vector_bytes"]),
        doc_count=int(payload["doc_count"]),
    )


def decode_collections(payload: Any) -> list[str]:
    return [str(name) for name in payload or ()]


def decode_meta(payload: Any) -> dict[str, str]:
    return {str(k): str(val) for k, val in (payload or {}).items()}


def decode_upserted(payload: Any) -> int:
    """The count from an upsert response. Named so the field name appears once, here."""
    return _count(payload, "upserted")


def decode_deleted(payload: Any) -> int:
    """The count from either delete form's response."""
    return _count(payload, "deleted")


def extract_error(text: str, status: int) -> str:
    """Dig the message out of a failed response's ``{"error": …}`` body, or fall back.

    Best-effort by design: a body that is not JSON (a proxy's HTML error page, say) is
    more useful verbatim than replaced, and an empty body leaves only the status.
    """
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        message = parsed.get("error")
        if isinstance(message, str):
            return message
    return text or f"HTTP {status}"


# ── Connection-independent request setup ─────────────────────────────────────────────
#
# Not IO either: both clients need the same URL normalization, the same header set, the
# same body encoding, and the same words when the request never lands, so all of it lives
# here rather than being written twice.


def prepare(
    token: Optional[str],
    extra: Optional[Mapping[str, str]],
    body: Any,
) -> tuple[Optional[bytes], dict[str, str]]:
    """Everything a request needs before it touches a socket: ``(payload, headers)``.

    ``body=None`` means *no body*, not a JSON ``null`` — no endpoint accepts a bare
    ``null``, and the bodyless writes send ``{}`` explicitly via :func:`empty_body`. Whether
    there is a payload is also what decides the ``content-type`` header, which is why the
    two are computed together instead of by each client in turn.
    """
    payload = None if body is None else encode_body(body)
    return payload, request_headers(token, extra, has_body=payload is not None)


def transport_error(path: str, err: BaseException) -> NidusError:
    """The error for "the request never produced a response".

    Status ``0`` is the cross-SDK sentinel for that, and the wording is part of the
    contract: a caller reading a log should not be able to tell whether the sync or the
    async client wrote the line. Returned rather than raised so the caller can
    ``raise ... from err`` and keep the original cause.
    """
    return NidusError(f"request to {path} failed: {err}", 0)


def normalize_base_url(base_url: str) -> str:
    """Strip trailing slashes so path concatenation cannot produce ``//stats``."""
    if not base_url:
        raise ValueError("a nidus client requires a base_url, e.g. http://127.0.0.1:7700")
    return base_url.rstrip("/")


def request_headers(
    token: Optional[str],
    extra: Optional[Mapping[str, str]],
    *,
    has_body: bool,
) -> dict[str, str]:
    """Headers for one request: caller extras first, then the ones we own.

    Auth and content-type are applied last so a caller's ``headers`` cannot accidentally
    unset the token, and ``content-type`` is sent only when there is a body to describe.
    """
    headers: dict[str, str] = dict(extra or {})
    if token:
        headers["authorization"] = f"Bearer {token}"
    if has_body:
        headers["content-type"] = "application/json"
    return headers


def _scope(scope: Optional[Sequence[str]]) -> list[str]:
    """A ``scope`` argument as it goes on the wire: ``[]`` for unset, meaning "everywhere"."""
    return [] if scope is None else _guards.str_sequence(scope, "scope")


def _count(payload: Any, key: str) -> int:
    """Pull a write endpoint's count out of its response body."""
    return int((payload or {}).get(key, 0))


def _attrs_of(payload: Mapping[str, Any]) -> Mapping[str, Value]:
    """A row's wire attrs, tolerating an absent map (a record with no metadata)."""
    attrs = payload.get("attrs")
    return attrs if isinstance(attrs, dict) else {}


def _opt_int(value: Any) -> Optional[int]:
    """An optional integer knob: absent (``None``) stays absent, meaning "not applicable"."""
    return None if value is None else int(value)
