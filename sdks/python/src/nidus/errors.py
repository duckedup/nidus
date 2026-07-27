"""The one error type the SDK raises, carrying the HTTP status the server reported.

A failed request comes back as ``{"error": "<message>"}`` with a status the server
chose deliberately (``src/server/mod.rs#classify``): 400 dimension mismatch, 403
read-only store, 409 writer-lock conflict, 507 capacity/OOM, 500 otherwise. Callers
branch on ``.status`` to tell a client fault from a server fault, which is why there is
one exception type with a status rather than a tree of subclasses — the server's
taxonomy already lives in the status code, and duplicating it as a class hierarchy would
be a second thing to keep in sync.

Status ``0`` is reserved for "no response at all" (connection refused, DNS failure,
timeout). The JS SDK uses the same sentinel; keeping it identical means the three SDKs
answer "was this even reachable?" the same way.
"""

from __future__ import annotations


class NidusError(Exception):
    """An error returned by a nidus server, or a transport failure reaching it."""

    #: The server's message, also the exception's ``str()``.
    message: str
    #: The HTTP status code, or ``0`` for a transport/timeout failure (no response).
    status: int

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.message = message
        self.status = status

    def __repr__(self) -> str:
        return f"NidusError(status={self.status}, message={self.message!r})"

    @property
    def is_transport_error(self) -> bool:
        """No response was received at all — refused, unreachable, or timed out."""
        return self.status == 0

    @property
    def is_bad_request(self) -> bool:
        """A malformed request the server rejected (HTTP 400)."""
        return self.status == 400

    @property
    def is_read_only(self) -> bool:
        """The store is read-only (HTTP 403)."""
        return self.status == 403

    @property
    def is_locked(self) -> bool:
        """The writer lock is held by another process (HTTP 409)."""
        return self.status == 409

    @property
    def is_out_of_capacity(self) -> bool:
        """Out of capacity: ``max_vector_bytes`` exceeded, or OOM (HTTP 507)."""
        return self.status == 507
