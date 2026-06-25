//! Error type carrying the HTTP status the server reported.
//
// The server replies to a failed request with `{ "error": <message> }` and a
// meaningful status (`src/server/mod.rs#classify`): 400 dimension mismatch,
// 403 read-only store, 409 writer-lock conflict, 507 capacity/OOM, 500 otherwise.
// Callers branch on `.status` to tell a client fault from a server fault.

/** An error returned by a `nidus` server, or a transport failure reaching it. */
export class NidusError extends Error {
  /** The HTTP status code, or `0` for a transport/timeout failure (no response). */
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "NidusError";
    this.status = status;
  }

  /** A malformed request the server rejected (HTTP 400). */
  get isBadRequest(): boolean {
    return this.status === 400;
  }
  /** The store is read-only (HTTP 403). */
  get isReadOnly(): boolean {
    return this.status === 403;
  }
  /** The writer lock is held by another process (HTTP 409). */
  get isLocked(): boolean {
    return this.status === 409;
  }
  /** Out of capacity: `max_vector_bytes` exceeded or OOM (HTTP 507). */
  get isOutOfCapacity(): boolean {
    return this.status === 507;
  }
}
