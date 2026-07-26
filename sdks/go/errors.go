// Error — the one error type this SDK returns, carrying the HTTP status.
//
// A failed request answers with a status chosen deliberately, and the status is the
// part a caller can act on — 409 and 503 are worth retrying, 400 and 422 never are —
// so it travels with the message rather than being flattened into prose. What
// produces each, verified against a running server:
//
//	400  a JSON *syntax* error in the body, plus the store's own client faults from
//	     src/server/mod.rs#classify (a vector whose length does not match the store
//	     dimension, a memory route on a server with no embedder). These carry the
//	     server's {"error": …} envelope; axum's syntax rejection is plain text.
//	401  missing or wrong bearer token (/health is exempt).
//	403  read-only store.
//	404  a route the binary does not have — /remember and /recall on a build without
//	     the `memory` feature. Bodyless, so the message is just "HTTP 404".
//	409  writer lock held elsewhere, or a collection pinned to a different embedder.
//	422  a body whose *types* are wrong: "top_k": -1, a Glob pattern sent as a tagged
//	     Value, {"Int": "5"}. This is axum's Json extractor refusing to deserialize,
//	     and it is plain text rather than the error envelope. Extra fields are NOT in
//	     this class — the DTOs carry no deny_unknown_fields, so an unknown field is
//	     accepted and ignored.
//	503  shed under backpressure, or the store is not open yet.
//	507  out of capacity: max_vector_bytes exceeded, or an allocation failed.
//	500  anything else.
//
// The classifiers below cover the statuses a caller branches on. Four of them
// (IsBadRequest, IsReadOnly, IsLocked, IsOutOfCapacity) mirror the JS SDK's getters
// one for one; IsTransport is Go-specific because Status 0 has no JS counterpart; and
// IsUnauthorized/IsUnavailable exist because 401 and 503 are the two statuses this
// SDK's own tests were otherwise comparing by hand. (The JS SDK should grow the last
// two as well — that is a change to both SDKs, filed rather than smuggled in here.)
// There is no helper for 500: a caller has no specific recovery for it.
//
// Status 0 means the request never got an answer at all: connection refused, DNS
// failure, timeout, cancelled context. Keeping that in the same type as a server
// error (the JS SDK does the same) means a caller has one thing to check, and can
// still tell "nidus said no" from "nidus was not reachable" by looking at Status.

package nidus

import "strconv"

// An Error is a failure reported by a nidus server, or a transport failure reaching
// one.
//
// Client methods return it as a *Error, so errors.As recovers the status:
//
//	var nerr *nidus.Error
//	if errors.As(err, &nerr) && nerr.IsLocked() {
//		// another process holds the writer lock — worth retrying
//	}
type Error struct {
	// Message is the server's error text, or a description of the transport failure.
	Message string
	// Status is the HTTP status code, or 0 when there was no response.
	Status int
}

func (e *Error) Error() string {
	if e.Status == 0 {
		return "nidus: " + e.Message
	}
	return "nidus: " + e.Message + " (HTTP " + strconv.Itoa(e.Status) + ")"
}

// IsTransport reports a failure with no HTTP response behind it — unreachable
// server, timeout, cancelled context. Unlike the status-carrying cases, this one
// says nothing about whether the request was applied: a timeout can fire after the
// server committed the write.
func (e *Error) IsTransport() bool { return e.Status == 0 }

// IsBadRequest reports a request the server refused to act on because the request
// itself is wrong: HTTP 400 (a JSON syntax error, or a store-level client fault such
// as a dimension mismatch or a memory route with no embedder) and HTTP 422 (a body
// whose types do not deserialize — a negative top_k, a wrong-shaped Value).
//
// Both are in one predicate because they are one thing to a caller: retrying will
// never help. A retry loop written as `if !nerr.IsBadRequest() { retry() }` would
// otherwise spin forever on a 422, which is the status axum returns for most
// malformed bodies. Note that an *unknown* field is in neither class — the server
// ignores extra fields.
func (e *Error) IsBadRequest() bool { return e.Status == 400 || e.Status == 422 }

// IsUnauthorized reports a missing or wrong bearer token (HTTP 401) — the value
// `nidus serve --token` was given, passed with [WithToken]. /health is exempt from
// auth, so [Client.Health] never fails for this reason alone.
func (e *Error) IsUnauthorized() bool { return e.Status == 401 }

// IsReadOnly reports a write against a read-only store (HTTP 403).
func (e *Error) IsReadOnly() bool { return e.Status == 403 }

// IsLocked reports a conflict with state the store has already committed (HTTP 409):
// another process holds the writer lock, or the target collection is pinned to a
// different embedding model. The lock case is worth retrying; the embedder case is
// not, and the message says which.
func (e *Error) IsLocked() bool { return e.Status == 409 }

// IsUnavailable reports that the server took the request but would not serve it yet
// (HTTP 503): it shed the request under backpressure, or the store is not open. Both
// are transient by construction, so this is the other status — with 409 — that a
// retry with backoff is the right answer to.
func (e *Error) IsUnavailable() bool { return e.Status == 503 }

// IsOutOfCapacity reports that the store refused to grow (HTTP 507): the configured
// max_vector_bytes was exceeded, or an allocation failed. The store is intact — the
// refusal happens before anything is written.
func (e *Error) IsOutOfCapacity() bool { return e.Status == 507 }
