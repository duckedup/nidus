// The single request path: URL building, headers, auth, timeout, and error mapping.
//
// Every method on [Client] goes through request() — no method builds an
// http.Request by hand. That is not tidiness for its own sake: the details that are
// easy to get subtly wrong here are the ones that are invisible when they *are*
// wrong. A Content-Type on a bodyless POST, a collection name with a slash in it that
// silently addresses a different route, a response body left undrained so the
// connection never returns to the keep-alive pool, an error body that is HTML from a
// proxy rather than the server's JSON — each is one line, in one place, tested once.
//
// Two decisions worth keeping straight:
//
//   - The per-request timeout is a *context* deadline, not http.Client.Timeout. A
//     deadline composes with the caller's own (the earlier one wins) instead of
//     fighting it, and it leaves a caller-supplied http.Client free of SDK policy.
//   - A failure with no HTTP response behind it becomes an [Error] with Status 0,
//     exactly as the JavaScript SDK does. Callers then have one error type to check
//     and can still tell "nidus said no" from "nidus was not reachable".

package nidus

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const (
	// errorBodyLimit is how much of a failed response is read to build its message.
	// 64 KiB is far more than the server's one-line {"error": …} envelope and enough of
	// a proxy's HTML page to recognise it.
	errorBodyLimit = 64 << 10
	// errorDrainLimit is how much of the remainder is discarded to keep the connection
	// reusable. Bounded for the same reason as the read above.
	errorDrainLimit = 1 << 20
)

// request issues one call and decodes its JSON response.
//
// A non-nil body is marshalled as JSON; a nil body sends none (and no Content-Type).
// A non-nil out receives the decoded response; a nil out discards it, which is how the
// endpoints that answer a bookkeeping {"ok": true} are called.
func (c *Client) request(ctx context.Context, method, path string, body, out any) error {
	var payload io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			// Deliberately not an *Error: nothing was sent, so there is no status to
			// report and Status 0 ("never got an answer") would be a lie. This is also
			// where a Filter built from a value the store has no attribute type for
			// surfaces — see the note on the builders in filter.go — and that is a
			// caller mistake, not a server or network failure.
			return fmt.Errorf("nidus: encoding the request body for %s: %w", path, err)
		}
		payload = bytes.NewReader(encoded)
	}

	// Attribute a timeout honestly. context.WithTimeout already picks the earlier of
	// the two deadlines, but the error message should only blame *our* timeout when
	// ours is the one that can fire first; otherwise the caller's deadline expired and
	// saying "timed out after 5s" would send them looking in the wrong place.
	applied := time.Duration(0)
	if c.timeout > 0 {
		if deadline, ok := ctx.Deadline(); !ok || time.Until(deadline) > c.timeout {
			applied = c.timeout
		}
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, payload)
	if err != nil {
		return fmt.Errorf("nidus: building the request for %s: %w", path, err)
	}

	// Caller headers first, then the SDK's own, so the SDK's contract with the server
	// cannot be broken by a WithHeader typo. Content-Type only when there is a body:
	// a bodyless POST that claims application/json is a lie, and some proxies act on
	// it. Authorization is skipped entirely when no token is configured rather than
	// sent empty — `nidus serve` without --token accepts anything, but an empty bearer
	// header is the sort of thing an intermediary rejects.
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.hc.Do(req)
	if err != nil {
		return transportError(path, applied, err)
	}
	defer resp.Body.Close()

	// The status decides how much of the body is worth reading, and the asymmetry is
	// deliberate. A success body has to be read whole — Records returns a collection —
	// but an error body only ever becomes a short message, so reading it unbounded lets
	// whatever is on the other end of the socket (a gateway streaming a multi-megabyte
	// error document, the reverse-proxy HTML page extractError anticipates) decide how
	// much memory this client allocates. The request side is bounded by the server's
	// DefaultBodyLimit; this is the response side's equivalent.
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, errorBodyLimit))
		// Drain a bounded remainder so the connection returns to the keep-alive pool in
		// the ordinary case; a pathologically large error body loses the connection
		// instead of the process's memory, which is the cheaper of the two mistakes.
		// A read failure here needs no special handling: we hold the status, and
		// extractError falls back to "HTTP <status>" when it has no bytes to work with.
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, errorDrainLimit))
		return &Error{Message: extractError(raw, resp.StatusCode), Status: resp.StatusCode}
	}

	// Read to completion even when out is nil. An undrained body is a connection
	// net/http cannot reuse, and a client that leaks one per flush() ends up opening a
	// fresh socket for every request under load.
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		// The status line arrived but the body did not, so we hold no complete answer:
		// Status 0 is the truthful classification, and it carries the right warning
		// with it — a write may well have been applied.
		return &Error{
			Message: fmt.Sprintf("reading the response to %s failed: %v", path, err),
			Status:  0,
		}
	}

	if out == nil || len(bytes.TrimSpace(raw)) == 0 {
		return nil
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("nidus: decoding the %s response: %w", path, err)
	}
	return nil
}

// transportError maps a failure that never produced a response onto an [Error] with
// Status 0 — connection refused, DNS failure, TLS failure, timeout, cancellation.
//
// applied is the SDK-configured timeout when it was the earlier deadline, and zero
// otherwise; naming the duration in the message is the difference between a caller
// tuning WithTimeout and a caller hunting a phantom network problem.
func transportError(path string, applied time.Duration, err error) *Error {
	switch {
	case errors.Is(err, context.Canceled):
		return &Error{Message: fmt.Sprintf("request to %s was cancelled", path), Status: 0}
	case errors.Is(err, context.DeadlineExceeded) && applied > 0:
		return &Error{
			Message: fmt.Sprintf("request to %s timed out after %s", path, applied),
			Status:  0,
		}
	case errors.Is(err, context.DeadlineExceeded):
		return &Error{Message: fmt.Sprintf("request to %s timed out", path), Status: 0}
	default:
		return &Error{Message: fmt.Sprintf("request to %s failed: %v", path, err), Status: 0}
	}
}

// extractError pulls the message out of a failed response, best-effort.
//
// The server answers {"error": "<message>"} (src/server/mod.rs, ApiError), but the
// thing on the other end of the socket is not always the server: a reverse proxy
// returns an HTML page, axum's own body-limit rejection is plain text, and a dropped
// upstream returns nothing at all. So: the JSON message when there is one, the raw
// body when it is not JSON, and "HTTP <status>" when the body is empty. An error path
// that can itself fail would hide the error the caller actually needs.
func extractError(raw []byte, status int) string {
	var envelope struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err == nil && envelope.Error != "" {
		return envelope.Error
	}
	if msg := strings.TrimSpace(string(raw)); msg != "" {
		return msg
	}
	return "HTTP " + strconv.Itoa(status)
}

// collPath builds /collections/{name}{suffix} — the one place a collection name
// enters a URL, so it is escaped exactly once.
//
// Names are opaque strings that may hold slashes ("notes/2024") or spaces, and the
// route is /collections/{name}: an unescaped slash would quietly address a different
// route (or a 404) instead of the collection the caller named. url.PathEscape is the
// net/url counterpart of the JS SDK's encodeURIComponent for a single segment — it
// escapes / ; , ? # and space. It leaves : @ & = + $ literal, which is legal inside a
// path segment (RFC 3986) and percent-decodes back to the same bytes on the server,
// so the two SDKs address the same collection even where their encodings differ.
func collPath(name, suffix string) string {
	return "/collections/" + url.PathEscape(name) + suffix
}

// aliasPath builds /aliases/{name}, escaped the same way collPath escapes a
// collection name — an alias name is just as opaque a string.
func aliasPath(name string) string {
	return "/aliases/" + url.PathEscape(name)
}
