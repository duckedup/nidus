//! Bearer-token authentication for `nidus serve` (nidus-abx.5).
//!
//! Two things live here, together on purpose: the token *type*, which knows how to compare
//! itself, and the middleware, which is its only caller. Keeping the comparison inside a
//! newtype rather than exposing the secret as a `String` is what stops a future second call
//! site from reintroducing `presented == expected` — the plain comparison is no longer
//! reachable, because `Token` has no accessor that hands the bytes back.

use axum::{
    Json,
    extract::{Request, State},
    http::{StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;

use super::AppState;

/// The configured bearer token, comparable only in constant time.
///
/// `str` equality short-circuits on the first differing byte, so its running time varies
/// with how many leading bytes of a guess were right — a textbook timing oracle on a
/// secret. The practical severity is low (lifting that signal out of network jitter, over
/// a handful of bytes, takes an enormous number of samples, and an attacker who can do it
/// has easier options), but it is unambiguously the wrong primitive for a secret and the
/// fix is ten lines, so the trade is trivially favourable.
///
/// **Dependency-free by choice.** `subtle` is the usual answer and is tiny, but so is this;
/// the same reasoning keeps `crc.rs` hand-rolled.
#[derive(Clone)]
pub(super) struct Token(std::sync::Arc<str>);

impl Token {
    pub(super) fn new(secret: impl Into<std::sync::Arc<str>>) -> Token {
        Token(secret.into())
    }

    /// Whether `presented` is the configured token, in time independent of how many
    /// leading bytes match.
    ///
    /// The length check is deliberate and comes first. Comparing different-length inputs
    /// leaks the length whatever you do — a fixed-time loop over the longer of the two
    /// still runs longer — so there is nothing to be gained by hiding it, and folding
    /// unequal lengths into the loop only invites an indexing bug. Leaking the token's
    /// *length* is acceptable; leaking a *prefix match* is not, and that is what the fold
    /// below prevents.
    pub(super) fn verify(&self, presented: &str) -> bool {
        let expected = self.0.as_bytes();
        let presented = presented.as_bytes();
        if expected.len() != presented.len() {
            return false;
        }
        // Fold every byte pair into one accumulator and test it once at the end, so the
        // work done — and every branch taken — is the same whether the first byte differs
        // or none of them do. `black_box` keeps a future optimiser from noticing that the
        // accumulator can be abandoned as soon as it is nonzero.
        let mut diff: u8 = 0;
        for (a, b) in expected.iter().zip(presented) {
            diff |= a ^ b;
        }
        std::hint::black_box(diff) == 0
    }
}

/// Paths that answer without a credential.
///
/// `/health` and `/ready` because an orchestrator would read a `401` as "not ready" and
/// never route to a perfectly healthy instance. `/metrics` for the same reason — a scraper
/// that gets a `401` reports the target as down — and because the endpoint is deliberately
/// label-free of collection names, so what it exposes is traffic shape, not data. That is
/// a real disclosure and it is documented as one: put `/metrics` on a scrape-only path,
/// alongside the TLS stance in the deployment guide.
pub(super) fn is_public(path: &str) -> bool {
    matches!(path, "/health" | "/ready" | "/metrics")
}

/// Reject any request lacking a valid `Authorization: Bearer <token>` when a token is
/// configured. A no-op when the server is unauthenticated.
pub(super) async fn auth(State(st): State<AppState>, req: Request, next: Next) -> Response {
    if let Some(expected) = &st.token
        && !is_public(req.uri().path())
    {
        let presented = req
            .headers()
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        let ok = presented.is_some_and(|p| expected.verify(p));
        if !ok {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "missing or invalid bearer token" })),
            )
                .into_response();
        }
    }
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Correctness, not timing. A timing assertion would flake on any shared runner and
    /// prove nothing; what must hold is that the constant-time path still accepts and
    /// rejects exactly what the plain comparison did.
    #[test]
    fn verify_accepts_only_the_exact_token() {
        let t = Token::new("s3cret-token");
        assert!(t.verify("s3cret-token"));
        assert!(!t.verify("wrong-token!!"), "same length, different bytes");
        assert!(!t.verify(""), "empty");
        assert!(!t.verify("s3cret-token-and-more"), "superstring");
    }

    /// A strict prefix of the real token must be rejected. This is the case a broken
    /// constant-time comparison most often gets wrong — folding over the shorter input, or
    /// over `min(len)`, would accept it.
    #[test]
    fn verify_rejects_a_prefix_of_the_token() {
        let t = Token::new("s3cret-token");
        for n in 0.."s3cret-token".len() {
            assert!(
                !t.verify(&"s3cret-token"[..n]),
                "prefix of length {n} must not authenticate"
            );
        }
    }

    /// Multi-byte characters must not panic or mis-compare: the fold is over bytes, and a
    /// token is arbitrary caller-supplied text.
    #[test]
    fn verify_is_byte_exact_for_non_ascii() {
        let t = Token::new("pässwörd");
        assert!(t.verify("pässwörd"));
        // Same byte length, differing in a continuation byte — this one reaches the fold
        // rather than being caught by the length check.
        assert!(!t.verify("pässwörf"));
        // Different byte length (`ä` is two bytes) — caught by the length check.
        assert!(!t.verify("passwörd"));
    }
}
