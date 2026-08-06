//! Bearer-token authentication for `nidus serve` (nidus-abx.5).

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
#[derive(Clone)]
pub(super) struct Token(std::sync::Arc<str>);

impl Token {
    pub(super) fn new(secret: impl Into<std::sync::Arc<str>>) -> Token {
        Token(secret.into())
    }

    /// Whether `presented` is the configured token, in time independent of how many
    /// leading bytes match.
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
