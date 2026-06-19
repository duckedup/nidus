//! [`S3`]: an Amazon S3 (and S3-compatible: R2, MinIO, …) [`Persistence`] backend.
//!
//! Selected by `s3://<bucket>[/<prefix>]`, with credentials/region/endpoint from the
//! standard AWS environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
//! `AWS_SESSION_TOKEN`, `AWS_REGION`/`AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL`). Object
//! ops are whole-object `get`/`put`/`delete`/`list`; there is **no native append**
//! (`appender` returns `None`), so S3 is for snapshots and whole-object use, not as a
//! live append-backed `data`/`log` store.
//!
//! [`rusty-s3`](rusty_s3) is sans-IO: it builds and Sigv4-signs each request into a
//! presigned [`Url`], which [`Http`] then executes. Signing is pure RustCrypto HMAC —
//! no network, no async — so it is unit-testable offline, and the execution path is
//! covered by a localhost mock in the tests.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use http::HeaderMap;
use rusty_s3::actions::{DeleteObject, GetObject, ListObjectsV2, PutObject};
use rusty_s3::{Bucket, Credentials, S3Action, UrlStyle};
use url::Url;

use super::{BackendLock, CasOutcome, Persistence, validate_key};
use crate::backend::cloud::Http;

/// How long a presigned URL is valid. Each request signs a fresh one; this only bounds
/// the window between signing and the (immediate) execution.
const PRESIGN_TTL: Duration = Duration::from_secs(300);

/// The `If-None-Match: *` precondition that makes a PUT a create-if-absent (it succeeds
/// only when no object matches — i.e. none exists). The header name is **lowercase**:
/// SigV4 signs canonical (lowercased) header names, and rusty-s3 signs the name as given,
/// so it must already be lowercase here to match what AWS recomputes. HTTP header names
/// are case-insensitive on the wire, so sending it lowercase is equally fine.
const IF_NONE_MATCH_HEADER: &str = "if-none-match";
const IF_NONE_MATCH_ANY: &str = "*";

/// The `If-Match: <etag>` precondition that makes a PUT a compare-and-swap (it succeeds only
/// when the object's current `ETag` equals the one this writer last saw). Lowercase for the
/// same SigV4 reason as [`IF_NONE_MATCH_HEADER`].
const IF_MATCH_HEADER: &str = "if-match";

/// An S3 (or S3-compatible) persistence backend rooted at a bucket and optional key
/// prefix.
pub struct S3 {
    bucket: Bucket,
    creds: Credentials,
    /// Key prefix within the bucket (`""` or `"a/b"`, never trailing-slashed). Object
    /// keys map to `<prefix>/<key>`.
    prefix: String,
    http: Http,
}

impl S3 {
    /// Build from the part of an `s3://` URL after the scheme: `<bucket>[/<prefix>]`.
    /// Credentials, region, and endpoint come from the AWS environment.
    pub(crate) fn from_url(rest: &str) -> Result<S3> {
        let (bucket_name, prefix) = match rest.split_once('/') {
            Some((b, p)) => (b, p.trim_end_matches('/')),
            None => (rest, ""),
        };
        if bucket_name.is_empty() {
            bail!("s3:// URL is missing a bucket name (expected s3://<bucket>[/<prefix>])");
        }

        let key = env("AWS_ACCESS_KEY_ID")
            .context("AWS_ACCESS_KEY_ID is not set (required for the s3:// backend)")?;
        let secret = env("AWS_SECRET_ACCESS_KEY")
            .context("AWS_SECRET_ACCESS_KEY is not set (required for the s3:// backend)")?;
        let creds = match env("AWS_SESSION_TOKEN") {
            Some(token) => Credentials::new_with_token(key, secret, token),
            None => Credentials::new(key, secret),
        };
        let region = env("AWS_REGION")
            .or_else(|| env("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|| "us-east-1".to_string());

        // A custom endpoint (MinIO/R2/localstack) is addressed path-style; plain AWS is
        // virtual-host style (`https://<bucket>.s3.<region>.amazonaws.com`).
        let (endpoint, style) = match env("AWS_ENDPOINT_URL") {
            Some(e) => (e, UrlStyle::Path),
            None => (
                format!("https://s3.{region}.amazonaws.com"),
                UrlStyle::VirtualHost,
            ),
        };
        Self::build(&endpoint, style, bucket_name, prefix, creds, region)
    }

    /// The endpoint/style/credentials-explicit constructor shared by [`from_url`] and
    /// the tests (which point it at a localhost mock).
    fn build(
        endpoint: &str,
        style: UrlStyle,
        bucket: &str,
        prefix: &str,
        creds: Credentials,
        region: String,
    ) -> Result<S3> {
        let endpoint = Url::parse(endpoint)
            .with_context(|| format!("invalid S3 endpoint URL {endpoint:?}"))?;
        let bucket = Bucket::new(endpoint, style, bucket.to_string(), region)
            .context("failed to construct S3 bucket")?;
        Ok(S3 {
            bucket,
            creds,
            prefix: prefix.to_string(),
            http: Http::new(),
        })
    }

    /// Map a flat object key to its full in-bucket path (`<prefix>/<key>`).
    fn path(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}/{}", self.prefix, key)
        }
    }
}

impl Persistence for S3 {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        validate_key(key)?;
        let path = self.path(key);
        let url = GetObject::new(&self.bucket, Some(&self.creds), &path).sign(PRESIGN_TTL);
        let (status, body) = self.http.get(url.as_str())?;
        match status {
            200 => Ok(Some(body)),
            404 => Ok(None),
            s => bail!("S3 GET {path} failed: HTTP {s}: {}", show(&body)),
        }
    }

    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        validate_key(key)?;
        let path = self.path(key);
        let url = PutObject::new(&self.bucket, Some(&self.creds), &path).sign(PRESIGN_TTL);
        let (status, body) = self.http.put(url.as_str(), &[], bytes)?;
        if (200..300).contains(&status) {
            Ok(())
        } else {
            bail!("S3 PUT {path} failed: HTTP {status}: {}", show(&body))
        }
    }

    fn get_cas(&self, key: &str) -> Result<Option<(Vec<u8>, Option<String>)>> {
        validate_key(key)?;
        let path = self.path(key);
        let url = GetObject::new(&self.bucket, Some(&self.creds), &path).sign(PRESIGN_TTL);
        let (status, body, headers) = self.http.get_h(url.as_str())?;
        match status {
            200 => Ok(Some((body, etag(&headers)))),
            404 => Ok(None),
            s => bail!("S3 GET {path} failed: HTTP {s}: {}", show(&body)),
        }
    }

    fn put_cas(&self, key: &str, bytes: &[u8], expected: Option<&str>) -> Result<CasOutcome> {
        validate_key(key)?;
        let path = self.path(key);
        // `If-Match: <etag>` makes the PUT a compare-and-swap; `If-None-Match: *` makes it a
        // create-if-absent (`expected: None`). Either header is *signed* (added before signing)
        // so it must be sent verbatim on the wire. A 412 means the precondition failed — a peer
        // changed the object since `expected` (or it already exists), i.e. we are fenced.
        let (name, value) = match expected {
            Some(etag) => (IF_MATCH_HEADER, etag),
            None => (IF_NONE_MATCH_HEADER, IF_NONE_MATCH_ANY),
        };
        let mut action = PutObject::new(&self.bucket, Some(&self.creds), &path);
        action.headers_mut().insert(name, value);
        let url = action.sign(PRESIGN_TTL);
        let (status, body, headers) = self.http.put_h(url.as_str(), &[(name, value)], bytes)?;
        match status {
            s if (200..300).contains(&s) => Ok(CasOutcome::Written(etag(&headers))),
            412 => Ok(CasOutcome::Stale),
            s => bail!(
                "S3 conditional PUT {path} failed: HTTP {s}: {}",
                show(&body)
            ),
        }
    }

    fn delete(&self, key: &str) -> Result<()> {
        validate_key(key)?;
        let path = self.path(key);
        let url = DeleteObject::new(&self.bucket, Some(&self.creds), &path).sign(PRESIGN_TTL);
        let (status, body) = self.http.delete(url.as_str())?;
        // 204 on success; 404 means already gone — both fine (delete is idempotent).
        if (200..300).contains(&status) || status == 404 {
            Ok(())
        } else {
            bail!("S3 DELETE {path} failed: HTTP {status}: {}", show(&body))
        }
    }

    fn list(&self) -> Result<Vec<String>> {
        let want_prefix = if self.prefix.is_empty() {
            String::new()
        } else {
            format!("{}/", self.prefix)
        };
        let mut keys = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut action = self.bucket.list_objects_v2(Some(&self.creds));
            if !want_prefix.is_empty() {
                action.query_mut().insert("prefix", want_prefix.clone());
            }
            if let Some(t) = &token {
                action.query_mut().insert("continuation-token", t.clone());
            }
            let url = action.sign(PRESIGN_TTL);
            let (status, body) = self.http.get(url.as_str())?;
            if status != 200 {
                bail!("S3 LIST failed: HTTP {status}: {}", show(&body));
            }
            let parsed = ListObjectsV2::parse_response(&body)
                .context("failed to parse S3 list-objects-v2 response")?;
            for c in parsed.contents {
                // Strip the prefix back off so callers see flat keys.
                let k = c.key.strip_prefix(&want_prefix).unwrap_or(&c.key);
                if !k.is_empty() {
                    keys.push(k.to_string());
                }
            }
            match parsed.next_continuation_token {
                Some(t) => token = Some(t),
                None => break,
            }
        }
        keys.sort();
        Ok(keys)
    }

    fn try_lock(&self, _key: &str, _ttl: Duration) -> Result<Option<Box<dyn BackendLock>>> {
        bail!(
            "the S3 backend has no native writer lock — a live object-backed store uses \
             the advisory lock (Store::acquire_lock) instead of calling try_lock"
        )
    }

    fn has_native_lock(&self) -> bool {
        false
    }
}

/// Read an env var, treating empty as unset.
fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

/// The object's `ETag` (S3's CAS version token) from a response, if present. Returned
/// verbatim, quotes and all, so it round-trips back into a signed `If-Match`.
fn etag(headers: &HeaderMap) -> Option<String> {
    headers
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// A short, lossy view of an error response body for messages.
fn show(body: &[u8]) -> String {
    let s = String::from_utf8_lossy(body);
    s.chars().take(300).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn creds() -> Credentials {
        Credentials::new("AKIDEXAMPLE", "secret")
    }

    // ── Offline: presigned-URL construction (Miri-clean) ────────────────────────

    #[test]
    fn presigned_get_url_carries_bucket_key_and_signature() {
        let s3 = S3::build(
            "https://s3.us-east-1.amazonaws.com",
            UrlStyle::VirtualHost,
            "my-bucket",
            "",
            creds(),
            "us-east-1".to_string(),
        )
        .unwrap();
        let url = GetObject::new(&s3.bucket, Some(&s3.creds), &s3.path("data")).sign(PRESIGN_TTL);
        let u = url.as_str();
        assert!(
            u.starts_with("https://my-bucket.s3.us-east-1.amazonaws.com/data?"),
            "{u}"
        );
        assert!(u.contains("X-Amz-Signature="), "{u}");
        assert!(u.contains("X-Amz-Credential="), "{u}");
    }

    #[test]
    fn prefix_is_prepended_to_keys() {
        let s3 = S3::build(
            "https://s3.us-east-1.amazonaws.com",
            UrlStyle::VirtualHost,
            "b",
            "snapshots/nightly",
            creds(),
            "us-east-1".to_string(),
        )
        .unwrap();
        assert_eq!(s3.path("snap.tar.gz"), "snapshots/nightly/snap.tar.gz");
    }

    #[test]
    fn conditional_put_url_signs_the_if_none_match_header() {
        let s3 = S3::build(
            "https://s3.us-east-1.amazonaws.com",
            UrlStyle::VirtualHost,
            "b",
            "",
            creds(),
            "us-east-1".to_string(),
        )
        .unwrap();
        let path = s3.path("lock");
        let mut action = PutObject::new(&s3.bucket, Some(&s3.creds), &path);
        action
            .headers_mut()
            .insert(IF_NONE_MATCH_HEADER, IF_NONE_MATCH_ANY);
        let url = action.sign(PRESIGN_TTL);
        let u = url.as_str();
        // The conditional header is folded into the SigV4 signed-headers set (lowercased,
        // as AWS canonicalisation requires), so it must travel with the request — assert it
        // was signed under its canonical name (not silently dropped or upper-cased).
        assert!(
            u.contains("X-Amz-SignedHeaders=") && u.contains("if-none-match"),
            "{u}"
        );
    }

    #[test]
    fn try_lock_is_unsupported() {
        let s3 = S3::build(
            "https://s3.us-east-1.amazonaws.com",
            UrlStyle::VirtualHost,
            "b",
            "",
            creds(),
            "us-east-1".to_string(),
        )
        .unwrap();
        assert!(s3.try_lock("lock", Duration::from_secs(1)).is_err());
    }

    // ── Online: round-trip against a localhost mock S3 (Miri-ignored) ───────────

    #[cfg_attr(miri, ignore)]
    #[test]
    fn round_trip_against_mock_s3() {
        let server = mock::MockS3::start();
        let s3 = S3::build(
            &server.endpoint(),
            UrlStyle::Path,
            "bucket",
            "",
            creds(),
            "us-east-1".to_string(),
        )
        .unwrap();

        assert!(s3.get("data").unwrap().is_none(), "absent object → None");
        s3.put("data", b"\x00\x01\x02hello").unwrap();
        s3.put("log", b"frames").unwrap();
        assert_eq!(
            s3.get("data").unwrap().as_deref(),
            Some(&b"\x00\x01\x02hello"[..])
        );

        let mut keys = s3.list().unwrap();
        keys.sort();
        assert_eq!(keys, vec!["data".to_string(), "log".to_string()]);

        s3.delete("data").unwrap();
        assert!(s3.get("data").unwrap().is_none());
        s3.delete("data").unwrap(); // idempotent
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn conditional_create_is_create_if_absent_against_mock_s3() {
        let server = mock::MockS3::start();
        let s3 = S3::build(
            &server.endpoint(),
            UrlStyle::Path,
            "bucket",
            "",
            creds(),
            "us-east-1".to_string(),
        )
        .unwrap();

        // First create wins (object absent → 200 → Some(true)).
        assert_eq!(s3.try_create_exclusive("lock", b"1").unwrap(), Some(true));
        // Second create loses the race (object present → 412 → Some(false)).
        assert_eq!(s3.try_create_exclusive("lock", b"2").unwrap(), Some(false));
        // The body is the first writer's — the losing create did not overwrite it.
        assert_eq!(s3.get("lock").unwrap().as_deref(), Some(&b"1"[..]));
        // Once deleted, a create can win again.
        s3.delete("lock").unwrap();
        assert_eq!(s3.try_create_exclusive("lock", b"3").unwrap(), Some(true));
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn compare_and_swap_round_trips_against_mock_s3() {
        let server = mock::MockS3::start();
        let s3 = S3::build(
            &server.endpoint(),
            UrlStyle::Path,
            "bucket",
            "",
            creds(),
            "us-east-1".to_string(),
        )
        .unwrap();

        // Create-if-absent (`expected: None`) wins, then loses once the object exists.
        let CasOutcome::Written(tok0) = s3.put_cas("m", b"v0", None).unwrap() else {
            panic!("create-if-absent should win on an empty key");
        };
        assert!(tok0.is_some(), "the write reports the new ETag");
        assert!(matches!(
            s3.put_cas("m", b"vX", None).unwrap(),
            CasOutcome::Stale
        ));

        // get_cas returns the current token; a CAS against it succeeds and mints a new token.
        let (bytes, tok) = s3.get_cas("m").unwrap().unwrap();
        assert_eq!(bytes, b"v0");
        assert_eq!(
            tok, tok0,
            "get_cas reports the same ETag the write returned"
        );
        let CasOutcome::Written(tok1) = s3.put_cas("m", b"v1", tok.as_deref()).unwrap() else {
            panic!("a CAS against the current token should win");
        };
        assert_ne!(tok0, tok1, "the token advances on each write");

        // A CAS against the now-stale token is fenced; the current value is unchanged.
        assert!(matches!(
            s3.put_cas("m", b"v2", tok0.as_deref()).unwrap(),
            CasOutcome::Stale
        ));
        assert_eq!(s3.get("m").unwrap().as_deref(), Some(&b"v1"[..]));
    }

    /// A minimal in-process S3-shaped HTTP server: just enough of the object + list
    /// API to exercise the backend's request execution and response handling. It
    /// ignores auth (the signature is rusty-s3's concern, asserted offline above).
    mod mock {
        use std::collections::HashMap;
        use std::io::{BufRead, BufReader, Read, Write};
        use std::net::{TcpListener, TcpStream};
        use std::sync::{Arc, Mutex};
        use std::thread;

        pub(super) struct MockS3 {
            port: u16,
        }

        /// Object bytes + a monotonic generation (the mock's `ETag` source), plus the next
        /// generation to mint — enough to model S3's `If-Match`/`If-None-Match` conditionals.
        #[derive(Default)]
        pub(super) struct State {
            objects: HashMap<String, (Vec<u8>, u64)>,
            next_gen: u64,
        }

        impl MockS3 {
            pub(super) fn start() -> MockS3 {
                let listener = TcpListener::bind("127.0.0.1:0").unwrap();
                let port = listener.local_addr().unwrap().port();
                let store: Arc<Mutex<State>> = Arc::new(Mutex::new(State::default()));
                thread::spawn(move || {
                    for conn in listener.incoming() {
                        let Ok(conn) = conn else { break };
                        let store = Arc::clone(&store);
                        // One request per connection (we reply Connection: close).
                        thread::spawn(move || serve(conn, store));
                    }
                });
                MockS3 { port }
            }

            pub(super) fn endpoint(&self) -> String {
                format!("http://127.0.0.1:{}", self.port)
            }
        }

        fn serve(mut stream: TcpStream, store: Arc<Mutex<State>>) {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).is_err() || request_line.is_empty() {
                return;
            }
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or("").to_string();
            let target = parts.next().unwrap_or("").to_string();

            // Read headers; capture Content-Length and the conditional preconditions.
            let mut content_length = 0usize;
            let mut if_none_match = false;
            let mut if_match: Option<String> = None;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    return;
                }
                let line = line.trim_end();
                if line.is_empty() {
                    break;
                }
                let lower = line.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap_or(0);
                }
                if let Some(v) = lower.strip_prefix("if-none-match:") {
                    if_none_match = v.trim() == "*";
                }
                if let Some(v) = lower.strip_prefix("if-match:") {
                    if_match = Some(v.trim().to_string());
                }
            }
            let mut body = vec![0u8; content_length];
            if content_length > 0 && reader.read_exact(&mut body).is_err() {
                return;
            }

            // `/bucket/key?query` → key; `/bucket?query` (no key) → a list request.
            let path = target.split('?').next().unwrap_or("");
            let is_list = target.contains("list-type");
            let key: String = path
                .trim_start_matches('/')
                .split_once('/')
                .map(|(_bucket, k)| k)
                .unwrap_or("")
                .to_string();

            // The `ETag` of the object written/read, echoed in the response (the CAS token).
            let mut etag: Option<String> = None;
            let (status, payload): (&str, Vec<u8>) = if is_list {
                let st = store.lock().unwrap();
                let mut xml =
                    String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult>");
                let mut names: Vec<&String> = st.objects.keys().collect();
                names.sort();
                for k in names {
                    let size = st.objects[k].0.len();
                    xml.push_str(&format!(
                        "<Contents><Key>{k}</Key><ETag>\"x\"</ETag>\
                         <LastModified>1970-01-01T00:00:00.000Z</LastModified>\
                         <Size>{size}</Size></Contents>"
                    ));
                }
                xml.push_str("</ListBucketResult>");
                ("200 OK", xml.into_bytes())
            } else {
                let mut st = store.lock().unwrap();
                let cur_etag = |st: &State| st.objects.get(&key).map(|(_, g)| format!("\"{g}\""));
                match method.as_str() {
                    // `If-None-Match: *` → create-if-absent: 412 when the key already exists.
                    "PUT" if if_none_match && st.objects.contains_key(&key) => {
                        ("412 Precondition Failed", Vec::new())
                    }
                    // `If-Match: <etag>` → compare-and-swap: 412 unless it matches the current one.
                    "PUT" if if_match.is_some() && if_match != cur_etag(&st) => {
                        ("412 Precondition Failed", Vec::new())
                    }
                    "PUT" => {
                        st.next_gen += 1;
                        let g = st.next_gen;
                        st.objects.insert(key, (body, g));
                        etag = Some(format!("\"{g}\""));
                        ("200 OK", Vec::new())
                    }
                    "DELETE" => {
                        st.objects.remove(&key);
                        ("204 No Content", Vec::new())
                    }
                    "GET" => match st.objects.get(&key) {
                        Some((v, g)) => {
                            etag = Some(format!("\"{g}\""));
                            ("200 OK", v.clone())
                        }
                        None => ("404 Not Found", Vec::new()),
                    },
                    _ => ("405 Method Not Allowed", Vec::new()),
                }
            };

            let etag_header = match &etag {
                Some(e) => format!("ETag: {e}\r\n"),
                None => String::new(),
            };
            let _ = write!(
                stream,
                "HTTP/1.1 {status}\r\n{etag_header}Content-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            let _ = stream.write_all(&payload);
            let _ = stream.flush();
        }
    }
}
