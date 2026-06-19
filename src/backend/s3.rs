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
use rusty_s3::actions::{DeleteObject, GetObject, ListObjectsV2, PutObject};
use rusty_s3::{Bucket, Credentials, S3Action, UrlStyle};
use url::Url;

use super::{BackendLock, Persistence, validate_key};
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

    fn try_create_exclusive(&self, key: &str, bytes: &[u8]) -> Result<Option<bool>> {
        validate_key(key)?;
        let path = self.path(key);
        // `If-None-Match: *` makes the PUT a create-if-absent. The header is *signed*
        // (added to the request before signing), so it must be sent verbatim on the wire.
        let mut action = PutObject::new(&self.bucket, Some(&self.creds), &path);
        action
            .headers_mut()
            .insert(IF_NONE_MATCH_HEADER, IF_NONE_MATCH_ANY);
        let url = action.sign(PRESIGN_TTL);
        let (status, body) = self.http.put(
            url.as_str(),
            &[(IF_NONE_MATCH_HEADER, IF_NONE_MATCH_ANY)],
            bytes,
        )?;
        match status {
            s if (200..300).contains(&s) => Ok(Some(true)), // created — we won the lock
            412 => Ok(Some(false)), // precondition failed: it already exists — lost the race
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

        impl MockS3 {
            pub(super) fn start() -> MockS3 {
                let listener = TcpListener::bind("127.0.0.1:0").unwrap();
                let port = listener.local_addr().unwrap().port();
                let store: Arc<Mutex<HashMap<String, Vec<u8>>>> =
                    Arc::new(Mutex::new(HashMap::new()));
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

        fn serve(mut stream: TcpStream, store: Arc<Mutex<HashMap<String, Vec<u8>>>>) {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).is_err() || request_line.is_empty() {
                return;
            }
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or("").to_string();
            let target = parts.next().unwrap_or("").to_string();

            // Read headers; capture Content-Length and any If-None-Match precondition.
            let mut content_length = 0usize;
            let mut if_none_match = false;
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

            let (status, payload): (&str, Vec<u8>) = if is_list {
                let map = store.lock().unwrap();
                let mut xml =
                    String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult>");
                let mut names: Vec<&String> = map.keys().collect();
                names.sort();
                for k in names {
                    let size = map[k].len();
                    xml.push_str(&format!(
                        "<Contents><Key>{k}</Key><ETag>\"x\"</ETag>\
                         <LastModified>1970-01-01T00:00:00.000Z</LastModified>\
                         <Size>{size}</Size></Contents>"
                    ));
                }
                xml.push_str("</ListBucketResult>");
                ("200 OK", xml.into_bytes())
            } else {
                match method.as_str() {
                    // `If-None-Match: *` → create-if-absent: 412 when the key already exists.
                    "PUT" if if_none_match && store.lock().unwrap().contains_key(&key) => {
                        ("412 Precondition Failed", Vec::new())
                    }
                    "PUT" => {
                        store.lock().unwrap().insert(key, body);
                        ("200 OK", Vec::new())
                    }
                    "DELETE" => {
                        store.lock().unwrap().remove(&key);
                        ("204 No Content", Vec::new())
                    }
                    "GET" => match store.lock().unwrap().get(&key) {
                        Some(v) => ("200 OK", v.clone()),
                        None => ("404 Not Found", Vec::new()),
                    },
                    _ => ("405 Method Not Allowed", Vec::new()),
                }
            };

            let _ = write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            let _ = stream.write_all(&payload);
            let _ = stream.flush();
        }
    }
}
