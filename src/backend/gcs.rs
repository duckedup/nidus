//! [`Gcs`]: a Google Cloud Storage [`Persistence`] backend.
//!
//! Selected by `gs://<bucket>[/<prefix>]` (alias `gcs://`), authenticated with a
//! service-account key whose path is in `GOOGLE_APPLICATION_CREDENTIALS` (or the JSON
//! inline in `GOOGLE_APPLICATION_CREDENTIALS_JSON`). Like [`S3`](super::S3) it is a
//! whole-object backend (`get`/`put`/`delete`/`list`, no native append) — for
//! snapshots and whole-object use.
//!
//! [`tame-gcs`](tame_gcs) and [`tame-oauth`](tame_oauth) are sans-IO: they build the
//! GCS request (and the OAuth2 token-exchange request) as [`http::Request`]s, which
//! [`Http`] executes. `tame-oauth` signs the service-account JWT (RSA via `ring`) and
//! caches the access token internally.
//!
//! Note: unlike S3, the request/auth path is exercised only by construction-level unit
//! tests here — there is no clean local mock for GCS's fixed OAuth token endpoint, so
//! end-to-end behaviour is verified against a real bucket out of band.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use http::header::{AUTHORIZATION, HeaderValue};
use tame_gcs::objects::{ListOptional, ListResponse, Object};
use tame_gcs::{ApiResponse, BucketName, ObjectId};
use tame_oauth::gcp::{ServiceAccountInfo, ServiceAccountProvider, TokenOrRequest, TokenProvider};

use super::{BackendLock, Persistence, validate_key};
use crate::backend::cloud::Http;

/// The OAuth2 scope for reading and writing GCS objects.
const SCOPE: &str = "https://www.googleapis.com/auth/devstorage.read_write";

/// A Google Cloud Storage persistence backend rooted at a bucket and optional key
/// prefix.
pub struct Gcs {
    bucket: String,
    /// Key prefix within the bucket (`""` or `"a/b"`, never trailing-slashed).
    prefix: String,
    auth: ServiceAccountProvider,
    http: Http,
}

impl Gcs {
    /// Build from the part of a `gs://` URL after the scheme: `<bucket>[/<prefix>]`.
    /// Service-account credentials come from `GOOGLE_APPLICATION_CREDENTIALS` (a path)
    /// or `GOOGLE_APPLICATION_CREDENTIALS_JSON` (the key JSON inline).
    pub(crate) fn from_url(rest: &str) -> Result<Gcs> {
        let (bucket, prefix) = match rest.split_once('/') {
            Some((b, p)) => (b, p.trim_end_matches('/')),
            None => (rest, ""),
        };
        if bucket.is_empty() {
            bail!("gs:// URL is missing a bucket name (expected gs://<bucket>[/<prefix>])");
        }

        let json = match env("GOOGLE_APPLICATION_CREDENTIALS_JSON") {
            Some(j) => j,
            None => {
                let path = env("GOOGLE_APPLICATION_CREDENTIALS").context(
                    "neither GOOGLE_APPLICATION_CREDENTIALS nor \
                     GOOGLE_APPLICATION_CREDENTIALS_JSON is set (required for the gs:// backend)",
                )?;
                std::fs::read_to_string(&path)
                    .with_context(|| format!("failed to read service-account key at {path}"))?
            }
        };
        let info = ServiceAccountInfo::deserialize(json)
            .map_err(|e| anyhow::anyhow!("invalid GCS service-account key: {e}"))?;
        let auth = ServiceAccountProvider::new(info)
            .map_err(|e| anyhow::anyhow!("failed to initialise GCS service-account auth: {e}"))?;

        Ok(Gcs {
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            auth,
            http: Http::new(),
        })
    }

    /// Map a flat object key to its full in-bucket name (`<prefix>/<key>`).
    fn name(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}/{}", self.prefix, key)
        }
    }

    /// A currently-valid OAuth2 access token, fetching (and caching, inside the
    /// provider) one via the token endpoint when needed.
    fn token(&self) -> Result<String> {
        match self
            .auth
            .get_token(&[SCOPE])
            .map_err(|e| anyhow::anyhow!("GCS token request failed: {e}"))?
        {
            TokenOrRequest::Token(t) => Ok(t.access_token),
            TokenOrRequest::Request {
                request,
                scope_hash,
                ..
            } => {
                let (status, body) = self.http.run(request)?;
                let resp = http::Response::builder()
                    .status(status)
                    .body(body)
                    .context("build GCS token response")?;
                let token = self
                    .auth
                    .parse_token_response(scope_hash, resp)
                    .map_err(|e| anyhow::anyhow!("failed to parse GCS token response: {e}"))?;
                Ok(token.access_token)
            }
        }
    }

    /// Attach a fresh bearer token to a request and execute it.
    fn run_authed(&self, req: http::Request<Vec<u8>>) -> Result<(u16, Vec<u8>)> {
        let token = self.token()?;
        let (mut parts, body) = req.into_parts();
        parts.headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).context("build GCS auth header")?,
        );
        self.http.run(http::Request::from_parts(parts, body))
    }

    fn object_id(&self, key: &str) -> Result<ObjectId<'static>> {
        ObjectId::new(self.bucket.clone(), self.name(key))
            .map_err(|e| anyhow::anyhow!("invalid GCS object id: {e}"))
    }
}

impl Persistence for Gcs {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        validate_key(key)?;
        let oid = self.object_id(key)?;
        let req = Object::default()
            .download(&oid, None)
            .map_err(gcs_err)?
            .map(|_empty| Vec::new());
        let (status, body) = self.run_authed(req)?;
        match status {
            200 => Ok(Some(body)),
            404 => Ok(None),
            s => bail!("GCS download {key} failed: HTTP {s}: {}", show(&body)),
        }
    }

    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        validate_key(key)?;
        let oid = self.object_id(key)?;
        let req = Object::default()
            .insert_simple(&oid, bytes.to_vec(), bytes.len() as u64, None)
            .map_err(gcs_err)?;
        let (status, body) = self.run_authed(req)?;
        if (200..300).contains(&status) {
            Ok(())
        } else {
            bail!("GCS insert {key} failed: HTTP {status}: {}", show(&body))
        }
    }

    fn delete(&self, key: &str) -> Result<()> {
        validate_key(key)?;
        let oid = self.object_id(key)?;
        let req = Object::default()
            .delete(&oid, None)
            .map_err(gcs_err)?
            .map(|_empty| Vec::new());
        let (status, body) = self.run_authed(req)?;
        // 2xx on success; 404 means already gone — both fine (delete is idempotent).
        if (200..300).contains(&status) || status == 404 {
            Ok(())
        } else {
            bail!("GCS delete {key} failed: HTTP {status}: {}", show(&body))
        }
    }

    fn list(&self) -> Result<Vec<String>> {
        let bucket = BucketName::try_from(self.bucket.as_str())
            .map_err(|e| anyhow::anyhow!("invalid GCS bucket name: {e}"))?;
        let want_prefix = if self.prefix.is_empty() {
            String::new()
        } else {
            format!("{}/", self.prefix)
        };
        let mut keys = Vec::new();
        let mut page_token: Option<String> = None;
        loop {
            let optional = ListOptional {
                prefix: if want_prefix.is_empty() {
                    None
                } else {
                    Some(want_prefix.as_str())
                },
                page_token: page_token.as_deref(),
                ..Default::default()
            };
            let req = Object::default()
                .list(&bucket, Some(optional))
                .map_err(gcs_err)?
                .map(|_empty| Vec::new());
            let (status, body) = self.run_authed(req)?;
            if status != 200 {
                bail!("GCS list failed: HTTP {status}: {}", show(&body));
            }
            let resp = http::Response::builder()
                .status(status)
                .body(body.as_slice())
                .context("build GCS list response")?;
            let parsed = <ListResponse as ApiResponse<&[u8]>>::try_from_parts(resp)
                .map_err(|e| anyhow::anyhow!("failed to parse GCS list response: {e}"))?;
            for meta in parsed.objects {
                if let Some(name) = meta.name {
                    let k = name.strip_prefix(&want_prefix).unwrap_or(&name);
                    if !k.is_empty() {
                        keys.push(k.to_string());
                    }
                }
            }
            match parsed.page_token {
                Some(t) => page_token = Some(t),
                None => break,
            }
        }
        keys.sort();
        Ok(keys)
    }

    fn try_lock(&self, _key: &str, _ttl: Duration) -> Result<Option<Box<dyn BackendLock>>> {
        bail!(
            "the GCS backend does not support writer locking — it is for snapshots and \
             whole-object use, not a lock-coordinated live store"
        )
    }
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

fn gcs_err(e: tame_gcs::Error) -> anyhow::Error {
    anyhow::anyhow!("GCS request build failed: {e}")
}

fn show(body: &[u8]) -> String {
    String::from_utf8_lossy(body).chars().take(300).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal valid-shape service-account key (RSA key omitted-but-parseable is hard;
    // these tests cover the URL/prefix logic that doesn't require a usable key).

    #[test]
    fn name_prepends_prefix() {
        let gcs = Gcs {
            bucket: "b".to_string(),
            prefix: "snapshots/nightly".to_string(),
            auth: dummy_auth(),
            http: Http::new(),
        };
        assert_eq!(gcs.name("snap.tar.gz"), "snapshots/nightly/snap.tar.gz");
        let flat = Gcs {
            bucket: "b".to_string(),
            prefix: String::new(),
            auth: dummy_auth(),
            http: Http::new(),
        };
        assert_eq!(flat.name("data"), "data");
    }

    #[test]
    fn object_id_builds_for_valid_names() {
        let gcs = Gcs {
            bucket: "my-bucket".to_string(),
            prefix: String::new(),
            auth: dummy_auth(),
            http: Http::new(),
        };
        let oid = gcs.object_id("data").unwrap();
        // The download request targets the bucket + object path.
        let req = Object::default().download(&oid, None).unwrap();
        let uri = req.uri().to_string();
        assert!(uri.contains("/b/my-bucket/o/data"), "{uri}");
        assert!(uri.contains("alt=media"), "{uri}");
    }

    #[test]
    fn try_lock_is_unsupported() {
        let gcs = Gcs {
            bucket: "b".to_string(),
            prefix: String::new(),
            auth: dummy_auth(),
            http: Http::new(),
        };
        assert!(gcs.try_lock("lock", Duration::from_secs(1)).is_err());
    }

    /// A throwaway provider built from a syntactically-valid (but non-functional)
    /// service-account key — enough to construct a `Gcs` for the URL/path tests, which
    /// never actually request a token.
    fn dummy_auth() -> ServiceAccountProvider {
        // A minimal RSA private key (PKCS#8) generated for tests only — never used to
        // sign a real request here. Parsing it just has to succeed.
        let json = r#"{
            "type": "service_account",
            "project_id": "test",
            "private_key_id": "0",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIBVAIBADANBgkqhkiG9w0BAQEFAASCAT4wggE6AgEAAkEA\n-----END PRIVATE KEY-----\n",
            "client_email": "test@test.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        }"#;
        let info = ServiceAccountInfo::deserialize(json).unwrap();
        ServiceAccountProvider::new(info).unwrap()
    }
}
