//! AWS credential resolution for the [`S3`](super::S3) backend.
//!
//! Two classes, picked by the environment (mirroring the AWS SDK default chain):
//!
//! - **Static** — long-lived keys in `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
//!   (+ optional `AWS_SESSION_TOKEN`). Never expire; used verbatim.
//! - **Keyless / temporary** — no static keys, so credentials are fetched from a metadata
//!   source and **refreshed before they expire**:
//!   - **Web identity (IRSA)** — `AWS_WEB_IDENTITY_TOKEN_FILE` + `AWS_ROLE_ARN`: exchange the
//!     projected token at STS `AssumeRoleWithWebIdentity`. This is how EKS pods authenticate.
//!   - **ECS / Fargate** — `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` (or `…_FULL_URI`): the
//!     task-role endpoint.
//!   - **EC2 instance role** — IMDSv2 on `169.254.169.254` (token-protected). Tried last.
//!
//! The HTTP responses (STS XML, ECS/IMDS JSON, the ISO-8601 expiry) are parsed by the small,
//! dependency-free, unit-tested helpers at the bottom — the same "parse fixed wire shapes
//! against in-memory buffers" discipline the codecs use, so they stay Miri-clean.

use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rusty_s3::Credentials;

use crate::backend::cloud::Http;

/// Refresh temporary credentials once they are within this margin of expiring, so an
/// in-flight request never races the cutover.
const REFRESH_MARGIN_SECS: u64 = 300;
/// Fallback validity when a source omits (or we cannot parse) an expiry. All of these
/// sources mint credentials valid for ≥ 1h, so a 10-minute lease is safe and self-heals.
const FALLBACK_TTL_SECS: u64 = 600;
/// Short ceiling on metadata calls so an off-cloud host fails fast instead of hanging on
/// the unroutable link-local IMDS address.
const METADATA_TIMEOUT: Duration = Duration::from_secs(5);
/// IMDS / ECS link-local base addresses.
const IMDS_BASE: &str = "http://169.254.169.254";
const ECS_BASE: &str = "http://169.254.170.2";

/// How the S3 backend obtains credentials for each signing call.
pub(crate) enum AwsCredentials {
    /// Long-lived keys; cloned as-is.
    Static(Credentials),
    /// A metadata-sourced provider with a refresh cache behind a lock.
    Dynamic(Mutex<Dynamic>),
}

/// The refreshable state for keyless credentials.
pub(crate) struct Dynamic {
    source: Source,
    http: Http,
    cached: Option<Cached>,
}

struct Cached {
    creds: Credentials,
    /// Unix seconds at which these credentials expire.
    expires_at: u64,
}

/// A keyless credential source and the inputs it needs.
enum Source {
    WebIdentity {
        role_arn: String,
        token_file: String,
        session_name: String,
        sts_endpoint: String,
    },
    /// A fully-qualified credentials URL plus an optional `Authorization` header value.
    Container {
        url: String,
        auth: Option<String>,
    },
    Imds,
}

impl AwsCredentials {
    /// Resolve the credential provider from the environment, following the AWS default
    /// chain: static keys → web identity (IRSA) → ECS task role → EC2 IMDS.
    pub(crate) fn from_env() -> Result<AwsCredentials> {
        // 1. Static keys win.
        if let Some(key) = env("AWS_ACCESS_KEY_ID") {
            let secret = env("AWS_SECRET_ACCESS_KEY").context(
                "AWS_ACCESS_KEY_ID is set but AWS_SECRET_ACCESS_KEY is not (required for s3://)",
            )?;
            let creds = match env("AWS_SESSION_TOKEN") {
                Some(token) => Credentials::new_with_token(key, secret, token),
                None => Credentials::new(key, secret),
            };
            return Ok(AwsCredentials::Static(creds));
        }

        let source = Self::keyless_source()?;
        Ok(AwsCredentials::Dynamic(Mutex::new(Dynamic {
            source,
            http: Http::new_with_timeout(Some(METADATA_TIMEOUT)),
            cached: None,
        })))
    }

    /// Pick the keyless source from the environment (no static keys present).
    fn keyless_source() -> Result<Source> {
        // 2. Web identity (IRSA / any OIDC web-identity federation).
        if let (Some(role_arn), Some(token_file)) =
            (env("AWS_ROLE_ARN"), env("AWS_WEB_IDENTITY_TOKEN_FILE"))
        {
            let region = env("AWS_REGION")
                .or_else(|| env("AWS_DEFAULT_REGION"))
                .unwrap_or_else(|| "us-east-1".to_string());
            return Ok(Source::WebIdentity {
                role_arn,
                token_file,
                session_name: env("AWS_ROLE_SESSION_NAME").unwrap_or_else(|| "nidus".to_string()),
                sts_endpoint: format!("https://sts.{region}.amazonaws.com/"),
            });
        }

        // 3. ECS / Fargate task role.
        if let Some(rel) = env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") {
            return Ok(Source::Container {
                url: format!("{ECS_BASE}{rel}"),
                auth: container_auth_token(),
            });
        }
        if let Some(full) = env("AWS_CONTAINER_CREDENTIALS_FULL_URI") {
            return Ok(Source::Container {
                url: full,
                auth: container_auth_token(),
            });
        }

        // 4. EC2 instance role via IMDS, unless explicitly disabled.
        if env("AWS_EC2_METADATA_DISABLED").as_deref() == Some("true") {
            bail!(
                "no S3 credentials: AWS_ACCESS_KEY_ID is unset, no web-identity/ECS \
                 environment is present, and AWS_EC2_METADATA_DISABLED=true rules out IMDS"
            );
        }
        Ok(Source::Imds)
    }

    /// Current credentials to sign with, refreshing temporary ones that are at/near expiry.
    pub(crate) fn get(&self) -> Result<Credentials> {
        match self {
            AwsCredentials::Static(c) => Ok(c.clone()),
            AwsCredentials::Dynamic(state) => {
                let mut state = state
                    .lock()
                    .map_err(|_| anyhow::anyhow!("AWS credentials lock poisoned"))?;
                let now = now_unix();
                let fresh = state
                    .cached
                    .as_ref()
                    .is_some_and(|c| c.expires_at > now + REFRESH_MARGIN_SECS);
                if !fresh {
                    let cached = state.source.fetch(&state.http)?;
                    state.cached = Some(cached);
                }
                Ok(state.cached.as_ref().expect("just populated").creds.clone())
            }
        }
    }
}

impl Source {
    /// Fetch a fresh set of temporary credentials from this source.
    fn fetch(&self, http: &Http) -> Result<Cached> {
        match self {
            Source::WebIdentity {
                role_arn,
                token_file,
                session_name,
                sts_endpoint,
            } => {
                // The projected web-identity token rotates on disk, so re-read it every refresh.
                let token = std::fs::read_to_string(token_file)
                    .with_context(|| format!("reading web-identity token {token_file}"))?;
                let body = url::form_urlencoded::Serializer::new(String::new())
                    .append_pair("Action", "AssumeRoleWithWebIdentity")
                    .append_pair("Version", "2011-06-15")
                    .append_pair("RoleArn", role_arn)
                    .append_pair("RoleSessionName", session_name)
                    .append_pair("WebIdentityToken", token.trim())
                    .finish();
                let req = http::Request::builder()
                    .method("POST")
                    .uri(sts_endpoint)
                    .header("content-type", "application/x-www-form-urlencoded")
                    .header("accept", "application/xml")
                    .body(body.into_bytes())
                    .context("build STS request")?;
                let (status, resp) = http.run(req)?;
                let resp = String::from_utf8_lossy(&resp);
                if !(200..300).contains(&status) {
                    bail!("STS AssumeRoleWithWebIdentity failed (HTTP {status}): {resp}");
                }
                creds_from_xml(&resp)
            }
            Source::Container { url, auth } => {
                let mut builder = http::Request::builder().method("GET").uri(url);
                if let Some(token) = auth {
                    builder = builder.header("authorization", token);
                }
                let req = builder
                    .body(Vec::new())
                    .context("build ECS creds request")?;
                let (status, resp) = http.run(req)?;
                let resp = String::from_utf8_lossy(&resp);
                if !(200..300).contains(&status) {
                    bail!("ECS container-credentials request failed (HTTP {status}): {resp}");
                }
                creds_from_json(&resp)
            }
            Source::Imds => fetch_imds(http),
        }
    }
}

/// IMDSv2: PUT a session token, read the role name, then the role's credentials JSON.
fn fetch_imds(http: &Http) -> Result<Cached> {
    let token_req = http::Request::builder()
        .method("PUT")
        .uri(format!("{IMDS_BASE}/latest/api/token"))
        .header("x-aws-ec2-metadata-token-ttl-seconds", "21600")
        .body(Vec::new())
        .context("build IMDS token request")?;
    let (status, token) = http.run(token_req).context(
        "EC2 instance-metadata (IMDS) unreachable — set AWS_ACCESS_KEY_ID or run with an \
         attached instance role",
    )?;
    if !(200..300).contains(&status) {
        bail!("IMDS token request failed (HTTP {status})");
    }
    let token = String::from_utf8_lossy(&token).trim().to_string();

    let role = imds_get(
        http,
        &format!("{IMDS_BASE}/latest/meta-data/iam/security-credentials/"),
        &token,
    )?;
    let role = role.trim();
    if role.is_empty() {
        bail!("IMDS reports no IAM role attached to this instance");
    }
    let body = imds_get(
        http,
        &format!("{IMDS_BASE}/latest/meta-data/iam/security-credentials/{role}"),
        &token,
    )?;
    creds_from_json(&body)
}

/// A token-authenticated IMDS `GET`, returning the body as a string.
fn imds_get(http: &Http, url: &str, token: &str) -> Result<String> {
    let req = http::Request::builder()
        .method("GET")
        .uri(url)
        .header("x-aws-ec2-metadata-token", token)
        .body(Vec::new())
        .context("build IMDS request")?;
    let (status, body) = http.run(req)?;
    if !(200..300).contains(&status) {
        bail!("IMDS request to {url} failed (HTTP {status})");
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
}

/// The ECS authorization header value, from `AWS_CONTAINER_AUTHORIZATION_TOKEN` or the
/// file named by `AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE`.
fn container_auth_token() -> Option<String> {
    if let Some(t) = env("AWS_CONTAINER_AUTHORIZATION_TOKEN") {
        return Some(t);
    }
    let path = env("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE")?;
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

/// Build [`Cached`] credentials from an STS `AssumeRoleWithWebIdentity` XML response.
fn creds_from_xml(xml: &str) -> Result<Cached> {
    let key = xml_tag(xml, "AccessKeyId").context("STS response missing AccessKeyId")?;
    let secret = xml_tag(xml, "SecretAccessKey").context("STS response missing SecretAccessKey")?;
    let token = xml_tag(xml, "SessionToken").context("STS response missing SessionToken")?;
    let expires_at = xml_tag(xml, "Expiration")
        .and_then(|e| iso8601_to_unix(&e))
        .unwrap_or_else(|| now_unix() + FALLBACK_TTL_SECS);
    Ok(Cached {
        creds: Credentials::new_with_token(key, secret, token),
        expires_at,
    })
}

/// Build [`Cached`] credentials from an ECS/IMDS credentials JSON document. These use
/// `Token` (not `SessionToken`) for the session token.
fn creds_from_json(json: &str) -> Result<Cached> {
    let key = json_str(json, "AccessKeyId").context("credentials JSON missing AccessKeyId")?;
    let secret =
        json_str(json, "SecretAccessKey").context("credentials JSON missing SecretAccessKey")?;
    let token = json_str(json, "Token").context("credentials JSON missing Token")?;
    let expires_at = json_str(json, "Expiration")
        .and_then(|e| iso8601_to_unix(&e))
        .unwrap_or_else(|| now_unix() + FALLBACK_TTL_SECS);
    Ok(Cached {
        creds: Credentials::new_with_token(key, secret, token),
        expires_at,
    })
}

// ── Dependency-free parsing of the fixed wire shapes ────────────────────────────

/// Content of the first `<tag>…</tag>` element, trimmed. `None` if absent.
fn xml_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml[start..end].trim().to_string())
}

/// Value of a JSON string field: the first `"<key>"` then its `"…"` string value. Good
/// enough for the flat AWS credential documents (whose values carry no escape sequences).
fn json_str(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let after_key = json.find(&needle)? + needle.len();
    let rest = &json[after_key..];
    let colon = rest.find(':')?;
    let after_colon = &rest[colon + 1..];
    let open = after_colon.find('"')? + 1;
    let value = &after_colon[open..];
    let end = value.find('"')?;
    Some(value[..end].to_string())
}

/// Parse an ISO-8601 instant (`YYYY-MM-DDTHH:MM:SS[.fff]Z`, always UTC here) to Unix
/// seconds. Pure integer civil-date math (Howard Hinnant's `days_from_civil`); no date
/// dependency. Returns `None` on a malformed prefix.
fn iso8601_to_unix(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let n = |range: std::ops::Range<usize>| -> Option<i64> {
        std::str::from_utf8(&b[range]).ok()?.parse().ok()
    };
    // YYYY-MM-DDTHH:MM:SS
    let (year, month, day) = (n(0..4)?, n(5..7)?, n(8..10)?);
    let (hour, min, sec) = (n(11..13)?, n(14..16)?, n(17..19)?);
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    // days_from_civil: days since 1970-01-01.
    let y = if month <= 2 { year - 1 } else { year };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    let secs = days * 86400 + hour * 3600 + min * 60 + sec;
    u64::try_from(secs).ok()
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso8601_anchors() {
        assert_eq!(iso8601_to_unix("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(iso8601_to_unix("2021-01-01T00:00:00Z"), Some(1_609_459_200));
        // Leap year: 2024-03-01 is 60 days after 2024-01-01 (Jan 31 + Feb 29).
        assert_eq!(iso8601_to_unix("2024-03-01T00:00:00Z"), Some(1_709_251_200));
        // Fractional seconds and offsets past index 19 are ignored.
        assert_eq!(
            iso8601_to_unix("2021-01-01T00:00:00.123Z"),
            Some(1_609_459_200)
        );
        assert_eq!(iso8601_to_unix("nonsense"), None);
        assert_eq!(iso8601_to_unix("2021-13-01T00:00:00Z"), None);
    }

    #[test]
    fn xml_tag_extracts_credentials() {
        let xml = "<AssumeRoleWithWebIdentityResult><Credentials>\
            <AccessKeyId>ASIA123</AccessKeyId>\
            <SecretAccessKey>secret/with+slashes</SecretAccessKey>\
            <SessionToken>tok==</SessionToken>\
            <Expiration>2026-06-19T20:00:00Z</Expiration>\
            </Credentials></AssumeRoleWithWebIdentityResult>";
        let c = creds_from_xml(xml).unwrap();
        assert_eq!(
            c.expires_at,
            iso8601_to_unix("2026-06-19T20:00:00Z").unwrap()
        );
        assert_eq!(xml_tag(xml, "AccessKeyId").unwrap(), "ASIA123");
        assert_eq!(
            xml_tag(xml, "SecretAccessKey").unwrap(),
            "secret/with+slashes"
        );
        assert_eq!(xml_tag(xml, "Missing"), None);
    }

    #[test]
    fn json_str_extracts_credentials() {
        let json = r#"{"Code":"Success","AccessKeyId":"ASIA9","SecretAccessKey":"sk","Token":"tk","Expiration":"2026-06-19T20:00:00Z"}"#;
        assert_eq!(json_str(json, "AccessKeyId").as_deref(), Some("ASIA9"));
        assert_eq!(json_str(json, "Token").as_deref(), Some("tk"));
        assert_eq!(json_str(json, "Missing"), None);
        let c = creds_from_json(json).unwrap();
        assert_eq!(
            c.expires_at,
            iso8601_to_unix("2026-06-19T20:00:00Z").unwrap()
        );
    }

    #[test]
    fn json_missing_field_errors() {
        let json = r#"{"AccessKeyId":"a","SecretAccessKey":"b"}"#;
        assert!(creds_from_json(json).is_err());
    }
}
